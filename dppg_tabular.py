import tyro

from envs.riverswim import RiverSwimEnv

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from scipy.linalg import eig
from scipy.stats import ncx2

from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    num_seeds: int = 10
    num_episodes: int = 1000
    gamma: float = 0.99
    policy_lr: float = 12
    baseline_lr: float = 0.001
    min_policy_lr: float = 0.055
    policy_lr_update_freq: int = 50
    policy_update_factor: float = 5.0
    npg: bool = False
    normalize_features = False
    prob_stay_terminal = 0.9

    trust_region_radius: float = 3.5
    epsilon: float = 5.0
    delta: float = 1e-3
    beta: float = 0.4
    adaptive_clipping: str = "fisher"  # fisher, spg_quantile, spg_exp

    num_fisher_episodes: int = 25
    fisher_regularizer: float = 1e-3
    use_tensorboard: bool = True
    plot: bool = False


class LinearPolicy:
    def __init__(self, state_dim, action_dim):
        self.theta = np.random.randn(state_dim + action_dim) * 0.01
        self.state_dim = state_dim
        self.action_dim = action_dim

    def create_features(self, state, action):
        state_action_features = np.zeros(self.state_dim + self.action_dim)
        state_action_features[:self.state_dim] = state
        state_action_features[self.state_dim + action] = 1
        return state_action_features

    def get_action_probs(self, state):
        features = np.array([self.create_features(state, action) for action in range(self.action_dim)])
        z = np.dot(features, self.theta)
        exp_z = np.exp(z - np.max(z))  # For numerical stability
        return exp_z / np.sum(exp_z)

    def sample_action(self, state):
        action_probs = self.get_action_probs(state)
        log_probs = np.log(action_probs)
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, log_probs

    def update(self, grad, lr):
        self.theta += lr * grad


class LinearBaseline:
    def __init__(self, feature_dim):
        self.w = np.zeros(feature_dim)

    def predict(self, features):
        return np.dot(features, self.w)

    def update(self, features_batch, prediction_batch, targets, lr):
        td_error = (targets - prediction_batch)[:, np.newaxis]
        td_error = np.tile(td_error, features_batch.shape[1])
        grad = (td_error * features_batch).mean(axis=0)
        self.w += lr * grad


def reinforce_with_baseline(env, policy, baseline, args, seed, writer):
    np.random.seed(seed)
    total_reward_list = []
    total_regret_list = []
    cum_regret = 0
    grad_norm_agg = 0
    if args.adaptive_clipping is not None:
        adap_clip_norm_agg = 0
    n_in_trust_region = 0

    policy_lr = args.policy_lr
    noise_multiplier = np.sqrt(2 * np.log(1.25 / args.delta)) / args.epsilon
    problem_dim = policy.theta.shape[0]

    qVals, _ = env.compute_optVals()

    for episode in range(args.num_episodes):

        # Estimate Fisher "publicly"
        if args.adaptive_clipping == "fisher":
            fisher_ep_count = 0
            fisher = 0.0
            features_list = []
            for fisher_ep in range(args.num_fisher_episodes):
                state, _ = env.reset()
                done = False
                while not done:
                    action, log_probs = policy.sample_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    features = policy.create_features(state, action)
                    features_list.append(features)
                    fisher_ep_count += 1
            features_batch = np.array(features_list)
            if args.normalize_features:
                features_batch -= features_batch.mean(axis=0)
            fisher = np.matmul(features_batch.T, features_batch)
            fisher /= fisher_ep_count
            fisher += args.fisher_regularizer * np.eye(fisher.shape[0])

        # Generate an episode
        states, actions, rewards, features_list, log_probs_list = [], [], [], [], []
        state, _ = env.reset()
        done = False
        t = 0

        while not done:
            action, log_probs = policy.sample_action(state)
            next_state, reward, done, _, _ = env.step(action)

            features = policy.create_features(state, action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            features_list.append(features)
            log_probs_list.append(log_probs[action])

            regret = qVals[state.argmax(), t].max() - qVals[state.argmax(), t][action]
            cum_regret += regret

            state = next_state
            t += 1

        # Compute returns and baseline values
        returns = np.zeros_like(rewards, dtype=float)
        g = 0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + args.gamma * g
            returns[t] = g

        features_batch = np.array(features_list)
        prediction_list = []
        advantage_list = []
        for t in range(len(features_batch)):
            features = features_batch[t]
            prediction = baseline.predict(features)
            advantage = returns[t] - prediction
            prediction_list.append(prediction)
            advantage_list.append(advantage)
        prediction_batch = np.array(prediction_list)
        advantage_batch = np.array(advantage_list)[:, np.newaxis]
        centered_features_batch = features_batch - features_batch.mean(axis=0)  # Log-prob gradient = centered features

        # Update baseline
        baseline.update(features_batch, prediction_batch, returns, args.baseline_lr)

        # Update policy
        if args.normalize_features:
            grad = (advantage_batch * centered_features_batch).mean(axis=0)
        else:
            grad = (advantage_batch * features_batch).mean(axis=0)
        grad_norm = np.linalg.norm(grad)
        grad_norm_agg += grad_norm
        clipping_norm = args.trust_region_radius
        if args.trust_region_radius is not None:

            if args.adaptive_clipping == "fisher":
                max_ev_fisher = eig(fisher)[0].real.max()
                trace_fisher = np.trace(fisher)
                den = max_ev_fisher + noise_multiplier ** 2 * trace_fisher
                clipping_norm = (1 / policy_lr) * np.sqrt(args.beta) * np.sqrt(2 * args.trust_region_radius / den)
                adap_clip_norm_agg += clipping_norm
            elif args.adaptive_clipping == "spg_quantile":
                nc = 1 / (noise_multiplier ** 2)
                df = problem_dim
                ncx2_dist = ncx2(df=df, nc=nc)
                quantile = ncx2_dist.ppf(1 - args.beta)
                den = noise_multiplier ** 2 * quantile
                clipping_norm = (1 / policy_lr) * np.sqrt(2 * args.trust_region_radius / den)
                adap_clip_norm_agg += clipping_norm
            elif args.adaptive_clipping == "spg":
                den = 1 + noise_multiplier ** 2 * problem_dim
                clipping_norm = (1 / policy_lr) * np.sqrt(args.beta) * np.sqrt(2 * args.trust_region_radius / den)

            clip_factor = np.max([1.0, grad_norm / clipping_norm])
            grad /= clip_factor

            if noise_multiplier > 0:
                sigma = noise_multiplier * clipping_norm
                noise = np.random.normal(loc=0, scale=sigma, size=grad.shape)
                grad += noise
            noisy_grad_norm = np.linalg.norm(policy_lr * grad)

            if args.trust_region_radius is not None and noisy_grad_norm <= args.trust_region_radius:
                n_in_trust_region += 1
            policy.update(grad, policy_lr)

        # Logging progress
        total_reward = sum(rewards)
        total_reward_list.append(total_reward)
        total_regret_list.append(cum_regret)
        if args.use_tensorboard:
            writer.add_scalar(f"charts_seed_{seed}/return", total_reward, episode)
            writer.add_scalar(f"charts_seed_{seed}/cum_regret", cum_regret, episode)
            writer.add_scalar(f"charts_seed_{seed}/perc_tr", n_in_trust_region / 10, episode)
            writer.add_scalar(f"charts_seed_{seed}/grad_norm_agg", grad_norm_agg / 10, episode)
            writer.add_scalar(f"charts_seed_{seed}/adap_clip_norm_agg", adap_clip_norm_agg / 10, episode)
        if episode % 10 == 0 or episode == args.num_episodes - 1:
            print(f"Episode {episode}: Total Reward = {total_reward} / Grad Norm Avg.: {grad_norm_agg / 10:.2f}")
            if args.adaptive_clipping:
                adap_clip_norm_agg = 0
            grad_norm_agg = 0
            n_in_trust_region = 0

        if episode % args.policy_lr_update_freq == 0:
            policy_lr /= args.policy_update_factor
            policy_lr = np.max([policy_lr, args.min_policy_lr])

    return total_reward_list, total_regret_list


def main(args):
    env = gym.make("RiverSwim-v0", normalize=True, prob_stay_terminal=args.prob_stay_terminal)  # Ensure you have this environment available in Gymnasium

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    feature_dim = state_dim + action_dim

    if args.use_tensorboard:
        run_name = f'riverswim_dp_pg__{args.prob_stay_terminal}_{args.epsilon}_{args.adaptive_clipping}_{args.trust_region_radius}_{args.beta}'
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    total_ep_rewards_pg_list = []
    total_ep_regrets_pg_list = []
    for seed in range(args.num_seeds):
        policy = LinearPolicy(state_dim, action_dim)
        baseline = LinearBaseline(feature_dim)
        total_rewards_pg, total_regrets_pg = reinforce_with_baseline(env, policy, baseline, args, seed, writer)
        total_ep_rewards_pg_list.append(total_rewards_pg)
        total_ep_regrets_pg_list.append(total_regrets_pg)

    metric = 'regret'
    res_list = np.array(total_ep_rewards_pg_list) if metric == 'reward' else np.array(total_ep_regrets_pg_list)
    mean_pg = res_list.mean(axis=0)
    pg_ci = 1.96 * res_list.std(axis=0) / np.sqrt(args.num_seeds)
    ep_axis = np.arange(1, args.num_episodes + 1)

    for episode in range(len(mean_pg)):
        writer.add_scalar("charts/regret_avg", mean_pg[episode], episode)
        writer.add_scalar("charts/regret_ci", pg_ci[episode], episode)

    if args.plot:
        plt.plot(ep_axis, mean_pg)
        plt.fill_between(ep_axis, mean_pg - pg_ci, mean_pg + pg_ci, alpha=0.3)

        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    args = tyro.cli(Args)

    for epsilon in [1.0, 5.0]:
        setattr(args, "epsilon", epsilon)
        for prob_stay_terminal in [0.6, 0.9]:
            setattr(args, "prob_stay_terminal", prob_stay_terminal)
            main(args)