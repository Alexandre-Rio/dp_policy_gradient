import tyro

from envs.riverswim import RiverSwimEnv
from baselines.ucb_agent import UCBVI_JDP

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym


from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    num_seeds: int = 10
    num_episodes: int = 1000
    gamma: float = 0.99
    policy_lr: float = 10.0
    baseline_lr: float = 0.01
    min_policy_lr: float = 0.01
    policy_lr_update_freq: int = 100
    policy_update_factor: float = 2.0
    npg: bool = False
    max_grad_norm: float = 10.0
    epsilon: float = 5.0
    conf_radius: float = 1.0
    adaptive_clipping: bool = True
    num_fisher_episodes: int = 25
    fisher_regularizer: float = 1e-3
    use_tensorboard: bool = True
    prob_stay_terminal: float = 0.9


def ucb_jdp(env, policy, args, seed, writer):
    np.random.seed(seed)
    total_reward_list = []
    total_regret_list = []
    cum_regret = 0

    opt_qVals, opt_vVals = env.compute_optVals()

    for episode in range(1, args.num_episodes + 1):

        # Generate an episode
        states, actions, rewards, features_list, log_probs_list = [], [], [], [], []
        state, _ = env.reset()
        done = False
        t = 0

        policy.update_policy(ep=episode)

        while not done:
            action = policy.sample_action(state.argmax(), t)
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            regret = opt_qVals[state.argmax(), t].max() - opt_qVals[state.argmax(), t][action]
            cum_regret += regret

            policy.update_obs(state.argmax(), action, reward, next_state.argmax(), int(1 - done), t)

            state = next_state
            t += 1

        # Logging progress
        total_reward = sum(rewards)
        total_reward_list.append(total_reward)
        total_regret_list.append(cum_regret)
        if episode % 10 == 0 or episode == args.num_episodes - 1:
            print(f"Episode {episode}: Total Reward = {total_reward}")
        if args.use_tensorboard:
            writer.add_scalar(f"charts_seed_{seed}/return", total_reward, episode)
            writer.add_scalar(f"charts_seed_{seed}/cum_regret", cum_regret, episode)

    return total_reward_list, total_regret_list


def main(args):
    env = gym.make("RiverSwim-v0", normalize=True, prob_stay_terminal=args.prob_stay_terminal)  # Ensure you have this environment available in Gymnasium

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    feature_dim = state_dim + action_dim

    if args.use_tensorboard:
        run_name = f'riverswim_ucb__eps_{args.epsilon}_{args.prob_stay_terminal}'
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    total_ep_rewards_pg_list = []
    total_ep_regrets_pg_list = []
    for seed in range(args.num_seeds):
        policy = UCBVI_JDP(state_dim, action_dim, env.horizon, args.num_episodes, privEps=args.epsilon)
        total_rewards_pg, total_regrets_pg = ucb_jdp(env, policy, args, seed, writer)
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


if __name__ == '__main__':
    args = tyro.cli(Args)
    for epsilon in [1.0, 5.0]:
        setattr(args, "epsilon", epsilon)
        for prob_stay_terminal in [0.6, 0.9]:
            setattr(args, "prob_stay_terminal", prob_stay_terminal)
            main(args)


