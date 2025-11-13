# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import copy

# For diabetic env
import dosing_rl_gym

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Diabetic-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    env_per_pool: int = 1
    """the number of environments per pool (or virtual client)"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    local_update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.005
    """the maximum norm for the gradient clipping"""
    max_grad_norm_value: float = 0.5
    """the maximum norm for the gradient clipping of the value function"""
    target_kl: float = None
    """the target KL divergence threshold"""
    noise_multiplier: float = 0.5
    """the noise multiplier for the DP noise"""
    forward_adam_states: bool = True
    """whether to forward adam state from iteration to iteration or not"""
    num_hidden_neurons: int = 64
    """number of hidden neurons for the critic and actor"""
    no_ppo_clip: bool = True
    """whether to use PPO ratio clipping or not. if clip_vloss=True, also affects the value loss."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, num_hidden_neurons):
        super().__init__()
        self.num_hidden_neurons = num_hidden_neurons
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.num_hidden_neurons)),
            nn.Tanh(),
            layer_init(nn.Linear(self.num_hidden_neurons, self.num_hidden_neurons)),
            nn.Tanh(),
            layer_init(nn.Linear(self.num_hidden_neurons, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.num_hidden_neurons)),
            nn.Tanh(),
            layer_init(nn.Linear(self.num_hidden_neurons, self.num_hidden_neurons)),
            nn.Tanh(),
            layer_init(nn.Linear(self.num_hidden_neurons, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def train(args, save_model=False):
    list_envs = list(range(args.num_envs))
    pools = []
    for i in range(0, len(list_envs), args.env_per_pool):
        pools.append(list_envs[i: i + args.env_per_pool])
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if args.env_id.startswith("Diabetic") and args.env_id not in gym.registry:
        env_version = args.env_id.split("-v")[-1]
        gym.register(id=args.env_id, entry_point=f"dosing_rl_gym.envs:Diabetic{env_version}Env")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Initialize global agent, local agents and corresponding optimizers (DPPG logic)
    agent = Agent(envs, args.num_hidden_neurons).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=1, eps=1e-5)
    local_agents = [copy.deepcopy(agent) for _ in range(len(pools))]
    local_optimizers = [optim.Adam(local_agents[i].parameters(), lr=args.learning_rate, eps=1e-5) for i in
                        range(len(local_agents))]

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for opt in local_optimizers:
                opt.param_groups[0]["lr"] = lrnow

        # Forward private gradient to local Adam states for privacy (DPPG logic)
        if args.forward_adam_states and iteration >= 2:
            for opt in local_optimizers:
                for opt_param, param in zip(opt.param_groups[0]['params'], agg_gradient.values()):
                    state = opt.state[opt_param]
                    if state:  # Check if the state exists for the parameter
                        state['exp_avg'] = param.clone()
                        state['exp_avg_sq'] = param.clone() ** 2

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Optimizing the policy and value network

        # Initialize private agg gradient and store initial params (DPPG logic)
        init_params = {}
        agg_gradient = {}
        for name, param in agent.named_parameters():
            agg_gradient[name] = torch.zeros_like(param)
            init_params[name] = copy.deepcopy(param)

        # For monitoring
        for pool_id in range(len(pools)):
            local_agent = local_agents[pool_id]
            local_optimizer = local_optimizers[pool_id]
            local_agent.load_state_dict(agent.state_dict())

            b_obs = obs[:, pools[pool_id], :].reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs[:, pools[pool_id]].reshape(-1)
            b_actions = actions[:, pools[pool_id]].reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages[:, pools[pool_id]].reshape(-1)
            b_returns = returns[:, pools[pool_id]].reshape(-1)
            b_values = values[:, pools[pool_id]].reshape(-1)

            b_inds = np.arange(args.num_steps * len(pools[pool_id]))
            clipfracs = []
            for epoch in range(args.local_update_epochs):
                np.random.shuffle(b_inds)
                batch_size = len(b_inds)
                minibatch_size = int(batch_size // args.num_minibatches)
                for start in range(0, int(args.num_steps * len(pools[pool_id])), minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = local_agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    if not args.no_ppo_clip:
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    else:  # DP clipping can replace PPO clipping
                        pg_loss = pg_loss1.mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss and not args.no_ppo_clip:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    local_optimizer.zero_grad()
                    loss.backward()
                    local_optimizer.step()

                    # DP clipping step (DPPG logic)
                    with torch.no_grad():
                        local_update = {}
                        local_update_norm_policy = 0
                        local_update_norm_value = 0

                        # Compute local update norms
                        for name, param in local_agent.named_parameters():
                            diff = param - init_params[name]
                            local_update[name] = diff
                            diff_norm = torch.norm(diff) ** 2
                            if name.startswith('actor'):
                                local_update_norm_policy += diff_norm
                            else:
                                local_update_norm_value += diff_norm
                        local_update_norm_policy = local_update_norm_policy ** 0.5
                        local_update_norm_value = local_update_norm_value ** 0.5

                        # Compute clipping factors
                        clip_factor_policy = np.max([1, local_update_norm_policy.cpu() / (args.max_grad_norm)])
                        clip_factor_value = np.max(
                            [1, local_update_norm_value.cpu() / (args.max_grad_norm_value)])

                        # Update local parameters
                        for name, param in local_agent.named_parameters():
                            if name.startswith('actor'):
                                local_update[name] /= clip_factor_policy
                            else:
                                local_update[name] /= clip_factor_value
                            param.copy_(init_params[name] + local_update[name])

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            # Update aggregated gradient (DPPG Logic)
            for name, param in local_agent.named_parameters():
                agg_gradient[name] += (param - init_params[name]) / (len(pools))

        # Add DP noise and manually update parameters (DPPG logic)
        with torch.no_grad():
            sigma = args.noise_multiplier * args.max_grad_norm / (len(pools))
            for name, param in agent.named_parameters():
                if args.noise_multiplier > 0 and name.startswith('actor'):
                    agg_gradient[name] += torch.normal(mean=0, std=sigma, size=agg_gradient[name].size()).to(
                        device)
                param.copy_(init_params[name] + agg_gradient[name])

        # Log metrics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()


if __name__ == '__main__':
    args = tyro.cli(Args)
    n_seeds = 10
    for seed in range(n_seeds):
        print(f'Training env {args.env_id} / seed {seed + 1}/{n_seeds}')
        setattr(args, 'seed', seed)
        train(args)

