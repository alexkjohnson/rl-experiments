import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions.categorical import Categorical


@dataclass
class Args:
    # Experiment configuration
    exp_name: str = "ppo"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "rlfw"
    wandb_entity: str | None = None
    capture_video: bool = False

    # Environment configuration
    env_id: str = "CartPole-v1"
    num_envs: int = 4

    # Training configuration
    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    num_steps: int = 128

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    target_kl: float | None = 0.01
    max_grad_norm: float = 0.5


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01)
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(obs)
        return action, log_prob, entropy, value


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = (f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}")
    
    # Initialise Weights & Biases if tracking is enabled.
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
        )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n

    agent = Agent(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    # Training loop.
    for iteration in range(1, args.num_iterations + 1):
        # Anneal the learning rate linearly if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect one rollout.
        for step in range(args.num_steps):
            global_step += args.num_envs

            # Store the current observations.
            obs[step] = next_obs
            dones[step] = next_done

            # Select actions and calculate their log probabilities and values.
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step all environments and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminated, truncated)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs, next_done = torch.as_tensor(next_obs, dtype=torch.float32, device=device), torch.as_tensor(next_done, dtype=torch.float32, device=device)

            # Log completed episodes.
            if "final_info" in infos and "episode" in infos["final_info"]:
                episode = infos["final_info"]["episode"]
                for idx in np.flatnonzero(infos["_final_info"]):
                    print(f"global_step={global_step}, episodic_return={episode['r'][idx]}")
                    if args.track:
                        wandb.log({
                            "charts/episodic_return": episode["r"][idx],
                            "charts/episodic_length": episode["l"][idx],
                        }, step=global_step)

            # Bootstrap from the final observation when an episode is truncated.
            for idx in np.flatnonzero(truncated):
                final_obs = torch.as_tensor(infos["final_obs"][idx], dtype=torch.float32, device=device)
                with torch.no_grad():
                    final_value = agent.critic(final_obs).squeeze(-1)
                rewards[step, idx] += args.gamma * final_value

        # Calculate advantages and returns.
        with torch.no_grad():
            next_value = agent.critic(next_obs).squeeze(-1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * next_nonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            returns = advantages + values

        # Flatten the rollout data.
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimise the policy and value networks.
        approx_kls = []
        clipfracs = []
        pg_losses = []
        v_losses = []
        entropy_losses = []
        stop_update = False

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Recalculate log probabilities using the updated policy.
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                approx_kls.append(approx_kl.item())
                clipfracs.append(clipfrac.item())

                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop_update = True
                    break

                # Normalize advantages.
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Calculate the clipped policy loss.
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Calculate the value-function loss.
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Calculate the entropy bonus.
                entropy_loss = entropy.mean()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())

                # Combine the losses.
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if stop_update:
                break

        # Calculate explained variance.
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0
            else 1 - np.var(y_true - y_pred) / var_y
        )

        mean_approx_kl = np.mean(approx_kls)
        mean_clipfrac = np.mean(clipfracs)

        # Report training progress.
        print(
            f"iteration={iteration}/{args.num_iterations}, "
            f"global_step={global_step}, "
            f"approx_kl={mean_approx_kl:.5f}, "
            f"clipfrac={mean_clipfrac:.3f}, "
        )

        # Log metrics to Weights & Biases.
        if args.track:
            metrics = {
                "charts/approx_kl": mean_approx_kl,
                "charts/clipfrac": mean_clipfrac,
                "charts/explained_variance": explained_var,
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "losses/policy_loss": np.mean(pg_losses),
                "losses/value_loss": np.mean(v_losses),
                "losses/entropy": np.mean(entropy_losses),
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
            }
            wandb.log(metrics, step=global_step)

    envs.close()
