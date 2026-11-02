import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


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

# Environment setup
env_id = "CartPole-v1"
num_envs = 4
num_steps = 128

envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make(env_id) for _ in range(num_envs)],
    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
)

# Storage setup
obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape)
actions = torch.zeros((num_steps, num_envs), dtype=torch.long)
logprobs = torch.zeros((num_steps, num_envs))
rewards = torch.zeros((num_steps, num_envs))
dones = torch.zeros((num_steps, num_envs))
values = torch.zeros((num_steps, num_envs))

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

gamma = 0.99
gae_lambda = 0.95

# PPO hyperparameters
learning_rate = 2.5e-4
num_minibatches = 4
update_epochs = 4
clip_coef = 0.2
clip_vloss = True
ent_coef = 0.0
vf_coef = 0.5
target_kl = 0.01
max_grad_norm = 0.5
batch_size = num_envs * num_steps
minibatch_size = batch_size // num_minibatches
total_timesteps = 10_000
num_iterations = total_timesteps // batch_size

obs_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.n
agent = Agent(obs_dim, action_dim)
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

next_obs, _ = envs.reset(seed=seed)
next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
next_done = torch.zeros(num_envs)
episode_returns = np.zeros(num_envs)
episode_lengths = np.zeros(num_envs)

# Track training progress.
global_step = 0

for iteration in range(1, num_iterations + 1):
    # Collect one rollout.
    for step in range(num_steps):
        global_step += num_envs

        # Store the current observations.
        obs[step] = next_obs
        dones[step] = next_done

        # Select actions and calculate their log probabilities and values.
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)

        # Store the policy outputs.
        actions[step] = action
        logprobs[step] = logprob
        values[step] = value.squeeze(-1)

        # Step all environments.
        next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action.cpu().numpy())

        # Track episode statistics.
        episode_returns += reward_np
        episode_lengths += 1
        for idx in np.flatnonzero(terminated_np | truncated_np):
            print(f"global_step={global_step}, "f"env={idx}, "
                f"episodic_return={episode_returns[idx]:.1f}, "
                f"episodic_length={episode_lengths[idx]}"
            )
            episode_returns[idx], episode_lengths[idx] = 0, 0

        # Convert rewards into a tensor.
        reward = torch.as_tensor(reward_np, dtype=torch.float32)

        # Bootstrap from the final observation when an episode is truncated.
        for idx in np.flatnonzero(truncated_np):
            final_obs = torch.as_tensor(infos["final_obs"][idx], dtype=torch.float32)

            with torch.no_grad():
                final_value = agent.critic(final_obs).squeeze()

            reward[idx] += gamma * final_value

        rewards[step] = reward

        # Update the current observation and episode-ending flags.
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)
        next_done = torch.as_tensor(np.logical_or(terminated_np, truncated_np), dtype=torch.float32)

    # Calculate advantages and returns.
    with torch.no_grad():
        next_value = agent.critic(next_obs).squeeze(-1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_nonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * next_nonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        returns = advantages + values

    # Flatten the rollout data.
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    approx_kls = []
    clipfracs = []
    epochs_completed = 0
    stop_update = False

    # Optimise the policy and value networks.
    b_inds = np.arange(batch_size)
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        epochs_completed += 1

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            # Recalculate log probabilities using the updated policy.
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds])

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
            approx_kls.append(approx_kl.item())
            clipfracs.append(clipfrac.item())

            if approx_kl > target_kl:
                stop_update = True
                break

            # Normalize advantages.
            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Calculate the clipped policy loss.
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Calculate the value-function loss.
            newvalue = newvalue.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            # Calculate the entropy bonus.
            entropy_loss = entropy.mean()

            # Combine the losses.
            loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if stop_update:
            break

    print(
        f"iteration={iteration}/{num_iterations}, "
        f"global_step={global_step}, "
        f"approx_kl={np.mean(approx_kls):.5f}, "
        f"clipfrac={np.mean(clipfracs):.3f}, "
        f"epochs={epochs_completed}/{update_epochs}"
    )

envs.close()
