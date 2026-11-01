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
    [lambda: gym.make(env_id) for _ in range(num_envs)]
)

# Storage setup
obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape)
actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape, dtype=torch.long)
logprobs = torch.zeros((num_steps, num_envs))
rewards = torch.zeros((num_steps, num_envs))
dones = torch.zeros((num_steps, num_envs))
values = torch.zeros((num_steps, num_envs))

obs_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.n

agent = Agent(obs_dim, action_dim)

# Collect one rollout.
next_obs, _ = envs.reset(seed=1)
next_obs = torch.as_tensor(next_obs, dtype=torch.float32)

next_done = torch.zeros(num_envs)

for step in range(num_steps):
    # Store the current observations.
    obs[step] = next_obs

    # Select actions and calculate their log probabilities and values.
    with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(next_obs)

    # Store the policy outputs.
    actions[step] = action
    logprobs[step] = logprob
    values[step] = value.squeeze(-1)

    # Step all environments.
    next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action.cpu().numpy())

    # Convert environment outputs into tensors.
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)
    rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32)

    # Record whether each episode ended.
    next_done = torch.as_tensor(np.logical_or(terminated_np, truncated_np), dtype=torch.float32)
    dones[step] = next_done

print("Rollout collected!")
print("Observations:", obs.shape)
print("Actions:", actions.shape)
print("Log probabilities:", logprobs.shape)
print("Rewards:", rewards.shape)
print("Dones:", dones.shape)
print("Values:", values.shape)