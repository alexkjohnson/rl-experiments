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
        super().__init()
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
actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape)
logprobs = torch.zeros((num_steps, num_envs))
rewards = torch.zeros((num_steps, num_envs))
dones = torch.zeros((num_steps, num_envs))
values = torch.zeros((num_steps, num_envs))

