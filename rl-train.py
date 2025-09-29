import random

import numpy as np

import gymnasium as gym

import torch
from torch import nn


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    
    def forward(self, x):
        return self.stack(x)
    

def get_new_reward(obs):
    reward = 1
    return reward

env = gym.make("CartPole-v1")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

model_policy = PolicyNet(obs_dim, act_dim)

optimizer = torch.optim.Adam(model_policy.parameters(), lr=1e-3)

for episode in range(4000):
    obs, _ = env.reset()
    log_probs = []
    rewards = []

    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = model_policy(obs_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))

        obs, reward, terminated, truncated, _ = env.step(action.item())

        reward = get_new_reward(obs)

        done = terminated or truncated
        rewards.append(reward)

    # Calcular retorno total (suma de rewards)
    total_return = sum(rewards)

    # Actualizar policy (REINFORCE)
    loss = []
    for log_prob in log_probs:
        loss.append(-log_prob * total_return)
    loss = torch.stack(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode}, Return: {total_return}")

env.close()

# Guardar modelo entrenado
torch.save(model_policy.state_dict(), "policy_cartpole_frompy.pt")