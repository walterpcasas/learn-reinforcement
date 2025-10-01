import random

import numpy as np

import gymnasium as gym

import torch
from torch import nn

MODEL_WEIGHTS = "model/policy_cartpole_base.pt"


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


env = gym.make("CartPole-v1", render_mode="human")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

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

# Crear policy con la misma arquitectura
policy = PolicyNet(obs_dim, act_dim)
policy.load_state_dict(torch.load(MODEL_WEIGHTS))  # cargar pesos guardados
policy.eval()


for episode in range(5):
    obs, _ = env.reset()
    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy(obs_tensor)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()  # greedy (sin exploración)

        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated:
            done = terminated
            print(f"episodio: {episode} - terminado")
        elif truncated:
            done = truncated
            print(f"episodio: {episode} - truncado")

env.close()