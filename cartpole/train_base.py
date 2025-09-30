import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# --------------- Variables -------------------
MODEL_NAME = "model/policy_cartpole_frompy.pt"
RESULT_REWARD = "result/returns_base.png"
RESULT_LENGTH = "result/lengths_base.png"
NRO_EPISODES = 500


# --------------- RL Algorithm ----------------
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


# --------------- Basic Definition --------------
env = gym.make("CartPole-v1")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

model_policy = PolicyNet(obs_dim, act_dim)
optimizer = torch.optim.Adam(model_policy.parameters(), lr=1e-3)

returns, lengths = [], []


# --------------- Training --------------
for episode in range(NRO_EPISODES):
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
    
    returns.append(total_return)
    lengths.append(len(rewards))

env.close()

# Guardar modelo entrenado
torch.save(model_policy.state_dict(), MODEL_NAME)


# --------------- Guardar Graficas ---------------
plt.figure()
plt.plot(returns)
plt.xlabel("Episodio")
plt.ylabel("Return")
plt.title("Return por episodio")
plt.savefig(RESULT_REWARD)
plt.close()

plt.figure()
plt.plot(lengths)
plt.xlabel("Episodio")
plt.ylabel("Longitud del episodio")
plt.title("Duración de los episodios")
plt.savefig(RESULT_LENGTH)
plt.close()