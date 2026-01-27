from pyvirtualdisplay import Display
import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline


#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# gymnasium
import gymnasium as gym
import imageio



import random
import pickle5 as pickle
from tqdm.notebook import tqdm


virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# create env
env_id = "CartPole-v1"

env = gym.make(env_id)
eval_env = gym.make(env_id)

s_size = env.observation_space.shape[0]
a_size = env.action_space.n
print(f"State space: {s_size}")
print(f"Action space: {a_size}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


policy = Policy(s_size, a_size, 128).to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

def reinforce(n_training_episodes, max_t, gamma, print_every=10, batch_size=10):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state, info = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            if terminated:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        for t in reversed(range(n_steps)):
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(gamma * disc_return_t + rewards[t])
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        if i_episode % print_every == 0:
            print("Episode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))
    return scores


cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

cartpole_policy = Policy(
    cartpole_hyperparameters["state_space"],
    cartpole_hyperparameters["action_space"],
    cartpole_hyperparameters["h_size"],
).to(device)
cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

scores = reinforce(
    cartpole_hyperparameters["n_training_episodes"],
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["gamma"],
    100,
)

def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, info = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        for step in range(max_steps):
            action, _ = policy.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_rewards.append(reward)
            if terminated:
                break
        episode_rewards.append(sum(episode_rewards))
    return np.mean(episode_rewards), np.std(episode_rewards)

mean_reward, std_reward = evaluate_agent(eval_env, cartpole_hyperparameters["max_t"], cartpole_hyperparameters["n_evaluation_episodes"], cartpole_policy)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
