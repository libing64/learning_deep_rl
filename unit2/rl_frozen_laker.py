import os
from pyvirtualdisplay import Display
import numpy as np
import gymnasium as gym
import random
import pickle5 as pickle
from tqdm.notebook import tqdm


virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# create env
env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False, render_mode='rgb_array')

print("observation space: ", env.observation_space.n)
print("action space: ", env.action_space.n)

state_space = env.observation_space.n
action_space = env.action_space.n
print("state space: ", state_space)
print("action space: ", action_space)

# initialize Q-table
Q = np.zeros([state_space, action_space])
print("Q-table: ", Q)

# set hyperparameters
alpha = 0.1
gamma = 0.99


# init for Q table
def initialize_q_table(state_space, action_space):  
    Qtable = np.zeros([state_space, action_space])
    return Qtable


Qtable_frozenlake = initialize_q_table(state_space, action_space)
print("Qtable_frozenlake: ", Qtable_frozenlake)


# greedy policy
def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state])
    return action


# epsilon-greedy policy
def epsilon_greedy_policy(Qtable, state, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = greedy_policy(Qtable, state)
    return action


#training parameters    
n_training_episodes = 10000
learning_rate = 0.7

# evaluation parameters
n_eval_episodes = 100

# environment parameters
env_id = "FrozenLake-v1"
max_steps = 99
gamma = 0.95
eval_seed = []

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005


def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        # reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        # reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        # repeat
        for step in range(max_steps):
            # choose the action at using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            # take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)
            # update the Q-value Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )
            # if terminated or truncated finish the episode
            if terminated or truncated:
                break
            # our next state is the new state
            state = new_state
    return Qtable


Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)
print("Qtable_frozenlake: ", Qtable_frozenlake)


# evaluate the agent
def evaluate_agent(n_eval_episodes, max_steps, Qtable, eval_seed):
    episodes_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if eval_seed:
            state, info = env.reset(seed=eval_seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0
        for step in range(max_steps):
            action = greedy_policy(Qtable, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward
            if terminated or truncated:
                break
            state = new_state
        episodes_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episodes_rewards)
    std_reward = np.std(episodes_rewards)
    return mean_reward, std_reward


# evaluate our agent
mean_reward, std_reward = evaluate_agent(n_eval_episodes, max_steps, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
