import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = gym.make('LunarLander-v3')
print("observation space: ", env.observation_space.shape)
print("action space: ", env.action_space.n)

# reset the environment
observation, info = env.reset()
for _ in range(20):
    # random action
    action = env.action_space.sample()
    print('action: ', action)
    # step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    # if the episode is terminated or truncated, reset the environment
    if terminated or truncated:
        print('environment is reset')
        observation, info = env.reset()
env.close()


# try to train a model 
env = make_vec_env('LunarLander-v3', n_envs=16)
model = PPO(policy='MlpPolicy', env=env,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
             verbose=1)

model.learn(total_timesteps=10000)
model_name = 'ppo_lunar_lander_v3'
model.save(model_name)

# evaluate the agent
eval_env = Monitor(gym.make("LunarLander-v3", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


# model.save('ppo_lunar_lander')

# model = DQN.load('ppo_lunar_lander')