#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Car Racing Environment Wrapper
Car Racing 环境封装
"""

import gymnasium as gym
import numpy as np


class CarRacingWrapper:
    """
    Car Racing 环境封装类
    
    CarRacing-v2 是一个基于图像的强化学习控制问题：
    - 观察空间：96x96x3 RGB图像
    - 动作空间：Box(3,) 连续动作空间 [转向, 油门, 刹车]
    - 目标：在赛道上尽可能快地行驶，获得高分
    """
    
    def __init__(self, render_mode=None, frameskip=4):
        """
        初始化 Car Racing 环境
        
        Args:
            render_mode: 渲染模式 ('human' 用于可视化, None 用于训练)
            frameskip: 帧跳过数（用于加速训练）
        """
        self.env = gym.make('CarRacing-v2', render_mode=render_mode, frameskip=frameskip)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.frameskip = frameskip
        
    def reset(self, seed=None):
        """
        重置环境到初始状态
        
        Returns:
            observation: 96x96x3 RGB图像
            info: 额外信息
        """
        if seed is not None:
            return self.env.reset(seed=seed)
        return self.env.reset()
    
    def step(self, action):
        """
        执行动作
        
        Args:
            action: 动作 
                - 连续动作: [转向(-1到1), 油门(0到1), 刹车(0到1)]
                - 离散动作: 0-14 (映射到15个离散动作)
            
        Returns:
            observation: 观察 (96x96x3 RGB图像)
            reward: 奖励（前进获得正奖励，碰撞或偏离赛道获得负奖励）
            terminated: 是否终止
            truncated: 是否截断（达到最大步数1000）
            info: 额外信息
        """
        return self.env.step(action)
    
    def render(self):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    def get_observation_shape(self):
        """获取观察空间形状"""
        return self.observation_space.shape
    
    def get_action_dim(self):
        """获取动作空间维度（连续动作空间）"""
        return self.action_space.shape[0]
    
    def get_action_space_type(self):
        """获取动作空间类型"""
        return type(self.action_space).__name__
    
    def discretize_action(self, discrete_action):
        """
        将离散动作转换为连续动作
        
        Car Racing 的连续动作空间为 Box(3,):
        - action[0]: 转向 (-1.0 到 1.0)
        - action[1]: 油门 (0.0 到 1.0)
        - action[2]: 刹车 (0.0 到 1.0)
        
        我们将离散化为15个动作：
        - 0: 无操作 [0, 0, 0]
        - 1-4: 左转 [转向, 0, 0] (转向: -1.0, -0.5, -0.25, -0.1)
        - 5-8: 右转 [转向, 0, 0] (转向: 0.1, 0.25, 0.5, 1.0)
        - 9-12: 前进 [0, 油门, 0] (油门: 0.25, 0.5, 0.75, 1.0)
        - 13: 左转+前进 [-0.5, 0.5, 0]
        - 14: 右转+前进 [0.5, 0.5, 0]
        
        Args:
            discrete_action: 离散动作 (0-14)
            
        Returns:
            continuous_action: 连续动作 [转向, 油门, 刹车]
        """
        action_map = {
            0: [0.0, 0.0, 0.0],      # 无操作
            1: [-1.0, 0.0, 0.0],      # 左转
            2: [-0.5, 0.0, 0.0],
            3: [-0.25, 0.0, 0.0],
            4: [-0.1, 0.0, 0.0],
            5: [0.1, 0.0, 0.0],       # 右转
            6: [0.25, 0.0, 0.0],
            7: [0.5, 0.0, 0.0],
            8: [1.0, 0.0, 0.0],
            9: [0.0, 0.25, 0.0],      # 前进
            10: [0.0, 0.5, 0.0],
            11: [0.0, 0.75, 0.0],
            12: [0.0, 1.0, 0.0],
            13: [-0.5, 0.5, 0.0],     # 左转+前进
            14: [0.5, 0.5, 0.0],      # 右转+前进
        }
        return np.array(action_map[discrete_action], dtype=np.float32)
    
    def get_discrete_action_dim(self):
        """获取离散动作空间维度"""
        return 15
    
    def __str__(self):
        return (f"Car Racing Environment\n"
                f"  Observation space: {self.observation_space}\n"
                f"  Action space: {self.action_space}\n"
                f"  Observation shape: {self.get_observation_shape()}\n"
                f"  Action dimension: {self.get_action_dim()}\n"
                f"  Discrete action dimension: {self.get_discrete_action_dim()}")


def preprocess_observation(obs):
    """
    预处理观察（图像）
    
    Args:
        obs: 原始观察 (96x96x3, uint8, 0-255)
        
    Returns:
        processed_obs: 预处理后的观察 (84x84x3, float32, 0-1)
    """
    # 转换为float32并归一化到[0, 1]
    obs = obs.astype(np.float32) / 255.0
    
    # 可选：调整大小到84x84（减少计算量）
    # 这里保持原始96x96，但可以改为84x84
    # from PIL import Image
    # obs = np.array(Image.fromarray((obs * 255).astype(np.uint8)).resize((84, 84))) / 255.0
    
    return obs


def test_environment():
    """测试环境"""
    print("🧪 Testing Car Racing Environment...")
    
    # 创建环境
    env = CarRacingWrapper()
    print(env)
    
    # 测试随机策略
    print("\n🎮 Testing random policy for 2 episodes:")
    for episode in range(2):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"  Episode {episode + 1}:")
        print(f"    Initial state shape: {state.shape}")
        print(f"    State dtype: {state.dtype}")
        print(f"    State range: [{state.min():.2f}, {state.max():.2f}]")
        
        while not done and steps < 100:  # 限制步数用于测试
            # 随机选择离散动作
            discrete_action = np.random.randint(0, env.get_discrete_action_dim())
            continuous_action = env.discretize_action(discrete_action)
            
            state, reward, terminated, truncated, info = env.step(continuous_action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        print(f"    Steps = {steps}, Total Reward = {total_reward:.2f}")
    
    env.close()
    print("\n✅ Environment test completed!")


if __name__ == "__main__":
    test_environment()

