#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CartPole Environment Wrapper
CartPole 环境封装
"""

import gymnasium as gym
import numpy as np


class CartPoleWrapper:
    """
    CartPole 环境封装类
    
    CartPole-v1 是一个经典的强化学习控制问题：
    - 观察空间：4维连续空间 [位置, 速度, 角度, 角速度]
    - 动作空间：2个离散动作 [向左推, 向右推]
    - 目标：保持杆子竖直，不让它倒下
    """
    
    def __init__(self, render_mode=None):
        """
        初始化 CartPole 环境
        
        Args:
            render_mode: 渲染模式 ('human' 用于可视化, None 用于训练)
        """
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self, seed=None):
        """重置环境到初始状态"""
        if seed is not None:
            return self.env.reset(seed=seed)
        return self.env.reset()
    
    def step(self, action):
        """
        执行动作
        
        Args:
            action: 动作 (0: 向左推, 1: 向右推)
            
        Returns:
            observation: 观察 [位置, 速度, 角度, 角速度]
            reward: 奖励（每存活一步获得1分）
            terminated: 是否终止（杆子倾斜过大或小车移出边界）
            truncated: 是否截断（达到最大步数500）
            info: 额外信息
        """
        return self.env.step(action)
    
    def render(self):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    def get_state_dim(self):
        """获取状态空间维度"""
        return self.observation_space.shape[0]
    
    def get_action_dim(self):
        """获取动作空间维度"""
        return self.action_space.n
    
    def __str__(self):
        return (f"CartPole Environment\n"
                f"  State space: {self.observation_space}\n"
                f"  Action space: {self.action_space}\n"
                f"  State dimension: {self.get_state_dim()}\n"
                f"  Action dimension: {self.get_action_dim()}")


def test_environment():
    """测试环境"""
    print("🧪 Testing CartPole Environment...")
    
    # 创建环境
    env = CartPoleWrapper()
    print(env)
    
    # 测试随机策略
    print("\n🎮 Testing random policy for 5 episodes:")
    for episode in range(5):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # 随机选择动作
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        print(f"  Episode {episode + 1}: Steps = {steps}, Total Reward = {total_reward}")
    
    env.close()
    print("\n✅ Environment test completed!")


if __name__ == "__main__":
    test_environment()

