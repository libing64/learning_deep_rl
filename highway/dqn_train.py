#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN Training Script for Highway
基于DQN的Highway训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from highway_env import HighwayWrapper


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) 神经网络
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化DQN网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度 (Highway通常为5)
            hidden_dim: 隐藏层维度
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """
    经验回放缓冲区
    """
    
    def __init__(self, capacity=50000):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        从缓冲区随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            states, actions, rewards, next_states, dones
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)


class DQNAgent:
    """
    DQN智能体
    """
    
    def __init__(self, state_dim, action_dim, lr=0.0005, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=50000, batch_size=64, target_update=100):
        """
        初始化DQN智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr: 学习率
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减
            memory_size: 经验回放缓冲区大小
            batch_size: 批次大小
            target_update: 目标网络更新频率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # 创建主网络和目标网络
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(memory_size)
        
        # 训练历史
        self.episode_rewards = []
        self.episode_lengths = []
    
    def select_action(self, state, training=True):
        """
        使用epsilon-greedy策略选择动作
        
        Args:
            state: 当前状态
            training: 是否在训练模式（影响探索）
            
        Returns:
            action: 选择的动作
        """
        if training and random.random() < self.epsilon:
            # 随机探索
            return random.randrange(self.action_dim)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        执行一步训练 - DQN算法的核心训练步骤
        
        Returns:
            loss: 损失值（如果训练成功），否则返回None
        """
        # 检查缓冲区是否有足够的样本
        if len(self.memory) < self.batch_size:
            return None
        
        # 从经验回放缓冲区随机采样一批经验
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 使用主网络计算当前状态-动作对的Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 使用目标网络计算目标Q值（Bellman方程）
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播和参数更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 定期更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        print(f"✅ Model loaded from {filepath}")


def train_dqn(env, agent, num_episodes=500, max_steps=200, save_interval=50):
    """
    训练DQN智能体
    
    Args:
        env: Highway环境
        agent: DQN智能体
        num_episodes: 训练回合数
        max_steps: 每回合最大步数
        save_interval: 保存模型的间隔
    """
    print("🚀 Starting DQN Training for Highway...")
    print(f"   Episodes: {num_episodes}")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   Initial epsilon: {agent.epsilon:.3f}")
    print("-" * 60)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        # 展平观察
        state = env.flatten_observation(state)
        
        total_reward = 0
        steps = 0
        episode_losses = []
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 展平下一个观察
            next_state = env.flatten_observation(next_state)
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 训练
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # 记录历史
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        
        # 打印进度
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_reward = np.mean(agent.episode_rewards[-10:]) if len(agent.episode_rewards) >= 10 else total_reward
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Steps: {steps:3d} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Avg Reward (10): {avg_reward:7.2f} | "
                  f"Loss: {avg_loss:.4f}")
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            agent.save(f"dqn_model_episode_{episode + 1}.pth")
    
    print("-" * 60)
    print("✅ Training completed!")
    
    # 保存最终模型
    agent.save("dqn_model_final.pth")
    
    return agent


def plot_training_history(agent, save_path="training_history.png"):
    """
    绘制训练历史
    
    Args:
        agent: 训练好的智能体
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制奖励曲线
    ax1.plot(agent.episode_rewards, alpha=0.6, label='Episode Reward')
    if len(agent.episode_rewards) >= 10:
        window = 10
        moving_avg = np.convolve(agent.episode_rewards, 
                                 np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(agent.episode_rewards)), 
                moving_avg, 'r-', label=f'Moving Average ({window})', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制步数曲线
    ax2.plot(agent.episode_lengths, alpha=0.6, label='Episode Length')
    if len(agent.episode_lengths) >= 10:
        window = 10
        moving_avg = np.convolve(agent.episode_lengths, 
                                 np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(agent.episode_lengths)), 
                moving_avg, 'r-', label=f'Moving Average ({window})', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Training history saved to {save_path}")
    plt.close()


def test_agent(env, agent, num_episodes=5, render=False):
    """
    测试训练好的智能体
    
    Args:
        env: Highway环境
        agent: 训练好的智能体
        num_episodes: 测试回合数
        render: 是否渲染
    """
    print(f"\n🧪 Testing agent for {num_episodes} episodes...")
    
    test_rewards = []
    test_lengths = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = env.flatten_observation(state)
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # 使用训练好的策略（不探索）
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = env.flatten_observation(state)
            
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        print(f"  Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    avg_reward = np.mean(test_rewards)
    avg_length = np.mean(test_lengths)
    print(f"\n📊 Test Results:")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Average Steps: {avg_length:.2f}")
    print(f"   Max Reward: {max(test_rewards):.2f}")
    print(f"   Min Reward: {min(test_rewards):.2f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train DQN agent for highway driving',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--env', '-e',
        type=str,
        default='highway-v0',
        help='Environment name (default: highway-v0)'
    )
    
    parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=500,
        help='Number of training episodes (default: 500)'
    )
    
    parser.add_argument(
        '--max-steps', '-s',
        type=int,
        default=200,
        help='Maximum steps per episode (default: 200)'
    )
    
    args = parser.parse_args()
    
    # 创建环境
    print("🎮 Creating Highway environment...")
    try:
        env = HighwayWrapper(args.env, render_mode=None)
        print(env)
        print()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Please install highway-env: pip install highway-env")
        return
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return
    
    # 获取环境信息
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    # 创建DQN智能体
    print("🤖 Creating DQN agent...")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=50000,
        batch_size=64,
        target_update=100
    )
    print("✅ Agent created!")
    print()
    
    # 训练
    agent = train_dqn(env, agent, num_episodes=args.episodes, max_steps=args.max_steps, save_interval=50)
    
    # 绘制训练历史
    plot_training_history(agent)
    
    # 测试
    test_agent(env, agent, num_episodes=10, render=False)
    
    # 关闭环境
    env.close()
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()

