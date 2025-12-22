#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioral Cloning for Imitation Learning
行为克隆模仿学习

这个模块实现了行为克隆算法，通过监督学习从专家演示中学习策略。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
from typing import List, Dict, Tuple, Optional

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highway.highway_env import HighwayWrapper


class TrajectoryDataset(Dataset):
    """
    轨迹数据集类

    用于将专家轨迹转换为 PyTorch 数据集格式
    """

    def __init__(self, trajectories: List[Dict], env_wrapper: HighwayWrapper):
        """
        初始化数据集

        Args:
            trajectories: 专家轨迹列表
            env_wrapper: 环境封装器，用于处理观察数据
        """
        self.trajectories = trajectories
        self.env_wrapper = env_wrapper
        self.data = []

        # 处理轨迹数据
        for trajectory in trajectories:
            states = trajectory['states']
            actions = trajectory['actions']

            for state, action in zip(states, actions):
                # 展平状态
                state_flat = self.env_wrapper.flatten_observation(state)
                self.data.append((state_flat, action))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action = self.data[idx]
        return torch.FloatTensor(state), torch.LongTensor([action])


class BCPolicy(nn.Module):
    """
    行为克隆策略网络

    使用多层感知机将状态映射到动作
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        """
        初始化策略网络

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表
        """
        super(BCPolicy, self).__init__()

        # 构建网络层
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state):
        """
        前向传播

        Args:
            state: 状态张量

        Returns:
            logits: 动作 logits
        """
        return self.network(state)

    def get_action(self, state, deterministic=True):
        """
        获取动作

        Args:
            state: 状态
            deterministic: 是否使用确定性策略

        Returns:
            action: 选择的动作
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0)

            logits = self.forward(state)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

        return action

    def get_action_probs(self, state):
        """
        获取动作概率

        Args:
            state: 状态

        Returns:
            probs: 动作概率分布
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0)

            logits = self.forward(state)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        return probs.numpy()


class BehavioralCloning:
    """
    行为克隆算法实现
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128],
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        """
        初始化行为克隆

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 创建策略网络
        self.policy = BCPolicy(state_dim, action_dim, hidden_dims)

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.policy.parameters(),
                                   lr=learning_rate,
                                   weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        # 训练历史
        self.train_history = {
            'loss': [],
            'accuracy': []
        }

    def train(self, trajectories: List[Dict], env_wrapper: HighwayWrapper,
              batch_size: int = 64, epochs: int = 50, validation_split: float = 0.2):
        """
        训练行为克隆模型

        Args:
            trajectories: 专家轨迹
            env_wrapper: 环境封装器
            batch_size: 批次大小
            epochs: 训练轮数
            validation_split: 验证集比例
        """
        # 创建数据集
        dataset = TrajectoryDataset(trajectories, env_wrapper)

        # 划分训练集和验证集
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"📚 Dataset: {len(dataset)} samples")
        print(f"   Train: {train_size}, Validation: {val_size}")

        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            train_loss, train_acc = self._train_epoch(train_loader)

            # 验证阶段
            val_loss, val_acc = self._validate_epoch(val_loader)

            # 记录历史
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")

    def _train_epoch(self, data_loader):
        """训练一个epoch"""
        self.policy.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for states, actions in data_loader:
            self.optimizer.zero_grad()

            # 前向传播
            logits = self.policy(states)
            loss = self.criterion(logits, actions.squeeze())

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == actions.squeeze()).sum().item()
            total_samples += len(states)

        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def _validate_epoch(self, data_loader):
        """验证一个epoch"""
        self.policy.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for states, actions in data_loader:
                logits = self.policy(states)
                loss = self.criterion(logits, actions.squeeze())

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == actions.squeeze()).sum().item()
                total_samples += len(states)

        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def evaluate(self, env_name: str = 'highway-v0', num_episodes: int = 10,
                max_steps: int = 200, render: bool = False):
        """
        评估学习到的策略

        Args:
            env_name: 环境名称
            num_episodes: 评估回合数
            max_steps: 每回合最大步数
            render: 是否渲染

        Returns:
            eval_results: 评估结果字典
        """
        env = HighwayWrapper(env_name, render_mode='human' if render else None)

        episode_rewards = []
        episode_lengths = []
        success_count = 0

        print("🔍 Evaluating learned policy...")

        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done and steps < max_steps:
                # 使用学习到的策略选择动作
                state_flat = env.flatten_observation(state)
                action = self.policy.get_action(state_flat)

                if render:
                    env.render()

                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                steps += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

            # 定义成功标准（无碰撞且达到一定步数）
            success = steps >= max_steps * 0.8 and not terminated  # 假设 terminated 表示碰撞
            if success:
                success_count += 1

            if episode % 5 == 0:
                print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

        env.close()

        eval_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': success_count / num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }

        print("✅ Evaluation completed!")
        print(".2f")
        print(".2f")
        return eval_results

    def save_model(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'train_history': self.train_history
        }, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path: str):
        """
        加载模型

        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', {'loss': [], 'accuracy': []})
        print(f"📂 Model loaded from {path}")


def compare_with_expert(expert_trajectories: List[Dict], bc_results: Dict):
    """
    与专家策略比较性能

    Args:
        expert_trajectories: 专家轨迹
        bc_results: BC 评估结果
    """
    expert_rewards = [traj['total_reward'] for traj in expert_trajectories]
    expert_lengths = [traj['episode_length'] for traj in expert_trajectories]

    print("🏆 Performance Comparison:")
    print("Expert Policy:")
    print(".2f")
    print(".1f")
    print("Behavioral Cloning:")
    print(".2f")
    print(".1f")

    # 计算性能差距
    reward_gap = np.mean(expert_rewards) - bc_results['mean_reward']
    length_gap = np.mean(expert_lengths) - bc_results['mean_length']

    print("Performance Gap:")
    print(".2f")
    print(".1f")
    return reward_gap, length_gap
