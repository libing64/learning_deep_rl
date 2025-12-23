#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAgger Algorithm: Imitation Learning + Reinforcement Learning
DAgger算法：模仿学习 + 强化学习

DAgger (Dataset Aggregation) 是一种结合模仿学习和强化学习的算法：
1. 使用专家数据训练初始策略（IL阶段）
2. 使用当前策略在环境中运行，收集新轨迹（RL阶段）
3. 让专家对新轨迹中的状态进行标注
4. 将新数据加入数据集，重新训练策略
5. 重复步骤2-4，直到策略性能满足要求
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import deque
from typing import List, Dict, Tuple, Optional
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highway.highway_env import HighwayWrapper
from imitation_learning.behavioral_cloning import BCPolicy, TrajectoryDataset
from imitation_learning.expert_policy import HighwayExpert


class DAggerAgent:
    """
    DAgger 算法智能体
    
    结合模仿学习和强化学习，通过迭代改进策略
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 lr: float = 0.001,
                 hidden_dims: List[int] = [256, 256, 128],
                 device: str = 'cpu'):
        """
        初始化 DAgger 智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr: 学习率
            hidden_dims: 隐藏层维度列表
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 创建策略网络（使用BC策略网络结构）
        self.policy = BCPolicy(state_dim, action_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'expert_agreement': []
        }
        
        # 数据集（累积所有收集的数据）
        self.all_trajectories = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            training: 是否在训练模式
            
        Returns:
            action: 选择的动作
        """
        self.policy.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.policy(state_tensor)
            action = action_probs.argmax(dim=1).item()
        return action
    
    def train_on_dataset(self, 
                        trajectories: List[Dict],
                        env_wrapper: HighwayWrapper,
                        epochs: int = 10,
                        batch_size: int = 64,
                        validation_split: float = 0.2):
        """
        在数据集上训练策略
        
        Args:
            trajectories: 轨迹列表
            env_wrapper: 环境封装器
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例
        """
        # 创建数据集
        dataset = TrajectoryDataset(trajectories, env_wrapper)
        
        # 划分训练集和验证集
        dataset_size = len(dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.policy.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for states, actions in train_loader:
                states = states.to(self.device)
                actions = actions.squeeze().to(self.device)
                
                # 前向传播
                action_probs = self.policy(states)
                loss = F.cross_entropy(action_probs, actions)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = action_probs.max(1)
                train_total += actions.size(0)
                train_correct += predicted.eq(actions).sum().item()
            
            # 验证阶段
            self.policy.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for states, actions in val_loader:
                    states = states.to(self.device)
                    actions = actions.squeeze().to(self.device)
                    
                    action_probs = self.policy(states)
                    loss = F.cross_entropy(action_probs, actions)
                    
                    val_loss += loss.item()
                    _, predicted = action_probs.max(1)
                    val_total += actions.size(0)
                    val_correct += predicted.eq(actions).sum().item()
            
            # 记录历史
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            
            self.train_history['loss'].append({
                'train': avg_train_loss,
                'val': avg_val_loss,
                'epoch': epoch
            })
            self.train_history['accuracy'].append({
                'train': train_acc,
                'val': val_acc,
                'epoch': epoch
            })
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {avg_val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2f}%")
    
    def collect_trajectory_with_policy(self,
                                      env: HighwayWrapper,
                                      expert: HighwayExpert,
                                      max_steps: int = 200,
                                      beta: float = 0.5) -> Dict:
        """
        使用当前策略收集轨迹，并用专家标注
        
        Args:
            env: 环境
            expert: 专家策略（用于标注）
            max_steps: 最大步数
            beta: 专家动作混合比例（beta=1.0时完全使用专家，beta=0.0时完全使用策略）
            
        Returns:
            trajectory: 轨迹字典
        """
        state, info = env.reset()
        state = env.flatten_observation(state)
        
        trajectory = {
            'states': [],
            'actions': [],
            'expert_actions': [],
            'rewards': []
        }
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 使用当前策略选择动作
            policy_action = self.select_action(state, training=True)
            
            # 获取专家动作
            expert_action = expert.get_action(state)
            
            # 混合动作（beta控制专家参与度）
            if np.random.random() < beta:
                action = expert_action
            else:
                action = policy_action
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = env.flatten_observation(next_state)
            
            # 记录轨迹（使用专家动作作为标签）
            trajectory['states'].append(state.copy())
            trajectory['actions'].append(expert_action)  # 使用专家动作作为标签
            trajectory['expert_actions'].append(expert_action)
            trajectory['rewards'].append(reward)
            
            state = next_state
            step += 1
        
        return trajectory
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', {'loss': [], 'accuracy': []})
        print(f"✅ Model loaded from {filepath}")


def run_dagger_iteration(agent: DAggerAgent,
                         env: HighwayWrapper,
                         expert: HighwayExpert,
                         num_trajectories: int,
                         max_steps: int = 200,
                         beta: float = 0.5,
                         epochs: int = 10,
                         batch_size: int = 64) -> List[Dict]:
    """
    运行一次 DAgger 迭代
    
    Args:
        agent: DAgger 智能体
        env: 环境
        expert: 专家策略
        num_trajectories: 收集的轨迹数量
        max_steps: 每轨迹最大步数
        beta: 专家动作混合比例
        epochs: 训练轮数
        batch_size: 批次大小
        
    Returns:
        new_trajectories: 新收集的轨迹列表
    """
    print(f"📊 Collecting {num_trajectories} trajectories with current policy...")
    
    # 使用当前策略收集轨迹
    new_trajectories = []
    for i in range(num_trajectories):
        trajectory = agent.collect_trajectory_with_policy(
            env, expert, max_steps=max_steps, beta=beta
        )
        new_trajectories.append(trajectory)
        
        if (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/{num_trajectories} trajectories")
    
    # 将新轨迹加入总数据集
    agent.all_trajectories.extend(new_trajectories)
    
    # 在累积数据集上训练
    print(f"🎓 Training on {len(agent.all_trajectories)} total trajectories...")
    agent.train_on_dataset(
        agent.all_trajectories,
        env,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return new_trajectories


def dagger_algorithm(env_name: str = 'highway-v0',
                     initial_expert_trajectories: int = 100,
                     dagger_iterations: int = 5,
                     trajectories_per_iteration: int = 50,
                     max_steps: int = 200,
                     beta_schedule: Optional[List[float]] = None,
                     training_epochs: int = 10,
                     batch_size: int = 64,
                     device: str = 'cpu') -> DAggerAgent:
    """
    运行完整的 DAgger 算法
    
    Args:
        env_name: 环境名称
        initial_expert_trajectories: 初始专家轨迹数量
        dagger_iterations: DAgger迭代次数
        trajectories_per_iteration: 每次迭代收集的轨迹数
        max_steps: 每轨迹最大步数
        beta_schedule: beta值调度（如果为None，使用线性衰减）
        training_epochs: 每次迭代的训练轮数
        batch_size: 批次大小
        device: 计算设备
        
    Returns:
        agent: 训练好的DAgger智能体
    """
    print("=" * 70)
    print("🚀 DAgger Algorithm: Imitation Learning + Reinforcement Learning")
    print("=" * 70)
    
    # 创建环境
    env = HighwayWrapper(env_name, render_mode=None)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    print(f"\n📋 Environment: {env_name}")
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")
    
    # 创建专家策略
    expert = HighwayExpert(env)
    
    # 创建DAgger智能体
    agent = DAggerAgent(state_dim, action_dim, device=device)
    
    # Beta调度（线性衰减：从1.0到0.0）
    if beta_schedule is None:
        beta_schedule = np.linspace(1.0, 0.0, dagger_iterations + 1).tolist()
    
    # 阶段1: 初始专家数据收集和训练
    print(f"\n{'='*70}")
    print("📚 Phase 1: Initial Expert Data Collection")
    print(f"{'='*70}")
    
    from imitation_learning.expert_policy import generate_expert_trajectories
    
    initial_trajectories = generate_expert_trajectories(
        env_name=env_name,
        num_episodes=initial_expert_trajectories,
        max_steps=max_steps
    )
    
    agent.all_trajectories = initial_trajectories
    
    print(f"✅ Collected {len(initial_trajectories)} initial expert trajectories")
    print(f"🎓 Training initial policy...")
    agent.train_on_dataset(
        initial_trajectories,
        env,
        epochs=training_epochs,
        batch_size=batch_size
    )
    
    # 阶段2: DAgger迭代
    print(f"\n{'='*70}")
    print("🔄 Phase 2: DAgger Iterations")
    print(f"{'='*70}")
    
    for iteration in range(dagger_iterations):
        print(f"\n--- DAgger Iteration {iteration + 1}/{dagger_iterations} ---")
        print(f"Beta (expert mixing ratio): {beta_schedule[iteration]:.2f}")
        
        # 运行一次DAgger迭代
        new_trajectories = run_dagger_iteration(
            agent=agent,
            env=env,
            expert=expert,
            num_trajectories=trajectories_per_iteration,
            max_steps=max_steps,
            beta=beta_schedule[iteration],
            epochs=training_epochs,
            batch_size=batch_size
        )
        
        print(f"✅ Iteration {iteration + 1} completed")
        print(f"   New trajectories: {len(new_trajectories)}")
        print(f"   Total trajectories: {len(agent.all_trajectories)}")
    
    env.close()
    
    print(f"\n{'='*70}")
    print("🎉 DAgger Algorithm Completed!")
    print(f"{'='*70}")
    print(f"📊 Final Statistics:")
    print(f"   Total trajectories: {len(agent.all_trajectories)}")
    print(f"   Total training samples: {sum(len(t['states']) for t in agent.all_trajectories)}")
    
    return agent

