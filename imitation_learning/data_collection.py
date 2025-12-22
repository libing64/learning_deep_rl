#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Collection for Imitation Learning
模仿学习数据收集

这个模块负责生成和处理专家演示数据，包括：
- 生成专家轨迹
- 数据预处理和验证
- 数据保存和加载
- 数据集统计分析
"""

import numpy as np
import pickle
import json
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import argparse

from expert_policy import generate_expert_trajectories, HighwayExpert
from highway.highway_env import HighwayWrapper


class ImitationDataset:
    """
    模仿学习数据集管理类
    """

    def __init__(self, data_dir: str = "data"):
        """
        初始化数据集管理器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # 数据文件路径
        self.trajectories_file = self.data_dir / "trajectories.pkl"
        self.metadata_file = self.data_dir / "metadata.json"
        self.stats_file = self.data_dir / "statistics.json"

    def collect_expert_data(self, env_name: str = 'highway-v0',
                           num_episodes: int = 1000,
                           max_steps: int = 200,
                           save_frequency: int = 100):
        """
        收集专家演示数据

        Args:
            env_name: 环境名称
            num_episodes: 收集的回合数
            max_steps: 每回合最大步数
            save_frequency: 保存频率
        """
        print(f"🎯 Collecting {num_episodes} expert trajectories from {env_name}")

        all_trajectories = []
        total_samples = 0

        # 分批收集数据
        batch_size = save_frequency
        for start_episode in range(0, num_episodes, batch_size):
            end_episode = min(start_episode + batch_size, num_episodes)
            current_batch = end_episode - start_episode

            print(f"\n📊 Collecting episodes {start_episode + 1}-{end_episode}")

            # 生成当前批次的轨迹
            trajectories = generate_expert_trajectories(
                env_name=env_name,
                num_episodes=current_batch,
                max_steps=max_steps
            )

            all_trajectories.extend(trajectories)
            total_samples += sum(len(traj['states']) for traj in trajectories)

            # 中间保存
            if len(all_trajectories) % save_frequency == 0 or end_episode == num_episodes:
                self._save_trajectories(all_trajectories, env_name)
                print(f"💾 Saved {len(all_trajectories)} trajectories ({total_samples} samples)")

        # 保存最终数据和统计信息
        self._save_trajectories(all_trajectories, env_name)
        self._compute_and_save_statistics(all_trajectories, env_name)

        print("✅ Data collection completed!")
        print(f"📈 Total trajectories: {len(all_trajectories)}")
        print(f"📈 Total samples: {total_samples}")

        return all_trajectories

    def _save_trajectories(self, trajectories: List[Dict], env_name: str):
        """
        保存轨迹数据

        Args:
            trajectories: 轨迹列表
            env_name: 环境名称
        """
        # 保存轨迹数据
        with open(self.trajectories_file, 'wb') as f:
            pickle.dump({
                'trajectories': trajectories,
                'env_name': env_name,
                'num_trajectories': len(trajectories),
                'total_samples': sum(len(traj['states']) for traj in trajectories)
            }, f)

        # 保存元数据
        metadata = {
            'env_name': env_name,
            'num_trajectories': len(trajectories),
            'total_samples': sum(len(traj['states']) for traj in trajectories),
            'avg_episode_length': np.mean([len(traj['states']) for traj in trajectories]),
            'avg_episode_reward': np.mean([traj['total_reward'] for traj in trajectories]),
            'max_episode_reward': max(traj['total_reward'] for traj in trajectories),
            'min_episode_reward': min(traj['total_reward'] for traj in trajectories),
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _compute_and_save_statistics(self, trajectories: List[Dict], env_name: str):
        """
        计算并保存数据集统计信息

        Args:
            trajectories: 轨迹列表
            env_name: 环境名称
        """
        # 基本统计
        episode_lengths = [len(traj['states']) for traj in trajectories]
        episode_rewards = [traj['total_reward'] for traj in trajectories]

        # 动作分布统计
        all_actions = []
        for traj in trajectories:
            all_actions.extend(traj['actions'])
        action_counts = np.bincount(all_actions, minlength=5)  # 假设有5个动作

        # 状态统计
        all_states = []
        for traj in trajectories:
            for state in traj['states']:
                if isinstance(state, dict):
                    # 转换为数组
                    env_wrapper = HighwayWrapper(env_name)
                    state_flat = env_wrapper.flatten_observation(state)
                    all_states.append(state_flat)
                else:
                    all_states.append(state)

        all_states = np.array(all_states)

        statistics = {
            'basic_stats': {
                'num_trajectories': len(trajectories),
                'total_samples': len(all_states),
                'avg_episode_length': float(np.mean(episode_lengths)),
                'std_episode_length': float(np.std(episode_lengths)),
                'avg_episode_reward': float(np.mean(episode_rewards)),
                'std_episode_reward': float(np.std(episode_rewards)),
                'max_episode_reward': float(np.max(episode_rewards)),
                'min_episode_reward': float(np.min(episode_rewards)),
            },
            'action_distribution': {
                'action_counts': action_counts.tolist(),
                'action_probabilities': (action_counts / len(all_actions)).tolist(),
                'most_common_action': int(np.argmax(action_counts)),
            },
            'state_stats': {
                'state_dim': all_states.shape[1],
                'state_mean': all_states.mean(axis=0).tolist(),
                'state_std': all_states.std(axis=0).tolist(),
                'state_min': all_states.min(axis=0).tolist(),
                'state_max': all_states.max(axis=0).tolist(),
            },
            'episode_length_distribution': {
                'percentiles': {
                    '25th': float(np.percentile(episode_lengths, 25)),
                    '50th': float(np.percentile(episode_lengths, 50)),
                    '75th': float(np.percentile(episode_lengths, 75)),
                    '90th': float(np.percentile(episode_lengths, 90)),
                    '95th': float(np.percentile(episode_lengths, 95)),
                }
            }
        }

        with open(self.stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)

        print("📊 Dataset statistics computed and saved")

    def load_data(self) -> Tuple[List[Dict], Dict]:
        """
        加载数据集

        Returns:
            trajectories: 轨迹列表
            metadata: 元数据字典
        """
        if not self.trajectories_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.trajectories_file}")

        # 加载轨迹数据
        with open(self.trajectories_file, 'rb') as f:
            data = pickle.load(f)
            trajectories = data['trajectories']
            metadata = data

        # 如果元数据文件存在，加载更详细的元数据
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                detailed_metadata = json.load(f)
                metadata.update(detailed_metadata)

        print(f"📂 Loaded {len(trajectories)} trajectories with {metadata['total_samples']} samples")

        return trajectories, metadata

    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息

        Returns:
            statistics: 统计信息字典
        """
        if not self.stats_file.exists():
            raise FileNotFoundError(f"Statistics file not found: {self.stats_file}")

        with open(self.stats_file, 'r') as f:
            return json.load(f)

    def validate_data(self) -> bool:
        """
        验证数据集完整性和正确性

        Returns:
            is_valid: 数据集是否有效
        """
        try:
            trajectories, metadata = self.load_data()
            stats = self.get_statistics()

            # 基本验证
            assert len(trajectories) == metadata['num_trajectories']
            assert stats['basic_stats']['num_trajectories'] == len(trajectories)

            # 轨迹验证
            for i, traj in enumerate(trajectories):
                assert 'states' in traj
                assert 'actions' in traj
                assert 'rewards' in traj
                assert len(traj['states']) == len(traj['actions']) == len(traj['rewards'])
                assert len(traj['states']) == traj['episode_length']

            print("✅ Dataset validation passed")
            return True

        except Exception as e:
            print(f"❌ Dataset validation failed: {e}")
            return False

    def split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.2) \
            -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        将数据集分割为训练集、验证集和测试集

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例

        Returns:
            train_trajectories, val_trajectories, test_trajectories: 分割后的轨迹
        """
        trajectories, _ = self.load_data()

        # 打乱数据
        np.random.shuffle(trajectories)

        n_total = len(trajectories)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_trajectories = trajectories[:n_train]
        val_trajectories = trajectories[n_train:n_train + n_val]
        test_trajectories = trajectories[n_train + n_val:]

        print(f"📊 Data split: Train {len(train_trajectories)}, "
              f"Val {len(val_trajectories)}, Test {len(test_trajectories)}")

        return train_trajectories, val_trajectories, test_trajectories

    def sample_batch(self, batch_size: int, trajectories: Optional[List[Dict]] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        从轨迹中采样批次数据

        Args:
            batch_size: 批次大小
            trajectories: 轨迹列表（如果为None，使用全部数据）

        Returns:
            states: 状态批次
            actions: 动作批次
        """
        if trajectories is None:
            trajectories, _ = self.load_data()

        # 展平所有状态-动作对
        all_states = []
        all_actions = []

        for traj in trajectories:
            # 创建环境包装器来处理状态
            env_wrapper = HighwayWrapper('highway-v0')  # 默认环境

            for state, action in zip(traj['states'], traj['actions']):
                state_flat = env_wrapper.flatten_observation(state)
                all_states.append(state_flat)
                all_actions.append(action)

        # 转换为numpy数组
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)

        # 随机采样
        indices = np.random.choice(len(all_states), size=batch_size, replace=False)

        return all_states[indices], all_actions[indices]


def create_balanced_dataset(data_dir: str = "data", target_samples_per_action: int = 1000):
    """
    创建动作平衡的数据集

    Args:
        data_dir: 数据目录
        target_samples_per_action: 每个动作的目标样本数
    """
    dataset = ImitationDataset(data_dir)
    trajectories, metadata = dataset.load_data()

    # 统计当前动作分布
    action_counts = np.zeros(5)  # 假设5个动作
    for traj in trajectories:
        counts = np.bincount(traj['actions'], minlength=5)
        action_counts += counts

    print(f"Current action distribution: {action_counts}")

    # 找出需要额外采样的动作
    min_count = np.min(action_counts)
    if min_count >= target_samples_per_action:
        print("Dataset already balanced")
        return

    # 为每个动作生成额外数据
    additional_trajectories = []
    expert = HighwayExpert('highway-v0')

    for action in range(5):
        needed = target_samples_per_action - int(action_counts[action])
        if needed <= 0:
            continue

        print(f"Generating {needed} additional samples for action {action}")

        # 生成偏向特定动作的轨迹
        action_specific_trajectories = generate_action_specific_trajectories(
            expert, action, needed
        )
        additional_trajectories.extend(action_specific_trajectories)

    # 合并并保存
    balanced_trajectories = trajectories + additional_trajectories
    dataset._save_trajectories(balanced_trajectories, 'highway-v0')
    dataset._compute_and_save_statistics(balanced_trajectories, 'highway-v0')

    print(f"✅ Balanced dataset created with {len(balanced_trajectories)} trajectories")


def generate_action_specific_trajectories(expert: HighwayExpert, target_action: int,
                                        num_samples: int) -> List[Dict]:
    """
    生成偏向特定动作的轨迹

    Args:
        expert: 专家策略
        target_action: 目标动作
        num_samples: 需要的样本数

    Returns:
        trajectories: 生成的轨迹列表
    """
    trajectories = []
    collected_samples = 0

    while collected_samples < num_samples:
        # 重置环境
        state, info = expert.env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []

        done = False
        steps = 0
        max_steps = 100  # 限制episode长度

        while not done and steps < max_steps:
            episode_states.append(state.copy() if hasattr(state, 'copy') else state)

            # 有一定概率使用目标动作
            if np.random.random() < 0.7:  # 70%概率使用目标动作
                action = target_action
            else:
                action = expert.get_action(state)

            episode_actions.append(action)

            state, reward, terminated, truncated, info = expert.env.step(action)
            done = terminated or truncated
            episode_rewards.append(reward)
            steps += 1

        if len(episode_states) > 0:
            trajectories.append({
                'states': np.array(episode_states),
                'actions': np.array(episode_actions),
                'rewards': np.array(episode_rewards),
                'episode_length': len(episode_states),
                'total_reward': sum(episode_rewards)
            })
            collected_samples += len(episode_states)

    return trajectories


def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(description="Imitation Learning Data Collection")
    parser.add_argument('--env', type=str, default='highway-v0',
                       help='Environment name')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of episodes to collect')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum steps per episode')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing dataset')
    parser.add_argument('--balance', action='store_true',
                       help='Create balanced dataset')

    args = parser.parse_args()

    dataset = ImitationDataset(args.data_dir)

    if args.validate:
        # 验证数据集
        is_valid = dataset.validate_data()
        if is_valid:
            stats = dataset.get_statistics()
            print("📊 Dataset Statistics:")
            print(f"  Trajectories: {stats['basic_stats']['num_trajectories']}")
            print(f"  Total samples: {stats['basic_stats']['total_samples']}")
            print(f"  Avg episode length: {stats['basic_stats']['avg_episode_length']:.1f}")
            print(f"  Avg episode reward: {stats['basic_stats']['avg_episode_reward']:.2f}")
            print(f"  Action distribution: {stats['action_distribution']['action_probabilities']}")

    elif args.balance:
        # 创建平衡数据集
        create_balanced_dataset(args.data_dir)

    else:
        # 收集数据
        trajectories = dataset.collect_expert_data(
            env_name=args.env,
            num_episodes=args.episodes,
            max_steps=args.max_steps
        )

        # 显示统计信息
        stats = dataset.get_statistics()
        print("\n📊 Collection Summary:")
        print(f"  Total trajectories: {stats['basic_stats']['num_trajectories']}")
        print(f"  Total samples: {stats['basic_stats']['total_samples']}")
        print(f"  Average episode length: {stats['basic_stats']['avg_episode_length']:.1f}")
        print(f"  Average episode reward: {stats['basic_stats']['avg_episode_reward']:.2f}")


if __name__ == "__main__":
    main()
