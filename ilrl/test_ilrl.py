#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for DAgger (IL+RL) Model
DAgger模型测试脚本
"""

import argparse
import json
import glob
from pathlib import Path
import torch
import numpy as np
import time
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highway.highway_env import HighwayWrapper
from imitation_learning.expert_policy import HighwayExpert
from dagger_algorithm import DAggerAgent


def load_model_and_config(model_path: str):
    """
    加载模型和配置
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        agent: 加载的智能体
        config: 配置字典
    """
    model_path = Path(model_path)
    
    # 加载配置
    config_path = model_path.parent / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 加载模型
    checkpoint = torch.load(model_path, weights_only=False)
    state_dim = checkpoint.get('state_dim', 35)  # 默认值
    action_dim = checkpoint.get('action_dim', 5)  # 默认值
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DAggerAgent(state_dim, action_dim, device=device)
    agent.load(str(model_path))
    
    return agent, config


def evaluate_agent(agent: DAggerAgent,
                   env: HighwayWrapper,
                   num_episodes: int = 10,
                   render: bool = False,
                   delay: float = 0.05):
    """
    评估智能体性能
    
    Args:
        agent: DAgger智能体
        env: 环境
        num_episodes: 评估回合数
        render: 是否渲染
        delay: 渲染延迟
        
    Returns:
        results: 评估结果字典
    """
    print(f"\n🧪 Evaluating agent for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    episode_info = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = env.flatten_observation(state)
        
        total_reward = 0
        steps = 0
        done = False
        actions_taken = []
        
        while not done:
            # 选择动作
            action = agent.select_action(state, training=False)
            actions_taken.append(action)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = env.flatten_observation(next_state)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if render:
                env.render()
                time.sleep(delay)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_info.append({
            'reward': total_reward,
            'length': steps,
            'actions': actions_taken
        })
        
        print(f"  Episode {episode + 1:3d} | "
              f"Reward: {total_reward:7.2f} | "
              f"Steps: {steps:3d}")
    
    # 统计结果
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'episode_info': episode_info
    }
    
    return results


def compare_with_expert(agent: DAggerAgent,
                        env: HighwayWrapper,
                        num_episodes: int = 10):
    """
    与专家策略比较
    
    Args:
        agent: DAgger智能体
        env: 环境
        num_episodes: 比较回合数
        
    Returns:
        comparison: 比较结果字典
    """
    print(f"\n📊 Comparing with expert policy for {num_episodes} episodes...")
    
    # 获取环境名称
    env_name = env.env_name if hasattr(env, 'env_name') else 'highway-v0'
    expert = HighwayExpert(env_name)
    
    # 评估智能体
    agent_results = evaluate_agent(agent, env, num_episodes=num_episodes, render=False)
    
    # 评估专家
    expert_rewards = []
    expert_lengths = []
    
    for episode in range(num_episodes):
        obs_original, info = env.reset()
        state = env.flatten_observation(obs_original)
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = expert.get_action(obs_original)  # 使用原始观察
            obs_original, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = env.flatten_observation(obs_original)
            
            total_reward += reward
            steps += 1
        
        expert_rewards.append(total_reward)
        expert_lengths.append(steps)
    
    comparison = {
        'agent': {
            'mean_reward': agent_results['mean_reward'],
            'std_reward': agent_results['std_reward'],
            'mean_length': agent_results['mean_length']
        },
        'expert': {
            'mean_reward': np.mean(expert_rewards),
            'std_reward': np.std(expert_rewards),
            'mean_length': np.mean(expert_lengths)
        }
    }
    
    # 计算性能比率
    reward_ratio = comparison['agent']['mean_reward'] / comparison['expert']['mean_reward']
    comparison['reward_ratio'] = reward_ratio
    comparison['performance_percentage'] = reward_ratio * 100
    
    return comparison


def print_results(results: dict, comparison: dict = None):
    """
    打印评估结果
    
    Args:
        results: 评估结果
        comparison: 与专家的比较结果
    """
    print("\n" + "=" * 70)
    print("📊 Evaluation Results")
    print("=" * 70)
    
    print(f"\n🎯 Agent Performance:")
    print(f"   Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"   Min Reward: {results['min_reward']:.2f}")
    print(f"   Max Reward: {results['max_reward']:.2f}")
    
    if comparison:
        print(f"\n👨‍🏫 Expert Performance:")
        print(f"   Mean Reward: {comparison['expert']['mean_reward']:.2f} ± {comparison['expert']['std_reward']:.2f}")
        print(f"   Mean Length: {comparison['expert']['mean_length']:.2f}")
        
        print(f"\n📈 Comparison:")
        print(f"   Reward Ratio: {comparison['reward_ratio']:.2%}")
        print(f"   Performance: {comparison['performance_percentage']:.1f}% of expert")
        
        if comparison['performance_percentage'] >= 90:
            print("   ✅ Excellent performance!")
        elif comparison['performance_percentage'] >= 70:
            print("   ✅ Good performance!")
        elif comparison['performance_percentage'] >= 50:
            print("   ⚠️  Moderate performance")
        else:
            print("   ❌ Needs improvement")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Test trained DAgger (IL+RL) model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test model
  python test_ilrl.py --model-path models/dagger_exp1_*/dagger_model.pth
  
  # Test with rendering
  python test_ilrl.py --model-path models/dagger_exp1_*/dagger_model.pth --render
  
  # Compare with expert
  python test_ilrl.py --model-path models/dagger_exp1_*/dagger_model.pth --compare-expert
        """
    )
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=True,
        help='Path to model file (supports wildcards)'
    )
    
    parser.add_argument(
        '--env', '-e',
        type=str,
        default=None,
        help='Environment name (default: from config)'
    )
    
    parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=10,
        help='Number of test episodes (default: 10)'
    )
    
    parser.add_argument(
        '--render', '-r',
        action='store_true',
        help='Enable rendering'
    )
    
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=0.05,
        help='Rendering delay in seconds (default: 0.05)'
    )
    
    parser.add_argument(
        '--compare-expert', '-c',
        action='store_true',
        help='Compare with expert policy'
    )
    
    args = parser.parse_args()
    
    # 查找模型文件
    model_paths = glob.glob(args.model_path)
    if not model_paths:
        print(f"❌ Error: No model file found matching '{args.model_path}'")
        return
    
    if len(model_paths) > 1:
        print(f"⚠️  Multiple model files found, using: {model_paths[0]}")
    
    model_path = model_paths[0]
    print(f"📂 Loading model from {model_path}")
    
    # 加载模型和配置
    agent, config = load_model_and_config(model_path)
    
    # 获取环境名称
    env_name = args.env or config.get('env_name', 'highway-v0')
    
    # 创建环境
    render_mode = 'human' if args.render else None
    env = HighwayWrapper(env_name, render_mode=render_mode)
    
    print(f"🎮 Environment: {env_name}")
    print(f"   State dimension: {env.get_state_dim()}")
    print(f"   Action dimension: {env.get_action_dim()}")
    
    try:
        # 评估智能体
        results = evaluate_agent(
            agent,
            env,
            num_episodes=args.episodes,
            render=args.render,
            delay=args.delay
        )
        
        # 与专家比较
        comparison = None
        if args.compare_expert:
            comparison = compare_with_expert(agent, env, num_episodes=args.episodes)
        
        # 打印结果
        print_results(results, comparison)
        
        print("\n✅ Testing completed!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()

