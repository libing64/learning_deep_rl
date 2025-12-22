#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Trained DQN Model for Highway Control
使用训练好的DQN模型进行Highway控制测试和演示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
from highway_env import HighwayWrapper
from dqn_train import DQN, DQNAgent


def load_trained_agent(model_path, state_dim, action_dim):
    """
    加载训练好的DQN智能体
    
    Args:
        model_path: 模型文件路径
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        
    Returns:
        agent: 加载的智能体
    """
    print(f"📦 Loading model from {model_path}...")
    
    # 创建智能体（使用默认参数）
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        epsilon=0.0  # 测试时不探索
    )
    
    # 加载模型
    try:
        agent.load(model_path)
        print("✅ Model loaded successfully!")
        return agent
    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("   Please train a model first using: python dqn_train.py")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def test_episode(env, agent, render=False, delay=0.05):
    """
    运行一个测试回合
    
    Args:
        env: Highway环境
        agent: DQN智能体
        render: 是否渲染
        delay: 渲染延迟（秒）
        
    Returns:
        total_reward: 总奖励
        steps: 步数
        episode_info: 回合信息
    """
    state, info = env.reset()
    state = env.flatten_observation(state)
    total_reward = 0
    steps = 0
    done = False
    episode_info = {
        'states': [np.array(state)],
        'actions': [],
        'rewards': [],
        'q_values': []
    }
    
    action_names = ['IDLE', 'LANE_LEFT', 'LANE_RIGHT', 'FASTER', 'SLOWER']
    
    while not done:
        # 使用模型选择动作（不探索）
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.q_network(state_tensor)
            action = q_values.argmax().item()
            q_value = q_values.max().item()
        
        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = env.flatten_observation(next_state)
        
        # 记录信息
        episode_info['states'].append(np.array(next_state))
        episode_info['actions'].append(action)
        episode_info['rewards'].append(reward)
        episode_info['q_values'].append(q_value)
        
        state = next_state
        total_reward += reward
        steps += 1
        
        # 渲染
        if render:
            env.render()
            time.sleep(delay)
    
    return total_reward, steps, episode_info


def test_model(model_path, env_name='highway-v0', num_episodes=10, render=False, delay=0.05):
    """
    测试训练好的模型
    
    Args:
        model_path: 模型文件路径
        env_name: 环境名称
        num_episodes: 测试回合数
        render: 是否渲染（可视化）
        delay: 渲染延迟（秒）
    """
    print("=" * 70)
    print("🧪 Highway DQN Model Testing")
    print("=" * 70)
    
    # 创建环境
    print(f"\n🎮 Creating {env_name} environment...")
    try:
        render_mode = 'human' if render else None
        env = HighwayWrapper(env_name, render_mode=render_mode)
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
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")
    
    # 加载模型
    agent = load_trained_agent(model_path, state_dim, action_dim)
    if agent is None:
        env.close()
        return
    
    # 运行测试
    print(f"\n🚀 Running {num_episodes} test episodes...")
    if render:
        print("   (Rendering enabled - close window to continue)")
    print("-" * 70)
    
    test_rewards = []
    test_lengths = []
    all_episode_info = []
    
    action_names = ['IDLE', 'LANE_LEFT', 'LANE_RIGHT', 'FASTER', 'SLOWER']
    
    for episode in range(num_episodes):
        total_reward, steps, episode_info = test_episode(env, agent, render=render, delay=delay)
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        all_episode_info.append(episode_info)
        
        # 统计动作分布
        action_counts = np.bincount(episode_info['actions'], minlength=action_dim)
        most_common_action = np.argmax(action_counts)
        
        print(f"Episode {episode + 1:3d} | "
              f"Reward: {total_reward:7.2f} | "
              f"Steps: {steps:3d} | "
              f"Avg Q-value: {np.mean(episode_info['q_values']):.3f} | "
              f"Most used: {action_names[most_common_action]}")
        
        if render and episode < num_episodes - 1:
            print("   (Waiting 2 seconds before next episode...)")
            time.sleep(2)
    
    # 统计结果
    print("-" * 70)
    print("\n📊 Test Results Summary:")
    print(f"   Total Episodes: {num_episodes}")
    print(f"   Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"   Average Steps: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")
    print(f"   Max Reward: {max(test_rewards):.2f}")
    print(f"   Min Reward: {min(test_rewards):.2f}")
    
    # 分析动作分布
    all_actions = []
    for info in all_episode_info:
        all_actions.extend(info['actions'])
    action_counts = np.bincount(all_actions, minlength=action_dim)
    print(f"\n🎯 Action Distribution:")
    for i, name in enumerate(action_names[:action_dim]):
        count = action_counts[i]
        percentage = count / len(all_actions) * 100 if len(all_actions) > 0 else 0
        print(f"   {name:12s}: {count:5d} ({percentage:5.1f}%)")
    
    # 分析Q值
    all_q_values = []
    for info in all_episode_info:
        all_q_values.extend(info['q_values'])
    print(f"\n💡 Q-Value Statistics:")
    print(f"   Mean: {np.mean(all_q_values):.3f}")
    print(f"   Std: {np.std(all_q_values):.3f}")
    print(f"   Min: {np.min(all_q_values):.3f}")
    print(f"   Max: {np.max(all_q_values):.3f}")
    
    # 关闭环境
    env.close()
    print("\n✅ Testing completed!")


def demo_control(model_path, env_name='highway-v0', num_demos=3, delay=0.05):
    """
    演示模型控制（可视化）
    
    Args:
        model_path: 模型文件路径
        env_name: 环境名称
        num_demos: 演示回合数
        delay: 渲染延迟（秒）
    """
    print("=" * 70)
    print("🎬 Highway DQN Control Demonstration")
    print("=" * 70)
    print(f"\nRunning {num_demos} demonstration episodes...")
    print("Close the rendering window to proceed to next episode.\n")
    
    test_model(model_path, env_name=env_name, num_episodes=num_demos, render=True, delay=delay)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Test trained DQN model for highway control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test model without visualization
  python test_model.py --model dqn_model_final.pth
  
  # Test with visualization (demo)
  python test_model.py --model dqn_model_final.pth --demo
  
  # Test with custom number of episodes
  python test_model.py --model dqn_model_final.pth --episodes 20
  
  # Test on different environment
  python test_model.py --model dqn_model_final.pth --env merge-v0 --demo
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='dqn_model_final.pth',
        help='Path to trained model file (default: dqn_model_final.pth)'
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
        default=10,
        help='Number of test episodes (default: 10)'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demonstration with visualization'
    )
    
    parser.add_argument(
        '--render', '-r',
        action='store_true',
        help='Enable rendering during testing'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=0.05,
        help='Rendering delay in seconds (default: 0.05)'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # 演示模式：可视化
        demo_control(args.model, env_name=args.env, num_demos=args.episodes, delay=args.delay)
    else:
        # 测试模式
        test_model(args.model, env_name=args.env, num_episodes=args.episodes, 
                  render=args.render, delay=args.delay)


if __name__ == "__main__":
    main()

