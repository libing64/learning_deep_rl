#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Trained DQN Model for CartPole Control
使用训练好的DQN模型进行CartPole控制测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
from cart_pole_env import CartPoleWrapper
from dqn_train import DQN, DQNAgent


def load_trained_agent(model_path, state_dim=4, action_dim=2):
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


def test_episode(env, agent, render=False, delay=0.01):
    """
    运行一个测试回合
    
    Args:
        env: CartPole环境
        agent: DQN智能体
        render: 是否渲染
        delay: 渲染延迟（秒）
        
    Returns:
        total_reward: 总奖励
        steps: 步数
        episode_info: 回合信息
    """
    state, info = env.reset()
    total_reward = 0
    steps = 0
    done = False
    episode_info = {
        'states': [np.array(state)],
        'actions': [],
        'rewards': [],
        'q_values': []
    }
    
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


def test_model(model_path, num_episodes=10, render=False, delay=0.01):
    """
    测试训练好的模型
    
    Args:
        model_path: 模型文件路径
        num_episodes: 测试回合数
        render: 是否渲染（可视化）
        delay: 渲染延迟（秒）
    """
    print("=" * 60)
    print("🧪 CartPole DQN Model Testing")
    print("=" * 60)
    
    # 创建环境
    print("\n🎮 Creating CartPole environment...")
    render_mode = 'human' if render else None
    env = CartPoleWrapper(render_mode=render_mode)
    
    # 加载模型
    agent = load_trained_agent(model_path)
    if agent is None:
        env.close()
        return
    
    # 获取环境信息
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")
    
    # 运行测试
    print(f"\n🚀 Running {num_episodes} test episodes...")
    if render:
        print("   (Rendering enabled - close window to continue)")
    print("-" * 60)
    
    test_rewards = []
    test_lengths = []
    all_episode_info = []
    
    for episode in range(num_episodes):
        total_reward, steps, episode_info = test_episode(env, agent, render=render, delay=delay)
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        all_episode_info.append(episode_info)
        
        print(f"Episode {episode + 1:3d} | "
              f"Reward: {total_reward:6.1f} | "
              f"Steps: {steps:3d} | "
              f"Avg Q-value: {np.mean(episode_info['q_values']):.3f}")
    
    # 统计结果
    print("-" * 60)
    print("\n📊 Test Results Summary:")
    print(f"   Total Episodes: {num_episodes}")
    print(f"   Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"   Average Steps: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")
    print(f"   Max Reward: {max(test_rewards):.2f}")
    print(f"   Min Reward: {min(test_rewards):.2f}")
    print(f"   Success Rate (Reward >= 475): {(np.array(test_rewards) >= 475).sum() / num_episodes * 100:.1f}%")
    
    # 分析动作分布
    all_actions = []
    for info in all_episode_info:
        all_actions.extend(info['actions'])
    action_counts = np.bincount(all_actions, minlength=action_dim)
    print(f"\n🎯 Action Distribution:")
    print(f"   Action 0 (Left): {action_counts[0]} ({action_counts[0]/len(all_actions)*100:.1f}%)")
    print(f"   Action 1 (Right): {action_counts[1]} ({action_counts[1]/len(all_actions)*100:.1f}%)")
    
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


def demo_control(model_path, num_demos=3):
    """
    演示模型控制（可视化）
    
    Args:
        model_path: 模型文件路径
        num_demos: 演示回合数
    """
    print("=" * 60)
    print("🎬 CartPole DQN Control Demonstration")
    print("=" * 60)
    print(f"\nRunning {num_demos} demonstration episodes...")
    print("Close the rendering window to proceed to next episode.\n")
    
    test_model(model_path, num_episodes=num_demos, render=True, delay=0.02)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Test trained DQN model for CartPole control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test model without visualization
  python test_model.py --model dqn_model_final.pth
  
  # Test with visualization (demo)
  python test_model.py --model dqn_model_final.pth --demo
  
  # Test with custom number of episodes
  python test_model.py --model dqn_model_final.pth --episodes 20
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='dqn_model_final.pth',
        help='Path to trained model file (default: dqn_model_final.pth)'
    )
    
    parser.add_argument(
        '--episodes', '-e',
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
    
    args = parser.parse_args()
    
    if args.demo:
        # 演示模式：可视化
        demo_control(args.model, num_demos=args.episodes)
    else:
        # 测试模式
        test_model(args.model, num_episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()

