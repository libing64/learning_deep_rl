#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Highway Environment Demo
Highway 环境演示脚本 - 展示各种highway场景
"""

import time
import numpy as np
from highway_env import HighwayWrapper, get_available_environments


def demo_environment(env_name, num_episodes=2, render=True, max_steps=100):
    """
    演示特定环境
    
    Args:
        env_name: 环境名称
        num_episodes: 演示回合数
        render: 是否渲染
        max_steps: 每回合最大步数
    """
    print("=" * 70)
    print(f"🎬 Demonstrating {env_name}")
    print("=" * 70)
    
    try:
        # 创建环境
        render_mode = 'human' if render else None
        env = HighwayWrapper(env_name, render_mode=render_mode)
        print(f"\n📋 Environment Info:")
        state_dim = env.get_state_dim()
        if state_dim is not None:
            print(f"   State dimension: {state_dim}")
        else:
            # 如果无法获取，尝试从观察中推断
            try:
                obs, _ = env.reset()
                state_dim = len(env.flatten_observation(obs))
                print(f"   State dimension: {state_dim} (inferred)")
            except:
                print(f"   State dimension: Unknown")
        print(f"   Action dimension: {env.get_action_dim()}")
        print(f"   Action meanings: IDLE, LANE_LEFT, LANE_RIGHT, FASTER, SLOWER")
        print()
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            print(f"🚗 Episode {episode + 1}/{num_episodes}")
            state, info = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            # 统计动作使用情况
            action_counts = [0] * env.get_action_dim()
            
            while not done and steps < max_steps:
                # 随机策略演示
                action = env.action_space.sample()
                action_counts[action] += 1
                
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                if render:
                    env.render()
                    time.sleep(0.05)  # 控制渲染速度
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            print(f"   Steps: {steps}, Total Reward: {total_reward:.2f}")
            print(f"   Action distribution: {action_counts}")
            
            if render and episode < num_episodes - 1:
                print("   (Waiting 2 seconds before next episode...)")
                time.sleep(2)
        
        print(f"\n📊 Summary:")
        print(f"   Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"   Average Steps: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"   Max Reward: {max(episode_rewards):.2f}")
        print(f"   Min Reward: {min(episode_rewards):.2f}")
        
        env.close()
        print("\n✅ Demo completed!\n")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Please install highway-env: pip install highway-env\n")
    except Exception as e:
        print(f"❌ Error running demo: {e}\n")


def demo_all_scenarios(num_episodes=1, render=True):
    """
    演示所有可用的highway场景
    
    Args:
        num_episodes: 每个场景的演示回合数
        render: 是否渲染
    """
    print("=" * 70)
    print("🌉 Highway Environment Scenarios Demo")
    print("=" * 70)
    print("\nThis demo will showcase various highway driving scenarios:")
    print("  1. highway-v0: Basic highway with multiple lanes")
    print("  2. highway-fast-v0: Fast highway environment")
    print("  3. merge-v0: Lane merging scenario")
    print("  4. roundabout-v0: Roundabout navigation")
    print("  5. parking-v0: Parking scenario")
    print("  6. intersection-v0: Intersection navigation")
    print()
    
    environments = get_available_environments()
    
    for i, env_name in enumerate(environments, 1):
        print(f"\n[{i}/{len(environments)}]")
        try:
            demo_environment(env_name, num_episodes=num_episodes, render=render, max_steps=100)
        except Exception as e:
            print(f"⚠️  Skipping {env_name}: {e}\n")
            continue
    
    print("=" * 70)
    print("🎉 All scenarios demonstrated!")
    print("=" * 70)


def demo_specific_scenario(env_name, num_episodes=3, render=True):
    """
    演示特定场景
    
    Args:
        env_name: 环境名称
        num_episodes: 演示回合数
        render: 是否渲染
    """
    demo_environment(env_name, num_episodes=num_episodes, render=render, max_steps=200)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Demonstrate various highway driving scenarios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo all scenarios
  python highway_demo.py --all
  
  # Demo specific scenario
  python highway_demo.py --env highway-v0 --episodes 3
  
  # Demo without rendering (faster)
  python highway_demo.py --all --no-render
        """
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Demo all available scenarios'
    )
    
    parser.add_argument(
        '--env', '-e',
        type=str,
        default='highway-v0',
        help='Specific environment to demo (default: highway-v0)'
    )
    
    parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=2,
        help='Number of episodes per scenario (default: 2)'
    )
    
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering (faster execution)'
    )
    
    args = parser.parse_args()
    
    if args.all:
        demo_all_scenarios(num_episodes=args.episodes, render=not args.no_render)
    else:
        demo_specific_scenario(args.env, num_episodes=args.episodes, render=not args.no_render)


if __name__ == "__main__":
    main()

