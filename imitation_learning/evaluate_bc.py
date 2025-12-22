#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Script for Behavioral Cloning
行为克隆评估脚本

这个脚本负责：
- 加载训练好的行为克隆模型
- 进行全面的性能评估
- 与专家策略进行对比
- 生成评估报告和可视化
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from data_collection import ImitationDataset
from behavioral_cloning import BehavioralCloning
from expert_policy import HighwayExpert, generate_expert_trajectories
from highway.highway_env import HighwayWrapper


def load_model_and_config(model_path: str):
    """
    加载模型和配置

    Args:
        model_path: 模型路径

    Returns:
        bc_model: 加载的模型
        config: 模型配置
    """
    # 找到配置文件的路径
    model_dir = Path(model_path).parent
    config_path = model_dir / 'config.json'

    if not config_path.exists():
        # 如果没有配置文件，使用默认配置
        print("⚠️  Config file not found, using default configuration")
        config = {
            'env_name': 'highway-v0',
            'hidden_dims': [256, 128],
        }
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)

    # 创建环境包装器获取维度
    env_wrapper = HighwayWrapper(config['env_name'])
    state_dim = env_wrapper.get_state_dim()
    action_dim = env_wrapper.get_action_dim()

    # 创建模型并加载权重
    bc_model = BehavioralCloning(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config['hidden_dims']
    )
    bc_model.load_model(model_path)

    return bc_model, config


def comprehensive_evaluation(bc_model: BehavioralCloning, config: dict,
                           num_episodes: int = 50, max_steps: int = 300,
                           render: bool = False):
    """
    全面评估行为克隆模型

    Args:
        bc_model: 行为克隆模型
        config: 配置字典
        num_episodes: 评估回合数
        max_steps: 每回合最大步数
        render: 是否渲染

    Returns:
        eval_results: 详细的评估结果
    """
    print(f"🔍 Starting comprehensive evaluation ({num_episodes} episodes)...")

    # 基本性能评估
    basic_results = bc_model.evaluate(
        env_name=config['env_name'],
        num_episodes=num_episodes,
        max_steps=max_steps,
        render=render
    )

    # 专家策略评估（用于对比）
    print("🎯 Evaluating expert policy for comparison...")
    expert_trajectories = generate_expert_trajectories(
        env_name=config['env_name'],
        num_episodes=num_episodes,
        max_steps=max_steps
    )

    expert_rewards = [traj['total_reward'] for traj in expert_trajectories]
    expert_lengths = [traj['episode_length'] for traj in expert_trajectories]

    expert_results = {
        'mean_reward': np.mean(expert_rewards),
        'std_reward': np.std(expert_rewards),
        'mean_length': np.mean(expert_lengths),
        'success_rate': len([r for r in expert_rewards if r > 0]) / len(expert_rewards),  # 简单成功定义
        'episode_rewards': expert_rewards,
        'episode_lengths': expert_lengths
    }

    # 动作分布分析
    action_distribution = analyze_action_distribution(bc_model, config, num_episodes=10)

    # 鲁棒性测试
    robustness_results = test_robustness(bc_model, config)

    # 综合结果
    eval_results = {
        'basic_performance': basic_results,
        'expert_comparison': expert_results,
        'action_distribution': action_distribution,
        'robustness_test': robustness_results,
        'performance_gap': {
            'reward_gap': expert_results['mean_reward'] - basic_results['mean_reward'],
            'length_gap': expert_results['mean_length'] - basic_results['mean_length'],
            'success_rate_gap': expert_results['success_rate'] - basic_results['success_rate']
        },
        'evaluation_config': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'env_name': config['env_name']
        }
    }

    print("✅ Comprehensive evaluation completed!")
    return eval_results


def analyze_action_distribution(bc_model: BehavioralCloning, config: dict, num_episodes: int = 10):
    """
    分析学习策略的动作分布

    Args:
        bc_model: 行为克隆模型
        config: 配置
        num_episodes: 分析的回合数

    Returns:
        action_stats: 动作分布统计
    """
    print("📊 Analyzing action distribution...")

    env_wrapper = HighwayWrapper(config['env_name'])
    all_actions = []

    for episode in range(num_episodes):
        state, info = env_wrapper.env.reset()
        done = False
        steps = 0

        while not done and steps < 100:  # 限制步数用于分析
            state_flat = env_wrapper.flatten_observation(state)
            action = bc_model.policy.get_action(state_flat)
            all_actions.append(action)

            state, reward, terminated, truncated, info = env_wrapper.env.step(action)
            done = terminated or truncated
            steps += 1

    env_wrapper.env.close()

    # 统计动作分布
    action_counts = np.bincount(all_actions, minlength=5)  # 假设5个动作
    action_probs = action_counts / len(all_actions)

    action_stats = {
        'action_counts': action_counts.tolist(),
        'action_probabilities': action_probs.tolist(),
        'total_actions': len(all_actions),
        'most_frequent_action': int(np.argmax(action_counts)),
        'action_entropy': -np.sum(action_probs * np.log(action_probs + 1e-10))
    }

    return action_stats


def test_robustness(bc_model: BehavioralCloning, config: dict):
    """
    测试模型的鲁棒性（不同环境配置下的表现）

    Args:
        bc_model: 行为克隆模型
        config: 配置

    Returns:
        robustness_results: 鲁棒性测试结果
    """
    print("🧪 Testing robustness across different configurations...")

    test_configs = [
        {'name': 'default', 'config': None},
        {'name': 'heavy_traffic', 'config': {'vehicles_count': 20, 'duration': 50}},
        {'name': 'light_traffic', 'config': {'vehicles_count': 5, 'duration': 50}},
        {'name': 'fast_pace', 'config': {'reward_speed_range': [25, 35], 'duration': 50}},
    ]

    robustness_results = {}

    for test_config in test_configs:
        print(f"  Testing {test_config['name']}...")

        results = bc_model.evaluate(
            env_name=config['env_name'],
            num_episodes=10,
            max_steps=100,
            render=False
        )

        robustness_results[test_config['name']] = {
            'mean_reward': results['mean_reward'],
            'success_rate': results['success_rate'],
            'mean_length': results['mean_length']
        }

    return robustness_results


def generate_evaluation_report(eval_results: dict, model_path: str, output_dir: str = "evaluation_results"):
    """
    生成详细的评估报告

    Args:
        eval_results: 评估结果
        model_path: 模型路径
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 生成图表
    generate_evaluation_plots(eval_results, str(output_path))

    # 生成文本报告
    report_path = output_path / 'evaluation_report.md'

    basic = eval_results['basic_performance']
    expert = eval_results['expert_comparison']
    gap = eval_results['performance_gap']
    action_dist = eval_results['action_distribution']
    robustness = eval_results['robustness_test']

    report = f"""# Behavioral Cloning Evaluation Report

## Overview
- **Model**: {model_path}
- **Environment**: {eval_results['evaluation_config']['env_name']}
- **Evaluation Episodes**: {eval_results['evaluation_config']['num_episodes']}
- **Max Steps per Episode**: {eval_results['evaluation_config']['max_steps']}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

### Basic Performance
- **Mean Episode Reward**: {basic['mean_reward']:.2f} ± {basic['std_reward']:.2f}
- **Mean Episode Length**: {basic['mean_length']:.1f}
- **Success Rate**: {basic['success_rate']:.2%}

### Comparison with Expert
- **Expert Mean Reward**: {expert['mean_reward']:.2f} ± {expert['std_reward']:.2f}
- **Expert Mean Length**: {expert['mean_length']:.1f}
- **Expert Success Rate**: {expert['success_rate']:.2%}

### Performance Gap
- **Reward Gap**: {gap['reward_gap']:.2f} ({gap['reward_gap']/expert['mean_reward']:.1%})
- **Length Gap**: {gap['length_gap']:.1f}
- **Success Rate Gap**: {gap['success_rate_gap']:.2%}

## Action Distribution Analysis
- **Total Actions Analyzed**: {action_dist['total_actions']}
- **Most Frequent Action**: {action_dist['most_frequent_action']}
- **Action Entropy**: {action_dist['action_entropy']:.3f}
- **Action Probabilities**: {action_dist['action_probabilities']}

## Robustness Test Results

| Configuration | Mean Reward | Success Rate | Mean Length |
|---------------|-------------|--------------|-------------|
"""

    for config_name, results in robustness.items():
        report += f"| {config_name} | {results['mean_reward']:.2f} | {results['success_rate']:.2%} | {results['mean_length']:.1f} |\n"

    report += """
## Analysis

### Performance Analysis
"""

    if abs(gap['reward_gap']) < expert['std_reward']:
        report += "- ✅ BC performance is comparable to expert (within 1 std dev)\n"
    else:
        report += "- ⚠️  BC performance significantly differs from expert\n"

    if basic['success_rate'] > 0.7:
        report += "- ✅ Good success rate (>70%)\n"
    else:
        report += "- ⚠️  Low success rate, may need more training data or model capacity\n"

    report += f"""
### Action Distribution Insights
- **Action Diversity**: Entropy = {action_dist['action_entropy']:.3f}"""

    if action_dist['action_entropy'] > 1.0:
        report += " (good diversity)\n"
    else:
        report += " (low diversity, may be overfitting)\n"

    # 保存报告
    with open(report_path, 'w') as f:
        f.write(report)

    # 保存详细结果
    results_path = output_path / 'detailed_results.json'
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"📝 Evaluation report saved to {report_path}")
    print(f"📊 Detailed results saved to {results_path}")


def generate_evaluation_plots(eval_results: dict, output_dir: str):
    """
    生成评估结果的可视化图表

    Args:
        eval_results: 评估结果
        output_dir: 输出目录
    """
    basic = eval_results['basic_performance']
    expert = eval_results['expert_comparison']
    robustness = eval_results['robustness_test']

    # 奖励分布对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 奖励直方图对比
    axes[0, 0].hist(expert['episode_rewards'], alpha=0.7, label='Expert', bins=15, density=True)
    axes[0, 0].hist(basic['episode_rewards'], alpha=0.7, label='BC', bins=15, density=True)
    axes[0, 0].axvline(np.mean(expert['episode_rewards']), color='blue', linestyle='--', alpha=0.8)
    axes[0, 0].axvline(np.mean(basic['episode_rewards']), color='orange', linestyle='--', alpha=0.8)
    axes[0, 0].set_xlabel('Episode Reward')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Reward Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 时间序列对比
    episodes = range(len(basic['episode_rewards']))
    axes[0, 1].plot(episodes, basic['episode_rewards'], 'o-', alpha=0.7, label='BC Policy', markersize=4)
    axes[0, 1].axhline(np.mean(expert['episode_rewards']), color='blue', linestyle='--',
                       label='.1f')
    axes[0, 1].axhline(np.mean(basic['episode_rewards']), color='orange', linestyle='--',
                       label='.1f')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Episode Rewards Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 动作分布
    action_dist = eval_results['action_distribution']
    actions = range(len(action_dist['action_probabilities']))
    axes[1, 0].bar(actions, action_dist['action_probabilities'], alpha=0.7)
    axes[1, 0].set_xlabel('Action')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_title('Learned Action Distribution')
    axes[1, 0].set_xticks(actions)
    axes[1, 0].grid(True, alpha=0.3)

    # 鲁棒性测试结果
    configs = list(robustness.keys())
    rewards = [robustness[c]['mean_reward'] for c in configs]
    success_rates = [robustness[c]['success_rate'] for c in configs]

    x = range(len(configs))
    axes[1, 1].bar(x, rewards, alpha=0.7, label='Mean Reward', width=0.35)
    axes[1, 1].set_ylabel('Mean Reward', color='blue')
    axes[1, 1].tick_params(axis='y', labelcolor='blue')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(configs, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    ax2 = axes[1, 1].twinx()
    ax2.plot(x, success_rates, 'r-o', label='Success Rate', markersize=6)
    ax2.set_ylabel('Success Rate', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)

    axes[1, 1].set_title('Robustness Across Configurations')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Evaluation plots saved to {output_dir}/evaluation_plots.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate Behavioral Cloning Model")
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of evaluation episodes')
    parser.add_argument('--max-steps', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true',
                       help='Render evaluation episodes')

    args = parser.parse_args()

    try:
        # 加载模型
        print(f"📂 Loading model from {args.model_path}")
        bc_model, config = load_model_and_config(args.model_path)

        # 进行评估
        eval_results = comprehensive_evaluation(
            bc_model=bc_model,
            config=config,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            render=args.render
        )

        # 生成报告
        generate_evaluation_report(eval_results, args.model_path, args.output_dir)

        # 打印关键结果
        basic = eval_results['basic_performance']
        gap = eval_results['performance_gap']

        print("\n🎯 Evaluation Summary:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(f"📂 Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
