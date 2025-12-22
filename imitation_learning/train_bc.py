#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Behavioral Cloning
行为克隆训练脚本

这个脚本负责：
- 加载专家演示数据
- 训练行为克隆模型
- 保存训练好的模型
- 生成训练报告和可视化
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
from highway.highway_env import HighwayWrapper


def setup_experiment(exp_name: str, data_dir: str = "data", models_dir: str = "models"):
    """
    设置实验目录和配置

    Args:
        exp_name: 实验名称
        data_dir: 数据目录
        models_dir: 模型目录

    Returns:
        config: 实验配置字典
    """
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(models_dir) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 实验配置
    config = {
        'experiment_name': exp_name,
        'timestamp': timestamp,
        'exp_dir': str(exp_dir),
        'data_dir': data_dir,
        'model_path': str(exp_dir / 'bc_model.pth'),
        'config_path': str(exp_dir / 'config.json'),
        'log_path': str(exp_dir / 'training.log'),
        'plots_dir': str(exp_dir / 'plots'),
        # 训练超参数
        'env_name': 'highway-v0',
        'hidden_dims': [256, 128],
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'epochs': 100,
        'validation_split': 0.2,
        # 数据配置
        'use_split': True,
        'train_ratio': 0.7,
        'val_ratio': 0.2,
    }

    # 保存配置
    with open(config['config_path'], 'w') as f:
        json.dump(config, f, indent=2)

    # 创建plots目录
    Path(config['plots_dir']).mkdir(exist_ok=True)

    return config


def train_model(config: dict):
    """
    训练行为克隆模型

    Args:
        config: 实验配置

    Returns:
        bc_model: 训练好的模型
        eval_results: 评估结果
    """
    print("🚀 Starting Behavioral Cloning Training")
    print(f"📁 Experiment: {config['experiment_name']}")
    print(f"📂 Model will be saved to: {config['model_path']}")

    # 加载数据
    print("📚 Loading dataset...")
    dataset = ImitationDataset(config['data_dir'])
    trajectories, metadata = dataset.load_data()

    print(f"📊 Dataset: {metadata['num_trajectories']} trajectories, "
          f"{metadata['total_samples']} samples")
    print(".2f")
    # 创建环境包装器获取状态维度
    env_wrapper = HighwayWrapper(config['env_name'])
    state_dim = env_wrapper.get_state_dim()
    action_dim = env_wrapper.get_action_dim()

    print(f"🏗️  Model: State dim = {state_dim}, Action dim = {action_dim}")
    print(f"   Hidden layers: {config['hidden_dims']}")

    # 创建行为克隆模型
    bc_model = BehavioralCloning(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config['hidden_dims'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 准备训练数据
    if config['use_split']:
        train_trajectories, val_trajectories, test_trajectories = dataset.split_data(
            config['train_ratio'], config['val_ratio']
        )
        train_data = train_trajectories + val_trajectories  # 合并用于训练
    else:
        train_data = trajectories

    # 训练模型
    print(f"🎯 Training for {config['epochs']} epochs with batch size {config['batch_size']}...")
    bc_model.train(
        trajectories=train_data,
        env_wrapper=env_wrapper,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=config['validation_split']
    )

    # 保存模型
    bc_model.save_model(config['model_path'])

    # 评估模型
    print("🔍 Evaluating trained model...")
    eval_results = bc_model.evaluate(
        env_name=config['env_name'],
        num_episodes=20,
        max_steps=200,
        render=False
    )

    # 保存评估结果
    eval_path = Path(config['exp_dir']) / 'evaluation_results.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    # 生成训练曲线
    plot_training_curves(bc_model.train_history, config['plots_dir'])

    # 生成性能对比
    if 'expert_trajectories' in metadata:
        expert_trajectories = metadata['expert_trajectories']
        plot_performance_comparison(expert_trajectories, eval_results, config['plots_dir'])

    print("✅ Training completed!")
    print(".2f")
    return bc_model, eval_results


def plot_training_curves(train_history: dict, plots_dir: str):
    """
    绘制训练曲线

    Args:
        train_history: 训练历史
        plots_dir: 图表保存目录
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_history['loss']) + 1)

    # 损失曲线
    ax1.plot(epochs, train_history['loss'], 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    ax1.legend()

    # 准确率曲线
    ax2.plot(epochs, train_history['accuracy'], 'r-', label='Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Training curves saved to {plots_dir}/training_curves.png")


def plot_performance_comparison(expert_trajectories: list, bc_results: dict, plots_dir: str):
    """
    绘制性能对比图

    Args:
        expert_trajectories: 专家轨迹
        bc_results: BC评估结果
        plots_dir: 图表保存目录
    """
    expert_rewards = [traj['total_reward'] for traj in expert_trajectories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 奖励分布对比
    ax1.hist(expert_rewards, alpha=0.7, label='Expert', bins=20, density=True)
    ax1.axvline(np.mean(expert_rewards), color='blue', linestyle='--',
                label='.1f')
    ax1.axvline(bc_results['mean_reward'], color='red', linestyle='--',
                label='.1f')
    ax1.set_xlabel('Episode Reward')
    ax1.set_ylabel('Density')
    ax1.set_title('Reward Distribution Comparison')
    ax1.legend()
    ax1.grid(True)

    # 奖励时间序列
    bc_rewards = bc_results['episode_rewards']
    ax2.plot(range(len(bc_rewards)), bc_rewards, 'r-o', alpha=0.7, label='BC Policy', markersize=3)
    ax2.axhline(np.mean(expert_rewards), color='blue', linestyle='--',
                label='.1f')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('BC Policy Performance')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Performance comparison saved to {plots_dir}/performance_comparison.png")


def generate_training_report(config: dict, eval_results: dict, train_history: dict):
    """
    生成训练报告

    Args:
        config: 实验配置
        eval_results: 评估结果
        train_history: 训练历史
    """
    report_path = Path(config['exp_dir']) / 'training_report.md'

    report = f"""# Behavioral Cloning Training Report

## Experiment Overview
- **Experiment Name**: {config['experiment_name']}
- **Timestamp**: {config['timestamp']}
- **Environment**: {config['env_name']}

## Model Configuration
- **Hidden Layers**: {config['hidden_dims']}
- **Learning Rate**: {config['learning_rate']}
- **Weight Decay**: {config['weight_decay']}
- **Batch Size**: {config['batch_size']}
- **Epochs**: {config['epochs']}

## Training Results
- **Final Training Loss**: {train_history['loss'][-1]:.4f}
- **Final Training Accuracy**: {train_history['accuracy'][-1]:.4f}
- **Best Training Accuracy**: {max(train_history['accuracy']):.4f}

## Evaluation Results
- **Mean Episode Reward**: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}
- **Mean Episode Length**: {eval_results['mean_length']:.1f}
- **Success Rate**: {eval_results['success_rate']:.2%}

## Files Generated
- Model: `{config['model_path']}`
- Config: `{config['config_path']}`
- Training Curves: `{config['plots_dir']}/training_curves.png`
- Performance Comparison: `{config['plots_dir']}/performance_comparison.png`
- Evaluation Results: `{config['exp_dir']}/evaluation_results.json`

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"📝 Training report saved to {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train Behavioral Cloning Model")
    parser.add_argument('--exp-name', type=str, default='bc_highway',
                       help='Experiment name')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Models directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden-dims', type=str, default='256,128',
                       help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation after training')

    args = parser.parse_args()

    # 解析隐藏层维度
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    # 设置实验
    config = setup_experiment(
        exp_name=args.exp_name,
        data_dir=args.data_dir,
        models_dir=args.models_dir
    )

    # 更新配置
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'hidden_dims': hidden_dims,
    })

    # 重新保存配置
    with open(config['config_path'], 'w') as f:
        json.dump(config, f, indent=2)

    try:
        # 训练模型
        bc_model, eval_results = train_model(config)

        # 生成报告
        generate_training_report(config, eval_results, bc_model.train_history)

        print("🎉 Training pipeline completed successfully!")
        print(f"📂 Results saved to: {config['exp_dir']}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
