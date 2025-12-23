#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for DAgger (IL+RL) Algorithm
DAgger算法训练脚本
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

from dagger_algorithm import dagger_algorithm, DAggerAgent


def setup_experiment(exp_name: str, models_dir: str = "models"):
    """
    设置实验目录和配置
    
    Args:
        exp_name: 实验名称
        models_dir: 模型存储目录
        
    Returns:
        config: 实验配置字典
    """
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(models_dir) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (exp_dir / "plots").mkdir(exist_ok=True)
    
    config = {
        'exp_name': exp_name,
        'exp_dir': str(exp_dir),
        'timestamp': timestamp
    }
    
    return config


def save_training_curves(agent: DAggerAgent, save_path: str):
    """
    保存训练曲线
    
    Args:
        agent: 训练好的智能体
        save_path: 保存路径
    """
    if not agent.train_history['loss']:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 损失曲线
    epochs = [h['epoch'] for h in agent.train_history['loss']]
    train_losses = [h['train'] for h in agent.train_history['loss']]
    val_losses = [h['val'] for h in agent.train_history['loss']]
    
    axes[0].plot(epochs, train_losses, label='Train Loss', alpha=0.7)
    axes[0].plot(epochs, val_losses, label='Validation Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    train_accs = [h['train'] for h in agent.train_history['accuracy']]
    val_accs = [h['val'] for h in agent.train_history['accuracy']]
    
    axes[1].plot(epochs, train_accs, label='Train Accuracy', alpha=0.7)
    axes[1].plot(epochs, val_accs, label='Validation Accuracy', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Training curves saved to {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Train DAgger (IL+RL) algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_ilrl.py --exp-name dagger_exp1
  
  # Custom configuration
  python train_ilrl.py --exp-name dagger_exp2 \\
      --initial-expert 200 \\
      --iterations 10 \\
      --trajectories-per-iter 100 \\
      --epochs 20
        """
    )
    
    parser.add_argument(
        '--exp-name', '-n',
        type=str,
        required=True,
        help='Experiment name'
    )
    
    parser.add_argument(
        '--env', '-e',
        type=str,
        default='highway-v0',
        help='Environment name (default: highway-v0)'
    )
    
    parser.add_argument(
        '--initial-expert', '-i',
        type=int,
        default=100,
        help='Initial expert trajectories (default: 100)'
    )
    
    parser.add_argument(
        '--iterations', '-it',
        type=int,
        default=5,
        help='Number of DAgger iterations (default: 5)'
    )
    
    parser.add_argument(
        '--trajectories-per-iter', '-t',
        type=int,
        default=50,
        help='Trajectories per DAgger iteration (default: 50)'
    )
    
    parser.add_argument(
        '--max-steps', '-s',
        type=int,
        default=200,
        help='Maximum steps per episode (default: 200)'
    )
    
    parser.add_argument(
        '--epochs', '-ep',
        type=int,
        default=10,
        help='Training epochs per iteration (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # 设置实验
    config = setup_experiment(args.exp_name)
    
    # 保存配置
    config.update({
        'env_name': args.env,
        'initial_expert_trajectories': args.initial_expert,
        'dagger_iterations': args.iterations,
        'trajectories_per_iteration': args.trajectories_per_iter,
        'max_steps': args.max_steps,
        'training_epochs': args.epochs,
        'batch_size': args.batch_size,
        'device': args.device
    })
    
    config_path = Path(config['exp_dir']) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Configuration saved to {config_path}")
    
    try:
        # 运行DAgger算法
        agent = dagger_algorithm(
            env_name=args.env,
            initial_expert_trajectories=args.initial_expert,
            dagger_iterations=args.iterations,
            trajectories_per_iteration=args.trajectories_per_iter,
            max_steps=args.max_steps,
            training_epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # 保存模型
        model_path = Path(config['exp_dir']) / 'dagger_model.pth'
        agent.save(str(model_path))
        
        # 保存训练曲线
        curves_path = Path(config['exp_dir']) / 'plots' / 'training_curves.png'
        save_training_curves(agent, str(curves_path))
        
        # 保存训练历史
        history_path = Path(config['exp_dir']) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(agent.train_history, f, indent=2)
        
        print("\n" + "=" * 70)
        print("🎉 Training completed successfully!")
        print("=" * 70)
        print(f"📂 Results saved to: {config['exp_dir']}")
        print(f"📦 Model saved to: {model_path}")
        print(f"📊 Training curves: {curves_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

