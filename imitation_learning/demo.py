#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imitation Learning Demo
模仿学习演示脚本

这个脚本提供了一个完整的模仿学习流程演示，
从数据收集到训练和评估的完整示例。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demo_data_collection():
    """演示数据收集"""
    print("🎯 Demo: Expert Data Collection")
    print("=" * 50)

    from data_collection import ImitationDataset

    # 创建数据集管理器
    dataset = ImitationDataset("demo_data")

    # 收集少量演示数据用于演示
    print("📚 Collecting expert demonstrations...")
    trajectories = dataset.collect_expert_data(
        env_name='highway-v0',
        num_episodes=50,  # 演示用少量数据
        max_steps=100
    )

    print(f"✅ Collected {len(trajectories)} expert trajectories")

    # 显示统计信息
    stats = dataset.get_statistics()
    print(".2f")
    print(f"   Avg episode length: {stats['basic_stats']['avg_episode_length']:.1f}")
    print(".2f")
    print(f"   Action distribution: {stats['action_distribution']['action_probabilities']}")

    return trajectories


def demo_training(trajectories):
    """演示模型训练"""
    print("\n🎯 Demo: Behavioral Cloning Training")
    print("=" * 50)

    from behavioral_cloning import BehavioralCloning
    from highway.highway_env import HighwayWrapper

    # 创建环境包装器
    env_wrapper = HighwayWrapper('highway-v0')
    state_dim = env_wrapper.get_state_dim()
    action_dim = env_wrapper.get_action_dim()

    print(f"🏗️  Creating BC model: {state_dim} states -> {action_dim} actions")

    # 创建行为克隆模型
    bc_model = BehavioralCloning(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 64],  # 较小的网络用于演示
        learning_rate=1e-3
    )

    # 训练模型（少量轮数用于演示）
    print("🎓 Training model...")
    bc_model.train(
        trajectories=trajectories,
        env_wrapper=env_wrapper,
        batch_size=32,
        epochs=20,  # 演示用少量轮数
        validation_split=0.2
    )

    print(".4f")
    print(".3f")
    # 保存模型
    model_path = "demo_model.pth"
    bc_model.save_model(model_path)

    return bc_model, model_path


def demo_evaluation(bc_model, trajectories):
    """演示模型评估"""
    print("\n🎯 Demo: Model Evaluation")
    print("=" * 50)

    from behavioral_cloning import compare_with_expert

    # 评估训练好的模型
    print("🔍 Evaluating trained model...")
    eval_results = bc_model.evaluate(
        env_name='highway-v0',
        num_episodes=10,  # 演示用少量评估
        max_steps=100,
        render=False
    )

    print(".2f"
    print(".2f"
    print(".1f"
    # 与专家比较
    reward_gap, length_gap = compare_with_expert(trajectories, eval_results)

    print("🏆 Performance Analysis:")
    if abs(reward_gap) < 5:
        print("   ✅ BC performance is close to expert!")
    else:
        print("   ⚠️  BC performance differs from expert (may need more training)")

    return eval_results


def demo_expert_vs_random():
    """演示专家策略 vs 随机策略"""
    print("\n🎯 Demo: Expert vs Random Policy")
    print("=" * 50)

    from expert_policy import HighwayExpert
    from highway.highway_env import HighwayWrapper

    # 创建专家和环境
    expert = HighwayExpert('highway-v0')
    env = HighwayWrapper('highway-v0')

    policies = {
        'Expert': lambda state: expert.get_action(state),
        'Random': lambda state: env.action_space.sample()
    }

    results = {}

    for name, policy in policies.items():
        print(f"🧪 Testing {name} policy...")

        episode_rewards = []
        for episode in range(5):  # 少量测试
            state, info = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and steps < 50:
                action = policy(env.flatten_observation(state))
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            episode_rewards.append(total_reward)

        results[name] = {
            'mean_reward': sum(episode_rewards) / len(episode_rewards),
            'episodes': len(episode_rewards)
        }
        print(".2f"
    env.close()

    # 比较结果
    expert_reward = results['Expert']['mean_reward']
    random_reward = results['Random']['mean_reward']
    improvement = expert_reward - random_reward

    print(".2f"
    return results


def main():
    """主演示函数"""
    print("🚀 Imitation Learning Demo")
    print("=" * 50)
    print("This demo will walk you through the complete imitation learning pipeline")
    print("including data collection, training, and evaluation.\n")

    try:
        # 步骤1: 数据收集
        trajectories = demo_data_collection()

        # 步骤2: 模型训练
        bc_model, model_path = demo_training(trajectories)

        # 步骤3: 模型评估
        eval_results = demo_evaluation(bc_model, trajectories)

        # 步骤4: 专家 vs 随机策略对比
        comparison_results = demo_expert_vs_random()

        # 总结
        print("\n🎉 Demo Completed Successfully!")
        print("=" * 50)
        print("📊 Summary:")
        print(f"   - Collected {len(trajectories)} expert trajectories")
        print(".4f"        print(".2f"        print(f"   - Model saved to: {model_path}")
        print("\n🚀 Next Steps:")
        print("   1. Try collecting more data: python data_collection.py --episodes 1000")
        print("   2. Train with full pipeline: python train_bc.py --epochs 100")
        print("   3. Evaluate your model: python evaluate_bc.py --model-path models/*/bc_model.pth")
        print("   4. Experiment with different environments and hyperparameters!")

        # 清理演示文件
        print("\n🧹 Cleaning up demo files...")
        import shutil
        if os.path.exists("demo_data"):
            shutil.rmtree("demo_data")
        if os.path.exists("demo_model.pth"):
            os.remove("demo_model.pth")
        print("✅ Demo cleanup completed")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    main()
