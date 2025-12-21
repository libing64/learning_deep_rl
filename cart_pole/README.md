# CartPole DQN 强化学习项目

## 项目说明

1. 创建 CartPole 的仿真环境
2. 基于 DQN 的方式训练强化学习模型
3. 使用训练好的模型进行 CartPole 控制测试
4. 使用 conda env: deep-rl-class，需要的 Python 库都已经装好了

## 文件说明

- `cart_pole_env.py`: CartPole 环境封装类
- `dqn_train.py`: DQN 训练脚本
- `test_model.py`: 模型测试脚本（可视化控制）

## 使用方法

### 1. 训练模型

```bash
python dqn_train.py
```

训练完成后会生成：
- `dqn_model_final.pth`: 最终模型
- `dqn_model_episode_*.pth`: 中间检查点
- `training_history.png`: 训练历史曲线图

### 2. 测试模型（无可视化）

```bash
python test_model.py --model dqn_model_final.pth
```

或者指定测试回合数：
```bash
python test_model.py --model dqn_model_final.pth --episodes 20
```

### 3. 演示控制（可视化）

```bash
python test_model.py --model dqn_model_final.pth --demo
```

或者使用 `--render` 参数：
```bash
python test_model.py --model dqn_model_final.pth --render
```

### 4. 测试环境

```bash
python cart_pole_env.py
```

## 参数说明

### 训练参数（dqn_train.py）
- `num_episodes`: 训练回合数（默认：500）
- `max_steps`: 每回合最大步数（默认：500）
- `save_interval`: 保存模型的间隔（默认：50）

### 测试参数（test_model.py）
- `--model, -m`: 模型文件路径（默认：dqn_model_final.pth）
- `--episodes, -e`: 测试回合数（默认：10）
- `--demo, -d`: 运行演示模式（可视化）
- `--render, -r`: 启用渲染

## 预期结果

训练良好的模型应该能够：
- 平均奖励 >= 475（接近满分 500）
- 成功率（Reward >= 475）>= 80%
- 能够稳定控制 CartPole 保持平衡