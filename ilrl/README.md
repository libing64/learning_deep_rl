# Imitation Learning + Reinforcement Learning (IL+RL)

这个项目实现了 DAgger (Dataset Aggregation) 算法，结合了模仿学习（Imitation Learning）和强化学习（Reinforcement Learning）。

## 算法原理

DAgger 算法的工作流程：

1. **初始阶段（IL）**：使用专家演示数据训练初始策略
2. **迭代阶段（IL+RL）**：
   - 使用当前策略在环境中运行，收集新轨迹
   - 让专家对新轨迹中的状态进行标注
   - 将新数据加入数据集，重新训练策略
   - 重复上述步骤，逐步减少对专家的依赖

## 项目结构

```
ilrl/
├── README.md              # 项目说明
├── dagger_algorithm.py    # DAgger算法实现
├── train_ilrl.py          # 训练脚本
└── test_ilrl.py           # 测试/评估脚本
```

## 环境要求

使用 conda 环境 `highway`，需要安装以下依赖：
- gymnasium
- highway-env
- torch
- numpy
- matplotlib

## 使用方法

### 1. 训练模型

```bash
# 基础训练
python train_ilrl.py --exp-name dagger_exp1

# 自定义配置
python train_ilrl.py --exp-name dagger_exp2 \
    --env highway-v0 \
    --initial-expert 200 \
    --iterations 10 \
    --trajectories-per-iter 100 \
    --epochs 20 \
    --batch-size 64
```

**参数说明：**
- `--exp-name`: 实验名称（必需）
- `--env`: 环境名称（默认: highway-v0）
- `--initial-expert`: 初始专家轨迹数量（默认: 100）
- `--iterations`: DAgger迭代次数（默认: 5）
- `--trajectories-per-iter`: 每次迭代收集的轨迹数（默认: 50）
- `--max-steps`: 每回合最大步数（默认: 200）
- `--epochs`: 每次迭代的训练轮数（默认: 10）
- `--batch-size`: 批次大小（默认: 64）
- `--device`: 计算设备（cpu/cuda，默认: cpu）

### 2. 测试模型

#### 2.1 性能测试 (test_ilrl.py)

```bash
# 基础测试（无渲染，快速评估）
python test_ilrl.py --model-path models/dagger_exp1_*/dagger_model.pth --episodes 10

# 可视化测试
python test_ilrl.py --model-path models/dagger_exp1_*/dagger_model.pth --render --episodes 5

# 与专家策略比较
python test_ilrl.py --model-path models/dagger_exp1_*/dagger_model.pth --compare-expert --episodes 10
```

**参数说明：**
- `--model-path`: 模型文件路径（支持通配符，必需）
- `--env`: 环境名称（默认: 从配置读取）
- `--episodes`: 测试回合数（默认: 10）
- `--render`: 启用渲染（可视化）
- `--delay`: 渲染延迟（秒，默认: 0.05）
- `--compare-expert`: 与专家策略比较性能

#### 2.2 可视化演示 (demo_ilrl.py)

```bash
# 基础演示（可视化展示模型效果）
python demo_ilrl.py --model-path models/dagger_exp1_*/dagger_model.pth --episodes 3

# 与专家策略对比演示
python demo_ilrl.py --model-path models/dagger_exp1_*/dagger_model.pth --compare-expert --episodes 3

# 自定义演示参数
python demo_ilrl.py --model-path models/dagger_exp1_*/dagger_model.pth --episodes 5 --delay 0.1
```

**参数说明：**
- `--model-path`: 模型文件路径（支持通配符，必需）
- `--env`: 环境名称（默认: 从配置读取）
- `--episodes`: 演示回合数（默认: 3）
- `--delay`: 渲染延迟（秒，默认: 0.05）
- `--compare-expert`: 与专家策略对比演示（会依次展示模型和专家的表现）

**演示脚本特点：**
- 实时可视化模型行为
- 详细的每步信息输出
- 动作分布统计
- 与专家策略的直观对比

## 训练流程

1. **初始专家数据收集**：收集专家演示轨迹
2. **初始策略训练**：在专家数据上训练初始策略
3. **DAgger迭代**：
   - 使用当前策略收集新轨迹（beta控制专家参与度）
   - 专家标注新轨迹中的状态
   - 在累积数据集上重新训练策略
   - beta值逐步衰减（从1.0到0.0），减少对专家的依赖

## 输出文件

训练完成后，会在 `models/{exp_name}_{timestamp}/` 目录下生成：

- `dagger_model.pth`: 训练好的模型
- `config.json`: 训练配置
- `training_history.json`: 训练历史
- `plots/training_curves.png`: 训练曲线图

## 算法优势

相比纯模仿学习（Behavioral Cloning）：
- **更好的泛化能力**：通过与环境交互，学习处理专家未覆盖的状态
- **减少分布偏移**：逐步减少对专家的依赖，避免分布不匹配问题
- **更高的性能**：结合IL和RL的优势，通常能达到或超过专家性能

## 示例

### 快速开始

```bash
# 1. 训练模型（使用默认参数）
python train_ilrl.py --exp-name my_first_dagger

# 2. 测试模型
python test_ilrl.py --model-path models/my_first_dagger_*/dagger_model.pth --episodes 10

# 3. 与专家比较
python test_ilrl.py --model-path models/my_first_dagger_*/dagger_model.pth --compare-expert
```

### 完整训练流程

```bash
# 步骤1: 训练（收集更多数据，更多迭代）
python train_ilrl.py --exp-name dagger_full \
    --initial-expert 200 \
    --iterations 10 \
    --trajectories-per-iter 100 \
    --epochs 20

# 步骤2: 评估性能
python test_ilrl.py --model-path models/dagger_full_*/dagger_model.pth \
    --compare-expert \
    --episodes 20

# 步骤3: 可视化演示
python test_ilrl.py --model-path models/dagger_full_*/dagger_model.pth \
    --render \
    --episodes 3
```

## 注意事项

1. **训练时间**：DAgger算法需要多次迭代，训练时间较长
2. **数据量**：随着迭代进行，数据集会不断增长，注意内存使用
3. **Beta调度**：beta值控制专家参与度，可以根据需要调整
4. **环境一致性**：测试时使用与训练相同的环境配置

## 故障排除

### 常见问题

1. **ImportError**: 确保在 `highway` conda 环境中运行
2. **CUDA错误**: 如果没有GPU，使用 `--device cpu`
3. **内存不足**: 减少 `--batch-size` 或 `--trajectories-per-iter`
4. **训练不稳定**: 调整学习率或增加训练轮数

## 扩展方向

- **改进Beta调度**：使用更复杂的衰减策略
- **自适应采样**：根据策略性能动态调整数据收集
- **多专家融合**：结合多个专家的知识
- **在线学习**：实时更新策略

## 引用

如果在研究中使用了这个实现，请引用：

```bibtex
@misc{dagger_highway,
  title={DAgger Algorithm for Highway Driving},
  author={Your Name},
  year={2024}
}
```
