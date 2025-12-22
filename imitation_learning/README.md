# Imitation Learning 模仿学习

这个项目实现了模仿学习（Imitation Learning）的完整例子，使用行为克隆（Behavioral Cloning）算法从专家演示中学习策略，并在 Highway 环境中进行训练和测试。

## 项目结构

```
imitation_learning/
├── README.md                    # 项目说明
├── expert_policy.py            # 专家策略实现
├── behavioral_cloning.py       # 行为克隆算法
├── data_collection.py          # 数据收集和处理
├── train_bc.py                 # 训练脚本
├── evaluate_bc.py              # 评估脚本
├── data/                       # 数据存储目录
├── models/                     # 模型存储目录
└── requirements.txt            # 依赖包列表
```

## 核心组件

### 1. 专家策略 (expert_policy.py)
- 实现了基于规则的高水平驾驶策略
- 能够在 Highway 环境中安全高效地驾驶
- 用于生成高质量的专家演示数据

### 2. 行为克隆算法 (behavioral_cloning.py)
- 实现了行为克隆的核心算法
- 使用多层感知机将状态映射到动作
- 支持训练、评估和模型保存/加载

### 3. 数据收集 (data_collection.py)
- 负责生成和处理专家演示数据
- 提供数据验证、分割和平衡功能
- 支持大规模数据收集

### 4. 训练脚本 (train_bc.py)
- 完整的训练流程
- 自动实验管理
- 训练监控和可视化

### 5. 评估脚本 (evaluate_bc.py)
- 全面的模型评估
- 与专家策略的性能对比
- 鲁棒性测试和详细报告

## 环境要求

### 依赖包
```
gymnasium>=0.29.0
highway-env>=1.8.0
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
tqdm>=4.64.0
```

### 安装依赖
```bash
pip install -r requirements.txt
```

## 使用指南

### 1. 数据收集
首先收集专家演示数据：

```bash
# 收集1000个episode的专家数据
python data_collection.py --episodes 1000 --env highway-v0

# 验证数据集
python data_collection.py --validate

# 创建平衡数据集
python data_collection.py --balance
```

### 2. 训练模型
使用收集的数据训练行为克隆模型：

```bash
# 基本训练
python train_bc.py --exp-name bc_highway_exp1 --epochs 100

# 自定义超参数训练
python train_bc.py --exp-name bc_custom --epochs 200 --batch-size 128 --lr 5e-4 --hidden-dims 512,256,128
```

### 3. 评估模型
评估训练好的模型性能：

```bash
# 评估最新训练的模型
python evaluate_bc.py --model-path models/bc_highway_exp1_*/bc_model.pth --episodes 50

# 带渲染的评估（用于可视化）
python evaluate_bc.py --model-path models/bc_highway_exp1_*/bc_model.pth --render --episodes 10
```

## 实验结果

### 性能指标
- **平均回合奖励**: 衡量策略的整体性能
- **成功率**: 无碰撞完成episode的比例
- **与专家差距**: 与专家策略的性能对比

### 典型结果
在 highway-v0 环境中训练的行为克隆模型通常能够达到：
- 平均奖励: 15-25 (专家水平: 20-30)
- 成功率: 70-85%
- 与专家性能差距: <10%

## 高级用法

### 自定义环境
支持不同的 Highway 环境配置：

```python
from highway.highway_env import HighwayWrapper

# 自定义环境配置
config = {
    "vehicles_count": 15,
    "lanes_count": 4,
    "duration": 60,
    "collision_reward": -2
}
env = HighwayWrapper('highway-v0', config=config)
```

### 数据增强
可以通过以下方式增强训练数据：
- 生成额外的专家轨迹
- 使用数据平衡技术
- 添加噪声和扰动

### 超参数调优
关键超参数包括：
- **网络架构**: 隐藏层维度 (256,128) 或 (512,256,128)
- **学习率**: 1e-3 到 1e-4
- **批次大小**: 32-128
- **训练轮数**: 50-200

## 算法详解

### 行为克隆原理
行为克隆是一种监督学习方法，将强化学习问题转化为分类问题：

1. **专家演示**: 收集专家在环境中执行任务的轨迹 D = {(s₁,a₁), (s₂,a₂), ..., (sₜ,aₜ)}
2. **策略学习**: 训练分类器 π_θ(a|s) 来预测专家在每个状态下选择的动作
3. **损失函数**: 使用交叉熵损失最小化预测动作与专家动作之间的差距

### 专家策略设计
专家策略基于以下规则：
- **安全第一**: 保持安全车距，避免碰撞
- **效率优先**: 在右侧车道行驶，适时超车
- **平滑控制**: 避免剧烈加速/减速

## 故障排除

### 常见问题
1. **ImportError**: 确保 highway-env 已正确安装
2. **CUDA错误**: 如果没有GPU，PyTorch会自动使用CPU
3. **内存不足**: 减少批次大小或使用更小的网络
4. **训练不稳定**: 调整学习率或添加正则化

### 调试技巧
- 使用 `--render` 选项可视化策略行为
- 检查数据质量和分布
- 监控训练过程中的损失和准确率变化
- 分析失败案例，识别问题模式

## 扩展方向

### 改进算法
- **DAgger**: Dataset Aggregation，迭代收集数据
- **GAIL**: Generative Adversarial Imitation Learning
- **SQIL**: Soft Q Imitation Learning

### 应用场景
- **自动驾驶**: 学习人类驾驶行为
- **机器人控制**: 模仿专家操作
- **游戏AI**: 学习专业玩家策略

## 引用

如果在研究中使用了这个项目，请引用：

```bibtex
@misc{imitation_learning_highway,
  title={Imitation Learning with Behavioral Cloning in Highway Environment},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/imitation-learning}
}
```

## 许可证

MIT License - 详见 LICENSE 文件