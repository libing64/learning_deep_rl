#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Highway Environment Wrapper
Highway 环境封装
"""

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
import sys
import os

# 确保导入真正的 highway-env 包（不是本地模块）
# 保存当前文件目录
_current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 函数：安全导入 highway-env
def _import_highway_env():
    """安全导入 highway-env 包，避免导入本地模块"""
    # 临时修改 sys.path，优先使用 site-packages 中的 highway-env
    original_path = sys.path.copy()
    site_packages_paths = [p for p in sys.path if 'site-packages' in p]
    
    if site_packages_paths:
        # 将 site-packages 路径放在前面，移除当前目录避免导入本地模块
        new_path = site_packages_paths.copy()
        for p in sys.path:
            if p not in site_packages_paths and p != _current_file_dir:
                new_path.append(p)
        sys.path = new_path
    
    try:
        # 导入 highway-env 以注册环境
        import highway_env
        # 验证环境是否已注册
        import gymnasium as gym
        test_envs = ['highway-v0', 'merge-v0', 'roundabout-v0']
        registered = any(env_id in gym.envs.registry.keys() for env_id in test_envs)
        return registered
    except ImportError:
        return False
    except Exception:
        # 即使检查失败，如果导入成功也返回 True
        return True
    finally:
        # 恢复原始路径
        sys.path = original_path

# 在模块加载时导入 highway-env 以注册环境
_highway_env_available = _import_highway_env()


class HighwayWrapper:
    """
    Highway 环境封装类
    
    Highway-env 提供了多种高速公路驾驶场景：
    - 观察空间：通常为图像或特征向量
    - 动作空间：离散动作（车道变更、加速、减速等）
    - 目标：安全驾驶，避免碰撞，高效通行
    """
    
    def __init__(self, env_name='highway-v0', render_mode=None, config=None):
        """
        初始化 Highway 环境
        
        Args:
            env_name: 环境名称
                - 'highway-v0': 基础高速公路环境
                - 'highway-fast-v0': 快速高速公路环境
                - 'merge-v0': 并道场景
                - 'roundabout-v0': 环岛场景
                - 'parking-v0': 停车场景
                - 'intersection-v0': 交叉路口场景
            render_mode: 渲染模式 ('human' 用于可视化, None 用于训练)
            config: 环境配置字典
        """
        # 确保 highway-env 已导入并注册环境
        # 临时修改 sys.path，移除当前目录，导入 highway-env，然后恢复
        original_path = sys.path.copy()
        site_packages_paths = [p for p in sys.path if 'site-packages' in p]
        
        # 移除当前目录，避免导入本地模块
        if _current_file_dir in sys.path:
            sys.path.remove(_current_file_dir)
        
        # 如果 site-packages 不在最前面，将其移到前面
        if site_packages_paths:
            for sp in site_packages_paths:
                if sp in sys.path:
                    sys.path.remove(sp)
                sys.path.insert(0, sp)
        
        try:
            # 导入 highway-env 以注册环境
            # 使用 __import__ 确保导入真正的包
            if 'highway_env' in sys.modules:
                # 如果已经导入过（可能是本地模块），先删除
                del sys.modules['highway_env']
            __import__('highway_env')
        except ImportError as e:
            raise ImportError(
                f"highway-env is not installed or cannot be imported: {e}. "
                "Please install it with: pip install highway-env"
            )
        finally:
            # 恢复原始路径
            sys.path = original_path
        
        # 默认配置
        if config is None:
            config = self._get_default_config(env_name)
        
        # 创建环境
        try:
            self.env = gym.make(env_name, render_mode=render_mode, config=config)
        except Exception as e:
            # 提供更友好的错误信息
            available = get_available_environments()
            raise ValueError(
                f"Failed to create environment '{env_name}': {e}\n"
                f"Available environments: {available}"
            )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.env_name = env_name
        
    def _get_default_config(self, env_name):
        """获取环境的默认配置"""
        if 'highway' in env_name:
            return {
                "observation": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20]
                    },
                    "absolute": False,
                    "normalize": True,
                    "vehicles_count": 5
                },
                "action": {
                    "type": "DiscreteMetaAction"
                },
                "lanes_count": 4,
                "vehicles_count": 10,
                "duration": 40,
                "initial_spacing": 2,
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.4,
                "reward_speed_range": [20, 30],
                "normalize_reward": False
            }
        elif 'merge' in env_name:
            return {
                "observation": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "normalize": True
                },
                "action": {
                    "type": "DiscreteMetaAction"
                },
                "lanes_count": 3,
                "vehicles_count": 10,
                "duration": 40
            }
        elif 'roundabout' in env_name:
            return {
                "observation": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "normalize": True
                },
                "action": {
                    "type": "DiscreteMetaAction"
                },
                "duration": 40
            }
        elif 'intersection' in env_name:
            return {
                "observation": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "normalize": True
                },
                "action": {
                    "type": "DiscreteMetaAction"
                },
                "duration": 40
            }
        else:
            return {}
    
    def reset(self, seed=None):
        """
        重置环境到初始状态
        
        Returns:
            observation: 观察（特征向量或图像）
            info: 额外信息
        """
        if seed is not None:
            return self.env.reset(seed=seed)
        return self.env.reset()
    
    def step(self, action):
        """
        执行动作
        
        Args:
            action: 动作
                - 离散动作: 0-4 (IDLE, LANE_LEFT, LANE_RIGHT, FASTER, SLOWER)
            
        Returns:
            observation: 观察
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        return self.env.step(action)
    
    def render(self):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    def get_state_dim(self):
        """获取状态空间维度"""
        if hasattr(self.observation_space, 'shape'):
            # 如果是图像，返回展平后的维度
            if len(self.observation_space.shape) > 1:
                return int(np.prod(self.observation_space.shape))
            return int(self.observation_space.shape[0])
        elif hasattr(self.observation_space, 'spaces'):
            # 如果是字典空间，返回总维度
            total_dim = 0
            for space in self.observation_space.spaces.values():
                if hasattr(space, 'shape') and len(space.shape) > 0:
                    total_dim += int(space.shape[0])
                else:
                    total_dim += 1
            return total_dim
        # 如果无法确定，尝试从观察中推断
        try:
            obs, _ = self.env.reset()
            flattened = self.flatten_observation(obs)
            return len(flattened)
        except:
            return None
    
    def get_action_dim(self):
        """获取动作空间维度"""
        return self.action_space.n
    
    def get_observation_shape(self):
        """获取观察空间形状"""
        if hasattr(self.observation_space, 'shape'):
            return self.observation_space.shape
        return None
    
    def flatten_observation(self, obs):
        """
        将观察展平为一维向量（用于DQN）
        
        Args:
            obs: 原始观察
            
        Returns:
            flattened_obs: 展平后的观察
        """
        if isinstance(obs, dict):
            # 如果是字典，展平所有值
            return np.concatenate([np.array(obs[key]).flatten() for key in sorted(obs.keys())])
        elif isinstance(obs, np.ndarray):
            return obs.flatten()
        else:
            return np.array(obs).flatten()
    
    def __str__(self):
        return (f"Highway Environment: {self.env_name}\n"
                f"  Observation space: {self.observation_space}\n"
                f"  Action space: {self.action_space}\n"
                f"  State dimension: {self.get_state_dim()}\n"
                f"  Action dimension: {self.get_action_dim()}")


def get_available_environments():
    """获取可用的highway环境列表"""
    # 确保 highway-env 已导入
    original_path = sys.path.copy()
    site_packages_paths = [p for p in sys.path if 'site-packages' in p]
    
    # 移除当前目录，避免导入本地模块
    if _current_file_dir in sys.path:
        sys.path.remove(_current_file_dir)
    
    # 如果 site-packages 不在最前面，将其移到前面
    if site_packages_paths:
        for sp in site_packages_paths:
            if sp in sys.path:
                sys.path.remove(sp)
            sys.path.insert(0, sp)
    
    try:
        # 确保 highway-env 已导入
        if 'highway_env' in sys.modules:
            del sys.modules['highway_env']
        __import__('highway_env')
        
        # 获取所有注册的 highway 相关环境
        all_envs = [k for k in gym.envs.registry.keys() 
                   if any(x in k.lower() for x in ['highway', 'merge', 'roundabout', 'parking', 'intersection'])
                   and 'multi-agent' not in k.lower()  # 排除多智能体环境
                   and 'parked' not in k.lower()  # 排除 parked 变体
                   and 'ActionRepeat' not in k]  # 排除 ActionRepeat 变体
        # 返回主要的环境
        main_envs = ['highway-v0', 'highway-fast-v0', 'merge-v0', 'roundabout-v0', 'parking-v0', 'intersection-v0']
        # 只返回实际存在的环境
        available = [e for e in main_envs if e in all_envs]
        return available if available else main_envs
    except:
        # 如果无法检测，返回默认列表
        return [
            'highway-v0',
            'highway-fast-v0',
            'merge-v0',
            'roundabout-v0',
            'parking-v0',
            'intersection-v0'
        ]
    finally:
        # 恢复原始路径
        sys.path = original_path


def test_environment():
    """测试环境"""
    print("🧪 Testing Highway Environment...")
    
    try:
        # 测试基础highway环境
        print("\n1. Testing highway-v0:")
        env = HighwayWrapper('highway-v0')
        print(env)
        
        # 测试随机策略
        print("\n🎮 Testing random policy for 2 episodes:")
        for episode in range(2):
            state, info = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            # 展平观察
            state_flat = env.flatten_observation(state)
            print(f"  Episode {episode + 1}:")
            print(f"    Initial state shape: {state_flat.shape}")
            print(f"    State range: [{state_flat.min():.2f}, {state_flat.max():.2f}]")
            
            while not done and steps < 50:  # 限制步数用于测试
                action = env.action_space.sample()
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            print(f"    Steps = {steps}, Total Reward = {total_reward:.2f}")
        
        env.close()
        print("\n✅ Environment test completed!")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Please install highway-env: pip install highway-env")


if __name__ == "__main__":
    test_environment()

