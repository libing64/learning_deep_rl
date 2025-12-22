#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert Policy for Imitation Learning
模仿学习专家策略

这个模块实现了在 highway 环境中表现良好的专家策略，
用于生成演示数据供模仿学习算法使用。
"""

import numpy as np
import sys
import os

# 添加父目录到路径，以便导入 highway_env 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highway.highway_env import HighwayWrapper


class HighwayExpert:
    """
    Highway 环境专家策略

    专家策略基于规则实现安全高效的驾驶行为：
    - 保持在右侧车道（如果可能）
    - 保持安全距离
    - 适当加速以保持流量
    - 避免碰撞
    """

    def __init__(self, env_name='highway-v0', config=None):
        """
        初始化专家策略

        Args:
            env_name: 环境名称
            config: 环境配置
        """
        self.env_name = env_name
        self.config = config or self._get_expert_config()
        self.env = HighwayWrapper(env_name, config=self.config)

        # 专家策略参数
        self.min_safe_distance = 10.0  # 最小安全距离
        self.preferred_speed = 25.0    # 偏好速度
        self.speed_tolerance = 5.0     # 速度容忍度
        self.lane_change_threshold = 15.0  # 换道阈值

    def _get_expert_config(self):
        """获取专家策略的推荐配置"""
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
            "vehicles_count": 15,  # 更多的车辆使环境更具挑战性
            "duration": 60,  # 更长的 episode
            "initial_spacing": 2,
            "collision_reward": -5,  # 更高的碰撞惩罚
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.6,
            "reward_speed_range": [20, 30],
            "normalize_reward": False
        }

    def get_action(self, observation):
        """
        基于观察选择动作

        Args:
            observation: 环境观察

        Returns:
            action: 选择的动作 (0-4)
                0: IDLE - 保持当前状态
                1: LANE_LEFT - 左换道
                2: LANE_RIGHT - 右换道
                3: FASTER - 加速
                4: SLOWER - 减速
        """
        # 解析观察数据
        vehicles = self._parse_observation(observation)
        ego_vehicle = vehicles[0]  # 自己的车辆

        # 获取前方车辆信息
        front_vehicle = self._get_front_vehicle(vehicles, ego_vehicle)

        # 决策逻辑
        action = self._decision_logic(ego_vehicle, front_vehicle, vehicles)

        return action

    def _parse_observation(self, observation):
        """
        解析观察数据

        Args:
            observation: 观察数据 (numpy array 或 dict)

        Returns:
            vehicles: 车辆列表，每个车辆包含 [presence, x, y, vx, vy, cos_h, sin_h]
        """
        if isinstance(observation, dict):
            # 如果是字典格式
            features = []
            for key in sorted(observation.keys()):
                if key.startswith('vehicles_'):
                    features.append(observation[key])
            obs_array = np.array(features)
        else:
            # 如果是数组格式
            obs_array = np.array(observation)

        # 处理不同的观察格式
        if len(obs_array.shape) == 2:
            # 如果已经是二维数组 (vehicles_count, features_per_vehicle)
            obs_reshaped = obs_array
        elif len(obs_array.shape) == 1:
            # 如果是一维数组，需要重塑为 (vehicles_count, features_per_vehicle)
            vehicles_count = obs_array.shape[0] // 7  # 7 个特征 per vehicle
            if vehicles_count == 0:
                raise ValueError(
                    f"Cannot reshape observation array of size {obs_array.shape[0]} "
                    f"into shape (vehicles_count, 7). "
                    f"Observation shape: {obs_array.shape}, "
                    f"First 10 values: {obs_array[:10]}"
                )
            obs_reshaped = obs_array.reshape(vehicles_count, 7)
        else:
            raise ValueError(
                f"Unexpected observation shape: {obs_array.shape}. "
                f"Expected 1D or 2D array."
            )

        vehicles = []
        for i in range(obs_reshaped.shape[0]):
            vehicle = obs_reshaped[i]
            vehicles.append({
                'presence': vehicle[0],
                'x': vehicle[1],
                'y': vehicle[2],
                'vx': vehicle[3],
                'vy': vehicle[4],
                'cos_h': vehicle[5],
                'sin_h': vehicle[6]
            })

        return vehicles

    def _get_front_vehicle(self, vehicles, ego_vehicle):
        """
        获取前方车辆信息

        Args:
            vehicles: 所有车辆列表
            ego_vehicle: 自己的车辆

        Returns:
            front_vehicle: 前方车辆信息或 None
        """
        ego_lane = self._get_vehicle_lane(ego_vehicle)
        min_distance = float('inf')
        front_vehicle = None

        for vehicle in vehicles[1:]:  # 跳过自己的车辆
            if vehicle['presence'] < 0.5:  # 车辆不存在
                continue

            vehicle_lane = self._get_vehicle_lane(vehicle)

            # 只考虑同车道的车辆
            if vehicle_lane != ego_lane:
                continue

            # 计算相对距离 (前方为正)
            distance = vehicle['x'] - ego_vehicle['x']

            if 0 < distance < min_distance:
                min_distance = distance
                front_vehicle = vehicle

        return front_vehicle

    def _get_vehicle_lane(self, vehicle):
        """
        获取车辆所在车道

        Args:
            vehicle: 车辆信息

        Returns:
            lane: 车道编号 (0 为最右车道)
        """
        # 基于 y 坐标确定车道
        # highway-env 中车道宽度通常为 4
        lane_width = 4.0
        lane = int(round(vehicle['y'] / lane_width))

        # 确保车道编号在有效范围内
        max_lanes = self.config.get('lanes_count', 4)
        lane = max(0, min(lane, max_lanes - 1))

        return lane

    def _decision_logic(self, ego_vehicle, front_vehicle, all_vehicles):
        """
        决策逻辑

        Args:
            ego_vehicle: 自己的车辆
            front_vehicle: 前方车辆
            all_vehicles: 所有车辆

        Returns:
            action: 选择的动作
        """
        ego_speed = ego_vehicle['vx']
        ego_lane = self._get_vehicle_lane(ego_vehicle)

        # 1. 检查前方是否有车辆
        if front_vehicle is not None:
            front_distance = front_vehicle['x'] - ego_vehicle['x']
            front_speed = front_vehicle['vx']

            # 如果距离太近，需要减速或换道
            if front_distance < self.min_safe_distance:
                # 尝试换到右侧车道（如果不是已经在最右车道）
                if ego_lane > 0 and self._can_change_lane(ego_vehicle, all_vehicles, direction='right'):
                    return 2  # LANE_RIGHT
                else:
                    return 4  # SLOWER - 减速

            # 如果前方车辆较慢，考虑超车
            elif front_speed < ego_speed - 2 and front_distance < self.lane_change_threshold:
                # 尝试换到左侧车道超车
                if ego_lane < self.config.get('lanes_count', 4) - 1:
                    if self._can_change_lane(ego_vehicle, all_vehicles, direction='left'):
                        return 1  # LANE_LEFT

        # 2. 速度控制
        if ego_speed < self.preferred_speed - self.speed_tolerance:
            return 3  # FASTER - 加速
        elif ego_speed > self.preferred_speed + self.speed_tolerance:
            return 4  # SLOWER - 减速

        # 3. 车道选择：倾向于右车道
        if ego_lane > 0 and self._can_change_lane(ego_vehicle, all_vehicles, direction='right'):
            # 检查右车道是否更畅通
            right_lane_speed = self._get_lane_average_speed(all_vehicles, ego_lane - 1)
            current_lane_speed = self._get_lane_average_speed(all_vehicles, ego_lane)

            if right_lane_speed > current_lane_speed + 2:
                return 2  # LANE_RIGHT

        # 4. 默认保持当前状态
        return 0  # IDLE

    def _can_change_lane(self, ego_vehicle, all_vehicles, direction='left'):
        """
        检查是否可以安全换道

        Args:
            ego_vehicle: 自己的车辆
            all_vehicles: 所有车辆
            direction: 换道方向 ('left' 或 'right')

        Returns:
            can_change: 是否可以换道
        """
        ego_lane = self._get_vehicle_lane(ego_vehicle)
        target_lane = ego_lane + (1 if direction == 'right' else -1)

        # 检查目标车道是否有效
        max_lanes = self.config.get('lanes_count', 4)
        if target_lane < 0 or target_lane >= max_lanes:
            return False

        # 检查目标车道是否有车辆太近
        for vehicle in all_vehicles[1:]:
            if vehicle['presence'] < 0.5:
                continue

            vehicle_lane = self._get_vehicle_lane(vehicle)
            if vehicle_lane != target_lane:
                continue

            # 计算相对距离
            distance = abs(vehicle['x'] - ego_vehicle['x'])
            if distance < self.min_safe_distance * 0.8:  # 更严格的安全距离
                return False

        return True

    def _get_lane_average_speed(self, vehicles, lane):
        """
        获取车道平均速度

        Args:
            vehicles: 所有车辆
            lane: 车道编号

        Returns:
            avg_speed: 平均速度
        """
        lane_speeds = []
        for vehicle in vehicles[1:]:  # 跳过自己的车辆
            if vehicle['presence'] < 0.5:
                continue

            vehicle_lane = self._get_vehicle_lane(vehicle)
            if vehicle_lane == lane:
                lane_speeds.append(vehicle['vx'])

        return np.mean(lane_speeds) if lane_speeds else self.preferred_speed


def generate_expert_trajectories(env_name='highway-v0', num_episodes=100, max_steps=200):
    """
    生成专家轨迹数据

    Args:
        env_name: 环境名称
        num_episodes: 回合数量
        max_steps: 每回合最大步数

    Returns:
        trajectories: 轨迹列表，每个轨迹包含 states, actions, rewards
    """
    expert = HighwayExpert(env_name)
    trajectories = []

    print(f"🎯 Generating {num_episodes} expert trajectories...")

    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(f"  Episode {episode}/{num_episodes}")

        states = []
        actions = []
        rewards = []

        # 重置环境
        state, info = expert.env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            # 记录状态
            states.append(state.copy() if hasattr(state, 'copy') else state)

            # 选择动作
            action = expert.get_action(state)
            actions.append(action)

            # 执行动作
            next_state, reward, terminated, truncated, info = expert.env.step(action)
            done = terminated or truncated

            rewards.append(reward)

            state = next_state
            step += 1

        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'episode_length': len(states),
            'total_reward': sum(rewards)
        })

    expert.env.close()

    print("✅ Expert trajectories generated!")
    print(".1f")
    return trajectories


def test_expert_policy(env_name='highway-v0', num_episodes=5, render=False):
    """
    测试专家策略性能

    Args:
        env_name: 环境名称
        num_episodes: 测试回合数
        render: 是否渲染
    """
    expert = HighwayExpert(env_name)

    print("🧪 Testing expert policy...")

    for episode in range(num_episodes):
        state, info = expert.env.reset()
        total_reward = 0
        steps = 0
        done = False

        print(f"\nEpisode {episode + 1}:")
        while not done and steps < 100:  # 限制测试步数
            action = expert.get_action(state)

            if render:
                expert.env.render()

            state, reward, terminated, truncated, info = expert.env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

        print(f"  Steps: {steps}, Total Reward: {total_reward:.2f}")

    expert.env.close()
    print("✅ Expert policy test completed!")


if __name__ == "__main__":
    # 测试专家策略
    test_expert_policy(num_episodes=2, render=False)

    # 生成少量演示数据用于测试
    trajectories = generate_expert_trajectories(num_episodes=5, max_steps=50)

    print(f"\n📊 Generated {len(trajectories)} trajectories")
    for i, traj in enumerate(trajectories):
        print(f"  Trajectory {i}: {traj['episode_length']} steps, reward: {traj['total_reward']:.2f}")
