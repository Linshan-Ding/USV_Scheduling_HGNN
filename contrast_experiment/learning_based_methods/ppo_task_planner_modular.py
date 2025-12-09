#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO算法任务规划器（模块化版本）
基于BasePlanner基类实现，支持统一的接口和数据格式

算法描述：
    使用近端策略优化算法优化USV任务分配决策
    状态表示：USV位置、电量、任务状态
    动作空间：选择下一个要执行的任务
    奖励函数：基于任务完成率和时间效率
"""

import sys
import os
import random
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_planner import BasePlanner, Task, USV
from utils import calculate_distance, simple_energy_model, DataConverter
from data_adapter import load_and_adapt_data
from typing import Dict, List, Any, Tuple


class ActorNetwork(nn.Module):
    """Actor网络（策略网络）"""
    def __init__(self, state_dim: int, action_dim: int):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=-1)


class CriticNetwork(nn.Module):
    """Critic网络（价值网络）"""
    def __init__(self, state_dim: int):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class PPOTaskPlanner(BasePlanner):
    """PPO算法任务规划器"""

    def __init__(self, config: Dict = None):
        """
        初始化PPO规划器

        Args:
            config: 算法配置参数
        """
        super().__init__(config)

        # PPO算法参数
        self.episodes = self.config.get('episodes', 60)  # 训练轮数
        self.learning_rate = self.config.get('learning_rate', 0.001)  # 学习率
        self.gamma = self.config.get('gamma', 0.95)  # 折扣因子
        self.eps_clip = self.config.get('eps_clip', 0.2)  # PPO裁剪参数
        self.k_epochs = self.config.get('k_epochs', 4)  # PPO更新次数
        self.batch_size = self.config.get('batch_size', 32)  # 批次大小
        self.memory_size = self.config.get('memory_size', 10000)  # 经验回放池大小
        self.random_seed = self.config.get('random_seed', 42)

        # 环境参数
        self.energy_cost_per_unit_distance = self.config.get('energy_cost_per_unit_distance', 1.0)
        self.task_time_energy_ratio = self.config.get('task_time_energy_ratio', 0.25)
        self.usv_initial_position = self.config.get('usv_initial_position', [0.0, 0.0])

        # 设置随机种子
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

        # 算法组件
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = []

        # 日志和状态
        self.warnings = []
        self.failures = []

    def plan(self, env_data: Dict) -> Dict:
        """
        执行PPO算法调度规划

        Args:
            env_data: 环境数据

        Returns:
            调度结果字典
        """
        # 验证环境数据
        if not self.validate_env_data(env_data):
            return {
                'success': False,
                'error': '环境数据验证失败',
                'warnings': self.warnings,
                'failures': self.failures
            }

        # 提取数据
        tasks_data = DataConverter.env_data_to_tasks(env_data)
        usvs_data = DataConverter.env_data_to_usvs(env_data)
        config = DataConverter.extract_config(env_data)

        # 转换为对象
        tasks = [Task.from_dict(t) for t in tasks_data]
        usvs = [USV.from_dict(u) for u in usvs_data]

        # 执行PPO算法
        self._execute_ppo_optimization(tasks, usvs, config)

        # 计算结果
        schedule_result = {
            'tasks': [task.to_dict() for task in tasks],
            'usvs': [usv.to_dict() for usv in usvs]
        }

        # 计算性能指标
        metrics = self.compute_basic_metrics(schedule_result)
        metrics.update({
            'warnings': self.warnings,
            'failures': self.failures,
            'algorithm': 'PPO',
            'episodes': self.episodes
        })

        # 保存结果
        self.results = {
            'schedule': schedule_result,
            'makespan': metrics['makespan'],
            'metrics': metrics,
            'success': len(tasks) > 0
        }

        return self.results

    def _execute_ppo_optimization(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """执行PPO优化算法"""
        print(f"PPO算法开始优化 - 训练轮数: {self.episodes}, 设备: {self.device}")

        # 初始化网络
        state_dim = self._get_state_dim(tasks, usvs)
        action_dim = len(tasks) * len(usvs)  # 每个USV-任务组合都是一个动作

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        best_solution = None
        best_makespan = float('inf')

        # 训练循环
        for episode in range(self.episodes):
            # 重置环境
            episode_tasks = copy.deepcopy(tasks)
            episode_usvs = copy.deepcopy(usvs)

            # 执行一个episode，收集经验
            episode_reward, assigned_count, episode_data = self._run_episode(episode_tasks, episode_usvs, config)

            # 计算makespan
            makespan = max(usv.current_time for usv in episode_usvs)

            # 记录最佳解
            if assigned_count > 0 and makespan < best_makespan:
                best_makespan = makespan
                best_solution = self._extract_solution(episode_tasks)

            # 存储经验
            self.memory.extend(episode_data)

            # 定期更新网络
            if len(self.memory) >= self.batch_size and episode % 2 == 0:
                self._update_network()

            # 限制内存大小
            if len(self.memory) > self.memory_size:
                self.memory = self.memory[-self.memory_size:]

            if episode % 10 == 0:
                print(f"  PPO轮数 {episode+1}/{self.episodes}, 奖励: {episode_reward:.2f}, "
                      f"任务完成: {assigned_count}/{len(tasks)}, 最佳makespan: {best_makespan:.2f}")

        # 应用最佳解
        if best_solution is not None:
            self._apply_best_solution(tasks, usvs, best_solution, config)

        print(f"PPO算法优化完成，最佳makespan: {best_makespan:.2f}")

    def _get_state_dim(self, tasks: List[Task], usvs: List[USV]) -> int:
        """计算状态维度"""
        # USV状态：位置(2) + 电量(1) + 当前时间(1) = 4维/USV
        # 任务状态：位置(2) + 是否分配(1) + 开始时间(1) + 完成时间(1) = 5维/任务
        return len(usvs) * 4 + len(tasks) * 5

    def _get_state(self, tasks: List[Task], usvs: List[USV]) -> np.ndarray:
        """获取当前状态"""
        state = []

        # USV状态：位置、电量、当前时间
        for usv in usvs:
            state.extend([usv.position[0], usv.position[1], usv.battery_level, usv.current_time])

        # 任务状态：位置、是否已分配、开始时间、完成时间
        for task in tasks:
            assigned = 1 if task.assigned_usv is not None else 0
            start_time = task.start_time if task.start_time is not None else 0
            finish_time = task.finish_time if task.finish_time is not None else 0
            state.extend([task.position[0], task.position[1], assigned, start_time, finish_time])

        return np.array(state, dtype=np.float32)

    def _run_episode(self, tasks: List[Task], usvs: List[USV], config: Dict) -> Tuple[float, int, List]:
        """运行一个episode"""
        # 重置状态
        for usv in usvs:
            usv.battery_level = usv.battery_capacity
            usv.current_time = 0.0
            usv.position = config['env_paras']['start_point'].copy()
            usv.timeline = []

        for task in tasks:
            task.assigned_usv = None
            task.start_time = None
            task.finish_time = None

        total_reward = 0.0
        assigned_count = 0
        episode_data = []

        # 为每个USV分配任务
        for usv in usvs:
            while True:
                state = self._get_state(tasks, usvs)
                available_tasks = [t for t in tasks if t.assigned_usv is None]

                # 如果没有可用任务，结束该USV的分配
                if not available_tasks:
                    break

                # 选择动作
                action_idx = self._select_action(state, usv.usv_id, available_tasks, tasks)
                if action_idx == -1:  # 无效动作
                    break

                # 执行动作
                task_idx = action_idx % len(available_tasks)
                selected_task = available_tasks[task_idx]

                # 检查是否可以执行任务
                if self._can_execute_task(usv, selected_task, config):
                    self._execute_task(usv, selected_task)
                    assigned_count += 1

                    # 计算奖励
                    reward = 100.0  # 完成任务奖励
                    reward -= selected_task.service_time * 0.1  # 时间惩罚
                    total_reward += reward

                    # 存储经验
                    next_state = self._get_state(tasks, usvs)
                    done = len([t for t in tasks if t.assigned_usv is None]) == 0
                    episode_data.append((state, action_idx, reward, next_state, done, usv.usv_id))

                else:
                    # 不能执行任务的惩罚
                    reward = -10.0
                    total_reward += reward

        return total_reward, assigned_count, episode_data

    def _select_action(self, state: np.ndarray, usv_id: int, available_tasks: List[Task], all_tasks: List[Task]) -> int:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)

            # 只考虑可用任务的动作
            valid_actions = []
            for i, task in enumerate(all_tasks):
                if task in available_tasks:
                    valid_actions.append(usv_id * len(all_tasks) + i)

            if not valid_actions:
                return -1

            valid_probs = action_probs[0][valid_actions]
            # 确保概率和为正数
            if torch.sum(valid_probs) <= 0:
                # 如果概率和无效，随机选择
                return random.choice(valid_actions)

            # 归一化概率
            valid_probs = valid_probs / torch.sum(valid_probs)
            return valid_actions[torch.multinomial(valid_probs, 1).item()]

    def _can_execute_task(self, usv: USV, task: Task, config: Dict) -> bool:
        """检查是否可以执行任务"""
        travel_distance = calculate_distance(usv.position, task.position)
        energy_needed = simple_energy_model(
            travel_distance, task.service_time,
            self.energy_cost_per_unit_distance, self.task_time_energy_ratio
        )

        # 检查是否有足够的能量
        if usv.battery_level < energy_needed:
            # 检查是否可以先充电
            distance_to_base = calculate_distance(usv.position, config['env_paras']['start_point'])
            energy_to_base = distance_to_base * self.energy_cost_per_unit_distance
            return usv.battery_level >= energy_to_base

        return usv.battery_level >= energy_needed

    def _execute_task(self, usv: USV, task: Task):
        """执行任务"""
        travel_distance = calculate_distance(usv.position, task.position)
        travel_time = travel_distance / usv.speed
        start_time = usv.current_time + travel_time
        finish_time = start_time + task.service_time

        usv.current_time = finish_time
        usv.position = task.position

        task.assigned_usv = usv.usv_id
        task.start_time = start_time
        task.finish_time = finish_time

        usv.timeline.append({
            'type': 'task',
            'task_id': task.task_id,
            'start_service': start_time,
            'finish_service': finish_time
        })

    def _update_network(self):
        """更新Actor和Critic网络"""
        if len(self.memory) < self.batch_size:
            return

        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, usv_ids = zip(*batch)

        # 转换为tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算优势
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages = rewards + self.gamma * next_values * (1 - dones) - values

        # PPO更新
        for _ in range(self.k_epochs):
            # Actor损失
            action_probs = self.actor(states)
            action_log_probs = torch.log(action_probs.gather(1, actions) + 1e-8)
            ratio = torch.exp(action_log_probs - action_log_probs.detach())

            # 计算裁剪的损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic损失
            values = self.critic(states)
            returns = rewards + self.gamma * self.critic(next_states).detach() * (1 - dones)
            critic_loss = F.mse_loss(values, returns)

            # 更新网络
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

    def _extract_solution(self, tasks: List[Task]) -> List[int]:
        """从任务中提取分配方案"""
        solution = []
        for task in tasks:
            solution.append(task.assigned_usv if task.assigned_usv is not None else -1)
        return solution

    def _apply_best_solution(self, tasks: List[Task], usvs: List[USV], best_solution: List[int], config: Dict):
        """应用最佳解"""
        # 重置状态
        for usv in usvs:
            usv.battery_level = usv.battery_capacity
            usv.current_time = 0.0
            usv.position = config['env_paras']['start_point'].copy()
            usv.timeline = []

        for task in tasks:
            task.assigned_usv = None
            task.start_time = None
            task.finish_time = None

        # 应用最佳解
        for i, (task, usv_idx) in enumerate(zip(tasks, best_solution)):
            if usv_idx >= 0 and usv_idx < len(usvs):
                usv = usvs[usv_idx]
                if self._can_execute_task(usv, task, config):
                    self._execute_task(usv, task)
                else:
                    self.failures.append(f"任务 {task.task_id} 无法分配给USV {usv_idx}")


# 兼容性接口
def run_single_case(json_file: str, config: Dict = None) -> Dict:
    """运行单个测试案例"""
    if config is None:
        config = {
            'episodes': 50,         # 适中的训练轮数
            'learning_rate': 0.001,
            'gamma': 0.95,
            'eps_clip': 0.2,
            'k_epochs': 4,
            'batch_size': 32,
            'memory_size': 10000,
            'random_seed': 42,
            'energy_cost_per_unit_distance': 1.0,
            'task_time_energy_ratio': 0.25,
            'usv_initial_position': [0.0, 0.0]
        }

    planner = PPOTaskPlanner(config)
    env_data = load_and_adapt_data(json_file)
    return planner.plan(env_data)


def main():
    """主函数，用于测试"""
    # 测试案例文件
    test_case_file = "../../usv_data_dev/40_8/usv_case_40_8_instance_01.json"

    if os.path.exists(test_case_file):
        print(f"运行PPO算法测试案例: {test_case_file}")
        result = run_single_case(test_case_file)

        if result['success']:
            makespan = result.get('makespan')
            if makespan is not None:
                print(f"PPO调度成功！完成时间: {makespan:.2f}")
            else:
                print("PPO调度成功但没有完成时间")
            print(f"已分配任务: {result['metrics']['assigned_tasks']}")
            print(f"未分配任务: {result['metrics']['unassigned_tasks']}")
        else:
            print("PPO调度失败！")
            if result.get('error'):
                print(f"错误信息: {result['error']}")
    else:
        print(f"测试案例文件不存在: {test_case_file}")


if __name__ == "__main__":
    main()