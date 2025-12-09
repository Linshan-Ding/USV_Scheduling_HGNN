#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子群算法任务规划器（模块化版本）
基于BasePlanner基类实现，支持统一的接口和数据格式

算法描述：
    使用粒子群算法优化USV任务分配
    粒子代表任务分配方案
    通过速度和位置更新搜索最优解
    适应度函数基于总完成时间和任务完成率
"""

import sys
import os
import random
import numpy as np
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_planner import BasePlanner, Task, USV
from utils import calculate_distance, simple_energy_model, DataConverter
from data_adapter import load_and_adapt_data
from typing import Dict, List, Any, Tuple


class PSOTaskPlanner(BasePlanner):
    """粒子群算法任务规划器"""

    def __init__(self, config: Dict = None):
        """
        初始化粒子群算法规划器

        Args:
            config: 算法配置参数
        """
        super().__init__(config)

        # PSO算法参数
        self.swarm_size = self.config.get('swarm_size', 40)  # 粒子群大小
        self.iterations = self.config.get('iterations', 30)  # 迭代次数
        self.inertia_weight_max = self.config.get('inertia_weight_max', 0.9)  # 最大惯性权重
        self.inertia_weight_min = self.config.get('inertia_weight_min', 0.4)  # 最小惯性权重
        self.c1 = self.config.get('c1', 2.0)  # 认知学习因子
        self.c2 = self.config.get('c2', 2.0)  # 社会学习因子
        self.random_seed = self.config.get('random_seed', 42)

        # 环境参数
        self.energy_cost_per_unit_distance = self.config.get('energy_cost_per_unit_distance', 1.0)
        self.task_time_energy_ratio = self.config.get('task_time_energy_ratio', 0.25)
        self.usv_initial_position = self.config.get('usv_initial_position', [0.0, 0.0])

        # 设置随机种子
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # 算法状态
        self.particles = []      # 粒子群
        self.velocities = []     # 速度
        self.personal_best = []  # 个体最优位置
        self.personal_best_fitness = []  # 个体最优适应度
        self.global_best = None  # 全局最优位置
        self.global_best_fitness = float('inf')  # 全局最优适应度

        # 日志和状态
        self.warnings = []
        self.failures = []

    def plan(self, env_data: Dict) -> Dict:
        """
        执行粒子群算法调度规划

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

        # 执行PSO算法
        self._execute_pso_optimization(tasks, usvs, config)

        # 使用最优解执行调度
        if self.global_best is not None:
            self._apply_solution(tasks, usvs, config)

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
            'algorithm': 'PSO',
            'swarm_size': self.swarm_size,
            'iterations': self.iterations,
            'best_fitness': self.global_best_fitness
        })

        # 保存结果
        self.results = {
            'schedule': schedule_result,
            'makespan': metrics['makespan'],
            'metrics': metrics,
            'success': len(tasks) > 0 and self.global_best is not None
        }

        return self.results

    def _execute_pso_optimization(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """执行粒子群算法优化"""
        print(f"PSO算法开始优化 - 粒子群大小: {self.swarm_size}, 迭代次数: {self.iterations}")

        # 初始化粒子群
        self._initialize_swarm(tasks, usvs, config)

        # 主循环
        for iteration in range(self.iterations):
            # 计算当前惯性权重（线性递减）
            inertia_weight = (self.inertia_weight_max - self.inertia_weight_min) * \
                           (self.iterations - iteration) / self.iterations + self.inertia_weight_min

            # 更新每个粒子
            for i in range(self.swarm_size):
                # 评估当前粒子适应度
                fitness = self._evaluate_fitness(self.particles[i], tasks, usvs, config)

                # 更新个体最优
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = copy.deepcopy(self.particles[i])

                # 更新全局最优
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = copy.deepcopy(self.particles[i])

            # 更新粒子速度和位置
            for i in range(self.swarm_size):
                self._update_particle(i, inertia_weight, tasks, usvs, config)

            if iteration % 5 == 0:
                print(f"  PSO迭代 {iteration+1}/{self.iterations}, 最佳适应度: {self.global_best_fitness:.2f}")

        print(f"PSO算法优化完成，最佳适应度: {self.global_best_fitness:.2f}")

    def _initialize_swarm(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """初始化粒子群"""
        self.particles = []
        self.velocities = []
        self.personal_best = []
        self.personal_best_fitness = []

        for _ in range(self.swarm_size):
            # 生成随机粒子位置（任务-USV分配）
            position = [random.randint(0, len(usvs) - 1) for _ in tasks]
            self.particles.append(position)

            # 初始化速度（随机整数）
            velocity = [random.randint(-1, 1) for _ in tasks]
            self.velocities.append(velocity)

            # 初始化个体最优
            self.personal_best.append(position.copy())
            fitness = self._evaluate_fitness(position, tasks, usvs, config)
            self.personal_best_fitness.append(fitness)

            # 更新全局最优
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = position.copy()

    def _evaluate_fitness(self, position: List[int], tasks: List[Task], usvs: List[USV], config: Dict) -> float:
        """评估粒子位置的适应度"""
        # 创建临时副本
        temp_tasks = copy.deepcopy(tasks)
        temp_usvs = copy.deepcopy(usvs)

        # 应用位置
        self._apply_position_to_temp(temp_tasks, temp_usvs, position, config)

        # 计算总完成时间
        max_time = 0
        for usv in temp_usvs:
            if usv.current_time > max_time:
                max_time = usv.current_time

        # 计算未完成任务数
        unassigned = sum(1 for task in temp_tasks if task.assigned_usv is None)

        # 适应度 = 完成时间 + 未完成任务惩罚（适度惩罚）
        fitness = max_time + unassigned * 1000  # 适度的未完成任务惩罚

        # 大幅奖励完成更多任务的解
        completion_rate = (len(tasks) - unassigned) / len(tasks)
        if completion_rate > 0.9:  # 完成90%以上任务
            fitness *= 0.5  # 给予50%的奖励
        elif completion_rate > 0.8:  # 完成80%以上任务
            fitness *= 0.7  # 给予30%的奖励
        elif completion_rate > 0.7:  # 完成70%以上任务
            fitness *= 0.85  # 给予15%的奖励

        return fitness

    def _apply_position_to_temp(self, tasks: List[Task], usvs: List[USV], position: List[int], config: Dict):
        """将粒子位置应用到临时任务和USV（支持充电）"""
        # 重置USV状态
        for usv in usvs:
            usv.battery_level = usv.battery_capacity
            usv.current_time = 0.0
            usv.position = config['env_paras']['start_point'].copy()
            usv.timeline = []

        for task in tasks:
            task.assigned_usv = None
            task.start_time = None
            task.finish_time = None

        # 按位置分配任务（使用带充电功能的执行逻辑）
        for i, task in enumerate(tasks):
            if i < len(position):
                usv_idx = position[i]
                if usv_idx < len(usvs):
                    usv = usvs[usv_idx]
                    if self._can_execute_task_simple(usv, task, config):
                        self._execute_task_simple(usv, task, config)

    def _can_execute_task_simple(self, usv: USV, task: Task, config: Dict) -> bool:
        """简化版本的任务执行能力检查（支持充电）"""
        travel_distance = calculate_distance(usv.position, task.position)
        energy_needed = simple_energy_model(
            travel_distance, task.service_time,
            self.energy_cost_per_unit_distance, self.task_time_energy_ratio
        )

        # 如果电量不足，检查是否可以先充电再执行
        if usv.battery_level < energy_needed:
            # 计算返回充电站所需能量
            distance_to_base = calculate_distance(usv.position, config['env_paras']['start_point'])
            energy_to_base = distance_to_base * self.energy_cost_per_unit_distance

            # 如果能返回充电站，就认为可以执行任务（先充电）
            return usv.battery_level >= energy_to_base

        return usv.battery_level >= energy_needed

    def _execute_task_simple(self, usv: USV, task: Task, config: Dict):
        """简化版本的任务执行（支持自动充电）"""
        travel_distance = calculate_distance(usv.position, task.position)
        energy_needed = simple_energy_model(
            travel_distance, task.service_time,
            self.energy_cost_per_unit_distance, self.task_time_energy_ratio
        )

        # 如果电量不足，先返回充电站充电
        if usv.battery_level < energy_needed:
            self._return_to_base_and_recharge(usv, config)

        # 现在执行任务
        travel_time = travel_distance / usv.speed
        start_time = usv.current_time + travel_time
        finish_time = start_time + task.service_time

        # 更新状态
        usv.current_time = finish_time
        usv.battery_level -= energy_needed
        usv.position = task.position

        task.assigned_usv = usv.usv_id
        task.start_time = start_time
        task.finish_time = finish_time

        # 记录时间线
        usv.timeline.append({
            'type': 'task',
            'task_id': task.task_id,
            'start_service': start_time,
            'finish_service': finish_time,
            'energy_used': energy_needed
        })

    def _return_to_base_and_recharge(self, usv: USV, config: Dict):
        """返回基地充电"""
        base_position = config['env_paras']['start_point']
        distance_to_base = calculate_distance(usv.position, base_position)
        travel_time = distance_to_base / usv.speed
        energy_to_base = distance_to_base * self.energy_cost_per_unit_distance

        # 返回充电站
        usv.current_time += travel_time
        usv.battery_level -= energy_to_base
        usv.position = base_position

        # 记录返回基地
        usv.timeline.append({
            'type': 'return_to_base',
            'start_time': usv.current_time - travel_time,
            'finish_time': usv.current_time,
            'energy_used': energy_to_base
        })

        # 充电
        charge_time = config['env_paras']['charge_time']
        usv.current_time += charge_time
        usv.battery_level = usv.battery_capacity

        # 记录充电
        usv.timeline.append({
            'type': 'recharge',
            'start_time': usv.current_time - charge_time,
            'finish_time': usv.current_time,
            'energy_gained': usv.battery_capacity
        })

    def _update_particle(self, particle_idx: int, inertia_weight: float, tasks: List[Task], usvs: List[USV], config: Dict):
        """更新粒子的速度和位置"""
        # 更新速度
        for j in range(len(self.particles[particle_idx])):
            r1 = random.random()
            r2 = random.random()

            # 认知部分
            cognitive = self.c1 * r1 * (self.personal_best[particle_idx][j] - self.particles[particle_idx][j])

            # 社会部分
            social = self.c2 * r2 * (self.global_best[j] - self.particles[particle_idx][j])

            # 更新速度
            self.velocities[particle_idx][j] = (inertia_weight * self.velocities[particle_idx][j] +
                                               cognitive + social)

            # 限制速度范围
            self.velocities[particle_idx][j] = max(-2, min(2, self.velocities[particle_idx][j]))

        # 更新位置
        for j in range(len(self.particles[particle_idx])):
            new_position = int(self.particles[particle_idx][j] + self.velocities[particle_idx][j])
            # 确保位置在有效范围内
            new_position = max(0, min(len(usvs) - 1, new_position))
            self.particles[particle_idx][j] = new_position

    def _apply_solution(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """应用最优解到实际任务和USV"""
        # 重置USV状态
        for usv in usvs:
            usv.battery_level = usv.battery_capacity
            usv.current_time = 0.0
            usv.position = config['env_paras']['start_point'].copy()
            usv.timeline = []

        for task in tasks:
            task.assigned_usv = None
            task.start_time = None
            task.finish_time = None

        # 应用全局最优解
        for i, task in enumerate(tasks):
            if i < len(self.global_best):
                usv_idx = self.global_best[i]
                if usv_idx < len(usvs):
                    usv = usvs[usv_idx]
                    if self._can_execute_task_simple(usv, task, config):
                        self._execute_task_simple(usv, task, config)
                    else:
                        self.failures.append(f"任务 {task.task_id} 无法分配给USV {usv_idx}")


# 兼容性接口
def run_single_case(json_file: str, config: Dict = None) -> Dict:
    """运行单个测试案例"""
    if config is None:
        config = {
            'swarm_size': 50,         # 增加粒子群大小
            'iterations': 35,         # 增加迭代次数
            'inertia_weight_max': 0.9,
            'inertia_weight_min': 0.4,
            'c1': 2.0,
            'c2': 2.0,
            'random_seed': 42,
            'energy_cost_per_unit_distance': 1.0,  # 保持原始物理参数
            'task_time_energy_ratio': 0.25,        # 与数据文件保持一致
            'usv_initial_position': [0.0, 0.0]
        }

    planner = PSOTaskPlanner(config)
    env_data = load_and_adapt_data(json_file)
    return planner.plan(env_data)


def main():
    """主函数，用于测试"""
    # 测试案例文件
    test_case_file = "../../usv_data_dev/40_8/usv_case_40_8_instance_01.json"

    if os.path.exists(test_case_file):
        print(f"运行PSO算法测试案例: {test_case_file}")
        result = run_single_case(test_case_file)

        if result['success']:
            makespan = result.get('makespan')
            if makespan is not None:
                print(f"PSO调度成功！完成时间: {makespan:.2f}")
            else:
                print("PSO调度成功但没有完成时间")
            print(f"已分配任务: {result['metrics']['assigned_tasks']}")
            print(f"未分配任务: {result['metrics']['unassigned_tasks']}")
            print(f"最佳适应度: {result['metrics']['best_fitness']:.2f}")
        else:
            print("PSO调度失败！")
            if result.get('error'):
                print(f"错误信息: {result['error']}")
    else:
        print(f"测试案例文件不存在: {test_case_file}")


if __name__ == "__main__":
    main()