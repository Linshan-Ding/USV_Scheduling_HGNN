#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工蜂群任务规划算法（模块化版本）
基于BasePlanner基类实现，支持统一的接口和数据格式

算法描述：
    使用人工蜂群算法优化USV任务分配
    食物源代表任务分配方案
    通过雇佣蜂、观察蜂和侦察蜂三种角色搜索最优解
    适应度函数基于总完成时间和任务完成率
"""

import sys
import os
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_planner import BasePlanner, Task, USV
from utils import calculate_distance, simple_energy_model, DataConverter
from data_adapter import load_and_adapt_data
from typing import Dict, List, Any, Tuple
import copy


class ABCTaskPlanner(BasePlanner):
    """人工蜂群任务规划器"""

    def __init__(self, config: Dict = None):
        """
        初始化人工蜂群规划器

        Args:
            config: 算法配置参数
        """
        super().__init__(config)

        # ABC算法参数
        self.colony_size = self.config.get('colony_size', 10)  # 蜂群大小
        self.max_iterations = self.config.get('max_iterations', 5)  # 最大迭代次数
        self.limit = self.config.get('limit', 30)  # 食物源废弃阈值
        self.random_seed = self.config.get('random_seed', 42)

        # 环境参数
        self.energy_cost_per_unit_distance = self.config.get('energy_cost_per_unit_distance', 1.0)
        self.task_time_energy_ratio = self.config.get('task_time_energy_ratio', 0.5)
        self.usv_initial_position = self.config.get('usv_initial_position', [0.0, 0.0])

        # 设置随机种子
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # 算法状态
        self.food_sources = []  # 食物源（解）
        self.fitness = []       # 适应度值
        self.trial_count = []   # 试验次数
        self.best_solution = None
        self.best_fitness = float('inf')

        # 日志和状态
        self.warnings = []
        self.failures = []

    def plan(self, env_data: Dict) -> Dict:
        """
        执行人工蜂群调度规划

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

        # 执行ABC算法
        self._execute_abc_optimization(tasks, usvs, config)

        # 使用最优解执行调度
        if self.best_solution is not None:
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
            'algorithm': 'ABC',
            'colony_size': self.colony_size,
            'iterations': self.max_iterations,
            'best_fitness': self.best_fitness
        })

        # 保存结果
        self.results = {
            'schedule': schedule_result,
            'makespan': metrics['makespan'],
            'metrics': metrics,
            'success': len(tasks) > 0 and self.best_solution is not None
        }

        return self.results

    def _execute_abc_optimization(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """执行人工蜂群优化算法"""
        print(f"ABC算法开始优化 - 蜂群大小: {self.colony_size}, 迭代次数: {self.max_iterations}")

        # 初始化食物源
        self._initialize_food_sources(tasks, usvs, config)

        # 主循环
        for iteration in range(self.max_iterations):
            # 雇佣蜂阶段
            self._employed_bee_phase(tasks, usvs, config)

            # 观察蜂阶段
            self._onlooker_bee_phase(tasks, usvs, config)

            # 侦察蜂阶段
            self._scout_bee_phase(tasks, usvs, config)

            # 更新最优解
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_solution = copy.deepcopy(self.food_sources[best_idx])

            if iteration % 2 == 0:
                print(f"  ABC迭代 {iteration+1}/{self.max_iterations}, 最佳适应度: {self.best_fitness:.2f}")

        print(f"ABC算法优化完成，最佳适应度: {self.best_fitness:.2f}")

    def _initialize_food_sources(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """初始化食物源"""
        self.food_sources = []
        self.fitness = []
        self.trial_count = []

        for _ in range(self.colony_size):
            # 生成随机解
            solution = self._generate_random_solution(tasks, usvs)
            fitness_value = self._evaluate_fitness(solution, tasks, usvs, config)

            self.food_sources.append(solution)
            self.fitness.append(fitness_value)
            self.trial_count.append(0)

        # 初始化最优解
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_solution = copy.deepcopy(self.food_sources[best_idx])

    def _generate_random_solution(self, tasks: List[Task], usvs: List[USV]) -> List[int]:
        """生成随机解（任务-USV分配）"""
        solution = []
        for _ in tasks:
            solution.append(random.randint(0, len(usvs) - 1))
        return solution

    def _evaluate_fitness(self, solution: List[int], tasks: List[Task], usvs: List[USV], config: Dict) -> float:
        """评估解的适应度"""
        # 创建临时副本
        temp_tasks = copy.deepcopy(tasks)
        temp_usvs = copy.deepcopy(usvs)

        # 应用解
        self._apply_solution_to_temp(temp_tasks, temp_usvs, solution, config)

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

    def _apply_solution_to_temp(self, tasks: List[Task], usvs: List[USV], solution: List[int], config: Dict):
        """将解应用到临时任务和USV（支持充电）"""
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

        # 按解分配任务（使用带充电功能的执行逻辑）
        for i, task in enumerate(tasks):
            if i < len(solution):
                usv_idx = solution[i]
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

    def _employed_bee_phase(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """雇佣蜂阶段"""
        for i in range(self.colony_size):
            # 生成新解
            k = random.randint(0, self.colony_size - 1)
            while k == i:
                k = random.randint(0, self.colony_size - 1)

            j = random.randint(0, len(tasks) - 1)

            new_solution = copy.deepcopy(self.food_sources[i])
            new_solution[j] = self.food_sources[k][j]  # 交叉操作

            # 评估新解
            new_fitness = self._evaluate_fitness(new_solution, tasks, usvs, config)

            # 贪婪选择
            if new_fitness < self.fitness[i]:
                self.food_sources[i] = new_solution
                self.fitness[i] = new_fitness
                self.trial_count[i] = 0
            else:
                self.trial_count[i] += 1

    def _onlooker_bee_phase(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """观察蜂阶段"""
        # 计算选择概率
        max_fitness = max(self.fitness) if max(self.fitness) > 0 else 1
        probabilities = [(max_fitness - f) / max_fitness for f in self.fitness]

        # 归一化概率
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities)] * len(probabilities)

        for i in range(self.colony_size):
            if random.random() < probabilities[i]:
                # 生成新解
                k = random.randint(0, self.colony_size - 1)
                while k == i:
                    k = random.randint(0, self.colony_size - 1)

                j = random.randint(0, len(tasks) - 1)

                new_solution = copy.deepcopy(self.food_sources[i])
                new_solution[j] = self.food_sources[k][j]

                # 评估新解
                new_fitness = self._evaluate_fitness(new_solution, tasks, usvs, config)

                # 贪婪选择
                if new_fitness < self.fitness[i]:
                    self.food_sources[i] = new_solution
                    self.fitness[i] = new_fitness
                    self.trial_count[i] = 0
                else:
                    self.trial_count[i] += 1

    def _scout_bee_phase(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """侦察蜂阶段"""
        for i in range(self.colony_size):
            if self.trial_count[i] > self.limit:
                # 放弃食物源，生成新的随机解
                self.food_sources[i] = self._generate_random_solution(tasks, usvs)
                self.fitness[i] = self._evaluate_fitness(self.food_sources[i], tasks, usvs, config)
                self.trial_count[i] = 0

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

        # 应用最优解
        for i, task in enumerate(tasks):
            if i < len(self.best_solution):
                usv_idx = self.best_solution[i]
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
            'colony_size': 80,     # 大幅增加蜂群大小
            'max_iterations': 50,  # 大幅增加迭代次数
            'limit': 2,            # 更激进的侦察蜂策略
            'random_seed': None,   # 移除随机种子限制，增加多样性
            'energy_cost_per_unit_distance': 1.0,  # 保持原始物理参数
            'task_time_energy_ratio': 0.25,        # 与数据文件保持一致
            'usv_initial_position': [0.0, 0.0]
        }

    planner = ABCTaskPlanner(config)
    env_data = load_and_adapt_data(json_file)
    return planner.plan(env_data)


def main():
    """主函数，用于测试"""
    # 测试案例文件
    test_case_file = "../../usv_data_dev/40_8/usv_case_40_8_instance_01.json"

    if os.path.exists(test_case_file):
        print(f"运行ABC算法测试案例: {test_case_file}")
        result = run_single_case(test_case_file)

        if result['success']:
            makespan = result.get('makespan')
            if makespan is not None:
                print(f"ABC调度成功！完成时间: {makespan:.2f}")
            else:
                print("ABC调度成功但没有完成时间")
            print(f"已分配任务: {result['metrics']['assigned_tasks']}")
            print(f"未分配任务: {result['metrics']['unassigned_tasks']}")
            print(f"最佳适应度: {result['metrics']['best_fitness']:.2f}")
        else:
            print("ABC调度失败！")
            if result.get('error'):
                print(f"错误信息: {result['error']}")
    else:
        print(f"测试案例文件不存在: {test_case_file}")


if __name__ == "__main__":
    main()