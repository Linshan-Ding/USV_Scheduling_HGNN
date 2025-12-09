#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遗传算法任务规划器（模块化版本）
基于BasePlanner基类实现，支持统一的接口和数据格式

算法描述：
    使用遗传算法优化USV任务分配
    个体代表任务分配方案
    通过选择、交叉、变异操作搜索最优解
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


class GATaskPlanner(BasePlanner):
    """遗传算法任务规划器"""

    def __init__(self, config: Dict = None):
        """
        初始化遗传算法规划器

        Args:
            config: 算法配置参数
        """
        super().__init__(config)

        # GA算法参数
        self.population_size = self.config.get('population_size', 50)  # 种群大小
        self.generations = self.config.get('generations', 30)  # 迭代代数
        self.crossover_rate = self.config.get('crossover_rate', 0.8)  # 交叉概率
        self.mutation_rate = self.config.get('mutation_rate', 0.2)  # 变异概率
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
        self.population = []     # 种群
        self.fitness = []        # 适应度值
        self.best_individual = None
        self.best_fitness = float('inf')

        # 日志和状态
        self.warnings = []
        self.failures = []

    def plan(self, env_data: Dict) -> Dict:
        """
        执行遗传算法调度规划

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

        # 执行GA算法
        self._execute_ga_optimization(tasks, usvs, config)

        # 使用最优解执行调度
        if self.best_individual is not None:
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
            'algorithm': 'GA',
            'population_size': self.population_size,
            'generations': self.generations,
            'best_fitness': self.best_fitness
        })

        # 保存结果
        self.results = {
            'schedule': schedule_result,
            'makespan': metrics['makespan'],
            'metrics': metrics,
            'success': len(tasks) > 0 and self.best_individual is not None
        }

        return self.results

    def _execute_ga_optimization(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """执行遗传算法优化"""
        print(f"GA算法开始优化 - 种群大小: {self.population_size}, 迭代代数: {self.generations}")

        # 初始化种群
        self._initialize_population(tasks, usvs, config)

        # 主循环
        for generation in range(self.generations):
            # 评估适应度
            self.fitness = [self._evaluate_fitness(individual, tasks, usvs, config)
                           for individual in self.population]

            # 更新最优解
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_individual = copy.deepcopy(self.population[best_idx])

            if generation % 5 == 0:
                print(f"  GA代数 {generation+1}/{self.generations}, 最佳适应度: {self.best_fitness:.2f}")

            # 选择、交叉、变异
            self._selection()
            self._crossover()
            self._mutation()

        print(f"GA算法优化完成，最佳适应度: {self.best_fitness:.2f}")

    def _initialize_population(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            # 生成随机个体（任务-USV分配）
            individual = [random.randint(0, len(usvs) - 1) for _ in tasks]
            self.population.append(individual)

    def _evaluate_fitness(self, individual: List[int], tasks: List[Task], usvs: List[USV], config: Dict) -> float:
        """评估个体的适应度"""
        # 创建临时副本
        temp_tasks = copy.deepcopy(tasks)
        temp_usvs = copy.deepcopy(usvs)

        # 应用个体
        self._apply_individual_to_temp(temp_tasks, temp_usvs, individual, config)

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

    def _apply_individual_to_temp(self, tasks: List[Task], usvs: List[USV], individual: List[int], config: Dict):
        """将个体应用到临时任务和USV（支持充电）"""
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

        # 按个体分配任务（使用带充电功能的执行逻辑）
        for i, task in enumerate(tasks):
            if i < len(individual):
                usv_idx = individual[i]
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

    def _selection(self):
        """选择操作：锦标赛选择"""
        selected = []
        for _ in range(self.population_size):
            # 随机选择两个个体
            idx1, idx2 = random.sample(range(self.population_size), 2)
            # 选择适应度更好的个体（makespan更小）
            if self.fitness[idx1] < self.fitness[idx2]:
                selected.append(copy.deepcopy(self.population[idx1]))
            else:
                selected.append(copy.deepcopy(self.population[idx2]))
        self.population = selected

    def _crossover(self):
        """交叉操作：顺序交叉 (Order Crossover, OX)"""
        new_population = []
        for i in range(0, self.population_size, 2):
            parent1 = self.population[i]
            parent2 = self.population[i+1] if i+1 < self.population_size else self.population[0]

            if random.random() < self.crossover_rate:
                child1, child2 = self._order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            new_population.extend([child1, child2])

        # 保持种群大小
        self.population = new_population[:self.population_size]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """顺序交叉操作"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))

        child1 = [-1] * size
        child2 = [-1] * size

        # 复制父代片段
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # 填充剩余位置
        def fill_child(child, parent, start, end):
            pointer = end
            for gene in parent[end:] + parent[:end]:
                if gene not in child:
                    if pointer >= size:
                        pointer = 0
                    if pointer == start:
                        break
                    child[pointer] = gene
                    pointer += 1

        fill_child(child1, parent2, start, end)
        fill_child(child2, parent1, start, end)

        return child1, child2

    def _mutation(self):
        """变异操作：交换变异"""
        for individual in self.population:
            if random.random() < self.mutation_rate:
                idx1, idx2 = random.sample(range(len(individual)), 2)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

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
            if i < len(self.best_individual):
                usv_idx = self.best_individual[i]
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
            'population_size': 60,   # 增加种群大小
            'generations': 40,       # 增加迭代代数
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'random_seed': 42,
            'energy_cost_per_unit_distance': 1.0,  # 保持原始物理参数
            'task_time_energy_ratio': 0.25,        # 与数据文件保持一致
            'usv_initial_position': [0.0, 0.0]
        }

    planner = GATaskPlanner(config)
    env_data = load_and_adapt_data(json_file)
    return planner.plan(env_data)


def main():
    """主函数，用于测试"""
    # 测试案例文件
    test_case_file = "../../usv_data_dev/40_8/usv_case_40_8_instance_01.json"

    if os.path.exists(test_case_file):
        print(f"运行GA算法测试案例: {test_case_file}")
        result = run_single_case(test_case_file)

        if result['success']:
            makespan = result.get('makespan')
            if makespan is not None:
                print(f"GA调度成功！完成时间: {makespan:.2f}")
            else:
                print("GA调度成功但没有完成时间")
            print(f"已分配任务: {result['metrics']['assigned_tasks']}")
            print(f"未分配任务: {result['metrics']['unassigned_tasks']}")
            print(f"最佳适应度: {result['metrics']['best_fitness']:.2f}")
        else:
            print("GA调度失败！")
            if result.get('error'):
                print(f"错误信息: {result['error']}")
    else:
        print(f"测试案例文件不存在: {test_case_file}")


if __name__ == "__main__":
    main()