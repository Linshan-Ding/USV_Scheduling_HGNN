#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机任务规划算法（模块化版本）
基于BasePlanner基类实现，支持统一的接口和数据格式

算法描述：
    1. 按照任务出现的顺序进行处理
    2. 为每个任务随机选择一个能够执行的USV
    3. 考虑USV电量约束，必要时进行充电
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_planner import BasePlanner, Task, USV
from utils import calculate_distance, simple_energy_model, DataConverter
from data_adapter import load_and_adapt_data
import math
from typing import Dict, List, Any


class RandomTaskPlanner(BasePlanner):
    """随机任务规划器"""

    def __init__(self, config: Dict = None):
        """
        初始化随机任务规划器

        Args:
            config: 算法配置参数，包含：
                   - energy_cost_per_unit_distance: 单位距离能耗
                   - task_time_energy_ratio: 任务时间能耗比例
                   - usv_initial_position: USV初始位置
                   - random_seed: 随机种子
        """
        super().__init__(config)

        # 默认参数
        self.energy_cost_per_unit_distance = self.config.get('energy_cost_per_unit_distance', 1.0)
        self.task_time_energy_ratio = self.config.get('task_time_energy_ratio', 0.5)
        self.usv_initial_position = self.config.get('usv_initial_position', [0.0, 0.0])
        self.random_seed = self.config.get('random_seed', 42)

        # 设置随机种子
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # 日志和状态
        self.warnings = []
        self.failures = []

    def plan(self, env_data: Dict) -> Dict:
        """
        执行随机任务调度规划

        Args:
            env_data: 环境数据，包含config、tasks、usvs

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

        # 执行调度
        self._execute_scheduling(tasks, usvs, config)

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
            'algorithm': 'RandomTaskPlanner',
            'random_seed': self.random_seed
        })

        # 保存结果
        self.results = {
            'schedule': schedule_result,
            'makespan': metrics['makespan'],
            'metrics': metrics,
            'success': len(self.failures) == 0
        }

        return self.results

    def _execute_scheduling(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        执行具体的调度逻辑

        Args:
            tasks: 任务列表
            usvs: USV列表
            config: 环境配置
        """
        # 按任务ID顺序处理
        tasks.sort(key=lambda t: t.task_id)

        # 为每个任务分配USV
        for task in tasks:
            assigned_usv, needs_charge = self._select_usv_for_task(task, usvs, config)

            if assigned_usv:
                if needs_charge:
                    assigned_usv.charge_full()
                assigned_usv.execute_task(task, config, self.energy_model)
            else:
                self.failures.append(f"任务 {task.task_id} 无法分配给任何USV")

    def _select_usv_for_task(self, task: Task, usvs: List[USV], config: Dict) -> tuple:
        """
        为任务随机选择一个合适的USV

        Args:
            task: 待分配的任务
            usvs: USV列表
            config: 环境配置

        Returns:
            (选中的USV, 是否需要充电)
        """
        capable_usvs_now = []
        capable_usvs_after_charge = []

        for usv in usvs:
            if self._can_execute_task(usv, task, config):
                capable_usvs_now.append(usv)
            else:
                # 检查充电后是否能执行
                if self._can_execute_after_charge(usv, task, config):
                    capable_usvs_after_charge.append(usv)
                else:
                    self.warnings.append(
                        f"USV {usv.usv_id} 即使充满电也无法执行任务 {task.task_id}"
                    )

        # 随机选择当前就能执行的USV
        if capable_usvs_now:
            chosen_usv = random.choice(capable_usvs_now)
            return chosen_usv, False

        # 随机选择需要充电后能执行的USV
        if capable_usvs_after_charge:
            chosen_usv = random.choice(capable_usvs_after_charge)
            return chosen_usv, True

        return None, False

    def _can_execute_task(self, usv: USV, task: Task, config: Dict) -> bool:
        """检查USV当前是否能执行任务"""
        travel_distance = calculate_distance(usv.position, task.position)
        energy_needed = simple_energy_model(
            travel_distance, task.service_time,
            self.energy_cost_per_unit_distance, self.task_time_energy_ratio
        )
        return usv.battery_level >= energy_needed

    def _can_execute_after_charge(self, usv: USV, task: Task, config: Dict) -> bool:
        """检查USV充满电后是否能执行任务"""
        travel_distance = calculate_distance(usv.position, task.position)
        energy_needed = simple_energy_model(
            travel_distance, task.service_time,
            self.energy_cost_per_unit_distance, self.task_time_energy_ratio
        )
        return usv.battery_capacity >= energy_needed

    def energy_model(self, distance: float, service_time: float) -> float:
        """能耗模型"""
        return simple_energy_model(
            distance, service_time,
            self.energy_cost_per_unit_distance, self.task_time_energy_ratio
        )


# 为了兼容性，添加执行任务的方法到USV类
def execute_task_method(self, task: Task, config: Dict, energy_model):
    """USV执行任务的方法"""
    travel_distance = calculate_distance(self.position, task.position)
    travel_time = travel_distance / self.speed
    energy_needed = energy_model(travel_distance, task.service_time)

    start_time = self.current_time + travel_time
    finish_time = start_time + task.service_time

    self.current_time = finish_time
    self.battery_level -= energy_needed
    self.position = task.position

    self.timeline.append({
        'type': 'task',
        'task_id': task.task_id,
        'depart_time': start_time - travel_time,
        'arrive_time': start_time,
        'start_service': start_time,
        'finish_service': finish_time,
        'energy_used': energy_needed,
        'battery_after': self.battery_level
    })

    task.assigned_usv = self.usv_id
    task.start_time = start_time
    task.finish_time = finish_time


def charge_full_method(self):
    """USV充满电的方法"""
    start_charge = self.current_time
    self.current_time += self.charge_time
    self.battery_level = self.battery_capacity

    self.timeline.append({
        'type': 'charge',
        'start_charge': start_charge,
        'finish_charge': self.current_time,
        'battery_after': self.battery_level
    })


# 动态添加方法到USV类
USV.execute_task = execute_task_method
USV.charge_full = charge_full_method


# 兼容性接口，用于直接运行
def run_single_case(json_file: str, config: Dict = None) -> Dict:
    """
    运行单个测试案例

    Args:
        json_file: 测试案例文件路径
        config: 算法配置

    Returns:
        调度结果
    """
    planner = RandomTaskPlanner(config)

    # 使用数据适配器加载和转换数据
    env_data = load_and_adapt_data(json_file)

    return planner.plan(env_data)


def main():
    """主函数，用于测试"""
    # 示例配置
    config = {
        'energy_cost_per_unit_distance': 1.0,
        'task_time_energy_ratio': 0.5,
        'usv_initial_position': [0.0, 0.0],
        'random_seed': 42
    }

    # 测试案例文件
    test_case_file = "../../save/40_2_usv_case_40_2_instance_01.json"

    if os.path.exists(test_case_file):
        print(f"运行测试案例: {test_case_file}")
        result = run_single_case(test_case_file, config)

        if result['success']:
            makespan = result.get('makespan')
            if makespan is not None:
                print(f"调度成功！完成时间: {makespan:.2f}")
            else:
                print("调度成功但没有完成时间")
            print(f"已分配任务: {result['metrics']['assigned_tasks']}")
            print(f"未分配任务: {result['metrics']['unassigned_tasks']}")
            print(f"总任务数: {result['metrics']['total_tasks']}")
            print(f"总USV数: {result['metrics']['total_usvs']}")
            print(f"随机种子: {result['metrics']['random_seed']}")
        else:
            print("调度失败！")
            if result.get('error'):
                print(f"错误信息: {result['error']}")
            for failure in result.get('failures', []):
                print(f"失败信息: {failure}")

        # 显示警告信息
        warnings = result.get('warnings', [])
        if warnings:
            print("警告信息:")
            for warning in warnings:
                print(f"  {warning}")
    else:
        print(f"测试案例文件不存在: {test_case_file}")


if __name__ == "__main__":
    main()