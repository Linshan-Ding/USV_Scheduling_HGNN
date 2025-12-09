#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USV调度算法工具函数
提供数据转换、距离计算、能耗模型等通用功能
"""

import math
import json
import os
from typing import List, Dict, Tuple, Any, Optional


def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """
    计算两点之间的欧几里得距离

    Args:
        point1: 点1坐标 [x, y]
        point2: 点2坐标 [x, y]

    Returns:
        两点间距离
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def simple_energy_model(distance: float, service_time: float,
                       energy_per_distance: float = 1.0,
                       energy_per_time: float = 0.5) -> float:
    """
    简单能耗模型

    Args:
        distance: 移动距离
        service_time: 服务时间
        energy_per_distance: 单位距离能耗
        energy_per_time: 单位时间能耗

    Returns:
        总能耗
    """
    return distance * energy_per_distance + service_time * energy_per_time


def validate_schedule_result(result: Dict) -> Tuple[bool, List[str]]:
    """
    验证调度结果的合理性

    Args:
        result: 调度结果

    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []

    # 检查必需字段
    required_fields = ['schedule', 'makespan', 'metrics', 'success']
    for field in required_fields:
        if field not in result:
            errors.append(f"缺少必需字段: {field}")

    # 检查调度方案
    if 'schedule' in result:
        schedule = result['schedule']
        if 'tasks' not in schedule or 'usvs' not in schedule:
            errors.append("调度方案缺少tasks或usvs字段")
        else:
            # 检查任务分配
            tasks = schedule['tasks']
            usvs = schedule['usvs']
            usv_ids = {usv['usv_id'] for usv in usvs}

            for task in tasks:
                if task.get('assigned_usv') is not None:
                    if task['assigned_usv'] not in usv_ids:
                        errors.append(f"任务{task['task_id']}分配给了不存在的USV: {task['assigned_usv']}")

    # 检查完成时间
    if 'makespan' in result and result['makespan'] is not None:
        if result['makespan'] < 0:
            errors.append("完成时间不能为负数")

    return len(errors) == 0, errors


def compare_results(baseline: Dict, comparison: Dict) -> Dict:
    """
    比较两个调度结果

    Args:
        baseline: 基准结果
        comparison: 对比结果

    Returns:
        比较结果字典
    """
    comparison_result = {
        'baseline_makespan': baseline.get('makespan'),
        'comparison_makespan': comparison.get('makespan'),
        'improvement': None,
        'better': None
    }

    if (baseline.get('makespan') is not None and
        comparison.get('makespan') is not None):

        baseline_time = baseline['makespan']
        comparison_time = comparison['makespan']
        improvement = ((baseline_time - comparison_time) / baseline_time) * 100

        comparison_result.update({
            'improvement': improvement,
            'better': improvement > 0  # 正值表示改进
        })

    return comparison_result


def save_comparison_results(results: Dict, output_file: str):
    """
    保存对比结果

    Args:
        results: 对比结果数据
        output_file: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_test_case(file_path: str) -> Dict:
    """
    加载测试案例

    Args:
        file_path: 测试案例文件路径

    Returns:
        测试案例数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"测试案例文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_time(time_value: float) -> str:
    """
    格式化时间显示

    Args:
        time_value: 时间值

    Returns:
        格式化后的时间字符串
    """
    return f"{time_value:.2f}"


def generate_summary_report(all_results: List[Dict]) -> Dict:
    """
    生成汇总报告

    Args:
        all_results: 所有算法的结果列表

    Returns:
        汇总报告字典
    """
    summary = {
        'total_algorithms': len(all_results),
        'total_test_cases': 0,
        'algorithm_performance': {},
        'best_algorithm': None,
        'comparison_table': []
    }

    if not all_results:
        return summary

    # 统计测试案例数量
    if all_results[0].get('results'):
        summary['total_test_cases'] = len(all_results[0]['results'])

    # 收集各算法性能
    algorithm_stats = {}
    for result in all_results:
        alg_name = result['algorithm']['name']
        algorithm_stats[alg_name] = {
            'avg_makespan': 0,
            'success_rate': 0,
            'total_cases': 0
        }

        if result.get('results'):
            makespans = []
            successful_cases = 0

            for case_result in result['results']:
                if case_result.get('success') and case_result.get('makespan') is not None:
                    makespans.append(case_result['makespan'])
                    successful_cases += 1

            if makespans:
                algorithm_stats[alg_name]['avg_makespan'] = sum(makespans) / len(makespans)
                algorithm_stats[alg_name]['success_rate'] = successful_cases / len(result['results'])
            algorithm_stats[alg_name]['total_cases'] = len(result['results'])

    summary['algorithm_performance'] = algorithm_stats

    # 找出最佳算法（基于平均完成时间）
    best_alg = None
    best_makespan = float('inf')

    for alg_name, stats in algorithm_stats.items():
        if stats['avg_makespan'] > 0 and stats['avg_makespan'] < best_makespan:
            best_makespan = stats['avg_makespan']
            best_alg = alg_name

    summary['best_algorithm'] = best_alg

    return summary


class DataConverter:
    """数据转换工具类"""

    @staticmethod
    def env_data_to_tasks(env_data: Dict) -> List[Dict]:
        """从环境数据提取任务列表"""
        return env_data.get('tasks', [])

    @staticmethod
    def env_data_to_usvs(env_data: Dict) -> List[Dict]:
        """从环境数据提取USV列表"""
        return env_data.get('usvs', [])

    @staticmethod
    def extract_config(env_data: Dict) -> Dict:
        """从环境数据提取配置信息"""
        return env_data.get('config', {})

    @staticmethod
    def normalize_positions(data: List[Dict]) -> List[Dict]:
        """标准化位置坐标格式"""
        for item in data:
            if 'position' in item and isinstance(item['position'], list):
                # 确保坐标是二维的
                if len(item['position']) == 1:
                    item['position'] = [item['position'][0], 0.0]
                elif len(item['position']) == 0:
                    item['position'] = [0.0, 0.0]
        return data