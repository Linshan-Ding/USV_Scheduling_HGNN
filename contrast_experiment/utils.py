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
from base_planner import BasePlanner, Task, USV
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_gantt_chart(tasks: List[Task], usvs: List[USV], config: Dict):
    """
    绘制最优解的甘特图
    """
    print("生成甘特图...")

    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.suptitle('USV任务调度甘特图 - 遗传算法优化结果（含任务顺序）',
                 fontsize=16, fontweight='bold')

    # 颜色配置
    colors = plt.cm.Set3(np.linspace(0, 1, len(tasks)))
    usv_colors = plt.cm.tab10(np.linspace(0, 1, len(usvs)))

    # 按USV分组任务
    usv_tasks = {}
    for usv in usvs:
        usv_tasks[usv.usv_id] = []

    for task in tasks:
        if task.assigned_usv is not None and task.start_time is not None and task.finish_time is not None:
            usv_tasks[task.assigned_usv].append(task)

    # 按开始时间排序每个USV的任务
    for usv_id in usv_tasks:
        usv_tasks[usv_id].sort(key=lambda x: x.start_time)

    # 绘制每个USV的任务条
    y_ticks = []
    y_labels = []

    for i, usv in enumerate(usvs):
        usv_id = usv.usv_id
        y_pos = len(usvs) - i - 1  # 从上到下排列

        y_ticks.append(y_pos)
        y_labels.append(f'USV-{usv_id}')

        # 绘制USV的任务时间线
        current_time = 0
        tasks_for_usv = usv_tasks.get(usv_id, [])

        for j, task in enumerate(tasks_for_usv):
            # 计算旅行时间（从前一个位置到任务位置）
            travel_time = task.start_time - current_time
            if travel_time > 0:
                # 绘制旅行段
                ax.barh(y_pos, travel_time, left=current_time,
                        color='lightgray', alpha=0.7,
                        label='旅行' if i == 0 and j == 0 else "")

            # 绘制任务执行段
            task_duration = task.finish_time - task.start_time
            ax.barh(y_pos, task_duration, left=task.start_time,
                    color=colors[task.task_id % len(colors)],
                    edgecolor='black', linewidth=0.5,
                    label=f'任务{task.task_id}' if i == 0 and j == 0 else "")

            # 添加任务标签
            ax.text(task.start_time + task_duration / 2, y_pos,
                    f'T{task.task_id}', ha='center', va='center',
                    fontsize=8, fontweight='bold')

            current_time = task.finish_time

        # 绘制返回基地的旅行时间（如果有）
        return_time = getattr(usv, 'return_time', 0)
        if return_time > 0 and tasks_for_usv:
            makespan = max([task.finish_time for task in tasks_for_usv])
            ax.barh(y_pos, return_time, left=makespan,
                    color='lightblue', alpha=0.7, label='返回基地' if i == 0 else "")

    # 设置Y轴
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('USV编号', fontsize=12)

    # 设置X轴
    max_time = max([task.finish_time for task in tasks if task.finish_time is not None]) if tasks else 0
    ax.set_xlabel('时间', fontsize=12)
    ax.set_xlim(0, max_time * 1.1)

    # 添加网格
    ax.grid(True, alpha=0.3, axis='x')

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(),
                  loc='upper right', bbox_to_anchor=(1.15, 1))

    # 调整布局
    plt.tight_layout()

    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gantt_chart_ga_with_order_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"甘特图已保存: {filename}")

    # 显示图片
    plt.show()


def plot_trajectory(tasks: List[Task], usvs: List[USV], config: Dict,
                    figsize: tuple = (15, 12), dpi: int = 300):
    """
    绘制多无人船多任务轨迹图

    Args:
        tasks: 任务列表
        usvs: 无人船列表
        config: 配置参数
        figsize: 图像大小
        dpi: 图像分辨率
    """
    print("生成轨迹图...")

    # 创建图形和子图
    fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=figsize,
                                             gridspec_kw={'width_ratios': [4, 1]})

    # # 设置主标题
    # fig.suptitle('多无人船任务执行轨迹图 - 遗传算法优化结果',
    #              fontsize=18, fontweight='bold', y=0.95)

    # 颜色配置 - 为每个USV分配独特颜色
    usv_colors = plt.cm.tab10(np.linspace(0, 1, len(usvs)))
    task_color = 'red'
    base_color = 'green'

    # 获取起点坐标
    start_point = config['env_paras']['start_point']
    base_x, base_y = start_point[0], start_point[1]

    # 绘制起点（基地）
    ax_main.scatter(base_x, base_y, color=base_color, s=200, zorder=5,
                    label='基地', marker='s', edgecolors='black', linewidth=2)
    ax_main.annotate('基地', (base_x, base_y), xytext=(10, 10),
                     textcoords='offset points', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # 按USV分组任务
    usv_tasks = {}
    for usv in usvs:
        usv_tasks[usv.usv_id] = []

    for task in tasks:
        if task.assigned_usv is not None:
            usv_tasks[task.assigned_usv].append(task)

    # 按开始时间排序每个USV的任务
    for usv_id in usv_tasks:
        usv_tasks[usv_id].sort(key=lambda x: x.start_time if x.start_time is not None else 0)

    # 绘制每个USV的轨迹
    for usv_idx, usv in enumerate(usvs):
        usv_id = usv.usv_id
        color = usv_colors[usv_idx % len(usv_colors)]

        # 获取该USV的任务列表
        assigned_tasks = usv_tasks.get(usv_id, [])

        if not assigned_tasks:
            continue

        # 构建轨迹点序列：起点 -> 任务1 -> 任务2 -> ... -> 返回起点
        trajectory_points = [start_point]  # 从起点开始

        for task in assigned_tasks:
            trajectory_points.append(task.position)

        trajectory_points.append(start_point)  # 返回起点

        # 转换为numpy数组便于处理
        points = np.array(trajectory_points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # 绘制轨迹线
        line = ax_main.plot(x_coords, y_coords, color=color, linewidth=2.5,
                            alpha=0.7, label=f'USV-{usv_id}')[0]

        # 添加箭头指示方向
        for i in range(len(x_coords) - 1):
            # 计算箭头位置（在线段的中点）
            mid_x = (x_coords[i] + x_coords[i + 1]) / 2
            mid_y = (y_coords[i] + y_coords[i + 1]) / 2

            # 计算箭头方向
            dx = x_coords[i + 1] - x_coords[i]
            dy = y_coords[i + 1] - y_coords[i]

            # 绘制箭头
            ax_main.arrow(mid_x, mid_y, dx * 0.3, dy * 0.3,
                          head_width=0.5, head_length=0.8, fc=color, ec=color,
                          alpha=0.7, length_includes_head=True)

        # 标注距离
        for i in range(len(x_coords) - 1):
            start = points[i]
            end = points[i + 1]
            distance = calculate_distance(start, end)

            # 计算标注位置
            text_x = (start[0] + end[0]) / 2
            text_y = (start[1] + end[1]) / 2

            # 稍微偏移文本避免重叠
            offset_x = (end[1] - start[1]) * 0.1  # 垂直方向偏移
            offset_y = (start[0] - end[0]) * 0.1  # 水平方向偏移

            ax_main.annotate(f'{distance:.1f}', (text_x, text_y),
                             xytext=(offset_x, offset_y), textcoords='offset points',
                             fontsize=9, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                             ha='center', va='center')

        # 标记USV位置点
        ax_main.scatter(x_coords, y_coords, color=color, s=80, zorder=4, alpha=0.7)

        # 在任务点标注USV ID
        for i, point in enumerate(points[1:-1]):  # 跳过起点和终点（基地）
            ax_main.annotate(f'USV-{usv_id}', (point[0], point[1]),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, color=color, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # 绘制所有任务点（用统一样式）
    task_x = [task.position[0] for task in tasks]
    task_y = [task.position[1] for task in tasks]
    task_ids = [task.task_id for task in tasks]

    # 绘制任务点
    scatter = ax_main.scatter(task_x, task_y, color=task_color, s=100, zorder=5,
                              label='任务点', marker='o', edgecolors='black', linewidth=1.5)

    # 标注任务ID和坐标
    for i, (x, y, task_id) in enumerate(zip(task_x, task_y, task_ids)):
        ax_main.annotate(f'T{task_id}\n({x:.1f},{y:.1f})', (x, y),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=9, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='black'))

    # 设置坐标轴属性
    ax_main.set_xlabel('X 坐标', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Y 坐标', fontsize=12, fontweight='bold')
    ax_main.set_title('任务执行轨迹', fontsize=14, fontweight='bold')

    # 设置网格
    ax_main.grid(True, alpha=0.3, linestyle='--')

    # 等比例显示
    ax_main.set_aspect('equal', adjustable='datalim')

    # 计算合适的坐标范围
    all_x = [base_x] + task_x
    all_y = [base_y] + task_y
    x_margin = (max(all_x) - min(all_x)) * 0.1
    y_margin = (max(all_y) - min(all_y)) * 0.1

    ax_main.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax_main.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    # 在右侧创建图例
    ax_legend.axis('off')  # 隐藏坐标轴

    # 创建自定义图例
    legend_elements = []

    # 基地图例
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                      markerfacecolor=base_color, markersize=10,
                                      label='基地', markeredgecolor='black', markeredgewidth=1))

    # 任务点图例
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=task_color, markersize=10,
                                      label='任务点', markeredgecolor='black', markeredgewidth=1))

    # 添加每个USV的图例
    for usv_idx, usv in enumerate(usvs):
        color = usv_colors[usv_idx % len(usv_colors)]
        usv_id = usv.usv_id

        # 统计该USV的任务数
        task_count = len(usv_tasks.get(usv_id, []))

        legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3,
                                          label=f'USV-{usv_id} ({task_count}个任务)'))

    # 添加统计信息
    total_tasks = len(tasks)
    assigned_tasks = sum(1 for task in tasks if task.assigned_usv is not None)
    completion_time = max([usv.completion_time for usv in usvs if usv.completion_time is not None])

    stats_text = f"统计信息:\n总任务数: {total_tasks}\n已分配任务: {assigned_tasks}\n完成时间: {completion_time:.1f}"

    # 创建图例
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=10, framealpha=0.9, fancybox=True)

    # 添加统计信息文本框
    ax_legend.text(0.5, 0.2, stats_text, transform=ax_legend.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.7))

    # 调整布局
    plt.tight_layout()

    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_plot_{timestamp}.png"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"轨迹图已保存: {filename}")

    # 显示图片
    plt.show()

    return fig, ax_main

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