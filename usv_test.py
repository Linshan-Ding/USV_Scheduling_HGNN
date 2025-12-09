#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USV对比实验统一测试框架
支持所有算法的统一测试、对比分析和结果生成

算法支持：
- 调度规则算法（5个）：最近任务优先、最低电量优先、最远任务优先、最高电量优先、随机规划器
- 元启发式算法（3个）：人工蜂群算法、遗传算法、粒子群算法
- 学习算法（2个）：DQN算法、PPO算法

功能特性：
- 统一接口调用所有算法
- 自动选择代表性测试案例
- 生成详细的对比分析报告
- 支持结果可视化和导出
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import traceback

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'contrast_experiment'))

# 导入调度规则算法
from contrast_experiment.dispatching_rule_methods.task_nearest_distant_first_modular import run_single_case as run_nearest_first
from contrast_experiment.dispatching_rule_methods.usv_lowest_battery_first_modular import run_single_case as run_lowest_battery
from contrast_experiment.dispatching_rule_methods.task_farthest_distant_first_modular import run_single_case as run_farthest_first
from contrast_experiment.dispatching_rule_methods.usv_highest_battery_first_modular import run_single_case as run_highest_battery
from contrast_experiment.dispatching_rule_methods.usv_task_random_planner_modular import run_single_case as run_random_planner

# 导入元启发式算法
from contrast_experiment.meta_heuristic_methods.abc_task_planner_modular import run_single_case as run_abc
from contrast_experiment.meta_heuristic_methods.ga_task_planner_modular import run_single_case as run_ga
from contrast_experiment.meta_heuristic_methods.pso_task_planner_modular import run_single_case as run_pso

# 导入学习算法
from contrast_experiment.learning_based_methods.dqn_task_planner_modular import run_single_case as run_dqn
from contrast_experiment.learning_based_methods.ppo_task_planner_modular import run_single_case as run_ppo


class USVTestFramework:
    """USV统一测试框架"""

    def __init__(self):
        """初始化测试框架"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'usv_data_dev')
        self.save_dir = os.path.join(self.base_dir, 'save')
        self.results_dir = os.path.join(self.base_dir, 'test_results')

        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)

        # 算法配置
        self.algorithms = {
            # 调度规则算法
            '最近任务优先': {
                'func': run_nearest_first,
                'type': 'dispatch_rule',
                'description': '优先分配距离USV最近的任务'
            },
            '最低电量优先': {
                'func': run_lowest_battery,
                'type': 'dispatch_rule',
                'description': '优先分配给电量最低的USV'
            },
            '最远任务优先': {
                'func': run_farthest_first,
                'type': 'dispatch_rule',
                'description': '优先分配距离USV最远的任务'
            },
            '最高电量优先': {
                'func': run_highest_battery,
                'type': 'dispatch_rule',
                'description': '优先分配给电量最高的USV'
            },
            '随机规划器': {
                'func': run_random_planner,
                'type': 'dispatch_rule',
                'description': '随机分配任务给USV'
            },

            # 元启发式算法
            '人工蜂群算法': {
                'func': run_abc,
                'type': 'meta_heuristic',
                'description': '使用人工蜂群算法优化任务分配'
            },
            '遗传算法': {
                'func': run_ga,
                'type': 'meta_heuristic',
                'description': '使用遗传算法优化任务分配'
            },
            '粒子群算法': {
                'func': run_pso,
                'type': 'meta_heuristic',
                'description': '使用粒子群算法优化任务分配'
            },

            # 学习算法
            'DQN算法': {
                'func': run_dqn,
                'type': 'learning_based',
                'description': '使用深度Q学习优化任务分配'
            },
            'PPO算法': {
                'func': run_ppo,
                'type': 'learning_based',
                'description': '使用近端策略优化算法优化任务分配'
            }
        }

        # 测试结果
        self.test_results = []

    def select_test_cases(self, max_cases: int = 20) -> List[str]:
        """选择代表性测试案例"""
        print("正在选择代表性测试案例...")

        test_cases = []

        # 遍历所有目录结构
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.json') and 'instance_01' in file:
                    full_path = os.path.join(root, file)

                    # 解析文件名获取任务和USV数量
                    dir_name = os.path.basename(root)
                    if '_' in dir_name:
                        try:
                            tasks, usvs = dir_name.split('_')
                            tasks, usvs = int(tasks), int(usvs)

                            # 选择不同规模的案例
                            if (tasks, usvs) in [(40, 2), (40, 4), (40, 6), (40, 8),
                                               (60, 2), (60, 4), (60, 6), (60, 8),
                                               (80, 2), (80, 4), (80, 6), (80, 8),
                                               (100, 2), (100, 4), (100, 6), (100, 8),
                                               (120, 2), (120, 4), (120, 6), (120, 8)]:
                                test_cases.append(full_path)

                                if len(test_cases) >= max_cases:
                                    break
                        except ValueError:
                            continue

            if len(test_cases) >= max_cases:
                break

        print(f"已选择 {len(test_cases)} 个代表性测试案例")
        return test_cases[:max_cases]

    def run_algorithm(self, algorithm_name: str, test_case: str) -> Dict[str, Any]:
        """运行单个算法"""
        print(f"  正在运行 {algorithm_name}...")

        try:
            start_time = time.time()

            # 调用算法
            algorithm_info = self.algorithms[algorithm_name]
            result = algorithm_info['func'](test_case)

            end_time = time.time()
            execution_time = end_time - start_time

            # 提取结果信息
            if result.get('success', False):
                makespan = result.get('makespan', float('inf'))
                metrics = result.get('metrics', {})
                assigned_tasks = metrics.get('assigned_tasks', 0)
                unassigned_tasks = metrics.get('unassigned_tasks', 0)

                # 计算任务完成率
                total_tasks = assigned_tasks + unassigned_tasks
                completion_rate = assigned_tasks / total_tasks if total_tasks > 0 else 0

                return {
                    'success': True,
                    'makespan': makespan,
                    'assigned_tasks': assigned_tasks,
                    'unassigned_tasks': unassigned_tasks,
                    'completion_rate': completion_rate,
                    'execution_time': execution_time,
                    'metrics': metrics
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', '未知错误'),
                    'execution_time': execution_time
                }

        except Exception as e:
            print(f"    错误: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'traceback': traceback.format_exc()
            }

    def run_comprehensive_test(self, test_cases: List[str] = None,
                             selected_algorithms: List[str] = None) -> pd.DataFrame:
        """运行综合测试"""
        print("开始运行综合测试...")
        print("=" * 60)

        if test_cases is None:
            test_cases = self.select_test_cases()

        if selected_algorithms is None:
            selected_algorithms = list(self.algorithms.keys())

        # 准备结果存储
        results = []

        for i, test_case in enumerate(test_cases):
            print(f"\n测试案例 {i+1}/{len(test_cases)}: {os.path.basename(test_case)}")
            print("-" * 50)

            # 解析测试案例信息
            dir_name = os.path.basename(os.path.dirname(test_case))
            tasks, usvs = dir_name.split('_')
            tasks, usvs = int(tasks), int(usvs)

            case_result = {
                'test_case': os.path.basename(test_case),
                'tasks': tasks,
                'usvs': usvs
            }

            # 运行每个算法
            for algorithm_name in selected_algorithms:
                result = self.run_algorithm(algorithm_name, test_case)

                # 存储结果
                case_result[f'{algorithm_name}_success'] = result['success']
                case_result[f'{algorithm_name}_makespan'] = result.get('makespan', float('inf'))
                case_result[f'{algorithm_name}_completion_rate'] = result.get('completion_rate', 0)
                case_result[f'{algorithm_name}_execution_time'] = result.get('execution_time', 0)
                case_result[f'{algorithm_name}_assigned'] = result.get('assigned_tasks', 0)
                case_result[f'{algorithm_name}_unassigned'] = result.get('unassigned_tasks', 0)

                if not result['success']:
                    case_result[f'{algorithm_name}_error'] = result.get('error', '')

            results.append(case_result)

        # 转换为DataFrame
        df = pd.DataFrame(results)
        self.test_results = df

        print(f"\n综合测试完成！共测试 {len(test_cases)} 个案例，{len(selected_algorithms)} 个算法")
        return df

    def generate_analysis_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成分析报告"""
        print("\n正在生成分析报告...")

        report = {
            'summary': {},
            'algorithm_comparison': {},
            'detailed_results': {}
        }

        # 算法列表
        algorithms = [alg for alg in self.algorithms.keys() if f'{alg}_success' in df.columns]

        # 基本统计
        report['summary'] = {
            'total_test_cases': len(df),
            'total_algorithms': len(algorithms),
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 算法对比统计
        for alg in algorithms:
            success_col = f'{alg}_success'
            makespan_col = f'{alg}_makespan'
            completion_col = f'{alg}_completion_rate'

            # 成功率
            success_rate = df[success_col].mean()

            # 平均完成时间（只考虑成功的案例）
            successful_cases = df[df[success_col] == True]
            avg_makespan = successful_cases[makespan_col].mean() if len(successful_cases) > 0 else float('inf')

            # 平均完成率
            avg_completion = df[completion_col].mean()

            # 平均执行时间
            exec_time_col = f'{alg}_execution_time'
            avg_exec_time = df[exec_time_col].mean()

            report['algorithm_comparison'][alg] = {
                'success_rate': success_rate,
                'avg_makespan': avg_makespan if avg_makespan != float('inf') else None,
                'avg_completion_rate': avg_completion,
                'avg_execution_time': avg_exec_time,
                'type': self.algorithms[alg]['type'],
                'description': self.algorithms[alg]['description']
            }

        # 排名分析
        rankings = {}

        # 完成时间排名（越小越好）
        makespan_ranking = []
        for alg in algorithms:
            avg_makespan = report['algorithm_comparison'][alg]['avg_makespan']
            if avg_makespan is not None:
                makespan_ranking.append((alg, avg_makespan))

        makespan_ranking.sort(key=lambda x: x[1])
        rankings['makespan_ranking'] = makespan_ranking

        # 完成率排名（越大越好）
        completion_ranking = []
        for alg in algorithms:
            completion_rate = report['algorithm_comparison'][alg]['avg_completion_rate']
            completion_ranking.append((alg, completion_rate))

        completion_ranking.sort(key=lambda x: x[1], reverse=True)
        rankings['completion_ranking'] = completion_ranking

        # 成功率排名（越大越好）
        success_ranking = []
        for alg in algorithms:
            success_rate = report['algorithm_comparison'][alg]['success_rate']
            success_ranking.append((alg, success_rate))

        success_ranking.sort(key=lambda x: x[1], reverse=True)
        rankings['success_ranking'] = success_ranking

        report['rankings'] = rankings

        return report

    def save_results(self, df: pd.DataFrame, report: Dict[str, Any]):
        """保存测试结果"""
        print("正在保存测试结果...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存详细结果
        results_file = os.path.join(self.results_dir, f'test_results_{timestamp}.csv')
        df.to_csv(results_file, index=False, encoding='utf-8-sig')

        # 保存分析报告
        report_file = os.path.join(self.results_dir, f'analysis_report_{timestamp}.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 保存综合报告
        comprehensive_file = os.path.join(self.results_dir, f'comprehensive_report_{timestamp}.txt')
        self._generate_text_report(df, report, comprehensive_file)

        print(f"结果已保存到:")
        print(f"  详细结果: {results_file}")
        print(f"  分析报告: {report_file}")
        print(f"  综合报告: {comprehensive_file}")

    def _generate_text_report(self, df: pd.DataFrame, report: Dict[str, Any], output_file: str):
        """生成文本格式的综合报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("USV任务调度算法对比实验综合报告\n")
            f.write("=" * 50 + "\n\n")

            # 基本信息
            summary = report['summary']
            f.write(f"测试时间: {summary['test_date']}\n")
            f.write(f"测试案例数量: {summary['total_test_cases']}\n")
            f.write(f"算法数量: {summary['total_algorithms']}\n\n")

            # 算法类型统计
            f.write("算法类型分布:\n")
            type_count = {}
            for alg, info in self.algorithms.items():
                alg_type = info['type']
                type_count[alg_type] = type_count.get(alg_type, 0) + 1

            for alg_type, count in type_count.items():
                f.write(f"  {alg_type}: {count}个\n")
            f.write("\n")

            # 排名结果
            rankings = report['rankings']

            f.write("算法性能排名:\n\n")

            f.write("1. 完成时间排名（越小越好）:\n")
            for i, (alg, avg_time) in enumerate(rankings['makespan_ranking'][:5], 1):
                f.write(f"   {i}. {alg}: {avg_time:.2f}\n")
            f.write("\n")

            f.write("2. 任务完成率排名（越大越好）:\n")
            for i, (alg, rate) in enumerate(rankings['completion_ranking'][:5], 1):
                f.write(f"   {i}. {alg}: {rate:.2%}\n")
            f.write("\n")

            f.write("3. 成功率排名（越大越好）:\n")
            for i, (alg, rate) in enumerate(rankings['success_ranking'][:5], 1):
                f.write(f"   {i}. {alg}: {rate:.2%}\n")
            f.write("\n")

            # 详细算法信息
            f.write("算法详细信息:\n")
            f.write("-" * 30 + "\n")

            for alg, comparison in report['algorithm_comparison'].items():
                f.write(f"\n{alg}:\n")
                f.write(f"  类型: {comparison['type']}\n")
                f.write(f"  描述: {comparison['description']}\n")
                f.write(f"  成功率: {comparison['success_rate']:.2%}\n")

                if comparison['avg_makespan'] is not None:
                    f.write(f"  平均完成时间: {comparison['avg_makespan']:.2f}\n")
                else:
                    f.write(f"  平均完成时间: 无成功案例\n")

                f.write(f"  平均任务完成率: {comparison['avg_completion_rate']:.2%}\n")
                f.write(f"  平均执行时间: {comparison['avg_execution_time']:.2f}秒\n")

    def run_full_test(self):
        """运行完整测试流程"""
        print("USV任务调度算法对比实验")
        print("=" * 60)

        # 选择测试案例
        test_cases = self.select_test_cases(20)

        # 运行测试
        df = self.run_comprehensive_test(test_cases)

        # 生成分析报告
        report = self.generate_analysis_report(df)

        # 保存结果
        self.save_results(df, report)

        # 显示简要结果
        self._display_summary(report)

        return df, report

    def _display_summary(self, report: Dict[str, Any]):
        """显示简要结果摘要"""
        print("\n" + "=" * 60)
        print("测试结果摘要")
        print("=" * 60)

        rankings = report['rankings']

        print("\n🏆 性能最佳算法:")

        print("\n1. 完成时间最快（前3名）:")
        for i, (alg, time) in enumerate(rankings['makespan_ranking'][:3], 1):
            print(f"   {i}. {alg}: {time:.2f}")

        print("\n2. 任务完成率最高（前3名）:")
        for i, (alg, rate) in enumerate(rankings['completion_ranking'][:3], 1):
            print(f"   {i}. {alg}: {rate:.2%}")

        print("\n3. 成功率最高（前3名）:")
        for i, (alg, rate) in enumerate(rankings['success_ranking'][:3], 1):
            print(f"   {i}. {alg}: {rate:.2%}")

        print("\n" + "=" * 60)


def main():
    """主函数"""
    framework = USVTestFramework()

    # 选择运行模式
    print("请选择运行模式:")
    print("1. 快速测试（5个案例，所有算法）")
    print("2. 标准测试（20个案例，所有算法）")
    print("3. 自定义测试")

    choice = input("请输入选择（1-3）: ").strip()

    if choice == '1':
        # 快速测试
        test_cases = framework.select_test_cases(5)
        df = framework.run_comprehensive_test(test_cases)
        report = framework.generate_analysis_report(df)
        framework.save_results(df, report)
        framework._display_summary(report)

    elif choice == '2':
        # 标准测试
        framework.run_full_test()

    elif choice == '3':
        # 自定义测试
        print("\n可用算法:")
        for i, alg in enumerate(framework.algorithms.keys(), 1):
            print(f"{i}. {alg}")

        selected_indices = input("请选择要测试的算法编号（用逗号分隔，如1,3,5）: ").strip()
        try:
            indices = [int(x.strip()) for x in selected_indices.split(',')]
            selected_algorithms = [list(framework.algorithms.keys())[i-1] for i in indices]

            num_cases = int(input("请输入测试案例数量（建议5-20）: ").strip())
            test_cases = framework.select_test_cases(num_cases)

            df = framework.run_comprehensive_test(test_cases, selected_algorithms)
            report = framework.generate_analysis_report(df)
            framework.save_results(df, report)
            framework._display_summary(report)

        except ValueError:
            print("输入格式错误，请重新运行程序")

    else:
        print("无效选择，运行标准测试")
        framework.run_full_test()


if __name__ == "__main__":
    main()