import sys
import os
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Tuple
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_planner import BasePlanner, Task, USV
from utils import calculate_distance, simple_energy_model, DataConverter, plot_gantt_chart, plot_trajectory
from data_adapter import load_and_adapt_data

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class ABCTaskPlanner(BasePlanner):
    """
    人工蜂群算法 (ABC) 任务规划器 - 用于多无人船任务调度

    算法特点：
    1. 模拟蜜蜂采蜜行为的群体智能算法
    2. 包含雇佣蜂、观察蜂和侦察蜂三种角色
    3. 全局搜索和局部搜索平衡良好
    4. 参数少，收敛速度快
    """

    def __init__(self, config: Dict = None):
        """
        初始化人工蜂群算法规划器
        """
        super().__init__(config)

        # ABC算法参数
        self.colony_size = self.config.get('colony_size', 100)  # 蜂群大小
        self.max_cycles = self.config.get('max_cycles', 200)  # 最大循环次数
        self.limit = self.config.get('limit', 50)  # 限制次数
        self.random_seed = self.config.get('random_seed', 42)

        # 环境参数
        self.energy_cost_per_unit_distance = self.config.get('energy_cost_per_unit_distance', 1.0)
        self.task_time_energy_ratio = self.config.get('task_time_energy_ratio', 0.25)
        self.usv_initial_position = self.config.get('usv_initial_position', [0.0, 0.0])
        self.charge_time = self.config.get('charge_time', 10.0)

        # 设置随机种子
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # 算法状态
        self.food_sources = []  # 食物源（解）
        self.fitness_values = []  # 适应度值
        self.trial_counts = []  # 试验次数
        self.best_solution = None  # 最优解
        self.best_fitness = float('inf')  # 最优适应度
        self.fitness_history = []  # 适应度历史

        # 日志和状态
        self.warnings = []
        self.failures = []

    def plan(self, env_data: Dict) -> Dict:
        """
        执行人工蜂群算法调度规划
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

        print(f"开始ABC优化: {len(tasks)}个任务, {len(usvs)}艘USV")

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
            'max_cycles': self.max_cycles,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history
        })

        # 保存结果
        self.results = {
            'schedule': schedule_result,
            'makespan': metrics['makespan'],
            'metrics': metrics,
            'success': len(tasks) > 0 and self.best_solution is not None
        }

        # 生成甘特图和轨迹图
        if self.results['success']:
            plot_gantt_chart(tasks, usvs, config)
            plot_trajectory(tasks, usvs, config)

        return self.results

    def _encode_solution(self, task_assignments: List[List[int]]) -> List[int]:
        """
        将任务分配编码为解向量

        Args:
            task_assignments: 每个USV的任务列表

        Returns:
            编码后的解向量，使用-1作为分隔符
        """
        solution = []
        for usv_tasks in task_assignments:
            solution.extend(usv_tasks)
            solution.append(-1)  # 添加分隔符

        return solution[:-1]  # 移除最后一个分隔符

    def _decode_solution(self, solution: List[int], num_tasks: int) -> List[List[int]]:
        """
        从解向量解码任务分配
        """
        task_assignments = []
        current_usv_tasks = []

        for gene in solution:
            if gene == -1:  # 分隔符
                if current_usv_tasks:  # 避免空USV
                    task_assignments.append(current_usv_tasks)
                current_usv_tasks = []
            elif 0 <= gene < num_tasks:  # 有效任务ID
                current_usv_tasks.append(gene)

        # 添加最后一个USV的任务
        if current_usv_tasks:
            task_assignments.append(current_usv_tasks)

        return task_assignments

    def _initialize_food_sources(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        初始化食物源（随机解）
        """
        self.food_sources = []
        num_tasks = len(tasks)
        num_usvs = len(usvs)

        # 生成初始食物源
        for _ in range(self.colony_size):
            # 随机分配任务到USV
            assignments = [[] for _ in range(num_usvs)]
            task_indices = list(range(num_tasks))
            random.shuffle(task_indices)

            # 随机分配任务
            for task_idx in task_indices:
                usv_idx = random.randint(0, num_usvs - 1)
                assignments[usv_idx].append(task_idx)

            # 对每个USV的任务随机排序
            for usv_tasks in assignments:
                random.shuffle(usv_tasks)

            # 编码为解向量
            solution = self._encode_solution(assignments)
            self.food_sources.append(solution)

        # 初始化试验次数
        self.trial_counts = [0] * self.colony_size

    def _evaluate_fitness(self, solution: List[int], tasks: List[Task],
                          usvs: List[USV], config: Dict) -> float:
        """
        评估解的适应度
        """
        # 解码解向量
        task_assignments = self._decode_solution(solution, len(tasks))

        # 创建临时副本
        temp_tasks = copy.deepcopy(tasks)
        temp_usvs = copy.deepcopy(usvs)

        # 应用解并执行完整调度
        completion_times = self._apply_solution_complete(temp_tasks, temp_usvs,
                                                         task_assignments, config)

        # 适应度 = 最大完工时间
        fitness = max(completion_times) if completion_times else float('inf')

        # 检查所有任务是否完成
        unassigned_tasks = sum(1 for task in temp_tasks if task.assigned_usv is None)
        if unassigned_tasks > 0:
            fitness += 10000 * unassigned_tasks  # 严重惩罚

        # 检查所有USV是否返回起点
        for usv in temp_usvs:
            distance_to_base = calculate_distance(usv.position, config['env_paras']['start_point'])
            if distance_to_base > 1.0:
                fitness += 5000  # 未返回起点的惩罚

        return fitness

    def _apply_solution_complete(self, tasks: List[Task], usvs: List[USV],
                                 task_assignments: List[List[int]], config: Dict) -> List[float]:
        """
        应用解并执行完整调度
        """
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

        # 为每个USV执行任务序列
        completion_times = []
        for usv_idx, task_indices in enumerate(task_assignments):
            if usv_idx >= len(usvs) or not task_indices:
                completion_times.append(0.0)
                continue

            usv = usvs[usv_idx]
            current_time = 0.0
            current_position = config['env_paras']['start_point'].copy()
            current_battery = usv.battery_capacity

            # 记录出发事件
            usv.timeline.append({
                'time': current_time,
                'type': 'departure',
                'position': current_position.copy(),
                'battery': current_battery
            })

            # 执行任务序列
            for task_idx in task_indices:
                if task_idx >= len(tasks):
                    continue

                task = tasks[task_idx]

                # 检查是否需要充电
                if not self._can_execute_task_with_return(usv, task, current_position,
                                                          current_battery, config):
                    # 返回充电
                    travel_time, energy_used = self._return_to_base(usv, current_position, config)
                    current_time += travel_time + self.charge_time
                    current_battery = usv.battery_capacity
                    current_position = config['env_paras']['start_point'].copy()

                    # 记录充电事件
                    usv.timeline.append({
                        'time': current_time - self.charge_time,
                        'type': 'charging',
                        'duration': self.charge_time,
                        'battery_after': current_battery
                    })

                # 执行任务
                travel_time, task_time, energy_used = self._execute_single_task(
                    usv, task, current_position, current_battery, config)

                current_time += travel_time + task_time
                current_battery -= energy_used
                current_position = task.position.copy()

                # 记录任务分配
                task.assigned_usv = usv.usv_id
                task.start_time = current_time - task_time
                task.finish_time = current_time

            # 返回起点
            return_time, return_energy = self._return_to_base(usv, current_position, config)
            current_time += return_time
            current_battery -= return_energy

            usv.return_time = return_time
            usv.completion_time = current_time
            completion_times.append(current_time)

        return completion_times

    def _generate_neighbor_solution(self, solution: List[int], tasks: List[Task]) -> List[int]:
        """
        生成邻域解 - ABC算法的关键操作
        """
        neighbor = solution.copy()

        # 随机选择邻域操作
        operation = random.choice(['swap', 'inversion', 'migration', 'insertion'])

        if operation == 'swap' and len(neighbor) >= 2:
            # 交换两个任务的位置
            task_positions = [i for i, gene in enumerate(neighbor) if gene != -1]
            if len(task_positions) >= 2:
                idx1, idx2 = random.sample(task_positions, 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

        elif operation == 'inversion' and len(neighbor) >= 2:
            # 反转一段任务序列
            task_positions = [i for i, gene in enumerate(neighbor) if gene != -1]
            if len(task_positions) >= 2:
                start_idx, end_idx = sorted(random.sample(task_positions, 2))
                neighbor[start_idx:end_idx + 1] = reversed(neighbor[start_idx:end_idx + 1])

        elif operation == 'migration':
            # 迁移任务到另一个USV
            task_assignments = self._decode_solution(neighbor, len(tasks))
            if len(task_assignments) >= 2:
                source_usv = random.randint(0, len(task_assignments) - 1)
                target_usv = random.randint(0, len(task_assignments) - 1)

                if source_usv != target_usv and task_assignments[source_usv]:
                    task_to_move = random.choice(task_assignments[source_usv])
                    task_assignments[source_usv].remove(task_to_move)
                    task_assignments[target_usv].append(task_to_move)

                    neighbor = self._encode_solution(task_assignments)

        elif operation == 'insertion' and len(neighbor) >= 2:
            # 插入操作：将一个任务移动到新位置
            task_positions = [i for i, gene in enumerate(neighbor) if gene != -1]
            if len(task_positions) >= 2:
                from_idx, to_idx = random.sample(task_positions, 2)
                task = neighbor[from_idx]
                # 删除原位置
                del neighbor[from_idx]
                # 插入新位置
                neighbor.insert(to_idx, task)

        return neighbor

    def _calculate_probabilities(self) -> List[float]:
        """
        计算选择概率 - 基于适应度值
        """
        # 将适应度转换为适合轮盘赌选择的概率
        # 注意：我们是最小化问题，所以适应度值越小越好
        max_fitness = max(self.fitness_values)
        adjusted_fitness = [max_fitness - fit + 1e-10 for fit in self.fitness_values]

        total = sum(adjusted_fitness)
        probabilities = [fit / total for fit in adjusted_fitness]

        return probabilities

    def _execute_abc_optimization(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        执行人工蜂群算法优化
        """
        print(f"ABC算法开始优化 - 蜂群大小: {self.colony_size}, 最大循环: {self.max_cycles}")

        # 1. 初始化阶段
        self._initialize_food_sources(tasks, usvs, config)

        # 评估初始食物源
        self.fitness_values = [self._evaluate_fitness(solution, tasks, usvs, config)
                               for solution in self.food_sources]

        # 记录最优解
        best_idx = np.argmin(self.fitness_values)
        self.best_fitness = self.fitness_values[best_idx]
        self.best_solution = copy.deepcopy(self.food_sources[best_idx])
        self.fitness_history.append(self.best_fitness)

        print(f"  初始最佳适应度: {self.best_fitness:.2f}")

        # 主循环
        for cycle in range(self.max_cycles):
            # 2. 雇佣蜂阶段 - 开发已知食物源
            self._employed_bee_phase(tasks, usvs, config)

            # 3. 观察蜂阶段 - 基于概率选择食物源进行开发
            self._onlooker_bee_phase(tasks, usvs, config)

            # 4. 侦察蜂阶段 - 放弃劣质食物源，探索新区域
            self._scout_bee_phase(tasks, usvs, config)

            # 记录最优解
            best_idx = np.argmin(self.fitness_values)
            if self.fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness_values[best_idx]
                self.best_solution = copy.deepcopy(self.food_sources[best_idx])

            self.fitness_history.append(self.best_fitness)

            if cycle % 20 == 0:
                print(f"  循环 {cycle + 1}/{self.max_cycles}, 最佳适应度: {self.best_fitness:.2f}")

        print(f"ABC算法优化完成，最佳适应度: {self.best_fitness:.2f}")

    def _employed_bee_phase(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        雇佣蜂阶段：每个雇佣蜂在其食物源附近搜索新解
        """
        for i in range(self.colony_size):
            # 生成邻域解
            neighbor_solution = self._generate_neighbor_solution(self.food_sources[i], tasks)

            # 评估新解
            neighbor_fitness = self._evaluate_fitness(neighbor_solution, tasks, usvs, config)

            # 贪婪选择：如果新解更好，则替换原解
            if neighbor_fitness < self.fitness_values[i]:
                self.food_sources[i] = neighbor_solution
                self.fitness_values[i] = neighbor_fitness
                self.trial_counts[i] = 0  # 重置试验次数
            else:
                self.trial_counts[i] += 1  # 增加试验次数

    def _onlooker_bee_phase(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        观察蜂阶段：基于概率选择食物源进行开发
        """
        # 计算选择概率
        probabilities = self._calculate_probabilities()

        for _ in range(self.colony_size):
            # 轮盘赌选择食物源
            selected_idx = np.random.choice(range(self.colony_size), p=probabilities)

            # 生成邻域解
            neighbor_solution = self._generate_neighbor_solution(
                self.food_sources[selected_idx], tasks)

            # 评估新解
            neighbor_fitness = self._evaluate_fitness(neighbor_solution, tasks, usvs, config)

            # 贪婪选择
            if neighbor_fitness < self.fitness_values[selected_idx]:
                self.food_sources[selected_idx] = neighbor_solution
                self.fitness_values[selected_idx] = neighbor_fitness
                self.trial_counts[selected_idx] = 0
            else:
                self.trial_counts[selected_idx] += 1

    def _scout_bee_phase(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        侦察蜂阶段：放弃试验次数过多的食物源，随机生成新解
        """
        max_trials = max(self.trial_counts) if self.trial_counts else 0

        # 如果有食物源达到限制次数，则重新初始化
        if max_trials > self.limit:
            # 找到试验次数最多的食物源
            worst_idx = np.argmax(self.trial_counts)

            if self.trial_counts[worst_idx] > self.limit:
                # 重新初始化该食物源
                num_tasks = len(tasks)
                num_usvs = len(usvs)

                # 随机生成新解
                assignments = [[] for _ in range(num_usvs)]
                task_indices = list(range(num_tasks))
                random.shuffle(task_indices)

                for task_idx in task_indices:
                    usv_idx = random.randint(0, num_usvs - 1)
                    assignments[usv_idx].append(task_idx)

                for usv_tasks in assignments:
                    random.shuffle(usv_tasks)

                new_solution = self._encode_solution(assignments)
                new_fitness = self._evaluate_fitness(new_solution, tasks, usvs, config)

                # 替换旧解
                self.food_sources[worst_idx] = new_solution
                self.fitness_values[worst_idx] = new_fitness
                self.trial_counts[worst_idx] = 0

                print(f"    侦察蜂替换食物源 {worst_idx}, 新适应度: {new_fitness:.2f}")

    def _apply_solution(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        应用最优解到实际任务和USV
        """
        if self.best_solution is None:
            return

        # 解码最优解
        task_assignments = self._decode_solution(self.best_solution, len(tasks))

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
        completion_times = self._apply_solution_complete(tasks, usvs, task_assignments, config)

        max_completion_time = max(completion_times) if completion_times else 0
        print(f"最优解完成时间: {max_completion_time:.2f}")

        # 检查所有任务是否分配
        unassigned = sum(1 for task in tasks if task.assigned_usv is None)
        if unassigned > 0:
            self.failures.append(f"{unassigned}个任务未分配")
            print(f"警告: {unassigned}个任务未分配")

        # 打印分配结果
        print("任务分配结果:")
        for usv_idx, task_indices in enumerate(task_assignments):
            if usv_idx < len(usvs) and task_indices:
                usv_id = usvs[usv_idx].usv_id
                task_ids = [tasks[i].task_id for i in task_indices if i < len(tasks)]
                print(f"  USV {usv_id}: 任务 {task_ids}")

    # 辅助方法（与遗传算法相同）
    def _can_execute_task_with_return(self, usv: USV, task: Task, current_position: List[float],
                                      current_battery: float, config: Dict) -> bool:
        """检查USV能否执行任务并返回起点"""
        distance_to_task = calculate_distance(current_position, task.position)
        energy_to_task = distance_to_task * self.energy_cost_per_unit_distance
        energy_for_task = task.service_time * self.task_time_energy_ratio

        distance_to_base = calculate_distance(task.position, config['env_paras']['start_point'])
        energy_to_base = distance_to_base * self.energy_cost_per_unit_distance

        total_energy = energy_to_task + energy_for_task + energy_to_base

        return current_battery >= total_energy

    def _execute_single_task(self, usv: USV, task: Task, current_position: List[float],
                             current_battery: float, config: Dict) -> Tuple[float, float, float]:
        """执行单个任务"""
        distance_to_task = calculate_distance(current_position, task.position)
        travel_time = distance_to_task / usv.speed

        travel_energy = distance_to_task * self.energy_cost_per_unit_distance
        task_energy = task.service_time * self.task_time_energy_ratio
        total_energy = travel_energy + task_energy

        return travel_time, task.service_time, total_energy

    def _return_to_base(self, usv: USV, current_position: List[float], config: Dict) -> Tuple[float, float]:
        """返回起点"""
        distance = calculate_distance(current_position, config['env_paras']['start_point'])
        travel_time = distance / usv.speed
        energy_used = distance * self.energy_cost_per_unit_distance
        return travel_time, energy_used

    def validate_env_data(self, env_data: Dict) -> bool:
        """验证环境数据"""
        required_keys = ['config', 'tasks', 'usvs']
        for key in required_keys:
            if key not in env_data:
                self.warnings.append(f"缺少必需键: {key}")
                return False
        return True

    def compute_basic_metrics(self, schedule_result: Dict) -> Dict:
        """计算基本性能指标"""
        tasks = schedule_result['tasks']
        assigned_tasks = sum(1 for task in tasks if task.get('assigned_usv') is not None)
        unassigned_tasks = len(tasks) - assigned_tasks

        finish_times = [task.get('finish_time', 0) for task in tasks if task.get('finish_time')]
        makespan = max(finish_times) if finish_times else 0

        return {
            'assigned_tasks': assigned_tasks,
            'unassigned_tasks': unassigned_tasks,
            'makespan': makespan
        }


def run_single_case_abc(json_file: str, config: Dict = None) -> Dict:
    """运行ABC算法测试案例"""
    if config is None:
        config = {
            'colony_size': 1000,  # 蜂群大小
            'max_cycles': 200,  # 最大循环次数
            'limit': 50,  # 限制次数
            'random_seed': 42,
            'energy_cost_per_unit_distance': 1.0,
            'task_time_energy_ratio': 0.25,
            'usv_initial_position': [0.0, 0.0],
            'charge_time': 10.0
        }

    planner = ABCTaskPlanner(config)
    env_data = load_and_adapt_data(json_file)
    return planner.plan(env_data)


def main():
    """主函数，用于测试ABC算法"""
    test_case_file = "../../usv_data_dev/40_8/usv_case_40_8_instance_02.json"

    if os.path.exists(test_case_file):
        print(f"运行ABC算法测试案例: {test_case_file}")
        result = run_single_case_abc(test_case_file)

        if result['success']:
            makespan = result.get('makespan')
            if makespan is not None:
                print(f"ABC调度成功！完成时间: {makespan:.2f}")
            else:
                print("ABC调度成功但没有完成时间")
            print(f"已分配任务: {result['metrics']['assigned_tasks']}")
            print(f"未分配任务: {result['metrics']['unassigned_tasks']}")
            print(f"最佳适应度: {result['metrics']['best_fitness']:.2f}")

            # 检查所有USV是否返回起点
            for usv in result['schedule']['usvs']:
                position = usv.get('position', [0, 0])
                distance = calculate_distance(position, [0, 0])
                if distance > 1.0:
                    print(f"警告: USV {usv['usv_id']} 未返回起点")
        else:
            print("ABC调度失败！")
            if result.get('error'):
                print(f"错误信息: {result['error']}")
    else:
        print(f"测试案例文件不存在: {test_case_file}")


if __name__ == "__main__":
    main()