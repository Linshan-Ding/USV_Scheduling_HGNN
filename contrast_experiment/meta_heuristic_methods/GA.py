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
from utils import calculate_distance, simple_energy_model, DataConverter
from data_adapter import load_and_adapt_data

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class GATaskPlannerWithOrder(BasePlanner):
    """
    重构的遗传算法任务规划器 - 同时优化任务分配和执行顺序
    关键改进：
    1. 染色体编码同时包含任务分配和任务执行顺序
    2. 支持任务顺序的交叉和变异操作
    3. 更精确的适应度评估，考虑任务执行顺序
    """

    def __init__(self, config: Dict = None):
        """
        初始化遗传算法规划器
        """
        super().__init__(config)

        # GA算法参数
        self.population_size = self.config.get('population_size', 100)
        self.generations = self.config.get('generations', 200)
        self.crossover_rate = self.config.get('crossover_rate', 0.8)
        self.mutation_rate = self.config.get('mutation_rate', 0.15)
        self.elite_count = self.config.get('elite_count', 5)
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
        self.population = []
        self.fitness = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.fitness_history = []

        # 日志和状态
        self.warnings = []
        self.failures = []

    def plan(self, env_data: Dict) -> Dict:
        """
        执行遗传算法调度规划
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

        print(f"开始GA优化: {len(tasks)}个任务, {len(usvs)}艘USV")

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
            'algorithm': 'GA_with_Order',
            'population_size': self.population_size,
            'generations': self.generations,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history
        })

        # 保存结果
        self.results = {
            'schedule': schedule_result,
            'makespan': metrics['makespan'],
            'metrics': metrics,
            'success': len(tasks) > 0 and self.best_individual is not None
        }

        # 生成甘特图
        if self.results['success']:
            # 生成甘特图
            self.plot_gantt_chart(tasks, usvs, config)
            # 生成轨迹图
            self.plot_trajectory(tasks, usvs, config)

        return self.results

    def _encode_individual(self, task_assignments: List[List[int]]) -> List[int]:
        """
        将任务分配编码为染色体

        Args:
            task_assignments: 每个USV的任务列表，如[[0,1,2], [3,4], [5,6,7]]

        Returns:
            编码后的染色体，如[0,1,2,-1,3,4,-1,5,6,7]
            -1作为分隔符表示USV切换
        """
        chromosome = []
        for usv_tasks in task_assignments:
            chromosome.extend(usv_tasks)
            chromosome.append(-1)  # 添加分隔符

        return chromosome[:-1]  # 移除最后一个分隔符

    def _decode_individual(self, chromosome: List[int], num_tasks: int) -> List[List[int]]:
        """
        从染色体解码任务分配

        Args:
            chromosome: 编码的染色体
            num_tasks: 任务总数

        Returns:
            每个USV的任务列表
        """
        task_assignments = []
        current_usv_tasks = []

        for gene in chromosome:
            # print(gene, num_tasks, len(chromosome))
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

    def _initialize_population(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        初始化种群 - 包含任务分配和顺序
        """
        self.population = []
        num_tasks = len(tasks)
        num_usvs = len(usvs)

        # 方法1: 随机分配和随机顺序
        for _ in range(self.population_size // 3):
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

            # 编码为染色体
            chromosome = self._encode_individual(assignments)
            self.population.append(chromosome)

        # 方法2: 基于距离的启发式分配和排序
        for _ in range(self.population_size // 3):
            assignments = [[] for _ in range(num_usvs)]

            # 计算任务到起点的距离
            task_distances = []
            for task_idx, task in enumerate(tasks):
                distance = calculate_distance(config['env_paras']['start_point'], task.position)
                task_distances.append((task_idx, distance))

            # 按距离排序
            task_distances.sort(key=lambda x: x[1])

            # 分配任务
            for i, (task_idx, _) in enumerate(task_distances):
                usv_idx = i % num_usvs  # 轮询分配
                assignments[usv_idx].append(task_idx)

            # 对每个USV的任务按距离排序
            for usv_idx, usv_tasks in enumerate(assignments):
                # 计算任务到起点的距离并排序
                task_dists = []
                for task_idx in usv_tasks:
                    distance = calculate_distance(config['env_paras']['start_point'],
                                                  tasks[task_idx].position)
                    task_dists.append((task_idx, distance))

                task_dists.sort(key=lambda x: x[1])
                assignments[usv_idx] = [task_idx for task_idx, _ in task_dists]

            chromosome = self._encode_individual(assignments)
            self.population.append(chromosome)

        # 方法3: 平衡负载分配
        for _ in range(self.population_size - len(self.population)):
            assignments = [[] for _ in range(num_usvs)]
            task_indices = list(range(num_tasks))
            random.shuffle(task_indices)

            # 平衡分配
            for task_idx in task_indices:
                # 找到任务数最少的USV
                usv_sizes = [len(tasks) for tasks in assignments]
                usv_idx = np.argmin(usv_sizes)
                assignments[usv_idx].append(task_idx)

            # 对每个USV的任务随机排序
            for usv_tasks in assignments:
                random.shuffle(usv_tasks)

            chromosome = self._encode_individual(assignments)
            self.population.append(chromosome)

    def _evaluate_fitness(self, chromosome: List[int], tasks: List[Task],
                          usvs: List[USV], config: Dict) -> float:
        """
        评估个体的适应度 - 考虑任务执行顺序
        """
        # 解码染色体
        task_assignments = self._decode_individual(chromosome, len(tasks))

        # 创建临时副本
        temp_tasks = copy.deepcopy(tasks)
        temp_usvs = copy.deepcopy(usvs)

        # 应用个体并执行完整调度
        completion_times = self._apply_individual_with_order(temp_tasks, temp_usvs,
                                                             task_assignments, config)

        # 适应度 = 最后一个USV返回起点的时间（最大完工时间）
        fitness = max(completion_times) if completion_times else float('inf')

        # 检查所有任务是否完成
        unassigned_tasks = sum(1 for task in temp_tasks if task.assigned_usv is None)
        if unassigned_tasks > 0:
            # 如果有未完成任务，给予严重惩罚
            fitness += 10000 * unassigned_tasks

        # 检查所有USV是否返回起点
        for usv in temp_usvs:
            distance_to_base = calculate_distance(usv.position, config['env_paras']['start_point'])
            if distance_to_base > 1.0:  # 允许小的浮点误差
                fitness += 5000  # 未返回起点的惩罚

        return fitness

    def _apply_individual_with_order(self, tasks: List[Task], usvs: List[USV],
                                     task_assignments: List[List[int]], config: Dict) -> List[float]:
        """
        应用个体并执行调度，考虑任务执行顺序
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

            # 按指定顺序执行任务
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

                # 记录旅行事件
                usv.timeline.append({
                    'time': current_time,
                    'type': 'travel',
                    'duration': travel_time,
                    'from_position': current_position.copy(),
                    'to_position': task.position.copy(),
                    'task_id': task.task_id
                })

                current_time += travel_time
                current_battery -= energy_used

                # 记录任务开始事件
                usv.timeline.append({
                    'time': current_time,
                    'type': 'task_start',
                    'task_id': task.task_id,
                    'duration': task_time
                })

                current_time += task_time
                current_position = task.position.copy()

                # 记录任务完成事件
                usv.timeline.append({
                    'time': current_time,
                    'type': 'task_complete',
                    'task_id': task.task_id
                })

                # 记录任务分配
                task.assigned_usv = usv.usv_id
                task.start_time = current_time - task_time
                task.finish_time = current_time

            # 任务完成后返回起点
            return_time, return_energy = self._return_to_base(usv, current_position, config)

            # 记录返回旅行
            usv.timeline.append({
                'time': current_time,
                'type': 'return_travel',
                'duration': return_time,
                'from_position': current_position.copy(),
                'to_position': config['env_paras']['start_point'].copy()
            })

            current_time += return_time
            current_battery -= return_energy

            # 记录返回事件
            usv.timeline.append({
                'time': current_time,
                'type': 'return',
                'position': config['env_paras']['start_point'].copy(),
                'battery': current_battery
            })

            usv.return_time = return_time
            completion_times.append(current_time)

        return completion_times

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

            self.fitness_history.append(self.best_fitness)

            if generation % 20 == 0:
                print(f"  代数 {generation + 1}/{self.generations}, 最佳适应度: {self.best_fitness:.2f}")

            # 选择、交叉、变异
            self._selection()
            self._crossover(tasks)
            self._mutation(tasks)

        print(f"GA算法优化完成，最佳适应度: {self.best_fitness:.2f}")

    def _selection(self):
        """选择操作：锦标赛选择 + 精英保留"""
        # 保留精英个体
        elite_indices = np.argsort(self.fitness)[:self.elite_count]
        new_population = [copy.deepcopy(self.population[i]) for i in elite_indices]

        # 锦标赛选择填充剩余种群
        tournament_size = 3
        while len(new_population) < self.population_size:
            # 随机选择参赛个体
            contestants = random.sample(range(len(self.population)), tournament_size)
            # 选择适应度最好的（数值小的）
            winner_idx = min(contestants, key=lambda x: self.fitness[x])
            new_population.append(copy.deepcopy(self.population[winner_idx]))

        self.population = new_population

        if [] in self.population:
            print("选择后种群数量", len(self.population))
            print("选择后存在非法空染色体")

    def _crossover(self, tasks: List[Task]):
        """交叉操作：顺序交叉 (Order Crossover, OX) 和基于分隔符的交叉"""
        new_population = copy.deepcopy(self.population[:self.elite_count])  # 保留精英

        for i in range(self.elite_count, self.population_size, 2):
            if i + 1 >= self.population_size:
                break

            parent1 = self.population[i]
            parent2 = self.population[i + 1]

            if random.random() < self.crossover_rate:
                # 随机选择交叉方法
                if random.random() < 0.5:
                    child1, child2 = self._order_crossover(parent1, parent2, tasks)
                else:
                    child1, child2 = self._separator_based_crossover(parent1, parent2, tasks)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            new_population.extend([child1, child2])

        # 保持种群大小
        self.population = new_population[:self.population_size + 1]

        if [] in self.population:
            print("交叉后种群数量", len(self.population))
            print("交叉后存在非法空染色体")

    def _order_crossover(self, parent1: List[int], parent2: List[int], tasks: List[Task]) -> Tuple[
        List[int], List[int]]:
        """顺序交叉操作 - 保持任务顺序"""
        # 移除分隔符，只对任务ID进行交叉
        parent1_tasks = [gene for gene in parent1 if gene != -1]
        parent2_tasks = [gene for gene in parent2 if gene != -1]

        size = len(parent1_tasks)
        # 选择交叉点
        start, end = sorted(random.sample(range(size), 2))

        # 执行顺序交叉
        child1_tasks = [-1] * size
        child2_tasks = [-1] * size

        # 复制父代片段
        child1_tasks[start:end] = parent1_tasks[start:end]
        child2_tasks[start:end] = parent2_tasks[start:end]

        child1_tasks = self._fill_remaining(child1_tasks, parent2_tasks, start, end)
        child2_tasks = self._fill_remaining(child2_tasks, parent1_tasks, start, end)

        # 重新添加分隔符（使用父代的分隔符模式）
        # 找到分隔符位置
        separators1 = [i for i, gene in enumerate(parent1) if gene == -1]
        separators2 = [i for i, gene in enumerate(parent2) if gene == -1]
        for sq1, sq2 in zip(separators1, separators2):
            child1_tasks.insert(sq2, -1)
            child2_tasks.insert(sq1, -1)

        return child1_tasks, child2_tasks

    def _fill_remaining(self, child, parent, start, end):
        remain_genes = []
        for gene in parent:
            if gene not in child[start:end]:
                remain_genes.append(gene)
        # # 输出信息
        # print(f"The number of gene to add is {len(remain_genes)}")
        # print(f"start is {start}, end is {end}")
        # 染色体信息替换
        child[:start] = remain_genes[:start]
        child[end:] = remain_genes[start:]

        return child

    def _separator_based_crossover(self, parent1: List[int], parent2: List[int], tasks: List[Task]) -> Tuple[
        List[int], List[int]]:
        """基于分隔符的交叉 - 保持USV分配结构"""
        # 找到分隔符位置
        separators1 = [i for i, gene in enumerate(parent1) if gene == -1]
        separators2 = [i for i, gene in enumerate(parent2) if gene == -1]

        # 选择交叉点（在USV边界）
        if not separators1 or not separators2:
            return parent1.copy(), parent2.copy()

        # 移除parent1和parent2中所有的分割符
        parent1 = [gene for gene in parent1 if gene != -1]
        parent2 = [gene for gene in parent2 if gene != -1]

        # 执行交叉
        for sq1, sq2 in zip(separators1, separators2):
            parent1.insert(sq2, -1)
            parent2.insert(sq1, -1)
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        return child1, child2

    def _mutation(self, tasks: List[Task]):
        """变异操作：交换变异 + 反转变异 + 迁移变异"""
        for i in range(self.elite_count, len(self.population)):  # 精英个体不变异
            individual = self.population[i]

            if random.random() < self.mutation_rate:
                # 随机选择变异方法
                mutation_type = random.random()

                if mutation_type < 0.4:  # 交换变异
                    self._swap_mutation(individual)
                elif mutation_type < 0.7:  # 反转变异
                    self._inversion_mutation(individual)
                else:  # 迁移变异
                    self._migration_mutation(individual, tasks)

    def _swap_mutation(self, individual: List[int]):
        """交换变异：交换两个任务的位置"""
        # 找到所有任务位置（非分隔符）
        task_positions = [i for i, gene in enumerate(individual) if gene != -1]

        if len(task_positions) >= 2:
            idx1, idx2 = random.sample(task_positions, 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    def _inversion_mutation(self, individual: List[int]):
        """反转变异：反转一段任务序列"""
        # 找到所有任务位置（非分隔符）
        task_positions = [i for i, gene in enumerate(individual) if gene != -1]

        if len(task_positions) >= 2:
            start_idx, end_idx = sorted(random.sample(task_positions, 2))

            # 找到对应的实际索引
            start_pos = individual.index(individual[start_idx])
            end_pos = individual.index(individual[end_idx])

            # 反转序列
            individual[start_pos:end_pos + 1] = reversed(individual[start_pos:end_pos + 1])

    def _migration_mutation(self, individual: List[int], tasks: List[Task]):
        """迁移变异：将任务从一个USV迁移到另一个USV"""
        # 解码染色体
        task_assignments = self._decode_individual(individual, len(tasks))

        if len(task_assignments) < 2:
            return

        # 选择源USV和目标USV
        source_usv = random.randint(0, len(task_assignments) - 1)
        target_usv = random.randint(0, len(task_assignments) - 1)

        if source_usv == target_usv or not task_assignments[source_usv]:
            return

        # 迁移任务
        task_to_move = random.choice(task_assignments[source_usv])
        task_assignments[source_usv].remove(task_to_move)
        task_assignments[target_usv].append(task_to_move)

        # 重新编码
        new_individual = self._encode_individual(task_assignments)
        individual[:] = new_individual

    def _apply_solution(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """应用最优解到实际任务和USV"""
        if self.best_individual is None:
            return

        # 解码最优解
        task_assignments = self._decode_individual(self.best_individual, len(tasks))

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
        completion_times = self._apply_individual_with_order(tasks, usvs, task_assignments, config)

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

    def plot_gantt_chart(self, tasks: List[Task], usvs: List[USV], config: Dict):
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

    def plot_trajectory(self, tasks: List[Task], usvs: List[USV], config: Dict,
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

        # 设置主标题
        fig.suptitle('多无人船任务执行轨迹图 - 遗传算法优化结果',
                     fontsize=18, fontweight='bold', y=0.95)

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
        completion_time = max([task.finish_time for task in tasks if task.finish_time is not None]) if any(
            task.finish_time is not None for task in tasks) else 0

        stats_text = f"统计信息:\n总任务数: {total_tasks}\n已分配任务: {assigned_tasks}\n完成时间: {completion_time:.1f}"

        # 创建图例
        legend = ax_legend.legend(handles=legend_elements, loc='center',
                                  fontsize=10, framealpha=0.9, fancybox=True)

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

    # 辅助方法（与之前相同）
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
        """验证环境数据（简化实现）"""
        required_keys = ['config', 'tasks', 'usvs']
        for key in required_keys:
            if key not in env_data:
                self.warnings.append(f"缺少必需键: {key}")
                return False
        return True

    def compute_basic_metrics(self, schedule_result: Dict) -> Dict:
        """计算基本性能指标（简化实现）"""
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


def run_single_case(json_file: str, config: Dict = None) -> Dict:
    """运行单个测试案例"""
    if config is None:
        config = {
            'population_size': 1000,
            'generations': 200,
            'crossover_rate': 0.8,
            'mutation_rate': 0.15,
            'elite_count': 5,
            'random_seed': 42,
            'energy_cost_per_unit_distance': 1.0,
            'task_time_energy_ratio': 0.25,
            'usv_initial_position': [0.0, 0.0],
            'charge_time': 10.0
        }

    planner = GATaskPlannerWithOrder(config)
    env_data = load_and_adapt_data(json_file)
    return planner.plan(env_data)


def main():
    """主函数，用于测试"""
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

            # 检查所有USV是否返回起点
            for usv in result['schedule']['usvs']:
                position = usv.get('position', [0, 0])
                distance = calculate_distance(position, [0, 0])
                if distance > 1.0:
                    print(f"警告: USV {usv['usv_id']} 未返回起点")
        else:
            print("GA调度失败！")
            if result.get('error'):
                print(f"错误信息: {result['error']}")
    else:
        print(f"测试案例文件不存在: {test_case_file}")


if __name__ == "__main__":
    main()