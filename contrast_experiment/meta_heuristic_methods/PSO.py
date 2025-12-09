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


class PSOTaskPlanner(BasePlanner):
    """
    粒子群优化算法 (PSO) 任务规划器 - 用于多无人船任务调度

    算法特点：
    1. 模拟鸟群觅食行为的群体智能算法
    2. 每个粒子在解空间中飞行，根据个体经验和群体经验调整飞行方向
    3. 参数少，收敛速度快，实现简单
    4. 平衡全局搜索和局部搜索能力
    """

    def __init__(self, config: Dict = None):
        """
        初始化粒子群优化算法规划器
        """
        super().__init__(config)

        # PSO算法参数
        self.swarm_size = self.config.get('swarm_size', 100)  # 粒子群大小
        self.max_iterations = self.config.get('max_iterations', 200)  # 最大迭代次数
        self.inertia_weight = self.config.get('inertia_weight', 0.8)  # 惯性权重
        self.cognitive_weight = self.config.get('cognitive_weight', 2.0)  # 认知权重
        self.social_weight = self.config.get('social_weight', 2.0)  # 社会权重
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
        self.particles = []  # 粒子位置（解）
        self.velocities = []  # 粒子速度
        self.personal_best = []  # 个体最优位置
        self.personal_best_fitness = []  # 个体最优适应度
        self.global_best = None  # 全局最优位置
        self.global_best_fitness = float('inf')  # 全局最优适应度
        self.fitness_history = []  # 适应度历史

        # 日志和状态
        self.warnings = []
        self.failures = []

    def plan(self, env_data: Dict) -> Dict:
        """
        执行粒子群优化算法调度规划
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

        print(f"开始PSO优化: {len(tasks)}个任务, {len(usvs)}艘USV")

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
            'max_iterations': self.max_iterations,
            'best_fitness': self.global_best_fitness,
            'fitness_history': self.fitness_history
        })

        # 保存结果
        self.results = {
            'schedule': schedule_result,
            'makespan': metrics['makespan'],
            'metrics': metrics,
            'success': len(tasks) > 0 and self.global_best is not None
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
            task极assignments: 每个USV的任务列表，如[[0,1,2], [3,4], [5,6,7]]

        Returns:
            编码后的解向量，如[0,1,2,-1,3,4,-1,5,6,7]
            -1作为分隔符表示USV切换
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

    def _initialize_swarm(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        初始化粒子群
        """
        self.particles = []
        self.velocities = []
        num_tasks = len(tasks)
        num_usvs = len(usvs)

        # 生成初始粒子
        for _ in range(self.swarm_size):
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
            self.particles.append(solution)

            # 初始化速度（随机交换操作）
            velocity = []
            for _ in range(len(solution)):
                # 速度表示交换操作的概率和方向
                if random.random() < 0.1:  # 10%的概率有速度
                    # 随机选择一个交换目标位置
                    target_pos = random.randint(0, len(solution) - 1)
                    velocity.append((random.random(), target_pos))
                else:
                    velocity.append((0, -1))  # 无速度

            self.velocities.append(velocity)

        # 初始化个体最优
        self.personal_best = copy.deepcopy(self.particles)
        self.personal_best_fitness = [float('inf')] * self.swarm_size

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
            if distance_to_base > 1.0:  # 允许小的浮点误差
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
            task.assigned_usv极 = None
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
            usv.completion_time = current_time
            completion_times.append(current_time)

        return completion_times

    def _update_particle_velocity_and_position(self, particle_idx: int, tasks: List[Task]):
        """
        更新粒子速度和位置 - PSO核心算法
        """
        particle = self.particles[particle_idx]
        velocity = self.velocities[particle_idx]
        personal_best = self.personal_best[particle_idx]
        global_best = self.global_best

        new_particle = particle.copy()
        new_velocity = []

        # 对每个基因位置进行更新
        for i in range(len(particle)):
            current_gene = particle[i]

            # 跳过分隔符
            if current_gene == -1:
                new_particle[i] = -1
                new_velocity.append((0, -1))
                continue

            # 获取当前速度
            current_vel_prob, current_vel_target = velocity[i]

            # 生成随机因子
            r1, r2 = random.random(), random.random()

            # PSO速度更新公式
            new_vel_prob = (self.inertia_weight * current_vel_prob +
                           self.cognitive_weight * r1 * (1 if personal_best[i] != current_gene else 0) +
                           self.social_weight * r2 * (1 if global_best[i] != current_gene else 0))

            # 确定新的目标位置
            new_vel_target = -1
            if new_vel_prob > 0.5:  # 高概率执行交换
                # 优先向全局最优学习
                if global_best[i] != -1 and global_best[i] != current_gene:
                    new_vel_target = self._find_gene_position(new_particle, global_best[i])
                # 其次向个体最优学习
                elif personal_best[i] != -1 and personal_best[i] != current_gene:
                    new_vel_target = self._find_gene_position(new_particle, personal_best[i])

            new_velocity.append((new_vel_prob, new_vel_target))

            # 执行交换操作
            if new_vel_target != -1 and new_vel_target < len(new_particle):
                # 确保目标位置不是分隔符
                if new_particle[new_vel_target] != -1:
                    new_particle[i], new_particle[new_vel_target] = new_particle[new_vel_target], new_particle[i]

        # 应用变异操作防止早熟收敛
        if random.random() < 0.1:
            new_particle = self._mutate_particle(new_particle, tasks)

        self.particles[particle_idx] = new_particle
        self.velocities[particle_idx] = new_velocity

    def _find_gene_position(self, particle: List[int], gene: int) -> int:
        """
        在粒子中找到基因的位置
        """
        try:
            return particle.index(gene)
        except ValueError:
            return -1

    def _mutate_particle(self, particle: List[int], tasks: List[Task]) -> List[int]:
        """
        对粒子进行变异操作
        """
        mutated = particle.copy()

        # 随机选择变异操作
        mutation_type = random.choice(['swap', 'inversion', 'migration'])

        if mutation_type == 'swap' and len(mutated) >= 2:
            # 交换两个任务的位置
            task_positions = [i for i, gene in enumerate(mutated) if gene != -1]
            if len(task_positions) >= 2:
                idx1, idx2 = random.sample(task_positions, 2)
                mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

        elif mutation_type == 'inversion' and len(mutated) >= 2:
            # 反转一段任务序列
            task_positions = [i for i, gene in enumerate(mutated) if gene != -1]
            if len(task_positions) >= 2:
                start_idx, end_idx = sorted(random.sample(task_positions, 2))
                mutated[start_idx:end_idx+1] = reversed(mutated[start_idx:end_idx+1])

        elif mutation_type == 'migration':
            # 迁移任务到另一个USV
            task_assignments = self._decode_solution(mutated, len(tasks))
            if len(task_assignments) >= 2:
                source_usv = random.randint(0, len(task_assignments) - 1)
                target_usv = random.randint(0, len(task_assignments) - 1)

                if source_usv != target_usv and task_assignments[source_usv]:
                    task_to_move = random.choice(task_assignments[source_usv])
                    task_assignments[source_usv].remove(task_to_move)
                    task_assignments[target_usv].append(task_to_move)

                    mutated = self._encode_solution(task_assignments)

        return mutated

    def _execute_pso_optimization(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        执行粒子群优化算法
        """
        print(f"PSO算法开始优化 - 粒子群大小: {self.swarm_size}, 最大迭代: {self.max_iterations}")

        # 1. 初始化粒子群
        self._initialize_swarm(tasks, usvs, config)

        # 评估初始粒子群
        fitness_values = [self._evaluate_fitness(particle, tasks, usvs, config)
                         for particle in self.particles]

        # 初始化个体最优和全局最优
        self.personal_best_fitness = fitness_values.copy()
        self.personal_best = copy.deepcopy(self.particles)

        best_idx = np.argmin(fitness_values)
        self.global_best_fitness = fitness_values[best_idx]
        self.global_best = copy.deepcopy(self.particles[best_idx])
        self.fitness_history.append(self.global_best_fitness)

        print(f"  初始最佳适应度: {self.global_best_fitness:.2f}")

        # 2. 主循环
        for iteration in range(self.max_iterations):
            # 更新每个粒子
            for i in range(self.swarm_size):
                # 更新粒子速度和位置
                self._update_particle_velocity_and_position(i, tasks)

                # 评估新粒子
                new_fitness = self._evaluate_fitness(self.particles[i], tasks, usvs, config)

                # 更新个体最优
                if new_fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = new_fitness
                    self.personal_best[i] = copy.deepcopy(self.particles[i])

                # 更新全局最优
                if new_fitness < self.global_best_fitness:
                    self.global_best_fitness = new_fitness
                    self.global_best = copy.deepcopy(self.particles[i])

            # 动态调整惯性权重（线性递减）
            self.inertia_weight = 0.9 - (0.5 * iteration / self.max_iterations)

            # 记录适应度历史
            self.fitness_history.append(self.global_best_fitness)

            if iteration % 2 == 0:
                print(f"  迭代 {iteration + 1}/{self.max_iterations}, "
                      f"最佳适应度: {self.global_best_fitness:.2f}, "
                      f"惯性权重: {self.inertia_weight:.3f}")

        print(f"PSO算法优化完成，最佳适应度: {self.global_best_fitness:.2f}")

    def _apply_solution(self, tasks: List[Task], usvs: List[USV], config: Dict):
        """
        应用最优解到实际任务和USV
        """
        if self.global_best is None:
            return

        # 解码最优解
        task_assignments = self._decode_solution(self.global_best, len(tasks))

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


def run_single_case_pso(json_file: str, config: Dict = None) -> Dict:
    """运行PSO算法测试案例"""
    if config is None:
        config = {
            'swarm_size': 100,        # 粒子群大小
            'max_iterations': 200,     # 最大迭代次数
            'inertia_weight': 0.8,    # 惯性权重
            'cognitive_weight': 2.0,  # 认知权重
            'social_weight': 2.0,     # 社会权重
            'random_seed': 42,
            'energy_cost_per_unit_distance': 1.0,
            'task_time_energy_ratio': 0.25,
            'usv_initial_position': [0.0, 0.0],
            'charge_time': 10.0
        }

    planner = PSOTaskPlanner(config)
    env_data = load_and_adapt_data(json_file)
    return planner.plan(env_data)


def main():
    """主函数，用于测试PSO算法"""
    test_case_file = "../../usv_data_dev/40_8/usv_case_40_8_instance_02.json"

    if os.path.exists(test_case_file):
        print(f"运行PSO算法测试案例: {test_case_file}")
        result = run_single_case_pso(test_case_file)

        if result['success']:
            makespan = result.get('makespan')
            if makespan is not None:
                print(f"PSO调度成功！完成时间: {makespan:.2f}")
            else:
                print("PSO调度成功但没有完成时间")
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
            print("PSO调度失败！")
            if result.get('error'):
                print(f"错误信息: {result['error']}")
    else:
        print(f"测试案例文件不存在: {test_case_file}")


if __name__ == "__main__":
    main()

