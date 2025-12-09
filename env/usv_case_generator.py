
import random
import json
import csv
import os
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


def calculate_fuzzy_expectation(a: float, m: float, b: float) -> float:
    """
    计算三角模糊数的期望值
    :param a: 最小值
    :param m: 最可能值
    :param b: 最大值
    :return: 期望值
    """
    return (a + 2 * m + b) / 4


def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    计算两点间欧几里得距离
    :param pos1: 位置1 (x, y)
    :param pos2: 位置2 (x, y)
    :return: 距离
    """
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def calculate_navigation_time(distance: float, speed: float) -> float:
    """
    计算航行时间
    :param distance: 距离
    :param speed: 航速
    :return: 航行时间
    """
    return distance / speed


@dataclass
class USVCaseData:
    """
    USV案例数据结构
    """
    # 基本信息
    num_usvs: int
    num_tasks: int
    case_id: str

    # 环境参数
    map_size: Tuple[int, int]
    start_point: Tuple[float, float]
    randomization_level: str
    seed: int

    # 环境固定参数
    environment_parameters: Dict[str, Any]

    # USV数据
    usv_positions: List[Tuple[float, float]]  # USV位置列表
    usv_initial_energy: List[float]           # USV初始电量

    # 任务数据
    task_positions: List[Tuple[float, float]]      # 任务位置列表
    task_types: List[str]                          # 任务类型列表
    task_execution_times: List[float]              # 任务执行时间期望值
    task_fuzzy_times: List[Tuple[float, float, float]]  # 任务模糊时间
    task_navigation_times: List[float]             # 任务从起点航行时间

    # 邻接矩阵
    task_usv_adjacency: List[List[int]]    # 任务-USV邻接矩阵
    task_predecessor: List[List[int]]      # 任务前驱关系矩阵


class USVCaseGenerator:
    """
    USV案例生成器核心类

    设计原则：
    1. 内存优先：默认在内存中生成数据，可选保存到文件
    2. 中等随机化：4×4分区策略，每区200×200，位置偏差±50
    3. 受控任务分配：温和随机分布，避免极端情况
    4. 完全兼容：与usv_env.py数据格式完全匹配
    """

    def __init__(self, num_usvs: int, num_tasks: int, path: str = '../data/',
                 flag_doc: bool = False, randomization_level: str = "medium",
                 seed: Optional[int] = None):
        """
        初始化USV案例生成器

        :param num_usvs: USV数量
        :param num_tasks: 任务数量
        :param path: 数据保存路径
        :param flag_doc: 是否保存到文件（默认False，内存优先）
        :param randomization_level: 随机化程度 ("low", "medium", "high")
        :param seed: 随机种子
        """
        self.num_usvs = num_usvs
        self.num_tasks = num_tasks
        self.path = path
        self.flag_doc = flag_doc
        self.randomization_level = randomization_level

        # 设置随机种子
        if seed is None:
            self.seed = int(time.time() * 1000) % 10000
        else:
            self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # 固定环境参数（来自usv_env.py）
        self.FIXED_PARAMETERS = {
            # 地图参数
            "map_size": (800, 800),
            "start_point": (0.0, 0.0),

            # USV物理参数
            "battery_capacity": 1200,
            "usv_speed": 5,
            "charge_time": 10,
            "energy_cost_per_distance": 1.0,
            "task_time_energy_ratio": 0.25,

            # 任务执行时间（三角模糊数）
            "task_service_time_fuzzy": {
                "Type1": (10.0, 20.0, 30.0),  # 期望值: 20.0
                "Type2": (30.0, 50.0, 80.0),  # 期望值: 52.5
                "Type3": (15.0, 25.0, 40.0)   # 期望值: 26.25
            },

            # 特征维度（4维x4维方案）
            "task_feat_dim": 4,
            "usv_feat_dim": 4
        }

        # 随机化配置
        self.RANDOMIZATION_CONFIG = self._get_randomization_config(randomization_level)

        # 确保保存路径存在
        if self.flag_doc and not os.path.exists(self.path):
            os.makedirs(self.path)

    def _get_randomization_config(self, level: str) -> Dict[str, Any]:
        """
        获取随机化配置

        :param level: 随机化程度
        :return: 随机化配置字典
        """
        if level == "low":
            return {
                "partition_strategy": "2x2_grid",
                "zone_size": (400, 400),
                "position_deviation": 25,
                "task_distribution": {
                    "Type1": (0.4, 0.6),
                    "Type2": (0.2, 0.4),
                    "Type3": (0.2, 0.4)
                }
            }
        elif level == "medium":
            return {
                "partition_strategy": "4x4_grid",
                "zone_size": (200, 200),
                "position_deviation": 50,
                "task_distribution": {
                    "Type1": (0.3, 0.5),  # 温和随机分布
                    "Type2": (0.2, 0.4),
                    "Type3": (0.2, 0.4)
                }
            }
        else:  # high
            return {
                "partition_strategy": "8x8_grid",
                "zone_size": (100, 100),
                "position_deviation": 100,
                "task_distribution": {
                    "Type1": (0.2, 0.8),
                    "Type2": (0.1, 0.7),
                    "Type3": (0.1, 0.7)
                }
            }

    def get_case(self, idx: int = 0) -> USVCaseData:
        """
        生成USV案例（核心方法）

        :param idx: 案例编号
        :return: USV案例数据对象
        """
        # 生成案例ID
        self.case_id = f"USV_N{self.num_usvs}_M{self.num_tasks}_E{str(idx+1).zfill(3)}"

        # 在内存中生成所有数据
        case_data = self._generate_case_data_in_memory()

        # 可选保存到文件
        if self.flag_doc:
            self._optional_save_to_file(case_data)

        return case_data

    def _generate_case_data_in_memory(self) -> USVCaseData:
        """
        在内存中生成完整的案例数据

        :return: USV案例数据对象
        """
        # 生成USV数据
        usv_positions, usv_initial_energy = self._generate_usvs_in_memory()

        # 生成任务数据
        (task_positions, task_types, task_execution_times,
         task_fuzzy_times, task_navigation_times) = self._generate_tasks_in_memory(usv_positions)

        # 生成邻接矩阵
        task_usv_adjacency, task_predecessor = self._generate_matrices_in_memory()

        # 创建案例数据对象
        case_data = USVCaseData(
            num_usvs=self.num_usvs,
            num_tasks=self.num_tasks,
            case_id=self.case_id,
            map_size=self.FIXED_PARAMETERS["map_size"],
            start_point=self.FIXED_PARAMETERS["start_point"],
            randomization_level=self.randomization_level,
            seed=self.seed,
            environment_parameters=self.FIXED_PARAMETERS.copy(),
            usv_positions=usv_positions,
            usv_initial_energy=usv_initial_energy,
            task_positions=task_positions,
            task_types=task_types,
            task_execution_times=task_execution_times,
            task_fuzzy_times=task_fuzzy_times,
            task_navigation_times=task_navigation_times,
            task_usv_adjacency=task_usv_adjacency,
            task_predecessor=task_predecessor
        )

        return case_data

    def _generate_usvs_in_memory(self) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        在内存中生成USV数据

        :return: USV位置列表, USV初始电量列表
        """
        usv_positions = []
        usv_initial_energy = []

        start_point = self.FIXED_PARAMETERS["start_point"]

        for usv_id in range(self.num_usvs):
            # 所有USV都从起点开始
            position = start_point
            energy = 1.0  # 满电状态

            usv_positions.append(position)
            usv_initial_energy.append(energy)

        return usv_positions, usv_initial_energy

    def _generate_tasks_in_memory(self, usv_positions: List[Tuple[float, float]]) -> Tuple[
        List[Tuple[float, float]], List[str], List[float],
        List[Tuple[float, float, float]], List[float]
    ]:
        """
        在内存中生成任务数据

        :param usv_positions: USV位置列表
        :return: 任务位置列表, 任务类型列表, 执行时间列表, 模糊时间列表, 航行时间列表
        """
        task_positions = []
        task_types = []
        task_execution_times = []
        task_fuzzy_times = []
        task_navigation_times = []

        map_size = self.FIXED_PARAMETERS["map_size"]
        start_point = self.FIXED_PARAMETERS["start_point"]
        fuzzy_times_config = self.FIXED_PARAMETERS["task_service_time_fuzzy"]

        # 根据随机化配置生成分区
        config = self.RANDOMIZATION_CONFIG
        zone_size = config["zone_size"]
        deviation = config["position_deviation"]

        for task_id in range(self.num_tasks):
            # 生成任务位置（有界随机化）
            position = self._generate_position_with_bounds(
                task_id, map_size, zone_size, deviation
            )
            task_positions.append(position)

            # 分配任务类型（受控随机化）
            task_type = self._assign_task_type_with_control(task_id, config["task_distribution"])
            task_types.append(task_type)

            # 计算执行时间
            fuzzy_time = fuzzy_times_config[task_type]
            execution_time = calculate_fuzzy_expectation(*fuzzy_time)
            task_execution_times.append(execution_time)
            task_fuzzy_times.append(fuzzy_time)

            # 计算从起点的航行时间
            distance = calculate_distance(start_point, position)
            navigation_time = calculate_navigation_time(
                distance, self.FIXED_PARAMETERS["usv_speed"]
            )
            task_navigation_times.append(navigation_time)

        return (task_positions, task_types, task_execution_times,
                task_fuzzy_times, task_navigation_times)

    def _generate_position_with_bounds(self, task_id: int, map_size: Tuple[int, int],
                                     zone_size: Tuple[int, int], deviation: int) -> Tuple[float, float]:
        """
        使用有界随机化策略生成位置

        :param task_id: 任务ID
        :param map_size: 地图尺寸
        :param zone_size: 分区尺寸
        :param deviation: 位置偏差
        :return: 位置坐标 (x, y)
        """
        # 计算任务所属分区
        zones_x = map_size[0] // zone_size[0]
        zones_y = map_size[1] // zone_size[1]
        total_zones = zones_x * zones_y

        # 根据任务ID均匀分配到不同分区
        zone_id = task_id % total_zones
        zone_x = zone_id % zones_x
        zone_y = zone_id // zones_x

        # 计算分区中心点
        center_x = zone_x * zone_size[0] + zone_size[0] // 2
        center_y = zone_y * zone_size[1] + zone_size[1] // 2

        # 在中心点周围±deviation范围内随机生成位置
        x = center_x + self._random_float(-deviation, deviation)
        y = center_y + self._random_float(-deviation, deviation)

        # 确保位置在地图边界内
        x = max(0, min(map_size[0] - 1, x))
        y = max(0, min(map_size[1] - 1, y))

        return (x, y)

    def _assign_task_type_with_control(self, task_id: int, distribution_config: Dict[str, Tuple[float, float]]) -> str:
        """
        使用受控策略分配任务类型

        :param task_id: 任务ID
        :param distribution_config: 分布配置
        :return: 任务类型
        """
        # 为每个任务类型生成随机比例
        type1_ratio = self._random_float(*distribution_config["Type1"])
        type2_ratio = self._random_float(*distribution_config["Type2"])
        type3_ratio = self._random_float(*distribution_config["Type3"])

        # 归一化比例
        total_ratio = type1_ratio + type2_ratio + type3_ratio
        type1_ratio /= total_ratio
        type2_ratio /= total_ratio
        type3_ratio /= total_ratio

        # 根据比例随机选择任务类型
        rand_val = self._random_probability()
        if rand_val < type1_ratio:
            return "Type1"
        elif rand_val < type1_ratio + type2_ratio:
            return "Type2"
        else:
            return "Type3"

    def _generate_task_usv_adjacency(self) -> List[List[int]]:
        """
        生成任务-USV邻接矩阵

        设计原则：所有USV都可以执行所有任务
        这表示一个完全连通的任务执行能力矩阵

        :return: 任务-USV邻接矩阵 (num_tasks × num_usvs)
        """
        # 创建完全连通矩阵：每个任务都可以由任何USV执行
        task_usv_adjacency = [[1 for _ in range(self.num_usvs)] for _ in range(self.num_tasks)]
        return task_usv_adjacency

    def _generate_task_predecessor_matrix(self) -> List[List[int]]:
        """
        生成任务前驱关系矩阵

        设计原则：任务间无前驱关系，所有任务可以并行执行
        这简化了调度问题的复杂度，适合作为基础测试案例

        :return: 任务前驱关系矩阵 (num_tasks × num_tasks)
        """
        # 创建零矩阵：表示任务间无前驱关系
        task_predecessor = [[0 for _ in range(self.num_tasks)] for _ in range(self.num_tasks)]
        return task_predecessor

    def _generate_matrices_in_memory(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        在内存中生成所有矩阵

        调用拆分后的矩阵生成方法，保持接口兼容性

        :return: 任务-USV邻接矩阵, 任务前驱关系矩阵
        """
        task_usv_adjacency = self._generate_task_usv_adjacency()
        task_predecessor = self._generate_task_predecessor_matrix()
        return task_usv_adjacency, task_predecessor

    def _optional_save_to_file(self, case_data: USVCaseData):
        """
        可选的文件保存功能

        :param case_data: 案例数据
        """
        # 保存JSON格式
        self._save_json_format(case_data)

        # 保存CSV格式
        self._save_csv_format(case_data)

    def _random_float(self, min_val: float, max_val: float) -> float:
        """
        生成指定范围内的随机浮点数

        :param min_val: 最小值
        :param max_val: 最大值
        :return: 随机浮点数
        """
        return random.uniform(min_val, max_val)

    def _random_probability(self) -> float:
        """
        生成0到1之间的随机概率值

        :return: 随机概率值 (0.0 <= value < 1.0)
        """
        return random.random()

    def _prepare_case_data_for_saving(self, case_data: USVCaseData) -> Dict[str, Any]:
        """
        准备用于保存的案例数据格式化方法

        :param case_data: 案例数据
        :return: 格式化后的数据字典
        """
        return {
            "case_id": case_data.case_id,
            "metadata": {
                "num_usvs": case_data.num_usvs,
                "num_tasks": case_data.num_tasks,
                "map_size": case_data.map_size,
                "start_point": case_data.start_point,
                "randomization_level": case_data.randomization_level,
                "seed": case_data.seed,
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "environment_parameters": case_data.environment_parameters,
            "usvs": [
                {
                    "usv_id": usv_id,
                    "start_position": list(case_data.usv_positions[usv_id]),
                    "initial_energy_ratio": case_data.usv_initial_energy[usv_id]
                }
                for usv_id in range(case_data.num_usvs)
            ],
            "tasks": [
                {
                    "task_id": task_id,
                    "position": list(case_data.task_positions[task_id]),
                    "type": case_data.task_types[task_id],
                    "execution_time": {
                        "fuzzy": list(case_data.task_fuzzy_times[task_id]),
                        "expected": case_data.task_execution_times[task_id]
                    },
                    "navigation_time_from_start": case_data.task_navigation_times[task_id]
                }
                for task_id in range(case_data.num_tasks)
            ],
            "matrices": {
                "task_usv_adjacency": case_data.task_usv_adjacency,
                "task_predecessor": case_data.task_predecessor
            }
        }

    def _save_json_format(self, case_data: USVCaseData):
        """
        保存为JSON格式

        :param case_data: 案例数据
        """
        # 使用统一的数据格式化方法
        json_data = self._prepare_case_data_for_saving(case_data)

        # 保存到文件
        filename = f"{case_data.case_id}.json"
        filepath = os.path.join(self.path, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    def _save_csv_format(self, case_data: USVCaseData):
        """
        保存为CSV格式（单文件完整格式）

        :param case_data: 案例数据
        """
        filename = f"{case_data.case_id}.csv"
        filepath = os.path.join(self.path, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入元数据
            writer.writerow(['# 元数据'])
            writer.writerow(['case_id', case_data.case_id])
            writer.writerow(['num_usvs', case_data.num_usvs])
            writer.writerow(['num_tasks', case_data.num_tasks])
            writer.writerow(['map_size', case_data.map_size[0], case_data.map_size[1]])
            writer.writerow(['start_point', case_data.start_point[0], case_data.start_point[1]])
            writer.writerow(['randomization_level', case_data.randomization_level])
            writer.writerow(['seed', case_data.seed])
            writer.writerow([])  # 空行

            # 写入环境参数
            writer.writerow(['# 环境参数'])
            for key, value in case_data.environment_parameters.items():
                if key != "task_service_time_fuzzy":  # 模糊时间单独处理
                    writer.writerow([key, value])
            writer.writerow([])  # 空行

            # 写入USV数据
            writer.writerow(['# USV数据'])
            writer.writerow(['usv_id', 'start_pos_x', 'start_pos_y', 'initial_energy_ratio'])
            for usv_id in range(case_data.num_usvs):
                writer.writerow([
                    usv_id,
                    case_data.usv_positions[usv_id][0],
                    case_data.usv_positions[usv_id][1],
                    case_data.usv_initial_energy[usv_id]
                ])
            writer.writerow([])  # 空行

            # 写入任务数据
            writer.writerow(['# 任务数据'])
            writer.writerow(['task_id', 'position_x', 'position_y', 'type',
                           'fuzzy_min', 'fuzzy_mode', 'fuzzy_max', 'expected_time', 'nav_time_from_start'])
            for task_id in range(case_data.num_tasks):
                fuzzy_min, fuzzy_mode, fuzzy_max = case_data.task_fuzzy_times[task_id]
                writer.writerow([
                    task_id,
                    case_data.task_positions[task_id][0],
                    case_data.task_positions[task_id][1],
                    case_data.task_types[task_id],
                    fuzzy_min, fuzzy_mode, fuzzy_max,
                    case_data.task_execution_times[task_id],
                    case_data.task_navigation_times[task_id]
                ])


# 预定义配置支持
CASE_COMBINATIONS = [
    # 2 USV × [40, 60, 80, 100, 120] 任务
    (2, 40), (2, 60), (2, 80), (2, 100), (2, 120),
    # 4 USV × [40, 60, 80, 100, 120] 任务
    (4, 40), (4, 60), (4, 80), (4, 100), (4, 120),
    # 6 USV × [40, 60, 80, 100, 120] 任务
    (6, 40), (6, 60), (6, 80), (6, 100), (6, 120),
    # 8 USV × [40, 60, 80, 100, 120] 任务
    (8, 40), (8, 60), (8, 80), (8, 100), (8, 120)
]


def create_generator(num_usvs: int, num_tasks: int, path: str = '../data/',
                    flag_doc: bool = False, randomization_level: str = "medium",
                    seed: Optional[int] = None) -> USVCaseGenerator:
    """
    便捷函数：创建USV案例生成器

    :param num_usvs: USV数量
    :param num_tasks: 任务数量
    :param path: 数据保存路径
    :param flag_doc: 是否保存到文件
    :param randomization_level: 随机化程度
    :param seed: 随机种子
    :return: USV案例生成器实例
    """
    return USVCaseGenerator(
        num_usvs=num_usvs,
        num_tasks=num_tasks,
        path=path,
        flag_doc=flag_doc,
        randomization_level=randomization_level,
        seed=seed
    )


def generate_all_cases(path: str = '../data/', flag_doc: bool = False,
                      randomization_level: str = "medium") -> Dict[str, USVCaseData]:
    """
    便捷函数：生成所有预定义案例组合

    :param path: 数据保存路径
    :param flag_doc: 是否保存到文件
    :param randomization_level: 随机化程度
    :return: 案例ID到案例数据的映射字典
    """
    all_cases = {}

    for idx, (num_usvs, num_tasks) in enumerate(CASE_COMBINATIONS):
        # 为每个组合创建一个唯一的种子
        seed = 1000 + idx

        # 创建生成器并生成案例
        generator = create_generator(
            num_usvs=num_usvs,
            num_tasks=num_tasks,
            path=path,
            flag_doc=flag_doc,
            randomization_level=randomization_level,
            seed=seed
        )

        # 生成案例
        case_data = generator.get_case(idx=0)
        all_cases[case_data.case_id] = case_data

        print(f"[PASS] 已生成案例: {case_data.case_id} ({num_usvs} USV, {num_tasks} 任务)")

    print(f"[SUCCESS] 总共生成 {len(all_cases)} 个案例")
    return all_cases


if __name__ == "__main__":
    # 示例用法：生成单个案例
    print("[START] 开始生成USV案例...")

    # 创建生成器（内存优先，不保存文件）
    generator = create_generator(
        num_usvs=4,
        num_tasks=80,
        path='../data/',
        flag_doc=True,  # 保存到文件用于调试
        randomization_level="medium"
    )

    # 生成案例
    case_data = generator.get_case(idx=0)

    # 打印基本信息
    print(f"📊 案例信息:")
    print(f"   案例ID: {case_data.case_id}")
    print(f"   USV数量: {case_data.num_usvs}")
    print(f"   任务数量: {case_data.num_tasks}")
    print(f"   地图尺寸: {case_data.map_size}")
    print(f"   随机化程度: {case_data.randomization_level}")
    print(f"   随机种子: {case_data.seed}")

    # 打印任务类型分布
    type_counts = {"Type1": 0, "Type2": 0, "Type3": 0}
    for task_type in case_data.task_types:
        type_counts[task_type] += 1
    print(f"   任务类型分布: {type_counts}")

    print(f"[PASS] 单个案例生成完成！")
    print("\n" + "="*50 + "\n")

    # 示例用法：生成所有组合案例
    print("[START] 开始生成所有组合案例...")
    all_cases = generate_all_cases(
        path='../data/',
        flag_doc=False,  # 仅在内存中生成
        randomization_level="medium"
    )

    print(f"[SUCCESS] 所有案例生成完成！总共 {len(all_cases)} 个案例")