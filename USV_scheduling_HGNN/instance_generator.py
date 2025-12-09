# ============================================================================
# 1. instance_generator.py - 实例生成模块
# ============================================================================
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class InstanceConfig:
    """实例配置参数"""
    n_usvs: int = 4  # USV数量
    n_tasks: int = 40  # 任务数量
    map_size: Tuple[int, int] = (800, 800)  # 地图大小
    battery_capacity: float = 400.0  # 电池容量
    usv_speed: float = 5.0  # USV速度
    charge_time: float = 10.0  # 充电时间
    energy_cost_per_distance: float = 1.0  # 单位距离能耗
    task_time_per_energy: float = 0.25  # 任务时间系数


class InstanceGenerator:
    """USV调度问题实例生成器"""

    def __init__(self, config: InstanceConfig):
        self.config = config

    def generate(self, seed: int = None) -> dict:
        """生成一个随机实例"""
        if seed is not None:
            np.random.seed(seed)

        # 生成任务坐标（均匀分布）
        task_coords = np.random.uniform(
            0, self.config.map_size[0],
            (self.config.n_tasks, 2)
        )

        # 生成三角模糊处理时间 (t1, t2, t3)
        # t2是最可能时间，t1和t3是最短和最长
        t2 = np.random.uniform(5, 20, self.config.n_tasks)
        t1 = t2 * np.random.uniform(0.7, 0.9, self.config.n_tasks)
        t3 = t2 * np.random.uniform(1.1, 1.3, self.config.n_tasks)
        fuzzy_times = np.stack([t1, t2, t3], axis=1)

        instance = {
            'n_usvs': self.config.n_usvs,
            'n_tasks': self.config.n_tasks,
            'task_coords': task_coords,
            'fuzzy_times': fuzzy_times,
            'config': self.config
        }

        return instance