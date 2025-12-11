# ============================================================================
# 1. instance_generator.py - 实例生成模块 (生成20组混合算例)
# ============================================================================
import numpy as np
import pickle
import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class InstanceConfig:
    """实例配置参数"""
    n_usvs: int = 4
    n_tasks: int = 40
    map_size: Tuple[int, int] = (1000, 1000)
    battery_capacity: float = 500.0
    usv_speed: float = 5.0
    charge_time: float = 10.0
    energy_cost_per_distance: float = 1.0
    task_time_per_energy: float = 0.25


class InstanceGenerator:
    def __init__(self, config: InstanceConfig):
        self.config = config

    def generate(self, seed: int = None) -> dict:
        if seed is not None:
            np.random.seed(seed)

        # 任务坐标
        task_coords = np.random.uniform(0, self.config.map_size[0], (self.config.n_tasks, 2))

        # 模糊时间
        t2 = np.random.uniform(5, 20, self.config.n_tasks)
        t1 = t2 * np.random.uniform(0.7, 0.9, self.config.n_tasks)
        t3 = t2 * np.random.uniform(1.1, 1.3, self.config.n_tasks)
        fuzzy_times = np.stack([t1, t2, t3], axis=1)

        instance = {
            'n_usvs': self.config.n_usvs,
            'n_tasks': self.config.n_tasks,
            'task_coords': task_coords,
            'fuzzy_times': fuzzy_times,
            'config': self.config,
            'seed': seed
        }
        return instance

    def save_instances(self, num_instances, save_dir='./data'):
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"u{self.config.n_usvs}_t{self.config.n_tasks}"
        print(f"Generating {num_instances} instances for: {prefix}...")

        for i in range(num_instances):
            # 这里的seed策略：为了避免不同配置间生成完全一样的分布，加入配置参数偏移
            seed = 1000 + i + self.config.n_usvs * 100 + self.config.n_tasks
            instance = self.generate(seed=seed)

            filename = os.path.join(save_dir, f"{prefix}_{i}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(instance, f)


def generate_benchmark_datasets():
    """生成 20 组不同规模的基准数据集"""

    # 定义 20 种组合
    usv_counts = [4, 6, 8, 10]
    task_counts = [20, 40, 60, 80, 100]

    # 组合列表
    configs = []
    for u in usv_counts:
        for t in task_counts:
            configs.append((u, t))

    print(f"Total Configurations: {len(configs)}")

    save_dir = './data/train_eval_mixed'

    for n_usvs, n_tasks in configs:
        cfg = InstanceConfig(n_usvs=n_usvs, n_tasks=n_tasks)
        generator = InstanceGenerator(cfg)

        # 为每种配置生成 50 个算例 (40训练 + 10测试)
        # 总共 20 * 50 = 1000 个文件
        generator.save_instances(50, save_dir=save_dir)


if __name__ == "__main__":
    generate_benchmark_datasets()