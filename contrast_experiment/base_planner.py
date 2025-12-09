#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USV调度算法基类
为所有对比算法提供统一的接口和数据结构
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
import os


class BasePlanner(ABC):
    """USV调度算法基类"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化规划器

        Args:
            config: 算法配置参数
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.results = {}

    @abstractmethod
    def plan(self, env_data: Dict) -> Dict:
        """
        执行调度规划

        Args:
            env_data: 环境数据，包含任务、USV等信息

        Returns:
            调度结果字典，包含：
            - schedule: 调度方案
            - makespan: 总完成时间
            - metrics: 其他性能指标
            - success: 是否成功完成调度
        """
        pass

    def load_env_from_json(self, json_file: str) -> Dict:
        """
        从JSON文件加载环境数据

        Args:
            json_file: JSON文件路径

        Returns:
            环境数据字典
        """
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"环境数据文件不存在: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def validate_env_data(self, env_data: Dict) -> bool:
        """
        验证环境数据的完整性

        Args:
            env_data: 环境数据

        Returns:
            验证是否通过
        """
        required_keys = ['config', 'tasks', 'usvs']
        for key in required_keys:
            if key not in env_data:
                print(f"错误：环境数据缺少必需字段: {key}")
                return False
        return True

    def compute_basic_metrics(self, schedule_result: Dict) -> Dict:
        """
        计算基本性能指标

        Args:
            schedule_result: 调度结果

        Returns:
            性能指标字典
        """
        metrics = {
            'total_tasks': len(schedule_result.get('tasks', [])),
            'total_usvs': len(schedule_result.get('usvs', [])),
            'assigned_tasks': 0,
            'unassigned_tasks': 0,
            'makespan': None,
            'total_distance': 0.0,
            'total_energy': 0.0
        }

        # 计算已分配和未分配任务数
        for task in schedule_result.get('tasks', []):
            if task.get('assigned_usv') is not None:
                metrics['assigned_tasks'] += 1
            else:
                metrics['unassigned_tasks'] += 1

        # 计算最大完成时间
        finish_times = []
        for usv in schedule_result.get('usvs', []):
            if usv.get('current_time', 0) > 0:
                finish_times.append(usv['current_time'])

        if finish_times:
            metrics['makespan'] = max(finish_times)

        return metrics

    def get_algorithm_info(self) -> Dict:
        """
        获取算法信息

        Returns:
            算法信息字典
        """
        return {
            'name': self.name,
            'type': self.__class__.__bases__[0].__name__,
            'config': self.config,
            'description': self.__doc__ or "无描述"
        }

    def save_results(self, output_file: str):
        """
        保存调度结果

        Args:
            output_file: 输出文件路径
        """
        output_data = {
            'algorithm': self.get_algorithm_info(),
            'results': self.results
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


class Task:
    """任务数据类"""

    def __init__(self, task_id: int, position: List[float], service_time: float,
                 priority: int = 0, assigned_usv: Optional[int] = None,
                 start_time: Optional[float] = None, finish_time: Optional[float] = None):
        self.task_id = task_id
        self.position = position
        self.service_time = service_time
        self.priority = priority
        self.assigned_usv = assigned_usv
        self.start_time = start_time
        self.finish_time = finish_time

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'task_id': self.task_id,
            'position': self.position,
            'service_time': self.service_time,
            'priority': self.priority,
            'assigned_usv': self.assigned_usv,
            'start_time': self.start_time,
            'finish_time': self.finish_time
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """从字典创建任务对象"""
        return cls(**data)


class USV:
    """USV数据类"""

    def __init__(self, usv_id: int, position: List[float], battery_capacity: float,
                 battery_level: float, speed: float, charge_time: float,
                 timeline: Optional[List[Dict]] = None, current_time: float = 0.0):
        self.usv_id = usv_id
        self.position = position
        self.battery_capacity = battery_capacity
        self.battery_level = battery_level
        self.speed = speed
        self.charge_time = charge_time
        self.timeline = timeline or []
        self.current_time = current_time

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'usv_id': self.usv_id,
            'position': self.position,
            'battery_capacity': self.battery_capacity,
            'battery_level': self.battery_level,
            'speed': self.speed,
            'charge_time': self.charge_time,
            'timeline': self.timeline,
            'current_time': self.current_time
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'USV':
        """从字典创建USV对象"""
        return cls(**data)

    def distance_to(self, point: List[float]) -> float:
        """计算到指定点的距离"""
        import math
        return math.sqrt((self.position[0] - point[0])**2 +
                        (self.position[1] - point[1])**2)


def create_standard_env_data(json_file: str) -> Dict:
    """
    从JSON文件创建标准格式的环境数据

    Args:
        json_file: JSON文件路径

    Returns:
        标准格式的环境数据
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 转换为标准格式
    env_data = {
        'config': raw_data.get('config', {}),
        'tasks': [Task.from_dict(t).to_dict() for t in raw_data.get('tasks', [])],
        'usvs': [USV.from_dict(u).to_dict() for u in raw_data.get('usvs', [])]
    }

    return env_data