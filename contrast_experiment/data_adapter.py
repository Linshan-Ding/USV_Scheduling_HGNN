#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USV数据适配器
将不同格式的USV案例数据转换为标准格式，供算法使用
"""

import json
from typing import Dict, List, Any
from base_planner import Task, USV


class USVDataAdapter:
    """USV数据适配器"""

    @staticmethod
    def adapt_usv_case_format(raw_data: Dict) -> Dict:
        """
        适配USV案例格式到标准格式

        Args:
            raw_data: 原始USV案例数据

        Returns:
            标准格式的环境数据
        """
        # 提取环境参数
        env_params = raw_data.get("环境参数", {})
        fixed_params = env_params.get("环境固定参数", {})

        # 构建标准config
        config = {
            "env_paras": {
                "num_usvs": raw_data.get("基本信息", {}).get("USV数量", 2),
                "num_tasks": raw_data.get("基本信息", {}).get("任务数量", 40),
                "map_size": fixed_params.get("map_size", [800, 800]),
                "battery_capacity": fixed_params.get("battery_capacity", 1200),
                "usv_speed": fixed_params.get("usv_speed", 5),
                "charge_time": fixed_params.get("charge_time", 10),
                "energy_cost_per_distance": fixed_params.get("energy_cost_per_distance", 1.0),
                "task_time_energy_ratio": fixed_params.get("task_time_energy_ratio", 0.25),
                "start_point": fixed_params.get("start_point", [0.0, 0.0])
            },
            "random_seed": raw_data.get("基本信息", {}).get("种子", 42)
        }

        # 转换任务数据
        tasks_data_dict = raw_data.get("任务数据", {})
        task_positions = tasks_data_dict.get("任务位置", [])
        task_types = tasks_data_dict.get("任务类型", ["Type1"] * len(task_positions))
        task_service_times = tasks_data_dict.get("任务服务时间", [20.0] * len(task_positions))

        tasks = []
        for i, position in enumerate(task_positions):
            # 从模糊服务时间中选择一个值（这里简化处理，取中间值）
            service_time = task_service_times[i] if i < len(task_service_times) else 20.0
            if isinstance(service_time, list):
                service_time = service_time[1] if len(service_time) > 1 else service_time[0]

            task = {
                "task_id": i + 1,
                "position": position,
                "service_time": float(service_time),
                "priority": 0,
                "assigned_usv": None,
                "start_time": None,
                "finish_time": None
            }
            tasks.append(task)

        # 转换USV数据
        usv_data_dict = raw_data.get("USV数据", {})
        usv_positions = usv_data_dict.get("USV位置", [])
        usv_battery_levels = usv_data_dict.get("USV初始电量", [1.0] * len(usv_positions))

        if not usv_positions:
            # 如果没有USV数据，根据配置创建默认USV
            num_usvs = config["env_paras"]["num_usvs"]
            start_point = config["env_paras"]["start_point"]
            battery_capacity = config["env_paras"]["battery_capacity"]
            usv_speed = config["env_paras"]["usv_speed"]
            charge_time = config["env_paras"]["charge_time"]

            usvs = []
            for i in range(num_usvs):
                usv = {
                    "usv_id": i,
                    "position": start_point.copy(),
                    "battery_capacity": battery_capacity,
                    "battery_level": battery_capacity,  # 初始满电
                    "speed": usv_speed,
                    "charge_time": charge_time,
                    "timeline": [],
                    "current_time": 0.0
                }
                usvs.append(usv)
        else:
            usvs = []
            battery_capacity = config["env_paras"]["battery_capacity"]
            usv_speed = config["env_paras"]["usv_speed"]
            charge_time = config["env_paras"]["charge_time"]

            for i, position in enumerate(usv_positions):
                # 将电量比例转换为实际电量值
                battery_level = usv_battery_levels[i] if i < len(usv_battery_levels) else 1.0
                if isinstance(battery_level, float) and battery_level <= 1.0:
                    battery_level = battery_level * battery_capacity

                usv = {
                    "usv_id": i,
                    "position": position,
                    "battery_capacity": battery_capacity,
                    "battery_level": battery_level,
                    "speed": usv_speed,
                    "charge_time": charge_time,
                    "timeline": [],
                    "current_time": 0.0
                }
                usvs.append(usv)

        # 返回标准格式数据
        return {
            "config": config,
            "tasks": tasks,
            "usvs": usvs,
            "metadata": {
                "case_id": raw_data.get("基本信息", {}).get("案例ID", "Unknown"),
                "original_format": "usv_case"
            }
        }

    @staticmethod
    def adapt_env_backup_format(raw_data: Dict) -> Dict:
        """
        适配env备份格式到标准格式（用于兼容原始算法）

        Args:
            raw_data: 原始env备份数据

        Returns:
            标准格式的环境数据
        """
        # 检查是否已经是标准格式
        if "config" in raw_data and "tasks" in raw_data and "usvs" in raw_data:
            return raw_data

        # 否则进行转换
        return USVDataAdapter.adapt_usv_case_format(raw_data)


def load_and_adapt_data(file_path: str) -> Dict:
    """
    加载并适配数据文件

    Args:
        file_path: 数据文件路径

    Returns:
        标准格式的环境数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 自动检测数据格式并进行适配
    if "基本信息" in raw_data and "环境参数" in raw_data:
        # USV案例格式
        return USVDataAdapter.adapt_usv_case_format(raw_data)
    elif "config" in raw_data or "env_paras" in raw_data:
        # 可能是env备份格式
        return USVDataAdapter.adapt_env_backup_format(raw_data)
    else:
        # 尝试直接作为标准格式处理
        return raw_data