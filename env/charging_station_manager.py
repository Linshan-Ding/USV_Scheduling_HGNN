
import torch
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class ChargingRecord:
    """充电记录数据结构"""
    usv_id: int
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    action: str = "start_charging"  # "start_charging" 或 "finish_charging"


class ChargingStationManager:
    """
    单充电站管理器 - 状态增强方案

    功能：
    - 管理单充电站的充放电状态
    - 支持多USV同时充电（无限充电能力）
    - 维护USV充电时长和可用状态
    """

    def __init__(self, location: Tuple[float, float] = (0.0, 0.0),
                 max_concurrent_usvs: float = float('inf')):
        """
        初始化充电站管理器

        参数：
            location: 充电站坐标 (x, y)
            max_concurrent_usvs: 最大同时充电USV数量，inf表示无限制
        """
        self.location = location
        self.max_concurrent_usvs = max_concurrent_usvs

        # USV充电状态管理
        self.charging_usvs = {}      # {usv_id: charging_start_time}
        self.available_usvs = set()  # 可用USV集合
        self.charging_history = []   # 充电历史记录

    def can_start_charging(self, usv_id: int, current_time: float) -> bool:
        """
        检查USV是否可以开始充电

        参数：
            usv_id: USV标识
            current_time: 当前时间

        返回：
            bool: 是否可以开始充电
        """
        if usv_id in self.charging_usvs:
            return False  # 已在充电

        if len(self.charging_usvs) >= self.max_concurrent_usvs:
            return False  # 充电站已满

        return True

    def start_charging(self, usv_id: int, current_time: float) -> bool:
        """
        开始充电

        参数：
            usv_id: USV标识
            current_time: 充电开始时间

        返回：
            bool: 充电是否成功开始
        """
        if not self.can_start_charging(usv_id, current_time):
            return False

        self.charging_usvs[usv_id] = current_time

        # 添加充电记录
        charging_record = ChargingRecord(
            usv_id=usv_id,
            start_time=current_time,
            action='start_charging'
        )
        self.charging_history.append(charging_record)

        return True

    def finish_charging(self, usv_id: int, current_time: float) -> float:
        """
        完成充电

        参数：
            usv_id: USV标识
            current_time: 充电结束时间

        返回：
            float: 充电时长
        """
        if usv_id not in self.charging_usvs:
            return 0.0

        charging_start_time = self.charging_usvs[usv_id]
        charging_duration = current_time - charging_start_time

        # 移除充电状态
        del self.charging_usvs[usv_id]

        # 添加到可用USV集合
        self.available_usvs.add(usv_id)

        # 添加充电完成记录
        charging_record = ChargingRecord(
            usv_id=usv_id,
            start_time=charging_start_time,
            end_time=current_time,
            duration=charging_duration,
            action='finish_charging'
        )
        self.charging_history.append(charging_record)

        return charging_duration

    def get_charging_duration(self, usv_id: int, current_time: float) -> float:
        """
        获取USV当前充电时长

        参数：
            usv_id: USV标识
            current_time: 当前时间

        返回：
            float: 充电时长，如果未充电则返回0.0
        """
        if usv_id not in self.charging_usvs:
            return 0.0

        return current_time - self.charging_usvs[usv_id]

    def is_usv_available(self, usv_id: int) -> bool:
        """
        检查USV是否可用（未在充电）

        参数：
            usv_id: USV标识

        返回：
            bool: USV是否可用
        """
        return usv_id not in self.charging_usvs

    def get_charging_status(self) -> Dict:
        """
        获取当前充电状态

        返回：
            dict: 包含充电统计信息
        """
        return {
            'charging_count': len(self.charging_usvs),
            'available_count': len(self.available_usvs),
            'charging_usvs': list(self.charging_usvs.keys()),
            'available_usvs': list(self.available_usvs),
            'location': self.location,
            'max_concurrent_usvs': self.max_concurrent_usvs
        }

    def reset(self):
        """重置充电站状态"""
        self.charging_usvs.clear()
        self.available_usvs.clear()
        self.charging_history.clear()

    def get_usv_charging_feature(self, usv_id: int, current_time: float) -> Tuple[float, float]:
        """
        获取USV充电特征（用于USV特征增强）

        参数：
            usv_id: USV标识
            current_time: 当前时间

        返回：
            tuple: (is_charging, charging_duration)
        """
        if usv_id in self.charging_usvs:
            charging_duration = current_time - self.charging_usvs[usv_id]
            return 1.0, charging_duration
        else:
            return 0.0, 0.0

    def get_all_charging_features(self, usv_ids: List[int], current_time: float,
                                 device: torch.device) -> torch.Tensor:
        """
        获取所有USV的充电特征（批量处理）

        参数：
            usv_ids: USV ID列表
            current_time: 当前时间
            device: 计算设备

        返回：
            torch.Tensor: 充电特征矩阵 (num_usvs, 2)
                      - [0]: 是否在充电 (0.0/1.0)
                      - [1]: 充电时长
        """
        num_usvs = len(usv_ids)
        charging_features = torch.zeros(num_usvs, 2, dtype=torch.float32, device=device)

        for i, usv_id in enumerate(usv_ids):
            is_charging, charging_duration = self.get_usv_charging_feature(usv_id, current_time)
            charging_features[i, 0] = is_charging
            charging_features[i, 1] = charging_duration

        return charging_features

    def get_charging_summary(self) -> Dict:
        """
        获取充电统计摘要

        返回：
            dict: 充电统计信息
        """
        if not self.charging_history:
            return {
                'total_charging_sessions': 0,
                'total_charging_time': 0.0,
                'average_charging_time': 0.0,
                'unique_usvs_charged': 0,
                'charging_frequency': {}
            }

        total_sessions = len([r for r in self.charging_history if r.action == 'start_charging'])
        completed_sessions = len([r for r in self.charging_history if r.action == 'finish_charging'])

        total_charging_time = 0.0
        unique_usvs = set()
        charging_frequency = {}

        for record in self.charging_history:
            if record.action == 'finish_charging' and record.duration is not None:
                total_charging_time += record.duration
                unique_usvs.add(record.usv_id)

                if record.usv_id not in charging_frequency:
                    charging_frequency[record.usv_id] = 0
                charging_frequency[record.usv_id] += 1

        average_charging_time = total_charging_time / completed_sessions if completed_sessions > 0 else 0.0

        return {
            'total_charging_sessions': total_sessions,
            'completed_charging_sessions': completed_sessions,
            'total_charging_time': total_charging_time,
            'average_charging_time': average_charging_time,
            'unique_usvs_charged': len(unique_usvs),
            'charging_frequency': charging_frequency,
            'currently_charging': len(self.charging_usvs)
        }