import sys
import gymnasium as gym  # 从gym迁移到gymnasium
from gymnasium.utils import seeding
import torch
import numpy as np
import copy
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Union


@dataclass
class ResetResult:
    """Container for Gymnasium reset results."""

    observation: Any
    info: Dict[str, Any]


@dataclass
class StepBatch:
    """Container for Gymnasium step results with cached batch arrays."""

    observation: Any
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: Dict[str, Any]


def ensure_numpy(array_like: Any, *, dtype: Optional[Any] = None) -> Union[np.ndarray, Any]:
    """Convert input to numpy.ndarray when possible."""
    if isinstance(array_like, np.ndarray):
        arr = array_like
    elif isinstance(array_like, torch.Tensor):
        arr = array_like.detach().cpu().numpy()
    else:
        try:
            arr = np.asarray(array_like)
        except Exception:
            return array_like
    if isinstance(arr, np.ndarray) and dtype is not None:
        if arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
    return arr


def pack_reset(observation: Any, info: Optional[Dict[str, Any]] = None) -> ResetResult:
    """Build a Gymnasium-compliant reset payload."""
    obs_np = ensure_numpy(observation)
    info_dict = {} if info is None else dict(info)
    return ResetResult(observation=obs_np, info=info_dict)


def pack_step(
    observation: Any,
    rewards: Any,
    terminated: Any,
    truncated: Any,
    info: Optional[Dict[str, Any]] = None,
) -> StepBatch:
    """Build a Gymnasium-compliant step payload and cache batch arrays."""
    obs_np = ensure_numpy(observation)
    reward_np = ensure_numpy(rewards, dtype=np.float64)
    terminated_np = ensure_numpy(terminated, dtype=bool)
    truncated_np = ensure_numpy(truncated, dtype=bool)

    info_dict: Dict[str, Any] = {} if info is None else dict(info)
    info_dict.setdefault("batch_rewards", reward_np)
    info_dict.setdefault("batch_terminated", terminated_np)
    info_dict.setdefault("batch_truncated", truncated_np)

    return StepBatch(
        observation=obs_np,
        rewards=reward_np,
        terminated=terminated_np,
        truncated=truncated_np,
        info=info_dict,
    )


# 导入新的约束验证器和充电站管理器
try:
    from .usv_constraint_validator import USVConstraintValidator
    from .charging_station_manager import ChargingStationManager
except ImportError:
    # 如果作为独立脚本运行，使用绝对导入
    from usv_constraint_validator import USVConstraintValidator
    from charging_station_manager import ChargingStationManager


@dataclass
class USVState:
    """
    USV环境状态类
    """
    # 静态特征（移除前驱关系，专注空间关系）
    task_usv_adj_batch: torch.Tensor = None  # 任务-USV邻接矩阵，表示哪些USV可以执行哪些任务
    task_task_adj_batch: torch.Tensor = None  # 任务-任务空间邻接矩阵（基于空间距离）
    task_types_batch: torch.Tensor = None    # 任务类型矩阵 (Type1/2/3)
    end_task_biases_batch: torch.Tensor = None  # 每个USV的最后一个任务的偏移量
    nums_tasks_batch: torch.Tensor = None    # 每个batch的任务数量

    # 动态特征
    batch_idxes: torch.Tensor = None
    feat_tasks_batch: torch.Tensor = None    # 任务特征矩阵
    feat_usvs_batch: torch.Tensor = None     # USV特征矩阵
    proc_times_batch: torch.Tensor = None    # 任务执行时间矩阵
    task_usv_adj_dynamic_batch: torch.Tensor = None  # 动态任务-USV邻接矩阵
    time_batch: torch.Tensor = None

    mask_task_procing_batch: torch.Tensor = None  # 任务处理中掩码
    mask_task_finish_batch: torch.Tensor = None   # 任务完成掩码
    mask_usv_procing_batch: torch.Tensor = None   # USV处理中掩码
    task_step_batch: torch.Tensor = None          # 任务步骤计数器

    # 充电站管理器引用
    charging_manager = None  # 充电站管理器实例

    def update(self, batch_idxes, feat_tasks_batch, feat_usvs_batch, proc_times_batch,
               task_usv_adj_dynamic_batch, mask_task_procing_batch, mask_task_finish_batch,
               mask_usv_procing_batch, task_step_batch, time, charging_manager=None):
        """更新状态信息"""
        self.batch_idxes = batch_idxes
        self.feat_tasks_batch = feat_tasks_batch
        self.feat_usvs_batch = feat_usvs_batch
        self.proc_times_batch = proc_times_batch
        self.task_usv_adj_dynamic_batch = task_usv_adj_dynamic_batch
        self.mask_task_procing_batch = mask_task_procing_batch
        self.mask_task_finish_batch = mask_task_finish_batch
        self.mask_usv_procing_batch = mask_usv_procing_batch
        self.task_step_batch = task_step_batch
        self.time_batch = time
        self.charging_manager = charging_manager


class USVEvent:
    """
    USV事件类,用于事件驱动机制
    """
    def __init__(self, event_type: str, usv_id: int, task_id: Optional[int] = None,
                 timestamp: float = 0.0, location: Optional[Tuple[float, float]] = None):
        """
        初始化事件
        :param event_type: 事件类型 ('arrive', 'complete', 'charge', 'return')
        :param usv_id: USV ID
        :param task_id: 任务ID (对于充电和返回事件可能为None)
        :param timestamp: 事件发生时间
        :param location: 事件发生位置
        """
        self.event_type = event_type
        self.usv_id = usv_id
        self.task_id = task_id
        self.timestamp = timestamp
        self.location = location

    def __lt__(self, other):
        """用于事件队列排序，按时间戳从小到大排序"""
        return self.timestamp < other.timestamp


class USVEnv(gym.Env):
    """
    USV任务调度环境
    """
    def __init__(self, case, env_paras, data_source='case'):
        """
        初始化USV环境
        :param case: 实例生成器或实例文件地址
        :param env_paras: 环境参数字典
        :param data_source: 实例来源 ('case' 或 'file')
        """
        super().__init__()

        # 初始化随机数生成器（符合 Gymnasium seeding 约定）
        self.np_random, self._initial_seed = seeding.np_random(None)
        self._last_reset_seed = self._initial_seed
        self._last_reset_options: Dict[str, Any] = {}

        # 加载环境参数
        self.show_mode = env_paras["show_mode"]  # 结果显示模式
        self.batch_size = env_paras["batch_size"]  # 并行实例数量
        self.num_usvs = env_paras["num_usvs"]  # USV数量
        self.num_tasks = env_paras["num_tasks"]  # 任务数量
        self.paras = env_paras  # 完整参数
        self.device = env_paras["device"]  # 计算设备

        # USV特定参数
        self.map_size = env_paras.get("map_size", (800, 800))  # 地图尺寸
        self.battery_capacity = env_paras.get("battery_capacity", 1200)  # 电池容量
        self.usv_speed = env_paras.get("usv_speed", 5)  # USV航速
        self.charge_time = env_paras.get("charge_time", 10)  # 充电时间
        self.energy_cost_per_distance = env_paras.get("energy_cost_per_distance", 1.0)  # 单位距离能耗
        self.task_time_energy_ratio = env_paras.get("task_time_energy_ratio", 0.25)  # 任务执行时间能耗比

        # 起始点位置
        self.start_point = (0.0, 0.0)

        # 预验证开关（默认开启以获得训练效率提升）
        self.enable_pre_validation = env_paras.get("enable_pre_validation", True)

        # 任务执行时间类型（三角模糊数）
        self.task_service_time_fuzzy = {
            "Type1": (10.0, 20.0, 30.0),  # 期望值: 20.0
            "Type2": (30.0, 50.0, 80.0),  # 期望值: 52.5
            "Type3": (15.0, 25.0, 40.0)   # 期望值: 26.25
        }

        # 加载实例数据
        self._load_instance_data(case, data_source)

        # 初始化状态特征
        self._initialize_features()

        # 初始化事件队列
        self.event_queues = [[] for _ in range(self.batch_size)]

        # 初始化USV状态
        self._initialize_usv_states()

        # 创建约束验证器实例（方案3A：基础版本）
        self.constraint_validator = USVConstraintValidator(
            num_usvs=self.num_usvs,
            num_tasks=self.num_tasks,
            start_point=self.start_point
        )

        # 创建充电站管理器实例
        self.charging_manager = ChargingStationManager(
            location=self.start_point,
            max_concurrent_usvs=float('inf')  # 无限充电能力
        )

        # 创建初始状态对象 - 修复：确保charging_manager被正确设置
        self.state = USVState(
            batch_idxes=torch.arange(self.batch_size),
            feat_tasks_batch=self.feat_tasks_batch,
            feat_usvs_batch=self.feat_usvs_batch,
            proc_times_batch=self.proc_times_batch,
            task_usv_adj_dynamic_batch=self.task_usv_adj_dynamic_batch,
            task_usv_adj_batch=self.task_usv_adj_batch,
            task_types_batch=self.task_types_batch,
            end_task_biases_batch=self.end_task_biases_batch,
            nums_tasks_batch=self.nums_tasks_batch,
            mask_task_procing_batch=self.mask_task_procing_batch,
            mask_task_finish_batch=self.mask_task_finish_batch,
            mask_usv_procing_batch=self.mask_usv_procing_batch,
            task_step_batch=self.task_step_batch,
            time_batch=self.time_batch
        )

        # 修复：单独设置charging_manager，因为dataclass不支持通过构造函数设置
        self.state.charging_manager = self.charging_manager

        # 保存初始数据用于重置
        self._save_initial_state()

        # 定义动作空间和观察空间
        self.action_space = gym.spaces.MultiDiscrete([self.num_tasks, self.num_usvs])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.batch_size, self._get_observation_dim()),
            dtype=np.float32
        )

    def _load_instance_data(self, case, data_source):
        """
        简化版实例数据加载，移除前驱关系处理

        :param case: 案例数据（USVCaseData或字典格式）
        :param data_source: 数据来源标识
        """
        # 导入数据加载器
        try:
            from .usv_load_data import load_usv_data
        except ImportError:
            from usv_load_data import load_usv_data

        if data_source == 'case':
            # 使用数据加载器加载案例数据（B.1阶段已修改为生成空间邻接矩阵）
            loaded_data = load_usv_data(case, device=self.device)

            # 从加载的数据中提取环境所需的张量，确保批次大小匹配
            self.proc_times_batch = loaded_data.matrix_proc_time.unsqueeze(0).expand(self.batch_size, -1, -1)

            # 使用新的空间邻接矩阵
            self.task_usv_adj_batch = loaded_data.matrix_ope_ma_adj.unsqueeze(0).expand(self.batch_size, -1, -1)
            self.task_usv_adj_dynamic_batch = copy.deepcopy(self.task_usv_adj_batch)

            # 使用空间邻接矩阵作为任务-任务关系（移除前驱关系概念）
            self.task_task_adj_batch = loaded_data.matrix_pre_proc.unsqueeze(0).expand(self.batch_size, -1, -1)

            # 存储特征数据（将在_initialize_features中使用），确保批次大小匹配
            self._loaded_task_features = loaded_data.task_features.unsqueeze(0).expand(self.batch_size, -1, -1)
            self._loaded_usv_features = loaded_data.usv_features.unsqueeze(0).expand(self.batch_size, -1, -1)

            # 存储其他需要的数据
            self.task_positions = loaded_data.task_positions
            self.usv_positions = loaded_data.usv_positions
            self.task_types = loaded_data.task_types
            self.environment_parameters = loaded_data.environment_parameters

            # 存储任务结构数据（兼容load_data.py格式），确保批次大小匹配
            self.ope_appertain_batch = loaded_data.ope_appertain.unsqueeze(0).expand(self.batch_size, -1)
            self.num_ope_biases_batch = loaded_data.num_ope_biases.unsqueeze(0).expand(self.batch_size, -1)
            self.nums_ope_batch = loaded_data.nums_ope.unsqueeze(0).expand(self.batch_size, -1)
            self.matrix_cal_cumul_batch = loaded_data.matrix_cal_cumul.unsqueeze(0).expand(self.batch_size, -1, -1)

            # 移除前驱关系：不再使用task_pre_adj_batch
            # 现在所有约束都通过空间邻接矩阵和USV状态来管理

    def _initialize_features(self):
        """初始化特征矩阵"""
                # 特征维度定义（4维x4维方案）
        self.task_feat_dim = 4  # 任务特征维度
        self.usv_feat_dim = 4   # USV特征维度

        # 检查是否有从数据加载器加载的特征数据
        if hasattr(self, '_loaded_task_features') and hasattr(self, '_loaded_usv_features'):
            # 使用数据加载器提供的特征数据
            self.feat_tasks_batch = self._loaded_task_features.clone()
            self.feat_usvs_batch = self._loaded_usv_features.clone()

            # 确保批次大小正确
            if self.feat_tasks_batch.shape[0] != self.batch_size:
                self.feat_tasks_batch = self.feat_tasks_batch.expand(self.batch_size, -1, -1)
            if self.feat_usvs_batch.shape[0] != self.batch_size:
                self.feat_usvs_batch = self.feat_usvs_batch.expand(self.batch_size, -1, -1)

        else:
            # 兼容原有逻辑：初始化默认特征矩阵
            # 初始化任务特征矩阵 (batch_size, 4, num_tasks)
            # 维度说明（4维方案）：
            # [0]: 任务状态 (0=未分配, 1=已分配, 2=已完成)
            # [1]: 执行时间期望值
            # [2]: 任务x坐标
            # [3]: 任务y坐标
            # 说明：移除了任务类型（冗余）和距离起点航行时间（可动态计算）
            self.feat_tasks_batch = torch.zeros(
                size=(self.batch_size, self.task_feat_dim, self.num_tasks),
                dtype=torch.float32, device=self.device
            )

            # 初始化USV特征矩阵 (batch_size, 4, num_usvs)
            # 维度说明（4维方案）：
            # [0]: 可用状态 (0=忙碌/充电, 1=空闲)
            # [1]: 当前电量 (归一化到0-1)
            # [2]: USV x坐标
            # [3]: USV y坐标
            # 说明：移除了利用率维度，充电状态通过可用状态和电量组合体现
            self.feat_usvs_batch = torch.zeros(
                size=(self.batch_size, self.usv_feat_dim, self.num_usvs),
                dtype=torch.float32, device=self.device
            )

        # 初始化任务执行时间矩阵（如果没有通过数据加载器设置）
        if not hasattr(self, 'proc_times_batch') or self.proc_times_batch.numel() == 0:
            self.proc_times_batch = torch.zeros(
                size=(self.batch_size, self.num_tasks, self.num_usvs),
                dtype=torch.float32, device=self.device
            )

        # 初始化任务-USV邻接矩阵（如果没有通过数据加载器设置）
        if not hasattr(self, 'task_usv_adj_batch') or self.task_usv_adj_batch.numel() == 0:
            # 1表示该USV可以执行该任务，初始时所有USV都可以执行所有任务
            self.task_usv_adj_batch = torch.ones(
                size=(self.batch_size, self.num_tasks, self.num_usvs),
                dtype=torch.long, device=self.device
            )
        self.task_usv_adj_dynamic_batch = copy.deepcopy(self.task_usv_adj_batch)

        # 移除前驱关系矩阵：使用空间邻接矩阵代替

        # 基于实例数据填充特征（兼容原有逻辑）
        self._populate_features_from_instances()

    def _populate_features_from_instances(self):
        """基于实例数据填充特征矩阵"""
        # 检查是否已经从数据加载器加载了数据
        if hasattr(self, '_loaded_task_features') and hasattr(self, '_loaded_usv_features'):
            # 数据已经从数据加载器加载，无需再次填充
            # 只需要确保任务类型矩阵正确设置
            self.task_types_batch = torch.zeros(
                size=(self.batch_size, self.num_tasks),
                dtype=torch.long, device=self.device
            )

            for batch_idx in range(self.batch_size):
                for task_id in range(self.num_tasks):
                    self.task_types_batch[batch_idx, task_id] = int(
                        self.feat_tasks_batch[batch_idx, 2, task_id]
                    )
        else:
            # 使用原有的随机生成逻辑作为后备方案
            for batch_idx in range(self.batch_size):
                # 填充任务特征
                for task_id in range(self.num_tasks):
                    # 随机生成任务位置 (在800x800地图内)
                    task_x = random.uniform(0, self.map_size[0])
                    task_y = random.uniform(0, self.map_size[1])

                    # 随机分配任务类型（用于计算执行时间）
                    task_type = random.choice([1, 2, 3])  # Type1/2/3

                    # 计算执行时间期望值
                    fuzzy_times = self.task_service_time_fuzzy[f"Type{task_type}"]
                    exec_time = calculate_fuzzy_expectation(*fuzzy_times)

                    # 填充任务特征（4维方案：去掉类型和航行时间）
                    self.feat_tasks_batch[batch_idx, 0, task_id] = 0  # 初始状态：未分配
                    self.feat_tasks_batch[batch_idx, 1, task_id] = exec_time
                    self.feat_tasks_batch[batch_idx, 2, task_id] = task_x
                    self.feat_tasks_batch[batch_idx, 3, task_id] = task_y

                    # 所有USV执行该任务的时间相同
                    for usv_id in range(self.num_usvs):
                        self.proc_times_batch[batch_idx, task_id, usv_id] = exec_time

                # 填充USV特征（4维方案：去掉利用率维度）
                for usv_id in range(self.num_usvs):
                    # 所有USV从起点开始，满电状态
                    self.feat_usvs_batch[batch_idx, 0, usv_id] = 1  # 可用状态：空闲
                    self.feat_usvs_batch[batch_idx, 1, usv_id] = 1  # 电量：满电
                    self.feat_usvs_batch[batch_idx, 2, usv_id] = self.start_point[0]  # x坐标
                    self.feat_usvs_batch[batch_idx, 3, usv_id] = self.start_point[1]  # y坐标

            # 设置任务类型矩阵（需要从临时变量获取，因为4维方案中移除了类型维度）
            self.task_types_batch = torch.zeros(
                size=(self.batch_size, self.num_tasks),
                dtype=torch.long, device=self.device
            )
            # 重新生成任务类型信息用于兼容性
            for batch_idx in range(self.batch_size):
                for task_id in range(self.num_tasks):
                    # 重新随机分配类型用于兼容性（实际使用时可以基于执行时间推断）
                    task_type = random.choice([1, 2, 3])
                    self.task_types_batch[batch_idx, task_id] = task_type

            # 计算任务-任务空间邻接矩阵（基于η-近邻构图，不包含自身）
            self._compute_task_task_spatial_adjacency()

    def _compute_task_task_spatial_adjacency(self):
        """计算基于位置的任务-任务空间邻接矩阵（η-近邻，不包含自身）"""
        # 默认η=3，如果没有设置则使用默认值
        if not hasattr(self, 'eta') or self.eta <= 0:
            self.eta = 3  # 默认η=3近邻

        self.task_task_adj_batch = torch.zeros(
            size=(self.batch_size, self.num_tasks, self.num_tasks),
            dtype=torch.float32, device=self.device
        )

        for batch_idx in range(self.batch_size):
            # 提取任务位置坐标 (num_tasks, 2)
            task_positions = torch.stack([
                self.feat_tasks_batch[batch_idx, 2, :],  # x坐标
                self.feat_tasks_batch[batch_idx, 3, :]   # y坐标
            ], dim=1)  # (num_tasks, 2)

            # 计算任务间欧氏距离矩阵 (num_tasks, num_tasks)
            diff = task_positions.unsqueeze(1) - task_positions.unsqueeze(0)  # (num_tasks, num_tasks, 2)
            distances = torch.norm(diff, dim=2)  # (num_tasks, num_tasks)

            # 向量化选择η个最近邻居（不包含自身）
            k = min(self.eta, self.num_tasks - 1)
            if k > 0:
                # 排除自身：将对角线设为无穷大
                distances.fill_diagonal_(float('inf'))

                # 对每行选择最近的k个邻居
                _, neighbor_indices = torch.topk(distances, k=k, largest=False, dim=1)  # (num_tasks, k)

                # 构建双向连接的邻接矩阵
                for i in range(self.num_tasks):
                    for j in neighbor_indices[i]:
                        self.task_task_adj_batch[batch_idx, i, j] = 1.0
                        self.task_task_adj_batch[batch_idx, j, i] = 1.0

    def _initialize_legacy_data_loading(self, case, data_source):
        """
        兼容性方法：处理其他数据源的加载
        保留原有的随机生成逻辑作为后备方案

        :param case: 案例数据
        :param data_source: 数据来源标识
        """
        # 对于其他数据源，暂时使用原有的随机生成逻辑
        # 这确保了向后兼容性
        pass

    def _initialize_usv_states(self):
        """初始化USV状态和掩码"""
        # 初始化掩码矩阵
        # 任务处理中掩码 (batch_size, num_tasks)
        self.mask_task_procing_batch = torch.full(
            size=(self.batch_size, self.num_tasks),
            dtype=torch.bool, fill_value=False, device=self.device
        )

        # 任务完成掩码 (batch_size, num_tasks)
        self.mask_task_finish_batch = torch.full(
            size=(self.batch_size, self.num_tasks),
            dtype=torch.bool, fill_value=False, device=self.device
        )

        # USV处理中掩码 (batch_size, num_usvs)
        self.mask_usv_procing_batch = torch.full(
            size=(self.batch_size, self.num_usvs),
            dtype=torch.bool, fill_value=False, device=self.device
        )

        # 初始化任务步骤计数器 (用于前驱任务约束)
        # 这里简化处理，每个任务都是独立的，没有前驱关系
        # 在实际应用中可以根据需要设置任务的前驱关系
        self.task_step_batch = torch.zeros(
            size=(self.batch_size, self.num_tasks),
            dtype=torch.long, device=self.device
        )

        # 初始化时间向量
        self.time_batch = torch.zeros(self.batch_size, device=self.device)

        # 移除前驱关系矩阵：使用空间邻接矩阵代替（已在_load_instance_data中设置task_task_adj_batch）

        # 任务类型矩阵 (batch_size, num_tasks)
        self.task_types_batch = torch.zeros(
            size=(self.batch_size, self.num_tasks),
            dtype=torch.long, device=self.device
        )
        for batch_idx in range(self.batch_size):
            for task_id in range(self.num_tasks):
                self.task_types_batch[batch_idx, task_id] = int(
                    self.feat_tasks_batch[batch_idx, 2, task_id]
                )

        # 每个USV的最后一个任务偏移量 (简化处理)
        self.end_task_biases_batch = torch.zeros(
            size=(self.batch_size, self.num_usvs),
            dtype=torch.long, device=self.device
        )
        for batch_idx in range(self.batch_size):
            # 简化处理：平均分配任务给USV
            tasks_per_usv = self.num_tasks // self.num_usvs
            for usv_id in range(self.num_usvs):
                if usv_id == self.num_usvs - 1:  # 最后一个USV处理剩余任务
                    self.end_task_biases_batch[batch_idx, usv_id] = self.num_tasks - 1
                else:
                    self.end_task_biases_batch[batch_idx, usv_id] = (usv_id + 1) * tasks_per_usv - 1

        # 每个batch的任务数量
        self.nums_tasks_batch = torch.full(
            size=(self.batch_size,), fill_value=self.num_tasks,
            dtype=torch.long, device=self.device
        )

        # 初始化调度信息
        # 任务调度状态 (batch_size, num_tasks, 4)
        # [0]: 状态 (0=未调度, 1=已调度)
        # [1]: 分配的USV ID
        # [2]: 开始时间
        # [3]: 完成时间
        self.schedules_batch = torch.zeros(
            size=(self.batch_size, self.num_tasks, 4),
            dtype=torch.float32, device=self.device
        )

        # USV调度状态 (batch_size, num_usvs, 4)
        # [0]: 空闲状态 (0=忙碌, 1=空闲)
        # [1]: 可用时间
        # [2]: 累计工作时间
        # [3]: 当前执行的任务ID
        self.usvs_batch = torch.zeros(
            size=(self.batch_size, self.num_usvs, 4),
            dtype=torch.float32, device=self.device
        )
        self.usvs_batch[:, :, 0] = 1  # 初始所有USV都空闲

        # 初始化makespan
        self.makespan_batch = torch.zeros(self.batch_size, device=self.device)
        self.done_batch = torch.full(
            size=(self.batch_size,), dtype=torch.bool, fill_value=False, device=self.device
        )

    def _save_initial_state(self):
        """保存初始状态用于重置"""
        # 方案C：混合优化 - 保留最重要的state对象，只保存关键补充数据
        self.initial_state = copy.deepcopy(self.state)
        self.initial_state_complement = {
            # 调度状态（state对象中不包含的重要数据）
            'schedules_batch': copy.deepcopy(self.schedules_batch),
            'usvs_batch': copy.deepcopy(self.usvs_batch),
        }

    def _calculate_navigation_energy(self, distance):
        """
        计算航行电量消耗

        :param distance: 航行距离
        :return: navigation_energy 航行电量消耗
        """
        return distance * self.energy_cost_per_distance

    def _calculate_task_energy(self, execution_time):
        """
        计算任务执行电量消耗

        :param execution_time: 任务执行时间
        :return: task_energy 任务执行电量消耗
        """
        return execution_time * self.task_time_energy_ratio

    def _get_observation_dim(self):
        """计算观察空间维度（4维x4维方案）"""
        # 观察空间包括：
        # - 任务特征：task_feat_dim * num_tasks (4 * num_tasks)
        # - USV特征：usv_feat_dim * num_usvs (4 * num_usvs)
        # - 静态邻接矩阵：num_tasks * num_usvs
        # - 掩码信息：num_tasks + num_usvs
        # - 时间信息：1

        task_feat_total = self.task_feat_dim * self.num_tasks    # 4 * num_tasks
        usv_feat_total = self.usv_feat_dim * self.num_usvs     # 4 * num_usvs
        adj_matrix_total = self.num_tasks * self.num_usvs
        mask_total = self.num_tasks + self.num_usvs
        time_info = 1

        total_dim = task_feat_total + usv_feat_total + adj_matrix_total + mask_total + time_info
        return total_dim

    def _state_to_numpy(self, state):
        """
        将USVState对象转换为numpy数组观察格式（gymnasium兼容）
        :param state: USVState对象
        :return: numpy.ndarray 观察数组，shape=(batch_size, observation_dim)
        """
        obs_list = []

        for batch_idx in range(self.batch_size):
            # 1. 任务特征：展平任务特征矩阵 (4 * num_tasks)
            task_features = state.feat_tasks_batch[batch_idx].flatten()  # (task_feat_dim * num_tasks,)

            # 2. USV特征：展平USV特征矩阵 (4 * num_usvs)
            usv_features = state.feat_usvs_batch[batch_idx].flatten()    # (usv_feat_dim * num_usvs,)

            # 3. 静态邻接矩阵：展平任务-USV邻接矩阵 (num_tasks * num_usvs)
            adj_matrix = state.task_usv_adj_batch[batch_idx].flatten()   # (num_tasks * num_usvs,)

            # 4. 掩码信息：任务处理掩码、任务完成掩码、USV处理掩码 (num_tasks + num_usvs)
            task_proc_mask = state.mask_task_procing_batch[batch_idx].float()  # 转换为float
            task_finish_mask = state.mask_task_finish_batch[batch_idx].float()
            usv_proc_mask = state.mask_usv_procing_batch[batch_idx].float()
            masks = torch.cat([task_proc_mask, task_finish_mask, usv_proc_mask])  # (num_tasks * 2 + num_usvs,)

            # 修正：为了保持一致性，只使用任务完成掩码和USV处理掩码
            masks = torch.cat([task_finish_mask, usv_proc_mask])  # (num_tasks + num_usvs,)

            # 5. 时间信息：当前时间
            time_info = state.time_batch[batch_idx:batch_idx+1].float()  # (1,)

            # 拼接所有特征
            single_obs = torch.cat([
                task_features,    # (task_feat_dim * num_tasks,)
                usv_features,     # (usv_feat_dim * num_usvs,)
                adj_matrix,       # (num_tasks * num_usvs,)
                masks,            # (num_tasks + num_usvs,)
                time_info         # (1,)
            ])  # (total_dim,)

            obs_list.append(single_obs)

        # 堆叠为批次观察
        observation = torch.stack(obs_list)  # (batch_size, total_dim)

        # 转换为numpy数组（gymnasium标准）
        return observation.cpu().numpy().astype(np.float32)

    def step(self, action):
        """
        环境步进函数（gymnasium兼容）
        :param action: 动作 (task_id, usv_id) 对，shape: (2, batch_size) 或类似的numpy数组
        :return: (observation, reward, terminated, truncated, info)
        """
        # 确保action是torch.Tensor（兼容numpy数组输入）
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device)

        # 确保action形状正确
        if action.dim() == 1:
            action = action.unsqueeze(1)  # 如果是1D，则添加维度
        elif action.dim() == 2 and action.shape[0] > action.shape[1]:
            action = action.T  # 如果形状是(batch_size, 2)，转置为(2, batch_size)

        task_ids = action[0, :]  # 任务ID
        usv_ids = action[1, :]   # USV ID

        # 保存旧的makespan用于计算奖励
        old_makespan_batch = copy.deepcopy(self.makespan_batch)

        # 初始化奖励
        total_reward = torch.zeros(self.batch_size, device=self.device)

        # 处理每个batch的动作
        for batch_idx in self.batch_idxes:
            task_id = task_ids[batch_idx]
            usv_id = usv_ids[batch_idx]

            # 检查任务和USV是否可用
            if not self._is_action_valid(batch_idx, task_id, usv_id):
                # 重新采样有效动作
                new_action = self._robust_resample_action(batch_idx)
                if new_action and new_action != (None, None):
                    task_id, usv_id = new_action
                    # 更新原始动作数组
                    action[0, batch_idx] = task_id
                    action[1, batch_idx] = usv_id
                else:
                    continue  # 重新采样失败，跳过

            # 预验证层（电池约束检查）
            if self.enable_pre_validation:
                if not self._check_battery_constraint(batch_idx, usv_id, task_id):
                    # 电量不足 → 返航充电 + 重新采样新动作
                    charging_penalty = self._handle_charging_decision(batch_idx, usv_id)
                    total_reward[batch_idx] += charging_penalty

                    # 重新采样有效动作
                    new_action = self._robust_resample_action(batch_idx)
                    if new_action and new_action != (None, None):
                        task_id, usv_id = new_action
                        # 更新动作数组
                        action[0, batch_idx] = task_id
                        action[1, batch_idx] = usv_id
                    else:
                        continue  # 找不到有效动作，跳过

            # 执行任务（只有通过所有检查才执行）
            task_reward = self._execute_task(batch_idx, task_id, usv_id)
            total_reward[batch_idx] += task_reward

            # 处理事件队列
            self._process_event_queue(batch_idx)

        # 更新环境状态
        self._update_environment_state()

        # 更新状态对象
        self.state.update(
            self.batch_idxes, self.feat_tasks_batch, self.feat_usvs_batch,
            self.proc_times_batch, self.task_usv_adj_dynamic_batch,
            self.mask_task_procing_batch, self.mask_task_finish_batch,
            self.mask_usv_procing_batch, self.task_step_batch, self.time_batch,
            self.charging_manager  # 传递充电站管理器引用
        )

        # 检查完成状态
        self._check_completion()

        # gymnasium新标准：将done分解为terminated和truncated
        terminated_tensor = self.done_batch.clone()
        truncated_tensor = torch.zeros_like(self.done_batch, dtype=torch.bool, device=self.done_batch.device)

        step_result = pack_step(
            self._state_to_numpy(self.state),
            total_reward,
            terminated_tensor,
            truncated_tensor,
            {
                "time_batch": ensure_numpy(self.time_batch, dtype=np.float32),
                "actions": ensure_numpy(action.long(), dtype=np.int64),
            },
        )

        return (
            step_result.observation,
            step_result.rewards,
            step_result.terminated,
            step_result.truncated,
            step_result.info,
        )

    def _robust_resample_action(self, batch_idx, max_attempts=3):
        """鲁棒的重新采样策略

        Args:
            batch_idx: 批次索引
            max_attempts: 最大尝试次数

        Returns:
            tuple: (task_id, usv_id) 或 (None, None)
        """
        import random

        for attempt in range(max_attempts):
            # 随机遍历所有可能的动作组合
            task_ids = list(range(self.num_tasks))
            usv_ids = list(range(self.num_usvs))
            random.shuffle(task_ids)
            random.shuffle(usv_ids)

            for task_id in task_ids:
                for usv_id in usv_ids:
                    # 双重检查：基础有效性 + 电池约束
                    if (self._is_action_valid(batch_idx, task_id, usv_id) and
                        self._check_battery_constraint(batch_idx, usv_id, task_id)):
                        return task_id, usv_id

        # 重新采样失败，返回None（不给负奖励）
        return None, None

    def _is_action_valid(self, batch_idx, task_id, usv_id):
        """检查动作是否有效（支持充电站）"""
        # 修复：添加索引边界检查，防止越界
        if batch_idx < 0 or batch_idx >= self.batch_size:
            return False
        if task_id < 0 or task_id > self.num_tasks:  # 注意：允许等于num_tasks（充电站）
            return False
        if usv_id < 0 or usv_id >= self.num_usvs:
            return False

        # 检查是否为充电站动作
        if task_id == self.num_tasks:  # 充电站索引为num_tasks
            # 检查USV是否空闲
            usv_status = self.feat_usvs_batch[batch_idx, 0, usv_id]  # 索引0：可用状态 (1=空闲, 0=忙碌)
            if not usv_status:  # 0表示忙碌
                return False

            # 检查USV是否已经在充电中
            if hasattr(self, 'charging_manager') and self.charging_manager:
                # 修复：usv_id参数就是真正的USV全局ID，不需要从特征中获取
                usv_id_global = usv_id

                # 调试信息
                if False:  # 设置为True启用调试
                    print(f"[DEBUG] 充电站检查: USV全局ID={usv_id_global}, 状态={usv_status}")
                    charging_status = self.charging_manager.get_charging_status()
                    print(f"[DEBUG] 当前充电状态: {charging_status}")

                if not self.charging_manager.is_usv_available(usv_id_global):
                    return False

            return True

        # 普通任务动作验证
        # 检查任务是否已经完成
        if self.mask_task_finish_batch[batch_idx, task_id]:
            return False

        # 检查任务是否正在处理
        if self.mask_task_procing_batch[batch_idx, task_id]:
            return False

        # 检查USV是否空闲
        if not self.feat_usvs_batch[batch_idx, 0, usv_id]:  # 0表示忙碌
            return False

        # 检查USV是否可以执行该任务
        if self.task_usv_adj_dynamic_batch[batch_idx, task_id, usv_id] == 0:
            return False

        return True

    def _execute_task(self, batch_idx, task_id, usv_id):
        """执行任务或充电并返回奖励（支持充电站）"""
        # 检查是否为充电站动作
        if task_id == self.num_tasks:  # 充电站索引为num_tasks
            return self._execute_charging(batch_idx, usv_id)

        # 普通任务执行
        # 计算航行时间和执行时间
        usv_pos = (
            self.feat_usvs_batch[batch_idx, 2, usv_id].item(),
            self.feat_usvs_batch[batch_idx, 3, usv_id].item()
        )
        task_pos = (
            self.feat_tasks_batch[batch_idx, 2, task_id].item(),  # x坐标
            self.feat_tasks_batch[batch_idx, 3, task_id].item()   # y坐标
        )

        # 航行到任务点
        travel_distance = calculate_distance(usv_pos, task_pos)
        travel_time = calculate_navigation_time(travel_distance, self.usv_speed)

        # 任务执行时间
        exec_time = self.proc_times_batch[batch_idx, task_id, usv_id].item()

        # 更新任务状态
        self.feat_tasks_batch[batch_idx, 0, task_id] = 1  # 标记为已分配

        # 更新USV状态（开始航行）
        self.feat_usvs_batch[batch_idx, 0, usv_id] = 0  # 标记为忙碌（航行中）

        # 创建到达任务点事件
        arrive_time = self.time_batch[batch_idx] + travel_time
        arrive_event = USVEvent(
            event_type='arrive',
            usv_id=usv_id,
            task_id=task_id,
            timestamp=arrive_time,
            location=task_pos
        )
        self._add_event_to_queue(batch_idx, arrive_event)

        # 创建任务完成事件
        completion_time = arrive_time + exec_time
        completion_event = USVEvent(
            event_type='complete',
            usv_id=usv_id,
            task_id=task_id,
            timestamp=completion_time,
            location=task_pos
        )
        self._add_event_to_queue(batch_idx, completion_event)

        # 更新调度信息
        self.schedules_batch[batch_idx, task_id, 0] = 1  # 状态：已调度
        self.schedules_batch[batch_idx, task_id, 1] = usv_id  # 分配的USV
        self.schedules_batch[batch_idx, task_id, 2] = self.time_batch[batch_idx]  # 开始时间
        self.schedules_batch[batch_idx, task_id, 3] = completion_time  # 完成时间

        self.usvs_batch[batch_idx, usv_id, 0] = 0  # 状态：忙碌
        self.usvs_batch[batch_idx, usv_id, 1] = completion_time  # 可用时间
        self.usvs_batch[batch_idx, usv_id, 2] += travel_time + exec_time  # 累计工作时间
        self.usvs_batch[batch_idx, usv_id, 3] = task_id  # 当前任务

        # 计算奖励（基于makespan改进）
        # 预估新的makespan
        estimated_completion_time = completion_time
        return_home_time = calculate_navigation_time(
            calculate_distance(task_pos, self.start_point), self.usv_speed
        )
        estimated_makespan = estimated_completion_time + return_home_time

        current_makespan = self.makespan_batch[batch_idx]
        reward = current_makespan - min(current_makespan, estimated_makespan)

        return reward

    def _execute_charging(self, batch_idx, usv_id):
        """执行充电动作并返回奖励"""
        # 获取USV当前位置
        usv_pos = (
            self.feat_usvs_batch[batch_idx, 2, usv_id].item(),
            self.feat_usvs_batch[batch_idx, 3, usv_id].item()
        )

        # 充电站位置（起点）
        charging_pos = self.start_point

        # 航行到充电站
        travel_distance = calculate_distance(usv_pos, charging_pos)
        travel_time = calculate_navigation_time(travel_distance, self.usv_speed)

        # 充电时间（固定值）
        charging_time = self.charge_time

        # 总充电完成时间
        completion_time = self.time_batch[batch_idx] + travel_time + charging_time

        # 更新USV状态（开始前往充电站）
        self.feat_usvs_batch[batch_idx, 0, usv_id] = 0  # 标记为忙碌

        # 如果有充电站管理器，更新充电状态
        if hasattr(self, 'charging_manager') and self.charging_manager:
            usv_id_global = int(self.feat_usvs_batch[batch_idx, 2, usv_id])  # 获取USV全局ID
            charging_start_time = self.time_batch[batch_idx] + travel_time
            self.charging_manager.start_charging(usv_id_global, charging_start_time)

            # 创建充电完成事件
            charging_complete_event = USVEvent(
                event_type='charge',  # 使用充电事件类型
                usv_id=usv_id,
                task_id=None,  # 充电事件没有任务ID
                timestamp=completion_time,
                location=charging_pos
            )
            self._add_event_to_queue(batch_idx, charging_complete_event)

        # 更新USV调度信息
        self.usvs_batch[batch_idx, usv_id, 0] = 0  # 状态：忙碌（充电中）
        self.usvs_batch[batch_idx, usv_id, 1] = completion_time  # 可用时间
        self.usvs_batch[batch_idx, usv_id, 2] += travel_time + charging_time  # 累计工作时间
        self.usvs_batch[batch_idx, usv_id, 3] = -1  # 特殊标记：充电中（-1表示充电）

        # 充电不提供额外奖励，保持对makespan优化的专注
        return 0.0

    def _add_event_to_queue(self, batch_idx, event):
        """添加事件到队列"""
        self.event_queues[batch_idx].append(event)
        self.event_queues[batch_idx].sort()  # 保持时间顺序

    def _check_completion(self):
        """检查是否所有任务完成"""
        for batch_idx in range(self.batch_size):
            if batch_idx in self.batch_idxes:
                # 检查是否所有任务都完成
                if self.mask_task_finish_batch[batch_idx].all():
                    # 计算最终的makespan（包括所有USV返回起点）
                    max_return_time = 0
                    for usv_id in range(self.num_usvs):
                        usv_pos = (
                            self.feat_usvs_batch[batch_idx, 2, usv_id].item(),
                            self.feat_usvs_batch[batch_idx, 3, usv_id].item()
                        )
                        return_distance = calculate_distance(usv_pos, self.start_point)
                        return_time = calculate_navigation_time(return_distance, self.usv_speed)
                        usv_available_time = self.usvs_batch[batch_idx, usv_id, 1].item()
                        total_return_time = usv_available_time + return_time
                        max_return_time = max(max_return_time, total_return_time)

                    self.makespan_batch[batch_idx] = max_return_time
                    self.done_batch[batch_idx] = True

        # 更新全局完成状态
        self.done = self.done_batch.all()

        # 更新batch_idxes，只包含未完成的实例
        unfinished_mask = ~self.done_batch
        if unfinished_mask.any():
            # 确保设备一致性：torch.arange使用与unfinished_mask相同的设备
            self.batch_idxes = torch.arange(self.batch_size, device=unfinished_mask.device)[unfinished_mask]
        else:
            self.batch_idxes = torch.tensor([], dtype=torch.long, device=unfinished_mask.device)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        重置环境到初始状态
        """
        if seed is not None:
            self.np_random, self._last_reset_seed = seeding.np_random(seed)
        else:
            generated_seed = int(self.np_random.integers(0, 2**32 - 1))
            self.np_random, _ = seeding.np_random(generated_seed)
            self._last_reset_seed = generated_seed
        self._last_reset_options = options or {}

        # 同步 Python / NumPy / Torch 随机数种子，确保复现性
        random.seed(self._last_reset_seed)
        np.random.seed(self._last_reset_seed % (2**32 - 1))
        torch.manual_seed(self._last_reset_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._last_reset_seed)

        # 方案C：混合优化 - 从state对象和补充数据中恢复状态
        self.state = copy.deepcopy(self.initial_state)

        # 从state对象中恢复所有主要数据
        self.feat_tasks_batch = copy.deepcopy(self.state.feat_tasks_batch)
        self.feat_usvs_batch = copy.deepcopy(self.state.feat_usvs_batch)
        self.task_usv_adj_dynamic_batch = copy.deepcopy(self.state.task_usv_adj_dynamic_batch)
        self.mask_task_procing_batch = copy.deepcopy(self.state.mask_task_procing_batch)
        self.mask_task_finish_batch = copy.deepcopy(self.state.mask_task_finish_batch)
        self.mask_usv_procing_batch = copy.deepcopy(self.state.mask_usv_procing_batch)
        self.time_batch = copy.deepcopy(self.state.time_batch)
        self.task_step_batch = copy.deepcopy(self.state.task_step_batch)
        self.batch_idxes = copy.deepcopy(self.state.batch_idxes)

        # task_task_adj_batch应该在initial_state中已经正确设置，直接恢复
        if hasattr(self.initial_state, 'task_task_adj_batch'):
            self.task_task_adj_batch = copy.deepcopy(self.initial_state.task_task_adj_batch)

        # 恢复补充数据（state对象中不包含的重要调度数据）
        self.schedules_batch = copy.deepcopy(self.initial_state_complement['schedules_batch'])
        self.usvs_batch = copy.deepcopy(self.initial_state_complement['usvs_batch'])

        # 设置充电站管理器引用
        self.state.charging_manager = self.charging_manager

        # 重置完成状态
        self.done_batch = torch.full(size=(self.batch_size,), dtype=torch.bool, fill_value=False, device=self.device)
        self.done = False

        # 重置makespan
        self.makespan_batch = torch.zeros(self.batch_size, device=self.device)

        # 清空事件队列
        self.event_queues = [[] for _ in range(self.batch_size)]

        reset_info = {
            "seed": self._last_reset_seed,
            "options": dict(self._last_reset_options),
            "batch_rewards": np.zeros(self.batch_size, dtype=np.float64),
            "batch_terminated": np.zeros(self.batch_size, dtype=bool),
            "batch_truncated": np.zeros(self.batch_size, dtype=bool),
        }
        if self.time_batch is not None:
            reset_info["time_batch"] = ensure_numpy(self.time_batch, dtype=np.float32)

        reset_result = pack_reset(self._state_to_numpy(self.state), reset_info)
        return reset_result.observation, reset_result.info

    def validate_constraints(self):
        """
        验证8大约束关系（使用新的约束验证器）

        Returns:
            tuple: (bool, dict) 是否满足约束和违反的约束信息
        """
        try:
            # 移除task_pre_adj_batch同步：已使用空间邻接矩阵代替前驱关系
            # 使用新的约束验证器进行验证
            validation_result = self.constraint_validator.validate_all(
                state=self.state,
                schedules_batch=self.schedules_batch,
                usvs_batch=self.usvs_batch,
                batch_idxes=torch.arange(self.batch_size)
            )

            # 提取传统格式的结果以保持向后兼容性
            all_valid = validation_result['overall_valid']
            violations = validation_result['violations']

            return all_valid, violations

        except Exception as e:
            # 如果验证器出现错误，返回错误信息
            print(f"约束验证过程中发生错误: {str(e)}")
            return False, {"validation_error": str(e)}

  
    def _check_battery_constraint(self, batch_idx, usv_id, task_id):
        """
        检查电池约束（支持充电站）
        :param batch_idx: 批次索引
        :param usv_id: USV ID
        :param task_id: 任务ID
        :return: 是否满足电池约束
        """
        # 检查是否为充电站动作
        if task_id == self.num_tasks:  # 充电站动作
            # 充电站动作不需要电池约束检查（或者使用简化逻辑）
            return True

        # 获取USV当前电量和位置
        current_energy_ratio = self.feat_usvs_batch[batch_idx, 1, usv_id].item()
        current_energy = current_energy_ratio * self.battery_capacity

        usv_pos = (
            self.feat_usvs_batch[batch_idx, 2, usv_id].item(),
            self.feat_usvs_batch[batch_idx, 3, usv_id].item()
        )

        # 获取任务位置和执行时间
        task_pos = (
            self.feat_tasks_batch[batch_idx, 2, task_id].item(),  # x坐标
            self.feat_tasks_batch[batch_idx, 3, task_id].item()   # y坐标
        )
        exec_time = self.proc_times_batch[batch_idx, task_id, usv_id].item()

        # 计算所需电量
        # 1. 航行到任务点
        distance_to_task = calculate_distance(usv_pos, task_pos)
        energy_to_task = self._calculate_navigation_energy(distance_to_task)

        # 2. 任务执行
        energy_for_task = self._calculate_task_energy(exec_time)

        # 3. 从任务点返回起始点
        distance_to_start = calculate_distance(task_pos, self.start_point)
        energy_to_start = self._calculate_navigation_energy(distance_to_start)

        total_energy_needed = energy_to_task + energy_for_task + energy_to_start

        # 检查是否满足电池约束
        return current_energy >= total_energy_needed

    def _handle_charging_decision(self, batch_idx, usv_id):
        """
        处理充电决策
        :param batch_idx: 批次索引
        :param usv_id: USV ID
        :return: 充电惩罚（基于时间增加）
        """
        usv_pos = (
            self.feat_usvs_batch[batch_idx, 2, usv_id].item(),
            self.feat_usvs_batch[batch_idx, 3, usv_id].item()
        )

        # 计算返回起点的航行时间
        distance_to_start = calculate_distance(usv_pos, self.start_point)
        travel_time_to_start = calculate_navigation_time(distance_to_start, self.usv_speed)

        # 充电时间
        charging_time = self.charge_time

        # 创建返回起点事件（USV到达起点）
        return_start_time = self.time_batch[batch_idx] + travel_time_to_start
        return_event = USVEvent(
            event_type='return',
            usv_id=usv_id,
            timestamp=return_start_time,
            location=self.start_point
        )
        self._add_event_to_queue(batch_idx, return_event)

        # 创建充电完成事件
        charge_completion_time = return_start_time + charging_time
        charge_complete_event = USVEvent(
            event_type='charge',
            usv_id=usv_id,
            timestamp=charge_completion_time,
            location=self.start_point
        )
        self._add_event_to_queue(batch_idx, charge_complete_event)

        # 更新USV状态（开始返回，标记为忙碌）
        self.feat_usvs_batch[batch_idx, 0, usv_id] = 0  # 标记为忙碌（返回中）
        # 注意：位置和电量在相应事件触发时更新，这里不立即更新

        self.usvs_batch[batch_idx, usv_id, 0] = 0  # 状态：忙碌（返回中）
        self.usvs_batch[batch_idx, usv_id, 1] = charge_completion_time  # 可用时间
        self.usvs_batch[batch_idx, usv_id, 2] += travel_time_to_start + charging_time  # 累计工作时间
        self.usvs_batch[batch_idx, usv_id, 3] = -2  # 特殊标记：充电中

        # 返回充电惩罚（充电时间增加了makespan）
        total_charging_time = travel_time_to_start + charging_time
        charging_penalty = -total_charging_time

        return charging_penalty

    def _process_event_queue(self, batch_idx):
        """
        处理事件队列
        :param batch_idx: 批次索引
        """
        while self.event_queues[batch_idx] and self.event_queues[batch_idx][0].timestamp <= self.time_batch[batch_idx]:
            event = self.event_queues[batch_idx].pop(0)

            if event.event_type == 'complete':
                self._handle_task_complete_event(batch_idx, event)
            elif event.event_type == 'charge':
                self._handle_charge_complete_event(batch_idx, event)
            elif event.event_type == 'return':
                self._handle_return_event(batch_idx, event)
            elif event.event_type == 'arrive':
                self._handle_arrive_event(batch_idx, event)

    def _handle_task_complete_event(self, batch_idx, event):
        """处理任务完成事件"""
        task_id = event.task_id
        usv_id = event.usv_id

        # 更新任务状态
        self.feat_tasks_batch[batch_idx, 0, task_id] = 2  # 标记为已完成
        self.mask_task_procing_batch[batch_idx, task_id] = False
        self.mask_task_finish_batch[batch_idx, task_id] = True

        # 更新USV状态
        self.feat_usvs_batch[batch_idx, 0, usv_id] = 1  # 标记为空闲
        self.usvs_batch[batch_idx, usv_id, 0] = 1  # 状态：空闲
        self.usvs_batch[batch_idx, usv_id, 3] = -1  # 清除当前任务

        # 移除利用率维度更新（4维方案中已去掉利用率特征）

    def _handle_charge_complete_event(self, batch_idx, event):
        """处理充电完成事件"""
        usv_id = event.usv_id

        # 更新USV状态：充电完成，变为可用
        self.feat_usvs_batch[batch_idx, 0, usv_id] = 1  # 标记为空闲
        self.feat_usvs_batch[batch_idx, 1, usv_id] = 1.0  # 电量充满
        self.feat_usvs_batch[batch_idx, 2, usv_id] = self.start_point[0]  # 位置：起点
        self.feat_usvs_batch[batch_idx, 3, usv_id] = self.start_point[1]  # 位置：起点

        # 更新调度状态：USV变为空闲，可以接受新任务
        self.usvs_batch[batch_idx, usv_id, 0] = 1  # 状态：空闲
        self.usvs_batch[batch_idx, usv_id, 1] = event.timestamp  # 可用时间
        self.usvs_batch[batch_idx, usv_id, 3] = -1  # 清除当前任务标记

    def _handle_return_event(self, batch_idx, event):
        """处理返回起点事件"""
        usv_id = event.usv_id

        # 更新USV位置到起点
        self.feat_usvs_batch[batch_idx, 2, usv_id] = self.start_point[0]
        self.feat_usvs_batch[batch_idx, 3, usv_id] = self.start_point[1]

        # USV到达起点，准备充电
        # 状态保持忙碌（充电中），将在charge事件中变为空闲
        self.feat_usvs_batch[batch_idx, 0, usv_id] = 0  # 标记为忙碌（充电中）

        # 消耗返回航行的电量
        # 电量计算已经在_handle_charging_decision中预先考虑了
        # 这里保持当前电量不变，充电将在charge事件中处理

        # 更新调度状态：到达起点，准备充电
        self.usvs_batch[batch_idx, usv_id, 3] = -2  # 特殊标记：充电中

    def _handle_arrive_event(self, batch_idx, event):
        """处理到达任务点事件"""
        usv_id = event.usv_id
        task_id = event.task_id
        task_pos = event.location

        # 更新USV位置到任务点
        self.feat_usvs_batch[batch_idx, 2, usv_id] = task_pos[0]  # x坐标
        self.feat_usvs_batch[batch_idx, 3, usv_id] = task_pos[1]  # y坐标

        # 更新任务状态：开始执行
        self.mask_task_procing_batch[batch_idx, task_id] = True

        # 消耗航行电量
        usv_pos_before = (
            self.feat_usvs_batch[batch_idx, 2, usv_id].item(),
            self.feat_usvs_batch[batch_idx, 3, usv_id].item()
        )
        # 注意：这里的位置已经是更新后的，需要从调度信息中获取起始位置
        # 简化处理：假设已经消耗了航行电量，在execute_task中已经计算

        # USV状态保持忙碌（执行任务中）
        self.feat_usvs_batch[batch_idx, 0, usv_id] = 0  # 标记为忙碌（执行任务中）

    def _update_environment_state(self):
        """更新环境状态"""
        # 推进时间到下一个事件或当前动作时间
        for batch_idx in self.batch_idxes:
            if self.event_queues[batch_idx]:
                # 推进到最早的事件时间
                next_event_time = self.event_queues[batch_idx][0].timestamp
                self.time_batch[batch_idx] = min(self.time_batch[batch_idx], next_event_time)

            # 处理所有当前时间的事件
            self._process_event_queue(batch_idx)

        # 更新makespan（基于当前最晚的预计完成时间）
        for batch_idx in self.batch_idxes:
            current_estimated_makespan = 0
            for task_id in range(self.num_tasks):
                if self.schedules_batch[batch_idx, task_id, 0] == 1:  # 已调度的任务
                    task_completion_time = self.schedules_batch[batch_idx, task_id, 3].item()
                    task_pos = (
                        self.feat_tasks_batch[batch_idx, 2, task_id].item(),  # x坐标
                        self.feat_tasks_batch[batch_idx, 3, task_id].item()   # y坐标
                    )
                    return_time = calculate_navigation_time(
                        calculate_distance(task_pos, self.start_point), self.usv_speed
                    )
                    estimated_makespan = task_completion_time + return_time
                    current_estimated_makespan = max(current_estimated_makespan, estimated_makespan)

            # 也考虑当前正在执行任务的USV返回时间
            for usv_id in range(self.num_usvs):
                if self.usvs_batch[batch_idx, usv_id, 0] == 0:  # 忙碌的USV
                    usv_available_time = self.usvs_batch[batch_idx, usv_id, 1].item()
                    usv_pos = (
                        self.feat_usvs_batch[batch_idx, 2, usv_id].item(),
                        self.feat_usvs_batch[batch_idx, 3, usv_id].item()
                    )
                    return_time = calculate_navigation_time(
                        calculate_distance(usv_pos, self.start_point), self.usv_speed
                    )
                    estimated_makespan = usv_available_time + return_time
                    current_estimated_makespan = max(current_estimated_makespan, estimated_makespan)

            if current_estimated_makespan > 0:
                self.makespan_batch[batch_idx] = current_estimated_makespan

    def _calculate_reward(self):
        """
        计算奖励（基于makespan改进）
        """
        # 奖励已经在step方法中计算，这里可以留空
        return torch.zeros(self.batch_size, device=self.device)

    def render(self, mode='human'):
        """
        渲染环境状态
        """
        if mode == 'human':
            for batch_idx in range(self.batch_size):
                print(f"\n=== Batch {batch_idx} ===")
                print(f"当前时间: {self.time_batch[batch_idx]:.2f}")
                print(f"Makespan: {self.makespan_batch[batch_idx]:.2f}")
                print(f"是否完成: {self.done_batch[batch_idx]}")

                # 显示USV状态（4维USV特征：可用/SOC/x/y）
                print("\nUSV状态:")
                for usv_id in range(self.num_usvs):
                    status = "空闲" if self.feat_usvs_batch[batch_idx, 0, usv_id] > 0.5 else "忙碌"
                    energy = self.feat_usvs_batch[batch_idx, 1, usv_id].item() * 100
                    pos_x = self.feat_usvs_batch[batch_idx, 2, usv_id].item()
                    pos_y = self.feat_usvs_batch[batch_idx, 3, usv_id].item()
                    print(f"  USV{usv_id}: {status}, 电量{energy:.1f}%, 位置({pos_x:.1f},{pos_y:.1f})")

                # 显示任务状态
                print("\n任务状态:")
                completed_count = 0
                for task_id in range(min(10, self.num_tasks)):  # 只显示前10个任务
                    status = int(self.feat_tasks_batch[batch_idx, 0, task_id].item())
                    status_str = ["未分配", "已分配", "已完成"][status]
                    # 在4维方案中，任务类型信息需要从临时变量获取（从task_types矩阵）
                    if hasattr(self, 'task_types') and self.task_types is not None:
                        task_type = int(self.task_types[batch_idx, task_id].item()) if self.task_types.dim() > 1 else int(self.task_types[task_id].item())
                    else:
                        task_type = 1  # 默认Type1
                    exec_time = self.feat_tasks_batch[batch_idx, 1, task_id].item()
                    pos_x = self.feat_tasks_batch[batch_idx, 2, task_id].item()  # x坐标
                    pos_y = self.feat_tasks_batch[batch_idx, 3, task_id].item()  # y坐标
                    print(f"  任务{task_id}: {status_str}, Type{task_type}, 时间{exec_time:.1f}, 位置({pos_x:.1f},{pos_y:.1f})")
                    if status == 2:
                        completed_count += 1

                if self.num_tasks > 10:
                    print(f"  ... 还有{self.num_tasks - 10}个任务")
                print(f"完成进度: {completed_count}/{self.num_tasks}")

    def close(self):
        """
        关闭环境
        """
        pass


# 辅助函数
def calculate_distance(point1, point2):
    """计算两点之间的欧几里得距离"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_navigation_time(distance, speed):
    """计算航行时间"""
    return distance / speed


def calculate_fuzzy_expectation(t1, t2, t3):
    """计算三角模糊数的期望值"""
    return (t1 + 2*t2 + t3) / 4


def unwrap_env(env: Any) -> Any:
    """Return the innermost Gymnasium environment instance."""
    return getattr(env, "unwrapped", env)


def get_env_attr(env: Any, attr: str) -> Any:
    """Safely retrieve an attribute from the innermost environment."""
    return getattr(unwrap_env(env), attr)


def set_env_attr(env: Any, attr: str, value: Any) -> None:
    """Safely set an attribute on the innermost environment."""
    setattr(unwrap_env(env), attr, value)
