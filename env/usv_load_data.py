
import torch
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

# 导入USV案例生成器的数据结构
try:
    from .usv_case_generator import USVCaseData, calculate_fuzzy_expectation, calculate_distance, calculate_navigation_time
except ImportError:
    # 如果作为独立脚本运行，使用绝对导入
    from usv_case_generator import USVCaseData, calculate_fuzzy_expectation, calculate_distance, calculate_navigation_time


@dataclass
class USVLoadedData:
    """
    USV加载后的数据结构
    包含环境所需的所有张量数据
    """
    # 基本矩阵（参考load_data.py的返回结构）
    matrix_proc_time: torch.Tensor          # 处理时间矩阵 (num_tasks, num_usvs)
    matrix_ope_ma_adj: torch.Tensor         # 任务-USV邻接矩阵 (num_tasks, num_usvs)
    matrix_pre_proc: torch.Tensor           # 前驱关系矩阵 (num_tasks, num_tasks)
    matrix_pre_proc_t: torch.Tensor         # 前驱关系矩阵转置 (num_tasks, num_tasks)

    # 任务相关数据（参考load_data.py结构）
    ope_appertain: torch.Tensor             # 任务归属向量 (num_tasks)
    num_ope_biases: torch.Tensor            # 任务偏移向量 (num_usvs)
    nums_ope: torch.Tensor                  # 每个USV的任务数 (num_usvs)
    matrix_cal_cumul: torch.Tensor          # 累积计算矩阵 (num_tasks, num_tasks)

    # USV特有的扩展数据
    task_features: torch.Tensor             # 任务特征矩阵 (6, num_tasks)
    usv_features: torch.Tensor              # USV特征矩阵 (4, num_usvs)
    task_positions: List[Tuple[float, float]]  # 任务位置列表
    usv_positions: List[Tuple[float, float]]   # USV位置列表
    task_types: List[str]                   # 任务类型列表
    environment_parameters: Dict[str, Any]  # 环境参数


class USVDataLoader:
    """
    USV数据加载器核心类

    功能：
    1. 将USVCaseData转换为环境所需张量格式
    2. 构建任务×USV处理时间矩阵
    3. 生成特征张量和邻接矩阵
    4. 实现特征标准化和格式转换
    5. 提供与load_data.py兼容的接口
    """

    def __init__(self, device: torch.device = torch.device('cpu'), normalize_features: bool = True):
        """
        初始化USV数据加载器

        :param device: 计算设备 (CPU/GPU)
        :param normalize_features: 是否标准化特征
        """
        self.device = device
        self.normalize_features = normalize_features

        # 特征维度定义（与usv_env.py保持一致）
        self.task_feat_dim = 4   # 任务特征维度
        self.usv_feat_dim = 4    # USV特征维度

        # 标准化参数（将在第一次加载数据时计算）
        self.feature_stats = {
            'task_pos_mean': None,
            'task_pos_std': None,
            'usv_pos_mean': None,
            'usv_pos_std': None,
            'exec_time_mean': None,
            'exec_time_std': None,
            'nav_time_mean': None,
            'nav_time_std': None
        }

    def load_usv_case(self, case_data: USVCaseData, batch_size: int = 1) -> USVLoadedData:
        """
        加载USV案例数据并转换为环境格式

        :param case_data: USV案例数据对象
        :param batch_size: 批次大小
        :return: USV加载后的数据结构
        """
        # 验证输入数据
        self._validate_case_data(case_data)

        # 计算标准化参数（如果需要）
        if self.normalize_features and self.feature_stats['task_pos_mean'] is None:
            self._compute_normalization_stats(case_data)

        # 构建核心矩阵
        matrix_proc_time = self._build_processing_time_matrix(case_data)

        # 使用新的空间邻接矩阵生成逻辑
        task_usv_adj, task_task_adj = self._build_spatial_adjacency_matrices(
            case_data.task_positions, case_data.usv_positions
        )

        # 用空间邻接矩阵替换原有邻接矩阵
        matrix_ope_ma_adj = task_usv_adj.long()

        # 空间邻接矩阵同时作为前驱关系矩阵（统一为纯空间关系）
        matrix_pre_proc = task_task_adj.long()
        matrix_pre_proc_t = matrix_pre_proc.t()

        # 构建任务相关数据
        ope_appertain, num_ope_biases, nums_ope, matrix_cal_cumul = self._build_task_structures(case_data)

        # 生成特征张量
        task_features = self._build_task_features(case_data)
        usv_features = self._build_usv_features(case_data)

        # 创建加载后的数据结构
        loaded_data = USVLoadedData(
            matrix_proc_time=matrix_proc_time,
            matrix_ope_ma_adj=matrix_ope_ma_adj,
            matrix_pre_proc=matrix_pre_proc,
            matrix_pre_proc_t=matrix_pre_proc_t,
            ope_appertain=ope_appertain,
            num_ope_biases=num_ope_biases,
            nums_ope=nums_ope,
            matrix_cal_cumul=matrix_cal_cumul,
            task_features=task_features,
            usv_features=usv_features,
            task_positions=case_data.task_positions,
            usv_positions=case_data.usv_positions,
            task_types=case_data.task_types,
            environment_parameters=case_data.environment_parameters
        )

        # 扩展到批次维度
        if batch_size > 1:
            loaded_data = self._expand_to_batch(loaded_data, batch_size)

        return loaded_data

    def load_usv_case_list(self, case_data_list: List[USVCaseData]) -> USVLoadedData:
        """
        加载多个USV案例数据（批次处理）

        :param case_data_list: USV案例数据列表
        :return: USV加载后的数据结构（批次格式）
        """
        if not case_data_list:
            raise ValueError("案例数据列表不能为空")

        batch_size = len(case_data_list)

        # 验证所有案例的兼容性
        self._validate_case_list_compatibility(case_data_list)

        # 逐个加载案例
        loaded_cases = []
        for case_data in case_data_list:
            loaded_case = self.load_usv_case(case_data, batch_size=1)
            loaded_cases.append(loaded_case)

        # 合并为批次张量
        batch_data = self._merge_cases_to_batch(loaded_cases)

        return batch_data

    def _validate_case_data(self, case_data: USVCaseData):
        """
        验证案例数据的有效性

        :param case_data: USV案例数据
        """
        if not isinstance(case_data, USVCaseData):
            raise TypeError("案例数据必须是USVCaseData类型")

        if case_data.num_usvs <= 0 or case_data.num_tasks <= 0:
            raise ValueError("USV数量和任务数量必须大于0")

        expected_task_pos_len = case_data.num_tasks
        expected_usv_pos_len = case_data.num_usvs

        if len(case_data.task_positions) != expected_task_pos_len:
            raise ValueError(f"任务位置数量({len(case_data.task_positions)})与任务数量({case_data.num_tasks})不匹配")

        if len(case_data.usv_positions) != expected_usv_pos_len:
            raise ValueError(f"USV位置数量({len(case_data.usv_positions)})与USV数量({case_data.num_usvs})不匹配")

        if len(case_data.task_types) != expected_task_pos_len:
            raise ValueError(f"任务类型数量({len(case_data.task_types)})与任务数量({case_data.num_tasks})不匹配")

    def _compute_normalization_stats(self, case_data: USVCaseData):
        """
        计算特征标准化参数

        :param case_data: USV案例数据
        """
        # 任务位置统计
        task_positions = np.array(case_data.task_positions)
        self.feature_stats['task_pos_mean'] = np.mean(task_positions, axis=0)
        self.feature_stats['task_pos_std'] = np.std(task_positions, axis=0) + 1e-8

        # USV位置统计
        usv_positions = np.array(case_data.usv_positions)
        self.feature_stats['usv_pos_mean'] = np.mean(usv_positions, axis=0)
        self.feature_stats['usv_pos_std'] = np.std(usv_positions, axis=0) + 1e-8

        # 执行时间统计
        exec_times = np.array(case_data.task_execution_times)
        self.feature_stats['exec_time_mean'] = np.mean(exec_times)
        self.feature_stats['exec_time_std'] = np.std(exec_times) + 1e-8

        # 航行时间统计
        nav_times = np.array(case_data.task_navigation_times)
        self.feature_stats['nav_time_mean'] = np.mean(nav_times)
        self.feature_stats['nav_time_std'] = np.std(nav_times) + 1e-8

    def _build_spatial_task_usv_adjacency(self, task_positions: List[Tuple[float, float]],
                                         usv_positions: List[Tuple[float, float]],
                                         distance_threshold: Optional[float] = None) -> torch.Tensor:
        """
        基于距离的任务-USV邻接矩阵：距离小于阈值为1，否则为0

        参数：
            task_positions: (N, 2) 任务坐标列表
            usv_positions: (M, 2) USV坐标列表
            distance_threshold: 距离阈值，None表示自动计算

        返回：
            task_usv_adj: (N, M) 任务-USV邻接矩阵
        """
        # 转换为PyTorch张量
        task_pos_tensor = torch.tensor(task_positions, dtype=torch.float32, device=self.device)
        usv_pos_tensor = torch.tensor(usv_positions, dtype=torch.float32, device=self.device)

        # 计算所有任务到所有USV的距离
        distances = torch.cdist(task_pos_tensor, usv_pos_tensor)  # (N, M)

        # 自动计算距离阈值：最大距离的70%
        if distance_threshold is None:
            distance_threshold = torch.max(distances) * 0.7

        # 构建邻接矩阵：距离小于阈值为1，否则为0
        task_usv_adj = torch.zeros(distances.shape, dtype=torch.float32, device=self.device)
        task_usv_adj[distances <= distance_threshold] = 1.0

        return task_usv_adj

    def _build_spatial_task_task_adjacency(self, task_positions: List[Tuple[float, float]],
                                          k_neighbors: int = 3) -> torch.Tensor:
        """
        基于K近邻的任务-任务邻接矩阵：每个任务连接最近的k个邻居

        参数：
            task_positions: (N, 2) 任务坐标列表
            k_neighbors: 近邻数量

        返回：
            task_task_adj: (N, N) 任务-任务邻接矩阵
        """
        N = len(task_positions)

        # 转换为PyTorch张量
        task_pos_tensor = torch.tensor(task_positions, dtype=torch.float32, device=self.device)

        # 计算任务间距离矩阵
        distances = torch.cdist(task_pos_tensor, task_pos_tensor)  # (N, N)

        # 初始化邻接矩阵
        task_task_adj = torch.zeros(N, N, dtype=torch.float32, device=self.device)

        for i in range(N):
            # 排除自身，选择最近的k个邻居
            distances_i = distances[i].clone()
            distances_i[i] = float('inf')  # 排除自身

            k = min(k_neighbors, N - 1)
            if k > 0:
                _, neighbor_indices = torch.topk(distances_i, k=k, largest=False)

                # 构建双向连接
                for j in neighbor_indices:
                    task_task_adj[i, j] = 1.0
                    task_task_adj[j, i] = 1.0

        return task_task_adj

    def _build_spatial_adjacency_matrices(self, task_positions: List[Tuple[float, float]],
                                       usv_positions: List[Tuple[float, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建纯空间的邻接矩阵（统一入口）

        参数：
            task_positions: (N, 2) 任务坐标列表
            usv_positions: (M, 2) USV坐标列表

        返回：
            task_usv_adj: (N, M) 任务-USV邻接矩阵
            task_task_adj: (N, N) 任务-任务邻接矩阵
        """
        task_usv_adj = self._build_spatial_task_usv_adjacency(task_positions, usv_positions)
        task_task_adj = self._build_spatial_task_task_adjacency(task_positions, k_neighbors=3)

        return task_usv_adj, task_task_adj

    def _build_processing_time_matrix(self, case_data: USVCaseData) -> torch.Tensor:
        """
        构建任务×USV处理时间矩阵

        设计原则：
        - 所有USV执行同一任务的时间相同（基于任务执行时间期望值）
        - 矩阵形状：(num_tasks, num_usvs)
        - 元素值：任务i由USV j执行的期望时间

        :param case_data: USV案例数据
        :return: 处理时间矩阵
        """
        num_tasks = case_data.num_tasks
        num_usvs = case_data.num_usvs

        # 创建处理时间矩阵
        proc_time_matrix = torch.zeros(num_tasks, num_usvs, dtype=torch.float32)

        # 填充处理时间（所有USV执行同一任务的时间相同）
        for task_id in range(num_tasks):
            exec_time = case_data.task_execution_times[task_id]
            proc_time_matrix[task_id, :] = exec_time

        return proc_time_matrix.to(self.device)

    def _build_adjacency_matrix(self, case_data: USVCaseData) -> torch.Tensor:
        """
        构建任务-USV邻接矩阵

        设计原则：
        - 基于case_data.task_usv_adjacency构建
        - 1表示该USV可以执行该任务
        - 矩阵形状：(num_tasks, num_usvs)

        :param case_data: USV案例数据
        :return: 邻接矩阵
        """
        adjacency_matrix = torch.tensor(
            case_data.task_usv_adjacency,
            dtype=torch.long,
            device=self.device
        )
        return adjacency_matrix

    def _build_precedence_matrices(self, case_data: USVCaseData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建前驱关系矩阵

        设计原则：
        - 基于case_data.task_predecessor构建
        - 1表示任务j是任务i的前驱
        - 矩阵形状：(num_tasks, num_tasks)

        :param case_data: USV案例数据
        :return: 前驱矩阵及其转置
        """
        pre_matrix = torch.tensor(
            case_data.task_predecessor,
            dtype=torch.long,
            device=self.device
        )
        pre_matrix_t = pre_matrix.t()

        return pre_matrix, pre_matrix_t

    def _build_task_structures(self, case_data: USVCaseData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建任务相关结构（参考load_data.py的返回结构）

        包括：
        - ope_appertain: 任务归属向量
        - num_ope_biases: 任务偏移向量
        - nums_ope: 每个USV的任务数
        - matrix_cal_cumul: 累积计算矩阵

        :param case_data: USV案例数据
        :return: 任务结构元组
        """
        num_tasks = case_data.num_tasks
        num_usvs = case_data.num_usvs

        # 任务归属向量（简化处理：平均分配任务给USV）
        tasks_per_usv = num_tasks // num_usvs
        ope_appertain = torch.zeros(num_tasks, dtype=torch.long)

        for task_id in range(num_tasks):
            usv_id = min(task_id // tasks_per_usv, num_usvs - 1)
            ope_appertain[task_id] = usv_id

        # 任务偏移向量（每个USV的第一个任务ID）
        num_ope_biases = torch.zeros(num_usvs, dtype=torch.long)
        for usv_id in range(num_usvs):
            num_ope_biases[usv_id] = usv_id * tasks_per_usv

        # 每个USV的任务数
        nums_ope = torch.full((num_usvs,), tasks_per_usv, dtype=torch.long)
        # 最后一个USV处理剩余任务
        nums_ope[-1] = num_tasks - (num_usvs - 1) * tasks_per_usv

        # 累积计算矩阵（简化为零矩阵，因为USV任务通常无前驱关系）
        matrix_cal_cumul = torch.zeros(num_tasks, num_tasks, dtype=torch.long)

        return ope_appertain.to(self.device), num_ope_biases.to(self.device), nums_ope.to(self.device), matrix_cal_cumul.to(self.device)

    def _build_task_features(self, case_data: USVCaseData) -> torch.Tensor:
        """
        构建任务特征矩阵

        特征维度（6维，与usv_env.py保持一致）：
        [0]: 任务状态 (0=未分配, 1=已分配, 2=已完成) - 初始为0
        [1]: 执行时间期望值（标准化后）
        [2]: 任务类型 (Type1=1.0, Type2=2.0, Type3=3.0)
        [3]: 距离起点的航行时间（标准化后）
        [4]: 任务x坐标（标准化后）
        [5]: 任务y坐标（标准化后）

        :param case_data: USV案例数据
        :return: 任务特征矩阵 (6, num_tasks)
        """
        num_tasks = case_data.num_tasks
        task_features = torch.zeros(self.task_feat_dim, num_tasks, dtype=torch.float32)

        for task_id in range(num_tasks):
            # [0]: 任务状态 - 初始为未分配
            task_features[0, task_id] = 0.0

            # [1]: 执行时间期望值
            exec_time = case_data.task_execution_times[task_id]
            if self.normalize_features:
                exec_time = (exec_time - self.feature_stats['exec_time_mean']) / self.feature_stats['exec_time_std']
            task_features[1, task_id] = exec_time

            # [2, 3]: 任务位置坐标（4维方案：去掉类型和航行时间）
            pos_x, pos_y = case_data.task_positions[task_id]
            if self.normalize_features:
                pos_x = (pos_x - self.feature_stats['task_pos_mean'][0]) / self.feature_stats['task_pos_std'][0]
                pos_y = (pos_y - self.feature_stats['task_pos_mean'][1]) / self.feature_stats['task_pos_std'][1]
            task_features[2, task_id] = pos_x
            task_features[3, task_id] = pos_y

        return task_features.to(self.device)

    def _build_usv_features(self, case_data: USVCaseData) -> torch.Tensor:
        """
        构建USV特征矩阵

        特征维度（4维，与usv_env.py保持一致）：
        [0]: 可用状态 (0=忙碌, 1=空闲) - 初始为1（空闲）
        [1]: 当前电量 (归一化到0-1) - 初始为1.0（满电）
        [2]: USV x坐标（标准化后）
        [3]: USV y坐标（标准化后）
        # 说明：利用率维度已移除，简化为4维方案

        :param case_data: USV案例数据
        :return: USV特征矩阵 (4, num_usvs)
        """
        num_usvs = case_data.num_usvs
        usv_features = torch.zeros(self.usv_feat_dim, num_usvs, dtype=torch.float32)

        for usv_id in range(num_usvs):
            # [0]: 可用状态 - 初始为空闲
            usv_features[0, usv_id] = 1.0

            # [1]: 当前电量 - 初始为满电
            usv_features[1, usv_id] = case_data.usv_initial_energy[usv_id]

            # [2, 3]: USV位置坐标（4维方案：去掉利用率维度）
            pos_x, pos_y = case_data.usv_positions[usv_id]
            if self.normalize_features:
                pos_x = (pos_x - self.feature_stats['usv_pos_mean'][0]) / self.feature_stats['usv_pos_std'][0]
                pos_y = (pos_y - self.feature_stats['usv_pos_mean'][1]) / self.feature_stats['usv_pos_std'][1]
            usv_features[2, usv_id] = pos_x
            usv_features[3, usv_id] = pos_y

        return usv_features.to(self.device)

    def _expand_to_batch(self, loaded_data: USVLoadedData, batch_size: int) -> USVLoadedData:
        """
        将单个案例数据扩展到批次维度

        :param loaded_data: 单个案例的加载数据
        :param batch_size: 批次大小
        :return: 批次格式的加载数据
        """
        # 扩展所有张量到批次维度
        batch_data = copy.deepcopy(loaded_data)

        # 核心矩阵扩展 (batch_size, ...)
        batch_data.matrix_proc_time = batch_data.matrix_proc_time.unsqueeze(0).expand(batch_size, -1, -1)
        batch_data.matrix_ope_ma_adj = batch_data.matrix_ope_ma_adj.unsqueeze(0).expand(batch_size, -1, -1)
        batch_data.matrix_pre_proc = batch_data.matrix_pre_proc.unsqueeze(0).expand(batch_size, -1, -1)
        batch_data.matrix_pre_proc_t = batch_data.matrix_pre_proc_t.unsqueeze(0).expand(batch_size, -1, -1)

        # 特征矩阵扩展 (batch_size, ...)
        batch_data.task_features = batch_data.task_features.unsqueeze(0).expand(batch_size, -1, -1)
        batch_data.usv_features = batch_data.usv_features.unsqueeze(0).expand(batch_size, -1, -1)

        # 向量扩展
        batch_data.ope_appertain = batch_data.ope_appertain.unsqueeze(0).expand(batch_size, -1)
        batch_data.num_ope_biases = batch_data.num_ope_biases.unsqueeze(0).expand(batch_size, -1)
        batch_data.nums_ope = batch_data.nums_ope.unsqueeze(0).expand(batch_size, -1)
        batch_data.matrix_cal_cumul = batch_data.matrix_cal_cumul.unsqueeze(0).expand(batch_size, -1, -1)

        return batch_data

    def _validate_case_list_compatibility(self, case_data_list: List[USVCaseData]):
        """
        验证案例列表的兼容性

        :param case_data_list: 案例数据列表
        """
        if not case_data_list:
            return

        # 检查所有案例的基本参数是否一致
        reference_case = case_data_list[0]

        for i, case_data in enumerate(case_data_list[1:], 1):
            if case_data.num_usvs != reference_case.num_usvs:
                raise ValueError(f"案例{i}的USV数量({case_data.num_usvs})与参考案例({reference_case.num_usvs})不匹配")

            if case_data.num_tasks != reference_case.num_tasks:
                raise ValueError(f"案例{i}的任务数量({case_data.num_tasks})与参考案例({reference_case.num_tasks})不匹配")

            if case_data.map_size != reference_case.map_size:
                raise ValueError(f"案例{i}的地图尺寸({case_data.map_size})与参考案例({reference_case.map_size})不匹配")

    def _merge_cases_to_batch(self, loaded_cases: List[USVLoadedData]) -> USVLoadedData:
        """
        将多个案例合并为批次张量

        :param loaded_cases: 加载后的案例列表
        :return: 批次格式的加载数据
        """
        if not loaded_cases:
            raise ValueError("案例列表不能为空")

        batch_size = len(loaded_cases)

        # 使用第一个案例作为参考
        batch_data = copy.deepcopy(loaded_cases[0])

        # 堆叠所有张量
        proc_time_list = []
        adj_list = []
        pre_proc_list = []
        pre_proc_t_list = []
        task_feat_list = []
        usv_feat_list = []
        ope_appertain_list = []
        num_ope_biases_list = []
        nums_ope_list = []
        cal_cumul_list = []

        for loaded_case in loaded_cases:
            proc_time_list.append(loaded_case.matrix_proc_time)
            adj_list.append(loaded_case.matrix_ope_ma_adj)
            pre_proc_list.append(loaded_case.matrix_pre_proc)
            pre_proc_t_list.append(loaded_case.matrix_pre_proc_t)
            task_feat_list.append(loaded_case.task_features)
            usv_feat_list.append(loaded_case.usv_features)
            ope_appertain_list.append(loaded_case.ope_appertain)
            num_ope_biases_list.append(loaded_case.num_ope_biases)
            nums_ope_list.append(loaded_case.nums_ope)
            cal_cumul_list.append(loaded_case.matrix_cal_cumul)

        # 堆叠为批次张量
        batch_data.matrix_proc_time = torch.stack(proc_time_list, dim=0)
        batch_data.matrix_ope_ma_adj = torch.stack(adj_list, dim=0)
        batch_data.matrix_pre_proc = torch.stack(pre_proc_list, dim=0)
        batch_data.matrix_pre_proc_t = torch.stack(pre_proc_t_list, dim=0)
        batch_data.task_features = torch.stack(task_feat_list, dim=0)
        batch_data.usv_features = torch.stack(usv_feat_list, dim=0)
        batch_data.ope_appertain = torch.stack(ope_appertain_list, dim=0)
        batch_data.num_ope_biases = torch.stack(num_ope_biases_list, dim=0)
        batch_data.nums_ope = torch.stack(nums_ope_list, dim=0)
        batch_data.matrix_cal_cumul = torch.stack(cal_cumul_list, dim=0)

        # 合并列表数据
        batch_data.task_positions = [case.task_positions for case in loaded_cases]
        batch_data.usv_positions = [case.usv_positions for case in loaded_cases]
        batch_data.task_types = [case.task_types for case in loaded_cases]
        batch_data.environment_parameters = [case.environment_parameters for case in loaded_cases]

        return batch_data


# 便捷函数（参考load_data.py的函数设计模式）
def load_usv_data(case_data: Union[USVCaseData, List[USVCaseData]],
                  device: torch.device = torch.device('cpu'),
                  normalize_features: bool = True) -> USVLoadedData:
    """
    便捷函数：加载USV案例数据

    :param case_data: 单个案例数据或案例列表
    :param device: 计算设备
    :param normalize_features: 是否标准化特征
    :return: 加载后的数据结构
    """
    loader = USVDataLoader(device=device, normalize_features=normalize_features)

    if isinstance(case_data, list):
        return loader.load_usv_case_list(case_data)
    else:
        return loader.load_usv_case(case_data, batch_size=1)


def load_usv_batch(case_data_list: List[USVCaseData],
                   device: torch.device = torch.device('cpu'),
                   normalize_features: bool = True) -> USVLoadedData:
    """
    便捷函数：加载USV案例批次

    :param case_data_list: 案例数据列表
    :param device: 计算设备
    :param normalize_features: 是否标准化特征
    :return: 加载后的批次数据结构
    """
    if not case_data_list:
        raise ValueError("案例数据列表不能为空")

    loader = USVDataLoader(device=device, normalize_features=normalize_features)
    return loader.load_usv_case_list(case_data_list)


# 兼容性函数（与load_data.py的load_fjs函数保持接口一致）
def load_usv_fjsp_format(case_data: USVCaseData,
                        device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, ...]:
    """
    兼容性函数：以FJSP格式返回USV数据

    返回格式与load_data.load_fjs保持一致：
    (matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc_t,
     ope_appertain, num_ope_biases, nums_ope, matrix_cal_cumul)

    :param case_data: USV案例数据
    :param device: 计算设备
    :return: FJSP格式的张量元组
    """
    loaded_data = load_usv_data(case_data, device=device)

    return (
        loaded_data.matrix_proc_time,
        loaded_data.matrix_ope_ma_adj,
        loaded_data.matrix_pre_proc,
        loaded_data.matrix_pre_proc_t,
        loaded_data.ope_appertain,
        loaded_data.num_ope_biases,
        loaded_data.nums_ope,
        loaded_data.matrix_cal_cumul
    )


if __name__ == "__main__":
    # 示例用法
    print("[START] 开始测试USV数据加载器...")

    # 创建测试案例
    from usv_case_generator import create_generator

    generator = create_generator(
        num_usvs=4,
        num_tasks=80,
        path='../data/',
        flag_doc=False,
        randomization_level="medium"
    )

    case_data = generator.get_case(idx=0)

    # 测试数据加载器
    loader = USVDataLoader(device=torch.device('cpu'), normalize_features=True)
    loaded_data = loader.load_usv_case(case_data, batch_size=1)

    print(f"   加载结果:")
    print(f"   处理时间矩阵形状: {loaded_data.matrix_proc_time.shape}")
    print(f"   邻接矩阵形状: {loaded_data.matrix_ope_ma_adj.shape}")
    print(f"   前驱矩阵形状: {loaded_data.matrix_pre_proc.shape}")
    print(f"   任务特征形状: {loaded_data.task_features.shape}")
    print(f"   USV特征形状: {loaded_data.usv_features.shape}")

    print(f"[SUCCESS] USV数据加载器测试完成！")
