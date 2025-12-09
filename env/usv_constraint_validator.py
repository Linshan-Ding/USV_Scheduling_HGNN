
import time
from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np


class ConstraintResult:
    """约束验证结果类"""

    def __init__(self, constraint_name: str, is_valid: bool, message: str = None,
                 details: Dict[str, Any] = None):
        """
        初始化约束验证结果

        Args:
            constraint_name: 约束名称
            is_valid: 是否满足约束
            message: 详细消息
            details: 额外的详细信息
        """
        self.constraint_name = constraint_name
        self.is_valid = is_valid
        self.message = message or ("满足约束" if is_valid else "违反约束")
        self.details = details or {}
        self.timestamp = time.time()

    def __str__(self):
        return f"{self.constraint_name}: {'✓' if self.is_valid else '✗'} - {self.message}"

    def __repr__(self):
        return (f"ConstraintResult(name='{self.constraint_name}', "
                f"valid={self.is_valid}, message='{self.message}')")


class USVConstraintValidator:
    """
    USV约束验证器类

    封装所有USV调度的约束验证逻辑，包括：
    1. 电池容量约束
    2. 任务开始时间约束
    3. 任务分配约束
    4. 单次出航任务约束
    5. 访问任务约束（出度/入度）
    6. 任务流平衡约束
    7. 起点任务时间约束
    """

    def __init__(self, num_usvs: int, num_tasks: int, start_point: Tuple[float, float] = (0.0, 0.0)):
        """
        初始化约束验证器

        Args:
            num_usvs: USV数量
            num_tasks: 任务数量
            start_point: 起始点坐标
        """
        # 固定的物理参数（根据用户要求）
        self.battery_capacity = 1200.0  # 电池容量
        self.map_size = 800.0           # 地图尺寸
        self.usv_speed = 5.0            # USV航速
        self.energy_cost_per_distance = 1.0      # 单位距离能耗
        self.task_time_energy_ratio = 0.25       # 任务执行时间能耗比

        # 环境参数
        self.num_usvs = num_usvs
        self.num_tasks = num_tasks
        self.start_point = start_point

        # 验证参数
        self.enable_strict_validation = True
        self.error_tolerance = 1e-6

    def validate_all(self, state: Any, schedules_batch: torch.Tensor,
                    usvs_batch: torch.Tensor, batch_idxes: torch.Tensor = None) -> Dict[str, Any]:
        """
        验证所有约束条件

        Args:
            state: USVState对象
            schedules_batch: 调度批次数据
            usvs_batch: USV批次数据
            batch_idxes: 批次索引

        Returns:
            dict: 验证结果，包含overall_valid和各约束的详细信息
        """
        if batch_idxes is None:
            batch_idxes = torch.arange(schedules_batch.shape[0])

        all_results = []
        batch_violations = {}
        all_valid = True

        for batch_idx in batch_idxes:
            batch_results = []
            batch_valid = True

            # 约束1：电池容量约束
            result = self.validate_battery_capacity(
                state, schedules_batch, usvs_batch, batch_idx
            )
            batch_results.append(result)
            if not result.is_valid:
                batch_valid = False

            # 约束2：任务开始时间约束
            result = self.validate_task_start_time(
                state, schedules_batch, usvs_batch, batch_idx
            )
            batch_results.append(result)
            if not result.is_valid:
                batch_valid = False

            # 约束3：任务分配约束
            result = self.validate_task_assignment(
                state, schedules_batch, usvs_batch, batch_idx
            )
            batch_results.append(result)
            if not result.is_valid:
                batch_valid = False

            # 约束4：单次出航任务约束
            result = self.validate_single_trip(
                state, schedules_batch, usvs_batch, batch_idx
            )
            batch_results.append(result)
            if not result.is_valid:
                batch_valid = False

            # 约束5：访问任务约束（出度）
            result = self.validate_visit_degree(
                state, schedules_batch, usvs_batch, batch_idx, 'out'
            )
            batch_results.append(result)
            if not result.is_valid:
                batch_valid = False

            # 约束6：访问任务约束（入度）
            result = self.validate_visit_degree(
                state, schedules_batch, usvs_batch, batch_idx, 'in'
            )
            batch_results.append(result)
            if not result.is_valid:
                batch_valid = False

            # 约束7：任务流平衡约束
            result = self.validate_task_flow_balance(
                state, schedules_batch, usvs_batch, batch_idx
            )
            batch_results.append(result)
            if not result.is_valid:
                batch_valid = False

            # 约束8：起点任务时间约束
            result = self.validate_start_point(
                state, schedules_batch, usvs_batch, batch_idx
            )
            batch_results.append(result)
            if not result.is_valid:
                batch_valid = False

            all_results.extend(batch_results)

            if not batch_valid:
                violations = [r.constraint_name for r in batch_results if not r.is_valid]
                batch_violations[f"batch_{batch_idx}"] = violations
                all_valid = False

        return {
            'overall_valid': all_valid,
            'violations': batch_violations,
            'constraint_results': all_results,
            'summary': self._generate_summary(all_results)
        }

    def validate_battery_capacity(self, state: Any, schedules_batch: torch.Tensor,
                                 usvs_batch: torch.Tensor, batch_idx: int) -> ConstraintResult:
        """
        验证电池容量约束：每个USV单次往返总航行距离不能超过最大电池容量

        Args:
            state: USVState对象
            schedules_batch: 调度批次数据
            usvs_batch: USV批次数据
            batch_idx: 批次索引

        Returns:
            ConstraintResult: 验证结果
        """
        try:
            # 按USV分组任务，构建每个USV的任务序列
            usv_task_sequences = self._get_usv_task_sequences(
                schedules_batch, batch_idx
            )

            # 如果返回None，表示存在无效USV ID，违反约束
            if usv_task_sequences is None:
                return ConstraintResult(
                    "电池容量约束",
                    False,
                    "存在无效的USV ID分配"
                )

            for usv_id, task_sequence in usv_task_sequences.items():
                if not task_sequence:  # 空序列，跳过
                    continue

                # 验证每个任务序列的电量消耗
                if not self._validate_task_sequence_battery(
                    state, schedules_batch, batch_idx, usv_id, task_sequence
                ):
                    return ConstraintResult(
                        "电池容量约束",
                        False,
                        f"USV{usv_id}的任务序列超出电池容量限制",
                        {'usv_id': usv_id, 'task_sequence': task_sequence}
                    )

            return ConstraintResult(
                "电池容量约束",
                True,
                "所有USV的任务序列都满足电池容量约束"
            )

        except Exception as e:
            return ConstraintResult(
                "电池容量约束",
                False,
                f"验证过程中发生错误: {str(e)}"
            )

    def validate_task_start_time(self, state: Any, schedules_batch: torch.Tensor,
                                usvs_batch: torch.Tensor, batch_idx: int) -> ConstraintResult:
        """
        验证任务开始时间约束：任务开始时间必须满足前驱任务完成时间约束

        Args:
            state: USVState对象
            schedules_batch: 调度批次数据
            usvs_batch: USV批次数据
            batch_idx: 批次索引

        Returns:
            ConstraintResult: 验证结果
        """
        try:
            # 检查任务开始时间是否满足前驱任务完成时间约束
            for task_id in range(self.num_tasks):
                if schedules_batch[batch_idx, task_id, 0] == 1:  # 已调度的任务
                    start_time = schedules_batch[batch_idx, task_id, 2].item()

                    # 检查前驱任务
                    for prev_task_id in range(self.num_tasks):
                        if (state.task_pre_adj_batch[batch_idx, prev_task_id, task_id] == 1 and
                            schedules_batch[batch_idx, prev_task_id, 0] == 1):  # 有前驱关系且前驱任务已调度
                            prev_completion_time = schedules_batch[batch_idx, prev_task_id, 3].item()
                            if start_time < prev_completion_time - self.error_tolerance:
                                return ConstraintResult(
                                    "任务开始时间约束",
                                    False,
                                    f"任务{task_id}的开始时间{start_time:.2f}早于前驱任务{prev_task_id}的完成时间{prev_completion_time:.2f}",
                                    {
                                        'task_id': task_id,
                                        'start_time': start_time,
                                        'prev_task_id': prev_task_id,
                                        'prev_completion_time': prev_completion_time
                                    }
                                )

            return ConstraintResult(
                "任务开始时间约束",
                True,
                "所有任务的开始时间都满足前驱任务约束"
            )

        except Exception as e:
            return ConstraintResult(
                "任务开始时间约束",
                False,
                f"验证过程中发生错误: {str(e)}"
            )

    def validate_task_assignment(self, state: Any, schedules_batch: torch.Tensor,
                                usvs_batch: torch.Tensor, batch_idx: int) -> ConstraintResult:
        """
        验证任务分配约束：每个任务只能分配给一个USV

        Args:
            state: USVState对象
            schedules_batch: 调度批次数据
            usvs_batch: USV批次数据
            batch_idx: 批次索引

        Returns:
            ConstraintResult: 验证结果
        """
        try:
            task_assignments = {}  # 记录任务分配情况

            for task_id in range(self.num_tasks):
                if schedules_batch[batch_idx, task_id, 0] == 1:  # 已调度的任务
                    assigned_usv_id = int(schedules_batch[batch_idx, task_id, 1].item())

                    # 检查分配的USV ID是否在有效范围内
                    if assigned_usv_id < 0 or assigned_usv_id >= self.num_usvs:
                        return ConstraintResult(
                            "任务分配约束",
                            False,
                            f"任务{task_id}分配给无效的USV ID {assigned_usv_id}",
                            {'task_id': task_id, 'invalid_usv_id': assigned_usv_id}
                        )

                    # 检查是否有重复分配
                    if task_id in task_assignments:
                        return ConstraintResult(
                            "任务分配约束",
                            False,
                            f"任务{task_id}被重复分配给多个USV",
                            {
                                'task_id': task_id,
                                'assignments': task_assignments[task_id] + [assigned_usv_id]
                            }
                        )

                    task_assignments[task_id] = [assigned_usv_id]

            return ConstraintResult(
                "任务分配约束",
                True,
                "所有任务都正确分配给唯一的USV"
            )

        except Exception as e:
            return ConstraintResult(
                "任务分配约束",
                False,
                f"验证过程中发生错误: {str(e)}"
            )

    def validate_single_trip(self, state: Any, schedules_batch: torch.Tensor,
                            usvs_batch: torch.Tensor, batch_idx: int) -> ConstraintResult:
        """
        验证单次出航任务约束：每个任务仅能属于该USV的一次出航任务序列

        Args:
            state: USVState对象
            schedules_batch: 调度批次数据
            usvs_batch: USV批次数据
            batch_idx: 批次索引

        Returns:
            ConstraintResult: 验证结果
        """
        try:
            # 获取每个USV的任务序列
            usv_task_sequences = self._get_usv_task_sequences(
                schedules_batch, batch_idx
            )

            # 如果返回None，表示存在无效USV ID，违反约束
            if usv_task_sequences is None:
                return ConstraintResult(
                    "单次出航任务约束",
                    False,
                    "存在无效的USV ID分配"
                )

            for usv_id, task_sequence in usv_task_sequences.items():
                if not task_sequence:  # 空序列，跳过
                    continue

                # 验证任务序列是否满足单次出航约束
                if not self._validate_single_trip_sequence(
                    state, schedules_batch, batch_idx, usv_id, task_sequence
                ):
                    return ConstraintResult(
                        "单次出航任务约束",
                        False,
                        f"USV{usv_id}的任务序列不满足单次出航约束",
                        {'usv_id': usv_id, 'task_sequence': task_sequence}
                    )

            return ConstraintResult(
                "单次出航任务约束",
                True,
                "所有USV的任务序列都满足单次出航约束"
            )

        except Exception as e:
            return ConstraintResult(
                "单次出航任务约束",
                False,
                f"验证过程中发生错误: {str(e)}"
            )

    def validate_visit_degree(self, state: Any, schedules_batch: torch.Tensor,
                             usvs_batch: torch.Tensor, batch_idx: int,
                             degree_type: str = 'out') -> ConstraintResult:
        """
        验证访问任务约束（度数）：每个任务节点的入度或出度约束

        Args:
            state: USVState对象
            schedules_batch: 调度批次数据
            usvs_batch: USV批次数据
            batch_idx: 批次索引
            degree_type: 度数类型 ('out' 验证出度约束, 'in' 验证入度约束)

        Returns:
            ConstraintResult: 验证结果
        """
        try:
            if degree_type not in ['out', 'in']:
                raise ValueError(f"无效的度数类型参数: {degree_type}")

            # 获取每个USV的任务序列
            usv_task_sequences = self._get_usv_task_sequences(
                schedules_batch, batch_idx
            )

            # 如果返回None，表示存在无效USV ID，违反约束
            if usv_task_sequences is None:
                return ConstraintResult(
                    f"访问任务约束（{degree_type}度）",
                    False,
                    "存在无效的USV ID分配"
                )

            for usv_id, task_sequence in usv_task_sequences.items():
                if not task_sequence or len(task_sequence) <= 1:
                    continue  # 空序列或单任务序列满足约束

                if degree_type == 'out':
                    # 验证出度约束：除最后一个任务外，每个任务应该只有一个后继
                    for i in range(len(task_sequence) - 1):
                        current_task = task_sequence[i]
                        expected_successor = task_sequence[i + 1]

                        # 检查当前任务的实际后继数量
                        actual_successors = self._count_task_connections(
                            schedules_batch, batch_idx, current_task, usv_id, 'successor'
                        )

                        if actual_successors != 1:
                            return ConstraintResult(
                                f"访问任务约束（出度）",
                                False,
                                f"USV{usv_id}的任务{current_task}后继数量为{actual_successors}，应为1",
                                {
                                    'usv_id': usv_id,
                                    'task_id': current_task,
                                    'actual_successors': actual_successors,
                                    'expected_successors': 1
                                }
                            )

                        # 验证后继任务是否正确
                        if not self._is_task_connected(
                            schedules_batch, batch_idx, current_task, expected_successor, usv_id, 'successor'
                        ):
                            return ConstraintResult(
                                f"访问任务约束（出度）",
                                False,
                                f"USV{usv_id}的任务{current_task}的后继任务不正确",
                                {
                                    'usv_id': usv_id,
                                    'current_task': current_task,
                                    'expected_successor': expected_successor
                                }
                            )

                elif degree_type == 'in':
                    # 验证入度约束：除第一个任务外，每个任务应该只有一个前驱
                    for i in range(1, len(task_sequence)):
                        current_task = task_sequence[i]
                        expected_predecessor = task_sequence[i - 1]

                        # 检查当前任务的实际前驱数量
                        actual_predecessors = self._count_task_connections(
                            schedules_batch, batch_idx, current_task, usv_id, 'predecessor'
                        )

                        if actual_predecessors != 1:
                            return ConstraintResult(
                                f"访问任务约束（入度）",
                                False,
                                f"USV{usv_id}的任务{current_task}前驱数量为{actual_predecessors}，应为1",
                                {
                                    'usv_id': usv_id,
                                    'task_id': current_task,
                                    'actual_predecessors': actual_predecessors,
                                    'expected_predecessors': 1
                                }
                            )

                        # 验证前驱任务是否正确
                        if not self._is_task_connected(
                            schedules_batch, batch_idx, current_task, expected_predecessor, usv_id, 'predecessor'
                        ):
                            return ConstraintResult(
                                f"访问任务约束（入度）",
                                False,
                                f"USV{usv_id}的任务{current_task}的前驱任务不正确",
                                {
                                    'usv_id': usv_id,
                                    'current_task': current_task,
                                    'expected_predecessor': expected_predecessor
                                }
                            )

            constraint_name = f"访问任务约束（{'出度' if degree_type == 'out' else '入度'}）"
            return ConstraintResult(
                constraint_name,
                True,
                f"所有任务的{degree_type}度约束都满足"
            )

        except Exception as e:
            constraint_name = f"访问任务约束（{'出度' if degree_type == 'out' else '入度'}）"
            return ConstraintResult(
                constraint_name,
                False,
                f"验证过程中发生错误: {str(e)}"
            )

    def validate_task_flow_balance(self, state: Any, schedules_batch: torch.Tensor,
                                  usvs_batch: torch.Tensor, batch_idx: int) -> ConstraintResult:
        """
        验证任务流平衡约束：任务的进入次数与离开次数相等

        Args:
            state: USVState对象
            schedules_batch: 调度批次数据
            usvs_batch: USV批次数据
            batch_idx: 批次索引

        Returns:
            ConstraintResult: 验证结果
        """
        try:
            # 获取每个USV的任务序列
            usv_task_sequences = self._get_usv_task_sequences(
                schedules_batch, batch_idx
            )

            # 如果返回None，表示存在无效USV ID，违反约束
            if usv_task_sequences is None:
                return ConstraintResult(
                    "任务流平衡约束",
                    False,
                    "存在无效的USV ID分配"
                )

            for usv_id, task_sequence in usv_task_sequences.items():
                if not task_sequence:
                    continue  # 空序列满足平衡约束

                # 验证流平衡：除首尾任务外，中间任务的入度应等于出度
                for i, task_id in enumerate(task_sequence):
                    in_degree = self._count_task_connections(
                        schedules_batch, batch_idx, task_id, usv_id, 'predecessor'
                    )
                    out_degree = self._count_task_connections(
                        schedules_batch, batch_idx, task_id, usv_id, 'successor'
                    )

                    if i == 0:
                        # 序列第一个任务：入度应为0，出度应为1（除非只有一个任务）
                        if len(task_sequence) == 1:
                            if in_degree != 0 or out_degree != 0:
                                return ConstraintResult(
                                    "任务流平衡约束",
                                    False,
                                    f"USV{usv_id}的单任务序列{task_id}度数不平衡：入度={in_degree}, 出度={out_degree}",
                                    {
                                        'usv_id': usv_id,
                                        'task_id': task_id,
                                        'in_degree': in_degree,
                                        'out_degree': out_degree,
                                        'sequence_length': 1
                                    }
                                )
                        else:
                            if in_degree != 0 or out_degree != 1:
                                return ConstraintResult(
                                    "任务流平衡约束",
                                    False,
                                    f"USV{usv_id}的首任务{task_id}度数不平衡：入度={in_degree}, 出度={out_degree}",
                                    {
                                        'usv_id': usv_id,
                                        'task_id': task_id,
                                        'in_degree': in_degree,
                                        'out_degree': out_degree,
                                        'position': 'first'
                                    }
                                )
                    elif i == len(task_sequence) - 1:
                        # 序列最后一个任务：入度应为1，出度应为0
                        if in_degree != 1 or out_degree != 0:
                            return ConstraintResult(
                                "任务流平衡约束",
                                False,
                                f"USV{usv_id}的尾任务{task_id}度数不平衡：入度={in_degree}, 出度={out_degree}",
                                {
                                    'usv_id': usv_id,
                                    'task_id': task_id,
                                    'in_degree': in_degree,
                                    'out_degree': out_degree,
                                    'position': 'last'
                                }
                            )
                    else:
                        # 中间任务：入度和出度都应为1
                        if in_degree != 1 or out_degree != 1:
                            return ConstraintResult(
                                "任务流平衡约束",
                                False,
                                f"USV{usv_id}的中间任务{task_id}度数不平衡：入度={in_degree}, 出度={out_degree}",
                                {
                                    'usv_id': usv_id,
                                    'task_id': task_id,
                                    'in_degree': in_degree,
                                    'out_degree': out_degree,
                                    'position': 'middle'
                                }
                            )

            return ConstraintResult(
                "任务流平衡约束",
                True,
                "所有任务序列的流平衡约束都满足"
            )

        except Exception as e:
            return ConstraintResult(
                "任务流平衡约束",
                False,
                f"验证过程中发生错误: {str(e)}"
            )

    def validate_start_point(self, state: Any, schedules_batch: torch.Tensor,
                            usvs_batch: torch.Tensor, batch_idx: int) -> ConstraintResult:
        """
        验证起点任务时间约束：起点的任务处理时间为0

        Args:
            state: USVState对象
            schedules_batch: 调度批次数据
            usvs_batch: USV批次数据
            batch_idx: 批次索引

        Returns:
            ConstraintResult: 验证结果
        """
        try:
            # 起点是(0,0)，不涉及任务处理时间，这个约束总是满足的
            return ConstraintResult(
                "起点任务时间约束",
                True,
                "起点不涉及任务处理时间，约束满足"
            )

        except Exception as e:
            return ConstraintResult(
                "起点任务时间约束",
                False,
                f"验证过程中发生错误: {str(e)}"
            )

    # ==================== 辅助方法 ====================

    def _get_usv_task_sequences(self, schedules_batch: torch.Tensor, batch_idx: int) -> Optional[Dict[int, List[int]]]:
        """获取每个USV的任务执行序列（按时间排序）"""
        usv_tasks = {}

        # 初始化每个USV的任务列表
        for usv_id in range(self.num_usvs):
            usv_tasks[usv_id] = []

        # 收集每个USV的任务
        for task_id in range(self.num_tasks):
            if schedules_batch[batch_idx, task_id, 0] == 1:  # 已调度的任务
                usv_id = int(schedules_batch[batch_idx, task_id, 1].item())
                start_time = schedules_batch[batch_idx, task_id, 2].item()

                # 检查USV ID是否有效
                if 0 <= usv_id < self.num_usvs:
                    usv_tasks[usv_id].append((task_id, start_time))
                else:
                    # 无效的USV ID，违反约束
                    return None

        # 按开始时间排序每个USV的任务序列
        for usv_id in usv_tasks:
            usv_tasks[usv_id].sort(key=lambda x: x[1])
            # 只保留任务ID
            usv_tasks[usv_id] = [task_id for task_id, _ in usv_tasks[usv_id]]

        return usv_tasks

    def _validate_task_sequence_battery(self, state: Any, schedules_batch: torch.Tensor,
                                       batch_idx: int, usv_id: int, task_sequence: List[int]) -> bool:
        """验证单个USV任务序列的电量约束"""
        current_position = self.start_point
        remaining_energy = self.battery_capacity  # 从满电开始

        for i, task_id in enumerate(task_sequence):
            # 获取任务位置
            task_pos = (
                state.feat_tasks_batch[batch_idx, 4, task_id].item(),
                state.feat_tasks_batch[batch_idx, 5, task_id].item()
            )

            # 计算到任务的航行距离和能耗
            travel_distance = self._calculate_distance(current_position, task_pos)
            travel_energy = self._calculate_navigation_energy(travel_distance)

            # 计算任务执行能耗
            exec_time = state.proc_times_batch[batch_idx, task_id, usv_id].item()
            task_energy = self._calculate_task_energy(exec_time)

            # 检查是否有足够电量到达任务点并执行
            energy_needed_to_task = travel_energy + task_energy

            # 如果电量不足，需要先返回起点充电
            if remaining_energy < energy_needed_to_task:
                # 计算返回起点的能耗
                return_distance = self._calculate_distance(current_position, self.start_point)
                return_energy = self._calculate_navigation_energy(return_distance)

                # 检查是否有足够电量返回起点
                if remaining_energy < return_energy:
                    return False  # 无法返回起点，违反约束

                # 返回起点充电，然后从起点前往任务
                remaining_energy = self.battery_capacity  # 充满电
                current_position = self.start_point

                # 重新计算从起点到任务的能耗
                travel_distance_from_start = self._calculate_distance(self.start_point, task_pos)
                travel_energy_from_start = self._calculate_navigation_energy(travel_distance_from_start)
                total_energy_needed = travel_energy_from_start + task_energy

                if remaining_energy < total_energy_needed:
                    return False  # 即使满电也无法完成任务

                remaining_energy -= total_energy_needed
            else:
                # 直接前往任务点
                remaining_energy -= energy_needed_to_task

            # 更新当前位置
            current_position = task_pos

        # 验证最后一个任务后能否返回起点
        if task_sequence:
            final_return_distance = self._calculate_distance(current_position, self.start_point)
            final_return_energy = self._calculate_navigation_energy(final_return_distance)

            if remaining_energy < final_return_energy:
                return False  # 无法返回起点，违反约束

        return True

    def _validate_single_trip_sequence(self, state: Any, schedules_batch: torch.Tensor,
                                       batch_idx: int, usv_id: int, task_sequence: List[int]) -> bool:
        """验证单个USV任务序列的单次出航约束"""
        # 根据电量约束，将任务序列分割为多个出航序列
        trip_segments = self._split_into_trip_segments(
            state, schedules_batch, batch_idx, usv_id, task_sequence
        )

        # 如果返回空列表，表示约束被违反
        if not trip_segments and task_sequence:
            return False

        # 验证每个trip segment是否合法
        for i, segment in enumerate(trip_segments):
            # 每个segment都应该是一次完整的出航（从起点出发，返回起点）
            if not self._validate_trip_segment(schedules_batch, batch_idx, usv_id, segment):
                return False

        return True

    def _split_into_trip_segments(self, state: Any, schedules_batch: torch.Tensor,
                                 batch_idx: int, usv_id: int, task_sequence: List[int]) -> List[List[int]]:
        """根据电量约束将任务序列分割为多个出航段"""
        if not task_sequence:
            return []

        segments = []
        current_segment = []
        current_position = self.start_point
        remaining_energy = self.battery_capacity

        for task_id in task_sequence:
            # 获取任务位置
            task_pos = (
                state.feat_tasks_batch[batch_idx, 4, task_id].item(),
                state.feat_tasks_batch[batch_idx, 5, task_id].item()
            )

            # 计算到达任务点并执行所需的电量
            travel_distance = self._calculate_distance(current_position, task_pos)
            travel_energy = self._calculate_navigation_energy(travel_distance)
            exec_time = state.proc_times_batch[batch_idx, task_id, usv_id].item()
            task_energy = self._calculate_task_energy(exec_time)

            # 计算从任务点返回起点所需的电量
            return_distance = self._calculate_distance(task_pos, self.start_point)
            return_energy = self._calculate_navigation_energy(return_distance)

            # 总所需电量 = 前往任务 + 执行任务 + 返回起点
            total_energy_needed = travel_energy + task_energy + return_energy

            # 检查电量是否足够
            if remaining_energy >= total_energy_needed:
                # 可以在当前出航中完成此任务
                current_segment.append(task_id)
                remaining_energy -= (travel_energy + task_energy)
                current_position = task_pos
            else:
                # 电量不足，需要开始新的出航
                if current_segment:  # 保存当前segment
                    segments.append(current_segment)

                # 开始新的出航（从起点出发）
                current_segment = [task_id]
                current_position = self.start_point
                remaining_energy = self.battery_capacity

                # 重新计算从起点到任务的电量需求
                travel_distance_from_start = self._calculate_distance(self.start_point, task_pos)
                travel_energy_from_start = self._calculate_navigation_energy(travel_distance_from_start)
                total_energy_from_start = travel_energy_from_start + task_energy + return_energy

                if remaining_energy < total_energy_from_start:
                    # 即使从起点出发也无法完成该任务，违反约束
                    # 返回空列表表示约束被违反
                    return []

                remaining_energy -= (travel_energy_from_start + task_energy)
                current_position = task_pos

        # 添加最后一个segment
        if current_segment:
            segments.append(current_segment)

        return segments

    def _validate_trip_segment(self, schedules_batch: torch.Tensor, batch_idx: int,
                              usv_id: int, segment: List[int]) -> bool:
        """验证单个出航段的合法性"""
        if not segment:  # 空段合法
            return True

        # 验证出航段中的任务顺序是否合理
        for i in range(len(segment) - 1):
            current_task = segment[i]
            next_task = segment[i + 1]

            # 检查任务时间顺序
            current_start_time = schedules_batch[batch_idx, current_task, 2].item()
            next_start_time = schedules_batch[batch_idx, next_task, 2].item()

            if current_start_time >= next_start_time:
                return False  # 任务时间顺序错误

        # 验证每个任务都分配给了正确的USV
        for task_id in segment:
            assigned_usv = int(schedules_batch[batch_idx, task_id, 1].item())
            if assigned_usv != usv_id:
                return False

        return True

    def _count_task_connections(self, schedules_batch: torch.Tensor, batch_idx: int,
                               task_id: int, usv_id: int, direction: str = 'successor') -> int:
        """计算指定任务的连接数量（前驱或后继）"""
        connection_count = 0

        for other_task_id in range(self.num_tasks):
            if other_task_id == task_id:
                continue

            if direction == 'successor':
                # 检查other_task_id是否是task_id的后继
                if self._is_task_connected(schedules_batch, batch_idx, task_id, other_task_id, usv_id, 'successor'):
                    connection_count += 1
            elif direction == 'predecessor':
                # 检查other_task_id是否是task_id的前驱
                if self._is_task_connected(schedules_batch, batch_idx, task_id, other_task_id, usv_id, 'predecessor'):
                    connection_count += 1
            else:
                raise ValueError(f"无效的方向参数: {direction}")

        return connection_count

    def _is_task_connected(self, schedules_batch: torch.Tensor, batch_idx: int,
                          current_task: int, related_task: int, usv_id: int, relation_type: str = 'successor') -> bool:
        """检查两个任务之间的连接关系（前驱或后继）"""
        # 前驱关系就是后继关系的反向
        if relation_type == 'predecessor':
            # 交换参数顺序，将前驱关系检查转换为后继关系检查
            current_task, related_task = related_task, current_task
            relation_type = 'successor'
        elif relation_type != 'successor':
            raise ValueError(f"无效的关系类型参数: {relation_type}")

        # 检查两个任务是否都分配给了同一个USV
        if (schedules_batch[batch_idx, current_task, 0] == 1 and
            schedules_batch[batch_idx, related_task, 0] == 1):

            current_usv = int(schedules_batch[batch_idx, current_task, 1].item())
            related_usv = int(schedules_batch[batch_idx, related_task, 1].item())

            if current_usv != usv_id or related_usv != usv_id:
                return False

            # 检查时间顺序：后继任务的开始时间应该晚于当前任务的完成时间
            current_completion_time = schedules_batch[batch_idx, current_task, 3].item()
            related_start_time = schedules_batch[batch_idx, related_task, 2].item()

            return related_start_time > current_completion_time

        return False

    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """计算两点之间的欧几里得距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _calculate_navigation_energy(self, distance: float) -> float:
        """计算航行电量消耗"""
        return distance * self.energy_cost_per_distance

    def _calculate_task_energy(self, execution_time: float) -> float:
        """计算任务执行电量消耗"""
        return execution_time * self.task_time_energy_ratio

    def _generate_summary(self, results: List[ConstraintResult]) -> Dict[str, Any]:
        """生成验证结果摘要"""
        total_constraints = len(results)
        passed_constraints = sum(1 for r in results if r.is_valid)
        failed_constraints = total_constraints - passed_constraints

        failed_types = [r.constraint_name for r in results if not r.is_valid]

        return {
            'total_constraints': total_constraints,
            'passed_constraints': passed_constraints,
            'failed_constraints': failed_constraints,
            'success_rate': passed_constraints / total_constraints if total_constraints > 0 else 0,
            'failed_constraint_types': failed_types,
            'validation_timestamp': time.time()
        }