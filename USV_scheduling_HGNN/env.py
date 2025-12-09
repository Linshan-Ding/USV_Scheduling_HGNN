# ============================================================================
# 2. env.py - 离散事件驱动仿真环境
# ============================================================================
import gym
from gym import spaces
import numpy as np

class USVSchedulingEnv(gym.Env):
    """USV调度环境 - 离散事件驱动"""

    def __init__(self, instance: dict):
        super().__init__()
        self.instance = instance
        self.config = instance['config']
        self.n_usvs = instance['n_usvs']
        self.n_tasks = instance['n_tasks']

        # 任务信息
        self.task_coords = instance['task_coords']
        self.fuzzy_times = instance['fuzzy_times']
        # 期望处理时间: (t1 + 2*t2 + t3) / 4
        self.task_durations = (
                                      self.fuzzy_times[:, 0] +
                                      2 * self.fuzzy_times[:, 1] +
                                      self.fuzzy_times[:, 2]
                              ) / 4.0

        # 动作空间：选择(task_id, usv_id)对
        self.action_space = spaces.Discrete(self.n_tasks * self.n_usvs)

        # 状态信息
        self.reset()

    def reset(self):
        """重置环境"""
        # USV状态 [x, y, battery, busy_until_time]
        self.usv_states = np.zeros((self.n_usvs, 4))
        self.usv_states[:, 2] = self.config.battery_capacity  # 初始电量

        # 任务状态 [scheduled, start_time, completion_time, assigned_usv]
        self.task_states = np.zeros((self.n_tasks, 4))
        self.task_states[:, 3] = -1  # 未分配

        self.current_time = 0.0
        self.n_scheduled_tasks = 0

        return self._get_state()

    def _get_state(self):
        """获取当前状态（异质图格式）"""
        # 返回状态字典，供HGNN使用
        state = {
            'usv_features': self.usv_states.copy(),  # [n_usvs, 4]
            'task_features': self._get_task_features(),  # [n_tasks, 5]
            'task_coords': self.task_coords.copy(),
            'n_scheduled': self.n_scheduled_tasks,
            'current_time': self.current_time
        }
        return state

    def _get_task_features(self):
        """获取任务特征 [x, y, duration, EST, scheduled]"""
        features = np.zeros((self.n_tasks, 5))
        features[:, 0:2] = self.task_coords
        features[:, 2] = self.task_durations
        features[:, 3] = self._compute_est()  # 最早开始时间
        features[:, 4] = self.task_states[:, 0]  # 是否已调度
        return features

    def _compute_est(self):
        """计算每个任务的最早开始时间(EST)"""
        est = np.full(self.n_tasks, self.current_time)
        # 已调度任务的EST是实际开始时间
        scheduled_mask = self.task_states[:, 0] == 1
        est[scheduled_mask] = self.task_states[scheduled_mask, 1]
        return est

    def get_available_actions(self):
        """获取当前可用动作（USV-Task对）"""
        available = []

        # 找到空闲的USV
        idle_usvs = np.where(self.usv_states[:, 3] <= self.current_time)[0]

        # 找到未调度的任务
        unscheduled_tasks = np.where(self.task_states[:, 0] == 0)[0]

        for usv_id in idle_usvs:
            usv_pos = self.usv_states[usv_id, 0:2]
            usv_battery = self.usv_states[usv_id, 2]

            for task_id in unscheduled_tasks:
                task_pos = self.task_coords[task_id]

                # 检查电量约束
                dist_to_task = np.linalg.norm(task_pos - usv_pos)
                dist_to_origin = np.linalg.norm(task_pos)
                total_dist = dist_to_task + dist_to_origin

                energy_needed = (total_dist / self.config.usv_speed *
                                 self.config.energy_cost_per_distance)

                if usv_battery >= energy_needed:
                    action_id = task_id * self.n_usvs + usv_id
                    available.append(action_id)

        return available

    def step(self, action: int):
        """执行动作"""
        # 解码动作
        task_id = action // self.n_usvs
        usv_id = action % self.n_usvs

        # 检查动作合法性
        if self.task_states[task_id, 0] == 1:
            return self._get_state(), -1000, False, {'error': 'task_scheduled'}

        if self.usv_states[usv_id, 3] > self.current_time:
            return self._get_state(), -1000, False, {'error': 'usv_busy'}

        # 计算移动时间和能耗
        usv_pos = self.usv_states[usv_id, 0:2]
        task_pos = self.task_coords[task_id]
        dist = np.linalg.norm(task_pos - usv_pos)

        travel_time = dist / self.config.usv_speed
        energy_cost = dist / self.config.usv_speed * self.config.energy_cost_per_distance

        # 更新状态
        old_makespan = self._compute_makespan()

        start_time = self.current_time + travel_time
        task_duration = self.task_durations[task_id]
        completion_time = start_time + task_duration

        # 更新任务状态
        self.task_states[task_id, 0] = 1  # 已调度
        self.task_states[task_id, 1] = start_time
        self.task_states[task_id, 2] = completion_time
        self.task_states[task_id, 3] = usv_id

        # 更新USV状态
        self.usv_states[usv_id, 0:2] = task_pos  # 新位置
        self.usv_states[usv_id, 2] -= energy_cost  # 消耗电量
        self.usv_states[usv_id, 3] = completion_time  # 忙碌到完成时间

        self.n_scheduled_tasks += 1

        # 如果电量不足返回原点，需要充电
        dist_to_origin = np.linalg.norm(task_pos)
        energy_to_origin = (dist_to_origin / self.config.usv_speed *
                            self.config.energy_cost_per_distance)

        if self.usv_states[usv_id, 2] < energy_to_origin * 1.2:  # 安全余量
            # 返回原点充电
            return_time = dist_to_origin / self.config.usv_speed
            self.usv_states[usv_id, 3] += return_time + self.config.charge_time
            self.usv_states[usv_id, 0:2] = [0, 0]
            self.usv_states[usv_id, 2] = self.config.battery_capacity

        # 更新当前时间到下一个决策点
        self.current_time = np.min(self.usv_states[:, 3])

        # 计算奖励（makespan的减少）
        new_makespan = self._compute_makespan()
        reward = old_makespan - new_makespan

        # 检查是否完成
        done = self.n_scheduled_tasks == self.n_tasks

        info = {
            'makespan': new_makespan,
            'n_scheduled': self.n_scheduled_tasks
        }

        return self._get_state(), reward, done, info

    def _compute_makespan(self):
        """计算当前makespan（最大完成时间）"""
        if self.n_scheduled_tasks == 0:
            # 估计初始makespan
            return np.sum(self.task_durations) / self.n_usvs + 500

        # 计算每个USV的完成时间
        usv_completion_times = []
        for usv_id in range(self.n_usvs):
            assigned_tasks = np.where(self.task_states[:, 3] == usv_id)[0]
            if len(assigned_tasks) > 0:
                max_completion = np.max(self.task_states[assigned_tasks, 2])
                # 加上返回原点的时间
                last_pos = self.usv_states[usv_id, 0:2]
                return_time = np.linalg.norm(last_pos) / self.config.usv_speed
                usv_completion_times.append(max_completion + return_time)
            else:
                usv_completion_times.append(0)

        return max(usv_completion_times) if usv_completion_times else self.current_time