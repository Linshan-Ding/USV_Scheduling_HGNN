# ============================================================================
# 2. env.py - 离散事件驱动仿真环境 (修正完工时间定义)
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
        self.task_durations = (
                                      self.fuzzy_times[:, 0] +
                                      2 * self.fuzzy_times[:, 1] +
                                      self.fuzzy_times[:, 2]
                              ) / 4.0

        self.action_space = spaces.Discrete(self.n_tasks * self.n_usvs)
        self.reset()

    def reset(self):
        """重置环境"""
        # USV状态 [x, y, battery, busy_until_time]
        self.usv_states = np.zeros((self.n_usvs, 4))
        self.usv_states[:, 2] = self.config.battery_capacity

        # 任务状态
        self.task_states = np.zeros((self.n_tasks, 4))
        self.task_states[:, 3] = -1

        self.current_time = 0.0
        self.n_scheduled_tasks = 0

        # 历史记录
        self.usv_history = {i: [] for i in range(self.n_usvs)}

        # 初始化 Makespan (包含返回时间)
        self.last_makespan = self._compute_makespan()

        return self._get_state()

    def _get_state(self):
        """获取当前状态"""
        state = {
            'usv_features': self._get_usv_features(),
            'task_features': self._get_task_features(),
            'task_coords': self.task_coords.copy(),
            'n_scheduled': self.n_scheduled_tasks,
            'current_time': self.current_time
        }
        return state

    def _get_usv_features(self):
        features = self.usv_states.copy()
        features[:, 3] = np.maximum(0, features[:, 3] - self.current_time)
        return features

    def _get_task_features(self):
        features = np.zeros((self.n_tasks, 5))
        features[:, 0:2] = self.task_coords
        features[:, 2] = self.task_durations
        est = self._compute_est()
        features[:, 3] = np.maximum(0, est - self.current_time)
        features[:, 4] = self.task_states[:, 0]
        return features

    def _compute_est(self):
        est = np.full(self.n_tasks, self.current_time)
        scheduled_mask = self.task_states[:, 0] == 1
        est[scheduled_mask] = self.task_states[scheduled_mask, 1]
        return est

    def get_available_actions(self):
        available = []
        idle_usvs = np.where(self.usv_states[:, 3] <= self.current_time + 1e-5)[0]
        unscheduled_tasks = np.where(self.task_states[:, 0] == 0)[0]

        if len(idle_usvs) == 0:
            return []

        for usv_id in idle_usvs:
            usv_pos = self.usv_states[usv_id, 0:2]
            usv_battery = self.usv_states[usv_id, 2]

            for task_id in unscheduled_tasks:
                task_pos = self.task_coords[task_id]
                dist_to_task = np.linalg.norm(task_pos - usv_pos)
                dist_to_origin = np.linalg.norm(task_pos)

                energy_to_task = (dist_to_task / self.config.usv_speed *
                                  self.config.energy_cost_per_distance)
                energy_return = (dist_to_origin / self.config.usv_speed *
                                 self.config.energy_cost_per_distance)

                if usv_battery >= energy_to_task + energy_return:
                    action_id = task_id * self.n_usvs + usv_id
                    available.append(action_id)
        return available

    def step(self, action: int):
        """执行动作"""
        task_id = action // self.n_usvs
        usv_id = action % self.n_usvs

        if self.task_states[task_id, 0] == 1:
            return self._get_state(), -10.0, True, {'error': 'task_scheduled'}

        # 1. 任务执行逻辑
        usv_pos = self.usv_states[usv_id, 0:2]
        task_pos = self.task_coords[task_id]

        dist = np.linalg.norm(task_pos - usv_pos)
        travel_time = dist / self.config.usv_speed
        energy_cost = dist * self.config.energy_cost_per_distance  # 注意：这里修正为距离*单位能耗

        move_start_time = self.current_time
        task_start_time = move_start_time + travel_time
        task_duration = self.task_durations[task_id]
        task_end_time = task_start_time + task_duration

        # 记录移动日志
        if travel_time > 0:
            self.usv_history[usv_id].append({
                'type': 'move',
                'start': move_start_time,
                'end': task_start_time,
                'info': 'Travel'
            })

        # 记录任务日志
        self.usv_history[usv_id].append({
            'type': 'task',
            'start': task_start_time,
            'end': task_end_time,
            'info': f'T{task_id}'
        })

        # 更新状态
        self.task_states[task_id, 0] = 1
        self.task_states[task_id, 1] = task_start_time
        self.task_states[task_id, 2] = task_end_time
        self.task_states[task_id, 3] = usv_id

        self.usv_states[usv_id, 0:2] = task_pos
        self.usv_states[usv_id, 2] -= energy_cost
        self.usv_states[usv_id, 3] = task_end_time

        self.n_scheduled_tasks += 1

        # 2. 充电逻辑检测 (中途充电)
        dist_to_origin = np.linalg.norm(task_pos)
        energy_to_origin = (dist_to_origin * self.config.energy_cost_per_distance)

        # 如果电量仅够回港（带一点缓冲），强制回港充电
        if self.usv_states[usv_id, 2] < energy_to_origin * 1.5:
            return_time = dist_to_origin / self.config.usv_speed
            return_start = task_end_time
            return_end = return_start + return_time
            charge_start = return_end
            charge_end = charge_start + self.config.charge_time

            self.usv_history[usv_id].append({
                'type': 'move',
                'start': return_start,
                'end': return_end,
                'info': 'Return(Charge)'
            })
            self.usv_history[usv_id].append({
                'type': 'charge',
                'start': charge_start,
                'end': charge_end,
                'info': 'Charge'
            })

            self.usv_states[usv_id, 3] = charge_end
            self.usv_states[usv_id, 0:2] = [0, 0]
            self.usv_states[usv_id, 2] = self.config.battery_capacity

        # 3. 检查是否结束 (所有任务调度完毕)
        done = self.n_scheduled_tasks == self.n_tasks

        # =========================================================
        # 核心修改：最终返航逻辑
        # =========================================================
        if done:
            # 遍历所有 USV，如果不在原点，强制添加一段“最终返航”行程
            for u in range(self.n_usvs):
                curr_pos = self.usv_states[u, 0:2]
                dist_home = np.linalg.norm(curr_pos)

                if dist_home > 1e-3:  # 如果不在原点
                    travel_home_time = dist_home / self.config.usv_speed
                    home_start = self.usv_states[u, 3]
                    home_end = home_start + travel_home_time

                    # 记录最终返航日志
                    self.usv_history[u].append({
                        'type': 'move',
                        'start': home_start,
                        'end': home_end,
                        'info': 'Final Return'
                    })

                    # 更新状态（确保 _compute_makespan 计算正确）
                    self.usv_states[u, 3] = home_end
                    self.usv_states[u, 0:2] = [0, 0]
                    # 最终返航不再扣减电量，假设电量足够（因为之前的get_available_actions已保证）

        # 4. 时间推进 (仅当未结束时)
        if not done:
            idle_usvs = np.where(self.usv_states[:, 3] <= self.current_time + 1e-5)[0]
            unscheduled_exists = np.any(self.task_states[:, 0] == 0)
            if len(idle_usvs) == 0 and unscheduled_exists:
                next_event_time = np.min(self.usv_states[:, 3])
                self.current_time = max(self.current_time, next_event_time)
        else:
            # 如果结束，当前时间推进到最后的完工时间
            self.current_time = self._compute_makespan()

        # 5. 计算奖励 (基于 Makespan 的差分)
        # 注意：此时 _compute_makespan 已经包含了最终返航时间
        current_makespan = self._compute_makespan()
        reward = (self.last_makespan - current_makespan) * 10.0
        self.last_makespan = current_makespan

        # 任务完成的小奖励
        reward += 1.0

        if done:
            reward += 20.0  # 完成所有任务的额外奖励

        info = {'makespan': current_makespan}
        return self._get_state(), reward, done, info

    def _compute_makespan(self):
        """
        计算当前的 Makespan。
        定义：所有 USV 完成任务并返回充电点的最晚时间。
        公式 = max_over_all_usvs( usv_busy_time + distance_to_depot / speed )
        """
        if self.n_scheduled_tasks == 0:
            return self._estimate_initial_makespan()

        finish_times = []
        for i in range(self.n_usvs):
            # 获取 USV 当前空闲时间
            busy_until = self.usv_states[i, 3]
            # 获取 USV 当前位置
            curr_pos = self.usv_states[i, 0:2]

            # 计算返回原点所需时间
            dist_to_home = np.linalg.norm(curr_pos)
            return_time = dist_to_home / self.config.usv_speed

            # 该 USV 的最终完工时间 = 忙碌结束 + 返航
            finish_times.append(busy_until + return_time)

        return max(finish_times)

    def _estimate_initial_makespan(self):
        # 估算包含往返的初始 Makespan
        total_task_time = np.sum(self.task_durations)
        # 假设平均每个任务都要往返（悲观估计）
        avg_dist = np.mean(np.linalg.norm(self.task_coords, axis=1))
        avg_travel = (avg_dist * 2 / self.config.usv_speed) * self.n_tasks

        return (total_task_time + avg_travel) / self.n_usvs