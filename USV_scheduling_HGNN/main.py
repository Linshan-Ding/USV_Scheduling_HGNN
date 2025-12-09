"""
完整的USV调度HGNN-DRL实现
包含所有必要的模块：环境、神经网络、PPO算法、实例生成器
"""

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


# ============================================================================
# 2. env.py - 离散事件驱动仿真环境
# ============================================================================
import gym
from gym import spaces


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


# ============================================================================
# 3. hgnn.py - 异质图神经网络模块
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNNLayer(nn.Module):
    """异质图神经网络层"""

    def __init__(self, hidden_dim=8, n_neighbors=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_neighbors = n_neighbors

        # USV节点编码
        self.usv_encoder = nn.Linear(4, hidden_dim)  # [x, y, battery, busy_time]

        # 任务节点编码
        self.task_encoder = nn.Linear(5, hidden_dim)  # [x, y, duration, EST, scheduled]

        # 边特征编码
        self.edge_encoder = nn.Linear(1, hidden_dim)  # [distance or duration]

        # 注意力机制参数
        self.attn_usv = nn.Linear(hidden_dim * 2, 1)
        self.attn_task = nn.Linear(hidden_dim * 2, 1)

        # 特征变换矩阵
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, usv_features, task_features, adj_matrix, edge_features):
        """
        前向传播

        Args:
            usv_features: [batch, n_usvs, 4]
            task_features: [batch, n_tasks, 5]
            adj_matrix: [batch, n_usvs, n_tasks] USV-Task邻接矩阵
            edge_features: [batch, n_usvs, n_tasks, 1] 边特征（距离）
        """
        batch_size = usv_features.size(0)
        n_usvs = usv_features.size(1)
        n_tasks = task_features.size(1)

        # ===== 阶段1: 更新USV节点嵌入 =====
        usv_embed = self.usv_encoder(usv_features)  # [batch, n_usvs, hidden]
        task_embed = self.task_encoder(task_features)  # [batch, n_tasks, hidden]

        # 计算USV对任务的注意力权重
        usv_query = self.W_Q(usv_embed).unsqueeze(2)  # [batch, n_usvs, 1, hidden]
        task_key = self.W_K(task_embed).unsqueeze(1)  # [batch, 1, n_tasks, hidden]

        # 注意力分数
        attn_scores = torch.tanh(usv_query + task_key)  # [batch, n_usvs, n_tasks, hidden]
        attn_scores = self.attn_usv(attn_scores).squeeze(-1)  # [batch, n_usvs, n_tasks]

        # 使用邻接矩阵mask
        attn_scores = attn_scores.masked_fill(adj_matrix == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, n_usvs, n_tasks]

        # 聚合任务特征到USV
        aggregated_task = torch.bmm(attn_weights, task_embed)  # [batch, n_usvs, hidden]

        # 更新USV嵌入（加上自注意力）
        usv_self_attn = torch.sigmoid(self.W_Q(usv_embed))
        usv_embed_updated = F.elu(usv_self_attn * usv_embed + aggregated_task)

        # ===== 阶段2: 更新任务节点嵌入 =====
        # 任务-任务邻接（n-neighborhood）
        task_query = self.W_V(task_embed).unsqueeze(2)  # [batch, n_tasks, 1, hidden]
        task_key_t = self.W_K(task_embed).unsqueeze(1)  # [batch, 1, n_tasks, hidden]

        # 计算任务间距离，找到n近邻
        task_coords = task_features[:, :, 0:2]  # [batch, n_tasks, 2]
        task_dist = torch.cdist(task_coords, task_coords)  # [batch, n_tasks, n_tasks]

        # 选择n个最近邻居
        _, nearest_indices = torch.topk(-task_dist, k=min(self.n_neighbors + 1, n_tasks), dim=-1)
        task_adj = torch.zeros(batch_size, n_tasks, n_tasks, device=task_features.device)
        for b in range(batch_size):
            for i in range(n_tasks):
                task_adj[b, i, nearest_indices[b, i, :]] = 1

        # 任务间注意力
        task_attn_scores = torch.tanh(task_query + task_key_t)
        task_attn_scores = self.attn_task(task_attn_scores).squeeze(-1)
        task_attn_scores = task_attn_scores.masked_fill(task_adj == 0, -1e9)
        task_attn_weights = F.softmax(task_attn_scores, dim=-1)

        # 聚合任务邻居
        aggregated_tasks = torch.bmm(task_attn_weights, task_embed)

        # 聚合USV特征到任务
        task_usv_query = self.W_V(task_embed).unsqueeze(1)  # [batch, 1, n_tasks, hidden]
        usv_key = self.W_Q(usv_embed_updated).unsqueeze(2)  # [batch, n_usvs, 1, hidden]

        task_usv_attn = torch.tanh(task_usv_query + usv_key)  # [batch, n_usvs, n_tasks, hidden]
        task_usv_attn = self.attn_task(task_usv_attn).squeeze(-1)  # [batch, n_usvs, n_tasks]
        task_usv_attn = F.softmax(task_usv_attn, dim=1)  # 在USV维度softmax

        aggregated_usv = torch.bmm(task_usv_attn.transpose(1, 2), usv_embed_updated)  # [batch, n_tasks, hidden]

        # 更新任务嵌入
        task_self_attn = torch.sigmoid(self.W_V(task_embed))
        task_embed_updated = F.elu(
            task_self_attn * task_embed + aggregated_tasks + aggregated_usv
        )

        return usv_embed_updated, task_embed_updated


class HGNN(nn.Module):
    """多层异质图神经网络"""

    def __init__(self, hidden_dim=8, n_layers=3, n_neighbors=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            HGNNLayer(hidden_dim, n_neighbors) for _ in range(n_layers)
        ])

    def forward(self, state_dict):
        """
        处理状态字典，返回嵌入

        Args:
            state_dict: 包含usv_features, task_features等的字典
        """
        usv_features = state_dict['usv_features']  # [batch, n_usvs, 4]
        task_features = state_dict['task_features']  # [batch, n_tasks, 5]

        batch_size = usv_features.size(0)
        n_usvs = usv_features.size(1)
        n_tasks = task_features.size(1)

        # 构建USV-Task邻接矩阵（基于可达性）
        adj_matrix = self._build_adjacency(state_dict)
        edge_features = self._compute_edge_features(state_dict)

        # 多层HGNN
        usv_embed = usv_features
        task_embed = task_features

        for layer in self.layers:
            usv_embed, task_embed = layer(
                usv_embed, task_embed, adj_matrix, edge_features
            )

        # 全局池化
        usv_global = torch.mean(usv_embed, dim=1)  # [batch, hidden]
        task_global = torch.mean(task_embed, dim=1)  # [batch, hidden]

        # 拼接全局嵌入
        global_embed = torch.cat([usv_global, task_global], dim=-1)  # [batch, 2*hidden]

        return {
            'usv_embed': usv_embed,
            'task_embed': task_embed,
            'global_embed': global_embed
        }

    def _build_adjacency(self, state_dict):
        """构建USV-Task邻接矩阵（可达性）"""
        usv_features = state_dict['usv_features']
        task_features = state_dict['task_features']

        batch_size = usv_features.size(0)
        n_usvs = usv_features.size(1)
        n_tasks = task_features.size(1)

        # 简化：所有未调度的任务对空闲的USV可达
        adj = torch.ones(batch_size, n_usvs, n_tasks, device=usv_features.device)

        # Mask已调度的任务
        scheduled = task_features[:, :, 4]  # [batch, n_tasks]
        adj = adj * (1 - scheduled.unsqueeze(1))  # [batch, n_usvs, n_tasks]

        return adj

    def _compute_edge_features(self, state_dict):
        """计算边特征（距离）"""
        usv_features = state_dict['usv_features']
        task_coords = state_dict['task_coords']

        batch_size = usv_features.size(0)
        n_usvs = usv_features.size(1)
        n_tasks = task_coords.size(1)

        usv_pos = usv_features[:, :, 0:2]  # [batch, n_usvs, 2]
        task_pos = task_coords  # [batch, n_tasks, 2]

        # 计算距离
        usv_pos_exp = usv_pos.unsqueeze(2)  # [batch, n_usvs, 1, 2]
        task_pos_exp = task_pos.unsqueeze(1)  # [batch, 1, n_tasks, 2]

        distances = torch.norm(usv_pos_exp - task_pos_exp, dim=-1, keepdim=True)

        return distances  # [batch, n_usvs, n_tasks, 1]


# ============================================================================
# 4. mlp.py - 多层感知机模块
# ============================================================================
class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# 5. ppo.py - PPO强化学习算法
# ============================================================================
class ActorCritic(nn.Module):
    """Actor-Critic网络"""

    def __init__(self, hidden_dim=8, hgnn_layers=3):
        super().__init__()

        self.hgnn = HGNN(hidden_dim=hidden_dim, n_layers=hgnn_layers, n_neighbors=2)

        # Actor网络：输出动作概率
        # 输入：[task_embed, usv_embed, global_embed]
        actor_input_dim = hidden_dim * 3
        self.actor = MLP(
            input_dim=actor_input_dim,
            hidden_dims=[64, 32],
            output_dim=1,
            activation='elu'
        )

        # Critic网络：输出状态价值
        critic_input_dim = hidden_dim * 2  # global_embed
        self.critic = MLP(
            input_dim=critic_input_dim,
            hidden_dims=[64, 32],
            output_dim=1,
            activation='elu'
        )

    def forward(self, state_dict, available_actions):
        """
        前向传播

        Args:
            state_dict: 状态字典
            available_actions: 可用动作列表 [(task_id, usv_id), ...]
        """
        # HGNN编码
        embeddings = self.hgnn(state_dict)

        usv_embed = embeddings['usv_embed']  # [batch, n_usvs, hidden]
        task_embed = embeddings['task_embed']  # [batch, n_tasks, hidden]
        global_embed = embeddings['global_embed']  # [batch, 2*hidden]

        # 计算每个可用动作的得分
        batch_size = usv_embed.size(0)
        n_actions = len(available_actions)

        action_scores = []
        for task_id, usv_id in available_actions:
            # 拼接task, usv, global嵌入
            task_emb = task_embed[:, task_id, :]  # [batch, hidden]
            usv_emb = usv_embed[:, usv_id, :]  # [batch, hidden]

            action_input = torch.cat([task_emb, usv_emb, global_embed], dim=-1)
            score = self.actor(action_input)  # [batch, 1]
            action_scores.append(score)

        if action_scores:
            action_scores = torch.cat(action_scores, dim=-1)  # [batch, n_actions]
            action_probs = F.softmax(action_scores, dim=-1)
        else:
            action_probs = torch.zeros(batch_size, 0)

        # 状态价值
        state_value = self.critic(global_embed)  # [batch, 1]

        return action_probs, state_value

    def get_action(self, state_dict, available_actions, deterministic=False):
        """选择动作"""
        with torch.no_grad():
            action_probs, state_value = self.forward(state_dict, available_actions)

            if len(available_actions) == 0:
                return None, None, None

            if deterministic:
                action_idx = torch.argmax(action_probs, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample().item()

            log_prob = torch.log(action_probs[0, action_idx] + 1e-8)

            return available_actions[action_idx], log_prob, state_value


class PPO:
    """PPO算法实现"""

    def __init__(self, hidden_dim=8, hgnn_layers=3, lr=3e-4,
                 gamma=1.0, epsilon=0.2, value_coef=0.5, entropy_coef=0.02):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = ActorCritic(hidden_dim, hgnn_layers).to(self.device)
        self.policy_old = ActorCritic(hidden_dim, hgnn_layers).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.buffer = []

    def select_action(self, env, state_dict, deterministic=False):
        """选择动作"""
        available_actions = env.get_available_actions()

        # 将动作ID转换为(task_id, usv_id)对
        action_pairs = [(a // env.n_usvs, a % env.n_usvs) for a in available_actions]

        # 准备state_dict为tensor
        state_tensor = self._prepare_state(state_dict)

        action_pair, log_prob, value = self.policy_old.get_action(
            state_tensor, action_pairs, deterministic
        )

        if action_pair is None:
            return None, None, None

        # 转换回动作ID
        task_id, usv_id = action_pair
        action_id = task_id * env.n_usvs + usv_id

        return action_id, log_prob, value

    def _prepare_state(self, state_dict):
        """将numpy状态转换为tensor"""
        return {
            'usv_features': torch.FloatTensor(state_dict['usv_features']).unsqueeze(0).to(self.device),
            'task_features': torch.FloatTensor(state_dict['task_features']).unsqueeze(0).to(self.device),
            'task_coords': torch.FloatTensor(state_dict['task_coords']).unsqueeze(0).to(self.device)
        }

    def store_transition(self, state, action, reward, log_prob, value):
        """存储转移"""
        self.buffer.append((state, action, reward, log_prob, value))

    def update(self, n_epochs=1):
        """更新策略"""
        if len(self.buffer) == 0:
            return {}

        # 计算折扣回报
        returns = []
        advantages = []
        R = 0

        for i in reversed(range(len(self.buffer))):
            _, _, reward, _, value = self.buffer[i]
            R = reward + self.gamma * R
            returns.insert(0, R)
            advantage = R - value.item()
            advantages.insert(0, advantage)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 更新策略
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0

        for epoch in range(n_epochs):
            for idx, (state, action, reward, old_log_prob, old_value) in enumerate(self.buffer):
                # 重新计算动作概率和价值
                state_tensor = self._prepare_state(state)

                # 获取可用动作
                task_id = action // len(state['usv_features'])
                usv_id = action % len(state['usv_features'])

                # 简化：假设动作仍然可用
                action_pairs = [(task_id, usv_id)]

                action_probs, value = self.policy.forward(state_tensor, action_pairs)

                if action_probs.size(-1) == 0:
                    continue

                log_prob = torch.log(action_probs[0, 0] + 1e-8)

                # PPO损失
                ratio = torch.exp(log_prob - old_log_prob.detach())
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2)

                # 价值损失
                critic_loss = F.mse_loss(value.squeeze(), returns[idx])

                # 熵损失（鼓励探索）
                entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()

                # 总损失
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空buffer
        self.buffer.clear()

        n_transitions = len(self.buffer) if self.buffer else 1

        return {
            'loss': total_loss / (n_epochs * n_transitions),
            'actor_loss': total_actor_loss / (n_epochs * n_transitions),
            'critic_loss': total_critic_loss / (n_epochs * n_transitions)
        }

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ============================================================================
# 6. train.py - 训练主程序
# ============================================================================
def train_hgnn_ppo(n_epochs=1000, test_every=10):
    """训练HGNN-PPO"""

    # 配置
    config = InstanceConfig(
        n_usvs=4,
        n_tasks=40,
        map_size=(800, 800),
        battery_capacity=400.0,
        usv_speed=5.0
    )

    # 初始化
    generator = InstanceGenerator(config)
    ppo = PPO(hidden_dim=8, hgnn_layers=3, lr=3e-4)

    # 训练循环
    train_rewards = []
    test_makespans = []

    print("开始训练 HGNN-PPO...")
    print(f"配置: {config.n_usvs} USVs, {config.n_tasks} 任务")

    for epoch in range(n_epochs):
        # 生成新实例
        instance = generator.generate(seed=epoch)
        env = USVSchedulingEnv(instance)

        # 训练episode
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, log_prob, value = ppo.select_action(env, state, deterministic=False)

            if action is None:
                break

            next_state, reward, done, info = env.step(action)

            ppo.store_transition(state, action, reward, log_prob, value)

            episode_reward += reward
            state = next_state
            steps += 1

            if steps > config.n_tasks * 2:  # 防止无限循环
                break

        # 更新策略
        if len(ppo.buffer) > 0:
            loss_info = ppo.update(n_epochs=1)

        train_rewards.append(episode_reward)

        # 测试
        if (epoch + 1) % test_every == 0:
            test_instance = generator.generate(seed=99999)
            test_env = USVSchedulingEnv(test_instance)
            test_makespan = test_policy(ppo, test_env)
            test_makespans.append(test_makespan)

            avg_reward = np.mean(train_rewards[-test_every:])
            print(f"Epoch {epoch + 1}/{n_epochs} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Test Makespan: {test_makespan:.2f}")

    return ppo, train_rewards, test_makespans


def test_policy(ppo, env, deterministic=True):
    """测试策略"""
    state = env.reset()
    done = False
    steps = 0

    while not done:
        action, _, _ = ppo.select_action(env, state, deterministic=deterministic)

        if action is None:
            break

        state, reward, done, info = env.step(action)
        steps += 1

        if steps > env.n_tasks * 2:
            break

    return info.get('makespan', float('inf'))


# ============================================================================
# 7. main.py - 主程序入口
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("USV调度 HGNN-DRL 系统")
    print("论文复现代码")
    print("=" * 60)

    # 训练
    trained_ppo, rewards, makespans = train_hgnn_ppo(
        n_epochs=200,
        test_every=10
    )

    print("\n训练完成！")
    print(f"最终测试 Makespan: {makespans[-1]:.2f}")

    # 保存模型
    trained_ppo.save("hgnn_ppo_model.pth")
    print("模型已保存至: hgnn_ppo_model.pth")

    # 可视化（如果有matplotlib）
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(rewards)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')

        ax2.plot(makespans)
        ax2.set_title('Test Makespan')
        ax2.set_xlabel('Test Episode')
        ax2.set_ylabel('Makespan')

        plt.tight_layout()
        plt.savefig('training_results.png')
        print("训练结果已保存至: training_results.png")
    except ImportError:
        print("未安装matplotlib，跳过可视化")