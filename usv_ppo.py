
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# 导入USV专用组件
from graph.usv_hgnn import USVBlock, TaskBlock
from usv_mlp import USVPairFeature, USVActionHead, USVValueHead

# 导入集中定义的常量
from utils.my_utils import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ACTION_LAYERS,
    DEFAULT_VALUE_LAYERS,
    MASK_FILL_VALUE,
    NUMERICAL_STABILITY_EPS,
    DEFAULT_MINIBATCH_SIZE
)


class Memory:
    """
    USV PPO经验存储类

    功能：存储rollout过程中的状态、动作、奖励等PPO训练所需数据
    适配USV特有的二维动作格式：(task_id, usv_id) - shape(2, batch_size)

    与PPO_model.py中Memory类的对应关系：
    - FJSP: actions(3,B) 存储(ope, mas, job)
    - USV: actions(2,B) 存储(task_id, usv_id)
    """

    def __init__(self):
        # === 核心PPO数据 ===
        self.states = []          # 环境状态列表
        self.logprobs = []        # 选中动作的对数概率
        self.rewards = []         # 即时奖励
        self.is_terminals = []    # 终止标记
        self.actions = []         # USV动作：(2, B)的(task_id, usv_id)

        # === HGNN相关数据（用于evaluate时的重计算） ===
        self.batch_idxes = []         # 批次索引
        self.task_features = []        # 任务特征张量
        self.usv_features = []        # USV特征张量
        self.task_usv_adj = []        # 任务-USV邻接矩阵
        self.task_task_adj = []        # 任务-任务邻接矩阵
        self.masks = []               # 动作掩码

        # === 辅助数据 ===
        self.values = []          # 状态价值函数V(s)
        self.entropies = []      # 策略熵

    def store(self, state, action, logprob, reward, done, value=None, entropy=None,
              task_features=None, usv_features=None, task_usv_adj=None,
              task_task_adj=None, mask=None):
        """
        存储单步经验

        参数：
            action: torch.Tensor, 形状(2, batch_size)，包含(task_ids, usv_ids)
            logprob: torch.Tensor, 动作的对数概率
            reward: torch.Tensor, 即时奖励
            done: torch.Tensor, 终止标记
            value: torch.Tensor, optional, 状态价值函数
            entropy: torch.Tensor, optional, 策略熵
            task_features: torch.Tensor, optional, 任务特征用于重计算
            usv_features: torch.Tensor, optional, USV特征用于重计算
            task_usv_adj: torch.Tensor, optional, 任务-USV邻接矩阵
            task_task_adj: torch.Tensor, optional, 任务-任务邻接矩阵
            mask: torch.Tensor, optional, 动作掩码
        """
        self.states.append(state)
        self.actions.append(action.detach().cpu())  # 确保在CPU上存储
        self.logprobs.append(logprob.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.is_terminals.append(done.detach().cpu())

        # 存储价值函数和熵（如果提供）
        if value is not None:
            self.values.append(value.detach().cpu())
        if entropy is not None:
            self.entropies.append(entropy.detach().cpu())

        # 存储HGNN相关数据（用于evaluate重计算）
        if task_features is not None:
            self.task_features.append(task_features.detach().cpu())
        if usv_features is not None:
            self.usv_features.append(usv_features.detach().cpu())
        if task_usv_adj is not None:
            self.task_usv_adj.append(task_usv_adj.detach().cpu())
        if task_task_adj is not None:
            self.task_task_adj.append(task_task_adj.detach().cpu())
        if mask is not None:
            self.masks.append(mask.detach().cpu())

    def clear(self):
        """清空经验存储"""
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.actions[:]
        del self.values[:]
        del self.entropies[:]
        del self.batch_idxes[:]
        del self.task_features[:]
        del self.usv_features[:]
        del self.task_usv_adj[:]
        del self.task_task_adj[:]
        del self.masks[:]

    def __len__(self):
        """返回存储的经验步数"""
        return len(self.states)


class MLPs(nn.Module):
    """
    USV MLP模块：整合策略头和价值头

    功能：将usv_mlp.py中的组件整合为统一的策略和价值网络
    对应PPO_model.py中MLPs类的职责，但适配USV场景

    === 核心方法 ===
    - policy_forward: 输出动作logits (B,N,M)
    - value_forward: 输出状态价值 (B,1)

    === 组件复用 ===
    - USVPairFeature: 成对特征构造器
    - USVActionHead: 策略头（继承MLPActor）
    - USVValueHead: 价值头（继承MLPCritic）
    """

    def __init__(self, d=DEFAULT_EMBEDDING_DIM, hidden_dim=DEFAULT_HIDDEN_DIM,
                 num_layers_action=DEFAULT_ACTION_LAYERS, num_layers_value=DEFAULT_VALUE_LAYERS):
        """
        初始化USV MLP模块

        参数：
            d (int): 嵌入维度，默认32（与usv_hgnn.py一致）
            hidden_dim (int): 隐藏层维度，默认128（等于4d）
            num_layers_action (int): 策略头层数，默认3
            num_layers_value (int): 价值头层数，默认2
        """
        super(MLPs, self).__init__()
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers_action = num_layers_action
        self.num_layers_value = num_layers_value

        # === 核心组件初始化 ===
        # 成对特征构造器：(B,N,d)+(B,M,d)+(B,2d) → (B,N,M,4d)
        self.pair_feature = USVPairFeature(d=d)

        # 策略头：(B,N,M,4d) → (B,N,M) logits
        self.action_head = USVActionHead(
            d=d,
            hidden_dim=hidden_dim,
            num_layers=num_layers_action
        )

        # 价值头：(B,2d) → (B,1) value
        self.value_head = USVValueHead(
            d=d,
            hidden_dim=hidden_dim,
            num_layers=num_layers_value
        )

    def policy_forward(self, task_emb, usv_emb, h_state):
        """
        策略前向传播：输出动作logits

        参数：
            task_emb (torch.Tensor): 任务嵌入，形状(B,N,d)
            usv_emb (torch.Tensor): USV嵌入，形状(B,M,d)
            h_state (torch.Tensor): 全局状态，形状(B,2d)

        返回：
            torch.Tensor: 动作logits，形状(B,N,M)
        """
        # 构造成对特征：(B,N,M,4d)
        pair_feat = self.pair_feature(task_emb, usv_emb, h_state)

        # 策略头输出：(B,N,M) logits
        logits = self.action_head(pair_feat)

        return logits

    def value_forward(self, h_state):
        """
        价值前向传播：输出状态价值

        参数：
            h_state (torch.Tensor): 全局状态，形状(B,2d)

        返回：
            torch.Tensor: 状态价值，形状(B,1)
        """
        return self.value_head(h_state)


class HGNNScheduler(nn.Module):
    """
    USV HGNN调度器：整合两阶段图神经网络处理

    功能：整合USVBlock和TaskBlock，实现USV场景的图神经网络编码
    对应PPO_model.py中HGNNScheduler类的职责

    === 两阶段处理 ===
    Stage-1: USVBlock实现"任务→USV"异构注意力更新
    Stage-2: TaskBlock实现η近邻任务聚合更新

    === 输出 ===
    task_emb: (B,N,d) 任务嵌入向量
    usv_emb: (B,M,d) USV嵌入向量
    h_state: (B,2d) 全局状态（任务和USV嵌入的平均池化拼接）
    """

    def __init__(self, d=DEFAULT_EMBEDDING_DIM, hidden_dim=DEFAULT_HIDDEN_DIM, num_layers=2, eta=3,
                 num_head=1, feat_drop=0.1, attn_drop=0.1,
                 negative_slope=0.2, activation=F.elu):
        """
        初始化USV HGNN调度器

        参数：
            d (int): 嵌入维度，默认32
            hidden_dim (int): 隐藏层维度，默认128
            num_layers (int): HGNN层数，默认2
            eta (int): 任务近邻数量，默认3
            num_head (int): 注意力头数，默认1
            feat_drop (float): 特征dropout率，默认0.1
            attn_drop (float): 注意力dropout率，默认0.1
            negative_slope (float): LeakyReLU负斜率，默认0.2
            activation: 激活函数，默认F.elu
        """
        super(HGNNScheduler, self).__init__()
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eta = eta
        self.num_head = num_head
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.activation = activation

        # === HGNN层初始化 ===
        # 每层包含USVBlock和TaskBlock
        self.usv_blocks = nn.ModuleList()
        self.task_blocks = nn.ModuleList()

        for layer in range(num_layers):
            # Stage-1: USVBlock实现"任务→USV"注意力
            usv_block = USVBlock(
                in_dims=(d, d),  # (任务特征维度d, USV特征维度d) - 保持原始设计
                out_dim=d,
                num_head=num_head,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                activation=activation
            )

            # Stage-2: TaskBlock实现η近邻任务聚合
            task_block = TaskBlock(
                in_dim=d,
                out_dim=d,
                hidden_dim=hidden_dim,
                num_head=num_head,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope
            )

            self.usv_blocks.append(usv_block)
            self.task_blocks.append(task_block)

    def encode(self, task_features, usv_features, task_usv_adj, task_task_adj):
        """
        HGNN编码：将原始特征编码为嵌入向量

        参数：
            task_features (torch.Tensor): 任务特征，形状(B,N,d_task)
            usv_features (torch.Tensor): USV特征，形状(B,M,d_usv)
            task_usv_adj (torch.Tensor): 任务-USV邻接矩阵，形状(B,N,M)
            task_task_adj (torch.Tensor): 任务-任务邻接矩阵，形状(B,N,N)

        返回：
            tuple: (task_emb(B,N,d), usv_emb(B,M,d), h_state(B,2d))
        """
        # === 维度验证 ===
        B, N, d_task = task_features.shape
        B_m, M, d_usv = usv_features.shape
        B_adj1, N_adj1, M_adj = task_usv_adj.shape
        B_adj2, N_adj2, N_adj2_check = task_task_adj.shape

        if B != B_m or B != B_adj1 or B != B_adj2:
            raise ValueError(f"批次大小不一致: task={B}, usv={B_m}, adj1={B_adj1}, adj2={B_adj2}")
        if N != N_adj1 or N != N_adj2:
            raise ValueError(f"任务数量不一致: features={N}, adj1={N_adj1}, adj2={N_adj2}")
        if M != M_adj:
            raise ValueError(f"USV数量不一致: features={M}, adj={M_adj}")

        # === 特征维度投影（如果输入维度不是d） ===
        # 确保输入特征维度等于嵌入维度d
        if d_task != self.d:
            if not hasattr(self, 'task_proj'):
                self.task_proj = nn.Linear(d_task, self.d).to(task_features.device)
            task_features = self.task_proj(task_features)

        if d_usv != self.d:
            # 强制重新创建投影层以适应新的USV特征维度
            self.usv_proj = nn.Linear(d_usv, self.d).to(usv_features.device)
            usv_features = self.usv_proj(usv_features)

        # === L层HGNN迭代 ===
        h_task = task_features  # (B,N,d)
        h_usv = usv_features   # (B,M,d)

        # 边特征：执行时间（从任务-USV邻接矩阵推导，或使用常数1）
        # 这里简化处理，假设边特征包含在邻接矩阵中
        edge_feat = torch.ones_like(task_usv_adj, dtype=torch.float32)  # (B,N,M)

        for layer in range(self.num_layers):
            # === Stage-1: USV节点更新 ===
            # USVBlock: 任务特征 + USV特征 + 边特征 + 邻接矩阵 → 新USV嵌入
            h_usv = self.usv_blocks[layer](h_task, h_usv, edge_feat, task_usv_adj)

            # === Stage-2: 任务节点更新 ===
            # TaskBlock: 任务特征 + 任务-任务邻接矩阵 → 新任务嵌入
            h_task = self.task_blocks[layer](h_task, task_task_adj)

        # === 全局状态构建 ===
        # 分别对任务和USV嵌入进行平均池化，然后拼接
        task_pooled = h_task.mean(dim=1)  # (B,d)
        usv_pooled = h_usv.mean(dim=1)   # (B,d)
        h_state = torch.cat([task_pooled, usv_pooled], dim=-1)  # (B,2d)

        return h_task, h_usv, h_state


class PPO:
    """
    USV PPO算法主类：完整的PPO-Clip强化学习算法实现

    功能：实现完整的PPO训练流程，包括采样、评估、更新
    适配USV场景的二维动作格式和环境接口

    === 核心方法 ===
    - select_action: 动作选择（采样/贪心）
    - rollout: 与环境交互收集轨迹
    - compute_gae: GAE优势估计
    - evaluate: 批量评估策略
    - update: PPO-Clip参数更新

    === 超参数配置 ===
    - gamma=1.0: 折扣因子（makespan目标等价）
    - lam=0.95: GAE λ参数
    - clip_eps=0.2: PPO裁剪范围
    - vf_coef=0.5: 价值函数损失系数
    - ent_coef=0.01: 熵正则化系数
    """

    def __init__(self, model_paras, train_paras, num_envs=None):
        """
        初始化PPO算法

        参数：
            model_paras (dict): 模型参数配置
            train_paras (dict): 训练参数配置
            num_envs (int): 并行环境数量
        """
        # === 基本配置 ===
        self.lr = train_paras["lr"]                      # 学习率
        self.betas = train_paras.get("betas", (0.9, 0.999))  # Adam优化器参数
        self.gamma = train_paras["gamma"]                  # 折扣因子
        self.lam = train_paras.get("lam", 0.95)         # GAE λ参数
        self.eps_clip = train_paras["eps_clip"]            # PPO裁剪范围
        self.K_epochs = train_paras["K_epochs"]            # 更新轮数
        self.A_coeff = train_paras.get("A_coeff", 1.0)    # 策略损失系数
        self.vf_coeff = train_paras.get("vf_coeff", 0.5)  # 价值损失系数
        self.entropy_coeff = train_paras.get("entropy_coeff", 0.01)  # 熵系数
        self.num_envs = num_envs                         # 并行环境数
        self.device = model_paras["device"]                 # 计算设备

        # === 模型组件初始化 ===
        # HGNN调度器：两阶段图神经网络处理
        self.hgnn_scheduler = HGNNScheduler(
            d=model_paras.get("d", DEFAULT_EMBEDDING_DIM),
            hidden_dim=model_paras.get("hidden_dim", DEFAULT_HIDDEN_DIM),
            num_layers=model_paras.get("num_layers", 2),
            eta=model_paras.get("eta", 3)
        ).to(self.device)

        # MLP模块：策略头和价值头
        self.mlps = MLPs(
            d=model_paras.get("d", DEFAULT_EMBEDDING_DIM),
            hidden_dim=model_paras.get("hidden_dim", DEFAULT_HIDDEN_DIM),
            num_layers_action=model_paras.get("num_layers_action", DEFAULT_ACTION_LAYERS),
            num_layers_value=model_paras.get("num_layers_value", DEFAULT_VALUE_LAYERS)
        ).to(self.device)

        # === 优化器 ===
        self.optimizer = torch.optim.Adam(
            list(self.hgnn_scheduler.parameters()) + list(self.mlps.parameters()),
            lr=self.lr,
            betas=self.betas
        )

        # === 损失函数 ===
        self.MseLoss = nn.MSELoss()

        # === 旧策略备份 ===
        self.hgnn_scheduler_old = copy.deepcopy(self.hgnn_scheduler)
        self.mlps_old = copy.deepcopy(self.mlps)
        self.hgnn_scheduler_old.load_state_dict(self.hgnn_scheduler.state_dict())
        self.mlps_old.load_state_dict(self.mlps.state_dict())

    def _prepare_policy_inputs(self, state):
        """
        从环境状态中提取策略网络所需的张量，并构建对应的动作掩码。

        返回：
            batch_idxes: 当前批次索引
            task_features: (B,N,d_task) 任务特征
            usv_features_enhanced: (B,M,d_usv) USV特征（含充电增强）
            task_usv_adj: (B,N,M) 任务-USV邻接
            task_task_adj: (B,N,N) 任务-任务邻接
            mask: (B,N或N+1,M) 动作掩码
        """
        batch_idxes = state.batch_idxes
        task_features = state.feat_tasks_batch[batch_idxes].transpose(1, 2)
        usv_features = state.feat_usvs_batch[batch_idxes].transpose(1, 2)
        task_usv_adj = state.task_usv_adj_dynamic_batch[batch_idxes]

        if hasattr(state, 'task_task_adj_batch') and state.task_task_adj_batch is not None:
            task_task_adj = state.task_task_adj_batch[batch_idxes]
        else:
            B, N = len(batch_idxes), task_features.size(1)
            device = task_features.device
            task_task_adj = self._ensure_task_task_adj(B, N, device, adjacency_strategy='default')

        # 彻底修复：强制统一使用增强掩码，确保与USVActionHead的N+1输出维度一致
        # USVActionHead设计为N+1动作空间，必须始终使用增强掩码
        charging_manager = getattr(state, 'charging_manager', None)
        if charging_manager is not None:
            usv_features_enhanced = self._enhance_usv_features(
                usv_features, charging_manager, state.time_batch[batch_idxes]
            )
        else:
            # 如果charging_manager确实缺失，使用原始USV特征
            usv_features_enhanced = usv_features

        # 始终使用增强掩码：确保与USVActionHead的N+1输出维度匹配
        if charging_manager is not None:
            mask = self._build_enhanced_action_mask(state, batch_idxes, charging_manager)
        else:
            # 创建默认charging_manager以确保维度一致性
            try:
                from env.charging_station_manager import ChargingStationManager
                default_charging_manager = ChargingStationManager(location=(0.0, 0.0), max_concurrent_usvs=float('inf'))
                mask = self._build_enhanced_action_mask(state, batch_idxes, default_charging_manager)
            except ImportError as e:
                print(f"警告：无法导入ChargingStationManager: {e}")
                # 使用简单的action mask作为fallback
                mask = self._build_basic_action_mask(state, batch_idxes)

        return batch_idxes, task_features, usv_features_enhanced, task_usv_adj, task_task_adj, mask

    def select_action(self, state, deterministic=False):
        """
        动作选择：采样或贪心选择

        参数：
            state: USV环境状态对象
            deterministic (bool): 是否使用贪心策略

        返回：
            torch.Tensor: 动作对，形状(2, batch_size)，包含(task_ids, usv_ids)
            torch.Tensor: 动作对数概率
            torch.Tensor: 状态价值
        """
        # === 提取状态特征 ===
        (batch_idxes, task_features, usv_features_enhanced,
         task_usv_adj, task_task_adj, mask) = self._prepare_policy_inputs(state)

        # === HGNN编码 ===
        task_emb, usv_emb, h_state = self.hgnn_scheduler.encode(
            task_features, usv_features_enhanced, task_usv_adj, task_task_adj
        )

        # === 策略前向传播 + 统一分布构建 ===
        logits, _, dist = self._build_action_distribution(
            task_emb, usv_emb, h_state, mask
        )

        B, _, M = logits.shape

        if deterministic:
            flat_action_ids = torch.argmax(dist.logits, dim=1)
        else:
            flat_action_ids = dist.sample()

        # === 动作解码 ===
        task_ids, usv_ids = self._decode_index(flat_action_ids, M)
        actions = torch.stack([task_ids, usv_ids], dim=0)  # (2,B)

        # === 计算对数概率（贪心模式下为None） ===
        if deterministic:
            logprobs = None
        else:
            logprobs = self._logprob_of_actions(dist, task_ids, usv_ids, M)

        # === 动作解释 ===
        # 如果使用充电站掩码（N+1动作空间）：
        # - task_ids < N-1: 分配USV到对应任务
        # - task_ids == N-1: 将USV分配到充电站（充电动作）
        # 如果使用标准掩码（N动作空间）：
        # - task_ids < N: 分配USV到对应任务

        # === 计算状态价值 ===
        values = self.mlps.value_forward(h_state).squeeze(-1)  # (B,)

        return actions, logprobs, values

    def _ensure_task_task_adj(self, batch_size, num_tasks, device, adjacency_strategy='default'):
        """
        统一的任务-任务邻接矩阵兜底逻辑

        参数：
            batch_size (int): 批次大小
            num_tasks (int): 任务数量
            device (torch.device): 计算设备
            adjacency_strategy (str): 邻接策略，支持：
                - 'default': 全连接图（去除自环）
                - 'ring': 环形连接图
                - 'sparse': 稀疏连接图（每个节点最多连接3个邻居）

        返回：
            torch.Tensor: 任务-任务邻接矩阵，形状(B,N,N)，去除对角线
        """
        if adjacency_strategy == 'default':
            # 全连接图（去除自环）
            task_task_adj = torch.ones(batch_size, num_tasks, num_tasks, dtype=torch.float32, device=device)
            eye = torch.eye(num_tasks, dtype=torch.float32, device=device).unsqueeze(0)
            task_task_adj = task_task_adj - eye

        elif adjacency_strategy == 'ring':
            # 环形连接图
            task_task_adj = torch.zeros(batch_size, num_tasks, num_tasks, dtype=torch.float32, device=device)
            for i in range(num_tasks):
                next_node = (i + 1) % num_tasks
                prev_node = (i - 1) % num_tasks
                task_task_adj[:, i, next_node] = 1.0
                task_task_adj[:, i, prev_node] = 1.0

        elif adjacency_strategy == 'sparse':
            # 稀疏连接图（每个节点最多连接3个邻居）
            task_task_adj = torch.zeros(batch_size, num_tasks, num_tasks, dtype=torch.float32, device=device)
            for i in range(num_tasks):
                # 连接到接下来的几个任务（循环连接）
                for offset in range(1, min(4, num_tasks)):  # 最多连接3个邻居
                    j = (i + offset) % num_tasks
                    task_task_adj[:, i, j] = 1.0
                    task_task_adj[:, j, i] = 1.0  # 双向连接
        else:
            raise ValueError(f"不支持的邻接策略: {adjacency_strategy}")

        return task_task_adj

    def _build_base_masks(self, state, batch_idxes):
        """
        构建通用掩码骨架：任务/USV合法性与邻接信息。
        返回：
            task_mask_base (B,N): 可调度任务标记
            idle_usvs (B,M): 空闲USV标记
            task_usv_adj (B,N,M): 任务-USV邻接矩阵（布尔）
        """
        device = self.device
        B = len(batch_idxes)
        N = state.feat_tasks_batch.size(2)
        M = state.feat_usvs_batch.size(2)

        task_status = state.feat_tasks_batch[batch_idxes, 0, :]
        completed_tasks = (task_status == 2)
        processing_tasks = (task_status == 1)

        task_mask_base = torch.ones(B, N, dtype=torch.bool, device=device)
        task_mask_base[completed_tasks] = False
        task_mask_base[processing_tasks] = False

        usv_status = state.feat_usvs_batch[batch_idxes, 0, :]
        idle_usvs = (usv_status == 1)

        task_usv_adj = state.task_usv_adj_dynamic_batch[batch_idxes].bool()

        return task_mask_base, idle_usvs, task_usv_adj

    def _ensure_has_legal_action(self, mask):
        """保证每个批次至少存在一个合法动作，若无则放宽约束以避免训练崩溃。"""
        has_legal_action = mask.any(dim=-1).any(dim=-1)
        for b in range(mask.size(0)):
            if not has_legal_action[b]:
                print(f"警告：批次{b}无合法动作，使用紧急兜底策略")
                mask[b] = True
        return mask

    def _flat_index(self, task_ids, usv_ids, num_usvs):
        """二维索引编码为一维：task_id * M + usv_id。"""
        return task_ids * num_usvs + usv_ids

    def _decode_index(self, flat_ids, num_usvs):
        """将一维索引还原为(task_id, usv_id)。"""
        # 修复：添加索引范围验证，防止解码出无效索引
        if not isinstance(flat_ids, torch.Tensor):
            flat_ids = torch.tensor(flat_ids, dtype=torch.long)

        # 确保索引非负
        flat_ids = torch.clamp(flat_ids, min=0)

        task_ids = flat_ids // num_usvs
        usv_ids = flat_ids % num_usvs
        return task_ids, usv_ids

    def _logprob_of_actions(self, dist, task_ids, usv_ids, num_usvs):
        """在给定分布下计算指定动作的对数概率。"""
        flat_ids = self._flat_index(task_ids, usv_ids, num_usvs)
        return dist.log_prob(flat_ids)

    def _build_action_distribution(self, task_emb, usv_emb, h_state, mask, mlps=None):
        """
        统一构建动作分布：应用掩码、softmax，并返回离散分布对象。

        返回：
            logits: 原始策略 logits，形状与 mask 一致
            probs: 归一化后的概率，形状与 mask 一致
            dist: torch.distributions.Categorical，展平后的一维分布
        """
        if mask is None:
            raise ValueError("构建动作分布需要显式提供掩码")

        model = mlps or self.mlps
        logits = model.policy_forward(task_emb, usv_emb, h_state)

        if logits.shape != mask.shape:
            raise ValueError(f"logits 和掩码形状不一致: {logits.shape} vs {mask.shape}")

        mask_bool = mask.bool()
        masked_logits = logits.masked_fill(~mask_bool, MASK_FILL_VALUE)

        flat_logits = masked_logits.view(masked_logits.size(0), -1)
        dist = Categorical(logits=flat_logits)
        probs = dist.probs.view_as(mask)

        return logits, probs, dist

    def _build_standard_action_mask(self, state, batch_idxes):
        """
        构建标准动作掩码：标记合法的(任务, USV)组合。
        返回形状(B,N,M)，True表示合法。
        """
        task_mask_base, idle_usvs, task_usv_adj = self._build_base_masks(state, batch_idxes)

        mask = task_mask_base.unsqueeze(-1).expand_as(task_usv_adj)
        mask = mask & idle_usvs.unsqueeze(1)
        mask = mask & task_usv_adj

        return self._ensure_has_legal_action(mask)

    def _build_enhanced_action_mask(self, state, batch_idxes, charging_manager):
        """
        构建包含充电动作的掩码：任务掩码 + 充电站掩码。
        返回形状(B,N+1,M)。
        """
        task_mask_base, idle_usvs, task_usv_adj = self._build_base_masks(state, batch_idxes)

        task_mask = task_mask_base.unsqueeze(-1).expand_as(task_usv_adj)
        task_mask = task_mask & task_usv_adj
        task_mask = task_mask & idle_usvs.unsqueeze(1)

        charging_mask = idle_usvs.unsqueeze(1)

        full_mask = torch.cat([task_mask, charging_mask], dim=1)

        return self._ensure_has_legal_action(full_mask)
    def rollout(self, env, horizon):
        """
        与环境交互收集轨迹

        参数：
            env: USV环境
            horizon (int): 最大步长

        返回：
            Memory: 存储的经验数据
        """
        memory = Memory()
        env.reset()
        state = env.state

        for step in range(horizon):
            # 选择动作
            actions, logprobs, values = self.select_action(state, deterministic=False)

            # 环境交互
            observations, _, _, _, info = env.step(actions)
            next_state = env.state
            rewards = torch.from_numpy(np.asarray(info["batch_rewards"], dtype=np.float64)).to(torch.float32)
            dones_np = np.asarray(info["batch_terminated"], dtype=bool) | np.asarray(info["batch_truncated"], dtype=bool)
            dones = torch.from_numpy(dones_np.astype(np.bool_))
            # 存储经验（提取HGNN数据用于后续evaluate）
            batch_idxes = state.batch_idxes
            task_features = state.feat_tasks_batch[batch_idxes]
            usv_features = state.feat_usvs_batch[batch_idxes]
            task_usv_adj = state.task_usv_adj_dynamic_batch[batch_idxes]
            # 检查task_task_adj_batch是否存在
            if hasattr(state, 'task_task_adj_batch') and state.task_task_adj_batch is not None:
                task_task_adj = state.task_task_adj_batch[batch_idxes]
            else:
                # 如果没有task_task_adj_batch，使用统一的默认邻接逻辑
                B, N = len(batch_idxes), task_features.size(1)
                device = task_features.device
                task_task_adj = self._ensure_task_task_adj(B, N, device, adjacency_strategy='default')
            # 构建mask用于存储（与select_action相同的逻辑）
            # 彻底修复：强制统一使用增强掩码，确保与USVActionHead的N+1输出维度一致
            charging_manager = getattr(state, 'charging_manager', None)
            if charging_manager is not None:
                mask = self._build_enhanced_action_mask(state, batch_idxes, charging_manager)
            else:
                # 创建默认charging_manager以确保维度一致性
                try:
                    from env.charging_station_manager import ChargingStationManager
                    default_charging_manager = ChargingStationManager(location=(0.0, 0.0), max_concurrent_usvs=float('inf'))
                    mask = self._build_enhanced_action_mask(state, batch_idxes, default_charging_manager)
                except ImportError as e:
                    print(f"警告：无法导入ChargingStationManager: {e}")
                    # 使用简单的action mask作为fallback
                    mask = self._build_basic_action_mask(state, batch_idxes)

            memory.store(
                state=state,
                action=actions,
                logprob=logprobs,
                reward=rewards,
                done=dones,
                value=values,
                task_features=task_features,
                usv_features=usv_features,
                task_usv_adj=task_usv_adj,
                task_task_adj=task_task_adj,
                mask=mask
            )

            state = next_state

            # 检查是否所有批次都结束
            if dones.all():
                break

        return memory

    def compute_gae(self, rewards, values, dones):
        """
        GAE优势估计：计算优势函数和回报

        参数：
            rewards (list): 奖励序列，每个元素为(B,)张量
            values (list): 价值函数序列，每个元素为(B,)张量
            dones (list): 终止标记序列，每个元素为(B,)张量

        返回：
            tuple: (advantages, returns) 优势函数和回报
        """
        # 转换为tensor并处理维度
        T = len(rewards)  # 序列长度
        if T == 0:
            return torch.tensor([]), torch.tensor([])

        # 堆叠为 (T, B) 形状
        rewards_tensor = torch.stack(rewards, dim=0)      # (T,B)
        values_tensor = torch.stack(values, dim=0)        # (T,B)
        dones_tensor = torch.stack(dones, dim=0).float()  # (T,B)

        # 计算GAE
        advantages = torch.zeros_like(rewards_tensor)  # (T,B)
        returns = torch.zeros_like(rewards_tensor)       # (T,B)

        # 最后一时间步的价值（用于bootstrap）
        next_value = values_tensor[-1] * (1 - dones_tensor[-1])

        # 从后向前计算
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0  # 最后一个时间步没有next_value
            else:
                next_value = values_tensor[t + 1] * (1 - dones_tensor[t])

            delta = rewards_tensor[t] + self.gamma * next_value - values_tensor[t]
            gae = delta + self.gamma * self.lam * (1 - dones_tensor[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values_tensor[t]

        return advantages, returns

    def evaluate(self, memory):
        """
        批量评估策略：计算旧策略下的logprob、熵、价值

        参数：
            memory (Memory): 存储的经验数据

        返回：
            tuple: (old_logprobs, state_values, dist_entropies)
        """
        if len(memory) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        # === 提取存储的数据 ===
        # 拉平所有时间步和批次的数据
        all_task_features = []
        all_usv_features = []
        all_task_usv_adj = []
        all_task_task_adj = []
        all_masks = []
        all_actions = []

        for t in range(len(memory)):
            if (t < len(memory.task_features) and t < len(memory.usv_features) and
                t < len(memory.task_usv_adj) and t < len(memory.task_task_adj) and
                t < len(memory.masks) and t < len(memory.actions)):

                # 获取当前时间步的数据
                task_features = memory.task_features[t]    # (B,N,6)
                usv_features = memory.usv_features[t]      # (B,M,5)
                task_usv_adj = memory.task_usv_adj[t]      # (B,N,M)
                task_task_adj = memory.task_task_adj[t]      # (B,N,N)
                masks = memory.masks[t]                      # (B,N,M)
                actions = memory.actions[t]                    # (2,B)

                # 移动到设备
                task_features = task_features.to(self.device)
                usv_features = usv_features.to(self.device)
                task_usv_adj = task_usv_adj.to(self.device)
                task_task_adj = task_task_adj.to(self.device)
                masks = masks.to(self.device)
                actions = actions.to(self.device)

                # 存储数据
                all_task_features.append(task_features)
                all_usv_features.append(usv_features)
                all_task_usv_adj.append(task_usv_adj)
                all_task_task_adj.append(task_task_adj)
                all_masks.append(masks)
                all_actions.append(actions)

        if not all_task_features:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        # === 堆叠所有数据 ===
        all_task_features = torch.cat(all_task_features, dim=0)    # (T*B,N,6)
        all_usv_features = torch.cat(all_usv_features, dim=0)      # (T*B,M,5)
        all_task_usv_adj = torch.cat(all_task_usv_adj, dim=0)      # (T*B,N,M)
        all_task_task_adj = torch.cat(all_task_task_adj, dim=0)      # (T*B,N,N)
        all_masks = torch.cat(all_masks, dim=0)                        # (T*B,N,M)
        all_actions = torch.cat(all_actions, dim=-1)                   # (2,T*B)

        # === 使用旧策略进行前向传播 ===
        with torch.no_grad():
            # HGNN编码
            task_emb, usv_emb, h_state = self.hgnn_scheduler_old.encode(
                all_task_features, all_usv_features,
                all_task_usv_adj, all_task_task_adj
            )

            # 策略前向传播
            logits, _, dist = self._build_action_distribution(
                task_emb, usv_emb, h_state, all_masks, mlps=self.mlps_old
            )

            # 价值前向传播
            state_values = self.mlps_old.value_forward(h_state).squeeze(-1)  # (T*B,)

            # === 计算logprob和熵 ===
            TB, _, M = logits.shape
            task_ids = all_actions[0]  # (T*B,)
            usv_ids = all_actions[1]   # (T*B,)

            old_logprobs = self._logprob_of_actions(dist, task_ids, usv_ids, M)
            dist_entropy = dist.entropy()

        return old_logprobs, state_values, dist_entropy

    def recompute_policy_logprobs(self, batch_indices, memory, states_batch):
        """
        重新计算新策略下的logprob

        参数：
            batch_indices (torch.Tensor): minibatch的索引
            memory (Memory): 存储的经验
            states_batch (list): 对应的状态列表

        返回：
            torch.Tensor: 新策略的logprob，形状与batch_indices一致
        """
        # 提取对应的状态和动作
        selected_states = [states_batch[i] for i in batch_indices]
        selected_actions = [memory.actions[i] for i in batch_indices]

        new_logprobs_list = []
        for state, action in zip(selected_states, selected_actions):
            (batch_idxes, task_features, usv_features_enhanced,
             task_usv_adj, task_task_adj, mask) = self._prepare_policy_inputs(state)

            task_emb, usv_emb, h_state = self.hgnn_scheduler.encode(
                task_features, usv_features_enhanced, task_usv_adj, task_task_adj
            )

            _, _, dist = self._build_action_distribution(task_emb, usv_emb, h_state, mask)
            M = mask.size(-1)

            action = action.to(self.device)
            task_ids = action[0].long()
            usv_ids = action[1].long()

            logprobs = self._logprob_of_actions(dist, task_ids, usv_ids, M)
            new_logprobs_list.append(logprobs)

        new_logprobs = torch.cat(new_logprobs_list, dim=0)

        # 确保返回的logprob数量与batch_indices一致
        assert len(new_logprobs) == len(batch_indices), \
            f"新策略logprob数量{len(new_logprobs)}与batch_indices数量{len(batch_indices)}不一致"

        return new_logprobs

    def update(self, optimizer, memory):
        """
        PPO参数更新：PPO-Clip损失函数

        参数：
            optimizer: 优化器
            memory (Memory): 存储的经验数据

        返回：
            dict: 损失统计信息
        """
        if len(memory) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # === 提取奖励和done标记 ===
        rewards = memory.rewards  # list of (B,) tensors
        values = memory.values if memory.values else [torch.zeros_like(r) for r in rewards]
        dones = memory.is_terminals  # list of (B,) tensors

        # === 计算GAE优势和回报 ===
        advantages, returns = self.compute_gae(rewards, values, dones)

        # === 展平所有数据 ===
        all_rewards = torch.cat(rewards, dim=0)      # (T*B,)
        all_values = torch.cat(values, dim=0)        # (T*B,)
        all_advantages = advantages.view(-1)            # (T*B,)
        all_returns = returns.view(-1)                 # (T*B,)

        # === 标准化优势函数 ===
        if len(all_advantages) > 1:
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + NUMERICAL_STABILITY_EPS)

        # === 准备状态数据 ===
        states_batch = memory.states  # 提取状态列表

        # === 评估旧策略 ===
        old_logprobs, state_values, dist_entropies = self.evaluate(memory)

        if len(old_logprobs) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # 移动到设备
        all_advantages = all_advantages.to(self.device)
        all_returns = all_returns.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        state_values = state_values.to(self.device)
        dist_entropies = dist_entropies.to(self.device)

        # === PPO更新循环 ===
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # 确定batch size
        batch_size = len(old_logprobs)
        minibatch_size = min(DEFAULT_MINIBATCH_SIZE, batch_size)  # 默认minibatch size

        for epoch in range(self.K_epochs):
            # 随机打乱数据
            indices = torch.randperm(batch_size)

            for i in range(0, batch_size, minibatch_size):
                batch_indices = indices[i:i + minibatch_size]
                if len(batch_indices) == 0:
                    continue

                # === 使用新策略计算 ===
                # 重新计算新策略下的logprob，确保算法正确性
                new_logprobs = self.recompute_policy_logprobs(batch_indices, memory, states_batch)
                batch_values = state_values[batch_indices]
                batch_advantages = all_advantages[batch_indices]
                batch_returns = all_returns[batch_indices]
                batch_entropies = dist_entropies[batch_indices]

                # === 计算PPO损失 ===
                # 重要性采样比率
                ratios = torch.exp(new_logprobs - old_logprobs[batch_indices].detach())

                # PPO裁剪损失
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                value_loss = self.MseLoss(batch_values, batch_returns)

                # 熵正则化
                entropy_loss = -batch_entropies.mean()

                # 总损失
                loss = (self.A_coeff * policy_loss +
                        self.vf_coeff * value_loss +
                        self.entropy_coeff * entropy_loss)

                # === 反向传播和优化 ===
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.hgnn_scheduler.parameters()) + list(self.mlps.parameters()),
                    max_norm=0.5
                )
                optimizer.step()

                # === 累计统计 ===
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()

        # === 同步新旧策略 ===
        self.hgnn_scheduler_old.load_state_dict(self.hgnn_scheduler.state_dict())
        self.mlps_old.load_state_dict(self.mlps.state_dict())

        # === 返回统计信息 ===
        num_updates = self.K_epochs * max(1, batch_size // minibatch_size)
        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates
        }

    def save(self, path):
        """
        保存模型参数

        参数：
            path (str): 保存路径
        """
        torch.save({
            "hgnn_scheduler": self.hgnn_scheduler.state_dict(),
            "mlps": self.mlps.state_dict(),
            "hgnn_scheduler_old": self.hgnn_scheduler_old.state_dict(),
            "mlps_old": self.mlps_old.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hyperparameters": {
                "lr": self.lr,
                "gamma": self.gamma,
                "lam": self.lam,
                "eps_clip": self.eps_clip,
                "K_epochs": self.K_epochs,
                "A_coeff": self.A_coeff,
                "vf_coeff": self.vf_coeff,
                "entropy_coeff": self.entropy_coeff
            }
        }, path)

    def load(self, path):
        """
        加载模型参数

        参数：
            path (str): 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.hgnn_scheduler.load_state_dict(checkpoint["hgnn_scheduler"])
        self.mlps.load_state_dict(checkpoint["mlps"])
        self.hgnn_scheduler_old.load_state_dict(checkpoint["hgnn_scheduler_old"])
        self.mlps_old.load_state_dict(checkpoint["mlps_old"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # 加载超参数（如果存在）
        if "hyperparameters" in checkpoint:
            hp = checkpoint["hyperparameters"]
            self.lr = hp.get("lr", self.lr)
            self.gamma = hp.get("gamma", self.gamma)
            self.lam = hp.get("lam", self.lam)
            self.eps_clip = hp.get("eps_clip", self.eps_clip)
            self.K_epochs = hp.get("K_epochs", self.K_epochs)
            self.A_coeff = hp.get("A_coeff", self.A_coeff)
            self.vf_coeff = hp.get("vf_coeff", self.vf_coeff)
            self.entropy_coeff = hp.get("entropy_coeff", self.entropy_coeff)

    def _enhance_usv_features(self, usv_features, charging_manager, current_time):
        """
        USV特征增强：添加充电状态和充电时长特征

        参数：
            usv_features (torch.Tensor): 原始USV特征，形状(B,M,4)
            charging_manager: 充电站管理器实例
            current_time (torch.Tensor): 当前时间，形状(B,)

        返回：
            torch.Tensor: 增强后的USV特征，形状(B,M,6)
                      - [0-3]: 原有4维特征
                      - [4]: 是否在充电 (0.0/1.0)
                      - [5]: 充电时长
        """
        B, M, D = usv_features.shape
        enhanced_features = torch.zeros(B, M, D + 2, device=usv_features.device, dtype=usv_features.dtype)

        # 复制原有特征
        enhanced_features[:, :, :D] = usv_features

        # 添加充电相关特征
        for b in range(B):
            # 获取当前批次的USV ID列表
            usv_ids = list(range(M))  # 简化处理：假设USV ID为0,1,2,...,M-1

            # 获取充电特征
            charging_features = charging_manager.get_all_charging_features(
                usv_ids, current_time[b].item(), usv_features.device
            )

            # 添加到增强特征中
            enhanced_features[b, :, D] = charging_features[:, 0]  # 是否在充电
            enhanced_features[b, :, D + 1] = charging_features[:, 1]  # 充电时长

        return enhanced_features
