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