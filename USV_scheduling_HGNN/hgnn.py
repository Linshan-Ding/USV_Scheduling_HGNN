import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNNLayer(nn.Module):
    """异质图神经网络层 (Hidden -> Hidden)"""

    def __init__(self, hidden_dim=8, n_neighbors=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_neighbors = n_neighbors

        # 特征变换矩阵 (W_Q, W_K, W_V)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

        # 边特征编码 (处理距离信息)
        self.edge_encoder = nn.Linear(1, hidden_dim)

        # 注意力机制参数
        # 修改：输入维度应为 hidden_dim (基于加性注意力 tanh(Wq + Wk + We))
        self.attn_usv = nn.Linear(hidden_dim, 1)
        self.attn_task = nn.Linear(hidden_dim, 1)

        # 层归一化 (可选，有助于深层网络稳定)
        self.norm_usv = nn.LayerNorm(hidden_dim)
        self.norm_task = nn.LayerNorm(hidden_dim)

    def forward(self, usv_embed, task_embed, adj_matrix, edge_features, task_coords):
        """
        前向传播
        Args:
            usv_embed: [batch, n_usvs, hidden]
            task_embed: [batch, n_tasks, hidden]
            adj_matrix: [batch, n_usvs, n_tasks] 0/1 掩码
            edge_features: [batch, n_usvs, n_tasks, 1] 距离
            task_coords: [batch, n_tasks, 2] 任务坐标(用于计算任务间邻居)
        """
        batch_size = usv_embed.size(0)
        n_usvs = usv_embed.size(1)
        n_tasks = task_embed.size(1)

        # ===== 阶段1: USV 更新 (聚合 Task -> USV) =====
        # Q: USV, K: Task, V: Task
        usv_query = self.W_Q(usv_embed).unsqueeze(2)  # [B, Nu, 1, H]
        task_key = self.W_K(task_embed).unsqueeze(1)  # [B, 1, Nt, H]
        task_val = self.W_V(task_embed).unsqueeze(1)  # [B, 1, Nt, H] (用于聚合)

        # 编码边特征 (距离)
        edge_emb = self.edge_encoder(edge_features)  # [B, Nu, Nt, H]

        # 计算注意力分数: tanh(Q + K + E)
        # 广播机制: [B, Nu, 1, H] + [B, 1, Nt, H] + [B, Nu, Nt, H] -> [B, Nu, Nt, H]
        attn_energy = torch.tanh(usv_query + task_key + edge_emb)
        attn_scores = self.attn_usv(attn_energy).squeeze(-1)  # [B, Nu, Nt]

        # Mask处理
        attn_scores = attn_scores.masked_fill(adj_matrix == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, Nu, Nt]

        # 聚合: weights * V
        # [B, Nu, Nt] x [B, Nt, H] -> [B, Nu, H]
        aggregated_task = torch.bmm(attn_weights, task_embed)

        # 残差连接 + 激活
        usv_embed_new = F.elu(usv_embed + aggregated_task)
        usv_embed_new = self.norm_usv(usv_embed_new)

        # ===== 阶段2: Task 更新 (聚合 Task邻居 + USV) =====

        # 1. 聚合 Task 邻居 (Task -> Task)
        # 动态计算任务的 KNN 图
        task_dist = torch.cdist(task_coords, task_coords)  # [B, Nt, Nt]
        # 获取 top-k 邻居索引 (包含自身)
        k = min(self.n_neighbors + 1, n_tasks)
        _, indices = torch.topk(-task_dist, k=k, dim=-1)

        # 构建 Task-Task 邻接掩码
        task_adj = torch.zeros(batch_size, n_tasks, n_tasks, device=task_embed.device)
        task_adj.scatter_(2, indices, 1.0)  # [B, Nt, Nt]

        # Task 自注意力
        t_query = self.W_Q(task_embed).unsqueeze(2)  # [B, Nt, 1, H]
        t_key = self.W_K(task_embed).unsqueeze(1)  # [B, 1, Nt, H]

        t_attn_energy = torch.tanh(t_query + t_key)
        t_attn_scores = self.attn_task(t_attn_energy).squeeze(-1)  # [B, Nt, Nt]
        t_attn_scores = t_attn_scores.masked_fill(task_adj == 0, -1e9)
        t_attn_weights = F.softmax(t_attn_scores, dim=-1)

        aggregated_tasks_neighbors = torch.bmm(t_attn_weights, task_embed)  # [B, Nt, H]

        # 2. 聚合关联的 USV (USV -> Task)
        # 使用阶段1计算的 attention weights 的转置
        # attn_weights: [B, Nu, Nt] -> transpose -> [B, Nt, Nu]
        # 注意：这里直接复用权重，表示相互影响；或者重新计算 attention 也可以
        # 这里为了对齐特征，重新计算反向注意力

        task_query_u = self.W_Q(task_embed).unsqueeze(2)  # [B, Nt, 1, H]
        usv_key_u = self.W_K(usv_embed_new).unsqueeze(1)  # [B, 1, Nu, H]
        edge_emb_t = edge_emb.transpose(1, 2)  # [B, Nt, Nu, H]

        t_u_energy = torch.tanh(task_query_u + usv_key_u + edge_emb_t)
        t_u_scores = self.attn_usv(t_u_energy).squeeze(-1)  # [B, Nt, Nu]

        # 掩码转置: adj_matrix [B, Nu, Nt] -> [B, Nt, Nu]
        t_u_scores = t_u_scores.masked_fill(adj_matrix.transpose(1, 2) == 0, -1e9)
        t_u_weights = F.softmax(t_u_scores, dim=-1)

        aggregated_usv = torch.bmm(t_u_weights, usv_embed_new)  # [B, Nt, H]

        # 合并更新
        task_embed_new = F.elu(task_embed + aggregated_tasks_neighbors + aggregated_usv)
        task_embed_new = self.norm_task(task_embed_new)

        return usv_embed_new, task_embed_new


class HGNN(nn.Module):
    """多层异质图神经网络"""

    def __init__(self, hidden_dim=8, n_layers=3, n_neighbors=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 修正：将原始特征编码器移至 HGNN 主类
        # USV原始特征: [x, y, battery, busy_time] -> 4
        self.usv_raw_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU()
        )

        # Task原始特征: [x, y, duration, EST, scheduled] -> 5
        self.task_raw_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU()
        )

        self.layers = nn.ModuleList([
            HGNNLayer(hidden_dim, n_neighbors) for _ in range(n_layers)
        ])

    def forward(self, state_dict):
        """
        处理状态字典，返回嵌入
        """
        usv_features = state_dict['usv_features']  # [batch, n_usvs, 4]
        task_features = state_dict['task_features']  # [batch, n_tasks, 5]

        # 修正：显式提取坐标用于距离和邻居计算
        # 假设前两维是坐标 x, y
        task_coords = task_features[:, :, 0:2]

        # 1. 原始特征编码 (Raw -> Hidden)
        usv_embed = self.usv_raw_encoder(usv_features)
        task_embed = self.task_raw_encoder(task_features)

        # 2. 构建图结构数据
        adj_matrix = self._build_adjacency(state_dict)
        edge_features = self._compute_edge_features(state_dict)

        # 3. 多层 HGNN 传播
        for layer in self.layers:
            usv_embed, task_embed = layer(
                usv_embed, task_embed, adj_matrix, edge_features, task_coords
            )

        # 4. 全局池化
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

        # 基础掩码：全1
        adj = torch.ones(batch_size, n_usvs, n_tasks, device=usv_features.device)

        # 逻辑：已调度(scheduled=1)的任务应当对USV不可见(或mask掉)
        # task_features最后一维是 scheduled 标记
        scheduled = task_features[:, :, 4]  # [batch, n_tasks]

        # 广播 scheduled: [B, 1, Nt]
        # 如果 scheduled=1, 也就是 1-1=0, mask 变为 0
        adj = adj * (1 - scheduled.unsqueeze(1))

        return adj

    def _compute_edge_features(self, state_dict):
        """计算边特征（距离）"""
        usv_features = state_dict['usv_features']
        # 修正：直接使用 state_dict 中的 features 提取坐标，或者如果 state_dict 有显式 coords key 则使用它
        # 这里假设从 features 提取
        task_coords = state_dict['task_features'][:, :, 0:2]

        usv_pos = usv_features[:, :, 0:2]  # [batch, n_usvs, 2]

        # 计算距离矩阵
        usv_pos_exp = usv_pos.unsqueeze(2)  # [batch, n_usvs, 1, 2]
        task_pos_exp = task_coords.unsqueeze(1)  # [batch, 1, n_tasks, 2]

        distances = torch.norm(usv_pos_exp - task_pos_exp, dim=-1, keepdim=True)

        return distances  # [batch, n_usvs, n_tasks, 1]