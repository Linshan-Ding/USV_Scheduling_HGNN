
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional


class USVBlock(nn.Module):
    """
    USV节点嵌入模块：基于GATedge架构实现"任务→USV"异构注意力更新

    功能：
    1. 基于GATedge的三元注意力机制实现"任务→USV"消息传递
    2. 处理异构关系：任务节点→USV节点
    3. 实现三元注意力：源节点特征 + 目标节点特征 + 边特征
    4. 输出更新后的USV嵌入μ′ₖ

    对应关系：相当于FJSP中的GATedge模块
    """

    def __init__(self,
                 in_dims: Tuple[int, int],  # (d_task, d_usv)
                 out_dim: int,
                 num_head: int = 1,
                 feat_drop: float = 0.1,
                 attn_drop: float = 0.1,
                 negative_slope: float = 0.2,
                 activation=None):
        """
        初始化USVBlock

        Args:
            in_dims: 输入特征维度 (d_task, d_usv)
            out_dim: 输出USV嵌入维度
            num_head: 注意力头数（单头用于保持与基线一致）
            feat_drop: 特征dropout率
            attn_drop: 注意力dropout率
            negative_slope: LeakyReLU负斜率
            activation: 激活函数
        """
        super(USVBlock, self).__init__()
        self._num_heads = num_head
        self._in_src_feats = in_dims[0]  # 任务特征维度
        self._in_dst_feats = in_dims[1]  # USV特征维度
        self._out_feats = out_dim

        # 线性变换层：任务特征、USV特征、边特征
        self.fc_src = nn.Linear(self._in_src_feats, out_dim * num_head, bias=False)
        self.fc_dst = nn.Linear(self._in_dst_feats, out_dim * num_head, bias=False)
        self.fc_edge = nn.Linear(1, out_dim * num_head, bias=False)  # 标量边特征

        # 注意力参数
        self.attn_l = nn.Parameter(torch.rand(size=(1, num_head, out_dim), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, num_head, out_dim), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, num_head, out_dim), dtype=torch.float))

        # Dropout层
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self, feat_src: torch.Tensor, feat_dst: torch.Tensor, feat_edge: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播：USV节点嵌入更新

        Args:
            feat_src: 源节点特征（任务特征）(B, N, d_task) 或 (N, d_task)
            feat_dst: 目标节点特征（USV特征）(B, M, d_usv) 或 (M, d_usv)
            feat_edge: 边特征（执行时间）(B, N, M) 或 (N, M)
            adj: 邻接矩阵 (B, N, M) 或 (N, M)

        Returns:
            usv_embeddings: 更新后的USV嵌入μ′ₖ (B, M, out_dim) 或 (M, out_dim)
        """
        # 确定输入维度
        if feat_src.dim() == 3:
            # 批次处理
            B, N, d_src = feat_src.shape
            _, M, d_dst = feat_dst.shape
            batch_mode = True
        else:
            # 单实例处理
            N, d_src = feat_src.shape
            M, d_dst = feat_dst.shape
            batch_mode = False

        # 应用dropout
        h_src = self.feat_drop(feat_src)
        h_dst = self.feat_drop(feat_dst)

        # 线性变换
        feat_src = self.fc_src(h_src)      # (B, N, out_dim*num_head) 或 (N, out_dim*num_head)
        feat_dst = self.fc_dst(h_dst)      # (B, M, out_dim*num_head) 或 (M, out_dim*num_head)
        feat_edge = self.fc_edge(feat_edge.unsqueeze(-1))  # (B, N, M, out_dim*num_head) 或 (N, M, out_dim*num_head)

        # 计算注意力系数
        # 源节点注意力
        if batch_mode:
            el = (feat_src * self.attn_l).sum(dim=-1)  # (B, N)
            er = (feat_dst * self.attn_r).sum(dim=-1)  # (B, M)
            ee = (feat_edge * self.attn_e).sum(dim=-1)  # (B, N, M)
        else:
            # 单实例模式：确保注意力参数正确广播
            el = (feat_src * self.attn_l.squeeze(0)).sum(dim=-1)  # (N)
            er = (feat_dst * self.attn_r.squeeze(0)).sum(dim=-1)  # (M)
            ee = (feat_edge * self.attn_e.squeeze(0)).sum(dim=-1)  # (N, M)

  
        # 计算任务→USV的注意力分数（三元注意力）
        if batch_mode:
            # 批次模式：(B, N, M)
            # el: (B, N), er: (B, M), ee: (B, N, M)
            # 扩展维度以支持广播：
            # el.unsqueeze(-1): (B, N) → (B, N, 1)
            # er.unsqueeze(-2): (B, M) → (B, 1, M)
            el_expanded = el.unsqueeze(-1)           # (B, N, 1)
            er_expanded = er.unsqueeze(-2)           # (B, 1, M)

            # 三元注意力计算：源节点注意力 + 边注意力 + 目标节点注意力
            a = el_expanded + ee + er_expanded       # (B, N, M)
        else:
            # 单实例模式：(N, M)
            # el: (N), er: (M), ee: (N, M)
            # 扩展维度以支持广播：
            # el.unsqueeze(-1): (N) → (N, 1)
            # er.unsqueeze(-2): (M) → (1, M)
            el_expanded = el.unsqueeze(-1)           # (N, 1)
            er_expanded = er.unsqueeze(-2)           # (1, M)

            # 三元注意力计算：源节点注意力 + 边注意力 + 目标节点注意力
            a = el_expanded + ee + er_expanded       # (N, M)

        e = self.leaky_relu(a)                                   # (B, N, M) 或 (N, M)

        # 构建掩码：仅任务-USV边
        mask = adj == 1  # (B, N, M) 或 (N, M)

        # 确保mask和e有相同的形状
        if e.dim() == 3 and mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (N, M) → (1, N, M)
        elif e.dim() == 2 and mask.dim() == 3:
            mask = mask.squeeze(0)  # (1, N, M) → (N, M)

        # 应用掩码：非法连接设为负无穷
        e_masked = e.clone()
        e_masked[~mask] = float('-inf')

        # Softmax归一化（处理全零邻接矩阵的情况）
        if mask.any():
            alpha = F.softmax(e_masked, dim=-2)  # (B, N, M) 或 (N, M)
        else:
            # 全零邻接矩阵：所有注意力权重为0
            alpha = torch.zeros_like(e_masked)  # (B, N, M) 或 (N, M)
      
        # 计算消息传递：任务→USV
        if batch_mode:
            # 批次模式
            Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)             # (B, N, M, out_dim*num_head)
            msg = Wmu_ijk * alpha.unsqueeze(-1)                      # (B, N, M, out_dim*num_head)
            aggregated = torch.sum(msg, dim=-3)                      # (B, M, out_dim*num_head)
        else:
            # 单实例模式
            Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)             # (N, M, out_dim*num_head)
            msg = Wmu_ijk * alpha.unsqueeze(-1)                      # (N, M, out_dim*num_head)
            aggregated = torch.sum(msg, dim=-3)                      # (M, out_dim*num_head)

        # 应用激活函数
        usv_embeddings = torch.sigmoid(aggregated)               # (B, M, out_dim*num_head) 或 (M, out_dim*num_head)

        # 重塑输出维度
        if self._num_heads > 1:
            if batch_mode:
                B, M, _ = usv_embeddings.shape
                usv_embeddings = usv_embeddings.view(B, M, self._num_heads, self._out_feats)
                usv_embeddings = usv_embeddings.mean(dim=2)  # 多头平均 (B, M, out_dim)
            else:
                M, _ = usv_embeddings.shape
                usv_embeddings = usv_embeddings.view(M, self._num_heads, self._out_feats)
                usv_embeddings = usv_embeddings.mean(dim=1)  # 多头平均 (M, out_dim)

        # 应用激活函数
        if self.activation is not None:
            usv_embeddings = self.activation(usv_embeddings)

        return usv_embeddings


class TaskBlock(nn.Module):
    """
    任务节点嵌入模块：基于MLPsim架构实现η近邻任务聚合

    功能：
    1. 基于MLPsim架构处理任务节点间的消息传递
    2. 处理η近邻任务聚合（邻接加权求和→MLP投影）
    3. 简单邻接聚合，不增加复杂注意力机制
    4. 输出更新后的任务嵌入τ′ᵢ

    对应关系：相当于FJSP中的MLPsim模块
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 num_head: int = 1,
                 feat_drop: float = 0.1,
                 attn_drop: float = 0.1,
                 negative_slope: float = 0.2):
        """
        初始化TaskBlock

        Args:
            in_dim: 输入特征维度
            out_dim: 输出任务嵌入维度
            hidden_dim: MLP隐藏层维度
            num_head: 注意力头数（保持与基线一致）
            feat_drop: 特征dropout率
            attn_drop: 注意力dropout率
            negative_slope: LeakyReLU负斜率
        """
        super(TaskBlock, self).__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self._num_heads = num_head

        # Dropout层
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # MLP投影层：邻接聚合→MLP投影
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

    def forward(self, feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播：任务节点嵌入更新

        Args:
            feat: 输入特征 (B, N, d_task) 或 (N, d_task)
            adj: 邻接矩阵 (B, N, N) 或 (N, N)

        Returns:
            task_embeddings: 更新后的任务嵌入τ′ᵢ (B, N, out_dim) 或 (N, out_dim)
        """
        # 应用dropout
        h = self.feat_drop(feat)

        # MLPsim核心：邻接加权求和 → MLP投影
        # 广播：邻接矩阵 * 特征张量
        a = adj.unsqueeze(-1) * h.unsqueeze(-3)   # (B, N, N, d_task) 或 (N, N, d_task)
        b = torch.sum(a, dim=-2)                  # (B, N, d_task) 或 (N, d_task)
        task_embeddings = self.project(b)          # (B, N, out_dim) 或 (N, out_dim)

        return task_embeddings