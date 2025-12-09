
# 导入集中定义的常量
from utils.my_utils import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ACTION_LAYERS,
    DEFAULT_VALUE_LAYERS,
    MASK_FILL_VALUE
)

import torch
import torch.nn as nn
import torch.nn.functional as F


class USVPairFeature(nn.Module):
    """
    USV成对特征构造器 - 支持N+1动作空间

    功能：将任务嵌入、USV嵌入和全局状态构造成成对特征
    输入：task_emb(B,N,d), usv_emb(B,M,d), h_state(B,2d)
    输出：pair_feat(B,N+1,M,4d) - 包含充电站

    === 核心机制 ===
    1. 任务嵌入广播：(B,N,d) → (B,N,1,d) → (B,N,M,d)
    2. 充电站嵌入生成：创建充电站的特殊嵌入
    3. USV嵌入广播：(B,M,d) → (B,1,M,d) → (B,N+1,M,d)
    4. 全局状态广播：(B,2d) → (B,1,1,2d) → (B,N+1,M,2d)
    5. 特征拼接：[task_feat, usv_feat, h_state] → (B,N+1,M,4d)

    === 充电站处理 ===
    - 充电站作为第N+1个"任务"，具有独特的嵌入表示
    - 充电站嵌入：基于全局状态 + 充电站特殊标识生成
    - 支持所有USV前往充电站进行充电操作

    === 内存优化策略 ===
    - 使用expand避免数据复制，零开销广播
    - 向量化操作替代Python循环
    - 保持输入张量的dtype和device
    """

    def __init__(self, d=DEFAULT_EMBEDDING_DIM):
        """
        初始化USV成对特征构造器

        参数：
            d (int): 嵌入维度，默认32（与usv_hgnn.py一致）
        """
        super(USVPairFeature, self).__init__()
        self.d = d
        # 验证维度合理性
        if d <= 0:
            raise ValueError(f"嵌入维度d必须为正数，当前值：{d}")

        # 充电站特殊嵌入参数（可学习）
        self.charging_bias = nn.Parameter(torch.zeros(1, 1, self.d))

    def forward(self, task_emb, usv_emb, h_state, edge_feat=None):
        """
        前向传播：构造包含充电站的成对特征

        参数：
            task_emb (torch.Tensor): 任务嵌入，形状(B,N,d)
            usv_emb (torch.Tensor): USV嵌入，形状(B,M,d)
            h_state (torch.Tensor): 全局状态，形状(B,2d)
            edge_feat (torch.Tensor, optional): 可选边特征，形状(B,N+1,M,edge_dim)

        返回：
            torch.Tensor: 成对特征，形状(B,N+1,M,4d)
        """
        # 输入维度验证
        B, N, d_task = task_emb.shape
        B_m, M, d_usv = usv_emb.shape
        B_h, d_hstate = h_state.shape

        # 验证batch一致性
        if B != B_m or B != B_h:
            raise ValueError(f"批次大小不一致：task_emb={B}, usv_emb={B_m}, h_state={B_h}")

        # 验证嵌入维度一致性
        if d_task != self.d or d_usv != self.d:
            raise ValueError(f"嵌入维度不匹配：task_emb={d_task}, usv_emb={d_usv}, 期望={self.d}")

        # 验证全局状态维度
        if d_hstate != 2 * self.d:
            raise ValueError(f"全局状态维度不匹配：h_state={d_hstate}, 期望={2*self.d}")

        # === 任务嵌入广播：(B,N,d) → (B,N,M,d) ===
        # 步骤1：添加USV维度 (B,N,d) → (B,N,1,d)
        task_expanded = task_emb.unsqueeze(2)
        # 步骤2：广播到所有USV (B,N,1,d) → (B,N,M,d) - 零复制操作
        task_feat = task_expanded.expand(-1, -1, M, -1)  # (B,N,M,d)

        # === 充电站嵌入生成：(B,1,d) → (B,1,M,d) ===
        # 充电站嵌入：基于全局状态前半部分 + 充电站特殊标识
        charging_emb = h_state[:, :self.d].unsqueeze(1)  # (B,1,d) - 使用全局状态前半部分
        # 充电站特殊偏置（可学习的参数，让充电站具有独特性）
        charging_emb = charging_emb + self.charging_bias  # 添加充电站特殊性
        # 添加USV维度然后广播到所有USV (B,1,d) → (B,1,1,d) → (B,1,M,d)
        charging_feat = charging_emb.unsqueeze(2).expand(B, 1, M, self.d)  # (B,1,M,d)

        # === 组合任务和充电站特征：(B,N,M,d) + (B,1,M,d) → (B,N+1,M,d) ===
        task_charging_feat = torch.cat([task_feat, charging_feat], dim=1)  # (B,N+1,M,d)

        # === USV嵌入广播：(B,M,d) → (B,N+1,M,d) ===
        # 步骤1：添加任务维度 (B,M,d) → (B,1,M,d)
        usv_expanded = usv_emb.unsqueeze(1)
        # 步骤2：广播到所有任务和充电站 (B,1,M,d) → (B,N+1,M,d) - 零复制操作
        usv_feat = usv_expanded.expand(-1, N + 1, -1, -1)  # (B,N+1,M,d)

        # === 全局状态广播：(B,2d) → (B,N+1,M,2d) ===
        # 步骤1：添加任务和USV维度 (B,2d) → (B,1,1,2d)
        h_state_expanded = h_state.unsqueeze(1).unsqueeze(1)
        # 步骤2：广播到所有任务-USV对 (B,1,1,2d) → (B,N+1,M,2d)
        h_state_feat = h_state_expanded.expand(-1, N + 1, M, -1)  # (B,N+1,M,2d)

        # === 特征拼接：(B,N+1,M,d) + (B,N+1,M,d) + (B,N+1,M,2d) → (B,N+1,M,4d) ===
        pair_feat = torch.cat([task_charging_feat, usv_feat, h_state_feat], dim=-1)

        # 可选：拼接边特征（预留扩展接口）
        if edge_feat is not None:
            # 验证边特征维度
            B_edge, N_edge, M_edge, edge_dim = edge_feat.shape
            if (B_edge, N_edge, M_edge) != (B, N + 1, M):
                raise ValueError(f"边特征空间维度不匹配：edge_feat=({B_edge},{N_edge},{M_edge}), 期望=({B},{N+1},{M})")

            # 拼接边特征：(B,N+1,M,4d) + (B,N+1,M,edge_dim) → (B,N+1,M,4d+edge_dim)
            pair_feat = torch.cat([pair_feat, edge_feat], dim=-1)

        return pair_feat


class MLPActor(nn.Module):
    """
    MLP策略网络 - 从mlp.py完全复制

    严格遵循mlp.py第56-105行实现，确保训练行为完全一致
    使用tanh激活函数，无BatchNorm层

    === 关键特性 ===
    - 激活函数：严格使用torch.tanh，无ReLU或其他激活函数
    - 归一化：完全禁用BatchNorm，确保训练一致性
    - 架构：支持线性模型(num_layers=1)和多层模型
    - 设备兼容：自动保持输入张量的dtype和device
    """

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
        初始化MLP策略网络

        参数：
            num_layers: 神经网络层数（不包括输入层）。如果num_layers=1，则退化为线性模型
            input_dim: 输入特征维度
            hidden_dim: 所有隐藏层的隐藏单元维度
            output_dim: 预测类别数
        '''

        super(MLPActor, self).__init__()

        self.linear_or_not = True  # 默认线性模型
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # 线性模型
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # 多层模型
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        """前向传播 - 严格遵循mlp.py实现"""
        if self.linear_or_not:
            # 线性模型
            return self.linear(x)
        else:
            # MLP模型
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    """
    MLP价值网络 - 从mlp.py完全复制

    严格遵循mlp.py第108-158行实现，确保训练行为完全一致
    使用tanh激活函数，无BatchNorm层

    === 关键特性 ===
    - 激活函数：严格使用torch.tanh，无ReLU或其他激活函数
    - 归一化：完全禁用BatchNorm，确保训练一致性
    - 架构：支持线性模型(num_layers=1)和多层模型
    - 设备兼容：自动保持输入张量的dtype和device
    """

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
        初始化MLP价值网络

        参数：
            num_layers: 神经网络层数（不包括输入层）。如果num_layers=1，则退化为线性模型
            input_dim: 输入特征维度
            hidden_dim: 所有隐藏层的隐藏单元维度
            output_dim: 预测类别数
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # 默认线性模型
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # 线性模型
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # 多层模型
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        """前向传播 - 严格遵循mlp.py实现"""
        if self.linear_or_not:
            # 线性模型
            return self.linear(x)
        else:
            # MLP模型
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class USVActionHead(MLPActor):
    """
    USV策略头 - 继承MLPActor，支持N+1动作空间

    功能：将成对特征映射为动作logits
    输入：pair_feat(B,N+1,M,4d) - 包含充电站
    输出：logits_pair(B,N+1,M) - 包含充电站logits

    === 处理流程 ===
    1. 展平：(B,N+1,M,4d) → (B*(N+1)*M,4d)
    2. MLP处理：继承tanh激活架构
    3. 重塑：(B*(N+1)*M,1) → (B,N+1,M)

    === 充电站支持 ===
    - 第N+1行对应充电站的logits
    - 支持所有USV选择前往充电站
    - 与PPO层的N+1动作空间设计完全兼容
    """

    def __init__(self, d=DEFAULT_EMBEDDING_DIM, hidden_dim=DEFAULT_HIDDEN_DIM, num_layers=DEFAULT_ACTION_LAYERS):
        """
        初始化USV策略头

        参数：
            d (int): 嵌入维度，默认32
            hidden_dim (int): 隐藏层维度，默认128（等于4d）
            num_layers (int): 网络层数，默认3
        """
        in_dim = 4 * d  # 输入维度为4d
        super(USVActionHead, self).__init__(num_layers, in_dim, hidden_dim, 1)
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, pair_feat):
        """
        前向传播：成对特征到动作logits（包含充电站）

        参数：
            pair_feat (torch.Tensor): 成对特征，形状(B,N+1,M,4d)

        返回：
            torch.Tensor: 动作logits，形状(B,N+1,M)
        """
        # 验证输入维度
        B, N_plus_1, M, feat_dim = pair_feat.shape
        expected_dim = 4 * self.d
        if feat_dim != expected_dim:
            raise ValueError(f"成对特征维度不匹配：输入={feat_dim}, 期望={expected_dim}")

        # 展平：(B,N+1,M,4d) → (B*(N+1)*M,4d)
        pair_flat = pair_feat.view(-1, feat_dim)

        # MLP处理
        logits_flat = super().forward(pair_flat)  # (B*(N+1)*M,1)

        # 重塑：(B*(N+1)*M,1) → (B,N+1,M)
        logits = logits_flat.view(B, N_plus_1, M)

        return logits


class USVValueHead(MLPCritic):
    """
    USV价值头 - 继承MLPCritic

    功能：将全局状态映射为状态价值
    输入：h_state(B,2d)
    输出：value(B,1)

    === 设计特点 ===
    - 直接继承MLPCritic的tanh激活架构
    - 输入维度为2d（全局状态维度）
    - 输出标量价值函数
    """

    def __init__(self, d=DEFAULT_EMBEDDING_DIM, hidden_dim=DEFAULT_HIDDEN_DIM, num_layers=DEFAULT_VALUE_LAYERS):
        """
        初始化USV价值头

        参数：
            d (int): 嵌入维度，默认32
            hidden_dim (int): 隐藏层维度，默认128
            num_layers (int): 网络层数，默认2
        """
        in_dim = 2 * d  # 输入维度为2d（全局状态维度）
        super(USVValueHead, self).__init__(num_layers, in_dim, hidden_dim, 1)
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, h_state):
        """
        前向传播：全局状态到价值

        参数：
            h_state (torch.Tensor): 全局状态，形状(B,2d)

        返回：
            torch.Tensor: 状态价值，形状(B,1)
        """
        # 验证输入维度
        B, state_dim = h_state.shape
        expected_dim = 2 * self.d
        if state_dim != expected_dim:
            raise ValueError(f"全局状态维度不匹配：输入={state_dim}, 期望={expected_dim}")

        # 直接使用MLPCritic的前向传播
        value = super().forward(h_state)  # (B,1)

        return value