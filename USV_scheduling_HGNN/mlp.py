# ============================================================================
# 4. mlp.py - 多层感知机模块
# ============================================================================
import torch.nn as nn


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # 确保 hidden_dims 是列表，如果是 None 则转换为空列表
        if hidden_dims is None:
            hidden_dims = []

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # 添加激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            else:
                # 抛出异常而不是忽略，便于调试
                raise ValueError(f"Unsupported activation function: {activation}")

            prev_dim = hidden_dim

        # 输出层（通常不加激活函数，或者根据任务具体需求添加）
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)