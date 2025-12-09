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