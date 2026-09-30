"""Attention modules: Efficient Channel Attention (ECA) and the Temporal Attention Module (TAM)."""

import torch
import torch.nn as nn


class ECABlock2D(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020).

    Global average pooling, a 1-D convolution across channels and a sigmoid produce one
    weight in [0, 1] per feature map, which rescales the CNN output.
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F)
        y = self.avg_pool(x)                                  # (B, C, 1, 1)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))      # (B, 1, C)
        y = torch.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # (B, C, 1, 1)
        return x * y.expand_as(x)


class TemporalAttention(nn.Module):
    """Additive attention pooling over the BiGRU hidden states.

    A two-layer MLP scores each time step, a softmax over time turns the scores into
    weights, and the output is the weighted sum of the hidden states.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2H)
        weights = torch.softmax(self.attention(x).squeeze(-1), dim=1)  # (B, T)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)            # (B, 2H)
