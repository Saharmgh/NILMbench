"""COLD: stacked Transformer encoder with per-block sequence pooling.

Operates on a sequence of FITPS frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import TransformerEncoderLayer

from nilmbench.models.base import NILMRegressor


class _AvgSeqAdaptivePool(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_size == x.shape[1]:
            return x
        kernel_size = (int(x.shape[-2] // self.output_size), 1)
        return F.avg_pool2d(x, kernel_size)


class COLD(NILMRegressor):
    def __init__(self,
                 n_categories: int,
                 input_size: int = 50,
                 seq_size: int = 60,
                 hidden_size: int = 512,
                 n_head: int = 8,
                 pools: tuple[int, ...] = (30, 10, 1),
                 activation: str = "relu",
                 dropout: float = 0.2):
        super().__init__()
        blocks: list[nn.Module] = []
        for pool_size in pools:
            if input_size != hidden_size:
                blocks.append(nn.Linear(input_size, hidden_size, bias=False))
            blocks.append(TransformerEncoderLayer(
                hidden_size, dropout=dropout, batch_first=True,
                dim_feedforward=4 * hidden_size,
                activation=activation, nhead=n_head,
            ))
            if pool_size != seq_size:
                blocks.append(_AvgSeqAdaptivePool(pool_size))
                seq_size = pool_size
            input_size = hidden_size
        self.back = nn.Sequential(*blocks, nn.Flatten(1))
        self.head = nn.Linear(seq_size * hidden_size, n_categories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(self.back(x))
        return torch.softmax(h, dim=-1)
