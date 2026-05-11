"""DeepDFML: stacked 1-D convolutional baseline operating on aggregate current."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nilmbench.models.base import NILMRegressor


class DeepDFML(NILMRegressor):
    """One-D conv baseline operating on the aggregate current channel.

    The first FC layer is :class:`torch.nn.LazyLinear` so the network adapts
    to any input length.
    """

    def __init__(self, n_categories: int):
        super().__init__()
        self.back = nn.Sequential(
            nn.Conv1d(1, 60, kernel_size=9),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(60, 40, kernel_size=9),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(40, 40, kernel_size=9),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(40, 40, kernel_size=9),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(40, 40, kernel_size=9),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=4),
        )
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.LazyLinear(300),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(300, 300),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(300, n_categories),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(self.back(x))
        return F.softmax(h, dim=-1)
