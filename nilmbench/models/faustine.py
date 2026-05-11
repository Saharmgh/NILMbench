"""FaustineCNN: 2-D convolutional baseline operating on a V/I representation.

Reference: Faustine et al., 2-D convolutional NILM regression on aggregate
voltage/current (see paper citations).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nilmbench.models.base import NILMRegressor


class FaustineCNN(NILMRegressor):
    def __init__(self, n_categories: int, fc_hidden: int = 1024,
                 dropout: float = 0.25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 2 * n_categories),
        )
        self.n_categories = n_categories

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        h = self.fc(h).view(x.size(0), self.n_categories, 2)
        return F.softmax(h, dim=-1)[..., 0]
