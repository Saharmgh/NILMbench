"""SchirmerCNN: lightweight 2-D CNN baseline for NILM regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nilmbench.models.base import NILMRegressor


class SchirmerCNN(NILMRegressor):
    """Two-D convolutional baseline.

    The first fully-connected layer is :class:`torch.nn.LazyLinear` so the
    network adapts to the spatial size of the chosen preprocessing
    (Spectrogram, FITPS, raw image reshape, ...).
    """

    def __init__(self, n_categories: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding="same"),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding="same"),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding="same"),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_categories),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.fc(h)
        return F.softmax(h, dim=-1)
