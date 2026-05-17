"""Starter template for your own NILMbench submission.

Copy this file, rename ``MyModel``, replace the network body with your
architecture, and point the CLI at it::

    nilmbench benchmark \
        --module examples.byom_template:MyModel \
        --weights ./my_checkpoint.pt \
        --data hf:Pybunny/nilmbench-ukdale \
        --out ./report/

Model contract (enforced by :mod:`nilmbench.runner`):

    Input  x : torch.Tensor, shape (B, 2, 96000), float32
              x[:, 0, :] -- voltage trace (V)
              x[:, 1, :] -- current trace (A)
    Output y : torch.Tensor, shape (B, K), float32, non-negative
              per-category active power in WATTS.

If your model emits per-category shares in ``[0, 1]`` rather than watts,
pass ``--shares`` to the CLI and the runner will rescale by the per-frame
aggregate active power. Otherwise return watts directly.

The optional ``n_categories`` constructor keyword is filled in automatically
from the benchmark dataset; you don't have to hard-code it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MyModel(nn.Module):
    """Replace the body with your architecture."""

    def __init__(self, n_categories: int):
        super().__init__()
        # Demo: a single-layer 1-D conv + global pool + linear head.
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=64, stride=16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, n_categories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 96000) -- 6 seconds of 16 kHz V/I
        h = self.encoder(x).flatten(1)  # (B, 32)
        y = self.head(h)                # (B, K) -- arbitrary sign
        # Watts must be non-negative; let NILMbench see softplus output.
        return torch.nn.functional.softplus(y)
