"""Trivial "bring your own model" example.

This is the minimum working interface a third-party NILMbench submission
needs to implement. It is intentionally useless as a model (it predicts
random per-category power) so you can see the contract without distractions.

Run it end-to-end with::

    nilmbench benchmark \
        --module examples.byom_random:RandomPredictor \
        --data hf:Pybunny/nilmbench-ukdale \
        --out ./report/

No ``--weights`` is needed because this model has no learned parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RandomPredictor(nn.Module):
    """Predicts a uniform random per-category active power in ``[0, max_W]``.

    The constructor's ``n_categories`` keyword is filled in automatically by
    :mod:`nilmbench.runner`: it inspects the dense benchmark dataset to
    determine the number of scored appliance categories, and passes the count
    to any model whose ``__init__`` accepts an argument named
    ``n_categories``, ``num_categories``, ``num_classes``, ``n_classes``,
    or ``K``.
    """

    def __init__(self, n_categories: int, max_W: float = 200.0,
                 seed: int = 0):
        super().__init__()
        self.n_categories = n_categories
        self.max_W = float(max_W)
        # Stored as a buffer so the seed survives state_dict round-trips,
        # though this model has no learned parameters.
        self.register_buffer("_seed", torch.tensor(int(seed)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 96000) -- ignored on purpose.
        g = torch.Generator(device=x.device).manual_seed(
            int(self._seed.item()) + int(x.shape[0]))
        return torch.rand(x.shape[0], self.n_categories,
                          generator=g, device=x.device) * self.max_W
