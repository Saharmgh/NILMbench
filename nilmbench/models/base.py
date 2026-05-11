"""Shared base class for the NILMbench baseline models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class NILMRegressor(ABC, nn.Module):
    """Common interface for the four baseline models.

    The forward pass returns a non-negative per-category score in ``[0, 1]``.
    Two output families are used in the literature:

    * **Softmax over categories** (SchirmerCNN, COLD, DeepDFML). Outputs sum
      to one across categories and can be interpreted directly as power
      shares.
    * **Per-category Bernoulli** (FaustineCNN). Outputs are independent
      activation probabilities; renormalisation across categories is applied
      at evaluation time to obtain shares.

    Either way, the final per-category power is obtained by multiplying the
    renormalised shares by an estimate of the total active power, e.g. the
    aggregate measured at the meter (see :meth:`shares_to_power`).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a non-negative per-category score in ``[0, 1]``."""

    @staticmethod
    def renormalise_shares(scores: torch.Tensor,
                           eps: float = 1e-9) -> torch.Tensor:
        return scores / (scores.sum(dim=-1, keepdim=True) + eps)

    def shares_to_power(self,
                        scores: torch.Tensor,
                        aggregate_power: torch.Tensor) -> torch.Tensor:
        """Distribute a predicted aggregate power across categories.

        ``scores`` are renormalised across categories before scaling, so this
        works for both softmax and Bernoulli output families.
        """
        shares = self.renormalise_shares(scores)
        return shares * aggregate_power.unsqueeze(-1)
