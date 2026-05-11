"""Recall-constrained per-category cutoff calibration.

After training, raw model predictions can be noisy near zero. The benchmark
allows tightening the prediction support B_t by replacing the activation
threshold theta_k with a per-class cutoff c_k learned only on the House-1
validation split, subject to a recall constraint that prevents low-power
appliances from being filtered out.

The cutoffs are then applied unchanged at evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CutoffResult:
    classes: list[str]
    cutoffs_W: dict[str, float]
    recall_floor: float
    sweep_grid_W: dict[str, np.ndarray] = field(default_factory=dict)


def _recall_at_cutoff(y_true_W: np.ndarray,
                     y_pred_W: np.ndarray,
                     theta: float,
                     cutoff: float) -> float:
    A = y_true_W > theta
    B = y_pred_W > cutoff
    tp = int((A & B).sum())
    fn = int((A & ~B).sum())
    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)


def calibrate_recall_constrained_cutoffs(
    y_true_val_W: np.ndarray,
    y_pred_val_W: np.ndarray,
    classes: list[str],
    theta_W: dict[str, float],
    recall_floor: float = 0.5,
    sweep_low_W: float = 0.0,
    sweep_high_factor: float = 5.0,
    n_grid: int = 100,
) -> CutoffResult:
    """Sweep a grid of candidate cutoffs per class on the validation set.

    For each class, the cutoff is the highest grid point that still satisfies
    ``recall_at_cutoff >= recall_floor``. Classes with no validation activity
    fall back to their original ``theta_W``.
    """
    cutoffs = {}
    grids = {}
    for k, cls in enumerate(classes):
        theta = float(theta_W[cls])
        active_max = float(y_pred_val_W[:, k].max(initial=0.0))
        hi = max(theta, active_max) * sweep_high_factor
        grid = np.linspace(sweep_low_W, hi, n_grid, dtype=np.float32)
        grids[cls] = grid

        # Default: leave at theta
        chosen = theta
        for c in grid[::-1]:
            r = _recall_at_cutoff(y_true_val_W[:, k], y_pred_val_W[:, k],
                                  theta, float(c))
            if r >= recall_floor:
                chosen = float(c)
                break
        # Never go below theta
        cutoffs[cls] = max(chosen, theta)
    return CutoffResult(classes=list(classes), cutoffs_W=cutoffs,
                        recall_floor=recall_floor, sweep_grid_W=grids)


def apply_cutoffs(y_pred_W: np.ndarray,
                  cutoffs_W: dict[str, float],
                  classes: list[str]) -> np.ndarray:
    """Zero-out predicted power below each per-class cutoff."""
    out = y_pred_W.copy()
    for k, cls in enumerate(classes):
        out[:, k] = np.where(out[:, k] > cutoffs_W[cls], out[:, k], 0.0)
    return out
