"""End-to-end dense House-2 evaluation loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from nilmbench.metrics import (
    modified_jaccard_frame,
    modified_jaccard_per_class,
    modified_f1_frame,
    f1_frame,
    jaccard_frame,
    teca_frame,
    mae_W,
    state_accuracy_hamming,
    DEFAULT_THRESHOLDS_W,
)


@dataclass
class BenchmarkResult:
    model: str
    classes: list[str]
    n_frames: int
    MJ_20W: float
    MJ_20pct: float
    MF_20pct: float
    F1: float
    Jaccard: float
    TECA: float
    MAE_W: float
    StateAcc_Hamming: float
    MJ_per_class_20W: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_dense(
    y_true_W: np.ndarray,
    y_pred_W: np.ndarray,
    classes: list[str],
    model_name: str = "model",
    thresholds_W: dict[str, float] | None = None,
    cutoffs_W: dict[str, float] | None = None,
) -> BenchmarkResult:
    """Compute the full NILMbench score sheet for a single model's predictions.

    Parameters
    ----------
    y_true_W, y_pred_W : array, shape (T, K)
        Per-frame ground-truth and predicted power in watts.
    classes : list of str, length K
        Category names matching the column order.
    thresholds_W :
        Per-class activation thresholds theta_k. Defaults to
        :data:`nilmbench.metrics.DEFAULT_THRESHOLDS_W`.
    cutoffs_W :
        Per-class prediction cutoffs c_k. If None, c_k = theta_k.
    """
    thr = thresholds_W or {c: DEFAULT_THRESHOLDS_W[c] for c in classes}
    theta = np.array([thr[c] for c in classes], dtype=np.float32)
    cut = np.array([(cutoffs_W or thr)[c] for c in classes], dtype=np.float32)

    mj = modified_jaccard_frame(y_true_W, y_pred_W, theta, 20.0, cut)
    mj_rel = modified_f1_frame(y_true_W, y_pred_W, theta, 0.2, cut)
    mf_rel = modified_f1_frame(y_true_W, y_pred_W, theta, 0.2, cut)  # MF == MJ form check

    return BenchmarkResult(
        model=model_name,
        classes=list(classes),
        n_frames=int(y_true_W.shape[0]),
        MJ_20W=mj["mean"],
        MJ_20pct=mj_rel["mean"],
        MF_20pct=mf_rel["mean"],
        F1=f1_frame(y_true_W, y_pred_W, theta, cut),
        Jaccard=jaccard_frame(y_true_W, y_pred_W, theta, cut),
        TECA=teca_frame(y_true_W, y_pred_W),
        MAE_W=mae_W(y_true_W, y_pred_W),
        StateAcc_Hamming=state_accuracy_hamming(y_true_W, y_pred_W, theta, cut),
        MJ_per_class_20W=dict(zip(
            classes,
            (float(v) for v in modified_jaccard_per_class(
                y_true_W, y_pred_W, theta, 20.0, cut)),
        )),
    )


def save_result(result: BenchmarkResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)
