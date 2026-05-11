"""Frame-level and per-class metrics for NILMbench.

All routines operate on float NumPy arrays shaped ``(T, K)`` where ``T`` is the
number of 6-second frames and ``K`` is the number of scored appliance
categories. Ground-truth and predicted power are in watts.

The headline NILMbench metric is the modified Jaccard index with a 20 W
absolute tolerance, ``MJ_{20W}``: a frame-level Jaccard whose true-positive
cell is restricted to predictions that lie within ``|p_hat - p| <= delta``.
True negatives are excluded from the average, so a "predict zero everywhere"
baseline cannot inflate the score.
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLDS_W = {
    "always on": 3.0,
    "cooking": 50.0,
    "dishwasher": 10.0,
    "electrical heating": 20.0,
    "electronics & lighting": 5.0,
    "fridge": 5.0,
    "misc": 10.0,
    "washing machine": 10.0,
}


def _coerce_supports(y_true_W: np.ndarray,
                     y_pred_W: np.ndarray,
                     theta_W: np.ndarray | float,
                     cutoff_W: np.ndarray | float | None = None):
    """Build the active-category supports A_t and B_t."""
    if cutoff_W is None:
        cutoff_W = theta_W
    theta = np.broadcast_to(np.asarray(theta_W, dtype=np.float32),
                            y_true_W.shape[1:])
    cut = np.broadcast_to(np.asarray(cutoff_W, dtype=np.float32),
                          y_pred_W.shape[1:])
    A = y_true_W > theta[None, :]
    B = y_pred_W > cut[None, :]
    return A, B


def modified_jaccard_frame(y_true_W: np.ndarray,
                           y_pred_W: np.ndarray,
                           theta_W: np.ndarray | float,
                           delta_W: float = 20.0,
                           cutoff_W: np.ndarray | float | None = None) -> dict:
    """Frame-averaged MJ with absolute power tolerance ``delta_W``.

    Returns a dict with the per-frame ``mj`` series, the scalar mean over the
    frames where ``A_t ∪ B_t`` is non-empty, and the counts (ATP, ITP, FP, FN).
    """
    A, B = _coerce_supports(y_true_W, y_pred_W, theta_W, cutoff_W)
    err_ok = np.abs(y_pred_W - y_true_W) <= delta_W
    C = A & B & err_ok
    union = A | B
    union_sz = union.sum(axis=1).astype(np.float32)
    mj = np.full(y_true_W.shape[0], np.nan, dtype=np.float32)
    has = union_sz > 0
    mj[has] = C[has].sum(axis=1) / union_sz[has]
    return {
        "mj": mj,
        "mean": float(np.nanmean(mj)),
        "atp": int((A & B & err_ok).sum()),
        "itp": int((A & B & ~err_ok).sum()),
        "fp": int((~A & B).sum()),
        "fn": int((A & ~B).sum()),
        "n_frames_scored": int(has.sum()),
        "n_frames_total": int(y_true_W.shape[0]),
    }


def modified_jaccard_per_class(y_true_W: np.ndarray,
                               y_pred_W: np.ndarray,
                               theta_W: np.ndarray | float,
                               delta_W: float = 20.0,
                               cutoff_W: np.ndarray | float | None = None) -> np.ndarray:
    """Per-class one-vs-rest MJ. Returns one score per category."""
    A, B = _coerce_supports(y_true_W, y_pred_W, theta_W, cutoff_W)
    err_ok = np.abs(y_pred_W - y_true_W) <= delta_W
    atp = (A & B & err_ok).sum(axis=0)
    itp = (A & B & ~err_ok).sum(axis=0)
    fp = (~A & B).sum(axis=0)
    fn = (A & ~B).sum(axis=0)
    denom = atp + itp + fp + fn
    out = np.zeros_like(atp, dtype=np.float32)
    np.divide(atp, denom, out=out, where=denom > 0)
    return out


def modified_f1_frame(y_true_W: np.ndarray,
                      y_pred_W: np.ndarray,
                      theta_W: np.ndarray | float,
                      r_rel: float = 0.2,
                      cutoff_W: np.ndarray | float | None = None) -> dict:
    """Frame-averaged modified F1 with relative-power tolerance ``r_rel``.

    The admissible-error rule is ``|p_hat - p| <= r_rel * p``.
    """
    A, B = _coerce_supports(y_true_W, y_pred_W, theta_W, cutoff_W)
    rel = np.minimum(np.abs(y_pred_W - y_true_W) / (y_true_W + 1e-9), 1.0)
    err_ok = rel <= r_rel
    atp = (A & B & err_ok).sum(axis=1).astype(np.float32)
    itp = (A & B & ~err_ok).sum(axis=1).astype(np.float32)
    fp = (~A & B).sum(axis=1).astype(np.float32)
    fn = (A & ~B).sum(axis=1).astype(np.float32)
    denom = atp + itp + 0.5 * (fp + fn)
    mf = np.full(y_true_W.shape[0], np.nan, dtype=np.float32)
    has = denom > 0
    mf[has] = atp[has] / denom[has]
    return {"mf": mf, "mean": float(np.nanmean(mf))}


def f1_frame(y_true_W: np.ndarray,
             y_pred_W: np.ndarray,
             theta_W: np.ndarray | float,
             cutoff_W: np.ndarray | float | None = None) -> float:
    A, B = _coerce_supports(y_true_W, y_pred_W, theta_W, cutoff_W)
    tp = (A & B).sum(axis=1).astype(np.float32)
    fp = (~A & B).sum(axis=1).astype(np.float32)
    fn = (A & ~B).sum(axis=1).astype(np.float32)
    denom = tp + 0.5 * (fp + fn)
    scores = np.full(y_true_W.shape[0], np.nan, dtype=np.float32)
    has = denom > 0
    scores[has] = tp[has] / denom[has]
    return float(np.nanmean(scores))


def jaccard_frame(y_true_W: np.ndarray,
                  y_pred_W: np.ndarray,
                  theta_W: np.ndarray | float,
                  cutoff_W: np.ndarray | float | None = None) -> float:
    A, B = _coerce_supports(y_true_W, y_pred_W, theta_W, cutoff_W)
    inter = (A & B).sum(axis=1).astype(np.float32)
    union = (A | B).sum(axis=1).astype(np.float32)
    scores = np.full(y_true_W.shape[0], np.nan, dtype=np.float32)
    has = union > 0
    scores[has] = inter[has] / union[has]
    return float(np.nanmean(scores))


def teca_frame(y_true_W: np.ndarray, y_pred_W: np.ndarray) -> float:
    """Total Energy Correctly Assigned, per the standard NILM definition."""
    P = y_true_W.sum(axis=1)
    denom = 2.0 * P
    err = np.abs(y_true_W - y_pred_W).sum(axis=1)
    scores = np.where(denom > 0, 1.0 - err / np.maximum(denom, 1e-9), np.nan)
    return float(np.nanmean(scores))


def mae_W(y_true_W: np.ndarray, y_pred_W: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true_W - y_pred_W)))


def state_accuracy_hamming(y_true_W: np.ndarray,
                           y_pred_W: np.ndarray,
                           theta_W: np.ndarray | float,
                           cutoff_W: np.ndarray | float | None = None) -> float:
    """Mean on/off agreement over (frame, category) pairs."""
    A, B = _coerce_supports(y_true_W, y_pred_W, theta_W, cutoff_W)
    return float((A == B).mean())
