"""Smoke tests for the metric definitions."""

from __future__ import annotations

import numpy as np
import pytest

from nilmbench.metrics import (
    modified_jaccard_frame,
    modified_jaccard_per_class,
    f1_frame,
    jaccard_frame,
    teca_frame,
    DEFAULT_THRESHOLDS_W,
)


CLASSES = ["always on", "cooking", "fridge"]
THETA = np.array([DEFAULT_THRESHOLDS_W[c] for c in CLASSES], dtype=np.float32)


def test_perfect_prediction_gives_mj_one():
    y = np.array([[100.0, 0.0, 50.0],
                  [80.0, 0.0, 0.0]], dtype=np.float32)
    out = modified_jaccard_frame(y, y, THETA, delta_W=20.0)
    assert out["mean"] == pytest.approx(1.0)


def test_predict_zero_gives_zero_mj():
    y = np.array([[100.0, 0.0, 50.0]], dtype=np.float32)
    p = np.zeros_like(y)
    out = modified_jaccard_frame(y, p, THETA, delta_W=20.0)
    assert out["mean"] == pytest.approx(0.0)


def test_tn_only_frame_is_excluded_from_mj_mean():
    y = np.array([[0.0, 0.0, 0.0],
                  [100.0, 0.0, 50.0]], dtype=np.float32)
    p = y.copy()
    out = modified_jaccard_frame(y, p, THETA, delta_W=20.0)
    # Only one frame contributes; perfect on that one
    assert out["mean"] == pytest.approx(1.0)
    assert out["n_frames_scored"] == 1
    assert out["n_frames_total"] == 2


def test_mj_is_at_most_f1():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 200, size=(64, 3)).astype(np.float32)
    p = y + rng.normal(0, 30, size=y.shape).astype(np.float32)
    p = np.clip(p, 0, None)
    mj = modified_jaccard_frame(y, p, THETA, 20.0)["mean"]
    f1 = f1_frame(y, p, THETA)
    assert mj <= f1 + 1e-6


def test_mj_is_at_most_jaccard():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 200, size=(64, 3)).astype(np.float32)
    p = y + rng.normal(0, 30, size=y.shape).astype(np.float32)
    p = np.clip(p, 0, None)
    mj = modified_jaccard_frame(y, p, THETA, 20.0)["mean"]
    jac = jaccard_frame(y, p, THETA)
    assert mj <= jac + 1e-6


def test_per_class_mj_matches_frame_mj_when_single_category_active():
    y = np.array([[100.0, 0.0, 0.0],
                  [110.0, 0.0, 0.0]], dtype=np.float32)
    p = np.array([[105.0, 0.0, 0.0],
                  [200.0, 0.0, 0.0]], dtype=np.float32)
    pc = modified_jaccard_per_class(y, p, THETA, 20.0)
    # First frame ATP, second frame ITP -> per-class MJ = 1/2
    assert pc[0] == pytest.approx(0.5)
    assert pc[1] == 0.0
    assert pc[2] == 0.0


def test_teca_predict_zero_gives_half():
    y = np.array([[100.0, 50.0]], dtype=np.float32)
    p = np.zeros_like(y)
    assert teca_frame(y, p) == pytest.approx(0.5)
