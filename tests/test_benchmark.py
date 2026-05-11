"""End-to-end smoke test of the benchmark scoring loop."""

from __future__ import annotations

import numpy as np

from nilmbench.benchmark import evaluate_dense
from nilmbench.metrics import DEFAULT_THRESHOLDS_W


def test_perfect_predictions_score_one():
    rng = np.random.default_rng(42)
    classes = ["always on", "fridge", "cooking"]
    y = rng.uniform(0, 200, size=(50, 3)).astype(np.float32)
    res = evaluate_dense(y, y.copy(), classes,
                         thresholds_W={c: DEFAULT_THRESHOLDS_W[c] for c in classes})
    assert res.MJ_20W == 1.0
    assert res.F1 == 1.0
    assert res.Jaccard == 1.0


def test_cutoffs_change_score():
    classes = ["always on", "fridge", "cooking"]
    y = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
    # 8 W false positive on fridge: above theta=5 W (so it would count as a
    # support member), but below the recall-constrained cutoff of 20 W.
    p = np.array([[100.0, 8.0, 0.0]], dtype=np.float32)
    base = evaluate_dense(y, p, classes)
    with_cut = evaluate_dense(y, p, classes,
                              cutoffs_W={"always on": 3.0, "fridge": 20.0,
                                         "cooking": 50.0})
    assert with_cut.MJ_20W > base.MJ_20W
