"""Shared test fixtures.

Tiny synthetic benchmark dataset mimicking the on-disk layout of the
:class:`nilmbench.data.dataset.DenseHouseDataset` so tests can exercise the
runner / evaluator without a 5 GB download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

N_FRAMES = 16
N_CATEGORIES = 7
T_SAMPLES = 96_000
CLASSES = [
    "always on", "cooking", "dishwasher", "electronics & lighting",
    "fridge", "misc", "washing machine",
]


@pytest.fixture
def tiny_benchmark(tmp_path: Path) -> Path:
    """Build a synthetic benchmark/ split and return its parent dir."""
    bench = tmp_path / "benchmark"
    bench.mkdir(parents=True)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_FRAMES, 2, T_SAMPLES), dtype=np.float32).astype(np.float16)
    y_power = rng.uniform(0, 200, size=(N_FRAMES, N_CATEGORIES)).astype(np.float32)
    # Activate a couple of categories per frame for non-trivial F1/Jaccard.
    mask = (rng.random((N_FRAMES, N_CATEGORIES)) < 0.4)
    y_power = y_power * mask
    y_state = (y_power > 0).astype(np.int8)

    np.save(bench / "x_vi_6s.npy", x)
    np.savez_compressed(
        bench / "labels_and_index.npz",
        y_power=y_power,
        y_state=y_state,
        x_agg=y_power.sum(axis=1).astype(np.float32),
        timestamp=np.arange(N_FRAMES, dtype=np.int64) * 6,
        sample_idx=np.arange(N_FRAMES, dtype=np.int32),
        window_id=np.array([f"w{i//4:02d}" for i in range(N_FRAMES)]),
        class_names=np.array(CLASSES),
        padded_hf_segment=np.zeros(N_FRAMES, dtype=bool),
    )
    return tmp_path
