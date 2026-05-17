"""Runner: load a user model, iterate dataset, save predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from nilmbench.runner import run_user_model


# A tiny module accessible by dotted-path; we register it at import time.
class _ConstantPredictor(nn.Module):
    def __init__(self, n_categories: int, value_W: float = 17.0):
        super().__init__()
        self.n_categories = n_categories
        self.value_W = value_W

    def forward(self, x):
        return torch.full(
            (x.shape[0], self.n_categories),
            self.value_W, device=x.device, dtype=torch.float32,
        )


sys.modules[__name__].__dict__["_ConstantPredictor"] = _ConstantPredictor


def test_runner_basic(tiny_benchmark: Path):
    out = tiny_benchmark / "preds.npz"
    res = run_user_model(
        module_spec=f"{__name__}:_ConstantPredictor",
        weights_path=None,
        data_root=tiny_benchmark,
        out_path=out,
        batch_size=4,
        device="cpu",
    )
    assert out.exists()
    assert res.n_frames == 16
    assert len(res.class_names) == 7

    preds = np.load(out, allow_pickle=True)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    assert y_true.shape == (16, 7)
    assert y_pred.shape == (16, 7)
    # Constant predictor: every prediction should be 17 W.
    np.testing.assert_allclose(y_pred, 17.0, atol=1e-6)


def test_runner_rejects_bad_output_shape(tiny_benchmark: Path):
    class _WrongShape(nn.Module):
        def __init__(self, n_categories):
            super().__init__()
            self.n_categories = n_categories
        def forward(self, x):
            return torch.zeros(x.shape[0], self.n_categories + 1)
    sys.modules[__name__].__dict__["_WrongShape"] = _WrongShape

    with pytest.raises(ValueError, match=r"Model output must be"):
        run_user_model(
            module_spec=f"{__name__}:_WrongShape",
            weights_path=None,
            data_root=tiny_benchmark,
            out_path=tiny_benchmark / "bad.npz",
            batch_size=4,
            device="cpu",
        )
