"""Frame-level datasets for the NILMbench train/val/benchmark splits.

The on-disk layout produced by :func:`nilmbench.data.prepare.run` is::

    data/
    └── sparse_hf_6s/
        ├── train/
        │   ├── x_vi_6s.npy          # (N, 2, 96000) float16
        │   └── labels_and_index.npz # y_power (N, K), y_state, x_agg, ...
        ├── val/    ...
        ├── benchmark/ ...
        └── summary.json

The two dataset classes here wrap that layout and serve ``(x, y_power)`` pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SparseHFDataset(Dataset):
    """Sub-sampled 6-second 16 kHz V/I frames with per-category power labels.

    Used for training and validation (House 1) and the sparse benchmark
    (House 2).
    """

    def __init__(self, root: str | Path, transform=None):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(root)
        self.root = root
        self.x = np.load(root / "x_vi_6s.npy", mmap_mode="r")
        lab = np.load(root / "labels_and_index.npz", allow_pickle=True)
        self.y_power = lab["y_power"].astype(np.float32)
        self.y_state = lab["y_state"]
        self.x_agg = lab["x_agg"].astype(np.float32)
        self.timestamp = lab["timestamp"]
        self.sample_idx = lab["sample_idx"]
        self.window_id = lab["window_id"]
        self.class_names = [str(c) for c in lab["class_names"]]
        self.transform = transform

    def __len__(self) -> int:
        return self.y_power.shape[0]

    def __getitem__(self, idx: int):
        x = torch.as_tensor(self.x[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y_power[idx], dtype=torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class DenseHouseDataset(Dataset):
    """All labelled 6-second frames of every selected House-2 window.

    No sub-sampling: 100 windows x 600 frames = 60,000 frames by default.
    """

    def __init__(self, root: str | Path, transform=None):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(root)
        self.root = root
        self.x = np.load(root / "x_vi_6s.npy", mmap_mode="r")
        lab = np.load(root / "labels_and_index.npz", allow_pickle=True)
        self.y_power = lab["y_power"].astype(np.float32)
        self.y_state = lab["y_state"]
        self.timestamp = lab["timestamp"]
        self.sample_idx = lab["sample_idx"]
        self.window_id = lab["window_id"]
        self.padded = lab["padded_hf_segment"].astype(bool) if "padded_hf_segment" in lab.files else np.zeros(len(self.y_power), dtype=bool)
        self.class_names = [str(c) for c in lab["class_names"]]
        self.transform = transform

    def __len__(self) -> int:
        return self.y_power.shape[0]

    def __getitem__(self, idx: int):
        x = torch.as_tensor(self.x[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y_power[idx], dtype=torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def load_split_summary(root: str | Path) -> dict:
    """Read ``summary.json`` produced by the prepare-data pipeline."""
    return json.loads(Path(root, "summary.json").read_text())
