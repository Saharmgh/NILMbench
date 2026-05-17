"""Data preparation entry points.

Two stages are exposed:

1. :func:`prepare_sparse_6s` -- builds 6-second per-category power labels and
   their surrounding aggregate context from the UK-DALE 2015 windows,
   producing the sparse train / val / benchmark splits.
2. :func:`prepare_sparse_hf_6s` -- for each sparse frame, extracts the matching
   16 kHz V/I segment and stores them in a ``(N, 2, 96000)`` memory-mapped
   array.

Both stages are also exposed through the ``nilmbench`` CLI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.lib.format import open_memmap

FS = 16_000
SEGMENT_SECONDS = 6
SEGMENT_SAMPLES = FS * SEGMENT_SECONDS


def _build_flac_index(root: Path) -> dict[str, Path]:
    index = {p.stem: p for p in root.rglob("vi-*.flac")}
    if not index:
        raise FileNotFoundError(f"No vi-*.flac files under {root}")
    return index


def _read_segment(path: Path, start_sample: int, dtype: str) -> np.ndarray:
    data, fs = sf.read(str(path), start=start_sample, frames=SEGMENT_SAMPLES,
                       dtype="float32", always_2d=True)
    if fs != FS:
        raise ValueError(f"Unexpected sample rate in {path}: {fs}")
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2 channels in {path}, got {data.shape[1]}")
    if data.shape[0] < SEGMENT_SAMPLES:
        pad = np.zeros((SEGMENT_SAMPLES - data.shape[0], 2), dtype=np.float32)
        data = np.vstack([data, pad])
    data = data[:SEGMENT_SAMPLES].T
    return data.astype(np.float16 if dtype == "float16" else np.float32)


def prepare_sparse_hf_6s(sparse_root: Path,
                         ukdale_hf_root: Path,
                         out_root: Path,
                         splits: tuple[str, ...] = ("train", "val", "benchmark"),
                         dtype: str = "float16",
                         limit: int | None = None) -> dict:
    """Materialise the 16 kHz V/I segments for each sparse frame."""
    sparse_root = Path(sparse_root)
    ukdale_hf_root = Path(ukdale_hf_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    flac_index = _build_flac_index(ukdale_hf_root)
    summaries = {}
    for split in splits:
        src = np.load(sparse_root / f"{split}.npz")
        n_src = src["x_agg"].shape[0]
        n = min(n_src, limit) if limit else n_src
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        x_path = out_dir / "x_vi_6s.npy"
        x = open_memmap(x_path, mode="w+", dtype=np.dtype(dtype),
                        shape=(n, 2, SEGMENT_SAMPLES))

        y_power = src["y_power"][:n].astype(np.float32)
        y_state = src["y_state"][:n].astype(np.bool_)
        x_agg = src["x_agg"][:n].astype(np.float32)
        timestamps = src["timestamp"][:n]
        sample_idx = src["sample_idx"][:n]
        window_id = src["window_id"][:n]
        class_names = src["class_names"]

        missing = []
        t0 = time.time()
        for i in range(n):
            wid = str(window_id[i])
            flac_stem = wid.replace("ukdale_", "")
            path = flac_index.get(flac_stem)
            if path is None:
                missing.append(wid)
                x[i] = 0
                continue
            start = int(sample_idx[i]) * SEGMENT_SECONDS * FS
            x[i] = _read_segment(path, start, dtype)
            if (i + 1) % 500 == 0 or i + 1 == n:
                print(f"{split}: {i + 1}/{n} segments ({time.time() - t0:.1f}s)",
                      flush=True)
        x.flush()
        np.savez_compressed(
            out_dir / "labels_and_index.npz",
            y_power=y_power, y_state=y_state, x_agg=x_agg,
            timestamp=timestamps, sample_idx=sample_idx,
            window_id=window_id, class_names=class_names,
            sample_rate=np.asarray(FS, dtype=np.int32),
            segment_seconds=np.asarray(SEGMENT_SECONDS, dtype=np.int16),
            segment_samples=np.asarray(SEGMENT_SAMPLES, dtype=np.int32),
        )
        summaries[split] = {
            "n_samples": int(n),
            "dtype": dtype,
            "shape": [int(n), 2, SEGMENT_SAMPLES],
            "missing_window_ids": sorted(set(missing)),
        }

    summary = {
        "sparse": str(sparse_root),
        "ukdale_hf": str(ukdale_hf_root),
        "output": str(out_root),
        "sample_rate": FS,
        "segment_seconds": SEGMENT_SECONDS,
        "segment_samples": SEGMENT_SAMPLES,
        "splits": summaries,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2,
                                                       sort_keys=True))
    return summary


def main_cli(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sparse", type=Path, required=True,
                   help="Root containing {train,val,benchmark}.npz from "
                        "prepare_sparse_6s.")
    p.add_argument("--ukdale-hf", type=Path, required=True,
                   help="UK-DALE 16 kHz root with vi-*.flac files.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output root for the dense HF segments.")
    p.add_argument("--splits", nargs="+",
                   default=["train", "val", "benchmark"])
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional per-split sample limit for dry runs.")
    args = p.parse_args(argv)

    prepare_sparse_hf_6s(args.sparse, args.ukdale_hf, args.out,
                         tuple(args.splits), args.dtype, args.limit)


if __name__ == "__main__":
    main_cli()
