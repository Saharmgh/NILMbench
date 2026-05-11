"""Score a checkpoint on the dense House-2 benchmark.

Usage::

    python scripts/evaluate_dense.py \
        --checkpoint runs/faustine/best.pt \
        --data data/sparse_hf_6s/benchmark \
        --out results/faustine/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nilmbench.benchmark import evaluate_dense, save_result
from nilmbench.data import DenseHouseDataset
from nilmbench.models import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True,
                   help="Dense House-2 split root.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cutoffs", type=Path, default=None,
                   help="Optional JSON with recall-constrained cutoffs.")
    return p.parse_args()


def run() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    classes = ckpt["classes"]

    model = build_model(ckpt["model"], n_categories=len(classes)).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = DenseHouseDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)

    ys, yps = [], []
    with torch.no_grad():
        for x, y in dl:
            scores = model(x.to(args.device))
            power = model.shares_to_power(scores, y.sum(dim=1).to(args.device))
            yps.append(power.cpu().numpy())
            ys.append(y.numpy())

    y_true = np.concatenate(ys).astype(np.float32)
    y_pred = np.concatenate(yps).astype(np.float32)

    np.savez_compressed(args.out / "predictions.npz",
                        y_true=y_true, y_pred=y_pred,
                        class_names=np.array(classes),
                        window_id=ds.window_id,
                        sample_idx=ds.sample_idx,
                        timestamp=ds.timestamp,
                        padded_hf_segment=ds.padded)

    cutoffs = None
    if args.cutoffs is not None:
        cutoffs = json.loads(args.cutoffs.read_text())["cutoffs_W"]

    result = evaluate_dense(y_true, y_pred, classes,
                            model_name=ckpt["model"], cutoffs_W=cutoffs)
    save_result(result, args.out / f"{ckpt['model']}_score.json")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
