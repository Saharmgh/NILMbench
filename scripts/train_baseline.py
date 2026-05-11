"""Reference training loop for the NILMbench baselines.

Run::

    python scripts/train_baseline.py \
        --model faustine \
        --data ./data/sparse_hf_6s \
        --epochs 30 \
        --out ./runs/faustine/

The script is intentionally short; it is meant as a reproducible reference
and a fork starting point, not a full training framework.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nilmbench.data import SparseHFDataset
from nilmbench.models import build_model
from nilmbench.metrics import modified_jaccard_frame, DEFAULT_THRESHOLDS_W


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   choices=["faustine", "schirmer", "cold", "deepdfml"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def run() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    train_ds = SparseHFDataset(args.data / "train")
    val_ds = SparseHFDataset(args.data / "val")
    n_categories = train_ds.y_power.shape[1]

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, n_categories=n_categories).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCELoss()

    classes = train_ds.class_names
    theta = np.array([DEFAULT_THRESHOLDS_W[c] for c in classes], dtype=np.float32)

    best_mj = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for x, y in train_dl:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            shares_true = y / (y.sum(dim=1, keepdim=True) + 1e-9)
            # BCE is per-category, matching the per-category Bernoulli output
            # head of FaustineCNN; softmax-output models simply see this as
            # a categorical share target.
            shares_pred = model(x)
            loss = loss_fn(shares_pred, shares_true)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        # Validation MJ_20W
        model.eval()
        ys, yps = [], []
        with torch.no_grad():
            for x, y in val_dl:
                scores = model(x.to(args.device))
                power = model.shares_to_power(scores, y.sum(dim=1).to(args.device))
                yps.append(power.cpu().numpy())
                ys.append(y.numpy())
        y_true = np.concatenate(ys).astype(np.float32)
        y_pred = np.concatenate(yps).astype(np.float32)
        mj = modified_jaccard_frame(y_true, y_pred, theta, 20.0)["mean"]

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_MJ_20W": float(mj),
                        "elapsed_s": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)

        if mj > best_mj:
            best_mj = mj
            torch.save({"state_dict": model.state_dict(),
                        "model": args.model,
                        "classes": classes,
                        "epoch": epoch,
                        "val_MJ_20W": float(mj)},
                       args.out / "best.pt")

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val MJ_20W = {best_mj:.4f}; checkpoint at {args.out / 'best.pt'}")


if __name__ == "__main__":
    run()
