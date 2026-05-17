"""A small trained NILM regressor you can drop into ``nilmbench benchmark``.

The architecture is intentionally tiny so the weights stay small (≈ 1 kB)
and you can read it in one screen:

* Compute six summary statistics per frame (RMS, mean |x|, std) of the
  voltage and current waveforms.
* Feed them through a single linear layer to per-category power (W).
* Softplus to enforce non-negativity.

It is *not* meant to match the FaustineCNN paper baseline — it's the
smallest model that produces non-trivial benchmark numbers so users can
verify the full BYOM pipeline (CLI + HF dataset download + report
generation) works on their machine.

Train it yourself in ~30 seconds on the HF dataset:

    python examples/byom_demo.py train \
        --train hf:Pybunny/nilmbench-ukdale \
        --out   examples/byom_demo.pt

Or just score the bundled checkpoint:

    nilmbench benchmark \
        --module  examples.byom_demo:DemoRegressor \
        --weights examples/byom_demo.pt \
        --data    hf:Pybunny/nilmbench-ukdale \
        --out     ./demo-report/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemoRegressor(nn.Module):
    """Six hand-crafted V/I summary stats → linear → softplus."""

    INPUT_LENGTH = 96_000   # 6 s × 16 kHz
    N_FEATURES = 6          # rms_v, rms_i, absmean_v, absmean_i, std_v, std_i

    def __init__(self, n_categories: int = 7):
        super().__init__()
        self.n_categories = n_categories
        self.head = nn.Linear(self.N_FEATURES, n_categories)

    @staticmethod
    def _features(x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T)
        rms = (x * x).mean(dim=-1).clamp_min(0).sqrt()  # (B, 2)
        absmean = x.abs().mean(dim=-1)                  # (B, 2)
        std = x.std(dim=-1)                             # (B, 2)
        return torch.cat([rms, absmean, std], dim=-1)   # (B, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.head(self._features(x)))  # (B, K) ≥ 0 W


# ----------------------------------------------------------------------
# Training (used to produce the bundled byom_demo.pt; rerun any time)
# ----------------------------------------------------------------------
def _train(train_path: str, out_path: Path, epochs: int = 3,
           batch_size: int = 64, lr: float = 1e-2,
           split: str = "train") -> None:
    from nilmbench.data.dataset import SparseHFDataset
    from torch.utils.data import DataLoader, random_split

    # Fetch only the split we need (val/ is 1/10 the size of train/ -- ideal
    # for a quick demo run).
    if train_path.startswith("hf:"):
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=train_path[3:],
            repo_type="dataset",
            allow_patterns=[f"{split}/*", "summary.json"],
        )
        root = Path(local)
    else:
        root = Path(train_path).expanduser()

    candidate = root / split
    if not candidate.exists():
        raise FileNotFoundError(f"No {split}/ split under {root}")
    ds = SparseHFDataset(candidate)

    # House 1 has 8 categories; the dense House 2 benchmark scores 7
    # (electrical heating never activates in House 2 and is excluded).
    # Filter the training labels to the benchmark's 7-class subset so the
    # saved weights load cleanly into the benchmark runner.
    BENCH_CLASSES = [
        "always on", "cooking", "dishwasher", "electronics & lighting",
        "fridge", "misc", "washing machine",
    ]
    keep_idx = [ds.class_names.index(c) for c in BENCH_CLASSES]
    ds.y_power = ds.y_power[:, keep_idx]
    ds.class_names = BENCH_CLASSES
    n_categories = len(ds.class_names)
    print(f"[train] {len(ds)} frames, {n_categories} classes "
          f"(filtered to benchmark): {ds.class_names}")

    # Tiny train/val cut so the loop is honest.
    val_n = max(64, len(ds) // 10)
    train_n = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0)

    model = DemoRegressor(n_categories=n_categories)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    for ep in range(epochs):
        model.train()
        tloss = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tloss += loss.item() * x.shape[0]
        tloss /= train_n

        model.eval()
        vloss = 0.0
        with torch.inference_mode():
            for x, y in val_dl:
                vloss += loss_fn(model(x), y).item() * x.shape[0]
        vloss /= val_n
        print(f"[train] epoch {ep+1}/{epochs}  train={tloss:.2f}  val={vloss:.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[train] wrote {out_path}")


def _main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--train", required=True,
                    help="Local path to a SparseHFDataset root, or "
                         "'hf:Pybunny/nilmbench-ukdale'")
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--split", default="val",
                    help="Which HF split to train on. Default 'val' "
                         "(1000 frames, ~500 MB) for a quick demo; use "
                         "'train' for the full 10000-frame split (~5 GB).")
    args = p.parse_args()
    if args.cmd == "train":
        _train(args.train, args.out, args.epochs, args.batch_size,
               split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
