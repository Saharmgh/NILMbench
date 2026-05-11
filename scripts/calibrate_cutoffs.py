"""Learn recall-constrained per-class cutoffs on the validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nilmbench.metrics import DEFAULT_THRESHOLDS_W
from nilmbench.postprocess import calibrate_recall_constrained_cutoffs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-predictions", type=Path, required=True)
    p.add_argument("--recall-floor", type=float, default=0.5)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def run() -> None:
    args = parse_args()
    data = np.load(args.val_predictions, allow_pickle=True)
    classes = [str(c) for c in data["class_names"]]
    res = calibrate_recall_constrained_cutoffs(
        y_true_val_W=data["y_true"].astype(np.float32),
        y_pred_val_W=data["y_pred"].astype(np.float32),
        classes=classes,
        theta_W=DEFAULT_THRESHOLDS_W,
        recall_floor=args.recall_floor,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"classes": res.classes, "cutoffs_W": res.cutoffs_W,
         "recall_floor": res.recall_floor},
        indent=2, sort_keys=True))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    run()
