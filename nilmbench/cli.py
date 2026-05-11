"""``nilmbench`` command-line entry point.

Subcommands::

    nilmbench prepare-data ...    -- build sparse + HF splits
    nilmbench train ...           -- train one of the baseline models
    nilmbench evaluate ...        -- score predictions on the dense House-2 set
    nilmbench calibrate ...       -- learn recall-constrained per-class cutoffs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nilmbench.benchmark import evaluate_dense, save_result
from nilmbench.metrics import DEFAULT_THRESHOLDS_W
from nilmbench.postprocess import calibrate_recall_constrained_cutoffs, apply_cutoffs


def _cmd_prepare_data(args) -> int:
    from nilmbench.data.prepare import prepare_sparse_hf_6s
    prepare_sparse_hf_6s(args.sparse, args.ukdale_hf, args.out,
                         tuple(args.splits), args.dtype, args.limit)
    return 0


def _cmd_train(args) -> int:
    # The training loop is intentionally kept in scripts/train_baseline.py
    # so users can fork it without diving into the package internals.
    print("Invoke `python scripts/train_baseline.py` for the reference loop, "
          "or import nilmbench.models.build_model in your own trainer.",
          file=sys.stderr)
    return 0


def _cmd_evaluate(args) -> int:
    preds = np.load(args.predictions, allow_pickle=True)
    y_true = preds["y_true"].astype(np.float32)
    y_pred = preds["y_pred"].astype(np.float32)
    class_names = [str(c) for c in preds["class_names"]]

    cutoffs = None
    if args.cutoffs is not None:
        cutoffs = json.loads(Path(args.cutoffs).read_text())["cutoffs_W"]

    result = evaluate_dense(
        y_true_W=y_true,
        y_pred_W=y_pred,
        classes=class_names,
        model_name=args.model,
        cutoffs_W=cutoffs,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_result(result, out_dir / f"{args.model}_score.json")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_calibrate(args) -> int:
    preds = np.load(args.val_predictions, allow_pickle=True)
    y_true = preds["y_true"].astype(np.float32)
    y_pred = preds["y_pred"].astype(np.float32)
    class_names = [str(c) for c in preds["class_names"]]

    res = calibrate_recall_constrained_cutoffs(
        y_true_val_W=y_true,
        y_pred_val_W=y_pred,
        classes=class_names,
        theta_W=DEFAULT_THRESHOLDS_W,
        recall_floor=args.recall_floor,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"classes": res.classes, "cutoffs_W": res.cutoffs_W,
         "recall_floor": res.recall_floor},
        indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nilmbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare-data", help="Build sparse + HF splits")
    p_prep.add_argument("--sparse", type=Path, required=True)
    p_prep.add_argument("--ukdale-hf", type=Path, required=True)
    p_prep.add_argument("--out", type=Path, required=True)
    p_prep.add_argument("--splits", nargs="+",
                        default=["train", "val", "benchmark"])
    p_prep.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p_prep.add_argument("--limit", type=int, default=None)
    p_prep.set_defaults(func=_cmd_prepare_data)

    p_train = sub.add_parser("train", help="Train a baseline model")
    p_train.add_argument("--model", required=True,
                          choices=["faustine", "schirmer", "cold", "deepdfml"])
    p_train.add_argument("--data", type=Path, required=True)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--out", type=Path, required=True)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Score predictions on House 2")
    p_eval.add_argument("--predictions", type=Path, required=True,
                        help="NPZ with y_true, y_pred, class_names columns")
    p_eval.add_argument("--model", default="model")
    p_eval.add_argument("--cutoffs", type=Path, default=None,
                        help="Optional JSON file from `nilmbench calibrate`")
    p_eval.add_argument("--out", type=Path, required=True)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_cal = sub.add_parser("calibrate", help="Recall-constrained cutoffs")
    p_cal.add_argument("--val-predictions", type=Path, required=True,
                       help="NPZ of validation-set y_true/y_pred")
    p_cal.add_argument("--recall-floor", type=float, default=0.5)
    p_cal.add_argument("--out", type=Path, required=True)
    p_cal.set_defaults(func=_cmd_calibrate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
