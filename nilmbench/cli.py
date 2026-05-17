"""``nilmbench`` command-line entry point.

Subcommands::

    nilmbench prepare-data ...    -- build sparse + HF splits
    nilmbench train ...           -- train one of the baseline models
    nilmbench run ...             -- run a user model on the dense set,
                                     save predictions.npz
    nilmbench evaluate ...        -- score saved predictions on the dense set
    nilmbench benchmark ...       -- one-shot: run + evaluate + render report
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
from nilmbench.runner import run_user_model
from nilmbench.data.fetch import resolve_data_root
from nilmbench.io.report import write_report


# ----------------------------------------------------------------------
# command handlers
# ----------------------------------------------------------------------

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


def _cmd_run(args) -> int:
    data_root = resolve_data_root(args.data)
    res = run_user_model(
        module_spec=args.module,
        weights_path=args.weights,
        data_root=data_root,
        out_path=args.out,
        batch_size=args.batch_size,
        device=args.device,
        output_kind="shares" if args.shares else "watts",
        model_args=args.model_arg,
        strict_load=args.strict_load,
        model_name=args.model_name,
    )
    print(f"Wrote {res.predictions_path}  ({res.n_frames} frames, "
          f"{len(res.class_names)} categories)")
    return 0


def _load_cutoffs(path: str | Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text())["cutoffs_W"]


def _cmd_evaluate(args) -> int:
    preds = np.load(args.predictions, allow_pickle=True)
    y_true = preds["y_true"].astype(np.float32)
    y_pred = preds["y_pred"].astype(np.float32)
    class_names = [str(c) for c in preds["class_names"]]

    result = evaluate_dense(
        y_true_W=y_true,
        y_pred_W=y_pred,
        classes=class_names,
        model_name=args.model,
        cutoffs_W=_load_cutoffs(args.cutoffs),
    )
    paths = write_report(result, args.out, title=f"NILMbench report — {args.model}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")
    return 0


def _cmd_benchmark(args) -> int:
    """One-shot user-model → predictions → report."""
    data_root = resolve_data_root(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.npz"

    run = run_user_model(
        module_spec=args.module,
        weights_path=args.weights,
        data_root=data_root,
        out_path=predictions_path,
        batch_size=args.batch_size,
        device=args.device,
        output_kind="shares" if args.shares else "watts",
        model_args=args.model_arg,
        strict_load=args.strict_load,
        model_name=args.model_name,
    )

    preds = np.load(predictions_path, allow_pickle=True)
    result = evaluate_dense(
        y_true_W=preds["y_true"].astype(np.float32),
        y_pred_W=preds["y_pred"].astype(np.float32),
        classes=[str(c) for c in preds["class_names"]],
        model_name=run.model_name,
        cutoffs_W=_load_cutoffs(args.cutoffs),
    )

    extra = {
        "module": args.module,
        "weights": str(args.weights) if args.weights else "(none)",
        "data": str(data_root),
        "n_frames": run.n_frames,
        "output_kind": "shares" if args.shares else "watts",
    }
    paths = write_report(result, out_dir, title=f"NILMbench — {run.model_name}",
                         extra=extra)
    print()
    print(f"Score sheet:  {paths['json']}")
    print(f"Markdown:     {paths['markdown']}")
    print(f"Predictions:  {predictions_path}")
    print()
    print(f"  MJ_20W   = {result.MJ_20W:.4f}   (headline)")
    print(f"  F1       = {result.F1:.4f}")
    print(f"  TECA     = {result.TECA:.4f}")
    print(f"  MAE (W)  = {result.MAE_W:.2f}")
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


# ----------------------------------------------------------------------
# argparse wiring
# ----------------------------------------------------------------------

def _add_run_args(p):
    p.add_argument("--module", required=True,
                   help="module.path:ClassName pointing at a torch.nn.Module")
    p.add_argument("--weights", type=Path, default=None,
                   help="PyTorch state-dict (or full checkpoint with "
                        "'state_dict' key). Omit to evaluate uninitialised "
                        "weights for a sanity check.")
    p.add_argument("--data", default="hf:Pybunny/nilmbench-ukdale",
                   help="Path to a benchmark/ root, or 'hf:<repo>' "
                        "(default: hf:Pybunny/nilmbench-ukdale)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None,
                   help="cuda | cpu | mps; default: auto-detect")
    p.add_argument("--shares", action="store_true",
                   help="Model outputs per-category shares in [0,1] rather "
                        "than watts; runner scales by aggregate.")
    p.add_argument("--model-arg", action="append", default=[],
                   help="Positional arg to the model constructor; repeatable.")
    p.add_argument("--strict-load", type=lambda s: s.lower() != "false",
                   default=True,
                   help="Pass strict=... to load_state_dict (default true)")
    p.add_argument("--model-name", default=None,
                   help="Friendly name used in the report (default: class name)")


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

    p_run = sub.add_parser("run",
                           help="Run a user model on the dense set, save predictions.npz")
    _add_run_args(p_run)
    p_run.add_argument("--out", type=Path, required=True,
                       help="Output path for predictions.npz")
    p_run.set_defaults(func=_cmd_run)

    p_eval = sub.add_parser("evaluate", help="Score predictions on House 2")
    p_eval.add_argument("--predictions", type=Path, required=True,
                        help="NPZ with y_true, y_pred, class_names columns")
    p_eval.add_argument("--model", default="model")
    p_eval.add_argument("--cutoffs", type=Path, default=None,
                        help="Optional JSON file from `nilmbench calibrate`")
    p_eval.add_argument("--out", type=Path, required=True)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_bench = sub.add_parser("benchmark",
                             help="One-shot: run a user model and produce a full report")
    _add_run_args(p_bench)
    p_bench.add_argument("--cutoffs", type=Path, default=None,
                         help="Optional JSON from `nilmbench calibrate`")
    p_bench.add_argument("--out", type=Path, required=True,
                         help="Output directory for predictions, score.json, report.md")
    p_bench.set_defaults(func=_cmd_benchmark)

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
