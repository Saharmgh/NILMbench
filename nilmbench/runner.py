"""Bring-Your-Own-Model runner.

Loads a user-supplied PyTorch ``nn.Module``, iterates the dense House-2
benchmark dataset, and writes a ``predictions.npz`` that the
:mod:`nilmbench.benchmark` evaluator consumes.

Model contract
==============
Implement a ``torch.nn.Module`` whose forward pass takes a batch of
6-second 16 kHz voltage/current waveform segments and returns per-category
predictions::

    Input:  x  -- torch.Tensor, shape (B, 2, 96000), float32
              x[:, 0, :]  voltage  in volts (UK-DALE House-2 calibration)
              x[:, 1, :]  current  in amps
    Output: y  -- torch.Tensor, shape (B, K), float32, non-negative
              per-category active power in watts.

If your model emits per-category *shares* in ``[0, 1]`` instead of watts, pass
``--shares`` to the CLI; the runner will scale them by the per-frame aggregate
active power read from the benchmark dataset.

Specify your model on the command line with a ``module.path:ClassName``
pointer; any optional positional constructor arguments may be passed via
``--model-arg``. Weights are loaded with ``torch.load(weights, map_location)``
and ``model.load_state_dict(...)`` (set ``--strict-load false`` to ignore
missing/unexpected keys).
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from nilmbench.data.dataset import DenseHouseDataset


@dataclass
class RunResult:
    predictions_path: Path
    n_frames: int
    class_names: list[str]
    model_name: str


def _import_object(spec: str) -> Any:
    """Resolve ``package.module:Attr`` to a Python object."""
    if ":" not in spec:
        raise ValueError(
            f"Expected 'module.path:ClassName', got {spec!r}. "
            "Example: examples.byom_random:RandomPredictor"
        )
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"Module {mod_name!r} has no attribute {attr!r}"
        ) from exc


def _build_model(model_cls: Callable[..., torch.nn.Module],
                 n_categories: int,
                 model_args: list[str] | None,
                 model_kwargs: dict[str, Any] | None) -> torch.nn.Module:
    """Instantiate the user's model class.

    Convention: if the class accepts ``n_categories`` as a keyword argument,
    we pass it; otherwise the user is responsible for fixing the output width
    via ``--model-arg`` / ``--model-kwarg``.
    """
    args = list(model_args or [])
    kwargs = dict(model_kwargs or {})

    # Best-effort: detect a `n_categories`/`num_classes` keyword in __init__.
    import inspect
    try:
        sig = inspect.signature(model_cls)
        for name in ("n_categories", "num_categories", "num_classes",
                     "n_classes", "K"):
            if name in sig.parameters and name not in kwargs:
                kwargs[name] = n_categories
                break
    except (TypeError, ValueError):
        pass

    return model_cls(*args, **kwargs)


def _coerce_arg(s: str) -> Any:
    """Best-effort literal parse for CLI strings (int, float, bool, str)."""
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    for caster in (int, float):
        try:
            return caster(s)
        except ValueError:
            continue
    return s


def run_user_model(
    module_spec: str,
    weights_path: str | Path | None,
    data_root: str | Path,
    out_path: str | Path,
    *,
    batch_size: int = 32,
    device: str | None = None,
    output_kind: str = "watts",   # "watts" or "shares"
    model_args: list[str] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    strict_load: bool = True,
    model_name: str | None = None,
) -> RunResult:
    """Run a user model on the dense benchmark set and dump predictions.

    Parameters
    ----------
    module_spec
        ``my_package.my_module:MyModelClass`` pointer to a ``torch.nn.Module``
        subclass.
    weights_path
        Path to a PyTorch state dict (or full checkpoint with a
        ``"state_dict"`` key). May be ``None`` to evaluate a freshly
        initialised model (useful for sanity checks).
    data_root
        Path to a ``DenseHouseDataset`` root (the ``benchmark/`` directory
        produced by :func:`nilmbench.data.prepare`) or to its parent.
    out_path
        Where to write ``predictions.npz``.
    batch_size, device, output_kind, strict_load
        Runtime knobs (see CLI help).
    """
    if output_kind not in {"watts", "shares"}:
        raise ValueError(f"output_kind must be 'watts' or 'shares', got {output_kind!r}")

    data_root = Path(data_root)
    if (data_root / "benchmark").exists():
        data_root = data_root / "benchmark"

    ds = DenseHouseDataset(data_root)
    n_categories = len(ds.class_names)

    model_cls = _import_object(module_spec)
    parsed_args = [_coerce_arg(a) for a in (model_args or [])]
    model = _build_model(model_cls, n_categories,
                         parsed_args, model_kwargs)

    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=strict_load)
        if missing:
            print(f"[runner] {len(missing)} missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"[runner] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    model = model.to(device_t).eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device_t.type == "cuda"))

    preds: list[np.ndarray] = []
    aggs_for_shares = None
    with torch.inference_mode():
        for batch_idx, (x, _y) in enumerate(loader):
            x = x.to(device_t, non_blocking=True)
            out = model(x)
            if not torch.is_tensor(out):
                raise TypeError(
                    f"Model forward returned {type(out)}; expected torch.Tensor"
                )
            if out.ndim != 2 or out.shape[1] != n_categories:
                raise ValueError(
                    f"Model output must be (B, {n_categories}); got {tuple(out.shape)}"
                )
            out_np = out.float().clamp_min(0.0).cpu().numpy()
            preds.append(out_np)

    y_pred = np.concatenate(preds, axis=0)
    if y_pred.shape[0] != len(ds):
        raise RuntimeError(
            f"Got {y_pred.shape[0]} prediction rows for {len(ds)} frames"
        )

    if output_kind == "shares":
        # Scale per-category shares by aggregate power per frame.
        x_agg = ds.x_agg if hasattr(ds, "x_agg") else None
        if x_agg is None:
            # DenseHouseDataset does not currently expose x_agg; fall back to
            # the row-sum of ground truth as a stand-in for total active power.
            agg = ds.y_power.sum(axis=1, dtype=np.float32)
        else:
            agg = np.asarray(x_agg, dtype=np.float32)
        norm = y_pred / np.maximum(y_pred.sum(axis=1, keepdims=True), 1e-9)
        y_pred = norm * agg[:, None]

    y_true = ds.y_power.astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        y_true=y_true,
        y_pred=y_pred.astype(np.float32),
        class_names=np.array(ds.class_names),
        window_id=np.asarray(ds.window_id),
        sample_idx=np.asarray(ds.sample_idx),
        timestamp=np.asarray(ds.timestamp),
    )

    return RunResult(
        predictions_path=out_path,
        n_frames=int(y_pred.shape[0]),
        class_names=list(ds.class_names),
        model_name=model_name or module_spec.split(":")[-1],
    )
