"""NILMbench HuggingFace Space.

Three tabs:

1. **Built-in example** – run the FaustineCNN baseline on a packaged
   6-second 16 kHz V/I frame from UK-DALE House 2.
2. **Upload V/I frame** – run FaustineCNN on a user-supplied single frame.
3. **Benchmark your model** – upload a ``.py`` model definition + a ``.pt``
   weights file and score it on the dense UK-DALE House 2 benchmark (full
   60,000 frames; the Space defaults to a 500-frame quick check to stay
   within the free-tier compute budget).

Model weights, classes, and recall-constrained cutoffs for the baseline are
pulled from the HF model repo ``Pybunny/nilmbench-faustine`` at startup.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, snapshot_download

# nilmbench is installed from the companion GitHub repo (see requirements.txt).
from nilmbench.runner import run_user_model
from nilmbench.benchmark import evaluate_dense
from nilmbench.io.report import render_markdown_report

HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE / "examples"
MODEL_REPO = "Pybunny/nilmbench-faustine"
DATASET_REPO = "Pybunny/nilmbench-ukdale"

# UK-DALE House 2 calibration constants (from calibration_house_2.cfg).
V_PER_ADC = 1.88296904357e-7
I_PER_ADC = 4.77518864497e-8
ADC_FULL_SCALE = 2 ** 31
V_FACTOR = ADC_FULL_SCALE * V_PER_ADC   # ~404.4
I_FACTOR = ADC_FULL_SCALE * I_PER_ADC   # ~102.5


# ----------------------------------------------------------------------
# Baseline model (self-contained for the single-frame demo)
# ----------------------------------------------------------------------
class FaustineCNN(nn.Module):
    def __init__(self, n_categories: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, 2 * n_categories),
        )
        self.n_categories = n_categories

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_layers(x).flatten(1)
        h = self.fc_layers(h).view(x.size(0), self.n_categories, 2)
        return F.softmax(h, dim=-1)[..., 0]


# ----------------------------------------------------------------------
# Asset loading (Hub)
# ----------------------------------------------------------------------
def load_assets():
    classes_path = hf_hub_download(MODEL_REPO, "classes.json")
    cutoffs_path = hf_hub_download(MODEL_REPO, "cutoffs.json")
    weights_path = hf_hub_download(MODEL_REPO, "faustine_best.pt")

    classes = json.loads(Path(classes_path).read_text())
    cutoffs = json.loads(Path(cutoffs_path).read_text())["cutoffs_W"]

    model = FaustineCNN(n_categories=len(classes))
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, classes, cutoffs


MODEL, CLASSES, CUTOFFS = load_assets()


# ----------------------------------------------------------------------
# Single-frame inference (tabs 1 and 2)
# ----------------------------------------------------------------------
def _to_2d_image(vi_norm: np.ndarray) -> torch.Tensor:
    if vi_norm.shape != (2, 96000):
        raise ValueError(f"Expected (2, 96000), got {vi_norm.shape}")
    img = vi_norm.reshape(2, 240, 400).astype(np.float32)
    return torch.as_tensor(img).unsqueeze(0)


def predict(vi_norm: np.ndarray, aggregate_W: float) -> dict[str, float]:
    with torch.no_grad():
        scores = MODEL(_to_2d_image(vi_norm)).cpu().numpy().squeeze(0)
    shares = scores / (scores.sum() + 1e-9)
    raw = shares * float(aggregate_W)
    out = {}
    for k, cls in enumerate(CLASSES):
        cut = CUTOFFS.get(cls, 0.0)
        out[cls] = float(raw[k]) if raw[k] > cut else 0.0
    return out


def make_overview_plot(vi_norm: np.ndarray, preds: dict[str, float],
                       truth: dict[str, float] | None) -> plt.Figure:
    v = vi_norm[0].astype(np.float32) * V_FACTOR
    i = vi_norm[1].astype(np.float32) * I_FACTOR
    t = np.arange(len(v)) / 16000

    fig = plt.figure(figsize=(8.0, 6.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1.6], hspace=0.55)

    ax_v = fig.add_subplot(gs[0])
    ax_v.plot(t, v, color="#1a4f8a", lw=0.4)
    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_xlim(0, 6); ax_v.grid(True, linestyle=":", alpha=0.4)

    ax_i = fig.add_subplot(gs[1])
    ax_i.plot(t, i, color="#7a1a1a", lw=0.4)
    ax_i.set_ylabel("Current (A)"); ax_i.set_xlabel("Time (s)")
    ax_i.set_xlim(0, 6); ax_i.grid(True, linestyle=":", alpha=0.4)

    ax_p = fig.add_subplot(gs[2])
    active = [(c, w) for c, w in preds.items() if w > 0]
    active.sort(key=lambda kv: -kv[1])
    if not active:
        active = [("(all categories below cutoff)", 0.0)]
    names = [c for c, _ in active]
    vals = [w for _, w in active]
    y_pos = np.arange(len(names))
    ax_p.barh(y_pos, vals, color="#a63d40", edgecolor="#222", linewidth=0.4,
              label="prediction")
    if truth is not None:
        tvals = [truth.get(c, 0.0) for c in names]
        ax_p.barh(y_pos + 0.32, tvals, height=0.32,
                  color="#1a4f8a", alpha=0.6, edgecolor="#222", linewidth=0.4,
                  label="ground truth")
    ax_p.set_yticks(y_pos); ax_p.set_yticklabels(names)
    ax_p.invert_yaxis()
    ax_p.set_xlabel("Predicted power (W)")
    ax_p.grid(True, axis="x", linestyle=":", alpha=0.4)
    if truth is not None:
        ax_p.legend(loc="lower right", frameon=False, fontsize=9)
    return fig


def list_examples() -> list[str]:
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(p.stem for p in EXAMPLES_DIR.glob("*.npy"))


def load_example(name: str):
    npy = EXAMPLES_DIR / f"{name}.npy"
    meta = EXAMPLES_DIR / f"{name}.json"
    vi = np.load(npy)
    truth = None
    aggregate = 0.0
    if meta.exists():
        m = json.loads(meta.read_text())
        truth = m.get("truth")
        aggregate = float(m.get("aggregate_W", 0.0))
    if aggregate == 0.0 and truth is not None:
        aggregate = sum(truth.values())
    return vi, truth, aggregate


def run_example(name: str):
    if not name:
        return None, {}
    vi, truth, agg = load_example(name)
    preds = predict(vi, agg)
    return make_overview_plot(vi, preds, truth), preds


def run_upload(file_obj, aggregate_W: float):
    if file_obj is None:
        return None, {}
    vi = np.load(file_obj.name)
    preds = predict(vi, aggregate_W)
    return make_overview_plot(vi, preds, None), preds


# ----------------------------------------------------------------------
# Tab 3: full benchmark, with the user's uploaded model
# ----------------------------------------------------------------------
_BENCHMARK_DATA_DIR: Path | None = None


def _ensure_benchmark_data() -> Path:
    """Snapshot-download the dense House-2 split (cached after first call)."""
    global _BENCHMARK_DATA_DIR
    if _BENCHMARK_DATA_DIR is not None:
        return _BENCHMARK_DATA_DIR
    local = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["benchmark/*", "summary.json", "README.md"],
    )
    _BENCHMARK_DATA_DIR = Path(local)
    return _BENCHMARK_DATA_DIR


def _import_user_module(file_path: Path, class_name: str):
    """Dynamically import a user-uploaded ``.py`` and return the class."""
    spec = importlib.util.spec_from_file_location("user_model_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model_module"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, class_name):
        raise AttributeError(
            f"Uploaded module has no attribute '{class_name}'. "
            f"Available: {[n for n in dir(mod) if not n.startswith('_')]}"
        )
    return getattr(mod, class_name)


def _subset_dataset(data_root: Path, max_frames: int) -> Path:
    """Make a temporary benchmark/ directory with the first N frames only.

    Lets us cap compute time on the free Space tier.
    """
    src = data_root / "benchmark"
    n_total = int(np.load(src / "x_vi_6s.npy", mmap_mode="r").shape[0])
    if max_frames >= n_total:
        return data_root  # use full set

    tmp_root = Path(tempfile.mkdtemp(prefix="nilmbench_subset_"))
    sub = tmp_root / "benchmark"
    sub.mkdir(parents=True)

    x = np.load(src / "x_vi_6s.npy", mmap_mode="r")
    np.save(sub / "x_vi_6s.npy", np.asarray(x[:max_frames]))

    lab = np.load(src / "labels_and_index.npz", allow_pickle=True)
    sliced = {}
    for k in lab.files:
        v = lab[k]
        if v.ndim >= 1 and v.shape[0] == n_total:
            sliced[k] = v[:max_frames]
        else:
            sliced[k] = v
    np.savez_compressed(sub / "labels_and_index.npz", **sliced)
    return tmp_root


def run_benchmark_upload(model_file, weights_file, class_name: str,
                          output_kind: str, max_frames: int, batch_size: int):
    """Run the user's model on the dense House-2 set and render a report."""
    if model_file is None:
        return "**Please upload a Python file defining your model.**", None
    class_name = (class_name or "Model").strip() or "Model"

    try:
        ModelCls = _import_user_module(Path(model_file.name), class_name)
    except Exception as exc:
        return (f"**Failed to import model class `{class_name}`:**\n\n"
                f"```\n{traceback.format_exc()}\n```"), None

    try:
        data_root = _ensure_benchmark_data()
    except Exception:
        return (f"**Could not download benchmark data:**\n\n"
                f"```\n{traceback.format_exc()}\n```"), None

    try:
        active_root = _subset_dataset(data_root, int(max_frames))
    except Exception:
        return (f"**Could not prepare data subset:**\n\n"
                f"```\n{traceback.format_exc()}\n```"), None

    tmpdir = Path(tempfile.mkdtemp(prefix="nilmbench_report_"))
    preds_path = tmpdir / "predictions.npz"

    try:
        # We already have the class; rebind via a temporary module name so
        # nilmbench.runner's importer can find it.
        sys.modules["__nilmbench_user__"] = sys.modules["user_model_module"]
        run = run_user_model(
            module_spec=f"__nilmbench_user__:{class_name}",
            weights_path=weights_file.name if weights_file is not None else None,
            data_root=active_root,
            out_path=preds_path,
            batch_size=int(batch_size),
            device="cpu",
            output_kind=output_kind,
            strict_load=False,
            model_name=class_name,
        )
    except Exception:
        return (f"**Model failed during inference:**\n\n"
                f"```\n{traceback.format_exc()}\n```"), None

    preds = np.load(preds_path, allow_pickle=True)
    result = evaluate_dense(
        y_true_W=preds["y_true"].astype(np.float32),
        y_pred_W=preds["y_pred"].astype(np.float32),
        classes=[str(c) for c in preds["class_names"]],
        model_name=class_name,
    )

    extra = {
        "Model class": class_name,
        "Weights file": Path(weights_file.name).name if weights_file else "(none)",
        "Frames scored": f"{run.n_frames} / 60,000",
        "Output kind": output_kind,
    }
    md = render_markdown_report(
        result,
        title=f"NILMbench report — {class_name}",
        extra=extra,
    )

    score_json_path = tmpdir / "score.json"
    score_json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    return md, str(score_json_path)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    examples = list_examples()
    with gr.Blocks(title="NILMbench") as demo:
        gr.Markdown(
            "# NILMbench\n"
            "Open benchmark for high-frequency NILM regression on UK-DALE 2015 "
            "(House 1 → House 2). Headline metric: modified Jaccard index "
            "**MJ$_{20W}$** with hybrid tolerance.\n\n"
            "Source code: <https://github.com/Saharmgh/NILMbench> · "
            "Baseline model: <https://huggingface.co/Pybunny/nilmbench-faustine> · "
            "Dataset: <https://huggingface.co/datasets/Pybunny/nilmbench-ukdale>"
        )
        with gr.Tabs():
            with gr.TabItem("Single frame · built-in example"):
                ex = gr.Dropdown(examples, label="Example frame",
                                  value=examples[0] if examples else None)
                btn = gr.Button("Run FaustineCNN", variant="primary")
                plot_a = gr.Plot()
                lab_a = gr.JSON(label="Predicted power per category (W)")
                btn.click(run_example, ex, [plot_a, lab_a])

            with gr.TabItem("Single frame · upload V/I"):
                up = gr.File(label="V/I segment (.npy, shape (2, 96000), "
                                    "FLAC-normalised float in [-1, 1])")
                agg = gr.Slider(0, 8000, value=300, step=10,
                                 label="Aggregate active power (W)")
                btn2 = gr.Button("Run FaustineCNN", variant="primary")
                plot_b = gr.Plot()
                lab_b = gr.JSON(label="Predicted power per category (W)")
                btn2.click(run_upload, [up, agg], [plot_b, lab_b])

            with gr.TabItem("Benchmark your model"):
                gr.Markdown(
                    "Upload a `.py` file defining a `torch.nn.Module` "
                    "subclass and its trained weights `.pt`. The Space "
                    "downloads the dense House-2 benchmark split from "
                    "[`Pybunny/nilmbench-ukdale`](https://huggingface.co/datasets/Pybunny/nilmbench-ukdale) "
                    "on first run (cached afterwards), runs your model on the "
                    "selected number of frames, and produces a full score "
                    "sheet.\n\n"
                    "**Model contract** (see "
                    "[`examples/byom_template.py`](https://github.com/Saharmgh/NILMbench/blob/main/examples/byom_template.py)):\n"
                    "* `forward(x)` takes `x` shape `(B, 2, 96000)` (V then I).\n"
                    "* Returns non-negative `(B, K)` per-category power in "
                    "watts. If your model emits shares in [0, 1] instead, "
                    "select **shares** below and the runner will scale by the "
                    "per-frame aggregate.\n"
                    "* If the constructor accepts a keyword named "
                    "`n_categories` (or `num_classes` / `K`), it is filled in "
                    "automatically.\n"
                )
                with gr.Row():
                    with gr.Column():
                        model_py = gr.File(label="Model definition (.py)")
                        weights = gr.File(label="Weights (.pt, optional)")
                        class_name = gr.Textbox(label="Class name",
                                                value="Model")
                        output_kind = gr.Radio(
                            ["watts", "shares"],
                            value="watts",
                            label="Output kind (watts = per-category power; "
                                  "shares = renormalise + scale by aggregate)")
                        max_frames = gr.Slider(50, 60000, value=500, step=50,
                                                label="Frames to score "
                                                      "(free CPU: 500 ≈ 1–2 min)")
                        batch_size = gr.Slider(1, 64, value=16, step=1,
                                                label="Batch size")
                        run_btn = gr.Button("Run benchmark", variant="primary")
                    with gr.Column():
                        report_md = gr.Markdown()
                        score_file = gr.File(label="Download score.json")
                run_btn.click(
                    run_benchmark_upload,
                    [model_py, weights, class_name, output_kind, max_frames, batch_size],
                    [report_md, score_file],
                )
    return demo


if __name__ == "__main__":
    # show_api=False bypasses gradio 4.44's broken JSON-schema introspector
    # which hits `additionalProperties: True` on the new BYOM-tab schemas
    # and crashes the /info endpoint. The handlers still work over the
    # normal WebSocket; only the auto-generated API docs are disabled.
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
