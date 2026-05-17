"""NILMbench HuggingFace Space.

Single-frame demo of the FaustineCNN baseline. Model weights, classes, and
recall-constrained cutoffs are pulled from the HF model repo
``Pybunny/nilmbench-faustine`` at startup. Example frames are bundled with
the Space so the demo works offline of the laptop.
"""

# ----------------------------------------------------------------------
# Monkey-patch gradio_client schema walker BEFORE importing gradio.
# Newer gradio_client (auto-installed by pip's resolution of gradio>=4.44)
# crashes at startup with `TypeError: argument of type 'bool' is not
# iterable` when it walks a schema with `additionalProperties: True`
# (which gr.JSON outputs produce). This brings the / route down and
# launch() then errors with "localhost is not accessible". Returning
# "Any" for bool schemas is what the unbroken upstream code does.
# ----------------------------------------------------------------------
import gradio_client.utils as _gc_utils  # noqa: E402

_orig_get_type = _gc_utils.get_type
_orig_to_python = _gc_utils._json_schema_to_python_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    return _orig_get_type(schema)


def _safe_to_python(schema, defs):
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    return _orig_to_python(schema, defs)


_gc_utils.get_type = _safe_get_type
_gc_utils._json_schema_to_python_type = _safe_to_python

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE / "examples"
MODEL_REPO = "Pybunny/nilmbench-faustine"

# UK-DALE House 2 calibration constants (from calibration_house_2.cfg).
V_PER_ADC = 1.88296904357e-7
I_PER_ADC = 4.77518864497e-8
ADC_FULL_SCALE = 2 ** 31
V_FACTOR = ADC_FULL_SCALE * V_PER_ADC   # ~404.4
I_FACTOR = ADC_FULL_SCALE * I_PER_ADC   # ~102.5


# ----------------------------------------------------------------------
# Model (self-contained so the Space has no dependency on the nilmbench pkg)
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
# Inference + plotting
# ----------------------------------------------------------------------
def _to_2d_image(vi_norm: np.ndarray) -> torch.Tensor:
    if vi_norm.shape != (2, 96000):
        raise ValueError(f"Expected (2, 96000), got {vi_norm.shape}")
    img = vi_norm.reshape(2, 240, 400).astype(np.float32)
    return torch.as_tensor(img).unsqueeze(0)


def predict(vi_norm: np.ndarray, aggregate_W: float) -> dict[str, float]:
    with torch.no_grad():
        scores = MODEL(_to_2d_image(vi_norm)).cpu().numpy().squeeze(0)
    # FaustineCNN outputs per-category Bernoulli activations; renormalise
    # across categories to obtain shares, then scale by the aggregate.
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


# ----------------------------------------------------------------------
# Gradio handlers
# ----------------------------------------------------------------------
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
# UI
# ----------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    examples = list_examples()
    with gr.Blocks(title="NILMbench demo") as demo:
        gr.Markdown(
            "# NILMbench demo\n"
            "FaustineCNN trained on UK-DALE House 1, applied to a single "
            "6-second 16 kHz V/I segment from House 2. Predicted power is "
            "post-processed with the recall-constrained cutoffs from the paper.\n\n"
            "Source code: <https://github.com/Saharmgh/NILMbench> · "
            "Model: <https://huggingface.co/Pybunny/nilmbench-faustine>"
        )
        with gr.Tabs():
            with gr.TabItem("Built-in example"):
                ex = gr.Dropdown(examples, label="Example frame",
                                  value=examples[0] if examples else None)
                btn = gr.Button("Run", variant="primary")
                plot_a = gr.Plot()
                lab_a = gr.JSON(label="Predicted power per category (W)")
                btn.click(run_example, ex, [plot_a, lab_a])
            with gr.TabItem("Upload your own"):
                up = gr.File(label="V/I segment (.npy, shape (2, 96000), "
                                    "FLAC-normalised float in [-1, 1])")
                agg = gr.Slider(0, 8000, value=300, step=10,
                                 label="Aggregate active power (W)")
                btn2 = gr.Button("Run", variant="primary")
                plot_b = gr.Plot()
                lab_b = gr.JSON(label="Predicted power per category (W)")
                btn2.click(run_upload, [up, agg], [plot_b, lab_b])
    return demo


if __name__ == "__main__":
    build_ui().launch()
