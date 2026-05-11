"""NILMbench HuggingFace Space.

A single-frame demo of the FaustineCNN baseline. The user supplies a
``(2, 96000)`` UK-DALE V/I 6-second segment (16 kHz, float32) and the model
returns the per-category predicted active power.

The local layout expected by this script is::

    model/
        faustine_best.pt        # checkpoint compatible with FaustineCNN
        cutoffs.json            # recall-constrained per-class cutoffs
        classes.json            # ordered list of category names

    examples/
        <name>.npy              # (2, 96000) float32 V/I segments
        <name>.json             # {"truth": {"category": Watts, ...}, ...}
"""

from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
EXAMPLES_DIR = HERE / "examples"

# UK-DALE House 2 calibration constants (from calibration_house_2.cfg).
V_PER_ADC = 1.88296904357e-7
I_PER_ADC = 4.77518864497e-8
ADC_FULL_SCALE = 2 ** 31
V_FACTOR = ADC_FULL_SCALE * V_PER_ADC   # ~404.4
I_FACTOR = ADC_FULL_SCALE * I_PER_ADC   # ~102.5


# ----------------------------------------------------------------------
# Model definition (kept self-contained so the Space has no NILMbench dep)
# ----------------------------------------------------------------------
class FaustineCNN(nn.Module):
    def __init__(self, n_categories: int):
        super().__init__()
        self.conv = nn.Sequential(
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
        self.fc = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, 2 * n_categories),
        )
        self.n_categories = n_categories

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        h = self.fc(h).view(x.size(0), self.n_categories, 2)
        return F.softmax(h, dim=-1)[..., 0]


# ----------------------------------------------------------------------
# Asset loading
# ----------------------------------------------------------------------
def load_assets():
    classes = json.loads((MODEL_DIR / "classes.json").read_text())
    cutoffs = json.loads((MODEL_DIR / "cutoffs.json").read_text())["cutoffs_W"]

    model = FaustineCNN(n_categories=len(classes))
    ckpt = torch.load(MODEL_DIR / "faustine_best.pt", map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, classes, cutoffs


MODEL, CLASSES, CUTOFFS = (None, None, None)
if MODEL_DIR.exists() and (MODEL_DIR / "faustine_best.pt").exists():
    MODEL, CLASSES, CUTOFFS = load_assets()


# ----------------------------------------------------------------------
# Inference + plotting
# ----------------------------------------------------------------------
def _to_2d_image(vi_norm: np.ndarray) -> torch.Tensor:
    """Reshape a (2, 96000) waveform into a (1, 2, 240, 400) image for the CNN.

    The exact 2-D reshape is part of the FaustineCNN input contract used in the
    paper: 96000 = 240 * 400, and the model treats the result as a 2-channel
    image.
    """
    if vi_norm.shape != (2, 96000):
        raise ValueError(f"Expected (2, 96000), got {vi_norm.shape}")
    img = vi_norm.reshape(2, 240, 400).astype(np.float32)
    return torch.as_tensor(img).unsqueeze(0)


def predict(vi_norm: np.ndarray, aggregate_W: float) -> dict[str, float]:
    if MODEL is None:
        raise RuntimeError("Model checkpoint missing in this Space.")
    with torch.no_grad():
        shares = MODEL(_to_2d_image(vi_norm)).cpu().numpy().squeeze(0)
    raw = shares * float(aggregate_W)
    final = {}
    for k, cls in enumerate(CLASSES):
        cut = CUTOFFS.get(cls, 0.0)
        final[cls] = float(raw[k]) if raw[k] > cut else 0.0
    return final


def make_overview_plot(vi_norm: np.ndarray, preds: dict[str, float],
                       truth: dict[str, float] | None) -> plt.Figure:
    v = vi_norm[0] * V_FACTOR
    i = vi_norm[1] * I_FACTOR
    t = np.arange(len(v)) / 16000

    fig = plt.figure(figsize=(8.0, 6.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1.6], hspace=0.55)

    ax_v = fig.add_subplot(gs[0])
    ax_v.plot(t, v, color="#1a4f8a", lw=0.4)
    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_xlim(0, 6)
    ax_v.grid(True, linestyle=":", alpha=0.4)

    ax_i = fig.add_subplot(gs[1])
    ax_i.plot(t, i, color="#7a1a1a", lw=0.4)
    ax_i.set_ylabel("Current (A)")
    ax_i.set_xlabel("Time (s)")
    ax_i.set_xlim(0, 6)
    ax_i.grid(True, linestyle=":", alpha=0.4)

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
    ax_p.set_yticks(y_pos)
    ax_p.set_yticklabels(names)
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


def load_example(name: str) -> tuple[np.ndarray, dict[str, float] | None,
                                     float]:
    npy = EXAMPLES_DIR / f"{name}.npy"
    meta = EXAMPLES_DIR / f"{name}.json"
    vi = np.load(npy).astype(np.float32)
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
    vi, truth, agg = load_example(name)
    preds = predict(vi, agg)
    fig = make_overview_plot(vi, preds, truth)
    return fig, preds


def run_upload(file_obj, aggregate_W: float):
    if file_obj is None:
        return None, {}
    vi = np.load(file_obj.name).astype(np.float32)
    preds = predict(vi, aggregate_W)
    fig = make_overview_plot(vi, preds, None)
    return fig, preds


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    examples = list_examples()
    with gr.Blocks(title="NILMbench demo") as demo:
        gr.Markdown("# NILMbench demo\n"
                    "FaustineCNN trained on UK-DALE House 1, evaluated on a "
                    "single 6-second 16 kHz V/I segment.")
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
                                   "FLAC-normalised float32 in [-1, 1])")
                agg = gr.Slider(0, 8000, value=300, step=10,
                                label="Aggregate active power (W)")
                btn2 = gr.Button("Run", variant="primary")
                plot_b = gr.Plot()
                lab_b = gr.JSON(label="Predicted power per category (W)")
                btn2.click(run_upload, [up, agg], [plot_b, lab_b])
        gr.Markdown(
            "Source code: <https://github.com/Saharmgh/NILMbench>. "
            "If you use this demo, please cite the NILMbench paper.")
    return demo


if __name__ == "__main__":
    build_ui().launch()
