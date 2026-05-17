"""NILMbench HuggingFace Space.

Three tabs:
1. Built-in single-frame example (FaustineCNN baseline, V/I bundled).
2. Single-frame upload (user supplies a V/I segment).
3. Benchmark your model: user uploads a .pt for the bundled
   ``DemoRegressor`` architecture (see examples/byom_demo.py in the GitHub
   repo); the Space scores it on a subset of the dense House-2 set and
   renders the same Markdown report the CLI produces.

Asset sources: model weights for the baseline come from
``Pybunny/nilmbench-faustine``; the dense benchmark split for tab 3 is
fetched once from ``Pybunny/nilmbench-ukdale`` and cached.
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
# Tab 3: full benchmark with a user-uploaded .pt for DemoRegressor
# ----------------------------------------------------------------------
# Self-contained copy of examples.byom_demo.DemoRegressor so the Space
# does not have to import the nilmbench package at module load time
# (lighter dep tree, faster cold start).
class DemoRegressor(nn.Module):
    """6 V/I stats -> linear -> softplus. Output: per-category power (W)."""
    N_FEATURES = 6

    def __init__(self, n_categories: int = 7):
        super().__init__()
        self.n_categories = n_categories
        self.head = nn.Linear(self.N_FEATURES, n_categories)

    @staticmethod
    def _feats(x):
        rms = (x * x).mean(dim=-1).clamp_min(0).sqrt()
        absmean = x.abs().mean(dim=-1)
        std = x.std(dim=-1)
        return torch.cat([rms, absmean, std], dim=-1)

    def forward(self, x):
        return F.softplus(self.head(self._feats(x)))


_BENCH_DATA_DIR = None


def _bench_data_root():
    """Cache-aware snapshot_download of the benchmark/ split."""
    global _BENCH_DATA_DIR
    if _BENCH_DATA_DIR is not None:
        return _BENCH_DATA_DIR
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id="Pybunny/nilmbench-ukdale", repo_type="dataset",
        allow_patterns=["benchmark/*", "summary.json"],
    )
    _BENCH_DATA_DIR = Path(local)
    return _BENCH_DATA_DIR


def _bench_subset(n_frames):
    """Memory-mapped read of the first n_frames frames from benchmark/.

    Filters the labels to the 7-category benchmark scoring set
    (electrical heating is listed in the file but never activates in House 2
    and is excluded by the official protocol). This matches the shape of
    the bundled byom_demo.pt and any other DemoRegressor checkpoint
    trained via examples/byom_demo.py.
    """
    BENCH_CLASSES = [
        "always on", "cooking", "dishwasher", "electronics & lighting",
        "fridge", "misc", "washing machine",
    ]
    root = _bench_data_root() / "benchmark"
    total = int(np.load(root / "x_vi_6s.npy", mmap_mode="r").shape[0])
    n = max(1, min(int(n_frames), total))
    x = np.asarray(np.load(root / "x_vi_6s.npy", mmap_mode="r")[:n],
                   dtype=np.float32)
    lab = np.load(root / "labels_and_index.npz", allow_pickle=True)
    all_cls = [str(c) for c in lab["class_names"]]
    keep = [all_cls.index(c) for c in BENCH_CLASSES if c in all_cls]
    y_all = lab["y_power"][:n].astype(np.float32)
    y = y_all[:, keep]
    classes = [all_cls[i] for i in keep]
    return x, y, classes, total


def _score_demo_pt(weights_file, n_frames):
    """Load the user's .pt into DemoRegressor and produce a Markdown report."""
    import json as _json
    if weights_file is None:
        return ("**Please upload a .pt file trained on the "
                "`DemoRegressor` architecture** (see "
                "[examples/byom_demo.py](https://github.com/Saharmgh/NILMbench/blob/main/examples/byom_demo.py)). "
                "A bundled checkpoint is at "
                "[examples/byom_demo.pt](https://github.com/Saharmgh/NILMbench/blob/main/examples/byom_demo.pt).",
                None)
    try:
        x, y_true, classes, total = _bench_subset(n_frames)
    except Exception as exc:
        return (f"**Benchmark data download failed.**\n\n```\n{exc}\n```", None)

    K = len(classes)
    model = DemoRegressor(n_categories=K)
    try:
        state = torch.load(weights_file.name, map_location="cpu",
                           weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        return (f"**Weights failed to load** (does the checkpoint match "
                f"`DemoRegressor(n_categories={K})`?).\n\n"
                f"```\n{exc}\n```", None)
    model.eval()

    with torch.inference_mode():
        x_t = torch.as_tensor(x)
        y_pred = model(x_t).cpu().numpy().astype(np.float32)

    # Use the nilmbench scorer, but installing it as a dep is heavy. Compute
    # the headline numbers inline. theta_k defaults from the paper.
    THETA = np.array([3, 50, 10, 5, 5, 10, 10], dtype=np.float32)
    if K != 7:
        THETA = np.full(K, 10.0, dtype=np.float32)

    A = y_true > THETA
    B = y_pred > THETA
    err_ok = np.abs(y_pred - y_true) <= 20.0
    union = (A | B).sum(axis=1)
    keep = union > 0
    inter = (A & B).sum(axis=1).astype(np.float32)
    correct = (A & B & err_ok).sum(axis=1).astype(np.float32)
    mj = float((correct[keep] / np.maximum(union[keep], 1)).mean()) if keep.any() else 0.0
    jacc = float((inter[keep] / np.maximum(union[keep], 1)).mean()) if keep.any() else 0.0

    tp = (A & B).sum(axis=1).astype(np.float32)
    fp = (~A & B).sum(axis=1).astype(np.float32)
    fn = (A & ~B).sum(axis=1).astype(np.float32)
    f1d = tp + 0.5 * (fp + fn)
    f1 = float(np.nanmean(np.where(f1d > 0, tp / np.maximum(f1d, 1), np.nan)))
    P = y_true.sum(axis=1)
    teca = float(np.nanmean(np.where(P > 0,
                                     1.0 - np.abs(y_true - y_pred).sum(axis=1) / np.maximum(2 * P, 1e-9),
                                     np.nan)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    per_class = []
    for k, c in enumerate(classes):
        Ak = A[:, k]; Bk = B[:, k]
        eok = np.abs(y_pred[:, k] - y_true[:, k]) <= 20.0
        unionk = (Ak | Bk).sum()
        cork = (Ak & Bk & eok).sum()
        per_class.append((c, float(cork / unionk) if unionk > 0 else 0.0))

    # Paper baselines (Table 3 of the NILMbench manuscript; full 60 000 frames).
    PAPER_BASELINES = [
        # name, MJ_20W, F1, Jaccard, TECA, MAE_W
        ("DeepDFML",                                0.316, 0.658, 0.532,  0.513, 38.64),
        ("COLD",                                    0.375, 0.714, 0.600,  0.580, 37.53),
        ("SchirmerCNN",                             0.412, 0.766, 0.667,  0.622, 45.25),
        ("FaustineCNN",                             0.504, 0.790, 0.698,  0.706, 29.64),
        ("FaustineCNN + recall-constr. cutoffs",    0.562, 0.811, 0.729,  0.739, 27.09),
        ("predict zero (trivial)",                  0.000, 0.000, 0.000,  0.500, 67.60),
        ("predict House-2 mean (trivial)",          0.227, 0.579, 0.450, -0.105, 60.70),
        ("all to 'always on' (trivial)",            0.019, 0.557, 0.412,  0.165, 76.40),
    ]

    md = []
    md.append(f"# NILMbench — uploaded .pt\n")
    md.append(f"_Your model scored on {len(x)} of {total} dense House-2 frames._\n")

    md.append("## Comparison to paper baselines")
    md.append("")
    md.append("Baselines below are from Table 3 of the NILMbench paper, computed "
              "on the full 60 000-frame dense House-2 set. **Your model is scored "
              f"on the first {len(x)} frames only** (Space free-tier compute budget); "
              "treat the comparison as directional. Use the `nilmbench` CLI locally "
              "to score on all 60 000 frames for a fair comparison.\n")
    md.append("| Model | MJ\\_{20W} | F1 | Jaccard | TECA | MAE (W) |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| **Your model (uploaded)** | **{mj:.4f}** | **{f1:.4f}** | "
              f"**{jacc:.4f}** | **{teca:.4f}** | **{mae:.2f}** |")
    for name, b_mj, b_f1, b_j, b_teca, b_mae in PAPER_BASELINES:
        md.append(f"| {name} | {b_mj:.4f} | {b_f1:.4f} | {b_j:.4f} | "
                  f"{b_teca:.4f} | {b_mae:.2f} |")
    md.append("")

    md.append("## Per-category MJ\\_{20W} (your model)\n")
    md.append("| Category | MJ_20W |")
    md.append("|---|---|")
    for c, v in per_class:
        md.append(f"| {c} | {v:.4f} |")
    md.append("")

    import tempfile as _t
    out = Path(_t.mkdtemp(prefix="nbench_report_")) / "score.json"
    out.write_text(_json.dumps({
        "MJ_20W": mj, "F1": f1, "Jaccard": jacc, "TECA": teca, "MAE_W": mae,
        "n_frames": int(len(x)), "n_total": int(total),
        "per_class_MJ_20W": dict(per_class),
    }, indent=2, sort_keys=True))
    return "\n".join(md), str(out)


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
            with gr.TabItem("Benchmark your model"):
                gr.Markdown(
                    "Upload a `.pt` checkpoint trained on the bundled "
                    "[`DemoRegressor`](https://github.com/Saharmgh/NILMbench/blob/main/examples/byom_demo.py) "
                    "architecture (V/I summary stats → linear head, 7 outputs). "
                    "A sample checkpoint is in the repo at "
                    "[`examples/byom_demo.pt`](https://github.com/Saharmgh/NILMbench/blob/main/examples/byom_demo.pt). "
                    "The Space downloads the dense House-2 benchmark from "
                    "`Pybunny/nilmbench-ukdale` on first run (cached) and "
                    "scores your model on the selected number of frames. "
                    "For full 60 000-frame scoring or your own model "
                    "architecture, use the `nilmbench` CLI from the GitHub repo."
                )
                pt = gr.File(label="Trained .pt for DemoRegressor")
                nf = gr.Slider(50, 5000, value=500, step=50,
                                label="Frames to score (free CPU; 500 ≈ 1 min)")
                bb = gr.Button("Run benchmark", variant="primary")
                rep = gr.Markdown()
                jf = gr.File(label="score.json")
                bb.click(_score_demo_pt, [pt, nf], [rep, jf])
    return demo


if __name__ == "__main__":
    build_ui().launch()
