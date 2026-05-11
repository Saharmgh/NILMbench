"""Regenerate every figure in the NILMbench paper from saved predictions.

Expected input layout::

    --results/
        cold/predictions.npz
        deepdfml/predictions.npz
        faustine/predictions.npz
        schirmer/predictions.npz
        faustine/cutoffs.json    (optional, for calibrated variant)

Output is written to ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilmbench.metrics import (
    modified_jaccard_frame,
    modified_jaccard_per_class,
    modified_f1_frame,
    f1_frame,
    jaccard_frame,
    teca_frame,
    DEFAULT_THRESHOLDS_W,
)

MODELS = ["deepdfml", "schirmer", "faustine", "cold"]
PUBLIC = {"deepdfml": "DeepDFML", "schirmer": "SchirmerCNN",
          "faustine": "FaustineCNN", "cold": "COLD"}

# Nemenyi q-values for alpha = 0.05 (standard CD tables)
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------
def load_predictions(results_root: Path) -> dict:
    out = {}
    for m in MODELS:
        d = np.load(results_root / m / "predictions.npz", allow_pickle=True)
        out[m] = {
            "y_true": d["y_true"].astype(np.float32),
            "y_pred": d["y_pred"].astype(np.float32),
            "window_id": d["window_id"],
            "padded": d["padded_hf_segment"].astype(bool) if "padded_hf_segment" in d.files else np.zeros(len(d["y_true"]), bool),
            "classes": [str(c) for c in d["class_names"]],
        }
    return out


# ----------------------------------------------------------------------
# Critical-difference diagram
# ----------------------------------------------------------------------
def per_window_mj(preds: dict) -> pd.DataFrame:
    cls = preds[MODELS[0]]["classes"]
    theta = np.array([DEFAULT_THRESHOLDS_W[c] for c in cls], dtype=np.float32)
    wids = preds[MODELS[0]]["window_id"]
    pad = preds[MODELS[0]]["padded"]
    unique = pd.unique(wids[~pad])

    cols = []
    for m in MODELS:
        out = modified_jaccard_frame(preds[m]["y_true"], preds[m]["y_pred"],
                                     theta, 20.0)
        mj = out["mj"]
        keep = (~np.isnan(mj)) & (~preds[m]["padded"])
        df = pd.DataFrame({"wid": preds[m]["window_id"][keep], "mj": mj[keep]})
        cols.append(df.groupby("wid")["mj"].mean().rename(m))
    return pd.concat(cols, axis=1).reindex(unique)


def critical_difference(k: int, n: int) -> float:
    return NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))


def make_cd_diagram(window_mat: pd.DataFrame, out_path: Path) -> None:
    ranks = (-window_mat).rank(axis=1, method="average")
    avg_rank = ranks.mean(axis=0)
    k, n = window_mat.shape[1], window_mat.shape[0]
    cd = critical_difference(k, n)

    order = avg_rank.sort_values().index.tolist()
    rvals = avg_rank.loc[order].values
    rmin, rmax = 1, k

    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.set_xlim(rmin - 0.3, rmax + 0.3)
    ax.set_ylim(-1.6, 1.0)
    ax.axis("off")
    ax.plot([rmin, rmax], [0, 0], color="#222", lw=1.0)
    for r in range(rmin, rmax + 1):
        ax.plot([r, r], [0, 0.08], color="#222", lw=1.0)
        ax.text(r, 0.22, str(r), ha="center", va="bottom", fontsize=9.5)
    ax.text((rmin + rmax) / 2, 0.50,
            r"Average rank under $\mathrm{MJ}_{20\mathrm{W}}$ (lower = better)",
            ha="center", va="bottom", fontsize=10)

    mid = (rmin + rmax) / 2
    for i, (m, r) in enumerate(zip(order, rvals)):
        ax.plot([r, r], [0, -0.25], color="#222", lw=0.8)
        x_lab = rmax + 0.15 if r > mid else rmin - 0.15
        ha = "left" if r > mid else "right"
        y_lab = -0.45 - 0.30 * i
        ax.plot([r, x_lab], [-0.25, y_lab], color="#222", lw=0.8)
        ax.text(x_lab, y_lab, f"{PUBLIC[m]}  ({r:.2f})", ha=ha, va="center",
                fontsize=10)

    cd_x1 = rmin + cd
    ax.plot([rmin, cd_x1], [0.85, 0.85], color="#222", lw=2.0)
    ax.text((rmin + cd_x1) / 2, 0.95, f"CD = {cd:.3f}",
            ha="center", va="bottom", fontsize=9.5)

    # Maximal-clique bars for non-significantly-different groups
    groups = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and rvals[j + 1] - rvals[i] < cd:
            j += 1
        if j > i:
            groups.append((rvals[i], rvals[j]))
        i += 1
    for idx, (lo, hi) in enumerate(groups):
        y = -0.05 - 0.07 * idx
        ax.plot([lo - 0.04, hi + 0.04], [y, y], color="#a63d40", lw=3.5,
                solid_capstyle="butt")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Bump chart
# ----------------------------------------------------------------------
def make_bump_chart(preds: dict, out_path: Path) -> None:
    cls = preds[MODELS[0]]["classes"]
    theta = np.array([DEFAULT_THRESHOLDS_W[c] for c in cls], dtype=np.float32)
    rows = []
    for m in MODELS:
        y_t = preds[m]["y_true"]
        y_p = preds[m]["y_pred"]
        rows.append({
            "model": PUBLIC[m],
            "MJ20W": modified_jaccard_frame(y_t, y_p, theta, 20.0)["mean"],
            "MJ20p": modified_f1_frame(y_t, y_p, theta, 0.2)["mean"],
            "MF20p": modified_f1_frame(y_t, y_p, theta, 0.2)["mean"],
            "F1": f1_frame(y_t, y_p, theta),
            "J": jaccard_frame(y_t, y_p, theta),
            "TECA": teca_frame(y_t, y_p),
        })
    df = pd.DataFrame(rows).set_index("model")
    cols = ["MJ20W", "MJ20p", "MF20p", "F1", "J", "TECA"]
    rank = df[cols].rank(ascending=False, method="min").astype(int)

    colors = {"FaustineCNN": "#a63d40", "SchirmerCNN": "#1f6f3a",
              "COLD": "#1a4f8a", "DeepDFML": "#7a4f10"}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(cols))
    for model in rank.index:
        y = rank.loc[model].values
        ax.plot(x, y, "-o", color=colors[model], lw=2.4, markersize=8,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.text(x[-1] + 0.10, y[-1], model, color=colors[model],
                fontsize=10, fontweight="bold", va="center")
        ax.text(x[0] - 0.10, y[0], model, color=colors[model],
                fontsize=10, fontweight="bold", va="center", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\mathrm{MJ}_{20\mathrm{W}}$", r"$\mathrm{MJ}_{20\%}$",
                       r"$\mathrm{MF}_{20\%}$", "F1", "Jaccard", "TECA"],
                      fontsize=10)
    ax.set_yticks(range(1, rank.values.max() + 1))
    ax.set_yticklabels([f"#{i}" for i in range(1, rank.values.max() + 1)])
    ax.invert_yaxis()
    ax.set_ylabel("Rank (1 = best)", fontsize=10.5)
    ax.set_xlim(-1.1, len(cols) - 0.1 + 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Per-model x per-class heatmap
# ----------------------------------------------------------------------
def make_heatmap(preds: dict, out_path: Path) -> None:
    cls = preds[MODELS[0]]["classes"]
    theta = np.array([DEFAULT_THRESHOLDS_W[c] for c in cls], dtype=np.float32)
    mat = np.zeros((len(MODELS), len(cls)), dtype=np.float32)
    for r, m in enumerate(MODELS):
        mat[r] = modified_jaccard_per_class(preds[m]["y_true"],
                                            preds[m]["y_pred"], theta, 20.0)
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cls)))
    ax.set_xticklabels([c.replace(" ", "\n", 1) for c in cls], fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([PUBLIC[m] for m in MODELS])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] < 0.5 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=9.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\mathrm{MJ}_{20\mathrm{W}}$", fontsize=10)
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    preds = load_predictions(args.results)
    win = per_window_mj(preds)

    make_cd_diagram(win, args.out / "cd_diagram.pdf")
    make_bump_chart(preds, args.out / "bump_chart.pdf")
    make_heatmap(preds, args.out / "class_model_heatmap.pdf")
    print(f"Wrote 3 figures to {args.out}")


if __name__ == "__main__":
    main()
