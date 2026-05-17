# How to test NILMbench's "bring your own model" workflow

Three levels of testing, from fastest (no downloads, no network) to most
realistic (full dense House-2 score).

## 1. Unit + end-to-end tests (no downloads)

These exercise the runner, the CLI `benchmark` command, and the report
renderer on a tiny synthetic dataset built on the fly by the test fixtures.

```bash
git clone https://github.com/Saharmgh/NILMbench
cd NILMbench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -e . pytest

pytest tests/ -q
```

You should see `3 passed` in a few seconds (one warning about a non-writable
NumPy array is benign — it's the mmap-backed dataset).

What this proves:
* `nilmbench.runner.run_user_model` correctly imports a user class,
  iterates the dataset, validates output shape, and saves `predictions.npz`.
* `nilmbench benchmark` (the CLI one-shot) chains run → evaluate → report
  and writes `score.json` + `report.md` in the expected schema.
* The report contains a headline score sheet and a per-category MJ table.

## 2. Local CLI on a tiny benchmark subset

Score the included random-output example model on a synthetic dataset that
the package generates internally:

```bash
# From the repo root
python - <<'PY'
import sys, os, tempfile, numpy as np
from pathlib import Path

# Build a synthetic benchmark/ split (16 frames, the same as the test fixture)
root = Path(tempfile.mkdtemp(prefix="nilmbench_demo_"))
bench = root / "benchmark"; bench.mkdir(parents=True)
rng = np.random.default_rng(0)
N, K, T = 16, 7, 96_000
x = rng.standard_normal((N, 2, T), dtype=np.float32).astype(np.float16)
y = (rng.uniform(0, 200, (N, K)).astype(np.float32) *
     (rng.random((N, K)) < 0.4).astype(np.float32))
np.save(bench / "x_vi_6s.npy", x)
np.savez_compressed(
    bench / "labels_and_index.npz",
    y_power=y, y_state=(y > 0).astype(np.int8),
    x_agg=y.sum(axis=1).astype(np.float32),
    timestamp=np.arange(N, dtype=np.int64) * 6,
    sample_idx=np.arange(N, dtype=np.int32),
    window_id=np.array([f"w{i//4:02d}" for i in range(N)]),
    class_names=np.array(["always on","cooking","dishwasher",
                          "electronics & lighting","fridge","misc",
                          "washing machine"]),
    padded_hf_segment=np.zeros(N, dtype=bool),
)
print(root)
PY
# Copy the printed path into DATA_DIR.
export DATA_DIR=...

nilmbench benchmark \
    --module examples.byom_random:RandomPredictor \
    --data $DATA_DIR \
    --out /tmp/nilmbench-demo \
    --batch-size 4 --device cpu

ls /tmp/nilmbench-demo/
cat /tmp/nilmbench-demo/report.md
```

Expected output: `predictions.npz`, `score.json`, `report.md` with the headline
sheet (MJ\_{20W} ~ 0.0–0.3 for the random predictor, perfectly fine — the
point is the pipeline works).

## 3. Real benchmark — your model on UK-DALE House 2

This is the workflow a third-party user follows.

```bash
git clone https://github.com/Saharmgh/NILMbench
cd NILMbench
pip install -e .

# 3a. Sanity check with the trivial random model (no weights needed).
#     Auto-downloads ~5 GB from HuggingFace on first run.
nilmbench benchmark \
    --module examples.byom_random:RandomPredictor \
    --data   hf:Pybunny/nilmbench-ukdale \
    --out    ./report-random/ \
    --batch-size 32

# 3b. Drop in your own model.
#     Copy examples/byom_template.py to my_model.py and replace
#     `MyModel.forward` with your architecture.
nilmbench benchmark \
    --module my_model:MyModel \
    --weights ./my_checkpoint.pt \
    --data   hf:Pybunny/nilmbench-ukdale \
    --out    ./report-mymodel/
```

Inspect `report-mymodel/report.md` — that's what to put in a paper or PR.

## 4. HuggingFace Space (no clone required)

Visit <https://huggingface.co/spaces/Pybunny/NILMbench> and open the
**Benchmark your model** tab.

1. Upload your model definition as a `.py` file. It must define a class
   (default name `Model`) that is a `torch.nn.Module` subclass implementing
   the contract in [`examples/byom_template.py`](https://github.com/Saharmgh/NILMbench/blob/main/examples/byom_template.py).
2. Upload the matching weights `.pt` file.
3. Pick `watts` or `shares` for the output kind.
4. Choose how many frames to score (50–60 000). The free CPU tier handles
   ~500 frames in 1–2 min; the full 60 000 may take 30+ min.
5. Click **Run benchmark**. The right column shows the rendered Markdown
   report and a downloadable `score.json`.

## What "passing" looks like

For the random-predictor sanity check on the real benchmark, expect
roughly:

| Metric | Random predictor | Reference baseline |
|---|---|---|
| MJ\_{20W} | ~ 0.02 | 0.504 (FaustineCNN, paper) |
| F1 | ~ 0.30 | 0.790 |
| TECA | ~ 0.40 | 0.706 |

A real submission should clearly beat the random and "predict-mean" floors.
If your model doesn't, the runner is still working — your model isn't.

## Troubleshooting

* **`ImportError: No module named 'huggingface_hub'`** — `pip install -e .`
  (or `pip install huggingface_hub`) — `huggingface_hub` is an optional
  runtime dependency only needed for the `hf:` prefix.
* **`Model output must be (B, 7)`** — your `forward` returned the wrong
  shape. Inspect with `print(model(torch.zeros(1, 2, 96000)).shape)`.
* **Out-of-memory on CPU** — drop `--batch-size`. The dense set has 60 k
  frames; small batches are fine.
* **`weights_only=False` warning** — expected (we load full checkpoints,
  not just tensors).
