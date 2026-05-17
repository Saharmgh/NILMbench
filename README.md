# NILMbench

A reproducible benchmark for high-frequency NILM (Non-Intrusive Load Monitoring)
regression models. NILMbench evaluates a model's ability to predict per-category
active power from a 16 kHz aggregate voltage/current waveform, on a strict
cross-household split of UK-DALE 2015 (House 1 → House 2).

The benchmark introduces the **modified Jaccard index** `MJ_{20W}` as the headline
metric, which couples appliance identification with a power tolerance so that
predicting the right appliance with the wrong wattage is not rewarded.

## Companion artifacts

| Where             | What                                                               |
| ----------------- | ------------------------------------------------------------------ |
| **This repo**     | Code (package, scripts, tests, figure-reproduction)                |
| [🤗 Dataset](https://huggingface.co/datasets/Pybunny/nilmbench-ukdale) | Pre-processed UK-DALE 16 kHz V/I splits (5 GB)         |
| [🤗 Model](https://huggingface.co/Pybunny/nilmbench-faustine)         | Trained FaustineCNN checkpoint + recall-constrained cutoffs |
| [🤗 Space](https://huggingface.co/spaces/Pybunny/NILMbench)           | Gradio demo: classify a single 6-s V/I frame in the browser |

## Quick start — benchmark your model in one command

NILMbench is built so a third-party model can be scored end-to-end without
having to re-implement the data pipeline or the metrics. The bundled
benchmark data is downloaded from the HuggingFace dataset
[`Pybunny/nilmbench-ukdale`](https://huggingface.co/datasets/Pybunny/nilmbench-ukdale)
automatically on first run.

```bash
git clone https://github.com/Saharmgh/NILMbench
cd NILMbench
pip install -e .

# Score a model in one shot.
# (RandomPredictor is the toy example shipped in examples/byom_random.py.)
nilmbench benchmark \
    --module examples.byom_random:RandomPredictor \
    --data hf:Pybunny/nilmbench-ukdale \
    --out  ./report/
```

That call:

1. snapshot-downloads the dense House-2 benchmark split from HuggingFace,
2. instantiates your model class, loads weights if given,
3. iterates every labelled 6-second 16 kHz V/I frame,
4. writes `report/predictions.npz`, `report/score.json`, and `report/report.md`.

## Bring your own model

Implement a `torch.nn.Module` whose forward pass conforms to the contract:

```
Input  x : torch.Tensor, shape (B, 2, 96000), float32
           x[:, 0, :]  voltage trace (V)
           x[:, 1, :]  current trace (A)
Output y : torch.Tensor, shape (B, K), float32, non-negative
           per-category active power in WATTS.
```

`K` is the number of scored appliance categories (7 on UK-DALE House 2). The
runner inspects your `__init__` signature and, if it accepts a keyword
called `n_categories` (or `num_categories`, `num_classes`, `n_classes`, `K`),
fills it in automatically — so you don't have to hard-code the count.

If your model emits per-category *shares* in `[0, 1]` rather than watts, pass
`--shares` and the runner will rescale by the per-frame aggregate active
power read from the benchmark dataset.

A minimal starter template lives in
[`examples/byom_template.py`](examples/byom_template.py); a trivial
random-output example in
[`examples/byom_random.py`](examples/byom_random.py) lets you verify the
pipeline works before you wire up your own architecture.

```bash
# Score your model end-to-end.
nilmbench benchmark \
    --module     my_pkg.my_module:MyModel \
    --weights    ./my_checkpoint.pt \
    --data       hf:Pybunny/nilmbench-ukdale \
    --batch-size 32 \
    --device     cuda \
    --out        ./report/

# Two-step variant: run once, score (and re-score) cheaply afterwards.
nilmbench run \
    --module my_pkg.my_module:MyModel \
    --weights ./my_checkpoint.pt \
    --data    hf:Pybunny/nilmbench-ukdale \
    --out     ./report/predictions.npz

nilmbench evaluate \
    --predictions ./report/predictions.npz \
    --model       MyModel \
    --out         ./report/
```

The output `report/report.md` contains the headline score sheet
(`MJ_{20W}`, `MJ_{20%}`, `MF_{20%}`, F1, Jaccard, TECA, MAE, state accuracy)
and the per-category `MJ_{20W}` breakdown, ready to drop into a paper or PR.

## Reproducing the paper's baselines from scratch

```bash
# Optional: rebuild the splits yourself from raw UK-DALE 2015.
export UKDALE_ROOT=/path/to/UK-DALE-2015
nilmbench prepare-data --ukdale-root $UKDALE_ROOT --out ./data/

# Train one of the four reference baselines.
nilmbench train --model faustine --data ./data/sparse_hf_6s \
                --epochs 30 --out ./runs/faustine/

# Calibrate recall-constrained post-processing cutoffs on House-1 val.
nilmbench calibrate \
    --val-predictions ./runs/faustine/val_predictions.npz \
    --recall-floor 0.5 \
    --out ./runs/faustine/cutoffs.json

# Score with cutoffs applied.
nilmbench benchmark \
    --module nilmbench.models.faustine:FaustineCNN \
    --weights ./runs/faustine/best.pt \
    --shares \
    --cutoffs ./runs/faustine/cutoffs.json \
    --out ./results/faustine/
```

## Repository layout

```
nilmbench/
├── nilmbench/              # importable package
│   ├── models/             # FaustineCNN, SchirmerCNN, COLD, DeepDFML
│   ├── data/               # UK-DALE loaders, FITPS transforms, data-prep scripts
│   ├── metrics.py          # MJ, MF, F1, Jaccard, TECA, per-class variants
│   ├── postprocess.py      # recall-constrained per-category cutoffs
│   ├── benchmark.py        # dense House-2 evaluation loop
│   └── cli.py              # `nilmbench` CLI
├── scripts/
│   ├── train_baseline.py
│   ├── evaluate_dense.py
│   ├── calibrate_cutoffs.py
│   └── reproduce_figures.py
├── tests/                  # metric correctness, benchmark protocol
├── figures/                # standalone figure-generation scripts
├── huggingface_space/      # Gradio demo (separate deployment)
├── docs/                   # extended documentation
├── LICENSE
├── CITATION.cff
└── pyproject.toml
```

## Headline results (dense House 2, 60 000 frames)

| Model       | `MJ_{20W}` | `MJ_{20%}` | `MF_{20%}` | F1     | Jaccard | MAE (W) |
| ----------- | ---------- | ---------- | ---------- | ------ | ------- | ------- |
| DeepDFML    | 0.316      | 0.107      | 0.124      | 0.658  | 0.532   | 38.46   |
| SchirmerCNN | 0.412      | 0.156      | 0.176      | 0.765  | 0.667   | 45.04   |
| COLD        | 0.375      | 0.141      | 0.166      | 0.714  | 0.600   | 37.28   |
| FaustineCNN | **0.504**  | **0.239**  | **0.264**  | **0.790** | **0.698** | **29.38** |
| FaustineCNN + recall-constrained cutoffs | **0.559** | **0.263** | **0.282** | **0.839** | **0.772** | **26.70** |

See the paper for the per-class breakdown, statistical comparison (CD diagram),
and metric-sensitivity analysis.

## Citation

```bibtex
@article{nilmbench2026,
  title   = {NILMbench: A Novel Benchmark For High-Frequency NILM Regression Models},
  author  = {Moghimian Hoosh, Sahar and Kamyshev, Ilia and Penuela, Javier and Mahmood, Farhat and Al-Ansari, Tareq and Ouerdane, Henni},
  journal = {(submitted)},
  year    = {2026},
}
```

## License

MIT, see [LICENSE](LICENSE).

## Contributors

The repository follows the authorship of the paper. See `CITATION.cff`.
