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

## Quick start

```bash
git clone https://github.com/Saharmgh/nilmbench
cd nilmbench
pip install -e .

# Download UK-DALE 2015 (16 kHz V/I + 1/6 Hz submeters) and point NILMbench at it.
export UKDALE_ROOT=/path/to/UK-DALE-2015

# 1. Prepare the sparse 6-second frames and the dense House-2 evaluation set.
nilmbench prepare-data --ukdale-root $UKDALE_ROOT --out ./data/

# 2. Train a baseline (FaustineCNN, SchirmerCNN, COLD, DeepDFML).
nilmbench train --model faustine --data ./data/sparse_hf_6s --epochs 30 --out ./runs/faustine/

# 3. Score on the dense House-2 set.
nilmbench evaluate --model faustine --checkpoint ./runs/faustine/best.pt --data ./data/sparse_hf_6s/benchmark --out ./results/faustine/

# 4. Calibrate the recall-constrained post-processing cutoffs on House-1 val
#    and re-score with them.
nilmbench calibrate --predictions ./results/faustine/predictions.npz --val ./data/sparse_hf_6s/val --out ./results/faustine/

# 5. Reproduce every figure in the paper from the saved predictions.
python scripts/reproduce_figures.py --results ./results/ --out ./figures/
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
