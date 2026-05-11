# Data preparation

NILMbench is built directly on top of the UK-DALE 2015 release. No synthetic
or augmented data is used.

The pipeline is split into two stages, both of which can be invoked through
the `nilmbench` CLI:

1. **Sparse 6-second frames.** A one-hour *window* is a contiguous block of
   600 frames sampled on the same 6-second grid as UK-DALE's submeter
   channels. We draw 1,000 House-1 training windows and 100 House-2
   benchmark windows by quota sampling over five time-of-day strata.
   Inside each window, submeters are aligned to the aggregate 6-second
   timestamps by nearest-neighbour match with a `±10 s` tolerance, and
   gaps of `≤ 30 s` are forward-filled. Raw appliance names are then
   collapsed to the 8 benchmark categories via a deterministic per-house
   mapping `φ_h`. A category is *active* in a frame when its aggregated
   power exceeds the per-class threshold `θ_k` (see
   `nilmbench.metrics.DEFAULT_THRESHOLDS_W`).

2. **High-frequency 16 kHz V/I segments.** For each sparse frame, the
   matching 16 kHz waveform segment (`96000` samples × 2 channels) is read
   from the UK-DALE FLAC archive and saved in a memory-mapped
   `x_vi_6s.npy`. Voltage and current are kept in the original
   FLAC-normalised range `[-1, 1]`; the calibration constants from
   `calibration_house_2.cfg` recover engineering units on demand.

## On-disk layout

```
data/
└── sparse_hf_6s/
    ├── train/
    │   ├── x_vi_6s.npy            # (N, 2, 96000) float16
    │   └── labels_and_index.npz   # y_power, y_state, x_agg, ...
    ├── val/    ...
    ├── benchmark/   ...
    └── summary.json
```

`labels_and_index.npz` carries:

| Key            | Shape          | Meaning                                |
| -------------- | -------------- | -------------------------------------- |
| `y_power`      | `(N, K)`       | per-category active power in watts     |
| `y_state`      | `(N, K)`       | per-category on/off label              |
| `x_agg`        | `(N, 2r+1)`    | aggregate context (`r=5`, ±66 s)       |
| `timestamp`    | `(N,)`         | Unix seconds of frame centre           |
| `sample_idx`   | `(N,)`         | 0..599 index inside the window         |
| `window_id`    | `(N,)`         | UK-DALE window identifier              |
| `class_names`  | `(K,)`         | ordered category names                 |

## Sparse vs dense

The benchmark runs in two modes:

* **Sparse benchmark**: 2,000 class-balanced steady-state frames sampled
  from House 2. Used for fast iteration during training.
* **Dense benchmark** (official scoring): all `100 × 600 = 60,000` House-2
  frames. The dense set is what is reported in the paper.

For the dense set, the `_hf_segment` is padded when fewer than 96000 raw
samples are available; the `padded_hf_segment` flag in
`labels_and_index.npz` marks those frames so they can be skipped or
weighted differently at evaluation time.
