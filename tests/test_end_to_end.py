"""End-to-end: ``nilmbench benchmark`` produces a valid report."""

from __future__ import annotations

import json
from pathlib import Path

from nilmbench.cli import main as cli_main


def test_benchmark_cli_end_to_end(tiny_benchmark: Path):
    out_dir = tiny_benchmark / "report"
    rc = cli_main([
        "benchmark",
        "--module", "examples.byom_random:RandomPredictor",
        "--data", str(tiny_benchmark),
        "--out", str(out_dir),
        "--batch-size", "4",
        "--device", "cpu",
    ])
    assert rc == 0

    assert (out_dir / "predictions.npz").exists()
    assert (out_dir / "score.json").exists()
    md = (out_dir / "report.md").read_text()
    assert "# NILMbench" in md
    assert "MJ\\_{20W} (headline)" in md
    assert "Per-category MJ" in md

    score = json.loads((out_dir / "score.json").read_text())
    for key in ("MJ_20W", "MJ_20pct", "MF_20pct", "F1", "Jaccard",
                "TECA", "MAE_W", "StateAcc_Hamming", "MJ_per_class_20W"):
        assert key in score
    for k in score:
        if k.startswith("MJ_") and k != "MJ_per_class_20W":
            assert 0.0 <= score[k] <= 1.0
    assert 0.0 <= score["F1"] <= 1.0
    assert 0.0 <= score["Jaccard"] <= 1.0
    # MAE is unbounded above; just check it's non-negative.
    assert score["MAE_W"] >= 0.0
