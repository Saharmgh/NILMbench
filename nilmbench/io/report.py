"""Render a :class:`nilmbench.benchmark.BenchmarkResult` as JSON + Markdown."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from nilmbench.benchmark import BenchmarkResult


def _fmt(v, n: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return str(v)


def render_markdown_report(result: BenchmarkResult,
                           *,
                           title: str | None = None,
                           extra: dict | None = None) -> str:
    """Return a one-page Markdown summary of the score sheet."""
    title = title or f"NILMbench report — {result.model}"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated {ts} on {result.n_frames:,} dense House-2 frames._")
    lines.append("")

    lines.append("## Headline score sheet")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    sheet: list[tuple[str, float, int]] = [
        ("MJ\\_{20W} (headline)", result.MJ_20W, 4),
        ("MJ\\_{20%}",            result.MJ_20pct, 4),
        ("MF\\_{20%}",            result.MF_20pct, 4),
        ("F1",                    result.F1, 4),
        ("Jaccard",               result.Jaccard, 4),
        ("TECA",                  result.TECA, 4),
        ("MAE (W)",               result.MAE_W, 2),
        ("State accuracy (Hamming)", result.StateAcc_Hamming, 4),
    ]
    for name, val, n in sheet:
        lines.append(f"| {name} | {_fmt(val, n)} |")
    lines.append("")

    lines.append("## Per-category MJ\\_{20W}")
    lines.append("")
    lines.append("| Category | MJ\\_{20W} |")
    lines.append("|---|---|")
    for cls in result.classes:
        v = result.MJ_per_class_20W.get(cls)
        lines.append(f"| {cls} | {_fmt(v, 4)} |")
    lines.append("")

    if extra:
        lines.append("## Run details")
        lines.append("")
        lines.append("| Key | Value |")
        lines.append("|---|---|")
        for k, v in extra.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("## What these mean (one-line each)")
    lines.append("")
    lines.append("- **MJ\\_{20W}** — Modified Jaccard with a 20 W absolute "
                 "tolerance. Joint identification + power-estimation score. "
                 "Headline NILMbench metric.")
    lines.append("- **MJ\\_{20%}** / **MF\\_{20%}** — relative-tolerance "
                 "variants, useful as sensitivity diagnostics.")
    lines.append("- **F1 / Jaccard** — on/off classification only, blind to "
                 "wattage.")
    lines.append("- **TECA** — energy-allocation accuracy; can be inflated by "
                 "an all-zero predictor (floors at 0.5).")
    lines.append("- **MAE** — mean absolute per-category power error.")
    lines.append("- **State accuracy** — fraction of on/off bits agreeing "
                 "across (frame, category) pairs.")
    lines.append("")
    return "\n".join(lines)


def write_report(result: BenchmarkResult,
                 out_dir: str | Path,
                 *,
                 title: str | None = None,
                 extra: dict | None = None) -> dict[str, Path]:
    """Write ``score.json`` and ``report.md`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "score.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    md_path = out_dir / "report.md"
    md_path.write_text(render_markdown_report(result, title=title, extra=extra))

    return {"json": json_path, "markdown": md_path}
