"""Resolve a NILMbench data location, downloading from HuggingFace on demand.

A ``--data`` argument to the CLI can be:

* A local filesystem path that already contains the ``benchmark/`` split.
* The literal id ``hf:Pybunny/nilmbench-ukdale`` (or any other dataset repo
  with the same on-disk layout). The dataset is fetched once via
  :func:`huggingface_hub.snapshot_download` and cached under
  ``~/.cache/huggingface/datasets``.

The function returns a local :class:`pathlib.Path` pointing at the
resolved root.
"""

from __future__ import annotations

import os
from pathlib import Path

HF_PREFIX = "hf:"
DEFAULT_HF_REPO = "Pybunny/nilmbench-ukdale"


def resolve_data_root(spec: str | Path) -> Path:
    """Return a local Path that contains the benchmark/ split.

    ``spec`` may be:
      * a local directory; returned as-is after a quick existence check.
      * a string starting with ``hf:`` whose tail is an HF dataset repo id;
        the dataset is snapshot-downloaded and the local cache directory is
        returned.
      * the literal string ``"benchmark"`` (or no argument) which is a
        shorthand for the default repo ``Pybunny/nilmbench-ukdale``.
    """
    s = str(spec)
    if s == "benchmark":
        s = HF_PREFIX + DEFAULT_HF_REPO

    if s.startswith(HF_PREFIX):
        repo_id = s[len(HF_PREFIX):]
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Fetching from HuggingFace needs the optional "
                "`huggingface_hub` package: `pip install huggingface_hub`."
            ) from exc

        local = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[
                "benchmark/*",
                "val/*",
                "train/*",
                "summary.json",
                "README.md",
            ],
        )
        return Path(local)

    p = Path(s).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"--data {p} does not exist. Use a local path or "
            f"'hf:{DEFAULT_HF_REPO}' to auto-download."
        )
    return p
