"""NILMbench: high-frequency NILM regression benchmark."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nilmbench")
except PackageNotFoundError:
    __version__ = "0.1.0"

from nilmbench.metrics import (
    modified_jaccard_frame,
    modified_jaccard_per_class,
    modified_f1_frame,
    f1_frame,
    jaccard_frame,
    teca_frame,
)
from nilmbench.benchmark import evaluate_dense
from nilmbench.postprocess import calibrate_recall_constrained_cutoffs

__all__ = [
    "modified_jaccard_frame",
    "modified_jaccard_per_class",
    "modified_f1_frame",
    "f1_frame",
    "jaccard_frame",
    "teca_frame",
    "evaluate_dense",
    "calibrate_recall_constrained_cutoffs",
]
