"""Baseline NILM regression models.

All four baselines share the same I/O contract:
    Input:  ``x`` of shape ``(B, 2, T)`` or ``(B, 2, F, T)`` (V/I waveform or
            its complex spectrogram, depending on the model).
    Output: per-category power *shares* in ``[0, 1]``, summing to one.

The actual per-category active power is recovered at evaluation time by
multiplying the shares by the predicted total active power.
"""

from nilmbench.models.faustine import FaustineCNN
from nilmbench.models.schirmer import SchirmerCNN
from nilmbench.models.cold import COLD
from nilmbench.models.deepdfml import DeepDFML

MODEL_REGISTRY = {
    "faustine": FaustineCNN,
    "schirmer": SchirmerCNN,
    "cold": COLD,
    "deepdfml": DeepDFML,
}


def build_model(name: str, n_categories: int, **kwargs):
    """Factory: ``build_model("faustine", n_categories=7)``."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](n_categories=n_categories, **kwargs)


__all__ = ["FaustineCNN", "SchirmerCNN", "COLD", "DeepDFML",
           "MODEL_REGISTRY", "build_model"]
