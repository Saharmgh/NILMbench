"""Light-weight signal transforms used by the baselines.

Provided:
    * Spectrogram      -- STFT magnitude
    * FITPS            -- per-cycle voltage-phase folding (the COLD/Faustine
                          input representation)
    * Normalise        -- per-sample z-score
    * AsTensor         -- ensures torch tensor with given dtype
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Spectrogram(nn.Module):
    """Short-time Fourier transform magnitude of a 1-D or 2-D mini-batch."""

    def __init__(self, window_size: int, hop_size: int, n_fft: int,
                 power: bool = True, eps: float = 1e-9):
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.power = power
        self.eps = eps
        self.register_buffer("window", torch.hann_window(window_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        b, c, t = x.shape
        x = x.reshape(b * c, t)
        z = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=self.window,
            center=True,
            return_complex=True,
        )
        mag = z.abs()
        if self.power:
            mag = mag.pow(2)
        mag = mag.clamp_min(self.eps).log()
        mag = mag.reshape(b, c, mag.shape[-2], mag.shape[-1])
        return mag.squeeze(1) if squeeze else mag


class Normalise(nn.Module):
    """Per-sample channel-wise z-score normalisation."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std


class AsTensor:
    """Callable that returns a torch tensor with the given dtype."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    def __call__(self, x):
        if torch.is_tensor(x):
            return x.to(self.dtype)
        return torch.as_tensor(np.asarray(x), dtype=self.dtype)


class FITPS(nn.Module):
    """Frame-Interval Triggered Phase Synchronisation.

    Re-samples a V/I waveform onto a per-cycle grid by detecting voltage zero
    crossings, producing a ``(2, n_cycles, samples_per_cycle)`` representation.
    Used by the COLD and FaustineCNN models.
    """

    def __init__(self, samples_per_cycle: int = 200, fs: int = 16_000,
                 mains_hz: float = 50.0):
        super().__init__()
        self.spc = samples_per_cycle
        self.fs = fs
        self.mains_hz = mains_hz
        self.cycle_len = int(round(fs / mains_hz))

    @staticmethod
    def _zero_crossings(v: np.ndarray) -> np.ndarray:
        return np.where(np.diff(np.signbit(v)))[0]

    def _process_single(self, vi: np.ndarray) -> np.ndarray:
        v = vi[0]
        i = vi[1]
        zc = self._zero_crossings(v)
        if len(zc) < 2:
            return np.zeros((2, 1, self.spc), dtype=np.float32)
        cycles_v = []
        cycles_i = []
        for s, e in zip(zc[:-1], zc[1:]):
            seg_v = v[s:e]
            seg_i = i[s:e]
            if len(seg_v) < 5:
                continue
            xp = np.linspace(0, 1, len(seg_v))
            xq = np.linspace(0, 1, self.spc)
            cycles_v.append(np.interp(xq, xp, seg_v))
            cycles_i.append(np.interp(xq, xp, seg_i))
        if not cycles_v:
            return np.zeros((2, 1, self.spc), dtype=np.float32)
        v_arr = np.stack(cycles_v).astype(np.float32)
        i_arr = np.stack(cycles_i).astype(np.float32)
        return np.stack([v_arr, i_arr])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            arr = x.detach().cpu().numpy()
            return torch.as_tensor(self._process_single(arr))
        outs = [self._process_single(s.detach().cpu().numpy()) for s in x]
        n_cycles = max(o.shape[1] for o in outs)
        padded = np.zeros((len(outs), 2, n_cycles, self.spc), dtype=np.float32)
        for k, o in enumerate(outs):
            padded[k, :, : o.shape[1]] = o
        return torch.as_tensor(padded)
