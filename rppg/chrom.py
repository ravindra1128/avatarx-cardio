"""
CHROM — chrominance-based rPPG (de Haan & Jeanne, IEEE TBME 2013).
Classical, training-free, always-on Layer 1; the second, independent
projection alongside POS.

Same contract as rppg/pos.py: one ROI in, one peak-up waveform out; fusion
is beat-level only.
"""
from __future__ import annotations

import numpy as np

from ._filters import bandpass, orient_peaks_up

CHROM_WINDOW_S = 1.6


def chrom_pulse(rgb: np.ndarray, fps: float, win_s: float = CHROM_WINDOW_S,
                band: tuple[float, float] = (0.7, 4.0)) -> np.ndarray:
    """RGB trace (n, 3) -> pulse waveform (n,).

    Per overlap-added window: temporal normalisation, X = 3R - 2G,
    Y = 1.5R + G - 1.5B, S = X - (sd(X)/sd(Y)) * Y.
    """
    x = np.asarray(rgb, float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("rgb must have shape (n, 3)")
    n = x.shape[0]
    h = np.zeros(n)
    w = max(int(round(win_s * fps)), 4)
    if n < w:
        return h
    for start in range(0, n - w + 1):
        seg = x[start:start + w]
        mu = seg.mean(axis=0)
        if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
            continue
        cn = seg / mu
        xs = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
        ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
        sdy = ys.std()
        s = xs - (xs.std() / sdy) * ys if sdy > 1e-12 else xs
        h[start:start + w] += s - s.mean()
    if band is not None:
        h = bandpass(h, fps, band[0], band[1])
    return orient_peaks_up(h)
