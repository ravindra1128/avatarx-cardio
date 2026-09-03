"""
POS — plane-orthogonal-to-skin rPPG (Wang, den Brinker, Stuijk, de Haan,
IEEE TBME 2017). Classical, training-free, always-on Layer 1.

Operates on ONE ROI's mean-RGB trace and returns one pulse waveform; ROIs
are fused later at the beat level, never at the pixel or waveform level.

The output is oriented systolic-peak-UP (positive skewness) so the beat
detector's peak convention holds regardless of the projection's sign.
"""
from __future__ import annotations

import numpy as np

from ._filters import bandpass, orient_peaks_up

POS_WINDOW_S = 1.6            # the paper's 32 frames at 20 fps


def pos_pulse(rgb: np.ndarray, fps: float, win_s: float = POS_WINDOW_S,
              band: tuple[float, float] = (0.7, 4.0)) -> np.ndarray:
    """RGB trace (n, 3) -> pulse waveform (n,).

    Overlap-added sliding windows; within each window the trace is
    temporally normalised (divided by its mean), projected on
    [[0, 1, -1], [-2, 1, 1]], and the two projections are alpha-combined.
    `band=None` skips the final bandpass — the SQI computes in-band
    fractions and needs the unfiltered projection.
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
        cn = seg / mu - 1.0
        s1 = cn[:, 1] - cn[:, 2]                      # G - B
        s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]     # G + B - 2R
        sd2 = s2.std()
        p = s1 + (s1.std() / sd2) * s2 if sd2 > 1e-12 else s1
        h[start:start + w] += p - p.mean()
    if band is not None:
        h = bandpass(h, fps, band[0], band[1])
    return orient_peaks_up(h)


def orient_rois_consistently(waves: dict, min_abs_skew: float = 0.15) -> dict:
    """Give all ROI waveforms ONE polarity (v0.1.2, audit-confirmed defect).

    POS output has a single physical polarity across skin regions, but
    `orient_peaks_up` decides per ROI from that ROI's own skewness — on a
    weak ROI that skew is noise-level and the ROI can be inverted, so its
    'peaks' are systolic troughs half a period off the true beats and can
    never fuse with the good ROIs. Reference = the ROI with the largest
    |skewness| (the most clearly pulse-shaped); every other ROI is flipped
    to correlate positively with it. ROIs whose own skew is decisive
    (|skew| >= min_abs_skew) and already agree are left alone.
    """
    out = {}
    names = list(waves)
    if not names:
        return out
    def _skew(x):
        z = np.asarray(x, float); z = z - z.mean(); sd = z.std()
        return float(np.mean((z / sd) ** 3)) if sd > 1e-12 else 0.0
    sk = {r: _skew(waves[r]) for r in names}
    ref = max(names, key=lambda r: abs(sk[r]))
    ref_w = np.asarray(waves[ref], float)
    if sk[ref] < 0:                                   # reference itself peak-up
        ref_w = -ref_w
    out[ref] = ref_w
    for r in names:
        if r == ref:
            continue
        w = np.asarray(waves[r], float)
        if w.size != ref_w.size or w.std() <= 1e-12:
            out[r] = w
            continue
        c = float(np.corrcoef(ref_w, w)[0, 1])
        out[r] = -w if c < 0 else w
    return out
