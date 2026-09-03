"""
Shared filtering utilities for the classical rPPG extractors.

The band limits express PHYSIOLOGY (24-240 bpm pulse), not rhythm
expectations: bandpassing to a wide physiological band removes drift and
out-of-band noise without imposing any periodicity — an aperiodic AF pulse
lives entirely inside this band. scipy is used when present; a plain FFT
brick-wall fallback keeps the module importable without it.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.signal import butter, filtfilt
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def bandpass(x: np.ndarray, fps: float, lo_hz: float, hi_hz: float) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size < 16 or fps <= 0:
        return x
    nyq = fps / 2.0
    hi = min(hi_hz, 0.99 * nyq)
    lo = min(lo_hz, 0.5 * hi)
    if _HAVE_SCIPY:
        b, a = butter(3, [lo / nyq, hi / nyq], btype="band")
        return filtfilt(b, a, x)
    # FFT brick wall fallback
    spec = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(x.size, 1.0 / fps)
    spec[(f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(spec, n=x.size)


def orient_peaks_up(x: np.ndarray) -> np.ndarray:
    """Flip sign so systolic peaks point up (positive skewness).

    The extractor projections fix the pulse shape only up to sign; the beat
    detector expects peaks up. Skewness of a pulse waveform is dominated by
    its sharp systolic upstroke, so its sign identifies the orientation.
    """
    x = np.asarray(x, float)
    if x.size < 8:
        return x
    z = x - x.mean()
    sd = z.std()
    if sd <= 1e-12:
        return x
    skew = float(np.mean((z / sd) ** 3))
    return -x if skew < 0 else x


def moving_average_detrend(y, k: int):
    """y minus an edge-NORMALIZED moving average of window k samples.

    np.convolve(..., mode="same") zero-pads, so near the edges a plain
    moving average of a series with a large DC offset produces ~offset/2
    ramps that dwarf the oscillation of interest (v0.4 review finding:
    the cadence counter's peak threshold, 0.3*std, exceeded the true rep
    amplitude whenever baseline/amplitude was large). Normalizing by the
    convolved window mass makes the edge average an average of the
    samples actually present.
    """
    import numpy as _np
    y = _np.asarray(y, float)
    k = max(int(k), 1)
    kern = _np.ones(k)
    mass = _np.convolve(_np.ones(y.size), kern, mode="same")
    trend = _np.convolve(y, kern, mode="same") / mass
    return y - trend
