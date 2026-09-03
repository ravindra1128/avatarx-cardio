"""
Respiratory rate from torso motion (v0.4 T6) — REST PHASE ONLY.

Breathing moves the chest/shoulders, not the face colour, so this is a
MOTION measurement: the brightness-weighted vertical centroid of the
lower part of the frame, band-limited to the validated resting range
(8-25 breaths/min). It runs as a second pass over the recorded clip
(ingest keeps no frames and no torso region — deliberately; this module
is the only consumer of torso motion and it never touches pulse).

The recovery-phase rate can be computed with the same function but is
RESEARCH-TAGGED by the session (post-exercise breathing at the camera
has no validation on record) — it never renders.
"""
from __future__ import annotations

import numpy as np

from capture.video_reader import iter_frames

RR_BAND_BRPM = (8.0, 25.0)
MIN_SECONDS = 20.0
MIN_CONCENTRATION = 0.25
TORSO_TOP_FRACTION = 0.55        # rows below this fraction of the frame


def torso_motion_series(video_path: str) -> tuple:
    """(t, vertical brightness centroid of the torso band, fps)."""
    ts, ys = [], []
    for t, frame in iter_frames(video_path):
        g = frame.mean(axis=2)
        torso = g[int(g.shape[0] * TORSO_TOP_FRACTION):, :]
        rows = torso.mean(axis=1)
        rows = rows - rows.min()
        tot = float(rows.sum()) or 1.0
        ys.append(float((rows * np.arange(rows.size)).sum() / tot))
        ts.append(float(t))
    t = np.asarray(ts)
    y = np.asarray(ys)
    fps = 1.0 / float(np.median(np.diff(t))) if t.size > 1 else 30.0
    return t, y, fps


def respiratory_rate_from_motion(t, y, fps) -> tuple:
    """(rate_brpm | None, confidence). None outside the validated band
    or when the motion has no concentrated breathing periodicity —
    fail-closed, never a guess."""
    y = np.asarray(y, float)
    if y.size < int(MIN_SECONDS * fps) or not np.all(np.isfinite(y)):
        return None, 0.0
    # detrend OUTSIDE the validated band: a 3 s moving average sat inside
    # 8-25/min (periods 2.4-7.5 s) and suppressed slow breathers ~3x
    # relative to their 2nd harmonic, doubling reported rates (v0.4
    # review finding). 12 s passes the whole band; edge-normalized.
    from rppg._filters import moving_average_detrend
    x = moving_average_detrend(y, int(12.0 * fps))
    n = x.size
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    lo, hi = RR_BAND_BRPM[0] / 60.0, RR_BAND_BRPM[1] / 60.0
    band = (freqs >= lo) & (freqs <= hi)
    wide = (freqs >= 0.05) & (freqs <= 2.0)
    if not band.any() or float(spec[wide].sum()) <= 0:
        return None, 0.0
    f_dom = float(freqs[band][np.argmax(spec[band])])
    conc = float(spec[band & (np.abs(freqs - f_dom) <= 0.04)].sum()
                 / spec[wide].sum())
    if conc < MIN_CONCENTRATION:
        return None, round(conc, 3)
    return round(f_dom * 60.0, 1), round(conc, 3)


def resting_respiratory_rate(video_path: str):
    """The session's rr_rest source. None whenever the clip carries no
    credible breathing motion — a missing RR is a result."""
    try:
        t, y, fps = torso_motion_series(video_path)
    except (OSError, IOError):
        return None
    rate, _ = respiratory_rate_from_motion(t, y, fps)
    return rate
