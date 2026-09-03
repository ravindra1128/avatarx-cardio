"""
Launch-level overrides for the measure service (split out of
measure_api.py to keep that module under the 500-line rule).

Every override here is applied ONCE at import, from the environment, and
is echoed in /healthz and in every response under `launch_overrides` -
so a loosened or experimental service can never be mistaken for a stock
one. None of them are defaults. See each function's docstring for the
measured evidence behind it, including the negative results.
"""
from __future__ import annotations

import os

# ---------------------------------------------- launch-level tolerances
# Some ingest gates read MODULE CONSTANTS, not the YAML (the YAML value of
# the same name governs the live demo's readiness check, a different
# surface). They cannot be overridden per request: with MAX_CONCURRENT
# workers a per-request monkeypatch would race across scans. So they are
# applied ONCE at launch from the environment, and echoed in /healthz and
# in every response under `launch_overrides`, so a loosened service can
# never be mistaken for a stock one.
#
# The one that bites phones: MAX_COLLAPSED_INTERVAL_FRACTION. Measured on a
# real mobile capture: 25 of 1184 intervals (2.11%) were under half the
# median frame time — ordinary mobile frame-rate jitter — against a 0.02
# floor. Smoothing the timestamps instead would fabricate timing for a
# beat-timing pipeline, which is precisely what the gate exists to stop;
# raising the floor is a judgement the pipeline owner should make, and
# this knob only lets that judgement be measured before it is made.
LAUNCH_OVERRIDES = {}


def _narrowband_orient_factory(fps: float):
    """Decide each ROI's polarity at the CARDIAC frequency, not broadband.

    rppg.pos.orient_rois_consistently correlates each ROI against a
    reference across the whole 0.7-3.0 Hz band. Measured on real browser
    captures that broadband correlation was 0.04-0.36 (a coin toss at the
    low end) while the same pairs correlated at |r| 0.63-0.71 in a narrow
    band around the shared pulse peak. Deciding the sign where the
    evidence is 3-17x stronger moved cross-ROI coherence 0.065 -> 0.186 and
    timing precision 56 -> 28 ms on one recording. On a SECOND recording
    (a phone) it went the other way: coherence 0.174 -> 0.079, timing
    40 -> 52 ms. Two recordings, opposite signs. This is NOT a fix; it is
    an experiment knob. Leave it OFF unless you are measuring it."""
    import numpy as np
    from scipy.signal import butter, filtfilt, welch

    def orient(waves, min_abs_skew=0.15):
        names = list(waves)
        if not names:
            return {}
        n = len(next(iter(waves.values())))
        peaks = []
        for r in names:
            fr, S = welch(waves[r], fps, nperseg=min(512, n))
            m = (fr >= 0.7) & (fr <= 3.0)
            peaks.append(fr[m][np.argmax(S[m])])
        f0 = float(np.median(peaks))
        lo, hi = max(0.5, f0 - 0.12), min(fps / 2 - 0.01, f0 + 0.12)
        b, a = butter(2, [lo / (fps / 2), hi / (fps / 2)], btype="band")
        nb = {r: filtfilt(b, a, np.asarray(waves[r], float)) for r in names}
        ref = max(names, key=lambda r: float(np.std(nb[r])))
        out = {}
        for r in names:
            w = np.asarray(waves[r], float)
            if r == ref or w.std() <= 1e-12:
                out[r] = w
                continue
            c = float(np.corrcoef(nb[ref], nb[r])[0, 1])
            out[r] = -w if c < 0 else w
        return out
    return orient


def _apply_launch_overrides():
    v = os.environ.get("AFIB_NARROWBAND_ORIENT", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        try:
            import inference.evidence as _ev
            fps = float(os.environ.get("AFIB_NARROWBAND_FPS", "30.0"))
            _ev.orient_rois_consistently = _narrowband_orient_factory(fps)
            LAUNCH_OVERRIDES["inference.evidence.orient_rois_consistently"] = \
                f"broadband -> narrowband (fps={fps})"
        except Exception as e:                   # noqa: BLE001
            LAUNCH_OVERRIDES["inference.evidence.orient_rois_consistently"] = \
                f"FAILED ({type(e).__name__}: {e})"
    v = os.environ.get("AFIB_MAX_COLLAPSED_FRACTION", "").strip()
    if v:
        try:
            import capture.ingest as _ing
            before = getattr(_ing, "MAX_COLLAPSED_INTERVAL_FRACTION", None)
            _ing.MAX_COLLAPSED_INTERVAL_FRACTION = float(v)
            LAUNCH_OVERRIDES["capture.ingest.MAX_COLLAPSED_INTERVAL_FRACTION"] = \
                f"{before} -> {float(v)}"
        except Exception as e:                   # noqa: BLE001
            LAUNCH_OVERRIDES["capture.ingest.MAX_COLLAPSED_INTERVAL_FRACTION"] = \
                f"FAILED ({type(e).__name__}: {e})"


_apply_launch_overrides()
