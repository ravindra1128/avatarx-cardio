"""
Rep/cadence counting (v0.4 T3) — workload verification ONLY.

Primary tracker: an optional pose-landmarker model (mediapipe tasks API,
same env-var pattern as the face model — AVATARX_POSE_MODEL points at a
downloaded pose_landmarker.task). Documented fallback: vertical
MOTION-ENERGY periodicity — the brightness-weighted vertical centroid of
the frame oscillates once per rep for every supported challenge
(sit-to-stand, stepping, marching all move the body vertically). The
tracker that actually ran is recorded in provenance, exactly like the
face-tracker chain.

Nothing in this module reads pulse. That is a contract, not an accident.
"""
from __future__ import annotations

import os

import numpy as np

from capture.video_reader import iter_frames

CADENCE_BAND_HZ = (0.10, 1.60)      # 6-96 reps/min
MIN_DURATION_S = 20.0


def _centroid_series(video_path: str) -> tuple:
    """(t, vertical brightness centroid per frame, fps estimate)."""
    ts, ys = [], []
    for t, frame in iter_frames(video_path):
        g = frame.mean(axis=2)                       # gray
        rows = g.mean(axis=1)
        rows = rows - rows.min()
        tot = float(rows.sum()) or 1.0
        ys.append(float((rows * np.arange(rows.size)).sum() / tot))
        ts.append(float(t))
    t = np.asarray(ts)
    y = np.asarray(ys)
    fps = 1.0 / float(np.median(np.diff(t))) if t.size > 1 else 30.0
    return t, y, fps


def _pose_series(video_path: str, model_path: str) -> tuple:
    """Hip-center vertical position via the pose landmarker. Guarded: any
    failure falls back to motion energy (recorded)."""
    from mediapipe.tasks import python as mp_python           # noqa
    from mediapipe.tasks.python import vision                 # noqa
    import mediapipe as mp                                    # noqa
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)
    lm = vision.PoseLandmarker.create_from_options(opts)
    ts, ys = [], []
    for t, frame in iter_frames(video_path):
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=frame[:, :, ::-1].copy())
        res = lm.detect_for_video(img, int(t * 1000))
        if res.pose_landmarks:
            hips = res.pose_landmarks[0]
            ys.append(float((hips[23].y + hips[24].y) / 2.0))
            ts.append(float(t))
    t = np.asarray(ts)
    y = np.asarray(ys)
    fps = 1.0 / float(np.median(np.diff(t))) if t.size > 1 else 30.0
    return t, y, fps


def _count_cycles(t: np.ndarray, y: np.ndarray, fps: float) -> dict:
    if t.size < int(MIN_DURATION_S * fps * 0.5):
        return {"reps": None, "cadence_per_min": None, "confidence": 0.0,
                "note": "clip too short to verify the activity"}
    # detrend with a ~4 s edge-normalized moving average (a zero-padded
    # 'same' convolution on a large-DC series creates edge ramps that
    # inflate std and bury small rep amplitudes), then smooth lightly
    from rppg._filters import moving_average_detrend
    x = moving_average_detrend(y, int(4.0 * fps))
    ks = max(int(0.15 * fps), 1)
    x = np.convolve(x, np.ones(ks) / ks, mode="same")
    # dominant cadence frequency
    n = x.size
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    band = (freqs >= CADENCE_BAND_HZ[0]) & (freqs <= CADENCE_BAND_HZ[1])
    if not band.any() or float(spec[band].sum()) <= 0:
        return {"reps": None, "cadence_per_min": None, "confidence": 0.0,
                "note": "no periodic motion in the cadence band"}
    f_dom = float(freqs[band][np.argmax(spec[band])])
    concentration = float(
        spec[band & (np.abs(freqs - f_dom) <= 0.05)].sum()
        / spec[band].sum())
    # count actual cycles: peaks with a refractory of 60% of the period
    thr = 0.3 * float(np.std(x))
    min_gap = int(0.6 / f_dom * fps)
    peaks = []
    i = 1
    while i < n - 1:
        if x[i] > thr and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            if not peaks or i - peaks[-1] >= min_gap:
                peaks.append(i)
                i += min_gap
                continue
        i += 1
    reps = len(peaks)
    span = float(t[-1] - t[0]) or 1.0
    return {"reps": reps,
            "cadence_per_min": round(f_dom * 60.0, 1),
            "confidence": round(concentration, 3),
            "analysed_seconds": round(span, 1),
            "note": None}


def count_cycles_from_series(t, y, fps: float) -> dict:
    """Public entry for LIVE series (the three-phase orchestrator feeds
    per-frame vertical centroids): same counter, same contract."""
    return _count_cycles(np.asarray(t, float), np.asarray(y, float),
                         float(fps))


def count_reps(video_path: str) -> dict:
    """The activity verifier. Returns reps, cadence, confidence and the
    tracker that actually ran (provenance, like the face-tracker chain)."""
    model = os.environ.get("AVATARX_POSE_MODEL")
    if model and os.path.exists(model):
        try:
            t, y, fps = _pose_series(video_path, model)
            out = _count_cycles(t, y, fps)
            out["tracker"] = "pose-landmarker"
            return out
        except Exception as e:                    # documented fallback
            fallback_note = f"pose landmarker failed ({e.__class__.__name__})"
    else:
        fallback_note = None
    t, y, fps = _centroid_series(video_path)
    out = _count_cycles(t, y, fps)
    out["tracker"] = "motion_energy"
    if fallback_note:
        out["note"] = ((out.get("note") or "") + " " + fallback_note).strip()
    return out
