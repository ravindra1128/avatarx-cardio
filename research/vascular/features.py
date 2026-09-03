"""Candidate pulse-morphology features (v0.4 vascular track, T1).

Per-beat features from a pulse waveform, computed IDENTICALLY for the
facial rPPG path and the simultaneous contact-PPG reference so the V0
fidelity study compares like with like. The facial waveform is rebuilt
from the production ingest's retained per-ROI RGB traces with a WIDER
morphology band than the beat detector's 0.7-4 Hz (a dicrotic notch does
not survive a 4 Hz low-pass); beats, gates and segments stay the
production ones — invariant V-c: features are computed only on
ACCEPT-grade scans and only on calibrated beats at or above the
production confidence floor. No demographic enters anything here.
"""
from __future__ import annotations

import numpy as np

from research.vascular import VascularError

# v0.8 (G-a discipline, as v0.7 did for the interval representation):
# the contour-morphology math is ONE implementation, in
# features/pulse_morphology.py, so `app/` and `inference/` — which are
# quarantined from research/ — can compute it too. This module keeps the
# research-track surface: the facial/contact session builders, the
# fidelity comparison and the V0 study. The names below are re-exported
# so every existing caller and test reads exactly what it read before.
from features.pulse_morphology import (  # noqa: F401
    FEATURE_NAMES, MIN_BEATS_FOR_SESSION, MIN_SDPPG_FS_HZ, MORPH_BAND_HZ,
    NOTCH_MIN_PROMINENCE, RESAMPLE_HZ, RR_PLAUSIBLE_S, beat_morphology,
    ensemble_beat, session_median, _beat_pairs, _beat_windows,
    _local_maxima, _local_minima, _resample)


# ------------------------------------------------- facial (production)
def facial_session_features(video_path: str, manifest=None,
                            config=None) -> dict:
    """Morphology features for one scan THROUGH the production pipeline.

    Invariant V-c enforced here: a scan that is not ACCEPT-grade yields
    NO features (available=False with the production no-read reasons) —
    feature extraction never runs on REPEAT_SCAN/NO_RESULT sessions."""
    from inference.pipeline import run_with_details
    result, det = run_with_details(video_path, manifest=manifest,
                                   config=config)
    return features_from_details(result, det)


def features_from_details(result, det) -> dict:
    """The extraction core, on an ALREADY-COMPLETED production run
    (result + details from inference.pipeline.run_with_details, or a
    live session's retained last_det). Same V-c contract."""
    from inference.evidence import capture_segments
    from preprocessing.roi import ROI_NAMES
    from rppg.pos import pos_pulse, orient_rois_consistently
    from datasets.schema import ScanOutcome

    if result.outcome is not ScanOutcome.ACCEPT:
        return {"available": False,
                "outcome": result.outcome.value,
                "reasons": ["V-c: morphology features are computed only "
                            "on ACCEPT-grade scans"]
                + list(result.no_read_reasons or [])}
    # the RESOLVED production confidence floor from the run itself; the
    # imported constant is only the legacy-det fallback — never a copy
    # that could drift from a config-tightened floor (review finding)
    from inference.pipeline import CALIBRATED_MIN_CONF
    min_conf = float(det.get("min_conf", CALIBRATED_MIN_CONF))
    ing = det["ingest"]
    lattice = det["lattice"]
    ts = np.asarray(ing.timestamps_s, float)
    fps = float(ing.meta.measured_fps_mean)
    hi = min(MORPH_BAND_HZ[1], 0.45 * fps)
    beat_t = np.asarray(lattice.beat_t_s, float)
    conf = np.asarray(lattice.beat_confidence, float)
    pairs = _beat_pairs(beat_t, conf, min_conf)
    roi_segments: dict = {r: [] for r in ROI_NAMES}
    per_beat: list = []
    for a, b in capture_segments(ts, fps):
        seg_ts = ts[a:b]
        waves = orient_rois_consistently(
            {r: pos_pulse(ing.traces[r][a:b], fps,
                          band=(MORPH_BAND_HZ[0], hi))
             for r in ROI_NAMES})
        in_seg = [(t0, t1) for t0, t1 in pairs
                  if t0 >= seg_ts[0] and t1 <= seg_ts[-1]]
        for roi in ROI_NAMES:
            wave = waves[roi]
            for f1, f2 in _beat_windows(wave, seg_ts, in_seg):
                roi_segments[roi].append(wave[f1:f2])
                per_beat.append(beat_morphology(wave[f1:f2], fps,
                                                native_fs=fps))
    # session features: median across per-ROI ENSEMBLE beats; per-beat
    # features feed only the dispersion/quality diagnostics (see
    # ensemble_beat docstring for the measured rationale)
    roi_feats = []
    for roi in ROI_NAMES:
        ens, dur = ensemble_beat(roi_segments[roi], fps)
        if ens is None:
            continue
        f = beat_morphology(ens, ens.size / dur, native_fs=fps)
        if f is not None:
            roi_feats.append(f)
    beats_per_roi = int(np.median([len(v) for v in roi_segments.values()]))
    if len(roi_feats) < 2 or beats_per_roi < MIN_BEATS_FOR_SESSION:
        return {"available": False, "outcome": result.outcome.value,
                "reasons": [f"only {beats_per_roi} morphology-usable "
                            f"beats across {len(roi_feats)} readable "
                            f"ROIs (need >= {MIN_BEATS_FOR_SESSION} "
                            "beats and 2 ROIs)"]}
    scatter = session_median(per_beat)
    features = {}
    for name in FEATURE_NAMES:
        vals = [r[name] for r in roi_feats if r.get(name) is not None]
        features[name] = (float(np.median(vals)) if vals else None)
    # notch_present is the PER-BEAT detection fraction on both arms —
    # a comparable continuous quantity (the ensemble's 0/1 is not)
    features["notch_present"] = scatter["features"].get("notch_present")
    return {"available": True, "outcome": result.outcome.value,
            "fps": fps, "band_hz": [MORPH_BAND_HZ[0], hi],
            "sdppg_derivable": fps >= MIN_SDPPG_FS_HZ,
            "recording_id": result.recording_id,
            "features": features,
            "n_beats_used": beats_per_roi,
            "n_rois_used": len(roi_feats),
            "quality": {"per_beat_scatter": scatter.get("quality", {}),
                        "notch_detect_fraction_beats":
                            scatter["features"].get("notch_present")}}


# ------------------------------------------------- contact reference
def contact_session_features(ppg) -> dict:
    """The IDENTICAL feature set from a synchronized contact-PPG
    reference (datasets.schema.ContactPpg). Its own beat segmentation —
    the reference must not inherit facial-path failure modes."""
    from scipy.signal import butter, filtfilt, find_peaks
    x = np.asarray(ppg.samples, float)
    fs = float(ppg.fs_hz)
    if x.size < int(2 * fs):
        raise VascularError("contact reference too short to segment")
    # the IDENTICAL band cap as the facial arm — the fidelity study
    # compares like with like, so a wider reference band would measure
    # a filter mismatch, not the camera (review finding)
    hi = min(MORPH_BAND_HZ[1], 0.45 * fs)
    b, a = butter(3, [MORPH_BAND_HZ[0] / (fs / 2), hi / (fs / 2)],
                  btype="band")
    w = filtfilt(b, a, x)
    peaks, _ = find_peaks(w, distance=int(RR_PLAUSIBLE_S[0] * fs),
                          prominence=0.25 * float(np.std(w)))
    t_peaks = peaks / fs
    segments, per_beat = [], []
    for (i0, i1), (t0, t1) in zip(zip(peaks, peaks[1:]),
                                  zip(t_peaks, t_peaks[1:])):
        rr = t1 - t0
        if not (RR_PLAUSIBLE_S[0] <= rr <= RR_PLAUSIBLE_S[1]):
            continue
        a0 = max(int(i0 - 0.35 * rr * fs), 0)
        b0 = max(int(i1 - 0.35 * rr * fs), 0)
        foot1 = a0 + int(np.argmin(w[a0:i0])) if i0 > a0 else a0
        foot2 = b0 + int(np.argmin(w[b0:i1])) if i1 > b0 else b0
        if foot2 - foot1 >= 6:
            segments.append(w[foot1:foot2])
            per_beat.append(beat_morphology(w[foot1:foot2], fs,
                                            native_fs=fs))
    if len(segments) < MIN_BEATS_FOR_SESSION:
        return {"available": False,
                "reasons": [f"only {len(segments)} contact beats"]}
    # the IDENTICAL session statistic as the facial arm: ensemble beat
    ens, dur = ensemble_beat(segments, fs)
    if ens is None:
        return {"available": False,
                "reasons": ["contact ensemble could not be formed"]}
    feats = beat_morphology(ens, ens.size / dur, native_fs=fs)
    if feats is None:
        return {"available": False,
                "reasons": ["contact ensemble beat unreadable"]}
    scatter = session_median(per_beat)
    feats = dict(feats)
    feats["notch_present"] = scatter["features"].get("notch_present")
    return {"available": True, "fs": fs,
            "sdppg_derivable": fs >= MIN_SDPPG_FS_HZ,
            "features": feats, "n_beats_used": len(segments),
            "quality": {"per_beat_scatter": scatter.get("quality", {}),
                        "notch_detect_fraction_beats":
                            scatter["features"].get("notch_present")}}
