"""
Public paired PPG+ECG corpus for the regularity track (v0.7 Task 3).

MIMIC PERform AF (Zenodo 6807403; data_cache/mimic_perform_{af,non_af}_csv,
125 Hz, columns Time/PPG/ECG/resp, ~20 min per subject): contact
fingertip PPG with simultaneous ECG and a respiration channel. It is a
SURROGATE — contact PPG, not a camera — so its runs carry signal_domain
"public_ppg" and can never open a gate (the qualification disqualifier
applies to every gate). What it IS good for: the ceiling test on real
human hearts. The PPG-derived intervals and the ECG R-R intervals go
through the SAME regularity code, and the question "does the pulse
reproduce the ECG's own regularity verdict?" gets an answer on people,
with AF and non-AF subjects both present, before a single facial
recording exists.

Windows are 90 s, non-overlapping, one row each; the subject is the
participant (splits stay participant-disjoint).
"""
from __future__ import annotations

import pathlib

import numpy as np

from beats.detector import BeatSeries, detect_beats_single_roi
from beats.ibi import clean_runs
from configs import load_config
from datasets.regularity_reference import (resolve_min_conf, classify_index,
                                           load_reference_definition,
                                           reference_from_rpeaks)
from features.regularity import regularity_from_runs

_REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CACHE = _REPO / "data_cache"
FS_HZ = 125.0
WINDOW_S = 90.0
SUBSETS = (("mimic_perform_af_csv", "AFIB", 1),
           ("mimic_perform_non_af_csv", "NON_AF", 0))


def available(cache=None) -> bool:
    c = pathlib.Path(cache or DEFAULT_CACHE)
    return all((c / sub).is_dir() and any((c / sub).glob("*_data.csv"))
               for sub, _, _ in SUBSETS)


def _respiration_from_column(t, resp, fps=FS_HZ):
    """Rate + spectral concentration of the corpus' own respiration
    channel, on the same validated band the torso reader uses."""
    from rppg.respiration import respiratory_rate_from_motion
    from rppg._filters import moving_average_detrend
    y = np.asarray(resp, float)
    if y.size < int(20.0 * fps) or not np.all(np.isfinite(y)):
        return None
    rate, conc = respiratory_rate_from_motion(t, y, fps)
    return {"t": np.asarray(t, float),
            "y": moving_average_detrend(y, int(12.0 * fps)),
            "rate_brpm": rate, "quality": conc}


def load_subject(path) -> dict:
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    t = arr[:, 0]
    return {"t": t, "ppg": arr[:, 1], "ecg": arr[:, 2],
            "resp": arr[:, 3] if arr.shape[1] > 3 else None,
            "fs": float(1.0 / np.median(np.diff(t)))}


def finite_segments(x, *, min_samples: int = 1):
    """[(start, end)) index ranges over which `x` is finite. A NaN block
    is a gap in the record: detection runs per finite stretch on the
    REAL clock, never on a spliced series (review finding: dropping NaN
    samples shifted every later beat earlier)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    out, start = [], None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start >= min_samples:
                out.append((start, i))
            start = None
    if start is not None and x.size - start >= min_samples:
        out.append((start, int(x.size)))
    return out


def _refine_subsample(seg, idx, fs):
    """Parabolic sub-sample refinement of integer peak indices on the
    rectified signal: at 125 Hz an integer-sample R-R difference is a
    multiple of 8 ms and the median |dRR| of a regular heart reads
    exactly 0 (review finding); the PPG side is refined, so the
    reference must be too."""
    y = np.abs(np.asarray(seg, float) - float(np.nanmedian(seg)))
    out = []
    for i in np.asarray(idx, int):
        if 1 <= i < y.size - 1:
            a, b, c = y[i - 1], y[i], y[i + 1]
            den = a - 2.0 * b + c
            off = 0.5 * (a - c) / den if den < 0 else 0.0
            off = float(np.clip(off, -0.5, 0.5))
        else:
            off = 0.0
        out.append((i + off) / fs)
    return np.asarray(out, float)


def rpeaks_segmented(ecg, fs) -> np.ndarray:
    """R-peak times (s, record clock) detected per finite segment, with
    sub-sample refinement."""
    from scripts.e6_degradation import detect_rpeaks_s
    x = np.asarray(ecg, float)
    out = []
    for a, b in finite_segments(x, min_samples=int(2.0 * fs)):
        seg = x[a:b]
        idx = np.rint(np.asarray(detect_rpeaks_s(seg, fs=fs), float) * fs)
        out.append(_refine_subsample(seg, idx, fs) + a / fs)
    return np.concatenate(out) if out else np.zeros(0)


def ppg_runs(ppg, fps=FS_HZ, config=None):
    """Beats from the contact PPG through the production detector +
    clean-run discipline (confidence is the detector's local quality —
    contact PPG has no ROI fusion, so no calibrated fusion confidence
    exists and the local quality stands in; recorded as a surrogate
    caveat). Detection runs per finite stretch of the window on the
    real clock; a NaN block is a gap between runs, not a spliced-out
    stretch."""
    cfg = config or load_config()
    rc = cfg["runs"]
    x = np.asarray(ppg, float)
    beats = []
    for a, b in finite_segments(x, min_samples=int(5.0 * fps)):
        seg = detect_beats_single_roi(x[a:b], fps, roi="ppg")
        for bt in seg:
            bt.t_s = float(bt.t_s) + a / fps
            bt.confidence = float(bt.signal_quality)
            bt.roi_agreement = 1.0
        beats.extend(seg)
    beats.sort(key=lambda bt: bt.t_s)
    series = BeatSeries(beats, float(fps), float(x.size / fps))
    lo_ms, hi_ms = rc["ibi_physiologic_ms"]
    return clean_runs(series, min_conf=resolve_min_conf(cfg),
                      min_run_beats=int(rc["min_run_beats"]),
                      max_physiologic_ibi_ms=float(hi_ms),
                      min_physiologic_ibi_ms=float(lo_ms),
                      missed_beat_ratio=float(rc["missed_ratio"]))


def mimic_perform_rows(*, cache=None, max_subjects=None,
                       max_windows_per_subject=None, window_s=WINDOW_S,
                       gates_path=None, config=None) -> list:
    """One row per 90 s window, in the same schema as the camera rows
    (evaluation/regularity_metrics.rows_from_dataset) so the ceiling
    test and the head run unchanged — except that the "camera" side is
    contact PPG, flagged by dataset/surrogate fields."""
    from heads import get_head
    c = pathlib.Path(cache or DEFAULT_CACHE)
    definition = load_reference_definition(gates_path)
    head = get_head("regularity")
    # the head classifies under the same definition object the
    # reference uses (review finding)
    head_cfg = dict(config or {})
    head_cfg["regularity"] = dict(head_cfg.get("regularity") or {})
    head_cfg["regularity"]["reference_label"] = dict(definition)
    rows = []
    for sub, rhythm, label in SUBSETS:
        files = sorted((c / sub).glob("*_data.csv"))
        if max_subjects is not None:
            files = files[:max_subjects]
        for f in files:
            pid = f.stem.replace("_data", "")
            d = load_subject(f)
            fs = d["fs"]
            rp = rpeaks_segmented(d["ecg"], fs)
            n_win = int(d["t"][-1] // window_s)
            if max_windows_per_subject is not None:
                n_win = min(n_win, max_windows_per_subject)
            for w in range(n_win):
                t0, t1 = w * window_s, (w + 1) * window_s
                sl = slice(int(t0 * fs), int(t1 * fs))
                rs = ppg_runs(d["ppg"][sl], fps=fs, config=config)
                resp = (_respiration_from_column(d["t"][sl] - t0,
                                                 d["resp"][sl], fps=fs)
                        if d["resp"] is not None else None)
                reg = regularity_from_runs(rs.runs, rs.run_confidences,
                                           rs.dropout_rate,
                                           run_times=rs.run_times, fps=fs,
                                           respiration=resp)
                ref = reference_from_rpeaks(rp[(rp >= t0) & (rp < t1)] - t0,
                                            definition=definition,
                                            config=config, fps=fs)
                # a window with no clean run is a NO_RESULT, not an
                # ACCEPT carrying zero intervals (review finding)
                outcome = "ACCEPT" if rs.runs else "NO_RESULT"
                hv = head.run(None, {"scan_outcome": outcome,
                                     "cfg": head_cfg,
                                     "respiration": resp,
                                     "regularity": reg}).value
                idx = reg.index.get("value")
                rows.append({
                    "recording_id": f"{pid}_w{w:02d}",
                    "participant_id": pid, "session_id": f"{pid}_w{w:02d}",
                    "dataset": sub, "surrogate": "public_ppg",
                    "rhythm": rhythm, "label_af": label,
                    "fps": fs, "sqi": None, "sqi_grade": "unknown",
                    "fitzpatrick_group": None, "age_years": None,
                    "sex": None, "age_band": "unknown",
                    "scan_outcome": outcome,
                    "camera_index": idx,
                    "camera_ci95": reg.index.get("ci95"),
                    "camera_class": (classify_index(idx, reg.n_intervals,
                                                    definition)
                                     if outcome == "ACCEPT"
                                     else "indeterminate"),
                    "camera_rmssd_ms": reg.values.get("rmssd"),
                    "camera_n_intervals": int(reg.n_intervals),
                    "camera_features": reg.to_dict(),
                    "ecg_index": ref["index"], "ecg_class": ref["class"],
                    "ecg_rmssd_ms": ref["rmssd_ms"],
                    "ecg_n_intervals": ref["n_intervals"],
                    "head": hv,
                    "resp_rate_brpm": (resp or {}).get("rate_brpm"),
                    "resp_quality": (resp or {}).get("quality"),
                })
    return rows
