"""
Fiducial and interval measurement (M4.14b) — the test that matters.
Waveform correlation flatters a generated ECG (the QRS dominates the
variance and its TIMING comes free from the pulse); clinical reading
happens at fiducials and intervals. Interval-level error in ms against
the reference ECG is therefore the primary metric; correlation may be
reported only alongside it.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from datasets.reference import detect_rpeaks


def rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    t = np.arange(np.asarray(ecg).size) / fs
    return detect_rpeaks(t, np.asarray(ecg, float))


def _smooth(x: np.ndarray, fs: float, hi: float = 20.0) -> np.ndarray:
    b, a = butter(2, hi / (fs / 2.0), btype="low")
    return filtfilt(b, a, np.asarray(x, float))


def fiducials(ecg: np.ndarray, fs: float, r_times: np.ndarray) -> list:
    """Per matched beat: P-peak and T-peak times (s) and prominences
    relative to the QRS amplitude, from guideline search windows around
    each R (P: R-260..R-100 ms; T: R+140..R+380 ms)."""
    x = _smooth(ecg, fs)
    out = []
    n = x.size
    for r in np.asarray(r_times, float):
        ri = int(round(r * fs))
        if ri < int(0.3 * fs) or ri > n - int(0.45 * fs):
            continue
        qrs_amp = float(np.max(np.abs(
            x[ri - int(0.05 * fs):ri + int(0.05 * fs)])) or 1.0)
        pa, pb = ri - int(0.26 * fs), ri - int(0.10 * fs)
        ta, tb = ri + int(0.14 * fs), ri + int(0.38 * fs)
        pi = pa + int(np.argmax(x[pa:pb]))
        ti = ta + int(np.argmax(x[ta:tb]))
        out.append({"r_t": float(r),
                    "p_t": pi / fs, "t_t": ti / fs,
                    "p_prom": float(max(x[pi], 0.0) / qrs_amp),
                    "t_prom": float(max(x[ti], 0.0) / qrs_amp),
                    "pr_ms": (r - pi / fs) * 1000.0,
                    "rt_ms": (ti / fs - r) * 1000.0})
    return out


def interval_errors(ref_ecg: np.ndarray, gen_ecg: np.ndarray, fs: float
                    ) -> dict:
    """Matched-beat fiducial/interval errors (ms) of a generated ECG
    against the reference, plus P/T prominences on each side."""
    rr = rpeaks(ref_ecg, fs)
    rg = rpeaks(gen_ecg, fs)
    fr = {round(f["r_t"], 3): f for f in fiducials(ref_ecg, fs, rr)}
    r_err, pr_err, rt_err = [], [], []
    p_ref, p_gen, t_ref, t_gen = [], [], [], []
    fg_all = fiducials(gen_ecg, fs, rg)
    for f in fg_all:
        if not fr:
            break
        keys = np.array(list(fr))
        j = float(keys[np.argmin(np.abs(keys - f["r_t"]))])
        if abs(j - f["r_t"]) > 0.08:
            continue
        m = fr[round(j, 3)]
        r_err.append(abs(f["r_t"] - m["r_t"]) * 1000.0)
        pr_err.append(abs(f["pr_ms"] - m["pr_ms"]))
        rt_err.append(abs(f["rt_ms"] - m["rt_ms"]))
        p_ref.append(m["p_prom"]); p_gen.append(f["p_prom"])
        t_ref.append(m["t_prom"]); t_gen.append(f["t_prom"])

    def med(v):
        return round(float(np.median(v)), 2) if v else None
    return {"n_matched_beats": len(r_err),
            "n_ref_beats": int(rr.size), "n_gen_beats": int(rg.size),
            "r_timing_mae_ms": med(r_err),
            "pr_interval_mae_ms": med(pr_err),
            "rt_interval_mae_ms": med(rt_err),
            "p_prominence_ref": med(p_ref),
            "p_prominence_gen": med(p_gen),
            "t_prominence_ref": med(t_ref),
            "t_prominence_gen": med(t_gen)}
