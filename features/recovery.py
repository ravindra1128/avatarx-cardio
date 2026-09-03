"""
Recovery HR tracker (v0.4 T2) — consumes the BeatLattice's beat times +
confidences directly. Deliberately NOT `clean_runs`/rhythm features:
those assume stationary IBI statistics, and recovery HR falls
20-40 bpm/min by design, so run-splitting and dispersion features would
read the recovery itself as artifact.

Method: short overlapping windows (8 s, 50% overlap) of trimmed-median
HR from verified-beat IBIs; a hard physiological slew bound
(|dHR/dt| <= 3 bpm/s) on the window series; a robust monotone
(non-increasing) trend via pool-adjacent-violators; `hr_end_proxy`
back-extrapolated to t = 0 from a Theil-Sen fit of the first 15 s
(the first window center already sits several seconds into recovery);
HRR30/60/120 as proxy-minus-trend. The mono-exponential tau is computed
but RESEARCH-TELEMETRY-ONLY (field CV ~25-35% — never user-facing).

Fail-closed: not enough verified windows -> None metrics with reasons.
"""
from __future__ import annotations

import numpy as np

WINDOW_S = 8.0
STEP_S = 4.0                      # 50% overlap
MAX_SLEW_BPM_S = 3.0              # physiological recovery slew bound
TRIM_FRACTION = 0.2               # trimmed median: drop top/bottom 20%
VERIFIED_CONF = 0.5
MIN_IBIS_PER_WINDOW = 4
PHYSIO_IBI_S = (0.25, 2.2)
FIT_SPAN_S = 15.0                 # back-extrapolation fit window
MIN_USABLE_WINDOWS = 6
MIN_USABLE_SPAN_S = 45.0


def _trimmed_median(v: np.ndarray, trim: float = TRIM_FRACTION) -> float:
    v = np.sort(np.asarray(v, float))
    k = int(np.floor(trim * v.size))
    core = v[k:v.size - k] if v.size - 2 * k >= 1 else v
    return float(np.median(core))


def slew_bound(t, hr, max_slew: float = MAX_SLEW_BPM_S) -> np.ndarray:
    """Forward clamp: no step may exceed max_slew * dt. A one-window
    artifact cannot drag the series faster than physiology allows."""
    t = np.asarray(t, float)
    out = np.asarray(hr, float).copy()
    for i in range(1, out.size):
        lim = max_slew * (t[i] - t[i - 1])
        out[i] = float(np.clip(out[i], out[i - 1] - lim, out[i - 1] + lim))
    return out


def window_hr_series(beat_t, beat_conf, *, duration_s: float,
                     window_s: float = WINDOW_S, step_s: float = STEP_S,
                     conf_floor: float = VERIFIED_CONF) -> list:
    """[(t_center, hr_bpm | None, confidence | None)] — one entry per
    window over [0, duration_s]; unusable windows stay in the series
    (hr None) so gaps are visible to callers and to the quality vector.
    HR values are slew-bounded across usable windows."""
    beat_t = np.asarray(beat_t, float)
    conf = np.asarray(beat_conf, float)
    ok = conf >= conf_floor
    bt, bc = beat_t[ok], conf[ok]
    entries = []
    t0 = 0.0
    while t0 + window_s <= float(duration_s) + 1e-9:
        c = t0 + window_s / 2.0
        m = (bt >= t0) & (bt < t0 + window_s)
        wt, wc = bt[m], bc[m]
        ibis = np.diff(wt)
        keep = (ibis >= PHYSIO_IBI_S[0]) & (ibis <= PHYSIO_IBI_S[1])
        ibis = ibis[keep]
        if ibis.size >= MIN_IBIS_PER_WINDOW:
            med = _trimmed_median(ibis)
            coverage = float(np.clip(ibis.size * med / window_s, 0.0, 1.0))
            entries.append([c, 60.0 / med, float(np.mean(wc)) * coverage])
        else:
            entries.append([c, None, None])
        t0 += step_s
    usable = [(i, e) for i, e in enumerate(entries) if e[1] is not None]
    if usable:
        ts = np.array([e[0] for _, e in usable])
        hrs = slew_bound(ts, np.array([e[1] for _, e in usable]))
        for (i, _), hr in zip(usable, hrs):
            entries[i][1] = float(hr)
    return [tuple(e) for e in entries]


def _pava_decreasing(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for a NON-INCREASING trend."""
    blocks = [[float(v), 1] for v in y]           # [mean, weight]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0] - 1e-12:     # violation
            m = (blocks[i][0] * blocks[i][1]
                 + blocks[i + 1][0] * blocks[i + 1][1])
            w = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [m / w, w]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for mean, w in blocks:
        out += [mean] * w
    return np.asarray(out)


def _theil_sen(t: np.ndarray, y: np.ndarray) -> tuple:
    slopes = [(y[j] - y[i]) / (t[j] - t[i])
              for i in range(t.size) for j in range(i + 1, t.size)
              if t[j] > t[i]]
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * t))
    return slope, intercept


def _fit_tau(t: np.ndarray, y: np.ndarray) -> dict:
    """Grid-search mono-exponential y = a + b*exp(-t/tau)."""
    best = None
    for tau in np.arange(15.0, 120.5, 1.0):
        X = np.stack([np.ones(t.size), np.exp(-t / tau)], 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        sse = float(resid @ resid)
        if best is None or sse < best[0]:
            best = (sse, float(tau), coef)
    sse, tau, coef = best
    tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
    return {"tau_s": tau, "tau_fit_r2": round(1.0 - sse / tot, 3),
            "asymptote_bpm": round(float(coef[0]), 1)}


def recovery_metrics(beat_t, beat_conf, *, duration_s: float) -> dict:
    """The T2 contract. All user-surfaceable values are MEASURED-class
    window statistics; anything model-ish (tau) lives under
    `research_only` and must never render."""
    series = window_hr_series(beat_t, beat_conf, duration_s=duration_s)
    usable = [(t, hr, c) for t, hr, c in series if hr is not None]
    reasons = []
    out = {"hr_series": series, "trend": [], "hr_end_proxy": None,
           "hrr30": None, "hrr60": None, "hrr120": None,
           "recovery_slope_bpm_min": None, "research_only": {},
           "reasons": reasons,
           "quality": {"n_windows": len(series), "n_usable": len(usable),
                       "coverage_fraction": (round(len(usable)
                                                   / len(series), 3)
                                             if series else 0.0),
                       "median_conf": (round(float(np.median(
                           [c for _, _, c in usable])), 3)
                           if usable else None)}}
    span = (usable[-1][0] - usable[0][0]) if len(usable) >= 2 else 0.0
    if len(usable) < MIN_USABLE_WINDOWS or span < MIN_USABLE_SPAN_S:
        reasons.append(f"insufficient usable HR windows for recovery "
                       f"metrics ({len(usable)} windows spanning "
                       f"{span:.0f} s)")
        return out

    ut = np.array([t for t, _, _ in usable])
    uh = np.array([hr for _, hr, _ in usable])
    trend = _pava_decreasing(ut, uh)
    out["trend"] = [(float(a), float(b)) for a, b in zip(ut, trend)]

    # the 0-15 s fit gets its own DENSER window series (6 s / 2 s step):
    # three coarse points make the Theil-Sen intercept jitter-sensitive
    early_series = window_hr_series(beat_t, beat_conf,
                                    duration_s=min(float(duration_s),
                                                   FIT_SPAN_S + 3.0),
                                    window_s=6.0, step_s=2.0)
    et = np.array([t for t, hr, c in early_series
                   if hr is not None and t <= FIT_SPAN_S])
    eh = np.array([hr for t, hr, c in early_series
                   if hr is not None and t <= FIT_SPAN_S])
    if et.size >= 3:
        slope, intercept = _theil_sen(et, eh)
        out["hr_end_proxy"] = round(intercept, 1)
    else:
        reasons.append("no usable windows in the first 15 s — cannot "
                       "back-extrapolate the end-exercise proxy")
        return out

    def trend_at(x):
        if ut[-1] < x:
            return None
        return float(np.interp(x, ut, trend))

    for key, at in (("hrr30", 30.0), ("hrr60", 60.0), ("hrr120", 120.0)):
        v = trend_at(at)
        if v is None:
            reasons.append(f"recovery scan does not span {at:.0f} s — "
                           f"{key} unavailable")
        else:
            out[key] = round(out["hr_end_proxy"] - v, 1)
    t60 = trend_at(60.0)
    if t60 is not None:
        out["recovery_slope_bpm_min"] = round(t60 - out["hr_end_proxy"], 1)
    out["research_only"] = _fit_tau(ut, uh)
    return out
