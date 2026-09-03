"""Pulse-CONTOUR morphology: the ONE implementation (v0.8).

Per-beat and ensemble-beat features from a pulse waveform — rise time,
upstroke slope, width, dicrotic notch, reflection index and the
second-derivative (SDPPG) a/b/c/d/e ratios — computed IDENTICALLY
wherever they are needed:

- the v0.4 vascular research track (`research/vascular/features.py`,
  now an adapter over this module) for the V0 fidelity study and the
  cfPWV model, on the facial and contact-PPG arms alike;
- the v0.8 resting hemodynamic indices (`features/hemodynamics.py`)
  that the production path can compute from a single scan.

It lives in `features/` rather than `research/` because `app/` and
`inference/` are quarantined from `research/` and must still be able to
compute contour morphology. Nothing here knows about models, gates or
demographics: it is waveform in, fiducials out. The invariants that
govern WHERE the numbers may be shown live with the heads and gates,
never here.

Moved verbatim from research/vascular/features.py (v0.4 T1) so the
fidelity study's numbers are unchanged by the move; the docstrings
below record the review findings that shaped each fiducial rule.
"""
from __future__ import annotations

import numpy as np

# Wide-band morphology extraction (Hz). The high edge is capped below
# Nyquist at capture time; second-derivative (SDPPG) features additionally
# require MIN_SDPPG_FS_HZ of true sampling — below that they are reported
# as None ("not derivable at our sampling rate"), never interpolated into
# existence.
MORPH_BAND_HZ = (0.5, 10.0)
MIN_SDPPG_FS_HZ = 50.0
RESAMPLE_HZ = 240.0
RR_PLAUSIBLE_S = (0.4, 1.6)
MIN_BEATS_FOR_SESSION = 8
NOTCH_MIN_PROMINENCE = 0.04    # reflected wave must clear the notch by
                               # this fraction of pulse amplitude, or the
                               # "notch" is a noise dip (review finding)

FEATURE_NAMES = (
    "rise_time_s",             # foot -> systolic peak
    "norm_upstroke_slope",     # max dW/dt / pulse amplitude (1/s)
    "pulse_width50_s",         # width at 50% amplitude
    "notch_present",           # dicrotic notch detected (0/1; session: frac)
    "notch_rel_amp",           # (notch - foot) / (peak - foot)
    "notch_time_frac",         # notch time / beat duration
    "reflection_index",        # (diastolic peak - foot) / (peak - foot)
    "sdppg_b_over_a",          # 2nd-derivative aging-index components —
    "sdppg_c_over_a",          # all four ratios are CANDIDATES; the V0
    "sdppg_d_over_a",          # fidelity study adjudicates which (if
    "sdppg_e_over_a",          # any) survive the camera
)


# ------------------------------------------------------------ per beat
def _resample(seg: np.ndarray, dur_s: float) -> tuple[np.ndarray, float]:
    # endpoint=False on BOTH grids (samples sit at k/fs): a closed-end
    # grid stretched the time base by size/(size-1) — ~4% at 30 fps —
    # inflating every per-beat time feature (review finding)
    n = max(int(round(dur_s * RESAMPLE_HZ)), 8)
    t_old = np.linspace(0.0, dur_s, seg.size, endpoint=False)
    t_new = np.linspace(0.0, dur_s, n, endpoint=False)
    return np.interp(t_new, t_old, seg), n / max(dur_s, 1e-9)


def _local_minima(w: np.ndarray) -> np.ndarray:
    return np.where((w[1:-1] < w[:-2]) & (w[1:-1] <= w[2:]))[0] + 1


def _local_maxima(w: np.ndarray) -> np.ndarray:
    return np.where((w[1:-1] > w[:-2]) & (w[1:-1] >= w[2:]))[0] + 1


def beat_morphology(seg: np.ndarray, fs: float,
                    native_fs: float | None = None) -> dict | None:
    """Features for ONE foot-to-foot beat segment. Returns None when the
    segment cannot support a systolic read (fail closed, never guessed).
    `native_fs` is the true capture rate governing SDPPG derivability
    (resampling adds no information)."""
    seg = np.asarray(seg, float)
    if seg.size < 8 or not np.all(np.isfinite(seg)):
        return None
    dur = seg.size / float(fs)
    if not (RR_PLAUSIBLE_S[0] * 0.8 <= dur <= RR_PLAUSIBLE_S[1] * 1.2):
        return None
    w, rfs = _resample(seg, dur)
    dt = 1.0 / rfs
    # fiducials -------------------------------------------------------
    foot_i = int(np.argmin(w[: max(int(0.15 * w.size), 2)]))
    sys_hi = int(0.60 * w.size)
    peak_i = foot_i + int(np.argmax(w[foot_i:sys_hi]))
    amp = float(w[peak_i] - w[foot_i])
    if peak_i <= foot_i or amp <= 1e-9:
        return None
    rise_time = (peak_i - foot_i) * dt
    slope = float(np.max(np.diff(w[foot_i:peak_i + 1]))) / dt / amp \
        if peak_i - foot_i >= 1 else None
    # width at 50% amplitude
    half = w[foot_i] + 0.5 * amp
    above = np.where(w >= half)[0]
    width50 = float((above[-1] - above[0]) * dt) if above.size >= 2 else None
    # dicrotic notch + reflected wave: the EARLIEST post-systolic local
    # minimum whose following local maximum rises with real prominence.
    # Review findings pinned both choices: deepest-min selection let a
    # ~1.5%-amplitude late dip relocate the notch, and a prominence-free
    # test saturated notch_present to 1.0 under modest noise. The search
    # window ends at 0.75 of the beat (physiological), and the reflected
    # wave must clear the notch by NOTCH_MIN_PROMINENCE of amplitude.
    notch_present, notch_rel, notch_frac, refl_index = 0.0, None, None, None
    tail_lo, tail_hi = peak_i + 2, int(0.75 * w.size)
    if tail_hi - tail_lo > 4:
        mins = [i for i in _local_minima(w[tail_lo:tail_hi]) + tail_lo]
        for i in sorted(mins):
            later_max = _local_maxima(w[i:tail_hi])
            if not later_max.size:
                continue
            j = i + int(later_max[0])
            if (w[j] - w[i]) < NOTCH_MIN_PROMINENCE * amp:
                continue
            notch_present = 1.0
            notch_rel = float((w[i] - w[foot_i]) / amp)
            notch_frac = float(i) / float(w.size)
            refl_index = float((w[j] - w[foot_i]) / amp)
            break
    # SDPPG a/b/c/d/e waves — only at reference-grade native sampling
    b_over_a, c_over_a, d_over_a, e_over_a = None, None, None, None
    nfs = float(native_fs if native_fs is not None else fs)
    if nfs >= MIN_SDPPG_FS_HZ:
        k = max(int(round(0.025 * rfs)), 1)
        smooth = np.convolve(w, np.ones(k) / k, mode="same")
        d2 = np.gradient(np.gradient(smooth))
        sys_zone = d2[foot_i:sys_hi]
        if sys_zone.size > 8:
            a_i = foot_i + int(np.argmax(sys_zone))
            a_v = float(d2[a_i])
            if a_v > 1e-12 and a_i + 2 < w.size:
                after = d2[a_i:int(0.7 * w.size)]
                if after.size > 4:
                    b_i = a_i + int(np.argmin(after))
                    b_over_a = float(d2[b_i] / a_v)
                    rest = d2[b_i:int(0.9 * w.size)]
                    rmins = _local_minima(rest)
                    rmaxs = _local_maxima(rest)
                    if rmaxs.size and rmins.size:
                        c_off = int(rmaxs[0])
                        c_over_a = float(d2[b_i + c_off] / a_v)
                        after_c = rmins[rmins > rmaxs[0]]
                        if after_c.size:
                            d_off = int(after_c[0])
                            d_over_a = float(d2[b_i + d_off] / a_v)
                            after_d = rmaxs[rmaxs > d_off]
                            if after_d.size:
                                e_over_a = float(
                                    d2[b_i + int(after_d[0])] / a_v)
    return {"rise_time_s": float(rise_time),
            "norm_upstroke_slope": slope,
            "pulse_width50_s": width50,
            "notch_present": notch_present,
            "notch_rel_amp": notch_rel,
            "notch_time_frac": notch_frac,
            "reflection_index": refl_index,
            "sdppg_b_over_a": b_over_a,
            "sdppg_c_over_a": c_over_a,
            "sdppg_d_over_a": d_over_a,
            "sdppg_e_over_a": e_over_a}


def ensemble_beat(segments: list, fs: float) -> tuple:
    """(ensemble waveform, duration_s) from min-max-normalized beat
    segments on a common 240 Hz grid of the median beat duration.

    Repo-measured rationale (spec B.20): single-beat fiducials at camera
    bandwidth are noise-limited — on synthetic 60 fps clips the per-beat
    reflection-index IQR reached 0.36 and the stiffness contrast
    inverted, while the ensemble beat recovers the injected morphology.
    Session-level features therefore come from the ensemble; per-beat
    features remain computed as dispersion/quality diagnostics."""
    durs = [s.size / fs for s in segments]
    if not durs:
        return None, None
    dur = float(np.median(durs))
    n = max(int(round(dur * RESAMPLE_HZ)), 8)
    grid = np.linspace(0.0, 1.0, n, endpoint=False)
    rows = []
    for s in segments:
        s = np.asarray(s, float)
        if s.size < 8 or not np.all(np.isfinite(s)):
            continue
        t = np.linspace(0.0, 1.0, s.size, endpoint=False)
        r = np.interp(grid, t, s)
        span = float(np.ptp(r))
        if span > 1e-9:
            rows.append((r - r.min()) / span)
    if len(rows) < 3:
        return None, None
    return np.mean(rows, axis=0), dur


def session_median(per_beat: list) -> dict:
    """Per-beat aggregation: median over beats (notch_present becomes
    the detection fraction) plus dispersion diagnostics."""
    rows = [r for r in per_beat if r is not None]
    out: dict = {"n_beats_used": len(rows)}
    feats: dict = {}
    quality: dict = {}
    for name in FEATURE_NAMES:
        vals = np.asarray([r[name] for r in rows
                           if r.get(name) is not None], float)
        if name == "notch_present":
            feats[name] = float(np.mean(vals)) if vals.size else None
        elif vals.size >= max(3, len(rows) // 3):
            feats[name] = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            quality[name + "_iqr"] = float(q3 - q1)
        else:
            feats[name] = None
    out["features"] = feats
    out["quality"] = quality
    return out


# --------------------------------------- beat pairing / windowing
def _beat_pairs(beat_t: np.ndarray, conf: np.ndarray,
                min_conf: float) -> list:
    """(t0, t1) pairs of beats that are ADJACENT IN THE LATTICE with both
    endpoints at/above the production confidence floor. Pairing after
    filtering would silently span dropped low-confidence beats and embed
    sub-floor morphology in a two-cycle 'beat' (review finding, V-c)."""
    out = []
    for k in range(len(beat_t) - 1):
        if conf[k] >= min_conf and conf[k + 1] >= min_conf:
            out.append((float(beat_t[k]), float(beat_t[k + 1])))
    return out


def _beat_windows(wave: np.ndarray, seg_ts: np.ndarray,
                  pairs: list) -> list:
    """Foot-to-foot segments around adjacent production beat pairs."""
    out = []
    for t0, t1 in pairs:
        rr = t1 - t0
        if not (RR_PLAUSIBLE_S[0] <= rr <= RR_PLAUSIBLE_S[1]):
            continue
        a0 = np.searchsorted(seg_ts, t0 - 0.35 * rr)
        a1 = np.searchsorted(seg_ts, t0)
        b0 = np.searchsorted(seg_ts, t1 - 0.35 * rr)
        b1 = np.searchsorted(seg_ts, t1)
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        foot1 = a0 + int(np.argmin(wave[a0:a1]))
        foot2 = b0 + int(np.argmin(wave[b0:b1]))
        if foot2 - foot1 >= 6:
            out.append((foot1, foot2))
    return out
