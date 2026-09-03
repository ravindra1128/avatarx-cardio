"""Vasomotor tone features (v0.5 vasotone track, T1) — QUARANTINED.

Everything here is a WITHIN-SESSION comparison: every feature is emitted
as (baseline, response, delta, delta_norm) and never as an absolute
level (invariant W-c — absolute cross-session tone is presumed
non-identifiable, threat T1). Amplitude-derived features additionally
require AE/AWB lock verified in the manifest; without it the session is
flagged `uncontrolled_optics` and those features are None (W-d).

Signal choices, stated for the record:
- AC/DC (normalized pulse amplitude, the perfusion-index analog) comes
  from the GREEN channel of the production ingest's ROI traces — the
  POS/CHROM projections normalize per window and partially destroy the
  amplitude scale they would otherwise carry. AC is the per-beat
  peak-to-peak of the band-passed green; DC is the raw green mean over
  the same beat. Dimensionless by construction.
- ROI combination is an SNR-screened median: ROIs whose in-band SNR
  falls below half the best ROI's are dropped, the rest contribute
  equally per beat.
- Vasomotor LF power uses only the frequencies a window can actually
  resolve (f >= 2 cycles per window); with 25 s windows that is the
  upper half of the 0.04-0.15 Hz band — reported honestly, never
  extrapolated.
- Notch relative amplitude and reflection index reuse the v0.4
  morphology machinery (ensemble beats per phase window).
"""
from __future__ import annotations

import numpy as np

from research.vascular.features import (MORPH_BAND_HZ, RR_PLAUSIBLE_S,
                                        _beat_pairs, _beat_windows,
                                        beat_morphology, ensemble_beat)

TONE_FEATURES = (
    "norm_pulse_amplitude",    # median per-beat AC/DC (green channel)
    "amplitude_trend",         # normalized within-window slope, 1/min
    "vasomotor_lf_power",      # LF fraction of the amplitude envelope
    "notch_rel_amp",           # morphology shift under provocation
    "reflection_index",
)
AMPLITUDE_FEATURES = ("norm_pulse_amplitude", "amplitude_trend",
                      "vasomotor_lf_power")
# already-normalized rates whose natural baseline is zero-centered: a
# ratio to a near-zero noise slope is ill-posed (review finding), so
# their delta_norm IS the delta
DIFFERENCE_SCALE_FEATURES = ("amplitude_trend",)
MIN_BEATS_PER_WINDOW = 6
LF_BAND_HZ = (0.04, 0.15)
_EPS = 1e-9


def _delta(baseline, response, difference_scale: bool = False) -> dict:
    """The ONLY emission shape for a tone feature (W-c). For
    difference-scale features (already-normalized zero-centered rates)
    delta_norm is the delta itself — a ratio to a near-zero baseline
    would be numerology (review finding)."""
    if baseline is None or response is None:
        return {"baseline": baseline, "response": response,
                "delta": None, "delta_norm": None}
    d = float(response) - float(baseline)
    dn = d if difference_scale else d / max(abs(float(baseline)), _EPS)
    return {"baseline": round(float(baseline), 5),
            "response": round(float(response), 5),
            "delta": round(d, 5),
            "delta_norm": round(dn, 5)}


def acdc_beats(green: np.ndarray, seg_ts: np.ndarray, pairs: list,
               fps: float) -> dict:
    """{lattice pair -> ac_over_dc} per adjacent accepted beat pair,
    from ONE ROI's raw green trace. Keyed by the SHARED production
    lattice pair — per-ROI foot midpoints almost never coincide across
    ROIs, so a midpoint-keyed merge silently skipped the cross-ROI
    median and inflated the beat series k-fold (review finding). AC
    from the band-passed green; DC the raw mean — both green-channel
    units, so the ratio is dimensionless."""
    from rppg._filters import bandpass
    hi = min(MORPH_BAND_HZ[1], 0.45 * fps)
    bp = bandpass(green - float(np.mean(green)), fps,
                  MORPH_BAND_HZ[0], hi)
    out = {}
    for pair, (f1, f2) in _beat_windows_keyed(bp, seg_ts, pairs):
        dc = float(np.mean(green[f1:f2]))
        if dc <= _EPS:
            continue
        ac = float(np.ptp(bp[f1:f2]))
        out[pair] = ac / dc
    return out


def _beat_windows_keyed(wave, seg_ts, pairs) -> list:
    """[(pair, (foot1, foot2))] — _beat_windows plus the identity of the
    lattice pair each window came from."""
    out = []
    for pair in pairs:
        win = _beat_windows(wave, seg_ts, [pair])
        if win:
            out.append((tuple(pair), win[0]))
    return out


def _roi_snr(green: np.ndarray, fps: float):
    """In-band (0.7-4 Hz) over out-of-band power of the green trace;
    None when the segment is too short to measure (fail closed — an
    unmeasurable SNR must not slip through the screen)."""
    x = np.asarray(green, float)
    x = x - float(np.mean(x))
    if x.size < int(4 * fps):
        return None
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / fps)
    inb = spec[(f >= 0.7) & (f <= 4.0)].sum()
    outb = spec[(f > 0.1) & (f < 0.7)].sum() + \
        spec[(f > 4.0) & (f < 8.0)].sum()
    return float(inb / max(outb, _EPS))


def windowed_median(series: list, span) -> tuple:
    """(median value, n) of an amplitude series inside [t0, t1)."""
    t0, t1 = float(span[0]), float(span[1])
    vals = [v for t, v in series if t0 <= t < t1]
    if len(vals) < MIN_BEATS_PER_WINDOW:
        return None, len(vals)
    return float(np.median(vals)), len(vals)


def windowed_trend(series: list, span) -> float:
    """Normalized amplitude slope inside the window (fraction of the
    window median per minute) — Theil-Sen for robustness."""
    t0, t1 = float(span[0]), float(span[1])
    pts = [(t, v) for t, v in series if t0 <= t < t1]
    if len(pts) < MIN_BEATS_PER_WINDOW:
        return None
    med = float(np.median([v for _, v in pts]))
    if abs(med) < _EPS:
        return None
    slopes = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dt = pts[j][0] - pts[i][0]
            if dt > 1e-3:
                slopes.append((pts[j][1] - pts[i][1]) / dt)
    if not slopes:
        return None
    return float(np.median(slopes) * 60.0 / med)


def lf_power_fraction(series: list, span) -> float:
    """Fraction of amplitude-envelope power in the resolvable part of
    the vasomotor LF band inside the window. Resolvability and the
    analysis grid come from the ACTUAL data coverage, never the nominal
    window — interpolation must not extrapolate beyond the beats or
    bridge long interior gaps (review finding: fabricated LF power)."""
    t0, t1 = float(span[0]), float(span[1])
    pts = sorted((t, v) for t, v in series if t0 <= t < t1)
    if len(pts) < MIN_BEATS_PER_WINDOW:
        return None
    tt = np.asarray([t for t, _ in pts])
    vv = np.asarray([v for _, v in pts])
    covered = float(tt[-1] - tt[0])
    if covered < 10.0:
        return None
    if float(np.max(np.diff(tt))) > 5.0:
        return None                    # interior gap: no honest bridge
    f_lo_resolvable = 2.0 / covered
    lo = max(LF_BAND_HZ[0], f_lo_resolvable)
    if lo >= LF_BAND_HZ[1]:
        return None
    fs_u = 4.0
    grid = np.arange(float(tt[0]), float(tt[-1]), 1.0 / fs_u)
    if grid.size < 8:
        return None
    u = np.interp(grid, tt, vv)
    u = u - float(np.mean(u))
    spec = np.abs(np.fft.rfft(u)) ** 2
    f = np.fft.rfftfreq(u.size, 1.0 / fs_u)
    # bin-overlap selection: a bin whose width straddles the resolvable
    # band edge still carries that band's power (a hard >= cut dropped
    # the peak bin at coarse spectral resolution)
    df = float(f[1]) if f.size > 1 else 0.0
    band = spec[(f + df / 2 >= lo)
                & (f - df / 2 <= LF_BAND_HZ[1])].sum()
    total = spec[(f >= 0.01) & (f <= 0.4)].sum()
    if total <= _EPS:
        return None
    return float(band / total)


def _window_morphology(waves: dict, seg_ts: np.ndarray, pairs: list,
                       span, fps: float) -> dict:
    """Ensemble-beat morphology per phase window, median across ROIs
    (the v0.4 session statistic, window-restricted)."""
    t0, t1 = float(span[0]), float(span[1])
    in_win = [(a, b) for a, b in pairs if t0 <= a and b < t1]
    vals = {"notch_rel_amp": [], "reflection_index": []}
    for roi, wave in waves.items():
        segs = [wave[f1:f2] for f1, f2 in
                _beat_windows(wave, seg_ts, in_win)]
        if len(segs) < MIN_BEATS_PER_WINDOW:
            continue
        ens, dur = ensemble_beat(segs, fps)
        if ens is None:
            continue
        m = beat_morphology(ens, ens.size / dur, native_fs=fps)
        if m is None:
            continue
        for k in vals:
            if m.get(k) is not None:
                vals[k].append(m[k])
    return {k: (float(np.median(v)) if v else None)
            for k, v in vals.items()}


def tone_session_features(result, det, provocation, capture=None) -> dict:
    """Tone features for ONE provocation session through the production
    path. V-c (ACCEPT only) and W-d (AE/AWB lock for amplitude
    features) enforced here; every feature is a within-session
    (baseline, response, delta, delta_norm) — nothing absolute (W-c)."""
    from datasets.schema import ScanOutcome
    from inference.pipeline import CALIBRATED_MIN_CONF
    from inference.evidence import capture_segments
    from preprocessing.roi import ROI_NAMES
    from rppg.pos import pos_pulse, orient_rois_consistently

    if result.outcome is not ScanOutcome.ACCEPT:
        return {"available": False, "outcome": result.outcome.value,
                "reasons": ["V-c: tone features are computed only on "
                            "ACCEPT-grade scans"]
                + list(result.no_read_reasons or [])}
    cap = capture or {}
    locked = bool(cap.get("exposure_locked")) and \
        bool(cap.get("awb_locked"))
    marks = provocation.phase_marks
    base_span = marks["baseline"]
    resp_span = marks["stimulus"]
    ing = det["ingest"]
    lattice = det["lattice"]
    ts = np.asarray(ing.timestamps_s, float)
    fps = float(ing.meta.measured_fps_mean)
    min_conf = float(det.get("min_conf", CALIBRATED_MIN_CONF))
    pairs = _beat_pairs(np.asarray(lattice.beat_t_s, float),
                        np.asarray(lattice.beat_confidence, float),
                        min_conf)
    hi = min(MORPH_BAND_HZ[1], 0.45 * fps)

    namp_series: list = []
    morph_base: dict = {"notch_rel_amp": [], "reflection_index": []}
    morph_resp: dict = {"notch_rel_amp": [], "reflection_index": []}
    for a, b in capture_segments(ts, fps):
        seg_ts = ts[a:b]
        in_seg = [(p0, p1) for p0, p1 in pairs
                  if p0 >= seg_ts[0] and p1 <= seg_ts[-1]]
        greens = {r: np.asarray(ing.traces[r][a:b, 1], float)
                  for r in ROI_NAMES}
        snrs = {r: _roi_snr(g, fps) for r, g in greens.items()}
        measured = {r: s for r, s in snrs.items() if s is not None}
        best = max(measured.values()) if measured else 0.0
        # fail closed: an unmeasurable or zero-power segment
        # contributes no beats (review finding: 0.5*0 kept everything)
        keep = ([] if best <= 0.0 else
                [r for r, s in measured.items() if s >= 0.5 * best])
        per_roi = {r: acdc_beats(greens[r], seg_ts, in_seg, fps)
                   for r in keep}
        # merge per PHYSICAL lattice pair: one series point per beat,
        # median across the ROIs that read it (review finding: float
        # midpoint keys almost never matched across ROIs)
        for pair in in_seg:
            key = tuple(pair)
            vs = [m[key] for m in per_roi.values() if key in m]
            if vs:
                namp_series.append(((key[0] + key[1]) / 2.0,
                                    float(np.median(vs))))
        waves = orient_rois_consistently(
            {r: pos_pulse(ing.traces[r][a:b], fps,
                          band=(MORPH_BAND_HZ[0], hi))
             for r in ROI_NAMES})
        mb = _window_morphology(waves, seg_ts, in_seg, base_span, fps)
        mr = _window_morphology(waves, seg_ts, in_seg, resp_span, fps)
        for k in morph_base:
            if mb.get(k) is not None:
                morph_base[k].append(mb[k])
            if mr.get(k) is not None:
                morph_resp[k].append(mr[k])

    features: dict = {}
    n_base = n_resp = 0
    if locked:
        b_amp, n_base = windowed_median(namp_series, base_span)
        r_amp, n_resp = windowed_median(namp_series, resp_span)
        features["norm_pulse_amplitude"] = _delta(b_amp, r_amp)
        features["amplitude_trend"] = _delta(
            windowed_trend(namp_series, base_span),
            windowed_trend(namp_series, resp_span),
            difference_scale=True)
        features["vasomotor_lf_power"] = _delta(
            lf_power_fraction(namp_series, base_span),
            lf_power_fraction(namp_series, resp_span))
    else:
        for k in AMPLITUDE_FEATURES:
            features[k] = _delta(None, None)
    for k in ("notch_rel_amp", "reflection_index"):
        features[k] = _delta(
            float(np.median(morph_base[k])) if morph_base[k] else None,
            float(np.median(morph_resp[k])) if morph_resp[k] else None)

    usable = any(v["delta_norm"] is not None for v in features.values())
    return {"available": usable,
            "uncontrolled_optics": not locked,
            "outcome": result.outcome.value,
            "maneuver": provocation.maneuver,
            "features": features,
            "n_beats": {"baseline": n_base, "response": n_resp,
                        "series": len(namp_series)},
            "reasons": ([] if usable else
                        ["no feature produced a within-session delta "
                         "(too few usable beats per window)"])
            + (["W-d: uncontrolled optics — amplitude features "
                "withheld and this session is excluded from gate "
                "evaluations"] if not locked else [])}


def pi_response(pi, base_span, resp_span) -> dict:
    """The contact reference's within-session response, in the SAME
    (baseline, response, delta, delta_norm) shape (W-c applies to the
    reference too)."""
    v = np.asarray(pi.values, float)
    t = pi.t0_video_s + np.arange(v.size) / float(pi.fs_hz)

    def med(span):
        m = (t >= float(span[0])) & (t < float(span[1]))
        return float(np.median(v[m])) if int(m.sum()) >= 3 else None

    return _delta(med(base_span), med(resp_span))
