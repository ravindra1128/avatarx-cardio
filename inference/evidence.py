"""
Shared signal-EVIDENCE computation (v0.1.3).

ONE function turns per-ROI RGB traces + the capture clock into the evidence
the decision gates consume — and the live demo runs the SAME function on
its rolling window to decide whether it may START the scan. That is what
makes the pre-scan gate a real predictor of the post-scan verdict: same
extractor, same detector, same fusion, same SQI, same timing-precision
proxy, same thresholds (configs/default.yaml → decision.evidence /
decision.readiness). If readiness said READY on the live window, the
recording made under those conditions clears the same gates.

READINESS THRESHOLDS = THE ANY-RESULT GATES (v0.1.4, docs/READINESS.md).
The parity rule: the gate may demand exactly the evidence a RESULT needs
(decision.evidence any-class values) — never the stricter AF-call bars,
which the decision applies only to the AF call itself on the full
recording, where that evidence actually accrues; and never a quantity the
decision does not gate at all. v0.1.3 violated both (AF bars per 8 s
window; jitter/multi-ROI/hole vetoes with no decision counterpart) and
measured READY on 0% of evaluations under a routine browser frame-drop
pattern whose full scan was ACCEPT/SINUS.
  frame_rate       ≥ 24 fps    consumer capture floor (research 30); below
                               it the RECORDING would be invalid
  timestamps       analysable COVERAGE of the window ≥ min_coverage_any
                   (0.60): gap-aware segments, exactly the pipeline's own
                   treatment of holes. Isolated drops are forgiven (the
                   pipeline splits segments); recurring drops that shred
                   every segment fail. Jitter is diagnostic only — its
                   beat-level effect is what beat_timing measures.
  face / framing   found; width 0.20-0.70 of frame; centred within 22%
  lighting         FACE-REGION luma ≥ 60/255 and face-region lux proxy
                   ≥ 100 (spec capture gate; the photons that matter fall
                   on the skin, not the background)
  exposure         no ≥12% step between ADJACENT 0.5 s luma bins in the
                   last 2 s (an active AE transient); slow drift is removed
                   by the 0.7 Hz high-pass and does not block
  motion           median centre move < 3.5% face width per frame
  tracking         raw-observation stability ≥ 0.6
  signal_snr       ≥ 2 ROIs with in-band SNR component ≥ 0.5 (= +3 dB, the
                   T4 logistic centre: pulse-band energy > 2x out-of-band)
  cross_roi_coh.   ≥ coherence_floor (0.20) — what ANY result needs
  beat_timing      best ROI-pair median |Δt| ≤ max_timing_precision_ms_any
                   (40 ms, the any-class timing gate)
  prelim_beats     ≥ 4 fused beats in the window (presence only — no
                   rhythm/burden test, no multi-ROI quota: cross-ROI
                   membership is what coherence measures)
  sqi              composite ≥ sqi_floor (0.30; noise ~0.2)
Held for hold_s (3 s); a check must fail 2 consecutive evaluations to
break the hold (engine debounce — estimator flicker on one 0.5 s window
must not discard accumulated readiness).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from preprocessing.roi import ROI_NAMES
from rppg.pos import pos_pulse, orient_rois_consistently
from rppg.chrom import chrom_pulse
from rppg.signal_quality import compute_sqi
from beats.detector import detect_beats_single_roi, fuse_multi_roi
from beats.ibi import clean_runs
from inference.decision_logic import timing_precision_from_trains, \
    harmonic_fraction

_EXTRACTORS = {"pos": pos_pulse, "chrom": chrom_pulse}

_DEFAULT_READINESS = {
    "mode": "advisory",           # v0.1.5: 'advisory' starts on blocking
                                  # checks only; 'blocking' = pre-v0.1.5
    "blocking_checks": ["face", "framing", "frame_rate_floor"],
    "min_fps_hard": 15.0,
    "hold_s": 3.0,
    "window_s": 8.0,
    "min_fps": 24.0,
    "min_snr_rois": 2,
    "min_roi_snr_component": 0.5,
    "min_prelim_beats": 4,
    "min_tracking_stability": 0.6,
    "max_collapsed_interval_fraction": 0.02,
    "debounce_evals": 2,          # consecutive failing evals to break hold
    "max_paused_s": 20.0,
    "max_pauses": 4,
    "resume_hold_s": 1.0,
}


def capture_segments(ts: np.ndarray, fps: float, gap_factor: float = 1.5,
                     min_seconds: float = 3.0) -> list:
    """[(start_idx, end_idx)] contiguous stretches of the capture clock.

    A frame-to-frame step > gap_factor x the nominal period is a hole
    (dropped frames or an intentional pause). Segments shorter than
    `min_seconds` cannot support extraction and are skipped.

    The 1.5x default is deliberate: one missing 30-fps frame creates a 2x
    interval and must not be silently analysed as uniformly sampled data.
    """
    ts = np.asarray(ts, float)
    if ts.size and (not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0)):
        return []
    if ts.size < 2:
        return [(0, ts.size)] if ts.size else []
    period = 1.0 / fps if fps > 0 else float(np.median(np.diff(ts)))
    cut = np.flatnonzero(np.diff(ts) > gap_factor * period) + 1
    bounds = [0] + cut.tolist() + [ts.size]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= 2 and (ts[b - 1] - ts[a]) >= min_seconds:
            out.append((a, b))
    return out


def extract_and_detect(traces: dict, ts: np.ndarray, fps: float, cfg: dict,
                       *, for_window: bool = False) -> tuple[dict, dict, list]:
    """Per-ROI raw waveforms, per-ROI beats ON THE CAPTURE CLOCK, segments.

    Traces are split at capture gaps; each segment is extracted/detected on
    its own; ROI polarity is made consistent per segment; beat times are
    mapped index -> sidecar timestamp within the segment.

    `for_window` (readiness): pieces truncated by the WINDOW edge are real
    contiguous data that continues beyond it — analyse them from 2 s (the
    extractor's floor) instead of the 3 s full-segment floor, else a hole
    phasing through the sliding window starves the beat/timing estimators
    at some phases. Interior sub-3 s pieces stay excluded everywhere: the
    full pipeline discards them, and the window must predict the pipeline.
    """
    ts = np.asarray(ts, float)
    extractor = _EXTRACTORS[cfg["decision"]["extractor"]]
    band = tuple(cfg["sqi"]["band_hz"])
    if for_window:
        segments = []
        for a, b in capture_segments(ts, fps, min_seconds=0.0):
            d = float(ts[b - 1] - ts[a])
            if d >= 3.0 or ((a == 0 or b == ts.size) and d >= 2.0):
                segments.append((a, b))
    else:
        segments = capture_segments(ts, fps)
    per_roi_beats = {r: [] for r in ROI_NAMES}
    # SQI must only see analysable samples.  The previous zero-filled array
    # treated discarded short fragments as a flat physiological signal and
    # introduced artificial edges at capture gaps.
    raw_parts = {r: [] for r in ROI_NAMES}
    for a, b in segments:
        filt = {}
        for roi in ROI_NAMES:
            tr = np.asarray(traces[roi])[a:b]
            raw_parts[roi].append(extractor(tr, fps, band=None))
            filt[roi] = extractor(tr, fps, band=band)
        filt = orient_rois_consistently(filt)
        for roi in ROI_NAMES:
            for beat in detect_beats_single_roi(filt[roi], fps, roi):
                idx = beat.t_s * fps
                beat.t_s = float(np.interp(idx, np.arange(b - a), ts[a:b]))
                per_roi_beats[roi].append(beat)
    raw = {r: (np.concatenate(raw_parts[r]) if raw_parts[r]
               else np.array([], dtype=float)) for r in ROI_NAMES}
    return raw, per_roi_beats, segments


def window_evidence(traces: dict, ts: np.ndarray, fps: float, cfg: dict, *,
                    tracking_stability: float = 1.0) -> dict:
    """Evidence dict for a window/recording: SQI + components, per-ROI SNR,
    coherence, timing precision, preliminary beats, clean-run burden,
    frame-rate/timestamp integrity. Pure function of its inputs."""
    ts = np.asarray(ts, float)
    n = ts.size
    dt = np.diff(ts) if n > 1 else np.array([])
    timestamp_valid = bool(n > 1 and np.all(np.isfinite(ts)) and
                           np.all(dt > 0))
    period = 1.0 / fps if fps > 0 else (
        float(np.median(dt)) if timestamp_valid else 0.0)
    med_dt = float(np.median(dt)) if timestamp_valid else 0.0
    fps_meas = float(1.0 / med_dt) if med_dt > 0 else 0.0
    collapsed = float(np.mean(dt < 0.5 * period)) \
        if timestamp_valid and period > 0 else 1.0
    ev = {"n_frames": int(n), "seconds": float(ts[-1] - ts[0])
          if timestamp_valid else 0.0,
          "fps": fps_meas,
          "max_gap_ms": float(np.max(dt) * 1000.0) if dt.size else 0.0,
          "jitter_ms": float(np.std(dt[dt < 2.5 * period]) * 1000.0)
          if timestamp_valid and period > 0 and np.any(dt < 2.5 * period)
          else 0.0,
          "timestamp_valid": timestamp_valid,
          "collapsed_interval_fraction": collapsed}
    trace_valid = all(
        r in traces and np.asarray(traces[r]).ndim == 2 and
        np.asarray(traces[r]).shape == (n, 3) and
        np.all(np.isfinite(np.asarray(traces[r]))) for r in ROI_NAMES)
    if not timestamp_valid or n < int(3.0 * fps) or not trace_valid:
        ev.update({"sqi": 0.0, "components": {}, "per_roi_snr": {},
                   "cross_roi_coherence": 0.0, "timing_precision_ms": float("nan"),
                   "timing_matched_fraction": 0.0, "n_beats": 0,
                   "frac_multi_roi": 0.0, "split_fraction": 0.0,
                   "harmonic_fraction": 0.0, "n_intervals": 0,
                   "window_coverage": 0.0, "insufficient": True,
                   "insufficient_reason": (
                       "invalid/non-monotone timestamps" if not timestamp_valid
                       else "ROI traces missing, non-finite, or wrong length"
                       if not trace_valid else "window shorter than 3 seconds")})
        return ev
    raw, per_roi_beats, segments = extract_and_detect(traces, ts, fps, cfg,
                                                      for_window=True)
    fused = fuse_multi_roi(per_roi_beats, fps, float(ts[-1] - ts[0]) + period,
                           min_rois=2)
    sq = compute_sqi(raw, fps, fused, tracking_stability=tracking_stability)
    agree = np.array([b.roi_agreement for b in fused.beats]) if fused.beats \
        else np.array([])
    tp = timing_precision_from_trains(
        {r: np.array([b.t_s for b in bs]) for r, bs in per_roi_beats.items()})
    ibi = np.diff(fused.times()) * 1000.0 if len(fused.beats) > 1 else np.array([])
    rs = clean_runs(fused, min_conf=0.0, min_run_beats=4)   # structure only
    ev.update({
        "sqi": float(sq.sqi), "components": dict(sq.components),
        "per_roi_snr": {r: float(v.get("snr_in_band", 0.0))
                        for r, v in sq.per_roi.items()},
        "cross_roi_coherence": float(sq.components.get("cross_roi_coherence", 0.0)),
        "timing_precision_ms": tp["timing_precision_ms"],
        "timing_matched_fraction": tp["timing_matched_fraction"],
        "timing_pair": tp.get("timing_pair"),
        "n_beats": int(len(fused.beats)),
        "frac_multi_roi": float(np.mean(agree >= 0.75)) if agree.size else 0.0,
        "split_fraction": float(rs.split_fraction),
        "harmonic_fraction": harmonic_fraction(ibi),
        "n_intervals": int(rs.n_intervals),
        "n_segments": len(segments),
        # analysable fraction of the window — an estimator of FULL-STREAM
        # coverage from this window. Interior pieces < 3 s (bounded by
        # holes on both sides) are what the pipeline itself discards and
        # do not count; pieces touching a window edge are truncated by
        # the WINDOW, not by holes — they continue beyond it and count at
        # their visible length (else a hole every ~4 s phases the sliding
        # window into 1 s + 3.9 s + 2.9 s cuts and coverage reads 0.48
        # for a stream whose full-scan coverage is 0.97).
        "window_coverage": _window_coverage(ts, fps),
        "insufficient": False,
    })
    return ev


def _window_coverage(ts: np.ndarray, fps: float) -> float:
    if ts.size < 2:
        return 0.0
    span = float(ts[-1] - ts[0])
    if span <= 0:
        return 0.0
    covered = 0.0
    for a, b in capture_segments(ts, fps, min_seconds=0.0):
        d = float(ts[b - 1] - ts[a])
        if d >= 3.0 or a == 0 or b == ts.size:
            covered += d
    return float(covered / span)


def readiness_from_evidence(ev: dict, cfg: dict, *, face_ok: bool = True,
                            framing: str = "ok", lighting_ok: bool = True,
                            exposure_ok: bool = True, motion_ok: bool = True,
                            tracking_stability: Optional[float] = None,
                            lighting_status: Optional[str] = None,
                            delivery_ok: bool = True) -> dict:
    """The READY / NOT READY checklist. Every check: value, threshold, pass,
    and one actionable hint for the first failing check (in priority order:
    the user must fix framing before light before stillness before signal)."""
    dc = cfg["decision"]
    ec = dc.get("evidence") or {}
    rc = {**_DEFAULT_READINESS, **(dc.get("readiness") or {})}
    checks: dict = {}

    def chk(name, value, thr, op, ok=None, hint=""):
        v = value
        if ok is None:
            try:
                ok = bool(np.isfinite(float(v))) and (
                    (float(v) >= thr) if op == ">=" else (float(v) <= thr))
            except (TypeError, ValueError):
                ok = False
        checks[name] = {"value": (None if v is None else v), "threshold": thr,
                        "op": op, "pass": bool(ok), "hint": hint}

    # root-cause order: darkness also loses the face, so lighting is judged
    # (and hinted) first; then face/framing; then exposure/motion; then signal
    light_hint = {
        "dark": "Improve lighting: face a bright, even light",
        "overexposed": "Face is overexposed — move away from direct light",
        "uneven": "Use even light across both sides of your face",
    }.get(lighting_status or ("ok" if lighting_ok else "dark"),
          "Improve lighting: face a bright, even light")
    chk("lighting", lighting_status or lighting_ok, "ok", "is",
        ok=lighting_ok, hint=light_hint)
    chk("face", face_ok, True, "is", ok=face_ok,
        hint="Position your face inside the oval")
    fr_hint = {"too_far": "Move a little closer", "too_close": "Move back a little",
               "off_centre": "Centre your face in the oval",
               "no_face": "Position your face inside the oval"}.get(framing, "")
    chk("framing", framing, "ok", "is", ok=(framing == "ok"), hint=fr_hint)
    chk("exposure", exposure_ok, True, "is", ok=exposure_ok,
        hint="Lighting is changing — hold still and wait for the camera to settle")
    chk("motion", motion_ok, True, "is", ok=motion_ok, hint="Hold still")
    chk("frame_delivery", delivery_ok, True, "is", ok=delivery_ok,
        hint="Camera frames are repeating or timestamps are invalid — close "
             "other camera apps and retry")
    chk("frame_rate", ev.get("fps", 0.0), rc["min_fps"], ">=",
        hint="Camera frame rate is too low — close other apps using the camera")
    # hard floor (v0.1.5, BLOCKING): below this the camera is unusable;
    # the 24 fps capture floor above stays advisory + post-scan gate
    chk("frame_rate_floor", ev.get("fps", 0.0), rc["min_fps_hard"], ">=",
        hint="Camera is not delivering usable video — close other apps or "
             "try a different camera")
    if tracking_stability is not None:
        chk("tracking", tracking_stability, rc["min_tracking_stability"], ">=",
            hint="Hold still — face tracking is unstable")
    # PARITY (v0.1.4): every signal threshold below is the ANY-RESULT gate
    # the decision itself applies (decision.evidence), read from the same
    # config keys so the two can never drift apart. Jitter and multi-ROI
    # quotas are gone: the decision gates neither (beat_timing measures
    # jitter's effect; coherence measures cross-ROI membership).
    cov_thr = float(ec.get("min_coverage_any", 0.60))
    coh_thr = float(ec.get("coherence_floor", 0.20))
    tp_thr = float(ec.get("max_timing_precision_ms_any", 40.0))
    if ev.get("insufficient"):
        time_hint = ("Camera timestamps are invalid — restart the camera"
                     if "timestamp" in str(ev.get("insufficient_reason", ""))
                     else "Pulse signal stabilizing…")
        chk("timestamps", None, cov_thr, ">=", ok=False,
            hint=time_hint)
        chk("signal_snr", None, rc["min_snr_rois"], ">=", ok=False,
            hint="Pulse signal stabilizing…")
        chk("cross_roi_coherence", None, coh_thr,
            ">=", ok=False, hint="Pulse signal stabilizing…")
        chk("beat_timing", None, tp_thr,
            "<=", ok=False, hint="Pulse signal stabilizing…")
        chk("prelim_beats", 0, rc["min_prelim_beats"], ">=", ok=False,
            hint="Pulse signal stabilizing…")
        chk("sqi", 0.0, dc["sqi_floor"], ">=", ok=False,
            hint="Pulse signal stabilizing…")
    else:
        # coverage, not hole-existence: an isolated dropped batch is a
        # segment boundary the pipeline handles (measured: 4-frame hole
        # every 4 s -> full scan ACCEPT); only drops that shred the
        # window's segments block readiness.
        cov = round(float(ev.get("window_coverage", 0.0)), 3)
        collapsed = float(ev.get("collapsed_interval_fraction", 0.0))
        timestamp_ok = (bool(ev.get("timestamp_valid", True)) and
                        cov >= cov_thr and
                        collapsed <= rc["max_collapsed_interval_fraction"])
        chk("timestamps", cov, cov_thr, ">=", ok=timestamp_ok,
            hint="Frames are being dropped — close other apps, "
                 "reduce browser load")
        checks["timestamps"]["collapsed_interval_fraction"] = collapsed
        checks["timestamps"]["max_collapsed_interval_fraction"] = \
            rc["max_collapsed_interval_fraction"]
        n_snr = sum(1 for v in ev["per_roi_snr"].values()
                    if v >= rc["min_roi_snr_component"])
        chk("signal_snr", n_snr, rc["min_snr_rois"], ">=",
            hint="Pulse signal weak — more light on your face, no hair over "
                 "the forehead")
        # two-region verification (v0.1.4.2, mirrors decision_logic): with
        # exactly two strong ROIs the >=3-ROI coherence component reads
        # 0.00 by construction; two regions placing the same beats within
        # the AF-grade timing budget are independent verification (the
        # real FP scans failed that budget: 36-49 ms, matched 0.63-0.75)
        tpv = ev["timing_precision_ms"]
        tmv = ev["timing_matched_fraction"]
        two_region = (np.isfinite(tpv) and
                      tpv <= float(ec.get("afib_max_timing_precision_ms", 30.0))
                      and np.isfinite(tmv) and
                      tmv >= float(ec.get("afib_min_timing_matched", 0.75)))
        coh_ok = float(ev["cross_roi_coherence"]) >= coh_thr
        chk("cross_roi_coherence", ev["cross_roi_coherence"], coh_thr, ">=",
            ok=(coh_ok or two_region),
            hint="Pulse not seen consistently across your face — even light on "
                 "both cheeks and forehead")
        if not coh_ok and two_region:
            checks["cross_roi_coherence"]["mode"] = "two_region_verified"
        tp_ok = (np.isfinite(ev["timing_precision_ms"]) and
                 ev["timing_precision_ms"] <= tp_thr)
        chk("beat_timing", ev["timing_precision_ms"], tp_thr, "<=", ok=tp_ok,
            hint="Beat timing not yet precise — hold still, breathe normally")
        # NOTE: no split/harmonic burden here — on an 8 s window (~12
        # intervals) one genuine AF short pair reads as 17% and would gate
        # AF users out (a periodicity penalty). Burden is judged on the
        # whole scan by the decision; readiness judges beat PRESENCE only.
        chk("prelim_beats", ev["n_beats"], rc["min_prelim_beats"], ">=",
            hint="Pulse signal stabilizing…")
        chk("sqi", ev["sqi"], dc["sqi_floor"], ">=",
            hint="Signal quality low — improve lighting and hold still")

    failing = [k for k, v in checks.items() if not v["pass"]]
    hint = next((checks[k]["hint"] for k in checks if not checks[k]["pass"]
                 and checks[k]["hint"]), "Great — hold still")
    # v0.1.5 two-tier split: blocking checks are "you are not pointing a
    # working camera at a face"; everything else is ADVISORY — same
    # thresholds, same hints, but it grades the scan (confidence stars)
    # instead of holding the Start button. mode 'blocking' preserves the
    # pre-v0.1.5 behaviour: every check gates the start.
    mode = str(rc.get("mode", "advisory"))
    blocking_names = list(rc.get("blocking_checks",
                                 ["face", "framing", "frame_rate_floor"]))
    blocking_fail = [k for k in failing if k in blocking_names]
    advisory_fail = [k for k in failing if k not in blocking_names]
    can_start = (not blocking_fail) if mode == "advisory" else (not failing)
    return {"ready": not failing, "checks": checks, "failing": failing,
            "hint": hint, "mode": mode,
            "blocking_pass": not blocking_fail,
            "advisory_pass": not advisory_fail,
            "can_start": bool(can_start)}
