"""Resting hemodynamic indices from ONE scan (v0.8).

The v0.4/v0.5 vascular tracks were built around studies: arterial
stiffness needs a model trained against carotid-femoral PWV, and
vasomotor REACTIVITY needs a timed provocation. Neither is derivable
from a single resting scan, and neither is what this module claims.

What IS computable from one resting facial scan, and is computed here:

* **Contour markers of arterial stiffness.** The pulse contour carries
  the reflected wave. Its two established single-waveform markers are
  the REFLECTION INDEX (diastolic-peak height over systolic-peak
  height) and the second-derivative AGING INDEX, (b - c - d - e) / a
  (Takazawa et al. 1998). Both are direct formulas over fiducials this
  repository already extracts: no model, no training set, no
  calibration. They are dimensionless indices that move with arterial
  stiffness. They are not a pulse-wave velocity and cannot be converted
  to one without the referenced study the vascular gates describe.
* **Resting vasomotor indices.** Beat-to-beat pulse AMPLITUDE varies
  with small-vessel tone. Absolute amplitude is the most
  optics-confounded quantity in this pipeline (v0.5 T1), so nothing
  absolute is reported: the amplitude series is normalised by its own
  median, making both outputs dimensionless - its coefficient of
  variation, and the fraction of its power in the vasomotion band
  (0.04-0.15 Hz). A capture without exposure and white-balance lock is
  flagged, because unlocked optics move amplitude on their own.
* **Resting cardiorespiratory indices.** Resting heart rate and
  interval dispersion, from the one canonical representation
  (`features/regularity.py`), plus a bounded resting-fitness proxy built
  from them.  The proxy is surfaced as a numeric Research Estimate /
  Prototype so paired-reference collection can start now.  It is not
  relabelled as oxygen uptake: the mL/kg/min slot remains empty until a
  fitted, versioned model artifact exists.

EVERY value here is UNCALIBRATED. Each family carries `calibrated:
False`, the definition it was computed from, and what a validated
number would require. The report serializer must preserve that research
label, the method, and the scan-evidence confidence beside every value.
"""
from __future__ import annotations

import numpy as np

# Vasomotion / low-frequency band of the amplitude envelope (Hz): the
# classic Mayer-wave band, the same 0.04-0.15 Hz the HRV literature
# calls LF, applied to AMPLITUDE rather than to intervals.
VASOMOTION_BAND_HZ = (0.04, 0.15)
MIN_AMPLITUDE_BEATS = 12          # below this the envelope has no spectrum
MIN_ENVELOPE_SPAN_S = 30.0        # under ~2 cycles of the slowest band edge

# PLANNING reference points for the autonomic index. They are population
# anchors, not a calibration: an adult resting pulse near 60 with RMSSD
# near 40 ms sits mid-scale. They live here, named and visible, so the
# index is reproducible and so replacing them with fitted values is a
# one-line change once a reference cohort exists.
HR_REF_BPM = 60.0
HR_SPREAD_BPM = 12.0
RMSSD_REF_MS = 40.0
RESEARCH_ESTIMATE_LABEL = "Research Estimate / Prototype"


def aging_index(contour: dict):
    """Takazawa second-derivative aging index, (b - c - d - e) / a.

    Each ratio is already normalised by a, so the index is their signed
    combination. None unless every component survived extraction: the
    SDPPG waves need real sampling rate and are never interpolated into
    existence."""
    keys = ("sdppg_b_over_a", "sdppg_c_over_a", "sdppg_d_over_a",
            "sdppg_e_over_a")
    vals = [contour.get(k) for k in keys]
    if any(v is None or not np.isfinite(float(v)) for v in vals):
        return None
    b, c, d, e = (float(v) for v in vals)
    return round(b - c - d - e, 5)


def stiffness_contour(contour: dict) -> dict:
    """The single-scan arterial-stiffness readout: dimensionless contour
    markers, never a velocity."""
    def finite(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    ri = finite(contour.get("reflection_index"))
    agi = aging_index(contour)
    rise = finite(contour.get("rise_time_s"))
    slope = finite(contour.get("norm_upstroke_slope"))
    width = finite(contour.get("pulse_width50_s"))
    out = {
        "aging_index": agi,
        "reflection_index": (None if ri is None else round(float(ri), 5)),
        "rise_time_s": rise,
        "norm_upstroke_slope": slope,
        "pulse_width50_s": width,
        "notch_time_frac": contour.get("notch_time_frac"),
        "notch_detect_fraction": contour.get("notch_present"),
        "calibrated": False,
        "definition": ("aging_index = (b - c - d - e)/a on the pulse "
                       "second derivative (Takazawa 1998); "
                       "reflection_index = (diastolic peak - foot) / "
                       "(systolic peak - foot). Dimensionless contour "
                       "markers that move with arterial stiffness."),
        "not_a_velocity": ("a carotid-femoral pulse-wave velocity in m/s "
                           "requires the referenced study the vascular "
                           "gates describe: simultaneous tonometry on a "
                           "participant-disjoint cohort. These indices "
                           "are the camera-observable part of it."),
    }
    # Prefer the two established contour indices.  At ordinary consumer
    # frame rates the SDPPG index is structurally unavailable and a clean
    # pulse does not always expose a dicrotic notch.  In that case expose
    # the directly-computed systolic rise time rather than turning a valid
    # scan into a blank card.  It is explicitly a morphology proxy, not a
    # calibrated stiffness or pulse-wave-velocity measurement.
    candidates = (
        ("second_derivative_aging_index", out["aging_index"],
         "dimensionless", "(b - c - d - e) / a on the ensemble SDPPG"),
        ("reflection_index", out["reflection_index"],
         "ratio", "reflected-wave height / systolic-wave height"),
        ("pulse_rise_time", (None if rise is None else
                             round(float(rise) * 1000.0, 1)),
         "ms", "pulse foot to systolic peak on the ensemble waveform"),
    )
    primary = next((x for x in candidates if x[1] is not None), None)
    out["estimate"] = (None if primary is None else {
        "label": RESEARCH_ESTIMATE_LABEL,
        "name": primary[0], "value": primary[1], "unit": primary[2],
        "method": primary[3],
    })
    out["available"] = primary is not None
    out["marker_completeness"] = sum(
        out.get(k) is not None for k in
        ("aging_index", "reflection_index", "rise_time_s",
         "norm_upstroke_slope", "pulse_width50_s"))
    if primary is None:
        out["reason"] = ("no supported pulse-contour marker survived "
                         "extraction; no arterial-stiffness proxy was "
                         "computed")
    return out


MIN_SEGMENT_BEATS = 4             # enough for a segment's own median to mean something


def segment_amplitudes(waves: dict, seg_ts: np.ndarray,
                       windows: dict) -> dict:
    """Per-ROI (time, amplitude / this segment's ROI median) for ONE capture
    segment. Normalising within the segment removes the per-ROI optical
    scale AND the camera-gain jump between segments (auto-exposure is not
    locked on a phone), which is the confound that makes an ABSOLUTE
    amplitude unusable across sessions (v0.5 T1/W-c).

    No 12-beat floor here. That floor exists so the scan's pooled envelope
    has a spectrum; applied per segment it dropped every region of a
    5-segment phone recording whose regions held 23-27 usable beats in
    total, because no single fragment reached 12."""
    out: dict = {}
    for roi, wave in waves.items():
        amps = []
        for f1, f2 in windows.get(roi, []):
            seg = np.asarray(wave[f1:f2], float)
            if seg.size >= 4 and np.all(np.isfinite(seg)):
                amps.append((float(seg_ts[f1]),
                             float(np.max(seg) - np.min(seg))))
        if len(amps) < MIN_SEGMENT_BEATS:
            continue
        med = float(np.median([a for _, a in amps]))
        if med <= 1e-12:
            continue
        out[roi] = [(t, a / med) for t, a in amps]
    return out


def pool_amplitudes(per_roi: dict) -> list:
    """Pool the (already normalised) per-ROI series beat by beat. A ROI
    needs MIN_AMPLITUDE_BEATS over the whole scan. Region order is the
    insertion order of `per_roi` (ROI_NAMES) — it decides the reference
    series on ties, so it must not change."""
    pooled: list = []
    for roi, amps in per_roi.items():
        if len(amps) < MIN_AMPLITUDE_BEATS:
            continue
        pooled.append(sorted(amps, key=lambda x: x[0]))
    if not pooled:
        return []
    longest = max(pooled, key=len)
    out = []
    for t, _ in longest:
        vals = []
        for series in pooled:
            near = [a for tt, a in series if abs(tt - t) < 0.05]
            if near:
                vals.append(near[0])
        if vals:
            out.append((t, float(np.median(vals))))
    return out


def amplitude_series(waves: dict, seg_ts: np.ndarray,
                     windows: dict) -> list:
    """One-segment convenience, identical to the pre-split behaviour for a
    single segment: normalise within the segment, pool with the 12 floor."""
    return pool_amplitudes(segment_amplitudes(waves, seg_ts, windows))


def vasomotor_indices(series: list, locked: bool) -> dict:
    """Dimensionless resting tone readout from the normalised amplitude
    series: its coefficient of variation and its vasomotion-band power
    fraction."""
    out = {"available": False, "calibrated": False,
           "optics_locked": bool(locked),
           "n_beats": len(series or []),
           "amplitude_cv": None,
           "vasomotion_index": None,
           "estimate": None,
           "definition": ("amplitude_cv = SD/mean of per-beat pulse "
                          "amplitude after normalising each ROI by its "
                          "own median; vasomotion_index = fraction of "
                          "that series' power in 0.04-0.15 Hz. Both "
                          "dimensionless: nothing absolute is reported "
                          "(v0.5 W-c)."),
           "not_reactivity": ("vasomotor REACTIVITY is a response to a "
                              "timed provocation and is not derivable "
                              "from a resting scan; these are resting "
                              "tone indices only.")}
    if not locked:
        out["caveat"] = ("exposure and white balance were not locked for "
                         "this capture: the camera's own gain changes "
                         "move pulse amplitude, so these indices carry "
                         "an optical confound (v0.5 W-d)")
    if len(series or []) < MIN_AMPLITUDE_BEATS:
        out["reason"] = (f"{len(series or [])} usable beats "
                         f"(need at least {MIN_AMPLITUDE_BEATS})")
        return out
    t = np.asarray([x for x, _ in series], float)
    a = np.asarray([y for _, y in series], float)
    span = float(t[-1] - t[0])
    mean = float(np.mean(a))
    if mean <= 1e-12:
        out["reason"] = "degenerate amplitude series"
        return out
    out["available"] = True
    out["amplitude_cv"] = round(float(np.std(a) / mean), 5)
    out["estimate"] = {
        "label": RESEARCH_ESTIMATE_LABEL,
        "name": "normalized_pulse_amplitude_variability",
        "value": round(100.0 * out["amplitude_cv"], 2),
        "unit": "% CV",
        "method": ("coefficient of variation of per-beat facial pulse "
                   "amplitude after within-ROI median normalization"),
    }
    out["envelope_span_s"] = round(span, 1)
    if span < MIN_ENVELOPE_SPAN_S:
        out["reason_vasomotion"] = (
            f"envelope spans {span:.0f} s: under "
            f"{MIN_ENVELOPE_SPAN_S:.0f} s the vasomotion band cannot be "
            "resolved")
        return out
    fs = 4.0
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    y = np.interp(grid, t, a)
    y = y - y.mean()
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
    fr = np.fft.rfftfreq(y.size, 1.0 / fs)
    wide = (fr > 0.0) & (fr <= 0.5)
    band = (fr >= VASOMOTION_BAND_HZ[0]) & (fr <= VASOMOTION_BAND_HZ[1])
    total = float(spec[wide].sum())
    out["vasomotion_index"] = (round(float(spec[band].sum() / total), 5)
                               if total > 0 else None)
    return out


def autonomic_index(hr_bpm, rmssd_ms):
    """A bounded 0-1 composite of the two resting quantities a camera
    can measure: a lower resting pulse and a higher RMSSD both move it
    up. UNCALIBRATED by construction - the reference points above are
    population anchors, not fitted values - and deliberately NOT named
    after any physiological quantity it has not been validated against.
    None unless both inputs exist."""
    if hr_bpm is None or rmssd_ms is None:
        return None
    try:
        hr = float(hr_bpm)
        rm = float(rmssd_ms)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(hr) and np.isfinite(rm)) or hr <= 0 or rm <= 0:
        return None
    z_hr = (HR_REF_BPM - hr) / HR_SPREAD_BPM
    z_hrv = float(np.log(rm / RMSSD_REF_MS))
    z = 0.5 * z_hr + 0.5 * z_hrv
    return round(float(1.0 / (1.0 + np.exp(-z))), 5)


def resting_rate_index(hr_bpm):
    """Bounded HR-only prototype used when interval variability is absent.

    This is not an imputed-RMSSD version of :func:`autonomic_index`: it is a
    separate, explicitly named one-input transform over the measured resting
    pulse rate.  It lets paired-reference collection retain a real computed
    feature when the rhythm cleaner cannot support RMSSD.
    """
    if hr_bpm is None:
        return None
    try:
        hr = float(hr_bpm)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(hr) or hr <= 0:
        return None
    z = (HR_REF_BPM - hr) / HR_SPREAD_BPM
    return round(float(1.0 / (1.0 + np.exp(-z))), 5)


def cardiorespiratory_indices(regularity, hr_bpm, participant=None) -> dict:
    """Resting cardiorespiratory values, and an explicit account of why
    an oxygen-uptake number is not among them.

    What a resting scan gives is resting heart rate and interval
    dispersion, plus the bounded autonomic index built from them. Every
    published non-exercise oxygen-uptake equation is dominated by age,
    sex and body composition, which no camera measures; emitting one
    here would report demographics as a camera measurement. The honest
    routes are the three-phase recovery session, which measures
    heart-rate recovery and is already built, or a model fitted on
    captured scans against a reference, which is what `model_artifact`
    is reserved for."""
    disp = (getattr(regularity, "dispersion", None) or {}) if regularity \
        else {}
    pc = participant or {}
    rmssd = disp.get("rmssd_ms")
    autonomic = autonomic_index(hr_bpm, rmssd)
    rate_only = resting_rate_index(hr_bpm) if autonomic is None else None
    idx = autonomic if autonomic is not None else rate_only
    demographics = {
        "age_years": pc.get("age_years") or pc.get("age"),
        "sex": pc.get("sex"),
        "height_cm": pc.get("height_cm"),
        "weight_kg": pc.get("measured_weight_kg") or pc.get("weight_kg"),
        "habitual_activity": pc.get("activity_ipaq"),
    }
    missing = [k for k, v in demographics.items() if v is None]
    proxy = None if idx is None else round(100.0 * float(idx), 1)
    if autonomic is not None:
        estimate_name = "resting_cardiorespiratory_fitness_proxy"
        estimate_method = (
            "logistic transform of equal-weight resting-HR and RMSSD "
            "terms: 0.5*(60-HR)/12 + 0.5*ln(RMSSD/40)"
        )
        basis = "resting_hr_and_rmssd"
    else:
        estimate_name = "resting_rate_cardiorespiratory_proxy"
        estimate_method = (
            "heart-rate-only logistic research transform: (60-HR)/12; "
            "RMSSD was unavailable and was not imputed"
        )
        basis = "resting_hr_only"
    return {
        "available": idx is not None,
        "calibrated": False,
        "resting_hr_bpm": (None if hr_bpm is None
                           else round(float(hr_bpm), 2)),
        "rmssd_ms": rmssd,
        "sdnn_ms": disp.get("sdnn_ms"),
        "autonomic_index": autonomic,
        "resting_rate_index": rate_only,
        "fitness_proxy_score": proxy,
        "fitness_proxy_basis": basis if proxy is not None else None,
        "estimate": (None if proxy is None else {
            "label": RESEARCH_ESTIMATE_LABEL,
            "name": estimate_name,
            "value": proxy,
            "unit": "/100",
            "method": estimate_method,
        }),
        "autonomic_index_definition": (
            "bounded 0-1 composite: 0.5*(60 - resting HR)/12 + "
            "0.5*ln(RMSSD/40), through a logistic. A lower resting "
            "pulse and a higher RMSSD move it up. The reference points "
            "are population anchors, not fitted values."),
        "oxygen_uptake_estimate": None,
        "model_artifact": None,
        "demographics_present": demographics,
        "missing_for_a_fitted_model": missing,
        "why_no_oxygen_uptake_value": (
            "not derivable from a resting scan. The camera contributes "
            "resting heart rate and its variability; every published "
            "non-exercise equation is dominated by age, sex and body "
            "composition. Reporting one now would present demographics "
            "as a camera measurement. Two honest routes: the "
            "three-phase recovery session, which measures heart-rate "
            "recovery, or a model fitted on captured scans against a "
            "reference."),
    }


def _evidence_quality(det: dict, *, fps: float, n_beats: int,
                      n_rois: int, outcome: str) -> dict:
    """Existing scan-quality evidence, without inventing endpoint accuracy.

    `confidence_score` and stars are the production evidence grade.  They
    describe whether the camera signal supports the calculation; they are
    not a confidence interval or accuracy claim for an uncalibrated proxy.
    """
    sqi_obj = (det or {}).get("sqi")
    sqi = getattr(sqi_obj, "sqi", None)
    components = getattr(sqi_obj, "components", {}) or {}
    ev = (det or {}).get("evidence") or {}
    ing = (det or {}).get("ingest")
    track = getattr(ing, "track", None)
    c = (((det or {}).get("rationale") or {}).get("confidence") or {})
    accepted = str(outcome) == "ACCEPT"
    return {
        "kind": "signal_evidence_not_endpoint_accuracy",
        "signal_quality_index": (None if sqi is None else round(float(sqi), 4)),
        "signal_quality_components": {
            str(k): round(float(v), 4) for k, v in components.items()
            if v is not None and np.isfinite(float(v))
        },
        "confidence_score": (None if c.get("score") is None else
                             round(float(c["score"]), 4)),
        "confidence_stars": c.get("stars"),
        "confidence_limiting_factor": c.get("limiting_factor"),
        "fps": round(float(fps), 3),
        "n_beats_used": int(n_beats),
        "n_rois_used": int(n_rois),
        "tracking_stability": getattr(track, "stability", None),
        "cross_roi_coherence": ev.get("cross_roi_coherence"),
        "timing_precision_ms": ev.get("timing_precision_ms"),
        "duplicate_frame_fraction": ev.get("duplicate_frame_fraction"),
        "collapsed_interval_fraction": ev.get("collapsed_interval_fraction"),
        "photometric": dict(ev.get("photometric") or {}),
        "scan_outcome": str(outcome),
        "overall_rhythm_accepted": accepted,
        "interpretation": (
            "capture and signal-evidence confidence only; endpoint accuracy "
            "is not established until paired-reference calibration" if
            accepted else
            "low-confidence research measurement from surviving signal "
            "evidence; the overall rhythm scan was not accepted and this "
            "value must not be treated as a clinical result"
        ),
    }


def resting_hemodynamics(det, *, outcome, participant=None, capture=None,
                         min_conf=None) -> dict:
    """Compute each resting family from its own surviving scan evidence.

    The overall rhythm verdict is carried into confidence but is not an
    endpoint input.  This is important for research collection: morphology
    can remain measurable even when rhythm-specific interval gates abstain.
    Missing waveform/lattice evidence still fails visibly.
    """
    from features.pulse_morphology import (FEATURE_NAMES,
                                           MIN_BEATS_FOR_SESSION,
                                           MIN_SDPPG_FS_HZ, MORPH_BAND_HZ,
                                           _beat_pairs, _beat_windows,
                                           beat_morphology, ensemble_beat,
                                           session_median)
    from inference.evidence import capture_segments
    from preprocessing.roi import ROI_NAMES
    from rppg.pos import orient_rois_consistently, pos_pulse

    ing, lattice = det.get("ingest"), det.get("lattice")
    if ing is None or lattice is None:
        return {"available": False, "outcome": str(outcome),
                "reasons": ["the scan produced no waveform or beat "
                            "lattice"]}
    # Endpoint availability is independent of the rhythm *classification*,
    # but never independent of measurement quality.  Morphology from one ROI
    # or from incoherent/noisy beats is not a real prototype estimate.  Apply
    # the same non-periodicity signal-evidence floors used by the production
    # decision; an interval-specific rhythm failure can still coexist with a
    # valid contour, but an unverified pulse cannot.
    from configs import load_config
    cfg = (det or {}).get("config") or load_config()
    ec = cfg["decision"].get("evidence") or {}
    ev = (det or {}).get("evidence") or {}
    sqi_obj = (det or {}).get("sqi")
    sqi_value = getattr(sqi_obj, "sqi", None)
    tracking = getattr(getattr(ing, "track", None), "stability", None)
    coh = ev.get("cross_roi_coherence")
    tp = ev.get("timing_precision_ms")
    tm = ev.get("timing_matched_fraction")

    def finite(v):
        try:
            return bool(np.isfinite(float(v)))
        except (TypeError, ValueError):
            return False

    coherence_ok = finite(coh) and float(coh) >= float(
        ec.get("coherence_floor", 0.20))
    two_region_ok = (finite(tp) and
                     float(tp) <= float(ec.get(
                         "afib_max_timing_precision_ms", 30.0)) and
                     finite(tm) and
                     float(tm) >= float(ec.get(
                         "afib_min_timing_matched", 0.75)))
    # Rhythm classification deliberately needs strong whole-scan agreement.
    # Resting contour/amplitude endpoints can also use a LIMITED evidence
    # path: some independently matched cross-region beats plus the downstream
    # requirement for >=2 morphology-readable ROIs and >=8 beats/ROI.  This
    # preserves a low-confidence research measurement from a usable scan
    # without turning a weak or single-region oscillation into a value.
    endpoint_coh_floor = float(ec.get("endpoint_min_coherence", 0.10))
    endpoint_tp_ceiling = float(ec.get(
        "endpoint_max_timing_precision_ms", 40.0))
    endpoint_tm_floor = float(ec.get("endpoint_min_timing_matched", 0.35))
    limited_region_ok = (
        finite(coh) and float(coh) >= endpoint_coh_floor and
        finite(tp) and float(tp) <= endpoint_tp_ceiling and
        finite(tm) and float(tm) >= endpoint_tm_floor
    )
    endpoint_evidence_ok = coherence_ok or two_region_ok or limited_region_ok
    evidence_mode = ("rhythm_grade" if (coherence_ok or two_region_ok)
                     else "limited_endpoint_morphology" if limited_region_ok
                     else "unverified")
    quality_reasons = []
    if not finite(sqi_value) or float(sqi_value) < float(
            cfg["decision"].get("sqi_floor", 0.30)):
        quality_reasons.append("signal quality is below the biomarker floor")
    if not finite(tracking) or float(tracking) < float(
            (cfg["decision"].get("readiness") or {}).get(
                "min_tracking_stability", 0.60)):
        quality_reasons.append("facial ROI tracking is unstable")
    if not endpoint_evidence_ok:
        quality_reasons.append(
            "pulse beats are not independently verified across facial regions")
    if quality_reasons:
        return {"available": False, "outcome": str(outcome),
                "reasons": quality_reasons,
                "quality_gate": {
                    "signal_quality_index": sqi_value,
                    "tracking_stability": tracking,
                    "cross_roi_coherence": coh,
                    "timing_precision_ms": tp,
                    "timing_matched_fraction": tm,
                    "evidence_mode": evidence_mode,
                    "thresholds": {
                        "minimum_cross_roi_coherence": endpoint_coh_floor,
                        "maximum_timing_precision_ms": endpoint_tp_ceiling,
                        "minimum_timing_matched_fraction": endpoint_tm_floor,
                    },
                    "pass": False,
                }}
    from inference.pipeline import CALIBRATED_MIN_CONF
    mc = float(min_conf if min_conf is not None
               else det.get("min_conf", CALIBRATED_MIN_CONF))
    ts = np.asarray(ing.timestamps_s, float)
    fps = float(ing.meta.measured_fps_mean)
    hi = min(MORPH_BAND_HZ[1], 0.45 * fps)
    pairs = _beat_pairs(np.asarray(lattice.beat_t_s, float),
                        np.asarray(lattice.beat_confidence, float), mc)
    roi_segments = {r: [] for r in ROI_NAMES}
    per_beat: list = []
    seg_amps: dict = {r: [] for r in ROI_NAMES}     # insertion order = ROI_NAMES
    for a, b in capture_segments(ts, fps):
        seg_ts = ts[a:b]
        waves = orient_rois_consistently(
            {r: pos_pulse(ing.traces[r][a:b], fps,
                          band=(MORPH_BAND_HZ[0], hi)) for r in ROI_NAMES})
        in_seg = [(t0, t1) for t0, t1 in pairs
                  if t0 >= seg_ts[0] and t1 <= seg_ts[-1]]
        windows = {r: _beat_windows(waves[r], seg_ts, in_seg)
                   for r in ROI_NAMES}
        for roi in ROI_NAMES:
            for f1, f2 in windows[roi]:
                roi_segments[roi].append(waves[roi][f1:f2])
                per_beat.append(beat_morphology(waves[roi][f1:f2], fps,
                                                native_fs=fps))
        for roi, amps in segment_amplitudes(waves, seg_ts, windows).items():
            seg_amps[roi] += amps
    # Pool ONCE over the whole scan; the 12-beat floor applies to the total
    # (see segment_amplitudes for why not per segment).
    amp_series = pool_amplitudes({r: v for r, v in seg_amps.items() if v})
    roi_feats = []
    for roi in ROI_NAMES:
        ens, dur = ensemble_beat(roi_segments[roi], fps)
        if ens is None:
            continue
        f = beat_morphology(ens, ens.size / dur, native_fs=fps)
        if f is not None:
            roi_feats.append(f)
    beats_per_roi = (int(np.median([len(v) for v in roi_segments.values()]))
                     if roi_segments else 0)
    if len(roi_feats) < 2 or beats_per_roi < MIN_BEATS_FOR_SESSION:
        return {"available": False, "outcome": str(outcome),
                "reasons": [f"only {beats_per_roi} morphology-usable "
                            f"beats across {len(roi_feats)} readable ROIs "
                            f"(need at least {MIN_BEATS_FOR_SESSION} "
                            "beats and 2 ROIs)"]}
    contour = {}
    for name in FEATURE_NAMES:
        vals = [r[name] for r in roi_feats if r.get(name) is not None]
        contour[name] = (float(np.median(vals)) if vals else None)
    contour["notch_present"] = (session_median(per_beat)["features"]
                                .get("notch_present"))
    reg = det.get("regularity")
    med_ibi = ((getattr(reg, "values", None) or {}).get("median_ibi")
               if reg is not None else None)
    rate_method = "clean_interval_median"
    if med_ibi:
        hr = 60000.0 / float(med_ibi)
        rate_intervals = int(getattr(reg, "n_intervals", 0) or 0)
    else:
        # Rhythm cleaning may reject every interval because it cannot make a
        # safe rhythm statement, while the calibrated lattice still contains
        # enough beats for a coarse resting-rate feature.  Use only adjacent,
        # confidence-qualified, physiologically plausible measured intervals.
        # No interval is repaired and RMSSD is never computed on this fallback.
        ibi = np.asarray([(b - a) * 1000.0 for a, b in pairs], float)
        ibi = ibi[(ibi >= 250.0) & (ibi <= 2200.0)]
        if ibi.size >= 5:
            med_ibi = float(np.median(ibi))
            hr = 60000.0 / med_ibi
            rate_intervals = int(ibi.size)
            rate_method = "calibrated_fused_beat_median"
        else:
            hr = None
            rate_intervals = int(ibi.size)
            rate_method = None
    cap = capture or {}
    locked = bool(cap.get("exposure_locked")) and bool(cap.get("awb_locked"))
    amp_series.sort(key=lambda x: x[0])
    quality = _evidence_quality(det, fps=fps, n_beats=beats_per_roi,
                                n_rois=len(roi_feats), outcome=str(outcome))
    quality["endpoint_evidence"] = {
        "pass": True,
        "mode": evidence_mode,
        "cross_roi_coherence": coh,
        "timing_precision_ms": tp,
        "timing_matched_fraction": tm,
        "thresholds": {
            "minimum_cross_roi_coherence": endpoint_coh_floor,
            "maximum_timing_precision_ms": endpoint_tp_ceiling,
            "minimum_timing_matched_fraction": endpoint_tm_floor,
        },
        "meaning": (
            "meets the rhythm-grade cross-region evidence floor" if
            evidence_mode == "rhythm_grade" else
            "limited cross-region evidence; value is retained only because "
            "multi-ROI beat morphology also survived"
        ),
    }
    stiffness = stiffness_contour(contour)
    stiffness["confidence"] = dict(
        quality, sdppg_supported=bool(fps >= MIN_SDPPG_FS_HZ),
        marker_completeness=stiffness.get("marker_completeness"))
    tone = vasomotor_indices(amp_series, locked)
    tone["confidence"] = dict(
        quality, optics_locked=bool(locked),
        amplitude_beats=int(tone.get("n_beats") or 0))
    fitness = cardiorespiratory_indices(reg, hr, participant)
    fitness["resting_rate_method"] = rate_method
    fitness["resting_rate_intervals"] = rate_intervals
    fitness["confidence"] = dict(
        quality,
        required_inputs_present=bool(
            fitness.get("resting_hr_bpm") is not None and
            fitness.get("rmssd_ms") is not None),
        proxy_basis=fitness.get("fitness_proxy_basis"),
        resting_rate_method=rate_method,
        resting_rate_intervals=rate_intervals)
    return {
        "available": True,
        "outcome": str(outcome),
        "fps": fps,
        "sdppg_derivable": fps >= MIN_SDPPG_FS_HZ,
        "n_beats_used": beats_per_roi,
        "n_rois_used": len(roi_feats),
        "quality": quality,
        "contour": contour,
        "arterial_stiffness": stiffness,
        "vascular_tone": tone,
        "cardiorespiratory_fitness": fitness,
    }
