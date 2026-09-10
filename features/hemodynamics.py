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
# The fitness card's resting rate must clear the same floor the pipeline uses
# to decide whether a rate or rhythm statement may be published at all
# (heads/head_rate_flags.MIN_INTERVALS, decision.evidence.min_intervals_any).
# Until 2026-09-09 the card had no floor of its own: one production scan
# published a score from FOUR clean intervals while the same response refused
# to publish a pulse from those same four.
MIN_RATE_INTERVALS = 15
MIN_ENVELOPE_SPAN_S = 30.0        # under ~2 cycles of the slowest band edge

# PLANNING reference points for the autonomic index. They are population
# anchors, not a calibration: an adult resting pulse near 60 with RMSSD
# near 40 ms sits mid-scale. They live here, named and visible, so the
# index is reproducible and so replacing them with fitted values is a
# one-line change once a reference cohort exists.
HR_REF_BPM = 72.0        # adult resting-rate centre; 60 put an athlete's rate at 50/100
HR_SPREAD_BPM = 14.0     # about one SD of adult resting rate
RMSSD_REF_MS = 40.0
RESEARCH_ESTIMATE_LABEL = "Research Estimate / Prototype"
# Iteration 12 (owner-approved 2026-09-09): the beat-count pulse is checked
# against the waveform's dominant rhythm (inference/evidence.py::
# spectral_pulse). The verdict travels with every scan (evidence, cards'
# confidence, tracking sheet). Mode "gate" additionally abstains the cards
# on a RESOLVED disagreement — an additional fail-closed check, never a
# loosening. It is BINDING as of 2026-09-09 (owner decision, on production
# evidence): a phone scan reported a beat count of 84 bpm against a spectral
# rhythm of 60 bpm, with 2 of 4 regions agreeing on the spectrum, while the
# reference device measured 64 bpm. The spectrum was right to within 4 bpm,
# the beat count was 20 bpm wrong, and all three cards computed on the wrong
# rate. Only a RESOLVED disagreement abstains: an unread or unresolved
# spectrum leaves the cards exactly as they were. The corpus cannot score
# this - an abstention rule can only lower the gate's `cards` metric - so it
# ships on the owner's decision, with every verdict written to the tracking
# sheet for audit.
PULSE_AGREEMENT_TOL = 0.15        # |count - spectral| / spectral
PULSE_MIN_ROI_AGREE = 2           # the spectral rhythm is resolved when >= 2 of 4 regions agree with it
PULSE_CHECK_MODE = "gate"         # "report" | "gate"

# ---- Display tiers (owner decision 2026-09-09, evening) ---------------------
# "A result on every scan, in a normal-looking range." Every card carries a
# 0-100 SCORE derived from the marker it measures, the typical range of that
# score, and a TIER:
#   measured    - every floor and gate held; the value is a measurement to the
#                 extent this prototype can make one.
#   provisional - a beat lattice existed but a floor or gate did not hold (too
#                 few beats, no dicrotic notch, an unverified pulse): the score
#                 is computed from the beats the scan DID yield and is labelled
#                 as thin. It will move between scans; that is the thinness of
#                 the evidence showing, not hidden.
# A card is unavailable only when there is nothing to compute from: no
# waveform, no beats. Nothing is imputed, defaulted or invented. The score
# maps are monotone in the measured marker, uncalibrated, and documented
# beside each card; the raw marker travels with every value.
MIN_PROVISIONAL_BEATS = 4             # morphology beats (median per ROI) for a provisional contour
MIN_PROVISIONAL_AMPLITUDE_BEATS = 5   # amplitude beats for a provisional tone score
MIN_PROVISIONAL_RATE_INTERVALS = 5    # clean intervals for a provisional clean-interval rate
# Stiffness keeps its own natural units (owner, 2026-09-09): the reflection
# index as a ratio, the way it read before the 0-100 rescale. Tone and fitness
# stay on 0-100, where a percentage and a bounded proxy already belong.
RI_TYPICAL = (0.4, 0.8)               # reflection index, healthy adults
AGING_TYPICAL = (-2.0, 1.0)           # SDPPG aging index (Takazawa), young -> old
STIFFNESS_TYPICAL = (40.0, 80.0)      # the shared 0-100 scale's typical band
# Pulse crest time, the no-notch fallback marker. Direction (checked against
# the PPG literature 2026-09-09, and OPPOSITE to what this file assumed at
# first): the pulse wave reaches the periphery sooner through stiffer
# arteries, so a SHORTER crest time reads as higher stiffness. These bounds
# are population rules of thumb, not calibrated thresholds, and raw crest time
# also shortens with a faster pulse - both reasons this marker is only ever
# reported as a band, and only as provisional.
RISE_TYPICAL_MS = (120.0, 320.0)


# One 0-100 stiffness index (owner, 2026-09-10), so every scan shows a number
# in the same unit. Each marker reaches it by its OWN monotone map, and every
# map puts that marker's typical range on 40-80, so the scales line up:
#
#   reflection index   x100, so 0.35 reads 35 and 0.55 reads 55 (typical 40-80)
#   pulse crest time   320 ms -> 40, 120 ms -> 80; SHORTER is stiffer, because
#                      a stiffer artery carries the wave faster
#   SDPPG aging index  its -2..+1 range across 10..90
#
# This is a rescale, NOT a conversion between markers: a crest-time score is
# not a reflection index and never claims to be. Which marker produced the
# number is in raw_name/raw_value, shown under the card and kept on the sheet,
# and a crest-time score is always tier "provisional". Uncalibrated throughout.
RISE_MS_TO_SCORE = ((320.0, 40.0), (120.0, 80.0))
AGI_TO_SCORE = ((-2.0, 10.0), (1.0, 90.0))


def _lerp_score(x, pair):
    (x0, y0), (x1, y1) = pair
    return None if x is None else round(_clip100(y0 + (float(x) - x0) * (y1 - y0) / (x1 - x0)), 1)


def stiffness_score_from_reflection_index(ri):
    """Reflection index x100: a stiffer artery returns a larger reflected wave."""
    return None if ri is None else round(_clip100(100.0 * float(ri)), 1)


def stiffness_score_from_rise_time(rise_s):
    """Crest time on the same 0-100 scale. Shorter crest time, higher score."""
    return None if rise_s is None else _lerp_score(float(rise_s) * 1000.0, RISE_MS_TO_SCORE)


def stiffness_band_from_reflection_index(ri):
    """"High" / "Typical" / "Low" from the reflection index, or None.

    A stiffer artery returns a larger reflected wave, so a HIGHER index reads
    as higher stiffness - the opposite direction to the crest time below."""
    if ri is None:
        return None
    lo, hi = RI_TYPICAL
    v = float(ri)
    if v > hi:
        return "High"
    return "Low" if v < lo else "Typical"


def stiffness_band_from_rise_time(rise_s):
    """"High" / "Typical" / "Low" from the pulse crest time, or None.

    A word rather than a number, because the crest time is a different
    quantity from the reflection index this card normally reports and the two
    must never share a number slot (owner, 2026-09-09)."""
    if rise_s is None:
        return None
    ms = float(rise_s) * 1000.0
    lo, hi = RISE_TYPICAL_MS
    if ms < lo:
        return "High"
    return "Low" if ms > hi else "Typical"
TONE_TYPICAL = (20.0, 60.0)           # amplitude CV %: the range seen on this prototype's resting scans
FITNESS_TYPICAL = (30.0, 70.0)        # by construction of the resting-rate logistic


def _clip100(x):
    return None if x is None else float(min(100.0, max(0.0, float(x))))


def tone_score_from_cv(cv):
    """Amplitude coefficient of variation in percent, clipped to 0-100."""
    return None if cv is None else round(_clip100(100.0 * float(cv)), 1)


def pulse_check(ev: dict) -> dict:
    """Verdict of the beat count against the waveform's dominant rhythm.

    verdict: "not_evaluated" (evidence without the fields: fixtures, legacy
    callers), "unresolved" (no estimate, or the regions disagree on the
    rhythm), "agree", "disagree". Pure function of the evidence dict."""
    pl, ps = ev.get("pulse_lattice_bpm"), ev.get("pulse_spectral_bpm")
    pa, ra = ev.get("pulse_agreement"), ev.get("pulse_spectral_roi_agree")
    out = {"pulse_lattice_bpm": pl, "pulse_spectral_bpm": ps,
           "pulse_agreement": pa, "pulse_spectral_roi_agree": ra,
           "tolerance": PULSE_AGREEMENT_TOL, "mode": PULSE_CHECK_MODE,
           "verdict": "not_evaluated", "reason": None}
    if "pulse_agreement" not in ev:
        return out

    def finite(v):
        try:
            return bool(np.isfinite(float(v)))
        except (TypeError, ValueError):
            return False

    if not (finite(pl) and finite(ps) and finite(pa)):
        out["verdict"] = "unresolved"
        out["reason"] = ("the beat count could not be checked: "
                         + ("no clean beat intervals" if not finite(pl)
                            else "no dominant rhythm could be read from the waveform"))
    elif ra is None:
        out["verdict"] = "unresolved"
        out["reason"] = "the regions' agreement on the waveform's rhythm was not measured"
    elif int(ra) < PULSE_MIN_ROI_AGREE:
        out["verdict"] = "unresolved"
        out["reason"] = (f"the facial regions do not agree on the waveform's rhythm "
                         f"({int(ra)} of 4 within 10 % of {float(ps):.0f} bpm)")
    elif float(pa) <= PULSE_AGREEMENT_TOL:
        out["verdict"] = "agree"
        out["reason"] = (f"beat count {float(pl):.0f} bpm agrees with the waveform's "
                         f"dominant rhythm {float(ps):.0f} bpm")
    else:
        out["verdict"] = "disagree"
        out["reason"] = (f"the beat count ({float(pl):.0f} bpm) disagrees with the "
                         f"waveform's dominant rhythm ({float(ps):.0f} bpm)")
    return out


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
    # Iteration 13 (owner-approved 2026-09-09): ONE metric per capture class.
    # At research frame rates (>= MIN_SDPPG_FS_HZ) the card is the SDPPG
    # aging index; at consumer rates it is the reflection index — or nothing.
    # Falling through to the rise time (ms) when no notch was found put two
    # different quantities on the same card within one device (phone scans
    # 2026-09-09: 0.83 ratio, then 210 ms, then 0.57). The rise time stays in
    # the details as a morphology marker; it is never the stiffness value.
    # ONE FORMAT ON EVERY SCAN (owner, 2026-09-10): a band - High, Typical or
    # Low - and never a number. This card's markers live in different units (a
    # dimensionless ratio, a dimensionless index, a time in milliseconds) and
    # no validated conversion exists between them, so any shared number slot
    # would either change units between scans or present one marker as
    # another. A band is the one form all three can honestly take, and it is
    # also the honest resolution of the marker's instability: this person's
    # reflection index moved 0.28 -> 0.97 in four minutes at 30 fps. The
    # measured marker travels beside the band in raw_value and reaches the
    # tracking sheet unchanged, so nothing is lost for analysis.
    tier_reasons = []
    band = None
    if agi is not None:
        # ONE metric per capture class (iteration 13): at a research frame rate
        # the second-derivative aging index is the better marker and is the
        # card's value; a device always sits in one class, so its card never
        # changes quantity between scans.
        score, unit, typical = _lerp_score(agi, AGI_TO_SCORE), "/100", STIFFNESS_TYPICAL
        band = ("High" if float(agi) > AGING_TYPICAL[1]
                else "Low" if float(agi) < AGING_TYPICAL[0] else "Typical")
        tier = "measured"
        name = "second_derivative_aging_index"
        method = ("second-derivative aging index (b - c - d - e)/a on the "
                  "ensemble pulse (Takazawa 1998); dimensionless, rises with "
                  "arterial stiffness. Needs a research frame rate")
        out["raw_value"], out["raw_unit"], out["raw_name"] = (
            round(float(agi), 5), "index", "second_derivative_aging_index")
    elif ri is not None:
        score, unit, typical = stiffness_score_from_reflection_index(ri), "/100", STIFFNESS_TYPICAL
        band = stiffness_band_from_reflection_index(ri)
        out["raw_value"], out["raw_unit"], out["raw_name"] = (
            round(float(ri), 5), "ratio", "reflection_index")
        tier = "measured"
        name = "reflection_index"
        method = ("reflection index: reflected-wave height / systolic-wave "
                  "height on the ensemble pulse; dimensionless and "
                  "uncalibrated, rises with arterial stiffness")
    elif rise is not None:
        # The weakest of the three markers, and in different units again, so it
        # is provisional - but it takes the same band form as the others.
        score, unit, typical = stiffness_score_from_rise_time(rise), "/100", STIFFNESS_TYPICAL
        band = stiffness_band_from_rise_time(rise)
        tier = "provisional"
        tier_reasons.append("no dicrotic notch was found, so this is a band "
                            "from the pulse crest time, a different and weaker "
                            "marker than the reflection index")
        name = "pulse_crest_time_band"
        method = ("band from the pulse crest time (foot to systolic peak) "
                  "against a 120-320 ms rule-of-thumb range: a shorter crest "
                  "time reads as stiffer. Uncalibrated, confounded by pulse "
                  "rate, and reported only when no dicrotic notch was found")
        out["raw_value"], out["raw_unit"], out["raw_name"] = (
            round(float(rise) * 1000.0, 1), "ms", "pulse_rise_time")
    else:
        score, unit, typical, tier, name, method = None, None, None, None, None, None
    out["estimate"] = (None if (score is None and band is None) else {
        "label": RESEARCH_ESTIMATE_LABEL,
        "name": name, "value": score, "unit": unit, "band": band,
        "method": method,
    })
    out["available"] = score is not None or band is not None
    out["tier"] = tier
    out["tier_reasons"] = tier_reasons
    out["score"] = score
    out["band"] = band
    out["score_typical_range"] = None if typical is None else list(typical)
    out["marker_completeness"] = sum(
        out.get(k) is not None for k in
        ("aging_index", "reflection_index", "rise_time_s",
         "norm_upstroke_slope", "pulse_width50_s"))
    if score is None:
        out["reason"] = ("neither a dicrotic notch nor a usable pulse upstroke "
                         "was found on the ensemble pulse, so no stiffness "
                         "marker could be computed")
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
    """Pool the (already normalised) per-ROI series beat by beat. A ROI needs
    MIN_PROVISIONAL_AMPLITUDE_BEATS over the whole scan. Region order is the
    insertion order of `per_roi` (ROI_NAMES) — it decides the reference
    series on ties, so it must not change.

    The floor here is the PROVISIONAL one, not the measured one (fixed
    2026-09-09): pooling at 12 meant a scan whose beats were spread across
    capture segments had every region dropped and reached the card with ZERO
    amplitude beats, so the card could never produce the provisional score the
    lower floor was meant to allow. A real scan showed exactly that — 18 usable
    beats over 4 segments, and "0 usable beats" on the card. Deciding measured
    from provisional is the CARD's job, on the pooled count."""
    pooled: list = []
    for roi, amps in per_roi.items():
        if len(amps) < MIN_PROVISIONAL_AMPLITUDE_BEATS:
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
    """One-segment convenience: normalise within the segment, then pool at the
    provisional floor (see pool_amplitudes)."""
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
    n_amp = len(series or [])
    if n_amp < MIN_PROVISIONAL_AMPLITUDE_BEATS:
        out["reason"] = (f"{n_amp} usable beats (need at least "
                         f"{MIN_PROVISIONAL_AMPLITUDE_BEATS} for a provisional "
                         f"score, {MIN_AMPLITUDE_BEATS} for a measured one)")
        return out
    # Display tiers (owner decision 2026-09-09): below the measured floor the
    # CV is still computed and shown, labelled provisional.
    out["tier"] = "measured" if n_amp >= MIN_AMPLITUDE_BEATS else "provisional"
    out["tier_reasons"] = ([] if out["tier"] == "measured" else
                           [f"only {n_amp} amplitude beats (a measured value "
                            f"needs {MIN_AMPLITUDE_BEATS})"])
    t = np.asarray([x for x, _ in series], float)
    a = np.asarray([y for _, y in series], float)
    span = float(t[-1] - t[0])
    mean = float(np.mean(a))
    if mean <= 1e-12:
        out["reason"] = "degenerate amplitude series"
        return out
    out["available"] = True
    out["amplitude_cv"] = round(float(np.std(a) / mean), 5)
    out["score"] = tone_score_from_cv(out["amplitude_cv"])
    out["score_typical_range"] = list(TONE_TYPICAL)
    out["raw_value"] = round(100.0 * out["amplitude_cv"], 2)
    out["raw_unit"] = "% CV"
    out["raw_name"] = "normalized_pulse_amplitude_variability"
    out["estimate"] = {
        "label": RESEARCH_ESTIMATE_LABEL,
        "name": "vascular_tone_index",
        "value": out["score"],
        "unit": "/100",
        "method": ("coefficient of variation of per-beat facial pulse "
                   "amplitude after within-ROI median normalization, in "
                   "percent (0-100); uncalibrated"),
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


def cardiorespiratory_indices(regularity, hr_bpm, participant=None, *,
                              rate_method=None, rate_intervals=None,
                              spectral_hr_bpm=None, pulse_verdict=None) -> dict:
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
    # ONE basis, always (owner decision 2026-09-09). The card used to switch
    # between a heart-rate+RMSSD composite and a heart-rate-only transform
    # depending on whether RMSSD survived cleaning. Those are different scales,
    # so the same person 14 minutes apart read 27.1 then 50.1 out of 100 - and
    # the 50.1 needed an RMSSD of 286 ms to appear while the reference device
    # measured 36 ms, an eightfold inflation cancelling a pulse 20 bpm too
    # fast. RMSSD stays in the payload as measured evidence; it never moves
    # the score.
    autonomic = None
    # Display tiers (owner decision 2026-09-09): ONE resting rate is chosen,
    # and the tier says how well it is evidenced.
    #   measured    - clean-interval median from >= MIN_RATE_INTERVALS clean
    #                 intervals, and the beat count agrees with (or could not
    #                 be checked against) the waveform's rhythm.
    #   provisional - the waveform's dominant rhythm when the beat count
    #                 disagreed with it or was too thin; a clean-interval rate
    #                 from 5-14 intervals; or, last, the unverified fused-beat
    #                 median. Each names its reason.
    tier_reasons = []
    rate_source = rate_method
    hr = hr_bpm
    n_int = None if rate_intervals is None else int(rate_intervals)
    clean = (rate_method == "clean_interval_median" and hr is not None)
    if clean and (n_int is None or n_int >= MIN_RATE_INTERVALS) and pulse_verdict != "disagree":
        tier = "measured"
    else:
        tier = "provisional"
        if pulse_verdict == "disagree" and spectral_hr_bpm is not None:
            hr, rate_source = float(spectral_hr_bpm), "waveform_rhythm"
            tier_reasons.append("the beat count disagreed with the waveform's "
                                "dominant rhythm, so the rate is taken from the "
                                "waveform")
        elif clean and n_int is not None and n_int >= MIN_PROVISIONAL_RATE_INTERVALS:
            tier_reasons.append(f"only {n_int} clean beat intervals (a measured "
                                f"value needs {MIN_RATE_INTERVALS})")
        elif spectral_hr_bpm is not None:
            hr, rate_source = float(spectral_hr_bpm), "waveform_rhythm"
            tier_reasons.append("too few clean beat intervals, so the rate is "
                                "taken from the waveform's dominant rhythm")
        elif hr is not None and rate_method is not None and rate_method != "clean_interval_median":
            # Adjacent lattice pairs inside the 250-2200 ms window, never seen
            # by the missed/false-beat splitter: the weakest admissible rate.
            tier_reasons.append("the rate comes from unverified beat intervals")
        elif clean and n_int is not None:
            tier_reasons.append(f"only {n_int} clean beat intervals (a measured "
                                f"value needs {MIN_RATE_INTERVALS})")
    rate_only = resting_rate_index(hr)
    idx = rate_only
    reason = reason_code = None
    if idx is None:
        reason_code = "no_resting_rate"
        reason = "no resting pulse rate could be measured from this scan"
        tier = None
    demographics = {
        "age_years": pc.get("age_years") or pc.get("age"),
        "sex": pc.get("sex"),
        "height_cm": pc.get("height_cm"),
        "weight_kg": pc.get("measured_weight_kg") or pc.get("weight_kg"),
        "habitual_activity": pc.get("activity_ipaq"),
    }
    missing = [k for k, v in demographics.items() if v is None]
    proxy = None if idx is None else round(100.0 * float(idx), 1)
    estimate_name = "resting_rate_cardiorespiratory_proxy"
    estimate_method = (
        "heart-rate-only logistic research transform: (60-HR)/12, over a "
        f"resting rate taken from at least {MIN_RATE_INTERVALS} clean beat "
        "intervals. Interval variability is reported beside it and never "
        "enters the score: on camera captures it is dominated by beat-timing "
        "noise, so including it would change the scale rather than the meaning."
    )
    basis = "resting_hr_only"
    return {
        "available": idx is not None,
        "calibrated": False,
        "resting_hr_bpm": (None if hr is None else round(float(hr), 2)),
        "rmssd_ms": rmssd,
        "sdnn_ms": disp.get("sdnn_ms"),
        "autonomic_index": autonomic,
        "resting_rate_index": rate_only,
        "fitness_proxy_score": proxy,
        "fitness_proxy_basis": basis if proxy is not None else None,
        "minimum_clean_intervals": MIN_RATE_INTERVALS,
        "reason": reason,
        "reason_code": reason_code,
        "tier": tier,
        "tier_reasons": tier_reasons,
        "score": proxy,
        "score_typical_range": list(FITNESS_TYPICAL),
        "raw_value": (None if hr is None else round(float(hr), 1)),
        "raw_unit": "bpm",
        "raw_name": "resting_heart_rate",
        "resting_rate_source": rate_source,
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
    # Pulse cross-check (iteration 12): reported on every scan; abstains
    # the cards only in "gate" mode and only on a RESOLVED disagreement.
    pulse_gate = pulse_check(ev)
    if PULSE_CHECK_MODE == "gate" and pulse_gate["verdict"] == "disagree":
        quality_reasons.append(pulse_gate["reason"])
    # Display tiers (owner decision 2026-09-09): a failed scan-level gate no
    # longer blanks the cards. The scan yielded a beat lattice, so every card
    # computes from it and carries the tier "provisional" with these reasons;
    # "measured" is reserved for scans on which every gate held.
    scan_tier_reasons = list(quality_reasons)
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
    if not roi_feats or beats_per_roi < MIN_PROVISIONAL_BEATS:
        # Nothing to compute a contour from: the one case that stays blank.
        return {"available": False, "outcome": str(outcome),
                "reasons": [f"only {beats_per_roi} morphology-usable "
                            f"beats across {len(roi_feats)} readable ROIs "
                            f"(a provisional value needs at least "
                            f"{MIN_PROVISIONAL_BEATS} beats in one region)"]}
    if len(roi_feats) < 2 or beats_per_roi < MIN_BEATS_FOR_SESSION:
        scan_tier_reasons.append(
            f"only {beats_per_roi} morphology-usable beats across "
            f"{len(roi_feats)} readable ROIs (a measured value needs "
            f"{MIN_BEATS_FOR_SESSION} beats in 2 regions)")
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
        "pass": not scan_tier_reasons,
        "scan_tier_reasons": list(scan_tier_reasons),
        "mode": evidence_mode,
        "cross_roi_coherence": coh,
        "timing_precision_ms": tp,
        "timing_matched_fraction": tm,
        "pulse_check": pulse_gate,
        "thresholds": {
            "minimum_cross_roi_coherence": endpoint_coh_floor,
            "maximum_timing_precision_ms": endpoint_tp_ceiling,
            "minimum_timing_matched_fraction": endpoint_tm_floor,
            "maximum_pulse_disagreement": PULSE_AGREEMENT_TOL,
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
    fitness = cardiorespiratory_indices(
        reg, hr, participant, rate_method=rate_method,
        rate_intervals=rate_intervals,
        spectral_hr_bpm=ev.get("pulse_spectral_bpm"),
        pulse_verdict=pulse_gate.get("verdict"))
    # The scan-level tier reasons apply to every card: a card is "measured"
    # only when both the scan and its own floors held.
    for card in (stiffness, tone, fitness):
        if card.get("available"):
            own = list(card.get("tier_reasons") or [])
            card["tier_reasons"] = list(scan_tier_reasons) + own
            card["tier"] = ("provisional" if (scan_tier_reasons or own)
                            else "measured")
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
        "tier": "provisional" if scan_tier_reasons else "measured",
        "tier_reasons": list(scan_tier_reasons),
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
