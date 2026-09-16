"""Guard the resting pulse against beat-doubling / halving.

The resting-rate feature the fitness card rides on is the median of the clean
inter-beat intervals (features/hemodynamics.py). When the beat detector splits
one beat into two it produces half-length intervals; when it merges two it
produces double-length ones. Either drags the interval median off toward a
factor of 2x or 0.5x, and a resting scan reads 100+ bpm when the true pulse is
~65 (seen live: 101 vs a 65 reference, 35 % half-length intervals).

The spectral rate (inference/evidence.spectral_pulse) is computed with a
subharmonic check, so it resists that error. When the interval rate shows a
doubling signature AGAINST a trustworthy spectral rate, the spectral rate is
the safer resting rate.

This is a PURE detector. It decides whether the interval rate looks doubled
against a spectral anchor it can defend (finite, positive, and — when the SNR
is given — above a floor) and, if so, hands back the spectral rate with the
reason. It never invents a rate and never fires without that anchor, so on a
clean scan it is a no-op. It does NOT touch the clinical rhythm/flutter head
(features/flutter.py), whose harmonic handling is its own; this is the
resting-rate / fitness path only.
"""
from __future__ import annotations

import math

# The interval rate is a clean "double"/"half" when it sits within this
# fraction of 2x / 0.5x the spectral rate — a detector split/merge, not
# physiology (no resting adult sits at exactly twice a trustworthy rhythm).
HARMONIC_RATIO_TOL = 0.15
# A split can contaminate only SOME intervals, lifting the median part-way
# rather than cleanly to 2x (the 101-vs-65 case sat at ratio ~1.5). Above this
# share of half-length intervals the median is not trustworthy even then.
HARMONIC_FRACTION_HI = 0.30
# Below this ratio the interval rate is not even "inflated"; the split branch
# needs the rate above the anchor before it will act.
INFLATED_RATIO_LO = 1.0 + HARMONIC_RATIO_TOL
# The spectral anchor must clear this in-band SNR to stand in as the truth.
SPECTRAL_SNR_FLOOR = 1.5
# Facial regions that must independently back the spectral rhythm before the
# reported rate (rate head) or the fitness rate may switch to it on a
# count/spectrum disagreement. The single source of truth for both consumers
# (heads/head_rate_flags.py and features/hemodynamics.py). On the tracking
# sheet the spectral rate is within 5 bpm of the reference on ~0.80 of scans
# with >= 3 regions agreeing, and no better than a coin flip with <= 2 (which
# is how a 2-of-4 spectral of 54 bpm inflated a fitness card on 2026-09-15).
MIN_ROI_AGREE_FOR_RATE = 3
# A clean 2x/0.5x count-vs-waveform fold may replace the count without the
# region quota only when this share of intervals is half-length (audit #7).
FOLD_MIN_HARMONIC_FRACTION = 0.20


def _finite(*xs) -> bool:
    try:
        return all(math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def guard_resting_pulse(lattice_bpm, spectral_bpm, *, harmonic_fraction=0.0,
                        spectral_snr=None, ref_bpm=None) -> dict:
    """Decide whether `lattice_bpm` (interval-median resting rate) is a
    doubling / halving error against `spectral_bpm`, and if so return the
    spectral rate to use instead.

    `ref_bpm` (an external reference such as ShenAI's HR) is recorded for
    telemetry only and never required — inference stays camera-only and
    replayable. Returns a dict with `guarded`, `reported_bpm` (spectral when
    guarded, else the lattice rate), `signature` ('double' | 'half' |
    'split_inflation' | None), `ratio`, and a one-line `reason`.
    """
    out = {"guarded": False, "reported_bpm": lattice_bpm,
           "raw_lattice_bpm": lattice_bpm, "spectral_bpm": spectral_bpm,
           "ratio": None, "harmonic_fraction": float(harmonic_fraction or 0.0),
           "spectral_snr": spectral_snr, "ref_bpm": ref_bpm,
           "signature": None, "reason": None}
    # A trustworthy spectral anchor and a finite interval rate are required.
    if not (_finite(lattice_bpm, spectral_bpm) and float(spectral_bpm) > 0
            and float(lattice_bpm) > 0):
        return out
    if spectral_snr is not None and _finite(spectral_snr) \
            and float(spectral_snr) < SPECTRAL_SNR_FLOOR:
        return out

    lat, spec = float(lattice_bpm), float(spectral_bpm)
    ratio = lat / spec
    out["ratio"] = round(ratio, 4)
    hf = float(harmonic_fraction or 0.0)

    def near(x, target):
        return abs(x - target) <= HARMONIC_RATIO_TOL * target

    if near(ratio, 2.0):
        sig = "double"
    elif near(ratio, 0.5):
        sig = "half"
    elif hf >= HARMONIC_FRACTION_HI and ratio >= INFLATED_RATIO_LO:
        sig = "split_inflation"
    else:
        return out

    tail = (f", {hf * 100:.0f}% half-length intervals"
            if sig == "split_inflation" else "")
    out.update(
        guarded=True, reported_bpm=round(spec, 2), signature=sig,
        reason=(f"resting rate {lat:.0f} bpm looks {sig.replace('_', ' ')} "
                f"against the waveform's dominant rhythm {spec:.0f} bpm "
                f"(ratio {ratio:.2f}{tail}); using the rhythm rate"))
    return out


# Audit 2026-09-17 #7 (CALC-3/CALC-4/CALC-2). Three resolvers produced three
# resting pulses from one scan: the decision's raw interval median (shown on
# the rhythm card, unguarded), the rate head (count/spectral reconciliation,
# "verified" only with >= 3 regions) and the fitness rate (guard-first, which
# let a spectral peak backed by 0-2 regions replace the count). This is the
# single resolver all three consume, so the card, the sheet's "Pulse bpm" and
# the fitness card are one number, and +/-8 bpm across scans is testable on
# one column. Rules are the rate head's (heads/head_rate_flags.py, iteration
# 20/21) with the doubling guard folded in behind the same region-backing bar.
def resolve_resting_rate(count_bpm, n_int, ev: dict, *,
                         min_intervals: int) -> dict:
    """One resting pulse for the scan, or None with the reason.

    count_bpm: median of 60000/ibi over the clean intervals (None when thin).
    ev: the pipeline evidence dict (pulse_spectral_bpm, pulse_spectral_roi_agree,
        pulse_spectral_snr, harmonic_fraction, pulse_agreement).
    Returns {bpm, source, confidence, reasons, verdict, spectral_backed, guard}.
      confidence: 'verified'   count and waveform agree (mean of the two)
                  'provisional' waveform rate replaced a disagreeing or doubled
                                count, backed by >= MIN_ROI_AGREE_FOR_RATE regions
                  'unverified' count alone, no cross-check available
                  'uncertain'  nothing trustworthy -> bpm None (fail closed)
    """
    from features.hemodynamics import pulse_check          # local: no import cycle
    ev = ev or {}
    verdict = pulse_check(ev).get("verdict")
    spec = ev.get("pulse_spectral_bpm")
    try:
        spec = float(spec) if spec is not None and math.isfinite(float(spec)) and float(spec) > 0 else None
    except (TypeError, ValueError):
        spec = None
    try:
        roia = int(ev.get("pulse_spectral_roi_agree"))
    except (TypeError, ValueError):
        roia = None
    backed = spec is not None and roia is not None and roia >= MIN_ROI_AGREE_FOR_RATE
    count = float(count_bpm) if count_bpm is not None and _finite(count_bpm) else None
    reasons: list = []
    guard = guard_resting_pulse(count, spec, harmonic_fraction=ev.get("harmonic_fraction", 0.0),
                                spectral_snr=ev.get("pulse_spectral_snr"))
    out = {"verdict": verdict, "spectral_backed": backed, "guard": guard,
           "n_intervals": n_int, "count_bpm": count, "spectral_bpm": spec}
    if guard.get("guarded"):
        # A clean 2x/0.5x fold may override WITHOUT the 3-region quota only
        # when the intervals themselves corroborate split beats (harmonic
        # fraction >= the AF gate's own 0.20): a doubled count carries many
        # half-length intervals; a genuinely fast rate paired with a 1-region
        # subharmonic spectral peak does not (heads tests: 100 vs 55, 1-2
        # regions, hf 0 -> uncertain, as iterations 20/21 decided).
        try:
            hf = float(ev.get("harmonic_fraction") or 0.0)
        except (TypeError, ValueError):
            hf = 0.0
        clean_fold = guard.get("signature") in ("double", "half") and hf >= FOLD_MIN_HARMONIC_FRACTION
        if backed or clean_fold:
            reasons.append(guard["reason"] + (f"; {roia} of 4 facial regions back the waveform rhythm"
                                              if backed else "; the count is a clean multiple of the "
                                              "waveform rate, which is evidence in itself"))
            return {**out, "bpm": round(spec, 1), "source": "waveform_rhythm_doubling_guard",
                    "confidence": "provisional", "reasons": reasons}
        reasons.append("the beat count shows a doubling/halving signature and too few facial "
                       "regions back the waveform rhythm to replace it, so no rate is reported")
        return {**out, "bpm": None, "source": None, "confidence": "uncertain", "reasons": reasons}
    if verdict in (None, "not_evaluated"):
        return {**out, "bpm": (round(count, 1) if count is not None else None),
                "source": "clean_interval_median" if count is not None else None,
                "confidence": "unverified" if count is not None else "uncertain", "reasons": reasons}
    if verdict == "agree":
        pair = [v for v in (count, spec) if v is not None]
        if not pair:
            return {**out, "bpm": None, "source": None, "confidence": "uncertain", "reasons": reasons}
        src = ("count_spectral_mean" if len(pair) == 2 else
               "clean_interval_median" if count is not None else "waveform_rhythm")
        return {**out, "bpm": round(float(sum(pair) / len(pair)), 1), "source": src,
                "confidence": "verified", "reasons": reasons}
    if verdict == "disagree" and backed:
        reasons.append("the beat count disagreed with the waveform's dominant rhythm; "
                       f"reporting the waveform rate ({spec:.0f} bpm), which {roia} of 4 "
                       "facial regions agree on")
        return {**out, "bpm": round(spec, 1), "source": "waveform_rhythm",
                "confidence": "provisional", "reasons": reasons}
    reasons.append("the resting rate could not be verified across facial regions, "
                   "so no rate is reported")
    return {**out, "bpm": None, "source": None, "confidence": "uncertain", "reasons": reasons}
