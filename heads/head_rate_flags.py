"""
head_rate_flags (M1.3) — the reported resting rate and sustained
bradycardia (< 50 bpm) / tachycardia (> 100 bpm) flags, with the same
abstention discipline as everything else: too few clean intervals, or a
rate the cross-check cannot verify -> no number and no flag, with the
reason named. Sanctioned sentences live in
datasets/schema.RATE_FLAG_SENTENCES (spec changelog B.15) — this head
never composes text.

Reliability-first rate selection (2026-09-14, owner-directed). The clean-
interval median (a COUNT of detected beats) doubles when the detector
splits beats and halves when it drops them; at ~0 dB per-ROI SNR this made
the reported pulse wrong on ~half of real phone scans, and a doubled count
could raise a FALSE tachycardia flag. The pipeline already computes an
independent estimate — the waveform's dominant rhythm (spectral) and its
agreement with the count (inference/evidence.py::spectral_pulse, carried in
context["evidence"]) — but this head ignored it. It no longer does:

  verdict "agree"     -> the MEAN of the count and the spectral rate (two
                         concordant estimates average to less error than
                         either alone), flags allowed from the interval
                         distribution.
  verdict "disagree"  -> if >= MIN_ROI_AGREE_FOR_RATE facial regions back
     & regions agree     the spectral rhythm, report the spectral rate as
                         PROVISIONAL; no flag (a single-number rhythm
                         cannot support a sustained-fraction test).
  otherwise            -> the rate could not be verified across regions:
     (unresolved,         report NOTHING (uncertain) and raise no flag.
      thin agreement)
  no cross-check       -> legacy/fixture callers with no evidence keep the
     (not_evaluated)      pre-2026-09-14 behaviour (clean-interval median,
                          flags allowed) so unit fixtures and the CLI are
                          unaffected.

Measured retrospectively on 82 reference-carrying tracking-sheet scans:
of the rates shown, within 5 bpm of the reference 0.53 -> 0.76; it stops
showing a rate on the scans where both estimators disagree, rather than
showing a wrong one. It does NOT by itself reach the 80 % pulse-agreement
target — the scans it blanks are signal-starved (low cross-ROI agreement),
which is a capture-quality problem, not a rate-selection one.
"""
from __future__ import annotations

import numpy as np

from datasets.schema import MeasurementClass, RATE_FLAG_SENTENCES
from features.regularity import regularity_from_runs
from heads.base import EndpointHead, HeadResult, register_head

BRADY_BPM = 50.0
TACHY_BPM = 100.0
MIN_INTERVALS = 15            # same floor as the any-class decision gate
MIN_SUSTAINED_FRACTION = 0.75  # flag only if >= 75% of clean intervals agree
# Regions that must back the spectral rhythm before it may REPLACE the beat
# count on a disagreement. Stricter than the verdict's own 2-of-4 resolution
# floor (features.hemodynamics.PULSE_MIN_ROI_AGREE): on the tracking sheet the
# spectral rate is within 5 bpm of the reference on ~0.80 of scans with >= 3
# regions agreeing, and no better than a coin-flip with <= 2.
MIN_ROI_AGREE_FOR_RATE = 3


def _finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


class RateFlagsHead(EndpointHead):
    name = "rate_flags"
    version = "1.1.0"
    required_inputs = ("regularity",)

    def run(self, lattice, context: dict) -> HeadResult:
        # v0.7 (G-a): the intervals come from the ONE representation (a
        # direct caller without one gets it built by the same module);
        # the rate itself is a location statistic on them, which is the
        # one thing a head may still compute.
        ctx = context or {}
        reg = ctx.get("regularity")
        if reg is None:
            reg = regularity_from_runs(
                list(getattr(lattice, "runs", []) or []),
                run_times=list(getattr(lattice, "run_times", []) or []),
                fps=getattr(lattice, "fps", None))
        ibi = np.concatenate(reg.runs) if reg.runs else np.array([])
        ibi = ibi[np.isfinite(ibi) & (ibi > 0)]
        n_int = int(ibi.size)
        lat_med = float(np.median(60000.0 / ibi)) if n_int >= MIN_INTERVALS \
            else None

        # The independent cross-check the pipeline already computed. Absent for
        # fixture/CLI callers (no evidence) -> verdict "not_evaluated".
        ev = ctx.get("evidence") or {}
        from features.hemodynamics import pulse_check
        verdict = pulse_check(ev).get("verdict")
        spec = ev.get("pulse_spectral_bpm")
        spec = float(spec) if _finite(spec) and float(spec) > 0 else None
        roia = ev.get("pulse_spectral_roi_agree")
        try:
            roia = int(roia)
        except (TypeError, ValueError):
            roia = None
        spec_backed = spec is not None and roia is not None \
            and roia >= MIN_ROI_AGREE_FOR_RATE

        reasons: list[str] = []
        # ---- reliability-first rate selection --------------------------------
        if verdict in (None, "not_evaluated"):
            reported, source, confidence, flags_ok = (
                lat_med, "clean_interval_median", "unverified", True)
        elif verdict == "agree":
            pair = [v for v in (lat_med, spec) if v is not None]
            reported = float(np.mean(pair)) if pair else None
            source = ("count_spectral_mean" if len(pair) == 2
                      else "clean_interval_median" if lat_med is not None
                      else "waveform_rhythm" if spec is not None else None)
            confidence = "verified"
            flags_ok = lat_med is not None      # need the interval distribution
        elif verdict == "disagree" and spec_backed:
            reported, source, confidence, flags_ok = (
                spec, "waveform_rhythm", "provisional", False)
            reasons.append(
                "the beat count disagreed with the waveform's dominant rhythm; "
                f"reporting the waveform rate ({spec:.0f} bpm), which "
                f"{roia} of 4 facial regions agree on")
        else:
            reported, source, confidence, flags_ok = (
                None, None, "uncertain", False)
            reasons.append(
                "the resting rate could not be verified across facial regions, "
                "so no rate is reported")

        if reported is not None and n_int < MIN_INTERVALS \
                and confidence != "provisional":
            reasons.append(f"only {n_int} clean intervals (< {MIN_INTERVALS})")

        # ---- sustained brady/tachy, only from a VERIFIED interval rate --------
        flag = key = None
        sustained = None
        if flags_ok and reported is not None and n_int >= MIN_INTERVALS:
            bpm = 60000.0 / ibi
            frac_low = float(np.mean(bpm < BRADY_BPM))
            frac_high = float(np.mean(bpm > TACHY_BPM))
            med = float(np.median(bpm))
            if med < BRADY_BPM and frac_low >= MIN_SUSTAINED_FRACTION:
                flag, key = "BRADY", "brady"
            elif med > TACHY_BPM and frac_high >= MIN_SUSTAINED_FRACTION:
                flag, key = "TACHY", "tachy"
            sustained = round(max(frac_low, frac_high), 3)
        elif n_int < MIN_INTERVALS and not reasons:
            reasons.append(f"only {n_int} clean intervals (< {MIN_INTERVALS}) "
                           "— no rate flag")
        assert key is None or key in RATE_FLAG_SENTENCES

        return HeadResult(
            head=self.name, version=self.version,
            measurement_class=MeasurementClass.MEASURED,
            value={"flag": flag,
                   "median_bpm": (round(reported, 1) if reported is not None
                                  else None),
                   "sustained_fraction": sustained,
                   "sentence_key": key,
                   "rate_source": source,
                   "rate_confidence": confidence,
                   "pulse_check_verdict": verdict,
                   "pulse_lattice_bpm": (round(lat_med, 1)
                                         if lat_med is not None else None),
                   "pulse_spectral_bpm": (round(spec, 1)
                                          if spec is not None else None)},
            confidence=None,
            reasons=reasons,
        )


register_head(RateFlagsHead)
