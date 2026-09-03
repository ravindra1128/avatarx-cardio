"""
head_flutter (v0.6 T2) — the regular-tachyarrhythmia PATTERN FLAG.

This head does not detect atrial flutter and does not name a rhythm. It
cannot: a metronomic pulse at ~150 is produced by 2:1 flutter, by
SVT/AVNRT and by sinus tachycardia alike, and telling them apart needs
sawtooth F waves on an ECG, which facial video does not carry. What it
emits is the SUSPICION of a sustained regular tachyarrhythmia plus a
route to the instrument that can name it (invariant F-a).

Two outputs, one head:
  regular_tachy_flag     per-scan: a sustained banded rate that is also
                         too regular for the rate, with the respiratory
                         modulation that sinus rhythm keeps ABSENT;
  series_ratio_signature across a registered scan series: rates sitting
                         at integer-ratio steps of one latent atrial
                         rate (null below three scans).

It ABSTAINS — flag None, not False — whenever it cannot see what it
needs: a non-ACCEPT scan, too few clean intervals, no camera-derived
respiration to test coupling against, or an unknown frame rate (the
dispersion floor is set by frame quantization, so without fps there is
no floor). Abstention is the honest answer; "no flag" and "could not
look" are different statements and this head never conflates them.

F-b: research-flagged, MeasurementClass.RESEARCH_RHYTHM, so the payload
boundary strips it while the §F gates are red or unsigned. F-d: the
flag is measured-path only — nothing here reads, imports or infers from
the gated ECG-reconstruction track.
"""
from __future__ import annotations

from datasets.schema import (MeasurementClass, REGULAR_TACHY_SENTENCES,
                             ScanOutcome)
from features.flutter import (DEFAULT_BANDS, flutter_features,
                              MIN_INTERVALS)  # noqa: F401
from heads.base import EndpointHead, HeadResult, register_head

WATERMARK = ("RESEARCH ARTIFACT — regular-tachyarrhythmia pattern flag, "
             "§F-gated, NOT VALIDATED, NOT A DIAGNOSIS")
# Only the 2:1 band carries a specific claim. 3:1 (~100) and 4:1 (~75)
# are computed and reported, and are expected to be non-specific: they
# collide with ordinary rhythms, which is exactly why slow fixed-block
# flutter is a documented permanent miss (F-c).
FLAGGING_BANDS = ("2:1",)
# Above this share of tachogram power at the breathing frequency, the
# interval series is respiratory-coupled — i.e. sinus. PLANNING VALUE
# (measured on v0.6 fixtures: metronomic 0.03, sinus tach 0.71-0.91).
MAX_COUPLED_FRACTION = 0.20


def user_facing_text(value: dict, *, render_allowed: bool = False):
    """The ONLY route from this head to a human sentence (F-a).

    Fail-closed by design: `render_allowed` must be passed True by a
    caller that has consulted the §F gate scoreboard. Anything else —
    the default, a missing key, an abstention — returns None, so a
    forgotten check cannot leak a rhythm statement.
    """
    if not render_allowed:
        return None
    key = (value or {}).get("sentence_key")
    if key not in REGULAR_TACHY_SENTENCES:
        return None
    return REGULAR_TACHY_SENTENCES[key]


class FlutterHead(EndpointHead):
    name = "flutter"
    version = "0.6.0-research"
    # `respiration` is listed because it DECIDES the answer: without it
    # the head abstains on every scan. (Nothing enforces required_inputs
    # today — heads/base.py says the pipeline should verify it — so this
    # is documentation of the contract, and the abstention is what
    # actually enforces it.)
    required_inputs = ("runs", "run_times", "scan_outcome",
                       "respiration")
    research_only = True

    def run(self, lattice, context: dict) -> HeadResult:
        ctx = context or {}
        cfg = ctx.get("cfg") or {}
        fc = (cfg.get("flutter") or {})
        reasons = [WATERMARK]
        value = {"regular_tachy_flag": None,
                 "series_ratio_signature": None,
                 "conduction_band": None,      # internal only
                 "sentence_key": None,         # never while §F is red
                 "user_facing": None,
                 "abstained": True,
                 "watermark": WATERMARK}

        outcome = ctx.get("scan_outcome")
        if outcome != ScanOutcome.ACCEPT.value:
            reasons.append(
                f"scan outcome {outcome!r} is not ACCEPT — no rhythm "
                "pattern is read from beats the gates rejected"
                if outcome is not None else
                "no scan outcome in context (head ordered before the "
                "decision head?) — failing closed")
            return self._result(value, reasons)

        # the pipeline hands heads the gate VERDICT, not the ScanResult;
        # the verdict was checked above, so the extractor's own
        # ACCEPT-grade guard is satisfied by a stand-in carrying it
        # the representation the pipeline published (G-a): read, not
        # rebuilt (review finding: the head never passed it through)
        feats = flutter_features(
            _Accepted(), lattice, cfg=cfg,
            respiration=ctx.get("respiration"),
            age_years=ctx.get("age_years"),
            series_rates_bpm=ctx.get("series_rates_bpm"),
            regularity=ctx.get("regularity"))
        if not feats.get("available"):
            reasons.extend(feats.get("reasons") or ["no features"])
            return self._result(value, reasons)

        rate = feats["rate"]
        reg = feats["regularity"]
        coup = reg["coupling"]
        series = feats["series"]
        min_int = int(fc.get("min_intervals", MIN_INTERVALS))
        if int(rate.get("n_intervals") or 0) < min_int:
            reasons.append(f"only {rate.get('n_intervals')} clean "
                           f"intervals (< {min_int}) — abstaining")
            return self._result(value, reasons)
        if reg.get("unknown_measurement_floor"):
            reasons.append("frame rate unknown, so the dispersion floor "
                           "is unknown — abstaining")
            return self._result(value, reasons)
        if not coup.get("available"):
            reasons.append(
                "respiratory coupling could not be tested "
                f"({coup.get('reason')}) — abstaining: an untested "
                "coupling is not an absent one")
            return self._result(value, reasons)

        # only past every abstention exit is anything published: a scan
        # the head could not read must not hand back a fitted atrial
        # rate or a rate fingerprint (review finding)
        value["series_ratio_signature"] = (
            series if series.get("n_scans", 0) >= int(
                fc.get("series_min_scans", 3)) else None)
        value["evidence"] = {"rate": rate, "regularity": reg,
                             "variable_block": feats["variable_block"]}
        bands = tuple(fc.get("flagging_bands") or FLAGGING_BANDS)
        unknown = [b for b in bands if b not in DEFAULT_BANDS]
        if unknown:
            reasons.append(f"config names unknown conduction band(s) "
                           f"{unknown} — abstaining")
            return self._result(value, reasons)
        if set(bands) - set(FLAGGING_BANDS):
            # widening past 2:1 is an owner decision, not a default: at
            # 3:1 (~100) and 4:1 (~75) the sanctioned sentence's own
            # words ("sustained FAST") stop being true
            reasons.append(
                f"config widened the flagging bands to {sorted(bands)} "
                "beyond the pre-registered 2:1 — the sanctioned "
                "sentence describes a FAST pulse and does not apply")
            return self._result(value, reasons)
        max_coupled = float(fc.get("max_coupled_fraction",
                                   MAX_COUPLED_FRACTION))
        in_flagging_band = bool(rate.get("in_band")
                                and rate.get("band") in bands)
        decoupled = bool(coup["tachogram_resp_fraction"] <= max_coupled)
        flag = bool(in_flagging_band and reg.get("below_floor")
                    and decoupled)
        value.update({
            "regular_tachy_flag": flag,
            "abstained": False,
            "conduction_band": rate.get("band") if flag else None,
            "criteria": {"sustained_band": in_flagging_band,
                         "below_dispersion_floor": bool(
                             reg.get("below_floor")),
                         "respiratory_modulation_absent": decoupled,
                         "max_coupled_fraction": max_coupled,
                         "measurement_limited": bool(
                             reg.get("measurement_limited"))},
            # the sentence KEY only; the sentence itself is produced by
            # user_facing_text() and only once §F is green and signed
            "sentence_key": "regular_tachy" if flag else None})
        if flag and reg.get("measurement_limited"):
            reasons.append(
                "dispersion floor was set by frame quantization, not "
                "physiology — this scan's regularity claim is "
                "measurement-limited")
        if not flag:
            reasons.append("no sustained regular-tachyarrhythmia "
                           "pattern in this scan")
        # the head never composes text and never names a rhythm
        assert value["user_facing"] is None
        return self._result(value, reasons,
                            confidence=ctx.get("confidence"))

    @staticmethod
    def _result(value, reasons, confidence=None) -> HeadResult:
        stars = getattr(confidence, "stars", None)
        return HeadResult(
            head=FlutterHead.name, version=FlutterHead.version,
            measurement_class=MeasurementClass.RESEARCH_RHYTHM,
            value=value,
            # no flutter-specific calibration exists: this is the scan's
            # own evidence grade, and §F0 must measure the rest
            confidence=(float(stars) / 5.0 if stars is not None else None),
            reasons=list(reasons))


class _Accepted:
    """Stand-in for the ScanResult the pipeline does not hand to heads,
    carrying the one field the extractor's gate check reads."""
    outcome = ScanOutcome.ACCEPT


register_head(FlutterHead)

# re-exported so callers do not have to know where the bands live
BANDS = DEFAULT_BANDS
