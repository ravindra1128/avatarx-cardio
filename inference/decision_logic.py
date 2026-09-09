"""
Decision logic (T5, hardened v0.1.2): quality gates FIRST, then beat-EVIDENCE
gates, then a transparent interim rule — with a full rationale.

GATE ORDER IS THE CONTRACT. No class may be emitted unless every gate
passed; NaN in any gate input fails closed. Property tests sweep this.

v0.1.2 — after a REAL false positive (a live scan read AFIB_SUGGESTIVE
while an Apple Watch check did not detect AF; reproduced on the lossless
recording): the irregularity came from DETECTION ERRORS on a weak,
incoherent signal, not from the heart. Rhythm features cannot tell those
apart; only the EVIDENCE around the beats can. So an accepted result — and
above all an AF call — now requires verified beats:

  * cross-ROI coherence ≥ floor  — the pulse was seen consistently across
    the face (the spec's "strong artifact veto"); 2-ROI chance
    coincidences do not count as verified beats;
  * enough clean intervals AND coverage — dispersion features are
    high-variance below the 25-interval knee (features/rhythm.py warns
    exactly this), and half a scan is not a scan;
  * irregularity NOT explained by detection-error harmonics — half/double
    intervals and short-pair splits (spec T5) are the signature of
    false/missed beats; AF's irregularity is distributed;
  * the pulse-deficit rule needs GOOD coherence — dropout is AF evidence
    only under good signal; otherwise it is detection failure.

When these fail on an irregular-looking series the outcome is REPEAT_SCAN
— never AFIB_SUGGESTIVE, and not OTHER_IRREGULAR either: artifact vs
arrhythmia is undecidable from that data. Abstention is the clinically
responsible answer; a false AF alert costs trust that a repeat scan does
not.

Every decision carries a RATIONALE: each gate's value/threshold/verdict,
the features the rule saw, the evidence, and which rule fired. It is
attached to pipeline details, the CLI JSON (`debug`) and the demo's
Details panel, so any result is explainable and reproducible.

Reference note: an Apple Watch and this prototype are not interchangeable
diagnostic references; ECG-confirmed rhythm remains the only ground truth
for AF performance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from datasets.schema import ScanResult, ScanOutcome
from features.rhythm import RhythmFeatures

INTERIM_MODEL_VERSION = "interim-rules-v0.1.2"

# Fallback evidence thresholds if a config predates the `evidence` block.
_DEFAULT_EVIDENCE = {
    "coherence_floor": 0.20,
    "min_intervals_any": 15,
    "min_coverage_any": 0.60,
    "max_split_fraction_any": 0.30,
    "afib_min_intervals": 20,
    "afib_min_coherence": 0.35,
    "afib_max_harmonic_fraction": 0.20,
    "afib_max_split_fraction": 0.15,
    "deficit_rule_min_coherence": 0.50,
    "max_timing_precision_ms_any": 40.0,
    "afib_max_timing_precision_ms": 30.0,
    "afib_min_timing_matched": 0.75,
}


def _finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------- evidence
def harmonic_fraction(ibi_ms: np.ndarray, half_tol: float = 0.15,
                      double_tol: float = 0.25) -> float:
    """Fraction of intervals near ½× or 2× the series median — the harmonic
    signature of false (½×) and missed (2×) beats. Genuine AF produces some
    by chance (~10% on AF-like series); detection error produces many."""
    x = np.asarray(ibi_ms, float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    med = float(np.median(x))
    if med <= 0:
        return 0.0
    r = x / med
    return float(np.mean((np.abs(r - 0.5) < half_tol) |
                         (np.abs(r - 2.0) < double_tol)))


def timing_precision_from_trains(trains: dict, tol_s: float = 0.10) -> dict:
    """Beat-TIMING precision without ECG (Gate 1's precondition, proxied).

    For every pair of per-ROI beat trains, match beats within `tol_s` and
    take the median |Δt| of matched pairs; report the pair with the highest
    matched fraction (ties: smaller |Δt|). Two independent skin regions
    placing the same beat consistently is the strongest available evidence
    that beat times are precise. Measured: synthetic/portrait references
    1-5 ms with matched 1.00; the real false-positive scans 36-49 ms with
    matched 0.63-0.75 — timing jitter that alone trips the AF rule.
    Returns NaN precision / 0 matched when no pair has >= 5 beats each.
    """
    import itertools
    best = None
    names = [r for r, t in trains.items() if np.asarray(t).size >= 5]
    for a, b in itertools.combinations(names, 2):
        ta, tb = np.sort(np.asarray(trains[a], float)), np.sort(np.asarray(trains[b], float))
        d = []
        for t in ta:
            j = int(np.argmin(np.abs(tb - t)))
            if abs(tb[j] - t) <= tol_s:
                d.append(abs(tb[j] - t))
        frac = len(d) / max(min(ta.size, tb.size), 1)
        med = float(np.median(d)) * 1000.0 if d else float("nan")
        key = (round(frac, 3), -(med if np.isfinite(med) else 1e9))
        if best is None or key > best[0]:
            best = (key, a, b, frac, med)
    if best is None:
        return {"timing_precision_ms": float("nan"),
                "timing_matched_fraction": 0.0, "timing_pair": None}
    return {"timing_precision_ms": best[4], "timing_matched_fraction": best[3],
            "timing_pair": f"{best[1]}-{best[2]}"}


def beat_evidence_from_series(series, runset, sqi_components: Optional[dict] = None,
                              per_roi_trains: Optional[dict] = None
                              ) -> dict:
    """Assemble the beat-evidence dict the decision gates consume from the
    calibrated fused series, the clean-run set and the SQI components."""
    comps = sqi_components or {}
    t = series.times() if series is not None else np.array([])
    ibi = np.diff(t) * 1000.0 if t.size > 1 else np.array([])
    agree = np.array([b.roi_agreement for b in series.beats]) \
        if series is not None and series.beats else np.array([])
    coh = comps.get("cross_roi_coherence")
    if coh is None:
        coh = float(np.mean(np.clip((agree - 0.5) / 0.5, 0, 1))) if agree.size else 0.0
    # iteration 12: the clean-interval pulse the rate head and the cards
    # use, as evidence, so it can be checked against the waveform's
    # dominant rhythm (inference/evidence.py::spectral_pulse)
    runs = list(getattr(runset, "runs", []) or [])
    ibi_clean = (np.concatenate([np.asarray(r, float).ravel() for r in runs])
                 if runs else np.array([]))
    ibi_clean = ibi_clean[np.isfinite(ibi_clean) & (ibi_clean > 0)]
    return {
        "cross_roi_coherence": float(coh) if _finite(coh) else 0.0,
        "pulse_lattice_bpm": (float(np.median(60000.0 / ibi_clean))
                              if ibi_clean.size >= 4 else None),
        "pulse_lattice_n_intervals": int(ibi_clean.size),
        "frac_multi_roi": float(np.mean(agree >= 0.75)) if agree.size else 0.0,
        "n_beats": int(t.size),
        "n_intervals": int(getattr(runset, "n_intervals", 0)),
        "harmonic_fraction": harmonic_fraction(ibi),
        "split_fraction": float(getattr(runset, "split_fraction", 0.0)),
        "n_false_pair_splits": int(getattr(runset, "n_false_pair_splits", 0)),
        "n_missed_splits": int(getattr(runset, "n_missed_splits", 0)),
        "dropout_rate": float(getattr(runset, "dropout_rate", 0.0)),
        **(timing_precision_from_trains(per_roi_trains)
           if per_roi_trains else {}),
    }


# --------------------------------------------------------------- decision
def decide(features: RhythmFeatures, sqi: float, coverage: float,
           config: dict, *, recording_id: str = "unspecified",
           evidence: Optional[dict] = None,
           confidence=None) -> ScanResult:
    """(features, sqi, coverage[, evidence]) -> ScanResult. Never a diagnosis."""
    return decide_with_rationale(features, sqi, coverage, config,
                                 recording_id=recording_id,
                                 evidence=evidence, confidence=confidence)[0]


def decide_with_rationale(features: RhythmFeatures, sqi: float,
                          coverage: float, config: dict, *,
                          recording_id: str = "unspecified",
                          evidence: Optional[dict] = None,
                          confidence=None
                          ) -> tuple[ScanResult, dict]:
    """As `decide`, plus the full rationale dict (gates, features, evidence,
    rule). `evidence` is None when a caller has no beat-level evidence (unit
    fixtures); then the evidence gates are recorded as NOT EVALUATED and
    the classic v0.1 gates apply — the production pipeline always supplies
    evidence.

    `confidence` (v0.1.5, optional ConfidenceStars): attached to the
    result, and it COUPLES to the AF call — a result below 3 stars can
    never be AFIB_SUGGESTIVE; such a call downgrades to the existing
    inconclusive path (REPEAT_SCAN), no new outcome value."""
    dc = config["decision"]
    ec = {**_DEFAULT_EVIDENCE, **(dc.get("evidence") or {})}
    ev = evidence or {}
    gates: list[dict] = []
    reasons: list[str] = []
    why: dict = {"recording_id": recording_id, "gates": gates,
                 "features": {}, "evidence": dict(ev), "rule": {},
                 "model_version": INTERIM_MODEL_VERSION}

    def gate(name, value, thr, op, reason, outcome):
        """Record a gate; return True if it FAILED (and the outcome)."""
        ok = _finite(value) and (
            (value >= thr) if op == ">=" else (value <= thr))
        gates.append({"name": name, "value": (float(value) if _finite(value)
                                             else None),
                      "threshold": thr, "op": op, "pass": bool(ok)})
        if not ok:
            reasons.append(reason if _finite(value)
                           else f"{name} is NaN — failing closed")
        return not ok

    # ---- 1. signal quality / coverage gates (v0.1) ------------------------
    # Every gate in a group is evaluated (not short-circuited) so the
    # rationale lists EVERY failing condition — a repeat-scan reason that
    # names only the first problem hides the others.
    n_int = features.values.get("n_intervals", 0.0)
    fail_nan = (not _finite(sqi)) or (not _finite(coverage))
    f1 = gate("sqi", sqi, float(dc["sqi_floor"]), ">=",
              f"signal quality {sqi:.2f} below floor {dc['sqi_floor']}"
              if _finite(sqi) else "", ScanOutcome.REPEAT_SCAN)
    f2 = gate("coverage", coverage, float(dc["coverage_floor"]), ">=",
              f"clean-interval coverage {coverage:.2f} below floor "
              f"{dc['coverage_floor']}" if _finite(coverage) else "",
              ScanOutcome.REPEAT_SCAN)
    f3 = gate("n_intervals_min", n_int, float(dc["min_clean_intervals"]), ">=",
              f"only {int(n_int) if _finite(n_int) else 0} clean intervals "
              f"(< {dc['min_clean_intervals']})", ScanOutcome.NO_RESULT)
    f4 = False
    for k in ("median_abs_succ_diff", "pnn50", "median_ibi"):
        if not _finite(features.values.get(k)):
            reasons.append(f"required rhythm feature {k} is missing or NaN — "
                           "failing closed")
            gates.append({"name": f"feature:{k}", "value": None,
                          "threshold": "finite", "op": "is", "pass": False})
            f4 = True

    # ---- 2. beat-evidence gates (v0.1.2) — any class ---------------------
    have_ev = bool(evidence)
    coh = ev.get("cross_roi_coherence")
    # v0.1.4.2 TWO-REGION VERIFICATION: with exactly two strong ROIs the
    # >=3-ROI coherence component reads 0.00 BY CONSTRUCTION — a decent
    # home-lit scan (one cheek shadowed, weak nose) is indistinguishable
    # from garbage on that metric alone. Two regions that place the same
    # beats within the AF-grade timing budget ARE independent
    # verification; the two real false-positive scans failed exactly
    # there (36-49 ms, matched 0.63-0.75). No new thresholds — this
    # reuses the AF timing keys as the price of missing 3-ROI coherence.
    _tp0 = ev.get("timing_precision_ms")
    _tm0 = ev.get("timing_matched_fraction")
    two_region_verified = (
        _tp0 is not None and _finite(_tp0) and
        _tp0 <= float(ec["afib_max_timing_precision_ms"]) and
        _tm0 is not None and _finite(_tm0) and
        _tm0 >= float(ec["afib_min_timing_matched"]))
    f5 = False
    if have_ev:
        sf_any = ev.get("split_fraction", 0.0)
        coh_ok = _finite(coh) and coh >= float(ec["coherence_floor"])
        if not coh_ok and two_region_verified:
            gates.append({"name": "two_region_verification", "value": _tp0,
                          "threshold": ec["afib_max_timing_precision_ms"],
                          "op": "<=", "pass": True,
                          "matched_fraction": _tm0,
                          "note": "cross-ROI coherence below floor, but two "
                                  "regions place the same beats within the "
                                  "AF-grade timing budget — independent "
                                  "verification (v0.1.4.2)"})
        else:
            f5 |= gate("cross_roi_coherence", coh, float(ec["coherence_floor"]), ">=",
                       f"pulse not seen consistently across the face (cross-ROI "
                       f"coherence {coh:.2f} < {ec['coherence_floor']}): beats are "
                       "not verified" if _finite(coh) else "",
                       ScanOutcome.REPEAT_SCAN)
        f5 |= gate("coverage_any_class", coverage, float(ec["min_coverage_any"]),
                   ">=", f"only {coverage:.0%} of the scan yielded clean "
                         f"intervals (< {ec['min_coverage_any']:.0%})"
                   if _finite(coverage) else "", ScanOutcome.REPEAT_SCAN)
        f5 |= gate("n_intervals_any_class", n_int, float(ec["min_intervals_any"]),
                   ">=", f"only {int(n_int) if _finite(n_int) else 0} clean "
                         f"intervals — too few to assess rhythm "
                         f"(< {ec['min_intervals_any']})", ScanOutcome.REPEAT_SCAN)
        f5 |= gate("split_fraction_any_class", sf_any,
                   float(ec["max_split_fraction_any"]), "<=",
                   f"{sf_any:.0%} of intervals were broken by missed/false-beat "
                   "detection: the beat sequence is not trustworthy enough for "
                   "any rhythm statement", ScanOutcome.REPEAT_SCAN)
        tp = ev.get("timing_precision_ms")
        if tp is not None:
            f5 |= gate("timing_precision_any_class", tp,
                       float(ec["max_timing_precision_ms_any"]), "<=",
                       f"beat timing too imprecise for any rhythm statement "
                       f"(regions disagree by {tp:.0f} ms per beat > "
                       f"{ec['max_timing_precision_ms_any']:.0f} ms)"
                       if _finite(tp) else "", ScanOutcome.REPEAT_SCAN)
    if f1 or f2 or f3 or f4 or f5:
        # NaN in a gate input or too few intervals -> NO_RESULT (structural);
        # quality-shaped failures -> REPEAT_SCAN (re-scan can fix them).
        outcome = ScanOutcome.NO_RESULT if (fail_nan or f3 or f4) \
            else ScanOutcome.REPEAT_SCAN
        why["rule"]["fired"] = "ABSTAIN (quality/evidence gates)"
        return _finish(why, _abstain(recording_id, outcome, sqi, reasons),
                       confidence=confidence, sqi=sqi)
    if not have_ev:
        gates.append({"name": "beat_evidence", "value": None,
                      "threshold": None, "op": "n/a", "pass": True,
                      "note": "not evaluated (no evidence supplied)"})

    # ---- 3. classifier stage ---------------------------------------------
    r = dc["interim_rules"]
    mad = float(features.values["median_abs_succ_diff"])
    pnn50 = float(features.values["pnn50"])
    med_ibi = float(features.values["median_ibi"])
    dropout = float(features.values.get("dropout_rate", 0.0))
    bpm = 60000.0 / med_ibi if med_ibi > 0 else float("nan")
    why["features"] = {"median_abs_succ_diff": mad, "pnn50": pnn50,
                       "median_ibi": med_ibi, "dropout_rate": dropout,
                       "n_intervals": n_int, "bpm": bpm,
                       "irregularity_index": features.values.get(
                           "irregularity_index"),
                       "rmssd": features.values.get("rmssd"),
                       "n_runs": features.values.get("n_runs"),
                       "longest_run": features.values.get("longest_run")}

    afib_prob: Optional[float] = None
    model_version = INTERIM_MODEL_VERSION
    if dc.get("classifier") == "model_a":
        from models.baseline import load_model_a
        art = dc.get("model_a") or _load_artifact(dc)
        model, art = load_model_a(art)
        vec = np.array([[features.values.get(k, float("nan"))
                         for k in art["feature_names"]]])
        afib_prob = float(model.predict_proba(vec)[0])
        model_version = art["version"]
        why["model_version"] = model_version
        afib_pattern = afib_prob >= float(art["threshold"])
        deficit_path = False
        why["rule"]["afib_probability"] = afib_prob
    else:
        afib_pattern = (mad >= r["afib_median_abs_ms"] and pnn50 >= r["afib_pnn50"])
        deficit_path = (mad >= r["afib_deficit_median_abs_ms"] and
                        dropout >= r["afib_deficit_dropout"])
        why["rule"]["afib_pattern_rule"] = bool(afib_pattern)
        why["rule"]["afib_deficit_rule"] = bool(deficit_path)

    # deficit is AF evidence only under good coherence
    if deficit_path and have_ev:
        ok_def = _finite(coh) and coh >= float(ec["deficit_rule_min_coherence"])
        gates.append({"name": "deficit_rule_coherence", "value": coh,
                      "threshold": ec["deficit_rule_min_coherence"],
                      "op": ">=", "pass": bool(ok_def)})
        if not ok_def:
            deficit_path = False
            why["rule"]["deficit_rule_suppressed"] = (
                "dropout not counted as pulse deficit: coherence too low")
    afib = bool(afib_pattern or deficit_path)

    # ---- 4. AF-call evidence gates (v0.1.2) ------------------------------
    if afib and have_ev:
        hf = ev.get("harmonic_fraction", 0.0)
        sf = ev.get("split_fraction", 0.0)
        failed = False
        failed |= gate("afib_n_intervals", n_int, float(ec["afib_min_intervals"]),
                       ">=", f"irregular intervals seen, but only {int(n_int)} "
                             f"clean intervals (< {ec['afib_min_intervals']}) — "
                             "not enough verified beats for an irregular-rhythm "
                             "result", ScanOutcome.REPEAT_SCAN)
        if not (_finite(coh) and coh >= float(ec["afib_min_coherence"])) \
                and two_region_verified:
            # the AF timing/matched gates below enforce the same budget the
            # two-region mode rests on; the >=3-ROI quota is waived
            gates.append({"name": "afib_two_region_verification",
                          "value": _tp0,
                          "threshold": ec["afib_max_timing_precision_ms"],
                          "op": "<=", "pass": True,
                          "matched_fraction": _tm0,
                          "note": "AF-call coherence quota waived: beats "
                                  "verified by two regions within the AF "
                                  "timing budget (v0.1.4.2)"})
        else:
            failed |= gate("afib_coherence", coh, float(ec["afib_min_coherence"]),
                           ">=", f"irregular intervals seen, but cross-ROI coherence "
                                 f"{coh:.2f} < {ec['afib_min_coherence']}: the beats "
                                 "behind the irregularity are not verified",
                           ScanOutcome.REPEAT_SCAN)
        failed |= gate("afib_harmonic_fraction", hf,
                       float(ec["afib_max_harmonic_fraction"]), "<=",
                       f"irregularity carried by half/double intervals "
                       f"({hf:.0%} harmonic): detection-error signature, not "
                       "rhythm evidence", ScanOutcome.REPEAT_SCAN)
        failed |= gate("afib_split_fraction", sf,
                       float(ec["afib_max_split_fraction"]), "<=",
                       f"{sf:.0%} of intervals broken by missed/false-beat "
                       "splitters: detection-error burden too high to call an "
                       "irregular rhythm", ScanOutcome.REPEAT_SCAN)
        tp = ev.get("timing_precision_ms")
        if tp is not None:
            tm = ev.get("timing_matched_fraction", 0.0)
            failed |= gate("afib_timing_precision", tp,
                           float(ec["afib_max_timing_precision_ms"]), "<=",
                           f"irregular intervals seen, but beat timing is not "
                           f"precise enough to call them (regions disagree by "
                           f"{tp:.0f} ms per beat > "
                           f"{ec['afib_max_timing_precision_ms']:.0f} ms): "
                           "timing jitter alone can look like irregularity"
                           if _finite(tp) else "", ScanOutcome.REPEAT_SCAN)
            failed |= gate("afib_timing_matched", tm,
                           float(ec["afib_min_timing_matched"]), ">=",
                           f"irregular intervals seen, but only {tm:.0%} of "
                           "beats were placed consistently by two regions",
                           ScanOutcome.REPEAT_SCAN)
        if failed:
            why["rule"]["fired"] = "ABSTAIN (irregular but unverified)"
            return _finish(why, _abstain(recording_id,
                                         ScanOutcome.REPEAT_SCAN, sqi,
                                         reasons),
                           confidence=confidence, sqi=sqi)

    other = mad >= r["other_median_abs_ms"] and pnn50 >= r["other_pnn50"]
    if afib:
        cls = "AFIB_SUGGESTIVE"
    elif other:
        cls = "OTHER_IRREGULAR"
    elif _finite(bpm) and bpm >= r["high_rate_bpm"]:
        cls = "HIGH_RATE"
    else:
        cls = "SINUS"
    why["rule"]["fired"] = cls

    res = ScanResult(
        recording_id=recording_id, outcome=ScanOutcome.ACCEPT,
        afib_probability=afib_prob,
        predicted_class=cls, signal_quality_index=float(sqi),
        mean_pulse_rate_bpm=float(bpm) if _finite(bpm) else None,
        no_read_reasons=[], model_version=model_version)
    return _finish(why, res, confidence=confidence, sqi=sqi)


def _finish(why: dict, res: ScanResult, *, confidence=None,
            sqi: float = float("nan")) -> tuple[ScanResult, dict]:
    if confidence is not None:
        res.confidence_stars = int(confidence.stars)
        res.confidence_limiting_factor = confidence.limiting_factor
        # THE COUPLING (v0.1.5): stars express the evidence gates — a scan
        # below 3 stars can never return AFIB_SUGGESTIVE. Downgrade to the
        # existing inconclusive path; no new outcome value.
        if res.predicted_class == "AFIB_SUGGESTIVE" and \
                int(confidence.stars) < 3:
            why.setdefault("gates", []).append(
                {"name": "afib_confidence_floor",
                 "value": int(confidence.stars), "threshold": 3,
                 "op": ">=", "pass": False})
            why.setdefault("rule", {})["fired"] = \
                "ABSTAIN (confidence below 3-star floor)"
            down = _abstain(res.recording_id, ScanOutcome.REPEAT_SCAN, sqi,
                            [f"irregular rhythm seen, but scan confidence "
                             f"{int(confidence.stars)} of 5 is below the "
                             "3-star floor an AF call requires"])
            down.confidence_stars = int(confidence.stars)
            down.confidence_limiting_factor = confidence.limiting_factor
            res = down
    why["outcome"] = res.outcome.value
    why["predicted_class"] = res.predicted_class
    why["no_read_reasons"] = list(res.no_read_reasons)
    why["gates_failed"] = [g["name"] for g in why["gates"] if not g["pass"]]
    return res, why


def _load_artifact(dc: dict) -> dict:
    import json
    import pathlib
    p = pathlib.Path(dc.get("model_a_path", "models/model_a_v01.json"))
    if not p.is_absolute():
        p = pathlib.Path(__file__).resolve().parents[1] / p
    if not p.exists():
        raise RuntimeError(f"model_a artifact missing: {p} — run "
                           "scripts/e6_degradation.py or set "
                           "decision.classifier: interim_rules")
    with open(p) as f:
        return json.load(f)


def _abstain(rid: str, outcome: ScanOutcome, sqi: float,
             reasons: list[str]) -> ScanResult:
    return ScanResult(recording_id=rid, outcome=outcome,
                      signal_quality_index=float(sqi) if _finite(sqi) else None,
                      no_read_reasons=list(reasons),
                      model_version=INTERIM_MODEL_VERSION)
