"""
head_regularity (v0.7 T4) — the substrate head: "is this pulse regular
or irregular, and how confident are we?"

It emits, from the ONE representation (features/regularity.py):
  index + CI     the continuous irregularity index with its bootstrap
                 CI — MEASURED-adjacent, always present with any class
                 (invariant G-b: a class without an index and CI is a
                 bug);
  class          regular / irregular under the SAME published definition
                 the ECG reference uses (configs/gates.yaml
                 regularity.reference_label);
  benign_pattern_evidence
                 on every irregular call (invariant G-c): respiration_
                 coupled / ectopy_pattern / chaotic / indeterminate.
                 Indeterminate is allowed; silent omission is not. These
                 are EXPLANATIONS attached to an irregular finding, never
                 standalone rhythm claims (G-d).

It ABSTAINS — class None, index still reported — when clean runs are
insufficient or the respiration channel is too poor to test coupling
against: without that channel the dominant false positive (a healthy
person breathing) cannot be told from a pathological irregularity, so
the head declines rather than guesses. Today's consumer path supplies
no respiration channel, so the head abstains on every production scan
by design (as head_flutter does), and that is pinned by test.

Research-flagged (MeasurementClass.RESEARCH_RHYTHM) until §R is green
and signed; the sanctioned sentence exists, names no rhythm, and leaves
through user_facing_text() only. This head never escalates to an
AFib-suggestive sentence — that is head_afib's job.
"""
from __future__ import annotations

import math

from datasets.schema import (MeasurementClass, REGULARITY_SENTENCES,
                             ScanOutcome)
from features.regularity import regularity_from_runs
from heads.base import EndpointHead, HeadResult, register_head

WATERMARK = ("RESEARCH ARTIFACT — pulse-regularity index, §R-gated, "
             "NOT VALIDATED, NAMES NO RHYTHM, NOT A DIAGNOSIS")
BENIGN_EVIDENCE = ("respiration_coupled", "ectopy_pattern", "chaotic",
                   "indeterminate")
# Planning thresholds for the explanations (owner may revise from data;
# the class threshold itself lives in gates.yaml and is read at run time
# so the head and the ECG reference can never disagree about it).
RESP_COUPLED_MIN_FRACTION = 0.50
ECTOPY_ALTERNATION_MIN = 0.50
# phase-locking value (3 segments) below which the tachogram is not
# moving WITH the breath; measured on the fixtures: locked RSA 0.999,
# AF 0.37, an unrelated periodic series up to ~0.80 by chance
PLV_LOCKED_MIN = 0.85
# the reported-rate fraction must at least be corroborated for the
# tachogram peak to be read as "that rate, slightly mis-estimated"
RATE_CORROBORATION_MIN = 0.20
# the breath waveform must itself carry power at the tachogram's peak
# before a rate-estimate error is blamed
WAVEFORM_AT_PEAK_MIN = 0.30
# median |successive difference| of a white residual with std sigma is
# ~ 0.6745 * sqrt(2) * sigma
RESIDUAL_MEDIAN_ABS_DIFF_FACTOR = 0.954
DEFAULT_RESIDUAL_THRESHOLD = 0.06
_DEFINITION_CACHE: dict = {}


def user_facing_text(value: dict, *, render_allowed: bool = False):
    """The ONLY route from this head to a human sentence (G-d).
    Fail-closed: the default cannot render."""
    if not render_allowed:
        return None
    key = (value or {}).get("sentence_key")
    if key not in REGULARITY_SENTENCES:
        return None
    return REGULARITY_SENTENCES[key]


def _definition(cfg: dict) -> dict:
    """The published reference definition — the same one the ECG label
    uses. Read from gates.yaml (cached), overridable by a caller
    passing cfg["regularity"]["reference_label"] for tests."""
    override = ((cfg or {}).get("regularity") or {}).get("reference_label")
    if isinstance(override, dict):
        return override
    if "d" not in _DEFINITION_CACHE:
        from datasets.regularity_reference import load_reference_definition
        _DEFINITION_CACHE["d"] = load_reference_definition()
    return _DEFINITION_CACHE["d"]


def _explained_fraction(coup: dict) -> tuple:
    """How much of the tachogram's power the breath can claim, and on
    what evidence. Three routes, each needing the tachogram to MOVE
    WITH the breath (phase locking) where a waveform exists:
    1. the fraction at the reported breathing rate, corroborated;
    2. the fraction at the tachogram's own peak when the reported rate
       was slightly off (locked at the reported rate, peak within the
       band tolerance);
    3. the fraction at the tachogram's own peak when the breath
       WAVEFORM is itself at that peak and locked there (a rate-estimate
       error the waveform exposes).
    Returns (fraction or None, route or None)."""
    plv = coup.get("phase_locking")
    frac_r = coup.get("tachogram_resp_fraction")
    if frac_r is None and coup.get("status") == "rate_uncorroborated":
        frac_r = coup.get("tachogram_resp_fraction_at_reported_rate")
    frac_p = coup.get("tachogram_peak_fraction")
    plv_p = coup.get("phase_locking_at_peak")
    wave_p = coup.get("resp_waveform_fraction_at_peak")
    locked_r = plv is None or float(plv) >= PLV_LOCKED_MIN
    routes = []
    if frac_r is not None and float(frac_r) >= RESP_COUPLED_MIN_FRACTION \
            and locked_r:
        routes.append((float(frac_r), "reported_rate"))
    if plv is not None and float(plv) >= PLV_LOCKED_MIN and \
            frac_p is not None and float(frac_p) >= RESP_COUPLED_MIN_FRACTION \
            and frac_r is not None and float(frac_r) > RATE_CORROBORATION_MIN \
            and plv_p is None:
        routes.append((float(frac_p), "peak_near_reported_rate"))
    if plv_p is not None and float(plv_p) >= PLV_LOCKED_MIN and \
            wave_p is not None and float(wave_p) >= WAVEFORM_AT_PEAK_MIN \
            and frac_p is not None and float(frac_p) >= RESP_COUPLED_MIN_FRACTION:
        routes.append((float(frac_p), "waveform_at_peak"))
    if not routes:
        return None, None
    return max(routes)


def benign_pattern_evidence(reg, threshold: float = None) -> dict:
    """The explanation for an irregular interval series, from the
    structure family.

    respiration_coupled needs TWO things: the breath must claim at
    least half the tachogram's power on evidence that the tachogram
    moves WITH the breath (phase locking, where a waveform exists), AND
    what is left after the breath is removed must itself be regular by
    the published definition — a chaotic series with some respiratory
    modulation is not "explained by breathing" (review finding). The
    residual is ESTIMATED from the unexplained power and the
    tachogram's spread (median |Δ| of white residual ≈ 0.95 σ).
    Then a short-period alternation or paired short/long outliers is
    the ectopy signature; pervasive scatter with neither is chaotic;
    anything else is indeterminate."""
    thr = float(threshold if threshold is not None else
                DEFAULT_RESIDUAL_THRESHOLD)
    st = reg.structure or {}
    coup = st.get("coupling") or {}
    per = st.get("periodicity") or {}
    outl = st.get("outliers") or {}
    alt = per.get("alternation_index")
    topo = outl.get("topology")
    explained, route = _explained_fraction(coup)
    rel_std = coup.get("tachogram_rel_std")
    residual = None
    if explained is not None and rel_std is not None:
        residual = round(RESIDUAL_MEDIAN_ABS_DIFF_FACTOR *
                         math.sqrt(max(0.0, 1.0 - explained)) *
                         float(rel_std), 5)
    detail = {"coupling_available": bool(coup.get("available")),
              "tachogram_resp_fraction": coup.get("tachogram_resp_fraction"),
              "explained_fraction": explained, "coupling_route": route,
              "phase_locking": coup.get("phase_locking"),
              "residual_index_estimate": residual,
              "residual_threshold": thr,
              "alternation_index": alt, "outlier_topology": topo}
    coupled = explained is not None and (residual is None or residual < thr)
    if coupled:
        return {"evidence": "respiration_coupled", **detail}
    if (alt is not None and float(alt) >= ECTOPY_ALTERNATION_MIN) or \
            topo == "paired":
        return {"evidence": "ectopy_pattern", **detail}
    if topo == "pervasive" and (alt is None or float(alt) <
                                ECTOPY_ALTERNATION_MIN):
        return {"evidence": "chaotic", **detail}
    return {"evidence": "indeterminate", **detail}


class RegularityHead(EndpointHead):
    name = "regularity"
    version = "0.7.0-research"
    required_inputs = ("regularity", "scan_outcome", "respiration")
    research_only = True

    def run(self, lattice, context: dict) -> HeadResult:
        ctx = context or {}
        cfg = ctx.get("cfg") or {}
        reasons = [WATERMARK]
        value = {"index": None, "class": None,
                 "benign_pattern_evidence": None,
                 "sentence_key": None, "user_facing": None,
                 "abstained": True, "watermark": WATERMARK}
        outcome = ctx.get("scan_outcome")
        if outcome != ScanOutcome.ACCEPT.value:
            reasons.append(
                f"scan outcome {outcome!r} is not ACCEPT — no regularity "
                "verdict is read from beats the gates rejected"
                if outcome is not None else
                "no scan outcome in context (head ordered before the "
                "decision head?) — failing closed")
            return self._result(value, reasons)
        reg = ctx.get("regularity")
        if reg is None:
            # rebuilt from the lattice WITH its confidences, so the
            # jitter budget carries the confidence penalty (review
            # finding)
            reg = regularity_from_runs(
                list(getattr(lattice, "runs", []) or []),
                list(getattr(lattice, "run_confidences", []) or []) or None,
                float(getattr(lattice, "dropout_rate", 0.0) or 0.0),
                run_times=list(getattr(lattice, "run_times", []) or []),
                fps=getattr(lattice, "fps", None),
                respiration=ctx.get("respiration"))
        elif ctx.get("respiration") is not None and \
                not (reg.structure.get("coupling") or {}).get("available"):
            # the pipeline builds the representation without a
            # respiration channel; a research caller that has one
            # re-derives the structure family with it
            reg = regularity_from_runs(
                reg.runs, getattr(reg, "run_confidences", None),
                reg.values.get("dropout_rate", 0.0),
                reg.values.get("mean_sqi", float("nan")),
                run_times=reg.run_times, fps=reg.fps,
                respiration=ctx.get("respiration"))
        d = _definition(cfg)
        # G-b: the index and its CI are reported whenever they exist,
        # abstention or not — the MEASUREMENT stands on its own
        value["index"] = {"value": reg.index.get("value"),
                          "ci95": reg.index.get("ci95"),
                          "n_diffs": reg.index.get("n_diffs"),
                          "definition": reg.index.get("definition")}
        value["jitter"] = (reg.confidence or {}).get("jitter")
        n = int(reg.n_intervals)
        if n < int(d["min_intervals"]):
            reasons.append(f"only {n} clean intervals "
                           f"(< {int(d['min_intervals'])}) — abstaining")
            return self._result(value, reasons)
        coup = (reg.structure or {}).get("coupling") or {}
        if not coup.get("available") and \
                coup.get("status") != "rate_uncorroborated":
            # an ABSENT channel (none, poor, or no resolvable tachogram)
            # is the abstention case. A tachogram dominated by a
            # non-breathing periodicity is not: that is what an ectopy
            # pattern looks like, and the head must still judge it.
            reasons.append(
                "respiratory coupling could not be tested "
                f"({coup.get('reason')}) — abstaining: without a "
                "breathing channel a healthy person breathing cannot be "
                "told from a pathological irregularity")
            return self._result(value, reasons)
        from datasets.regularity_reference import classify_index
        cls = classify_index(reg.index.get("value"), n, d)
        value.update({"class": cls, "abstained": False,
                      "threshold": float(d["irregular_if_index_at_least"])})
        if cls == "irregular":
            value["benign_pattern_evidence"] = benign_pattern_evidence(
                reg, threshold=float(d["irregular_if_index_at_least"]))
            value["sentence_key"] = "irregular"
        else:
            value["benign_pattern_evidence"] = {"evidence": "not_applicable"}
        assert value["user_facing"] is None
        return self._result(value, reasons, confidence=ctx.get("confidence"))

    @staticmethod
    def _result(value, reasons, confidence=None) -> HeadResult:
        stars = getattr(confidence, "stars", None)
        return HeadResult(
            head=RegularityHead.name, version=RegularityHead.version,
            measurement_class=MeasurementClass.RESEARCH_RHYTHM,
            value=value,
            confidence=(float(stars) / 5.0 if stars is not None else None),
            reasons=list(reasons))


register_head(RegularityHead)
