"""head_vasotone — vasomotor-reactivity research head (v0.5, §W).

RESEARCH-FLAGGED (invariant W-a): while any §W gate is red or the
signoff is unsigned, this head produces NO user-facing output of any
kind. On any surface the production pipeline can reach it returns an
inert, number-free result; real reactivity readings exist ONLY when a
research surface supplies precomputed inputs via the context
(research/vascular/tone_runner.py or the evaluation harness).

Quarantine posture: NO import of research/* anywhere in this module
(the transitive walker audits every import, function-scoped and
dynamic included). The research side computes tone features and loads
the run artifact; this head only applies the recorded decision rule —
dict arithmetic on within-session deltas.

Invariant W-c holds structurally: every emitted quantity is a
within-session delta or a dimensionless normalized index — there is no
code path that could emit an absolute cross-session tone level.
"""
from __future__ import annotations

from datasets.schema import MeasurementClass
from heads.base import EndpointHead, HeadResult, register_head

# Must equal research.vascular.WATERMARK (pinned by test; duplicated so
# the head never imports the quarantined package).
WATERMARK = ("VASCULAR MORPHOLOGY — RESEARCH ARTIFACT — "
             "NOT A MEASUREMENT")


class VasotoneHead(EndpointHead):
    name = "vasotone"
    version = "0.1.0-research"
    research_only = True
    required_inputs = ()

    def run(self, lattice, context: dict) -> HeadResult:
        ctx = context or {}
        if ctx.get("vasotone_research_surface") is not True:
            return HeadResult(
                head=self.name, version=self.version,
                measurement_class=MeasurementClass.RESEARCH_VASCULAR,
                value={"available": False},
                confidence=None,
                reasons=["research-only head (W-a): outputs exist only "
                         "on research surfaces while the §W gates "
                         "are red — nothing renders to users"])
        reasons = []
        feats = ctx.get("vasotone_features") or {}
        model = ctx.get("vasotone_model") or {}
        surviving = list(ctx.get("surviving_features") or [])
        if not feats.get("available"):
            reasons.append("no tone features for this session (V-c or "
                           "quality refusal)")
        if feats.get("uncontrolled_optics"):
            reasons.append("W-d: uncontrolled optics — no reactivity "
                           "reading from an unlocked capture")
        if model.get("kind") != "reactivity_primary" or \
                not model.get("primary_feature"):
            reasons.append("no recorded reactivity rule on record — "
                           "run evaluate-vasotone on a provocation "
                           "dataset")
        if not surviving:
            reasons.append("no W1-surviving features — the null-arm "
                           "study blocks all readings (fail closed)")
        primary = model.get("primary_feature")
        if not reasons and primary not in surviving:
            reasons.append("the recorded primary feature did not "
                           "survive the null arms — stale rule, "
                           "refusing")
        floor = model.get("detection_floor_delta_norm")
        if not reasons and (floor is None or float(floor) <= 0):
            reasons.append("no natural-drift detection floor on record "
                           "— a response cannot be distinguished from "
                           "drift, refusing")
        dn = None
        if not reasons:
            entry = (feats.get("features") or {}).get(primary) or {}
            dn = entry.get("delta_norm")
            if dn is None:
                reasons.append(f"session lacks a within-session delta "
                               f"for {primary!r}")
        if reasons:
            return HeadResult(
                head=self.name, version=self.version,
                measurement_class=MeasurementClass.RESEARCH_VASCULAR,
                value={"available": False, "watermark": WATERMARK},
                confidence=None, reasons=reasons)
        floor = float(floor)
        dn = float(dn)
        detected = abs(dn) >= floor
        pi = ctx.get("pi_response") or {}
        pi_dn = pi.get("delta_norm")
        agrees = None
        if pi_dn is not None and detected:
            agrees = bool((dn < 0) == (float(pi_dn) < 0))
        value = {"available": True, "watermark": WATERMARK,
                 "maneuver": feats.get("maneuver"),
                 "response_delta_norm": round(dn, 4),
                 "detected": detected,
                 "magnitude_vs_drift": round(abs(dn) / floor, 2),
                 "direction": ("decrease" if dn < 0 else "increase")
                 if detected else "none",
                 "primary_feature": primary,
                 "surviving_deltas": {
                     k: (feats["features"].get(k) or {}).get("delta_norm")
                     for k in surviving},
                 "contact_reference": (
                     {"delta_norm": pi_dn,
                      "direction_agrees": agrees}
                     if pi_dn is not None else None),
                 "basis": "within-session delta vs the recorded "
                          "natural-drift floor — dimensionless, never "
                          "an absolute level (W-c)"}
        return HeadResult(
            head=self.name, version=self.version,
            measurement_class=MeasurementClass.RESEARCH_VASCULAR,
            value=value, confidence=None,
            reasons=["research artifact — never a measurement, never "
                     "user-facing while the §W gates are red "
                     "(W-a)"])


register_head(VasotoneHead)
