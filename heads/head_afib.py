"""
head_afib (M1.2) — endpoint head #1: the existing gated AFib decision.

This head IS the v0.1 production path (quality gates -> beat-evidence
gates -> interim rules / Model A -> confidence coupling), wrapped in the
head contract. Behaviour is locked by regression tests: identical
ScanResult (sanctioned sentence, star grade, outcome, class) to calling
the decision directly.
"""
from __future__ import annotations

from datasets.schema import MeasurementClass
from heads.base import EndpointHead, HeadResult, register_head


class AfibHead(EndpointHead):
    name = "afib"
    version = "1.0.0"
    required_inputs = ("regularity", "sqi", "coverage", "evidence",
                      "confidence", "recording_id", "cfg")

    def run(self, lattice, context: dict):
        from inference.decision_logic import decide_with_rationale
        # v0.7 (G-a): the decision reads the canonical RegularityFeatures
        # through its legacy view — the same dict object, so the two can
        # never drift. A caller supplying only the view (unit fixtures)
        # is still honoured.
        reg = context.get("regularity")
        features = (reg.as_rhythm_features() if reg is not None
                    else context["features"])
        result, rationale = decide_with_rationale(
            features, context["sqi"], context["coverage"],
            context["cfg"], recording_id=context.get("recording_id",
                                                     "unspecified"),
            evidence=context.get("evidence"),
            confidence=context.get("confidence"))
        return HeadResult(
            head=self.name, version=self.version,
            measurement_class=MeasurementClass.INFERRED_RHYTHM,
            value={"outcome": result.outcome.value,
                   "predicted_class": result.predicted_class,
                   "afib_probability": result.afib_probability,
                   "confidence_stars": result.confidence_stars,
                   "user_facing_text": result.user_facing_text()},
            confidence=(result.confidence_stars / 5.0
                        if result.confidence_stars is not None else None),
            reasons=list(result.no_read_reasons),
        ), result, rationale


register_head(AfibHead)
