"""
head_fitness v1 (v0.4 T5) — INFERRED_FITNESS, disabled by default, and
UNRENDERABLE while any §V gate is red (the report layer enforces the
invariant via evaluation.fitness_gates.fitness_render_allowed; this head
never checks gates itself and never decides to render).

What it emits when it runs at all: an age-referenced HRR60 CATEGORY
(below_typical / typical / above_typical) with a deliberately wide
uncertainty band, the exact inputs used, and a baseline-ladder audit
stub that stays honest about what has NOT been proven (V3 is the pivotal
open question). NEVER a number in mL/kg/min — a forbidden-output test
pins that at every surface.

Hard rule (decision logic, not labeling): rate-limiting medication
answers (beta-blocker / CCB / ivabradine) route this head to trend-only
— HR-response fitness inference is unreliable under rate control.

The reference bands below are PROVISIONAL literature anchors
(HRR declines with age; Cole 1999 marks <=12 bpm at 1 min as abnormal in
active recovery) and REQUIRE the §V validation campaign before any
rendering — which is precisely why the gates exist.
"""
from __future__ import annotations

from datasets.schema import MeasurementClass
from heads.base import EndpointHead, HeadResult, register_head

# (age_max, p20, p80) for HRR60 in bpm — provisional, §V-gated
_HRR60_BANDS = ((30, 25.0, 45.0), (40, 23.0, 42.0), (50, 20.0, 38.0),
                (60, 17.0, 34.0), (70, 14.0, 30.0), (200, 12.0, 26.0))


def _reference_band(age) -> tuple:
    a = 45.0 if age is None else float(age)
    for age_max, p20, p80 in _HRR60_BANDS:
        if a < age_max:
            return p20, p80
    return _HRR60_BANDS[-1][1:]


class FitnessHead(EndpointHead):
    name = "fitness"
    version = "0.1.0-gated"
    required_inputs = ("recovery_metrics", "participant_context")
    research_only = True          # not enabled by default; §V decides ever

    def run(self, lattice, context: dict) -> HeadResult:
        ctx = context or {}
        pc = ctx.get("participant_context")
        meds = getattr(pc, "meds", None)
        m = ctx.get("recovery_metrics") or {}
        inputs_used = {"hrr60_bpm": m.get("hrr60"),
                       "age": getattr(pc, "age", None),
                       "sex_used": False,      # v1: age-referenced only
                       "workload_verified":
                           (ctx.get("compliance") or {}).get("verdict")
                           == "compliant"}
        audit = {"baseline_ladder_note":
                 "V3 unproven: incremental value beyond demographics"
                 "+activity has not been demonstrated; this category is "
                 "unrenderable until §V is green and signed",
                 "reference_bands": "provisional literature anchors"}
        if (meds is not None and meds.rate_limiting) or \
                bool(ctx.get("meds_flagged")):
            # detailed meds context OR the screen-level BP/heart-
            # medication answer: either routes to trend-only, fail-closed
            return HeadResult(
                head=self.name, version=self.version,
                measurement_class=MeasurementClass.INFERRED_FITNESS,
                value={"category": None, "routed": "trend_only",
                       "uncertainty": None, "inputs_used": inputs_used,
                       "audit": audit},
                reasons=["rate-limiting medication reported — HR-based "
                         "fitness category unreliable; trend-only "
                         "routing (hard rule)"])
        hrr60 = m.get("hrr60")
        if hrr60 is None or not inputs_used["workload_verified"]:
            return HeadResult(
                head=self.name, version=self.version,
                measurement_class=MeasurementClass.INFERRED_FITNESS,
                value={"category": None, "routed": None,
                       "uncertainty": None, "inputs_used": inputs_used,
                       "audit": audit},
                reasons=["no comparable HRR60 (quality or compliance) — "
                         "no category"])
        p20, p80 = _reference_band(inputs_used["age"])
        category = ("below_typical" if hrr60 < p20 else
                    "above_typical" if hrr60 > p80 else "typical")
        return HeadResult(
            head=self.name, version=self.version,
            measurement_class=MeasurementClass.INFERRED_FITNESS,
            value={"category": category,
                   "uncertainty": "wide — single session, unvalidated "
                                  "reference bands",
                   "reference_band_bpm": [p20, p80],
                   "inputs_used": inputs_used, "audit": audit},
            reasons=["§V-gated: not renderable while any gate is red"])


register_head(FitnessHead)
