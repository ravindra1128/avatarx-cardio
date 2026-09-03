"""
head_recovery v1 (v0.4 T5) — MEASURED. The session's recovery physiology
as one classified head result: resting HR/RR, end-exercise HR proxy,
HRR30/60/120, recovery slope, plus the protocol + workload CONTEXT that
makes the numbers comparable. Runs only inside a three-phase session
(the context carries the Task-2 metrics); it never re-extracts signal
and never composes text.
"""
from __future__ import annotations

from datasets.schema import MeasurementClass
from heads.base import EndpointHead, HeadResult, register_head


class RecoveryHead(EndpointHead):
    name = "recovery"
    version = "0.1.0"
    required_inputs = ("recovery_metrics",)

    def run(self, lattice, context: dict) -> HeadResult:
        m = (context or {}).get("recovery_metrics") or {}
        reasons = list(m.get("reasons") or [])
        value = {
            "protocol_id": (context or {}).get("protocol_id"),
            "hr_rest_bpm": (context or {}).get("hr_rest_bpm"),
            "rr_rest_brpm": (context or {}).get("rr_rest_brpm"),
            "hr_end_proxy_bpm": m.get("hr_end_proxy"),
            "hrr30_bpm": m.get("hrr30"),
            "hrr60_bpm": m.get("hrr60"),
            "hrr120_bpm": m.get("hrr120"),
            "recovery_slope_bpm_min": m.get("recovery_slope_bpm_min"),
            "quality": m.get("quality"),
            "workload_context": (context or {}).get("workload"),
            "compliance_verdict": ((context or {}).get("compliance")
                                   or {}).get("verdict"),
        }
        return HeadResult(head=self.name, version=self.version,
                          measurement_class=MeasurementClass.MEASURED,
                          value=value, reasons=reasons)


register_head(RecoveryHead)
