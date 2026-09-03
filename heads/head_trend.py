"""
head_trend v1 (v0.4 T5) — INFERRED_FITNESS, §V-gated like head_fitness.
Within-person change is the best-evidenced use of recovery physiology
(ΔHRR tracked ΔVO2peak at |r|~0.87 in cardiac rehab), so this head exists
from day one — but it claims at most a DIRECTION until the V5
training-response study is green (magnitude classes stay None), and it
renders nowhere until §V is green and signed.

Requires >= 3 accepted sessions of the SAME protocol inside the window.
"""
from __future__ import annotations

import numpy as np

from datasets.schema import MeasurementClass
from heads.base import EndpointHead, HeadResult, register_head

MIN_SESSIONS = 3
DIRECTION_DEADBAND_BPM = 2.0      # |net change| below this = stable


class TrendHead(EndpointHead):
    name = "trend"
    version = "0.1.0-gated"
    required_inputs = ("trend_sessions",)
    research_only = True

    def run(self, lattice, context: dict) -> HeadResult:
        rows = [r for r in (context or {}).get("trend_sessions") or []
                if r.get("hrr60_bpm") is not None]
        value = {"direction": None, "magnitude_class": None,
                 "n_sessions": len(rows),
                 "protocol_id": (context or {}).get("protocol_id")}
        if len(rows) < MIN_SESSIONS:
            return HeadResult(
                head=self.name, version=self.version,
                measurement_class=MeasurementClass.INFERRED_FITNESS,
                value=value,
                reasons=[f"needs >= {MIN_SESSIONS} accepted sessions "
                         f"of the same protocol ({len(rows)} on record)"])
        y = np.asarray([float(r["hrr60_bpm"]) for r in rows])
        x = np.arange(y.size, dtype=float)
        slopes = [(y[j] - y[i]) / (x[j] - x[i])
                  for i in range(y.size) for j in range(i + 1, y.size)]
        net = float(np.median(slopes)) * (y.size - 1)
        value["net_change_bpm"] = round(net, 1)
        if abs(net) < DIRECTION_DEADBAND_BPM:
            value["direction"] = "stable"
        else:                     # higher HRR60 = faster recovery
            value["direction"] = "improving" if net > 0 else "declining"
        # magnitude_class stays None until §V V5 is green (r >= 0.6,
        # AUROC >= 0.75 for a >= 1-MET change) — direction-only claim
        return HeadResult(
            head=self.name, version=self.version,
            measurement_class=MeasurementClass.INFERRED_FITNESS,
            value=value,
            reasons=["§V-gated: not renderable while any gate is red; "
                     "magnitude claims additionally require V5"])


register_head(TrendHead)
