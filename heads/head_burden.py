"""
head_burden (M5.17) — INTERNAL ONLY aggregation of head_afib results
across a registered series of scans: per-day AF-suggestive fraction with
ABSTENTION-AWARE denominators (a day of no-reads has an undefined
fraction, not a zero — treating abstention as absence is how a burden
claim lies). No user-facing output until a burden claim is validated.
"""
from __future__ import annotations

from datasets.schema import MeasurementClass
from heads.base import EndpointHead, HeadResult, register_head


class BurdenHead(EndpointHead):
    name = "burden"
    version = "0.1.0-internal"
    required_inputs = ("scan_series",)
    research_only = True

    def run(self, lattice, context: dict) -> HeadResult:
        series = list((context or {}).get("scan_series") or [])
        days: dict = {}
        for s in series:
            day = str(s.get("date") or "unknown")
            d = days.setdefault(day, {"n_scans": 0, "n_accepted": 0,
                                      "n_af_suggestive": 0})
            d["n_scans"] += 1
            if s.get("outcome") == "ACCEPT":
                d["n_accepted"] += 1
                if s.get("predicted_class") == "AFIB_SUGGESTIVE":
                    d["n_af_suggestive"] += 1
        for d in days.values():
            d["af_suggestive_fraction"] = (
                round(d["n_af_suggestive"] / d["n_accepted"], 3)
                if d["n_accepted"] else None)      # abstention-aware: no
            #                                        accepted scans -> no
            #                                        fraction, never 0
            d["abstention_rate"] = (
                round(1.0 - d["n_accepted"] / d["n_scans"], 3)
                if d["n_scans"] else None)
        return HeadResult(
            head=self.name, version=self.version,
            measurement_class=MeasurementClass.INFERRED_RHYTHM,
            value={"per_day": days, "n_scans": len(series),
                   "user_facing": None},
            reasons=["internal only — burden is not a validated claim"],
        )


register_head(BurdenHead)
