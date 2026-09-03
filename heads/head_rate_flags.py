"""
head_rate_flags (M1.3) — sustained bradycardia (< 50 bpm) / tachycardia
(> 100 bpm) flags from the CLEAN-RUN rate, with the same abstention
discipline as everything else: too few clean intervals -> no flag, with
the reason named. Sanctioned sentences live in
datasets/schema.RATE_FLAG_SENTENCES (spec changelog B.15) — this head
never composes text.
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


class RateFlagsHead(EndpointHead):
    name = "rate_flags"
    version = "1.0.0"
    required_inputs = ("regularity",)

    def run(self, lattice, context: dict) -> HeadResult:
        # v0.7 (G-a): the intervals come from the ONE representation (a
        # direct caller without one gets it built by the same module);
        # the rate itself is a location statistic on them, which is the
        # one thing a head may still compute.
        reg = (context or {}).get("regularity")
        if reg is None:
            reg = regularity_from_runs(
                list(getattr(lattice, "runs", []) or []),
                run_times=list(getattr(lattice, "run_times", []) or []),
                fps=getattr(lattice, "fps", None))
        ibi = np.concatenate(reg.runs) if reg.runs else np.array([])
        ibi = ibi[np.isfinite(ibi) & (ibi > 0)]
        if ibi.size < MIN_INTERVALS:
            return HeadResult(
                head=self.name, version=self.version,
                measurement_class=MeasurementClass.MEASURED,
                value={"flag": None, "median_bpm": None,
                       "sentence_key": None},
                reasons=[f"only {int(ibi.size)} clean intervals "
                         f"(< {MIN_INTERVALS}) — no rate flag"],
            )
        bpm = 60000.0 / ibi
        med = float(np.median(bpm))
        frac_low = float(np.mean(bpm < BRADY_BPM))
        frac_high = float(np.mean(bpm > TACHY_BPM))
        flag, key = None, None
        if med < BRADY_BPM and frac_low >= MIN_SUSTAINED_FRACTION:
            flag, key = "BRADY", "brady"
        elif med > TACHY_BPM and frac_high >= MIN_SUSTAINED_FRACTION:
            flag, key = "TACHY", "tachy"
        assert key is None or key in RATE_FLAG_SENTENCES
        return HeadResult(
            head=self.name, version=self.version,
            measurement_class=MeasurementClass.MEASURED,
            value={"flag": flag, "median_bpm": round(med, 1),
                   "sustained_fraction": round(max(frac_low, frac_high), 3),
                   "sentence_key": key},
            confidence=None,
            reasons=[],
        )


register_head(RateFlagsHead)
