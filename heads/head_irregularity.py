"""
head_irregularity (v0.3 T6) — RESEARCH FLAG ONLY. A rhythm-AGNOSTIC
irregularity flag: it says "these verified intervals are sustainedly
irregular", never what kind (AF vs ectopy vs sinus arrhythmia is the
afib decision head's gated territory, and beat-morphology typing is out
of scope by spec). Not enabled by default, no sanctioned sentence, never
rendered by app/; it logs evidence under the same promotion discipline
as every research head — validation before any user-facing claim.
"""
from __future__ import annotations

from datasets.schema import MeasurementClass
from features.regularity import regularity_from_runs
from heads.base import EndpointHead, HeadResult, register_head

MIN_INTERVALS = 15
REL_MAD_FLOOR = 0.08          # well above resting sinus HRV, not AF-specific
PNN80_FLOOR = 0.30            # fraction of successive diffs > 80 ms


class IrregularityHead(EndpointHead):
    name = "irregularity"
    version = "0.1.0-research"
    required_inputs = ("regularity",)
    research_only = True

    def run(self, lattice, context: dict) -> HeadResult:
        # v0.7 (G-a): no private interval statistics. The representation
        # arrives in context; a direct caller without one (unit fixtures)
        # gets it built from the lattice by the same module. pnn80 is now
        # a WITHIN-run statistic — the old concatenation counted a fake
        # difference at every run seam.
        reg = (context or {}).get("regularity")
        if reg is None:
            reg = regularity_from_runs(
                list(getattr(lattice, "runs", []) or []),
                run_times=list(getattr(lattice, "run_times", []) or []),
                fps=getattr(lattice, "fps", None))
        n = int(reg.n_intervals)
        ev = {"n_intervals": n}
        flag = False
        if n >= MIN_INTERVALS:
            med = float(reg.values["median_ibi"])
            rel_mad = float(reg.dispersion["rel_mad"])
            pnn80 = float(reg.dispersion["pnn80"] or 0.0)
            ev.update({"median_ibi_ms": round(med, 1),
                       "rel_mad": round(rel_mad, 4),
                       "pnn80": round(pnn80, 3)})
            flag = rel_mad >= REL_MAD_FLOOR or pnn80 >= PNN80_FLOOR
        else:
            ev["note"] = f"insufficient clean intervals (< {MIN_INTERVALS})"
        return HeadResult(
            head=self.name, version=self.version,
            measurement_class=MeasurementClass.INFERRED_RHYTHM,
            value={"research_flag": bool(flag), "evidence": ev,
                   "user_facing": None},     # deliberately: no sentence
            reasons=(["research flag only — rhythm-agnostic, not "
                      "validated, never user-facing"] if flag else []),
        )


register_head(IrregularityHead)
