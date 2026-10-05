"""Run the FPR bench sequence B0-B8 in order and render the verdict.

Overall verdict (owner brief section 16):

  PASS          every gate stage PASSed on real evidence -> the measurement
                system is scientifically suitable for the small-subject pilot
  FAIL          a hard-stop gate was violated -> not suitable; the stage's
                reason and corrective experiment say why and what to do
  INCONCLUSIVE  no hard stop, but at least one gate is missing its evidence
                -> name exactly which recordings are still required

The sequence HONORS STOP CONDITIONS: a hard-stop FAIL halts the run (the
report's "stop and diagnose" stages) rather than pressing on. The point of
the harness is to determine whether the instrument can detect the expected
signal without fooling ourselves -- a defensible INCONCLUSIVE beats a
cosmetic PASS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

from . import WATERMARK
from .contract import FprSession
from .stages import STAGES, run_B7, StageResult, PASS, FAIL, INCONCLUSIVE

# Canonical order. B7 (synthetic) is run first because it gates everything:
# a processing bug invalidates every hardware measurement that follows.
ORDER = ["B7", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B8"]


@dataclass
class Qualification:
    verdict: str
    stages: list = field(default_factory=list)       # list[dict]
    missing_inputs: dict = field(default_factory=dict)
    stopped_at: str = ""
    watermark: str = WATERMARK

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "watermark": self.watermark,
                "stopped_at": self.stopped_at,
                "missing_inputs": self.missing_inputs,
                "stages": self.stages,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def run_sequence(session: FprSession, *, stop_on_hard_fail: bool = True
                 ) -> Qualification:
    """Execute B7 then B0..B8; honor hard stops; collect results."""
    results: list[StageResult] = []
    stopped_at = ""
    for stage in ORDER:
        runner = STAGES[stage]
        try:
            res = runner(session)
        except Exception as e:                       # noqa: BLE001 fail closed
            res = StageResult(stage, f"{stage} (errored)", INCONCLUSIVE,
                              reason=f"stage raised {type(e).__name__}: {e}",
                              corrective_experiment="fix the harness or the "
                              "input before trusting this stage")
        results.append(res)
        if res.status == FAIL and res.hard_stop and stop_on_hard_fail:
            stopped_at = stage
            break

    verdict = _overall_verdict(results)
    return Qualification(
        verdict=verdict, stages=[r.to_dict() for r in results],
        missing_inputs={k: v for k, v in session.missing_inputs().items()
                        if v is not None},
        stopped_at=stopped_at)


def _overall_verdict(results: list) -> str:
    statuses = {r.stage: r.status for r in results}
    if any(r.status == FAIL and r.hard_stop for r in results):
        return FAIL
    ran = {r.stage for r in results}
    all_gates = set(STAGES) & set(ORDER)
    if ran >= all_gates and all(statuses.get(g) == PASS for g in all_gates):
        return PASS
    return INCONCLUSIVE


def qualify_session(session: FprSession, *, save: bool = True) -> Qualification:
    """Run the sequence and (optionally) write report/qualification.json."""
    q = run_sequence(session)
    if save:
        session.ensure_dirs()
        session.save_json("report/qualification.json", q.to_dict())
    return q


def format_report(q: Qualification) -> str:
    """Human-readable qualification summary."""
    lines = [WATERMARK, "",
             f"FPR QUALIFICATION VERDICT: {q.verdict}"]
    if q.stopped_at:
        lines.append(f"  (sequence stopped at a hard stop in {q.stopped_at})")
    lines.append("")
    for st in q.stages:
        mark = {"PASS": "PASS ", "FAIL": "FAIL ",
                "INCONCLUSIVE": "INCON"}.get(st["status"], "?????")
        lines.append(f"  [{mark}] {st['stage']}  {st['title']}")
        if st.get("reason"):
            lines.append(f"           {st['reason']}")
        if st["status"] != PASS and st.get("corrective_experiment"):
            lines.append(f"           -> {st['corrective_experiment']}")
    if q.missing_inputs:
        lines.append("")
        lines.append("MISSING EVIDENCE (record these to move past "
                     "INCONCLUSIVE):")
        for stage, rel in sorted(q.missing_inputs.items()):
            lines.append(f"  {stage}: {rel}")
    return "\n".join(lines)


def run_b7_only() -> StageResult:
    """Convenience: the one stage that qualifies from software alone."""
    return run_B7()
