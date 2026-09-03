"""
Three-phase session state machine (v0.4 T3): SAFETY_SCREEN -> REST_SCAN
-> GUIDED_ACTIVITY -> TRANSITION -> RECOVERY_SCAN -> gates -> heads.

Contracts enforced here, not in prose:
  * physiology comes ONLY from the rest and recovery scans, through the
    same production pipeline as every other scan — the activity clip
    goes to the workload verifier and nowhere else;
  * compliance grades the session: reps AND cadence within +/-10 % ->
    compliant; 10-20 % -> REPEAT_SCAN; > 20 % or transition > 10 s ->
    NO_RESULT for recovery/fitness outputs (resting vitals still
    render). An unverified workload makes HRR non-comparable, so the
    HRR fields stay None rather than inviting misuse;
  * the safety screen fails closed: any positive or unanswered question
    blocks the activity (a resting scan remains available);
  * INFERRED_FITNESS fields are never populated here — they belong to
    §V-gated heads.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

from configs import load_config
from datasets.io import _build
from datasets.schema import (Medications, ParticipantContext, ScanOutcome,
                             SessionResult)
from protocol.challenges import get_challenge
from protocol.safety import evaluate_screen

PHASES = ("SAFETY_SCREEN", "REST_SCAN", "GUIDED_ACTIVITY", "TRANSITION",
          "RECOVERY_SCAN", "GATES", "HEADS")
TRANSITION_TARGET_S = 5.0
TRANSITION_TIMEOUT_S = 10.0
COMPLIANT_TOL = 0.10
REPEAT_TOL = 0.20


class SessionError(ValueError):
    pass


def evaluate_compliance(challenge, *, reps, cadence_per_min,
                        transition_s) -> dict:
    """The compliance contract. Fail-closed: an unverifiable activity is
    a NO_RESULT for recovery/fitness outputs, never a shrug."""
    reasons = []
    out = {"protocol_id": challenge.protocol_id,
           "reps_prescribed": round(challenge.cadence_per_min / 60.0
                                    * challenge.duration_s, 1),
           "cadence_prescribed": challenge.cadence_per_min,
           "reps": reps, "cadence_achieved": cadence_per_min,
           "transition_s": transition_s, "reasons": reasons}
    if transition_s is None:
        # fail CLOSED: an unrecorded transition is an unverifiable one —
        # the steepest part of the recovery curve may already be gone
        # (v0.4 review finding: the None case skipped the gate entirely)
        reasons.append("transition time not recorded — early recovery "
                       "cannot be verified")
        out["verdict"] = "no_result"
        return out
    if transition_s > TRANSITION_TIMEOUT_S:
        reasons.append(f"transition took {transition_s:.0f} s (> "
                       f"{TRANSITION_TIMEOUT_S:.0f} s hard timeout) — "
                       "early recovery was not captured")
        out["verdict"] = "no_result"
        return out
    if reps is None or cadence_per_min is None:
        reasons.append("activity could not be verified (no periodic "
                       "motion counted)")
        out["verdict"] = "no_result"
        return out
    rep_dev = abs(reps - out["reps_prescribed"]) / out["reps_prescribed"]
    cad_dev = abs(cadence_per_min - challenge.cadence_per_min) \
        / challenge.cadence_per_min
    out["rep_deviation"] = round(rep_dev, 3)
    out["cadence_deviation"] = round(cad_dev, 3)
    worst = max(rep_dev, cad_dev)
    if worst <= COMPLIANT_TOL:
        out["verdict"] = "compliant"
    elif worst <= REPEAT_TOL:
        out["verdict"] = "repeat"
        reasons.append(f"pace {worst * 100:.0f}% off the prescription "
                       "(10-20% band) — repeat for comparable metrics")
    else:
        out["verdict"] = "no_result"
        reasons.append(f"pace {worst * 100:.0f}% off the prescription "
                       "(> 20%) — workload not standardized")
    return out


def load_session_manifest(path) -> dict:
    p = pathlib.Path(path)
    with open(p) as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise SessionError("session manifest must be an object")
    for k in ("session_id", "protocol_id", "phases"):
        if k not in m:
            raise SessionError(f"session manifest missing {k!r}")
    get_challenge(m["protocol_id"])                 # fail-closed id check
    phases = m["phases"]
    if not isinstance(phases, dict):
        raise SessionError("session manifest phases must be a mapping")
    if "rest" not in phases:
        raise SessionError("session manifest needs at least a rest phase")
    for name, ph in phases.items():
        if name not in ("rest", "activity", "recovery"):
            raise SessionError(f"unknown session phase {name!r}")
        if not isinstance(ph, dict):
            raise SessionError(f"session phase {name!r} must be a mapping")
        if "video" in ph:
            ph["video"] = str((p.parent / ph["video"]).resolve())
    pc = m.get("participant_context")
    if pc is not None:
        pc = dict(pc)
        if pc.get("meds") is not None:
            pc["meds"] = _build(Medications, dict(pc["meds"]))
        m["participant_context"] = _build(ParticipantContext, pc)
    return m


def run_session(manifest_path, *, config: Optional[dict] = None,
                videos_override: Optional[list] = None) -> tuple:
    """SessionResult + per-phase details. `videos_override` (CLI
    positionals) replaces the phase videos in rest/activity/recovery
    order when given."""
    from inference.pipeline import run_with_details

    m = load_session_manifest(manifest_path)
    cfg = config or load_config()
    challenge = get_challenge(m["protocol_id"])
    phases = m["phases"]
    if videos_override:
        order = [n for n in ("rest", "activity", "recovery") if n in phases]
        for name, video in zip(order, videos_override):
            phases[name]["video"] = str(pathlib.Path(video).resolve())

    pc = m.get("participant_context")
    screen = evaluate_screen((m.get("safety_screen") or {})
                             .get("answers") or {})
    det = {"screen": screen, "phases_run": []}
    reasons = []
    result = SessionResult(session_id=str(m["session_id"]),
                           outcome=ScanOutcome.NO_RESULT,
                           protocol_id=challenge.protocol_id,
                           no_read_reasons=reasons)

    # ---------------- REST_SCAN (always, when a video exists)
    stars = []
    res_rest = None
    if phases.get("rest", {}).get("video"):
        det["phases_run"].append("REST_SCAN")
        res_rest, det_rest = run_with_details(phases["rest"]["video"],
                                              config=cfg)
        det["rest"] = det_rest
        result.hr_rest_bpm = res_rest.mean_pulse_rate_bpm
        from rppg.respiration import resting_respiratory_rate
        result.rr_rest_brpm = resting_respiratory_rate(
            phases["rest"]["video"])
        if res_rest.confidence_stars is not None:
            stars.append(int(res_rest.confidence_stars))
        result.phases["rest"] = {
            "outcome": res_rest.outcome.value,
            "stars": res_rest.confidence_stars,
            "analysed_seconds": res_rest.analysed_seconds,
            "reasons": list(res_rest.no_read_reasons)}
        result.model_version = res_rest.model_version
        result.code_commit = res_rest.code_commit
        result.calibration_version = res_rest.calibration_version
        result.config_hash = res_rest.config_hash
        result.capture_meta = {"rest": res_rest.capture_meta}

    # ---------------- SAFETY gate on the activity
    if not screen.passed:
        result.safety_blocked = True
        reasons.append("safety screen not passed — guided activity "
                       "blocked (resting scan only)")
        return _finish(result, stars), det

    if not phases.get("activity", {}).get("video"):
        reasons.append("no guided activity in this session — fitness "
                       "metrics require the standardized workload")
        return _finish(result, stars), det

    # ---------------- GUIDED_ACTIVITY (workload verification ONLY)
    det["phases_run"] += ["GUIDED_ACTIVITY", "TRANSITION"]
    from activity.pose_cadence import count_reps
    act = count_reps(phases["activity"]["video"])
    det["activity"] = act
    result.activity_performed = True
    result.activity_tracker = act.get("tracker")
    transition_s = (phases.get("recovery") or {}).get("transition_s")
    comp = evaluate_compliance(challenge, reps=act.get("reps"),
                               cadence_per_min=act.get("cadence_per_min"),
                               transition_s=transition_s)
    result.compliance = comp
    result.phases["activity"] = {"tracker": act.get("tracker"),
                                 "reps": act.get("reps"),
                                 "cadence_per_min":
                                     act.get("cadence_per_min"),
                                 "verdict": comp["verdict"]}
    reasons.extend(comp["reasons"])

    # ---------------- RECOVERY_SCAN
    if not phases.get("recovery", {}).get("video"):
        reasons.append("no recovery scan in this session")
        result.outcome = ScanOutcome.NO_RESULT
        return _finish(result, stars), det
    det["phases_run"].append("RECOVERY_SCAN")
    res_rec, det_rec = run_with_details(phases["recovery"]["video"],
                                        config=cfg)
    det["recovery"] = det_rec
    if res_rec.confidence_stars is not None:
        stars.append(int(res_rec.confidence_stars))
    result.phases["recovery"] = {
        "outcome": res_rec.outcome.value,
        "stars": res_rec.confidence_stars,
        "analysed_seconds": res_rec.analysed_seconds,
        "reasons": list(res_rec.no_read_reasons)}
    if result.capture_meta is None:
        result.capture_meta = {}
    result.capture_meta["recovery"] = res_rec.capture_meta

    # recovery-phase respiratory rate: computed, RESEARCH-TAGGED, never
    # rendered (no validation on record for post-exercise camera RR)
    from rppg.respiration import resting_respiratory_rate as _rr
    det["research_only"] = {"rr_recovery_brpm":
                            _rr(phases["recovery"]["video"])}

    lat = det_rec.get("lattice")
    metrics = None
    if lat is not None:
        from features.recovery import recovery_metrics
        import numpy as np
        dur = float(det_rec["ingest"].meta.duration_s)
        metrics = recovery_metrics(np.asarray(lat.beat_t_s),
                                   np.asarray(lat.beat_confidence),
                                   duration_s=dur)
        det["recovery_metrics"] = metrics

    finalize_session(result, comp=comp, metrics=metrics, lattice=lat,
                     challenge=challenge, pc=pc, det=det, stars=stars,
                     meds_flagged=screen.meds_flagged)
    return result, det


def finalize_session(result: SessionResult, *, comp, metrics, lattice,
                     challenge, pc, det, stars,
                     meds_flagged: bool = False) -> SessionResult:
    """GATES (compliance x quality -> outcome + MEASURED fields) then
    HEADS — shared by the offline CLI path (run_session) and the live
    three-phase orchestrator (app/session_flow.py) so the two surfaces
    can never drift apart."""
    reasons = result.no_read_reasons
    if "phases_run" in det:
        det["phases_run"].append("GATES")
    if comp["verdict"] == "no_result":
        result.outcome = ScanOutcome.NO_RESULT
    elif metrics is None or metrics.get("hrr60") is None:
        result.outcome = ScanOutcome.REPEAT_SCAN
        reasons.append("recovery scan quality insufficient for recovery "
                       "metrics")
        reasons.extend((metrics or {}).get("reasons") or [])
    elif comp["verdict"] == "repeat":
        result.outcome = ScanOutcome.REPEAT_SCAN
    else:
        result.outcome = ScanOutcome.ACCEPT
        result.hr_end_proxy_bpm = metrics["hr_end_proxy"]
        result.hrr30_bpm = metrics["hrr30"]
        result.hrr60_bpm = metrics["hrr60"]
        result.hrr120_bpm = metrics["hrr120"]
        result.recovery_slope_bpm_min = metrics["recovery_slope_bpm_min"]
        reasons.extend(r for r in metrics["reasons"] if "120" in r)

    if "phases_run" in det:
        det["phases_run"].append("HEADS")
    _run_session_heads(result, lattice, metrics, comp, challenge, pc, det,
                       meds_flagged=meds_flagged)
    return _finish(result, stars)


def _run_session_heads(result, lattice, metrics, comp, challenge, pc,
                       det, *, meds_flagged: bool = False) -> None:
    """recovery (MEASURED) + fitness/trend (INFERRED_FITNESS). The
    INFERRED_FITNESS fields on the SessionResult stay None unless every
    §V gate is green AND signed (invariant 9, v0.4 extension) — the
    heads still run and their classified results ride head_results for
    telemetry and evaluation."""
    import os
    from activity.workload import workload_context
    from heads.base import get_head

    trend_rows = []
    store = None
    store_path = os.environ.get("AVATARX_TREND_STORE")
    if store_path:
        from trend.store import TrendStore
        store = TrendStore(store_path)
        if result.outcome is ScanOutcome.ACCEPT:
            store.append(session_id=result.session_id,
                         protocol_id=result.protocol_id,
                         hrr60_bpm=result.hrr60_bpm,
                         hr_rest_bpm=result.hr_rest_bpm,
                         stars=result.confidence_stars)
        trend_rows = store.sessions(protocol_id=result.protocol_id)

    ctx = {"recovery_metrics": metrics or {},
           "protocol_id": result.protocol_id,
           "hr_rest_bpm": result.hr_rest_bpm,
           "rr_rest_brpm": result.rr_rest_brpm,
           "compliance": comp,
           "workload": (workload_context(challenge, pc)
                        if pc is not None else {"available": False}),
           "participant_context": pc,
           # the screen-level BP/heart-medication answer routes fitness
           # to trend-only even when no detailed meds context exists —
           # fail-closed (v0.4 review finding: the live flow never
           # consumed meds_flagged, so the hard rule could not fire)
           "meds_flagged": bool(meds_flagged),
           "trend_sessions": trend_rows}
    hr_rec = get_head("recovery").run(lattice, ctx)
    hr_fit = get_head("fitness").run(lattice, ctx)
    hr_trd = get_head("trend").run(None, ctx)
    result.head_results = [hr_rec.to_dict(), hr_fit.to_dict(),
                           hr_trd.to_dict()]
    det["heads"] = {"fitness_routed": hr_fit.value.get("routed")}

    from evaluation.fitness_gates import fitness_render_allowed
    if fitness_render_allowed():
        # unreachable while §V is red — tested from both sides
        result.fitness_category = hr_fit.value.get("category")
        result.trend = {k: hr_trd.value.get(k)
                        for k in ("direction", "magnitude_class",
                                  "n_sessions")}


def _finish(result: SessionResult, stars: list) -> SessionResult:
    if stars:
        result.confidence_stars = min(stars)
    return result
