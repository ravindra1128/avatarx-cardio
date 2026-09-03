"""
Live three-phase orchestrator (v0.4 T7) — behind config
`protocol.three_phase` (or AVATARX_THREE_PHASE=1). Wraps two ordinary
ScanSessions (rest, recovery) and an activity verifier; every phase
keeps its own fail-closed gates, and the final assembly runs through the
SAME `protocol.session.finalize_session` as the offline CLI path.

The camera contract per phase is structural: physiology comes only from
the two still-subject ScanSessions; during the activity this class
receives frames and reduces each one to a single vertical-centroid
float — the pulse path is never invoked on a moving subject.
"""
from __future__ import annotations

import enum
import json
import pathlib
import threading
import time

import numpy as np

from activity.pose_cadence import count_cycles_from_series
from app.scan_engine import ScanSession, SessionState
from configs import load_config
from datasets.io import _build
from datasets.schema import (Medications, ParticipantContext, ScanOutcome,
                             SessionResult)
from protocol.challenges import DEFAULT_CHALLENGE, get_challenge
from protocol.safety import (SCREEN_QUESTIONS, STOP_RULES,
                             evaluate_screen)
from protocol.session import evaluate_compliance, finalize_session


def three_phase_enabled(cfg=None) -> bool:
    import os
    env = os.environ.get("AVATARX_THREE_PHASE", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    cfg = cfg or load_config()
    return bool((cfg.get("protocol") or {}).get("three_phase"))


class FlowPhase(str, enum.Enum):
    SAFETY = "safety"
    REST = "rest"
    ACTIVITY = "activity"
    TRANSITION = "transition"
    RECOVERY = "recovery"
    DONE = "done"
    FAILED = "failed"


class MultiPhaseSession:
    """One producer thread + status readers, like ScanSession."""

    def __init__(self, session_id: str, work_dir: str, *,
                 fps_hint: float = 30.0, config=None,
                 rest_seconds=None, recovery_seconds=None):
        self.id = session_id
        self.work_dir = pathlib.Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config or load_config()
        pcfg = self.cfg.get("protocol") or {}
        self.challenge = get_challenge(pcfg.get("challenge",
                                                DEFAULT_CHALLENGE))
        self.rest_seconds = float(rest_seconds
                                  or pcfg.get("rest_seconds", 60))
        self.recovery_seconds = float(recovery_seconds
                                      or pcfg.get("recovery_seconds", 150))
        self.phase = FlowPhase.SAFETY
        self.screen = None
        self.pc = None
        self.rest = ScanSession(f"{session_id}-rest",
                                str(self.work_dir / "rest"),
                                fps_hint=fps_hint,
                                scan_seconds=self.rest_seconds,
                                config=self.cfg)
        self.recovery = None
        self.result = None
        self._lock = threading.Lock()
        self._act_t, self._act_y = [], []
        self._act_live = {"reps": None, "cadence_per_min": None}
        self._t_activity_end = None
        self.transition_s = None
        self._act_final = None

    # ---------------- SAFETY_SCREEN
    def safety(self, answers: dict, participant: dict = None) -> dict:
        with self._lock:
            if self.phase not in (FlowPhase.SAFETY, FlowPhase.REST):
                raise RuntimeError(f"safety screen not editable in phase "
                                   f"{self.phase.value}")
            self.screen = evaluate_screen(answers or {})
            if participant:
                p = dict(participant)
                if p.get("meds") is not None:
                    p["meds"] = _build(Medications, dict(p["meds"]))
                self.pc = _build(ParticipantContext, p)
            self.phase = FlowPhase.REST      # rest scan always available
        return {"passed": self.screen.passed,
                "blockers": list(self.screen.blockers),
                "unanswered": list(self.screen.unanswered),
                "activity_allowed": self.screen.passed}

    # ---------------- GUIDED_ACTIVITY
    def begin_activity(self) -> dict:
        if self.screen is None or not self.screen.passed:
            raise RuntimeError("safety screen not passed — the guided "
                               "activity is blocked (resting scan only)")
        if self.rest.state is not SessionState.DONE:
            raise RuntimeError(f"rest scan not complete "
                               f"(state {self.rest.state.value})")
        with self._lock:
            self.phase = FlowPhase.ACTIVITY
            self._act_t, self._act_y = [], []
        return {"cadence_per_min": self.challenge.cadence_per_min,
                "duration_s": self.challenge.duration_s,
                "display_name": self.challenge.display_name,
                "stop_rules": list(STOP_RULES)}

    def push_activity_frames(self, frames_bgr: list,
                             timestamps_s: list) -> dict:
        """Workload verification ONLY: each frame becomes one vertical
        brightness-centroid float. No trace, no pulse, no exceptions."""
        with self._lock:
            if self.phase is not FlowPhase.ACTIVITY:
                return dict(self._act_live)
            for f, t in zip(frames_bgr, timestamps_s):
                g = np.asarray(f).mean(axis=2)
                rows = g.mean(axis=1)
                rows = rows - rows.min()
                tot = float(rows.sum()) or 1.0
                self._act_y.append(float(
                    (rows * np.arange(rows.size)).sum() / tot))
                self._act_t.append(float(t))
            if len(self._act_t) > 60:
                t = np.asarray(self._act_t)
                fps = 1.0 / float(np.median(np.diff(t)))
                live = count_cycles_from_series(t, self._act_y, fps)
                self._act_live = {"reps": live.get("reps"),
                                  "cadence_per_min":
                                      live.get("cadence_per_min")}
            return dict(self._act_live)

    def end_activity(self) -> dict:
        with self._lock:
            if self.phase is not FlowPhase.ACTIVITY:
                raise RuntimeError("no activity in progress")
            self.phase = FlowPhase.TRANSITION
            self._t_activity_end = time.time()
            if len(self._act_t) > 60:
                t = np.asarray(self._act_t)
                fps = 1.0 / float(np.median(np.diff(t)))
                self._act_final = count_cycles_from_series(
                    t, self._act_y, fps)
            else:
                self._act_final = {"reps": None, "cadence_per_min": None,
                                   "note": "activity clip too short",
                                   "tracker": "motion_energy"}
            self._act_final.setdefault("tracker", "motion_energy")
        return dict(self._act_final)

    # ---------------- TRANSITION -> RECOVERY
    def attach_recovery(self, scan: ScanSession) -> None:
        with self._lock:
            if self.phase not in (FlowPhase.TRANSITION,
                                  FlowPhase.RECOVERY):
                raise RuntimeError(
                    f"recovery scan cannot be attached in phase "
                    f"{self.phase.value} — complete the activity first")
            self.recovery = scan

    def mark_recovery_started(self) -> None:
        """Called on EVERY start of the recovery ScanSession — a restart
        (pause budget exceeded) discards the recording and re-records,
        so the transition clock must follow the LAST recording start,
        not freeze at the first (v0.4 review finding)."""
        with self._lock:
            if self.recovery is None:
                return                        # no recovery scan attached
            if self._t_activity_end is not None:
                self.transition_s = round(
                    time.time() - self._t_activity_end, 2)
            self.phase = FlowPhase.RECOVERY

    # ---------------- assembly
    def finalize(self) -> dict:
        with self._lock:
            if self.result is not None:       # idempotent: one verdict,
                return self.result            # one trend-store append
        from protocol.session import _finish
        rest_r = self.rest.result or {}
        activity_done = self._act_final is not None
        result = SessionResult(
            session_id=self.id, outcome=ScanOutcome.NO_RESULT,
            protocol_id=self.challenge.protocol_id,
            activity_performed=activity_done)
        reasons = result.no_read_reasons
        stars = []
        if rest_r:
            result.hr_rest_bpm = rest_r.get("pulse_bpm")
            if rest_r.get("confidence_stars") is not None:
                stars.append(int(rest_r["confidence_stars"]))
            result.phases["rest"] = {
                "outcome": rest_r.get("outcome"),
                "stars": rest_r.get("confidence_stars"),
                "sqi": rest_r.get("signal_quality"),
                "reasons": list(rest_r.get("no_read_reasons") or [])}
            prov = rest_r.get("provenance") or {}
            result.model_version = prov.get("model_version", "unknown")
            result.code_commit = prov.get("code_commit", "unknown")
            result.calibration_version = prov.get("calibration_version",
                                                  "unknown")
            result.config_hash = prov.get("config_hash", "unknown")
            result.capture_meta = {"rest": rest_r.get("capture")}
            from rppg.respiration import resting_respiratory_rate
            result.rr_rest_brpm = resting_respiratory_rate(
                self.rest.video_path)
        elif self.rest.error:
            reasons.append(f"rest scan failed: "
                           f"{(self.rest.error or {}).get('code')}")
        if self.screen is not None and not self.screen.passed:
            result.safety_blocked = True
            reasons.append("safety screen not passed — guided activity "
                           "blocked (resting scan only)")
        det = {"phases_run": ["REST_SCAN"]}

        # the activity verdict is part of the record whenever the
        # activity ran — even when the recovery scan never completed
        # (v0.4 review finding: the early exit dropped compliance)
        comp = None
        if activity_done:
            det["phases_run"] += ["GUIDED_ACTIVITY", "TRANSITION"]
            result.activity_tracker = self._act_final.get("tracker")
            comp = evaluate_compliance(
                self.challenge, reps=self._act_final.get("reps"),
                cadence_per_min=self._act_final.get("cadence_per_min"),
                transition_s=self.transition_s)
            result.compliance = comp
            reasons.extend(comp["reasons"])
            result.phases["activity"] = {
                "tracker": result.activity_tracker,
                "reps": self._act_final.get("reps"),
                "cadence_per_min": self._act_final.get("cadence_per_min"),
                "verdict": comp["verdict"]}

        meds_flagged = bool(self.screen and self.screen.meds_flagged)
        if not activity_done or self.recovery is None or \
                self.recovery.state is not SessionState.DONE:
            if not result.safety_blocked and not activity_done:
                reasons.append("no guided activity in this session — "
                               "fitness metrics require the "
                               "standardized workload")
            elif activity_done:
                reasons.append("recovery scan missing or incomplete")
            _finish(result, stars)
            self._emit(result, det)
            return self.result

        # ------------ full path: recovery metrics + gates + heads
        det["phases_run"].append("RECOVERY_SCAN")
        rec_r = self.recovery.result or {}
        if rec_r.get("confidence_stars") is not None:
            stars.append(int(rec_r["confidence_stars"]))
        result.phases["recovery"] = {
            "outcome": rec_r.get("outcome"),
            "stars": rec_r.get("confidence_stars"),
            "sqi": rec_r.get("signal_quality"),
            "transition_s": self.transition_s,
            "reasons": list(rec_r.get("no_read_reasons") or [])}
        if result.capture_meta is None:
            result.capture_meta = {}
        result.capture_meta["recovery"] = rec_r.get("capture")
        rec_det = getattr(self.recovery, "last_det", None) or {}
        lat = rec_det.get("lattice")
        metrics = None
        if lat is not None:
            from features.recovery import recovery_metrics
            dur = float(rec_det["ingest"].meta.duration_s)
            metrics = recovery_metrics(np.asarray(lat.beat_t_s),
                                       np.asarray(lat.beat_confidence),
                                       duration_s=dur)
        finalize_session(result, comp=comp, metrics=metrics, lattice=lat,
                         challenge=self.challenge, pc=self.pc, det=det,
                         stars=stars, meds_flagged=meds_flagged)
        self._emit(result, det)
        return self.result

    def _emit(self, result: SessionResult, det: dict) -> None:
        import dataclasses
        from app.report_session import render_session_fragment
        doc = dataclasses.asdict(result)
        doc["outcome"] = result.outcome.value
        # invariant 9 at the boundary: no RESEARCH_* head entry reaches
        # the browser payload, even as an inert stub
        from datasets.schema import public_head_results
        doc["head_results"] = public_head_results(doc.get("head_results"))
        doc["user_facing_text"] = result.user_facing_text()
        doc["report_html"] = render_session_fragment(result)
        doc["debug"] = {"phases_run": det.get("phases_run")}
        # v0.8 (owner-directed, spec B.24): the gated research tracks in
        # their own key, flag-gated and screen-only. The rhythm tracks
        # read the REST phase (the still-face scan the session already
        # ran); the fitness track reads this session's own head.
        import types as _types
        from app.research_tracks import research_tracks_block
        rest_r = (getattr(self.rest, "result", None) or {})
        rest_shim = _types.SimpleNamespace(
            outcome=_types.SimpleNamespace(value=rest_r.get("outcome")),
            no_read_reasons=list(rest_r.get("no_read_reasons") or []),
            recording_id=rest_r.get("recording_id"))
        blk = research_tracks_block(
            rest_shim, getattr(self.rest, "last_det", None) or {}, self.cfg,
            video_path=getattr(self.rest, "video_path", None),
            session_result=result)
        if blk:
            doc["research_tracks"] = blk
        with self._lock:
            self.result = doc
            self.phase = FlowPhase.DONE
        self._jlog({"kind": "session", "t": time.time(),
                    "outcome": doc["outcome"],
                    "compliance": result.compliance,
                    "transition_s": self.transition_s,
                    "per_phase_sqi": {k: (v or {}).get("sqi")
                                      for k, v in result.phases.items()},
                    "rejection_reasons": list(result.no_read_reasons)})

    def _jlog(self, rec: dict) -> None:
        try:
            with open(self.work_dir / "session_log.jsonl", "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass

    def status(self) -> dict:
        with self._lock:
            return {"id": self.id, "phase": self.phase.value,
                    "activity_live": dict(self._act_live),
                    "transition_s": self.transition_s,
                    "rest_session": self.rest.id,
                    "recovery_session": getattr(self.recovery, "id",
                                                None),
                    "result": self.result}
