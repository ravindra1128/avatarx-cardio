"""v0.4 T7 — the live three-phase orchestrator behind the config flag:
rest scan -> guided activity (workload only) -> transition -> recovery
scan -> findings report. The INFERRED_FITNESS render path exists and is
proven unreachable while §V is red."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.report_session import (SESSION_REPORT_TITLE,
                                render_session_fragment)
from app.scan_engine import ScanSession, SessionState
from app.session_flow import (FlowPhase, MultiPhaseSession,
                              three_phase_enabled)
from capture.video_reader import iter_frames
from configs import load_config
from datasets.schema import ScanOutcome, SessionResult

VO2_RE = re.compile(r"m[lL]\s*/\s*kg\s*/\s*min|vo2|vo₂", re.I)


def test_flag_defaults_off_and_env_overrides(monkeypatch):
    cfg = load_config()
    assert cfg["protocol"]["three_phase"] is False
    monkeypatch.delenv("AVATARX_THREE_PHASE", raising=False)
    assert three_phase_enabled(cfg) is False
    monkeypatch.setenv("AVATARX_THREE_PHASE", "1")
    assert three_phase_enabled(cfg) is True
    monkeypatch.setenv("AVATARX_THREE_PHASE", "0")   # review finding:
    assert three_phase_enabled(cfg) is False         # "0" must mean OFF


def test_session_report_escapes_hostile_strings():
    r = SessionResult("s", ScanOutcome.NO_RESULT, activity_performed=True,
                      compliance={"verdict": "no_result",
                                  "reasons": ["<script>alert(1)</script>"]},
                      activity_tracker="<img onerror=x>",
                      protocol_id="sts_1min")
    frag = render_session_fragment(r, _render_allowed=lambda: False)
    assert "<script>alert" not in frag and "&lt;script&gt;" in frag
    assert "<img onerror" not in frag


def _stub_scan(state, result):
    import types
    return types.SimpleNamespace(state=state, result=result, error=None,
                                 id="stub", last_det=None,
                                 video_path="/nonexistent.avi")


def test_finalize_parity_meds_routing_and_partial_paths(tmp_path,
                                                        monkeypatch):
    """Review findings bundle: the live flow must (a) route fitness to
    trend-only on the screen-level meds answer, (b) keep compliance +
    stars + calibration provenance on partial paths, (c) be idempotent."""
    from protocol.safety import SCREEN_QUESTIONS
    cfg = load_config()
    vs = MultiPhaseSession("vstub", str(tmp_path / "v"), config=cfg)
    ans = {q: False for q in SCREEN_QUESTIONS}
    ans["bp_heart_meds"] = True                  # routes, never blocks
    ok = vs.safety(ans)
    assert ok["passed"] and ok["activity_allowed"]
    rest_result = {"pulse_bpm": 70.0, "confidence_stars": 4,
                   "outcome": "ACCEPT", "signal_quality": 0.9,
                   "no_read_reasons": [],
                   "capture": {"profile": "consumer"},
                   "provenance": {"model_version": "m", "code_commit": "c",
                                  "calibration_version": "calv",
                                  "config_hash": "h"}}
    vs.rest = _stub_scan(SessionState.DONE, rest_result)
    vs._act_final = {"reps": 20, "cadence_per_min": 20.0,
                     "tracker": "motion_energy"}
    vs._t_activity_end = time.time() - 3.0
    vs.phase = FlowPhase.TRANSITION
    rec = _stub_scan(SessionState.DONE,
                     {"confidence_stars": 3, "outcome": "REPEAT_SCAN",
                      "signal_quality": 0.5, "no_read_reasons": [],
                      "capture": {"profile": "consumer"}})
    vs.attach_recovery(rec)
    vs.mark_recovery_started()
    doc = vs.finalize()
    assert doc["confidence_stars"] == 3          # min of phase stars
    assert doc["calibration_version"] == "calv"  # provenance parity
    assert doc["capture_meta"]["rest"]["profile"] == "consumer"
    assert doc["compliance"]["verdict"] == "compliant"
    fit = next(h for h in doc["head_results"] if h["head"] == "fitness")
    assert fit["value"]["routed"] == "trend_only"    # meds hard rule LIVE
    assert doc["fitness_category"] is None
    # idempotent: a second finalize returns the same verdict object
    assert vs.finalize() is doc


def test_fitness_render_path_exists_but_unreachable_while_v_red():
    r = SessionResult("s", ScanOutcome.ACCEPT, activity_performed=True,
                      hrr60_bpm=25.0, confidence_stars=4,
                      fitness_category="typical",
                      trend={"direction": "improving"})
    from datasets.schema import (FITNESS_CATEGORY_SENTENCES,
                                 TREND_DIRECTION_SENTENCES)
    closed = render_session_fragment(r)          # live §V check: red
    assert "typical range" not in closed
    assert "recovery trend" not in closed
    opened = render_session_fragment(r, _render_allowed=lambda: True)
    # even when open, ONLY sanctioned sentences render, keyed verbatim
    assert FITNESS_CATEGORY_SENTENCES["typical"] in opened
    assert TREND_DIRECTION_SENTENCES["improving"] in opened
    # field-gate: no category on the result -> nothing even when open
    r2 = SessionResult("s2", ScanOutcome.ACCEPT, activity_performed=True)
    assert "typical range" not in render_session_fragment(
        r2, _render_allowed=lambda: True)


def test_session_report_is_findings_only_and_number_free():
    r = SessionResult("s", ScanOutcome.ACCEPT, activity_performed=True,
                      hr_rest_bpm=71.0, hrr60_bpm=24.0,
                      confidence_stars=4,
                      compliance={"verdict": "compliant", "reasons": []},
                      protocol_id="sts_1min")
    frag = render_session_fragment(r, _render_allowed=lambda: False)
    assert SESSION_REPORT_TITLE in frag
    for tok in ("<svg", "<polyline", "<canvas", "<path", "NOT AN ECG"):
        assert tok not in frag
    assert not VO2_RE.search(frag)
    assert "Pulse-rate drop, 60 s" in frag and "24 bpm" in frag
    assert "★★★★☆" in frag
    assert "not a diagnosis" in frag


def _feed(sess, frames, chunk=15):
    fb = None
    for i in range(0, len(frames), chunk):
        batch = frames[i:i + chunk]
        fb = sess.push_frames([f for _, f in batch],
                              [t for t, _ in batch])
    return fb


def _run_scan(scan: ScanSession, frames, preview_n=120):
    _feed(scan, frames[:preview_n])
    scan.start_scan(camera_settings={})
    _feed(scan, frames[preview_n:])
    scan.finish_scan()
    for _ in range(600):
        if scan.state in (SessionState.DONE, SessionState.FAILED):
            break
        time.sleep(0.1)
    assert scan.state is SessionState.DONE, scan.error


@pytest.fixture(scope="module")
def live_clips(tmp_path_factory):
    from scripts.make_synth_recovery import make_recovery_session
    d = tmp_path_factory.mktemp("live")
    truth = make_recovery_session(d, "live", seed=12, rest_s=26.0,
                                  activity_s=60.0, recovery_s=78.0)
    return d, truth


def test_three_phase_flow_end_to_end(live_clips, tmp_path):
    d, truth = live_clips
    cfg = load_config()
    cfg["protocol"] = dict(cfg["protocol"], rest_seconds=18,
                           recovery_seconds=66)
    vs = MultiPhaseSession("vtest", str(tmp_path / "vs"), config=cfg)
    ok = vs.safety({q: False for q in
                    __import__("protocol.safety",
                               fromlist=["SCREEN_QUESTIONS"])
                    .SCREEN_QUESTIONS},
                   participant={"age": 44, "measured_weight_kg": 78.0,
                                "height_cm": 175.0})
    assert ok["passed"] and ok["activity_allowed"]

    rest_frames = [(t, f) for t, f in
                   iter_frames(str(d / "live_rest.avi"))]
    _run_scan(vs.rest, rest_frames)

    act = vs.begin_activity()
    assert act["cadence_per_min"] == 20.0 and act["stop_rules"]
    act_frames = [(t, f) for t, f in
                  iter_frames(str(d / "live_activity.avi"))]
    for i in range(0, len(act_frames), 30):
        batch = act_frames[i:i + 30]
        live = vs.push_activity_frames([f for _, f in batch],
                                       [t for t, _ in batch])
    assert live["reps"] is not None            # live rep counter works
    final = vs.end_activity()
    assert abs(final["reps"] - truth["reps_scripted"]) <= 1

    rec = ScanSession("vtest-rec", str(tmp_path / "vs" / "recovery"),
                      scan_seconds=66, config=cfg)
    vs.attach_recovery(rec)
    rec_frames = [(t, f) for t, f in
                  iter_frames(str(d / "live_recovery.avi"))]
    _feed(rec, rec_frames[:120])
    rec.start_scan(camera_settings={})
    vs.mark_recovery_started()                 # what /api/scan/start does
    assert vs.transition_s is not None and vs.transition_s < 10.0
    _feed(rec, rec_frames[120:])
    rec.finish_scan()
    for _ in range(600):
        if rec.state in (SessionState.DONE, SessionState.FAILED):
            break
        time.sleep(0.1)
    assert rec.state is SessionState.DONE, rec.error

    doc = vs.finalize()
    assert vs.phase is FlowPhase.DONE
    assert doc["outcome"] == "ACCEPT", doc["no_read_reasons"]
    assert doc["compliance"]["verdict"] == "compliant"
    # the recording starts a few seconds into the decay (preview time =
    # live transition cost); the honest reference is truth at that offset
    from scripts.make_synth_recovery import recovery_hr
    t0 = rec_frames[120][0] - rec_frames[0][0]
    kw = {"hr_rest": truth["hr_rest"], "hr0": truth["hr0"],
          "tau": truth["tau"]}
    expected = float(recovery_hr(t0, **kw) - recovery_hr(t0 + 60.0, **kw))
    assert abs(doc["hrr60_bpm"] - expected) <= 3.0, \
        (doc["hrr60_bpm"], expected, t0)
    assert doc["fitness_category"] is None and doc["trend"] is None
    assert [h["head"] for h in doc["head_results"]] == \
        ["recovery", "fitness", "trend"]
    rep = doc["report_html"]
    assert SESSION_REPORT_TITLE in rep and "<svg" not in rep
    assert not VO2_RE.search(json.dumps(doc))
    # telemetry: compliance, transition, per-phase SQI, rejections
    log = (tmp_path / "vs" / "session_log.jsonl").read_text()
    rec_log = json.loads(log.splitlines()[-1])
    assert rec_log["compliance"]["verdict"] == "compliant"
    assert rec_log["transition_s"] == vs.transition_s
    assert set(rec_log["per_phase_sqi"]) >= {"rest", "recovery"}


def test_safety_blocked_flow_is_rest_only(live_clips, tmp_path):
    d, _ = live_clips
    from protocol.safety import SCREEN_QUESTIONS
    cfg = load_config()
    cfg["protocol"] = dict(cfg["protocol"], rest_seconds=18)
    vs = MultiPhaseSession("vblk", str(tmp_path / "vb"), config=cfg)
    ans = {q: False for q in SCREEN_QUESTIONS}
    ans["chest_pain_activity"] = True
    ok = vs.safety(ans)
    assert not ok["activity_allowed"]
    with pytest.raises(RuntimeError, match="safety"):
        vs.begin_activity()
    rest_frames = [(t, f) for t, f in
                   iter_frames(str(d / "live_rest.avi"))]
    _run_scan(vs.rest, rest_frames)
    doc = vs.finalize()
    assert doc["safety_blocked"] is True
    assert doc["hr_rest_bpm"] is not None      # vitals still render
    assert "skip the activity portion" in doc["user_facing_text"]
