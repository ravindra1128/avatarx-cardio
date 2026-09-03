"""
Live scan engine (app/scan_engine.py): the state machine behind the demo.

Frames come from a browser or camera; the engine tracks the face live,
gives framing/lighting/motion/signal feedback, records the scan LOSSLESSLY
with honest timestamps, and hands the file to THE production pipeline
(`inference.pipeline.run_with_details`). Nothing here computes a result of
its own — every number on the results screen comes from that call.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.scan_engine import ScanSession, SessionState
from configs import load_config


def _blocking_cfg():
    # v0.1.5: the hard-abort semantics for movement/darkness belong to the
    # pre-v0.1.5 BLOCKING gate; in advisory mode these conditions grade the
    # scan (stars) instead of killing it
    cfg = load_config()
    cfg["decision"]["readiness"] = dict(cfg["decision"].get("readiness") or {})
    cfg["decision"]["readiness"]["mode"] = "blocking"
    return cfg
from capture.video_reader import iter_frames
from capture.face_tracking import yunet_available
from scripts.make_synth_video import synth_video, synth_portrait_video, \
    portrait_available


def _frames(path):
    return [(t, f) for t, f in iter_frames(path)]


@pytest.fixture(scope="module")
def sinus_frames(tmp_path_factory):
    d = tmp_path_factory.mktemp("eng")
    p = str(d / "s.avi")
    truth = synth_video(p, kind="sinus", fps=30.0, duration_s=44.0, seed=5)
    return _frames(p), truth


@pytest.fixture(scope="module")
def af_frames(tmp_path_factory):
    d = tmp_path_factory.mktemp("eng2")
    p = str(d / "a.avi")
    truth = synth_video(p, kind="af", fps=30.0, duration_s=44.0, seed=6)
    return _frames(p), truth


def _feed(sess, frames, phase, chunk=15, t_offset=0.0):
    fb = None
    for i in range(0, len(frames), chunk):
        batch = frames[i:i + chunk]
        fb = sess.push_frames([f for _, f in batch],
                              [t + t_offset for t, _ in batch])
    return fb


# ------------------------------------------------------------ preview
def test_preview_feedback_reports_face_and_readiness(sinus_frames, tmp_path):
    frames, _ = sinus_frames
    s = ScanSession("t1", str(tmp_path / "t1"),
                    fps_hint=30.0, scan_seconds=8.0)
    fb = _feed(s, frames[:240], "preview")           # 8 s of preview
    assert fb["face_found"] is True
    assert fb["framing"] == "ok"
    assert fb["lighting"] == "ok"
    assert fb["motion"] == "ok"
    assert fb["signal_quality"] is not None and fb["signal_quality"] > 0.3
    assert fb["ready"] is True
    assert s.state is SessionState.PREVIEW


def test_preview_no_face_and_dark(sinus_frames):
    frames, _ = sinus_frames
    s = ScanSession("t2", "/tmp", fps_hint=30.0)
    blank = [np.full((240, 320, 3), 40, np.uint8) for _ in range(30)]
    fb = s.push_frames(blank, list(np.arange(30) / 30.0))
    assert fb["face_found"] is False and fb["framing"] == "no_face"
    assert fb["ready"] is False
    dark = [(f * 0.15).astype(np.uint8) for _, f in frames[:30]]
    fb = s.push_frames(dark, list(np.arange(30, 60) / 30.0))
    assert fb["lighting"] == "dark" and fb["ready"] is False


def test_preview_flags_motion(sinus_frames, tmp_path):
    # v0.1.5 advisory mode: violent motion is FLAGGED (check fails, stars
    # drop) but no longer holds the Start button — motion is a quality
    # condition, not a you-have-no-camera condition.
    frames, _ = sinus_frames
    s = ScanSession("t3", str(tmp_path / "t3"), fps_hint=30.0)
    moved = []
    for i, (t, f) in enumerate(frames[:60]):
        dx = 40 if i % 2 else -40                     # violent alternation
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        moved.append(cv2.warpAffine(f, M, (f.shape[1], f.shape[0])))
    fb = s.push_frames(moved, [t for t, _ in frames[:60]])
    assert fb["motion"] == "moving"
    assert fb["readiness"]["checks"]["motion"]["pass"] is False
    assert fb["ready"] is True                        # advisory: still starts


# --------------------------------------------------------------- scan
def test_full_scan_runs_production_pipeline(tmp_path, sinus_frames):
    frames, truth = sinus_frames
    s = ScanSession("t4", str(tmp_path), fps_hint=30.0, scan_seconds=16.0)
    fb = _feed(s, frames[:420], "preview")         # 14 s: window + hold -> READY
    assert fb["ready"], fb["readiness"]
    s.start_scan(camera_settings={"exposureMode": "continuous"})
    assert s.state is SessionState.SCANNING
    fb = _feed(s, frames[420:1020], "scan")        # 20 s wall -> 16 s good
    assert fb["progress"] >= 0.99, fb
    s.finish_scan()
    for _ in range(600):
        if s.state in (SessionState.DONE, SessionState.FAILED):
            break
        time.sleep(0.1)
    assert s.state is SessionState.DONE, s.error
    r = s.result
    assert r["outcome"] == "ACCEPT", r
    assert r["predicted_class"] == "SINUS"
    assert r["user_facing_text"]                     # verbatim from ScanResult
    assert 60 < r["pulse_bpm"] < 90
    assert r["provenance"]["code_commit"] not in ("", "unknown")
    assert r["capture"]["profile"] == "consumer"
    assert any("exposure" in c for c in r["capture"]["caveats"])
    # honest timestamps: sidecar was written and used
    assert pathlib.Path(s.video_path + ".timestamps.json").exists()
    assert abs(r["capture"]["measured_fps"] - 30.0) < 1.0
    # v0.3: the unified report defaults to findings-only — NO waveform of
    # any kind on the consumer surface (owner requirement + §G)
    rep = r.get("report_html", "")
    assert "AvatarX Cardiac Rhythm Scan Report" in rep
    assert "FINDINGS" in rep
    assert "<polyline" not in rep and "<svg" not in rep


def test_research_tracks_ride_the_browser_payload(tmp_path, sinus_frames):
    """v0.8 (owner-directed, spec B.24): the five gated research tracks
    are in the payload the results screen renders — by default, with
    their gates RED beside them. Red gates do not suppress them."""
    frames, _ = sinus_frames
    s = ScanSession("t4rt", str(tmp_path), fps_hint=30.0, scan_seconds=16.0)
    _feed(s, frames[:420], "preview")
    s.start_scan()
    _feed(s, frames[420:1020], "scan")
    s.finish_scan()
    for _ in range(600):
        if s.state in (SessionState.DONE, SessionState.FAILED):
            break
        time.sleep(0.1)
    assert s.state is SessionState.DONE, s.error
    rt = s.result.get("research_tracks")
    assert rt, "the research tracks are missing from the browser payload"
    assert "RESEARCH ARTIFACT" in rt["watermark"]
    assert [t["key"] for t in rt["tracks"]] == [
        "rhythm_regularity", "atrial_flutter", "arterial_stiffness",
        "vascular_tone", "cardiorespiratory_fitness"]
    for t in rt["tracks"]:
        assert t["gates"]["promotion"] == "BLOCKED"      # red, and shown
        assert len(t["gates"]["gates"]) >= 5
        assert t["status"] in ("value", "abstained", "not_run",
                               "not_applicable", "not_available_here")
    # the rhythm head measured this scan and its index rode along
    assert rt["tracks"][0]["value"]["index"]["value"] is not None
    # The older gated research block remains outside the generated report;
    # the three separately-labelled prototype biomarker rows are tested below.
    assert "RESEARCH ARTIFACT" not in s.result.get("report_html", "")
    for h in s.result["head_results"]:
        assert not str(h["measurement_class"]).startswith("RESEARCH_")

    # Current product contract: one accepted resting scan also carries the
    # three explicit biomarker estimates through API state into the same
    # desktop/mobile/print report.  They are real computed values, never UI
    # defaults, and each exposes its method and signal-evidence confidence.
    b = s.result["biomarkers"]
    assert b["complete"] is True
    assert [x["key"] for x in b["items"]] == [
        "arterial_stiffness", "vascular_tone",
        "cardiorespiratory_fitness"]
    for item in b["items"]:
        assert item["status"] == "computed"
        assert item["value"] is not None and item["method"]
        assert item["label"] == "Research Estimate / Prototype"
        assert item["confidence"]["signal_quality_index"] is not None
        assert item["confidence"]["n_beats_used"] >= 8
        assert f'data-biomarker="{item["key"]}"' in s.result["report_html"]
    assert "@media (max-width:700px)" in s.result["report_html"]
    blog = pathlib.Path(s.work_dir) / "biomarker_log.jsonl"
    assert blog.exists()
    logged = json.loads(blog.read_text().splitlines()[-1])
    assert logged["biomarkers"] == b


def test_af_scan_reads_irregular(tmp_path, af_frames):
    frames, _ = af_frames
    # AF call needs >= 20 VERIFIED intervals (v0.1.2 evidence gate): scan
    # long enough to supply them, as the 30 s product scan does.
    s = ScanSession("t5", str(tmp_path), fps_hint=30.0, scan_seconds=22.0)
    fb = _feed(s, frames[:420], "preview")
    assert fb["ready"], fb["readiness"]
    s.start_scan()
    _feed(s, frames[420:1320], "scan")
    s.finish_scan()
    for _ in range(600):
        if s.state in (SessionState.DONE, SessionState.FAILED):
            break
        time.sleep(0.1)
    assert s.state is SessionState.DONE, s.error
    assert s.result["predicted_class"] in ("AFIB_SUGGESTIVE", "OTHER_IRREGULAR")


def test_scan_pauses_when_face_lost_then_restarts(tmp_path, sinus_frames):
    """v0.1.3: losing the face PAUSES the good-time timer (the recording
    stops; the hole is honest in the sidecar); when the pause budget is
    exhausted the session RESTARTS at readiness rather than delivering a
    doomed recording."""
    frames, _ = sinus_frames
    s = ScanSession("t6", str(tmp_path), fps_hint=30.0, scan_seconds=16.0)
    fb = _feed(s, frames[:420], "preview")
    assert fb["ready"]
    s.start_scan()
    _feed(s, frames[420:480], "scan")
    gone = [np.full((240, 320, 3), 120, np.uint8) for _ in range(90)]   # 3 s
    fb = s.push_frames(gone, list(16.0 + np.arange(90) / 30.0))
    assert s.state is SessionState.SCANNING and fb["paused"] is True
    budget = s.cfg["decision"]["readiness"]["max_paused_s"]
    n = int((budget + 1.0) * 30)
    fb = s.push_frames([np.full((240, 320, 3), 120, np.uint8) for _ in range(n)],
                       list(19.0 + np.arange(n) / 30.0))
    assert s.state is SessionState.PREVIEW
    assert fb["restarted"] is True


def test_scan_aborts_on_excessive_movement(tmp_path, sinus_frames):
    frames, _ = sinus_frames
    s = ScanSession("t7", str(tmp_path), fps_hint=30.0, scan_seconds=16.0,
                    config=_blocking_cfg())
    fb = _feed(s, frames[:420], "preview")
    assert fb["ready"]
    s.start_scan()
    moved = []
    for i, (t, f) in enumerate(frames[420:600]):              # 6 s violent
        dx = 40 if i % 2 else -40
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        moved.append(cv2.warpAffine(f, M, (f.shape[1], f.shape[0])))
    s.push_frames(moved, [t for t, _ in frames[420:600]])
    assert s.state is SessionState.FAILED
    assert s.error["code"] == "movement"


def test_scan_aborts_when_it_goes_dark(tmp_path, sinus_frames):
    frames, _ = sinus_frames
    s = ScanSession("t8", str(tmp_path), fps_hint=30.0, scan_seconds=16.0,
                    config=_blocking_cfg())
    fb = _feed(s, frames[:420], "preview")
    assert fb["ready"]
    s.start_scan()
    dark = [(f * 0.12).astype(np.uint8) for _, f in frames[420:510]]
    s.push_frames(dark, [t for t, _ in frames[420:510]])
    assert s.state is SessionState.FAILED
    assert s.error["code"] == "lighting"


def test_low_signal_scan_completes_but_never_forces_a_result(tmp_path):
    """Noise-only face: easy recording, then explicit quality abstention."""
    rng = np.random.default_rng(9)
    frames = []
    for i in range(720):
        f = np.full((240, 320, 3), 120, np.uint8)
        cv2.ellipse(f, (160, 120), (70, 96), 0, 0, 360, (95, 140, 180), -1)
        f = np.clip(f.astype(float) + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8)
        frames.append((i / 30.0, f))
    s = ScanSession("t9", str(tmp_path), fps_hint=30.0, scan_seconds=8.0)
    fb = _feed(s, frames[:420], "preview")
    # Advisory mode permits the attempt. Provisional pulse evidence guides
    # the user but does not deadlock completion before the full scan exists.
    assert fb["ready"] is True
    assert fb["stars"] is not None and fb["stars"]["stars"] <= 2, fb["stars"]
    s.start_scan()
    _feed(s, frames[420:690], "scan")
    fb = s.status()["feedback"]
    assert fb["paused"] is False
    assert fb["progress"] >= 0.99
    assert any(k in fb["readiness"]["failing"] for k in
               ("signal_snr", "cross_roi_coherence", "prelim_beats", "sqi"))
    s.finish_scan()
    for _ in range(600):
        if s.state in (SessionState.DONE, SessionState.FAILED):
            break
        time.sleep(0.1)
    assert s.state is SessionState.DONE, s.error
    assert s.result["outcome"] in ("REPEAT_SCAN", "NO_RESULT")
    assert not s.result["biomarkers"]["complete"]
    assert all(item["value"] is None for item in
               s.result["biomarkers"]["items"])


@pytest.mark.skipif(not (yunet_available() and portrait_available()),
                    reason="needs YuNet + portrait sample")
def test_portrait_scan_end_to_end(tmp_path):
    p = str(tmp_path / "p.avi")
    synth_portrait_video(p, kind="sinus", fps=30.0, duration_s=44.0, seed=8)
    frames = _frames(p)
    s = ScanSession("t10", str(tmp_path), fps_hint=30.0, scan_seconds=16.0)
    fb = _feed(s, frames[:420], "preview")
    assert fb["face_found"] and fb["tracker"] == "yunet"
    assert fb["roi_boxes"] and set(fb["roi_boxes"]) == {"forehead", "cheek_l",
                                                       "cheek_r", "nose"}
    assert fb["ready"], fb["readiness"]
    s.start_scan()
    _feed(s, frames[420:1020], "scan")
    s.finish_scan()
    for _ in range(900):
        if s.state in (SessionState.DONE, SessionState.FAILED):
            break
        time.sleep(0.1)
    assert s.state is SessionState.DONE, s.error
    assert s.result["outcome"] == "ACCEPT", s.result
    assert s.result["predicted_class"] == "SINUS"
