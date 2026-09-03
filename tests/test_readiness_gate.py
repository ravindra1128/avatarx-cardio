"""
Pre-scan readiness gate (v0.1.3): the scan timer must not start until the
live window already satisfies the evidence the post-scan decision will
demand — computed by the SAME code (`inference/evidence.py`) on the SAME
thresholds — and the timer only counts GOOD seconds once running.

Definition of done: a user should almost never complete a scan only to be
told the signal was inadequate.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.scan_engine import ScanSession, SessionState, READINESS_CHECKS
from capture.video_reader import iter_frames
from configs import load_config
from inference.evidence import window_evidence, readiness_from_evidence
from scripts.make_synth_video import synth_video


def _frames(path):
    return [(t, f) for t, f in iter_frames(path)]


@pytest.fixture(scope="module")
def sinus_frames(tmp_path_factory):
    d = tmp_path_factory.mktemp("rdy")
    p = str(d / "s.avi")
    synth_video(p, kind="sinus", fps=30.0, duration_s=60.0, seed=15)
    return _frames(p)


def _feed(s, frames, chunk=15):
    fb = None
    for i in range(0, len(frames), chunk):
        b = frames[i:i + chunk]
        fb = s.push_frames([f for _, f in b], [t for t, _ in b])
    return fb


# ------------------------------------------------- shared evidence function
def test_window_evidence_matches_pipeline_gate_vocabulary(sinus_frames):
    """The readiness gate must speak the decision's language: the same
    evidence keys and thresholds, so READY predicts ACCEPT."""
    cfg = load_config()
    from capture.ingest import ingest_video
    import tempfile
    d = tempfile.mkdtemp(); p = f"{d}/w.avi"
    synth_video(p, kind="sinus", fps=30.0, duration_s=12.0, seed=16)
    ing = ingest_video(p, capture_profile="consumer", assume_rig_locks=False)
    ev = window_evidence(ing.traces, ing.timestamps_s, ing.meta.measured_fps_mean,
                         cfg, tracking_stability=ing.track.stability)
    for k in ("sqi", "cross_roi_coherence", "timing_precision_ms",
              "timing_matched_fraction", "n_beats", "split_fraction",
              "harmonic_fraction", "fps", "max_gap_ms", "window_coverage",
              "per_roi_snr"):
        assert k in ev, k
    rd = readiness_from_evidence(ev, cfg)
    assert rd["ready"] is True, rd
    assert set(rd["checks"]) >= {"signal_snr", "cross_roi_coherence",
                                 "beat_timing", "prelim_beats", "sqi",
                                 "frame_rate", "timestamps"}
    # THE PARITY RULE (v0.1.4): the gate may demand exactly the evidence a
    # RESULT needs (the any-class gates) — not the stricter AF-call bars,
    # which the decision applies only to the AF call itself on the full
    # recording, where that evidence actually accrues.
    for name, thr_key in (("cross_roi_coherence", "coherence_floor"),
                          ("beat_timing", "max_timing_precision_ms_any"),
                          ("timestamps", "min_coverage_any")):
        assert rd["checks"][name]["threshold"] == cfg["decision"]["evidence"][thr_key]


def test_readiness_denies_noise_and_names_the_failing_checks():
    cfg = load_config()
    rng = np.random.default_rng(2)
    n = 240
    ts = np.arange(n) / 30.0
    traces = {r: 120 + rng.normal(0, 3, (n, 3)) for r in
              ("forehead", "cheek_l", "cheek_r", "nose")}
    ev = window_evidence(traces, ts, 30.0, cfg, tracking_stability=1.0)
    rd = readiness_from_evidence(ev, cfg)
    assert rd["ready"] is False
    failed = [k for k, v in rd["checks"].items() if not v["pass"]]
    assert "signal_snr" in failed or "cross_roi_coherence" in failed \
        or "prelim_beats" in failed, failed
    assert rd["hint"]                                    # actionable text


def test_isolated_hole_forgiven_dense_drops_and_low_fps_denied():
    """v0.1.4: the timestamps check judges what the pipeline judges —
    analysable COVERAGE of the window (gap-aware segments), not the mere
    existence of a hole. A single dropped batch is a segment boundary the
    pipeline handles (measured: full scan with a 4-frame hole every 4 s
    is ACCEPT/SINUS); recurring drops that shred every segment are not."""
    cfg = load_config()
    import tempfile
    from capture.ingest import ingest_video
    d = tempfile.mkdtemp(); p = f"{d}/w.avi"
    synth_video(p, kind="sinus", fps=30.0, duration_s=10.0, seed=17)
    ing = ingest_video(p, capture_profile="consumer", assume_rig_locks=False)
    ts = ing.timestamps_s.copy()
    ts[len(ts) // 2:] += 0.6                             # ONE 0.6 s hole
    ev = window_evidence(ing.traces, ts, ing.meta.measured_fps_mean, cfg,
                         tracking_stability=ing.track.stability)
    rd = readiness_from_evidence(ev, cfg)
    assert rd["checks"]["timestamps"]["pass"] is True, rd["checks"]["timestamps"]
    # Randomly sorting heavily jittered timestamps creates collapsed adjacent
    # intervals (near-duplicate frame times).  Those are now an explicit
    # delivery failure instead of being hidden behind a plausible median fps.
    rng = np.random.default_rng(3)
    tsj = np.sort(ing.timestamps_s +
                  rng.normal(0.0, 0.012, ing.timestamps_s.size))
    evj = window_evidence(ing.traces, tsj, ing.meta.measured_fps_mean, cfg,
                          tracking_stability=ing.track.stability)
    assert readiness_from_evidence(evj, cfg)["checks"]["timestamps"]["pass"] \
        is False
    # dense drops: a 5-frame hole every 1.5 s leaves no segment >= 3 s
    keep = np.ones(ing.timestamps_s.size, bool)
    for k in range(40, ing.timestamps_s.size, 45):
        keep[k:k + 5] = False
    trd = {r: ing.traces[r][keep] for r in ing.traces}
    evd = window_evidence(trd, ing.timestamps_s[keep],
                          ing.meta.measured_fps_mean, cfg,
                          tracking_stability=ing.track.stability)
    rdd = readiness_from_evidence(evd, cfg)
    assert rdd["checks"]["timestamps"]["pass"] is False
    ts2 = np.arange(len(ing.timestamps_s)) / 20.0          # 20 fps clock
    ev2 = window_evidence(ing.traces, ts2, 20.0, cfg,
                          tracking_stability=ing.track.stability)
    assert readiness_from_evidence(ev2, cfg)["checks"]["frame_rate"]["pass"] is False


# ------------------------------------------------- engine: readiness + hold
def test_engine_readiness_holds_before_ready(sinus_frames, tmp_path):
    """BLOCKING-mode contract (v0.1.5: preserved behind
    decision.readiness.mode: blocking; advisory is the new default and
    starts on the blocking checks alone — tests/test_confidence_stars)."""
    cfg = load_config()
    cfg["decision"]["readiness"] = dict(cfg["decision"].get("readiness") or {})
    cfg["decision"]["readiness"]["mode"] = "blocking"
    s = ScanSession("r1", str(tmp_path / "r1"), fps_hint=30.0, config=cfg)
    fb = _feed(s, sinus_frames[:120])                     # 4 s: not enough yet
    assert fb["ready"] is False
    assert fb["readiness"]["hold_s"] < s.cfg["decision"]["readiness"]["hold_s"]
    fb = _feed(s, sinus_frames[120:420])                  # +10 s
    assert fb["ready"] is True, fb["readiness"]
    assert fb["readiness"]["hold_s"] >= s.cfg["decision"]["readiness"]["hold_s"]
    assert all(v["pass"] for v in fb["readiness"]["checks"].values()), fb["readiness"]


def test_engine_readiness_checklist_has_actionable_hints(sinus_frames, tmp_path):
    s = ScanSession("r2", str(tmp_path / "r2"), fps_hint=30.0)
    dark = [(f * 0.15).astype(np.uint8) for _, f in sinus_frames[:60]]
    fb = s.push_frames(dark, [t for t, _ in sinus_frames[:60]])
    assert fb["ready"] is False
    assert "light" in fb["readiness"]["hint"].lower()
    blank = [np.full((240, 320, 3), 120, np.uint8) for _ in range(30)]
    fb = s.push_frames(blank, [2.0 + i / 30 for i in range(30)])
    assert "face" in fb["readiness"]["hint"].lower() or "oval" in fb["readiness"]["hint"].lower()


# ------------------------------------------------- scan: good-time timer
def test_scan_timer_counts_only_good_time_and_pauses(tmp_path, sinus_frames):
    s = ScanSession("r3", str(tmp_path), fps_hint=30.0, scan_seconds=15.0)
    _feed(s, sinus_frames[:420])
    assert s.status()["feedback"]["ready"]
    s.start_scan()
    fb = _feed(s, sinus_frames[420:540])                  # 4 s good
    assert 0.24 <= fb["progress"] <= 0.30, fb["progress"]
    # face leaves for 3 s: recording pauses, progress does not advance
    gone = [np.full((240, 320, 3), 120, np.uint8) for _ in range(90)]
    fb = s.push_frames(gone, [18.0 + i / 30 for i in range(90)])
    assert s.state is SessionState.SCANNING
    assert fb["paused"] is True
    assert 0.24 <= fb["progress"] <= 0.30
    # face returns: the evidence window refills (~window_s), then the timer
    # resumes and the remaining good seconds complete the scan
    # Preserve the capture clock: the three-second face-loss interval is a
    # real hole, so returned source frames need the same wall-clock offset.
    returned = [(t + 3.0, f) for t, f in sinus_frames[540:1200]]
    fb = _feed(s, returned)
    assert fb["paused"] is False
    assert fb["progress"] >= 0.99, fb["progress"]
    # honest record: the sidecar shows the pause as a capture gap
    s.finish_scan()
    for _ in range(600):
        if s.state in (SessionState.DONE, SessionState.FAILED):
            break
        time.sleep(0.1)
    assert s.state is SessionState.DONE, s.error
    ts = np.array(s._scan_ts)
    assert np.max(np.diff(ts)) > 2.0                      # the pause is a hole
    assert s.result["outcome"] == "ACCEPT", s.result["no_read_reasons"]
    assert s.result["scan_quality"]["paused_seconds"] > 2.0


def test_scan_restarts_when_pause_budget_exceeded(tmp_path, sinus_frames):
    s = ScanSession("r4", str(tmp_path), fps_hint=30.0, scan_seconds=8.0)
    _feed(s, sinus_frames[:420])
    s.start_scan()
    _feed(s, sinus_frames[420:480])
    budget = s.cfg["decision"]["readiness"]["max_paused_s"]
    n = int((budget + 2.0) * 30)
    gone = [np.full((240, 320, 3), 120, np.uint8) for _ in range(n)]
    fb = s.push_frames(gone, [16.0 + i / 30 for i in range(n)])
    assert s.state is SessionState.PREVIEW                # back to readiness
    assert fb["restarted"] is True and fb["restart_reason"]


# ------------------------------------------------- anti-periodicity of readiness
def test_readiness_is_rhythm_neutral_af_is_ready_as_often_as_sinus(tmp_path):
    """The gate must never keep AF users from starting: on noise-matched
    synthetic AF and sinus, the fraction of READY evaluations across the
    scan must be equal within 5 points (the anti-periodicity rule applied
    to readiness)."""
    from scripts.make_synth_video import synth_video
    rates = {}
    for kind in ("sinus", "af"):
        p = str(tmp_path / f"{kind}.avi")
        synth_video(p, kind=kind, fps=30.0, duration_s=40.0, seed=23)
        frames = _frames(p)
        s = ScanSession(f"rn-{kind}", str(tmp_path), fps_hint=30.0)
        n_ready = n_eval = 0
        for i in range(0, len(frames), 15):
            b = frames[i:i + 15]
            fb = s.push_frames([f for _, f in b], [t for t, _ in b])
            if b[0][0] >= 12.0:                        # after window + hold
                n_eval += 1
                n_ready += int(bool(fb["readiness"]["ready"]))
        rates[kind] = n_ready / max(n_eval, 1)
    assert rates["af"] >= rates["sinus"] - 0.05, rates
    assert rates["af"] > 0.9, rates


# ---------------------------------------------- v0.1.4: the real-world fixes
def test_recurring_dropped_frames_start_and_yield_result(tmp_path, sinus_frames):
    """ROOT-CAUSE REGRESSION (the 'cannot start after 3+ tries' report).
    A browser under transient load drops a small frame batch every few
    seconds. Measured: with a 4-frame hole every 4 s the v0.1.3 gate was
    READY on 0% of evaluations (the hole poisons the whole 8 s window)
    while the full-scan verdict on the SAME stream is ACCEPT/SINUS. The
    gate must start this scan, and the scan must produce a result."""
    src = [(t, f) for t, f in sinus_frames if int(round(t * 30)) % 120 not in
           (60, 61, 62, 63)]
    # the REAL product scan length: 30 s of good time is what guarantees
    # the >= 15 clean intervals the decision demands even with segment
    # boundaries at every drop (a 15 s scan yields only ~8 and REPEATs)
    s = ScanSession("rd1", str(tmp_path), fps_hint=30.0, scan_seconds=30.0)
    i = 0
    while i < len(src) and not s.status()["feedback"]["ready"]:
        b = src[i:i + 15]
        s.push_frames([f for _, f in b], [t for t, _ in b])
        i += 15
    fb = s.status()["feedback"]
    assert fb["ready"] is True, ("never READY; failing="
                                 + str(fb["readiness"]["failing"]))
    assert src[min(i, len(src) - 1)][0] <= 16.0, \
        f"took {src[min(i, len(src)-1)][0]:.1f}s to become ready"
    s.start_scan()
    fb = None
    while i < len(src):
        b = src[i:i + 15]
        fb = s.push_frames([f for _, f in b], [t for t, _ in b])
        i += 15
        if fb["progress"] >= 0.999:
            break
    assert fb["progress"] >= 0.99, (fb["progress"], fb["readiness"]["failing"])
    s.finish_scan()
    for _ in range(600):
        if s.state in (SessionState.DONE, SessionState.FAILED):
            break
        time.sleep(0.1)
    assert s.state is SessionState.DONE, s.error
    assert s.result["outcome"] == "ACCEPT", s.result["no_read_reasons"]


def test_exposure_slow_ae_ramp_passes_step_fails(tmp_path, sinus_frames):
    """AE that drifts slowly (~1.5%/s) is a <0.1 Hz trend the 0.7 Hz
    high-pass removes — it must not block readiness (v0.1.3 failed the
    whole window on a 12% first-half/second-half difference). An abrupt
    AE STEP is a real broadband transient and must still fail, and clear
    once the step leaves the recent-bin window."""
    s = ScanSession("re1", str(tmp_path), fps_hint=30.0)
    ramp = [(t, np.clip(f.astype(np.float32) * (1.0 + 0.015 * t), 0,
                        255).astype(np.uint8)) for t, f in sinus_frames[:600]]
    fb = _feed(s, ramp)
    assert fb["readiness"]["checks"]["exposure"]["pass"] is True, \
        fb["readiness"]["checks"]["exposure"]
    assert fb["ready"] is True, fb["readiness"]["failing"]
    # a 20% step at t=20s: exposure must fail within ~1.5 s...
    stepped, saw_fail, recovered = [], False, False
    for t, f in sinus_frames[600:900]:
        g = np.clip(f.astype(np.float32) * (1.0 + 0.015 * 20.0) * 1.20,
                    0, 255).astype(np.uint8)
        stepped.append((t, g))
    for j in range(0, len(stepped), 15):
        b = stepped[j:j + 15]
        fb = s.push_frames([f for _, f in b], [t for t, _ in b])
        ex = fb["readiness"]["checks"]["exposure"]["pass"]
        if not ex and b[-1][0] <= 21.5:
            saw_fail = True
        if saw_fail and ex and b[-1][0] >= 22.0:
            recovered = True
    assert saw_fail, "exposure step was not detected"
    assert recovered, "exposure check never recovered after the step settled"


def test_lighting_judged_on_face_not_background(tmp_path, sinus_frames):
    """A well-lit face in a dark room is analysable — the photons that
    matter fall on the skin. v0.1.3's whole-frame lux proxy called any
    frame with mean luma < 32/255 'dark' regardless of the face."""
    from capture.face_tracking import FaceTracker
    tr = FaceTracker()
    obs = None
    for _, f in sinus_frames[:30]:
        o = tr.process(f)
        if o.found:
            obs = o
    assert obs is not None
    bx = obs.box or (obs.cx - obs.ax, obs.cy - obs.ay, 2 * obs.ax, 2 * obs.ay)
    x0, y0 = max(0, int(bx[0])), max(0, int(bx[1]))
    x1, y1 = int(bx[0] + bx[2]), int(bx[1] + bx[3])
    # dim so the FACE-BOX luma sits at ~75 (above the 60 floor) while the
    # zeroed background pulls the whole-frame mean below 32 (the level at
    # which the v0.1.3 whole-frame proxy called the scene dark)
    f0 = sinus_frames[0][1].astype(np.float32)
    crop = f0[y0:y1, x0:x1]
    base = float(np.mean(0.114 * crop[..., 0] + 0.587 * crop[..., 1] +
                         0.299 * crop[..., 2]))
    dim = min(1.0, 75.0 / base)

    def dark_bg(f):
        g = (f.astype(np.float32) * dim)
        m = np.zeros(g.shape[:2], np.float32)
        m[y0:y1, x0:x1] = 1.0
        return (g * m[..., None]).astype(np.uint8)

    frames = [(t, dark_bg(f)) for t, f in sinus_frames[:420]]
    lumas = [float(np.mean(f)) for _, f in frames[:60]]
    assert np.mean(lumas) < 32.0, \
        f"fixture not discriminating: frame luma {np.mean(lumas):.0f}"
    s = ScanSession("rl1", str(tmp_path), fps_hint=30.0)
    fb = _feed(s, frames)
    assert fb["readiness"]["checks"]["lighting"]["pass"] is True, \
        (fb.get("lux_proxy"), fb["readiness"]["checks"]["lighting"])
    assert fb["readiness"]["checks"]["face"]["pass"] is True


def test_single_eval_blip_is_debounced_persistent_failure_is_not(tmp_path):
    """A check must fail two consecutive evaluations (1 s) to break the
    hold: estimator flicker on one 0.5 s evaluation must not throw away
    3 s of accumulated readiness. Persistent problems still do."""
    s = ScanSession("rb1", str(tmp_path), fps_hint=30.0)
    assert s._effective_failing(["sqi"]) == []            # first blip: forgiven
    assert s._effective_failing(["sqi"]) == ["sqi"]       # persists: fails
    assert s._effective_failing([]) == []
    assert s._effective_failing(["sqi"]) == []            # streak was reset
    assert s._effective_failing(["sqi", "motion"]) == ["sqi"]
    assert s._effective_failing(["motion"]) == ["motion"]


def test_readiness_two_region_verified_passes_coherence_check():
    """The user's real blocked attempt (screenshot 2026-08-25): 2 SNR
    ROIs, coherence 0.00 (structural — the metric counts only >=3-ROI
    beats), beat timing 27 ms, beats 5, SQI 0.40. That is a verifiable
    signal: two regions placing the same beats within the AF-grade
    timing budget. Readiness must pass coherence via the two-region
    mode; the FP timing profile (36 ms) must still fail."""
    cfg = load_config()
    ev = {"insufficient": False, "sqi": 0.40, "components": {},
          "per_roi_snr": {"forehead": 0.7, "cheek_l": 0.6,
                          "cheek_r": 0.2, "nose": 0.1},
          "cross_roi_coherence": 0.0, "timing_precision_ms": 27.0,
          "timing_matched_fraction": 0.85, "timing_pair": None,
          "n_beats": 5, "frac_multi_roi": 0.0, "split_fraction": 0.0,
          "harmonic_fraction": 0.0, "n_intervals": 4, "n_segments": 1,
          "fps": 30.0, "max_gap_ms": 33.3, "jitter_ms": 1.0,
          "window_coverage": 1.0, "seconds": 8.0, "n_frames": 240}
    rd = readiness_from_evidence(ev, cfg)
    c = rd["checks"]["cross_roi_coherence"]
    assert c["pass"] is True, c
    assert c.get("mode") == "two_region_verified"
    assert rd["ready"] is True, rd["failing"]
    ev2 = {**ev, "timing_precision_ms": 36.0,
           "timing_matched_fraction": 0.75}
    rd2 = readiness_from_evidence(ev2, cfg)
    assert rd2["checks"]["cross_roi_coherence"]["pass"] is False


def test_framing_allows_close_face_while_rois_intact(tmp_path, sinus_frames):
    """The same real attempt was ALSO blocked by 'Move back a little':
    face width > 0.70 of frame was a hard veto with no post-scan
    counterpart (tracking 0.98, beats present). Framing must veto only
    when ROI geometry actually breaks; a close, centred face with every
    ROI keeping its pixels is fine."""
    h, w = sinus_frames[0][1].shape[:2]
    c = 0.62                                   # zoom: face width ~0.71
    x0, y0 = int(w * (1 - c) / 2), int(h * (1 - c) / 2)
    zoom = [(t, cv2.resize(f[y0:y0 + int(h * c), x0:x0 + int(w * c)],
                           (w, h), interpolation=cv2.INTER_LINEAR))
            for t, f in sinus_frames[:420]]
    s = ScanSession("rf1", str(tmp_path / "rf1"), fps_hint=30.0)
    fb = _feed(s, zoom)
    assert fb["face_box"] is not None and fb["face_box"][2] > 0.70, \
        ("fixture not discriminating", fb["face_box"])
    assert fb["readiness"]["checks"]["framing"]["pass"] is True, \
        fb["readiness"]["checks"]["framing"]


def test_roi_integrity_math():
    """Unit: fraction of each ROI's nominal area inside the frame. A
    centred face scores 1.0 everywhere; a face high in the frame (real
    laptop close-ups via the landmark tracker) loses the forehead."""
    from preprocessing.roi import roi_integrity
    from capture.face_tracking import FaceObservation
    ok = roi_integrity(FaceObservation(found=True, cx=160, cy=120,
                                       ax=70, ay=96), (240, 320))
    assert all(v == 1.0 for v in ok.values()), ok
    high = roi_integrity(FaceObservation(found=True, cx=160, cy=40,
                                         ax=70, ay=96), (240, 320))
    assert high["forehead"] < 0.5, high
    assert high["cheek_l"] > 0.9 and high["cheek_r"] > 0.9, high


def test_readiness_log_persists_per_eval_evidence(tmp_path, sinus_frames):
    """A session that never becomes ready must still leave evidence of WHY
    (the Aug-20/21 sessions left empty directories — undiagnosable)."""
    import json as _json
    s = ScanSession("rt1", str(tmp_path / "rt1"), fps_hint=30.0)
    _feed(s, sinus_frames[:420])
    s.start_scan()
    log = tmp_path / "rt1" / "readiness_log.jsonl"
    assert log.exists(), "readiness_log.jsonl missing"
    lines = [_json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    evals = [x for x in lines if x.get("kind") == "eval"]
    events = [x for x in lines if x.get("kind") == "event"]
    assert len(evals) >= 5
    for e in evals[-3:]:
        assert "ready" in e and "failing" in e and "checks" in e
        assert "ev" in e and "state" in e
    assert any(e.get("event") == "start_scan" for e in events)
