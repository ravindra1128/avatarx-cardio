"""Regression matrix for realistic smartphone capture failures.

These are interface/robustness proofs on synthetic imagery, not biomedical
accuracy claims.  Every failure must either recover without contaminating the
trace or end in an explicit REPEAT_SCAN/NO_RESULT path; clean evidence must keep
its accepted result and three labelled prototype biomarkers.
"""
import copy
import json
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.report_data import report_biomarkers
from app.scan_engine import ScanSession, SessionState
from capture.face_tracking import (FaceObservation, FaceTracker, TrackSummary,
                                   _SkinBackend)
from capture.ingest import ingest_video
from capture.video_reader import iter_frames, probe_video
from configs import load_config
from evaluation.beat_metrics import match_beats
from features.hemodynamics import resting_hemodynamics
from inference.evidence import readiness_from_evidence, window_evidence
from inference.pipeline import run_with_details
from preprocessing.roi import ROI_NAMES, mean_rgb, roi_bounds_px, roi_photometry
from rppg.pos import pos_pulse
from beats.detector import detect_beats_single_roi, fuse_multi_roi
from scripts.make_synth_video import synth_video


@pytest.fixture(scope="module")
def clean_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("realworld")
    path = str(d / "clean.avi")
    truth = synth_video(path, kind="sinus", fps=30.0, duration_s=42.0,
                        seed=93)
    frames = list(iter_frames(path))
    ing = ingest_video(path, capture_profile="consumer",
                       assume_rig_locks=False)
    assert ing.ok, ing.reasons
    return path, truth, frames, ing


@pytest.fixture(scope="module")
def clean_pipeline(clean_clip):
    path, _, _, _ = clean_clip
    return run_with_details(
        path, manifest={"capture_profile": "consumer",
                        "assume_rig_locks": False},
        recording_id="realworld-clean")


def _feed(session, frames, chunk=15):
    fb = None
    for i in range(0, len(frames), chunk):
        b = frames[i:i + chunk]
        fb = session.push_frames([f for _, f in b], [t for t, _ in b])
    return fb


class _FixedTracker:
    """Deterministic geometry for isolating photometric gate tests."""
    def __init__(self, obs):
        self.obs = obs
        self.observations = []

    @property
    def tracker(self):
        return "test_fixed_geometry"

    def process(self, _frame):
        self.observations.append(self.obs)
        return self.obs

    def summary(self, _fps):
        n = len(self.observations)
        return TrackSummary(n, n, 0.0, 1.0, self.tracker)


# ------------------------------- timestamps, drops and duplicated delivery
def test_timestamp_sidecar_length_must_match_video_exactly(tmp_path):
    p = str(tmp_path / "clock.avi")
    synth_video(p, duration_s=2.0, fps=30.0, seed=1)
    n = probe_video(p).n_frames
    for count, message in ((n + 1, "longer"), (n - 1, "shorter")):
        pathlib.Path(p + ".timestamps.json").write_text(json.dumps(
            {"timestamps_s": (np.arange(count) / 30.0).tolist()}))
        with pytest.raises(IOError, match=message):
            probe_video(p)


def test_repeated_frames_fail_capture_visibly(tmp_path):
    p = str(tmp_path / "frozen.avi")
    frame = np.full((240, 320, 3), 120, np.uint8)
    cv2.ellipse(frame, (160, 120), (70, 96), 0, 0, 360,
                (70, 105, 145), -1)
    writer = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"FFV1"), 30.0,
                             (320, 240))
    for _ in range(120):
        writer.write(frame)
    writer.release()
    meta = probe_video(p)
    assert meta.duplicate_frame_fraction > 0.95
    out = ingest_video(p, capture_profile="consumer",
                       assume_rig_locks=False)
    assert not out.ok
    assert any("duplicated video frames" in r for r in out.reasons)
    result, det = run_with_details(
        p, manifest={"capture_profile": "consumer",
                     "assume_rig_locks": False})
    assert result.outcome.value == "NO_RESULT"
    assert result.capture_meta["duplicate_frame_fraction"] > 0.95
    payload = report_biomarkers(result, det)
    assert all(x["value"] is None and x["reason"] for x in payload["items"])


def test_live_frozen_frames_are_skipped_and_pause_capture(tmp_path):
    obs = FaceObservation(True, 160, 120, 70, 96, 1.0,
                          box=(90, 24, 140, 192))
    s = ScanSession("frozen-live", str(tmp_path / "frozen-live"),
                    fps_hint=30.0, scan_seconds=8.0)
    s._tracker = _FixedTracker(obs)
    s.start_scan(force=True)
    frame = np.full((240, 320, 3), (90, 135, 175), np.uint8)
    fb = s.push_frames([frame.copy() for _ in range(8)],
                       (np.arange(8) / 30.0).tolist())
    assert fb["duplicate_frames"] == 7
    assert fb["paused"] is True and fb["disposition"] == "REPEAT_SCAN"
    assert len(s._scan_ts) == 1                         # repeats never recorded
    s.abort()


def test_overexposed_file_fails_before_signal_extraction(tmp_path):
    p = str(tmp_path / "clipped.avi")
    writer = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"FFV1"), 30.0,
                             (320, 240))
    for i in range(90):
        frame = np.full((240, 320, 3), 255, np.uint8)
        frame[0:8, 0:8] = (i % 200, i % 170, i % 140)
        writer.write(frame)
    writer.release()
    out = ingest_video(p, capture_profile="consumer",
                       assume_rig_locks=False)
    assert not out.ok
    assert any("overexposed" in r for r in out.reasons)


def test_nonmonotone_and_mismatched_inputs_fail_closed(clean_clip):
    _, _, _, ing = clean_clip
    cfg = load_config()
    traces = {r: ing.traces[r][:240] for r in ROI_NAMES}
    ts = ing.timestamps_s[:240].copy()
    ts[100] = ts[99]
    ev = window_evidence(traces, ts, 30.0, cfg)
    assert ev["insufficient"] is True
    assert "timestamp" in ev["insufficient_reason"]
    assert readiness_from_evidence(ev, cfg)["checks"]["timestamps"]["pass"] \
        is False
    bad = dict(traces)
    bad["nose"] = bad["nose"][:-1]
    ev2 = window_evidence(bad, ing.timestamps_s[:240], 30.0, cfg)
    assert ev2["insufficient"] is True
    assert "wrong length" in ev2["insufficient_reason"]


# -------------------------------- appearance/occlusion without colour priors
def test_robust_roi_mean_tolerates_glasses_hair_and_highlights():
    obs = FaceObservation(True, 160, 120, 70, 96, 1.0,
                          box=(90, 24, 140, 192))
    frame = np.full((240, 320, 3), (90, 135, 175), np.uint8)
    x0, x1, y0, y1 = roi_bounds_px(obs, frame.shape)["cheek_l"]
    patch = frame[y0:y1, x0:x1]
    patch[:, :max(1, patch.shape[1] // 8)] = 0       # dark glasses/hair edge
    patch[:, -max(1, patch.shape[1] // 16):] = 255   # specular highlight
    rgb = mean_rgb(frame, obs)["cheek_l"]
    assert np.allclose(rgb, [175, 135, 90], atol=3.0), rgb
    photo = roi_photometry(frame, obs)
    assert photo["usable_rois"] >= 3                  # one bad area is tolerated


def test_dark_skin_fallback_localizer_has_no_absolute_red_floor():
    frame = np.full((240, 320, 3), 120, np.uint8)
    # BGR counterpart of the darkest synthetic Fitzpatrick optical fixture.
    cv2.ellipse(frame, (160, 120), (70, 96), 0, 0, 360,
                (40, 58, 85), -1)
    obs = _SkinBackend().detect(frame)
    assert obs.found and obs.ax > 50 and obs.ay > 70


@pytest.mark.parametrize("skin_rgb", [
    (225.0, 190.0, 160.0), (180.0, 140.0, 95.0), (85.0, 58.0, 40.0)])
def test_clean_pulse_recovery_across_synthetic_skin_reflectance(
        tmp_path, skin_rgb):
    p = str(tmp_path / ("skin-" + str(int(skin_rgb[0])) + ".avi"))
    truth = synth_video(p, kind="sinus", fps=30.0, duration_s=14.0,
                        seed=18, skin_rgb=skin_rgb)
    ing = ingest_video(p, capture_profile="consumer",
                       assume_rig_locks=False)
    assert ing.ok, ing.reasons
    per = {r: detect_beats_single_roi(pos_pulse(ing.traces[r], 30.0), 30.0, r)
           for r in ROI_NAMES}
    fused = fuse_multi_roi(per, 30.0, float(ing.timestamps_s[-1]), min_rois=2)
    score = match_beats(np.asarray(truth["peak_times_s"]), fused.times(), 50)
    assert score.f1 >= 0.80, (skin_rgb, score)


def test_one_occluded_roi_does_not_poison_three_clean_regions(clean_clip):
    _, _, _, ing = clean_clip
    cfg = load_config()
    n = 360
    traces = {r: ing.traces[r][:n].copy() for r in ROI_NAMES}
    clean = window_evidence(traces, ing.timestamps_s[:n], 30.0, cfg,
                            tracking_stability=1.0)
    rng = np.random.default_rng(9)
    traces["forehead"][:] = 20.0 + rng.normal(0, 0.5, traces["forehead"].shape)
    covered = window_evidence(traces, ing.timestamps_s[:n], 30.0, cfg,
                              tracking_stability=1.0)
    rd = readiness_from_evidence(covered, cfg)
    assert rd["checks"]["signal_snr"]["pass"] is True
    assert rd["checks"]["cross_roi_coherence"]["pass"] is True
    assert covered["n_beats"] >= clean["n_beats"] - 2
    assert covered["sqi"] >= 0.70 * clean["sqi"]


# ----------------------------------------- tracking, framing and illumination
def test_tracker_resets_stale_geometry_after_occlusion():
    tr = FaceTracker(smooth_alpha=0.2)
    tr._inject_for_test(FaceObservation(True, 100, 100, 50, 70, 1.0))
    for _ in range(3):
        tr._inject_for_test(FaceObservation(False))
    reacquired = tr._inject_for_test(
        FaceObservation(True, 210, 105, 50, 70, 1.0))
    assert reacquired.cx == pytest.approx(210.0)          # no stale EMA drag
    assert tr.summary(30.0).stability < 0.6              # gaps remain visible


@pytest.mark.parametrize("kind", ["overexposed", "uneven"])
def test_live_photometric_faults_are_actionable(kind, tmp_path):
    obs = FaceObservation(True, 160, 120, 70, 96, 1.0,
                          box=(90, 24, 140, 192))
    s = ScanSession("photo-" + kind, str(tmp_path / kind), fps_hint=30.0)
    s._tracker = _FixedTracker(obs)
    frames = []
    for i in range(45):
        if kind == "overexposed":
            f = np.full((240, 320, 3), 255, np.uint8)
        else:
            f = np.full((240, 320, 3), 120, np.uint8)
            f[:, :160] = 55
            f[:, 160:] = 220
        f[0, 0] = (i % 17, i % 13, i % 11)  # not a frozen transport frame
        frames.append(f)
    fb = s.push_frames(frames, (np.arange(len(frames)) / 30.0).tolist())
    assert fb["lighting"] == kind
    assert fb["readiness"]["checks"]["lighting"]["pass"] is False
    assert fb["disposition"] == "REPEAT_SCAN"
    assert ("light" in fb["readiness"]["hint"].lower() or
            "expos" in fb["readiness"]["hint"].lower())


def test_partial_face_is_rejected_before_roi_sampling(tmp_path):
    obs = FaceObservation(True, 20, 120, 70, 96, 1.0,
                          box=(-50, 24, 140, 192))
    s = ScanSession("partial", str(tmp_path / "partial"), fps_hint=30.0)
    s._tracker = _FixedTracker(obs)
    frames = []
    for i in range(20):
        f = np.full((240, 320, 3), 120, np.uint8)
        f[0, 0] = i
        frames.append(f)
    fb = s.push_frames(frames, (np.arange(20) / 30.0).tolist())
    assert fb["framing"] in ("off_centre", "too_close")
    assert fb["ready"] is False and fb["disposition"] == "REPEAT_SCAN"
    assert len(s._trace_t) == 0


def test_face_too_far_is_actionable_and_not_sampled(tmp_path):
    obs = FaceObservation(True, 160, 120, 18, 25, 1.0,
                          box=(142, 95, 36, 50))
    s = ScanSession("far", str(tmp_path / "far"), fps_hint=30.0)
    s._tracker = _FixedTracker(obs)
    frames = []
    for i in range(20):
        f = np.full((240, 320, 3), 120, np.uint8)
        f[0, 0] = i
        frames.append(f)
    fb = s.push_frames(frames, (np.arange(20) / 30.0).tolist())
    assert fb["framing"] == "too_far"
    assert "closer" in fb["readiness"]["hint"].lower()
    assert fb["disposition"] == "REPEAT_SCAN" and len(s._trace_t) == 0


def test_head_motion_pauses_advisory_recording(clean_clip, tmp_path):
    _, _, frames, _ = clean_clip
    s = ScanSession("motion", str(tmp_path / "motion"), fps_hint=30.0,
                    scan_seconds=8.0)
    _feed(s, frames[:420])
    assert s.status()["feedback"]["ready"]
    s.start_scan()
    moved = []
    for i, (t, f) in enumerate(frames[420:540]):
        dx = 42 if i % 2 else -42
        moved.append((t, cv2.warpAffine(
            f, np.float32([[1, 0, dx], [0, 1, 0]]),
            (f.shape[1], f.shape[0]))))
    fb = _feed(s, moved)
    assert fb["motion"] == "moving"
    assert fb["paused"] is True and fb["disposition"] == "REPEAT_SCAN"
    assert fb["progress"] < 0.30


def test_weak_signal_guides_without_blocking_full_capture(clean_clip, tmp_path):
    _, _, frames, _ = clean_clip
    s = ScanSession("recover", str(tmp_path / "recover"), fps_hint=30.0,
                    scan_seconds=20.0)
    _feed(s, frames[:420])
    s.start_scan()
    rng = np.random.default_rng(7)
    weak = []
    for i in range(300):
        f = np.full((240, 320, 3), 120.0)
        cv2.ellipse(f, (160, 120), (70, 96), 0, 0, 360,
                    (95, 140, 180), -1)
        f += rng.normal(0, 2, f.shape)
        weak.append((14.0 + i / 30.0, np.clip(f, 0, 255).astype(np.uint8)))
    fb = _feed(s, weak)
    assert fb["paused"] is False and fb["disposition"] == "READY"
    assert any(k in fb["readiness"]["failing"] for k in
               ("signal_snr", "cross_roi_coherence", "prelim_beats", "sqi"))
    assert fb["progress"] > 0.40
    returned = [(t + 10.0, f) for t, f in frames[420:960]]
    fb = _feed(s, returned)
    assert s.state is SessionState.SCANNING
    assert fb["paused"] is False and fb["disposition"] == "READY"
    assert fb["progress"] >= 0.99


def test_short_or_interrupted_scan_cannot_be_finalized(clean_clip, tmp_path):
    _, _, frames, _ = clean_clip
    s = ScanSession("short", str(tmp_path / "short"), fps_hint=30.0,
                    scan_seconds=8.0)
    _feed(s, frames[:420])
    s.start_scan()
    _feed(s, frames[420:450])
    with pytest.raises(RuntimeError, match="scan incomplete"):
        s.finish_scan()
    fb = s.status()["feedback"]
    assert fb["disposition"] == "REPEAT_SCAN"
    assert "more seconds" in fb["hint"]


# --------------------------------------- end-to-end outcome and biomarkers
def test_clean_signal_performance_and_all_biomarkers_are_preserved(
        clean_pipeline, clean_clip):
    result, det = clean_pipeline
    _, truth, _, _ = clean_clip
    assert result.outcome.value == "ACCEPT", result.no_read_reasons
    assert result.predicted_class == "SINUS"
    beat_score = match_beats(np.asarray(truth["peak_times_s"]),
                             det["fused"].times(), 50)
    assert beat_score.f1 >= 0.85
    assert det["sqi"].sqi >= 0.30
    hemo = resting_hemodynamics(
        det, outcome=result.outcome.value,
        capture={"exposure_locked": True, "awb_locked": True})
    payload = report_biomarkers(result, det, hemodynamics=hemo)
    assert payload["complete"] is True
    assert all(x["status"] == "computed"
               and (x["value"] is not None or x.get("band") is not None)
               for x in payload["items"])
    assert all(x["label"] == "Research Estimate / Prototype"
               for x in payload["items"])


def test_unverified_pulse_never_forces_biomarker_values(clean_pipeline):
    result, det = clean_pipeline
    bad = dict(det)
    bad["sqi"] = SimpleNamespace(sqi=0.05, components={})
    bad["evidence"] = dict(det["evidence"], cross_roi_coherence=0.0,
                           timing_precision_ms=90.0,
                           timing_matched_fraction=0.2)
    bad["ingest"] = copy.copy(det["ingest"])
    bad["ingest"].track = copy.copy(det["ingest"].track)
    bad["ingest"].track.stability = 0.2
    hemo = resting_hemodynamics(bad, outcome="NO_RESULT")
    # Display tiers (2026-09-09): the worst evidence still yields scores, but
    # never a MEASURED one, and every card says why it is provisional.
    assert hemo["available"] is True and hemo["tier"] == "provisional"
    assert len(hemo["tier_reasons"]) >= 3          # sqi, tracking, evidence
    payload = report_biomarkers(
        SimpleNamespace(outcome=SimpleNamespace(value="NO_RESULT")), bad,
        hemodynamics=hemo)
    for x in payload["items"]:
        if x["status"] == "computed":
            assert x["tier"] == "provisional" and x["tier_reasons"]
            # A card reports a 0-100 score or a band (owner, 2026-09-10);
            # the stiffness card is always a band.
            if x["value"] is not None:
                assert 0.0 <= float(x["value"]) <= 100.0
            else:
                assert x["band"] in ("High", "Typical", "Low")
        else:
            assert x["value"] is None and x["band"] is None and x["reason"]
