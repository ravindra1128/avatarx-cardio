"""
Live-demo support in the capture layer (v0.1.1):

* YuNet face detector backend (OpenCV zoo ONNX, Apache-2.0, vendored) with
  5 landmarks -> landmark-anchored ROIs on REAL faces;
* temporal smoothing of the tracked geometry (raw observations still feed
  the stability metric);
* honest per-frame timestamps from a sidecar file (browser-captured video
  is written at a nominal fps; the true capture clock lives beside it);
* an explicit `capture_profile="consumer"` in ingest that records the
  auto-exposure state truthfully and reports it as a caveat instead of
  failing the scan (documented deviation for the live demo; the research
  profile is unchanged).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from capture.face_tracking import FaceTracker, FaceObservation, yunet_available
from capture.video_reader import probe_video, iter_frames
from capture.ingest import ingest_video
from preprocessing.roi import roi_bounds_px, ROI_NAMES
from scripts.make_synth_video import synth_video, portrait_available, \
    synth_portrait_video

REPO = pathlib.Path(__file__).resolve().parents[1]


# ------------------------------------------------------------- YuNet
@pytest.mark.skipif(not yunet_available(), reason="YuNet model not vendored")
@pytest.mark.skipif(not portrait_available(), reason="no portrait sample")
def test_yunet_finds_real_face_with_landmarks():
    from scripts.make_synth_video import portrait_path
    img = cv2.imread(portrait_path())
    tr = FaceTracker()
    obs = tr.process(img)
    assert obs.found and tr.tracker == "yunet"
    assert obs.landmarks is not None and obs.landmarks.shape == (5, 2)
    # geometry sanity: eyes above nose above mouth, eyes left-right ordered
    lm = obs.landmarks
    assert lm[0, 1] < lm[2, 1] < lm[3, 1]
    assert lm[0, 0] < lm[1, 0]
    b = roi_bounds_px(obs, img.shape)
    x0, x1, y0, y1 = b["forehead"]
    assert y1 < lm[0, 1]                          # forehead above the eyes
    for c in ("cheek_l", "cheek_r"):
        cx0, cx1, cy0, cy1 = b[c]
        assert cy0 > lm[0, 1] and cy1 < lm[3, 1] + 5   # between eyes and mouth
    assert b["cheek_l"][1] < lm[2, 0] < b["cheek_r"][0]  # nose between cheeks


def test_no_face_on_blank_frame():
    tr = FaceTracker()
    obs = tr.process(np.full((240, 320, 3), 40, np.uint8))
    assert not obs.found


def test_smoothing_keeps_raw_for_stability_and_smooths_geometry():
    tr = FaceTracker(smooth_alpha=0.3)
    rng = np.random.default_rng(0)
    for i in range(30):
        cx = 160 + (5 if i % 2 else -5)                 # 10 px alternating jitter
        tr._locked = None
        tr._inject_for_test(FaceObservation(True, cx, 120, 40, 55, 1.0))
    smoothed = [o.cx for o in tr.smoothed_observations[-10:]]
    raw = [o.cx for o in tr.observations[-10:]]
    assert np.std(smoothed) < 0.5 * np.std(raw)
    s = tr.summary(30.0)
    assert s.stability < 0.5                            # judged on RAW jitter


# --------------------------------------------------------- timestamps
def test_timestamp_sidecar_overrides_container_clock(tmp_path):
    p = str(tmp_path / "v.avi")
    synth_video(p, kind="sinus", fps=30.0, duration_s=4.0, seed=1)
    meta0 = probe_video(p)
    n = meta0.n_frames
    # pretend the true clock ran at 25 fps with jitter
    ts = (np.arange(n) / 25.0 + np.random.default_rng(0).normal(0, 0.002, n))
    ts = np.sort(ts)
    with open(p + ".timestamps.json", "w") as f:
        json.dump({"timestamps_s": ts.tolist()}, f)
    meta = probe_video(p)
    assert abs(meta.measured_fps_mean - 25.0) < 0.5
    assert meta.measured_fps_jitter_ms > 1.0
    got = np.array([t for t, _ in iter_frames(p)])
    assert np.allclose(got, ts)


# ------------------------------------------------- consumer capture profile
def test_consumer_profile_reports_exposure_caveat_instead_of_failing(tmp_path):
    p = str(tmp_path / "v.avi")
    synth_video(p, kind="sinus", fps=30.0, duration_s=6.0, seed=2)
    strict = ingest_video(p, assume_rig_locks=False)
    assert not strict.ok and any("exposure" in w for w in strict.reasons)
    cons = ingest_video(p, assume_rig_locks=False, capture_profile="consumer")
    assert cons.ok, cons.reasons
    assert any("exposure" in w for w in cons.capture_caveats)
    assert cons.capture.exposure_locked is False        # recorded truthfully


def test_consumer_profile_still_fails_closed_on_darkness(tmp_path):
    p = str(tmp_path / "dark.avi")
    synth_video(p, kind="sinus", fps=30.0, duration_s=6.0, seed=3, lux_scale=0.15)
    cons = ingest_video(p, assume_rig_locks=False, capture_profile="consumer")
    assert not cons.ok and any("illuminance" in w for w in cons.reasons)


def test_consumer_profile_fps_tolerance_with_caveat(tmp_path):
    """A 25 fps webcam (below the 30 fps research floor, above the 24 fps
    consumer floor) passes with a caveat; 20 fps still fails closed."""
    p25 = str(tmp_path / "f25.avi")
    synth_video(p25, kind="sinus", fps=25.0, duration_s=6.0, seed=5)
    r25 = ingest_video(p25, assume_rig_locks=False, capture_profile="consumer")
    assert r25.ok, r25.reasons
    assert any("fps" in c for c in r25.capture_caveats)
    # research profile rejects the same capture
    assert not ingest_video(p25, assume_rig_locks=True).ok
    p20 = str(tmp_path / "f20.avi")
    synth_video(p20, kind="sinus", fps=20.0, duration_s=6.0, seed=6)
    r20 = ingest_video(p20, assume_rig_locks=False, capture_profile="consumer")
    assert not r20.ok and any("fps below" in w for w in r20.reasons)


# ------------------------------------------------------ portrait synth
@pytest.mark.skipif(not (yunet_available() and portrait_available()),
                    reason="needs YuNet + portrait sample")
def test_portrait_video_tracks_with_yunet_and_recovers_pulse(tmp_path):
    from preprocessing.roi import ROI_NAMES
    from rppg.pos import pos_pulse
    from beats.detector import detect_beats_single_roi, fuse_multi_roi
    from evaluation.beat_metrics import match_beats
    p = str(tmp_path / "portrait.avi")
    truth = synth_portrait_video(p, kind="sinus", fps=30.0, duration_s=15.0,
                                 seed=4)
    r = ingest_video(p, capture_profile="consumer")
    assert r.ok, r.reasons
    assert r.track.tracker == "yunet"
    fps = r.meta.measured_fps_mean
    per = {roi: detect_beats_single_roi(pos_pulse(r.traces[roi], fps), fps, roi)
           for roi in ROI_NAMES}
    s = fuse_multi_roi(per, fps, float(r.timestamps_s[-1]), min_rois=2)
    m = match_beats(np.asarray(truth["peak_times_s"]), s.times(), tolerance_ms=50)
    assert m.f1 >= 0.85, (m.f1, m.sensitivity, m.ppv)
