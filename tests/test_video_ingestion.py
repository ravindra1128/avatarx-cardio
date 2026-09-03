"""
T3 — real-video ingestion: reader -> face tracking -> ROI traces -> POS/CHROM
-> the EXISTING beat detector.

All numbers here are INTERFACE PROOFS on synthetic video (drawn face,
known pulse), not accuracy claims — the README says the same.

Fail-closed ordering under test: capture validity (codec/fps/lux) is
evaluated BEFORE face tracking, so a dark video fails with the illuminance
reason rather than a confusing tracking error; a bright video with no face
fails with the face reason.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from scripts.make_synth_video import synth_video
from capture.video_reader import probe_video, iter_frames, build_capture_config
from capture.ingest import ingest_video
from preprocessing.roi import ROI_NAMES
from rppg.pos import pos_pulse
from rppg.chrom import chrom_pulse
from beats.detector import detect_beats_single_roi, fuse_multi_roi
from evaluation.beat_metrics import match_beats


# ------------------------------------------------- fixtures (videos: conftest)
@pytest.fixture(scope="session")
def ingested(videos):
    cache = {}
    def get(name):
        if name not in cache:
            cache[name] = ingest_video(videos[name][0])
        return cache[name]
    return get


# ------------------------------------------------------------------ reader
def test_reader_measures_fps_and_codec(videos):
    path, truth = videos["sinus30"]
    meta = probe_video(path)
    assert abs(meta.measured_fps_mean - 30.0) < 0.5
    assert meta.width == 320 and meta.height == 240
    assert meta.n_frames >= 19 * 30
    assert meta.codec_fourcc.lower() in ("ffv1", "mjpg")
    ts = [t for t, _ in iter_frames(path)]
    assert len(ts) == meta.n_frames
    assert np.all(np.diff(ts) > 0)                       # monotone timestamps


def test_capture_config_from_bright_lossless_video_passes_gate(videos):
    meta = probe_video(videos["sinus30"][0])
    cfg = build_capture_config(meta)
    ok, why = cfg.is_valid_for_beat_analysis()
    assert ok, why
    assert cfg.codec != "unknown"


def test_dark_video_capture_config_fails_on_lux(videos):
    meta = probe_video(videos["dark30"][0])
    cfg = build_capture_config(meta)
    ok, why = cfg.is_valid_for_beat_analysis()
    assert not ok
    assert any("illuminance" in w for w in why), why


# ------------------------------------------------------------------ ingest
def test_ingest_finds_synthetic_face_and_records_tracker(ingested):
    r = ingested("sinus30")
    assert r.ok, r.reasons
    assert r.track.tracker in ("mediapipe", "opencv_haar", "skin_segmentation")
    assert r.track.n_found >= 0.95 * r.track.n_frames
    assert r.track.stability > 0.9


def test_ingest_no_face_aborts_with_face_reason(ingested):
    r = ingested("noface30")
    assert not r.ok
    assert any("face" in w.lower() for w in r.reasons), r.reasons


def test_ingest_dark_video_fails_closed_on_capture_not_tracking(ingested):
    r = ingested("dark30")
    assert not r.ok
    assert any("illuminance" in w for w in r.reasons), r.reasons


def test_ingest_lux_defers_to_face_region_when_scene_is_dark(videos, tmp_path):
    """v0.1.4: the lux proxy exists to guarantee photons on the SKIN. A
    well-lit face against a dark background (whole-frame luma < 32/255)
    is analysable and must pass — measured, the same fix as the live
    gate's lighting check. A face that is itself dark still fails with
    the illuminance reason (fail-closed ordering preserved)."""
    from capture.face_tracking import FaceTracker
    src, _ = videos["sinus30"]
    tr = FaceTracker()
    obs = None
    for _, f in iter_frames(src):
        o = tr.process(f)
        if o.found:
            obs = o
            break
    assert obs is not None
    bx = obs.box or (obs.cx - obs.ax, obs.cy - obs.ay, 2 * obs.ax, 2 * obs.ay)
    x0, y0 = max(0, int(bx[0])), max(0, int(bx[1]))
    x1, y1 = int(bx[0] + bx[2]), int(bx[1] + bx[3])
    p = str(tmp_path / "darkbg.avi")
    vw = None
    lumas = []
    dim = None
    for t, f in iter_frames(src):
        if dim is None:
            crop = f[y0:y1, x0:x1].astype(np.float32)
            base = float(np.mean(0.114 * crop[..., 0] + 0.587 * crop[..., 1]
                                 + 0.299 * crop[..., 2]))
            dim = min(1.0, 75.0 / base)
        g = f.astype(np.float32) * dim
        m = np.zeros(g.shape[:2], np.float32)
        m[y0:y1, x0:x1] = 1.0
        g = (g * m[..., None]).astype(np.uint8)
        lumas.append(float(np.mean(g)))
        if vw is None:
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"FFV1"), 30.0,
                                 (g.shape[1], g.shape[0]))
        vw.write(g)
    vw.release()
    assert np.mean(lumas) < 32.0, "fixture not discriminating"
    r = ingest_video(p, capture_profile="consumer", assume_rig_locks=False)
    assert r.ok, r.reasons
    assert any("face" in c.lower() and "illuminance" in c.lower()
               for c in r.capture_caveats), r.capture_caveats


def test_roi_traces_are_separate_channels(ingested):
    r = ingested("sinus30")
    assert set(r.traces) == set(ROI_NAMES)
    n = r.timestamps_s.size
    for roi, tr in r.traces.items():
        assert tr.shape == (n, 3)
        assert np.all(np.isfinite(tr))
    names = list(ROI_NAMES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.array_equal(r.traces[names[i]], r.traces[names[j]])


# ---------------------------------------------------------------- rPPG unit
def _synth_rgb_trace(fps=30.0, n_beats=24, noise=0.5, seed=3):
    rng = np.random.default_rng(seed)
    rr = np.clip(rng.normal(0.85, 0.03, n_beats), 0.5, 1.4)
    onsets = np.concatenate([[0.0], np.cumsum(rr)])
    n = int(onsets[-1] * fps) + 1
    pulse = np.zeros(n)
    for a, b in zip(onsets, onsets[1:]):
        i0, i1 = int(a * fps), int(b * fps)
        if i1 - i0 > 2:
            t = np.linspace(0, 1, i1 - i0, endpoint=False)
            pulse[i0:i1] = np.exp(-((t - 0.18) ** 2) / (2 * 0.075 ** 2))
    drift = 1.0 + 0.05 * np.sin(2 * np.pi * 0.1 * np.arange(n) / fps)
    base = np.array([180.0, 140.0, 95.0])                # RGB skin
    gains = np.array([0.3, 1.0, 0.6]) * 3.0
    rgb = (base[None, :] - pulse[:, None] * gains[None, :]) * drift[:, None]
    rgb += rng.normal(0, noise, rgb.shape)
    return rgb, pulse


@pytest.mark.parametrize("extractor", [pos_pulse, chrom_pulse])
def test_extractor_recovers_pulse_and_ignores_illumination(extractor):
    rgb, pulse = _synth_rgb_trace()
    out = extractor(rgb, 30.0)
    assert out.shape == (rgb.shape[0],)
    zp = (pulse - pulse.mean()) / pulse.std()
    zo = (out - out.mean()) / (out.std() + 1e-12)
    assert np.corrcoef(zp, zo)[0, 1] > 0.7               # peak-up orientation
    out2 = extractor(rgb * 2.0, 30.0)                    # illumination gain
    zo2 = (out2 - out2.mean()) / (out2.std() + 1e-12)
    assert np.corrcoef(zo, zo2)[0, 1] > 0.99


# ------------------------------------------------------------- end to end
@pytest.mark.parametrize("name", ["sinus30", "sinus60", "af30", "af60"])
@pytest.mark.parametrize("extractor", [pos_pulse, chrom_pulse])
def test_end_to_end_beat_f1(videos, ingested, name, extractor):
    """THE T3 acceptance: video file -> fused beats, F1@50ms >= 0.85,
    via BOTH POS and CHROM, at 30 and 60 fps, sinus and AF."""
    path, truth = videos[name]
    r = ingested(name)
    assert r.ok, r.reasons
    fps = r.meta.measured_fps_mean
    per_roi = {}
    for roi in ROI_NAMES:
        wave = extractor(r.traces[roi], fps)
        per_roi[roi] = detect_beats_single_roi(wave, fps, roi)
    series = fuse_multi_roi(per_roi, fps, float(r.timestamps_s[-1]), min_rois=2)
    m = match_beats(np.asarray(truth["peak_times_s"]), series.times(),
                    tolerance_ms=50)
    assert m.f1 >= 0.85, (name, extractor.__name__, m.f1,
                          m.sensitivity, m.ppv)
