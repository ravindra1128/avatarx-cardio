"""
T4 — composite SQI, with THE anti-periodicity test.

The open-rppg defect this module must never reproduce: an
autocorrelation-peak SQI scores clean AF 0.277 (blocked) vs clean sinus
0.838 (passed) — a quality gate that screens out exactly the patients
being screened. Our SQI measures signal PRESENCE (in-band energy, pulse
skewness, cross-ROI coherence, tracking) and never rhythm REGULARITY, so
clean AF must score at least as well as clean sinus minus 0.05. That bound
is a permanent acceptance criterion, not a tunable.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from beats.detector import detect_beats_single_roi, fuse_multi_roi
from rppg.pos import pos_pulse
from rppg.signal_quality import compute_sqi, SQI_COMPONENTS
from configs import load_config, config_hash

FPS = 60.0


# ---------------------------------------------------------------- helpers
def _pulse_train(rr_s, fps=FPS):
    onsets = np.concatenate([[0.0], np.cumsum(rr_s)])
    n = int(onsets[-1] * fps) + 1
    sig = np.zeros(n)
    for a, b in zip(onsets, onsets[1:]):
        i0, i1 = int(a * fps), int(b * fps)
        if i1 - i0 > 2:
            t = np.linspace(0, 1, i1 - i0, endpoint=False)
            sig[i0:i1] = np.exp(-((t - 0.18) ** 2) / (2 * 0.075 ** 2))
    return sig


def _roi_waveforms(base, noise, seed, motion=None):
    """Four ROI 'extracted' waveforms with matched noise level."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, roi in enumerate(("forehead", "cheek_l", "cheek_r", "nose")):
        nz = noise * (1.0 if roi == "forehead" else 1.4 if roi == "nose" else 1.15)
        w = base + rng.normal(0, nz, base.size)
        if motion is not None:
            w = w + motion
        out[roi] = w
    return out


def _sqi_for(waveforms, fps=FPS, stability=1.0):
    per_roi = {roi: detect_beats_single_roi(w, fps, roi)
               for roi, w in waveforms.items()}
    dur = len(next(iter(waveforms.values()))) / fps
    fused = fuse_multi_roi(per_roi, fps, dur, min_rois=2)
    return compute_sqi(waveforms, fps, fused, tracking_stability=stability)


def sinus_waveforms(noise=0.13, seed=5):
    rng = np.random.default_rng(seed)
    rr = np.clip(rng.normal(0.85, 0.03, 106), 0.5, 1.4)
    return _roi_waveforms(_pulse_train(rr), noise, seed + 1)


def af_waveforms(noise=0.13, seed=5):
    rng = np.random.default_rng(seed)
    rr = np.clip(rng.normal(0.62, 0.17, 145), 0.28, 1.35)
    return _roi_waveforms(_pulse_train(rr), noise, seed + 1)


# --------------------------------------------------- THE anti-periodicity
@pytest.mark.parametrize("noise", [0.10, 0.13, 0.18])
def test_clean_af_scores_at_least_clean_sinus_minus_005(noise):
    """NON-NEGOTIABLE (v0.1 invariant 6): AF is aperiodic; a quality index
    that rewards periodicity blocks the very patients being screened."""
    for seed in (5, 11):
        s = _sqi_for(sinus_waveforms(noise, seed))
        a = _sqi_for(af_waveforms(noise, seed))
        assert a.sqi >= s.sqi - 0.05, \
            f"noise {noise} seed {seed}: AF {a.sqi:.3f} vs sinus {s.sqi:.3f}"


def test_clean_signals_score_high():
    assert _sqi_for(sinus_waveforms()).sqi > 0.6
    assert _sqi_for(af_waveforms()).sqi > 0.6


# --------------------------------------------------------- low-quality < 0.3
def test_white_noise_scores_low():
    rng = np.random.default_rng(3)
    w = {roi: rng.normal(0, 1, 1200) for roi in
         ("forehead", "cheek_l", "cheek_r", "nose")}
    assert _sqi_for(w).sqi < 0.3


def test_no_pulse_scores_low():
    rng = np.random.default_rng(4)
    w = {roi: rng.normal(0, 0.02, 1200) for roi in
         ("forehead", "cheek_l", "cheek_r", "nose")}
    assert _sqi_for(w).sqi < 0.3


def test_heavy_motion_scores_low():
    """Large shared low-frequency lurches + poor tracking: even though the
    pulse is still present underneath, the scan is not analysable."""
    rng = np.random.default_rng(6)
    n = int(90 * FPS)
    lurch = np.zeros(n)
    for i in rng.integers(0, n - 60, 40):
        lurch[i:i + 60] += rng.uniform(3, 8) * np.hanning(60)
    base = _pulse_train(np.clip(rng.normal(0.85, 0.03, 106), 0.5, 1.4))
    w = _roi_waveforms(base, 0.6, 7, motion=lurch[:base.size])
    assert _sqi_for(w, stability=0.2).sqi < 0.3


# ------------------------------------------------------------- fail closed
def test_nan_waveform_fails_closed():
    """A corrupt ROI can only LOWER the composite, never raise it, and its
    own components must score zero. (Whole-scan NaN rejection happens
    upstream in ingest; gate-level NaN handling is decision logic's job.)"""
    clean = _sqi_for(sinus_waveforms())
    w = sinus_waveforms()
    w["forehead"][100:110] = np.nan
    r = _sqi_for(w)
    assert np.isfinite(r.sqi)
    assert r.sqi < clean.sqi
    assert r.per_roi["forehead"]["snr_in_band"] == 0.0
    assert r.per_roi["forehead"]["skewness"] == 0.0


def test_empty_input_scores_zero():
    r = compute_sqi({}, FPS, None, tracking_stability=1.0)
    assert r.sqi == 0.0


# ------------------------------------------------------------------ config
def test_weights_come_from_default_config():
    cfg = load_config()
    ws = cfg["sqi"]["weights"]
    assert set(ws) == set(SQI_COMPONENTS)
    assert all(v > 0 for v in ws.values())
    r = _sqi_for(sinus_waveforms())
    assert set(r.components) == set(SQI_COMPONENTS)


def test_config_hash_is_stable_and_content_sensitive():
    cfg = load_config()
    h1, h2 = config_hash(cfg), config_hash(load_config())
    assert h1 == h2 and len(h1) >= 12
    mutated = dict(cfg)
    mutated["sqi"] = {**cfg["sqi"], "floor": 0.99}
    assert config_hash(mutated) != h1


def test_spec_b6_values_survive_the_yaml_loader():
    cfg = load_config()
    assert cfg["capture"]["lux_floor"] == 100
    assert cfg["capture"]["bpp_floor"] == 0.12
    assert cfg["sync"]["min_marker_events"] == 10
    assert cfg["runs"]["min_conf"] == "CALIBRATED"
    assert cfg["runs"]["ibi_physiologic_ms"] == [250, 2200]
    assert cfg["gates"]["g1"]["confidence_ece"] == 0.10
    assert cfg["serial"]["rule"] == [2, 3]
