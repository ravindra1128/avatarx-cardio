"""Iteration 12: the beat-count pulse is checked against the waveform's
dominant rhythm. Spectral estimator on synthetic waveforms; verdict logic on
evidence dicts."""
import os

import numpy as np

os.environ.setdefault("AFIB_SHEET_ID", "test-sheet")

from inference.evidence import spectral_pulse, pulse_agreement  # noqa: E402
from features.hemodynamics import pulse_check, PULSE_AGREEMENT_TOL  # noqa: E402

FPS = 30.3
ROIS = ("forehead", "cheek_l", "cheek_r", "nose")


def _wave(bpm, n=1200, harmonic=0.0, noise=0.8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    f = bpm / 60.0
    return (np.sin(2 * np.pi * f * t) + harmonic * np.sin(2 * np.pi * 2 * f * t + 0.4)
            + noise * rng.standard_normal(n))


def test_subharmonic_check_recovers_the_pulse_under_a_dominant_second_harmonic():
    out = spectral_pulse({r: _wave(55, harmonic=1.3, seed=i) for i, r in enumerate(ROIS)}, FPS)
    assert abs(out["pulse_spectral_bpm"] - 55) <= 3       # 3 bpm bin resolution
    assert out["pulse_spectral_roi_agree"] == 4


def test_a_clean_fast_pulse_is_not_halved():
    out = spectral_pulse({r: _wave(100, harmonic=0.2, seed=i) for i, r in enumerate(ROIS)}, FPS)
    assert abs(out["pulse_spectral_bpm"] - 100) <= 3


def test_one_noise_region_cannot_dominate_the_fused_estimate():
    waves = {r: _wave(62, seed=i) for i, r in enumerate(ROIS[:3])}
    waves["nose"] = 50.0 * np.random.default_rng(9).standard_normal(1200)
    out = spectral_pulse(waves, FPS)
    assert abs(out["pulse_spectral_bpm"] - 62) <= 3
    assert out["pulse_spectral_roi_agree"] >= 3


def test_too_short_waveforms_give_no_estimate():
    out = spectral_pulse({r: _wave(60)[:200] for r in ROIS}, FPS)
    assert out["pulse_spectral_bpm"] is None
    assert out["pulse_spectral_roi_agree"] == 0


def test_agreement_is_relative_to_the_spectral_estimate():
    assert pulse_agreement(100, 55) > 0.8
    assert pulse_agreement(60.4, 54) < PULSE_AGREEMENT_TOL + 0.01
    assert pulse_agreement(None, 54) is None
    assert pulse_agreement(60, 0) is None


def test_verdicts():
    assert pulse_check({})["verdict"] == "not_evaluated"          # legacy evidence
    assert pulse_check({"pulse_agreement": None, "pulse_lattice_bpm": None,
                        "pulse_spectral_bpm": 60})["verdict"] == "unresolved"
    assert pulse_check({"pulse_agreement": 0.05, "pulse_lattice_bpm": 63,
                        "pulse_spectral_bpm": 60, "pulse_spectral_roi_agree": 1})["verdict"] == "unresolved"
    ok = pulse_check({"pulse_agreement": 0.05, "pulse_lattice_bpm": 63,
                      "pulse_spectral_bpm": 60, "pulse_spectral_roi_agree": 3})
    assert ok["verdict"] == "agree" and "63 bpm" in ok["reason"]
    bad = pulse_check({"pulse_agreement": 0.82, "pulse_lattice_bpm": 100,
                       "pulse_spectral_bpm": 55, "pulse_spectral_roi_agree": 4})
    assert bad["verdict"] == "disagree" and "100 bpm" in bad["reason"] and "55 bpm" in bad["reason"]
