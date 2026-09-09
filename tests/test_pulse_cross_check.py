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


# --- gate mode + the fitness card's single basis (iteration 14, 2026-09-09) ---

from features.hemodynamics import (  # noqa: E402
    PULSE_CHECK_MODE, PULSE_MIN_ROI_AGREE, MIN_RATE_INTERVALS,
    autonomic_index, resting_rate_index, cardiorespiratory_indices)


class _Reg:
    """Minimal regularity stand-in: only `dispersion` is read by the card."""
    def __init__(self, rmssd=None, sdnn=None):
        self.dispersion = {"rmssd_ms": rmssd, "sdnn_ms": sdnn}


def test_the_pulse_check_is_binding():
    assert PULSE_CHECK_MODE == "gate"


def test_the_motivating_production_scan_abstains():
    """2026-09-09: beat count 84, spectral 60, 2 of 4 regions backing the
    spectrum, reference device 64. This must be a resolved disagreement."""
    v = pulse_check({"pulse_lattice_bpm": 84.0, "pulse_spectral_bpm": 60.0,
                     "pulse_agreement": 0.40, "pulse_spectral_roi_agree": 2})
    assert v["verdict"] == "disagree"
    assert v["mode"] == "gate"


def test_a_disagreement_the_regions_do_not_back_is_not_a_veto():
    """The same numbers with one region behind the spectrum must NOT abstain:
    an unresolved spectrum leaves the cards exactly as they were."""
    for ra in range(PULSE_MIN_ROI_AGREE):
        v = pulse_check({"pulse_lattice_bpm": 84.0, "pulse_spectral_bpm": 60.0,
                         "pulse_agreement": 0.40, "pulse_spectral_roi_agree": ra})
        assert v["verdict"] == "unresolved", ra
    # and a missing count fails closed to unresolved, never to a veto
    assert pulse_check({"pulse_lattice_bpm": 84.0, "pulse_spectral_bpm": 60.0,
                        "pulse_agreement": 0.40})["verdict"] == "unresolved"


def test_fitness_uses_one_basis_regardless_of_rmssd():
    """The composite is gone: the same rate gives the same score whether or
    not an RMSSD survived cleaning. This is the 27.1-then-50.1 defect."""
    kw = {"rate_method": "clean_interval_median", "rate_intervals": 20}
    with_hrv = cardiorespiratory_indices(_Reg(rmssd=286.0), 83.5, None, **kw)
    without = cardiorespiratory_indices(_Reg(rmssd=None), 83.5, None, **kw)
    assert with_hrv["fitness_proxy_score"] == without["fitness_proxy_score"]
    assert with_hrv["fitness_proxy_score"] == round(100 * resting_rate_index(83.5), 1)
    assert with_hrv["fitness_proxy_score"] != 50.1      # the composite's answer
    assert with_hrv["fitness_proxy_basis"] == "resting_hr_only"
    assert with_hrv["autonomic_index"] is None
    assert with_hrv["rmssd_ms"] == 286.0                # reported, never scored


def test_fitness_needs_the_pipelines_own_publish_a_rate_floor():
    at = cardiorespiratory_indices(_Reg(), 60.0, None,
                                   rate_method="clean_interval_median",
                                   rate_intervals=MIN_RATE_INTERVALS)
    assert at["available"] is True and at["reason_code"] is None
    below = cardiorespiratory_indices(_Reg(), 60.0, None,
                                      rate_method="clean_interval_median",
                                      rate_intervals=MIN_RATE_INTERVALS - 1)
    assert below["available"] is False
    assert below["reason_code"] == "insufficient_clean_intervals"
    assert str(MIN_RATE_INTERVALS) in below["reason"]


def test_a_rate_from_unsplit_intervals_is_inadmissible_at_any_count():
    """The fallback median never sees the missed/false-beat splitter, so a
    doubled or halved interval survives it. Count cannot buy that back."""
    out = cardiorespiratory_indices(_Reg(), 60.0, None,
                                    rate_method="calibrated_fused_beat_median",
                                    rate_intervals=30)
    assert out["available"] is False
    assert out["reason_code"] == "rate_not_from_clean_intervals"


def test_legacy_callers_without_rate_provenance_still_work():
    """Fixtures that call the function positionally must not start abstaining."""
    out = cardiorespiratory_indices(_Reg(rmssd=40.0), 60.0)
    assert out["available"] is True
    assert out["fitness_proxy_basis"] == "resting_hr_only"


def test_autonomic_index_survives_as_a_function_but_not_as_the_basis():
    """Kept public for its own tests and for research use; never the card."""
    assert autonomic_index(60.0, 40.0) is not None
    out = cardiorespiratory_indices(_Reg(rmssd=40.0), 60.0, None,
                                    rate_method="clean_interval_median",
                                    rate_intervals=20)
    assert out["autonomic_index"] is None
