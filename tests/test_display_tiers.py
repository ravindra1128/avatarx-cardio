"""Display tiers (owner decision 2026-09-09): a result on every scan that
yielded beats, as a 0-100 score in a normal-looking range, labelled
"measured" when every floor held and "provisional" otherwise. Nothing is
invented: a scan with no beats stays blank."""
import os

import numpy as np
import pytest

os.environ.setdefault("AFIB_SHEET_ID", "test-sheet")

from features.hemodynamics import (  # noqa: E402
    MIN_AMPLITUDE_BEATS, MIN_PROVISIONAL_AMPLITUDE_BEATS, MIN_RATE_INTERVALS,
    MIN_PROVISIONAL_RATE_INTERVALS, RI_TYPICAL, TONE_TYPICAL, FITNESS_TYPICAL,
    stiffness_contour, vasomotor_indices, cardiorespiratory_indices,
    stiffness_score_from_reflection_index, stiffness_score_from_rise_time,
    tone_score_from_cv, resting_rate_index)


class _Reg:
    def __init__(self, rmssd=None):
        self.dispersion = {"rmssd_ms": rmssd, "sdnn_ms": None}


# ------------------------------------------------------------ score maps
def test_scores_are_monotone_bounded_and_documented():
    assert stiffness_score_from_reflection_index(0.5) == 50.0
    assert stiffness_score_from_reflection_index(1.3) == 100.0       # clipped
    assert stiffness_score_from_reflection_index(None) is None
    assert stiffness_score_from_rise_time(0.120) == 30.0
    assert stiffness_score_from_rise_time(0.320) == 90.0
    assert stiffness_score_from_rise_time(0.220) == 60.0
    assert 0.0 <= stiffness_score_from_rise_time(0.010) <= 100.0
    assert tone_score_from_cv(0.523) == 52.3
    assert tone_score_from_cv(1.7) == 100.0
    for lo, hi in (RI_TYPICAL, TONE_TYPICAL, FITNESS_TYPICAL):
        assert 0 <= lo < hi <= 100


def test_fitness_anchors_put_typical_resting_rates_in_the_typical_band():
    """Population anchors: a resting rate of 60-85 should not read as
    unfit. The old anchors put 84 bpm at 12/100."""
    for hr in (60, 65, 70, 75, 80):
        s = 100 * resting_rate_index(hr)
        assert FITNESS_TYPICAL[0] <= s <= FITNESS_TYPICAL[1] + 1, (hr, s)
    assert 100 * resting_rate_index(84) > 25
    assert resting_rate_index(50) > resting_rate_index(70) > resting_rate_index(90)


# ------------------------------------------------------------ stiffness
def test_stiffness_is_measured_from_the_reflection_index():
    out = stiffness_contour({"reflection_index": 0.62, "rise_time_s": 0.2})
    assert out["available"] and out["tier"] == "measured"
    assert out["estimate"]["value"] == 62.0 and out["estimate"]["unit"] == "/100"
    assert out["raw_name"] == "reflection_index" and out["raw_value"] == 0.62
    assert out["tier_reasons"] == []


def test_stiffness_is_provisional_from_crest_time_without_a_notch():
    out = stiffness_contour({"reflection_index": None, "rise_time_s": 0.220})
    assert out["available"] and out["tier"] == "provisional"
    assert out["estimate"]["value"] == 60.0
    assert out["raw_name"] == "pulse_rise_time" and out["raw_unit"] == "ms"
    assert "no dicrotic notch" in out["tier_reasons"][0]


def test_stiffness_stays_blank_with_no_marker_at_all():
    out = stiffness_contour({"reflection_index": None, "rise_time_s": None})
    assert out["available"] is False and out["estimate"] is None
    assert out["tier"] is None and "no stiffness marker" in out["reason"]


# ------------------------------------------------------------ tone
def _amps(n, cv=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return [(i * 1.0, 1.0 + cv * rng.standard_normal()) for i in range(n)]


def test_tone_is_measured_at_the_floor_and_provisional_below_it():
    full = vasomotor_indices(_amps(MIN_AMPLITUDE_BEATS), locked=True)
    assert full["available"] and full["tier"] == "measured"
    assert full["estimate"]["unit"] == "/100"
    assert full["estimate"]["value"] == full["score"] == tone_score_from_cv(full["amplitude_cv"])
    thin = vasomotor_indices(_amps(MIN_PROVISIONAL_AMPLITUDE_BEATS), locked=True)
    assert thin["available"] and thin["tier"] == "provisional"
    assert "amplitude beats" in thin["tier_reasons"][0]
    none = vasomotor_indices(_amps(MIN_PROVISIONAL_AMPLITUDE_BEATS - 1), locked=True)
    assert none["available"] is False


# ------------------------------------------------------------ fitness
def test_fitness_measured_needs_the_clean_rate_floor_and_an_agreeing_pulse():
    out = cardiorespiratory_indices(_Reg(), 70.0, None, rate_method="clean_interval_median",
                                    rate_intervals=MIN_RATE_INTERVALS, pulse_verdict="agree")
    assert out["tier"] == "measured" and out["tier_reasons"] == []
    assert out["estimate"]["value"] == round(100 * resting_rate_index(70.0), 1)
    assert out["raw_value"] == 70.0 and out["raw_unit"] == "bpm"


def test_fitness_takes_the_waveform_rate_when_the_beat_count_disagrees():
    out = cardiorespiratory_indices(_Reg(), 84.0, None, rate_method="clean_interval_median",
                                    rate_intervals=20, pulse_verdict="disagree",
                                    spectral_hr_bpm=60.0)
    assert out["tier"] == "provisional"
    assert out["resting_rate_source"] == "waveform_rhythm"
    assert out["resting_hr_bpm"] == 60.0
    assert out["estimate"]["value"] == round(100 * resting_rate_index(60.0), 1)
    assert "disagreed" in out["tier_reasons"][0]


def test_fitness_is_provisional_from_a_thin_clean_rate():
    out = cardiorespiratory_indices(_Reg(), 70.0, None, rate_method="clean_interval_median",
                                    rate_intervals=MIN_PROVISIONAL_RATE_INTERVALS,
                                    pulse_verdict="agree", spectral_hr_bpm=74.0)
    assert out["tier"] == "provisional"
    assert out["resting_hr_bpm"] == 70.0                    # clean rate kept, spectrum not needed
    assert f"only {MIN_PROVISIONAL_RATE_INTERVALS}" in out["tier_reasons"][0]


def test_fitness_falls_to_the_waveform_when_the_clean_rate_is_too_thin():
    out = cardiorespiratory_indices(_Reg(), 70.0, None, rate_method="clean_interval_median",
                                    rate_intervals=3, spectral_hr_bpm=74.0)
    assert out["tier"] == "provisional" and out["resting_rate_source"] == "waveform_rhythm"
    assert out["resting_hr_bpm"] == 74.0


def test_fitness_is_blank_only_with_no_rate_at_all():
    out = cardiorespiratory_indices(_Reg(), None, None)
    assert out["available"] is False and out["tier"] is None
    assert out["reason_code"] == "no_resting_rate"
    out = cardiorespiratory_indices(_Reg(), None, None, spectral_hr_bpm=66.0)
    assert out["available"] and out["tier"] == "provisional"
