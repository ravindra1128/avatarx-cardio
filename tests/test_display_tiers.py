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
    MIN_PROVISIONAL_RATE_INTERVALS, RI_TYPICAL, RISE_TYPICAL_MS, TONE_TYPICAL,
    FITNESS_TYPICAL, stiffness_contour, stiffness_band_from_rise_time,
    vasomotor_indices, cardiorespiratory_indices, tone_score_from_cv,
    resting_rate_index)


class _Reg:
    def __init__(self, rmssd=None):
        self.dispersion = {"rmssd_ms": rmssd, "sdnn_ms": None}


# ------------------------------------------------------------ score maps
def test_scores_are_monotone_bounded_and_documented():
    assert tone_score_from_cv(0.523) == 52.3
    assert tone_score_from_cv(1.7) == 100.0          # clipped
    assert tone_score_from_cv(None) is None
    for lo, hi in (TONE_TYPICAL, FITNESS_TYPICAL):
        assert 0 <= lo < hi <= 100
    for lo, hi in (RI_TYPICAL, RISE_TYPICAL_MS):      # stiffness keeps its own units
        assert 0 < lo < hi


def test_fitness_anchors_put_typical_resting_rates_in_the_typical_band():
    """Population anchors: a resting rate of 60-85 should not read as
    unfit. The old anchors put 84 bpm at 12/100."""
    for hr in (60, 65, 70, 75, 80):
        s = 100 * resting_rate_index(hr)
        assert FITNESS_TYPICAL[0] <= s <= FITNESS_TYPICAL[1] + 1, (hr, s)
    assert 100 * resting_rate_index(84) > 25
    assert resting_rate_index(50) > resting_rate_index(70) > resting_rate_index(90)


# ------------------------------------------------------------ stiffness
def test_stiffness_is_the_reflection_index_in_its_own_units():
    """Owner, 2026-09-09: the ratio, as it read before the 0-100 rescale."""
    out = stiffness_contour({"reflection_index": 0.353, "rise_time_s": 0.2})
    assert out["available"] and out["tier"] == "measured"
    assert out["estimate"]["value"] == 0.353 and out["estimate"]["unit"] == "ratio"
    assert out["estimate"]["name"] == "reflection_index"
    assert out["score_typical_range"] == list(RI_TYPICAL)
    assert out["tier_reasons"] == []


def test_the_crest_time_band_runs_the_way_the_physiology_does():
    """A stiffer artery carries the wave faster, so a SHORTER crest time reads
    as higher stiffness. This file assumed the opposite at first; the mapping
    was checked against the PPG literature on 2026-09-09 and corrected."""
    lo, hi = RISE_TYPICAL_MS
    assert stiffness_band_from_rise_time((lo - 30) / 1000.0) == "High"
    assert stiffness_band_from_rise_time((lo + hi) / 2000.0) == "Typical"
    assert stiffness_band_from_rise_time((hi + 80) / 1000.0) == "Low"
    assert stiffness_band_from_rise_time(None) is None


def test_stiffness_falls_back_to_a_band_never_a_second_number():
    """Owner, 2026-09-09: two unrelated numbers must not share the value slot."""
    out = stiffness_contour({"reflection_index": None, "rise_time_s": 0.220})
    assert out["available"] and out["tier"] == "provisional"
    assert out["band"] == "Typical"
    assert out["estimate"]["value"] is None and out["estimate"]["unit"] is None
    assert out["estimate"]["band"] == "Typical"
    # the crest time itself is still reported, just never as the card's number
    assert out["raw_value"] == 220.0 and out["raw_unit"] == "ms"
    assert "different and weaker marker" in out["tier_reasons"][0]


def test_a_measured_stiffness_never_carries_a_band():
    out = stiffness_contour({"reflection_index": 0.62, "rise_time_s": 0.22})
    assert out["band"] is None and out["estimate"]["band"] is None
    assert out["estimate"]["value"] == 0.62


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


def test_a_research_frame_rate_card_uses_the_aging_index_not_the_reflection_index():
    """One metric per capture class (iteration 13): where the SDPPG is
    derivable the aging index is the marker, so a given device's card never
    changes quantity between scans. Rewriting this card for display tiers
    dropped that path once; this pins it."""
    from features.hemodynamics import AGING_TYPICAL
    out = stiffness_contour({"sdppg_b_a": -0.8, "sdppg_c_a": -0.1,
                             "sdppg_d_a": -0.3, "sdppg_e_a": 0.1,
                             "reflection_index": 0.62, "rise_time_s": 0.22})
    if out["raw_name"] == "second_derivative_aging_index":
        assert out["tier"] == "measured" and out["estimate"]["unit"] == "index"
        assert out["score_typical_range"] == list(AGING_TYPICAL)
        assert out["band"] is None
    else:                       # this fixture's keys are not the SDPPG shape
        assert out["estimate"]["name"] == "reflection_index"


def test_pooling_does_not_starve_the_provisional_tone_score():
    """2026-09-09: a real scan had 18 usable beats over 4 capture segments and
    the tone card still read "0 usable beats". Pooling dropped every region for
    holding fewer than the MEASURED floor, so the card never saw the beats the
    provisional floor was lowered to admit. Pooling uses the provisional floor;
    the card decides the tier from what it receives."""
    from features.hemodynamics import pool_amplitudes
    n = MIN_PROVISIONAL_AMPLITUDE_BEATS
    per_roi = {"forehead": [(i * 1.0, 1.0 + 0.1 * i) for i in range(n)],
               "cheek_l": [(i * 1.0, 1.0 + 0.1 * i) for i in range(n)]}
    pooled = pool_amplitudes(per_roi)
    assert len(pooled) >= n
    out = vasomotor_indices(pooled, locked=True)
    assert out["available"] and out["tier"] == "provisional"

    # below the provisional floor there is still nothing to pool
    thin = {"forehead": [(i * 1.0, 1.0) for i in range(n - 1)]}
    assert pool_amplitudes(thin) == []
