"""Mathematical correctness controls; these do not validate clinical accuracy."""
import numpy as np
import pytest

from inference import evidence


FREQUENCIES = np.arange(81, dtype=float) * 0.05


@pytest.mark.parametrize('power', [
    1 / (FREQUENCIES + 0.1) ** 2,
    FREQUENCIES + 0.1,
    np.ones_like(FREQUENCIES),
    np.zeros_like(FREQUENCIES),
])
def test_no_pulse_reported_from_spectra_without_a_peak(power):
    assert evidence._fundamental(FREQUENCIES, power) == (None, None)


@pytest.mark.parametrize('bpm', [45, 48, 60, 90, 120, 150, 177])
def test_real_peaks_remain_valid_including_low_rate(bpm):
    power = 0.01 + np.exp(-0.5 * ((FREQUENCIES - bpm / 60) / 0.02) ** 2)
    frequency, ratio = evidence._fundamental(FREQUENCIES, power)
    assert frequency * 60 == pytest.approx(bpm)
    assert ratio > 1


@pytest.mark.parametrize('boundary', [0.7, 3.0])
def test_band_boundary_maximum_is_not_moved_inward(boundary):
    power = 0.01 + np.exp(-0.5 * ((FREQUENCIES - boundary) / 0.04) ** 2)
    assert evidence._fundamental(FREQUENCIES, power) == (None, None)


def test_real_peak_can_be_recovered_below_a_stronger_drift_boundary():
    power = 1 / (FREQUENCIES + 0.1) ** 2
    power += 0.6 * np.exp(-0.5 * ((FREQUENCIES - 1.5) / 0.02) ** 2)
    assert evidence._fundamental(FREQUENCIES, power)[0] * 60 == pytest.approx(90)


def test_declining_background_near_half_rate_does_not_halve_a_real_peak():
    power = 1 / (FREQUENCIES + 0.1) ** 2
    power += 1.8 * np.exp(-0.5 * ((FREQUENCIES - 1.5) / 0.02) ** 2)
    assert evidence._fundamental(FREQUENCIES, power)[0] * 60 == pytest.approx(90)


def test_supported_subharmonic_still_replaces_the_dominant_second_harmonic():
    power = np.full_like(FREQUENCIES, 0.01)
    power += 0.7 * np.exp(-0.5 * ((FREQUENCIES - 1.0) / 0.02) ** 2)
    power += np.exp(-0.5 * ((FREQUENCIES - 2.0) / 0.02) ** 2)
    assert evidence._fundamental(FREQUENCIES, power)[0] * 60 == pytest.approx(60)


def test_weak_subharmonic_does_not_change_existing_power_requirement():
    power = np.full_like(FREQUENCIES, 0.01)
    power += 0.2 * np.exp(-0.5 * ((FREQUENCIES - 1.0) / 0.02) ** 2)
    power += np.exp(-0.5 * ((FREQUENCIES - 2.0) / 0.02) ** 2)
    assert evidence._fundamental(FREQUENCIES, power)[0] * 60 == pytest.approx(120)


def test_flat_topped_interior_peak_has_a_deterministic_centre():
    power = np.full_like(FREQUENCIES, 0.01)
    power[29:32] = 1.0
    assert evidence._fundamental(FREQUENCIES, power)[0] * 60 == pytest.approx(90)


@pytest.mark.parametrize('f,p', [
    ([], []), ([1.0], [1.0]), ([1.0, 1.1], [2.0, 1.0]),
    ([1.0, 1.1, 1.2], [1.0, float('nan'), 1.0]),
    ([1.0, 1.1, 1.2], [1.0, float('inf'), 1.0]),
    ([1.0, 1.1, 1.2], [1.0, -1.0, 1.0]),
])
def test_insufficient_or_invalid_spectrum_gives_no_estimate(f, p):
    assert evidence._fundamental(np.array(f), np.array(p)) == (None, None)


def test_four_regions_with_shared_drift_cannot_corroborate_a_pulse(monkeypatch):
    power = 1 / (FREQUENCIES + 0.1) ** 2
    monkeypatch.setattr(evidence, '_welch_psd', lambda *_: (FREQUENCIES, power))
    out = evidence.spectral_pulse({r: np.zeros(1200) for r in evidence.ROI_NAMES}, 30)
    assert out['pulse_spectral_bpm'] is None
    assert out['pulse_spectral_roi_agree'] == 0
    assert all(x is None for x in out['pulse_spectral_roi_bpm'].values())
