"""The resting-rate doubling guard.

The interval-median resting rate inflates toward 2x when the beat detector
splits beats; the spectral rate is subharmonic-protected. When the interval
rate shows a doubling signature against a trustworthy spectral anchor, the
fitness rate must come from the spectral rhythm instead — the 101-vs-65 case
(35 % half-length intervals) that read a resting scan as tachycardic.

Two layers are pinned: the pure detector, and its effect on the fitness card
via cardiorespiratory_indices. The clinical rhythm/flutter head is out of
scope by construction and is not exercised here.
"""
import pytest

from features.rate_guard import (
    HARMONIC_FRACTION_HI,
    SPECTRAL_SNR_FLOOR,
    guard_resting_pulse,
)
from features.hemodynamics import cardiorespiratory_indices


# ------------------------------------------------------------ pure detector
def test_a_clean_scan_is_left_alone():
    g = guard_resting_pulse(66.0, 67.0, harmonic_fraction=0.05, spectral_snr=3.0)
    assert g["guarded"] is False
    assert g["reported_bpm"] == 66.0
    assert g["signature"] is None


def test_a_clean_double_is_folded_to_the_rhythm():
    g = guard_resting_pulse(130.0, 65.0, harmonic_fraction=0.1, spectral_snr=3.0)
    assert g["guarded"] is True and g["signature"] == "double"
    assert g["reported_bpm"] == 65.0


def test_a_clean_half_is_folded_to_the_rhythm():
    g = guard_resting_pulse(33.0, 66.0, harmonic_fraction=0.1, spectral_snr=3.0)
    assert g["guarded"] is True and g["signature"] == "half"
    assert g["reported_bpm"] == 66.0


def test_the_live_partial_split_case_101_vs_69():
    # ratio 1.46 — not a clean 2x, but 35 % half-length intervals says split.
    g = guard_resting_pulse(101.0, 69.0, harmonic_fraction=0.35, spectral_snr=3.0)
    assert g["guarded"] is True and g["signature"] == "split_inflation"
    assert g["reported_bpm"] == 69.0
    assert "using the rhythm rate" in g["reason"]


def test_inflation_without_the_harmonic_evidence_is_not_touched():
    # 90 vs 66 is a real possibility (exertion, anxiety); without many
    # half-length intervals the guard must not overwrite it.
    g = guard_resting_pulse(90.0, 66.0, harmonic_fraction=0.05, spectral_snr=3.0)
    assert g["guarded"] is False


def test_an_untrustworthy_spectral_anchor_is_not_used():
    g = guard_resting_pulse(101.0, 69.0, harmonic_fraction=0.35,
                            spectral_snr=SPECTRAL_SNR_FLOOR - 0.1)
    assert g["guarded"] is False


def test_no_anchor_no_guard():
    assert guard_resting_pulse(101.0, None, harmonic_fraction=0.9)["guarded"] is False
    assert guard_resting_pulse(101.0, 0.0, harmonic_fraction=0.9)["guarded"] is False
    assert guard_resting_pulse(None, 66.0)["guarded"] is False


def test_ref_bpm_is_telemetry_only_and_never_required():
    # present or absent, it does not change the decision (camera-only).
    a = guard_resting_pulse(130.0, 65.0, spectral_snr=3.0, ref_bpm=64.0)
    b = guard_resting_pulse(130.0, 65.0, spectral_snr=3.0)
    assert a["guarded"] == b["guarded"] == True
    assert a["ref_bpm"] == 64.0 and b["ref_bpm"] is None


def test_snr_is_optional():
    # when SNR is not reported, a clean doubling signature still guards.
    g = guard_resting_pulse(130.0, 65.0, harmonic_fraction=0.1)
    assert g["guarded"] is True


# ------------------------------------------------ effect on the fitness card
class _Reg:
    """Minimal regularity stub: dispersion only, like the real object."""
    def __init__(self, rmssd=40.0):
        self.dispersion = {"rmssd_ms": rmssd, "sdnn_ms": 50.0}


def test_the_fitness_card_takes_the_rhythm_rate_on_a_split_scan():
    # A clean-interval rate of 101 with 35 % harmonic and a 69 bpm rhythm:
    # the card's resting rate must become 69, not 101, and drop to provisional.
    out = cardiorespiratory_indices(
        _Reg(), 101.0, None, rate_method="clean_interval_median",
        rate_intervals=20, spectral_hr_bpm=69.0, spectral_roi_agree=3,
        harmonic_fraction=0.35, spectral_snr=3.0, pulse_verdict="agree")
    assert out["resting_hr_bpm"] == 69.0
    assert out["tier"] == "provisional"
    assert out["rate_guard"]["guarded"] is True and out["rate_guard"]["applied"] is True
    assert any("rhythm rate" in r for r in out["tier_reasons"])


def test_split_inflation_without_region_backing_reports_no_rate():
    # Audit #7 (CALC-4) + card robustness (2026-09-16): a split-inflation
    # signature (ratio ~1.5) may override the count only when >= 3 regions
    # back the waveform rhythm; with 2 the count is NOT published either -
    # on 2026-09-16 counts of 109/114 against a 78 bpm rhythm (reference
    # 72-74) had put fitness at 6.5/100 "provisional" while the pulse head
    # reported 78. The card abstains with both estimates in the reason.
    out = cardiorespiratory_indices(
        _Reg(), 101.0, None, rate_method="clean_interval_median",
        rate_intervals=20, spectral_hr_bpm=69.0, spectral_roi_agree=2,
        harmonic_fraction=0.35, spectral_snr=3.0, pulse_verdict="agree")
    assert out["available"] is False and out["fitness_proxy_score"] is None
    assert out["resting_hr_bpm"] is None and out["tier"] is None
    assert out["reason_code"] == "resting_rate_unverified"
    assert "101 bpm" in out["reason"] and "69 bpm" in out["reason"]
    assert out["rate_guard"]["guarded"] is True and out["rate_guard"]["applied"] is False
    assert any("too few facial regions" in r for r in out["tier_reasons"])


def test_a_clean_fitness_scan_is_unchanged_and_stays_measured():
    out = cardiorespiratory_indices(
        _Reg(), 66.0, None, rate_method="clean_interval_median",
        rate_intervals=20, spectral_hr_bpm=67.0,
        harmonic_fraction=0.05, spectral_snr=3.0, pulse_verdict="agree")
    assert out["resting_hr_bpm"] == 66.0
    assert out["tier"] == "measured"
    assert out["rate_guard"]["guarded"] is False


def test_the_guard_is_inert_when_the_caller_passes_no_evidence():
    # Legacy/back-compat call: no harmonic_fraction, no spectral_snr.
    out = cardiorespiratory_indices(
        _Reg(), 66.0, None, rate_method="clean_interval_median",
        rate_intervals=20)
    assert out["tier"] == "measured"
    assert out["rate_guard"]["guarded"] is False


def test_a_clean_double_needs_no_region_backing():
    # count/2 == spectral is two independent estimators agreeing on the halved
    # rate; that is evidence in itself (holdout clips 9b3023/ebe749).
    out = cardiorespiratory_indices(
        _Reg(), 130.0, None, rate_method="clean_interval_median",
        rate_intervals=20, spectral_hr_bpm=65.0, spectral_roi_agree=2,
        harmonic_fraction=0.3, spectral_snr=3.0, pulse_verdict="disagree")
    assert out["resting_hr_bpm"] == 65.0
    assert out["tier"] == "provisional" and out["rate_guard"]["applied"] is True
