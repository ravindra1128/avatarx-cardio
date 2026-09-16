"""Audit 2026-09-17 #2 — the interim irregularity rule is noise-aware.

median|dRR| and pNN50 are compared to ECG-scale thresholds (60 ms / 0.40),
but the phone's beat timing is 20-59 ms and successive differences amplify
per-beat timing error by sqrt(6): a perfectly regular rhythm tripped the rule
on most phone scans, after which only the AF-grade bar or an abstention was
reachable. The thresholds are unchanged; the observed irregularity must also
exceed NOISE_FLOOR_MARGIN x what the scan's own timing noise predicts.
"""
import numpy as np

from configs import load_config
from features.rhythm import RhythmFeatures
from inference.decision_logic import (NOISE_FLOOR_MARGIN, decide_with_rationale,
                                      irregularity_noise_floor)

CFG = load_config()


def test_floor_scales_with_timing_precision_and_fusion_count():
    a = irregularity_noise_floor({"timing_precision_ms": 40.0, "mean_roi_agreement": 0.5})   # k = 2
    b = irregularity_noise_floor({"timing_precision_ms": 40.0, "mean_roi_agreement": 1.0})   # k = 4
    c = irregularity_noise_floor({"timing_precision_ms": 20.0, "mean_roi_agreement": 0.5})
    assert a["k"] == 2.0 and b["k"] == 4.0
    assert a["mad_floor_ms"] > b["mad_floor_ms"] > 0          # more ROIs averaged -> less noise
    assert abs(a["mad_floor_ms"] / c["mad_floor_ms"] - 2.0) < 1e-6   # linear in precision
    # 40 ms, k=2: sigma_f = 40/0.954/sqrt(2) = 29.6; sd_d = sqrt(6)*29.6 = 72.6; median = 49.0
    assert abs(a["mad_floor_ms"] - 49.0) < 0.5


def test_no_precision_means_no_floor():
    f = irregularity_noise_floor({})
    assert f["mad_floor_ms"] is None
    f = irregularity_noise_floor({"timing_precision_ms": float("nan")})
    assert f["mad_floor_ms"] is None


def _decide(mad, pnn50, tp, *, coherence=0.30, matched=0.70, n=25, rmssd=None):
    vals = {"median_abs_succ_diff": mad, "pnn50": pnn50, "median_ibi": 900.0,
            "dropout_rate": 0.0, "n_intervals": n, "rmssd": rmssd or mad * 1.3}
    f = RhythmFeatures(values=vals, n_intervals=n, mean_confidence=0.9, estimator_warnings=[])
    ev = {"cross_roi_coherence": coherence, "timing_precision_ms": tp, "timing_matched_fraction": matched,
          "split_fraction": 0.0, "harmonic_fraction": 0.0, "mean_roi_agreement": 0.5,
          "pulse_lattice_bpm": 66.7, "pulse_spectral_bpm": 66.7, "pulse_spectral_roi_agree": 3,
          "pulse_agreement": 0.0, "pulse_spectral_snr": 3.0}
    return decide_with_rationale(f, 0.5, 0.65, CFG, recording_id="t", evidence=ev)


def test_irregularity_within_the_noise_floor_is_not_irregular():
    # 40 ms precision, k=2: floor 49 ms -> the rule needs mad >= 98 ms.
    # mad 65 / pNN50 0.5 used to fire the AF pattern and abstain; now SINUS.
    res, why = _decide(65.0, 0.50, 40.0)
    assert why["rule"]["irregularity_exceeds_noise"] is False
    assert why["rule"]["afib_pattern_rule"] is False
    assert res.outcome.value == "ACCEPT" and res.predicted_class == "SINUS"


def test_irregularity_well_above_the_floor_still_fires():
    res, why = _decide(140.0, 0.80, 40.0)          # AF-like: 140 >= 2 x 49
    assert why["rule"]["irregularity_exceeds_noise"] is True
    assert why["rule"]["afib_pattern_rule"] is True
    # at phone coherence 0.30 the AF-grade bar (0.35) still abstains: fail closed
    assert res.outcome.value == "REPEAT_SCAN"
    assert any("irregular intervals seen" in r for r in res.no_read_reasons)


def test_precise_timing_keeps_the_original_rule_exactly():
    # 5 ms precision: floor ~6 ms, margin 12 ms << 60 - the rule is unchanged.
    res, why = _decide(65.0, 0.50, 5.0, coherence=0.6, matched=0.9)
    assert why["rule"]["afib_pattern_rule"] is True


def test_the_other_class_uses_the_same_floor():
    res, why = _decide(50.0, 0.48, 40.0)          # 'other' box, inside the floor
    assert res.predicted_class == "SINUS"
    res, why = _decide(50.0, 0.48, 10.0, coherence=0.6, matched=0.9)   # precise: 'other' as before
    assert res.predicted_class == "OTHER_IRREGULAR"


def test_margin_is_the_documented_value():
    assert NOISE_FLOOR_MARGIN == 2.0
