"""Audit 2026-09-17 #7 — one resting pulse per scan.

resolve_resting_rate is the single resolver the rate head, the fitness card
and the decision's ACCEPT pulse all consume. Its rules are the rate head's
(count/waveform reconciliation, waveform may replace the count only with
>= 3 backing regions) with the doubling guard folded in behind the same bar,
and it fails closed to None rather than publish a number it cannot defend.
"""
from features.rate_guard import resolve_resting_rate, MIN_ROI_AGREE_FOR_RATE


def _ev(spec=66.0, roia=3, agreement=0.02, hf=0.05, snr=3.0):
    return {"pulse_lattice_bpm": 65.0, "pulse_spectral_bpm": spec,
            "pulse_spectral_roi_agree": roia, "pulse_agreement": agreement,
            "harmonic_fraction": hf, "pulse_spectral_snr": snr}


def test_agreeing_count_and_waveform_give_a_verified_mean():
    r = resolve_resting_rate(65.0, 25, _ev(), min_intervals=15)
    assert r["confidence"] == "verified" and r["source"] == "count_spectral_mean"
    assert r["bpm"] == 65.5


def test_no_evidence_gives_the_count_unverified():
    r = resolve_resting_rate(65.0, 25, {}, min_intervals=15)
    assert r["confidence"] == "unverified" and r["bpm"] == 65.0


def test_disagreement_with_backing_reports_the_waveform_provisionally():
    r = resolve_resting_rate(101.0, 25, _ev(spec=66.0, roia=3, agreement=0.53), min_intervals=15)
    assert r["confidence"] == "provisional" and r["bpm"] == 66.0


def test_disagreement_without_backing_reports_nothing():
    r = resolve_resting_rate(101.0, 25, _ev(spec=66.0, roia=2, agreement=0.53), min_intervals=15)
    assert r["bpm"] is None and r["confidence"] == "uncertain"


def test_doubling_signature_needs_the_same_backing_as_the_head():
    backed = resolve_resting_rate(130.0, 25, _ev(spec=65.0, roia=MIN_ROI_AGREE_FOR_RATE, agreement=1.0, hf=0.4), min_intervals=15)
    assert backed["bpm"] == 65.0 and backed["confidence"] == "provisional"
    # a clean 2x is self-consistent evidence and needs no region quota
    unbacked = resolve_resting_rate(130.0, 25, _ev(spec=65.0, roia=2, agreement=1.0, hf=0.4), min_intervals=15)
    assert unbacked["bpm"] == 65.0 and unbacked["confidence"] == "provisional"
    # split-inflation (ratio ~1.5) without backing reports nothing
    split = resolve_resting_rate(101.0, 25, _ev(spec=69.0, roia=2, agreement=0.46, hf=0.35), min_intervals=15)
    assert split["bpm"] is None and split["confidence"] == "uncertain"


def test_decision_high_rate_fails_closed_when_the_count_is_unverified():
    """A uniformly doubled count reads ~2x the true rate; the decision used to
    call HIGH_RATE on it. With the resolver saying 'uncertain' (doubling
    signature, 2 backing regions) the scan must abstain, never call HIGH_RATE."""
    import numpy as np
    from inference.decision_logic import decide_with_rationale
    from configs import load_config
    from features.rhythm import RhythmFeatures
    cfg = load_config()
    feats = RhythmFeatures(values={"median_abs_succ_diff": 10.0, "pnn50": 0.02,
                                   "median_ibi": 60000.0 / 130.0, "dropout_rate": 0.0,
                                   "rmssd": 12.0, "n_intervals": 30},
                           n_intervals=30, mean_confidence=0.9, estimator_warnings=[])
    ev = {"cross_roi_coherence": 0.6, "timing_precision_ms": 15.0, "timing_matched_fraction": 0.9,
          "split_fraction": 0.0, "harmonic_fraction": 0.4, "pulse_lattice_bpm": 130.0,
          "pulse_spectral_bpm": 65.0, "pulse_spectral_roi_agree": 2, "pulse_agreement": 1.0,
          "pulse_spectral_snr": 3.0}
    res, why = decide_with_rationale(feats, 0.8, 0.9, cfg, recording_id="t", evidence=ev)
    assert res.outcome.value == "REPEAT_SCAN"
    assert res.predicted_class is None
    assert any("rate unverified" in r for r in res.no_read_reasons)
