"""M1.3 — head_rate_flags: sustained brady/tachy from the CLEAN-RUN rate
with the same abstention discipline as every other output."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from beats.lattice import BeatLattice, LATTICE_VERSION
from datasets.schema import RATE_FLAG_SENTENCES
from heads.base import get_head


def _lat(runs):
    runs = [np.asarray(r, float) for r in runs]
    return BeatLattice(
        version=LATTICE_VERSION, fps=30.0, duration_s=30.0,
        beat_t_s=np.array([]), beat_confidence=np.array([]),
        beat_agreement=np.array([]), runs=runs,
        run_confidences=[np.ones(max(len(r) - 1, 1)) for r in runs],
        n_intervals=int(sum(len(r) for r in runs)), dropout_rate=0.0,
        split_fraction=0.0, per_roi_times={}, segments=[])


def test_sustained_brady_and_tachy_flag():
    h = get_head("rate_flags")
    r = h.run(_lat([np.full(20, 1400.0)]), {})        # ~43 bpm sustained
    assert r.value["flag"] == "BRADY"
    assert r.value["sentence_key"] == "brady"
    assert r.measurement_class.value == "MEASURED"
    r2 = h.run(_lat([np.full(20, 520.0)]), {})        # ~115 bpm sustained
    assert r2.value["flag"] == "TACHY"
    assert r2.value["sentence_key"] == "tachy"


def test_normal_rate_and_transients_do_not_flag():
    h = get_head("rate_flags")
    r = h.run(_lat([np.full(24, 850.0)]), {})         # ~71 bpm
    assert r.value["flag"] is None
    # median below 50 but NOT sustained (only 60% of intervals slow)
    mixed = np.concatenate([np.full(12, 1300.0), np.full(8, 900.0)])
    r2 = h.run(_lat([mixed]), {})
    assert r2.value["flag"] is None


def test_abstains_below_interval_floor_with_reason():
    h = get_head("rate_flags")
    r = h.run(_lat([np.full(6, 1400.0)]), {})         # only 6 intervals
    assert r.value["flag"] is None and r.value["sentence_key"] is None
    assert any("clean intervals" in w for w in r.reasons)


def test_sentences_are_sanctioned_and_never_diagnose():
    banned = ["you have", "you are in", "diagnosed with", "confirmed afib",
              "confirms", "is atrial fibrillation", "bradycardia",
              "tachycardia"]                     # no condition NAMES either
    assert set(RATE_FLAG_SENTENCES) == {"brady", "tachy"}
    for k, s in RATE_FLAG_SENTENCES.items():
        low = s.lower()
        assert not any(b in low for b in banned), (k, s)
        assert "clinician" in low                 # escalation path


# --- cross-check-verified reported rate (2026-09-14, owner-directed) ---------
def _ev(lattice_bpm, spectral_bpm, roi_agree):
    """A production-shaped evidence dict for the pulse cross-check."""
    pa = abs(lattice_bpm - spectral_bpm) / spectral_bpm
    return {"evidence": {
        "pulse_lattice_bpm": lattice_bpm, "pulse_spectral_bpm": spectral_bpm,
        "pulse_agreement": round(pa, 4), "pulse_spectral_roi_agree": roi_agree}}


def test_agree_reports_mean_of_count_and_spectral():
    h = get_head("rate_flags")
    r = h.run(_lat([np.full(20, 833.0)]), _ev(72, 70, 4))   # ~72 bpm, agree
    assert r.value["rate_confidence"] == "verified"
    assert r.value["rate_source"] == "count_spectral_mean"
    assert abs(r.value["median_bpm"] - 71.0) < 0.6          # mean(72, 70)
    assert r.value["flag"] is None


def test_disagree_with_regional_backing_reports_spectral_provisional():
    h = get_head("rate_flags")
    r = h.run(_lat([np.full(20, 600.0)]), _ev(100, 68, 3))  # count 100, spec 68
    assert r.value["median_bpm"] == 68.0                    # the waveform rate
    assert r.value["rate_source"] == "waveform_rhythm"
    assert r.value["rate_confidence"] == "provisional"
    assert r.value["flag"] is None                          # no flag off one number


def test_doubled_count_does_not_raise_a_false_tachy_flag():
    """The regression this change exists for: a doubled beat count (100 bpm)
    that the waveform (55 bpm, 4/4 regions) contradicts must NOT report 100 or
    flag TACHY."""
    h = get_head("rate_flags")
    r = h.run(_lat([np.full(20, 600.0)]), _ev(100, 55, 4))
    assert r.value["flag"] is None
    assert r.value["median_bpm"] == 55.0


def test_disagree_without_regional_backing_is_uncertain():
    h = get_head("rate_flags")
    r = h.run(_lat([np.full(20, 600.0)]), _ev(100, 55, 2))  # only 2/4 regions
    assert r.value["median_bpm"] is None
    assert r.value["flag"] is None
    assert r.value["rate_confidence"] == "uncertain"


def test_unresolved_crosscheck_is_uncertain():
    h = get_head("rate_flags")
    r = h.run(_lat([np.full(20, 600.0)]), _ev(100, 55, 1))  # 1/4 -> unresolved
    assert r.value["median_bpm"] is None
    assert r.value["rate_confidence"] == "uncertain"


def test_no_evidence_preserves_legacy_lattice_behaviour():
    """Fixture/CLI callers pass no evidence: the head must behave as before —
    clean-interval median with brady/tachy flags allowed."""
    h = get_head("rate_flags")
    r = h.run(_lat([np.full(20, 520.0)]), {})               # ~115 bpm, no ev
    assert r.value["flag"] == "TACHY"
    assert r.value["rate_confidence"] == "unverified"
    assert abs(r.value["median_bpm"] - 115.4) < 0.6
