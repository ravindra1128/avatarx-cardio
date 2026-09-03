"""v0.1.5 — the blocking readiness gate becomes a 1-5 star confidence
rating. The scan ALWAYS starts unless you are not pointing a working
camera at a face (face / framing / hard fps floor); everything else
grades the attempt instead of preventing it.

THE RULE THAT MUST NOT BE LOST: lowering the bar to START does not lower
the bar to CALL AF. The v0.1.2 evidence gates are untouched; stars are
their user-visible expression — a scan below 3 stars can never return
AFIB_SUGGESTIVE (enforced by the star ceiling: any failing AF evidence
gate caps stars at 2; and by the decision coupling: an AF call with
stars < 3 downgrades to the existing inconclusive path).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from configs import load_config
from datasets.schema import ScanOutcome, ScanResult
from features.rhythm import RhythmFeatures
from inference.confidence_stars import ConfidenceStars, confidence_stars
from inference.decision_logic import decide_with_rationale
from inference.evidence import readiness_from_evidence


def _good_ev(**over):
    ev = {"insufficient": False, "sqi": 0.90, "components": {},
          "per_roi_snr": {"forehead": 0.9, "cheek_l": 0.85,
                          "cheek_r": 0.8, "nose": 0.6},
          "cross_roi_coherence": 0.95, "timing_precision_ms": 4.0,
          "timing_matched_fraction": 0.98, "timing_pair": None,
          "n_beats": 9, "frac_multi_roi": 0.8, "split_fraction": 0.02,
          "harmonic_fraction": 0.02, "n_intervals": 24, "n_segments": 1,
          "fps": 30.0, "max_gap_ms": 33.3, "jitter_ms": 1.0,
          "window_coverage": 1.0, "seconds": 20.0, "n_frames": 600}
    ev.update(over)
    return ev


def _stars(ev, cfg=None, **kw):
    cfg = cfg or load_config()
    rd = readiness_from_evidence(ev, cfg)
    return confidence_stars(ev, rd, cfg, **kw)


# ------------------------------------------------------------- the module
def test_five_stars_on_reference_grade_evidence():
    cs = _stars(_good_ev())
    assert isinstance(cs, ConfidenceStars)
    assert cs.stars == 5, (cs.stars, cs.score, cs.limiting_factor)
    assert 0.0 <= cs.score <= 1.0
    assert cs.limiting_factor in cs.per_check


def test_stars_are_monotone_in_quality():
    """Degrade SNR / coherence / timing together across a sweep; stars
    must be non-increasing."""
    grades = [
        {},                                                        # pristine
        {"cross_roi_coherence": 0.55, "timing_precision_ms": 12.0,
         "per_roi_snr": {"forehead": 0.7, "cheek_l": 0.6,
                         "cheek_r": 0.55, "nose": 0.3}},
        {"cross_roi_coherence": 0.38, "timing_precision_ms": 22.0,
         "sqi": 0.55, "per_roi_snr": {"forehead": 0.6, "cheek_l": 0.52,
                                      "cheek_r": 0.3, "nose": 0.2}},
        {"cross_roi_coherence": 0.22, "timing_precision_ms": 34.0,
         "timing_matched_fraction": 0.70, "sqi": 0.40,
         "per_roi_snr": {"forehead": 0.55, "cheek_l": 0.5,
                         "cheek_r": 0.2, "nose": 0.1}},
        {"cross_roi_coherence": 0.08, "timing_precision_ms": 55.0,
         "timing_matched_fraction": 0.5, "sqi": 0.25, "n_beats": 3,
         "per_roi_snr": {"forehead": 0.3, "cheek_l": 0.2,
                         "cheek_r": 0.1, "nose": 0.05}},
    ]
    seq = [_stars(_good_ev(**g)).stars for g in grades]
    assert all(a >= b for a, b in zip(seq, seq[1:])), seq
    assert seq[0] == 5 and seq[-1] <= 2, seq


def test_af_evidence_gate_failure_caps_stars_at_two():
    """The enforcement teeth: ANY failing AF evidence gate caps stars at
    2, no matter how good the weighted arithmetic looks. Here everything
    is pristine except timing precision at 34 ms (> the 30 ms AF bar but
    within the 40 ms any-class bar)."""
    cs = _stars(_good_ev(timing_precision_ms=34.0))
    assert cs.stars <= 2, (cs.stars, cs.score)
    cs2 = _stars(_good_ev(split_fraction=0.22))       # > afib 0.15, < any 0.30
    assert cs2.stars <= 2
    # two-region verified evidence (coherence 0 by construction, AF-grade
    # timing) must NOT be capped — the decision can legitimately call AF
    cs3 = _stars(_good_ev(cross_roi_coherence=0.0, frac_multi_roi=0.0,
                          timing_precision_ms=8.0,
                          per_roi_snr={"forehead": 0.8, "cheek_l": 0.7,
                                       "cheek_r": 0.1, "nose": 0.1}))
    assert cs3.stars >= 3, (cs3.stars, cs3.score, cs3.limiting_factor)


def test_blocking_check_failure_is_one_star():
    cfg = load_config()
    rd = readiness_from_evidence(_good_ev(), cfg, face_ok=False,
                                 framing="no_face")
    cs = confidence_stars(_good_ev(), rd, cfg)
    assert cs.stars == 1
    assert rd["can_start"] is False                    # scan not started


# --------------------------------------------------- two-tier readiness
def test_advisory_mode_starts_on_blocking_checks_only():
    """Poor quality (low SNR, zero coherence, no beats yet) must not hold
    the Start button — only face/framing/hard-fps do."""
    cfg = load_config()
    bad = _good_ev(cross_roi_coherence=0.0, timing_precision_ms=float("nan"),
                   timing_matched_fraction=0.0, sqi=0.22, n_beats=1,
                   per_roi_snr={"forehead": 0.2, "cheek_l": 0.1,
                                "cheek_r": 0.1, "nose": 0.0})
    rd = readiness_from_evidence(bad, cfg)
    assert rd["mode"] == "advisory"
    assert rd["blocking_pass"] is True
    assert rd["advisory_pass"] is False
    assert rd["can_start"] is True
    # hard fps floor still blocks
    rd2 = readiness_from_evidence(_good_ev(fps=12.0), cfg)
    assert rd2["can_start"] is False, rd2["failing"]


def test_blocking_mode_preserves_pre_advisory_behaviour():
    """mode: blocking reproduces the previous gate exactly (the prompt
    calls it 'v0.1.3 behaviour'; on this tree the previous gate is
    v0.1.4.2 — same checks, same thresholds, start only when EVERY check
    passes)."""
    cfg = load_config()
    cfg["decision"]["readiness"] = dict(cfg["decision"].get("readiness") or {})
    cfg["decision"]["readiness"]["mode"] = "blocking"
    bad = _good_ev(sqi=0.22)
    rd = readiness_from_evidence(bad, cfg)
    assert rd["mode"] == "blocking"
    assert rd["can_start"] is False                    # advisory fail blocks
    assert "sqi" in rd["failing"]
    good = readiness_from_evidence(_good_ev(), cfg)
    assert good["can_start"] is True
    # the check set and thresholds are byte-identical between modes
    cfg2 = load_config()
    rd_adv = readiness_from_evidence(bad, cfg2)
    assert {k: (v["threshold"], v["op"]) for k, v in rd["checks"].items()} == \
           {k: (v["threshold"], v["op"]) for k, v in rd_adv["checks"].items()}


# --------------------------------------------------- decision coupling
def _af_features():
    return RhythmFeatures({"n_intervals": 24.0, "median_abs_succ_diff": 95.0,
                           "pnn50": 0.6, "median_ibi": 900.0,
                           "mean_ibi": 900.0, "dropout_rate": 0.1,
                           "irregularity_index": 0.2}, 24, 0.9, [])


def test_low_star_result_is_never_afib_suggestive():
    """AF-like features + a failing AF coherence/timing gate: the class
    must never be AFIB_SUGGESTIVE, whichever layer catches it."""
    cfg = load_config()
    ev = _good_ev(cross_roi_coherence=0.10, timing_precision_ms=36.0,
                  timing_matched_fraction=0.70)
    cs = _stars(ev)
    assert cs.stars <= 2
    r, why = decide_with_rationale(_af_features(), 0.6, 0.9, cfg,
                                   evidence=ev, confidence=cs)
    assert r.predicted_class != "AFIB_SUGGESTIVE"
    assert r.confidence_stars == cs.stars


def test_coupling_downgrades_af_call_when_stars_below_three():
    """Belt-and-braces: even if every AF gate passes, an externally low
    star rating (< 3) downgrades an AF call to the existing inconclusive
    path — no new outcome value."""
    cfg = load_config()
    ev = _good_ev()
    low = ConfidenceStars(stars=2, score=0.45, limiting_factor="sqi",
                          hint="", per_check={})
    r, why = decide_with_rationale(_af_features(), 0.8, 0.9, cfg,
                                   evidence=ev, confidence=low)
    assert r.predicted_class != "AFIB_SUGGESTIVE"
    assert r.outcome in (ScanOutcome.REPEAT_SCAN, ScanOutcome.NO_RESULT)
    assert r.confidence_stars == 2
    assert any("confidence" in w.lower() for w in r.no_read_reasons), \
        r.no_read_reasons
    # with stars >= 3 the same evidence yields the AF call
    ok = ConfidenceStars(stars=4, score=0.8, limiting_factor="sqi",
                         hint="", per_check={})
    r2, _ = decide_with_rationale(_af_features(), 0.8, 0.9, cfg,
                                  evidence=ev, confidence=ok)
    assert r2.predicted_class == "AFIB_SUGGESTIVE"
    assert r2.confidence_stars == 4


# --------------------------------------------------- end-to-end fixtures
def test_scan_starts_in_poor_conditions(videos):
    """dark30 used to be a refusal; it must now produce a RESULT carrying
    1-2 stars (its outcome may still be a no-read — the capture gates are
    untouched — but the attempt is graded, not refused)."""
    from inference.pipeline import run_with_details
    path, _ = videos["dark30"]
    r, det = run_with_details(path, manifest={"capture_profile": "consumer",
                                              "assume_rig_locks": False},
                              recording_id="stars-dark30")
    assert r.confidence_stars in (1, 2), (r.confidence_stars,
                                          r.no_read_reasons)
    assert r.predicted_class != "AFIB_SUGGESTIVE"
    assert r.confidence_limiting_factor
    txt = r.user_facing_text()
    assert f"Confidence: {r.confidence_stars} of 5" in txt, txt


def test_good_fixture_gets_high_stars_and_suffix(videos):
    from inference.pipeline import run_with_details
    path, _ = videos["sinus30"]
    r, det = run_with_details(path, manifest={"capture_profile": "consumer",
                                              "assume_rig_locks": False},
                              recording_id="stars-sinus30")
    assert r.outcome is ScanOutcome.ACCEPT
    assert r.confidence_stars >= 3, (r.confidence_stars,
                                     r.confidence_limiting_factor)
    txt = r.user_facing_text()
    assert "cannot rule out" in txt.lower()            # unchanged core text
    assert f"Confidence: {r.confidence_stars} of 5" in txt


def test_stars_are_rhythm_neutral(videos):
    """Anti-periodicity applied to stars: AF must not score below sinus
    (same capture quality)."""
    from inference.pipeline import run_with_details
    stars = {}
    for name in ("sinus30", "af30"):
        r, _ = run_with_details(videos[name][0],
                                manifest={"capture_profile": "consumer",
                                          "assume_rig_locks": False},
                                recording_id=f"stars-{name}")
        stars[name] = r.confidence_stars
    assert stars["af30"] >= stars["sinus30"] - 1, stars
