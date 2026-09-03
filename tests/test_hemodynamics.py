"""v0.8: resting hemodynamic indices from ONE scan.

What the physics allows, refused where it does not: the second
derivative needs real sampling rate, the vasomotion band needs a long
enough envelope, and no oxygen-uptake number is produced at all.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re

import numpy as np
import pytest

from features.hemodynamics import (MIN_AMPLITUDE_BEATS, aging_index,
                                   amplitude_series, autonomic_index,
                                   cardiorespiratory_indices,
                                   resting_hemodynamics, resting_rate_index,
                                   stiffness_contour,
                                   vasomotor_indices)

VO2_RE = re.compile(r"m[lL]\s*/\s*kg\s*/\s*min|vo2|vo₂", re.I)


# ------------------------------------------------------- stiffness
def test_aging_index_is_the_takazawa_formula():
    c = {"sdppg_b_over_a": -0.8, "sdppg_c_over_a": -0.1,
         "sdppg_d_over_a": -0.3, "sdppg_e_over_a": 0.2}
    assert aging_index(c) == pytest.approx(-0.8 + 0.1 + 0.3 - 0.2)
    # a stiffer contour (b less negative, d less negative) reads higher
    stiff = dict(c, sdppg_b_over_a=-0.4, sdppg_d_over_a=-0.05)
    assert aging_index(stiff) > aging_index(c)


def test_aging_index_refuses_a_missing_or_infinite_component():
    base = {"sdppg_b_over_a": -0.8, "sdppg_c_over_a": -0.1,
            "sdppg_d_over_a": -0.3, "sdppg_e_over_a": 0.2}
    for k in base:
        c = dict(base, **{k: None})
        assert aging_index(c) is None, k
    assert aging_index(dict(base, sdppg_c_over_a=float("nan"))) is None
    assert aging_index({}) is None


def test_stiffness_contour_says_what_it_is_and_is_not():
    out = stiffness_contour({"sdppg_b_over_a": -0.8, "sdppg_c_over_a": -0.1,
                             "sdppg_d_over_a": -0.3, "sdppg_e_over_a": 0.2,
                             "reflection_index": 0.4, "rise_time_s": 0.2})
    assert out["available"] is True
    assert out["calibrated"] is False
    assert out["aging_index"] is not None and out["reflection_index"] == 0.4
    assert "Takazawa" in out["definition"]
    assert "m/s" in out["not_a_velocity"]
    # nothing here is a velocity
    assert "pwv" not in json.dumps(out).lower()
    empty = stiffness_contour({})
    assert empty["available"] is False and empty["reason"]


# ------------------------------------------------------------ tone
def _series(n, *, base=1.0, mod=0.0, f=0.1, dt=0.85, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    a = base + mod * np.sin(2 * np.pi * f * t) + rng.normal(0, noise, n)
    return list(zip(t.tolist(), a.tolist()))


def test_amplitude_cv_and_vasomotion_band():
    flat = _series(80, mod=0.0, noise=0.001)
    out = vasomotor_indices(flat, True)
    assert out["available"] is True and out["optics_locked"] is True
    assert out["amplitude_cv"] < 0.01
    # a 0.1 Hz modulation sits in the vasomotion band and dominates it
    modulated = _series(80, mod=0.25, f=0.1, noise=0.002)
    out2 = vasomotor_indices(modulated, True)
    assert out2["amplitude_cv"] > out["amplitude_cv"]
    assert out2["vasomotion_index"] > 0.7
    # a 0.35 Hz modulation is OUTSIDE the band and must not count
    fast = _series(80, mod=0.25, f=0.35, noise=0.002)
    assert vasomotor_indices(fast, True)["vasomotion_index"] < 0.3


def test_tone_refuses_too_few_beats_and_too_short_an_envelope():
    few = vasomotor_indices(_series(MIN_AMPLITUDE_BEATS - 1), True)
    assert few["available"] is False and "usable beats" in few["reason"]
    short = vasomotor_indices(_series(20, dt=0.85), True)   # ~16 s span
    assert short["available"] is True
    assert short["amplitude_cv"] is not None
    assert short["vasomotion_index"] is None
    assert "cannot be resolved" in short["reason_vasomotion"]


def test_unlocked_optics_are_flagged_not_hidden():
    out = vasomotor_indices(_series(80), False)
    assert out["optics_locked"] is False
    assert "exposure and white balance" in out["caveat"]
    assert out["available"] is True          # reported, with the caveat
    assert "REACTIVITY" in out["not_reactivity"]


def test_amplitude_series_is_self_normalised_per_roi():
    """Two ROIs with wildly different optical scale must pool to the
    same normalised series: the scale is the confound being removed."""
    ts = np.arange(0, 40, 0.01)
    windows, waves = {}, {}
    for roi, scale in (("a", 1.0), ("b", 250.0)):
        w = np.zeros(ts.size)
        wins = []
        for k in range(20):
            i = k * 85
            w[i:i + 40] = scale * np.hanning(40)
            wins.append((i, i + 40))
        waves[roi], windows[roi] = w, wins
    out = amplitude_series(waves, ts, windows)
    assert len(out) == 20
    assert all(abs(a - 1.0) < 1e-6 for _, a in out)


# ------------------------------------------------- cardiorespiratory
def test_autonomic_index_is_bounded_and_monotone():
    assert autonomic_index(None, 40) is None
    assert autonomic_index(60, None) is None
    assert autonomic_index(0, 40) is None and autonomic_index(60, 0) is None
    mid = autonomic_index(60, 40)
    assert 0.0 < mid < 1.0 and mid == pytest.approx(0.5, abs=0.02)
    # a lower resting pulse reads higher; a higher RMSSD reads higher
    assert autonomic_index(50, 40) > mid > autonomic_index(80, 40)
    assert autonomic_index(60, 80) > mid > autonomic_index(60, 20)
    for hr in (35, 60, 120):
        for rm in (5, 40, 200):
            assert 0.0 < autonomic_index(hr, rm) < 1.0


def test_resting_fitness_proxy_is_numeric_without_fabricating_oxygen_uptake():
    """The real resting proxy is usable now; oxygen uptake stays null."""
    class _Reg:
        dispersion = {"rmssd_ms": 42.0, "sdnn_ms": 55.0}
    out = cardiorespiratory_indices(_Reg(), 58.0,
                                    {"age_years": 44, "sex": "F"})
    assert out["available"] is True
    assert out["resting_hr_bpm"] == 58.0 and out["rmssd_ms"] == 42.0
    assert out["autonomic_index"] is not None
    assert out["fitness_proxy_score"] == pytest.approx(
        100.0 * out["autonomic_index"], abs=0.11)
    assert out["estimate"]["value"] == out["fitness_proxy_score"]
    assert out["estimate"]["label"] == "Research Estimate / Prototype"
    assert out["estimate"]["unit"] == "/100"
    assert out["oxygen_uptake_estimate"] is None
    assert out["model_artifact"] is None
    assert out["calibrated"] is False
    assert "not derivable from a resting scan" in \
        out["why_no_oxygen_uptake_value"]
    # what a fitted model still needs is named, not hidden
    assert set(out["missing_for_a_fitted_model"]) == {
        "height_cm", "weight_kg", "habitual_activity"}
    # and the fenced token appears nowhere in the payload
    assert not VO2_RE.search(json.dumps(out))


def test_cardiorespiratory_without_inputs_reports_no_index():
    out = cardiorespiratory_indices(None, None)
    assert out["available"] is False and out["autonomic_index"] is None
    assert out["oxygen_uptake_estimate"] is None
    assert out["estimate"] is None


def test_cardiorespiratory_uses_measured_hr_without_imputing_rmssd():
    out = cardiorespiratory_indices(None, 66.0)
    assert out["available"] is True
    assert out["autonomic_index"] is None
    assert out["rmssd_ms"] is None
    assert out["resting_rate_index"] == resting_rate_index(66.0)
    assert out["fitness_proxy_score"] == pytest.approx(
        100.0 * out["resting_rate_index"], abs=0.11)
    assert out["fitness_proxy_basis"] == "resting_hr_only"
    assert "RMSSD was unavailable and was not imputed" in \
        out["estimate"]["method"]


# ------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def scans(tmp_path_factory):
    pytest.importorskip("cv2")
    from inference.pipeline import run_with_details
    from scripts.make_synth_video import synth_video
    d = tmp_path_factory.mktemp("hemo")
    out = {}
    for fps in (30.0, 60.0):
        p = d / f"s{int(fps)}.avi"
        synth_video(str(p), kind="sinus", fps=fps, duration_s=45.0, seed=11)
        res, det = run_with_details(str(p), manifest={},
                                    recording_id=f"h{int(fps)}")
        out[fps] = (res, det)
    return out


def test_all_three_families_come_from_one_resting_scan(scans):
    res, det = scans[60.0]
    h = resting_hemodynamics(det, outcome=res.outcome.value,
                             capture={"exposure_locked": True,
                                      "awb_locked": True})
    assert h["available"] is True and h["n_beats_used"] >= 8
    assert h["arterial_stiffness"]["available"] is True
    assert h["arterial_stiffness"]["estimate"]["value"] is not None
    assert h["arterial_stiffness"]["aging_index"] is not None
    assert h["vascular_tone"]["available"] is True
    assert h["vascular_tone"]["estimate"]["value"] is not None
    assert h["vascular_tone"]["amplitude_cv"] is not None
    assert h["cardiorespiratory_fitness"]["available"] is True
    assert h["cardiorespiratory_fitness"]["estimate"]["value"] is not None
    assert h["cardiorespiratory_fitness"]["resting_hr_bpm"] > 40
    for key in ("arterial_stiffness", "vascular_tone",
                "cardiorespiratory_fitness"):
        c = h[key]["confidence"]
        assert c["kind"] == "signal_evidence_not_endpoint_accuracy"
        assert c["signal_quality_index"] is not None
        assert c["n_beats_used"] >= 8 and c["n_rois_used"] >= 2
    assert not VO2_RE.search(json.dumps(h, default=str))


def test_the_second_derivative_is_refused_at_consumer_frame_rate(scans):
    """30 fps cannot carry the SDPPG waves: the aging index is None and
    says so, rather than being interpolated into existence."""
    res, det = scans[30.0]
    h = resting_hemodynamics(det, outcome=res.outcome.value)
    assert h["available"] is True
    assert h["sdppg_derivable"] is False
    assert h["arterial_stiffness"]["aging_index"] is None
    assert h["arterial_stiffness"]["available"] is True
    assert h["arterial_stiffness"]["estimate"]["value"] is not None
    res60, det60 = scans[60.0]
    h60 = resting_hemodynamics(det60, outcome=res60.outcome.value)
    assert h60["sdppg_derivable"] is True
    assert h60["arterial_stiffness"]["aging_index"] is not None


def test_rejected_rhythm_scan_keeps_endpoint_usable_research_values(scans):
    res, det = scans[60.0]
    h = resting_hemodynamics(det, outcome="NO_RESULT")
    assert h["available"] is True
    assert h["quality"]["overall_rhythm_accepted"] is False
    assert all(h[k]["available"] for k in
               ("arterial_stiffness", "vascular_tone",
                "cardiorespiratory_fitness"))
    assert all(h[k]["estimate"]["value"] is not None for k in
               ("arterial_stiffness", "vascular_tone",
                "cardiorespiratory_fitness"))
    assert "overall rhythm scan was not accepted" in \
        h["quality"]["interpretation"]
    from types import SimpleNamespace
    from app.report_data import report_biomarkers
    payload = report_biomarkers(
        SimpleNamespace(outcome=SimpleNamespace(value="NO_RESULT")), det,
        hemodynamics=h)
    assert payload["complete"] is True
    assert payload["scan_accepted"] is False
    assert all(x["status"] == "computed" and x["value"] is not None and
               x.get("warning") for x in payload["items"])
    # Hard-invalid input still fails visibly rather than manufacturing values.
    assert resting_hemodynamics({}, outcome="ACCEPT")["available"] is False


def test_limited_cross_region_evidence_keeps_real_multi_roi_values(scans):
    """A completed 2-star rhythm scan can retain endpoint morphology.

    These values mirror the evidence range seen in a real 30 fps browser
    scan: below the rhythm coherence quota, but with non-trivial agreement
    and morphology surviving independently in multiple facial regions.
    """
    res, det = scans[30.0]
    limited = dict(det)
    limited["evidence"] = dict(
        det["evidence"], cross_roi_coherence=0.17,
        timing_precision_ms=36.7, timing_matched_fraction=0.387)
    h = resting_hemodynamics(limited, outcome="REPEAT_SCAN")
    assert h["available"] is True
    assert h["n_beats_used"] >= 8 and h["n_rois_used"] >= 2
    assert h["quality"]["endpoint_evidence"]["mode"] == \
        "limited_endpoint_morphology"
    assert all(h[k]["available"] and h[k]["estimate"]["value"] is not None
               for k in ("arterial_stiffness", "vascular_tone",
                          "cardiorespiratory_fitness"))


def test_endpoint_limited_path_still_rejects_unverified_regions(scans):
    """Good lighting/SQI alone cannot manufacture cross-region evidence."""
    _, det = scans[30.0]
    unverified = dict(det)
    unverified["evidence"] = dict(
        det["evidence"], cross_roi_coherence=0.09,
        timing_precision_ms=41.0, timing_matched_fraction=0.34)
    h = resting_hemodynamics(unverified, outcome="REPEAT_SCAN")
    assert h["available"] is False
    assert h["quality_gate"]["evidence_mode"] == "unverified"
    assert "independently verified" in h["reasons"][0]


def test_abstention_still_exposes_method_and_signal_gate(scans):
    from types import SimpleNamespace
    from app.report_data import report_biomarkers

    _, det = scans[30.0]
    unverified = dict(det)
    unverified["evidence"] = dict(
        det["evidence"], cross_roi_coherence=0.0,
        timing_precision_ms=90.0, timing_matched_fraction=0.1)
    h = resting_hemodynamics(unverified, outcome="REPEAT_SCAN")
    payload = report_biomarkers(
        SimpleNamespace(outcome=SimpleNamespace(value="REPEAT_SCAN")),
        unverified, hemodynamics=h)
    assert all(x["status"] == "not_computed" and x["method"] and
               x["confidence"].get("pass") is False
               for x in payload["items"])
