"""The Vascular Tone card is computed from the live-frame traces after the
video job (app/measure_api._apply_vascular_tone, 2026-09-23): only that card
changes, the response contract holds, and a scan without traces says why
instead of showing the retired clip number."""
from __future__ import annotations

import copy
import os
import sys

import pytest

from app import measure_api as api
from app.report_data import replace_biomarker, report_biomarkers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_facial_perfusion import make_doc  # noqa: E402  (tests/ is not a package here)


def _legacy_biomarkers():
    """The payload the video job builds, with an old clip-based tone card."""
    hemo = {
        "quality": {"kind": "signal_evidence_not_endpoint_accuracy"},
        "arterial_stiffness": {"available": True, "tier": "provisional",
                               "estimate": {"value": 61.0, "unit": "/100", "name": "x"}},
        "vascular_tone": {"available": True, "tier": "provisional", "amplitude_cv": 0.453,
                          "score": 45.3, "n_beats": 9,
                          "estimate": {"value": 45.3, "unit": "/100", "name": "vascular_tone_index"},
                          "raw_name": "normalized_pulse_amplitude_variability",
                          "raw_value": 45.3, "raw_unit": "% CV"},
        "cardiorespiratory_fitness": {"available": True, "tier": "measured",
                                      "estimate": {"value": 55.2, "unit": "/100", "name": "y"}},
    }
    return report_biomarkers(None, {}, hemodynamics=hemo)


def _doc():
    return {"outcome": "REPEAT_SCAN", "biomarkers": _legacy_biomarkers(),
            "debug": {"shenai_input": {"sdk_hr_bpm": 72.0}}}


def _item(doc, key):
    return next(i for i in doc["biomarkers"]["items"] if i["key"] == key)


def test_tone_card_replaced_from_traces_and_others_untouched():
    doc = _doc()
    before = copy.deepcopy(doc["biomarkers"]["items"])
    api._apply_vascular_tone(doc, make_doc(pi_pct=0.5), {})
    tone = _item(doc, "vascular_tone")
    assert tone["status"] == "computed" and tone["unit"] == "/100"
    assert tone["raw"]["name"] == "facial_perfusion_index_percent"
    assert tone["raw"]["unit"] == "%" and 0.1 < tone["raw"]["value"] < 1.0
    assert tone["details"]["source"] == "live_frame_traces"
    assert tone["details"]["legacy_clip_cv"]["score"] == 45.3
    assert tone["tier"] == "measured"
    assert 1 <= tone["confidence"]["confidence_stars"] <= 5
    for key in ("arterial_stiffness", "cardiorespiratory_fitness"):
        assert _item(doc, key) == next(i for i in before if i["key"] == key)
    dbg = doc["debug"]["vascular_tone"]
    assert dbg["available"] and dbg["legacy_clip_score"] == 45.3
    assert dbg["f0_source"] in ("trace_peak_near_reference_rate", "live_frame_reference_rate")
    # The frozen contract: same row keys as every other card.
    assert set(tone) >= {"key", "title", "label", "status", "value", "unit", "metric",
                         "method", "calibrated", "confidence", "details", "tier",
                         "tier_reasons", "typical_range", "raw"}


def test_no_traces_is_not_computed_with_a_reason_never_the_clip_number():
    doc = _doc()
    api._apply_vascular_tone(doc, None, {})
    tone = _item(doc, "vascular_tone")
    assert tone["status"] == "not_computed" and tone["value"] is None
    assert "did not reach the service" in tone["reason"]
    assert doc["biomarkers"]["complete"] is False


def test_clip_source_switch_keeps_the_legacy_card():
    doc = _doc()
    old = api.TONE_SOURCE
    try:
        api.TONE_SOURCE = "clip"
        api._apply_vascular_tone(doc, make_doc(), {})
    finally:
        api.TONE_SOURCE = old
    assert _item(doc, "vascular_tone")["value"] == 45.3
    assert doc["debug"]["vascular_tone"]["source"] == "clip"


def test_estimator_failure_never_leaves_the_clip_number(monkeypatch):
    import features.facial_perfusion as fpm

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(fpm, "facial_perfusion", boom)
    doc = _doc()
    api._apply_vascular_tone(doc, make_doc(), {})
    tone = _item(doc, "vascular_tone")
    assert tone["status"] == "not_computed"
    assert "failed safely" in tone["reason"]


def test_hint_prefers_the_sdk_rate_then_the_request_reference():
    assert api._tone_hint_bpm({"debug": {"shenai_input": {"sdk_hr_bpm": 64.0}}},
                              {"reference": {"ref_hr": 80}}) == 64.0
    assert api._tone_hint_bpm({}, {"reference": {"ref_hr": "80"}}) == 80.0
    assert api._tone_hint_bpm({}, {}) is None


def test_replace_biomarker_is_a_no_op_on_an_error_payload():
    assert replace_biomarker({"error": "x"}, "vascular_tone", {}) == {"error": "x"}


def test_sheet_row_carries_the_index_and_the_legacy_score():
    from app import result_sheet as rs
    doc = _doc()
    api._apply_vascular_tone(doc, make_doc(pi_pct=0.5), {})
    row = rs.row_from_doc(doc)
    assert row["VT Source"] == "live_frame_traces"
    assert 0.1 < float(row["VT PI %"]) < 1.0
    assert row["VT Legacy Clip"] in (45.3, "45.3")
    assert row["Vascular Tone"] == _item(doc, "vascular_tone")["value"]
    assert set(k for k in row if k.startswith("VT ")) <= set(rs.COLUMNS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
