"""Evidence accounting and SDK assessment must never change a decision."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app import scan_evidence as audit, measure_api as api, result_sheet
from inference import shenai_route
from inference.evidence import capture_segments
from test_shenai_route import _video, _sidecar, _regular


def video_details(ts, runs=()):
    segments = capture_segments(ts, 30)
    return {
        "ingest": SimpleNamespace(timestamps_s=np.asarray(ts), meta=SimpleNamespace(measured_fps_mean=30.0)),
        "runset": SimpleNamespace(runs=[np.array(r) for r in runs]),
        "evidence": {"capture_segments": [(ts[a], ts[b - 1]) for a, b in segments],
                     "captured_seconds": sum(ts[b - 1] - ts[a] for a, b in segments)},
    }


def clock_doc(ts):
    return {"clock": {"first_frame_s": float(ts[0]), "last_frame_s": float(ts[-1]),
                      "n_frames": len(ts)}}


def test_duration_partition_exposes_high_coverage_on_short_processed_window():
    # Eight seconds survive; eighteen separated two-second fragments do not.
    ts = np.concatenate([np.arange(241) / 30] + [9 + 3 * i + np.arange(61) / 30 for i in range(18)])
    det = video_details(ts, [[762.5] * 8])
    out = audit.video_duration_summary(clock_doc(ts), det, 130000)
    assert out["state"] == "assessed"
    assert out["processing_segment_s"] == pytest.approx(8)
    assert out["discarded_fragment_s"] == pytest.approx(36)
    assert out["excluded_roi_step_s"] == pytest.approx(18)
    assert out["retained_span_s"] == pytest.approx(62)
    assert out["clean_fraction_processed"] == pytest.approx(6.1 / 8)
    assert out["clean_fraction_retained"] == pytest.approx(6.1 / 62)
    assert out["client_recording_elapsed_s"] == 130
    assert sum(out[k] for k in ("processing_segment_s", "discarded_fragment_s", "excluded_roi_step_s")) == pytest.approx(out["retained_span_s"])


def test_uniform_video_has_no_unaccounted_duration_and_no_decision_mutation():
    ts = np.arange(1201) / 30
    det = video_details(ts, [[800] * 40])
    doc = {**clock_doc(ts), "outcome": "REPEAT_SCAN", "debug": {"evidence": det["evidence"]}}
    before = copy.deepcopy(doc)
    out = audit.video_duration_summary(doc, det)
    assert doc == before
    assert out["retained_span_s"] == 40 and out["processing_segment_s"] == 40
    assert out["clean_fraction_retained"] == 0.8
    assert out["excluded_roi_steps_n"] == 0 and out["discarded_fragments_n"] == 0


def test_isolated_frames_and_no_segments_are_visible_not_relabelled_processed():
    ts = np.array([0.0, 1.0, 2.0])
    out = audit.video_duration_summary(clock_doc(ts), video_details(ts))
    assert out["isolated_frames_n"] == 3 and out["excluded_roi_steps_n"] == 2
    assert out["processing_segment_s"] == 0
    assert out["clean_fraction_processed"] is None
    assert out["clean_fraction_retained"] == 0


@pytest.mark.parametrize("ts", [[0, 0, 1], [0, 1, 0.5], [0, float("nan"), 1]])
def test_bad_clock_is_explicit(ts):
    out = audit.video_duration_summary({}, video_details(ts))
    assert out["state"] == "invalid_clock" and out["retained_span_s"] is None


def test_early_capture_failure_does_not_invent_duration_or_clean_beats():
    out = audit.video_duration_summary({}, {})
    assert out["state"] == "unavailable"
    assert out["clean_interval_s"] is None and out["clean_fraction_retained"] is None


def test_full_video_duration_includes_frames_omitted_by_face_tracking():
    full = np.arange(1501) / 30  # 50 s video
    roi = full[150:-150]        # only 40 s has usable face regions
    out = audit.video_duration_summary(clock_doc(full), video_details(roi, [[800] * 25]))
    assert out["retained_span_s"] == 50 and out["roi_span_s"] == 40
    assert out["roi_excluded_frames_n"] == 300 and out["unobserved_edge_s"] == 10
    assert out["clean_fraction_retained"] == 0.4
    assert out["clean_fraction_processed"] == 0.5


def test_missing_full_clock_never_substitutes_roi_span_for_retained_video():
    out = audit.video_duration_summary({}, video_details(np.arange(1201) / 30, [[800] * 25]))
    assert out["roi_span_s"] == 40
    assert out["retained_span_s"] is None and out["clean_fraction_retained"] is None


def test_reordered_or_foreign_segments_cannot_pass_accounting():
    det = video_details(np.arange(1201) / 30)
    det["evidence"]["capture_segments"] = [(0, 40), (0, 40)]
    assert audit.video_duration_summary({}, det)["state"] == "inconsistent_segments"


def test_full_sdk_interval_arm_keeps_final_interval_and_reproduces_its_hrv():
    raw = _sidecar([800, 820, 780, 810, 800] * 10)
    original = copy.deepcopy(raw)
    out = audit.shenai_evidence_summary(raw, {})
    assert raw == original
    assert out["state"] == "assessed"
    assert out["reported_intervals"]["intervals_n"] == 50
    assert out["current_route_train"]["intervals_n"] == 49
    assert out["internal_comparison"]["rmssd_relative_error"] == pytest.approx(0, abs=1e-10)
    assert out["integrity"]["gap_boundaries_n"] == 0
    assert out["timing"]["video_alignment"] == "unknown"
    assert out["timing"]["verified_overlap_s"] is None
    assert out["mode"] == "diagnostic_only" and "predicted_class" not in out
    json.dumps(out, allow_nan=False)
    assert "heartbeats" not in json.dumps(out) and "start_location_sec" not in json.dumps(out)


def test_hrv_never_bridges_a_missing_beat_or_invalid_entry():
    raw = _sidecar([800] * 20 + [1000] * 20, drop_at=20)
    out = audit.shenai_evidence_summary(raw, {})
    assert out["integrity"]["gap_boundaries_n"] == 1
    assert out["reported_intervals"]["rmssd_ms"] == pytest.approx(0, abs=1e-8)
    raw = _sidecar([800] * 20 + [1000] * 20)
    raw["heartbeats"].insert(20, {"start_location_sec": None})
    out = audit.shenai_evidence_summary(raw, {})
    assert out["integrity"]["invalid_beats_n"] == 1
    assert out["reported_intervals"]["rmssd_ms"] == pytest.approx(0, abs=1e-8)
    assert out["current_route_train"]["state"] == "not_evaluated_invalid_or_empty_train"


@pytest.mark.parametrize("bad", [None, True, "0.8", float("nan"), float("inf")])
def test_missing_or_malformed_sdk_values_are_not_zero_or_quality_evidence(bad):
    raw = _sidecar(_regular(50))
    raw["heartbeats"][10]["start_location_sec"] = bad
    raw["reference"]["average_signal_quality"] = bad
    out = audit.shenai_evidence_summary(raw, {})
    assert out["integrity"]["invalid_beats_n"] == 1
    assert out["sdk"]["quality"] is None
    assert "sdk_quality_unavailable" in out["issues"]
    json.dumps(out, allow_nan=False)


def test_duplicate_order_overlap_and_duration_defects_survive_assessment():
    raw = _sidecar(_regular(50))
    raw["heartbeats"].insert(10, copy.deepcopy(raw["heartbeats"][9]))
    raw["heartbeats"][0]["duration_ms"] = 10
    out = audit.shenai_evidence_summary(raw, {})
    assert out["integrity"]["non_increasing_starts_n"] == 1
    assert out["integrity"]["overlap_boundaries_n"] == 1
    assert "reported_duration_disagrees_with_timestamps" in out["issues"]
    assert out["current_route_train"]["state"] == "not_evaluated_invalid_or_empty_train"


@pytest.mark.parametrize("outcome,gates", [("ACCEPT", ()), ("REPEAT_SCAN", ("timing_precision_any_class",)),
                                         ("NO_RESULT", ("sqi",)), ("REPEAT_SCAN", ("coverage",))])
def test_every_route_outcome_gets_assessed_without_changing_the_existing_decision(outcome, gates, monkeypatch):
    monkeypatch.setattr(api, "SHENAI_ROUTE_ON", True)
    monkeypatch.setenv("AFIB_KEEP_UPLOADS", "0")
    raw = _sidecar(_regular(60))
    doc, det = _video(gates_failed=gates, outcome=outcome)
    baseline = copy.deepcopy(doc)
    expected_route = shenai_route.evaluate(baseline, det, raw)
    api._apply_shenai_route(doc, det, "audit-test", attachment={
        "signal_delivery": {"version": 1, "state": "attached"}, "shenai_signals": raw})
    assert doc["debug"]["shenai_assessment"]["current_route_train"]["intervals_n"] > 15
    assert doc["debug"].pop("shenai_route") == expected_route
    doc["debug"].pop("shenai_input")
    doc["debug"].pop("shenai_assessment")
    assert doc["debug"].pop("client_capture_diagnostics")["state"] == "unavailable"
    baseline.setdefault("rhythm_source", "video")
    assert doc == baseline


def test_disabled_route_still_assesses_attached_train(monkeypatch):
    monkeypatch.setattr(api, "SHENAI_ROUTE_ON", False)
    monkeypatch.setattr(shenai_route, "evaluate", lambda *a, **kw: pytest.fail("route ran"))
    doc, det = _video(gates_failed=["coverage"])
    api._apply_shenai_route(doc, det, "audit-test", attachment={
        "signal_delivery": {"version": 1, "state": "attached"}, "shenai_signals": _sidecar(_regular(50))})
    assert doc["debug"]["shenai_assessment"]["state"] == "assessed"
    assert doc["debug"]["shenai_route"]["used"] is False


def test_audit_failure_is_explicit_and_never_skips_the_route(monkeypatch):
    monkeypatch.setattr(api, "SHENAI_ROUTE_ON", True)
    monkeypatch.setattr(audit, "shenai_evidence_summary", lambda *a: (_ for _ in ()).throw(ValueError()))
    seen = []
    monkeypatch.setattr(shenai_route, "evaluate", lambda *a, **kw: seen.append(True) or {"used": False})
    doc = {"outcome": "REPEAT_SCAN"}
    api._apply_shenai_route(doc, {}, "audit-test", attachment={
        "signal_delivery": {"version": 1, "state": "missing_snapshot"}})
    assert seen and doc["outcome"] == "REPEAT_SCAN"
    assert doc["debug"]["shenai_assessment"] == {"version": 1, "state": "assessment_failed", "error_type": "ValueError"}


def test_nonfinite_diagnostic_cannot_break_json():
    doc = {"outcome": "ACCEPT"}
    audit.record_summary(doc, "test", lambda: {"value": float("inf")})
    assert doc["debug"]["test"]["state"] == "assessment_failed"
    json.dumps(doc, allow_nan=False)


def test_sheet_separates_video_duration_from_selected_sdk_rhythm_and_handles_legacy():
    doc = {"outcome": "ACCEPT", "rhythm_source": "shenai_train", "analysed_seconds": 38,
           "debug": {"video_duration": {"state": "assessed", "retained_span_s": 44.5,
                                        "processing_segment_s": 7.82, "clean_interval_s": 6.1,
                                        "clean_fraction_retained": 6.1 / 44.5},
                     "shenai_assessment": audit.shenai_evidence_summary(_sidecar(_regular(50)), {})}}
    row = result_sheet.row_from_doc(doc)
    assert row["Analysed s"] == 38 and row["Video Clean s"] == 6.1
    assert row["Video Whole Coverage"] == 0.1371
    assert row["SDK Video Alignment"] == "unknown" and row["SDK Invalid Beats"] == 0
    assert row["SDK Clean Intervals"] == 49
    legacy = result_sheet.row_from_doc({"outcome": "NO_RESULT"})
    assert legacy["Video Whole Coverage"] == "" and legacy["SDK Clean Intervals"] == ""
    assert set(row) == set(result_sheet.COLUMNS) and len(result_sheet.COLUMNS) == len(set(result_sheet.COLUMNS))
