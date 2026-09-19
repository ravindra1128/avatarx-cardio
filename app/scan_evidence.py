"""Compact, non-classifying evidence audits. No files, waveform retention or state.

These reports do not feed the classifier. In particular, SDK self-consistency
does not establish beat accuracy, and a shared scan ID does not align two clocks.
"""
from __future__ import annotations

import math
import json

import numpy as np


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (ValueError, OverflowError):
            pass
    return None


def _ratio(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def _stats(runs):
    """Descriptive interval statistics; successive differences never cross gaps."""
    intervals = np.concatenate(runs) if runs else np.array([])
    differences = [np.diff(r) for r in runs if len(r) > 1]
    differences = np.concatenate(differences) if differences else np.array([])
    return {
        "intervals_n": int(intervals.size),
        "seconds": float(np.sum(intervals) / 1000),
        "longest_run_s": max((float(np.sum(r) / 1000) for r in runs), default=0.0),
        "bpm": float(np.median(60000 / intervals)) if intervals.size else None,
        "sdnn_ms": float(np.std(intervals, ddof=1)) if intervals.size > 1 else None,
        "successive_pairs_n": int(differences.size),
        "rmssd_ms": float(np.sqrt(np.mean(differences ** 2))) if differences.size else None,
        "mad_ms": float(np.median(np.abs(differences))) if differences.size else None,
        "pnn50": float(np.mean(np.abs(differences) > 50)) if differences.size else None,
    }


def video_duration_summary(doc, det, client_duration_ms=None):
    """Partition the retained frame clock, not the recorder's elapsed wall time.

    Excluded steps are gaps in the ROI clock under the CURRENT extraction rule.
    They can include frame wander or unusable faces, not only missing frames.
    The full-video clock is measured before those ROI exclusions. Endpoint spans
    exclude a final frame period everywhere.
    """
    from inference.evidence import capture_segments

    out = {"version": 1, "source": "video", "state": "unavailable",
           "client_recording_elapsed_s": _ratio(_number(client_duration_ms), 1000),
           "retained_span_s": None, "processing_segment_s": None,
           "discarded_fragment_s": None, "excluded_roi_step_s": None,
           "clean_interval_s": None, "clean_fraction_retained": None,
           "clean_fraction_processed": None}
    ing = det.get("ingest")
    out["interval_rejections"] = dict(getattr(det.get("runset"), "rejection_audit", {}) or {})
    clock = doc.get("clock") or {}
    first, last = _number(clock.get("first_frame_s")), _number(clock.get("last_frame_s"))
    span = last - first if first is not None and last is not None and last >= first else None
    out.update(retained_span_s=span, retained_frames_n=clock.get("n_frames"),
               frame_clock_source=clock.get("source"),
               repaired_timestamps_n=clock.get("repaired"),
               frame_step_p99_ms=_number(clock.get("frame_step_p99_ms")),
               frame_step_max_ms=_number(clock.get("frame_step_max_ms")))
    ts = np.asarray(getattr(ing, "timestamps_s", []), dtype=float)
    ev = det.get("evidence") or {}
    if ts.ndim != 1 or len(ts) < 2:
        out["reason"] = "retained frame timestamps unavailable"
        return out
    if not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
        out.update(state="invalid_clock", reason="non-finite or non-increasing frame timestamps")
        return out
    roi_span = float(ts[-1] - ts[0])
    out["roi_span_s"] = roi_span
    if span is not None and (ts[0] < first - 1e-6 or ts[-1] > last + 1e-6):
        out.update(state="inconsistent_clocks", reason="ROI timestamps fall outside the full-video clock")
        return out
    if "capture_segments" not in ev:
        out["reason"] = "video extraction did not produce segment evidence"
        return out
    fps = _number(getattr(getattr(ing, "meta", None), "measured_fps_mean", None))
    if fps is None or fps <= 0:
        out["reason"] = "extraction frame rate unavailable"
        return out
    # Same implementation, with ONLY its short-fragment exclusion removed for
    # accounting. The production extraction and its thresholds stay unchanged.
    fragments = capture_segments(ts, fps, min_seconds=0.0)
    accepted = [(float(a), float(b)) for a, b in ev["capture_segments"]]
    fragment_spans = {(float(ts[a]), float(ts[b - 1])) for a, b in fragments}
    if len(set(accepted)) != len(accepted) or not set(accepted).issubset(fragment_spans):
        out.update(state="inconsistent_segments", reason="extraction segments do not match the frame clock")
        return out
    processed_s = sum(b - a for a, b in accepted)
    fragment_s = sum(ts[b - 1] - ts[a] for a, b in fragments)
    covered_steps = np.zeros(len(ts) - 1, dtype=bool)
    for a, b in fragments:
        covered_steps[a:b - 1] = True
    clean = getattr(det.get("runset"), "runs", None)
    clean_s = sum(float(np.sum(r)) for r in clean) / 1000 if clean is not None else None
    if clean_s is not None and (not math.isfinite(clean_s) or clean_s < 0):
        clean_s = None
    dt = np.diff(ts)
    out.update(
        state="assessed", roi_frames_n=len(ts),
        roi_excluded_frames_n=clock["n_frames"] - len(ts) if isinstance(clock.get("n_frames"), int) else None,
        unobserved_edge_s=span - roi_span if span is not None else None,
        processing_segment_s=float(processed_s),
        processing_segments_n=len(accepted),
        discarded_fragment_s=float(max(0, fragment_s - processed_s)),
        discarded_fragments_n=len(fragments) - len(accepted),
        isolated_frames_n=len(ts) - sum(b - a for a, b in fragments),
        excluded_roi_step_s=float(np.sum(dt[~covered_steps])),
        excluded_roi_steps_n=int(np.sum(~covered_steps)),
        unprocessed_s=float(span - processed_s) if span is not None else None,
        roi_step_median_ms=float(np.median(dt) * 1000),
        roi_step_p99_ms=float(np.percentile(dt, 99) * 1000),
        roi_step_max_ms=float(np.max(dt) * 1000),
        clean_interval_s=clean_s,
        clean_fraction_retained=_ratio(clean_s, span),
        clean_fraction_processed=_ratio(clean_s, processed_s),
        decision_coverage_denominator_s=_number(ev.get("captured_seconds")),
        clean_duration_exceeds_processed=clean_s is not None and clean_s > processed_s + 1e-6,
    )
    return out


def shenai_evidence_summary(raw, det):
    """Assess the supplied train even when video accepts or prevents routing.

    Report original-order defects before any sorting/cleaning. The raw-duration
    arm includes each reported start/end pair, including the final interval;
    the existing route arm uses start-to-start intervals and its exact cleaner.
    Neither is a clinical reference for the other.
    """
    from inference import shenai_route as route

    out = {"version": 1, "state": "not_received", "mode": "diagnostic_only",
           "timing": {"beat_clock": "sdk_relative_seconds",
                      "video_alignment": "unknown", "verified_overlap_s": None,
                      "comparison_scope": "same_scan_unmatched_windows"},
           "limitations": ["per_beat_confidence_unavailable",
                           "sdk_self_consistency_does_not_validate_rhythm",
                           "sdk_video_clock_mapping_unavailable"],
           "issues": []}
    if not isinstance(raw, dict):
        return out
    hb = raw.get("heartbeats")
    if not isinstance(hb, list):
        out.update(state="invalid_input", issues=["heartbeats_not_an_array"])
        return out
    out.update(state="assessed" if hb else "empty_train", input_beats_n=len(hb))
    ppg = raw.get("ppg") if isinstance(raw.get("ppg"), dict) else {}
    out["timing"].update(ppg_fs_hz=_number(ppg.get("fs_hz")),
                         ppg_fs_source=str(ppg.get("fs_source") or "unknown")[:80],
                         ppg_t0_source=str(ppg.get("t0_source") or "unknown")[:80])
    ref = raw.get("reference") if isinstance(raw.get("reference"), dict) else {}
    sdk = {name: _number(ref.get(key)) for name, key in (
        ("quality", "average_signal_quality"), ("bad_signal_s", "bad_signal_seconds"),
        ("hr_bpm", "heart_rate_bpm"), ("sdnn_ms", "hrv_sdnn_ms"),
        ("lnrmssd", "hrv_lnrmssd_ms"))}
    try:
        sdk["rmssd_ms"] = _number(math.exp(sdk["lnrmssd"])) if sdk["lnrmssd"] is not None else None
    except OverflowError:
        sdk["rmssd_ms"] = None
    out["sdk"] = sdk
    for key in ("quality", "bad_signal_s", "hr_bpm", "rmssd_ms"):
        if sdk[key] is None:
            out["issues"].append(f"sdk_{key}_unavailable")
    if sdk["quality"] is not None and not 0 <= sdk["quality"] <= 1:
        out["issues"].append("sdk_quality_out_of_range")
    if sdk["bad_signal_s"] is not None and sdk["bad_signal_s"] < 0:
        out["issues"].append("sdk_bad_signal_seconds_negative")

    runs, valid, prev, duration_errors = [], [], None, []
    invalid_n = order_n = gaps_n = overlaps_n = missing_duration_n = 0
    gap_s = 0.0
    for b in hb:
        s = _number(b.get("start_location_sec")) if isinstance(b, dict) else None
        e = _number(b.get("end_location_sec")) if isinstance(b, dict) else None
        if s is None or e is None or s < 0 or e <= s or not math.isfinite((e - s) * 1000):
            invalid_n += 1
            prev = None
            continue
        rr = (e - s) * 1000
        declared = _number(b.get("duration_ms"))
        if declared is None:
            missing_duration_n += 1
        else:
            duration_errors.append(abs(declared - rr))
        connected = False
        if prev is not None:
            delta = s - prev[1]
            order_n += int(s <= prev[0])
            gaps_n += int(delta > route.DROPPED_BEAT_GAP_S)
            overlaps_n += int(delta < -route.DROPPED_BEAT_GAP_S)
            gap_s += max(0, delta)
            connected = s > prev[0] and abs(delta) <= route.DROPPED_BEAT_GAP_S
        if not connected:
            runs.append([])
        runs[-1].append(rr)
        valid.append((s, e))
        prev = (s, e)
    span = max(e for _, e in valid) - min(s for s, _ in valid) if valid else None
    out["integrity"] = {
        "valid_beats_n": len(valid), "invalid_beats_n": invalid_n,
        "non_increasing_starts_n": order_n, "gap_boundaries_n": gaps_n,
        "overlap_boundaries_n": overlaps_n, "positive_boundary_gap_s": gap_s,
        "continuity_tolerance_s": route.DROPPED_BEAT_GAP_S,
        "span_s": span, "segments_n": len(runs),
        "first_beat_start_s": valid[0][0] if valid else None,
        "last_beat_end_s": valid[-1][1] if valid else None,
        "missing_duration_n": missing_duration_n,
        "duration_disagreement_max_ms": max(duration_errors) if duration_errors else None,
    }
    for issue, n in (("invalid_beats", invalid_n), ("non_increasing_starts", order_n),
                     ("overlapping_beats", overlaps_n), ("discontinuous_train", gaps_n)):
        if n:
            out["issues"].append(issue)
    if duration_errors and max(duration_errors) > 1.0:
        # A diagnostic precision check (1 ms), never an AFib validity gate.
        out["issues"].append("reported_duration_disagrees_with_timestamps")
    if ppg.get("truncated") is True:
        out["issues"].append("ppg_truncated")
    unfiltered = _stats([np.asarray(r, float) for r in runs])
    out["reported_intervals"] = unfiltered
    out["internal_comparison"] = {
        "hr_delta_bpm": unfiltered["bpm"] - sdk["hr_bpm"]
        if unfiltered["bpm"] is not None and sdk["hr_bpm"] is not None else None,
        "rmssd_relative_error": _ratio(abs(unfiltered["rmssd_ms"] - sdk["rmssd_ms"]), sdk["rmssd_ms"])
        if unfiltered["rmssd_ms"] is not None and sdk["rmssd_ms"] is not None else None,
        "sdnn_delta_ms": unfiltered["sdnn_ms"] - sdk["sdnn_ms"]
        if unfiltered["sdnn_ms"] is not None and sdk["sdnn_ms"] is not None else None,
        "matched_windows_verified": False,
    }
    if invalid_n or order_n or overlaps_n or not valid:
        out["current_route_train"] = {"state": "not_evaluated_invalid_or_empty_train"}
        return out
    from beats.ibi import clean_runs
    from configs import load_config

    cfg = det.get("config") or load_config()
    rc = cfg["runs"]
    train = route.train_from_sidecar(raw)
    lo, hi = rc["ibi_physiologic_ms"]
    rs = clean_runs(route.train_series(train, train.get("fs_hz")),
                    min_conf=float(det.get("min_conf", 0.5)),
                    min_run_beats=int(rc["min_run_beats"]),
                    max_physiologic_ibi_ms=float(hi), min_physiologic_ibi_ms=float(lo),
                    missed_beat_ratio=float(rc["missed_ratio"]))
    cleaned = _stats(rs.runs)
    cleaned.update(state="assessed", coverage=_ratio(cleaned["seconds"], train["span_s"]),
                   split_fraction=float(rs.split_fraction),
                   checks_failed=route.train_checks(train, cleaned["rmssd_ms"]))
    out["current_route_train"] = cleaned
    out["internal_comparison"]["clean_to_reported_rmssd_ratio"] = _ratio(
        cleaned["rmssd_ms"], unfiltered["rmssd_ms"])
    # Same-scan rate comparison only; neither overlap nor beat agreement is known.
    minimum = int(cfg.get("decision", {}).get("evidence", {}).get("min_intervals_any", 15))
    out["current_route_rate_check"] = route.corroborate_rate(
        cleaned["bpm"] if cleaned["intervals_n"] >= 4 else None, det.get("evidence") or {}, minimum)
    return out


def record_summary(doc, key, callback, *args):
    """An audit failure must never skip or change the scan's decision."""
    try:
        summary = callback(*args)
        # Non-finite input or an unexpected numpy object must not corrupt JSON.
        json.dumps(summary, allow_nan=False)
    except Exception as error:  # noqa: BLE001
        summary = {"version": 1, "state": "assessment_failed", "error_type": type(error).__name__}
    doc.setdefault("debug", {})[key] = summary
