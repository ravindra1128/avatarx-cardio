"""Independent source reports; never select or overwrite a published result."""
from copy import deepcopy
import math
from time import perf_counter

from app.afib_response import finalize_afib_response
from inference import shenai_route


def _decision(doc, rationale):
    outcome = doc.get("outcome")
    result = None
    if outcome in {"ACCEPT", "REPEAT_SCAN", "NO_RESULT"}:
        result = finalize_afib_response({"outcome": outcome,
                                        "predicted_class": doc.get("predicted_class")})["afib_result"]
    return {"state": "assessed" if result else "unavailable", "result": result,
            "outcome": outcome, "predicted_class": doc.get("predicted_class"),
            "gates_failed": deepcopy(rationale.get("gates_failed") or []),
            "reasons": list(doc.get("no_read_reasons") or []),
            "features": {key: float(value) if isinstance(value, (int, float))
                         and not isinstance(value, bool) and math.isfinite(value) else None
                         for key, value in (rationale.get("features") or {}).items()}}


def compare_sources(doc, det, raw, route_enabled):
    """Assess the SDK candidate even when source-selection policy blocks it.

    This is a conditional assessment using the existing video quality evidence,
    not an independent SDK diagnosis or a comparison of synchronized beats.
    Only a small isolated working document is passed to the mutating route core.
    """
    started = perf_counter()
    debug = doc.get("debug") or {}
    video = debug.get("video_outcome") or doc
    rationale = debug.get("video_rationale") or det.get("rationale") or debug.get("rationale") or {}
    allowed, reason = shenai_route.video_path_allows_route(rationale, video.get("outcome"))
    audit = debug.get("shenai_assessment") or {}
    sdk = {"state": "unavailable", "result": None, "gates_failed": [],
           "reason": "no ShenAI sidecar arrived", "publication_policy_eligible": bool(allowed and route_enabled),
           "publication_blocker": "route disabled (AFIB_SHENAI_ROUTE)" if not route_enabled else (None if allowed else reason),
           "integrity": deepcopy(audit.get("integrity") or {}),
           "clean_intervals": deepcopy(audit.get("current_route_train") or {}),
           "timing": deepcopy(audit.get("timing") or {})}
    out = {"version": 1, "mode": "diagnostic_only", "state": "assessed",
           "contributes_to_published_result": False, "video": _decision(video, rationale),
           "shenai_train": sdk, "window_alignment": "unknown", "verified_overlap_s": None,
           "decision_agreement": "not_comparable",
           "limitations": ["SDK candidate shares video quality checks",
                           "source windows are not synchronized", "SDK per-beat confidence is unavailable"]}
    out["video"]["window"] = deepcopy(debug.get("video_duration") or {})
    try:
        if isinstance(raw, dict):
            integrity = audit.get("integrity") or {}
            if audit.get("state") != "assessed":
                sdk.update(state="not_evaluated", reason="SDK integrity assessment unavailable or invalid")
            elif any(integrity.get(k) for k in ("invalid_beats_n", "non_increasing_starts_n", "overlap_boundaries_n")):
                sdk.update(state="not_evaluated", reason="SDK beat ordering or integrity is invalid")
            else:
                isolated_det = deepcopy({k: det[k] for k in
                    ("config", "evidence", "rationale", "sqi", "min_conf") if k in det})
                working = deepcopy({k: doc.get(k) for k in ("recording_id", "signal_quality_index")})
                working.update(deepcopy({k: video.get(k) for k in
                    ("outcome", "predicted_class", "no_read_reasons")}))
                working["debug"] = {"rationale": deepcopy(rationale),
                                    "evidence": deepcopy(isolated_det.get("evidence") or {})}
                rec = {"attempted": False, "used": False, "waited_s": 0.0}
                shenai_route._evaluate_train(working, isolated_det, raw, rec)
                sdk["reason"] = rec.get("reason")
                sdk["rate_check"] = rec.get("rate_check")
                if rec.get("used"):
                    sdk.update(_decision(working, working["debug"]["rationale"]))
                    sdk["reason"] = "candidate assessed under existing physiological checks; diagnostic only"
                else:
                    sdk["state"] = "not_evaluated"
        if out["video"]["result"] is not None and sdk["result"] is not None:
            out["decision_agreement"] = "agree" if out["video"]["result"] == sdk["result"] else "disagree"
    except Exception as error:  # Diagnostics cannot fail the completed scan.
        sdk.update(state="assessment_failed", result=None, reason=type(error).__name__)
    out["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
    return out
