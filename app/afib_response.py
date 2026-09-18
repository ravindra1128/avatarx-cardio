"""One public AFib decision after interval-source selection.

Presentation only: never reruns inference or supplies a physiological decision
for an operational error. Existing fields and non-AFib heads remain intact.
"""
from copy import deepcopy


def finalize_afib_response(doc):
    if doc.get("error"):
        return doc
    if doc.get("outcome") not in {"ACCEPT", "REPEAT_SCAN", "NO_RESULT"}:
        raise ValueError("analysis has no valid terminal outcome")
    result = "INCONCLUSIVE"
    if doc["outcome"] == "ACCEPT":
        if doc.get("predicted_class") == "AFIB_SUGGESTIVE":
            result = "AFIB_DETECTED"
        elif doc.get("predicted_class") in {"SINUS", "HIGH_RATE"}:
            result = "AFIB_NOT_DETECTED"
    doc["afib_result"] = result
    fields = ("outcome", "predicted_class", "afib_probability",
              "confidence_stars", "user_facing_text")
    for head in doc.get("head_results") or []:
        if not isinstance(head, dict) or head.get("head") != "afib":
            continue
        if doc.get("rhythm_source") == "shenai_train":
            doc.setdefault("debug", {}).setdefault("video_afib_head", deepcopy(head))
        head["value"] = {**(head.get("value") or {}), **{key: doc.get(key) for key in fields},
                         "afib_result": result}
        head["reasons"] = list(doc.get("no_read_reasons") or [])
        stars = doc.get("confidence_stars")
        head["confidence"] = stars / 5.0 if isinstance(stars, (int, float)) and not isinstance(stars, bool) else None
    return doc


def capture_diagnostic_summary(raw):
    """Keep a small allowlist of client summaries; never raw arrays or IDs."""
    import math
    source = raw.get("capture_diagnostics") if isinstance(raw, dict) else None
    if not isinstance(source, dict) or source.get("version") != 1:
        return {"version": 1, "state": "unavailable"}
    schema = {
        "version": None, "capture_version": None, "scope": None, "quality_semantics": None,
        "sdk_version": None, "probe_state": None, "frame_callback_supported": None,
        "state_transitions": None, "hidden_events": None, "hidden_ms": None, "elapsed_ms": None,
        "settings": dict.fromkeys(["width", "height", "frameRate", "aspectRatio", "facingMode", "exposureMode", "whiteBalanceMode"]),
        "quality": dict.fromkeys(["n", "min", "max", "mean", "invalid_n"]),
        "frames": dict.fromkeys(["observations", "unobserved_presented_frames", "backward_steps", "duplicate_steps",
                                 "media_span_s", "callback_span_s", "limitation"]),
    }
    distribution = dict.fromkeys(["n", "mean_ms", "p50_ms", "p99_ms", "max_ms", "histogram_resolution_ms", "histogram_cap_ms"])
    schema["frames"].update({k: distribution for k in ("observed_media_steps", "adjacent_frame_steps", "callback_steps")})
    def keep(values, keys):
        out = {}
        for key, nested in keys.items():
            value = values.get(key)
            if nested is not None:
                if isinstance(value, dict):
                    out[key] = keep(value, nested)
            elif value is None or isinstance(value, bool):
                out[key] = value
            elif isinstance(value, str):
                out[key] = value[:160]
            elif isinstance(value, (int, float)):
                try:
                    if math.isfinite(value):
                        out[key] = value
                except OverflowError:
                    pass
        return out
    return {**keep(source, schema), "state": "reported", "client_reported": True}
