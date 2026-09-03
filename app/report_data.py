"""Report/API input assembly.

The waveform, beat lattice, and intervals are measured pipeline details.
The three resting biomarker rows are explicitly uncalibrated Research
Estimate / Prototype outputs computed by ``features.hemodynamics`` from
the endpoint-usable details that survived the scan.  This module serializes them; it never
synthesizes a signal or substitutes a display default for a missing value.
"""
from __future__ import annotations

import numpy as np

from datasets.schema import RATE_FLAG_SENTENCES
from rppg.pos import pos_pulse

VERIFIED_CONF = 0.5
PHYSIO_MS = (250.0, 2200.0)
MAX_JOIN_GAP_S = 2.5

_BIOMARKERS = (
    ("arterial_stiffness", "Arterial Stiffness"),
    ("vascular_tone", "Vascular Tone"),
    ("cardiorespiratory_fitness",
     "VO₂ Max / Cardiorespiratory Fitness Estimate"),
)

_BIOMARKER_METHODS = {
    "arterial_stiffness": (
        "pulse-contour reflection index; at sufficient native frame rate, "
        "second-derivative aging index"
    ),
    "vascular_tone": (
        "coefficient of variation of per-beat facial pulse amplitude after "
        "within-region median normalization"
    ),
    "cardiorespiratory_fitness": (
        "resting-heart-rate and RMSSD research proxy; heart-rate-only "
        "fallback is identified explicitly and RMSSD is never imputed"
    ),
}


def _failure_reason(endpoint: dict, hemo: dict) -> str:
    for key in ("reason", "reason_vasomotion"):
        if endpoint.get(key):
            return str(endpoint[key])
    reasons = list(hemo.get("reasons") or [])
    return (str(reasons[0]) if reasons else
            "required waveform evidence was unavailable")


def report_biomarkers(scan_result, det: dict, *, capture: dict = None,
                      participant: dict = None,
                      hemodynamics: dict = None) -> dict:
    """Build the one canonical API/report payload for resting biomarkers.

    The calculation remains in ``features.hemodynamics``.  This function
    only serializes its real outputs into a stable UI contract and makes an
    abstention explicit.  No default, random, or imputed number is created.
    """
    outcome = getattr(getattr(scan_result, "outcome", None), "value", None)
    if outcome is None:
        outcome = str(getattr(scan_result, "outcome", "NO_RESULT"))
    if hemodynamics is None:
        try:
            from features.hemodynamics import resting_hemodynamics
            hemodynamics = resting_hemodynamics(
                det or {}, outcome=outcome, participant=participant,
                capture=capture or {})
        except Exception as exc:                         # fail visibly
            hemodynamics = {
                "available": False, "outcome": outcome,
                "reasons": [f"biomarker computation failed: "
                            f"{type(exc).__name__}: {exc}"],
            }
    hemo = hemodynamics or {}
    items = []
    for key, title in _BIOMARKERS:
        endpoint = dict(hemo.get(key) or {})
        estimate = dict(endpoint.get("estimate") or {})
        value = estimate.get("value")
        computed = bool(endpoint.get("available")) and value is not None
        confidence = dict(endpoint.get("confidence") or
                          hemo.get("quality") or
                          hemo.get("quality_gate") or {})
        if not computed:
            confidence.setdefault(
                "kind", "signal_evidence_not_endpoint_accuracy")
            confidence.setdefault(
                "interpretation",
                "The endpoint abstained because its measured signal "
                "requirements were not met; no display value was substituted.")
        row = {
            "key": key,
            "title": title,
            "label": "Research Estimate / Prototype",
            "status": "computed" if computed else "not_computed",
            "value": value if computed else None,
            "unit": estimate.get("unit") if computed else None,
            "metric": estimate.get("name") if computed else None,
            "method": (estimate.get("method") or
                       endpoint.get("definition") or
                       _BIOMARKER_METHODS[key]),
            "calibrated": bool(endpoint.get("calibrated", False)),
            "confidence": confidence,
            "details": endpoint,
        }
        if not computed:
            row["reason"] = _failure_reason(endpoint, hemo)
        elif outcome != "ACCEPT":
            row["warning"] = (
                f"Overall scan outcome was {outcome}. This numeric research "
                "feature survived its endpoint inputs, but the rhythm result "
                "was rejected; treat the value as low-confidence paired-data "
                "collection only."
            )
        if key == "cardiorespiratory_fitness":
            row["oxygen_uptake_ml_kg_min"] = endpoint.get(
                "oxygen_uptake_estimate")
            row["limitation"] = endpoint.get(
                "why_no_oxygen_uptake_value")
        elif key == "arterial_stiffness":
            row["limitation"] = endpoint.get("not_a_velocity")
        elif key == "vascular_tone":
            row["limitation"] = endpoint.get("not_reactivity")
        items.append(row)
    return {
        "schema_version": 1,
        "label": "Research Estimate / Prototype",
        "scope": "single resting facial rPPG scan",
        "outcome": outcome,
        "scan_accepted": outcome == "ACCEPT",
        "complete": all(x["status"] == "computed" for x in items),
        "quality": dict(hemo.get("quality") or {}),
        "items": items,
    }


def report_signal(det: dict, cfg: dict) -> dict:
    """The renderer's `signal` payload from run_with_details() details."""
    ing = det.get("ingest")
    lat = det.get("lattice")
    if ing is None or not getattr(ing, "ok", False):
        return {}
    fps = float(ing.meta.measured_fps_mean)
    ts = np.asarray(ing.timestamps_s, float)
    segments = list((det.get("evidence") or {}).get("capture_segments")
                    or ([(float(ts[0]), float(ts[-1]))] if ts.size else []))
    t_out, v_out = [], []
    band = tuple(cfg["sqi"]["band_hz"])
    for a_t, b_t in segments:
        keep = (ts >= a_t) & (ts <= b_t)
        if keep.sum() < int(2 * fps):
            continue
        wave = pos_pulse(ing.traces["forehead"][keep], fps, band=band)
        sd = float(np.std(wave)) or 1.0
        t_out += [round(float(x), 3) for x in ts[keep]]
        v_out += [round(float(v / sd), 3) for v in wave]

    beats = []
    intervals = []
    if lat is not None:
        bt = np.asarray(lat.beat_t_s, float)
        bc = np.asarray(lat.beat_confidence, float)
        beats = [{"t": round(float(t), 3), "conf": round(float(c), 3)}
                 for t, c in zip(bt, bc)]
        ok = bc >= VERIFIED_CONF
        vt, vc = bt[ok], bc[ok]
        for i in range(1, vt.size):
            dt_s = float(vt[i] - vt[i - 1])
            ms = dt_s * 1000.0
            if dt_s <= MAX_JOIN_GAP_S and PHYSIO_MS[0] <= ms <= PHYSIO_MS[1]:
                intervals.append({"t": round(float(vt[i]), 3),
                                  "ms": round(ms, 1),
                                  "conf": round(float(min(vc[i],
                                                          vc[i - 1])), 3)})
    return {"t_s": t_out, "values": v_out, "fps": fps,
            "beats": beats, "intervals": intervals,
            "segments": [[round(float(a), 3), round(float(b), 3)]
                         for a, b in segments]}


def report_capture_meta(scan_result, det: dict, cfg: dict,
                        extra: dict = None) -> dict:
    """Header fields + signal + presentation flags for the renderer."""
    cm = dict(scan_result.capture_meta or {})
    ing = det.get("ingest")
    if ing is not None and getattr(ing, "meta", None) is not None:
        cm.setdefault("measured_fps", float(ing.meta.measured_fps_mean))
        cm["lighting"] = f"~{ing.meta.lux_proxy:.0f} lux (proxy)"
        ts = np.asarray(ing.timestamps_s, float)
        if ts.size > 1:
            cm.setdefault("scan_seconds", round(float(ts[-1] - ts[0]), 1))
    cm.setdefault("device", cm.get("tracker"))
    for h in (scan_result.head_results or []):
        if h.get("head") == "rate_flags":
            key = (h.get("value") or {}).get("sentence_key")
            if key in RATE_FLAG_SENTENCES:
                cm["rate_flag_sentence"] = RATE_FLAG_SENTENCES[key]
    cm["report_mode"] = str((cfg.get("report") or {})
                            .get("mode", "findings_only"))
    cm["show_tachogram"] = bool(((cfg.get("report") or {})
                                 .get("show_tachogram", True)))
    cm["signal"] = report_signal(det, cfg)
    cm.update(extra or {})
    if "biomarkers" not in cm:
        cm["biomarkers"] = report_biomarkers(
            scan_result, det,
            capture={"exposure_locked": cm.get("exposure_locked"),
                     "awb_locked": cm.get("awb_locked")},
            participant=cm.get("participant"))
    return cm
