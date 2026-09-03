"""v0.8: the five GATED research tracks as a block of the results
screen's RESEARCH / DEBUG METRICS panel - screen-only, never printed,
never in the consumer payload's head_results.

Owner instruction (2026-09-01, spec B.24, quoted verbatim): "Provide
all VO2 max, atrial flutter, arterial stiffness, vascular tone,
rhythm-regularity in the report for research and investigation purposes
only."

The tracks are rhythm regularity (§R), the atrial-flutter pattern flag
(§F), arterial stiffness (vascular), vasomotor tone (§W) and
cardiorespiratory fitness (§V). Each value rides with its track's LIVE
gate status, so a number and the reason it may not be shown to anyone
are never separated.

A red gate does NOT suppress these values: it is printed beside them.
That is the point - an investigator must see the number and its
standing together.

Three of the five now carry values from a SINGLE resting scan
(`features/hemodynamics.py`, v0.8): contour markers of arterial
stiffness, dimensionless resting vasomotor indices, and resting
cardiorespiratory indices. All are UNCALIBRATED and say so. The two
things a resting scan cannot give - a carotid-femoral velocity in m/s
and vasomotor REACTIVITY - are named as such, with the command that
produces the full research document.

THE FENCES, none of which this module moves:
1. It is a RESEARCH surface, on by default at the owner's direction
   (`report.research_tracks`, env AVATARX_RESEARCH_TRACKS): the panel
   that carries it has been labelled "not validated outputs, not for
   participants, excluded from the printed report" since v0.1.
   AVATARX_RESEARCH_TRACKS=0 turns it off for a participant-facing
   session.
2. Screen only: the panel is `display:none` in print CSS, and
   `render_report()` - the Cardiac Rhythm Scan Report, the printed and
   exported document - never sees this block.
3. Its own payload key (`research_tracks`), never `head_results`: the
   consumer boundary `datasets.schema.public_head_results` keeps doing
   exactly what it did.
4. Numbers, labels and the heads' own reasons - no sentence about the
   person. Participant-facing wording still comes solely from
   `ScanResult.user_facing_text()`.

app/ cannot import research/ (quarantine), so the cfPWV model and the
provocation-based reactivity head stay behind `cli.py research-report`.
"""
from __future__ import annotations

import os

WATERMARK = ("GATED RESEARCH TRACKS — RESEARCH ARTIFACT — FOR RESEARCH "
             "AND INVESTIGATION PURPOSES ONLY — NOT VALIDATED — NOT A "
             "MEASUREMENT — NOT A DIAGNOSIS — NOT FOR PARTICIPANT USE")

PANEL_NOTE = ("Every track below is gated: while its promotion reads "
              "BLOCKED nothing in it is a finding about the person "
              "scanned, and none of it may be repeated to them. Every "
              "value is uncalibrated and carries the definition it was "
              "computed from. The printed report never carries this "
              "block.")

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def research_tracks_enabled(cfg=None) -> bool:
    """Explicit env wins in BOTH directions (the v0.4 lesson: "0" must
    mean OFF even if the config says true); otherwise the config."""
    env = os.environ.get("AVATARX_RESEARCH_TRACKS", "").strip().lower()
    if env in _TRUE:
        return True
    if env in _FALSE:
        return False
    if cfg is None:
        from configs import load_config
        cfg = load_config()
    return bool((cfg.get("report") or {}).get("research_tracks"))


def _gates(spec: str) -> dict:
    """The track's live gate status, FAIL-CLOSED: any error reads as
    BLOCKED with the error recorded."""
    import importlib
    mod, fn = spec.split(":")
    try:
        doc = getattr(importlib.import_module(mod), fn)()
        return {"promotion": doc.get("promotion", "BLOCKED"),
                "gates_version": doc.get("gates_version"),
                "clinical_signoff": bool(doc.get("clinical_signoff")),
                "gates": [{"gate": g.get("gate"),
                           "title": g.get("title"),
                           "status": g.get("status"),
                           "first_reason": (g.get("reasons") or [None])[0]}
                          for g in (doc.get("gates") or [])]}
    except Exception as e:                       # noqa: BLE001 — fail closed
        return {"promotion": "BLOCKED", "gates_version": None,
                "clinical_signoff": False, "gates": [],
                "error": f"{type(e).__name__}: {e} — fail closed"}


def _respiration(video_path):
    """The torso-motion second decode the rhythm tracks need (never the
    facial trace the intervals come from - that would be circular).
    None when there is no video or no decodable torso."""
    if not video_path:
        return None
    try:
        from rppg._filters import moving_average_detrend
        from rppg.respiration import (respiratory_rate_from_motion,
                                      torso_motion_series)
        t, y, fps = torso_motion_series(str(video_path))
        rate, conc = respiratory_rate_from_motion(t, y, fps)
        return {"t": t, "y": moving_average_detrend(y, int(12.0 * fps)),
                "rate_brpm": rate, "quality": conc}
    except Exception:                            # noqa: BLE001 — optional
        return None


def _track(key, title, head, gates, *, status, value=None, reasons=None,
           needs=None, note=None, extra=None) -> dict:
    row = {"key": key, "title": title, "head": head, "status": status,
           "gates": gates, "watermark": WATERMARK}
    if value is not None:
        row["value"] = value
    if reasons:
        row["reasons"] = list(reasons)
    if needs:
        row["needs"] = needs
    if note:
        row["note"] = note
    if extra:
        row.update(extra)
    return row


def research_tracks_block(scan_result, det, cfg=None, *, video_path=None,
                          session_result=None, capture=None,
                          participant=None) -> dict:
    """The block, or {} when the flag is off. Never raises: a research
    panel must not be able to break a results screen."""
    if not research_tracks_enabled(cfg):
        return {}
    try:
        return _build(scan_result, det, cfg, video_path, session_result,
                      capture, participant)
    except Exception as e:                       # noqa: BLE001
        return {"watermark": WATERMARK, "note": PANEL_NOTE,
                "error": f"research tracks unavailable "
                         f"({type(e).__name__}): {e}",
                "tracks": []}


def _build(scan_result, det, cfg, video_path, session_result, capture,
           participant) -> dict:
    from heads import get_head
    from heads.base import head_status

    det = det or {}
    lat = det.get("lattice")
    reg = det.get("regularity")
    outcome = getattr(getattr(scan_result, "outcome", None), "value", None)
    resp = _respiration(video_path)
    no_read = list(getattr(scan_result, "no_read_reasons", []) or [])
    rid = getattr(scan_result, "recording_id", None) or "<recording>"

    # the three families a SINGLE resting scan can carry, uncalibrated
    try:
        from features.hemodynamics import resting_hemodynamics
        hemo = resting_hemodynamics(det, outcome=outcome,
                                    participant=participant,
                                    capture=capture)
    except Exception as e:                       # noqa: BLE001 — fail soft
        hemo = {"available": False,
                "reasons": [f"resting hemodynamics unavailable "
                            f"({type(e).__name__}): {e}"]}
    hemo_reasons = list(hemo.get("reasons") or [])
    tracks = []

    # ---- rhythm regularity (§R)
    g = _gates("evaluation.regularity_gates:regularity_gate_status")
    if lat is not None:
        hr = get_head("regularity").run(lat, {
            "regularity": reg, "scan_outcome": outcome,
            "respiration": resp, "cfg": cfg or {}}).to_dict()
        tracks.append(_track(
            "rhythm_regularity", "Rhythm regularity (v0.7, §R)",
            "regularity", g, status=head_status(hr),
            value=hr.get("value"), reasons=hr.get("reasons")))
    else:
        tracks.append(_track(
            "rhythm_regularity", "Rhythm regularity (v0.7, §R)",
            "regularity", g, status="not_run", reasons=no_read,
            note=f"the scan produced no beat lattice (outcome {outcome}); "
                 "the head never ran"))

    # ---- atrial flutter (§F)
    g = _gates("evaluation.flutter_gates:flutter_gate_status")
    if lat is not None:
        hr = get_head("flutter").run(lat, {
            "runs": list(lat.runs), "run_times": list(lat.run_times),
            "scan_outcome": outcome, "respiration": resp,
            "regularity": reg, "cfg": cfg or {}}).to_dict()
        tracks.append(_track(
            "atrial_flutter", "Atrial flutter pattern flag (v0.6, §F)",
            "flutter", g, status=head_status(hr), value=hr.get("value"),
            reasons=hr.get("reasons")))
    else:
        tracks.append(_track(
            "atrial_flutter", "Atrial flutter pattern flag (v0.6, §F)",
            "flutter", g, status="not_run", reasons=no_read,
            note=f"the scan produced no beat lattice (outcome {outcome}); "
                 "the head never ran"))

    # ---- arterial stiffness (vascular)
    g = _gates("evaluation.vascular_gates:vascular_gate_status")
    st = (hemo.get("arterial_stiffness") or {}) if hemo.get("available") \
        else {}
    if st.get("available"):
        tracks.append(_track(
            "arterial_stiffness", "Arterial stiffness (v0.4 vascular)",
            "vascular", g, status="value", value=st,
            note="contour markers from THIS scan, uncalibrated. A "
                 "carotid-femoral velocity in m/s needs the referenced "
                 "study the gates describe; run `python3 cli.py "
                 f"research-report <video>` (recording {rid}) for the "
                 "full morphology set and the head's own verdict."))
    else:
        tracks.append(_track(
            "arterial_stiffness", "Arterial stiffness (v0.4 vascular)",
            "vascular", g, status="abstained", value=(st or None),
            reasons=(hemo_reasons
                     or ([st["reason"]] if st.get("reason") else [])),
            note="no contour marker survived extraction on this scan"))

    # ---- vasomotor tone (§W)
    g = _gates("evaluation.vasotone_gates:vasotone_gate_status")
    vt = (hemo.get("vascular_tone") or {}) if hemo.get("available") else {}
    if vt.get("available"):
        tracks.append(_track(
            "vascular_tone", "Vasomotor tone (v0.5, §W)", "vasotone", g,
            status="value", value=vt,
            note="RESTING tone indices from this scan, dimensionless and "
                 "uncalibrated. Vasomotor REACTIVITY is a response to a "
                 "timed provocation and is a different measurement: "
                 "`python3 cli.py research-report <video> --provocation "
                 "<file.provocation.json>`."))
    else:
        tracks.append(_track(
            "vascular_tone", "Vasomotor tone (v0.5, §W)", "vasotone", g,
            status="abstained", value=(vt or None),
            reasons=(hemo_reasons
                     or ([vt["reason"]] if vt.get("reason") else [])),
            note="the amplitude envelope did not support a resting tone "
                 "index on this scan"))

    # ---- cardiorespiratory fitness (§V)
    g = _gates("evaluation.fitness_gates:vo2_gate_status")
    cf = (hemo.get("cardiorespiratory_fitness") or {}) \
        if hemo.get("available") else {}
    fit = None
    for h in (getattr(session_result, "head_results", None) or []):
        if (h or {}).get("head") == "fitness":
            fit = h
    extra = {"session_head": fit} if fit else None
    if cf.get("available"):
        tracks.append(_track(
            "cardiorespiratory_fitness",
            "Cardiorespiratory fitness (v0.4, §V)", "fitness", g,
            status="value", value=cf, extra=extra,
            note="resting values from THIS scan. No oxygen-uptake number "
                 "is produced: the value carries the reason. Heart-rate "
                 "recovery, the measured route, needs the three-phase "
                 "session."))
    elif fit is not None:
        tracks.append(_track(
            "cardiorespiratory_fitness",
            "Cardiorespiratory fitness (v0.4, §V)", "fitness", g,
            status=head_status(fit), value=fit.get("value"),
            reasons=fit.get("reasons"), extra=extra))
    else:
        tracks.append(_track(
            "cardiorespiratory_fitness",
            "Cardiorespiratory fitness (v0.4, §V)", "fitness", g,
            status="abstained", value=(cf or None), reasons=hemo_reasons,
            note="resting cardiorespiratory indices were not computable "
                 "on this scan"))

    return {"watermark": WATERMARK, "note": PANEL_NOTE,
            "respiration_channel": (None if resp is None else {
                "rate_brpm": resp.get("rate_brpm"),
                "quality": resp.get("quality")}),
            "resting_hemodynamics_available": bool(hemo.get("available")),
            "tracks": tracks}
