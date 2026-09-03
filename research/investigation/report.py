"""Build, render and write the Research & Investigation Report.

`build_investigation_report(video, ...)` runs the PRODUCTION pipeline
once (the same `run_with_details` the consumer path uses), then asks
each gated research head for its raw output on that scan, exactly the
way each track's own research surface does today:

- rhythm regularity: `head_regularity` with the torso-derived
  respiration channel (the second decode rppg/respiration.py) — the
  head abstains, with its reason, when the channel is unusable;
- atrial flutter: `head_flutter` with the same channel;
- arterial stiffness: `research.vascular.head_runner.run_research_head`
  (the v0.4 research surface, unchanged);
- vascular tone: `research.vascular.tone_runner.run_research_tone_head`
  when timed provocation marks are supplied, else NOT APPLICABLE with
  what it needs;
- VO2 / fitness: `protocol.session.run_session` when a three-phase
  session manifest is supplied, else NOT APPLICABLE with what it needs.

Every section carries the track's LIVE gate status and the watermark;
the consumer result is QUOTED (verbatim `user_facing_text`) as context
and never re-authored. Nothing here is reachable from app/ or
inference/ (quarantine walker) and nothing here writes into a
ScanResult.
"""
from __future__ import annotations

import html as _html
import importlib
import json
import pathlib
import time

from research.investigation import TRACKS, WATERMARK, InvestigationError

_REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO / "research" / "runs" / "investigation"
REPORT_TITLE = "AvatarX Research & Investigation Report"
CONSUMER_REPORT_TITLE = "AvatarX Cardiac Rhythm Scan Report"
NOT_CONSUMER_NOTICE = (
    "This document is NOT the consumer scan report and nothing in it is "
    "for a participant, a patient or a clinician. It exists for research "
    "and investigation only. Every value below is a raw head payload: "
    "while its track's promotion reads BLOCKED the track may show nothing "
    "anywhere a participant can see, and if a track ever opens, only its "
    "sanctioned sentence may render, through that track's own gate — "
    "never these payloads.")


# ------------------------------------------------------------ helpers
def _gate_status(spec: str) -> dict:
    """The track's live gate status, FAIL-CLOSED: any error reads as
    BLOCKED with the error recorded."""
    mod, fn = spec.split(":")
    try:
        doc = getattr(importlib.import_module(mod), fn)()
        return {"promotion": doc.get("promotion", "BLOCKED"),
                "gates_version": doc.get("gates_version"),
                "clinical_signoff": bool(doc.get("clinical_signoff")),
                "gates": [{"gate": g.get("gate"), "title": g.get("title"),
                           "status": g.get("status"),
                           "first_reason": ((g.get("reasons") or [None])[0])}
                          for g in (doc.get("gates") or [])]}
    except Exception as e:                       # noqa: BLE001 — fail closed
        return {"promotion": "BLOCKED", "gates_version": None,
                "clinical_signoff": False, "gates": [],
                "error": f"{type(e).__name__}: {e} — fail closed"}


def _respiration_from_video(video: str):
    """The torso-motion second decode the flutter/regularity harnesses
    use (rppg/respiration.py); None when the video carries none."""
    try:
        from rppg._filters import moving_average_detrend
        from rppg.respiration import (respiratory_rate_from_motion,
                                      torso_motion_series)
        t, y, fps = torso_motion_series(video)
        rate, conc = respiratory_rate_from_motion(t, y, fps)
        return {"t": t, "y": moving_average_detrend(y, int(12.0 * fps)),
                "rate_brpm": rate, "quality": conc}
    except (OSError, IOError, ValueError):
        return None


def _section(key, title, head, gates, *, status, head_result=None,
             needs=None, note=None, extra=None) -> dict:
    out = {"watermark": WATERMARK, "key": key, "title": title,
           "head": head, "research_only": True, "status": status,
           "gates": gates}
    if head_result is not None:
        out["head_result"] = head_result
    if needs:
        out["needs"] = needs
    if note:
        out["note"] = note
    if extra:
        out.update(extra)
    return out


def _status_of(hr: dict) -> str:
    """The ONE classifier (heads.base.head_status) — never a second
    implementation of the same judgement."""
    from heads.base import head_status
    return head_status(hr)


def _not_run(key, title, head, gates, cls, reasons, note) -> dict:
    """A head that never executed is NOT an abstention: the section says
    `not_run` and carries the scan's or session's own reasons (review
    finding: synthesised abstentions misrepresented heads that never
    ran)."""
    return _section(key, title, head, gates, status="not_run",
                    note=note,
                    head_result={"head": head, "version": None,
                                 "measurement_class": cls, "value": None,
                                 "reasons": list(reasons or [])})



# ------------------------------------------------------------ builder
def build_investigation_report(video, *, manifest=None, age_years=None,
                               session_manifest=None, session_videos=None,
                               provocation=None, pi=None, config=None,
                               recording_id=None) -> dict:
    from configs import load_config
    from heads import get_head
    from inference.pipeline import run_with_details

    video = str(video)
    if not pathlib.Path(video).exists():
        raise InvestigationError(f"video not found: {video}")
    cfg = config or load_config()
    rid = recording_id or pathlib.Path(video).stem
    res, det = run_with_details(video, manifest=manifest or {},
                                config=cfg, recording_id=rid)
    resp = _respiration_from_video(video)
    lat = det.get("lattice")
    reg = det.get("regularity")
    outcome = res.outcome.value
    titles = {t[0]: t for t in TRACKS}
    tracks = {}

    # ---- rhythm regularity (§R)
    k, title, head, _, gs = titles["rhythm_regularity"]
    if lat is not None:
        hr = get_head(head).run(lat, {
            "regularity": reg, "scan_outcome": outcome,
            "respiration": resp, "cfg": cfg, "age_years": age_years})
        hrd = hr.to_dict()
        tracks[k] = _section(k, title, head, _gate_status(gs),
                             status=_status_of(hrd), head_result=hrd)
    else:
        tracks[k] = _not_run(k, title, head, _gate_status(gs),
                             "RESEARCH_RHYTHM", res.no_read_reasons,
                             "the scan produced no beat lattice "
                             f"(outcome {outcome}); the head never ran")

    # ---- atrial flutter (§F)
    k, title, head, _, gs = titles["atrial_flutter"]
    if lat is not None:
        hr = get_head(head).run(lat, {
            "runs": list(lat.runs), "run_times": list(lat.run_times),
            "scan_outcome": outcome, "respiration": resp,
            "regularity": reg, "cfg": cfg, "age_years": age_years,
            "series_rates_bpm": None})
        hrd = hr.to_dict()
        tracks[k] = _section(k, title, head, _gate_status(gs),
                             status=_status_of(hrd), head_result=hrd)
    else:
        tracks[k] = _not_run(k, title, head, _gate_status(gs),
                             "RESEARCH_RHYTHM", res.no_read_reasons,
                             "the scan produced no beat lattice "
                             f"(outcome {outcome}); the head never ran")

    # ---- arterial stiffness (vascular)
    k, title, head, _, gs = titles["arterial_stiffness"]
    try:
        from research.vascular.features import features_from_details
        from research.vascular.head_runner import run_research_head
        # the raw morphology features are the single-scan output of this
        # track (the head drops them on every refusal); they are carried
        # here, watermarked, beside the head's verdict
        try:
            morph = features_from_details(res, det)
        except Exception as e:                   # noqa: BLE001
            morph = {"available": False,
                     "reason": f"{type(e).__name__}: {e}"}
        rr = run_research_head(res, det)
        hrd = rr["head_result"].to_dict()
        tracks[k] = _section(k, title, head, _gate_status(gs),
                             status=_status_of(hrd), head_result=hrd,
                             extra={"morphology_features": morph,
                                    "track_report_path": rr["report_path"],
                                    "side_effect": (
                                        "the v0.4 vascular research runner "
                                        "also writes its own per-scan "
                                        "report under the vascular runs "
                                        "root (gitignored), exactly as "
                                        "`cli.py process --heads vascular` "
                                        "does; --out does not move it")})
    except Exception as e:                       # noqa: BLE001
        tracks[k] = _not_run(k, title, head, _gate_status(gs),
                             "RESEARCH_VASCULAR", [],
                             f"vascular research runner failed "
                             f"({type(e).__name__}): {e}")

    # ---- vascular tone (§W)
    k, title, head, _, gs = titles["vascular_tone"]
    if provocation is None:
        tracks[k] = _section(
            k, title, head, _gate_status(gs), status="not_applicable",
            needs=("timed provocation marks (a .provocation.json with "
                   "baseline / stimulus / recovery phase marks) and, "
                   "optionally, a contact perfusion-index reference — "
                   "vasomotor reactivity is a RESPONSE to a controlled "
                   "provocation; a single resting scan carries no tone "
                   "measurement"))
    else:
        try:
            from research.vascular.tone_runner import \
                run_research_tone_head
            cap = (manifest or {}).get("capture") if manifest else None
            rr = run_research_tone_head(res, det, provocation,
                                        capture=cap, pi=pi)
            hrd = rr["head_result"].to_dict()
            tracks[k] = _section(k, title, head, _gate_status(gs),
                                 status=_status_of(hrd), head_result=hrd,
                                 extra={"track_report_path":
                                        rr["report_path"],
                                        "side_effect": (
                                            "the v0.5 vasotone research "
                                            "runner also writes its own "
                                            "per-scan report under the "
                                            "vasotone runs root "
                                            "(gitignored); --out does not "
                                            "move it")})
        except Exception as e:                   # noqa: BLE001
            tracks[k] = _not_run(k, title, head, _gate_status(gs),
                                 "RESEARCH_VASCULAR", [],
                                 f"vasotone research runner failed "
                                 f"({type(e).__name__}): {e}")

    # ---- VO2 / fitness (§V)
    k, title, head, _, gs = titles["cardiorespiratory_fitness"]
    vo2_note = ("hard rule (spec B.18/B.19): an oxygen-uptake number never "
                "renders in any version and none exists in the session "
                "pipeline; the track yields a fitness CATEGORY from "
                "heart-rate recovery, shown here raw, never an estimate")
    if session_manifest is None:
        tracks[k] = _section(
            k, title, head, _gate_status(gs), status="not_applicable",
            note=vo2_note,
            needs=("the three-phase recovery session (rest scan, guided "
                   "activity, recovery scan) as a session manifest — "
                   "`cli.py research-report <video> --session "
                   "<manifest.json> [--videos rest activity recovery]`; "
                   "a single resting scan carries no VO2 or fitness "
                   "information"))
    else:
        try:
            from protocol.session import run_session
            sres, sdet = run_session(session_manifest,
                                     videos_override=session_videos or None)
            heads = {h.get("head"): h for h in (sres.head_results or [])}
            fit = heads.get("fitness")
            tracks[k] = _section(
                k, title, head, _gate_status(gs),
                status=(_status_of(fit) if fit else "not_run"),
                note=(vo2_note if fit else vo2_note + "; the session ended "
                      "before its heads ran"),
                head_result=fit or {"head": head, "version": None,
                                    "value": None,
                                    "measurement_class": "INFERRED_FITNESS",
                                    "reasons": list(sres.no_read_reasons)},
                extra={"session": {
                    "session_id": sres.session_id,
                    "outcome": sres.outcome.value,
                    "protocol_id": sres.protocol_id,
                    "phases_run": list(sdet.get("phases_run") or []),
                    "hr_rest_bpm": sres.hr_rest_bpm,
                    "hrr60_bpm": sres.hrr60_bpm,
                    "user_facing_text": sres.user_facing_text()},
                    "recovery_head": heads.get("recovery"),
                    "trend_head": heads.get("trend")})
        except Exception as e:                   # noqa: BLE001
            tracks[k] = _not_run(k, title, head, _gate_status(gs),
                                 "INFERRED_FITNESS", [],
                                 f"session failed ({type(e).__name__}): {e}")

    doc = {"WATERMARK": WATERMARK,
           "report_title": REPORT_TITLE,
           "notice": NOT_CONSUMER_NOTICE,
           "purpose": ("research artifact for investigators: every gated "
                       "head's raw output for one scan beside the live gate "
                       "status of its track; user-visibility of any track is "
                       "decided by its block of configs/gates.yaml, nowhere "
                       "else"),
           "recording_id": rid,
           "video": video,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "inputs": {"age_years": age_years,
                      "session_manifest": (str(session_manifest)
                                           if session_manifest else None),
                      "provocation": bool(provocation is not None),
                      "respiration_channel": (None if resp is None else {
                          "rate_brpm": resp.get("rate_brpm"),
                          "quality": resp.get("quality")})},
           "consumer_result": {
               "report_title": CONSUMER_REPORT_TITLE,
               "outcome": outcome,
               "predicted_class": res.predicted_class,
               "confidence_stars": res.confidence_stars,
               "signal_quality_index": res.signal_quality_index,
               "no_read_reasons": list(res.no_read_reasons),
               "user_facing_text": res.user_facing_text()},
           "tracks": tracks}
    return doc


# ------------------------------------------------------------ renderer
def _e(x) -> str:
    return _html.escape("" if x is None else str(x))


def _kv_table(d: dict) -> str:
    rows = []
    for k, v in (d or {}).items():
        if isinstance(v, (dict, list)):
            cell = f"<pre>{_e(json.dumps(v, indent=1, default=str))}</pre>"
        else:
            cell = _e(v)
        rows.append(f"<tr><th>{_e(k)}</th><td>{cell}</td></tr>")
    return "<table class='kv'>" + "".join(rows) + "</table>"


def render_investigation_html(doc: dict) -> str:
    wm = _e(doc.get("WATERMARK", WATERMARK))
    banner = f"<div class='wm'>{wm}</div>"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_e(REPORT_TITLE)}</title>"
        "<style>body{font-family:system-ui,sans-serif;color:#1f2937;"
        "max-width:980px;margin:24px auto;padding:0 16px;background:#fff}"
        ".wm{background:#111827;color:#f9fafb;padding:10px 14px;"
        "border-radius:8px;font-weight:700;margin:12px 0;font-size:13px}"
        ".notice{background:#f3f4f6;border-left:6px solid #6b7280;"
        "padding:10px 14px;margin:12px 0}"
        "section{border:1px solid #d1d5db;border-radius:10px;padding:14px;"
        "margin:18px 0}"
        "h1{font-size:22px} h2{font-size:17px;margin:6px 0}"
        ".gate{display:inline-block;padding:2px 8px;border-radius:6px;"
        "background:#374151;color:#f9fafb;font-weight:600;margin-right:6px}"
        "table.kv{border-collapse:collapse;width:100%;font-size:13px}"
        "table.kv th{text-align:left;padding:4px 8px;width:32%;"
        "background:#f9fafb;border-bottom:1px solid #e5e7eb}"
        "table.kv td{padding:4px 8px;border-bottom:1px solid #e5e7eb}"
        "pre{white-space:pre-wrap;margin:0;font-size:12px}"
        ".muted{color:#6b7280}</style></head><body>",
        banner,
        f"<h1>{_e(REPORT_TITLE)}</h1>",
        f"<div class='notice'>{_e(doc.get('notice'))} "
        "<b>NOT the consumer scan report.</b></div>",
        "<p class='muted'>recording "
        f"<b>{_e(doc.get('recording_id'))}</b> · generated "
        f"{_e(doc.get('generated_at'))} · inputs "
        f"{_e(json.dumps(doc.get('inputs'), default=str))}</p>",
        f"<section><div class='wm'>{wm}</div>"
        "<h2>Consumer result on this scan (quoted verbatim, "
        "for context)</h2>",
        _kv_table(doc.get("consumer_result") or {}),
        "</section>",
    ]
    for t in (doc.get("tracks") or {}).values():
        g = t.get("gates") or {}
        gates_html = "".join(
            f"<li><span class='gate'>{_e(x.get('status'))}</span>"
            f"{_e(x.get('title') or x.get('gate'))}"
            + (f" <span class='muted'>— {_e(x.get('first_reason'))}</span>"
               if x.get("first_reason") else "") + "</li>"
            for x in (g.get("gates") or []))
        parts.append("<section>")
        parts.append(f"<div class='wm'>{_e(t.get('watermark'))}</div>")
        parts.append(f"<h2>{_e(t.get('title'))}</h2>")
        parts.append(
            f"<p>head <b>{_e(t.get('head'))}</b> · status "
            f"<b>{_e(t.get('status'))}</b> · track promotion "
            f"<span class='gate'>{_e(g.get('promotion'))}</span> "
            f"gates {_e(g.get('gates_version'))} · clinical signoff "
            f"{_e(g.get('clinical_signoff'))}</p>")
        if g.get("error"):
            parts.append(f"<p class='muted'>{_e(g['error'])}</p>")
        if gates_html:
            parts.append(f"<ul>{gates_html}</ul>")
        if t.get("needs"):
            parts.append(f"<p><b>Not applicable on this scan — needs:</b> "
                         f"{_e(t['needs'])}</p>")
        if t.get("note"):
            parts.append(f"<p>{_e(t['note'])}</p>")
        hr = t.get("head_result")
        if hr:
            parts.append(_kv_table({
                "head": hr.get("head"), "version": hr.get("version"),
                "measurement_class": hr.get("measurement_class"),
                "reasons": hr.get("reasons"),
                "value": hr.get("value")}))
        for k2 in ("morphology_features", "session", "recovery_head",
                   "trend_head", "track_report_path", "side_effect"):
            if t.get(k2) is not None:
                parts.append(f"<h3>{_e(k2)}</h3>")
                parts.append(_kv_table(t[k2]) if isinstance(t[k2], dict)
                             else f"<p>{_e(t[k2])}</p>")
        parts.append("</section>")
    parts.append(banner)
    parts.append("<p class='muted'>Research pipeline — not a medical "
                 "device. No clinical performance is claimed or implied."
                 "</p></body></html>")
    return "".join(parts)


# ------------------------------------------------------------ writer
def write_investigation_report(doc: dict, out_dir=None) -> dict:
    d = pathlib.Path(out_dir or (DEFAULT_OUT / str(doc.get("recording_id")
                                                     or "unknown")))
    d.mkdir(parents=True, exist_ok=True)
    jp = d / "investigation_report.json"
    hp = d / "investigation_report.html"
    jp.write_text(json.dumps(doc, indent=2, default=str))
    hp.write_text(render_investigation_html(doc))
    return {"json_path": str(jp), "html_path": str(hp)}
