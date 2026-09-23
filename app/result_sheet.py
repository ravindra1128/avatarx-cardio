"""Fire-and-forget logging of every measure result to a Google Sheet.

WHY: the owner tracks how the cardio cards behave across real scans in one
spreadsheet, the same way the glucose service does. This module is the
cardio counterpart of eq-blood-glucose-estimation/db/vitals_sheet.py, cut
down to what a stateless batch service needs.

THE ONE RULE: the sheet never delays or breaks a scan result.
  - `schedule_append()` is called AFTER the HTTP response has been written
    and returns immediately; the gspread work runs on a single worker
    thread (rows land in request order).
  - Nothing here raises to the caller. Missing libraries, missing
    credentials, a revoked share, a Sheets outage — each is one log line
    and a counter in `status()` (surfaced by /healthz), never a 500.
  - Retries with backoff inside the worker; after that the row is dropped
    and counted. This service keeps no database, so there is no durable
    queue to replay from — /healthz `sheet.failed` is the signal to look.

CONFIG (env):
  AFIB_SHEET_ID                    spreadsheet id (default: the owner's tracking sheet)
  AFIB_SHEET_GID                   tab id (default: 140358874)
  GOOGLE_SHEETS_CREDENTIALS_JSON   service-account JSON *content* (Railway-friendly)
  GOOGLE_SHEETS_CREDENTIALS_FILE   or a path to that JSON (local dev)
  AFIB_SHEET_ENABLED=0             hard off switch

The service account's email must be shared on the sheet as an editor.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

SHEET_ID = os.environ.get("AFIB_SHEET_ID",
                          "1-CHxko3E4_xQBwxGbsUj-WjqY2_LFBVa3kKnkY3WaRo").strip()
SHEET_GID = os.environ.get("AFIB_SHEET_GID", "140358874").strip()
CREDS_JSON = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON", "").strip()
CREDS_FILE = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_FILE", "").strip()
ENABLED = os.environ.get("AFIB_SHEET_ENABLED", "1").strip().lower() not in ("0", "false", "no")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
RETRIES = (0.0, 2.0, 6.0)          # seconds to wait before each attempt

# Column order for a brand-new tab. On an existing tab the header row wins:
# values are placed by column NAME, missing names are appended to the header,
# so re-ordering columns in the sheet never breaks the writer.
COLUMNS = [
    # The three cards first (owner's request), then identity and verdict,
    # then the cards' unit/status/reason, then evidence, capture and timing.
    "Timestamp (UTC)", "Arterial Stiffness", "Vascular Tone", "Fitness",
    # 2026-09-17 (owner): the AFib result beside the cards, where it is read
    # first - AFIB_DETECTED | AFIB_NOT_DETECTED | INCONCLUSIVE and the
    # classifier's probability (inference/afib_result.py). The basis stays at
    # the end with the other long text.
    "AFib Result", "AFib p",
    "Session", "Outcome", "Stars",
    "AS Unit", "AS Status", "AS Reason",
    "VT Unit", "VT Status", "VT Reason",
    "Fit Unit", "Fit Status", "Fit Reason",
    "Limiting Factor", "SQI", "Coherence", "Timing ms", "Timing Matched",
    "Coverage", "Clean Intervals", "Usable Beats", "Analysed s",
    "Capture Segments", "Pulse bpm", "FPS", "Width", "Height", "Codec",
    "Clock Source", "Capture Profile", "No-Read Reasons",
    "Upload MB", "Upload s", "Trim s", "Trim Probe", "Window s", "Downscale s",
    "Downscale Note",
    "Analysis s", "Server Total s", "Client Duration ms", "Launch Overrides",
    "Build", "Config Hash", "User Agent",
    # Reference vitals the client attached (ShenAI) and the agreement column.
    "Ref HR", "Ref HRV", "Ref SBP", "Ref DBP", "Ref Source", "Pulse - Ref HR",
    # Capture state reported by the client (step 1: exposure lock + pre-check).
    "AE Locked", "AWB Locked", "Client FPS", "Face Luma", "Capture Note",
    "Pulse Spectral", "Pulse Check", "Spectral ROI Agree",
    "Pulse Lattice", "Pulse Lattice N", "Pulse Check Mode",
    "Fit Basis", "Rate Method", "Rate Intervals",
    "AS Tier", "VT Tier", "Fit Tier", "AS Raw", "VT Raw", "Fit HR",
    # ShenAI's own dense PPG waveform and beat train, posted as a sidecar by
    # /beta/cardio-staging and retained beside the clip (2026-09-11). Audit
    # only — nothing here is derived or compared, so the sheet never has to
    # re-implement the offline harness. They exist so a run of
    # scripts/compare_shenai_signal.py can be planned from the history: which
    # scans carry a second opinion, and how much of one.
    "ShenAI Sidecar", "ShenAI PPG N", "ShenAI PPG fs", "ShenAI Beats N",
    # 2026-09-14: what the trim actually did — the keyframe it kept from, how
    # much head and hole it dropped, or why it did nothing. Until now a cut
    # placed by duration alone landed in the rolling recorder's hole and
    # silently kept the whole clip; this column is how that is seen per scan.
    "Trim Note",
    # 2026-09-14: when the resting-rate doubling guard fired — the beat-
    # interval median looked doubled/halved against the subharmonic-protected
    # spectral rhythm, so the fitness rate was taken from the rhythm instead.
    # Blank on a clean scan (the guard is a no-op) and on builds before it.
    "Rate Guard",
    # 2026-09-16: the AFib rhythm result as the participant sees it. Outcome and
    # Stars were always here; these add the class the decision head assigned
    # (SINUS | AFIB_SUGGESTIVE | OTHER_IRREGULAR | HIGH_RATE, blank on abstain)
    # and the exact sanctioned sentence the results page renders, so a phone
    # test can be read back row by row without opening the response JSON.
    "Rhythm Class", "Rhythm Text",
    # 2026-09-17 (audit #1): the classifier's own inputs. The irregularity rule
    # is median|dRR| >= 60 ms AND pNN50 >= 0.40 on within-run successive
    # differences, compared to the scan's measured beat-timing precision
    # (Timing ms). Reading these three beside Timing ms per scan is what
    # decides whether a regular heart is being read as irregular because of
    # timing noise (audit finding #2) and by how much.
    "MAD ms", "pNN50", "Rate N",
    # ShenAI route (2026-09-16): which interval source the rhythm statement
    # above came from ("video" | "shenai_train"), what the route did on this
    # scan and why (used or not), and the train's own rate. When the route is
    # used, Rhythm Class/Text/MAD/pNN50/Rate N above are ITS numbers and the
    # video path's own rationale is kept under debug.video_rationale.
    "Rhythm Source", "ShenAI Route", "ShenAI Rate",
    # 2026-09-17: ONE AFib result per completed scan (inference/afib_result.py)
    # - AFIB_DETECTED | AFIB_NOT_DETECTED | INCONCLUSIVE - with the classifier's
    # probability when a probabilistic classifier ran, and the basis: for an
    # inconclusive, WHICH of capture / signal / rhythm was missing.
    "AFib Basis",
    # 2026-09-17: the trace path - THE pipeline run on ROI traces the phone
    # sampled from the live camera frames (no codec), recorded beside the
    # video path and the ShenAI train on every scan until the live comparison
    # says it may decide. inference/trace_ingest.py.
    "Trace Ran", "Trace Outcome", "Trace Result", "Trace p", "Trace SQI",
    "Trace Coherence", "Trace Timing ms", "Trace Intervals", "Trace Coverage",
    "Trace Pulse", "Trace FPS", "Trace Note",
    # 2026-09-23: the Vascular Tone card's measurement, the facial pulsatile
    # perfusion index from the live-frame traces (features/facial_perfusion.py).
    # "Vascular Tone" above is its 0-100 score; these are the index itself (%),
    # its 95 % window-bootstrap interval, the face regions and low-motion
    # windows it used, why it abstained when it did, and the retired clip-based
    # CV score of the same scan, so repeat scans can be compared per source.
    "VT Source", "VT PI %", "VT PI CI", "VT Regions", "VT Windows", "VT Why",
    "VT Legacy Clip",
    # 2026-09-20: the fitness card's second basis (features/vo2max.py). "Fitness"
    # above holds what the user saw - mL/kg/min on the profile basis, the 0-100
    # proxy otherwise, and "Fit Unit" says which. These say where the estimate's
    # rate came from (the clip, or the live-frame rate of the same scan) and keep
    # the clip's own 0-100 proxy beside it so the two bases stay comparable.
    # "Fit Profile" is the equation's user-entered inputs (sex/age/BMI/activity),
    # which a later treadmill comparison needs; there is no name or id in it.
    "Fit Estimator", "Fit HR Source", "Fit HR Own", "Fit HR Ref", "Fit Proxy", "Fit Profile",
    # Receipt from the job's memory, independent of retained sidecars.
    "Scan ID", "Signals State", "Signals Received", "Signals Transport",
    "Input Beats N", "Input PPG N", "Input PPG Missing N", "Input PPG Clock",
    "SDK Quality", "SDK Bad Signal s", "SDK HR", "SDK lnRMSSD",
    # Codex diagnostics: independent of selected rhythm source and retention.
    "Video Audit", "Video Retained s", "Video Processed s", "Video Short Fragments s",
    "Video Excluded ROI Steps s", "Video Clean s", "Video Whole Coverage", "Frame Step p99 ms",
    "SDK Train Audit", "SDK Train Span s", "SDK Invalid Beats", "SDK Order Errors",
    "SDK Gap Boundaries", "SDK Overlap Boundaries", "SDK Raw RMSSD ms",
    "SDK Clean Intervals", "SDK Clean s", "SDK Longest Clean Run s", "SDK Clean Coverage",
    "SDK Clean RMSSD ms", "SDK RMSSD Relative Error", "SDK Video Alignment", "SDK Audit Issues",
    "SDK Train Checks",
    "Video ROI Excluded Frames", "Video ROI Edge Loss s", "ROI Step p99 ms",
    "Capture Diagnostics", "SDK Version", "Capture Scope",
    "Client Adjacent Frame p99 ms", "Client Callback p99 ms", "Client Unobserved Frames",
    "Client Clock Reversals", "Client Hidden s", "SDK Live Quality Mean", "SDK Live Quality Min",
    "Rhythm Comparison State", "Video Diagnostic Result", "Video Diagnostic Gates",
    "SDK Diagnostic State", "SDK Diagnostic Result", "SDK Diagnostic Gates", "SDK Diagnostic Reason",
    "SDK Policy Eligible", "SDK Publication Blocker", "Source Decision Agreement",
    "Source Window Alignment", "Rhythm Diagnostic ms",
    "Spectral Strongest bpm", "Spectral Selection", "Spectral Peak Audit",
    "Video Interval Rejections", "Video SQI Exact",
]

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sheet")
_LOCK = threading.Lock()
_ws = None                          # cached gspread worksheet
_header: list = []                  # cached header row
_stats = {"queued": 0, "written": 0, "failed": 0, "last_error": None,
          "last_ok_at": None, "worksheet": None}


def is_configured() -> bool:
    return ENABLED and bool(SHEET_ID) and bool(CREDS_JSON or CREDS_FILE)


def status() -> dict:
    with _LOCK:
        return {"configured": is_configured(), "sheet_id": SHEET_ID or None,
                "gid": SHEET_GID or None, **_stats}


# ------------------------------------------------------------ row building
def _g(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _num(v, nd=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    # isfinite, not `f == f`: that only caught NaN and let ±inf through into a
    # cell. inf is as reachable as NaN was — the ShenAI sidecar columns below
    # read an unauthenticated /api/scan-signals body, and json.loads accepts the
    # non-standard literals Infinity/-Infinity/NaN, as does float("inf") on a
    # posted string. gspread serialises inf to bare `Infinity`, which is invalid
    # JSON to the Sheets API, so one such value fails all three retries and
    # DROPS the whole row — the scan's only audit trail (2026-09-11).
    return round(f, nd) if math.isfinite(f) else ""


def _count(v):
    """A count cell, written exactly as the sidecar reported it (0 and 1483 must
    stay ints, not become 0.0/1483.0), but only when it is a finite number.
    These two cells are the only numbers in the row that bypass `_num`, so they
    would otherwise be the remaining path for an `Infinity` posted to
    /api/scan-signals to reach the sheet and drop the row (2026-09-11)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return v if math.isfinite(f) else ""


def _first(*vals):
    """First value that is not None. The "Downscale Note" fallback chain uses
    `or`, which is right for strings but wrong for counts: a sidecar that
    honestly reports 0 PPG samples must not read as "field missing"."""
    for v in vals:
        if v is not None:
            return v
    return None


def _tier(items: dict, key: str) -> str:
    return str((items.get(key) or {}).get("tier") or "")


def _raw(items: dict, key: str, nd: int = 3):
    return _num(((items.get(key) or {}).get("raw") or {}).get("value"), nd)


def _card(items: dict, key: str) -> tuple:
    it = items.get(key) or {}
    # A card may report a band ("Typical") rather than a number; the cell holds
    # whichever it produced, and the raw marker is in its own column.
    value = _num(it.get("value"), 3)
    if value == "" and it.get("band"):
        value = str(it["band"])
    return (value, it.get("unit") or "", it.get("status") or "",
            it.get("reason") or "")


def _pulse_check_mode(ev: dict) -> str:
    """"gate" / "report" / "" — which regime this row was measured under."""
    if not ev or "pulse_agreement" not in ev:
        return ""
    try:
        from features.hemodynamics import PULSE_CHECK_MODE
        return str(PULSE_CHECK_MODE)
    except Exception:                       # the sheet never breaks a result
        return ""


def _pulse_check_cell(ev: dict) -> str:
    """'agree 63 vs 60' / 'disagree 100 vs 55' / 'unresolved: ...' / ''."""
    if not ev or "pulse_agreement" not in ev:
        return ""
    try:
        from features.hemodynamics import pulse_check
        pc = pulse_check(ev)
    except Exception:                       # the sheet never breaks a result
        return ""
    v = pc.get("verdict") or ""
    if v in ("agree", "disagree"):
        return (f"{v} {float(pc['pulse_lattice_bpm']):.0f} vs "
                f"{float(pc['pulse_spectral_bpm']):.0f}")
    if v == "unresolved":
        return f"unresolved: {pc.get('reason') or ''}"[:120]
    return ""


def _rate_guard_cell(g) -> str:
    """'split_inflation: 101->69' when the doubling guard fired, else ''."""
    if not isinstance(g, dict) or not g.get("guarded"):
        return ""
    raw, rep = g.get("raw_lattice_bpm"), g.get("reported_bpm")
    try:
        return f"{g.get('signature')}: {float(raw):.0f}->{float(rep):.0f}"
    except (TypeError, ValueError):
        return str(g.get("signature") or "guarded")


def _shenai_route_cell(r) -> str:
    """One glance: 'used: 56 clean intervals at coverage 0.98, rate 67 bpm
    backed by ...' or 'not used: <why>' (never blank once the build has the
    route, so an absent sidecar is visible per scan)."""
    if not isinstance(r, dict):
        return ""
    why = str(r.get("reason") or "")
    if r.get("used"):
        return (why if why.startswith("used") else f"used: {why}")[:300]
    return (f"not used: {why}" if why else "not used")[:300]


def _fit_profile(fit_d: dict) -> str:
    """The equation's inputs as one compact cell, e.g. "male/35y/BMI 24.0/PA 3"."""
    i = _g(fit_d, "vo2max", "inputs") or {}
    if not i:
        return ""
    return (f"{i.get('sex')}/{_num(i.get('age_years'), 0)}y/BMI {_num(i.get('bmi'), 1)}"
            f"/PA {i.get('activity_level')}")


def row_from_doc(doc: dict, extra: dict | None = None) -> dict:
    """Flatten one measure response into {column name: cell value}."""
    ev = _g(doc, "debug", "evidence", default={}) or {}
    ra = _g(doc, "debug", "rationale", default={}) or {}
    cm = doc.get("capture_meta") or {}
    tm = doc.get("timing") or {}
    dsc = doc.get("downscale") or {}
    vd = _g(doc, "debug", "video_duration", default={}) or {}
    saudit = _g(doc, "debug", "shenai_assessment", default={}) or {}
    cd = _g(doc, "debug", "client_capture_diagnostics", default={}) or {}
    comparison = _g(doc, "debug", "rhythm_comparison", default={}) or {}
    # Always original video evidence, including when SDK supplied the final rhythm.
    spectral = _g(doc, "debug", "evidence", "spectral_diagnostics", default={}) or {}
    items = {i.get("key"): i for i in _g(doc, "biomarkers", "items", default=[]) or []
             if isinstance(i, dict)}
    pulse = ""
    for h in doc.get("head_results") or []:
        if isinstance(h, dict) and h.get("head") == "rate_flags":
            pulse = _num(_g(h, "value", "median_bpm"), 1)
    a_s = _card(items, "arterial_stiffness")
    v_t = _card(items, "vascular_tone")
    fit = _card(items, "cardiorespiratory_fitness")
    fit_d = (items.get("cardiorespiratory_fitness") or {}).get("details") or {}
    tp = _g(doc, "debug", "trace_path") or {}
    vt = _g(doc, "debug", "vascular_tone") or {}
    vt = vt if isinstance(vt, dict) else {}
    # Standalone /beta/cardio-afib scan (rhythm_source client_traces): it has no
    # sideband trace_path, but the same Trace* columns should show its capture
    # so a short/weak mobile scan is diagnosable from the row. Map its own
    # trace_ingest + result into tp when the sideband is absent.
    if not tp and doc.get("rhythm_source") == "client_traces":
        ti = _g(doc, "trace_ingest") or {}
        ev = _g(doc, "debug", "evidence") or {}
        _rg = ti.get("roi_green") or {}
        _smp = ti.get("sampler") or {}
        _src = _smp.get("source") or "?"
        # head names the frontend build (sampler version) and the RAW frame clock
        # (fps + jitter): a large jitter is the bursty/thermal clock trace_ingest
        # resamples, so a row that once read coherence 0 is diagnosable at a glance.
        _head = f"src={_src} v={_smp.get('version') or '?'}"
        if ti.get("fps"):
            _head += f" fps={round(float(ti['fps']), 1)}"
        if ti.get("jitter_ms") is not None:
            _head += f" jit={round(float(ti['jitter_ms']))}ms"
        _green = (_head + "; green mean/std " + ", ".join(
            f"{r[:2]}={_rg[r]['mean']}/{_rg[r]['std']}" for r in
            ("forehead", "cheek_l", "cheek_r", "nose") if r in _rg)) if _rg else _head
        tp = {"ran": True, "outcome": doc.get("outcome"),
              "afib_result": doc.get("afib_result"), "afib_probability": doc.get("afib_probability"),
              "sqi": doc.get("signal_quality_index"),
              "coherence": ev.get("cross_roi_coherence"), "timing_ms": ev.get("timing_precision_ms"),
              "n_intervals": ev.get("n_intervals"),
              "coverage": (ti.get("duration_s") and ev.get("captured_seconds") is not None
                           and round(ev["captured_seconds"] / ti["duration_s"], 2)) or None,
              "pulse_bpm": doc.get("mean_pulse_rate_bpm"), "fps": ti.get("fps"),
              "no_read_reasons": (([_green] if _green else [])   # diagnostic FIRST so the 300-char note never truncates it
                                  + [f"{ti.get('n_frames')} frames / {ti.get('duration_s')}s captured"
                                     if ti.get("n_frames") else ""]
                                  + [c for c in (ti.get("reasons") or [])]
                                  + list(doc.get("no_read_reasons") or []))}
    ex = extra or {}
    ref = doc.get("reference") or {}
    cap = doc.get("client_capture") or {}
    # ShenAI's sidecar. Client-supplied audit data like `ref`/`cap` above, so it
    # is read the same tolerant way — and it may reach here in either of two
    # shapes: a small summary the HTTP layer builds, or the sidecar document
    # itself (schema_version 1: {ppg: {n, fs_hz}, heartbeats: [...]}). Accept
    # both, and accept a bare bool from a build that only records arrival.
    shen = doc.get("shenai")
    sa = shen if isinstance(shen, dict) else {}
    ppg_n = _first(sa.get("ppg_n"), _g(sa, "ppg", "n"))
    # fs is DERIVED, never measured: the SDK exposes no sample rate at all
    # (avatarxvitals/index.d.ts:374-377 — getFullPpgSignal() returns a bare
    # number[]). Which rule produced it lives in the retained JSON as
    # ppg.fs_source; it is deliberately not mirrored here, because a rate in a
    # sheet cell with no provenance beside it reads as a measured one.
    ppg_fs = _first(sa.get("ppg_fs_hz"), sa.get("ppg_fs"), _g(sa, "ppg", "fs_hz"))
    beats_n = _first(sa.get("beats_n"), _g(sa, "measurement", "beats_n"),
                     len(sa["heartbeats"]) if isinstance(sa.get("heartbeats"), list) else None)
    return {
        "Capture Diagnostics": cd.get("probe_state") or cd.get("state") or "",
        "SDK Version": cd.get("sdk_version") or "",
        "Capture Scope": cd.get("scope") or "",
        "Client Adjacent Frame p99 ms": _num(_g(cd, "frames", "adjacent_frame_steps", "p99_ms"), 1),
        "Client Callback p99 ms": _num(_g(cd, "frames", "callback_steps", "p99_ms"), 1),
        "Client Unobserved Frames": _g(cd, "frames", "unobserved_presented_frames", default=""),
        "Client Clock Reversals": _g(cd, "frames", "backward_steps", default=""),
        "Client Hidden s": _num(cd.get("hidden_ms") / 1000 if isinstance(cd.get("hidden_ms"), (int, float)) else None, 2),
        "SDK Live Quality Mean": _num(_g(cd, "quality", "mean"), 3),
        "SDK Live Quality Min": _num(_g(cd, "quality", "min"), 3),
        "Rhythm Comparison State": comparison.get("state", ""),
        "Video Diagnostic Result": _g(comparison, "video", "result", default=""),
        "Video Diagnostic Gates": " | ".join(_g(comparison, "video", "gates_failed", default=[]) or [])[:500],
        "SDK Diagnostic State": _g(comparison, "shenai_train", "state", default=""),
        "SDK Diagnostic Result": _g(comparison, "shenai_train", "result", default=""),
        "SDK Diagnostic Gates": " | ".join(_g(comparison, "shenai_train", "gates_failed", default=[]) or [])[:500],
        "SDK Diagnostic Reason": str(_g(comparison, "shenai_train", "reason", default="") or "")[:500],
        "SDK Policy Eligible": _g(comparison, "shenai_train", "publication_policy_eligible", default=""),
        "SDK Publication Blocker": str(_g(comparison, "shenai_train", "publication_blocker", default="") or "")[:500],
        "Source Decision Agreement": comparison.get("decision_agreement", ""),
        "Source Window Alignment": comparison.get("window_alignment", ""),
        "Rhythm Diagnostic ms": _num(comparison.get("elapsed_ms"), 3),
        "Spectral Strongest bpm": _num(_g(spectral, "fused", "strongest_peak_bpm"), 3),
        "Spectral Selection": _g(spectral, "fused", "selection", default=""),
        "Spectral Peak Audit": json.dumps(spectral, separators=(",", ":"), allow_nan=False) if spectral else "",
        "Video Interval Rejections": json.dumps(vd.get("interval_rejections"), separators=(",", ":"), allow_nan=False) if vd.get("interval_rejections") else "",
        "Video SQI Exact": _num(doc.get("signal_quality_index"), 8),
        "Timestamp (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "Session": doc.get("session") or "",
        "Build": ex.get("build") or "",
        "Config Hash": doc.get("config_hash") or "",
        "Outcome": doc.get("outcome") or "",
        "Stars": doc.get("confidence_stars") if doc.get("confidence_stars") is not None else "",
        "Limiting Factor": doc.get("confidence_limiting_factor") or "",
        "SQI": _num(doc.get("signal_quality_index"), 3),
        "Coherence": _num(ev.get("cross_roi_coherence"), 3),
        "Timing ms": _num(ev.get("timing_precision_ms"), 1),
        "Timing Matched": _num(ev.get("timing_matched_fraction"), 2),
        "Coverage": _num(ra.get("coverage"), 2),
        "Clean Intervals": ev.get("n_intervals") if ev.get("n_intervals") is not None else "",
        "Usable Beats": doc.get("usable_beats") if doc.get("usable_beats") is not None else "",
        "Analysed s": _num(doc.get("analysed_seconds"), 1),
        "Capture Segments": len(ev.get("capture_segments") or []) or "",
        "Pulse bpm": pulse,
        "FPS": _num(cm.get("measured_fps"), 2),
        "Width": cm.get("width") or "",
        "Height": cm.get("height") or "",
        "Codec": cm.get("codec") or "",
        "Clock Source": _g(doc, "clock", "source", default="") or "",
        "Capture Profile": cm.get("capture_profile") or "",
        "No-Read Reasons": " | ".join(doc.get("no_read_reasons") or []),
        "Arterial Stiffness": a_s[0], "AS Unit": a_s[1], "AS Status": a_s[2], "AS Reason": a_s[3],
        "Vascular Tone": v_t[0], "VT Unit": v_t[1], "VT Status": v_t[2], "VT Reason": v_t[3],
        "Fitness": fit[0], "Fit Unit": fit[1], "Fit Status": fit[2], "Fit Reason": fit[3],
        "Upload MB": _num((tm.get("upload_bytes") or 0) / 1e6, 1) if tm.get("upload_bytes") else "",
        "Upload s": _num(tm.get("upload_received_s"), 2),
        "Trim s": _num(tm.get("trim_s"), 2),
        "Trim Probe": _g(doc, "trim", "duration_probe", default="") or "",
        # Seconds of the clip actually analysed. The client asks for this, so
        # it must be recorded: comparing a 70 s scan against the 40 s history
        # is the whole point of asking.
        "Window s": _num(_g(doc, "trim", "window_s"), 0),
        "Downscale s": _num(tm.get("downscale_s"), 2),
        # A downscale that FAILS falls back to the native file and the scan
        # still returns, so the only trace is a suspiciously fast stage and a
        # resolution that did not change. One real scan (2026-09-10 12:26)
        # analysed at 480x720 with downscale_s 0.51 and nothing recorded why.
        "Downscale Note": (dsc.get("reason") or dsc.get("degraded")
                           or dsc.get("scaled_to") or ""),
        "Analysis s": _num(tm.get("analysis_s"), 2),
        "Server Total s": _num(tm.get("server_total_s"), 2),
        "Client Duration ms": ex.get("duration_ms") if ex.get("duration_ms") is not None else "",
        "Launch Overrides": json.dumps(doc.get("launch_overrides")) if doc.get("launch_overrides") else "",
        "User Agent": (ex.get("user_agent") or "")[:200],
        "Ref HR": _num(ref.get("ref_hr"), 1),
        "Ref HRV": _num(ref.get("ref_hrv"), 1),
        "Ref SBP": _num(ref.get("ref_sbp"), 0),
        "Ref DBP": _num(ref.get("ref_dbp"), 0),
        "Ref Source": ref.get("ref_source") or "",
        "Pulse - Ref HR": (_num(float(pulse) - float(ref["ref_hr"]), 1)
                           if pulse != "" and ref.get("ref_hr") is not None else ""),
        "AE Locked": ("TRUE" if cap.get("exposure_locked") else "FALSE") if "exposure_locked" in cap else "",
        "AWB Locked": ("TRUE" if cap.get("awb_locked") else "FALSE") if "awb_locked" in cap else "",
        "Client FPS": _num(cap.get("client_fps"), 1),
        "Face Luma": _num(cap.get("face_luma"), 0),
        "Capture Note": str(cap.get("note") or "")[:300],
        # iteration 12: the beat count checked against the waveform's dominant
        # rhythm; the verdict comes from the same function the cards use
        "Pulse Spectral": _num(ev.get("pulse_spectral_bpm"), 1),
        "Pulse Check": _pulse_check_cell(ev),
        "Spectral ROI Agree": (ev.get("pulse_spectral_roi_agree")
                               if ev.get("pulse_spectral_roi_agree") is not None else ""),
        # The other side of the cross-check, and the evidence behind it: a blank
        # "Pulse bpm" is explained by "Pulse Lattice N" below the 15-interval
        # publish floor. "Pulse Check Mode" tells report-mode rows from
        # gate-mode rows when reading the history back.
        "Pulse Lattice": _num(ev.get("pulse_lattice_bpm"), 1),
        "Pulse Lattice N": (ev.get("pulse_lattice_n_intervals")
                            if ev.get("pulse_lattice_n_intervals") is not None else ""),
        "Pulse Check Mode": _pulse_check_mode(ev),
        # Why the fitness card computed or abstained, without opening the JSON.
        "Fit Basis": str(fit_d.get("fitness_proxy_basis") or ""),
        "Rate Method": str(fit_d.get("resting_rate_method") or ""),
        "Rate Intervals": (fit_d.get("resting_rate_intervals")
                           if fit_d.get("resting_rate_intervals") is not None else ""),
        # Display tiers (2026-09-09): the card columns above hold the 0-100
        # score the user saw; these say how well it was evidenced and what
        # marker it came from, so "measured" availability can still be read.
        "AS Tier": _tier(items, "arterial_stiffness"),
        "VT Tier": _tier(items, "vascular_tone"),
        "Fit Tier": _tier(items, "cardiorespiratory_fitness"),
        "AS Raw": _raw(items, "arterial_stiffness", 4),
        "VT Raw": _raw(items, "vascular_tone", 2),
        "Fit HR": _raw(items, "cardiorespiratory_fitness", 1),
        # Route 6 (2026-09-11): a second, independent optical opinion on the
        # same beats. Ours is bottlenecked on cross-ROI fusion — four face
        # regions disagree on the pulse by 15-23 bpm at the phone's compression
        # level, cross-region waveform correlation is 0.16, coverage sits at
        # 0.47 against the 0.50 floor and 1 scan in 49 reaches ACCEPT — so it is
        # worth knowing, per scan and at a glance, whether a ShenAI waveform is
        # retained to compare against. Blank means the build predates the
        # sidecar or the scan never reported one; FALSE means it reported none.
        "ShenAI Sidecar": (("TRUE" if (sa or shen is True) else "FALSE")
                           if shen is not None else ""),
        # Counts, not statistics: the harness computes every comparison offline
        # from the retained JSON, and nothing derived is recomputed here where
        # it could delay or break a scan.
        "ShenAI PPG N": _count(ppg_n) if ppg_n is not None else "",
        "ShenAI PPG fs": _num(ppg_fs, 2),
        "ShenAI Beats N": _count(beats_n) if beats_n is not None else "",
        "Trim Note": str(_g(doc, "trim", "note") or
                         _g(doc, "trim", "reason") or "")[:300],
        "Rate Guard": _rate_guard_cell(fit_d.get("rate_guard")),
        "Rhythm Class": str(doc.get("predicted_class") or ""),
        "Rhythm Text": str(doc.get("user_facing_text") or "")[:500],
        "MAD ms": _num(_g(ra, "features", "median_abs_succ_diff"), 1),
        "pNN50": _num(_g(ra, "features", "pnn50"), 3),
        "Rate N": (_g(ra, "features", "n_intervals")
                   if _g(ra, "features", "n_intervals") is not None else ""),
        "Rhythm Source": str(doc.get("rhythm_source") or ""),
        "ShenAI Route": _shenai_route_cell(_g(doc, "debug", "shenai_route")),
        "ShenAI Rate": _num(_g(doc, "debug", "shenai_route", "train", "bpm"), 1),
        "Trace Ran": (("TRUE" if tp.get("ran") else "FALSE") if tp else ""),
        "Trace Outcome": str(tp.get("outcome") or "") if tp else "",
        "Trace Result": str(tp.get("afib_result") or "") if tp else "",
        "Trace p": _num(tp.get("afib_probability"), 3) if tp else "",
        "Trace SQI": _num(tp.get("sqi"), 3) if tp else "",
        "Trace Coherence": _num(tp.get("coherence"), 3) if tp else "",
        "Trace Timing ms": _num(tp.get("timing_ms"), 1) if tp else "",
        "Trace Intervals": (tp.get("n_intervals") if tp and tp.get("n_intervals") is not None else ""),
        "Trace Coverage": _num(tp.get("coverage"), 2) if tp else "",
        "Trace Pulse": _num(tp.get("pulse_bpm"), 1) if tp else "",
        "Trace FPS": _num(tp.get("fps"), 1) if tp else "",
        "Trace Note": ((" | ".join(tp.get("no_read_reasons") or tp.get("ingest_reasons") or [])
                        or str(tp.get("reason") or ""))[:300] if tp else ""),
        "VT Source": str(vt.get("source") or ""),
        "VT PI %": _num(vt.get("pi_percent"), 3),
        "VT PI CI": ("-".join(f"{x:.3f}" for x in vt["ci95_percent"])
                     if isinstance(vt.get("ci95_percent"), list) and len(vt["ci95_percent"]) == 2 else ""),
        "VT Regions": ",".join(vt.get("regions_used") or []),
        "VT Windows": (f"{_g(vt, 'windows', 'low_motion')}/{_g(vt, 'windows', 'total')}"
                       if isinstance(vt.get("windows"), dict) else ""),
        "VT Why": str(vt.get("reason") or "")[:200],
        "VT Legacy Clip": _num(vt.get("legacy_clip_score"), 1),
        "Fit Estimator": str(fit_d.get("estimator") or ""),
        "Fit HR Source": str(fit_d.get("resting_rate_source") or ""),
        "Fit HR Own": _num(_g(fit_d, "resting_rate_choice", "own_bpm"), 1),
        "Fit HR Ref": _num(_g(fit_d, "resting_rate_choice", "reference_bpm"), 1),
        "Fit Proxy": _num(fit_d.get("fitness_proxy_score"), 1),
        "Fit Profile": _fit_profile(fit_d),
        "AFib Result": str(doc.get("afib_result") or ""),
        "AFib p": _num(doc.get("afib_probability"), 3),
        "AFib Basis": (lambda b: (f"{b.get('category')}: {b.get('why')}" if b.get("category")
                                  else str(b.get("why") or ""))[:300]
                       if isinstance(b, dict) else "")(doc.get("afib_result_basis")),
        "Scan ID": str(doc.get("scan_id") or doc.get("upload_id") or "")[:80],
        "Signals State": str(_g(doc, "debug", "shenai_input", "state") or "")[:80],
        "Signals Received": ({True: "TRUE", False: "FALSE"}.get(
            _g(doc, "debug", "shenai_input", "received"), "")),
        "Signals Transport": str(_g(doc, "debug", "shenai_input", "transport") or "")[:40],
        "Input Beats N": _g(doc, "debug", "shenai_input", "beats_n", default=""),
        "Input PPG N": _g(doc, "debug", "shenai_input", "ppg_n", default=""),
        "Input PPG Missing N": _g(doc, "debug", "shenai_input", "ppg_missing_n", default=""),
        "Input PPG Clock": str(_g(doc, "debug", "shenai_input", "ppg_fs_source") or "")[:40],
        "SDK Quality": _num(_g(doc, "debug", "shenai_input", "sdk_quality"), 3),
        "SDK Bad Signal s": _num(_g(doc, "debug", "shenai_input", "sdk_bad_signal_s"), 2),
        "SDK HR": _num(_g(doc, "debug", "shenai_input", "sdk_hr_bpm"), 1),
        "SDK lnRMSSD": _num(_g(doc, "debug", "shenai_input", "sdk_lnrmssd"), 3),
        "Video Audit": str(vd.get("state") or ""),
        "Video Retained s": _num(vd.get("retained_span_s"), 3),
        "Video Processed s": _num(vd.get("processing_segment_s"), 3),
        "Video Short Fragments s": _num(vd.get("discarded_fragment_s"), 3),
        "Video Excluded ROI Steps s": _num(vd.get("excluded_roi_step_s"), 3),
        "Video Clean s": _num(vd.get("clean_interval_s"), 3),
        "Video Whole Coverage": _num(vd.get("clean_fraction_retained"), 4),
        "Frame Step p99 ms": _num(vd.get("frame_step_p99_ms"), 3),
        "SDK Train Audit": str(saudit.get("state") or ""),
        "SDK Train Span s": _num(_g(saudit, "integrity", "span_s"), 3),
        "SDK Invalid Beats": _g(saudit, "integrity", "invalid_beats_n", default=""),
        "SDK Order Errors": _g(saudit, "integrity", "non_increasing_starts_n", default=""),
        "SDK Gap Boundaries": _g(saudit, "integrity", "gap_boundaries_n", default=""),
        "SDK Overlap Boundaries": _g(saudit, "integrity", "overlap_boundaries_n", default=""),
        "SDK Raw RMSSD ms": _num(_g(saudit, "reported_intervals", "rmssd_ms"), 3),
        "SDK Clean Intervals": _g(saudit, "current_route_train", "intervals_n", default=""),
        "SDK Clean s": _num(_g(saudit, "current_route_train", "seconds"), 3),
        "SDK Longest Clean Run s": _num(_g(saudit, "current_route_train", "longest_run_s"), 3),
        "SDK Clean Coverage": _num(_g(saudit, "current_route_train", "coverage"), 4),
        "SDK Clean RMSSD ms": _num(_g(saudit, "current_route_train", "rmssd_ms"), 3),
        "SDK RMSSD Relative Error": _num(_g(saudit, "internal_comparison", "rmssd_relative_error"), 4),
        "SDK Video Alignment": str(_g(saudit, "timing", "video_alignment") or ""),
        "SDK Audit Issues": " | ".join(saudit.get("issues") or [])[:500],
        "SDK Train Checks": " | ".join(_g(saudit, "current_route_train", "checks_failed", default=[]) or [])[:500],
        "Video ROI Excluded Frames": vd.get("roi_excluded_frames_n") if vd.get("roi_excluded_frames_n") is not None else "",
        "Video ROI Edge Loss s": _num(vd.get("unobserved_edge_s"), 3),
        "ROI Step p99 ms": _num(vd.get("roi_step_p99_ms"), 3),
    }


def _extend_header(ws, header: list, missing: list) -> list:
    """Append `missing` column names to row 1, growing the grid first: a tab
    left exactly as wide as its header (e.g. after --reorder) rejects a
    header write past its last column with 'exceeds grid limits'."""
    from gspread.utils import rowcol_to_a1
    need = len(header) + len(missing)
    if ws.col_count < need:
        ws.add_cols(need - ws.col_count)
    start = len(header) + 1
    a1 = f"{rowcol_to_a1(1, start)}:{rowcol_to_a1(1, need)}"
    ws.update(range_name=a1, values=[missing], value_input_option="RAW")
    return header + missing


# ------------------------------------------------------------ sheet access
def _worksheet():
    """Authorise once, open the tab by gid, make sure every COLUMNS name is in
    the header (append the missing ones). Cached; a failure clears the cache
    so the next attempt re-opens."""
    global _ws, _header
    if _ws is not None:
        # Re-read the header on EVERY write. Someone (or --reorder) can move
        # columns while the service runs; a header cached at first write
        # then places every later value in the wrong column. Found live:
        # a row landed shifted after the tab was reordered. One cheap API
        # call per scan is the price of never doing that.
        _header = _ws.row_values(1) or list(_header)
        missing = [c for c in COLUMNS if c not in _header]
        if missing:
            _header = _extend_header(_ws, _header, missing)
        return _ws, _header
    import gspread
    from google.oauth2.service_account import Credentials
    if CREDS_JSON:
        creds = Credentials.from_service_account_info(json.loads(CREDS_JSON), scopes=SCOPES)
    else:
        creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPES)
    book = gspread.authorize(creds).open_by_key(SHEET_ID)
    ws = book.get_worksheet_by_id(int(SHEET_GID)) if SHEET_GID else book.sheet1
    header = ws.row_values(1)
    if not header:
        # A tab created by hand is 26 columns wide (A-Z) and COLUMNS is 78, so
        # widen before writing — the same "exceeds grid limits" rejection
        # _extend_header grows the grid to avoid, which until now only the
        # existing-header branch below was protected from. An empty tab is
        # exactly what a fresh environment points at, so this branch is the one
        # a new deployment hits first.
        if ws.col_count < len(COLUMNS):
            ws.add_cols(len(COLUMNS) - ws.col_count)
        ws.append_row(COLUMNS, value_input_option="RAW")
        header = list(COLUMNS)
    else:
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            header = _extend_header(ws, header, missing)
    _ws, _header = ws, header
    with _LOCK:
        _stats["worksheet"] = ws.title
    print(f"[sheet] target tab {ws.title!r} (gid {ws.id}), {len(header)} columns", flush=True)
    return ws, header


def _append(row: dict) -> None:
    global _ws, _header
    last = None
    for wait in RETRIES:
        if wait:
            time.sleep(wait)
        try:
            ws, header = _worksheet()
            ws.append_row([row.get(c, "") for c in header], value_input_option="RAW")
            with _LOCK:
                _stats["written"] += 1
                _stats["last_ok_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                _stats["last_error"] = None
            print(f"[sheet] row appended: session={row.get('Session')} outcome={row.get('Outcome')}",
                  flush=True)
            return
        except Exception as e:  # noqa: BLE001 — any failure is a retry, then a counter
            last = f"{type(e).__name__}: {str(e)[:160]}"
            _ws, _header = None, []               # re-open on the next attempt
    with _LOCK:
        _stats["failed"] += 1
        _stats["last_error"] = last
    print(f"[sheet] row DROPPED after {len(RETRIES)} attempts: {last}", flush=True)


def schedule_append(doc: dict, extra: dict | None = None) -> None:
    """Queue one result row. Never blocks, never raises."""
    try:
        if not is_configured():
            return
        row = row_from_doc(doc, extra)
        with _LOCK:
            _stats["queued"] += 1
        _POOL.submit(_append, row)
    except Exception as e:  # noqa: BLE001
        print(f"[sheet] could not queue row: {type(e).__name__}: {e}", flush=True)


# ------------------------------------------------------------ maintenance
def reorder_existing_tab() -> str:
    """Rewrite the target tab so its columns follow COLUMNS. Existing rows are
    re-mapped by header name (unknown columns are kept at the end). One-off
    tool: `python -m app.result_sheet --reorder`. Rows written after this by
    ANY build land correctly, because writes are placed by column name."""
    ws, header = _worksheet()
    rows = ws.get_all_values()
    if not rows:
        return "empty tab, nothing to do"
    old_header, body = rows[0], rows[1:]
    extra = [c for c in old_header if c not in COLUMNS]
    new_header = list(COLUMNS) + extra
    idx = {c: i for i, c in enumerate(old_header)}
    new_rows = [[(r[idx[c]] if c in idx and idx[c] < len(r) else "") for c in new_header]
                for r in body]
    ws.clear()
    if ws.col_count < len(new_header):
        ws.add_cols(len(new_header) - ws.col_count)
    ws.update(range_name="A1", values=[new_header] + new_rows, value_input_option="RAW")
    global _ws, _header
    _ws, _header = None, []                      # re-read the header next time
    return f"reordered {len(body)} row(s); {len(new_header)} columns" + (f" (kept extra: {extra})" if extra else "")


if __name__ == "__main__":
    import sys
    if "--reorder" in sys.argv:
        print(reorder_existing_tab())
    else:
        print(json.dumps(status(), indent=1))
