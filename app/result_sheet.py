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
    return round(f, nd) if f == f else ""      # NaN -> ""


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


def row_from_doc(doc: dict, extra: dict | None = None) -> dict:
    """Flatten one measure response into {column name: cell value}."""
    ev = _g(doc, "debug", "evidence", default={}) or {}
    ra = _g(doc, "debug", "rationale", default={}) or {}
    cm = doc.get("capture_meta") or {}
    tm = doc.get("timing") or {}
    dsc = doc.get("downscale") or {}
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
    ex = extra or {}
    ref = doc.get("reference") or {}
    cap = doc.get("client_capture") or {}
    return {
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
        # A tab created by hand is 26 columns wide (A-Z) and COLUMNS is 73, so
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
