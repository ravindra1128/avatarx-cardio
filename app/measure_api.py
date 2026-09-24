"""
Batch measure API (integration surface for the AvatarX webapp).

WHY THIS EXISTS
---------------
The webapp runs several vitals engines against ONE face scan, in parallel.
gm0513 streams JPEG frames over a WebSocket; this pipeline cannot do that
cheaply, because `/api/frames` on the live demo server takes RAW RGBA
(w*h*4 bytes per frame) and streaming that from a phone is ~20-40x the
bandwidth of the JPEG path. So the webapp instead hands us the recorded
clip once the scan ends, and we run THE production path on it:

    inference.pipeline.run_with_details  (same call cli.py process makes)

Nothing here reimplements analysis. This module is transport only.

THE CLOCK PROBLEM (the reason this file is not 40 lines)
--------------------------------------------------------
MediaRecorder .webm has no honest frame clock. On a real capture from the
webapp, OpenCV reports CAP_PROP_FPS = 1000.0 and a negative frame count.
capture.video_reader.iter_frames falls back to `index / fps` when the
container clock is degenerate — with fps=1000 that spaces frames 1 ms
apart and every interval downstream is nonsense.

iter_frames prefers a sidecar (`<video>.timestamps.json`) when one exists,
and calls it "the honest clock". So we build that sidecar here, in
priority order:

  1. client_timestamps_s  — per-frame timestamps measured in the browser
     (best: it is the capture clock itself)
  2. container clock      — CAP_PROP_POS_MSEC, repaired to be strictly
     increasing (measured: ~0.4% of steps are duplicates, worst backward
     jump 0.0 ms, so repair is a nudge, not a rewrite)
  3. uniform over duration_ms — wall-clock recording duration from the
     browser, spread across the decoded frame count

Every request reports which source was used and how many timestamps were
repaired, so a caller can tell a clean capture from a salvaged one.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import hmac
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from app import result_sheet

MAX_UPLOAD_BYTES = int(os.environ.get("AFIB_MAX_UPLOAD_MB", "256")) * 1024 * 1024
MAX_CONCURRENT = int(os.environ.get("AFIB_MAX_CONCURRENT", "2"))
ALLOW_ORIGIN = os.environ.get("AFIB_ALLOW_ORIGIN", "*")
WORK_DIR = pathlib.Path(os.environ.get("AFIB_WORK_DIR",
                                       tempfile.gettempdir())) / "afib_measure"

_POOL = ThreadPoolExecutor(max_workers=MAX_CONCURRENT,
                           thread_name_prefix="afib-measure")

from app.measure_overrides import LAUNCH_OVERRIDES  # noqa: E402
_INFLIGHT = threading.BoundedSemaphore(MAX_CONCURRENT)
# Observability for the failure mode a timed-out client can't see: when the
# browser aborts at its own deadline the server thread does NOT stop —
# fut.result() runs the whole job to completion regardless. Each abandoned
# scan is a zombie holding one of MAX_CONCURRENT slots and a full CPU share,
# and the next scan runs slower for it. /healthz now reports how many jobs
# are actually running, plus which build is serving, so this is visible.
_INFLIGHT_COUNT = 0
_INFLIGHT_LOCK = threading.Lock()

# A finished result outlives the connection that asked for it.
#
# WHY: a scan is ~10 s of upload followed by 20-45 s of analysis during which
# no bytes move, and a phone on a mobile radio does not always survive that
# quiet stretch — 2026-09-09 a scan died with ERR_HTTP2_PING_FAILED after the
# work was done, and the completed analysis was thrown away because the socket
# it belonged to was gone. Results are held briefly by unique upload ID so
# the client can collect its own scan without recording and uploading again.
# The bounded memory cache is backed by a TTL-limited SQLite result store so
# eviction does not lose unexpired results. AFIB_RESULT_DB must point
# to a persistent volume to survive container replacement; uploads stay transient.
RESULT_TTL_S = float(os.environ.get("AFIB_RESULT_TTL_S", "900"))    # 15 min
MAX_CACHED_RESULTS = int(os.environ.get("AFIB_MAX_CACHED_RESULTS", "32"))
_RESULTS: "OrderedDict[str, tuple]" = OrderedDict()
_RESULTS_LOCK = threading.Lock()
from app.result_store import ResultStore
_RESULT_STORE = ResultStore(os.environ.get("AFIB_RESULT_DB") or WORK_DIR / "results.sqlite3", RESULT_TTL_S)


def _remember_result(session, doc: dict) -> None:
    if not session or not isinstance(doc, dict):
        return
    # Delivery identity belongs to this upload, never a previous scan. These
    # additive fields do not classify a transport/runtime failure as a rhythm.
    doc["upload_id"] = doc["scan_id"] = str(session)
    doc["analysis_state"] = "failed" if doc.get("error") else "complete"
    created = time.time()
    doc["result_recovery"] = {"persisted": True, "expires_at_unix": created + RESULT_TTL_S,
                              "scope": "completed_results_on_this_filesystem"}
    stored = _RESULT_STORE.put(str(session), doc, created=created)
    doc["result_recovery"]["persisted"] = stored
    if not stored:
        print(f"[measure] result recovery storage unavailable: {_RESULT_STORE.last_error}", flush=True)
    with _RESULTS_LOCK:
        _RESULTS[str(session)] = (created, doc)
        _RESULTS.move_to_end(str(session))
        while len(_RESULTS) > MAX_CACHED_RESULTS:
            _RESULTS.popitem(last=False)


# ---- Chunked upload + detached processing (2026-09-09) ---------------------
# One request carrying 23 MB and then holding the line through 40 s of analysis
# is 80-105 s of connection on a phone, and it kept dying (ERR_HTTP2_PING_FAILED,
# and a 502 at Railway's ~300 s request ceiling on a large clip). Neither
# heartbeats nor a result cache can save an upload that dies mid-body, so the
# work is split:
#   POST /api/upload-part  one slice, retryable on its own, seconds long
#   POST /api/start        assembles and returns 202 at once; the job detaches
#   GET  /api/result       the finished document, or 202 while it runs
# The single-shot /api/process-video path is untouched, so an older client
# keeps working exactly as before.
UPLOAD_DIR = WORK_DIR / "parts"

# Retained scan clips, so a change can be replayed against REAL phone captures
# instead of costing a 3-minute mobile scan per candidate. The eval corpus has
# no 480x720 portrait clip - the only shape production records - which is why
# the optimizer gate could not judge the 2026-09-10 downscale change at all.
#
# BOTH env vars are required, and neither is set on production. This serves
# recorded video of someone's face from a PUBLIC URL, so it stays off unless
# deliberately switched on, and the token is not optional.
CLIPS_DIR = WORK_DIR / "clips"
CLIPS_TOKEN = os.environ.get("AFIB_CLIPS_TOKEN", "").strip()
CLIPS_KEEP = int(os.environ.get("AFIB_CLIPS_KEEP", "12"))       # newest N
CLIPS_MAX_BYTES = int(os.environ.get("AFIB_CLIPS_MAX_MB", "600")) * 1024 * 1024

# A sidecar travels with its clip: same basename plus a suffix. Listed once
# because three places defined "a clip" as "not .timestamps.json" - the
# retention filter, the prune victim tuple and the /api/clips listing - and a
# second suffix silently breaks all three: a 30 KB JSON would count as a clip,
# take a slot in the newest-CLIPS_KEEP window and evict a real recording, and
# pruning that recording would orphan its JSON forever on an ephemeral disk.
CLIP_SIDECAR_SUFFIXES = (".timestamps.json", ".shenai.json")

# ShenAI's own dense PPG waveform and beat train, posted to /api/scan-signals
# as JSON and parked under this FIXED name inside the upload's part directory:
# it must never match _assemble_parts' `part-*` glob, and it must not depend on
# whatever `ext` the client declares later.
SHENAI_PART_NAME = "shenai.json"
# The retained basename is not recoverable from upload_id (sid is truncated to
# 16 chars below), and the sidecar normally arrives AFTER retention has run, so
# _assemble_parts leaves the pairing here for the job to pick up.
RETAINED_POINTER_NAME = "retained"
# That route only ever carries JSON. Measured payload is ~20-50 KB (48 s of
# ShenAI PPG plus ~60 beats), so 4 MB is ~100x headroom - while keeping
# MAX_UPLOAD_BYTES' 256 MB away from a route that writes onto the same small
# ephemeral disk the retained clips live on.
MAX_SIDECAR_BYTES = (int(os.environ.get("AFIB_MAX_SIDECAR_MB", "4"))
                     * 1024 * 1024)


_TRUTHY = {"1", "true", "yes", "on"}


def _clips_token() -> str:
    """The read-back token, read LIVE.

    CLIPS_TOKEN above is frozen at import. On 2026-09-12 the owner added
    AFIB_CLIPS_TOKEN and AFIB_KEEP_UPLOADS in Railway, ran eight scans, and
    every ShenAI column came back blank: the process had been up 14 h, so the
    token it held was the empty string it started with. A variable set in the
    console must work on the next request, not after the next redeploy."""
    return (os.environ.get("AFIB_CLIPS_TOKEN") or CLIPS_TOKEN or "").strip()


def _clips_gate_reason():
    """Why retention is OFF, or None when it is on. For the PRIVATE log only -
    never the token itself, only whether one is present."""
    keep = (os.environ.get("AFIB_KEEP_UPLOADS") or "").strip()
    if keep.lower() not in _TRUTHY:
        # The old check was `== "1"`: `true` and `True` failed silently. Any
        # of these spellings is an unambiguous intent to switch retention ON;
        # nobody types `true` meaning off, so accepting them cannot leak.
        return f"AFIB_KEEP_UPLOADS is {keep!r} (need one of 1/true/yes/on)"
    if not _clips_token():
        return "AFIB_CLIPS_TOKEN is unset or empty"
    return None


def _clips_enabled() -> bool:
    return _clips_gate_reason() is None


# Live-frame trace retention WITHOUT any video (owner, 2026-09-24: "keep only
# traces"). The Vascular Tone card is computed from the trace document the
# phone posts to /api/scan-traces - four face regions' mean colour per frame,
# no image - which was held in memory only, so a phone pair that read 0.35 %
# and 0.81 % three minutes apart could not be examined. Its own switch, its
# own directory, independent of clip retention (AFIB_KEEP_UPLOADS may stay
# off); read back only with the same token as the clips. Newest N kept.
TRACES_DIR = WORK_DIR / "traces"
TRACES_KEEP = int(os.environ.get("AFIB_TRACES_KEEP", "50"))
TRACE_FILE_SUFFIX = ".traces.json"


def _traces_gate_reason():
    """Why trace retention is OFF, or None when it is on (private log only)."""
    keep = (os.environ.get("AFIB_KEEP_TRACES") or "").strip()
    if keep.lower() not in _TRUTHY:
        return f"AFIB_KEEP_TRACES is {keep!r} (need one of 1/true/yes/on)"
    if not _clips_token():
        return "AFIB_CLIPS_TOKEN is unset or empty"
    return None


def _traces_enabled() -> bool:
    return _traces_gate_reason() is None


def _retain_traces(upload_id: str, raw_traces, doc: dict) -> None:
    """Store this scan's trace document, plus a short note (scan id, the SDK's
    heart rate and quality, the Vascular Tone the service computed), as one
    JSON file. Never video, never the clip. Best effort: a scan returns
    whether or not this works."""
    if not _traces_enabled() or not isinstance(raw_traces, dict):
        return
    try:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in str(upload_id) if c.isalnum() or c in "-_")[:40] or "scan"
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dst = TRACES_DIR / f"{stamp}-{safe}{TRACE_FILE_SUFFIX}"
        debug = (doc or {}).get("debug") or {}
        shen = debug.get("shenai_input") or {}
        ref = (doc or {}).get("reference") if isinstance((doc or {}).get("reference"), dict) else {}
        note = {"upload_id": str(upload_id), "stored_at": stamp, "build": BUILD_SHA,
                "sdk_hr_bpm": shen.get("sdk_hr_bpm"), "sdk_quality": shen.get("sdk_quality"),
                "ref_hr": (ref or {}).get("ref_hr"), "outcome": (doc or {}).get("outcome"),
                "vascular_tone": debug.get("vascular_tone")}
        dst.write_text(json.dumps({"service": note, "trace_document": raw_traces}, default=str))
        files = sorted(TRACES_DIR.glob("*" + TRACE_FILE_SUFFIX),
                       key=lambda q: q.stat().st_mtime, reverse=True)
        for q in files[max(TRACES_KEEP, 1):]:
            try:
                q.unlink()
            except OSError:
                pass
        print(f"[traces] retained {dst.name} ({dst.stat().st_size / 1e3:.0f} kB)", flush=True)
    except Exception as e:                                     # noqa: BLE001
        print(f"[traces] retain failed: {type(e).__name__}: {e}", flush=True)


def _retain_clip(video_path: str, sid: str):
    """Keep an untouched copy of the upload before prep rewrites it in place.

    Called on BOTH upload paths. The chunked path is the one a phone actually
    uses, and it never retained anything - so AFIB_KEEP_UPLOADS looked enabled
    while quietly doing nothing for every real scan.

    Returns the retained path (or None). The caller needs it because the
    ShenAI sidecar is posted AFTER /api/start is accepted - so that it can
    never contribute to an upload failure - and retention runs at the TOP of
    /api/start. Without the returned name the late sidecar could not be paired
    with the clip it belongs to.
    """
    if not _clips_enabled():
        return None
    try:
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        src = pathlib.Path(video_path)
        ext = src.suffix.lstrip(".") or "webm"
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dst = CLIPS_DIR / f"{stamp}-{sid[:16]}.{ext}"
        shutil.copyfile(src, dst)
        side = src.parent / (src.name + ".timestamps.json")
        if side.exists():
            shutil.copyfile(side, str(dst) + ".timestamps.json")
        # ShenAI's signals, when the phone happened to post them before
        # /api/start. Named after the CLIP, not the upload_id, because that is
        # the only name that exists here - and the full upload_id is carried
        # INSIDE the JSON, so the pairing stays verifiable offline even though
        # sid is truncated to 16 chars above.
        shen = src.parent / SHENAI_PART_NAME
        if shen.exists():
            shutil.copyfile(shen, str(dst) + ".shenai.json")
        # Railway's disk is ephemeral and small: keep the newest N, and stay
        # under a byte cap. Oldest go first.
        kept = sorted(CLIPS_DIR.glob("*"), key=lambda q: q.stat().st_mtime,
                      reverse=True)
        vids = [q for q in kept if not q.name.endswith(CLIP_SIDECAR_SUFFIXES)]
        total = 0
        for i, q in enumerate(vids):
            total += q.stat().st_size
            if i >= CLIPS_KEEP or total > CLIPS_MAX_BYTES:
                for victim in [q] + [pathlib.Path(str(q) + s)
                                     for s in CLIP_SIDECAR_SUFFIXES]:
                    try:
                        victim.unlink()
                    except OSError:
                        pass
        print(f"[clips] retained {dst.name} "
              f"({src.stat().st_size / 1e6:.1f} MB)", flush=True)
        return dst
    except Exception as e:                                    # noqa: BLE001
        # Never let retention affect a scan.
        print(f"[clips] retain failed: {e}", flush=True)
        return None


def _shenai_summary(raw: dict) -> dict:
    """Counts only, never the waveform: a sheet cell must not carry
    physiological data, and the sheet must not re-derive anything the offline
    harness owns (scripts/compare_shenai_signal.py). The document itself is
    stored VERBATIM - this is the bounded view of it that rides the result doc.

    Every field is coerced and truncated the same way _parse_envelope treats
    ref_source/capture_note: the body is an anonymous 4 MB POST, so a string
    copied straight through could carry megabytes into a doc and a sheet cell.
    """
    def _i(v):
        try:
            return int(v)
        # OverflowError, not ValueError, is what int(float('inf')) raises, and
        # a JSON body may legally say 1e999. Before 2026-09-11 that escaped
        # do_POST after the sidecar had already been written: the phone got
        # RemoteDisconnected instead of its documented 200.
        except (TypeError, ValueError, OverflowError):
            return None

    def _f(v):
        try:
            f = float(v)
        except (TypeError, ValueError, OverflowError):
            return None
        # NaN AND +/-inf are both rejected: neither is a count, and either one
        # reaching json.dumps emits a bare non-JSON token (see _json_bytes).
        return f if math.isfinite(f) else None

    def _s(v, n):
        return str(v)[:n] if isinstance(v, (str, int, float)) else None

    ppg = raw.get("ppg") if isinstance(raw.get("ppg"), dict) else {}
    beats = raw.get("heartbeats")
    hist = raw.get("hr_history_10s")
    return {
        "present": True,
        "schema_version": _i(raw.get("schema_version")),
        "upload_id": _s(raw.get("upload_id"), 80),
        "ppg_n": _i(ppg.get("n")),
        "ppg_fs_hz": _f(ppg.get("fs_hz")),
        # "sdk" | "derived_from_beats" | "derived_from_duration" | "unknown".
        # Kept because a DERIVED rate must never be mistaken for a measured
        # one: the SDK exposes no sample rate at all (index.d.ts:374-377).
        "ppg_fs_source": _s(ppg.get("fs_source"), 40),
        "ppg_truncated": bool(ppg.get("truncated")),
        "beats_n": len(beats) if isinstance(beats, list) else 0,
        "hr_history_n": len(hist) if isinstance(hist, list) else 0,
    }


def _pair_shenai(part_dir, doc: dict, retained=None) -> None:
    """Pair a late-arriving ShenAI sidecar with its retained clip, and record
    its counts on the result doc.

    WHY this is not just two lines inside _retain_clip: the client posts the
    signals only AFTER /api/start has been accepted (a sidecar that competed
    with the upload could cost a scan whose bytes are already on the server),
    while retention runs at the top of /api/start. So on a normal scan
    _retain_clip has already run and seen nothing, and this second pass - at
    the end of the 20-45 s job, by which time the POST has landed - is where
    the pairing actually happens.

    Best effort throughout: a scan must return its cards whether or not any of
    this works.

    GATED, like every other member of this family (2026-09-11). _retain_clip,
    _upload_signals and _serve_clip all check _clips_enabled() before they
    touch disk; this function did not, and it is the LAST writer in the chain.
    An adversarial review proved the gap: with retention OFF, a stale part-dir
    sidecar plus a clip left over from a gate-ON period made this copy a full
    waveform into CLIPS_DIR and set doc["shenai"], which the sheet then
    rendered as ShenAI Sidecar=TRUE. That made the invariant the retention
    tests assert in prose - "_clips_enabled() is the single thing standing
    between a public URL and persisted physiological data" - false in fact.
    """
    if not _clips_enabled():
        return
    try:
        src = pathlib.Path(part_dir) / SHENAI_PART_NAME
        if not src.exists():
            return
    except (OSError, TypeError, ValueError):
        return
    try:
        if retained is None:
            # _assemble_parts left the retained basename here: it is the only
            # place the clip's real name survives (sid is truncated to 16
            # chars, so the retained name can be pure session prefix).
            ptr = pathlib.Path(part_dir) / RETAINED_POINTER_NAME
            if ptr.exists():
                name = os.path.basename(ptr.read_text().strip())
                retained = CLIPS_DIR / name if name else None
        if retained is not None:
            paired = pathlib.Path(str(retained) + ".shenai.json")
            if pathlib.Path(retained).exists() and not paired.exists():
                shutil.copyfile(src, paired)
                print(f"[clips] paired {paired.name}", flush=True)
    except Exception as e:                                    # noqa: BLE001
        print(f"[clips] shenai pairing failed: {e}", flush=True)
    try:
        raw = json.loads(src.read_text())
        if isinstance(raw, dict):
            doc["shenai"] = _shenai_summary(raw)               # additive; audit only
    except Exception as e:                                    # noqa: BLE001
        print(f"[measure] shenai summary failed: {e}", flush=True)
# Slices are sized to the client's measured link, down to 256 KB on a slow
# one, so a clip near MAX_UPLOAD_BYTES can arrive as ~1000 parts.
MAX_UPLOAD_PARTS = 2048
_STARTED: dict = {}                  # upload_id -> True while a job is running
_STARTED_LOCK = threading.Lock()


def _discard_part_dir(upload_id: str) -> None:
    _drop_signals(upload_id)
    """Remove a finished job's working directory (assembled clip, FFV1
    intermediate, timestamp/ShenAI sidecars). Kept only when AFIB_KEEP_UPLOADS
    asks for on-disk debugging; the retained clip lives in CLIPS_DIR."""
    keep = (os.environ.get("AFIB_KEEP_UPLOADS") or "").strip().lower()
    if keep in _TRUTHY:
        return
    try:
        d = _part_dir(upload_id)
    except Exception:                                     # noqa: BLE001
        return
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)


def _part_dir(upload_id: str) -> pathlib.Path:
    # A single path segment, so an id can never escape the parts directory.
    safe = "".join(c for c in str(upload_id) if c.isalnum() or c in "-_")[:80]
    if not safe:
        raise ValueError("bad upload_id")
    return UPLOAD_DIR / safe


def _assemble_parts(upload_id: str, ext: str) -> str:
    """Concatenate the received slices in index order. Raises if any is
    missing: a gap means silent corruption, never a shorter video."""
    d = _part_dir(upload_id)
    parts = sorted(d.glob("part-*"))
    if not parts:
        raise ValueError("no parts received for that upload")
    # The phone's real upload duration: first slice landing -> now. The
    # single-shot path reports this as upload_received_s and the sheet reads
    # it; without it the sliced path silently lost both upload columns.
    first_at = min(p.stat().st_mtime for p in parts)
    total = int((d / "total").read_text().strip())
    if len(parts) != total:
        have = {int(p.name.split("-")[1]) for p in parts}
        missing = sorted(set(range(total)) - have)[:8]
        raise ValueError(f"upload incomplete: {len(parts)}/{total} parts, "
                         f"missing {missing}")
    out = d / f"scan.{ext}"
    with open(out, "wb") as fh:
        for part in parts:
            fh.write(part.read_bytes())
            part.unlink()
    # Retain BEFORE prep: trim and downscale rewrite this file in place, and
    # the whole point is to replay the native capture later.
    retained = _retain_clip(str(out), d.name)
    if retained is not None:
        # Leave the retained basename in the part dir. The ShenAI sidecar
        # arrives after this point (the client posts it once /api/start is
        # accepted), and _run_assembled has `path` - hence this dir - but no
        # other way to learn the name the clip was retained under. Not
        # "part-*", so a re-assembly can never mistake it for a slice.
        try:
            (d / RETAINED_POINTER_NAME).write_text(retained.name)
        except OSError:
            pass
    return str(out), max(0.0, time.time() - first_at)


def _sweep_parts(max_age_s: float = 3600.0) -> None:
    """Drop abandoned part directories: a phone that never came back."""
    try:
        now = time.time()
        for d in UPLOAD_DIR.glob("*"):
            with _STARTED_LOCK:
                if d.name not in _STARTED and d.is_dir() and now - d.stat().st_mtime > max_age_s:
                    shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


def _recall_result(session):
    """The most recent completed result for `session`, or None. Expired
    entries are dropped on the way past."""
    now = time.time()
    with _RESULTS_LOCK:
        for k in [k for k, (ts, _) in _RESULTS.items()
                  if now - ts > RESULT_TTL_S]:
            _RESULTS.pop(k, None)
        item = _RESULTS.get(str(session or ""))
    if item is not None:
        return item[1]
    return _RESULT_STORE.get(str(session)) if session else None
_STARTED_AT = time.time()
# Say which way the gate is set, ONCE, in the private service log. A gate that
# failed silently cost eight real scans of nothing on 2026-09-12. The public
# /healthz must never carry this: whether a URL holds face video is exactly
# the oracle the token exists to deny.
print("[clips] retention " + ("ON" if _clips_enabled()
                              else f"OFF: {_clips_gate_reason()}"), flush=True)
print("[traces] retention " + ("ON (traces only, no video)" if _traces_enabled()
                               else f"OFF: {_traces_gate_reason()}"), flush=True)
BUILD_SHA = (os.environ.get("RAILWAY_GIT_COMMIT_SHA")      # set by Railway
             or os.environ.get("AFIB_BUILD_SHA") or "unknown")[:12]


def _inflight(delta: int) -> int:
    global _INFLIGHT_COUNT
    with _INFLIGHT_LOCK:
        _INFLIGHT_COUNT += delta
        return _INFLIGHT_COUNT


from app.measure_prep import DEFAULT_SCALE, DEFAULT_WINDOW_S, trim_tail, downscale  # noqa: E402,F401

# ----------------------------------------------------------------- clock
def build_timestamp_sidecar(video_path: str, *,
                            client_timestamps_s=None,
                            duration_ms=None) -> dict:
    """Write `<video>.timestamps.json` and return a provenance dict.

    Never raises: a caller that cannot produce a sidecar still gets a
    result, it just carries a weaker clock and says so."""
    import cv2
    import numpy as np

    info = {"source": None, "n_frames": 0, "repaired": 0,
            "median_dt_ms": None, "implied_fps": None, "warnings": []}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        info["warnings"].append("cannot open video to build clock")
        return info
    pos = []
    n = 0
    while True:
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, _ = cap.read()
        if not ok:
            break
        pos.append(t_ms)
        n += 1
    cap.release()
    info["n_frames"] = n
    if n == 0:
        info["warnings"].append("video decoded zero frames")
        return info

    ts = None
    if client_timestamps_s is not None and len(client_timestamps_s) == n:
        ts = np.asarray(client_timestamps_s, float)
        info["source"] = "client_timestamps"
    elif client_timestamps_s is not None:
        info["warnings"].append(
            f"client sent {len(client_timestamps_s)} timestamps for {n} "
            f"decoded frames — ignoring them")

    if ts is None:
        p = np.asarray(pos, float) / 1000.0
        usable = (np.all(np.isfinite(p)) and p.size > 1
                  and float(p[-1]) > 0.0 and float(np.max(np.diff(p))) > 0.0)
        if usable:
            ts = p
            info["source"] = "container_clock"

    if ts is None and duration_ms:
        dur = float(duration_ms) / 1000.0
        if dur > 0:
            ts = np.arange(n, dtype=float) * (dur / max(n, 1))
            info["source"] = "uniform_over_duration"

    if ts is None:
        info["warnings"].append(
            "no usable clock (container degenerate, no client timestamps, "
            "no duration) — falling back to pipeline default")
        return info

    # Strictly increasing, as iter_frames' sidecar loader demands.
    d = np.diff(ts)
    step = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0 / 30.0
    repaired = 0
    for i in range(1, ts.size):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + step
            repaired += 1
    info["repaired"] = repaired
    if repaired:
        info["warnings"].append(
            f"repaired {repaired} non-increasing timestamps "
            f"({100.0 * repaired / max(ts.size - 1, 1):.2f}% of steps)")

    d2 = np.diff(ts)
    # Scalar provenance for whole-video coverage. Ingest later omits frames
    # without a usable face, so its ROI timestamps cannot define this span.
    info["first_frame_s"] = float(ts[0])
    info["last_frame_s"] = float(ts[-1])
    info["span_s"] = float(ts[-1] - ts[0])
    info["frame_step_p99_ms"] = float(np.percentile(d2, 99) * 1000) if d2.size else None
    info["frame_step_max_ms"] = float(np.max(d2) * 1000) if d2.size else None
    info["median_dt_ms"] = round(float(np.median(d2)) * 1000.0, 3)
    info["implied_fps"] = round(1.0 / float(np.median(d2)), 2)
    with open(video_path + ".timestamps.json", "w") as f:
        json.dump({"timestamps_s": [float(x) for x in ts]}, f)
    return info


# ------------------------------------------------------------- pipeline
def _apply_overrides(cfg: dict, overrides: dict) -> list:
    """Apply {"capture.fps_min": 25} style dotted overrides onto a loaded
    config. Returns a list of human-readable notes.

    These exist so an integrator can MEASURE what a looser capture
    threshold would do without editing configs/default.yaml. They are
    opt-in per request, never a default, and every applied override is
    echoed back in the response so a loosened run can never be mistaken
    for a clean one."""
    notes = []
    for dotted, value in (overrides or {}).items():
        parts = str(dotted).split(".")
        node = cfg
        try:
            for k in parts[:-1]:
                node = node[k]
            before = node.get(parts[-1])
            node[parts[-1]] = value
            notes.append(f"{dotted}: {before} -> {value}")
        except (KeyError, TypeError, AttributeError):
            notes.append(f"{dotted}: IGNORED (no such config key)")
    return notes


_REPO_DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def _protect_input(video_path: str, protected_root: pathlib.Path = _REPO_DATA) -> str:
    """measure_video trims and downscales IN PLACE — the upload is a temp
    file, so that is the cheap thing to do. A recording under the repo's
    data/ directory is never a temp file: on 2026-09-09 a diagnostic called
    measure_video on the evaluation corpus and destroyed eight phone
    recordings. Such an input (and its sidecar) is copied to a temp dir and
    the copy is measured; the caller's file is never touched."""
    try:
        p = pathlib.Path(video_path).resolve()
        root = pathlib.Path(protected_root).resolve()
        if root not in p.parents:
            return video_path
    except OSError:
        return video_path
    d = tempfile.mkdtemp(prefix="afib_measure_copy-")
    dst = os.path.join(d, p.name)
    shutil.copy(p, dst)
    side = str(p) + ".timestamps.json"
    if os.path.exists(side):
        shutil.copy(side, dst + ".timestamps.json")
    print(f"[measure] input under {root}: measuring a copy at {dst}", flush=True)
    return dst


# The user-entered profile behind the fitness card's VO2max estimate
# (features/vo2max.py). It travels in a request BODY - the /api/start JSON, the
# measure envelope's header, the trace document - and never in a query string:
# age, sex, height and weight are personal data, and URLs are what proxies and
# platform logs keep. Only the keys the equation reads survive, as plain
# scalars, so nothing else a client sends can ride along into the response or
# the sheet.
PARTICIPANT_KEYS = ("age", "age_years", "sex", "height_cm", "weight_kg",
                    "measured_weight_kg", "activity_level")
MAX_START_BODY_BYTES = 16 * 1024


def _participant_block(doc):
    raw = doc.get("participant") if isinstance(doc, dict) else None
    if not isinstance(raw, dict):
        return None
    out = {}
    for k in PARTICIPANT_KEYS:
        v = raw.get(k)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, str) and len(v) <= 16:
            out[k] = v.strip()
    return out or None


def measure_video(video_path: str, *, manifest=None,
                  client_timestamps_s=None, duration_ms=None,
                  config_overrides=None, window_s=None, scale=None,
                  participant=None, reference=None) -> dict:
    """Run THE production path and return a JSON-ready ScanResult."""
    doc, _ = measure_video_details(
        video_path, manifest=manifest, client_timestamps_s=client_timestamps_s,
        duration_ms=duration_ms, config_overrides=config_overrides,
        window_s=window_s, scale=scale, participant=participant,
        reference=reference)
    return doc


def measure_video_details(video_path: str, *, manifest=None,
                          client_timestamps_s=None, duration_ms=None,
                          config_overrides=None, window_s=None,
                          scale=None, participant=None,
                          reference=None) -> tuple:
    """measure_video, plus the pipeline's own details dict (`det`: config,
    evidence, sqi, runset, min_conf ...) for a caller that runs a second
    interval source through the same decision (inference/shenai_route.py)
    after the sidecar it needs has landed. The doc is what measure_video
    returns, unchanged."""
    video_path = _protect_input(video_path)
    # TRIM BEFORE DOWNSCALE. Both orders give the same analysis window, but
    # trim_tail is an ffmpeg stream COPY (no re-encode, ~instant) while
    # downscale is a full FFV1 re-encode — the most expensive step in the
    # request. Downscaling first re-encoded the whole clip and then threw
    # away everything outside the window: with a 75 s recording trimmed to
    # 40 s that was ~47% of the encode wasted. Trimming first hands the
    # encoder only the frames that will actually be analysed.
    # Stage timing, printed as each stage finishes (not batched at the end):
    # a request that times out client-side still leaves this trail in the
    # platform's log stream, which is otherwise the only way to tell an
    # upload-bound scan from a downscale-bound one from an analysis-bound
    # one on a host whose CPU hasn't been measured yet (Railway's shared
    # vCPU vs the Mac these components were originally tuned on).
    timing = {}                      # server-side stage durations, seconds
    t0 = time.perf_counter()
    trim = trim_tail(video_path,
                     DEFAULT_WINDOW_S if window_s is None else float(window_s))
    timing["trim_s"] = round(time.perf_counter() - t0, 2)
    print(f"[measure] trim: {timing['trim_s']}s applied={trim.get('applied')} "
          f"{trim.get('note') or trim.get('reason') or ''}", flush=True)

    t0 = time.perf_counter()
    scaled = downscale(video_path, DEFAULT_SCALE if scale is None else scale)
    timing["downscale_s"] = round(time.perf_counter() - t0, 2)
    print(f"[measure] downscale: {timing['downscale_s']}s "
          f"applied={scaled.get('applied')} reason={scaled.get('reason')}",
          flush=True)
    if scaled.get("applied") and scaled.get("path"):
        video_path = scaled["path"]              # container changed on downscale
    if scaled.get("applied") or trim.get("applied"):
        # Client-side per-frame timestamps describe the untrimmed clip.
        client_timestamps_s = None
    clock = build_timestamp_sidecar(video_path,
                                    client_timestamps_s=client_timestamps_s,
                                    duration_ms=duration_ms)
    from inference.pipeline import run_with_details

    cfg = None
    override_notes = []
    # Launch-level config overrides (2026-09-17): AFIB_CONFIG_OVERRIDES is a
    # JSON object of dotted keys, e.g. {"decision.classifier": "model_a",
    # "decision.model_a_path": "models/model_a_v02_45s.json"} - the staging
    # switch that turns the validated RR classifier on without editing
    # configs/*.yaml. Applied through the SAME _apply_overrides as a request's
    # own overrides (a request's keys win), and echoed on every response and
    # on /healthz under launch_overrides, so a run under it can never be
    # mistaken for the yaml default.
    merged = dict(LAUNCH_CONFIG_OVERRIDES)
    merged.update(config_overrides or {})
    if merged:
        from configs import load_config
        cfg = load_config()
        override_notes = _apply_overrides(cfg, merged)

    t0 = time.perf_counter()
    result, det = run_with_details(video_path, manifest=manifest or {},
                                   config=cfg)
    timing["analysis_s"] = round(time.perf_counter() - t0, 2)
    timing["server_total_s"] = round(sum(timing.values()), 2)
    print(f"[measure] pipeline analysis: {timing['analysis_s']}s "
          f"outcome={result.outcome.value}", flush=True)
    doc = dataclasses.asdict(result)
    doc["outcome"] = result.outcome.value
    doc["user_facing_text"] = result.user_facing_text()
    doc["clock"] = clock
    doc["trim"] = trim
    doc["downscale"] = scaled
    # The same stage durations the log carries, returned to the caller: the
    # split between trim / downscale / analysis is otherwise only visible
    # in a platform log stream the client (and anyone without log access)
    # can't see. Upload time is NOT here — it belongs to the request, not
    # the job — the client measures that side itself.
    doc["timing"] = timing
    # cardio v0.8: the resting biomarker cards and the five gated research
    # tracks. Both are the pipeline's OWN research surfaces (spec B.24) —
    # computed from the same `det`, carrying their own labels, limitations,
    # warnings and live gate status, so a value and the reason it may not
    # be shown are never separated. Their own payload keys, never
    # head_results; the consumer boundary is untouched. Fail-soft: a
    # research panel must never break a measure response.
    try:
        from app.report_data import report_biomarkers
        # The client's lock state must reach the cards, not only the sheet:
        # without it the vascular-tone card records optics_locked=False on
        # every scan and carries an "optics not locked" caveat that its own
        # request contradicts (found 2026-09-09). The request handler merges
        # header["manifest"] — which carries exposure_locked / awb_locked —
        # into `manifest`, so that is the dict the cards need. NOTE: this
        # whole block is fail-soft, so a NameError here would silently
        # replace every card with an error dict; tests/test_biomarker_wiring.py
        # pins it.
        # `participant` (the user-entered profile) and `reference` (the
        # live-frame rate of the same scan) reach the FITNESS card only, and
        # never the pipeline above: the rhythm decision, the pulse and the
        # other two cards are computed exactly as they were without them.
        doc["biomarkers"] = report_biomarkers(
            result, det, capture=manifest or {}, participant=participant,
            reference=reference)
    except Exception as e:                       # noqa: BLE001
        doc["biomarkers"] = {"error": f"{type(e).__name__}: {e}"}
    # The five gated research tracks are OFF by default here. They cost a
    # SECOND full video decode (torso motion for the respiration channel),
    # which roughly doubles request time, and the /beta/cardio results page
    # renders only the three resting biomarker cards above — so that work
    # was being computed and thrown away. Set AFIB_RESEARCH_TRACKS=1 to
    # compute them again; the payload key and contents are unchanged then.
    if os.environ.get("AFIB_RESEARCH_TRACKS", "").strip().lower() in (
            "1", "true", "yes", "on"):
        try:
            from app.research_tracks import research_tracks_block
            doc["research_tracks"] = research_tracks_block(
                result, det, cfg, video_path=video_path) or None
        except Exception as e:                   # noqa: BLE001
            doc["research_tracks"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        doc["research_tracks"] = None
    doc["config_overrides"] = override_notes or None
    doc["launch_overrides"] = LAUNCH_OVERRIDES or None
    doc["debug"] = {"rationale": det.get("rationale"),
                    "evidence": det.get("evidence")}
    from app.scan_evidence import record_summary, video_duration_summary
    record_summary(doc, "video_duration", video_duration_summary, doc, det, duration_ms)
    # Every completed scan carries exactly one AFib result. The assembled job
    # recomputes it after the ShenAI route has run; here it is the video
    # path's own answer, so single-shot callers and replays get one too.
    _apply_afib_result(doc, det)
    return doc, det


def measure_traces(payload: dict, *, manifest=None, config_overrides=None) -> tuple:
    """Standalone AFib path: THE pipeline on live-frame ROI traces the client
    sampled itself (no video clip, no ShenAI). inference/trace_ingest.py turns
    the document into an IngestResult and runs the pipeline on it; the response
    has the SAME shape as a video scan (afib_result, biomarkers,
    user_facing_text), so /beta/cardio-afib renders it the same way. Honors
    AFIB_CONFIG_OVERRIDES so the standalone scan uses the validated classifier
    and its decision band, exactly as the video path does."""
    from inference import trace_ingest
    m = dict(manifest or {})
    m.setdefault("capture_profile", "consumer")
    cfg = None
    override_notes = []
    merged = dict(LAUNCH_CONFIG_OVERRIDES)
    merged.update(config_overrides or {})
    if merged:
        from configs import load_config
        cfg = load_config()
        override_notes = _apply_overrides(cfg, merged)
    upload_id = str((payload or {}).get("upload_id") or uuid.uuid4().hex[:12])
    t0 = time.perf_counter()
    result, det = trace_ingest.run_on_traces(payload, manifest=m, config=cfg,
                                             upload_id=upload_id)
    analysis_s = round(time.perf_counter() - t0, 2)
    doc = dataclasses.asdict(result)
    doc["outcome"] = result.outcome.value
    doc["user_facing_text"] = result.user_facing_text()
    doc["rhythm_source"] = "client_traces"
    doc["timing"] = {"analysis_s": analysis_s, "server_total_s": analysis_s}
    try:
        from app.report_data import report_biomarkers
        doc["biomarkers"] = report_biomarkers(
            result, det, capture=m, participant=_participant_block(payload))
    except Exception as e:                       # noqa: BLE001
        doc["biomarkers"] = {"error": f"{type(e).__name__}: {e}"}
    doc["research_tracks"] = None
    doc["config_overrides"] = override_notes or None
    doc["launch_overrides"] = LAUNCH_OVERRIDES or None
    doc["debug"] = {"rationale": det.get("rationale"),
                    "evidence": det.get("evidence")}
    _apply_vascular_tone(doc, payload, payload if isinstance(payload, dict) else {})
    ing = (det or {}).get("ingest")
    meta = getattr(ing, "meta", None)
    # Per-region signal diagnostic (2026-09-18): the std and peak-to-peak of
    # each region's green channel over the kept frames. A near-zero std means
    # the frames the client sent do not vary (frozen/static or a dead region);
    # a healthy pulse rides ~0.5-2 counts on a ~100-160 base. This is what
    # tells a misplaced-region bug apart from a static-frame bug on mobile.
    roi_green = {}
    try:
        import numpy as _np
        tr = getattr(ing, "traces", None) or {}
        for _r, _a in tr.items():
            _a = _np.asarray(_a, float)
            if _a.ndim == 2 and _a.shape[0] > 1:
                g = _a[:, 1][_np.isfinite(_a[:, 1])]
                if g.size > 1:
                    roi_green[_r] = {"n": int(g.size), "mean": round(float(_np.mean(g)), 2),
                                     "std": round(float(_np.std(g)), 3),
                                     "ptp": round(float(_np.ptp(g)), 2)}
    except Exception:                                          # noqa: BLE001
        pass
    doc["trace_ingest"] = {
        "roi_green": roi_green,
        "ingest_ok": bool(getattr(ing, "ok", False)),
        "fps": getattr(meta, "measured_fps_mean", None),
        "jitter_ms": getattr(meta, "measured_fps_jitter_ms", None),   # raw clock; high => bursty, resampled
        "n_frames": getattr(meta, "n_frames", None),
        "duration_s": getattr(meta, "duration_s", None),
        "tracking_stability": getattr(getattr(ing, "track", None), "stability", None),
        "reasons": list(getattr(ing, "reasons", []) or []),
        "sampler": (payload or {}).get("sampler") if isinstance((payload or {}).get("sampler"), dict) else None,
    }
    _apply_afib_result(doc, det)
    print(f"[measure] traces: outcome={doc.get('outcome')} class={doc.get('predicted_class')} "
          f"result={doc.get('afib_result')} pulse={doc.get('mean_pulse_rate_bpm')} "
          f"in {analysis_s}s", flush=True)
    print(f"[measure] traces roi_green: {roi_green}", flush=True)
    return doc, det


# ---------------------------------------------------- ShenAI route (2026-09-16)
# The sidecar the phone posts to /api/scan-signals is HELD IN MEMORY for the
# job that is running for that upload, so the rhythm decision can use the
# train (inference/shenai_route.py) whether or not retention is on. Nothing
# here touches disk: the retention gate (_clips_enabled) stays the single
# thing between a public URL and persisted physiological data. Bounded to a
# handful of in-flight uploads; an entry lives until its job's part dir is
# discarded.
_SIGNALS: dict = {}
_SIGNALS_LOCK = threading.Lock()
SIGNALS_HOLD_MAX = 8
SHENAI_ROUTE_ON = os.environ.get("AFIB_SHENAI_ROUTE", "1").strip().lower() in (
    "1", "true", "yes", "on")
SHENAI_WAIT_S = float(os.environ.get("AFIB_SHENAI_WAIT_S", "8"))


def _launch_config_overrides() -> dict:
    raw = os.environ.get("AFIB_CONFIG_OVERRIDES", "").strip()
    if not raw:
        return {}
    try:
        d = json.loads(raw)
        if not isinstance(d, dict):
            raise ValueError("not a JSON object")
    except ValueError as e:
        LAUNCH_OVERRIDES["config"] = f"AFIB_CONFIG_OVERRIDES IGNORED ({e})"
        return {}
    for k, v in d.items():
        LAUNCH_OVERRIDES[f"config.{k}"] = str(v)
    return {str(k): v for k, v in d.items()}


LAUNCH_CONFIG_OVERRIDES = _launch_config_overrides()


def _apply_afib_result(doc: dict, det: dict) -> None:
    """ONE AFib result per completed scan (inference/afib_result.py), after
    every interval source has had its say. Best effort, never raises."""
    try:
        from inference.afib_result import afib_result
        r = afib_result(doc, (det or {}).get("config"))
        doc["afib_result"] = r["result"]
        doc["afib_result_basis"] = r["basis"]
    except Exception as e:                                     # noqa: BLE001
        doc["afib_result"] = "INCONCLUSIVE"
        doc["afib_result_basis"] = {"category": "signal",
                                    "why": f"result layer failed safely: {type(e).__name__}: {e}"}


# Live-frame ROI traces (inference/trace_ingest.py, 2026-09-17): held in
# memory exactly like the ShenAI sidecar, never written; dropped with the
# job's part dir. AFIB_TRACE_PATH=0 disables the (recorded-only) trace path.
_TRACES: dict = {}
TRACE_PATH_ON = os.environ.get("AFIB_TRACE_PATH", "1").strip().lower() in (
    "1", "true", "yes", "on")


def _hold_traces(upload_id: str, payload: dict) -> None:
    with _SIGNALS_LOCK:
        _TRACES[upload_id] = payload
        while len(_TRACES) > SIGNALS_HOLD_MAX:
            _TRACES.pop(next(iter(_TRACES)))


def _peek_traces(upload_id: str):
    with _SIGNALS_LOCK:
        return _TRACES.get(upload_id)


def _trace_summary(res, det, cfg) -> dict:
    """The trace path's answer, flattened for debug.trace_path and the sheet."""
    from inference.afib_result import afib_result
    ing = (det or {}).get("ingest")
    out = {"ran": True, "ingest_ok": bool(getattr(ing, "ok", False)),
           "ingest_reasons": list(getattr(ing, "reasons", []) or [])}
    if res is None:
        return out
    ev = (det or {}).get("evidence") or {}
    ra = (det or {}).get("rationale") or {}
    meta = getattr(ing, "meta", None)
    doc = {"outcome": res.outcome.value, "predicted_class": res.predicted_class,
           "afib_probability": res.afib_probability, "confidence_stars": res.confidence_stars,
           "no_read_reasons": list(res.no_read_reasons),
           "debug": {"rationale": ra}, "rhythm_source": "client_traces"}
    r = afib_result(doc, cfg)
    feats = ra.get("features") or {}
    out.update({
        "outcome": res.outcome.value, "predicted_class": res.predicted_class,
        "afib_probability": res.afib_probability, "stars": res.confidence_stars,
        "afib_result": r["result"], "afib_basis": r["basis"].get("why"),
        "no_read_reasons": list(res.no_read_reasons),
        "gates_failed": list(ra.get("gates_failed") or []),
        "sqi": res.signal_quality_index,
        "coherence": ev.get("cross_roi_coherence"),
        "timing_ms": ev.get("timing_precision_ms"),
        "timing_matched": ev.get("timing_matched_fraction"),
        "n_intervals": ev.get("n_intervals"), "n_beats": ev.get("n_beats"),
        "coverage": ra.get("coverage"), "pulse_bpm": res.mean_pulse_rate_bpm,
        "spectral_bpm": ev.get("pulse_spectral_bpm"),
        "mad_ms": feats.get("median_abs_succ_diff"), "pnn50": feats.get("pnn50"),
        "fps": getattr(meta, "measured_fps_mean", None),
        "jitter_ms": getattr(meta, "measured_fps_jitter_ms", None),
        "duration_s": getattr(meta, "duration_s", None),
        "n_frames": getattr(meta, "n_frames", None),
        "tracking_stability": getattr(getattr(ing, "track", None), "stability", None),
    })
    return out


def _apply_trace_path(doc: dict, det: dict, upload_id: str, manifest: dict) -> None:
    """Run THE pipeline on the client's live-frame traces when they arrived,
    and record the answer beside the video path's. Recorded only: nothing
    here changes afib_result (owner decision pending the live comparison)."""
    if not TRACE_PATH_ON:
        doc.setdefault("debug", {})["trace_path"] = {"ran": False, "reason": "disabled (AFIB_TRACE_PATH)"}
        return
    try:
        from inference import shenai_route, trace_ingest
        raw, waited = shenai_route.wait_for(lambda: _peek_traces(upload_id), SHENAI_WAIT_S)
        if not isinstance(raw, dict):
            doc.setdefault("debug", {})["trace_path"] = {
                "ran": False, "reason": "no trace document arrived"
                + (f" (waited {waited:.1f} s)" if waited else "")}
            return
        t0 = time.perf_counter()
        res, tdet = trace_ingest.run_on_traces(raw, manifest=manifest,
                                               config=(det or {}).get("config"),
                                               upload_id=upload_id)
        rec = _trace_summary(res, tdet, (det or {}).get("config"))
        rec["waited_s"] = round(waited, 2)
        rec["analysis_s"] = round(time.perf_counter() - t0, 2)
        rec["sampler"] = raw.get("sampler") if isinstance(raw.get("sampler"), dict) else None
        doc.setdefault("debug", {})["trace_path"] = rec
        print(f"[measure] trace path: ok={rec.get('ingest_ok')} outcome={rec.get('outcome')} "
              f"class={rec.get('predicted_class')} result={rec.get('afib_result')} "
              f"coh={rec.get('coherence')} tp={rec.get('timing_ms')} n={rec.get('n_intervals')} "
              f"pulse={rec.get('pulse_bpm')} in {rec['analysis_s']}s", flush=True)
    except Exception as e:                                     # noqa: BLE001
        doc.setdefault("debug", {})["trace_path"] = {
            "ran": False, "reason": f"trace path failed safely: {type(e).__name__}: {e}"}


# Vascular Tone from the live-frame traces (features/facial_perfusion.py,
# 2026-09-23). The clip-based amplitude CV it replaces measured capture noise
# (a pulse with zero amplitude variation read 16-20 through it; the same scan
# read 45 from the clip and 100 from its traces). AFIB_TONE_SOURCE=clip keeps
# the old card for an A/B; anything else computes the facial perfusion index.
TONE_SOURCE = os.environ.get("AFIB_TONE_SOURCE", "traces").strip().lower()


def _tone_hint_bpm(doc: dict, header: dict):
    """The scan's live-frame heart rate: the SDK's own figure, else the
    request's reference rate. Only a hint - the estimator locks onto the
    traces' own pulse peak within +/- 15 bpm of it."""
    sdk = ((doc.get("debug") or {}).get("shenai_input") or {}).get("sdk_hr_bpm")
    ref = (header or {}).get("reference") if isinstance((header or {}).get("reference"), dict) else {}
    for v in (sdk, (ref or {}).get("ref_hr")):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return None


def _apply_vascular_tone(doc: dict, raw_traces, header: dict) -> None:
    """Replace the Vascular Tone card with the facial perfusion index measured
    on this scan's live-frame traces. Best effort: a scan returns whether or
    not this works, and a failure leaves an explicit reason on the card, never
    the old clip number presented as the new measurement."""
    if TONE_SOURCE == "clip":
        doc.setdefault("debug", {})["vascular_tone"] = {"source": "clip",
                                                        "reason": "AFIB_TONE_SOURCE=clip"}
        return
    try:
        from app.report_data import replace_biomarker
        from features.facial_perfusion import facial_perfusion, vascular_tone_card
        bio = doc.get("biomarkers")
        if not isinstance(bio, dict) or not isinstance(bio.get("items"), list):
            return
        legacy = next((dict(it.get("details") or {}) for it in bio["items"]
                       if isinstance(it, dict) and it.get("key") == "vascular_tone"), None)
        if isinstance(raw_traces, dict):
            fp = facial_perfusion(raw_traces, hr_hint_bpm=_tone_hint_bpm(doc, header))
        else:
            fp = {"available": False, "version": None,
                  "reason": ("the live camera traces for this scan did not reach the "
                             "service, so facial perfusion could not be measured")}
        card = vascular_tone_card(fp, legacy=legacy)
        doc["biomarkers"] = replace_biomarker(bio, "vascular_tone", card)
        doc.setdefault("debug", {})["vascular_tone"] = {
            "source": "live_frame_traces", "version": fp.get("version"),
            "available": bool(fp.get("available")), "pi_percent": fp.get("pi_percent"),
            "score": card.get("score"), "tier": card.get("tier"),
            "ci95_percent": (fp.get("uncertainty") or {}).get("ci95_percent"),
            "regions_used": fp.get("regions_used"), "windows": fp.get("windows"),
            "seconds_used": fp.get("seconds_used"), "f0_source": fp.get("f0_source"),
            "f0_bpm": (round(fp["f0_hz"] * 60.0, 1) if fp.get("f0_hz") else None),
            "reason": fp.get("reason"),
            "legacy_clip_score": (legacy or {}).get("score"),
        }
        print(f"[measure] vascular tone: pi={fp.get('pi_percent')} score={card.get('score')} "
              f"tier={card.get('tier')} regions={fp.get('regions_used')} "
              f"windows={fp.get('windows')} legacy_clip={(legacy or {}).get('score')} "
              f"{fp.get('reason') or ''}", flush=True)
    except Exception as e:                                     # noqa: BLE001
        reason = f"vascular tone failed safely: {type(e).__name__}: {e}"
        doc.setdefault("debug", {})["vascular_tone"] = {"source": "live_frame_traces",
                                                        "reason": reason}
        try:            # never leave the old clip number standing in for the new card
            from app.report_data import replace_biomarker
            from features.facial_perfusion import vascular_tone_card
            doc["biomarkers"] = replace_biomarker(
                doc.get("biomarkers"), "vascular_tone",
                vascular_tone_card({"available": False, "reason": reason}))
        except Exception:                                      # noqa: BLE001
            pass


def _held_traces(upload_id: str):
    """The client's trace document for this job. The trace path has already
    waited for it when it is enabled; otherwise wait here, just as long."""
    raw = _peek_traces(upload_id)
    if raw is None and not TRACE_PATH_ON:
        from inference import shenai_route
        raw, _ = shenai_route.wait_for(lambda: _peek_traces(upload_id), SHENAI_WAIT_S)
    return raw if isinstance(raw, dict) else None


def _hold_signals(upload_id: str, payload: dict) -> None:
    with _SIGNALS_LOCK:
        _SIGNALS[upload_id] = payload
        while len(_SIGNALS) > SIGNALS_HOLD_MAX:
            _SIGNALS.pop(next(iter(_SIGNALS)))


def _peek_signals(upload_id: str):
    with _SIGNALS_LOCK:
        return _SIGNALS.get(upload_id)


def _drop_signals(upload_id: str) -> None:
    with _SIGNALS_LOCK:
        _SIGNALS.pop(upload_id, None)
        _TRACES.pop(upload_id, None)


def _validate_signal_attachment(header: dict, upload_id: str) -> dict:
    """Bounded optional input; its identity must agree with the video job."""
    delivery = header.get("signal_delivery")
    raw = header.get("shenai_signals")
    if delivery is None and raw is None:
        return {}                           # legacy separate-sidecar client
    states = {"attached", "empty_snapshot", "missing_snapshot", "snapshot_too_large",
              "snapshot_serialization_failed"}
    if not isinstance(delivery, dict) or type(delivery.get("version")) is not int \
            or delivery.get("version") != 1 or not isinstance(delivery.get("state"), str) \
            or delivery.get("state") not in states:
        raise ValueError("invalid signal delivery metadata")
    if raw is not None and (not isinstance(raw, dict) or not upload_id
                            or raw.get("upload_id") != upload_id):
        raise ValueError("ShenAI snapshot upload_id does not match the video")
    if (delivery["state"] in {"attached", "empty_snapshot"}) != isinstance(raw, dict):
        raise ValueError("signal delivery state does not match its payload")
    return {"signal_delivery": {"version": 1, "state": delivery["state"]},
            "shenai_signals": raw}


def _signal_input_summary(raw, delivery, transport: str, waited_s: float) -> dict:
    """Receipt is independent of retention. No sample arrays leave the job."""
    received = isinstance(raw, dict)
    out = {"received": received, "transport": transport,
           "state": delivery.get("state") if delivery else ("received" if received else "not_received"),
           "waited_s": round(waited_s, 2)}
    if not received:
        return out
    out.update(_shenai_summary(raw))
    ppg = raw.get("ppg") if isinstance(raw.get("ppg"), dict) else {}
    signal = ppg.get("signal")
    out["ppg_n"] = len(signal) if isinstance(signal, list) else 0
    out["ppg_missing_n"] = sum(not isinstance(x, (int, float)) or isinstance(x, bool)
                               or not math.isfinite(x) for x in signal) if isinstance(signal, list) else 0
    # These are the SDK's reported figures, not an independent validation of
    # beat timing or an AFib probability. Keep missing quality distinct from 0.
    ref = raw.get("reference") if isinstance(raw.get("reference"), dict) else {}
    for dest, src in (("sdk_quality", "average_signal_quality"),
                      ("sdk_bad_signal_s", "bad_signal_seconds"),
                      ("sdk_hr_bpm", "heart_rate_bpm"),
                      ("sdk_lnrmssd", "hrv_lnrmssd_ms")):
        value = ref.get(src)
        out[dest] = (float(value) if isinstance(value, (int, float)) and
                     not isinstance(value, bool) and math.isfinite(value) else None)
    return out


def _apply_shenai_route(doc: dict, det: dict, upload_id: str, part_dir=None,
                        *, attachment=None) -> None:
    """Wait (bounded) for the sidecar, run the route, record the outcome on
    the doc. Best effort: a scan must return whether or not any of this works."""
    from inference import shenai_route

    def _get():
        raw = _peek_signals(upload_id)
        if raw is None and part_dir is not None:
            # Retention on and the memory hold missed (a restart between the
            # POST and the job): the file is on disk, gated as ever.
            try:
                src = pathlib.Path(part_dir) / SHENAI_PART_NAME
                if _clips_enabled() and src.exists():
                    raw = json.loads(src.read_text())
            except Exception:                                  # noqa: BLE001
                raw = None
        return raw if isinstance(raw, dict) else None

    delivery = (attachment or {}).get("signal_delivery")
    raw = (attachment or {}).get("shenai_signals")
    waited = 0.0
    transport = "job_request" if delivery else "sidecar"
    try:
        if not delivery:
            if SHENAI_ROUTE_ON:
                raw, waited = shenai_route.wait_for(_get, SHENAI_WAIT_S)
            else:
                raw = _get()
        try:
            doc.setdefault("debug", {})["shenai_input"] = _signal_input_summary(
                raw, delivery, transport, waited)
        except Exception as summary_error:                  # diagnostics cannot change a decision
            doc.setdefault("debug", {})["shenai_input"] = {
                "received": isinstance(raw, dict), "transport": transport,
                "state": "summary_failed", "error_type": type(summary_error).__name__}
        from app.scan_evidence import record_summary, shenai_evidence_summary
        from app.afib_response import capture_diagnostic_summary
        record_summary(doc, "client_capture_diagnostics", capture_diagnostic_summary, raw)
        record_summary(doc, "shenai_assessment", shenai_evidence_summary, raw, det)
        if not SHENAI_ROUTE_ON:
            rec = {"route": shenai_route.ROUTE_NAME, "attempted": False, "used": False,
                   "reason": "route disabled (AFIB_SHENAI_ROUTE)"}
        else:
            rec = shenai_route.evaluate(doc, det, raw, waited_s=waited)
            # One rate per scan: the fitness card follows the live-frame
            # train's rate when the route is used, and abstains when a sound
            # train contradicts the video path's rate (iteration 32).
            rec["fitness_reconciliation"] = shenai_route.reconcile_fitness_rate(doc, rec)
        doc.setdefault("rhythm_source", "video")
        doc.setdefault("debug", {})["shenai_route"] = rec
        print(f"[measure] shenai route: used={rec.get('used')} "
              f"{rec.get('reason')} -> outcome={doc.get('outcome')} "
              f"class={doc.get('predicted_class')}", flush=True)
    except Exception as e:                                     # noqa: BLE001
        doc.setdefault("debug", {}).setdefault("shenai_input", {
            "received": isinstance(raw, dict), "transport": transport,
            "state": "summary_failed", "error_type": type(e).__name__})
        doc.setdefault("rhythm_source", "video")
        doc.setdefault("debug", {})["shenai_route"] = {
            "used": False, "reason": f"route failed safely: {type(e).__name__}: {e}"}


    # Run after publication selection, on isolated inputs. Observability cannot
    # change the selected source, final decision, retention, or recovery behavior.
    try:
        from app.scan_evidence import record_summary
        from app.rhythm_diagnostics import compare_sources
        record_summary(doc, "rhythm_comparison", compare_sources, doc, det, raw, SHENAI_ROUTE_ON)
    except Exception as error:
        doc.setdefault("debug", {})["rhythm_comparison"] = {
            "version": 1, "mode": "diagnostic_only", "state": "assessment_failed",
            "contributes_to_published_result": False, "error_type": type(error).__name__}


# ------------------------------------------------------------ transport
def _parse_envelope(body: bytes, ctype: str, query: dict) -> tuple:
    """Two accepted shapes.

    `application/x-afib-measure` uses the same length-prefixed envelope the
    live server's /api/frames uses, so per-frame timestamps (~30 KB of JSON
    for a 90 s scan) travel with the clip instead of being crammed into a
    query string:

        [4-byte LE header length][JSON header][video bytes]

    Anything else is treated as a raw video body with metadata in the query
    string — the simple path, and the one the webapp uses by default."""
    import struct

    if ctype.startswith("application/x-afib-measure"):
        if len(body) < 4:
            raise ValueError("envelope shorter than its length prefix")
        (hlen,) = struct.unpack("<I", body[:4])
        if hlen <= 0 or hlen > MAX_SIDECAR_BYTES or 4 + hlen > len(body):
            raise ValueError(f"bad envelope header length {hlen}")
        header = json.loads(body[4:4 + hlen].decode())
        if not isinstance(header, dict):
            raise ValueError("envelope header is not a JSON object")
        # The browser keeps video metadata in the query and attaches the
        # signal in the envelope. Preserve timestamps/metadata from older
        # envelope clients while retaining the query's scan identity.
        _, query_header = _parse_envelope(b"", "video/webm", query)
        if query_header.get("upload_id") and header.get("upload_id") not in (
                None, query_header["upload_id"]):
            raise ValueError("envelope upload_id does not match query")
        header = {**query_header, **header}
        header.update(_validate_signal_attachment(header, header.get("upload_id")))
        return body[4 + hlen:], header

    def _one(k):
        v = query.get(k, [None])[0]
        return v

    header = {}
    if _one("duration_ms"):
        header["duration_ms"] = float(_one("duration_ms"))
    if _one("lux"):
        header["illuminance_lux"] = float(_one("lux"))
    if _one("session"):
        header["session"] = _one("session")
    if _one("capture_profile"):
        header["capture_profile"] = _one("capture_profile")
    # Reference vitals from the client's OTHER measurement (ShenAI) for the
    # tracking sheet's agreement columns. Never used by the pipeline; echoed
    # back under doc["reference"] so a row can be audited. Non-numeric -> ignored.
    ref = {}
    for k in ("ref_hr", "ref_hrv", "ref_sbp", "ref_dbp"):
        v = _one(k)
        if v not in (None, ""):
            try:
                ref[k] = float(v)
            except ValueError:
                pass
    if _one("ref_source"):
        ref["ref_source"] = str(_one("ref_source"))[:40]
    if ref:
        header["reference"] = ref
    # Capture state from the client (step 1 of the repeatability plan). The lock
    # flags go into the manifest the pipeline already understands, so the
    # capture record and the vascular-tone "optics locked" caveat are truthful;
    # fps and face brightness are echoed for the tracking sheet only.
    cap = {}
    for k in ("exposure_locked", "awb_locked"):
        v = _one(k)
        if v in ("0", "1", "true", "false"):
            cap[k] = v in ("1", "true")
    for k in ("client_fps", "face_luma"):
        v = _one(k)
        if v not in (None, ""):
            try:
                cap[k] = float(v)
            except ValueError:
                pass
    # The client's own account of the lock (gain steps, brightness before/after,
    # mode read back, or why it was skipped) — free text for the sheet only.
    v = _one("capture_note")
    if isinstance(v, str) and v.strip():
        cap["note"] = v.strip()[:300]
    # A per-attempt id, used ONLY as the key a dropped client collects its
    # result under. The session id cannot serve: the webapp sends a constant
    # one in development, so keying on it could hand back a previous scan.
    v = _one("upload_id")
    if isinstance(v, str) and v.strip():
        header["upload_id"] = v.strip()[:80]
    if cap:
        header["client_capture"] = cap
        m = header.setdefault("manifest", {}) if isinstance(header.get("manifest"), (dict, type(None))) else {}
        if isinstance(m, dict):
            for k in ("exposure_locked", "awb_locked"):
                if k in cap:
                    m[k] = cap[k]
            header["manifest"] = m
    if _one("window_s"):
        header["window_s"] = float(_one("window_s"))
    if _one("scale") is not None:
        header["scale"] = _one("scale")            # "WxH" or "0" to disable
    if _one("ext"):
        # Safari/iOS records MP4, Chrome/Android WebM; the temp file needs the
        # matching extension or the decoder can fail to sniff the container.
        header["ext"] = "".join(c for c in _one("ext") if c.isalnum())[:8]
    # Opt-in threshold experiments from the simple path, e.g. ?fps_min=25.
    ov = {}
    if _one("fps_min"):
        ov["capture.fps_min"] = float(_one("fps_min"))
    if _one("bpp_floor"):
        ov["capture.bpp_floor"] = float(_one("bpp_floor"))
    if ov:
        header["config_overrides"] = ov
    return body, header


def _finite(o):
    """Replace non-finite floats with None, recursively. Only ever reached
    from _json_bytes's fallback path, so the walk costs nothing on a normal
    scan."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _json_bytes(doc: dict) -> bytes:
    """Serialise a doc for the wire without ever emitting a bare Infinity.

    json.dumps defaults to allow_nan=True, which writes the BARE tokens
    Infinity/-Infinity/NaN. Those are not JSON. Traced end to end on
    2026-09-11 from a sidecar body carrying fs_hz=inf: the phone's
    `await res.json()` throws, its bare catch swallows the error, and all 30
    result polls re-fetch the same poisoned doc - the scan is lost after
    minutes of polling; on the single-shot path JSON.parse(text) throws at
    once and the card reads "non-JSON response (200)".

    _shenai_summary's _f now rejects non-finite values at the source. This is
    the boundary belt-and-braces, so no future additive doc key can reach a
    browser the same way. A scan must still get a response if it fires, hence
    the scrub-and-retry rather than letting the handler raise.
    """
    try:
        return json.dumps(doc, default=str, allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError) as e:
        print(f"[measure] doc held a non-JSON value ({e}); scrubbing",
              flush=True)
        try:
            return json.dumps(_finite(doc), default=str,
                              allow_nan=False).encode()
        except Exception as e2:                               # noqa: BLE001
            print(f"[measure] result doc not serialisable: {e2}", flush=True)
            return json.dumps({"error": "result could not be serialised"}
                              ).encode()


class MeasureHandler(BaseHTTPRequestHandler):
    server_version = "AvatarXMeasure/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[measure] {self.address_string()} {fmt % args}", flush=True)
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, X-Afib-Session")
        self.send_header("Access-Control-Max-Age", "86400")

    def _json(self, code: int, doc: dict):
        body = _json_bytes(doc)
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # The client gave up (its own timeout, a locked phone, a dropped
            # link) while the job was still running. The work is done and
            # thrown away; say so in one line instead of a traceback.
            print(f"[measure] client gone before response could be sent "
                  f"(would have been HTTP {code})", flush=True)
            self.close_connection = True

    def _json_streaming(self, fut, *, heartbeat_s: float = 5.0):
        """Answer a long job without letting the connection go quiet.

        The response starts NOW, chunked, and a single space is emitted every
        `heartbeat_s` until the job's JSON is ready. Leading whitespace is
        valid JSON, so every client still parses the body unchanged — but the
        socket never sits idle for the 20-45 s the analysis takes, which is
        what killed a real phone scan on 2026-09-09 (ERR_HTTP2_PING_FAILED).

        The status line goes out before the job finishes, so a job that raises
        cannot become an HTTP 500 here; it becomes a 200 whose body is
        {"error": ...}. Clients already treat a body-level `error` as a
        failure, so this is the same outcome by a different route.

        Returns (doc, delivered): `doc` is always the job's result — the
        caller still records it even when `delivered` is False because the
        client vanished mid-heartbeat.
        """
        gone = False

        def _write(data: bytes) -> bool:
            try:
                self.wfile.write(data)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError,
                    ConnectionAbortedError, OSError):
                return False

        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError,
                ConnectionAbortedError, OSError):
            gone = True

        beats = 0
        while True:
            try:
                doc = fut.result(timeout=heartbeat_s)
                break
            except _FutureTimeout:
                if gone:
                    continue                      # finish the job regardless
                beats += 1
                if not _write(b"1\r\n \r\n"):
                    gone = True
                    print("[measure] client gone mid-analysis; finishing the "
                          "job so the result can be collected later",
                          flush=True)
            except Exception as e:                # noqa: BLE001
                traceback.print_exc()
                doc = {"error": f"{type(e).__name__}: {e}"}
                break

        if gone:
            self.close_connection = True
            return doc, False
        body = _json_bytes(doc)
        ok = _write(f"{len(body):X}\r\n".encode() + body + b"\r\n")
        ok = _write(b"0\r\n\r\n") and ok
        if not ok:
            self.close_connection = True
            print("[measure] client gone before the result could be written",
                  flush=True)
        elif beats:
            print(f"[measure] kept the connection warm with {beats} "
                  f"heartbeat(s)", flush=True)
        return doc, ok

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if urlparse(self.path).path in ("/api/traces", "/api/trace"):
            self._serve_traces()
            return
        if urlparse(self.path).path in ("/healthz", "/api/healthz"):
            _RESULT_STORE.purge()
            self._json(200, {"ok": True, "service": "afib-measure",
                             "build": BUILD_SHA,
                             "uptime_s": int(time.time() - _STARTED_AT),
                             "inflight_jobs": _inflight(0),
                             "result_recovery": _RESULT_STORE.status(),
                             "launch_overrides": LAUNCH_OVERRIDES or None,
                             "sheet": result_sheet.status(),
                             "max_concurrent": MAX_CONCURRENT,
                             "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)})
            return
        if urlparse(self.path).path in ("/api/result", "/api/results"):
            # Collect a result whose connection died. Additive: the normal
            # path is unchanged and nothing depends on this being called.
            q = parse_qs(urlparse(self.path).query)
            sid = (q.get("upload_id") or q.get("session") or [""])[0]
            doc = _recall_result(sid)
            if doc is None:
                with _STARTED_LOCK:
                    running = sid in _STARTED
                if running:
                    # Distinct from "unknown": the client should keep waiting.
                    self._json(202, {"status": "processing", "upload_id": sid})
                    return
                self._json(404, {"error": "no completed result for that "
                                          "session", "session": sid,
                                 "ttl_s": RESULT_TTL_S})
                return
            self._json(200, doc)
            return
        if urlparse(self.path).path in ("/api/clips", "/api/clip"):
            self._serve_clip()
            return
        self._json(404, {"error": "not found"})

    def _serve_traces(self):
        """List (/api/traces) or download (/api/trace?id=) a retained trace
        document. Same rules as _serve_clip: 404 unless retention is on AND
        the token matches (compared as bytes, in constant time), basename-only
        ids, nothing outside TRACES_DIR."""
        u = urlparse(self.path)
        q = parse_qs(u.query)
        token = (q.get("token") or [""])[0]
        if not _traces_enabled() or not hmac.compare_digest(
                token.encode("utf-8", "ignore"), _clips_token().encode("utf-8", "ignore")):
            self._json(404, {"error": "not found"})
            return
        try:
            TRACES_DIR.mkdir(parents=True, exist_ok=True)
            files = sorted(TRACES_DIR.glob("*" + TRACE_FILE_SUFFIX),
                           key=lambda f: f.stat().st_mtime, reverse=True)
        except OSError as e:
            self._json(500, {"error": f"traces unavailable: {e}"})
            return
        if u.path == "/api/traces":
            self._json(200, {"traces": [
                {"id": f.name, "bytes": f.stat().st_size,
                 "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_mtime))}
                for f in files], "keep": TRACES_KEEP})
            return
        tid = os.path.basename((q.get("id") or [""])[0])
        target = TRACES_DIR / tid
        if not tid.endswith(TRACE_FILE_SUFFIX) or not target.is_file():
            self._json(404, {"error": "no such trace", "id": tid})
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
        self.end_headers()
        self.wfile.write(data)

    def _serve_clip(self):
        """List or download a retained scan clip.

        Gated on AFIB_KEEP_UPLOADS=1 AND a matching AFIB_CLIPS_TOKEN, because
        this serves video of someone's face from a public URL. Disabled looks
        like 404, not 403: an endpoint that is off should not advertise that it
        exists. Never enabled on production.
        """
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not _clips_enabled():
            self._json(404, {"error": "not found"})
            return
        token = (q.get("token") or [""])[0]
        # Compare BYTES. compare_digest refuses two str arguments unless both
        # are pure ASCII ("comparing strings with non-ASCII characters is not
        # supported") and the query string is attacker-chosen, so
        # ?token=%C3%A9 raised TypeError here, killed the request thread and
        # returned zero bytes. Measured 2026-09-11: clips off -> HTTP 404,
        # clips on -> connection closed. One anonymous probe therefore told
        # anybody whether this public URL currently holds face video and PPG
        # waveforms - the exact opposite of the design stated above.
        if not hmac.compare_digest(token.encode("utf-8", "ignore"),
                                   _clips_token().encode("utf-8", "ignore")):
            print("[clips] rejected: bad or missing token", flush=True)
            self._json(404, {"error": "not found"})
            return
        try:
            CLIPS_DIR.mkdir(parents=True, exist_ok=True)
            vids = sorted((f for f in CLIPS_DIR.glob("*")
                           if not f.name.endswith(CLIP_SIDECAR_SUFFIXES)),
                          key=lambda f: f.stat().st_mtime, reverse=True)
        except OSError as e:
            self._json(500, {"error": f"clips unavailable: {e}"})
            return

        if u.path == "/api/clips":
            self._json(200, {"clips": [
                {"id": f.name,
                 "bytes": f.stat().st_size,
                 "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(f.stat().st_mtime)),
                 "sidecar": os.path.exists(str(f) + ".timestamps.json"),
                 # Sidecars get no rows of their own (filtered out above);
                 # this is how scripts/pull_scan_clips.py knows to ask for
                 # "<id>.shenai.json" through the same gated /api/clip.
                 "shenai": os.path.exists(str(f) + ".shenai.json")}
                for f in vids], "keep": CLIPS_KEEP})
            return

        cid = (q.get("id") or [""])[0]
        # Basename only: a caller must not be able to walk out of CLIPS_DIR.
        target = CLIPS_DIR / os.path.basename(cid)
        if not cid or not target.exists() or not target.is_file():
            self._json(404, {"error": "no such clip", "id": cid})
            return
        try:
            data = target.read_bytes()
        except OSError as e:
            self._json(500, {"error": f"unreadable: {e}"})
            return
        print(f"[clips] serving {target.name} ({len(data) / 1e6:.1f} MB)",
              flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{target.name}"')
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self, limit: int):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return None
        if length <= 0:
            self._json(400, {"error": "empty body"})
            return None
        if length > limit:
            self._json(413, {"error": f"body exceeds "
                                      f"{limit // (1024 * 1024)} MB"})
            return None
        return self.rfile.read(length)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/upload-part":
            self._upload_part(parse_qs(u.query))
            return
        if u.path == "/api/start":
            self._start_job(parse_qs(u.query))
            return
        if u.path == "/api/scan-signals":
            self._upload_signals(parse_qs(u.query))
            return
        if u.path == "/api/scan-traces":
            self._upload_traces(parse_qs(u.query))
            return
        if u.path == "/api/measure-traces":
            self._measure_traces_request(parse_qs(u.query))
            return
        if u.path not in ("/api/process-video", "/api/measure"):
            self._json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return
        if length <= 0:
            self._json(400, {"error": "empty body"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._json(413, {"error": f"upload exceeds "
                                      f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB"})
            return

        # Real mobile scan timed out client-side at 240s with no visibility
        # into where server time went — the client aborts before any
        # response reaches it, so `trim`/`downscale` never get returned.
        # This is the fix: stage timing that lands in the platform's log
        # stream (Railway, etc.) as it happens, not just in a response body
        # that a timed-out request never receives.
        t_upload0 = time.perf_counter()
        body = self.rfile.read(length)
        upload_s = time.perf_counter() - t_upload0
        got = len(body)
        print(f"[measure] upload received: {got / 1e6:.1f} MB of "
              f"{length / 1e6:.1f} MB declared, in {upload_s:.1f}s "
              f"({got / 1e6 / max(upload_s, 0.001):.2f} MB/s)", flush=True)
        # rfile.read() returns SHORT at EOF — it does not raise. Measured:
        # a client that aborted after sending 6.3 MB of a declared 37 MB
        # body had the 6.3 MB analysed as a whole clip, through trim,
        # downscale and the full pipeline, to a verdict with NaN rhythm
        # features. A phone upload cut by a network blip would do the
        # same. Refuse an incomplete body before any work is done — and
        # before the worker slot is taken.
        if got != length:
            print(f"[measure] REJECTED incomplete upload: {got} of {length} "
                  f"bytes — client disconnected mid-body", flush=True)
            self._json(400, {"error": "incomplete upload",
                             "received_bytes": got,
                             "declared_bytes": length})
            # The peer is gone. With HTTP/1.1 keep-alive the handler would
            # otherwise loop to read the NEXT request line on this socket
            # and raise ConnectionResetError from the stdlib — outside any
            # handler code. Ending the connection here means there is no
            # next read to fail.
            self.close_connection = True
            return
        try:
            video_bytes, header = _parse_envelope(
                body, (self.headers.get("Content-Type") or "").lower(),
                parse_qs(u.query))
        except Exception as e:
            self._json(400, {"error": f"bad request envelope: {e}"})
            return
        # How long THIS process spent receiving the body. Behind a platform
        # edge proxy that is the proxy->app relay, not the client's uplink:
        # measured, a 37 MB body a fast client finished sending in <1 s took
        # ~55 s to arrive here while the job itself took 14 s. Without this
        # number in the response, that gap is invisible from outside.
        header["_upload_received_s"] = round(upload_s, 2)
        header["_upload_bytes"] = got
        if not video_bytes:
            self._json(400, {"error": "no video bytes in request"})
            return

        # Bound concurrency: each measure holds a full decode + pipeline run.
        if not _INFLIGHT.acquire(blocking=False):
            self._json(503, {"error": "measure workers busy — retry",
                             "max_concurrent": MAX_CONCURRENT})
            return
        n = _inflight(+1)
        print(f"[measure] job start (inflight now {n}/{MAX_CONCURRENT})",
              flush=True)
        t_job = time.perf_counter()
        try:
            fut = _POOL.submit(self._run, video_bytes, header)
            # Streamed: the connection stays warm while the analysis runs, and
            # the job is finished even if the client disappears mid-way.
            doc, delivered = self._json_streaming(fut)
            # Hold the finished result briefly so a client whose connection
            # died can collect it from GET /api/result?session=... instead of
            # losing a completed scan.
            _remember_result(header.get("upload_id") or header.get("session"),
                             doc)
            if not delivered:
                print(f"[measure] result kept as "
                      f"{(header.get('upload_id') or header.get('session'))!r} "
                      f"for {RESULT_TTL_S:.0f}s — client may collect it",
                      flush=True)
            # AFTER the response is on the wire: queue the tracking-sheet row.
            # Runs on its own thread, never delays or fails the scan result.
            result_sheet.schedule_append(doc, extra={
                "build": BUILD_SHA,
                "duration_ms": header.get("duration_ms"),
                "user_agent": self.headers.get("User-Agent", "")})
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": str(e)})
        finally:
            n = _inflight(-1)
            print(f"[measure] job end after {time.perf_counter() - t_job:.1f}s "
                  f"(inflight now {n}/{MAX_CONCURRENT})", flush=True)
            _INFLIGHT.release()

    def _upload_part(self, q: dict):
        """One slice of a clip. Small and independently retryable: a phone
        that loses its link re-sends this part only, not 23 MB."""
        one = lambda k: (q.get(k) or [None])[0]                  # noqa: E731
        try:
            upload_id = str(one("upload_id") or "")
            index = int(one("index"))
            total = int(one("total"))
            d = _part_dir(upload_id)
        except (TypeError, ValueError) as e:
            self._json(400, {"error": f"bad part parameters: {e}"})
            return
        if not (0 <= index < total <= MAX_UPLOAD_PARTS):
            self._json(400, {"error": "part index/total out of range"})
            return
        body = self._read_body(MAX_UPLOAD_BYTES)
        if body is None:
            return
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / "total").write_text(str(total))
            # Write beside, then rename: a part is either wholly there or not
            # there at all, so a retry of a half-written part is safe.
            tmp = d / f".part-{index:04d}.tmp"
            tmp.write_bytes(body)
            tmp.replace(d / f"part-{index:04d}")
        except OSError as e:
            self._json(500, {"error": f"could not store part: {e}"})
            return
        have = len(list(d.glob("part-*")))
        print(f"[measure] part {index + 1}/{total} ({len(body)} B) for "
              f"{upload_id!r}; {have}/{total} held", flush=True)
        self._json(200, {"ok": True, "upload_id": upload_id, "index": index,
                         "received": have, "total": total})

    def _upload_signals(self, q: dict):
        """ShenAI's own dense PPG waveform and beat train for one scan.

        WHY store a second opinion at all (measured 2026-09-10): our four face
        ROIs disagree on the pulse by 15-23 bpm at the phone's compression
        level and cross-region waveform correlation is 0.16, which shatters the
        interval series, holds coverage at 0.47 against the 0.50 floor and
        leaves ACCEPT on 1 scan in 49. ShenAI builds a 3D face model and
        extracts ONE dense signal, so its waveform and beat train are an
        independent view of the SAME beats, captured by the same camera under
        the same illuminant. Retained for OFFLINE comparison only: nothing
        written here is ever read by the pipeline, and this route can neither
        start, delay nor fail a scan.

        THE WRITE IS UNAUTHENTICATED, deliberately. It arrives from an
        anonymous phone mid-scan exactly like /api/upload-part, and
        /beta/cardio-staging has no login - so demanding AFIB_CLIPS_TOKEN here
        would mean shipping that token inside a public JS bundle, weakening the
        READ gate it exists to protect. The write is made harmless instead: no
        semaphore (MAX_CONCURRENT is 2 and a JSON POST must never be able to
        503 a real scan), no decoding, a 4 MB cap, and nothing is persisted
        unless _clips_enabled(). Production sets neither switch, so there the
        waveform is accepted, counted and dropped without ever touching disk.
        Read-back needs no new endpoint and no new gate: the retained copy
        lives in CLIPS_DIR and is served by the existing GET /api/clip behind
        _clips_enabled() + hmac.compare_digest + basename-only ids.
        """
        # Every OTHER writer's flow ends at /api/start, which sweeps. This
        # route is a dead end that reaches disk, so it sweeps for itself.
        _sweep_parts()
        one = lambda k: (q.get(k) or [None])[0]                  # noqa: E731
        try:
            upload_id = str(one("upload_id") or "")
            d = _part_dir(upload_id)          # the ONLY sanitizer on this path
        except (TypeError, ValueError) as e:
            self._json(400, {"error": f"bad signal parameters: {e}"})
            # protocol_version is HTTP/1.1 and nothing has drained the request
            # body, so the unread bytes would desync this keep-alive socket for
            # the next request - same reason as the single-shot path above.
            self.close_connection = True
            return
        body = self._read_body(MAX_SIDECAR_BYTES)
        if body is None:
            self.close_connection = True      # _read_body answered 400/413
            return
        try:
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body is not a JSON object")
        except (UnicodeDecodeError, ValueError, RecursionError) as e:
            self._json(400, {"error": f"bad signal parameters: {e}"})
            self.close_connection = True
            return
        # ShenAI route (2026-09-16): hold the document IN MEMORY for the job
        # running on this upload - no directory is created, nothing is
        # written, and the hold dies with the job's part dir. Only an upload
        # that is actually in progress is held; a flood of anonymous POSTs is
        # bounded by SIGNALS_HOLD_MAX entries of at most MAX_SIDECAR_BYTES.
        held = False
        if d.is_dir():
            _hold_signals(d.name, payload)
            held = True
        # NEVER 404 of our own accord. The client treats 404 from an upload
        # route as "this service predates the route" and stops; 404 here stays
        # reserved for a service that genuinely has no such path (the catch-all
        # below). Retention off means accepted-and-dropped, not refused.
        if not _clips_enabled():
            print(f"[signals] dropped {len(body)} B for {upload_id!r}: "
                  f"retention OFF: {_clips_gate_reason()}"
                  f"{' (held for the route)' if held else ''}", flush=True)
            self._json(200, {"ok": True, "upload_id": d.name,
                             "bytes": len(body), "stored": False, "held": held})
            return
        # This route never CREATES a part dir, only writes into one an upload
        # already opened. It is unauthenticated: measured 2026-09-11 against a
        # live server with clips on, 40 anonymous POSTs of 4,194,136 B each all
        # answered {"stored":true} and left 167.8 MB across 40 directories that
        # nothing reclaimed. UPLOAD_DIR and CLIPS_DIR share one small ephemeral
        # Railway disk, and filling it fails real scans at _upload_part's write
        # (OSError -> 500). A sidecar for an upload that never sent a byte
        # could not be paired anyway - _pair_shenai needs the retained pointer
        # that only _assemble_parts writes - so nothing is lost by refusing it.
        # Still a 200: the client must see success and never retry.
        if not d.is_dir():
            print(f"[signals] dropped {len(body)} B for {upload_id!r}: "
                  f"no upload in progress", flush=True)
            self._json(200, {"ok": True, "upload_id": d.name,
                             "bytes": len(body), "stored": False})
            return
        try:
            # Write beside, then rename, like _upload_part: the file is either
            # wholly there or not there at all, so a retry is safe and
            # _retain_clip can never copy half a document.
            tmp = d / f".{SHENAI_PART_NAME}.tmp"
            tmp.write_bytes(body)
            tmp.replace(d / SHENAI_PART_NAME)
        except OSError as e:
            self._json(500, {"error": f"could not store signals: {e}"})
            self.close_connection = True
            return
        # The document is already on disk and _pair_shenai summarises it again
        # at pairing time, so a summary that raises must cost nothing here. It
        # is only a log line, and the handler owes this client one of its
        # documented 200/400/413/500 answers.
        try:
            s = _shenai_summary(payload)
        except Exception as e:                                # noqa: BLE001
            print(f"[signals] stored {len(body)} B for {upload_id!r}: "
                  f"summary unavailable ({type(e).__name__}: {e})", flush=True)
        else:
            # The observed getFullPpgSignal() length is unrecorded anywhere in
            # either repo as of 2026-09-11, and both the client's 200_000-sample
            # cap and AFIB_MAX_SIDECAR_MB were sized from a plausible range, not
            # a measurement. This log line is how that gets settled.
            print(f"[signals] stored {len(body)} B for {upload_id!r}: "
                  f"ppg_n={s['ppg_n']} fs={s['ppg_fs_hz']}"
                  f"({s['ppg_fs_source']}) beats={s['beats_n']}", flush=True)
        self._json(200, {"ok": True, "upload_id": d.name,
                         "bytes": len(body), "stored": True})

    def _measure_traces_request(self, q: dict) -> None:
        """POST /api/measure-traces: run the pipeline on a standalone trace
        document (no clip) and return the full result synchronously. The body
        is the trace document (schema_version, t_s, traces, bbox, ...); an
        optional `manifest` and `config_overrides` may ride in it. Takes a
        worker slot so it cannot overrun the box, and writes a sheet row like
        every other scan."""
        body = self._read_body(MAX_SIDECAR_BYTES)
        if body is None:
            self.close_connection = True                 # _read_body answered 400/413
            return
        try:
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body is not a JSON object")
        except (UnicodeDecodeError, ValueError, RecursionError) as e:
            self._json(400, {"error": f"bad trace document: {e}"})
            self.close_connection = True
            return
        if not _INFLIGHT.acquire(blocking=False):
            self._json(503, {"error": "measure workers busy — retry",
                             "max_concurrent": MAX_CONCURRENT})
            return
        ua = self.headers.get("User-Agent", "")
        n = _inflight(+1)
        t0 = time.perf_counter()
        print(f"[measure] traces request start "
              f"({len(payload.get('t_s') or []) if isinstance(payload.get('t_s'), list) else 0} frames, "
              f"inflight {n}/{MAX_CONCURRENT})", flush=True)
        try:
            doc, _ = measure_traces(
                payload,
                manifest=MeasureHandler._build_manifest(payload),
                config_overrides=payload.get("config_overrides"))
            doc["session"] = payload.get("session") or ""
            if payload.get("reference"):
                doc["reference"] = payload["reference"]
            try:
                result_sheet.schedule_append(doc, extra={"build": BUILD_SHA, "user_agent": ua})
            except Exception as e:                       # noqa: BLE001
                print(f"[measure] traces sheet append failed: {e}", flush=True)
            self._json(200, doc)
        except Exception as e:                           # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            _inflight(-1)
            _INFLIGHT.release()
            print(f"[measure] traces request end after {time.perf_counter() - t0:.1f}s", flush=True)

    def _upload_traces(self, q: dict) -> None:
        """POST /api/scan-traces?upload_id=...: the client's live-frame ROI
        traces (inference/trace_ingest.py). Held in memory for the job on
        that upload, never written to disk; accepted-and-dropped when no
        upload is in progress. Always 200 on a well-formed document, so a
        client can never mistake this for a missing route."""
        one = lambda k: (q.get(k) or [None])[0]                  # noqa: E731
        try:
            upload_id = str(one("upload_id") or "")
            d = _part_dir(upload_id)
        except (TypeError, ValueError) as e:
            self._json(400, {"error": f"bad trace parameters: {e}"})
            self.close_connection = True
            return
        body = self._read_body(MAX_SIDECAR_BYTES)
        if body is None:
            self.close_connection = True
            return
        try:
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body is not a JSON object")
        except (UnicodeDecodeError, ValueError, RecursionError) as e:
            self._json(400, {"error": f"bad trace document: {e}"})
            self.close_connection = True
            return
        held = False
        if d.is_dir():
            _hold_traces(d.name, payload)
            held = True
        n = len(payload.get("t_s") or []) if isinstance(payload.get("t_s"), list) else 0
        print(f"[traces] {'held' if held else 'dropped'} {len(body)} B ({n} frames) for "
              f"{upload_id!r}", flush=True)
        self._json(200, {"ok": True, "upload_id": d.name, "bytes": len(body),
                         "frames": n, "held": held})

    def _read_small_json(self, limit: int):
        """The request body as a JSON object, or None. Never raises."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0:
            return None
        if length > limit:
            # Too large to be a profile: not read, so the socket cannot be reused.
            self.close_connection = True
            return None
        try:
            raw = self.rfile.read(length)
            doc = json.loads(raw.decode("utf-8")) if len(raw) == length else None
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        return doc if isinstance(doc, dict) else None

    def _start_job(self, q: dict):
        """Assemble the slices and detach the analysis. Returns immediately:
        the client polls GET /api/result, so no connection is held through the
        job and a dropped link costs nothing."""
        # Drain the bounded body even on a cached/503 response. Leaving it on
        # an HTTP/1.1 connection would corrupt the next request's framing.
        attachment = {}
        if self.headers.get("Content-Length", "0") != "0":
            body = self._read_body(MAX_SIDECAR_BYTES)
            if body is None:
                self.close_connection = True
                return
            if len(body) != int(self.headers["Content-Length"]):
                self._json(400, {"error": "incomplete start input"})
                self.close_connection = True
                return
            try:
                attachment = json.loads(body)
                if not isinstance(attachment, dict):
                    raise ValueError("start body is not a JSON object")
            except (ValueError, RecursionError):
                # Legacy staging profiles were optional: malformed profile-only
                # bodies do not discard an uploaded video. Valid signal metadata
                # still passes the strict identity/state validator below.
                attachment = {}
        _sweep_parts()
        upload_id = str((q.get("upload_id") or [""])[0])
        try:
            _, header = _parse_envelope(b"", "video/webm", q)
            header.update(_validate_signal_attachment(attachment, upload_id))
            part = _participant_block(attachment)
            if part:
                header["participant"] = part
            ext = str(header.get("ext") or "webm").lstrip(".")
            if _part_dir(upload_id).name != upload_id or not upload_id.isascii():
                raise ValueError("upload_id must be an ASCII identifier of at most 80 characters")
            if not ext.isascii() or not ext.isalnum() or len(ext) > 10:
                raise ValueError("invalid video extension")
        except ValueError as e:
            self._json(400, {"error": str(e)})
            return
        poll = f"/api/result?upload_id={upload_id}"
        # Check and reserve atomically. A lost 202 or simultaneous retry must
        # find the accepted job BEFORE testing capacity or consuming parts.
        with _STARTED_LOCK:
            completed = _recall_result(upload_id)
            if completed is not None:
                reply = (200, {"status": "complete", "upload_id": upload_id,
                               "scan_id": upload_id, "poll": poll})
            elif upload_id in _STARTED:
                reply = (202, {"status": "processing", "upload_id": upload_id,
                               "scan_id": upload_id, "poll": poll})
            elif not _INFLIGHT.acquire(blocking=False):
                reply = (503, {"error": "measure workers busy — retry",
                               "max_concurrent": MAX_CONCURRENT})
            else:
                _STARTED[upload_id] = True
                reply = None
        if reply is not None:
            self._json(*reply)
            return
        # Take the worker slot BEFORE assembling: _assemble_parts deletes the
        # slices, so a 503 issued after it left the client's retry with "no
        # parts" -> 400 and the scan was lost (audit 2026-09-17, #6a). Now a
        # busy service answers 503 with the slices intact, and the retry works.
        try:
            path, upload_s = _assemble_parts(upload_id, ext)
            size = os.path.getsize(path)
        except (ValueError, OSError) as e:
            with _STARTED_LOCK:
                _STARTED.pop(upload_id, None)
            _INFLIGHT.release()
            self._json(400, {"error": str(e), "upload_id": upload_id})
            return
        header["upload_id"] = upload_id
        header.setdefault("session", (q.get("session") or [upload_id])[0])
        ua = self.headers.get("User-Agent", "")

        def job():
            n = _inflight(+1)
            t0 = time.perf_counter()
            print(f"[measure] detached job start for {upload_id!r} "
                  f"({size} B, inflight {n}/{MAX_CONCURRENT})", flush=True)
            try:
                doc = self._run_assembled(path, header)
                if not isinstance(doc, dict) or (not doc.get("error") and
                        doc.get("outcome") not in {"ACCEPT", "REPEAT_SCAN", "NO_RESULT"}):
                    raise TypeError("analysis did not return a valid result document")
            except Exception as e:                        # noqa: BLE001
                traceback.print_exc()
                doc = {"error": f"{type(e).__name__}: {e}",
                       "error_code": "analysis_failed"}
            try:
                doc["size_bytes"] = size
                # Keep the upload columns meaningful on this path too.
                t = doc.setdefault("timing", {}) if isinstance(doc.get("timing"), (dict, type(None))) else {}
                if isinstance(t, dict):
                    t.setdefault("upload_bytes", size)
                    t.setdefault("upload_received_s", round(upload_s, 2))
                    doc["timing"] = t
                _remember_result(upload_id, doc)
                result_sheet.schedule_append(doc, extra={
                    "build": BUILD_SHA,
                    "duration_ms": header.get("duration_ms"),
                    "user_agent": ua})
            finally:
                with _STARTED_LOCK:
                    _STARTED.pop(upload_id, None)
                _inflight(-1)
                _INFLIGHT.release()
                # Audit 2026-09-17 #6b: this path left scan.<ext>, the ~220 MB
                # FFV1 intermediate and the sidecars in the part dir until the
                # hourly sweep, on the small disk CLIPS_DIR shares. The retained
                # copy (when retention is on) was taken at /api/start and
                # _pair_shenai has already read shenai.json inside
                # _run_assembled, so nothing here is still needed.
                _discard_part_dir(upload_id)
                print(f"[measure] detached job end for {upload_id!r} after "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)

        try:
            _POOL.submit(job)
        except Exception as e:                            # noqa: BLE001
            doc = {"error": f"Could not schedule analysis: {type(e).__name__}",
                   "error_code": "job_submission_failed"}
            _remember_result(upload_id, doc)
            with _STARTED_LOCK:
                _STARTED.pop(upload_id, None)
            _INFLIGHT.release()
            _discard_part_dir(upload_id)
            self._json(503, doc)
            return
        # Tell the client roughly when to bother asking. Measured on Railway:
        # ~1.6 s of work per MB (downscale + analysis), floor 12 s. Without a
        # hint the client polls blindly from t=0 and burns a dozen requests
        # before the answer can possibly exist.
        eta = max(12.0, round(size / (1024 * 1024) * 1.6, 1))
        self._json(202, {"status": "processing", "upload_id": upload_id,
                         "scan_id": upload_id,
                         "size_bytes": size, "eta_s": eta,
                         "poll": f"/api/result?upload_id={upload_id}"})

    @staticmethod
    def _build_manifest(header: dict) -> dict:
        """The manifest EVERY upload path must send.

        A browser scan IS a consumer capture: the camera will not lock AE/AWB
        and real webm lands at 24-30 fps, not the research profile's 30+.
        capture/ingest.py exists for exactly this and surfaces caveats instead
        of silently accepting research-grade claims. Callers may still ask for
        "research" explicitly.

        Shared because it was not: the detached path built its own and omitted
        the profile, so the pipeline defaulted to "research" and the same clip
        that returned REPEAT_SCAN with cards came back NO_RESULT (2026-09-09).
        """
        manifest = {"capture_profile": "consumer"}
        if header.get("illuminance_lux") is not None:
            manifest["illuminance_lux"] = float(header["illuminance_lux"])
        if header.get("capture_profile"):
            manifest["capture_profile"] = str(header["capture_profile"])
        if isinstance(header.get("manifest"), dict):
            manifest.update(header["manifest"])
        return manifest

    @staticmethod
    def _run_assembled(path: str, header: dict) -> dict:
        """The same production path as `_run`, on a clip already on disk."""
        doc, det = measure_video_details(
            path, manifest=MeasureHandler._build_manifest(header),
            client_timestamps_s=header.get("timestamps_s"),
            duration_ms=header.get("duration_ms"),
            config_overrides=header.get("config_overrides"),
            window_s=header.get("window_s"), scale=header.get("scale"),
            participant=_participant_block(header),
            reference=header.get("reference"))
        doc["session"] = header.get("session") or ""
        # ShenAI route (2026-09-16): the sidecar posted after /api/start was
        # accepted is held in memory under the part dir's name; the route runs
        # the SAME decision on the train when the video path abstained on
        # interval gates alone. Recorded on the doc used or not.
        _apply_shenai_route(doc, det, os.path.basename(os.path.dirname(path)),
                            part_dir=os.path.dirname(path), attachment=header)
        _apply_trace_path(doc, det, os.path.basename(os.path.dirname(path)),
                          MeasureHandler._build_manifest(header))
        _apply_vascular_tone(doc, _held_traces(os.path.basename(os.path.dirname(path))),
                             header)
        _apply_afib_result(doc, det)
        if header.get("reference"):
            doc["reference"] = header["reference"]
        if header.get("client_capture"):
            doc["client_capture"] = header["client_capture"]
        # `path` still points inside the part dir, where /api/scan-signals
        # parks the ShenAI JSON. It is posted after /api/start was accepted, so
        # it has had the whole 20-45 s of this job to land - and _retain_clip,
        # which ran before the job was even queued, will have missed it.
        _pair_shenai(os.path.dirname(path), doc)
        from app.afib_response import finalize_afib_response
        out = finalize_afib_response(doc)
        upload_id = os.path.basename(os.path.dirname(path))
        _retain_traces(upload_id, _peek_traces(upload_id), out)
        return out

    @staticmethod
    def _run(video_bytes: bytes, header: dict) -> dict:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        sid = str(header.get("session") or uuid.uuid4().hex[:12])
        ext = str(header.get("ext") or "webm").lstrip(".")
        d = WORK_DIR / f"{sid}-{uuid.uuid4().hex[:6]}"
        d.mkdir(parents=True, exist_ok=True)
        path = str(d / f"scan.{ext}")
        with open(path, "wb") as f:
            f.write(video_bytes)
        # Prep (downscale/trim) rewrites scan.<ext> IN PLACE. When uploads
        # are retained for diagnosis, keep the untouched original too —
        # otherwise a live scan can never be re-run at native resolution or
        # with a different encoder setting afterwards.
        if os.environ.get("AFIB_KEEP_UPLOADS") == "1":
            shutil.copyfile(path, str(d / f"scan.orig.{ext}"))
        retained = _retain_clip(path, sid)

        try:
            # Both transports use the same video + optional train route.
            doc = MeasureHandler._run_assembled(path, header)
            doc["session"] = sid
            doc["size_bytes"] = len(video_bytes)
            if header.get("reference"):
                doc["reference"] = header["reference"]     # additive; audit only
            if header.get("client_capture"):
                doc["client_capture"] = header["client_capture"]   # additive
            # Keyed on upload_id, NOT session: /api/scan-signals parks the file
            # under the sanitized upload_id, while this path retains under the
            # session (sid, line above). Expect it to find nothing in practice
            # - the single-shot client posts the sidecar only after THIS
            # request has already returned - but a retry or a client that got
            # in early should not silently lose its second opinion.
            if header.get("upload_id"):
                try:
                    _pair_shenai(_part_dir(header["upload_id"]), doc, retained)
                except ValueError:
                    pass
            if isinstance(doc.get("timing"), dict):
                doc["timing"]["upload_received_s"] = header.get("_upload_received_s")
                doc["timing"]["upload_bytes"] = header.get("_upload_bytes")
            # Ground truth in the server log: without this the access log only
            # says 200, which is indistinguishable from "the pipeline refused
            # the clip". Callers debugging an integration need the verdict.
            print(f"[measure] {sid}: outcome={doc.get('outcome')} "
                  f"stars={doc.get('confidence_stars')} "
                  f"clock={(doc.get('clock') or {}).get('source')}/"
                  f"{(doc.get('clock') or {}).get('implied_fps')}fps "
                  f"reasons={doc.get('no_read_reasons')}", flush=True)
            return doc
        finally:
            if os.environ.get("AFIB_KEEP_UPLOADS") != "1":
                for p in d.glob("*"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
                try:
                    d.rmdir()
                except OSError:
                    pass


class _MeasureServer(ThreadingHTTPServer):
    """handle_error is a SERVER hook (socketserver.BaseServer), called from
    the request thread — not a handler method. A peer that resets the
    connection before a request line even arrives raises from the stdlib's
    own readline, before any handler code runs, so this is the only place
    it can be made quiet. One line for a dropped connection; anything else
    is a real error and gets the default traceback."""

    def handle_error(self, request, client_address):
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            print(f"[measure] connection dropped by {client_address[0]}: "
                  f"{type(exc).__name__}", flush=True)
            return
        super().handle_error(request, client_address)


def _maybe_reorder_sheet() -> None:
    """One-time maintenance at startup, opt-in: AFIB_SHEET_REORDER_ON_START
    truthy makes THIS service rewrite its own sheet tab so the columns follow
    result_sheet.COLUMNS (rows re-mapped by header name, nothing lost). It
    runs where the sheet credentials already live, so no key has to leave
    Railway; the outcome is echoed on /healthz under launch_overrides. Remove
    the variable after the deploy that ran it."""
    v = os.environ.get("AFIB_SHEET_REORDER_ON_START", "").strip().lower()
    if v not in ("1", "true", "yes", "on"):
        return
    try:
        if not result_sheet.is_configured():
            LAUNCH_OVERRIDES["sheet.reorder"] = "requested, but the sheet is not configured"
            return
        LAUNCH_OVERRIDES["sheet.reorder"] = result_sheet.reorder_existing_tab()
    except Exception as e:                                     # noqa: BLE001
        LAUNCH_OVERRIDES["sheet.reorder"] = f"FAILED ({type(e).__name__}: {str(e)[:120]})"
    print(f"[sheet] reorder on start: {LAUNCH_OVERRIDES['sheet.reorder']}", flush=True)


def serve(host: str = "127.0.0.1", port: int = 8790):
    _maybe_reorder_sheet()
    httpd = _MeasureServer((host, port), MeasureHandler)
    print(f"[measure] afib measure API on http://{host}:{port}", flush=True)
    print(f"[measure]   POST /api/process-video   GET /healthz", flush=True)
    print(f"[measure]   allow-origin={ALLOW_ORIGIN} "
          f"max_concurrent={MAX_CONCURRENT}", flush=True)
    if LAUNCH_OVERRIDES:
        print(f"[measure]   LAUNCH OVERRIDES ACTIVE: {LAUNCH_OVERRIDES}",
              flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[measure] shutting down", flush=True)
    finally:
        httpd.server_close()
