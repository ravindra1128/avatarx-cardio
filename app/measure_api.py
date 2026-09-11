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
# it belonged to was gone. Results are held briefly by session id so the
# client can come back and collect one instead of re-recording and re-uploading
# 30 MB. In memory only, small and short-lived: this is a delivery retry, not
# storage, and the tracking sheet remains the durable record.
RESULT_TTL_S = float(os.environ.get("AFIB_RESULT_TTL_S", "900"))    # 15 min
MAX_CACHED_RESULTS = int(os.environ.get("AFIB_MAX_CACHED_RESULTS", "32"))
_RESULTS: "OrderedDict[str, tuple]" = OrderedDict()
_RESULTS_LOCK = threading.Lock()


def _remember_result(session, doc: dict) -> None:
    if not session or not isinstance(doc, dict):
        return
    with _RESULTS_LOCK:
        _RESULTS[str(session)] = (time.time(), doc)
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


def _clips_enabled() -> bool:
    return os.environ.get("AFIB_KEEP_UPLOADS") == "1" and bool(CLIPS_TOKEN)


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
            if d.is_dir() and now - d.stat().st_mtime > max_age_s:
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
    return None if item is None else item[1]
_STARTED_AT = time.time()
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


def measure_video(video_path: str, *, manifest=None,
                  client_timestamps_s=None, duration_ms=None,
                  config_overrides=None, window_s=None, scale=None) -> dict:
    """Run THE production path and return a JSON-ready ScanResult."""
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
    print(f"[measure] trim: {timing['trim_s']}s applied={trim.get('applied')}",
          flush=True)

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
    if config_overrides:
        from configs import load_config
        cfg = load_config()
        override_notes = _apply_overrides(cfg, config_overrides)

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
        doc["biomarkers"] = report_biomarkers(
            result, det, capture=manifest or {})
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
    return doc


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
        if hlen <= 0 or 4 + hlen > len(body):
            raise ValueError(f"bad envelope header length {hlen}")
        header = json.loads(body[4:4 + hlen].decode())
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
        if urlparse(self.path).path in ("/healthz", "/api/healthz"):
            self._json(200, {"ok": True, "service": "afib-measure",
                             "build": BUILD_SHA,
                             "uptime_s": int(time.time() - _STARTED_AT),
                             "inflight_jobs": _inflight(0),
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
                                   CLIPS_TOKEN.encode("utf-8", "ignore")):
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
        # NEVER 404 of our own accord. The client treats 404 from an upload
        # route as "this service predates the route" and stops; 404 here stays
        # reserved for a service that genuinely has no such path (the catch-all
        # below). Retention off means accepted-and-dropped, not refused.
        if not _clips_enabled():
            print(f"[signals] dropped {len(body)} B for {upload_id!r}: "
                  f"retention disabled", flush=True)
            self._json(200, {"ok": True, "upload_id": d.name,
                             "bytes": len(body), "stored": False})
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

    def _start_job(self, q: dict):
        """Assemble the slices and detach the analysis. Returns immediately:
        the client polls GET /api/result, so no connection is held through the
        job and a dropped link costs nothing."""
        _sweep_parts()
        _, header = _parse_envelope(b"", "video/webm", q)
        upload_id = str((q.get("upload_id") or [""])[0])
        ext = str(header.get("ext") or (q.get("ext") or ["webm"])[0]).lstrip(".")
        try:
            path, upload_s = _assemble_parts(upload_id, ext)
        except (ValueError, OSError) as e:
            self._json(400, {"error": str(e), "upload_id": upload_id})
            return
        size = os.path.getsize(path)
        if not _INFLIGHT.acquire(blocking=False):
            self._json(503, {"error": "measure workers busy — retry",
                             "max_concurrent": MAX_CONCURRENT})
            return
        with _STARTED_LOCK:
            _STARTED[upload_id] = True
        header.setdefault("session", (q.get("session") or [upload_id])[0])
        ua = self.headers.get("User-Agent", "")

        def job():
            n = _inflight(+1)
            t0 = time.perf_counter()
            print(f"[measure] detached job start for {upload_id!r} "
                  f"({size} B, inflight {n}/{MAX_CONCURRENT})", flush=True)
            try:
                doc = self._run_assembled(path, header)
            except Exception as e:                        # noqa: BLE001
                traceback.print_exc()
                doc = {"error": f"{type(e).__name__}: {e}"}
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
                print(f"[measure] detached job end for {upload_id!r} after "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)

        _POOL.submit(job)
        # Tell the client roughly when to bother asking. Measured on Railway:
        # ~1.6 s of work per MB (downscale + analysis), floor 12 s. Without a
        # hint the client polls blindly from t=0 and burns a dozen requests
        # before the answer can possibly exist.
        eta = max(12.0, round(size / (1024 * 1024) * 1.6, 1))
        self._json(202, {"status": "processing", "upload_id": upload_id,
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
        doc = measure_video(
            path, manifest=MeasureHandler._build_manifest(header),
            client_timestamps_s=header.get("timestamps_s"),
            duration_ms=header.get("duration_ms"),
            config_overrides=header.get("config_overrides"),
            window_s=header.get("window_s"), scale=header.get("scale"))
        doc["session"] = header.get("session") or ""
        if header.get("reference"):
            doc["reference"] = header["reference"]
        if header.get("client_capture"):
            doc["client_capture"] = header["client_capture"]
        # `path` still points inside the part dir, where /api/scan-signals
        # parks the ShenAI JSON. It is posted after /api/start was accepted, so
        # it has had the whole 20-45 s of this job to land - and _retain_clip,
        # which ran before the job was even queued, will have missed it.
        _pair_shenai(os.path.dirname(path), doc)
        return doc

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
            doc = measure_video(
                path, manifest=MeasureHandler._build_manifest(header),
                client_timestamps_s=header.get("timestamps_s"),
                duration_ms=header.get("duration_ms"),
                config_overrides=header.get("config_overrides"),
                window_s=header.get("window_s"),
                scale=header.get("scale"))
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


def serve(host: str = "127.0.0.1", port: int = 8790):
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
