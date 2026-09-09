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
import os
import pathlib
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
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
        body = json.dumps(doc, default=str).encode()
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
        self._json(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
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
            doc = fut.result()
            self._json(200, doc)
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
            import shutil
            shutil.copyfile(path, str(d / f"scan.orig.{ext}"))

        # A browser scan IS a consumer capture: the camera won't lock AE/AWB
        # and real webm lands at 24-30 fps, not the research profile's 30+.
        # capture/ingest.py exists for exactly this and surfaces caveats
        # instead of silently accepting research-grade claims. Callers may
        # still ask for "research" explicitly.
        manifest = {"capture_profile": "consumer"}
        if header.get("illuminance_lux") is not None:
            manifest["illuminance_lux"] = float(header["illuminance_lux"])
        if header.get("capture_profile"):
            manifest["capture_profile"] = str(header["capture_profile"])
        if isinstance(header.get("manifest"), dict):
            manifest.update(header["manifest"])

        try:
            doc = measure_video(
                path, manifest=manifest or None,
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
