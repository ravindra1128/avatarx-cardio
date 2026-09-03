"""
Local HTTP server for the live scan demo (v0.1.1). Standard library only.

    GET  /                        the single-page app (app/static/index.html)
    GET  /api/config              {scan_seconds, ...}
    POST /api/session             {fps_hint?, scan_seconds?} -> {id}
    POST /api/frames              binary batch (see below) -> feedback
    POST /api/scan/start          {session, camera_settings?} -> {state}
    POST /api/scan/finish         {session} -> {state}
    POST /api/scan/abort          {session, code?, message?}
    GET  /api/status?session=ID   {state, feedback, error, result}
    GET  /api/testsource?kind=    a labelled synthetic clip for SELF-TEST
                                  (the page plays it instead of the camera
                                  only when explicitly asked via ?source=test)

Frame batch wire format (POST /api/frames, application/octet-stream):
    uint32 LE header_len | header JSON {session, w, h, n, ts:[...]} |
    n * h * w * 4 bytes RGBA (top-left origin, as canvas getImageData)
Frames are RAW — no JPEG/WebM in the path. Lossy compression is the one
thing this programme's own evidence says destroys the pulse.

The camera never touches this process: the browser owns it (getUserMedia),
which is also what makes "ask the user for permission" a real, native step.
"""
from __future__ import annotations

import json
import pathlib
import struct
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse, parse_qs

import numpy as np

from app.scan_engine import ScanSession, DEFAULT_SCAN_SECONDS

_STATIC = pathlib.Path(__file__).resolve().parent / "static"
_STARTED_AT = time.strftime("%Y-%m-%d %H:%M:%S")
_SESSIONS: dict[str, ScanSession] = {}
_SESSIONS_LOCK = threading.Lock()
_VSESSIONS: dict = {}                    # v0.4 three-phase orchestrators
_VSESSION_BY_SCAN: dict[str, str] = {}   # recovery ScanSession id -> vsid
_WORK_DIR = pathlib.Path("/tmp/avatarx_live")
_TESTSRC_CACHE: dict[str, pathlib.Path] = {}
_TESTSRC_LOCK = threading.Lock()


def _gc_sessions() -> None:
    """Keep memory bounded (caller holds _SESSIONS_LOCK): drop scans
    older than 1 h UNLESS a still-running vsession owns them, and prune
    finished vsessions (with their phase scans + last_det arrays) the
    same way."""
    now = time.time()
    done_vs = [k for k, v in _VSESSIONS.items()
               if v.phase.value in ("done", "failed")
               and now - getattr(v.rest, "created_at", now) > 3600]
    for k in done_vs:
        vs = _VSESSIONS.pop(k)
        for sid in (vs.rest.id, getattr(vs.recovery, "id", None)):
            _SESSIONS.pop(sid or "", None)
            _VSESSION_BY_SCAN.pop(sid or "", None)
    owned = set()
    for v in _VSESSIONS.values():
        owned.add(v.rest.id)
        if v.recovery is not None:
            owned.add(v.recovery.id)
    stale = [k for k, v in _SESSIONS.items()
             if now - v.created_at > 3600 and k not in owned]
    for k in stale:
        _SESSIONS.pop(k, None)
        _VSESSION_BY_SCAN.pop(k, None)


def _testsource(kind: str) -> pathlib.Path:
    """Generate (once) a browser-playable synthetic clip for self-test."""
    from scripts.make_synth_video import synth_video, synth_portrait_video, \
        portrait_available
    key = kind
    with _TESTSRC_LOCK:
        if key in _TESTSRC_CACHE and _TESTSRC_CACHE[key].exists():
            return _TESTSRC_CACHE[key]
        d = _WORK_DIR / "testsource"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"testsource_{kind}.webm"
        rhythm = "af" if kind.startswith("af") else "sinus"
        # Kept short (14 s ≈ 3 s countdown grace + a shortened self-test
        # scan) so the clip stays small enough to download quickly; the
        # server does not serve byte ranges, and seek-based self-test needs
        # the whole clip buffered.
        if kind.endswith("_portrait") and portrait_available():
            synth_portrait_video(str(p), kind=rhythm, fps=30.0,
                                 duration_s=14.0, seed=77, codec="VP90")
        else:
            synth_video(str(p), kind=rhythm, fps=30.0, duration_s=14.0,
                        seed=77, codec="VP90")
        _TESTSRC_CACHE[key] = p
        return p


class _Handler(BaseHTTPRequestHandler):
    server_version = "AvatarXLive/0.1.1"

    def log_message(self, fmt, *args):            # quiet by default
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    # ---------------------------------------------------------- helpers
    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0"))
        buf = bytearray()
        while len(buf) < n:
            chunk = self.rfile.read(min(1 << 20, n - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    def _session(self, sid: Optional[str]) -> Optional[ScanSession]:
        with _SESSIONS_LOCK:
            return _SESSIONS.get(sid or "")

    # -------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            body = (_STATIC / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/api/config":
            from configs import load_config, config_hash
            cfg = load_config()
            from app.session_flow import three_phase_enabled
            self._json(200, {"scan_seconds": DEFAULT_SCAN_SECONDS,
                             "sqi_floor": cfg["decision"]["sqi_floor"],
                             "readiness": cfg["decision"].get("readiness", {}),
                             "evidence": cfg["decision"].get("evidence", {}),
                             "config_hash": config_hash(cfg),
                             "capture_profile": "consumer",
                             "three_phase": three_phase_enabled(cfg)})
            return
        if u.path == "/api/vsession/status":
            vs = _VSESSIONS.get((q.get("session") or [None])[0] or "")
            if vs is None:
                self._json(404, {"error": "unknown vsession"})
                return
            self._json(200, vs.status())
            return
        if u.path == "/api/meta":
            # which BUILD is this page talking to? (v0.1.4: a stale demo
            # server once served days-old code on the default port)
            import os
            from inference.pipeline import _git_commit
            self._json(200, {"code_commit": _git_commit(), "pid": os.getpid(),
                             "started_at": _STARTED_AT})
            return
        if u.path == "/api/status":
            s = self._session((q.get("session") or [None])[0])
            if s is None:
                self._json(404, {"error": "unknown session"})
                return
            self._json(200, s.status())
            return
        if u.path == "/api/sessions":
            # localhost diagnostics: enumerate sessions and their outcome.
            # Used by ops and by the headless self-test to observe a run.
            with _SESSIONS_LOCK:
                rows = [{"id": k, "state": v.state.value,
                         "outcome": (v.result or {}).get("outcome"),
                         "predicted_class": (v.result or {}).get("predicted_class"),
                         "error": (v.error or {}).get("code")}
                        for k, v in _SESSIONS.items()]
                vrows = [{"id": k, "phase": v.phase.value,
                          "outcome": (v.result or {}).get("outcome")}
                         for k, v in _VSESSIONS.items()]
            self._json(200, {"sessions": rows, "vsessions": vrows})
            return
        if u.path == "/api/testsource":
            kind = (q.get("kind") or ["sinus"])[0]
            if kind not in ("sinus", "af", "sinus_portrait", "af_portrait"):
                self._json(400, {"error": "unknown kind"})
                return
            try:
                p = _testsource(kind)
            except Exception as e:
                self._json(500, {"error": str(e)})
                return
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/webm")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(404, {"error": "not found"})

    # ------------------------------------------------------------- POST
    def do_POST(self):
        u = urlparse(self.path)
        if u.path.startswith("/api/vsession"):
            self._post_vsession(u)
            return
        if u.path == "/api/session":
            body = self._read_body()
            opts = json.loads(body.decode() or "{}") if body else {}
            sid = uuid.uuid4().hex[:12]
            sess = ScanSession(sid, str(_WORK_DIR / sid),
                               fps_hint=float(opts.get("fps_hint", 30.0)),
                               scan_seconds=float(opts.get("scan_seconds",
                                                           DEFAULT_SCAN_SECONDS)))
            with _SESSIONS_LOCK:
                _gc_sessions()
                _SESSIONS[sid] = sess
            self._json(200, {"id": sid, "scan_seconds": sess.scan_seconds})
            return

        if u.path == "/api/client_error":
            body = self._read_body()
            try:
                rec = json.loads(body.decode() or "{}")
            except ValueError:
                rec = {"kind": "unparseable"}
            try:
                with open(_WORK_DIR / "client_errors.jsonl", "a") as f:
                    f.write(json.dumps(rec, default=str)[:2000] + "\n")
            except OSError:
                pass
            self._json(200, {"ok": True})
            return

        if u.path == "/api/frames":
            body = self._read_body()
            if len(body) < 4:
                self._json(400, {"error": "empty batch"})
                return
            (hlen,) = struct.unpack("<I", body[:4])
            header = json.loads(body[4:4 + hlen].decode())
            s = self._session(header.get("session"))
            if s is None:
                self._json(404, {"error": "unknown session"})
                return
            w, h, n = int(header["w"]), int(header["h"]), int(header["n"])
            raw = body[4 + hlen:]
            expected = n * h * w * 4
            if len(raw) != expected:
                self._json(400, {"error": f"expected {expected} bytes of RGBA, "
                                          f"got {len(raw)}"})
                return
            rgba = np.frombuffer(raw, np.uint8).reshape(n, h, w, 4)
            frames = [np.ascontiguousarray(rgba[i, :, :, [2, 1, 0]]
                                           .transpose(1, 2, 0))
                      for i in range(n)]
            ts = [float(t) for t in header.get("ts", [])]
            if len(ts) != n:
                self._json(400, {"error": "timestamps/frames mismatch"})
                return
            s.note_client_dropped(int(header.get("dropped", 0) or 0))
            fb = s.push_frames(frames, ts)
            fb = dict(fb)
            fb["state"] = s.state.value
            fb["error"] = s.error
            self._json(200, fb)
            return

        if u.path in ("/api/scan/start", "/api/scan/finish", "/api/scan/abort"):
            body = self._read_body()
            opts = json.loads(body.decode() or "{}") if body else {}
            s = self._session(opts.get("session"))
            if s is None:
                self._json(404, {"error": "unknown session"})
                return
            try:
                if u.path.endswith("start"):
                    s.start_scan(camera_settings=opts.get("camera_settings"),
                                 force=bool(opts.get("force", False)))
                    # v0.4: a recovery scan starting stops the transition
                    # clock on its orchestrator
                    vsid = _VSESSION_BY_SCAN.get(s.id)
                    if vsid and vsid in _VSESSIONS:
                        _VSESSIONS[vsid].mark_recovery_started()
                elif u.path.endswith("finish"):
                    s.finish_scan()
                else:
                    s.abort(opts.get("code", "cancelled"),
                            opts.get("message", "Scan cancelled."))
            except RuntimeError as e:
                self._json(409, {"error": str(e), "state": s.state.value})
                return
            self._json(200, {"state": s.state.value})
            return
        self._json(404, {"error": "not found"})


    # ---------------------------------------------- v0.4 three-phase flow
    def _post_vsession(self, u):
        from app.session_flow import (MultiPhaseSession,
                                      three_phase_enabled)
        from protocol.safety import SCREEN_QUESTIONS, STOP_RULES
        if u.path == "/api/vsession":
            if not three_phase_enabled():
                self._json(409, {"error": "three-phase sessions are not "
                                          "enabled (config protocol."
                                          "three_phase)"})
                return
            body = self._read_body()
            opts = json.loads(body.decode() or "{}") if body else {}
            vsid = "v" + uuid.uuid4().hex[:11]
            vs = MultiPhaseSession(
                vsid, str(_WORK_DIR / vsid),
                rest_seconds=opts.get("rest_seconds"),
                recovery_seconds=opts.get("recovery_seconds"))
            with _SESSIONS_LOCK:
                _VSESSIONS[vsid] = vs
                _SESSIONS[vs.rest.id] = vs.rest
            self._json(200, {
                "id": vsid, "rest_session": vs.rest.id,
                "rest_seconds": vs.rest_seconds,
                "recovery_seconds": vs.recovery_seconds,
                "challenge": {"protocol_id": vs.challenge.protocol_id,
                              "display_name": vs.challenge.display_name,
                              "cadence_per_min":
                                  vs.challenge.cadence_per_min,
                              "duration_s": vs.challenge.duration_s},
                "safety_questions": SCREEN_QUESTIONS,
                "stop_rules": list(STOP_RULES)})
            return

        if u.path == "/api/vsession/activity_frames":
            body = self._read_body()
            if len(body) < 4:
                self._json(400, {"error": "empty batch"})
                return
            (hlen,) = struct.unpack("<I", body[:4])
            header = json.loads(body[4:4 + hlen].decode())
            vs = _VSESSIONS.get(header.get("vsession") or "")
            if vs is None:
                self._json(404, {"error": "unknown vsession"})
                return
            w, h, n = int(header["w"]), int(header["h"]), int(header["n"])
            raw = body[4 + hlen:]
            if len(raw) != n * h * w * 4:
                self._json(400, {"error": "bad frame payload"})
                return
            rgba = np.frombuffer(raw, np.uint8).reshape(n, h, w, 4)
            frames = [rgba[i, :, :, :3] for i in range(n)]
            live = vs.push_activity_frames(frames,
                                           [float(t) for t in
                                            header.get("ts", [])])
            self._json(200, live)
            return

        body = self._read_body()
        opts = json.loads(body.decode() or "{}") if body else {}
        vs = _VSESSIONS.get(opts.get("session") or "")
        if vs is None:
            self._json(404, {"error": "unknown vsession"})
            return
        try:
            if u.path == "/api/vsession/safety":
                self._json(200, vs.safety(opts.get("answers") or {},
                                          opts.get("participant")))
            elif u.path == "/api/vsession/activity":
                if opts.get("action") == "begin":
                    self._json(200, vs.begin_activity())
                else:
                    self._json(200, vs.end_activity())
            elif u.path == "/api/vsession/recovery":
                sid = uuid.uuid4().hex[:12]
                scan = ScanSession(sid, str(_WORK_DIR / vs.id / "recovery"),
                                   fps_hint=float(opts.get("fps_hint",
                                                           30.0)),
                                   scan_seconds=vs.recovery_seconds,
                                   config=vs.cfg)
                with _SESSIONS_LOCK:
                    _SESSIONS[sid] = scan
                    _VSESSION_BY_SCAN[sid] = vs.id
                vs.attach_recovery(scan)
                self._json(200, {"recovery_session": sid,
                                 "scan_seconds": scan.scan_seconds})
            elif u.path == "/api/vsession/finish":
                self._json(200, {"result": vs.finalize()})
            else:
                self._json(404, {"error": "not found"})
        except (RuntimeError, ValueError) as e:
            self._json(409, {"error": str(e), "phase": vs.phase.value})


def start_server(port: int = 8765, work_dir: Optional[str] = None,
                 open_browser: bool = True, verbose: bool = False
                 ) -> tuple[ThreadingHTTPServer, int]:
    """Start the server on a background thread; returns (server, port).

    If the requested port is taken (v0.1.4: a stale `cli.py demo` from an
    older build once held 8765 for days — the browser then talks to OLD
    code while the user runs new code), fall back to an OS-assigned free
    port with a loud warning instead of crashing with EADDRINUSE. The
    browser is opened with the REAL port either way, and /api/meta lets
    the page display which build it is talking to.
    """
    global _WORK_DIR
    if work_dir:
        _WORK_DIR = pathlib.Path(work_dir)
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    except OSError:
        print(f"WARNING: port {port} is already in use — most likely a "
              f"STALE `cli.py demo` from an earlier session (find it with: "
              f"lsof -nP -iTCP:{port} -sTCP:LISTEN). It may be running OLD "
              "code; kill it to avoid confusion. Starting on a free port "
              "instead.")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    srv.verbose = verbose
    real_port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(
            f"http://127.0.0.1:{real_port}/")).start()
    return srv, real_port


def main(port: int = 8765, open_browser: bool = True) -> None:
    srv, real_port = start_server(port=port, open_browser=open_browser)
    url = f"http://127.0.0.1:{real_port}/"
    print(f"AvatarX AFib live scan demo — open {url}  (Ctrl-C to stop)")
    print("Recordings are kept lossless under", _WORK_DIR)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
