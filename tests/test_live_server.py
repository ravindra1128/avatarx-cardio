"""
Live demo HTTP server (app/server.py): the page talks to these endpoints.

The test plays the browser's role exactly — creates a session, streams
raw RGBA frame batches in the wire format the page uses, starts/finishes a
scan, polls status — against a real server thread on a free port.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import struct
import time
import urllib.request

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.server import start_server
from capture.video_reader import iter_frames
from scripts.make_synth_video import synth_video


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    d = tmp_path_factory.mktemp("srv")
    srv, port = start_server(port=0, work_dir=str(d), open_browser=False)
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return r.status, json.loads(r.read().decode())


def _post_json(base, path, obj):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read().decode())


def _post_frames(base, sid, frames_bgr, ts):
    """Wire format: uint32 LE header length, JSON header, raw RGBA bytes."""
    h, w = frames_bgr[0].shape[:2]
    rgba = np.zeros((len(frames_bgr), h, w, 4), np.uint8)
    for i, f in enumerate(frames_bgr):
        rgba[i, ..., :3] = f[..., ::-1]
        rgba[i, ..., 3] = 255
    header = json.dumps({"session": sid, "w": w, "h": h, "n": len(frames_bgr),
                         "ts": list(map(float, ts))}).encode()
    body = struct.pack("<I", len(header)) + header + rgba.tobytes()
    req = urllib.request.Request(base + "/api/frames", data=body,
                                 headers={"Content-Type":
                                          "application/octet-stream"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read().decode())


def test_index_and_config(server):
    with urllib.request.urlopen(server + "/", timeout=10) as r:
        html = r.read().decode()
    assert "getUserMedia" in html and "AvatarX" in html
    # Advisory start permission is deliberately easier than downstream
    # signal acceptance. The browser must not wait for result disposition.
    assert "phase === 'preview' && fb.ready) beginCountdown()" in html
    assert "lastFb.disposition !== 'READY'" not in html
    assert "captureState" not in html
    code, cfg = _get(server, "/api/config")
    assert code == 200 and cfg["scan_seconds"] == 30.0


def test_meta_reports_build_and_busy_port_falls_back(server, tmp_path):
    """v0.1.4: a stale `cli.py demo` held the default port for days and the
    browser silently talked to old code. /api/meta lets the page show which
    build answers; a busy port must yield a WORKING server on a free port,
    not an EADDRINUSE crash."""
    code, meta = _get(server, "/api/meta")
    assert code == 200 and meta["code_commit"] and meta["started_at"]
    # occupy a fixed port with one server, then request the SAME port again
    srv1, p1 = start_server(port=0, work_dir=str(tmp_path / "a"),
                            open_browser=False)
    try:
        srv2, p2 = start_server(port=p1, work_dir=str(tmp_path / "b"),
                                open_browser=False)
        try:
            assert p2 != p1
            code, meta2 = _get(f"http://127.0.0.1:{p2}", "/api/meta")
            assert code == 200 and meta2["code_commit"] == meta["code_commit"]
        finally:
            srv2.shutdown()
    finally:
        srv1.shutdown()


def test_full_journey_over_http(server, tmp_path):
    p = str(tmp_path / "s.avi")
    synth_video(p, kind="sinus", fps=30.0, duration_s=32.0, seed=21)
    frames = [(t, f) for t, f in iter_frames(p)]
    code, s = _post_json(server, "/api/session", {"fps_hint": 30.0,
                                                  "scan_seconds": 16.0})
    assert code == 200
    sid = s["id"]
    # preview until READY (window fill + hold), through the real gate
    fb = None
    for i in range(0, 420, 15):
        batch = frames[i:i + 15]
        _, fb = _post_frames(server, sid, [f for _, f in batch],
                             [t for t, _ in batch])
    assert fb["face_found"] and fb["ready"], fb["readiness"]
    code, st = _post_json(server, "/api/scan/start",
                          {"session": sid,
                           "camera_settings": {"exposureMode": "continuous"}})
    assert code == 200 and st["state"] == "scanning"
    for i in range(420, 960, 15):
        batch = frames[i:i + 15]
        _, fb = _post_frames(server, sid, [f for _, f in batch],
                             [t for t, _ in batch])
    assert fb["progress"] >= 0.99, fb
    code, st = _post_json(server, "/api/scan/finish", {"session": sid})
    assert st["state"] in ("processing", "done")
    for _ in range(600):
        _, st = _get(server, f"/api/status?session={sid}")
        if st["state"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert st["state"] == "done", st
    r = st["result"]
    assert r["outcome"] == "ACCEPT" and r["predicted_class"] == "SINUS"
    # v0.2.1: the unified Cardiac Rhythm Scan Report rides the result;
    # v0.3: findings-only by default — no waveform of any kind
    rep = r.get("report_html", "")
    assert "AvatarX Cardiac Rhythm Scan Report" in rep
    assert "FINDINGS" in rep
    assert "<polyline" not in rep and "<svg" not in rep
    assert [h["head"] for h in r.get("head_results", [])][:1] == ["afib"]
    assert r["user_facing_text"]
    assert r["provenance"]["config_hash"] not in ("", "unknown")


def test_unknown_session_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(server, "/api/status?session=nope")
    assert e.value.code == 404


def test_sessions_diagnostics_endpoint(server):
    code, s = _post_json(server, "/api/session", {"scan_seconds": 5.0})
    sid = s["id"]
    code, doc = _get(server, "/api/sessions")
    assert code == 200
    row = next((r for r in doc["sessions"] if r["id"] == sid), None)
    assert row is not None and row["state"] == "preview"


def test_test_source_endpoint_serves_labelled_video(server):
    """The in-app self-test hook: a portrait/synthetic clip the page can
    play INSTEAD of the camera, clearly labelled — never the default."""
    with urllib.request.urlopen(server + "/api/testsource?kind=sinus",
                                timeout=300) as r:
        data = r.read()
        ctype = r.headers.get("Content-Type", "")
    assert len(data) > 10000
    assert "video/" in ctype
