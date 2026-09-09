"""Delivering a finished scan over a connection that may not survive it.

A scan is ~10 s of upload then 20-45 s of analysis with no bytes moving. On
2026-09-09 a real phone scan died with ERR_HTTP2_PING_FAILED during that quiet
stretch and a completed analysis was discarded. Two defences, both pinned here:
the response is streamed with heartbeats so the socket never goes quiet, and
the finished result is held briefly so a client can come back for it.
"""
import inspect
import io
import json
import os
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

os.environ.setdefault("AFIB_SHEET_ID", "test-sheet")

from app import measure_api  # noqa: E402
from app.measure_api import MeasureHandler  # noqa: E402


class _Wire(io.BytesIO):
    """A socket that can be told to break at the Nth write."""

    def __init__(self, break_after=None):
        super().__init__()
        self.break_after = break_after
        self.writes = 0

    def write(self, b):
        self.writes += 1
        if self.break_after is not None and self.writes > self.break_after:
            raise BrokenPipeError("client gone")
        return super().write(b)

    def flush(self):
        return None


def _handler(wire):
    h = MeasureHandler.__new__(MeasureHandler)
    h.wfile = wire
    h.close_connection = False
    h.headers = {}
    h.sent = []
    h.send_response = lambda code, *a: h.sent.append(code)
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda: None
    h._cors = lambda: None
    return h


def _decode_chunked(raw: bytes) -> bytes:
    """Minimal HTTP/1.1 chunked decoder — proves the framing is well formed."""
    out, i = b"", 0
    while True:
        j = raw.index(b"\r\n", i)
        size = int(raw[i:j], 16)
        i = j + 2
        if size == 0:
            assert raw[i:] == b"\r\n", "chunked body not terminated correctly"
            return out
        out += raw[i:i + size]
        assert raw[i + size:i + size + 2] == b"\r\n", "bad chunk terminator"
        i += size + 2


def test_a_slow_job_is_answered_with_heartbeats_and_still_parses_as_json():
    wire = _Wire()
    h = _handler(wire)
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(lambda: (time.sleep(0.25), {"outcome": "ACCEPT"})[1])
        doc, delivered = h._json_streaming(fut, heartbeat_s=0.05)
    assert delivered is True
    assert doc == {"outcome": "ACCEPT"}
    body = _decode_chunked(wire.getvalue())
    assert body.startswith(b" "), "no heartbeat was sent during the wait"
    assert json.loads(body) == {"outcome": "ACCEPT"}    # leading space is legal JSON


def test_a_fast_job_needs_no_heartbeat_and_frames_identically():
    wire = _Wire()
    h = _handler(wire)
    with ThreadPoolExecutor(max_workers=1) as pool:
        doc, delivered = h._json_streaming(pool.submit(lambda: {"outcome": "ACCEPT"}),
                                           heartbeat_s=5.0)
    assert delivered is True
    assert json.loads(_decode_chunked(wire.getvalue())) == {"outcome": "ACCEPT"}


def test_a_job_that_raises_becomes_a_body_level_error_not_a_torn_response():
    """The status line is already sent, so a failure cannot be an HTTP 500
    here. Clients treat a body-level `error` as a failure either way."""
    wire = _Wire()
    h = _handler(wire)

    def boom():
        raise RuntimeError("pipeline exploded")

    with ThreadPoolExecutor(max_workers=1) as pool:
        doc, delivered = h._json_streaming(pool.submit(boom), heartbeat_s=0.05)
    assert delivered is True
    assert "pipeline exploded" in doc["error"]
    assert "pipeline exploded" in json.loads(_decode_chunked(wire.getvalue()))["error"]
    assert h.sent == [200]


def test_the_job_finishes_even_when_the_client_disappears_mid_analysis():
    """The whole point: the analysis is expensive, so a dead socket must not
    abandon it. The caller still gets the doc, marked undelivered."""
    ran = threading.Event()

    def slow():
        time.sleep(0.3)
        ran.set()
        return {"outcome": "ACCEPT"}

    wire = _Wire(break_after=0)          # the very first heartbeat write fails
    h = _handler(wire)
    with ThreadPoolExecutor(max_workers=1) as pool:
        doc, delivered = h._json_streaming(pool.submit(slow), heartbeat_s=0.05)
    assert ran.is_set(), "the job was abandoned when the client went away"
    assert delivered is False
    assert doc == {"outcome": "ACCEPT"}
    assert h.close_connection is True


def test_a_finished_result_can_be_collected_after_the_connection_dies():
    measure_api._RESULTS.clear()
    doc = {"outcome": "REPEAT_SCAN", "session": "s-1"}
    measure_api._remember_result("s-1", doc)
    assert measure_api._recall_result("s-1") == doc
    assert measure_api._recall_result("s-2") is None
    assert measure_api._recall_result(None) is None


def test_the_cache_is_bounded_and_expires():
    measure_api._RESULTS.clear()
    for i in range(measure_api.MAX_CACHED_RESULTS + 5):
        measure_api._remember_result(f"s{i}", {"outcome": "ACCEPT", "n": i})
    assert len(measure_api._RESULTS) == measure_api.MAX_CACHED_RESULTS
    assert measure_api._recall_result("s0") is None            # oldest evicted
    assert measure_api._recall_result(f"s{measure_api.MAX_CACHED_RESULTS + 4}")

    measure_api._RESULTS.clear()
    measure_api._remember_result("old", {"outcome": "ACCEPT"})
    measure_api._RESULTS["old"] = (time.time() - measure_api.RESULT_TTL_S - 1,
                                   {"outcome": "ACCEPT"})
    assert measure_api._recall_result("old") is None


def test_a_non_dict_result_is_never_cached():
    measure_api._RESULTS.clear()
    measure_api._remember_result("s", None)
    measure_api._remember_result("s", "not a doc")
    assert measure_api._recall_result("s") is None


def test_the_collection_key_is_per_attempt_not_per_session():
    """The webapp sends a constant session id in development, so a result must
    be collected under a per-upload id or a client could be handed a previous
    scan's answer."""
    _, header = measure_api._parse_envelope(
        b"", "video/webm",
        {"session": ["1234567890"], "upload_id": ["  1234567890-abc123  "]})
    assert header["upload_id"] == "1234567890-abc123"

    _, header = measure_api._parse_envelope(b"", "video/webm", {"upload_id": ["x" * 200]})
    assert len(header["upload_id"]) == 80

    _, header = measure_api._parse_envelope(b"", "video/webm", {"upload_id": ["  "]})
    assert "upload_id" not in header


# --- chunked upload + detached processing (2026-09-09) -----------------------
# One request carrying ~23 MB and then holding the line through ~40 s of
# analysis is 80-105 s of connection on a phone, and it kept dying inside the
# body, where neither a heartbeat nor the result cache can help.

def test_parts_assemble_in_index_order(tmp_path, monkeypatch):
    monkeypatch.setattr(measure_api, "UPLOAD_DIR", tmp_path)
    d = measure_api._part_dir("up-1")
    d.mkdir(parents=True)
    (d / "total").write_text("3")
    for i, chunk in enumerate((b"aaa", b"bbb", b"ccc")):
        (d / f"part-{i:04d}").write_bytes(chunk)
    path, upload_s = measure_api._assemble_parts("up-1", "webm")
    assert upload_s >= 0
    assert pathlib.Path(path).read_bytes() == b"aaabbbccc"
    assert not list(d.glob("part-*"))          # slices freed after assembly


def test_a_missing_slice_is_an_error_not_a_shorter_video(tmp_path, monkeypatch):
    """Silently analysing 2 of 3 slices would be a truncated scan reported as
    a real one."""
    monkeypatch.setattr(measure_api, "UPLOAD_DIR", tmp_path)
    d = measure_api._part_dir("up-2")
    d.mkdir(parents=True)
    (d / "total").write_text("3")
    (d / "part-0000").write_bytes(b"aaa")
    (d / "part-0002").write_bytes(b"ccc")
    with pytest.raises(ValueError, match="incomplete"):
        measure_api._assemble_parts("up-2", "webm")


def test_an_upload_id_cannot_escape_the_parts_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(measure_api, "UPLOAD_DIR", tmp_path)
    assert measure_api._part_dir("../../etc/passwd").parent == tmp_path
    assert measure_api._part_dir("a/b").parent == tmp_path
    with pytest.raises(ValueError):
        measure_api._part_dir("../..")


def test_both_upload_paths_send_the_same_manifest():
    """The detached path once omitted capture_profile, so the pipeline
    defaulted to "research" and the same clip that returned REPEAT_SCAN with
    cards came back NO_RESULT. One builder, used by both."""
    h = {"session": "s"}
    assert measure_api.MeasureHandler._build_manifest(h)["capture_profile"] == "consumer"
    h2 = {"capture_profile": "research"}
    assert measure_api.MeasureHandler._build_manifest(h2)["capture_profile"] == "research"
    h3 = {"manifest": {"exposure_locked": True}, "illuminance_lux": 120}
    m = measure_api.MeasureHandler._build_manifest(h3)
    assert m["exposure_locked"] is True and m["illuminance_lux"] == 120.0
    assert m["capture_profile"] == "consumer"
    src = inspect.getsource(measure_api.MeasureHandler._run)
    assert "_build_manifest(header)" in src, "the single-shot path must use it too"


def test_slices_may_differ_in_size_and_the_last_total_wins(tmp_path, monkeypatch):
    """The client sends a small timed probe first, then slices sized to the
    link it measured, and only the LAST slice knows the true total. Assembly
    must concatenate whatever sizes arrived, in index order, and trust the
    last-written total."""
    monkeypatch.setattr(measure_api, "UPLOAD_DIR", tmp_path)
    d = measure_api._part_dir("up-3")
    d.mkdir(parents=True)
    (d / "total").write_text("9")                  # the probe's provisional guess
    (d / "part-0000").write_bytes(b"a" * 4)         # probe
    (d / "part-0001").write_bytes(b"b" * 16)        # bigger, after measuring
    (d / "total").write_text("3")                  # the last slice's true total
    (d / "part-0002").write_bytes(b"c" * 5)
    path, _ = measure_api._assemble_parts("up-3", "webm")
    assert pathlib.Path(path).read_bytes() == b"a" * 4 + b"b" * 16 + b"c" * 5
