"""Delivering a finished scan over a connection that may not survive it.

A scan is ~10 s of upload then 20-45 s of analysis with no bytes moving. On
2026-09-09 a real phone scan died with ERR_HTTP2_PING_FAILED during that quiet
stretch and a completed analysis was discarded. Two defences, both pinned here:
the response is streamed with heartbeats so the socket never goes quiet, and
the finished result is held briefly so a client can come back for it.
"""
import io
import json
import os
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
