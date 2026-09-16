"""Audit 2026-09-17 #6a/#6b — availability around the detached job.

#6a: /api/start used to consume (assemble + delete) the upload slices BEFORE
taking a worker slot; a 503 then left the client's retry with "no parts" ->
400 and the scan was lost. The slot is now taken first, so a busy service
answers 503 with the slices intact.

#6b: a finished job's part directory (assembled clip, ~220 MB FFV1
intermediate, sidecars) is removed at the end of the job unless
AFIB_KEEP_UPLOADS asks for on-disk debugging.
"""
import io
import pathlib

import pytest

from app import measure_api as api


def _handler(path="/", body=b""):
    h = api.MeasureHandler.__new__(api.MeasureHandler)
    h.path = path
    h.rfile = io.BytesIO(body)
    h.headers = {"Content-Length": str(len(body)), "User-Agent": "t"}
    h.close_connection = False
    h.sent = []
    h._json = lambda code, doc: h.sent.append((code, doc))
    return h


def _upload_one_part(upload_id: str, payload: bytes = b"x"):
    h = _handler("/api/upload-part", payload)
    h._upload_part({"upload_id": [upload_id], "index": ["0"], "total": ["1"]})
    assert h.sent and h.sent[-1][0] == 200, h.sent


def test_busy_service_answers_503_and_keeps_the_slices(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "parts", raising=False)
    uid = "busy-test-upload"
    _upload_one_part(uid)
    d = api._part_dir(uid)
    before = sorted(p.name for p in d.iterdir())
    assert before, "no part written"
    # exhaust every worker slot
    taken = 0
    while api._INFLIGHT.acquire(blocking=False):
        taken += 1
    try:
        h = _handler("/api/start")
        h._start_job({"upload_id": [uid]})
        code, doc = h.sent[-1]
        assert code == 503 and "busy" in doc["error"]
        # the slices must still be there for the client's retry
        assert sorted(p.name for p in d.iterdir()) == before
    finally:
        for _ in range(taken):
            api._INFLIGHT.release()


def test_discard_part_dir_removes_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "parts", raising=False)
    monkeypatch.delenv("AFIB_KEEP_UPLOADS", raising=False)
    uid = "done-upload"
    d = api._part_dir(uid); d.mkdir(parents=True, exist_ok=True)
    (d / "scan.webm").write_bytes(b"v")
    (d / "scan.webm.scaled.avi").write_bytes(b"f" * 10)
    (d / "scan.webm.timestamps.json").write_text("{}")
    api._discard_part_dir(uid)
    assert not d.exists()


def test_discard_part_dir_keeps_it_when_debugging_is_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "parts", raising=False)
    monkeypatch.setenv("AFIB_KEEP_UPLOADS", "1")
    uid = "kept-upload"
    d = api._part_dir(uid); d.mkdir(parents=True, exist_ok=True)
    (d / "scan.webm").write_bytes(b"v")
    api._discard_part_dir(uid)
    assert (d / "scan.webm").exists()
