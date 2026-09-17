"""A lost start acknowledgement must not cost a finished scan or a second job."""
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from app import measure_api as api


def handler():
    h = api.MeasureHandler.__new__(api.MeasureHandler)
    h.headers = {}
    h._json = Mock()
    return h


@pytest.fixture
def upload(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(api, "_STARTED", {})
    monkeypatch.setattr(api, "_RESULTS", api.OrderedDict())
    monkeypatch.setattr(api, "_INFLIGHT", threading.BoundedSemaphore(1))
    monkeypatch.setattr(api, "_POOL", Mock())
    monkeypatch.setattr(api, "_retain_clip", Mock(return_value=None))
    monkeypatch.setattr(api.result_sheet, "schedule_append", Mock())
    monkeypatch.delenv("AFIB_KEEP_UPLOADS", raising=False)
    d = api._part_dir("scan-one")
    d.mkdir()
    (d / "part-0000").write_bytes(b"video")
    (d / "total").write_text("1")
    return {"upload_id": [d.name], "ext": ["webm"]}


def test_lost_ack_retry_returns_existing_job_even_when_workers_are_full(upload):
    first, retry = handler(), handler()
    first._start_job(upload)
    retry._start_job(upload)
    assert first._json.call_args.args[0] == 202
    assert retry._json.call_args.args[0] == 202
    assert retry._json.call_args.args[1]["upload_id"] == "scan-one"
    assert api._POOL.submit.call_count == 1


def test_simultaneous_starts_schedule_exactly_one_job(upload):
    callers = [handler(), handler()]
    barrier = threading.Barrier(2)
    def start(h):
        barrier.wait()
        h._start_job(upload)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(start, callers))
    assert [h._json.call_args.args[0] for h in callers] == [202, 202]
    assert api._POOL.submit.call_count == 1


def test_completed_retry_collects_same_scan_without_scheduling(upload, monkeypatch):
    h = handler()
    monkeypatch.setattr(h, "_run_assembled", Mock(return_value={"outcome": "REPEAT_SCAN"}))
    h._start_job(upload)
    api._POOL.submit.call_args.args[0]()   # execute the detached job
    result = api._recall_result("scan-one")
    assert result["scan_id"] == result["upload_id"] == "scan-one"
    retry = handler()
    retry._start_job(upload)
    assert retry._json.call_args.args[0] == 200
    assert retry._json.call_args.args[1]["status"] == "complete"
    assert api._POOL.submit.call_count == 1
    assert api.result_sheet.schedule_append.call_count == 1


def test_rejected_submission_is_explicit_collectable_failure_and_releases_slot(upload):
    api._POOL.submit.side_effect = RuntimeError("executor stopped")
    h = handler()
    h._start_job(upload)
    assert h._json.call_args.args[0] == 503
    failure = api._recall_result("scan-one")
    assert failure["analysis_state"] == "failed"
    assert failure["error_code"] == "job_submission_failed"
    assert failure["scan_id"] == "scan-one"
    assert "outcome" not in failure   # operational failure is not a rhythm class
    assert not api._STARTED
    assert api._INFLIGHT.acquire(blocking=False)


def test_sweep_does_not_delete_an_active_upload(upload, monkeypatch):
    d = api._part_dir("scan-one")
    api._STARTED[d.name] = True
    monkeypatch.setattr(api.time, "time", lambda: d.stat().st_mtime + 7200)
    api._sweep_parts()
    assert (d / "part-0000").exists()


@pytest.mark.parametrize("invalid", [None, {}, {"status": "processing"}])
def test_an_invalid_worker_return_is_a_collectable_failure(upload, monkeypatch, invalid):
    h = handler()
    monkeypatch.setattr(h, "_run_assembled", Mock(return_value=invalid))
    h._start_job(upload)
    api._POOL.submit.call_args.args[0]()
    result = api._recall_result("scan-one")
    assert result["analysis_state"] == "failed"
    assert result["error_code"] == "analysis_failed"
    assert result["scan_id"] == "scan-one"
    assert "outcome" not in result
    assert not api._STARTED


@pytest.mark.parametrize("bad_id", ["", "a/b", "../scan-one", "x" * 81])
def test_start_rejects_ambiguous_ids_before_any_work(upload, bad_id):
    h = handler()
    h._start_job({"upload_id": [bad_id]})
    assert h._json.call_args.args[0] == 400
    assert api._POOL.submit.call_count == 0
