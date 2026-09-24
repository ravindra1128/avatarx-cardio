"""Trace-only retention (owner, 2026-09-24: "keep only traces").

The live-frame trace document - four face regions' mean colour per frame, no
image - is what the Vascular Tone card is computed from. With AFIB_KEEP_TRACES
on and a token set, the job keeps it (plus a short note) in its own directory,
independent of clip retention, and it is read back only with the token."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import app.measure_api as api  # noqa: E402

DOC = {"schema_version": 1, "t_s": [0.0, 0.033], "upload_id": "upl-1",
       "traces": {"forehead": [[150.1, 120.2, 100.3], None]}}
RESULT = {"outcome": "ACCEPT", "reference": {"ref_hr": 72.0},
          "debug": {"shenai_input": {"sdk_hr_bpm": 71.5, "sdk_quality": 0.9},
                    "vascular_tone": {"pi_percent": 0.56, "score": 47.9}}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.delenv("AFIB_KEEP_TRACES", raising=False)
    monkeypatch.delenv("AFIB_KEEP_UPLOADS", raising=False)
    monkeypatch.setattr(api, "CLIPS_TOKEN", "", raising=False)
    monkeypatch.delenv("AFIB_CLIPS_TOKEN", raising=False)
    monkeypatch.setattr(api, "TRACES_DIR", tmp_path / "traces", raising=False)
    monkeypatch.setattr(api, "CLIPS_DIR", tmp_path / "clips", raising=False)
    monkeypatch.setattr(api, "TRACES_KEEP", 3, raising=False)


def _enable(monkeypatch, token="s3cret"):
    monkeypatch.setenv("AFIB_KEEP_TRACES", "1")
    monkeypatch.setattr(api, "CLIPS_TOKEN", token, raising=False)


def _handler(path):
    h = api.MeasureHandler.__new__(api.MeasureHandler)
    h.path = path
    h.sent = []
    h._json = lambda code, doc: h.sent.append((code, doc))
    return h


def test_nothing_is_stored_unless_the_switch_and_a_token_are_both_set(monkeypatch):
    api._retain_traces("upl-1", DOC, RESULT)
    assert not api.TRACES_DIR.exists()
    monkeypatch.setenv("AFIB_KEEP_TRACES", "1")            # switch without a token
    api._retain_traces("upl-1", DOC, RESULT)
    assert not api.TRACES_DIR.exists()
    monkeypatch.setenv("AFIB_KEEP_UPLOADS", "1")           # clip retention does not enable it
    monkeypatch.delenv("AFIB_KEEP_TRACES")
    monkeypatch.setattr(api, "CLIPS_TOKEN", "s3cret", raising=False)
    api._retain_traces("upl-1", DOC, RESULT)
    assert not api.TRACES_DIR.exists()


def test_traces_are_kept_without_any_video_and_with_their_note(monkeypatch):
    _enable(monkeypatch)                                    # AFIB_KEEP_UPLOADS stays off
    api._retain_traces("upl-1", DOC, RESULT)
    files = list(api.TRACES_DIR.iterdir())
    assert len(files) == 1 and files[0].name.endswith(".traces.json")
    stored = json.loads(files[0].read_text())
    assert stored["trace_document"] == DOC
    assert stored["service"]["upload_id"] == "upl-1"
    assert stored["service"]["sdk_hr_bpm"] == 71.5 and stored["service"]["ref_hr"] == 72.0
    assert stored["service"]["vascular_tone"]["pi_percent"] == 0.56
    assert not api.CLIPS_DIR.exists(), "trace retention must never write the clip store"


def test_only_the_newest_are_kept(monkeypatch):
    _enable(monkeypatch)
    for i in range(5):
        api._retain_traces(f"upl-{i}", DOC, RESULT)
        f = sorted(api.TRACES_DIR.iterdir(), key=lambda q: q.stat().st_mtime)[-1]
        f.touch()
    assert len(list(api.TRACES_DIR.iterdir())) == api.TRACES_KEEP


def test_read_back_needs_the_switch_and_the_right_token(monkeypatch):
    h = _handler("/api/traces?token=s3cret")
    h._serve_traces()
    assert h.sent[-1][0] == 404                             # off looks like not there
    _enable(monkeypatch)
    api._retain_traces("upl-1", DOC, RESULT)
    for bad in ("/api/traces", "/api/traces?token=nope", "/api/traces?token=%C3%A9"):
        h = _handler(bad)
        h._serve_traces()
        assert h.sent[-1][0] == 404, bad
    h = _handler("/api/traces?token=s3cret")
    h._serve_traces()
    code, doc = h.sent[-1]
    assert code == 200 and len(doc["traces"]) == 1 and doc["traces"][0]["id"].endswith(".traces.json")


def test_a_trace_id_cannot_escape_the_traces_directory(monkeypatch, tmp_path):
    _enable(monkeypatch)
    api._retain_traces("upl-1", DOC, RESULT)
    (tmp_path / "secret.traces.json").write_text("{}")
    for tid in ("../secret.traces.json", "/etc/passwd", "x.webm"):
        h = _handler(f"/api/trace?token=s3cret&id={tid}")
        h._serve_traces()
        assert h.sent and h.sent[-1][0] == 404, tid


def test_a_failure_never_breaks_a_scan(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(api, "TRACES_DIR", pathlib.Path("/dev/null/not-a-dir"), raising=False)
    api._retain_traces("upl-1", DOC, RESULT)                # must not raise
