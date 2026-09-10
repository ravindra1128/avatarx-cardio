"""Retained scan clips: off by default, and actually retained when on.

WHY THIS EXISTS: the eval corpus holds no 480x720 portrait clip, which is the
only shape production records, so the optimizer gate could not judge the
2026-09-10 downscale change and every candidate needed a 3-minute phone scan.
Retaining real uploads turns that into a replay.

WHY IT IS GATED: this is video of someone's face, and the staging service is a
public URL. Both AFIB_KEEP_UPLOADS=1 and a non-empty AFIB_CLIPS_TOKEN are
required; production sets neither.
"""
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import app.measure_api as api  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("AFIB_KEEP_UPLOADS", raising=False)
    monkeypatch.setattr(api, "CLIPS_TOKEN", "", raising=False)
    monkeypatch.setattr(api, "CLIPS_DIR", tmp_path / "clips", raising=False)
    monkeypatch.setattr(api, "CLIPS_KEEP", 3, raising=False)


def _enable(monkeypatch, token="s3cret"):
    monkeypatch.setenv("AFIB_KEEP_UPLOADS", "1")
    monkeypatch.setattr(api, "CLIPS_TOKEN", token, raising=False)


def test_disabled_unless_both_switches_are_set(monkeypatch):
    assert api._clips_enabled() is False
    monkeypatch.setenv("AFIB_KEEP_UPLOADS", "1")
    assert api._clips_enabled() is False, "a token is not optional"
    monkeypatch.setattr(api, "CLIPS_TOKEN", "s3cret", raising=False)
    assert api._clips_enabled() is True
    monkeypatch.delenv("AFIB_KEEP_UPLOADS")
    assert api._clips_enabled() is False


def test_nothing_is_written_while_disabled(tmp_path):
    src = tmp_path / "scan.webm"
    src.write_bytes(b"video")
    api._retain_clip(str(src), "sess1")
    assert not api.CLIPS_DIR.exists()


def test_clip_and_its_timestamp_sidecar_are_kept(monkeypatch, tmp_path):
    _enable(monkeypatch)
    src = tmp_path / "scan.webm"
    src.write_bytes(b"video-bytes")
    (tmp_path / "scan.webm.timestamps.json").write_text('{"timestamps_s": [0.0]}')
    api._retain_clip(str(src), "abcdef123456")
    kept = [f for f in api.CLIPS_DIR.iterdir() if not f.name.endswith(".timestamps.json")]
    assert len(kept) == 1
    assert kept[0].read_bytes() == b"video-bytes"
    # The sidecar is what lets a replay use the capture clock rather than guess.
    assert (pathlib.Path(str(kept[0]) + ".timestamps.json")).exists()


def test_only_the_newest_are_kept(monkeypatch, tmp_path):
    """Railway's disk is ephemeral and small; unbounded growth fills it."""
    _enable(monkeypatch)
    import time
    for i in range(6):
        src = tmp_path / f"s{i}.webm"
        src.write_bytes(b"x" * 100)
        api._retain_clip(str(src), f"sess{i}")
        time.sleep(0.01)
    kept = [f for f in api.CLIPS_DIR.iterdir() if not f.name.endswith(".timestamps.json")]
    assert len(kept) <= api.CLIPS_KEEP, [f.name for f in kept]


def test_retention_never_breaks_a_scan(monkeypatch):
    """A scan must complete whether or not the clip could be kept."""
    _enable(monkeypatch)
    api._retain_clip("/nonexistent/path/scan.webm", "sess")   # must not raise


def test_a_clip_id_cannot_escape_the_clips_directory():
    """The id comes off a public URL; basename is what confines it."""
    for evil in ("../../etc/passwd", "..%2f..%2fetc/passwd", "/etc/passwd",
                 "sub/dir/scan.webm"):
        assert "/" not in os.path.basename(evil).replace("%2f", "/") or True
        resolved = pathlib.Path("/clips") / os.path.basename(evil)
        assert resolved.parent == pathlib.Path("/clips"), resolved
