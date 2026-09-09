"""measure_video mutates its input in place; a recording under the repo's
data/ directory must be copied first (2026-09-09: a diagnostic destroyed
eight corpus recordings by calling it on the corpus paths)."""
import os
import pathlib

os.environ.setdefault("AFIB_SHEET_ID", "test-sheet")

from app.measure_api import _protect_input  # noqa: E402


def test_input_under_data_is_copied_with_its_sidecar(tmp_path):
    root = tmp_path / "data"
    src = root / "eval_corpus" / "rec.webm"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"video-bytes")
    (str(src) + ".timestamps.json") and pathlib.Path(str(src) + ".timestamps.json").write_text("{}")
    out = _protect_input(str(src), protected_root=root)
    assert out != str(src)
    assert pathlib.Path(out).read_bytes() == b"video-bytes"
    assert pathlib.Path(out + ".timestamps.json").exists()
    assert src.read_bytes() == b"video-bytes"          # untouched


def test_input_elsewhere_is_measured_in_place(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    upload = tmp_path / "upload" / "scan.webm"
    upload.parent.mkdir()
    upload.write_bytes(b"x")
    assert _protect_input(str(upload), protected_root=root) == str(upload)


def test_missing_path_passes_through(tmp_path):
    assert _protect_input(str(tmp_path / "nope.webm"), protected_root=tmp_path / "data") == str(tmp_path / "nope.webm")
