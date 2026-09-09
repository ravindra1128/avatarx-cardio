"""Capture state from the client (step 1 of the repeatability plan) travels
from the upload's query string into the response and the tracking sheet."""
import os

os.environ.setdefault("AFIB_SHEET_ID", "test-sheet")

from app import measure_api, result_sheet  # noqa: E402


def _q(**kw):
    return {k: [v] for k, v in kw.items()}


def test_capture_flags_and_note_are_parsed():
    _, header = measure_api._parse_envelope(
        b"", "video/webm",
        _q(exposure_locked="1", awb_locked="0", client_fps="29.9", face_luma="121",
           capture_note="  fps delivered; iso 566 -> luma 90; ae locked (none) luma 127 -> 123  "))
    cap = header["client_capture"]
    assert cap["exposure_locked"] is True and cap["awb_locked"] is False
    assert cap["client_fps"] == 29.9 and cap["face_luma"] == 121.0
    assert cap["note"] == "fps delivered; iso 566 -> luma 90; ae locked (none) luma 127 -> 123"
    # the lock flags also reach the manifest the pipeline reads
    assert header["manifest"]["exposure_locked"] is True
    assert header["manifest"]["awb_locked"] is False


def test_capture_note_is_bounded_and_optional():
    _, header = measure_api._parse_envelope(
        b"", "video/webm", _q(capture_note="x" * 1000, client_fps="30"))
    assert len(header["client_capture"]["note"]) == 300
    _, header = measure_api._parse_envelope(b"", "video/webm", _q(capture_note="   "))
    assert "client_capture" not in header


def test_sheet_row_carries_the_note():
    doc = {"outcome": "NO_RESULT", "biomarkers": {"items": []},
           "client_capture": {"exposure_locked": False, "awb_locked": True,
                              "client_fps": 26.7, "face_luma": 127,
                              "note": "ae reverted: luma 127 -> 31, mode none"}}
    row = result_sheet.row_from_doc(doc, extra={})
    assert row["AE Locked"] == "FALSE" and row["AWB Locked"] == "TRUE"
    assert row["Capture Note"] == "ae reverted: luma 127 -> 31, mode none"
    assert all(k in result_sheet.COLUMNS for k in row)
    assert "Capture Note" in result_sheet.COLUMNS
