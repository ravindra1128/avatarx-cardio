"""Route 6 (2026-09-11): ShenAI posts its own dense PPG waveform and beat train
as a sidecar, retained beside the clip for OFFLINE comparison against our
four-ROI POS lattice. The sheet carries four audit cells so a comparison run can
be planned from the history — which scans have a second opinion, and how big it
is. These tests pin that the cells exist, that they read both shapes the doc can
carry, and that a scan with no sidecar still writes a row."""
import math
import os

os.environ.setdefault("AFIB_SHEET_ID", "test-sheet")

from app import result_sheet  # noqa: E402

SHENAI_COLUMNS = ["ShenAI Sidecar", "ShenAI PPG N", "ShenAI PPG fs", "ShenAI Beats N"]


def _doc(**kw):
    # The minimal doc shape the other sheet tests use: row_from_doc must survive
    # every optional sub-dict being absent.
    return {"outcome": "NO_RESULT", "biomarkers": {"items": []}, **kw}


def test_the_four_columns_exist():
    for name in SHENAI_COLUMNS:
        assert name in result_sheet.COLUMNS
    # one cell per name, no stray key: the same 1:1 invariant
    # tests/test_client_capture.py pins for the whole row.
    row = result_sheet.row_from_doc(_doc(), extra={})
    assert all(k in result_sheet.COLUMNS for k in row)
    assert all(name in row for name in SHENAI_COLUMNS)


def test_summary_shape_is_read():
    """The compact summary the HTTP layer builds from the posted sidecar."""
    row = result_sheet.row_from_doc(
        _doc(shenai={"ppg_n": 1483, "ppg_fs_hz": 29.7, "beats_n": 57}), extra={})
    assert row["ShenAI Sidecar"] == "TRUE"
    assert row["ShenAI PPG N"] == 1483
    assert row["ShenAI PPG fs"] == 29.7
    assert row["ShenAI Beats N"] == 57


def test_sidecar_document_shape_is_read():
    """The schema_version 1 document itself, should the doc carry it verbatim:
    ppg.n / ppg.fs_hz, and the beat count taken from heartbeats[]."""
    row = result_sheet.row_from_doc(
        _doc(shenai={"schema_version": 1, "sdk": "shenai",
                     "ppg": {"n": 1483, "fs_hz": 29.684, "fs_source": "derived_from_beats"},
                     "heartbeats": [{"start_location_sec": 1.0, "end_location_sec": 1.8,
                                     "duration_ms": 800},
                                    {"start_location_sec": 1.8, "end_location_sec": 2.65,
                                     "duration_ms": 850}]}),
        extra={})
    assert row["ShenAI Sidecar"] == "TRUE"
    assert row["ShenAI PPG N"] == 1483
    assert row["ShenAI PPG fs"] == 29.68          # _num(..., 2)
    assert row["ShenAI Beats N"] == 2


def test_absent_sidecar_is_blank_not_an_error():
    row = result_sheet.row_from_doc(_doc(), extra={})
    for name in SHENAI_COLUMNS:
        assert row[name] == "", name


def test_reported_absent_reads_false_while_unknown_stays_blank():
    """FALSE ("this scan sent nothing") and blank ("this build says nothing")
    are different facts: a run of blanks means the service predates route 6,
    a run of FALSEs means the phone side is not snapshotting."""
    assert result_sheet.row_from_doc(_doc(shenai=False), extra={})["ShenAI Sidecar"] == "FALSE"
    assert result_sheet.row_from_doc(_doc(shenai={}), extra={})["ShenAI Sidecar"] == "FALSE"
    assert result_sheet.row_from_doc(_doc(), extra={})["ShenAI Sidecar"] == ""


def test_bare_arrival_flag_does_not_break_the_row():
    """A build that only records arrival writes a bool, not a dict."""
    row = result_sheet.row_from_doc(_doc(shenai=True), extra={})
    assert row["ShenAI Sidecar"] == "TRUE"
    assert row["ShenAI PPG N"] == "" and row["ShenAI Beats N"] == ""


def test_zero_counts_are_recorded_as_zero():
    """An empty waveform is evidence, not a missing field — `or` would erase it,
    which is why the cells go through _first rather than the Downscale Note
    fallback chain."""
    row = result_sheet.row_from_doc(
        _doc(shenai={"ppg_n": 0, "beats_n": 0, "ppg_fs_hz": None}), extra={})
    assert row["ShenAI PPG N"] == 0 and row["ShenAI Beats N"] == 0
    assert row["ShenAI PPG fs"] == ""


def test_garbage_values_do_not_raise():
    """The sheet never breaks a result: a malformed sidecar writes blanks."""
    row = result_sheet.row_from_doc(
        _doc(shenai={"ppg": "not-a-dict", "heartbeats": "not-a-list",
                     "ppg_fs_hz": "nonsense"}), extra={})
    assert row["ShenAI Sidecar"] == "TRUE"
    assert row["ShenAI PPG N"] == "" and row["ShenAI Beats N"] == ""
    assert row["ShenAI PPG fs"] == ""


def test_num_blanks_every_non_finite_value():
    """`f == f` only rejected NaN; ±inf went into the cell verbatim. Reachable:
    the sidecar body is unauthenticated and unvalidated, and json.loads accepts
    Infinity/-Infinity/NaN, as does float() on those strings."""
    for bad in (float("inf"), float("-inf"), float("nan"),
                "Infinity", "-Infinity", "nan"):
        assert result_sheet._num(bad) == "", bad
        assert result_sheet._num(bad, 0) == "", bad
    # the guard must not cost the finite values their rounding
    assert result_sheet._num(29.684, 2) == 29.68
    assert result_sheet._num(0) == 0


def test_non_finite_shenai_fields_never_reach_a_cell():
    """A posted Infinity must not raise and must not be written: gspread turns
    it into bare `Infinity`, which the Sheets API rejects as invalid JSON, so
    one such value drops the entire row after the three retries."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        row = result_sheet.row_from_doc(
            _doc(shenai={"ppg_n": bad, "ppg_fs_hz": bad, "beats_n": bad}), extra={})
        assert row["ShenAI Sidecar"] == "TRUE", bad
        for name in ("ShenAI PPG N", "ShenAI PPG fs", "ShenAI Beats N"):
            assert row[name] == "", (name, bad)
    # and the same through the sidecar-document shape, where fs is nested
    row = result_sheet.row_from_doc(
        _doc(shenai={"schema_version": 1,
                     "ppg": {"n": float("inf"), "fs_hz": float("inf")},
                     "heartbeats": [{"duration_ms": 800}]}), extra={})
    assert row["ShenAI PPG N"] == "" and row["ShenAI PPG fs"] == ""
    assert row["ShenAI Beats N"] == 1
    # nothing anywhere in the row is a non-finite float, whatever the doc says
    for name, cell in row.items():
        assert not (isinstance(cell, float) and not math.isfinite(cell)), name
