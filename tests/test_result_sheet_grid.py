"""A fresh tab is 26 columns wide; the tracking sheet is 73.

This is the first thing a new deployment does — point AFIB_SHEET_GID at a tab
someone just created, then run a scan. The write fails soft (the queue never
raises), so the scan still returns its cards and the row is simply gone: the
one outcome that looks like success and isn't. Both branches of _worksheet()
must therefore widen the grid before writing, not just the one that was.
"""
import sys, types, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import app.result_sheet as rs


class FakeWorksheet:
    """Mimics the one gspread behaviour that matters: a write past the last
    column is rejected rather than silently growing the sheet."""

    def __init__(self, col_count=26, header=None):
        self.col_count = col_count
        self.id = 342232654
        self.title = "cardio-staging"
        self._header = list(header or [])
        self.appended = None
        self.added_cols = 0

    def row_values(self, _row):
        return list(self._header)

    def col_values(self, _col):
        return ["x"] if self._header else []

    def add_cols(self, n):
        self.col_count += n
        self.added_cols += n

    def append_row(self, values, value_input_option=None):
        if len(values) > self.col_count:
            raise RuntimeError(
                f"Range ('{self.title}'!A1) exceeds grid limits. "
                f"Max columns: {self.col_count}, requested: {len(values)}"
            )
        self._header = list(values)
        self.appended = list(values)

    def update(self, range_name=None, values=None, value_input_option=None):
        wanted = len(self._header) + len(values[0])
        if wanted > self.col_count:
            raise RuntimeError("exceeds grid limits")
        self._header += list(values[0])


def _open(monkeypatch, ws):
    """Run _worksheet() against a fake book, with the module cache cleared."""
    monkeypatch.setattr(rs, "_ws", None, raising=False)
    monkeypatch.setattr(rs, "_header", None, raising=False)
    book = types.SimpleNamespace(get_worksheet_by_id=lambda _gid: ws, sheet1=ws)
    monkeypatch.setattr(rs, "gspread", types.SimpleNamespace(
        authorize=lambda _c: types.SimpleNamespace(open_by_key=lambda _k: book)),
        raising=False)
    fake_creds = types.SimpleNamespace(
        from_service_account_info=lambda *a, **k: None,
        from_service_account_file=lambda *a, **k: None)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name, *a, **k):
        if name == "gspread":
            return rs.gspread
        if name == "google.oauth2.service_account":
            return types.SimpleNamespace(Credentials=fake_creds)
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr(rs, "CREDS_JSON", "", raising=False)
    monkeypatch.setattr(rs, "CREDS_FILE", "/dev/null", raising=False)
    monkeypatch.setattr(rs, "SHEET_GID", "342232654", raising=False)
    return rs._worksheet()


def test_fresh_26_column_tab_is_widened_before_the_header_is_written(monkeypatch):
    ws = FakeWorksheet(col_count=26, header=[])
    _, header = _open(monkeypatch, ws)
    assert ws.added_cols == len(rs.COLUMNS) - 26, "grid was not grown"
    assert ws.appended == list(rs.COLUMNS)
    assert header == list(rs.COLUMNS)


def test_a_tab_already_wide_enough_is_not_resized(monkeypatch):
    ws = FakeWorksheet(col_count=200, header=[])
    _open(monkeypatch, ws)
    assert ws.added_cols == 0


def test_existing_narrow_header_still_grows_for_new_columns(monkeypatch):
    # The branch that was already protected — pinned so a refactor of the new
    # one cannot quietly drop it.
    ws = FakeWorksheet(col_count=len(rs.COLUMNS) - 2, header=list(rs.COLUMNS[:-2]))
    _, header = _open(monkeypatch, ws)
    assert ws.added_cols >= 2
    assert header == list(rs.COLUMNS)
