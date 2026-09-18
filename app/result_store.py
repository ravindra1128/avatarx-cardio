"""Short-lived completed-result recovery. No uploads or SDK sample inputs.

SQLite survives a Python-process restart on the same filesystem. A container
replacement requires AFIB_RESULT_DB on an attached persistent volume. This does
not resume interrupted inference or share a job queue between replicas.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import time


_RAW_KEYS = frozenset({"shenai_signals", "heartbeats", "ppg", "signal", "waveform",
                       "raw_signal", "timestamps_s", "rr_ms", "ibi_ms", "video_bytes"})


def _result_only(value):
    if isinstance(value, dict):
        return {k: _result_only(v) for k, v in value.items() if isinstance(k, str) and k not in _RAW_KEYS}
    if isinstance(value, (list, tuple)):
        return [_result_only(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


class ResultStore:
    def __init__(self, path, ttl_s=900, *, clock=time.time, max_bytes=64 * 1024 * 1024):
        self.path = Path(path)
        self.ttl_s = ttl_s
        self.clock = clock
        self.max_bytes = max_bytes
        self.last_error = None

    @contextmanager
    def _db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(self.path), timeout=1.0)
        try:
            self.path.chmod(0o600)
            db.execute("PRAGMA secure_delete=ON")
            page_size = db.execute("PRAGMA page_size").fetchone()[0]
            db.execute(f"PRAGMA max_page_count={max(16, self.max_bytes // page_size)}")
            db.execute("CREATE TABLE IF NOT EXISTS results (scan_id TEXT PRIMARY KEY, created REAL NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS results_created ON results(created)")
            with db:
                db.execute("DELETE FROM results WHERE created <= ?", (self.clock() - self.ttl_s,))
                yield db
        finally:
            db.close()

    def put(self, scan_id, doc, *, created=None):
        try:
            payload = json.dumps(_result_only(doc), allow_nan=False)
            with self._db() as db:
                db.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?)",
                           (str(scan_id), self.clock() if created is None else created, payload))
            self.last_error = None
            return True
        except (OSError, sqlite3.Error, TypeError, ValueError, RecursionError) as error:
            self.last_error = type(error).__name__
            return False

    def get(self, scan_id):
        try:
            with self._db() as db:
                row = db.execute("SELECT payload FROM results WHERE scan_id=?", (str(scan_id),)).fetchone()
            self.last_error = None
            return json.loads(row[0]) if row else None
        except (OSError, sqlite3.Error, ValueError) as error:
            self.last_error = type(error).__name__
            return None

    def purge(self):
        try:
            with self._db():
                pass
            self.last_error = None
        except (OSError, sqlite3.Error) as error:
            self.last_error = type(error).__name__

    def status(self):
        return {"backend": "sqlite", "ttl_s": self.ttl_s,
                "scope": "completed_results_on_this_filesystem", "last_error": self.last_error}
