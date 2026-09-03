"""
Trend store (v0.4 T5): accepted sessions' recovery summaries, appended
locally, versioned, exportable. head_trend reads windows of it; nothing
else does. Future store versions fail closed on read.
"""
from __future__ import annotations

import json
import pathlib
import time

STORE_VERSION = 1


class TrendStoreError(ValueError):
    pass


class TrendStore:
    def __init__(self, path):
        self.path = pathlib.Path(path)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"store_version": STORE_VERSION, "sessions": []}
        with open(self.path) as f:
            doc = json.load(f)
        ver = int(doc.get("store_version", 1))
        if ver > STORE_VERSION:
            raise TrendStoreError(
                f"trend store version {ver} is newer than this build "
                f"(reads <= {STORE_VERSION})")
        doc.setdefault("sessions", [])
        return doc

    def append(self, *, session_id: str, when: str = None,
               protocol_id: str = None, hrr60_bpm: float = None,
               hr_rest_bpm: float = None, stars: int = None) -> dict:
        doc = self._load()
        row = {"session_id": session_id,
               "when": when or time.strftime("%Y-%m-%dT%H:%M:%S"),
               "protocol_id": protocol_id, "hrr60_bpm": hrr60_bpm,
               "hr_rest_bpm": hr_rest_bpm, "stars": stars}
        # one row per session: a re-run of the same session replaces its
        # row rather than double-counting it in the trend
        doc["sessions"] = [r for r in doc["sessions"]
                           if r.get("session_id") != session_id]
        doc["sessions"].append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(doc, f, indent=1)
        return row

    def sessions(self, *, protocol_id: str = None) -> list:
        rows = self._load()["sessions"]
        if protocol_id:
            rows = [r for r in rows if r.get("protocol_id") == protocol_id]
        return rows

    def export(self) -> str:
        """The user's data, portable: the raw JSON document."""
        return json.dumps(self._load(), indent=1)
