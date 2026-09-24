"""Download the live-frame trace documents a service retained (AFIB_KEEP_TRACES).

Each file is one scan: {"service": {upload_id, sdk_hr_bpm, ref_hr, outcome,
vascular_tone, ...}, "trace_document": <what the phone posted>}. No video.

Usage:
  AFIB_CLIPS_TOKEN=... python scripts/pull_scan_traces.py               # list
  AFIB_CLIPS_TOKEN=... python scripts/pull_scan_traces.py --get all     # download
  --base https://avatarx-cardio-staging.up.railway.app (default)
  --out data/eval_cache/vascular_tone_live/staging_phone (default)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request

DEFAULT_BASE = "https://avatarx-cardio-staging.up.railway.app"
DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "eval_cache" / "vascular_tone_live" / "staging_phone"


def _get(base: str, path: str, params: dict) -> bytes:
    url = f"{base.rstrip('/')}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--token", default=os.environ.get("AFIB_CLIPS_TOKEN", ""))
    ap.add_argument("--get", default=None, help="'all' or one id")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    if not a.token:
        print("set AFIB_CLIPS_TOKEN (or --token)", file=sys.stderr)
        return 2
    try:
        listing = json.loads(_get(a.base, "/api/traces", {"token": a.token}))
    except Exception as e:                                   # noqa: BLE001
        print(f"listing failed ({e}): trace retention off, or the token is wrong", file=sys.stderr)
        return 1
    items = listing.get("traces") or []
    print(f"{len(items)} trace document(s) on {a.base} (keeps newest {listing.get('keep')}):")
    for t in items:
        print(f"   {t['id']:60} {t['bytes'] / 1e3:7.0f} kB  {t['mtime']}")
    if not a.get:
        return 0
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = items if a.get == "all" else [t for t in items if t["id"] == a.get]
    for t in wanted:
        dst = out / t["id"]
        if dst.exists():
            print(f"   have {t['id']}")
            continue
        dst.write_bytes(_get(a.base, "/api/trace", {"token": a.token, "id": t["id"]}))
        print(f"   got  {t['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
