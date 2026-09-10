"""Pull retained scan clips off the staging service into the eval corpus.

The point: a candidate change currently costs a 3-minute phone scan to judge,
and the corpus it COULD be replayed against holds no 480x720 portrait clip —
the only shape production records. That is why the optimizer gate had nothing
to say about the 2026-09-10 downscale change. Pull a few real captures in and
every future change gets measured in seconds instead of scans.

  export AFIB_CLIPS_TOKEN=<the value set in Railway>
  python scripts/pull_scan_clips.py                 # list what is on staging
  python scripts/pull_scan_clips.py --get all       # download into the corpus

Requires AFIB_KEEP_UPLOADS=1 and AFIB_CLIPS_TOKEN set on the staging service;
neither is set on production, and the endpoint 404s without both.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request

STAGING = "https://avatarx-cardio-staging.up.railway.app"
CORPUS = pathlib.Path(__file__).resolve().parents[1] / "data" / "eval_corpus"


def _call(base: str, path: str, token: str, params: dict | None = None):
    q = dict(params or {})
    q["token"] = token
    url = f"{base}{path}?{urllib.parse.urlencode(q)}"
    with urllib.request.urlopen(url, timeout=180) as r:
        return r.read(), r.headers.get("Content-Type", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.environ.get("AFIB_CLIPS_BASE", STAGING))
    ap.add_argument("--get", metavar="ID", help="clip id, or 'all'")
    ap.add_argument("--out", default=str(CORPUS),
                    help=f"destination directory (default: {CORPUS})")
    a = ap.parse_args(argv)

    token = os.environ.get("AFIB_CLIPS_TOKEN", "").strip()
    if not token:
        print("AFIB_CLIPS_TOKEN is not set — it must match the value in Railway.",
              file=sys.stderr)
        return 2

    try:
        body, _ = _call(a.base, "/api/clips", token)
    except Exception as e:                                    # noqa: BLE001
        print(f"could not list clips: {e}\n"
              f"  a 404 means the service has AFIB_KEEP_UPLOADS or "
              f"AFIB_CLIPS_TOKEN unset, or the token does not match.",
              file=sys.stderr)
        return 1
    clips = json.loads(body).get("clips", [])
    if not clips:
        print("no clips retained yet — run a scan on /beta/cardio-staging first")
        return 0

    print(f"{len(clips)} clip(s) on {a.base}:")
    for c in clips:
        print(f"   {c['id']:44} {c['bytes']/1e6:7.1f} MB  {c['mtime']}"
              f"{'  +timestamps' if c.get('sidecar') else ''}")
    if not a.get:
        print("\nre-run with --get all (or --get <id>) to download")
        return 0

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = clips if a.get == "all" else [c for c in clips if c["id"] == a.get]
    if not wanted:
        print(f"no clip with id {a.get!r}", file=sys.stderr)
        return 1
    for c in wanted:
        for name in ([c["id"]] + ([c["id"] + ".timestamps.json"] if c.get("sidecar") else [])):
            dst = out / name
            if dst.exists():
                print(f"   have {name}")
                continue
            data, _ = _call(a.base, "/api/clip", token, {"id": name})
            dst.write_bytes(data)
            print(f"   saved {name}  ({len(data)/1e6:.1f} MB)")
    print(f"\nin {out}. Next:")
    print("   python .claude/skills/cardio-pipeline-optimizer/scripts/build_eval_set.py "
          "--holdout-frac 0.4 --seed 13")
    print("   ...then replay/score as usual — the corpus now carries the real "
          "portrait capture shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
