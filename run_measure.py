"""One-command launcher for the batch measure API.

    python3 run_measure.py                 # 127.0.0.1:8790
    python3 run_measure.py --port 8800
    python3 run_measure.py --allow-origin http://localhost:5173

This is the service the AvatarX webapp posts a recorded scan to so the
rhythm pipeline can run in parallel with ShenAI. It runs THE production
path (inference.pipeline.run_with_details) — the same call `cli.py
process` makes — and adds nothing but transport and an honest frame clock.

Env overrides: AFIB_ALLOW_ORIGIN, AFIB_MAX_UPLOAD_MB, AFIB_MAX_CONCURRENT,
AFIB_WORK_DIR, AFIB_KEEP_UPLOADS=1 (keep uploads on disk for debugging).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    # Railway (and most PaaS hosts) assign the port at runtime via $PORT and
    # expect the process to bind 0.0.0.0, not localhost. Local dev has no
    # $PORT set, so it falls back to the previous 127.0.0.1:8790 default —
    # an explicit --host/--port flag still wins over both.
    default_host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    default_port = int(os.environ.get("PORT", "8790"))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=default_host)
    ap.add_argument("--port", type=int, default=default_port)
    ap.add_argument("--allow-origin", default=None,
                    help="CORS origin (default *, or AFIB_ALLOW_ORIGIN)")
    ap.add_argument("--keep-uploads", action="store_true",
                    help="don't delete uploaded clips after processing")
    args = ap.parse_args()

    if args.allow_origin:
        os.environ["AFIB_ALLOW_ORIGIN"] = args.allow_origin
    if args.keep_uploads:
        os.environ["AFIB_KEEP_UPLOADS"] = "1"

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("opencv (cv2) is required. Install it into THIS interpreter:\n"
              f"  {sys.executable} -m pip install opencv-python",
              file=sys.stderr)
        return 2

    from app.measure_api import serve
    serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
