"""Isolated delivery/confidence probes; no network, real jobs, or sheet writes.

Uses the existing ShenAI-route test fixtures as constructed inputs. This does
not measure performance on AF patients. All service mutations are mocked or
scoped to this short-lived process and its temporary directory.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import runpy
import sys
import tempfile
import threading
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from app import measure_api as api
    from inference.confidence_stars import confidence_stars, _DEFAULT_WEIGHTS
    from inference import shenai_route

    with tempfile.TemporaryDirectory(prefix="afib-delivery-probe-") as temp:
        with patch.object(api, "UPLOAD_DIR", Path(temp)), \
             patch.object(api, "_POOL") as pool, \
             patch.object(api, "_INFLIGHT", threading.BoundedSemaphore(2)), \
             patch.object(api, "_STARTED", {}), \
             patch.object(api, "_retain_clip", return_value=None):
            directory = api._part_dir("probe-lost-start-response")
            directory.mkdir()
            (directory / "part-0000").write_bytes(b"fake-video-not-analysed")
            (directory / "total").write_text("1")
            handler = api.MeasureHandler.__new__(api.MeasureHandler)
            handler.headers = {}
            handler._json = Mock()
            query = {"upload_id": [directory.name], "ext": ["webm"]}
            handler._start_job(query)
            first = handler._json.call_args.args
            # Simulate a lost 202: the client retries the same start request.
            handler._start_job(query)
            second = handler._json.call_args.args
            starts = {"first_http": first[0], "retry_http": second[0],
                      "retry_body": second[1], "submitted_jobs": pool.submit.call_count}

    fixtures = runpy.run_path(str(ROOT / "tests/test_shenai_route.py"))
    readiness = {"blocking_pass": True, "checks": {
        k: {"value": 1, "threshold": 1, "op": ">=", "pass": True}
        for k in _DEFAULT_WEIGHTS}}
    routes = []
    for kind in ("regular", "afib"):
        doc, details = fixtures["_video"](n_int=8, coherence=.4, tp=20, matched=.8)
        conf = confidence_stars(details["evidence"], readiness, fixtures["CFG"], final=True)
        details["rationale"]["confidence"] = dataclasses.asdict(conf)
        doc["confidence_stars"] = conf.stars
        raw = fixtures["_sidecar"](fixtures[f"_{kind}"]())
        route = shenai_route.evaluate(doc, details, raw)
        routes.append({"constructed_train": kind, "video_intervals": 8,
                       "recomputed_video_stars": conf.stars, "route_used": route["used"],
                       "outcome": doc["outcome"], "class": doc["predicted_class"],
                       "reasons": doc["no_read_reasons"]})
    report = {"notice": "Software counterexamples, not clinical validation.",
              "lost_start_acknowledgement": starts, "inherited_confidence": routes}
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/audit_afib_contracts.py output.json")
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
