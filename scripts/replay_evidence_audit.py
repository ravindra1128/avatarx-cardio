"""Check additive audits against a saved replay, always analysing temp copies.

Writes aggregate diagnostics and equality checks only, never raw beat/PPG arrays.
This is an integration regression check, not the signal optimizer or clinical eval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    os.environ["AFIB_KEEP_UPLOADS"] = "0"
    os.environ["AFIB_MAX_COLLAPSED_FRACTION"] = "0.05"
    from app.measure_api import measure_video_details, _json_bytes, _apply_shenai_route
    from app.scan_evidence import shenai_evidence_summary
    from scripts.audit_afib_phase1 import decision_view

    baseline = json.loads(args.baseline.read_text())["records"]
    records = []
    for base in baseline:
        video = args.videos / base["clip"]
        before_hash = hashlib.sha256(video.read_bytes()).hexdigest()
        side = video.with_name(video.name + ".shenai.json")
        raw = json.loads(side.read_text()) if side.exists() else None
        side_hash = hashlib.sha256(side.read_bytes()).hexdigest() if side.exists() else None
        previous = None
        for repeat in range(args.repeat):
            print(f"Replay {video.name} ({repeat + 1}/{args.repeat})", flush=True)
            with tempfile.TemporaryDirectory(prefix="codex-evidence-audit-") as temp:
                copied = Path(temp) / video.name
                shutil.copy2(video, copied)
                doc, det = measure_video_details(str(copied), manifest={"capture_profile": "consumer"},
                                                 window_s=70, scale="640x480")
                view = decision_view(doc)
                _apply_shenai_route(doc, det, video.stem, attachment={
                    "signal_delivery": {"version": 1, "state": "attached" if raw else "missing_snapshot"},
                    "shenai_signals": raw})
                rs, series = det.get("runset"), det.get("series")
                # Canonicalize legacy nonfinite values exactly as HTTP serialization does.
                current = json.loads(_json_bytes({
                    "video": view, "final": decision_view(doc), "route": doc["debug"]["shenai_route"],
                    "evidence": det.get("evidence"),
                    "runs": [r.tolist() for r in getattr(rs, "runs", [])],
                    "run_times": [r.tolist() for r in getattr(rs, "run_times", [])],
                    "series_beats": [vars(b) for b in getattr(series, "beats", [])],
                }))
                checks = {key: value == base[key] for key, value in current.items()}
                summaries = {key: doc["debug"][key] for key in ("video_duration", "shenai_assessment")}
                for value in summaries.values():
                    assert value["state"] != "assessment_failed", value
                assert all(checks.values()), checks
                assert previous is None or previous == summaries, "non-deterministic audit"
                previous = summaries
                t0 = time.perf_counter()
                for _ in range(20):
                    shenai_evidence_summary(raw, det)
                audit_ms = (time.perf_counter() - t0) * 1000 / 20
                records.append({"clip": video.name, "repeat": repeat + 1, "baseline_equal": checks,
                                "input_sha256": before_hash, "sdk_audit_mean_ms": round(audit_ms, 3),
                                **summaries})
            assert hashlib.sha256(video.read_bytes()).hexdigest() == before_hash
            assert not side.exists() or hashlib.sha256(side.read_bytes()).hexdigest() == side_hash
    report = {"kind": "integration_diagnostics_regression", "signal_candidate": False,
              "baseline": str(args.baseline), "retention_enabled": False,
              "original_inputs_unchanged": True, "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"PASS: {len(baseline)} recordings x {args.repeat} repeats; saved {args.out}")


if __name__ == "__main__":
    main()
