"""Read-only diagnostic replay of paired facial video / ShenAI captures.

This is an audit, not a training, tuning, or clinical-validation harness. Inputs
are hashed and copied before measure_video_details (which rewrites its input).
The SDK snapshot is supplied deterministically, so network delivery is NOT tested.
No server, sheet append, deployment, or classifier modification is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def probes():
    """Counterexamples to implementation assumptions, NOT physiological cases."""
    import numpy as np
    from configs import load_config
    from features.rhythm import RhythmFeatures
    from inference.decision_logic import decide_with_rationale, timing_precision_from_trains
    from inference.shenai_route import train_from_sidecar, train_checks

    features = RhythmFeatures(
        values={"median_abs_succ_diff": 65.0, "pnn50": 0.5,
                "median_ibi": 900.0, "dropout_rate": 0.0, "n_intervals": 40},
        n_intervals=40, mean_confidence=0.9, estimator_warnings=[])
    evidence = {"cross_roi_coherence": 0.6, "timing_matched_fraction": 0.9,
                "split_fraction": 0.0, "harmonic_fraction": 0.0,
                "mean_roi_agreement": 0.5, "pulse_lattice_bpm": 66.6667,
                "pulse_spectral_bpm": 66.6667, "pulse_spectral_roi_agree": 3,
                "pulse_spectral_snr": 3.0}
    decisions = []
    for precision in (5.0, 20.0, 30.0, 40.0):
        result, rationale = decide_with_rationale(
            features, 0.5, 0.9, load_config(), recording_id="audit-feature-vector",
            evidence={**evidence, "timing_precision_ms": precision})
        decisions.append({"timing_precision_ms": precision,
                          "outcome": result.outcome.value,
                          "predicted_class": result.predicted_class,
                          "rule": rationale["rule"]})
    # Two detections in A reuse one B beat. A matching fraction must not exceed 1.
    matching = timing_precision_from_trains({
        "a": np.array([0, .05, 1, 2, 3, 4, 5], float),
        "b": np.array([0, 1, 2, 3, 4, 5], float)})
    raw = {"heartbeats": [{"start_location_sec": i * .8,
                            "end_location_sec": (i + 1) * .8,
                            "duration_ms": 800} for i in range(40)]}
    missing_quality = train_from_sidecar(raw)
    return {"notice": "Constructed software counterexamples; no AF sensitivity/specificity claim.",
            "identical_features_with_worsening_timing": decisions,
            "many_to_one_timing_match": matching,
            "missing_sdk_quality_checks": train_checks(missing_quality, 0.0)}


def decision_view(doc):
    debug = doc.get("debug") or {}
    ra, ev = debug.get("rationale") or {}, debug.get("evidence") or {}
    return {"outcome": doc.get("outcome"), "class": doc.get("predicted_class"),
            "source": doc.get("rhythm_source", "video"),
            "reasons": doc.get("no_read_reasons"), "stars": doc.get("confidence_stars"),
            "pulse_bpm": doc.get("mean_pulse_rate_bpm"),
            "gates_failed": ra.get("gates_failed"), "coverage": ra.get("coverage"),
            "n_intervals": ev.get("n_intervals"), "sqi": doc.get("signal_quality_index"),
            "coherence": ev.get("cross_roi_coherence"),
            "timing_precision_ms": ev.get("timing_precision_ms"),
            "features": ra.get("features"), "rule": ra.get("rule")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--window-s", type=float, default=70.0)
    parser.add_argument("--scale", default="640x480")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    # Explicitly reproduce the documented launch override; do not claim it
    # reflects a deployment whose environment has not been verified.
    os.environ["AFIB_MAX_COLLAPSED_FRACTION"] = "0.05"
    from app.measure_api import measure_video_details, _json_bytes
    from inference.shenai_route import evaluate

    report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=ROOT, text=True).strip(),
              "settings": {"window_s": args.window_s, "scale": args.scale,
                           "capture_profile": "consumer", "repeat": args.repeat,
                           "max_collapsed_fraction": 0.05,
                           "sidecar_delivery": "supplied locally before evaluation"},
              "probes": probes(), "records": [],
              "limitations": ["No simultaneous ECG labels or verified participant identifiers.",
                              "Paired sidecar replay is outside the legacy optimizer gate.",
                              "Repeat equality is determinism, not human test-retest stability."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_bytes(_json_bytes(report))

    save()
    sidecars = sorted(args.corpus.glob("*.shenai.json")) if args.corpus else []
    for sidecar in sidecars:
        video = Path(str(sidecar).removesuffix(".shenai.json"))
        if not video.is_file():
            continue
        sources = [video, sidecar]
        clock = Path(str(video) + ".timestamps.json")
        if clock.is_file():
            sources.append(clock)
        before = {p.name: digest(p) for p in sources}
        raw = json.loads(sidecar.read_text())
        record = {"clip": video.name, "sha256": before, "runs": []}
        for iteration in range(args.repeat):
            print(f"AUDIT {video.name} repeat {iteration + 1}/{args.repeat}", flush=True)
            started = time.perf_counter()
            try:
                with tempfile.TemporaryDirectory(prefix="afib-phase1-audit-") as temp:
                    copied = Path(temp) / video.name
                    shutil.copyfile(video, copied)
                    if clock.is_file():
                        shutil.copyfile(clock, str(copied) + ".timestamps.json")
                    doc, details = measure_video_details(
                        str(copied), manifest={"capture_profile": "consumer"},
                        window_s=args.window_s, scale=args.scale)
                    video_view = decision_view(doc)
                    route = evaluate(doc, details, raw)
                    record["runs"].append({"returned": True, "video": video_view,
                                           "final": decision_view(doc), "route": route,
                                           "head_results": doc.get("head_results"),
                                           "clock": doc.get("clock"), "trim": doc.get("trim"),
                                           "timing": doc.get("timing"),
                                           "elapsed_s": round(time.perf_counter() - started, 3)})
            except Exception as exc:
                record["runs"].append({"returned": False,
                                       "error": f"{type(exc).__name__}: {exc}"})
        record["inputs_unchanged"] = all(digest(p) == before[p.name] for p in sources)
        if not record["inputs_unchanged"]:
            raise RuntimeError("An audit source changed; stop and investigate")
        record["decision_deterministic"] = (all(r["returned"] for r in record["runs"]) and
            all(r["video"] == record["runs"][0]["video"] and
                r["final"] == record["runs"][0]["final"] for r in record["runs"][1:])) \
            if args.repeat > 1 else None
        report["records"].append(record)
        for which in ("video", "final"):
            report[f"{which}_counts"] = dict(Counter(
                f"{r['runs'][0][which]['outcome']}/{r['runs'][0][which]['class']}"
                if r["runs"][0]["returned"] else "ERROR" for r in report["records"]))
        save()
        print(json.dumps({"completed": len(report["records"]),
                          "counts": report.get("final_counts")}), flush=True)
    print(f"Audit saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
