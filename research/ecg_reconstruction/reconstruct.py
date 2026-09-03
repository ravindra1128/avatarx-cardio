"""
`cli.py reconstruct` backend (v0.3) — one face scan through the
PRODUCTION measurement path, then the latest trained reconstruction
model. The output is a RESEARCH ARTIFACT, never a consumer output: it is
watermarked at every rendering site, carries the live §G gate scoreboard
in its footer, and is written only under research/runs/.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import numpy as np

from configs import load_config
from evaluation.inferred_ecg.decoder import FS, _znorm
from research.ecg_reconstruction import WATERMARK
from research.ecg_reconstruction.decoder import (ReconstructionError,
                                                 reconstruct)
from research.ecg_reconstruction.fidelity import _strip_svg, interval_mae
from research.ecg_reconstruction.gates import (DEFAULT_RUNS, gate_status,
                                               latest_scoreboard_entry)


def latest_model(runs_root=None) -> tuple:
    """(artifact, run_id) of the most recent training run, fail-closed."""
    entry = latest_scoreboard_entry(runs_root)
    if entry is None:
        raise ReconstructionError(
            "no trained reconstruction model on record — run:\n"
            "  python3 cli.py train configs/train_reconstruction.yaml")
    p = pathlib.Path(runs_root or DEFAULT_RUNS) / entry["run_id"] \
        / "model.json"
    if not p.exists():
        raise ReconstructionError(f"model artifact missing for run "
                                  f"{entry['run_id']} ({p})")
    with open(p) as f:
        return json.load(f), entry["run_id"]


def reconstruct_video(video_path, manifest: dict = None, *,
                      runs_root=None, out_root=None) -> dict:
    from inference.pipeline import run_with_details
    from rppg.pos import pos_pulse

    artifact, model_run = latest_model(runs_root)
    result, det = run_with_details(str(video_path), manifest=manifest)
    ing = det.get("ingest")
    if ing is None or not getattr(ing, "ok", False):
        raise ReconstructionError(
            "production ingest failed on this video — the reconstruction "
            "track consumes the measured path, it never bypasses it")
    cfg = load_config()
    fps = float(ing.meta.measured_fps_mean)
    ts = np.asarray(ing.timestamps_s, float)
    wave = pos_pulse(ing.traces["forehead"], fps,
                     band=tuple(cfg["sqi"]["band_hz"]))
    grid = np.arange(float(ts[0]), float(ts[-1]), 1.0 / FS)
    ppg = _znorm(np.interp(grid, ts, wave))
    gen = reconstruct(artifact, ppg)

    # paired reference (if this scan has an ingested session next to its
    # manifest video) -> fidelity numbers; otherwise honest null
    fidelity = None
    ref_dir = (manifest or {}).get("reference_dir")
    if ref_dir and (pathlib.Path(ref_dir) / "ecg_export.csv").exists():
        from datasets.reference import read_ecg_csv
        ref_p = pathlib.Path(ref_dir) / "reference.json"
        if ref_p.exists():
            with open(ref_p) as f:
                ref = json.load(f)
            t_e, mv = read_ecg_csv(pathlib.Path(ref_dir) / "ecg_export.csv")
            drift = float(ref["sync"]["drift_ppm"] or 0.0)
            t_ev = t_e * (1.0 + drift * 1e-6) \
                + float(ref["sync"]["offset_ms"]) / 1000.0
            ecg = _znorm(np.interp(grid, t_ev, mv))
            fidelity = interval_mae(ecg, gen, FS)

    gates = gate_status(runs_root=runs_root)
    rid = hashlib.sha256(
        (str(video_path) + model_run).encode()).hexdigest()[:12]
    out = pathlib.Path(out_root or DEFAULT_RUNS) / f"reconstruct-{rid}"
    out.mkdir(parents=True, exist_ok=True)

    doc = {"WATERMARK": WATERMARK,
           "purpose": "research artifact — the §G gates decide if this "
                      "class of output ever becomes user-visible; while "
                      "any gate is red it may not be rendered on any "
                      "user-facing surface",
           "video": str(video_path), "model_run": model_run,
           "architecture": artifact.get("architecture"),
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "scan_outcome": result.outcome.value,
           "fs_hz": FS, "n_samples": int(gen.size),
           "fidelity_vs_reference": fidelity or
           {"note": "no paired reference ECG for this scan"},
           "gate_status": {"promotion": gates["promotion"],
                           "gates": [{"gate": g["gate"],
                                      "status": g["status"]}
                                     for g in gates["gates"]]},
           "synthetic_ecg_estimate": [round(float(v), 4) for v in gen]}
    with open(out / "reconstruction.json", "w") as f:
        json.dump(doc, f, indent=1)
    _strip_svg(gen, out / "strip.svg", FS)
    footer = " · ".join(f"{g['gate'].upper()}:{g['status']}"
                        for g in gates["gates"])
    (out / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<div style='background:#b45309;color:#fff;padding:10px;"
        f"font:bold 14px sans-serif'>{WATERMARK}</div>"
        + (out / "strip.svg").read_text()
        + f"<pre style='font:12px monospace'>fidelity: "
        f"{json.dumps(doc['fidelity_vs_reference'], indent=1)}</pre>"
        f"<div style='font:bold 12px sans-serif;padding:8px'>"
        f"§G promotion: {gates['promotion']} — {footer}</div>"
        f"<div style='background:#b45309;color:#fff;padding:10px;"
        f"font:bold 14px sans-serif'>{WATERMARK}</div>")
    return {"out_dir": str(out), "model_run": model_run,
            "scan_outcome": result.outcome.value,
            "fidelity_vs_reference": doc["fidelity_vs_reference"],
            "promotion": gates["promotion"]}
