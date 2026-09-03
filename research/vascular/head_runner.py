"""Research-surface glue for head_vascular (v0.4 vascular, T3).

The head itself (heads/head_vascular.py) is quarantine-clean and inert
everywhere the production pipeline can reach it. THIS module is the one
sanctioned way to get a real research estimate: it computes morphology
features from an already-completed production run, loads the latest
evaluation run's model artifact and the V0-surviving set, hands all of
it to the head via ctx, and writes a WATERMARKED research report under
the vascular runs root. Nothing returned here may be merged into a
ScanResult or any consumer artifact (V-a).
"""
from __future__ import annotations

import json
import pathlib
import time

from research.vascular import WATERMARK
from research.vascular.features import features_from_details


def _latest_model_artifact(runs_root=None) -> dict:
    from evaluation.vascular_gates import (latest_scoreboard_entry,
                                           DEFAULT_RUNS)
    entry = latest_scoreboard_entry("evaluation", runs_root=runs_root)
    if not entry:
        return {}
    p = pathlib.Path(runs_root or DEFAULT_RUNS) / entry["run_id"] \
        / "model.json"
    if not p.exists():
        return {}
    try:
        art = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(art, dict) or not art.get("coef"):
        return {}
    n = len(art.get("features") or [])
    if not (n == len(art.get("mu") or []) == len(art.get("sd") or [])
            == len(art.get("coef") or []) > 0):
        return {}                      # incoherent artifact: fail closed
    return art


def run_research_head(result, det, *, runs_root=None,
                      gates_path=None) -> dict:
    """Run head_vascular on a completed production run; write the
    watermarked research report; return {"head_result", "report_path"}."""
    from evaluation.vascular_gates import (surviving_features,
                                           vascular_gate_status,
                                           DEFAULT_RUNS)
    from heads import get_head

    feats = features_from_details(result, det)
    ctx = {"vascular_research_surface": True,
           "vascular_features": feats,
           "vascular_model": _latest_model_artifact(runs_root),
           "surviving_features": surviving_features(
               runs_root=runs_root, gates_path=gates_path)}
    hr = get_head("vascular").run(det.get("lattice"), ctx)
    try:
        gates = vascular_gate_status(runs_root=runs_root,
                                     gates_path=gates_path)
        gate_line = {"promotion": gates["promotion"],
                     "gates_version": gates["gates_version"]}
    except Exception:
        gate_line = {"promotion": "BLOCKED",
                     "gates_version": "unreadable — fail closed"}
    rid = getattr(result, "recording_id", None) or "unknown"
    out_dir = pathlib.Path(runs_root or DEFAULT_RUNS) / f"head-{rid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"WATERMARK": WATERMARK,
           "purpose": ("research artifact for the vascular track; "
                       "user-visibility is decided by the vascular "
                       "block of configs/gates.yaml, nowhere else"),
           "recording_id": rid,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "gates": gate_line,
           "head_result": hr.to_dict()}
    path = out_dir / "vascular_head_report.json"
    path.write_text(json.dumps(doc, indent=2, default=str))
    return {"head_result": hr, "report_path": str(path)}
