"""Research-surface glue for head_vasotone (v0.5, T4).

The one sanctioned way to a real reactivity reading: compute tone
features from a completed production run + provocation record, load the
latest evaluation run's recorded reactivity rule and the W1-surviving
set, hand everything to the head via ctx, and write a WATERMARKED
research report under the vasotone runs root. Nothing returned here may
be merged into a ScanResult or any consumer artifact (W-a).
"""
from __future__ import annotations

import json
import pathlib
import time

from research.vascular import WATERMARK
from research.vascular.tone_features import (pi_response,
                                             tone_session_features)


def _latest_rule(runs_root=None) -> dict:
    from evaluation.vasotone_gates import (DEFAULT_RUNS,
                                           latest_scoreboard_entry)
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
    if not isinstance(art, dict) or \
            art.get("kind") != "reactivity_primary" or \
            not art.get("primary_feature"):
        return {}
    return art


def run_research_tone_head(result, det, provocation, *, capture=None,
                           pi=None, runs_root=None,
                           gates_path=None) -> dict:
    """Run head_vasotone on a completed production run; write the
    watermarked research report; return {"head_result",
    "report_path"}."""
    from evaluation.vasotone_gates import (DEFAULT_RUNS,
                                           surviving_tone_features,
                                           vasotone_gate_status)
    from heads import get_head

    feats = tone_session_features(result, det, provocation,
                                  capture=capture)
    ctx = {"vasotone_research_surface": True,
           "vasotone_features": feats,
           "vasotone_model": _latest_rule(runs_root),
           "surviving_features": surviving_tone_features(
               runs_root=runs_root, gates_path=gates_path)}
    if pi is not None:
        ctx["pi_response"] = pi_response(
            pi, provocation.phase_marks["baseline"],
            provocation.phase_marks["stimulus"])
    hr = get_head("vasotone").run(det.get("lattice"), ctx)
    try:
        gates = vasotone_gate_status(runs_root=runs_root,
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
           "purpose": ("research artifact for the vasotone track; "
                       "user-visibility is decided by the vasotone "
                       "block of configs/gates.yaml, nowhere else"),
           "recording_id": rid,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "gates": gate_line,
           "head_result": hr.to_dict()}
    path = out_dir / "vasotone_head_report.json"
    path.write_text(json.dumps(doc, indent=2, default=str))
    return {"head_result": hr, "report_path": str(path)}
