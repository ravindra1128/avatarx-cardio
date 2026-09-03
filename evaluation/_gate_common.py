"""Shared gate mechanics for the pre-registered promotion tracks
(vo2 §V, vascular, vasotone §W) — v0.5 refactor so a third track REUSES
the machinery instead of forking it. Pure stdlib + configs; importable
from app/ (fail-closed consumers depend on that). The per-track modules
own their CONTRACTS (gate keys, criteria, notes); this module owns only
the mechanics: explicit-threshold reads, the yaml-null convention,
evidence qualification, and scoreboard IO.
"""
from __future__ import annotations

import json
import pathlib


def make_req(block_label: str):
    """Explicit-threshold reader factory: thresholds must be IN
    gates.yaml — a silently applied coded default would let the
    pre-registered contract drift (v0.4 review discipline)."""
    def _req(section, key):
        if not isinstance(section, dict) or key not in section:
            raise ValueError(
                f"configs/gates.yaml {block_label} block missing "
                f"required threshold {key!r} — the pre-registered "
                "contract must be explicit")
        return section[key]
    return _req


def unset(v) -> bool:
    """True for every YAML null spelling the subset parser leaves as a
    string — 'None'/'NULL'/'~' must never count as a signed value."""
    return v is None or (isinstance(v, str)
                         and v.strip().lower() in ("", "null", "none",
                                                   "~"))


def qualification(gcfg: dict, data) -> list:
    """Disqualifying reasons for one run's evidence standing."""
    req = gcfg.get("evidence_requirements") or {}
    if not isinstance(data, dict) or not data:
        return ["no evaluation run on record"]
    out = []
    want = req.get("signal_domain", "facial_rppg")
    if data.get("signal_domain") != want:
        out.append(f"evidence domain {data.get('signal_domain')!r} is "
                   f"not {want!r} — machinery evidence only, can never "
                   "open a gate")
    for key, label in (("participant_disjoint", "participant-disjoint"),
                       ("session_disjoint", "session-disjoint"),
                       ("production_path", "production-path")):
        # the boolean True, never truthiness: a string "false" is not
        # evidence of disjointness (review finding)
        if req.get(key, True) and data.get(key) is not True:
            out.append(f"evidence is not {label}")
    return out


def append_scoreboard(entry: dict, runs_root, default_runs) -> pathlib.Path:
    root = pathlib.Path(runs_root or default_runs)
    root.mkdir(parents=True, exist_ok=True)
    p = root / "scoreboard.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return p


def latest_scoreboard_entry(kind, runs_root, default_runs):
    p = pathlib.Path(runs_root or default_runs) / "scoreboard.jsonl"
    if not p.exists():
        return None
    last = None
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("scoreboard line is not a JSON object: "
                             f"{line[:60]!r}")
        if kind is None or row.get("kind") == kind:
            last = row
    return last
