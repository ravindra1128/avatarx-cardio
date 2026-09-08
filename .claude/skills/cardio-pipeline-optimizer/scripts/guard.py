"""The structural guard: a candidate may not loosen a validity gate or edit out of scope.

Why this exists: every metric score.py computes is a proxy the pipeline exposes, and
every proxy can be inflated the same way — move a threshold until a number appears.
That is not an improvement, it is a fabricated one. So before any score is compared,
this script proves the thresholds behind the proxies did not move.

  --snapshot   record protected values + clean git HEAD (run BEFORE editing)
  --check      compare the working tree against the snapshot; write guard.json

Usage:
  python guard.py --snapshot
  python guard.py --check --out guard.json
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess

from _common import cache_dir, dump_json, load_json, repo

# (file, symbol, permissive direction). See references/decision-rules.md.
PROTECTED = [
    ("capture/ingest.py", "MIN_FINITE_FRACTION", "decrease"),
    ("capture/ingest.py", "MAX_COLLAPSED_INTERVAL_FRACTION", "increase"),
    ("capture/ingest.py", "CONSUMER_FPS_FLOOR", "decrease"),
    ("datasets/schema.py", "FPS_FLOOR_BEAT", "decrease"),
    ("datasets/schema.py", "BPP_FLOOR", "decrease"),
    ("datasets/schema.py", "CRF_CEILING", "increase"),
]
FROZEN_FILES = ["configs/default.yaml", "configs/gates.yaml"]
ALLOWED_DIRS = ("rppg/", "beats/", "preprocessing/", "features/", "inference/")
ALLOWED_FILES = ("app/measure_prep.py",)


def _read_symbol(rel: str, name: str):
    text = (repo() / rel).read_text()
    m = re.search(rf"^\s*{re.escape(name)}\s*[:=]\s*([0-9.]+)", text, re.M)
    return float(m.group(1)) if m else None


def _file_hash(rel: str) -> str | None:
    p = repo() / rel
    return hashlib.sha1(p.read_bytes()).hexdigest() if p.exists() else None


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=repo(), capture_output=True,
                          text=True).stdout.strip()


def _dirty() -> list:
    """Every path that differs from HEAD right now: modified + untracked."""
    changed = [p for p in _git("diff", "--name-only", "HEAD").splitlines() if p]
    changed += [p for p in _git("ls-files", "--others", "--exclude-standard").splitlines() if p]
    return sorted(set(changed))


def _snapshot() -> dict:
    return {
        "head": _git("rev-parse", "HEAD"),
        "symbols": {f"{f}::{s}": _read_symbol(f, s) for f, s, _ in PROTECTED},
        "frozen": {f: _file_hash(f) for f in FROZEN_FILES},
        # The working tree is rarely clean when an iteration starts (an
        # unrelated .gitignore edit, the skill's own files). Record what is
        # already dirty so --check judges only what the CANDIDATE changed,
        # not everything that differs from HEAD. Found on the first real run:
        # a pre-existing .gitignore edit failed the guard on a no-op.
        "dirty_at_snapshot": _dirty(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="guard.json")
    args = ap.parse_args()
    snap_path = cache_dir() / "guard_snapshot.json"

    if args.snapshot:
        snap = _snapshot()
        dump_json(snap, snap_path)
        for k, v in snap["symbols"].items():
            print(f"[guard] {k} = {v}")
        print(f"[guard] snapshot at HEAD {snap['head'][:10]} -> {snap_path}")
        return

    if not args.check:
        ap.error("pass --snapshot or --check")
    if not snap_path.exists():
        raise SystemExit("[guard] no snapshot — run `guard.py --snapshot` BEFORE editing")

    base = load_json(snap_path)
    now = _snapshot()
    violations = []

    # 1) protected thresholds must not move in the permissive direction (or vanish)
    for f, s, direction in PROTECTED:
        key = f"{f}::{s}"
        b, n = base["symbols"].get(key), now["symbols"].get(key)
        if b is None:
            continue                          # wasn't there at snapshot; nothing to protect
        if n is None:
            violations.append(f"{key} was removed")
        elif direction == "decrease" and n < b - 1e-12:
            violations.append(f"{key} loosened: {b} -> {n} (lower admits more scans)")
        elif direction == "increase" and n > b + 1e-12:
            violations.append(f"{key} loosened: {b} -> {n} (higher admits more scans)")

    # 2) frozen files must be byte-identical
    for f in FROZEN_FILES:
        if base["frozen"].get(f) != now["frozen"].get(f):
            violations.append(f"{f} changed (frozen — gate/config thresholds live here)")

    # 3) the diff must stay inside the signal chain — judged on what changed
    #    SINCE the snapshot, not on everything dirty relative to HEAD
    already = set(base.get("dirty_at_snapshot", []))
    changed = [p for p in _dirty() if p not in already]
    out_of_scope = [p for p in changed
                    if not (p.startswith(ALLOWED_DIRS) or p in ALLOWED_FILES
                            or p.startswith((".claude/", "data/")))]
    for p in out_of_scope:
        violations.append(f"out of scope: {p} (allowed: {', '.join(ALLOWED_DIRS + ALLOWED_FILES)})")

    ok = not violations
    dump_json({"ok": ok, "violations": violations, "changed_files": changed,
               "snapshot_head": base["head"], "symbols_now": now["symbols"]}, args.out)
    print(f"[guard] {'OK' if ok else 'FAIL'} — {len(changed)} file(s) changed")
    for v in violations:
        print(f"        - {v}")
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
