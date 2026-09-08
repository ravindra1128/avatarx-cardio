"""Append one iteration's outcome to data/eval_cache/iterations.jsonl.

A record of what was tried, what the gate said, and why — so the next iteration (or
the next person) doesn't repeat a rejected hypothesis. Optionally commits an ACCEPTED
diff with the verdict in the message.

Usage:
  python log_iteration.py --verdict verdict.json --candidate candidate_score.json \
      --hypothesis "one line" --action accept [--commit]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

from _common import cache_dir, load_json, repo


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=repo(), capture_output=True,
                          text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--action", choices=["accept", "reject"], required=True)
    ap.add_argument("--commit", action="store_true",
                    help="commit the working-tree diff (accept only)")
    args = ap.parse_args()

    verdict = load_json(args.verdict)
    cand = load_json(args.candidate)
    if args.action == "accept" and verdict.get("decision") != "accept":
        raise SystemExit("[log] refusing: --action accept but verdict.decision != accept")

    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "head": _git("rev-parse", "--short", "HEAD"),
        "action": args.action,
        "decision": verdict.get("decision"),
        "hypothesis": args.hypothesis,
        "delta_total": verdict.get("delta_total"),
        "reasons": verdict.get("reasons"),
        "candidate_metrics": cand.get("metrics"),
        "diff_stat": _git("diff", "--stat"),
    }
    log = cache_dir() / "iterations.jsonl"
    with open(log, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    print(f"[log] {args.action} recorded -> {log}")

    if args.commit and args.action == "accept":
        msg = (f"cardio-pipeline-optimizer: {args.hypothesis}\n\n"
               f"Gate: accept (delta_total {verdict.get('delta_total')}).\n"
               + "\n".join(f"- {r}" for r in verdict.get("reasons", [])))
        subprocess.run(["git", "add", "-A", "rppg", "beats", "preprocessing", "features",
                        "inference", "app/measure_prep.py"], cwd=repo(), check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=repo(), check=True)
        print(f"[log] committed {_git('rev-parse', '--short', 'HEAD')}")


if __name__ == "__main__":
    main()
