#!/bin/sh
# SessionStart hook: print the standing state so no session starts blind.
C=/Users/ravindrasinghbisht/Downloads/cardio
[ -d "$C" ] || exit 0
PY=/Users/ravindrasinghbisht/miniconda3/bin/python3
echo "=== cardio brief (from .claude/hooks/session-brief.sh) ==="
echo "goal: every quality-passing scan computes all 3 cards; repeat scans within 30 min agree; pulse agrees with the reference. Rules and paths: $C/CLAUDE.md"
( cd "$C" && echo "git: $(git rev-parse --short HEAD) on $(git branch --show-current); personal/main $(git rev-parse --short personal/main 2>/dev/null || echo ?)  personal/staging $(git rev-parse --short personal/staging 2>/dev/null || echo ?)  origin/main $(git rev-parse --short origin/main 2>/dev/null || echo ?); dirty: $(git status --short | grep -v '^?? data/\|^?? .claude/' | wc -l | tr -d ' ') file(s)" )
echo "deploy: work goes to STAGING first (personal/staging -> avatarx-cardio-staging), is measured there, and only then merges to main (-> avatarx-cardio-production, the service the owner's phone scans). Pushing an experiment straight to main is how a bitrate change that broke the SQI floor reached real scans on 2026-09-10."
if [ -f "$C/data/eval_cache/baseline_score.json" ]; then
  $PY - <<PYEOF 2>/dev/null
import json
m = json.load(open("$C/data/eval_cache/baseline_score.json")).get("metrics", {})
print("baseline (holdout):", {k: m.get(k) for k in ("signal","cards","determinism","consistency","latency","mean_job_s")})
PYEOF
fi
if [ -f "$C/data/eval_cache/iterations.jsonl" ]; then
  echo "last iterations:"; tail -3 "$C/data/eval_cache/iterations.jsonl" | $PY -c "
import sys, json
for l in sys.stdin:
    e = json.loads(l); print(f\"  {e['ts'][:10]} {e['action']:<7} {e['hypothesis'][:120]}\")" 2>/dev/null
fi
echo "next: state the proxy + one hypothesis, run the optimizer skill, honor the verdict."
