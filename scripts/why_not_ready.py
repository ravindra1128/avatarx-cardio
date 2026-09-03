"""Summarise a live session's readiness_log.jsonl: which checks blocked
READY, how often, with what values — the post-mortem for "the scan would
not start" (v0.1.4; the 2026-08-20/21 failures left no evidence at all).

Usage:
  python3 scripts/why_not_ready.py                  # newest session
  python3 scripts/why_not_ready.py /tmp/avatarx_live/<session-id>
"""
import json
import pathlib
import sys

import numpy as np

LIVE = pathlib.Path("/tmp/avatarx_live")


def find_session(arg: str = None) -> pathlib.Path:
    if arg:
        p = pathlib.Path(arg)
        return p if p.is_dir() else LIVE / arg
    cands = sorted((d for d in LIVE.iterdir() if d.is_dir()),
                   key=lambda d: d.stat().st_mtime) if LIVE.is_dir() else []
    if not cands:
        sys.exit("no sessions under /tmp/avatarx_live")
    return cands[-1]


def main() -> None:
    sess = find_session(sys.argv[1] if len(sys.argv) > 1 else None)
    log = sess / "readiness_log.jsonl"
    # v0.4: a three-phase vsession dir holds per-phase subdirs + its own
    # session_log.jsonl — summarise each phase, then the session verdict
    if not log.exists() and ((sess / "session_log.jsonl").exists()
                             or (sess / "rest").is_dir()):
        print(f"three-phase session {sess.name}:")
        for phase in ("rest", "recovery"):
            for d in sorted(sess.glob(f"{phase}*")):
                if (d / "readiness_log.jsonl").exists():
                    print(f"\n--- phase: {d.name} "
                          f"(python3 scripts/why_not_ready.py {d})")
        slog = sess / "session_log.jsonl"
        if slog.exists():
            import json as _j
            rec = _j.loads(slog.read_text().splitlines()[-1])
            print(f"\nsession verdict: {rec.get('outcome')}")
            print(f"compliance: {rec.get('compliance')}")
            print(f"transition_s: {rec.get('transition_s')}")
            print(f"per-phase SQI: {rec.get('per_phase_sqi')}")
            print(f"rejections: {rec.get('rejection_reasons')}")
        else:
            print("\nno session_log.jsonl — the session never reached "
                  "finalize (check each phase above)")
        print("\nFitness-track gate status (never a fitness claim while "
              "red): python3 cli.py gate-status --track vo2")
        return
    if not log.exists():
        import time as _t
        print(f"{log} does not exist — this session never received a single "
              "frame batch (or predates v0.1.4 telemetry).")
        print("\nAll sessions on this machine:")
        if LIVE.is_dir():
            for d in sorted((x for x in LIVE.iterdir() if x.is_dir()),
                            key=lambda x: x.stat().st_mtime, reverse=True):
                age_h = (_t.time() - d.stat().st_mtime) / 3600.0
                has = "log+" if (d / "readiness_log.jsonl").exists() else "EMPTY"
                print(f"  {d.name}  {_t.strftime('%Y-%m-%d %H:%M', _t.localtime(d.stat().st_mtime))}"
                      f"  ({age_h:.1f} h ago, {has})")
        print("\nIf no session matches your LATEST attempt, your browser was "
              "not talking to the current build. Most common cause: a STALE "
              "`cli.py demo` process from an earlier session still holds the "
              "port — check with:\n  lsof -nP -iTCP:8765 -sTCP:LISTEN\n"
              "kill it, run `python3 cli.py demo` again, and confirm the "
              "build commit shown under the page title matches `git "
              "rev-parse --short HEAD`.")
        sys.exit(1)
    evals, events = [], []
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        (evals if rec.get("kind") == "eval" else events).append(rec)
    print(f"session {sess.name}: {len(evals)} evaluations, "
          f"{len(events)} events")
    if not evals:
        return

    ready = [e for e in evals if e.get("ready")]
    print(f"READY on {len(ready)}/{len(evals)} evaluations "
          f"({100.0 * len(ready) / len(evals):.0f}%)")
    span = (evals[0].get("t"), evals[-1].get("t"))
    if all(isinstance(x, (int, float)) for x in span):
        print(f"capture clock t = {span[0]:.1f} .. {span[1]:.1f} s")
    lat = [e["eval_ms"] for e in evals if isinstance(e.get("eval_ms"),
                                                     (int, float))]
    if lat:
        print(f"evidence eval: median {np.median(lat):.0f} ms, "
              f"p90 {np.percentile(lat, 90):.0f} ms")

    # per-check: how often it blocked (post-debounce), and its values
    names = []
    for e in evals:
        for k in (e.get("checks") or {}):
            if k not in names:
                names.append(k)
    print(f"\n{'check':22s} {'blocked':>8s} {'flicker':>8s}   values (med / p10)")
    for k in names:
        blocked = sum(1 for e in evals if k in (e.get("failing") or []))
        flicker = sum(1 for e in evals
                      if k in (e.get("failing_now") or [])
                      and k not in (e.get("failing") or []))
        vals = [e["checks"][k][0] for e in evals
                if k in (e.get("checks") or {})
                and isinstance(e["checks"][k][0], (int, float))
                and not isinstance(e["checks"][k][0], bool)]
        if vals:
            v = f"{float(np.median(vals)):.3g} / {float(np.percentile(vals, 10)):.3g}"
        else:
            v = "-"
        flag = " <-- blocker" if blocked > 0.2 * len(evals) else ""
        print(f"{k:22s} {blocked:8d} {flicker:8d}   {v}{flag}")

    if events:
        print("\nevents:")
        for e in events:
            t = e.get("t")
            ts = f"t={t:.1f}s " if isinstance(t, (int, float)) else ""
            extra = {k: v for k, v in e.items()
                     if k not in ("kind", "event", "t")}
            print(f"  {ts}{e.get('event')} {json.dumps(extra) if extra else ''}")


if __name__ == "__main__":
    main()
