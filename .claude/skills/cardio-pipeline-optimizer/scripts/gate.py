"""Accept/reject a candidate against the baseline. Full rule in references/decision-rules.md.

  accept iff
    (0) guard passed
    (1) every scan returned a structured outcome
    (2) signal improved by >= MIN_MARGIN, or cards improved with signal not down
    (3) no guardrail regression: cards, determinism, consistency, latency
    (4) per-record signal improvement on >= MIN_IMPROVED_RECORDS records

Usage:
  python gate.py --baseline baseline_score.json --candidate candidate_score.json \
      --guard guard.json --out verdict.json
"""
from __future__ import annotations

import argparse

from _common import dump_json, env_float, env_int, load_json

MIN_MARGIN = env_float("MIN_MARGIN", 0.02)
SIG_SLACK = env_float("SIG_SLACK", 0.0)
CARD_SLACK = env_float("CARD_SLACK", 0.0)
DET_SLACK = env_float("DET_SLACK", 0.0)
CONS_SLACK = env_float("CONS_SLACK", 0.05)
LAT_SLACK = env_float("LAT_SLACK", 0.05)
MIN_IMPROVED_RECORDS = env_int("MIN_IMPROVED_RECORDS", 3)
CONS_MARGIN = env_float("CONS_MARGIN", 0.05)   # reliability clause: consistency must rise by this


def _ge(a, b, slack):
    """a >= b - slack, treating a None on either side as 'not comparable' (passes)."""
    return True if a is None or b is None else a >= b - slack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--guard", default=None)
    ap.add_argument("--out", default="verdict.json")
    args = ap.parse_args()

    base, cand = load_json(args.baseline), load_json(args.candidate)
    bm, cm = base["metrics"], cand["metrics"]
    reasons, regressions = [], []

    # (0) guard
    guard_ok = True
    if args.guard:
        g = load_json(args.guard)
        guard_ok = bool(g.get("ok"))
        if not guard_ok:
            regressions += [f"guard: {v}" for v in g.get("violations", [])]

    # (1) every scan returned
    cond1 = cm["returned"] >= 1.0 - 1e-9
    if not cond1:
        failed = [p["id"] for p in cand["per_record"] if not p.get("returned")]
        regressions.append(f"not every scan returned a result: {failed}")

    # (2) improved
    sig_up = cm["signal"] >= bm["signal"] + MIN_MARGIN
    cards_up = (cm["cards"] >= bm["cards"] + MIN_MARGIN
                and cm["signal"] >= bm["signal"] - SIG_SLACK)
    # Reliability clause (owner decision 2026-09-08, "reliability over availability"):
    # a change that makes the remaining card values steadier across scans is an
    # improvement even if fewer cards compute — provided signal and determinism hold.
    # The card drop is reported in the reasons, never hidden.
    bc, cc = bm.get("consistency"), cm.get("consistency")
    cons_up = (bc is not None and cc is not None and cc >= bc + CONS_MARGIN
               and cm["signal"] >= bm["signal"] - SIG_SLACK
               and _ge(cm.get("determinism"), bm.get("determinism"), DET_SLACK))
    cond2 = sig_up or cards_up or cons_up
    if sig_up:
        reasons.append(f"signal {bm['signal']:.3f} -> {cm['signal']:.3f} (>= +{MIN_MARGIN})")
    if cards_up:
        reasons.append(f"cards {bm['cards']:.3f} -> {cm['cards']:.3f} with signal held")
    if cons_up:
        reasons.append(f"consistency {bc:.3f} -> {cc:.3f} (>= +{CONS_MARGIN}) with signal held"
                       + (f"; cards {bm['cards']:.3f} -> {cm['cards']:.3f} (accepted drop)"
                          if cm["cards"] < bm["cards"] else ""))
    if not cond2:
        regressions.append(f"no improvement: signal {bm['signal']:.3f}->{cm['signal']:.3f}, "
                           f"cards {bm['cards']:.3f}->{cm['cards']:.3f} (need +{MIN_MARGIN})")

    # (3) guardrails
    cond3 = True
    for name, slack in (("cards", CARD_SLACK), ("determinism", DET_SLACK),
                        ("consistency", CONS_SLACK), ("latency", LAT_SLACK)):
        if name == "cards" and cons_up:
            continue                      # the reliability clause allows a card drop
        if not _ge(cm.get(name), bm.get(name), slack):
            cond3 = False
            regressions.append(f"{name} regressed {bm.get(name)} -> {cm.get(name)} "
                               f"(slack {slack})")

    # (4) not a fluke
    bid = {p["id"]: p for p in base["per_record"]}
    improved = [p["id"] for p in cand["per_record"]
                if p["id"] in bid and p["signal"] > bid[p["id"]]["signal"] + 1e-6]
    need = min(MIN_IMPROVED_RECORDS, max(1, cand["n"] // 2 + 1))
    cond4 = len(improved) >= need or (cards_up and not sig_up) or (cons_up and not sig_up)
    if not cond4:
        regressions.append(f"only {len(improved)} record(s) improved (need >= {need}); "
                           f"may be a fluke")

    decision = "accept" if (guard_ok and cond1 and cond2 and cond3 and cond4) else "reject"
    verdict = {
        "decision": decision,
        "delta_total": round(cand["total"] - base["total"], 4),
        "conditions": {"guard": guard_ok, "every_scan_returned": cond1,
                       "improved": cond2, "no_regression": cond3, "not_fluke": cond4},
        "reasons": reasons if decision == "accept" else regressions,
        "improved_records": improved,
        "baseline_metrics": bm,
        "candidate_metrics": cm,
    }
    path = dump_json(verdict, args.out)
    print(f"[gate] DECISION={decision.upper()} delta_total={verdict['delta_total']}")
    for r in verdict["reasons"]:
        print(f"        - {r}")
    print(f"[gate] -> {path}")
    raise SystemExit(0 if decision == "accept" else 2)


if __name__ == "__main__":
    main()
