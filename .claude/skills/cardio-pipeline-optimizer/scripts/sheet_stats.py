"""The owner's goal proxies, computed from the tracking sheet (`cardio-data` tab).

  availability     quality-passing scans that computed all three cards
  retest           same phone (User Agent) scanned again within 30 min: per-card CV
  pulse_agreement  |Pulse bpm - Ref HR| <= 5 on rows that carry a reference

Reads the sheet with the same credentials the service uses
(GOOGLE_SHEETS_CREDENTIALS_JSON or GOOGLE_SHEETS_CREDENTIALS_FILE). Read-only.

Usage:
  python sheet_stats.py [--since 2026-09-07] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))     # repo root
from app import result_sheet as rs  # noqa: E402

RETEST_WINDOW_MIN = 30
HR_TOL_BPM = 5.0
# The owner's stated target for a repeat scan of the same person under similar
# conditions: each card within +/-10-20 %. The metric below is the RELATIVE
# DIFFERENCE |a - b| / mean(a, b), not a coefficient of variation - with two
# samples the two differ by a constant factor, and the relative difference is
# what "within 20 %" actually means. Judged against both ends of the target.
RETEST_TARGET = 0.20
RETEST_STRETCH = 0.10
MIN_ROWS = 10
CARDS = ("Arterial Stiffness", "Vascular Tone", "Fitness")


def _f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def load_rows(since: str | None) -> list[dict]:
    ws, header = rs._worksheet()
    rows = [dict(zip(header, r)) for r in ws.get_all_values()[1:]]
    out = []
    for r in rows:
        try:
            r["_ts"] = datetime.strptime(r.get("Timestamp (UTC)", ""), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if since and r["_ts"] < datetime.strptime(since, "%Y-%m-%d"):
            continue
        out.append(r)
    return out


def availability(rows: list[dict]) -> dict:
    """Denominator: scans that produced evidence (a beat lattice); NO_RESULT with
    no beats is a capture failure, not the pipeline's to answer for."""
    quality = [r for r in rows if r.get("Outcome") and r.get("Outcome") != "NO_RESULT"
               and _f(r.get("Coherence")) is not None]
    status_cols = (("Arterial Stiffness", "AS Status", "AS Tier"),
                   ("Vascular Tone", "VT Status", "VT Tier"),
                   ("Fitness", "Fit Status", "Fit Tier"))
    full = [r for r in quality if all(r.get(sc) == "computed" for _, sc, _ in status_cols)]
    # Display tiers (2026-09-09): "computed" now includes provisional scores.
    # Measured-only availability is the number the standing goal is about;
    # rows written before the tier columns existed count as measured, since
    # every gate had to hold for them to compute at all.
    def measured(r, sc, tc):
        return r.get(sc) == "computed" and (r.get(tc) or "measured") == "measured"
    full_measured = [r for r in quality if all(measured(r, sc, tc) for _, sc, tc in status_cols)]
    per_card = {c: sum(1 for r in quality if r.get(sc) == "computed") for c, sc, _ in status_cols}
    per_card_measured = {c: sum(1 for r in quality if measured(r, sc, tc)) for c, sc, tc in status_cols}
    return {"n_scans": len(rows), "n_quality": len(quality),
            "n_all_three": len(full), "n_all_three_measured": len(full_measured),
            "availability": round(len(full) / len(quality), 3) if quality else None,
            "availability_measured": round(len(full_measured) / len(quality), 3) if quality else None,
            "per_card_computed": per_card, "per_card_measured": per_card_measured}


def _find_pairs(rows: list[dict]) -> list[tuple]:
    """Consecutive scans from the same phone inside the retest window."""
    by_ua: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["_ts"]):
        by_ua.setdefault(r.get("User Agent", ""), []).append(r)
    pairs = []
    for _ua, rs_ in by_ua.items():
        for a, b in zip(rs_, rs_[1:]):
            if b["_ts"] - a["_ts"] <= timedelta(minutes=RETEST_WINDOW_MIN):
                pairs.append((a, b))
    return pairs


def _rel_diff(x, y):
    if x is None or y is None:
        return None
    m = (x + y) / 2
    return None if not m else abs(x - y) / m


def retest(rows: list[dict]) -> dict:
    pairs = _find_pairs(rows)
    out = {"n_pairs": len(pairs), "window_min": RETEST_WINDOW_MIN,
           "target": RETEST_TARGET, "cards": {}}
    for c in CARDS:
        ds = [d for d in (_rel_diff(_f(a.get(c)), _f(b.get(c))) for a, b in pairs)
              if d is not None]
        out["cards"][c] = {
            "n_pairs_with_value": len(ds),
            "median_rel_diff": round(st.median(ds), 3) if ds else None,
            "within_20pct": round(sum(d <= RETEST_TARGET for d in ds) / len(ds), 3) if ds else None,
            "within_10pct": round(sum(d <= RETEST_STRETCH for d in ds) / len(ds), 3) if ds else None,
        }
    return out


def pair_detail(rows: list[dict]) -> list[dict]:
    """One row per pair, with the evidence needed to read it honestly.

    A card differing between two scans only means the MEASUREMENT is unstable
    if the person did not change. The reference pulse from the other device is
    the independent check on that, so it is reported beside every card."""
    out = []
    for a, b in _find_pairs(rows):
        gap = (b["_ts"] - a["_ts"]).total_seconds() / 60.0
        d = {"first": a["Timestamp (UTC)"], "second": b["Timestamp (UTC)"],
             "gap_min": round(gap, 1),
             "outcomes": [a.get("Outcome"), b.get("Outcome")],
             "ref_hr": [_f(a.get("Ref HR")), _f(b.get("Ref HR"))],
             "ref_hr_rel_diff": _rel_diff(_f(a.get("Ref HR")), _f(b.get("Ref HR"))),
             "pulse": [_f(a.get("Pulse bpm")), _f(b.get("Pulse bpm"))],
             "pulse_check": [a.get("Pulse Check"), b.get("Pulse Check")],
             "cards": {}}
        for c in CARDS:
            x, y = _f(a.get(c)), _f(b.get(c))
            rd = _rel_diff(x, y)
            d["cards"][c] = {"values": [x, y], "rel_diff": None if rd is None else round(rd, 3),
                             "verdict": ("no value" if rd is None else
                                         "within 10%" if rd <= RETEST_STRETCH else
                                         "within 20%" if rd <= RETEST_TARGET else "outside 20%")}
        out.append(d)
    return out


def pulse_agreement(rows: list[dict]) -> dict:
    d = [abs(_f(r.get("Pulse - Ref HR"))) for r in rows if _f(r.get("Pulse - Ref HR")) is not None]
    return {"n_with_reference": len(d), "tol_bpm": HR_TOL_BPM,
            "within_tol": round(sum(x <= HR_TOL_BPM for x in d) / len(d), 3) if d else None,
            "median_abs_diff": round(st.median(d), 1) if d else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--pairs", action="store_true",
                    help="print every pair in full - the view for a deliberate test-retest")
    args = ap.parse_args()
    rows = load_rows(args.since)
    rep = {"rows": len(rows), "availability": availability(rows), "retest": retest(rows),
           "pairs": pair_detail(rows),
           "pulse_agreement": pulse_agreement(rows), "meaningful": len(rows) >= MIN_ROWS}
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1, default=str))
    a, t, pa = rep["availability"], rep["retest"], rep["pulse_agreement"]
    print(f"[sheet] rows={rep['rows']}" + ("" if rep["meaningful"] else f"  (n < {MIN_ROWS}: not yet meaningful)"))
    print(f"  availability   any tier {a['availability']} ({a['n_all_three']}/{a['n_quality']}); "
          f"measured only {a['availability_measured']} ({a['n_all_three_measured']}/{a['n_quality']}); "
          f"per card {a['per_card_computed']} / measured {a['per_card_measured']}")
    print(f"  retest ({t['window_min']} min) {t['n_pairs']} pair(s), target ±{t['target']:.0%}: " + "; ".join(
        f"{c}: median |Δ|/mean {v['median_rel_diff']} within-20% {v['within_20pct']} (n={v['n_pairs_with_value']})"
        for c, v in t["cards"].items()))
    print(f"  pulse vs ref   within ±{pa['tol_bpm']} bpm: {pa['within_tol']}  median |Δ| {pa['median_abs_diff']}  (n={pa['n_with_reference']})")
    if args.pairs:
        for d in rep["pairs"]:
            print(f"\n  --- {d['first']} -> {d['second']}  ({d['gap_min']} min apart)")
            print(f"      outcomes {d['outcomes']}   pipeline pulse {d['pulse']}   pulse check {d['pulse_check']}")
            rd = d["ref_hr_rel_diff"]
            same = "unknown" if rd is None else ("the person barely changed" if rd <= 0.05
                                                 else f"the person's own pulse moved {rd:.0%}")
            print(f"      reference pulse {d['ref_hr']}  -> {same}")
            for c, v in d["cards"].items():
                pct = "" if v["rel_diff"] is None else f"{v['rel_diff']:.0%}"
                print(f"      {c:20} {str(v['values']):24} {pct:>5}  {v['verdict']}")


if __name__ == "__main__":
    main()
