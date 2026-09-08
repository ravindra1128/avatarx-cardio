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
CV_TOL = 0.5
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
    full = [r for r in quality if all(r.get(f"{k} Status" if k != "Fitness" else "Fit Status",
                                            "") == "computed"
                                      for k in ("AS", "VT", "Fitness"))]
    per_card = {c: sum(1 for r in quality if r.get(s) == "computed") for c, s in
                (("Arterial Stiffness", "AS Status"), ("Vascular Tone", "VT Status"),
                 ("Fitness", "Fit Status"))}
    return {"n_scans": len(rows), "n_quality": len(quality), "n_all_three": len(full),
            "availability": round(len(full) / len(quality), 3) if quality else None,
            "per_card_computed": per_card}


def retest(rows: list[dict]) -> dict:
    """Pairs = consecutive rows from the same User Agent within the window."""
    by_ua: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["_ts"]):
        by_ua.setdefault(r.get("User Agent", ""), []).append(r)
    pairs = []
    for ua, rs_ in by_ua.items():
        for a, b in zip(rs_, rs_[1:]):
            if b["_ts"] - a["_ts"] <= timedelta(minutes=RETEST_WINDOW_MIN):
                pairs.append((a, b))
    out = {"n_pairs": len(pairs), "window_min": RETEST_WINDOW_MIN, "cards": {}}
    for c in CARDS:
        cvs = []
        for a, b in pairs:
            x, y = _f(a.get(c)), _f(b.get(c))
            if x is None or y is None:
                continue
            m = (x + y) / 2
            if m:
                cvs.append(abs(x - y) / m)     # 2-sample CV proxy: |Δ| / mean
        out["cards"][c] = {"n_pairs_with_value": len(cvs),
                           "median_cv": round(st.median(cvs), 3) if cvs else None,
                           "within_tol": round(sum(v <= CV_TOL for v in cvs) / len(cvs), 3) if cvs else None}
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
    args = ap.parse_args()
    rows = load_rows(args.since)
    rep = {"rows": len(rows), "availability": availability(rows), "retest": retest(rows),
           "pulse_agreement": pulse_agreement(rows), "meaningful": len(rows) >= MIN_ROWS}
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1, default=str))
    a, t, pa = rep["availability"], rep["retest"], rep["pulse_agreement"]
    print(f"[sheet] rows={rep['rows']}" + ("" if rep["meaningful"] else f"  (n < {MIN_ROWS}: not yet meaningful)"))
    print(f"  availability   {a['availability']}  ({a['n_all_three']}/{a['n_quality']} quality scans; per card {a['per_card_computed']})")
    print(f"  retest ({t['window_min']} min) {t['n_pairs']} pair(s): " + "; ".join(
        f"{c}: median CV {v['median_cv']} within-tol {v['within_tol']} (n={v['n_pairs_with_value']})" for c, v in t["cards"].items()))
    print(f"  pulse vs ref   within ±{pa['tol_bpm']} bpm: {pa['within_tol']}  median |Δ| {pa['median_abs_diff']}  (n={pa['n_with_reference']})")


if __name__ == "__main__":
    main()
