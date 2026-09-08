"""Reduce a replay run to one comparable scalar + per-metric / per-record breakdown.

Definitions in references/decision-rules.md. There is no reference signal here: every
component is a proxy the pipeline itself exposes, which is why the gate pairs this with
guard.py — a proxy can only be trusted while the thresholds behind it are frozen.

Usage:
  python score.py --pred candidate.json --out candidate_score.json
"""
from __future__ import annotations

import argparse
import statistics

from _common import clip01, dump_json, env_float, env_int, load_json

W_SIG = env_float("W_SIG", 1.0)
W_CARD = env_float("W_CARD", 0.5)
W_DET = env_float("W_DET", 0.3)
W_CONS = env_float("W_CONS", 0.3)
W_LAT = env_float("W_LAT", 0.2)

COH_GATE = env_float("COH_GATE", 0.20)
COV_GATE = env_float("COV_GATE", 0.50)
TIM_GATE = env_float("TIM_GATE", 40.0)
SQI_GATE = env_float("SQI_GATE", 0.30)
CV_TOL = env_float("CV_TOL", 0.5)
MIN_CARDS_FOR_CV = env_int("MIN_CARDS_FOR_CV", 3)
LAT_GOAL_S = env_float("LAT_GOAL_S", 15.0)
LAT_TOL_S = env_float("LAT_TOL_S", 30.0)

CARD_KEYS = ("arterial_stiffness", "vascular_tone", "cardiorespiratory_fitness")


def _term(v, gate, lower_is_better=False):
    if v is None:
        return 0.0
    return clip01(gate / max(v, 1e-6)) if lower_is_better else clip01(v / gate)


def _per_record(r: dict) -> dict:
    if not r.get("returned"):
        return {"id": r["id"], "returned": 0.0, "signal": 0.0, "cards": 0.0,
                "determinism": None, "latency": 0.0}
    sig = statistics.fmean([
        _term(r.get("coherence"), COH_GATE),
        _term(r.get("coverage"), COV_GATE),
        _term(r.get("timing_ms"), TIM_GATE, lower_is_better=True),
        _term(r.get("sqi"), SQI_GATE),
    ])
    cards = sum(1 for k in CARD_KEYS
                if (r.get("cards") or {}).get(k, {}).get("status") == "computed") / 3.0
    det = None if "deterministic" not in r else (1.0 if r["deterministic"] else 0.0)
    job = (r.get("timing") or {}).get("server_total_s")
    lat = 0.0 if job is None else 1.0 - clip01((float(job) - LAT_GOAL_S) / LAT_TOL_S)
    return {"id": r["id"], "returned": 1.0, "signal": round(sig, 4), "cards": round(cards, 4),
            "determinism": det, "latency": round(lat, 4), "job_s": job,
            "outcome": r.get("outcome"), "gates_failed": r.get("gates_failed")}


def _consistency(rows: list) -> tuple:
    """1 - mean CV of card values across records, grouped by (key, unit)."""
    cvs = []
    for k in CARD_KEYS:
        by_unit = {}
        for r in rows:
            c = (r.get("cards") or {}).get(k) or {}
            if c.get("status") == "computed" and c.get("value") is not None:
                by_unit.setdefault(c.get("unit"), []).append(c["value"])
        for unit, vals in by_unit.items():
            if len(vals) >= MIN_CARDS_FOR_CV:
                m = statistics.fmean(vals)
                if abs(m) > 1e-9:
                    cvs.append(statistics.pstdev(vals) / abs(m))
    if not cvs:
        return None, []
    mean_cv = statistics.fmean(cvs)
    return round(1.0 - clip01(mean_cv / CV_TOL), 4), [round(c, 4) for c in cvs]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="score.json")
    args = ap.parse_args()

    rows = load_json(args.pred)["rows"]
    per = [_per_record(r) for r in rows]
    returned = _mean([p["returned"] for p in per]) or 0.0
    signal = _mean([p["signal"] for p in per]) or 0.0
    cards = _mean([p["cards"] for p in per]) or 0.0
    determinism = _mean([p["determinism"] for p in per])      # None if no --repeat
    consistency, cvs = _consistency(rows)                      # None if too few cards
    latency = _mean([p["latency"] for p in per]) or 0.0

    total = (W_SIG * signal + W_CARD * cards
             + W_DET * (determinism if determinism is not None else 0.0)
             + W_CONS * (consistency if consistency is not None else 0.0)
             + W_LAT * latency)
    result = {
        "n": len(per),
        "total": round(total, 4),
        "metrics": {"returned": returned, "signal": signal, "cards": cards,
                    "determinism": determinism, "consistency": consistency,
                    "latency": latency,
                    "mean_job_s": _mean([p.get("job_s") for p in per]),
                    "card_cvs": cvs},
        "per_record": per,
    }
    path = dump_json(result, args.out)
    m = result["metrics"]
    print(f"[score] total={result['total']} returned={m['returned']} signal={m['signal']} "
          f"cards={m['cards']} det={m['determinism']} cons={m['consistency']} "
          f"lat={m['latency']} (job {m['mean_job_s']}s) -> {path}")


if __name__ == "__main__":
    main()
