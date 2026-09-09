"""Replay one split through the CURRENT pipeline via the production measure path.

Each record is COPIED to a temp dir (measure_video rewrites its input in place during
trim/downscale) and run through app.measure_api.measure_video in-process — the same
trim → downscale → clock → analyse → biomarkers path the deployed service runs, with the
same launch overrides. No HTTP server is started.

`--repeat N` runs each record N times and records whether every run matched the first
(outcome, evidence numbers, card values). That is how determinism is measured.

Usage:
  python replay.py --split holdout --label baseline --repeat 2 --out baseline.json
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
import time
import traceback

from _common import apply_pipeline_env, dump_json, load_json, manifest_path, repo

apply_pipeline_env()                                   # before importing the pipeline
from app.measure_api import measure_video              # noqa: E402

CARD_KEYS = ("arterial_stiffness", "vascular_tone", "cardiorespiratory_fitness")


def _num(x):
    try:
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _collect(doc: dict) -> dict:
    ev = (doc.get("debug") or {}).get("evidence") or {}
    ra = (doc.get("debug") or {}).get("rationale") or {}
    items = {i.get("key"): i for i in (doc.get("biomarkers") or {}).get("items", [])
             if isinstance(i, dict)}
    cards = {}
    for k in CARD_KEYS:
        it = items.get(k) or {}
        cards[k] = {"status": it.get("status"), "value": _num(it.get("value")),
                    "unit": it.get("unit"), "reason": it.get("reason")}
    return {
        "outcome": doc.get("outcome"),
        "no_read_reasons": list(doc.get("no_read_reasons") or []),
        "sqi": _num(doc.get("signal_quality_index")),
        "stars": doc.get("confidence_stars"),
        "coherence": _num(ev.get("cross_roi_coherence")),
        "timing_ms": _num(ev.get("timing_precision_ms")),
        "n_intervals": ev.get("n_intervals"),
        "n_beats": ev.get("n_beats"),
        "dropout_rate": _num(ev.get("dropout_rate")),
        "coverage": _num(ra.get("coverage")),
        "gates_failed": [g.get("name") for g in (ra.get("gates") or [])
                         if isinstance(g, dict) and g.get("pass") is False],
        "cards": cards,
        "timing": doc.get("timing") or {},
        "prep": {"trim": doc.get("trim"), "downscale": doc.get("downscale"),
                 "clock": doc.get("clock")},
    }


def _same(a: dict, b: dict) -> bool:
    """Determinism: outcome, every evidence number, every card value."""
    if a["outcome"] != b["outcome"]:
        return False
    for k in ("sqi", "coherence", "timing_ms", "coverage"):
        x, y = a.get(k), b.get(k)
        if (x is None) != (y is None):
            return False
        if x is not None and abs(x - y) > 1e-6:
            return False
    for k in CARD_KEYS:
        x, y = a["cards"][k]["value"], b["cards"][k]["value"]
        if (x is None) != (y is None):
            return False
        if x is not None and abs(x - y) > 1e-6:
            return False
    return True


def _run_once(rel_path: str, pretrimmed: bool = False) -> dict:
    # `pretrimmed` (manifest flag, 2026-09-09 repair): the file already IS the
    # analysed 40 s window; re-trimming would drop leading frames and delete
    # the sidecar clock the baseline was scored with.
    src = repo() / rel_path
    d = tempfile.mkdtemp(prefix="cardio-replay-")
    try:
        v = os.path.join(d, src.name)
        shutil.copy(src, v)
        side = src.parent / f"{src.name}.timestamps.json"
        if side.exists():
            shutil.copy(side, v + ".timestamps.json")
        t0 = time.perf_counter()
        doc = measure_video(v, manifest={"capture_profile": "consumer"},
                            window_s=(0 if pretrimmed else None))
        row = _collect(doc)
        row["wall_s"] = round(time.perf_counter() - t0, 2)
        row["returned"] = True
        return row
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "holdout", "all"], default="holdout")
    ap.add_argument("--label", default="run")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default="run.json")
    args = ap.parse_args()

    manifest = load_json(manifest_path())["records"]
    rows = []
    for m in manifest:
        if args.split != "all" and m["split"] != args.split:
            continue
        runs, err = [], None
        for i in range(max(1, args.repeat)):
            try:
                runs.append(_run_once(m["path"], bool(m.get("pretrimmed"))))
            except Exception as e:  # noqa: BLE001 — a crash IS the finding
                err = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                break
        if not runs:
            rows.append({"id": m["id"], "split": m["split"], "source": m["source"],
                         "returned": False, "error": err, "outcome": None, "cards": {}})
            print(f"[replay] {m['id']}: RETURNED=False ({err})")
            continue
        first = runs[0]
        first["id"], first["split"], first["source"] = m["id"], m["split"], m["source"]
        if len(runs) > 1:
            first["deterministic"] = all(_same(first, r) for r in runs[1:])
            first["repeat_n"] = len(runs)
        rows.append(first)
        n_cards = sum(1 for k in CARD_KEYS if first["cards"][k]["status"] == "computed")
        print(f"[replay] {m['id']:<32} {first['outcome']:<12} coh={first['coherence']} "
              f"cov={first['coverage']} tim={first['timing_ms']} cards={n_cards}/3 "
              f"job={first['timing'].get('server_total_s')}s"
              + (f" det={first['deterministic']}" if 'deterministic' in first else ""))

    path = dump_json({"label": args.label, "split": args.split, "repeat": args.repeat,
                      "rows": rows}, args.out)
    print(f"[replay] {len(rows)} records -> {path}")


if __name__ == "__main__":
    main()
