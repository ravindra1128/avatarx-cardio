"""Repeatability of the fitness ("VO2 Max") card on the retained phone scans.

Replays every retained clip that has a ShenAI sidecar through the SAME job the
service runs (`MeasureHandler._run_assembled`: trim -> downscale -> pipeline ->
cards -> ShenAI route -> AFib result), with the request parameters the webapp
really sends (window_s=70, capture_profile=consumer, ref_hr from the SDK), and
records the card together with every input that feeds it. The clips are one
person, scanned in bursts, so scans closer than --group-min minutes form a
repeat group and the card's scan-to-scan spread is measurable per stage:

    SDK live-frame HR  ->  our rate (count / spectral / resolver)  ->  card

`--repo` points the harness at ANOTHER checkout of this repository (e.g. a
`git archive` of the production branch), so BEFORE and AFTER run the same
clips through the same harness. Corpus files are never handed to the
pipeline: each run works on a copy (measure_video trims its input in place).

Usage:
  python scripts/fitness_repeatability.py replay --label staging --out fit_staging.json
  python scripts/fitness_repeatability.py replay --repo /tmp/cardio-main --label production
  python scripts/fitness_repeatability.py stats fit_staging.json [fit_production.json ...]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import statistics as st
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TOL_UNITS = 8.0                     # the owner's repeatability target, card units
# What the deployed services launch with (GET /healthz -> launch_overrides).
# AFIB_TRACE_PATH=0: the recorded-only trace path would wait 8 s per clip for a
# trace document no retained scan has; it never touches the cards.
PRODUCTION_ENV = {"AFIB_MAX_COLLAPSED_FRACTION": "0.05", "AFIB_TRACE_PATH": "0"}
STAGING_ENV = dict(PRODUCTION_ENV, AFIB_CONFIG_OVERRIDES=json.dumps({
    "decision.classifier": "model_a",
    "decision.model_a_path": "models/model_a_v02_45s.json"}))


def _num(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _scan_time(name: str):
    try:
        return datetime.strptime(name[:16], "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None


def corpus_records(corpus: Path) -> list:
    out = []
    for side in sorted(corpus.glob("*.shenai.json")):
        clip = side.with_name(side.name[:-len(".shenai.json")])
        t = _scan_time(clip.name)
        if clip.exists() and t is not None:
            out.append({"id": clip.name.split(".")[0], "clip": clip, "sidecar": side, "t": t})
    return out


def assign_groups(records: list, group_min: float) -> None:
    """A repeat group is a run of scans each within `group_min` of the previous."""
    g, prev = 0, None
    for r in sorted(records, key=lambda r: r["t"]):
        if prev is not None and (r["t"] - prev).total_seconds() > group_min * 60:
            g += 1
        r["group"], prev = g, r["t"]


def _sidecar_summary(raw: dict) -> dict:
    ref = raw.get("reference") or {}
    hb = [b for b in (raw.get("heartbeats") or []) if isinstance(b, dict)]
    durs = [_num(b.get("duration_ms")) for b in hb]
    durs = [d for d in durs if d and 250.0 <= d <= 2200.0]
    ln = _num(ref.get("hrv_lnrmssd_ms"))
    return {"sdk_hr_bpm": _num(ref.get("heart_rate_bpm")),
            "sdk_sdnn_ms": _num(ref.get("hrv_sdnn_ms")),
            "sdk_rmssd_ms": (math.exp(ln) if ln is not None and ln < 7 else ln),
            "sdk_quality": _num(ref.get("average_signal_quality")),
            "sdk_bad_signal_s": _num(ref.get("bad_signal_seconds")),
            "train_beats": len(hb),
            "train_median_bpm": (round(60000.0 / st.median(durs), 2) if durs else None)}


def _collect(doc: dict) -> dict:
    dbg = doc.get("debug") or {}
    ev, ra = dbg.get("evidence") or {}, dbg.get("rationale") or {}
    items = {i.get("key"): i for i in (doc.get("biomarkers") or {}).get("items", [])
             if isinstance(i, dict)}
    fit = items.get("cardiorespiratory_fitness") or {}
    det = fit.get("details") if isinstance(fit.get("details"), dict) else {}
    guard = det.get("rate_guard") if isinstance(det.get("rate_guard"), dict) else {}
    route = dbg.get("shenai_route") if isinstance(dbg.get("shenai_route"), dict) else {}
    return {
        "outcome": doc.get("outcome"), "stars": doc.get("confidence_stars"),
        "biomarkers_error": (doc.get("biomarkers") or {}).get("error"),
        "pulse_bpm": _num(doc.get("mean_pulse_rate_bpm")),
        "sqi": _num(doc.get("signal_quality_index")),
        "coherence": _num(ev.get("cross_roi_coherence")),
        "timing_ms": _num(ev.get("timing_precision_ms")),
        "n_intervals": ev.get("n_intervals"), "coverage": _num(ra.get("coverage")),
        "pulse_lattice_bpm": _num(ev.get("pulse_lattice_bpm")),
        "pulse_spectral_bpm": _num(ev.get("pulse_spectral_bpm")),
        "pulse_spectral_snr": _num(ev.get("pulse_spectral_snr")),
        "spectral_roi_agree": ev.get("pulse_spectral_roi_agree"),
        "harmonic_fraction": _num(ev.get("harmonic_fraction")),
        "gates_failed": [g.get("name") for g in (ra.get("gates") or [])
                         if isinstance(g, dict) and g.get("pass") is False],
        "fit_status": fit.get("status"), "fit_value": _num(fit.get("value")),
        "fit_unit": fit.get("unit"), "fit_metric": fit.get("metric"),
        "fit_tier": fit.get("tier"), "fit_reason": fit.get("reason"),
        "fit_tier_reasons": list(fit.get("tier_reasons") or []),
        "fit_hr_bpm": _num((fit.get("raw") or {}).get("value")),
        "fit_rate_source": det.get("resting_rate_source"),
        "fit_rate_method": det.get("resting_rate_method"),
        "fit_rate_intervals": det.get("resting_rate_intervals"),
        "fit_rmssd_ms": _num(det.get("rmssd_ms")),
        "fit_details_keys": sorted(det.keys()),
        "fit_estimator": det.get("estimator"),
        "rate_guard_signature": guard.get("signature"),
        "rate_guard_applied": guard.get("applied", guard.get("guarded")),
        "route_used": route.get("used"), "route_reason": route.get("reason"),
        "route_fitness": route.get("fitness_reconciliation"),
        "reference": doc.get("reference"),
        "timing": doc.get("timing") or {},
        "trim_note": (doc.get("trim") or {}).get("note"),
    }


def replay(args) -> None:
    repo = Path(args.repo).expanduser().resolve()
    env = STAGING_ENV if args.launch == "staging" else PRODUCTION_ENV
    for k, v in env.items():
        os.environ.setdefault(k, v)
    if args.scale:
        os.environ["AFIB_SCALE"] = args.scale
    if args.no_route:
        os.environ["AFIB_SHENAI_ROUTE"] = "0"         # the service's own switch, read at import
    sys.path.insert(0, str(repo))
    os.chdir(repo)                                   # configs/ and models/ resolve from the tree
    from app import measure_api as api               # noqa: E402  (after env + path)

    has_route = hasattr(api, "_apply_shenai_route") and not args.no_route
    records = corpus_records(Path(args.corpus).expanduser().resolve())
    assign_groups(records, args.group_min)
    if args.only:
        records = [r for r in records if any(s in r["id"] for s in args.only)]
    rows = []
    for r in records:
        raw = json.loads(r["sidecar"].read_text())
        side = _sidecar_summary(raw)
        d = tempfile.mkdtemp(prefix="fit-repeat-")
        try:
            # The job's layout: <work>/<upload_id>/scan.<ext>; the route looks the
            # held sidecar up by the directory name.
            uid = f"replay-{r['id'][:24]}"
            part = Path(d) / uid
            part.mkdir()
            v = part / f"scan{r['clip'].suffix}"
            shutil.copy(r["clip"], v)
            header = {"capture_profile": "consumer", "window_s": float(args.window_s),
                      "session": "replay", "upload_id": uid}
            if side["sdk_hr_bpm"] is not None and not args.no_reference:   # what getReference() attaches
                header["reference"] = {"ref_hr": side["sdk_hr_bpm"], "ref_source": "shenai"}
                if side["sdk_sdnn_ms"] is not None:
                    header["reference"]["ref_hrv"] = side["sdk_sdnn_ms"]
            if args.participant:
                header["participant"] = json.loads(args.participant)
            t0 = time.perf_counter()
            if has_route:
                api._hold_signals(uid, raw)
            try:
                # ALWAYS the job the service runs, so the profile and the
                # reference reach the cards exactly as a real request's do.
                # (2026-09-20: a --no-route run once fell through to a bare
                # measure_video call here, silently dropped the profile, and
                # scored the legacy card under a VO2max label. Void, re-run.)
                doc = api.MeasureHandler._run_assembled(str(v), header)
            finally:
                if has_route:
                    api._drop_signals(uid)
            row = _collect(doc)
            if args.participant and row.get("fit_value") is not None \
                    and row.get("fit_unit") != "mL/kg/min":
                raise RuntimeError(f"a profile was sent but the card came back in "
                                   f"{row.get('fit_unit')!r}: the tree under test ignored it")
            row["wall_s"] = round(time.perf_counter() - t0, 2)
            row["returned"] = True
        except Exception as e:                            # noqa: BLE001 - a crash IS the finding
            row = {"returned": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            shutil.rmtree(d, ignore_errors=True)
        row.update(id=r["id"], t=r["t"].isoformat(), group=r["group"], **side)
        rows.append(row)
        print(f"[fit] {r['id'][:16]} g{r['group']} sdk={side['sdk_hr_bpm'] and round(side['sdk_hr_bpm'])} "
              f"pulse={row.get('pulse_bpm')} fitHR={row.get('fit_hr_bpm')} "
              f"fit={row.get('fit_value')} {row.get('fit_tier') or row.get('fit_status')} "
              f"src={row.get('fit_rate_source')} route={row.get('route_used')}", flush=True)
    out = Path(args.out or f"fit_{args.label}.json")
    if out.parent == Path("."):
        out = HERE / "data" / "eval_cache" / out
    doc = {"label": args.label, "repo": str(repo), "launch": args.launch,
           "window_s": args.window_s, "scale": args.scale, "route": has_route, "rows": rows}
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(f"[fit] {len(rows)} scans -> {out}")


# ------------------------------------------------------------------ statistics
def _spread(vals: list) -> dict:
    v = [x for x in vals if x is not None]
    if not v:
        return {"n": 0}
    out = {"n": len(v), "mean": round(st.fmean(v), 2), "min": round(min(v), 2),
           "max": round(max(v), 2), "range": round(max(v) - min(v), 2)}
    if len(v) > 1:
        out["sd"] = round(st.stdev(v), 2)
        if out["mean"]:
            out["cv_pct"] = round(100.0 * out["sd"] / abs(out["mean"]), 1)
    return out


def _pairs(vals: list) -> list:
    v = [x for x in vals if x is not None]
    return [abs(a - b) for a, b in itertools.combinations(v, 2)]


def window_pairs(rows: list, key, max_min: float, same=None) -> tuple:
    """(|a-b| for every pair of scans no more than `max_min` apart with a value
    on both sides, number of such pairs with a value MISSING on either side).
    Pairwise, not chained: a 13-scan session is not one 2-hour "repeat"."""
    get = key if callable(key) else (lambda r: r.get(key))
    diffs, missing = [], 0
    for a, b in itertools.combinations(rows, 2):
        if same is not None and same(a) != same(b):
            continue
        ta, tb = a["t"], b["t"]
        if isinstance(ta, str):
            ta, tb = datetime.fromisoformat(ta), datetime.fromisoformat(tb)
        if abs((tb - ta).total_seconds()) > max_min * 60:
            continue
        va, vb = get(a), get(b)
        if va is None or vb is None:
            missing += 1
        else:
            diffs.append(abs(va - vb))
    return diffs, missing


def pair_report(rows: list, value, max_min: float, tol: float, same=None) -> dict:
    """Repeat-pair statistics of one quantity over strictly time-windowed pairs."""
    d, missing = window_pairs(rows, value, max_min, same)
    out = {"pairs": len(d), "pairs_missing_a_value": missing}
    if d:
        out.update(within_tol=sum(1 for x in d if x <= tol),
                   pct_within_tol=round(100.0 * sum(1 for x in d if x <= tol) / len(d), 1),
                   mean_abs_diff=round(st.fmean(d), 2), median_abs_diff=round(st.median(d), 2),
                   p90_abs_diff=round(sorted(d)[max(0, math.ceil(0.9 * len(d)) - 1)], 2),
                   max_abs_diff=round(max(d), 2),
                   # SD of a single scan implied by the pair differences (Bland-Altman
                   # within-subject SD: sqrt(mean(d^2) / 2)).
                   within_subject_sd=round(math.sqrt(st.fmean([x * x for x in d]) / 2.0), 2))
    return out


def _fmt(rep: dict, tol: float, unit: str = "") -> str:
    if not rep.get("pairs"):
        return f"no pairs with a value on both sides ({rep.get('pairs_missing_a_value', 0)} pairs missing a value)"
    return (f"within +/-{tol:g}: {rep['within_tol']}/{rep['pairs']} = {rep['pct_within_tol']}%; "
            f"mean|d| {rep['mean_abs_diff']}{unit}, median {rep['median_abs_diff']}, p90 {rep['p90_abs_diff']}, "
            f"max {rep['max_abs_diff']}; within-subject SD {rep['within_subject_sd']}{unit}; "
            f"{rep['pairs_missing_a_value']} more pairs lack a value")


def summarize(doc: dict, value_key: str = "fit_value", tol: float = TOL_UNITS,
              max_min: float = 20.0) -> dict:
    rows = sorted([r for r in doc["rows"] if r.get("returned")], key=lambda r: r["t"])
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    n = len(rows)
    have = [r for r in rows if r.get(value_key) is not None]
    hr_err = [abs(r["fit_hr_bpm"] - r["sdk_hr_bpm"]) for r in rows
              if r.get("fit_hr_bpm") is not None and r.get("sdk_hr_bpm") is not None]
    return {
        "label": doc.get("label"), "value_key": value_key, "n_scans": n,
        "n_with_value": len(have), "coverage": round(len(have) / n, 3) if n else None,
        "card": pair_report(rows, value_key, max_min, tol),
        "card_hr": pair_report(rows, "fit_hr_bpm", max_min, 5.0),
        "sdk_hr": pair_report(rows, "sdk_hr_bpm", max_min, 5.0),
        "legacy_card_on_sdk_hr": pair_report(rows, lambda r: _legacy_card(r.get("sdk_hr_bpm")), max_min, tol),
        "card_hr_vs_sdk_mae": round(st.fmean(hr_err), 2) if hr_err else None,
        "card_hr_vs_sdk_within5": (sum(1 for e in hr_err if e <= 5.0), len(hr_err)),
        "card_hr_vs_sdk_off15": (sum(1 for e in hr_err if e > 15.0), len(hr_err)),
        "overall": _spread([r.get(value_key) for r in rows]),
        "groups": [{"group": g, "scans": [r["id"][9:15] for r in rs],
                    "values": [r.get(value_key) for r in rs],
                    "card": _spread([r.get(value_key) for r in rs]),
                    "card_hr": [r.get("fit_hr_bpm") for r in rs],
                    "sdk_hr": [None if r.get("sdk_hr_bpm") is None else round(r["sdk_hr_bpm"], 1) for r in rs],
                    "source": [r.get("fit_rate_source") for r in rs]}
                   for g, rs in sorted(groups.items())],
    }


def stats(args) -> None:
    for f in args.files:
        p = Path(f)
        if not p.exists():
            p = HERE / "data" / "eval_cache" / f
        doc = json.loads(p.read_text())
        s = summarize(doc, args.key, args.tol, args.pair_min)
        o = s["overall"]
        print(f"\n=== {s['label']}  ({p.name}; value = {args.key}; pairs <= {args.pair_min:g} min apart)")
        print(f"coverage      {s['n_with_value']}/{s['n_scans']} = {s['coverage']}   all scans: mean {o.get('mean')} "
              f"sd {o.get('sd')} min {o.get('min')} max {o.get('max')} cv {o.get('cv_pct')}%")
        print(f"card          {_fmt(s['card'], args.tol)}")
        print(f"card HR input {_fmt(s['card_hr'], 5.0, ' bpm')}")
        print(f"SDK HR        {_fmt(s['sdk_hr'], 5.0, ' bpm')}")
        print(f"legacy card on SDK HR (counterfactual): {_fmt(s['legacy_card_on_sdk_hr'], args.tol)}")
        print(f"card HR vs SDK HR: MAE {s['card_hr_vs_sdk_mae']} bpm, within 5 bpm "
              f"{s['card_hr_vs_sdk_within5'][0]}/{s['card_hr_vs_sdk_within5'][1]}, "
              f"off by > 15 bpm {s['card_hr_vs_sdk_off15'][0]}/{s['card_hr_vs_sdk_off15'][1]}")
        for g in s["groups"]:
            print(f"  g{g['group']} {','.join(g['scans'])}: card {g['values']} range {g['card'].get('range')} | "
                  f"cardHR {g['card_hr']} | sdkHR {g['sdk_hr']} | src {g['source']}")
        if args.json:
            Path(args.json).write_text(json.dumps(s, indent=1, default=str))


# ------------------------------------------------------- tracking-sheet history
def _legacy_card(hr):
    """The HR-only logistic the card used through 2026-09-20 (HR_REF 72, spread 14)."""
    return None if hr is None else 100.0 / (1.0 + math.exp(-(72.0 - hr) / 14.0))


def sheet_rows(csv_path: str, group_min: float) -> list:
    """Rows of a tracking-sheet CSV export, with display groups: consecutive
    scans from the SAME device (User Agent) no more than `group_min` apart."""
    import csv
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                t = datetime.strptime(r.get("Timestamp (UTC)", ""), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            rows.append({"t": t, "ua": r.get("User Agent") or "", "outcome": r.get("Outcome"),
                         "fit": _num(r.get("Fitness")), "fit_status": r.get("Fit Status"),
                         "fit_tier": r.get("Fit Tier"), "fit_hr": _num(r.get("Fit HR")),
                         "ref_hr": _num(r.get("Ref HR")), "pulse": _num(r.get("Pulse bpm")),
                         "rate_method": r.get("Rate Method"), "build": r.get("Build"),
                         "fit_reason": r.get("Fit Reason")})
    rows.sort(key=lambda r: (r["ua"], r["t"]))
    g, prev = -1, None
    for r in rows:
        if prev is None or prev["ua"] != r["ua"] or \
                (r["t"] - prev["t"]).total_seconds() > group_min * 60:
            g += 1
        r["group"], prev = g, r
    return rows


def sheet(args) -> None:
    for f in args.files:
        rows = sheet_rows(f, args.pair_min)
        if args.since:
            rows = [r for r in rows if r["t"] >= datetime.strptime(args.since, "%Y-%m-%d")]
        n = len(rows)
        done = [r for r in rows if r["ref_hr"] is not None]        # the SDK scan completed
        have = [r for r in rows if r["fit"] is not None]
        dev = lambda r: r["ua"]                                     # noqa: E731
        err = [abs(r["fit_hr"] - r["ref_hr"]) for r in rows
               if r["fit_hr"] is not None and r["ref_hr"] is not None]
        print(f"\n=== {Path(f).name}: {n} scans; repeat pairs = same device, <= {args.pair_min:g} min apart")
        print(f"coverage      card value on {len(have)}/{n} scans = {len(have) / max(n, 1):.2f}; on scans the SDK "
              f"completed (Ref HR present) {sum(1 for r in done if r['fit'] is not None)}/{len(done)}")
        print(f"card          {_fmt(pair_report(rows, 'fit', args.pair_min, args.tol, dev), args.tol)}")
        print(f"card HR input {_fmt(pair_report(rows, 'fit_hr', args.pair_min, 5.0, dev), 5.0, ' bpm')}")
        print(f"reference HR  {_fmt(pair_report(rows, 'ref_hr', args.pair_min, 5.0, dev), 5.0, ' bpm')}"
              f"   <- what the pulse really did between scans")
        print(f"legacy card on the reference HR (counterfactual: the estimator's own share): "
              f"{_fmt(pair_report(rows, lambda r: _legacy_card(r['ref_hr']), args.pair_min, args.tol, dev), args.tol)}")
        if err:
            print(f"card HR vs reference HR: MAE {st.fmean(err):.1f} bpm, within 5 bpm "
                  f"{sum(1 for e in err if e <= 5)}/{len(err)}, off by > 15 bpm {sum(1 for e in err if e > 15)}/{len(err)}")
        if args.verbose:
            groups = {}
            for r in rows:
                groups.setdefault(r["group"], []).append(r)
            for g, rs in sorted(groups.items(), key=lambda kv: kv[1][0]["t"]):
                if len(rs) < 2:
                    continue
                rnd = lambda v, k=0: None if v is None else round(v, k)   # noqa: E731
                print(f"  {rs[0]['t']:%m-%d %H:%M} n={len(rs)} card {[rnd(r['fit'], 1) for r in rs]} | "
                      f"cardHR {[rnd(r['fit_hr']) for r in rs]} | refHR {[rnd(r['ref_hr']) for r in rs]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    hp = sub.add_parser("sheet", help="repeatability from a tracking-sheet CSV export")
    hp.add_argument("files", nargs="+")
    hp.add_argument("--pair-min", type=float, default=20.0, help="max minutes between a repeat pair")
    hp.add_argument("--tol", type=float, default=TOL_UNITS)
    hp.add_argument("--since", default=None)
    hp.add_argument("--verbose", "-v", action="store_true")
    hp.set_defaults(fn=sheet)
    rp = sub.add_parser("replay")
    rp.add_argument("--repo", default=str(HERE))
    rp.add_argument("--corpus", default=str(HERE / "data" / "eval_corpus"))
    rp.add_argument("--label", default="run")
    rp.add_argument("--launch", choices=["production", "staging"], default="staging")
    rp.add_argument("--window-s", type=float, default=70.0)
    rp.add_argument("--scale", default=None, help="AFIB_SCALE, e.g. 640x480")
    rp.add_argument("--group-min", type=float, default=20.0)
    rp.add_argument("--no-route", action="store_true")
    rp.add_argument("--no-reference", action="store_true",
                    help="send no ref_hr: the card stands on the clip's own rate (the standalone scan)")
    rp.add_argument("--participant", default=None, help="JSON the client would send")
    rp.add_argument("--only", nargs="*", default=None)
    rp.add_argument("--out", default=None)
    rp.set_defaults(fn=replay)
    sp = sub.add_parser("stats")
    sp.add_argument("files", nargs="+")
    sp.add_argument("--key", default="fit_value")
    sp.add_argument("--pair-min", type=float, default=20.0, help="max minutes between a repeat pair")
    sp.add_argument("--tol", type=float, default=TOL_UNITS)
    sp.add_argument("--json", default=None)
    sp.set_defaults(fn=stats)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
