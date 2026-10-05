"""Repeatability of the Arterial Stiffness card on the retained phone scans.

`cache` replays every retained clip through the SAME job the service runs
(trim -> downscale -> pipeline -> cards; window_s=70, capture_profile=consumer,
staging launch overrides) ONCE and stores what the stiffness path consumes:
the per-region RGB traces, the capture clock, the fused beat lattice, the
evidence block and the card the current code publishes - plus the ShenAI
sidecar's own live-frame PPG and beat train. Everything downstream of the
traces (wide-band POS, beat windows, ensemble, fiducials, estimator) is then
recomputable in milliseconds, so a candidate estimator is scored on exactly
the inputs the deployed one saw, without decoding video again.

`stats` reports the published card's scan-to-scan spread per repeat group
(scans of one person closer than --pair-min minutes).

Corpus files are never handed to the pipeline: each run works on a copy
(measure_video trims its input in place).

Usage:
  python scripts/stiffness_repeatability.py cache            # ~1 min per clip
  python scripts/stiffness_repeatability.py stats
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / "data" / "eval_cache" / "stiffness"
# What the staging service launches with (GET /healthz -> launch_overrides).
STAGING_ENV = {"AFIB_MAX_COLLAPSED_FRACTION": "0.05", "AFIB_TRACE_PATH": "0",
               "AFIB_CONFIG_OVERRIDES": json.dumps({
                   "decision.classifier": "model_a",
                   "decision.model_a_path": "models/model_a_v02_45s.json"})}


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


def corpus_records(corpus: Path, include_untimed: bool) -> list:
    out = []
    for clip in sorted(corpus.iterdir()):
        if clip.suffix.lower() not in (".webm", ".avi", ".mp4", ".mkv"):
            continue
        t = _scan_time(clip.name)
        side = clip.with_name(clip.name + ".shenai.json")
        if t is None and not include_untimed:
            continue
        out.append({"id": clip.name.split(".")[0], "clip": clip, "t": t,
                    "sidecar": side if side.exists() else None})
    return out


def _card(doc: dict) -> dict:
    items = {i.get("key"): i for i in (doc.get("biomarkers") or {}).get("items", [])
             if isinstance(i, dict)}
    it = items.get("arterial_stiffness") or {}
    raw = it.get("raw") if isinstance(it.get("raw"), dict) else {}
    return {"status": it.get("status"), "value": _num(it.get("value")),
            "unit": it.get("unit"), "band": it.get("band"), "tier": it.get("tier"),
            "reason": it.get("reason"), "metric": it.get("metric"),
            "raw_name": raw.get("name"), "raw_value": _num(raw.get("value")),
            "tier_reasons": list(it.get("tier_reasons") or [])}


def cache(args) -> None:
    for k, v in STAGING_ENV.items():
        os.environ.setdefault(k, v)
    if args.scale:
        os.environ["AFIB_SCALE"] = args.scale
    sys.path.insert(0, str(HERE))
    os.chdir(HERE)
    from app import measure_api as api                 # noqa: E402 (after env + path)
    from preprocessing.roi import ROI_NAMES

    CACHE.mkdir(parents=True, exist_ok=True)
    records = corpus_records(Path(args.corpus).expanduser().resolve(), args.all)
    if args.only:
        records = [r for r in records if any(s in r["id"] for s in args.only)]
    for r in records:
        out = CACHE / f"{r['id']}.npz"
        if out.exists() and not args.force:
            print(f"[as] {r['id'][:22]} cached", flush=True)
            continue
        raw = json.loads(r["sidecar"].read_text()) if r["sidecar"] else {}
        ref = raw.get("reference") or {}
        d = tempfile.mkdtemp(prefix="as-repeat-")
        t0 = time.perf_counter()
        try:
            part = Path(d) / f"replay-{r['id'][:24]}"
            part.mkdir()
            v = part / f"scan{r['clip'].suffix}"
            shutil.copy(r["clip"], v)
            header = {"capture_profile": "consumer", "window_s": float(args.window_s),
                      "session": "replay"}
            if _num(ref.get("heart_rate_bpm")) is not None:
                header["reference"] = {"ref_hr": ref["heart_rate_bpm"], "ref_source": "shenai"}
            doc, det = api.measure_video_details(
                str(v), manifest=api.MeasureHandler._build_manifest(header),
                window_s=header["window_s"], reference=header.get("reference"))
            ing, lat = det.get("ingest"), det.get("lattice")
            meta = {"id": r["id"], "t": r["t"].isoformat() if r["t"] else None,
                    "outcome": doc.get("outcome"), "card": _card(doc),
                    "evidence": det.get("evidence"), "min_conf": det.get("min_conf"),
                    "sqi": _num(doc.get("signal_quality_index")),
                    "pulse_bpm": _num(doc.get("mean_pulse_rate_bpm")),
                    "sdk": ref, "trim": doc.get("trim"), "downscale": doc.get("downscale"),
                    "clock": doc.get("clock"), "wall_s": round(time.perf_counter() - t0, 1),
                    "hemodynamics": (doc.get("biomarkers") or {}).get("hemodynamics")}
            arrays = {}
            if ing is not None and getattr(ing, "traces", None):
                arrays["ts"] = np.asarray(ing.timestamps_s, float)
                for roi in ROI_NAMES:
                    arrays[f"rgb_{roi}"] = np.asarray(ing.traces[roi], float)
                meta["fps"] = float(ing.meta.measured_fps_mean)
                meta["width"], meta["height"] = ing.meta.width, ing.meta.height
            if lat is not None:
                arrays["beat_t"] = np.asarray(lat.beat_t_s, float)
                arrays["beat_conf"] = np.asarray(lat.beat_confidence, float)
                arrays["beat_agree"] = np.asarray(lat.beat_agreement, float)
                for roi, bt in (lat.per_roi_times or {}).items():
                    arrays[f"roibeats_{roi}"] = np.asarray(bt, float)
            if raw.get("ppg"):
                arrays["sdk_ppg"] = np.asarray(raw["ppg"].get("signal") or [], float)
                meta["sdk_ppg_fs"] = _num(raw["ppg"].get("fs_hz"))
            hb = [b for b in (raw.get("heartbeats") or []) if isinstance(b, dict)]
            arrays["sdk_beat_start"] = np.asarray([b.get("start_location_sec") for b in hb], float)
            arrays["sdk_beat_end"] = np.asarray([b.get("end_location_sec") for b in hb], float)
            np.savez_compressed(out, meta=json.dumps(meta, default=str), **arrays)
            c = meta["card"]
            print(f"[as] {r['id'][:22]} {meta['outcome']} card={c['value']} {c['tier']} "
                  f"raw={c['raw_name']}={c['raw_value']} fps={meta.get('fps')} "
                  f"({meta['wall_s']} s)", flush=True)
        except Exception as e:                             # noqa: BLE001 - a crash IS the finding
            print(f"[as] {r['id'][:22]} FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)


def load_cache(only_timed: bool = True) -> list:
    rows = []
    for f in sorted(CACHE.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if only_timed and not meta.get("t"):
            continue
        meta["arrays"] = {k: z[k] for k in z.files if k != "meta"}
        rows.append(meta)
    return rows


def pair_stats(rows: list, get, max_min: float) -> dict:
    """|a-b| over every pair of scans no more than `max_min` apart (pairwise, not
    chained) and the within-subject SD they imply: sqrt(mean(d^2)/2)."""
    import itertools
    d, missing = [], 0
    for a, b in itertools.combinations(rows, 2):
        ta, tb = datetime.fromisoformat(a["t"]), datetime.fromisoformat(b["t"])
        if abs((tb - ta).total_seconds()) > max_min * 60:
            continue
        va, vb = get(a), get(b)
        if va is None or vb is None:
            missing += 1
        else:
            d.append(abs(va - vb))
    out = {"pairs": len(d), "pairs_missing": missing}
    if d:
        out.update(mean_abs=float(np.mean(d)), median_abs=float(np.median(d)),
                   max_abs=float(np.max(d)),
                   wsd=float(math.sqrt(np.mean(np.square(d)) / 2.0)))
    return out


def stats(args) -> None:
    rows = load_cache()
    have = [r for r in rows if r["card"]["value"] is not None]
    print(f"scans {len(rows)}  card computed {len(have)}")
    for r in rows:
        c = r["card"]
        print(f"  {r['t'][5:16]} {r['outcome']:<11} card={c['value']} {c['tier'] or c['status']} "
              f"{c['raw_name']}={c['raw_value']} sqi={r['sqi']} "
              f"coh={_num((r['evidence'] or {}).get('cross_roi_coherence'))} "
              f"tp={_num((r['evidence'] or {}).get('timing_precision_ms'))} "
              f"sdkHR={_num((r['sdk'] or {}).get('heart_rate_bpm'))}")
    p = pair_stats(rows, lambda r: r["card"]["value"], args.pair_min)
    print(f"card repeat pairs <= {args.pair_min:g} min: {json.dumps(p)}")


RECORDED_ONLY_KEYS = ("pulse_harmonics_sdk", "pulse_harmonics_clip", "pulse_harmonics_traces")
# Wall-clock and temp-dir noise: differs between any two runs of the SAME code.
VOLATILE = ("timing", "wall_s", "waited_s", "analysis_s", "elapsed_ms", "diagnostic_ms", "path", "note")


def dump(args) -> None:
    """Every retained clip through the job the SERVICE runs (_run_assembled, the ShenAI
    snapshot held in memory as a real request's is), in the tree `--repo` names, with the
    WHOLE response written out. Run it on a clean export of HEAD and on the working tree,
    then `diff` the two: a recorded-only change may differ in its own debug keys, nowhere
    else. (Comparing against `cache` instead is apples to oranges: that is the video path
    alone, and the ShenAI route legitimately changes the outcome of some scans.)"""
    repo = Path(args.repo).expanduser().resolve()
    for k, v in STAGING_ENV.items():
        os.environ.setdefault(k, v)
    sys.path.insert(0, str(repo))
    os.chdir(repo)                                     # configs/ and models/ resolve from the tree
    from app import measure_api as api                 # noqa: E402

    assert Path(api.__file__).resolve().is_relative_to(repo), f"imported {api.__file__}, not {repo}"
    docs = {}
    for r in corpus_records(Path(args.corpus).expanduser().resolve(), False):
        if r["sidecar"] is None:
            continue
        raw = json.loads(r["sidecar"].read_text())
        ref = raw.get("reference") or {}
        d = tempfile.mkdtemp(prefix="as-dump-")
        try:
            uid = f"replay-{r['id'][:24]}"
            part = Path(d) / uid
            part.mkdir()
            v = part / f"scan{r['clip'].suffix}"
            shutil.copy(r["clip"], v)
            header = {"capture_profile": "consumer", "window_s": float(args.window_s),
                      "session": "replay", "upload_id": uid}
            if _num(ref.get("heart_rate_bpm")) is not None:
                header["reference"] = {"ref_hr": ref["heart_rate_bpm"], "ref_source": "shenai"}
            api._hold_signals(uid, raw)
            try:
                docs[r["id"]] = api.MeasureHandler._run_assembled(str(v), header)
            finally:
                api._drop_signals(uid)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        doc = docs[r["id"]]
        print(f"[dump] {r['id'][:16]} outcome={doc.get('outcome')} afib={doc.get('afib_result')} "
              f"source={doc.get('rhythm_source')} card={_card(doc)['value']}", flush=True)
    out = Path(args.out).expanduser().resolve()
    out.write_text(api._json_bytes(docs).decode())     # the service's own serializer: what a client receives
    print(f"[dump] {len(docs)} responses from {repo} -> {out}")


def _flatten(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, o


def diff(args) -> None:
    a, b = (json.loads(Path(p).expanduser().read_text()) for p in (args.before, args.after))
    assert set(a) == set(b), f"different scans: {set(a) ^ set(b)}"
    unexpected, recorded, volatile, provenance = [], 0, 0, 0
    for sid in sorted(a):
        fa, fb = dict(_flatten(a[sid])), dict(_flatten(b[sid]))
        for path in sorted(set(fa) | set(fb)):
            if fa.get(path, "<absent>") == fb.get(path, "<absent>"):
                continue
            if any(f"debug.{k}" in path for k in RECORDED_ONLY_KEYS):
                recorded += 1
            elif path == "code_commit":
                # WHERE the code ran, not what it computed: a git-archive export reports
                # "no-git" (as Railway does), a checkout its commit. Counted on its own
                # line so it is never mistaken for, or hidden among, timing noise.
                provenance += 1
            elif any(tok in path.split(".")[-1] or path.startswith(tok) for tok in VOLATILE):
                volatile += 1
            else:
                unexpected.append((sid[:16], path, fa.get(path, "<absent>"), fb.get(path, "<absent>")))
    n = sum(len(dict(_flatten(v))) for v in a.values())
    print(f"{len(a)} scans, {n} response fields compared")
    print(f"  differing only in the recorded-only debug keys: {recorded} fields")
    print(f"  wall-clock / temp-path noise:                   {volatile} fields")
    print(f"  code_commit (git checkout vs non-git export):   {provenance} fields")
    print(f"  ANY OTHER DIFFERENCE:                           {len(unexpected)}")
    for u in unexpected[:40]:
        print("   ", u)
    if unexpected:
        raise SystemExit("the change is NOT recorded-only")
    print("PASS: every published field is identical; the change is recorded-only")


def verify(args) -> None:
    """The recorded descriptors on every retained clip, through the service's job."""
    for k, v in STAGING_ENV.items():
        os.environ.setdefault(k, v)
    sys.path.insert(0, str(HERE))
    os.chdir(HERE)
    from app import measure_api as api                 # noqa: E402
    from app import result_sheet

    before = {r["id"]: r for r in load_cache()}
    moved, rows = [], []
    for r in corpus_records(Path(args.corpus).expanduser().resolve(), False):
        if r["id"] not in before or r["sidecar"] is None:
            continue
        raw = json.loads(r["sidecar"].read_text())
        ref = raw.get("reference") or {}
        d = tempfile.mkdtemp(prefix="as-verify-")
        try:
            uid = f"replay-{r['id'][:24]}"
            part = Path(d) / uid
            part.mkdir()
            v = part / f"scan{r['clip'].suffix}"
            shutil.copy(r["clip"], v)
            header = {"capture_profile": "consumer", "window_s": float(args.window_s),
                      "session": "replay", "upload_id": uid}
            if _num(ref.get("heart_rate_bpm")) is not None:
                header["reference"] = {"ref_hr": ref["heart_rate_bpm"], "ref_source": "shenai"}
            api._hold_signals(uid, raw)
            try:
                doc = api.MeasureHandler._run_assembled(str(v), header)
            finally:
                api._drop_signals(uid)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        # The CARD only: the cache is the video path alone, and the ShenAI route this job
        # also runs may legitimately change the OUTCOME. `dump` + `diff` is the full proof.
        card, was = _card(doc), before[r["id"]]["card"]
        same = card == was
        if not same:
            moved.append((r["id"], was, card))
        dbg = doc.get("debug") or {}
        row = result_sheet.row_from_doc(doc)
        rows.append({"id": r["id"], "t": r["t"].isoformat(), "same": same,
                     "sdk": dbg.get("pulse_harmonics_sdk"), "clip": dbg.get("pulse_harmonics_clip"),
                     "sheet": {k: row[k] for k in row if k.startswith("H2 ")}})
        s, c = rows[-1]["sdk"] or {}, rows[-1]["clip"] or {}
        print(f"[verify] {r['id'][:16]} card {'UNCHANGED' if same else 'MOVED'} ({card['value']}) | "
              f"sdk {s.get('state')} phi2={s.get('phi2_deg')} r2={s.get('r2')} p={s.get('p_value')} | "
              f"clip {c.get('state')} phi2={c.get('phi2_deg')} p={c.get('p_value')}", flush=True)
    (CACHE / "verify_record_only.json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"\n{len(rows)} scans replayed; cards moved: {len(moved)}")
    for m in moved:
        print("  MOVED", m)
    for src in ("sdk", "clip"):
        ok = [x[src] for x in rows if (x[src] or {}).get("state") == "ok"]
        if not ok:
            continue
        ph = np.radians([o["phi2_deg"] for o in ok])
        R = abs(np.mean(np.exp(1j * ph)))
        print(f"{src}: ok {len(ok)}/{len(rows)}; detectable (p<0.01) {sum(o['detectable'] for o in ok)}; "
              f"phi2 circular mean {np.degrees(np.angle(np.mean(np.exp(1j * ph)))):.0f} deg, "
              f"circular SD {np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))):.0f} deg, R={R:.2f} "
              f"(random phases: R~{1/np.sqrt(len(ok)):.2f})")
    print("sheet cells, last scan:", rows[-1]["sheet"] if rows else None)
    if moved:
        raise SystemExit("a recorded-only change moved a published card")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    vf = sub.add_parser("verify")
    vf.add_argument("--corpus", default=str(HERE / "data" / "eval_corpus"))
    vf.add_argument("--window-s", type=float, default=70.0)
    vf.set_defaults(fn=verify)
    dp = sub.add_parser("dump")
    dp.add_argument("--repo", default=str(HERE), help="the tree whose service code runs (e.g. a git-archive export of HEAD)")
    dp.add_argument("--corpus", default=str(HERE / "data" / "eval_corpus"))
    dp.add_argument("--window-s", type=float, default=70.0)
    dp.add_argument("--out", required=True)
    dp.set_defaults(fn=dump)
    df = sub.add_parser("diff")
    df.add_argument("before")
    df.add_argument("after")
    df.set_defaults(fn=diff)
    c = sub.add_parser("cache")
    c.add_argument("--corpus", default=str(HERE / "data" / "eval_corpus"))
    c.add_argument("--window-s", type=float, default=70.0)
    c.add_argument("--scale", default=None)
    c.add_argument("--only", nargs="*")
    c.add_argument("--all", action="store_true", help="also clips without a scan time in their name")
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cache)
    s = sub.add_parser("stats")
    s.add_argument("--pair-min", type=float, default=20.0)
    s.set_defaults(fn=stats)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
