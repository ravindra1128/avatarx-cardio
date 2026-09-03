"""
Noise-floor characterization (v0.7 Task 2) — the honesty task.

Camera pulse intervals are ECG R-R intervals PLUS pulse-transit and
pre-ejection jitter PLUS frame-timing quantization PLUS beat-detection
error. Before the product claims to measure anyone's rhythm it
quantifies its own floor, three ways:

  1. the timing-jitter BUDGET, propagated from first principles
     (features/regularity.timing_jitter_budget) and CALIBRATED against
     what metronomic references actually read through the pipeline;
  2. the MINIMUM DETECTABLE IRREGULARITY (MDI): the smallest true
     dispersion, as measured on the simultaneous ECG, that the camera
     path separates from a perfectly regular rhythm at a stated
     confidence and power — per fps, per SQI grade, per skin-tone
     group. The MDI is a PUBLISHED product parameter: it defines what
     the product can honestly claim to notice;
  3. FALSE IRREGULARITY FROM BEAT ERRORS: one missed beat inside a run
     inflates dispersion dramatically. The clean-run / no-repair
     architecture is supposed to contain that; this measures whether it
     does, as a function of beat-detection error rate, against the
     raw-series alternative it replaced.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from beats.detector import Beat, BeatSeries
from beats.ibi import clean_runs
from datasets.regularity_reference import (resolve_min_conf, classify_index,
                                           load_reference_definition)
from evaluation.afib_metrics import wilson_ci
from features.regularity import (DEFAULT_INTERPOLATION_GAIN,
                                 regularity_from_runs,
                                 timing_jitter_budget)

# ECG-RMSSD bins for the MDI search (ms). A cell's MDI is the smallest
# bin whose camera index clears the metronomic floor's p95 in at least
# `power` of its scans.
MDI_BINS_MS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0),
               (20.0, 30.0), (30.0, 50.0), (50.0, 1e9))
METRONOMIC_ECG_RMSSD_MS = 5.0     # a reference this regular is "regular"
MIN_SCANS_PER_BIN = 3
MIN_REFS_TO_COMPUTE_FLOOR = 3     # a quantile of fewer is not a floor


# ------------------------------------------------ 1. the jitter budget
def budget_table(fps_list=(24.0, 30.0, 60.0, 120.0),
                 interpolation_gain: float = DEFAULT_INTERPOLATION_GAIN
                 ) -> list:
    """The propagated budget per frame rate — recomputed, never typed
    in. The raw (gain 1.0) column is what naive peak picking would do;
    the calibrated column is what the pipeline's sub-sample refinement
    leaves."""
    out = []
    for fps in fps_list:
        raw = timing_jitter_budget(fps, interpolation_gain=1.0)
        cal = timing_jitter_budget(fps, interpolation_gain=interpolation_gain)
        out.append({"fps": float(fps), "frame_ms": raw["frame_ms"],
                    "raw_sigma_interval_ms": raw["sigma_interval_ms"],
                    "raw_rmssd_floor_ms": raw["rmssd_floor_ms"],
                    "calibrated_sigma_interval_ms": cal["sigma_interval_ms"],
                    "calibrated_rmssd_floor_ms": cal["rmssd_floor_ms"],
                    "interpolation_gain": interpolation_gain})
    return out


NOMINAL_FPS = (24.0, 25.0, 30.0, 50.0, 60.0, 120.0)
NOMINAL_FPS_TOLERANCE = 1.5


def nominal_fps(fps) -> float:
    """The capture's nominal frame-rate class. A measured 29.97 or 30.02
    is the 30 fps class; keying cells on the per-clip measured rate
    fragments real captures into cells nothing can fill (review
    finding). Outside every class the rate is kept, rounded."""
    f = float(fps)
    for c in NOMINAL_FPS:
        if abs(f - c) <= NOMINAL_FPS_TOLERANCE:
            return c
    return round(f, 1)


def _usable(r) -> bool:
    """A row the camera actually judged: an ACCEPT scan with an index.
    Rejected scans are still rows (they count as non-detections in the
    MDI power denominator) but never references (review finding)."""
    return (r.get("camera_index") is not None
            and str(r.get("scan_outcome", "ACCEPT")) == "ACCEPT")


def calibrate_interpolation_gain(rows) -> dict:
    """Measured gain = (camera RMSSD attributable to the CAMERA on
    metronomic references) / (raw quantization prediction), per nominal
    fps. The reference's own dispersion is removed in quadrature
    (camera^2 = ECG^2 + floor^2) — a "metronomic" reference is anything
    under 5 ms, not exactly 0 (review finding). None where no
    metronomic references exist — a gain must be measured, never
    assumed."""
    per = {}
    for r in rows:
        fps, ecg, cam = r.get("fps"), r.get("ecg_rmssd_ms"), \
            r.get("camera_rmssd_ms")
        if fps is None or ecg is None or cam is None or not _usable(r):
            continue
        if float(ecg) < METRONOMIC_ECG_RMSSD_MS:
            per.setdefault(nominal_fps(fps), []).append(
                (float(cam), float(ecg)))
    out = {}
    for fps, vals in sorted(per.items()):
        raw = timing_jitter_budget(fps, interpolation_gain=1.0)
        cam = np.asarray([c for c, _ in vals])
        ecg = np.asarray([e for _, e in vals])
        cam_only = np.sqrt(np.maximum(cam ** 2 - ecg ** 2, 0.0))
        med = float(np.median(cam_only))
        out[str(fps)] = {"n_metronomic_refs": len(vals),
                         "camera_rmssd_p50_ms": round(float(
                             np.median(cam)), 3),
                         "camera_rmssd_p95_ms": round(
                             float(np.percentile(cam, 95)), 3),
                         "camera_only_rmssd_p50_ms": round(med, 3),
                         "raw_prediction_ms": raw["rmssd_floor_ms"],
                         "interpolation_gain_measured": round(
                             med / raw["rmssd_floor_ms"], 4)}
    return out


def mdi_from_rows(rows, *, power: float = 0.80,
                  confidence: float = 0.95,
                  min_scans_per_bin: int = MIN_SCANS_PER_BIN,
                  min_regular_refs: int = 3) -> dict:
    """The MDI per (nominal fps, SQI grade, skin-tone group) cell.

    Floor = the `confidence` quantile of the camera index on ACCEPT
    scans whose ECG says metronomic (RMSSD < 5 ms) — the same fps, any
    grade/group, because a per-cell metronomic reference set is a
    luxury real data rarely affords (the fps floor is physics; grade/
    group modulate it and are reported per cell). MDI = the lowest
    ECG-RMSSD bin in which at least `power` of the cell's scans read
    ABOVE that floor, provided every better-populated bin above it does
    too (detection must be monotone in true dispersion — a first bin
    that happens to pass while higher ones fail is not a threshold).
    A scan the camera could not read is a NON-detection, in the
    denominator. A cell with no such bin, or a fps with no metronomic
    references, is "not characterized" — never a number.
    """
    q = 100.0 * confidence
    floors = {}
    with_ecg = [r for r in rows if r.get("fps") is not None
                and r.get("ecg_rmssd_ms") is not None]
    for fps in sorted({nominal_fps(r["fps"]) for r in with_ecg}):
        refs = [float(r["camera_index"]) for r in with_ecg
                if nominal_fps(r["fps"]) == fps and _usable(r)
                and float(r["ecg_rmssd_ms"]) < METRONOMIC_ECG_RMSSD_MS]
        floors[fps] = (None if len(refs) < min_regular_refs
                       else {"index_p95": float(np.percentile(refs, q)),
                             "n": len(refs)})
    cells = {}
    keys = sorted({(nominal_fps(r["fps"]), str(r.get("sqi_grade")),
                    str(r.get("fitzpatrick_group"))) for r in with_ecg})
    for fps, grade, fitz in keys:
        cell = [r for r in with_ecg
                if nominal_fps(r["fps"]) == fps
                and str(r.get("sqi_grade")) == grade
                and str(r.get("fitzpatrick_group")) == fitz]
        key = f"fps={fps:g}|sqi={grade}|fitz={fitz}"
        fl = floors.get(fps)
        if fl is None:
            cells[key] = {"mdi_ms": None, "n": len(cell),
                          "n_no_read": sum(1 for r in cell
                                           if not _usable(r)),
                          "reason": f"no metronomic references at "
                                    f"{fps:g} fps — floor unknown"}
            continue
        bins = []
        for lo, hi in MDI_BINS_MS:
            inb = [r for r in cell if lo <= float(r["ecg_rmssd_ms"]) < hi]
            if len(inb) < min_scans_per_bin:
                bins.append({"bin_ms": [lo, hi], "n": len(inb),
                             "detected_fraction": None})
                continue
            det = float(np.mean([
                _usable(r) and float(r["camera_index"]) > fl["index_p95"]
                for r in inb]))
            bins.append({"bin_ms": [lo, hi], "n": len(inb),
                         "n_no_read": sum(1 for r in inb
                                          if not _usable(r)),
                         "detected_fraction": round(det, 3),
                         "ecg_rmssd_median_ms": round(float(np.median(
                             [float(r["ecg_rmssd_ms"]) for r in inb])), 2)})
        rated = [b for b in bins if b["detected_fraction"] is not None
                 and b["bin_ms"][0] > 0.0]
        mdi, reason = None, "no ECG-dispersion bin reached the required " \
                            "power — not characterized"
        for i, b in enumerate(rated):
            if b["detected_fraction"] >= power:
                above = rated[i + 1:]
                if all(a["detected_fraction"] >= power for a in above):
                    mdi, reason = b["ecg_rmssd_median_ms"], None
                else:
                    failing = [a["bin_ms"] for a in above
                               if a["detected_fraction"] < power]
                    reason = (f"bin {b['bin_ms']} reaches power but a "
                              f"higher bin {failing[0]} does not — "
                              "detection is not monotone; not "
                              "characterized")
                break
        cells[key] = {"mdi_ms": mdi, "n": len(cell),
                      "n_no_read": sum(1 for r in cell if not _usable(r)),
                      "floor_index_p95": round(fl["index_p95"], 5),
                      "n_metronomic_refs": fl["n"], "bins": bins,
                      "reason": reason}
    return {"cells": cells, "floors_per_fps": {str(k): v for k, v in
                                               floors.items()},
            "power": power, "confidence": confidence,
            "min_regular_refs": min_regular_refs}


# ------------------------------------ 3. false irregularity from errors
def _series(times_s, confidences, fps):
    beats = [Beat(t_s=float(t), confidence=float(c), roi_agreement=1.0,
                  signal_quality=1.0, amplitude=1.0, prominence=1.0,
                  source_rois=["sim"]) for t, c in zip(times_s, confidences)]
    return BeatSeries(beats, float(fps),
                      float(times_s[-1] - times_s[0]) if len(times_s) > 1
                      else 0.0)


def inject_beat_errors(rr_s, *, miss_rate: float, false_rate: float,
                       fps: float, jitter_sigma_ms: float, rng,
                       false_confidence: float = 0.9) -> tuple:
    """Beat times from an RR series with camera timing jitter, then
    MISSED beats (dropped) and FALSE beats (inserted mid-interval). A
    false beat here is a CONFIDENT one — the case the confidence channel
    cannot catch and the short-pair splitter must."""
    t = np.concatenate([[0.0], np.cumsum(np.asarray(rr_s, float))])
    t = t + rng.normal(0.0, jitter_sigma_ms / 1000.0, t.size)
    keep = rng.random(t.size) >= miss_rate
    keep[0] = keep[-1] = True
    t = t[keep]
    conf = np.full(t.size, 0.95)
    extra = []
    for i in range(t.size - 1):
        if rng.random() < false_rate:
            extra.append(t[i] + rng.uniform(0.3, 0.7) * (t[i + 1] - t[i]))
    if extra:
        t = np.concatenate([t, np.asarray(extra)])
        conf = np.concatenate([conf, np.full(len(extra), false_confidence)])
    order = np.argsort(t)
    return t[order], conf[order]


def beat_error_study(*, fps_list=(30.0, 60.0),
                     error_rates=(0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40),
                     n_trials: int = 30, duration_s: float = 60.0,
                     bpm: float = 70.0, true_rmssd_ms: float = 8.0,
                     definition=None, gates_path=None, config=None,
                     seed: int = 20260901) -> dict:
    """P(read irregular | truly regular) as beat-detection error rate
    rises, for the clean-run architecture and for the raw series it
    replaced. The truly-regular source has a small physiological
    dispersion (default RMSSD 8 ms) so the test is not trivially easy.

    Two statistics are reported per path because they fail differently:
    the index (median |succ diff| / median IBI) is robust to SPARSE
    beat errors by construction — a median ignores a minority of wild
    differences — so on the raw series it only tips past the published
    threshold when errors are no longer sparse (≈30 % of beats). RMSSD
    has no such protection: one missed beat inside a 60 s window
    multiplies it. The clean-run architecture is what keeps every
    dispersion statistic honest, not just the index.
    """
    d = definition or load_reference_definition(gates_path)
    from configs import load_config
    cfg = config or load_config()
    rc = cfg["runs"]
    lo_ms, hi_ms = rc["ibi_physiologic_ms"]
    rng = np.random.default_rng(seed)
    base = 60.0 / bpm
    out = {"fps": {}, "definition": {
        "true_rmssd_ms": true_rmssd_ms, "bpm": bpm,
        "duration_s": duration_s, "n_trials": n_trials,
        "irregular_if_index_at_least":
            float(d["irregular_if_index_at_least"])}}
    for fps in fps_list:
        sig = timing_jitter_budget(fps)["sigma_beat_ms"]
        rows = []
        for rate in error_rates:
            n_clean_irr = n_raw_irr = n_clean_read = 0
            clean_idx, raw_idx, clean_rmssd, raw_rmssd = [], [], [], []
            for _ in range(n_trials):
                n = int(duration_s / base) + 4
                rr = np.clip(base + rng.normal(
                    0.0, true_rmssd_ms / np.sqrt(2.0) / 1000.0, n), 0.3, 2.0)
                rr = rr[np.cumsum(rr) <= duration_s]
                t, conf = inject_beat_errors(
                    rr, miss_rate=rate / 2.0, false_rate=rate / 2.0,
                    fps=fps, jitter_sigma_ms=sig, rng=rng)
                # clean-run architecture: the production path
                rs = clean_runs(_series(t, conf, fps),
                                min_conf=resolve_min_conf(cfg),
                                min_run_beats=int(rc["min_run_beats"]),
                                max_physiologic_ibi_ms=float(hi_ms),
                                min_physiologic_ibi_ms=float(lo_ms),
                                missed_beat_ratio=float(rc["missed_ratio"]))
                reg = regularity_from_runs(rs.runs, rs.run_confidences,
                                           run_times=rs.run_times, fps=fps)
                c_cls = classify_index(reg.index.get("value"),
                                       reg.n_intervals, d)
                # raw series: every interval, no splitting, no repair
                raw = regularity_from_runs([np.diff(t) * 1000.0], fps=fps)
                r_cls = classify_index(raw.index.get("value"),
                                       raw.n_intervals, d)
                if c_cls != "indeterminate":
                    n_clean_read += 1
                    n_clean_irr += int(c_cls == "irregular")
                    clean_idx.append(reg.index["value"])
                    clean_rmssd.append(reg.dispersion["rmssd_ms"])
                n_raw_irr += int(r_cls == "irregular")
                if raw.index.get("value") is not None:
                    raw_idx.append(raw.index["value"])
                    raw_rmssd.append(raw.dispersion["rmssd_ms"])
            rows.append({
                "beat_error_rate": rate,
                "clean_runs": {
                    "false_irregular_rate": (round(n_clean_irr /
                                                   n_clean_read, 4)
                                             if n_clean_read else None),
                    "ci95": ([round(x, 4) for x in
                              wilson_ci(n_clean_irr, n_clean_read)]
                             if n_clean_read else None),
                    "read_rate": round(n_clean_read / n_trials, 4),
                    "index_p50": (round(float(np.median(clean_idx)), 5)
                                  if clean_idx else None),
                    "rmssd_p50_ms": (round(float(np.median(clean_rmssd)), 2)
                                     if clean_rmssd else None)},
                "raw_series": {
                    "false_irregular_rate": round(n_raw_irr / n_trials, 4),
                    "index_p50": (round(float(np.median(raw_idx)), 5)
                                  if raw_idx else None),
                    "rmssd_p50_ms": (round(float(np.median(raw_rmssd)), 2)
                                     if raw_rmssd else None)}})
        out["fps"][str(float(fps))] = rows
    return out


# ------------------------------------------------------- the scoreboard
def regularity_floor_report(rows, *, runs_root=None, gates_path=None,
                            signal_domain: str = "synthetic",
                            beat_error_trials: int = 30) -> dict:
    """Everything the R1 gate reads, from paired rows: the budget, the
    measured gain, the MDI table, the beat-error study."""
    from evaluation.regularity_gates import (append_scoreboard,
                                             DEFAULT_RUNS,
                                             load_regularity_gates)
    # the pre-registered R1 parameters are READ here and recorded with
    # the run, so the gate can check the floor was computed at the
    # published power and confidence (review finding: they were never
    # read)
    t1 = load_regularity_gates(gates_path).get("r1_noise_floor") or {}
    params = {"mdi_power": float(t1.get("mdi_power", 0.80)),
              "mdi_confidence": float(t1.get("mdi_confidence", 0.95)),
              "min_regular_scans_per_cell": int(
                  t1.get("min_regular_scans_per_cell", 10)),
              # the statistical minimum to COMPUTE a floor at all; the
              # gate separately requires min_regular_scans_per_cell
              "min_refs_to_compute_floor": MIN_REFS_TO_COMPUTE_FLOOR}
    gain = calibrate_interpolation_gain(rows)
    mdi = mdi_from_rows(rows, power=params["mdi_power"],
                        confidence=params["mdi_confidence"],
                        min_regular_refs=MIN_REFS_TO_COMPUTE_FLOOR)
    errors = beat_error_study(n_trials=beat_error_trials,
                              gates_path=gates_path)
    doc = {"kind": "floor",
           "signal_domain": signal_domain,
           "parameters": params,
           "n_paired_scans": sum(1 for r in rows
                                 if r.get("ecg_rmssd_ms") is not None
                                 and r.get("camera_rmssd_ms") is not None),
           "n_scans": len(rows),
           "n_rejected_scans": sum(1 for r in rows if not _usable(r)),
           "budget": budget_table(),
           "interpolation_gain": gain,
           "mdi": mdi,
           "beat_errors": errors}
    run_id = "rfloor-" + hashlib.sha256(json.dumps(
        {"ids": sorted(str(r.get("recording_id")) for r in rows),
         "domain": signal_domain}, sort_keys=True).encode()).hexdigest()[:12]
    root = pathlib.Path(runs_root or DEFAULT_RUNS)
    (root / run_id).mkdir(parents=True, exist_ok=True)
    (root / run_id / "floor.json").write_text(json.dumps(doc, indent=1,
                                                        default=str))
    append_scoreboard({"kind": "floor", "run_id": run_id,
                       "signal_domain": signal_domain,
                       "gates_version": load_regularity_gates(
                           gates_path).get("gates_version"),
                       "parameters": params,
                       "mdi_cells": {k: v.get("mdi_ms")
                                     for k, v in mdi["cells"].items()},
                       "interpolation_gain": gain,
                       "false_irregular_at_5pct": {
                           fps: next((r["clean_runs"]["false_irregular_rate"]
                                      for r in rs
                                      if abs(r["beat_error_rate"] - 0.05)
                                      < 1e-9), None)
                           for fps, rs in errors["fps"].items()}},
                      runs_root=runs_root)
    doc["run_id"] = run_id
    doc["run_dir"] = str(root / run_id)
    return doc
