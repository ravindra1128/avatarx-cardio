"""
Diagnose a kept live scan (or any video) through the production pipeline
and print the full trace: capture → tracking → per-ROI beats → fusion →
calibration → clean runs → features → SQI → beat evidence → every decision
gate → the rule that fired.

    python3 scripts/diagnose_scan.py /tmp/avatarx_live/<id>/scan_<id>.avi
    python3 scripts/diagnose_scan.py --list      # kept live scans

This is the tool behind the v0.1.2 root-cause investigation: a result is
only trustworthy if it can be re-derived from the recording, stage by
stage. Nothing here recomputes a verdict — it calls
`inference.pipeline.run_with_details`, exactly what the demo and
`cli.py process` call.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import glob
import json

import numpy as np


def list_scans():
    for v in sorted(glob.glob("/tmp/avatarx_live/*/scan_*.avi")):
        p = pathlib.Path(v)
        print(f"{p.stat().st_size / 1e6:7.1f} MB  {v}")


def diagnose(path: str, profile: str = "consumer") -> dict:
    from inference.pipeline import run_with_details
    from capture.video_reader import load_timestamp_sidecar
    from rppg.pos import pos_pulse
    from beats.detector import detect_beats_single_roi
    from preprocessing.roi import ROI_NAMES

    manifest = {"capture_profile": profile, "assume_rig_locks": False}
    side = None
    try:
        side = load_timestamp_sidecar(path)
    except IOError as e:
        print("! timestamp sidecar unusable:", e)
    res, det = run_with_details(path, manifest=manifest)
    print("=" * 78)
    print(f"{path}")
    print(f"RESULT: {res.outcome.value} {res.predicted_class or ''}   "
          f"bpm={res.mean_pulse_rate_bpm}   sqi={res.signal_quality_index}   "
          f"beats={res.usable_beats}   analysed_s={res.analysed_seconds}")
    print("user-facing:", res.user_facing_text())
    if res.no_read_reasons:
        print("why no result:")
        for w in res.no_read_reasons:
            print("   -", w)
    prov = {"model": res.model_version, "commit": res.code_commit,
            "calibration": res.calibration_version, "config": res.config_hash}
    print("provenance:", prov)

    ing = det.get("ingest")
    if ing is None or not ing.ok:
        print("ingest failed:", getattr(ing, "reasons", None))
        return {"result": res.outcome.value}
    m = ing.meta
    print("\n[capture] fps %.2f jitter %.1f ms  n=%d  dur %.2f s  lux~%.0f  "
          "codec %s  caveats %s" % (m.measured_fps_mean, m.measured_fps_jitter_ms,
                                    m.n_frames, m.duration_s, m.lux_proxy,
                                    m.codec_fourcc, ing.capture_caveats))
    if side is not None:
        d = np.diff(side)
        print("[clock]   sidecar n=%d, max gap %.0f ms, dropped-frame gaps(>50ms)=%d"
              % (side.size, d.max() * 1000, int((d > 0.05).sum())))
    t = ing.track
    print("[track]   %s  found %d/%d  stability %.2f" % (t.tracker, t.n_found,
                                                        t.n_frames, t.stability))
    fps = m.measured_fps_mean
    print("[per-ROI] single-ROI detections (before fusion):")
    for roi in ROI_NAMES:
        b = detect_beats_single_roi(pos_pulse(ing.traces[roi], fps), fps, roi)
        tt = np.array([x.t_s for x in b]); dd = np.diff(tt) * 1000 if tt.size > 1 else np.array([])
        c = det["sqi"].per_roi.get(roi, {})
        print("   %-9s %3d beats  median IBI %5.0f ms  IQR %4.0f  snr %.2f skew %.2f"
              % (roi, len(b), np.median(dd) if dd.size else float("nan"),
                 (np.percentile(dd, 75) - np.percentile(dd, 25)) if dd.size else 0,
                 c.get("snr_in_band", float("nan")), c.get("skewness", float("nan"))))
    fused, series, rs = det["fused"], det["series"], det["runset"]
    agree = np.array([b.roi_agreement for b in fused.beats])
    print("[fusion]  %d beats; ROI-agreement histogram %s; raw conf mean %.2f; "
          "calibrated>=0.5: %d" % (len(fused.beats),
                                  {float(a): int((agree == a).sum()) for a in sorted(set(agree))},
                                  fused.confidences().mean() if len(fused.beats) else float("nan"),
                                  int((series.confidences() >= 0.5).sum())))
    ibi = np.diff(fused.times()) * 1000
    if ibi.size:
        med = np.median(ibi)
        print("          fused IBIs (ms): %s" % np.round(ibi).astype(int).tolist())
        print("          median %.0f; ≈½: %d  ≈2×: %d  within ±10%%: %.2f"
              % (med, int((np.abs(ibi / med - 0.5) < 0.15).sum()),
                 int((np.abs(ibi / med - 2.0) < 0.25).sum()),
                 float(np.mean(np.abs(ibi / med - 1) < 0.10))))
    print("[runs]    kept %d/%d  n_runs %d  dropout %.2f  missed-splits %d  "
          "false-pair-splits %d  split_fraction %.2f  coverage %.2f"
          % (rs.kept_beats, rs.total_beats, rs.n_runs, rs.dropout_rate,
             rs.n_missed_splits, rs.n_false_pair_splits, rs.split_fraction,
             det["coverage"]))
    for i, r in enumerate(rs.runs):
        print("          run%d (%d): %s" % (i, r.size, np.round(r).astype(int).tolist()))
    f = det["features"].values
    print("[features] " + ", ".join(f"{k}={f.get(k, float('nan')):.3g}" for k in
          ("n_intervals", "median_abs_succ_diff", "pnn50", "irregularity_index",
           "median_ibi", "rmssd", "dropout_rate", "longest_run")))
    print("[sqi]     %.3f  %s" % (det["sqi"].sqi,
                                  {k: round(v, 3) for k, v in det["sqi"].components.items()}))
    why = det["rationale"]
    print("[evidence] %s" % {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in why["evidence"].items()})
    print("[gates]")
    for g in why["gates"]:
        v = g["value"]
        print("   %s  %-26s %s %s %s" % ("PASS" if g["pass"] else "FAIL", g["name"],
                                       "NaN" if v is None else f"{v:.3f}", g["op"],
                                       g["threshold"]))
    print("[rule]    %s" % why["rule"])
    print("\nNote: a consumer wearable and this prototype are not interchangeable "
          "references; ECG-confirmed rhythm is the only ground truth for AF.")
    return {"result": res.outcome.value, "class": res.predicted_class,
            "rationale": why}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        list_scans()
        sys.exit(0)
    diagnose(sys.argv[1], profile=(sys.argv[2] if len(sys.argv) > 2 else "consumer"))
