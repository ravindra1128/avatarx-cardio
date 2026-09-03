"""
End-to-end smoke demo on synthetic data.

Purpose: prove the interfaces compose and the gates fire, BEFORE any real data
exists. Everything here is synthetic -- no accuracy claim is made or implied.
Run:  python3 scripts/demo_pipeline.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from beats.detector import (detect_beats_single_roi, fuse_multi_roi,
                            flag_suspected_missed_beats)
from beats.ibi import clean_runs, rmssd_from_runs
from features.rhythm import (compute_rhythm_features_from_runs, TRANSFERABILITY)
from evaluation.beat_metrics import (match_beats, ibi_agreement,
                                     GATE1_CONTROLLED, GATE1_ARRHYTHMIA,
                                     decompose_rmssd_error,
                                     beat_confidence_calibration)
from evaluation.afib_metrics import (evaluate_binary, risk_coverage_curve,
                                     projected_ppv, serial_confirmation,
                                     subgroup_report, fairness_gate, no_read_report)

RNG = np.random.default_rng(20260814)
FPS = 60.0


def pulse_template(n):
    t = np.linspace(0, 1, n, endpoint=False)
    return np.exp(-((t - 0.18) ** 2) / (2 * 0.075 ** 2))


def synth_roi(rr_s, fps=FPS, noise=0.15, amp_jitter=0.0, drop_short_ms=None):
    """Build one ROI waveform. `drop_short_ms` simulates PULSE DEFICIT:
    beats following a short RR eject too little volume to be seen."""
    onsets = np.concatenate([[0.0], np.cumsum(rr_s)])
    sig = np.zeros(int(onsets[-1] * fps) + 1)
    peaks = []
    for k, (a, b) in enumerate(zip(onsets, onsets[1:])):
        if drop_short_ms and (b - a) * 1000 < drop_short_ms:
            continue
        i0, i1 = int(a * fps), int(b * fps)
        if i1 - i0 > 2:
            amp = 1.0 + (RNG.normal(0, amp_jitter) if amp_jitter else 0.0)
            sig[i0:i1] = amp * pulse_template(i1 - i0)
            peaks.append(a + 0.18 * (b - a))
    return sig + RNG.normal(0, noise, sig.shape), np.array(peaks)


def run_case(title, rr_s, noise, drop_short_ms=None, gate=GATE1_CONTROLLED):
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)
    per_roi = {}
    for roi in ("forehead", "cheek_l", "cheek_r", "nose"):
        nz = noise * (1.0 if roi == "forehead" else 1.4 if roi == "nose" else 1.15)
        sig, ref = synth_roi(rr_s, noise=nz, drop_short_ms=drop_short_ms)
        per_roi[roi] = detect_beats_single_roi(sig, FPS, roi)
    dur = float(np.sum(rr_s))
    series = fuse_multi_roi(per_roi, FPS, dur, min_rois=2)

    print(f"  per-ROI detections : "
          + ", ".join(f"{k}={len(v)}" for k, v in per_roi.items()))
    print(f"  fused beats        : {len(series.beats)}  "
          f"(true {len(ref)})   mean conf {series.confidences().mean():.2f}")
    print(f"  suspected missed   : {len(flag_suspected_missed_beats(series))}")

    m50 = match_beats(ref, series.times(), tolerance_ms=50)
    ibi = ibi_agreement(ref, series.times(), m50)
    print(f"  beat F1@50ms       : {m50.f1:.3f}   "
          f"(Se {m50.sensitivity:.3f} / PPV {m50.ppv:.3f})")
    print(f"  IBI MAE            : {ibi.ibi_mae_ms:.1f} ms   "
          f"[LoA {ibi.loa_lower_ms:.0f}, {ibi.loa_upper_ms:.0f}]")
    print(f"  RMSSD ref/est      : {ibi.ref_rmssd_ms:.1f} / {ibi.est_rmssd_ms:.1f} ms "
          f"(err {ibi.rmssd_error_ms:+.1f})")

    ok, detail = gate.evaluate(m50, ibi)
    print(f"  {gate.name}: {'PASS' if ok else 'FAIL'}")
    for k, d in detail.items():
        if not d["pass"]:
            print(f"      x {k}: {d['value']:.3f} {d['op']} {d['threshold']}")

    # PRODUCTION PATH (v2): clean runs, never the raw fused series.
    # The confidence THRESHOLD is a calibrated quantity, not a constant: this
    # demo detector's confidences are honest but compressed (mean ~0.45 on
    # clean data), so a naive 0.5 cutoff abstains on everything. Production
    # calibrates the scale on DEV (Gate: calibration ECE <= 0.10); here the
    # demo shows the measurement and uses the scale it implies.
    cal = beat_confidence_calibration(m50, series.confidences())
    lo = [b for b in cal["bins"] if b["fraction_matched"] < 0.7]
    thr = max((b["mean_confidence"] for b in lo), default=0.30)
    print(f"  conf calibration   : ECE {cal['ece']:.3f}; bin PPVs "
          + "/".join(f"{b['fraction_matched']:.2f}" for b in cal["bins"])
          + f" -> threshold {thr:.2f}")
    raw_rmssd = float(np.sqrt(np.mean(np.diff(series.ibi_ms()) ** 2))) \
        if len(series.beats) > 2 else float("nan")
    rs = clean_runs(series, min_conf=thr, min_run_beats=4)
    true_rmssd = float(np.sqrt(np.mean(np.diff(np.asarray(rr_s) * 1000) ** 2)))
    print(f"  RMSSD paths        : true {true_rmssd:6.1f} | raw-series "
          f"{raw_rmssd:6.1f} | clean-run {rmssd_from_runs(rs):6.1f} ms   "
          f"(runs {rs.n_runs}, dropout {rs.dropout_rate*100:.0f}%)")

    f = compute_rhythm_features_from_runs(rs.runs, rs.run_confidences,
                                          rs.dropout_rate)
    keys = ["n_intervals", "longest_run", "pnn50", "median_abs_succ_diff",
            "rmssd", "cv_ibi", "sample_entropy", "markov_surprise"]
    print("  features (runs)    : " + ", ".join(
        f"{k}={f.values.get(k, float('nan')):.3g}" for k in keys))
    for w in f.estimator_warnings:
        print(f"      ! {w}")
    return series, ibi


# --------------------------------------------------------------------- cases
sinus = np.clip(RNG.normal(0.85, 0.030, 106), 0.5, 1.4)          # 90 s, RMSSD ~40 ms
run_case("A. Sinus rhythm, 90 s @60 fps, clean", sinus, noise=0.12)

af = np.clip(RNG.normal(0.62, 0.17, 145), 0.28, 1.35)            # 90 s AF
run_case("B. Atrial fibrillation, 90 s @60 fps, clean", af, noise=0.12,
         gate=GATE1_ARRHYTHMIA)

run_case("C. Atrial fibrillation WITH PULSE DEFICIT (<400 ms RR not perfused)",
         af, noise=0.12, drop_short_ms=400, gate=GATE1_ARRHYTHMIA)

short = np.clip(RNG.normal(0.85, 0.030, 36), 0.5, 1.4)           # 30 s only
run_case("D. Sinus rhythm, 30 s -- estimator sample-size floor", short, noise=0.12)

# --------------------------------------------------------------- error budget
print("\n" + "=" * 78)
print(" RMSSD ERROR DECOMPOSITION (prior AvatarX measurement, 30 fps)")
print("=" * 78)
d = decompose_rmssd_error(true_rmssd_ms=33.8, measured_rmssd_ms=65.9, fps=30)
for k, v in d.items():
    print(f"  {k:<34}{v:>10.3f}" if isinstance(v, float) else f"  {k:<34}{v}")
print("  -> the extractor term dominates; raising frame rate alone is NOT the fix")

# ------------------------------------------------------------ classifier eval
print("\n" + "=" * 78)
print(" AFib-LEVEL EVALUATION (synthetic labels -- interface demo only)")
print("=" * 78)
n = 600
pid = np.array([f"p{i//3}" for i in range(n)])                 # 3 recordings/person
y = RNG.binomial(1, 0.25, n)
qual = np.clip(RNG.beta(5, 2, n), 0, 1)
score = np.clip(0.5 * y + 0.35 * qual * RNG.normal(0.5, 0.3, n) + 0.2 * RNG.random(n), 0, 1)
pred = (score > 0.55).astype(int)
noread = qual < 0.35

res = evaluate_binary(y, pred, score, noread, pid, n_boot=400)
s = res.summary()
print(f"  analysed {s['n_analysed']}  no-read {s['n_no_read']} "
      f"({s['no_read_rate']*100:.1f}%)  prevalence {s['prevalence_in_analysed']:.3f}")
for k in ("sensitivity", "specificity", "ppv", "npv", "auroc"):
    v = s[k]
    if v: print(f"  {k:<12}{v['value']:.3f}  CI95 [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
print(f"  FP per 1000 analysed: {s['false_positives_per_1000_analysed']}")

print("\n  risk-coverage curve (the measurement the field omits):")
print(f"    {'cov':>6}{'n':>6}{'thr':>7}{'sens':>8}{'spec':>8}{'nAF':>6}")
for r in risk_coverage_curve(y[~noread], pred[~noread], qual[~noread]):
    print(f"    {r['coverage']:>6.2f}{r['n']:>6}{r['quality_threshold']:>7.2f}"
          f"{r['sensitivity']:>8.3f}{r['specificity']:>8.3f}{r['n_afib_retained']:>6}")

print("\n  projected PPV at deployment prevalence (Se .85 / Sp .94 planning point):")
print(f"    {'prev':>6}{'PPV':>8}{'NPV':>8}{'FP/1000':>10}{'FA:TP':>8}")
for r in projected_ppv(0.85, 0.94):
    print(f"    {r['prevalence']:>6.2f}{r['ppv']:>8.3f}{r['npv']:>8.3f}"
          f"{r['false_positives_per_1000_scans']:>10.1f}{r['false_alarms_per_true_case']:>8.1f}")

print("\n  serial confirmation vs persistent-FP share (v2 -- the honest model):")
print(f"    {'rule':>8}{'FPshare':>9}{'Se':>8}{'Sp':>9}{'PPV@2%':>9}")
for k, nn in [(2, 3), (3, 4)]:
    for share in (0.0, 0.5, 0.8):
        se, sp = serial_confirmation(0.85, 0.94, k, nn, fp_persistent_share=share)
        p2 = 0.02 * se / (0.02 * se + 0.98 * (1 - sp))
        print(f"    {f'{k}/{nn}':>8}{share:>9.1f}{se:>8.3f}{sp:>9.4f}{p2:>9.3f}")
print("    (share = fraction of FPs from persistent causes: ectopy, RSA, face)")

print("\n  subgroup + fairness gate:")
tone = np.array(["1-4", "5-7", "8-10"])[RNG.integers(0, 3, n)]
rep = subgroup_report(y[~noread], pred[~noread], {"monk_band": tone[~noread]})
for lev, v in rep["monk_band"].items():
    if v["status"] == "OK":
        print(f"    band {lev:<5} n={v['n']:<4} AF={v['n_afib']:<4} "
              f"sens {v['sensitivity']:.3f}  spec {v['specificity']:.3f}")
    else:
        print(f"    band {lev:<5} {v['status']} (n={v['n']}, AF={v['n_afib']})")
ok, fails = fairness_gate(rep)
print(f"    fairness gate: {'PASS' if ok else 'FAIL'}")
for f_ in fails: print(f"      x {f_}")

print("\n  no-read parity audit (v2 -- abstention is gameable, so audit it):")
ok_nr, rep_nr = no_read_report(noread, {"monk_band": tone,
                                        "afib": np.where(y==1, "AF", "non-AF")})
for dim, levels in rep_nr["levels"].items():
    line = "    " + dim + ": " + ", ".join(f"{k}={v['no_read_rate']*100:.1f}%" for k,v in levels.items())
    print(line)
print(f"    parity gate: {'PASS' if ok_nr else 'FAIL'}")
for f_ in rep_nr["failures"]: print(f"      x {f_}")

print("\n  feature transferability census:")
for t in ("HIGH", "MED", "LOW"):
    ks = [k for k, v in TRANSFERABILITY.items() if v == t]
    print(f"    {t:<5} ({len(ks):>2}): {', '.join(ks[:7])}{' ...' if len(ks) > 7 else ''}")
print("\ndone.")
