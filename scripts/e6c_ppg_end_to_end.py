"""
E6c — the RR classifier on REAL PPG beats found by OUR detector (2026-09-17).

E6/E6b degrade ECG-derived RR intervals synthetically. This runs the actual
front end instead: MIMIC PERform's contact PPG channel (125 Hz, 35 subjects,
19 AF / 16 non-AF, CC-BY 4.0) is resampled to the camera's 30 Hz, band-passed
like the pipeline's pulse waveform, and given to beats/detector.py's
single-waveform detector -> fuse_multi_roi(min_rois=1) -> the production
feature path -> the RR classifier -> the three-way decision band from
models/model_a_v02_45s.json.

Honest split: for every fold, the classifier is fitted on the ECG-derived,
synthetically degraded windows of the TRAINING participants (E6b's pooled
mix) and scored on the PPG-derived windows of the HELD-OUT participants. No
participant is ever on both sides; the PPG windows never train anything.

What this validates: the classifier's transfer from synthetic degradation to
real pulse-wave beats found by our own code, and the band's inconclusive
fraction on them. What it does NOT validate: faces, cameras, compression,
motion - contact PPG in an ICU is cleaner than any phone scan.

Run:  python3 scripts/e6c_ppg_end_to_end.py
Outputs: models/e6c_ppg_end_to_end.json
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

from beats.detector import BeatSeries, detect_beats_single_roi, fuse_multi_roi
from models.baseline import feature_vector, LogisticModel
from evaluation.afib_metrics import evaluate_binary
from scripts import e6_degradation as e6
from scripts.e6_window45 import (windows_of as ecg_windows_of, features_at, POOL,
                                 band_metrics, WINDOW_S, MIN_INTERVALS, SEED)

OUT = e6.OUT_DIR / "e6c_ppg_end_to_end.json"
FS_IN = 125.0
BAND_HZ = (0.7, 3.0)
RATES = (30.0, 125.0)


def load_ppg(cache=e6.CACHE):
    """[(pid, label, ppg@125Hz, rpeaks_s)] - the ECG R-peaks travel along so
    the detector's beat count can be checked window by window."""
    out = []
    for label, sub in ((1, "mimic_perform_af_csv"), (0, "mimic_perform_non_af_csv")):
        for f in sorted((cache / sub).glob("*_data.csv")):
            pid = f.stem.replace("_data", "")
            arr = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=(1, 2))
            ppg, ecg = arr[:, 0], arr[:, 1]
            ppg = np.where(np.isfinite(ppg), ppg, np.nanmedian(ppg))
            rp = e6.detect_rpeaks_s(ecg)
            out.append((pid, label, ppg, rp))
    return out


def pulse_wave(ppg, fs_out):
    x = np.asarray(ppg, float)
    if fs_out != FS_IN:
        from fractions import Fraction
        fr = Fraction(int(round(fs_out)), int(FS_IN)).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator)
    x = x - np.mean(x)
    b, a = butter(3, [BAND_HZ[0] / (fs_out / 2), BAND_HZ[1] / (fs_out / 2)], btype="band")
    return filtfilt(b, a, x)


def beat_windows(times_s, duration_s, window_s=WINDOW_S):
    """[(t0, t1)] non-overlapping windows over the record."""
    return [(t0, min(t0 + window_s, duration_s))
            for t0 in np.arange(0.0, duration_s - window_s + 1e-9, window_s)]


def ppg_windows(subjects, fs):
    """[(pid, label, BeatSeries, n_ecg_beats, n_det_beats)] per 45 s window."""
    out = []
    for pid, label, ppg, rp in subjects:
        w = pulse_wave(ppg, fs)
        dur = w.size / fs
        beats = detect_beats_single_roi(w, fs, "ppg")
        fused = fuse_multi_roi({"ppg": beats}, fs, dur, min_rois=1)
        for t0, t1 in beat_windows(None, dur):
            bs = [b for b in fused.beats if t0 <= b.t_s < t1]
            n_ecg = int(np.sum((rp >= t0) & (rp < t1)))
            if len(bs) < MIN_INTERVALS + 1:
                out.append((pid, label, None, n_ecg, len(bs)))
                continue
            out.append((pid, label, BeatSeries(bs, fs, t1 - t0), n_ecg, len(bs)))
    return out


def main():
    e6.ensure_download()
    print("loading PPG + ECG ...")
    subjects = load_ppg()
    art = json.load(open(e6.OUT_DIR / "model_a_v02_45s.json"))
    band = art["decision_band"]
    lo, hi = band["tau_lo"], band["tau_hi"]

    # ECG-derived degraded training pool (as E6b) with participant folds
    ecg_subjects = [(p, l, np.diff(rp) * 1000.0) for p, l, _, rp in subjects]
    ecg_wins = ecg_windows_of(ecg_subjects)
    Xs, ys, ps = [], [], []
    for sig, fr in POOL:
        X, y, pid = features_at(ecg_wins, sig, fr, seed=SEED + int(sig))
        Xs.append(X); ys.append(y); ps.append(pid)
    Xp, yp, pp = np.vstack(Xs), np.concatenate(ys), np.concatenate(ps)
    from datasets.splits import _stable_unit_interval
    uniq = sorted(set(pp.tolist()))
    fold_of = {p: int(_stable_unit_interval(str(p), SEED) * 5) for p in uniq}

    report = {"band": band, "rates": {}}
    for fs in RATES:
        wins = ppg_windows(subjects, fs)
        usable = [w for w in wins if w[2] is not None]
        print(f"fs {fs:.0f} Hz: {len(wins)} windows, {len(usable)} with >= {MIN_INTERVALS} beats "
              f"({len(wins) - len(usable)} too sparse -> INCONCLUSIVE/signal)")
        X = np.vstack([feature_vector(w[2]) for w in usable])
        y = np.asarray([w[1] for w in usable], int)
        pid = np.asarray([w[0] for w in usable])
        count_ratio = np.asarray([w[4] / max(w[3], 1) for w in usable], float)
        oof = np.full(y.size, np.nan)
        for k in range(5):
            tr = np.array([fold_of[p] != k for p in pp])
            te = np.array([fold_of.get(p, -1) == k for p in pid])
            if not te.any():
                continue
            m = LogisticModel().fit(Xp[tr], yp[tr])
            oof[te] = m.predict_proba(X[te])
        ok = np.isfinite(oof)
        res = evaluate_binary(y[ok], (oof[ok] >= 0.5).astype(int), oof[ok],
                              participant_ids=pid[ok], n_boot=500, seed=SEED)
        bm = band_metrics(y[ok], oof[ok], lo, hi)
        # three-way per window, counting sparse windows as inconclusive too
        n_all = len(wins)
        detected = int(np.sum(oof[ok] >= hi)); notdet = int(np.sum(oof[ok] <= lo))
        inconclusive = n_all - detected - notdet
        # per-participant majority of decided windows
        per_p = {}
        for p, yy, s in zip(pid[ok], y[ok], oof[ok]):
            d = per_p.setdefault(p, {"label": int(yy), "det": 0, "not": 0, "inc": 0})
            if s >= hi: d["det"] += 1
            elif s <= lo: d["not"] += 1
            else: d["inc"] += 1
        subj_correct = sum(1 for d in per_p.values()
                           if (d["det"] > d["not"]) == bool(d["label"]) and (d["det"] + d["not"]) > 0)
        subj_undecided = sum(1 for d in per_p.values() if d["det"] + d["not"] == 0)
        r = {"n_windows": n_all, "n_usable": int(y.size),
             "detector_count_ratio_median": float(np.median(count_ratio)),
             "detector_count_ratio_iqr": [float(np.percentile(count_ratio, 25)),
                                          float(np.percentile(count_ratio, 75))],
             "oof_auroc": float(res.auroc), "oof_auroc_ci": [float(res.auroc_ci[0]), float(res.auroc_ci[1])],
             "sens_spec_at_0.5": [float(res.sensitivity), float(res.specificity)],
             "band": bm,
             "three_way_all_windows": {"AFIB_DETECTED": detected, "AFIB_NOT_DETECTED": notdet,
                                       "INCONCLUSIVE": inconclusive},
             "subjects": {"n": len(per_p), "majority_correct": subj_correct,
                          "undecided": subj_undecided}}
        report["rates"][f"{fs:.0f}"] = r
        print(f"  detector beats/ECG beats: median {r['detector_count_ratio_median']:.3f} "
              f"IQR {r['detector_count_ratio_iqr'][0]:.3f}-{r['detector_count_ratio_iqr'][1]:.3f}")
        print(f"  OOF AUROC {res.auroc:.3f} [{res.auroc_ci[0]:.3f},{res.auroc_ci[1]:.3f}]  "
              f"sens/spec@0.5 {res.sensitivity:.3f}/{res.specificity:.3f}")
        print(f"  band: inconclusive {bm['inconclusive_fraction']:.2f}, decided sens "
              f"{bm['decided_sensitivity']:.3f} spec {bm['decided_specificity']:.3f}")
        print(f"  three-way over ALL {n_all} windows: {r['three_way_all_windows']}; "
              f"subjects majority-correct {subj_correct}/{len(per_p)}, undecided {subj_undecided}")
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
