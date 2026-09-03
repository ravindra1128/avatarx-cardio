"""
E6 — degradation experiment on REAL RR interval data (T6).

Data: MIMIC PERform AF (CC-BY 4.0; Charlton et al., via
https://ppg-beats.readthedocs.io — Zenodo record 6807403): 35 critically-ill
adults, 19 AF / 16 non-AF, 20 min each, ECG @ 125 Hz. R-peaks are detected
here (the dataset ships waveforms, not annotations) with a Pan-Tompkins-
style detector; per-subject plausibility is checked and reported.

Protocol: per subject, RR series -> non-overlapping 90 s windows (>= 30
intervals) -> inject measured-rPPG degradation (timing jitter sigma, pulse-
deficit dropout p on beats ending RR < 400 ms, 3% false insertions) ->
PRODUCTION feature path (clean_runs -> run features) -> logistic regression
AND gradient-boosted trees -> PARTICIPANT-level 5-fold CV -> window-level
AUROC with participant-cluster bootstrap CIs.

The degradation curve is the deliverable. If AUC <= 0.90 at the moderate
setting, that is the honest answer — nothing here is tuned on test folds.

Run:  python3 scripts/e6_degradation.py            # full grid + artifact
Outputs: models/e6_degradation_curve.csv, models/model_a_v01.json,
         models/e6_moderate_result.json
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import csv
import json
import urllib.request
import zipfile

import numpy as np

from models.baseline import (feature_vector, degrade_rr_to_beatseries,
                             cross_validate_participants, train_model_a)
from evaluation.afib_metrics import evaluate_binary

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = REPO / "data_cache"
OUT_DIR = REPO / "models"
FS = 125.0
WINDOW_S = 90.0
MIN_INTERVALS = 30
FALSE_RATE = 0.03
SIGMAS = (5.0, 10.0, 15.0, 20.0, 30.0)
DEFICITS = (0.3, 0.5, 0.7)
MODERATE = (15.0, 0.5)

ZENODO = "https://zenodo.org/record/6807403/files/{}?download=1"
FILES = ("mimic_perform_af_csv.zip", "mimic_perform_non_af_csv.zip")


def ensure_download(cache: pathlib.Path = CACHE) -> None:
    cache.mkdir(exist_ok=True)
    for name in FILES:
        z = cache / name
        if not z.exists():
            print(f"downloading {name} (CC-BY 4.0) ...")
            urllib.request.urlretrieve(ZENODO.format(name), z)
        d = cache / name.replace(".zip", "")
        if not d.exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(cache)


# ------------------------------------------------------------- R-peaks
def detect_rpeaks_s(ecg: np.ndarray, fs: float = FS) -> np.ndarray:
    """Pan-Tompkins-style: bandpass -> square -> integrate -> block-adaptive
    threshold -> refine on the filtered signal. Polarity-agnostic."""
    from scipy.signal import butter, filtfilt, find_peaks
    x = np.asarray(ecg, float)
    x = x[np.isfinite(x)] if not np.all(np.isfinite(x)) else x
    b, a = butter(3, [5.0 / (fs / 2), 20.0 / (fs / 2)], btype="band")
    bp = filtfilt(b, a, x - np.mean(x))
    e = bp ** 2
    k = max(int(0.12 * fs), 1)
    integ = np.convolve(e, np.ones(k) / k, mode="same")

    peaks_all = []
    block = int(10 * fs)
    for s0 in range(0, integ.size, block):
        seg = integ[s0:s0 + block]
        if seg.size < fs:
            continue
        thr = 0.25 * np.percentile(seg, 95)
        pk, _ = find_peaks(seg, height=thr, distance=int(0.25 * fs))
        peaks_all.extend((pk + s0).tolist())
    # refine each to the local |bp| maximum (true R location)
    half = int(0.08 * fs)
    refined = []
    for p in peaks_all:
        lo, hi = max(0, p - half), min(bp.size, p + half + 1)
        refined.append(lo + int(np.argmax(np.abs(bp[lo:hi]))))
    r = np.unique(refined)
    # collapse refinements that landed on the same R wave
    keep = [r[0]] if r.size else []
    for p in r[1:]:
        if p - keep[-1] >= int(0.25 * fs):
            keep.append(p)
    return np.asarray(keep) / fs


def load_mimic_rr(cache: pathlib.Path = CACHE) -> list:
    """[(pid, label, rr_ms array)] with per-subject plausibility report."""
    subjects = []
    for label, sub in ((1, "mimic_perform_af_csv"),
                       (0, "mimic_perform_non_af_csv")):
        for f in sorted((cache / sub).glob("*_data.csv")):
            pid = f.stem.replace("_data", "")
            ecg = np.genfromtxt(f, delimiter=",", skip_header=1,
                                usecols=(2,))
            rp = detect_rpeaks_s(ecg)
            rr = np.diff(rp) * 1000.0
            med = float(np.median(rr)) if rr.size else float("nan")
            frac_plaus = float(np.mean((rr > 250) & (rr < 2200))) \
                if rr.size else 0.0
            if not (300 <= med <= 1500) or frac_plaus < 0.85 or rr.size < 200:
                print(f"  ! {pid}: median RR {med:.0f} ms, plausible "
                      f"{frac_plaus:.0%}, n={rr.size} — INCLUDED but flagged")
            subjects.append((pid, label, rr))
    return subjects


def windows_of(subjects) -> list:
    """[(pid, label, rr_window)] — non-overlapping 90 s windows."""
    out = []
    for pid, label, rr in subjects:
        t = np.cumsum(rr) / 1000.0
        w0, start = 0, 0.0
        for i in range(rr.size):
            if t[i] - start >= WINDOW_S:
                seg = rr[w0:i + 1]
                if seg.size >= MIN_INTERVALS:
                    out.append((pid, label, seg))
                w0, start = i + 1, t[i]
    return out


def run_e6_setting(subjects, sigma_ms: float, deficit_p: float,
                   seed: int = 20260814, models=("logreg", "gbt")) -> dict:
    wins = windows_of(subjects)
    X, y, pid = [], [], []
    for k, (p, label, seg) in enumerate(wins):
        s = degrade_rr_to_beatseries(seg, sigma_ms, deficit_p, FALSE_RATE,
                                     seed=seed + 7919 * k)
        X.append(feature_vector(s))
        y.append(label)
        pid.append(p)
    X = np.vstack(X); y = np.asarray(y); pid = np.asarray(pid)
    out = {"sigma_ms": sigma_ms, "deficit_p": deficit_p,
           "false_rate": FALSE_RATE, "n_windows": int(y.size),
           "n_participants": int(len(set(pid.tolist())))}
    for m in models:
        cv = cross_validate_participants(X, y, pid, model=m, k=5, seed=seed)
        scores = cv["oof_scores"]
        res = evaluate_binary(y, (scores >= 0.5).astype(int), scores,
                              participant_ids=pid, n_boot=1000, seed=seed)
        out[m] = {"auroc": float(res.auroc),
                  "auroc_ci95": [float(res.auroc_ci[0]),
                                 float(res.auroc_ci[1])],
                  "sensitivity": float(res.sensitivity),
                  "specificity": float(res.specificity)}
    return out


def main():
    ensure_download()
    print("loading MIMIC PERform AF / non-AF and detecting R-peaks ...")
    subjects = load_mimic_rr()
    n_af = sum(1 for _, l, _ in subjects if l == 1)
    print(f"{len(subjects)} subjects ({n_af} AF / {len(subjects) - n_af} "
          f"non-AF), {len(windows_of(subjects))} windows of {WINDOW_S:.0f} s")

    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for sig in SIGMAS:
        for dp in DEFICITS:
            r = run_e6_setting(subjects, sig, dp)
            rows.append(r)
            print(f"  sigma {sig:4.0f} ms  deficit {dp:.1f}  "
                  f"LR AUC {r['logreg']['auroc']:.3f} "
                  f"[{r['logreg']['auroc_ci95'][0]:.3f},"
                  f"{r['logreg']['auroc_ci95'][1]:.3f}]  "
                  f"GBT AUC {r['gbt']['auroc']:.3f} "
                  f"[{r['gbt']['auroc_ci95'][0]:.3f},"
                  f"{r['gbt']['auroc_ci95'][1]:.3f}]")

    csv_path = OUT_DIR / "e6_degradation_curve.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_ms", "deficit_p", "false_rate", "n_windows",
                    "logreg_auroc", "logreg_ci_lo", "logreg_ci_hi",
                    "gbt_auroc", "gbt_ci_lo", "gbt_ci_hi"])
        for r in rows:
            w.writerow([r["sigma_ms"], r["deficit_p"], r["false_rate"],
                        r["n_windows"],
                        r["logreg"]["auroc"], *r["logreg"]["auroc_ci95"],
                        r["gbt"]["auroc"], *r["gbt"]["auroc_ci95"]])
    print(f"wrote {csv_path}")

    moderate = next(r for r in rows if (r["sigma_ms"], r["deficit_p"])
                    == MODERATE)
    with open(OUT_DIR / "e6_moderate_result.json", "w") as f:
        json.dump(moderate, f, indent=2)

    # deployment artifact: fitted on ALL windows at the moderate setting —
    # its numbers are NEVER quoted; the CV numbers above are the evidence.
    wins = windows_of(subjects)
    X = np.vstack([feature_vector(degrade_rr_to_beatseries(
        seg, MODERATE[0], MODERATE[1], FALSE_RATE, seed=20260814 + 7919 * k))
        for k, (_, _, seg) in enumerate(wins)])
    y = np.asarray([l for _, l, _ in wins])
    art = train_model_a(X, y, model="logreg")
    art["training_note"] = ("fitted on rPPG-degraded MIMIC PERform AF "
                            "(synthetic degradation of real RR data); no "
                            "clinical performance is claimed")
    with open(OUT_DIR / "model_a_v01.json", "w") as f:
        json.dump(art, f)
    print(f"wrote {OUT_DIR / 'model_a_v01.json'} ({art['version']})")

    gate = moderate["logreg"]["auroc"] > 0.90 and moderate["gbt"]["auroc"] > 0.90
    print(f"\nT6 acceptance (AUC > 0.90 at sigma=15 ms, deficit 0.5): "
          f"{'PASS' if gate else 'FAIL — task blocked, report the curve'}")


if __name__ == "__main__":
    main()
