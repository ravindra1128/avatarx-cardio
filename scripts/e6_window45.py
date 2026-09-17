"""
E6b — the RR classifier at OUR scan length, under phone-level beat timing
noise, with a three-way decision band (2026-09-17).

Same data, protocol and production feature path as scripts/e6_degradation.py
(MIMIC PERform AF, CC-BY 4.0: 35 critically-ill adults, 19 AF / 16 non-AF,
20 min ECG each; R-peaks detected here; participant-level 5-fold CV;
cluster-bootstrap CIs). What changes:

  * WINDOW: 45 s (>= 20 intervals) - the analysis window a phone scan
    actually yields (the recorder keeps the last 48 s), not E6's 90 s.
  * TIMING NOISE up to 60 ms - the phone's measured cross-region beat
    timing is 20-59 ms; the ShenAI train is ~15 ms; E6 stopped at 30.
  * FALSE-BEAT RATE 3 % and 8 % - split detections are the phone's main
    error mode (split fraction 0.15-0.38 on real scans).
  * A deployment fit on a POOLED mix of noise levels, so one artifact serves
    both interval sources, evaluated out-of-fold PER setting.
  * THE DECISION BAND: from the out-of-fold scores at the phone-like setting,
    the pair (tau_lo, tau_hi) such that among DECIDED windows sensitivity and
    specificity both clear TARGET, with the smallest inconclusive fraction.
    Everything between the two is INCONCLUSIVE. This is what turns a
    probability into "AFib Detected / AFib Not Detected / Inconclusive"
    (inference/afib_result.py) and it is set here, on public data, never on
    a scan.

Numbers quoted anywhere come from the out-of-fold CV, never from the
deployment fit.

Run:  python3 scripts/e6_window45.py
Outputs: models/e6b_window45_curve.csv, models/e6b_window45_band.json,
         models/model_a_v02_45s.json
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import csv
import json

import numpy as np

from models.baseline import (feature_vector, degrade_rr_to_beatseries,
                             cross_validate_participants, train_model_a)
from evaluation.afib_metrics import evaluate_binary
from scripts import e6_degradation as e6

OUT_DIR = e6.OUT_DIR
WINDOW_S = 45.0
MIN_INTERVALS = 20
SIGMAS = (10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0)
FALSE_RATES = (0.03, 0.08)
DEFICIT = 0.5
POOL = ((10.0, 0.03), (20.0, 0.03), (30.0, 0.05), (40.0, 0.08))   # deployment mix
PHONE_LIKE = (30.0, 0.08)          # band is set here: our video path's centre
TARGET_SENS = 0.95
TARGET_SPEC = 0.97
SEED = 20260917


def windows_of(subjects, window_s=WINDOW_S, min_intervals=MIN_INTERVALS):
    out = []
    for pid, label, rr in subjects:
        t = np.cumsum(rr) / 1000.0
        w0, start = 0, 0.0
        for i in range(rr.size):
            if t[i] - start >= window_s:
                seg = rr[w0:i + 1]
                if seg.size >= min_intervals:
                    out.append((pid, label, seg))
                w0, start = i + 1, t[i]
    return out


def features_at(wins, sigma_ms, false_rate, deficit=DEFICIT, seed=SEED):
    X, y, pid = [], [], []
    for k, (p, label, seg) in enumerate(wins):
        s = degrade_rr_to_beatseries(seg, sigma_ms, deficit, false_rate,
                                     seed=seed + 7919 * k)
        X.append(feature_vector(s))
        y.append(label)
        pid.append(p)
    return np.vstack(X), np.asarray(y, int), np.asarray(pid)


def band_from_scores(y, scores, target_sens=TARGET_SENS, target_spec=TARGET_SPEC):
    """Smallest inconclusive fraction whose decided cases clear both
    targets; ties broken toward the wider margin. Grid over quantiles."""
    y = np.asarray(y, int)
    s = np.asarray(scores, float)
    grid = np.unique(np.round(np.quantile(s, np.linspace(0.0, 1.0, 201)), 4))
    best = None
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            decided = (s <= lo) | (s >= hi)
            if decided.sum() == 0:
                continue
            pos = s >= hi
            tp = int(np.sum(pos & (y == 1)))
            fn = int(np.sum((s <= lo) & (y == 1)))
            tn = int(np.sum((s <= lo) & (y == 0)))
            fp = int(np.sum(pos & (y == 0)))
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            if sens >= target_sens and spec >= target_spec:
                inc = 1.0 - decided.mean()
                key = (inc, -(hi - lo))
                if best is None or key < best[0]:
                    best = (key, {"tau_lo": float(lo), "tau_hi": float(hi),
                                  "inconclusive_fraction": float(inc),
                                  "decided_sensitivity": float(sens),
                                  "decided_specificity": float(spec),
                                  "n_decided": int(decided.sum()),
                                  "n_windows": int(y.size)})
    return best[1] if best else None


def band_metrics(y, scores, lo, hi):
    y = np.asarray(y, int)
    s = np.asarray(scores, float)
    decided = (s <= lo) | (s >= hi)
    pos = s >= hi
    tp = int(np.sum(pos & (y == 1))); fn = int(np.sum((s <= lo) & (y == 1)))
    tn = int(np.sum((s <= lo) & (y == 0))); fp = int(np.sum(pos & (y == 0)))
    return {"inconclusive_fraction": float(1.0 - decided.mean()),
            "decided_sensitivity": float(tp / max(tp + fn, 1)),
            "decided_specificity": float(tn / max(tn + fp, 1)),
            "n_decided": int(decided.sum())}


def main():
    e6.ensure_download()
    print("loading MIMIC PERform AF / non-AF and detecting R-peaks ...")
    subjects = e6.load_mimic_rr()
    wins = windows_of(subjects)
    n_af = sum(1 for _, l, _ in subjects if l == 1)
    print(f"{len(subjects)} subjects ({n_af} AF / {len(subjects) - n_af} non-AF), "
          f"{len(wins)} windows of {WINDOW_S:.0f} s "
          f"({sum(1 for _, l, _ in wins if l == 1)} AF windows)")
    OUT_DIR.mkdir(exist_ok=True)

    # ---- 1. per-setting out-of-fold curve (model fitted per setting) -------
    rows = []
    for fr in FALSE_RATES:
        for sig in SIGMAS:
            X, y, pid = features_at(wins, sig, fr)
            cv = cross_validate_participants(X, y, pid, model="logreg", k=5, seed=SEED)
            sc = cv["oof_scores"]
            res = evaluate_binary(y, (sc >= 0.5).astype(int), sc,
                                  participant_ids=pid, n_boot=500, seed=SEED)
            rows.append({"sigma_ms": sig, "false_rate": fr, "deficit_p": DEFICIT,
                         "n_windows": int(y.size), "auroc": float(res.auroc),
                         "ci_lo": float(res.auroc_ci[0]), "ci_hi": float(res.auroc_ci[1]),
                         "sens_at_0.5": float(res.sensitivity),
                         "spec_at_0.5": float(res.specificity)})
            print(f"  sigma {sig:4.0f} ms  false {fr:.2f}  AUC {res.auroc:.3f} "
                  f"[{res.auroc_ci[0]:.3f},{res.auroc_ci[1]:.3f}]  "
                  f"sens/spec@0.5 {res.sensitivity:.3f}/{res.specificity:.3f}")
    with open(OUT_DIR / "e6b_window45_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 2. POOLED fit: one model across noise levels, OOF per setting -----
    Xs, ys, ps, tags = [], [], [], []
    for sig, fr in POOL:
        X, y, pid = features_at(wins, sig, fr, seed=SEED + int(sig))
        Xs.append(X); ys.append(y); ps.append(pid); tags += [(sig, fr)] * y.size
    Xp, yp, pp = np.vstack(Xs), np.concatenate(ys), np.concatenate(ps)
    cvp = cross_validate_participants(Xp, yp, pp, model="logreg", k=5, seed=SEED)
    scp = cvp["oof_scores"]
    tags = np.array(tags)
    pooled = {}
    for sig, fr in POOL:
        m = (tags[:, 0] == sig) & (tags[:, 1] == fr)
        res = evaluate_binary(yp[m], (scp[m] >= 0.5).astype(int), scp[m],
                              participant_ids=pp[m], n_boot=500, seed=SEED)
        pooled[f"{sig:.0f}ms/{fr:.2f}"] = {"auroc": float(res.auroc),
                                           "ci": [float(res.auroc_ci[0]), float(res.auroc_ci[1])]}
        print(f"  pooled model @ sigma {sig:.0f} false {fr:.2f}: AUC {res.auroc:.3f}")

    # ---- 3. the decision band, on the pooled model's OOF at the phone-like
    #         setting (which is NOT one of the pooled training settings) -----
    Xb, yb, pb = features_at(wins, *PHONE_LIKE, seed=SEED + 4242)
    # out-of-fold on the pooled model: refit pooled folds, score the phone-like
    # windows of the held-out participants
    from models.baseline import LogisticModel
    oof = np.full(yb.size, np.nan)
    # the SAME participant folds the pooled CV used (identity hash only);
    # the model imputes medians inside fit, so NaN features pass through raw
    fold_of = cvp["fold_of_participant"]
    for k in range(5):
        tr = np.array([fold_of[p] != k for p in pp])
        te = np.array([fold_of[p] == k for p in pb])
        if not te.any():
            continue
        m = LogisticModel().fit(Xp[tr], yp[tr])
        oof[te] = m.predict_proba(Xb[te])
    ok = np.isfinite(oof)
    band = band_from_scores(yb[ok], oof[ok])
    res = evaluate_binary(yb[ok], (oof[ok] >= 0.5).astype(int), oof[ok],
                          participant_ids=pb[ok], n_boot=500, seed=SEED)
    print(f"  phone-like {PHONE_LIKE}: pooled-model OOF AUC {res.auroc:.3f}; band {band}")
    # how the same band behaves at the other settings
    band_by_setting = {}
    if band:
        for sig in SIGMAS:
            for fr in FALSE_RATES:
                Xo, yo, po = features_at(wins, sig, fr, seed=SEED + 99)
                o = np.full(yo.size, np.nan)
                for k in range(5):
                    tr = np.array([fold_of[p] != k for p in pp])
                    te = np.array([fold_of[p] == k for p in po])
                    if te.any():
                        m = LogisticModel().fit(Xp[tr], yp[tr])
                        o[te] = m.predict_proba(Xo[te])
                bm = band_metrics(yo, o, band["tau_lo"], band["tau_hi"])
                band_by_setting[f"{sig:.0f}ms/{fr:.2f}"] = bm
                print(f"    band @ sigma {sig:.0f} false {fr:.2f}: inconclusive "
                      f"{bm['inconclusive_fraction']:.2f}, decided sens "
                      f"{bm['decided_sensitivity']:.3f} spec {bm['decided_specificity']:.3f}")

    # ---- 4. deployment artifact: pooled fit on ALL windows -----------------
    art = train_model_a(Xp, yp, model="logreg")
    art["window_s"] = WINDOW_S
    art["training_note"] = (
        "fitted on rPPG-degraded MIMIC PERform AF (real RR intervals, 35 "
        "subjects, 45 s windows) pooled over timing sigma 10/20/30/40 ms and "
        "false-beat rates 3-8 %; no clinical performance is claimed. "
        "Evaluation numbers are the participant-level out-of-fold ones in "
        "e6b_window45_band.json, never this fit.")
    art["decision_band"] = band
    art["decision_band_setting"] = {"sigma_ms": PHONE_LIKE[0], "false_rate": PHONE_LIKE[1],
                                    "deficit_p": DEFICIT, "target_sens": TARGET_SENS,
                                    "target_spec": TARGET_SPEC}
    with open(OUT_DIR / "model_a_v02_45s.json", "w") as f:
        json.dump(art, f)
    with open(OUT_DIR / "e6b_window45_band.json", "w") as f:
        json.dump({"window_s": WINDOW_S, "min_intervals": MIN_INTERVALS,
                   "n_windows": len(wins), "n_subjects": len(subjects),
                   "pooled_settings": [list(p) for p in POOL],
                   "pooled_oof_auroc_by_setting": pooled,
                   "phone_like_setting": list(PHONE_LIKE),
                   "phone_like_oof_auroc": float(res.auroc),
                   "band": band, "band_by_setting": band_by_setting,
                   "curve_rows": rows, "seed": SEED}, f, indent=1)
    print("wrote", OUT_DIR / "model_a_v02_45s.json", "and e6b_window45_band.json")


if __name__ == "__main__":
    main()
