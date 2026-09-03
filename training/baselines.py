"""
Mandatory-baselines harness (v0.2 M3.11) — invariant 13's teeth. A
candidate ships only if it beats every baseline on the participant- and
session-disjoint INTERNAL_TEST split, with every number derived from the
GATED PRODUCTION PATH's cached features (training/runs.py), never a side
harness.

The four mandatory baselines:
  tachogram_stats      LR on the classic tachogram statistics — the
                       "did the extra features earn anything" bar
  hr_only              LR on median IBI alone — the "is it just rate" bar
  participant_history  predicts each participant's TRAIN-split label —
                       the MEMORIZATION detector: on honest disjoint
                       splits it cannot see the test participants and
                       must sit near chance; a high AUC here means the
                       splits leak identity
  device_site          LR on device/site metadata one-hots alone — the
                       LEAKAGE detector: signal here means labels are
                       confounded with hardware/site, and any model can
                       cheat

Promotion margins (spec B.15): the candidate must beat tachogram_stats
and hr_only by >= 0.02 AUC (or both sit at the >= 0.97 ceiling with the
candidate not below the baseline), must exceed the two detector
baselines by >= 0.15, and each detector baseline must itself stay
<= 0.60 AUC — otherwise the DATA is refused, not just the model.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from training.runs import rank_auc, matrix

TACHO_FEATURES = ("median_abs_succ_diff", "pnn50", "rmssd",
                  "irregularity_index", "median_ibi")
SIGNAL_MARGIN = 0.02
DETECTOR_MARGIN = 0.15
DETECTOR_CEILING = 0.60
AUC_CEILING = 0.97


def _lr(X, y, seed=0):
    from models.baseline import LogisticModel
    np.random.seed(seed)
    return LogisticModel().fit(np.nan_to_num(np.asarray(X, float), nan=0.0),
                               np.asarray(y, int))


def _onehot(values: list) -> tuple:
    cats = sorted(set(values))
    X = np.zeros((len(values), max(len(cats), 1)))
    for i, v in enumerate(values):
        X[i, cats.index(v)] = 1.0
    return X, cats


def mandatory_baseline_report(run_dir) -> dict:
    run_dir = pathlib.Path(run_dir)
    with open(run_dir / "run_record.json") as f:
        record = json.load(f)
    records = [json.loads(x) for x in
               (run_dir / "features.jsonl").read_text().splitlines() if x]
    names = record["feature_names"]
    eval_split = "INTERNAL_TEST"
    Xtr, ytr, rtr = matrix(records, names, "TRAIN")
    Xte, yte, rte = matrix(records, names, eval_split)
    if Xte.shape[0] == 0 or len(set(yte.tolist())) < 2:
        raise ValueError(f"{eval_split} split has no class contrast — "
                         "cannot evaluate margins")

    from models.baseline import load_model_a
    with open(run_dir / "model.json") as f:
        model, art = load_model_a(json.load(f))
    cand = rank_auc(yte, model.predict_proba(np.nan_to_num(Xte, nan=0.0)))

    out = {"eval_split": eval_split, "n_eval": int(Xte.shape[0]),
           "candidate_auc": round(float(cand), 4), "baselines": {},
           "margins": {}, "leak_flags": [], "parity": {}}

    # tachogram statistics + HR-only
    for bname, feats in (("tachogram_stats", [n for n in TACHO_FEATURES
                                              if n in names]),
                         ("hr_only", ["median_ibi"])):
        idx = [names.index(f) for f in feats if f in names]
        m = _lr(Xtr[:, idx], ytr)
        auc = rank_auc(yte, m.predict_proba(
            np.nan_to_num(Xte[:, idx], nan=0.0)))
        out["baselines"][bname] = round(float(auc), 4)

    # participant-history (memorization detector)
    hist = {}
    for r in rtr:
        hist.setdefault(r["participant_id"], []).append(r["label_af"])
    prior = float(np.mean(ytr)) if ytr.size else 0.5
    ph = np.array([float(np.mean(hist[r["participant_id"]]))
                   if r["participant_id"] in hist else prior for r in rte])
    out["baselines"]["participant_history"] = round(float(
        0.5 if np.allclose(ph, ph[0]) else rank_auc(yte, ph)), 4)

    # device/site (metadata leakage detector)
    Xd_tr, cats = _onehot([f"{r['device']}|{r['site']}" for r in rtr])
    vals_te = [f"{r['device']}|{r['site']}" for r in rte]
    Xd_te = np.zeros((len(vals_te), len(cats)))
    for i, v in enumerate(vals_te):
        if v in cats:
            Xd_te[i, cats.index(v)] = 1.0
    if Xd_tr.shape[1] < 2:
        out["baselines"]["device_site"] = 0.5      # single stratum: no signal
    else:
        m = _lr(Xd_tr, ytr)
        out["baselines"]["device_site"] = round(float(
            rank_auc(yte, m.predict_proba(Xd_te))), 4)

    for bname, auc in out["baselines"].items():
        out["margins"][bname] = round(float(cand - auc), 4)
    for det in ("participant_history", "device_site"):
        if out["baselines"][det] > DETECTOR_CEILING:
            out["leak_flags"].append(
                f"{det} AUC {out['baselines'][det]:.2f} > "
                f"{DETECTOR_CEILING}: the {'splits leak identity' if det == 'participant_history' else 'labels are confounded with hardware/site'}")

    # no-read parity by Fitzpatrick group (when recorded)
    groups = {}
    for r in rte:
        g = r.get("fitzpatrick") or "unrecorded"
        groups.setdefault(str(g), []).append(r["outcome"] != "ACCEPT")
    out["parity"] = {g: {"n": len(v),
                         "no_read_rate": round(float(np.mean(v)), 3)}
                     for g, v in groups.items()}
    if set(out["parity"]) == {"unrecorded"}:
        out["parity_note"] = ("Fitzpatrick not recorded in this dataset — "
                              "parity by skin tone NOT evaluated; campaign "
                              "capture must record it (M2.7)")
    return out


def promotion_gate(report: dict, *, fairness_cfg: dict) -> tuple:
    """(ok, reasons). The spec-defined margins, applied."""
    reasons = []
    cand = report["candidate_auc"]
    for b in ("tachogram_stats", "hr_only"):
        auc = report["baselines"][b]
        near_ceiling = auc >= AUC_CEILING and cand >= auc
        if not near_ceiling and not (cand >= auc + SIGNAL_MARGIN):
            reasons.append(f"candidate {cand:.3f} does not beat {b} "
                           f"{auc:.3f} by {SIGNAL_MARGIN}")
    for b in ("participant_history", "device_site"):
        auc = report["baselines"][b]
        if auc > DETECTOR_CEILING:
            reasons.append(f"{b} AUC {auc:.3f} > {DETECTOR_CEILING} — "
                           "leakage detector fired; the DATA is refused")
        if not (cand >= auc + DETECTOR_MARGIN):
            reasons.append(f"candidate {cand:.3f} does not exceed {b} "
                           f"{auc:.3f} by {DETECTOR_MARGIN}")
    if report.get("leak_flags"):
        reasons += list(report["leak_flags"])
    rates = [v["no_read_rate"] for g, v in report["parity"].items()
             if g != "unrecorded" and v["n"] >= 3]
    if len(rates) >= 2 and min(rates) > 0:
        ratio = max(rates) / min(rates)
        if ratio > float(fairness_cfg.get("noread_max_ratio", 1.5)):
            reasons.append(f"no-read parity ratio {ratio:.2f} exceeds "
                           f"{fairness_cfg.get('noread_max_ratio', 1.5)}")
    return (not reasons), reasons
