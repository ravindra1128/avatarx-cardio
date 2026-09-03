"""
§R evaluation harness (v0.7 regularity track) — the ceiling test, the
benign-separation battery, the baselines and the fairness table.

The ceiling test (gate R0) runs BOTH paths — camera intervals and ECG
R-R — through the SAME feature code and asks one question: can the
optics reproduce the ECG's own regularity verdict? It isolates "is the
camera good enough" from "is the verdict clinically right", and it runs
on every paired recording there is, healthy volunteers included.

Rows carry, per recording: the camera-path representation and index,
the ECG reference (datasets/regularity_reference.py), the SQI grade,
the participant sidecar (age/sex/skin-tone group) and the respiration
channel from the torso-motion second decode — never the facial trace
the intervals are built from.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from datasets.regularity_reference import (classify_index,
                                           load_reference_definition,
                                           reference_from_rpeaks)
from evaluation.afib_metrics import wilson_ci

SEED = 20260901
SQI_GRADES = (("A", 0.70), ("B", 0.50), ("C", 0.0))
AGE_BANDS = (("<35", 0, 35), ("35-59", 35, 60), (">=60", 60, 200))


def sqi_grade(sqi) -> str:
    try:
        s = float(sqi)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(s):
        return "unknown"
    for name, floor in SQI_GRADES:
        if s >= floor:
            return name
    return "C"


def age_band(age) -> str:
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(a):
        return "unknown"
    for name, lo, hi in AGE_BANDS:
        if lo <= a < hi:
            return name
    return "unknown"


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


# ------------------------------------------------------------- agreement
def cohens_kappa(a, b, classes=("regular", "irregular")) -> dict:
    """Cohen's kappa between two class sequences, restricted to the
    rows where BOTH are in `classes` (indeterminate is reported apart,
    never silently counted as agreement)."""
    pairs = [(x, y) for x, y in zip(a, b) if x in classes and y in classes]
    n = len(pairs)
    if n == 0:
        return {"kappa": None, "n": 0, "observed_agreement": None}
    k = len(classes)
    m = np.zeros((k, k))
    ix = {c: i for i, c in enumerate(classes)}
    for x, y in pairs:
        m[ix[x], ix[y]] += 1
    po = float(np.trace(m) / n)
    pe = float(np.sum(m.sum(axis=0) * m.sum(axis=1)) / n / n)
    # kappa is UNDEFINED when expected agreement is 1 (both raters carry
    # one class): a cohort with no irregular scan on either side must
    # read "no class agreement on record", never a fabricated 1.0
    # (review finding)
    kappa = (po - pe) / (1.0 - pe) if pe < 1.0 else None
    return {"kappa": (None if kappa is None else round(float(kappa), 4)),
            "n": n, "observed_agreement": round(po, 4),
            "expected_agreement": round(pe, 4),
            "single_class": bool(pe >= 1.0),
            "confusion": {f"{x}->{y}": int(m[ix[x], ix[y]])
                          for x in classes for y in classes}}


def pearson_r(x, y) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": None, "n": int(x.size)}
    return {"r": round(float(np.corrcoef(x, y)[0, 1]), 4), "n": int(x.size)}


def bland_altman(camera, reference) -> dict:
    """Bias and 95% limits of agreement of camera - reference."""
    c = np.asarray(camera, float)
    r = np.asarray(reference, float)
    ok = np.isfinite(c) & np.isfinite(r)
    d = c[ok] - r[ok]
    if d.size < 3:
        return {"bias": None, "loa": None, "n": int(d.size)}
    sd = float(np.std(d, ddof=1))
    bias = float(np.mean(d))
    return {"bias": round(bias, 5),
            "sd": round(sd, 5),
            "loa": [round(bias - 1.96 * sd, 5), round(bias + 1.96 * sd, 5)],
            "loa_halfwidth": round(1.96 * sd, 5),
            "n": int(d.size)}


def ceiling_test(rows, *, min_pairs: int = 3) -> dict:
    """R0 evidence: class agreement (kappa), index correlation and
    Bland-Altman, overall and per SQI grade / per dataset."""
    # a PAIR is a scan judged on both sides: index present AND class
    # decided (regular / irregular) by camera and by ECG. Rows either
    # side left indeterminate are counted apart, never inside n —
    # otherwise 99 no-read scans plus one agreeing pair would open R0
    # on n = 100 (review finding)
    judged = ("regular", "irregular")
    paired = [r for r in rows if r.get("ecg_index") is not None
              and r.get("camera_index") is not None
              and r.get("camera_class") in judged
              and r.get("ecg_class") in judged]
    n_unpaired = len(rows) - len(paired)

    def _block(rs):
        return {"n": len(rs),
                "kappa": cohens_kappa([r["camera_class"] for r in rs],
                                      [r["ecg_class"] for r in rs]),
                "index_r": pearson_r([r["camera_index"] for r in rs],
                                     [r["ecg_index"] for r in rs]),
                "bland_altman": bland_altman(
                    [r["camera_index"] for r in rs],
                    [r["ecg_index"] for r in rs]),
                "n_camera_indeterminate": sum(
                    1 for r in rs if r["camera_class"] == "indeterminate"),
                "n_ecg_indeterminate": sum(
                    1 for r in rs if r["ecg_class"] == "indeterminate")}

    per_grade = {}
    for g in sorted({r.get("sqi_grade") for r in paired}):
        rs = [r for r in paired if r.get("sqi_grade") == g]
        per_grade[str(g)] = (_block(rs) if len(rs) >= min_pairs
                             else {"n": len(rs), "reason": "too few pairs"})
    per_ds = {}
    for ds in sorted({str(r.get("dataset") or "unlabeled") for r in paired}):
        rs = [r for r in paired if str(r.get("dataset") or
                                       "unlabeled") == ds]
        per_ds[ds] = (_block(rs) if len(rs) >= min_pairs
                      else {"n": len(rs), "reason": "too few pairs"})
    overall = _block(paired) if paired else {"n": 0}
    # worst grade governs: a grade with enough pairs whose kappa is
    # undefined (single class on one side) is UNRATED, and an unrated
    # grade makes the worst-grade kappa unrated — it is not skipped
    # (review finding)
    rated = [b for b in per_grade.values() if b.get("kappa")]
    kappas = [b["kappa"]["kappa"] for b in rated]
    worst_kappa = (None if (not kappas or any(k is None for k in kappas))
                   else min(kappas))
    # every scan seen on both sides is also counted for the reader,
    # with the reason it is not a pair
    n_indet = sum(1 for r in rows
                  if r.get("ecg_index") is not None
                  and r.get("camera_index") is not None
                  and r not in paired)
    return {"overall": overall, "per_sqi_grade": per_grade,
            "per_dataset": per_ds, "n_paired_scans": len(paired),
            "n_scans": len(rows),
            "n_indeterminate_either_side": n_indet,
            "n_unpaired": n_unpaired,
            "worst_grade_kappa": worst_kappa,
            "kappa": (overall.get("kappa") or {}).get("kappa"),
            "index_r": (overall.get("index_r") or {}).get("r"),
            "bias": (overall.get("bland_altman") or {}).get("bias"),
            "loa_halfwidth": (overall.get("bland_altman") or {})
            .get("loa_halfwidth")}


# ------------------------------------------------------------ rows
# the schema's recording-level precedence (datasets/schema.py
# Recording.recording_level_rhythm): clinical salience first, then the
# longest annotation — never annotation index 0 (review finding)
_RHYTHM_PRECEDENCE = ("AFIB", "ATRIAL_FLUTTER", "SVT", "PVC_FREQUENT",
                      "PAC_FREQUENT", "PVC", "PAC")


def recording_rhythm(man: dict):
    anns = [a for a in (man.get("rhythm_annotations") or [])
            if isinstance(a, dict) and a.get("rhythm")]
    if not anns:
        return None
    names = [str(a["rhythm"]).split(".")[-1] for a in anns]
    for r in _RHYTHM_PRECEDENCE:
        if r in names:
            return r

    def _dur(a):
        try:
            return float(a.get("t_end_s", 0.0)) - float(a.get("t_start_s",
                                                            0.0))
        except (TypeError, ValueError):
            return 0.0
    return str(max(anns, key=_dur)["rhythm"]).split(".")[-1]


def rows_from_dataset(dataset_dir, *, dataset_name=None, gates_path=None,
                      config=None) -> list:
    """Run the PRODUCTION path over a registered paired dataset and
    build one row per recording: camera representation + index +
    class, the ECG reference, SQI grade, participant metadata, and the
    head's verdict (the REAL head, with the torso-derived respiration
    channel)."""
    from heads import get_head
    from inference.pipeline import run_with_details
    from rppg._filters import moving_average_detrend
    from rppg.respiration import (respiratory_rate_from_motion,
                                  torso_motion_series)
    from features.regularity import regularity_from_runs
    d = pathlib.Path(dataset_dir)
    definition = load_reference_definition(gates_path)
    head = get_head("regularity")
    rows = []
    for mp in sorted(d.glob("*.recording.json")):
        man = json.loads(mp.read_text())
        video = d / str(man.get("video_path") or "")
        if not video.exists():
            continue
        rid = mp.name.split(".recording.json")[0]
        pp = d / f"{rid}.participant.json"
        part = json.loads(pp.read_text()) if pp.exists() else {}
        res, det = run_with_details(str(video),
                                    manifest=(man.get("capture") or {}),
                                    config=config)
        try:
            t, y, fps_r = torso_motion_series(str(video))
            rate, conc = respiratory_rate_from_motion(t, y, fps_r)
            resp = {"t": t, "y": moving_average_detrend(y, int(12.0 * fps_r)),
                    "rate_brpm": rate, "quality": conc}
        except (OSError, IOError, ValueError):
            resp = None
        lat = det.get("lattice")
        reg = None
        if lat is not None:
            rs_runs = list(lat.runs)
            reg = regularity_from_runs(
                rs_runs, list(lat.run_confidences),
                det["runset"].dropout_rate,
                mean_sqi=float(det["sqi"].sqi),
                run_times=list(lat.run_times), fps=lat.fps,
                respiration=resp)
        rpeaks = man.get("ecg_rpeaks_s")
        if not rpeaks:
            ep = d / str(man.get("ecg_path") or "")
            if ep.exists():
                rpeaks = (json.loads(ep.read_text()) or {}).get("rpeaks_s")
        # the reference is the ECG over the SPAN THE CAMERA SAW: R-peaks
        # mapped onto the video clock (video_t = ecg_t + offset) and
        # clipped to the video's duration. An ECG record longer than the
        # video would otherwise be compared against a shorter camera
        # series and the difference called optics (review finding).
        ecg_clip = None
        if rpeaks:
            rp = np.asarray([float(x) for x in rpeaks], float)
            offset_s = float(((man.get("sync") or {}).get("offset_ms")
                              or 0.0)) / 1000.0
            ing = det.get("ingest")
            dur = (float(ing.meta.duration_s) if ing is not None and
                   getattr(getattr(ing, "meta", None), "duration_s", None)
                   else _f(man.get("duration_s")))
            on_video = rp + offset_s
            if dur is not None:
                keep = (on_video >= 0.0) & (on_video <= dur + 1e-6)
                ecg_clip = {"offset_s": offset_s, "video_duration_s": dur,
                            "n_rpeaks_total": int(rp.size),
                            "n_rpeaks_in_span": int(keep.sum())}
                rp = on_video[keep]
            else:
                ecg_clip = {"offset_s": offset_s, "video_duration_s": None,
                            "n_rpeaks_total": int(rp.size),
                            "n_rpeaks_in_span": int(rp.size),
                            "note": "video duration unknown — unclipped"}
            rpeaks = rp.tolist()
        ref = reference_from_rpeaks(rpeaks, definition=definition,
                                    config=config) if rpeaks else None
        # the head classifies under the SAME definition object the
        # reference used — a gates_path given here reaches the head too
        # (review finding)
        head_cfg = dict(config or {})
        head_cfg["regularity"] = dict(head_cfg.get("regularity") or {})
        head_cfg["regularity"]["reference_label"] = dict(definition)
        ctx = {"scan_outcome": res.outcome.value, "cfg": head_cfg,
               "respiration": resp, "regularity": reg,
               "age_years": part.get("age_years")}
        hv = head.run(lat, ctx).value if lat is not None else {}
        cam_idx = (reg.index.get("value") if reg is not None else None)
        rows.append({
            "recording_id": man.get("recording_id"),
            "participant_id": man.get("participant_id"),
            "session_id": man.get("session_id"),
            "dataset": dataset_name or d.name,
            "rhythm": recording_rhythm(man),
            "fps": (lat.fps if lat is not None else None),
            "sqi": float(det["sqi"].sqi) if "sqi" in det else None,
            "sqi_grade": sqi_grade(det["sqi"].sqi) if "sqi" in det
            else "unknown",
            "fitzpatrick_group": part.get("fitzpatrick_group"),
            "age_years": part.get("age_years"), "sex": part.get("sex"),
            "age_band": age_band(part.get("age_years")),
            "scan_outcome": res.outcome.value,
            "camera_index": cam_idx,
            "camera_ci95": (reg.index.get("ci95") if reg else None),
            "camera_class": (classify_index(cam_idx, reg.n_intervals,
                                            definition)
                             if reg is not None and
                             res.outcome.value == "ACCEPT"
                             else "indeterminate"),
            "camera_rmssd_ms": (reg.values.get("rmssd") if reg else None),
            "camera_n_intervals": (int(reg.n_intervals) if reg else 0),
            "camera_features": (reg.to_dict() if reg else None),
            "ecg_index": (ref["index"] if ref else None),
            "ecg_class": (ref["class"] if ref else "indeterminate"),
            "ecg_rmssd_ms": (ref["rmssd_ms"] if ref else None),
            "ecg_n_intervals": (ref["n_intervals"] if ref else 0),
            "head": hv,
            "resp_rate_brpm": (resp or {}).get("rate_brpm"),
            "resp_quality": (resp or {}).get("quality"),
            "ecg_span": ecg_clip,
        })
    return rows


# ------------------------------------------ benign separation (R2)
# The reference label is a REGULARITY label; RSA is irregular by it and
# benign by rhythm. R2 therefore asks a clinical question of the
# annotated rhythm: how often does the head call a benign rhythm
# "clinically irregular" — an irregular class WITHOUT a benign
# explanation attached?
BENIGN_RHYTHMS = {"SINUS", "SINUS_BRADYCARDIA", "SINUS_TACHYCARDIA",
                  "RESPIRATORY_SINUS_ARRHYTHMIA", "PAC"}
RSA_RHYTHMS = {"RESPIRATORY_SINUS_ARRHYTHMIA"}


def clinically_irregular(head_value) -> bool:
    """An irregular class the head could NOT explain as breathing.
    Ectopy and chaotic explanations still count as clinically
    irregular — they are irregularities an ECG should look at."""
    v = head_value or {}
    if v.get("class") != "irregular":
        return None if v.get("class") is None else False
    ev = (v.get("benign_pattern_evidence") or {}).get("evidence")
    return ev != "respiration_coupled"


def benign_separation(rows) -> dict:
    """RSA flag rate and age-stratified specificity against annotated
    BENIGN rhythms, at session level. Every denominator is reported
    THREE ways — total, judged, no-read — because a head that abstains
    on the sessions it would get wrong can otherwise buy its
    specificity with silence (review finding)."""
    judged = [r for r in rows if clinically_irregular(r.get("head"))
              is not None]
    rsa_all = [r for r in rows if str(r.get("rhythm")) in RSA_RHYTHMS]
    rsa = [r for r in judged if str(r.get("rhythm")) in RSA_RHYTHMS]
    rsa_flag = [clinically_irregular(r["head"]) for r in rsa]
    strata = {}
    for name, _, _ in AGE_BANDS:
        rs_all = [r for r in rows if r.get("age_band") == name
                  and str(r.get("rhythm")) in BENIGN_RHYTHMS]
        rs = [r for r in rs_all if clinically_irregular(r.get("head"))
              is not None]
        fp = sum(1 for r in rs if clinically_irregular(r["head"]))
        strata[name] = {"n": len(rs), "n_total": len(rs_all),
                        "n_no_read": len(rs_all) - len(rs),
                        "no_read_rate": (round(1.0 - len(rs) / len(rs_all),
                                               4) if rs_all else None),
                        "n_false_irregular": fp,
                        "specificity": (round(1.0 - fp / len(rs), 4)
                                        if rs else None),
                        "specificity_ci95": ([round(x, 4) for x in
                                              wilson_ci(len(rs) - fp,
                                                        len(rs))]
                                             if rs else None)}
    benign_all = [r for r in rows if str(r.get("rhythm")) in BENIGN_RHYTHMS]
    benign = [r for r in judged if str(r.get("rhythm")) in BENIGN_RHYTHMS]
    n_fp = sum(1 for r in benign if clinically_irregular(r["head"]))
    return {"n_rsa_sessions": len(rsa),
            "n_rsa_sessions_total": len(rsa_all),
            "rsa_no_read_rate": (round(1.0 - len(rsa) / len(rsa_all), 4)
                                 if rsa_all else None),
            "rsa_flag_rate": (round(float(np.mean(rsa_flag)), 4)
                              if rsa_flag else None),
            "rsa_explained_rate": (round(float(np.mean(
                [(r["head"].get("benign_pattern_evidence") or {})
                 .get("evidence") == "respiration_coupled" for r in rsa])),
                4) if rsa else None),
            "age_strata": strata,
            "n_unknown_age_band": sum(
                1 for r in benign_all
                if r.get("age_band") not in {b for b, _, _ in AGE_BANDS}),
            "benign_specificity_overall": (round(1.0 - n_fp / len(benign), 4)
                                           if benign else None),
            "n_benign_sessions": len(benign),
            "n_benign_sessions_total": len(benign_all),
            "n_abstained": sum(1 for r in rows
                               if clinically_irregular(r.get("head"))
                               is None)}


# ------------------------------------------------- baselines (R3)
def _feat_value(row, key):
    cf = row.get("camera_features") or {}
    v = (cf.get("values") or {}).get(key)
    return _f(v)


def _label(row):
    c = row.get("ecg_class")
    return None if c not in ("regular", "irregular") else int(c == "irregular")


def _participant_split(rows, *, seed=SEED, frac=0.5):
    tr, te = [], []
    for r in rows:
        pid = str(r.get("participant_id"))
        h = hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()
        (tr if int(h[:8], 16) / 0xFFFFFFFF < frac else te).append(r)
    return tr, te


def _balanced_accuracy(y, pred):
    y = np.asarray(y, int)
    p = np.asarray(pred, int)
    if y.size == 0 or y.sum() == 0 or (1 - y).sum() == 0:
        return None
    sens = float(np.mean(p[y == 1] == 1))
    spec = float(np.mean(p[y == 0] == 0))
    return round(0.5 * (sens + spec), 4)


def _best_threshold(x, y):
    """Threshold on a scalar maximising balanced accuracy — chosen on
    TRAIN rows only."""
    x = np.asarray([np.nan if v is None else v for v in x], float)
    y = np.asarray(y, int)
    ok = np.isfinite(x)
    if ok.sum() < 4 or y[ok].sum() == 0 or (1 - y[ok]).sum() == 0:
        return None
    best = None
    for thr in np.unique(x[ok]):
        pred = np.where(ok, x >= thr, 0).astype(int)
        ba = _balanced_accuracy(y, pred)
        if ba is not None and (best is None or ba > best[0]):
            best = (ba, float(thr))
    return None if best is None else best[1]


def _logistic(Xtr, ytr, Xte):
    from models.baseline import LogisticModel
    ytr = np.asarray(ytr, int)
    if ytr.size < 4 or ytr.sum() == 0 or (1 - ytr).sum() == 0:
        return None
    m = LogisticModel().fit(np.asarray(Xtr, float), ytr)
    return np.asarray(m.predict_proba(np.asarray(Xte, float)), float)


def baselines(rows, *, seed=SEED) -> dict:
    """B1 RMSSD threshold; B2 Shannon-entropy threshold; B3 B1+B2+rate
    (logistic); B4 demographics only (age/sex); B5 SQI only. All on the
    SAME participant-disjoint split, scored on held-out rows by balanced
    accuracy against the ECG reference class. The head is scored on the
    rows it judged; its no-read rate rides beside it."""
    labeled = [r for r in rows if _label(r) is not None]
    tr, te = _participant_split(labeled, seed=seed)
    out = {"n_train_rows": len(tr), "n_test_rows": len(te),
           "n_test_participants": len({str(r.get("participant_id"))
                                       for r in te}),
           "split_seed": seed}
    if not tr or not te:
        out["reason"] = "no participant-disjoint split with rows on both sides"
        return out
    yte = [_label(r) for r in te]
    ytr = [_label(r) for r in tr]

    def _thr_baseline(name, key):
        thr = _best_threshold([_feat_value(r, key) for r in tr], ytr)
        if thr is None:
            out[f"{name}_balanced_accuracy"] = None
            return
        pred = [int(_feat_value(r, key) is not None and
                    _feat_value(r, key) >= thr) for r in te]
        out[f"{name}_balanced_accuracy"] = _balanced_accuracy(yte, pred)
        out[f"{name}_threshold"] = round(thr, 5)
        preds[name] = np.asarray(pred, int)

    preds = {}
    _thr_baseline("b1", "rmssd")
    _thr_baseline("b2", "shannon_entropy")

    def _X(rs, keys):
        return [[(_feat_value(r, k) if k in ("rmssd", "shannon_entropy")
                  else (60000.0 / _feat_value(r, "median_ibi")
                        if k == "rate" and _feat_value(r, "median_ibi")
                        else (_f(r.get("age_years")) if k == "age"
                              else (1.0 if str(r.get("sex") or "").upper()
                                    .startswith("F") else 0.0) if k == "sex"
                              else _f(r.get("sqi")) if k == "sqi" else None)))
                 if True else None for k in keys] for r in rs]

    def _nan(v):
        return [[np.nan if x is None else x for x in row] for row in v]

    for name, keys in (("b3", ("rmssd", "shannon_entropy", "rate")),
                       ("b4", ("age", "sex")), ("b5", ("sqi",))):
        Xtr, Xte = _nan(_X(tr, keys)), _nan(_X(te, keys))
        if all(all(not np.isfinite(x) for x in row) for row in Xtr):
            out[f"{name}_balanced_accuracy"] = None
            out[f"{name}_reason"] = f"no {keys} available on train rows"
            continue
        # the operating point is chosen on TRAIN for the metric the gate
        # reads (balanced accuracy), never a fixed 0.5 — under class
        # imbalance 0.5 collapses to the majority class and a leaking
        # baseline reads chance (review finding)
        ptr = _logistic(Xtr, ytr, Xtr)
        p = _logistic(Xtr, ytr, Xte)
        if p is None or ptr is None:
            out[f"{name}_balanced_accuracy"] = None
            continue
        cut = _best_threshold(list(ptr), ytr)
        cut = 0.5 if cut is None else cut
        pred = (p >= cut).astype(int)
        out[f"{name}_threshold"] = round(float(cut), 5)
        out[f"{name}_balanced_accuracy"] = _balanced_accuracy(yte, pred)
        preds[name] = pred
    # the head, on the rows it judged — and every baseline re-scored on
    # THOSE rows, so the gate compares like with like (review finding:
    # the head was scored on an easier subset than the baselines)
    mask = [(r.get("head") or {}).get("class") in ("regular", "irregular")
            for r in te]
    judged = [(r, yl) for r, yl, m in zip(te, yte, mask) if m]
    out["head_balanced_accuracy"] = _balanced_accuracy(
        [yl for _, yl in judged],
        [int(r["head"]["class"] == "irregular") for r, _ in judged]) \
        if judged else None
    out["head_no_read_rate"] = round(1.0 - len(judged) / len(te), 4)
    out["n_test_participants_judged"] = len({str(r.get("participant_id"))
                                             for r, _ in judged})
    yj = [yl for _, yl in judged]
    for name, pred in preds.items():
        pj = [int(x) for x, m in zip(pred, mask) if m]
        out[f"{name}_balanced_accuracy_on_judged"] = (
            _balanced_accuracy(yj, pj) if judged else None)
    return out


# ------------------------------------------------- fairness (R4)
def fairness(rows, *, min_group_participants: int = 5,
             darkest_bands=(5, 6)) -> dict:
    groups = {}
    for r in rows:
        g = r.get("fitzpatrick_group")
        if g is None:
            continue
        groups.setdefault(g, []).append(r)
    table, bias, flag, cov = {}, {}, {}, {}
    for g, rs in groups.items():
        pids = {str(r.get("participant_id")) for r in rs}
        paired = [(r["camera_index"], r["ecg_index"]) for r in rs
                  if r.get("camera_index") is not None
                  and r.get("ecg_index") is not None]
        judged = [r for r in rs if clinically_irregular(r.get("head"))
                  is not None]
        row = {"n_scans": len(rs), "n_participants": len(pids),
               "index_bias": (round(float(np.mean([c - e for c, e in
                                                   paired])), 5)
                              if paired else None),
               "flag_rate": (round(float(np.mean(
                   [clinically_irregular(r["head"]) for r in judged])), 4)
                   if judged else None),
               "coverage": (round(len(judged) / len(rs), 4) if rs else None),
               "rated": len(pids) >= min_group_participants}
        if row["rated"]:
            if row["index_bias"] is not None:
                bias[g] = row["index_bias"]
            if row["flag_rate"] is not None:
                flag[g] = row["flag_rate"]
            if row["coverage"] is not None:
                cov[g] = row["coverage"]
        table[str(g)] = row
    unrated = [str(g) for g, r in table.items() if not r["rated"]]

    def _ratio(d):
        return (round(min(d.values()) / max(d.values()), 4)
                if len(d) >= 2 and max(d.values()) > 0 else None)

    return {"fitzpatrick_group": {
        "groups": table,
        "worst_index_bias": (None if (unrated or not bias)
                             else round(max(bias.values(), key=abs), 5)),
        "flag_rate_parity_ratio_worst": (None if unrated else _ratio(flag)),
        "coverage_ratio_worst": (None if unrated else _ratio(cov)),
        "unrated_groups": unrated, "n_rated_groups": len(cov),
        "darkest_band_present": bool(any(g in groups
                                         for g in darkest_bands))}}


# ------------------------------------------------- the evaluation
def evaluate_regularity_dataset(rows, *, signal_domain, runs_root=None,
                                gates_path=None,
                                production_path: bool = False,
                                seed: int = SEED) -> dict:
    """Every §R evaluation over paired rows, recorded as an
    "evaluation" scoreboard entry. `signal_domain` is REQUIRED and
    `production_path` defaults False — a permissive default would let
    surrogate rows be recorded as the one evidence class that can open
    a gate."""
    from evaluation.regularity_gates import (append_scoreboard,
                                             DEFAULT_RUNS,
                                             evaluate_regularity_gates,
                                             latest_scoreboard_entry,
                                             load_regularity_gates)
    rows = list(rows or [])
    if not rows:
        raise ValueError("no paired scans supplied — an evaluation with "
                         "no cohort is not an evaluation")
    gcfg = load_regularity_gates(gates_path)
    t4 = gcfg.get("r4_fairness") or {}
    r0 = ceiling_test(rows)
    r2 = benign_separation(rows)
    r3 = baselines(rows, seed=seed)
    r4 = fairness(rows, min_group_participants=int(
        t4.get("min_group_participants", 5)))
    r5 = {"escalation_path_present": False,
          "sentence_family": "irregular_rhythm_notification"}
    pids = sorted({str(r.get("participant_id")) for r in rows})
    sids = sorted({str(r.get("session_id") or r.get("recording_id"))
                   for r in rows})
    evidence = {
        "data": {"signal_domain": signal_domain,
                 "participant_disjoint": len(pids) > 1,
                 "session_disjoint": len(sids) > 1,
                 "production_path": bool(production_path),
                 "n_scans": len(rows), "n_participants": len(pids)},
        "r0": r0, "r2": r2, "r3": r3, "r4": r4, "r5": r5}
    floor = latest_scoreboard_entry("floor", runs_root=runs_root) or {}
    verdict = evaluate_regularity_gates(gcfg, evidence, floor)
    run_id = "reg-" + hashlib.sha256(json.dumps(
        {"n": len(rows),
         "ids": sorted(str(r.get("recording_id")) for r in rows),
         "domain": signal_domain, "production_path": bool(production_path),
         "gates_version": gcfg.get("gates_version")},
        sort_keys=True).encode()).hexdigest()[:12]
    root = pathlib.Path(runs_root or DEFAULT_RUNS)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gate_results.json").write_text(json.dumps(verdict, indent=1,
                                                          default=str))
    # the evidence itself rides with the run so the registry door can
    # RE-EVALUATE it under the live gates.yaml rather than trust the
    # verdict file (review finding: a forged gate_results.json promoted)
    (run_dir / "evidence.json").write_text(json.dumps(
        {"evidence": evidence, "floor": floor,
         "gates_version": gcfg.get("gates_version")}, indent=1,
        default=str))
    (run_dir / "regularity_record.json").write_text(json.dumps(
        {"run_id": run_id, "n_scans": len(rows),
         "n_participants": len(pids),
         "gates_version": gcfg.get("gates_version")}, indent=1))
    (run_dir / "model.json").write_text(json.dumps(
        {"kind": "regularity_threshold_rule",
         "reference_label": gcfg.get("reference_label")}, indent=1,
        default=str))
    append_scoreboard({"kind": "evaluation", "run_id": run_id,
                       "gates_version": gcfg.get("gates_version"),
                       "reference_label": gcfg.get("reference_label"),
                       "floor_run_id": floor.get("run_id"),
                       "evidence": evidence,
                       "verdict": {"all_gates_green":
                                   verdict["all_gates_green"],
                                   "promotion_open":
                                   verdict["promotion_open"]}},
                      runs_root=runs_root)
    return {"run_id": run_id, "run_dir": str(run_dir),
            "evidence": evidence, "verdict": verdict}
