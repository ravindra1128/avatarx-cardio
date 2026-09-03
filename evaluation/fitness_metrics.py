"""
Fitness evaluation harness (v0.4 T6) — §6 of the track contract.

The heart is the MANDATORY BASELINE LADDER, computed in the same run on
identical participant-level splits: age; age+sex; +BMI; resting-HR-only;
activity-only; demographics+activity; static-face-image (NEGATIVE
CONTROL — if appearance alone matches the physiology model, the model is
reading faces, not fitness); resting physiology; recovery physiology;
full. §V's pivotal V3 is judged from this ladder: recovery physiology
must beat demographics+activity by ΔSEE >= 0.5 mL/kg/min with a CI
excluding zero AND category accuracy >= +5 pts AND beat the negative
control — otherwise head_fitness is never promoted and the product ships
recovery metrics only.

Falsification probes ride in the same module (the §G lab is quarantined
research code; these are harness-side): shuffled-workload (recovery
features must degrade when the workload labels are scrambled), the
demographics-only impostor (the ladder itself), and the transition-time
sensitivity sweep.

VO2 numbers live HERE and in labels only — never on a user surface.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import numpy as np

from configs import config_hash, load_config
from datasets.schema import ScanOutcome, cpet_from_dict
from datasets.splits import _stable_unit_interval
from evaluation.fitness_gates import (DEFAULT_RUNS, append_scoreboard,
                                      evaluate_vo2_gates, load_vo2_gates)

SYNTHETIC_NOTICE = ("Every number in this report derives from SYNTHETIC "
                    "interface-proof data unless the dataset provenance "
                    "says otherwise; nothing here is a performance claim.")
TRAIN_FRACTION = 0.7
RIDGE_L2 = 1.0

# provisional VO2peak category anchors (p33, p66) by age decade,
# mL/kg/min, sex-neutral v1 — evaluation-side only, REQUIRES_CLINICAL_
# SIGNOFF like everything §V touches
_VO2_BANDS = ((30, 38.0, 48.0), (40, 34.0, 44.0), (50, 30.0, 40.0),
              (60, 26.0, 35.0), (70, 22.0, 31.0), (200, 19.0, 27.0))


# ------------------------------------------------- agreement metrics
def see(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return round(float(np.sqrt(np.mean((y - yhat) ** 2))), 3)


def mae(y, yhat) -> float:
    return round(float(np.mean(np.abs(np.asarray(y, float)
                                      - np.asarray(yhat, float)))), 3)


def bland_altman(y, yhat) -> dict:
    d = np.asarray(yhat, float) - np.asarray(y, float)
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
    return {"bias": round(bias, 3),
            "loa_lower": round(bias - 1.96 * sd, 3),
            "loa_upper": round(bias + 1.96 * sd, 3),
            "n": int(d.size)}


def vo2_category(vo2, age) -> str:
    a = 45.0 if age is None else float(age)
    for age_max, p33, p66 in _VO2_BANDS:
        if a < age_max:
            break
    return ("below" if vo2 < p33 else "above" if vo2 > p66 else "typical")


def category_accuracy(y, yhat, ages) -> float:
    hits = [vo2_category(a, g) == vo2_category(b, g)
            for a, b, g in zip(y, yhat, ages)]
    return round(float(np.mean(hits)), 3) if hits else float("nan")


# ------------------------------------------------- the ladder
def _sex01(r):
    return {"female": 0.0, "male": 1.0}.get(r.get("sex"), 0.5)


def _bmi(r):
    w, h = r.get("weight_kg"), r.get("height_cm")
    if not w or not h:
        return None
    return float(w) / (float(h) / 100.0) ** 2


def _ipaq01(r):
    return {"low": 0.0, "moderate": 0.5, "high": 1.0}.get(
        r.get("activity_ipaq"), 0.5)


LADDER = (
    ("age", lambda r: [r.get("age")]),
    ("age_sex", lambda r: [r.get("age"), _sex01(r)]),
    ("age_sex_bmi", lambda r: [r.get("age"), _sex01(r), _bmi(r)]),
    ("resting_hr_only", lambda r: [r.get("hr_rest_bpm")]),
    ("activity_only", lambda r: [r.get("est_mets"), _ipaq01(r)]),
    ("demographics_activity",
     lambda r: [r.get("age"), _sex01(r), _bmi(r), r.get("est_mets"),
                _ipaq01(r)]),
    ("static_face_image",                      # NEGATIVE CONTROL
     lambda r: list(r.get("static_face") or [None] * 6)),
    ("resting_physiology",
     lambda r: [r.get("hr_rest_bpm"), r.get("rr_rest_brpm")]),
    ("recovery_physiology",
     lambda r: [r.get("hr_end_proxy_bpm"), r.get("hrr30_bpm"),
                r.get("hrr60_bpm"), r.get("recovery_slope_bpm_min")]),
    ("full",
     lambda r: [r.get("age"), _sex01(r), _bmi(r), r.get("est_mets"),
                _ipaq01(r), r.get("hr_end_proxy_bpm"),
                r.get("hrr30_bpm"), r.get("hrr60_bpm"),
                r.get("recovery_slope_bpm_min")]),
)


def _matrix(records, fn) -> np.ndarray:
    X = np.array([[np.nan if v is None else float(v) for v in fn(r)]
                  for r in records], float)
    return X


def _ridge(Xtr, ytr, Xte, l2: float = RIDGE_L2) -> np.ndarray:
    """Ridge with an UNPENALIZED intercept: y is centred and only the
    standardized slopes are shrunk. (v0.4 review finding: penalizing the
    ones column against uncentred VO2 shrank every prediction by
    ~mean(y)/(n+1), inflating every rung's SEE/bias at small n and
    making ladder numbers incomparable across cohort sizes.)"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN column
        mu = np.nanmean(Xtr, axis=0)                      # -> imputed 0

    mu = np.where(np.isfinite(mu), mu, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, mu)
    Xte = np.where(np.isfinite(Xte), Xte, mu)
    sd = Xtr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    y = np.asarray(ytr, float)
    ybar = float(np.mean(y))
    coef = np.linalg.solve(Ztr.T @ Ztr + l2 * np.eye(Ztr.shape[1]),
                           Ztr.T @ (y - ybar))
    return ybar + Zte @ coef


def _split(records, seed) -> tuple:
    """Participant-disjoint QUANTILE split by stable hash — deterministic
    and identity-only, and non-empty on both sides for n >= 2 pids (a
    raw hash threshold is lumpy at small n; same lesson as the v0.2
    MIMIC subject split)."""
    pids = sorted({str(r["participant_id"]) for r in records},
                  key=lambda p: (_stable_unit_interval(p, seed), p))
    n_test = max(1, int(round((1.0 - TRAIN_FRACTION) * len(pids)))) \
        if len(pids) >= 2 else 0
    test_pids = set(pids[:n_test])
    tr = [r for r in records
          if str(r["participant_id"]) not in test_pids]
    te = [r for r in records if str(r["participant_id"]) in test_pids]
    return tr, te


def fit_ladder(records: list, *, seed: int = 20260830,
               n_boot: int = 200) -> dict:
    """Every rung fitted on the SAME participant-disjoint split; ΔSEE and
    category gain (full vs demographics+activity) with participant-
    cluster bootstrap CIs; the static-face comparison alongside."""
    tr, te = _split(records, seed)
    if len(tr) < 3 or len(te) < 2:
        return {"available": False,
                "reason": f"too few participants for a disjoint ladder "
                          f"(train {len(tr)}, test {len(te)})"}
    ytr = [r["vo2"] for r in tr]
    yte = np.asarray([r["vo2"] for r in te], float)
    ages = [r.get("age") for r in te]
    rungs = {}
    preds = {}
    for name, fn in LADDER:
        yhat = _ridge(_matrix(tr, fn), ytr, _matrix(te, fn))
        preds[name] = yhat
        rungs[name] = {"see": see(yte, yhat), "mae": mae(yte, yhat),
                       **bland_altman(yte, yhat),
                       "category_accuracy":
                           category_accuracy(yte, yhat, ages)}
    base, full, static = (rungs["demographics_activity"], rungs["full"],
                          rungs["static_face_image"])
    delta_see = round(base["see"] - full["see"], 3)
    gain_pts = round((full["category_accuracy"]
                      - base["category_accuracy"]) * 100.0, 1)

    by_pid = {}
    for i, r in enumerate(te):
        by_pid.setdefault(str(r["participant_id"]), []).append(i)

    def _boot(stat_fn):
        # true with-replacement CLUSTER bootstrap: a participant drawn k
        # times contributes its rows k times (v0.4 review finding:
        # collapsing draws with set() deflated CI widths ~0.8x — anti-
        # conservative for the pivotal V3 "CI excludes zero" test)
        rng = np.random.default_rng(seed + 1)
        pids = sorted(by_pid)
        if len(pids) < 4:
            return None
        vals = []
        for _ in range(n_boot):
            draw = rng.choice(pids, size=len(pids), replace=True)
            idx = [i for p in draw for i in by_pid[p]]
            if len(idx) < 2:
                continue
            vals.append(stat_fn(np.asarray(idx)))
        if len(vals) < 10:
            return None
        return [round(float(np.percentile(vals, 2.5)), 3),
                round(float(np.percentile(vals, 97.5)), 3)]

    dsee_ci = _boot(lambda idx: see(yte[idx],
                                    preds["demographics_activity"][idx])
                    - see(yte[idx], preds["full"][idx]))
    gain_ci = _boot(lambda idx: (category_accuracy(
        yte[idx], preds["full"][idx], [ages[i] for i in idx])
        - category_accuracy(yte[idx],
                            preds["demographics_activity"][idx],
                            [ages[i] for i in idx])) * 100.0)
    return {"available": True, "n_train": len(tr), "n_test": len(te),
            "rungs": rungs,
            "delta_see_mlkgmin": delta_see, "delta_see_ci95": dsee_ci,
            "category_accuracy_gain_pts": gain_pts,
            "category_gain_ci95": gain_ci,
            # "beats" = strictly better on the primary metric (SEE) and
            # no worse on the coarse 3-class category accuracy
            "beats_static_face_control":
                bool(full["see"] < static["see"]
                     and full["category_accuracy"]
                     >= static["category_accuracy"])}


# ------------------------------------------------- falsification probes
def shuffled_workload_test(records: list, *, seed: int = 20260830) -> dict:
    """Scramble the workload context across participants: if the full
    model does NOT degrade, it never used the standardized workload and
    the protocol is decoration."""
    base = fit_ladder(records, seed=seed)
    if not base.get("available"):
        return {"available": False, "reason": base.get("reason")}
    rng = np.random.default_rng(seed + 7)
    mets = [r.get("est_mets") for r in records]
    perm = rng.permutation(len(mets))
    shuffled = [dict(r, est_mets=mets[j])
                for r, j in zip(records, perm)]
    alt = fit_ladder(shuffled, seed=seed)
    # STRICTLY worse, or the probe failed: an unchanged SEE means the
    # model never used the workload — exactly what this must catch
    return {"available": True,
            "see_full": base["rungs"]["full"]["see"],
            "see_full_shuffled_workload": alt["rungs"]["full"]["see"],
            "degrades": bool(alt["rungs"]["full"]["see"]
                             > base["rungs"]["full"]["see"] + 1e-9)}


def transition_sensitivity(beat_t, beat_conf, duration_s,
                           offsets_s=(0.0, 2.0, 5.0, 10.0)) -> dict:
    """How much a late recovery start costs: drop the first k seconds
    (as a slow transition would) and watch HRR60 shrink — the reason for
    the <=5 s target / 10 s hard timeout."""
    from features.recovery import recovery_metrics
    beat_t = np.asarray(beat_t, float)
    out = {}
    for k in offsets_s:
        keep = beat_t >= k
        m = recovery_metrics(beat_t[keep] - k,
                             np.asarray(beat_conf, float)[keep],
                             duration_s=float(duration_s) - k)
        out[f"+{k:.0f}s"] = m.get("hrr60")
    return out


# ------------------------------------------------- static-face control
def static_face_features(video_path: str):
    """Appearance-only features from ONE frame (centre-crop channel
    means + brightness moments). Deliberately crude — the point is a
    negative control that carries appearance information and zero
    physiology."""
    from capture.video_reader import iter_frames
    for _, frame in iter_frames(str(video_path)):
        h, w = frame.shape[:2]
        crop = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4, :]
        means = [float(crop[:, :, c].mean()) for c in (2, 1, 0)]  # RGB
        g = crop.mean(axis=2)
        rows = g.mean(axis=1)
        tot = float(rows.sum()) or 1.0
        cy = float((rows * np.arange(rows.size)).sum() / tot) / rows.size
        return means + [float(g.std()), cy, w / max(h, 1)]
    return None


# ------------------------------------------------- the harness
def evaluate_fitness_dataset(dataset_dir, out_dir=None, *,
                             signal_domain: str = "synthetic",
                             runs_root=None, gates_path=None) -> dict:
    """§6: sessions through the PRODUCTION session path, CPET labels
    fail-closed, EXCLUDED-with-reasons, the ladder + probes, and a §V
    gate verdict recorded to the fitness scoreboard (promote reads the
    run dir and refuses while red)."""
    from protocol.session import run_session

    d = pathlib.Path(dataset_dir)
    session_files = sorted(d.glob("*.session.json"))
    if not session_files:
        raise ValueError(f"no *.session.json files in {d}")
    cfg = load_config()
    rows, excluded = [], []
    for sp in session_files:
        sid = sp.name[:-len(".session.json")]
        cpet_p = d / f"{sid}.cpet.json"
        if not cpet_p.exists():
            excluded.append({"session_id": sid,
                             "reasons": ["no CPET label "
                                         f"({cpet_p.name} missing)"]})
            continue
        try:
            with open(cpet_p) as f:
                cpet = cpet_from_dict(json.load(f))
        except ValueError as e:
            excluded.append({"session_id": sid,
                             "reasons": [f"invalid CPET label: {e}"]})
            continue
        result, det = run_session(sp, config=cfg)
        if result.outcome is not ScanOutcome.ACCEPT:
            excluded.append({"session_id": sid,
                             "reasons": [f"session outcome "
                                         f"{result.outcome.value}"]
                             + list(result.no_read_reasons)})
            continue
        man = json.loads(sp.read_text())
        pc = man.get("participant_context") or {}
        rec_head = next((h for h in result.head_results
                         if h["head"] == "recovery"), {"value": {}})
        wc = rec_head["value"].get("workload_context") or {}
        rv = pathlib.Path((man["phases"]["rest"] or {}).get("video"))
        rest_video = rv if rv.is_absolute() else (d / rv)
        rows.append({
            "session_id": sid,
            "participant_id": str(man.get("participant_id", sid)),
            "age": pc.get("age"), "sex": pc.get("sex"),
            "weight_kg": pc.get("measured_weight_kg"),
            "height_cm": pc.get("height_cm"),
            "activity_ipaq": pc.get("activity_ipaq"),
            "est_mets": wc.get("est_mets"),
            "hr_rest_bpm": result.hr_rest_bpm,
            "rr_rest_brpm": result.rr_rest_brpm,
            "hr_end_proxy_bpm": result.hr_end_proxy_bpm,
            "hrr30_bpm": result.hrr30_bpm,
            "hrr60_bpm": result.hrr60_bpm,
            "recovery_slope_bpm_min": result.recovery_slope_bpm_min,
            "stars": result.confidence_stars,
            "static_face": static_face_features(rest_video),
            "vo2": cpet.vo2peak_mlkgmin,
        })

    ladder = (fit_ladder(rows) if len(rows) >= 5
              else {"available": False,
                    "reason": f"only {len(rows)} labeled ACCEPT sessions"})
    probes = {"shuffled_workload": (shuffled_workload_test(rows)
                                    if ladder.get("available")
                                    else {"available": False})}
    evidence = {
        "data": {"signal_domain": signal_domain,
                 "participant_disjoint": True, "session_disjoint": True,
                 "production_path": signal_domain == "facial_rppg",
                 "n_sessions": len(rows), "dataset_dir": str(d)},
        "v3": ({"delta_see_mlkgmin": ladder.get("delta_see_mlkgmin"),
                "delta_see_ci95": ladder.get("delta_see_ci95"),
                "category_accuracy_gain_pts":
                    ladder.get("category_accuracy_gain_pts"),
                "category_gain_ci95": ladder.get("category_gain_ci95"),
                "beats_static_face_control":
                    ladder.get("beats_static_face_control")}
               if ladder.get("available") else {}),
    }
    verdict = evaluate_vo2_gates(load_vo2_gates(gates_path), evidence)

    run_id = "fit-" + hashlib.sha256(
        (str(d) + signal_domain
         + ",".join(r["session_id"] for r in rows)).encode()
    ).hexdigest()[:12]
    run_dir = pathlib.Path(runs_root or DEFAULT_RUNS) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "track": "vo2_fitness",
              "dataset_dir": str(d), "n_included": len(rows),
              "n_excluded": len(excluded),
              "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(run_dir / "fitness_record.json", "w") as f:
        json.dump(record, f, indent=1)
    with open(run_dir / "gate_results.json", "w") as f:
        json.dump({"run_id": run_id, "evidence": evidence, **verdict},
                  f, indent=1)
    with open(run_dir / "model.json", "w") as f:
        json.dump({"ladder": {k: v for k, v in ladder.items()
                              if k != "rungs"} if ladder else None},
                  f, indent=1)
    append_scoreboard({"run_id": run_id, "evaluated_at":
                       record["evaluated_at"], "evidence": evidence,
                       "gates": verdict["gates"],
                       "all_gates_green": verdict["all_gates_green"],
                       "promotion_open": verdict["promotion_open"]},
                      runs_root=runs_root)

    doc = {"provenance": {"config_hash": config_hash(cfg),
                          "dataset_dir": str(d),
                          "signal_domain": signal_domain,
                          "synthetic_notice": SYNTHETIC_NOTICE,
                          "run_id": run_id},
           "excluded": excluded, "n_included": len(rows),
           "per_session": [{k: v for k, v in r.items()
                            if k != "static_face"} for r in rows],
           "baseline_ladder": ladder, "falsification": probes,
           "gates": {"promotion": ("OPEN" if verdict["promotion_open"]
                                   else "BLOCKED"),
                     "verdict": verdict}}
    out = pathlib.Path(out_dir) if out_dir else d / "report_fitness"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "evaluation_report.json", "w") as f:
        json.dump(doc, f, indent=2, default=str)
    (out / "evaluation_report.md").write_text(_render_markdown(doc))
    return {"report_json": str(out / "evaluation_report.json"),
            "report_md": str(out / "evaluation_report.md"),
            "run_dir": str(run_dir), "report": doc}


def _render_markdown(doc: dict) -> str:
    L = ["# Fitness-track evaluation (§6)", "",
         f"> {doc['provenance']['synthetic_notice']}", "",
         f"Run `{doc['provenance']['run_id']}` · domain "
         f"{doc['provenance']['signal_domain']} · included "
         f"{doc['n_included']} · excluded {len(doc['excluded'])}", "",
         "## Excluded sessions"]
    for e in doc["excluded"]:
        L.append(f"- {e['session_id']}: {'; '.join(e['reasons'])}")
    L += ["", "## Mandatory baseline ladder"]
    lad = doc["baseline_ladder"]
    if lad.get("available"):
        L.append("| rung | SEE | MAE | bias | LoA | category acc |")
        L.append("|---|---|---|---|---|---|")
        for name, _ in LADDER:
            r = lad["rungs"][name]
            L.append(f"| {name} | {r['see']} | {r['mae']} | {r['bias']} "
                     f"| [{r['loa_lower']}, {r['loa_upper']}] "
                     f"| {r['category_accuracy']} |")
        L += ["",
              f"ΔSEE (full vs demographics+activity): "
              f"**{lad['delta_see_mlkgmin']}** mL/kg/min "
              f"(CI {lad['delta_see_ci95']}); category gain "
              f"{lad['category_accuracy_gain_pts']} pts "
              f"(CI {lad['category_gain_ci95']}); beats static-face "
              f"control: {lad['beats_static_face_control']}"]
    else:
        L.append(f"unavailable: {lad.get('reason')}")
    sw = doc["falsification"]["shuffled_workload"]
    L += ["", "## Falsification",
          f"- shuffled workload: {json.dumps(sw)}"]
    L += ["", f"## §V verdict: {doc['gates']['promotion']}"]
    for g in doc["gates"]["verdict"]["gates"]:
        L.append(f"- {g['title']}: **{g['status']}** — "
                 + ("; ".join(g["reasons"][:2]) if g["reasons"]
                    else "pass"))
    return "\n".join(L) + "\n"
