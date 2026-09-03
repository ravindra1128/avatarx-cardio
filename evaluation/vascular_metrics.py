"""Vascular baseline battery + evaluation harness (v0.4 vascular, T3/T4).

The T2 defense lives here: every evaluation trains and reports the
MANDATORY baselines on identical participant-disjoint splits —
  B1 age · B2 age+sex · B3 age+sex+brachial-BP (the clinic bar) ·
  B4 HR-only · B5 age+sex+HR
— and the head (V0-surviving MORPHOLOGY features only; demographics may
never enter it) is judged against B3 with cluster-bootstrap CIs.
Invariant V-d is structural: `battery_report` refuses to build a report
without every baseline and the B3 deltas, so a result quoted without its
baseline context cannot exist.

Added variance explained is a held-out STATISTIC (squared partial
correlation of head predictions with cfPWV given B3 predictions) — no
model containing demographics+morphology together is ever trained, so
the no-demographics-inside-any-model invariant holds.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import numpy as np

from evaluation.fitness_metrics import _ridge, _split, bland_altman

SYNTHETIC_NOTICE = ("SYNTHETIC / SURROGATE DATA — machinery evidence "
                    "only; never quote as performance")

BASELINES = ("b1_age", "b2_age_sex", "b3_age_sex_bp", "b4_hr",
             "b5_age_sex_hr")


class VascularHarnessError(ValueError):
    pass


def _sex01(row) -> float:
    return {"female": 0.0, "male": 1.0}.get(row.get("sex"), 0.5)


def _baseline_vec(name: str, r: dict) -> list:
    if name == "b1_age":
        return [r.get("age")]
    if name == "b2_age_sex":
        return [r.get("age"), _sex01(r)]
    if name == "b3_age_sex_bp":
        return [r.get("age"), _sex01(r), r.get("sbp"), r.get("dbp")]
    if name == "b4_hr":
        return [r.get("hr")]
    if name == "b5_age_sex_hr":
        return [r.get("age"), _sex01(r), r.get("hr")]
    raise VascularHarnessError(f"unknown baseline {name!r}")


def _matrix(rows: list, fn) -> np.ndarray:
    out = []
    for r in rows:
        out.append([np.nan if v is None else float(v) for v in fn(r)])
    return np.asarray(out, float)


def _head_vec(r: dict, surviving: list) -> list:
    feats = r.get("features") or {}
    return [feats.get(name) for name in surviving]


def _rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def _r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = float(np.sum((y - y.mean()) ** 2))
    if ss < 1e-12:
        return 0.0
    return float(1.0 - np.sum((yhat - y) ** 2) / ss)


def _partial_r(y, head_pred, b3_pred) -> float:
    """SIGNED partial correlation of the head's predictions with the
    label, controlling for B3's predictions — a held-out statistic, not
    a trained combined model (T2 invariant). The sign matters: a
    bootstrap CI over the SQUARE is non-negative by construction and
    its exclude-zero test does no statistical work (review finding)."""
    y = np.asarray(y, float)
    h = np.asarray(head_pred, float)
    b = np.asarray(b3_pred, float)

    def _resid(v):
        A = np.stack([np.ones_like(b), b], 1)
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef

    ry, rh = _resid(y), _resid(h)
    denom = float(np.std(ry) * np.std(rh))
    if denom < 1e-12:
        return 0.0
    return float(np.mean((ry - ry.mean()) * (rh - rh.mean())) / denom)


def _partial_r2(y, head_pred, b3_pred) -> float:
    """Squared partial correlation (the added-variance point
    estimate)."""
    return float(_partial_r(y, head_pred, b3_pred) ** 2)


def fit_battery(records: list, surviving: list, *, seed: int = 20260831,
                n_boot: int = 200) -> dict:
    """Head + B1..B5 on ONE participant-disjoint split; deltas vs B3
    with multiplicity-correct participant-cluster bootstrap CIs."""
    rows = [r for r in records if r.get("cfpwv") is not None]
    if not surviving:
        return {"available": False,
                "reason": "no V0-surviving features — the fidelity gate "
                          "blocks all modeling (fail closed)"}
    tr, te = _split(rows, seed)
    if len(tr) < 4 or len(te) < 2:
        return {"available": False,
                "reason": f"split too small (train {len(tr)}, "
                          f"test {len(te)})"}
    ytr = np.asarray([r["cfpwv"] for r in tr], float)
    yte = np.asarray([r["cfpwv"] for r in te], float)
    preds: dict = {}
    models: dict = {}
    for name in BASELINES:
        Xtr = _matrix(tr, lambda r, n=name: _baseline_vec(n, r))
        Xte = _matrix(te, lambda r, n=name: _baseline_vec(n, r))
        preds[name] = _ridge(Xtr, ytr, Xte)
    Htr = _matrix(tr, lambda r: _head_vec(r, surviving))
    Hte = _matrix(te, lambda r: _head_vec(r, surviving))
    preds["head"] = _ridge(Htr, ytr, Hte)
    models["head"] = _train_head_artifact(Htr, ytr, surviving)

    table = {}
    for name, p in preds.items():
        table[name] = {"rmse": round(_rmse(yte, p), 3),
                       "mae": round(float(np.mean(np.abs(p - yte))), 3),
                       "r2": round(_r2(yte, p), 3),
                       **bland_altman(yte, p)}
    imp = table["b3_age_sex_bp"]["rmse"] - table["head"]["rmse"]
    added = _partial_r2(yte, preds["head"], preds["b3_age_sex_bp"])

    by_pid: dict = {}
    for i, r in enumerate(te):
        by_pid.setdefault(str(r["participant_id"]), []).append(i)
    pids = sorted(by_pid)

    def _boot(stat):
        if len(pids) < 4:
            return None
        rng = np.random.default_rng(seed + 1)
        vals = []
        for _ in range(n_boot):
            draw = rng.choice(pids, size=len(pids), replace=True)
            idx = [i for p in draw for i in by_pid[p]]
            if len(idx) < 2:
                continue
            v = stat(np.asarray(idx, int))
            if v is not None and np.isfinite(v):
                vals.append(v)
        if len(vals) < 10:
            return None
        return [round(float(np.percentile(vals, 2.5)), 3),
                round(float(np.percentile(vals, 97.5)), 3)]

    imp_ci = _boot(lambda idx: _rmse(yte[idx],
                                     preds["b3_age_sex_bp"][idx])
                   - _rmse(yte[idx], preds["head"][idx]))
    # CI over the SIGNED partial correlation (review finding: the
    # squared statistic's CI can never straddle zero)
    added_ci = _boot(lambda idx: _partial_r(
        yte[idx], preds["head"][idx], preds["b3_age_sex_bp"][idx]))
    return {"available": True, "n_train": len(tr), "n_test": len(te),
            "n_test_participants": len(pids),
            "surviving_features": list(surviving),
            "models": table,
            "rmse_improvement_vs_b3_mps": round(float(imp), 3),
            "rmse_improvement_ci95": imp_ci,
            "added_r2": round(float(added), 4),
            "added_r_ci95": added_ci,
            "head_artifact": models["head"],
            "_test_pred": {"head": preds["head"].tolist(),
                           "b3": preds["b3_age_sex_bp"].tolist()},
            "_test_rows": te}


def _train_head_artifact(Htr, ytr, surviving, kind: str = None) -> dict:
    """The deployable simple model on session morphology features:
    dispatched through _HEAD_TRAINERS keyed by configs/default.yaml
    `vascular.head_model` (config-swappable; an unknown kind is a loud
    refusal listing the registry). NEVER contains demographics."""
    if kind is None:
        from configs import load_config
        kind = str(((load_config().get("vascular") or {})
                    .get("head_model", "ridge")))
    trainer = _HEAD_TRAINERS.get(kind)
    if trainer is None:
        raise VascularHarnessError(
            f"unknown vascular head_model {kind!r} — supported: "
            f"{sorted(_HEAD_TRAINERS)} (configs/default.yaml "
            "vascular.head_model)")
    return trainer(Htr, ytr, surviving)


def _train_ridge(Htr, ytr, surviving) -> dict:
    """Regularized linear on session medians: coefficients + residual
    spread for the (naive, pre-validation) CI the head reports."""
    mu = np.nanmean(Htr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    X = np.where(np.isfinite(Htr), Htr, mu)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X - mu) / sd
    ybar = float(np.mean(ytr))
    coef = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(Z.shape[1]),
                           Z.T @ (ytr - ybar))
    resid = ytr - (ybar + Z @ coef)
    return {"kind": "ridge_morphology", "features": list(surviving),
            "mu": [round(float(v), 6) for v in mu],
            "sd": [round(float(v), 6) for v in sd],
            "coef": [round(float(v), 6) for v in coef],
            "intercept": round(ybar, 4),
            "residual_sd": round(float(np.std(resid, ddof=1))
                                 if resid.size > 1 else 0.0, 4),
            "n_train": int(len(ytr))}


# the config-swappable trainer registry: extension = a new entry here
# plus an apply path in heads/head_vascular.py keyed on artifact "kind"
_HEAD_TRAINERS = {"ridge": _train_ridge}


def battery_report(battery: dict) -> dict:
    """V-d, structurally: a reportable battery MUST carry every baseline
    and the B3 deltas — a result without its baseline context cannot be
    built, only refused."""
    if not battery.get("available"):
        return {"available": False,
                "reason": battery.get("reason", "unavailable")}
    models = battery.get("models") or {}
    missing = [b for b in BASELINES if b not in models]
    if missing or "head" not in models:
        raise VascularHarnessError(
            f"V-d violation: battery is missing {missing or ['head']} — "
            "a vascular result may not be reported without its full "
            "baseline battery")
    for key in ("rmse_improvement_vs_b3_mps", "added_r2"):
        if key not in battery:
            raise VascularHarnessError(
                f"V-d violation: battery lacks {key} — a result quoted "
                "without the B3 delta is a bug")
    return {k: v for k, v in battery.items()
            if not k.startswith("_") and k != "head_artifact"}


# ------------------------------------------------- auxiliary evidence
def estimate_retest(test_rows: list, preds: list) -> dict:
    """Same-visit scan pairs among held-out rows -> retest ICC + drift
    of the ESTIMATE (V2 evidence). Pairs are ordered by ACQUISITION
    TIME (manifest video_start_utc), never manifest sort order — a
    scrambled order attenuates |drift| toward zero, which is
    anti-fail-closed against the abs-drift gate (review finding).
    Visits without distinct timestamps are dropped, not guessed."""
    from research.vascular.fidelity import icc_2_1
    by_visit: dict = {}
    for r, p in zip(test_rows, preds):
        by_visit.setdefault((r["participant_id"], r["session_id"]),
                            []).append((r.get("video_start_utc"),
                                        float(p)))
    pairs = []
    for scans in by_visit.values():
        if len(scans) < 2:
            continue
        times = [t for t, _ in scans]
        if any(not isinstance(t, str) or not t.strip() for t in times) \
                or len(set(times)) < len(times):
            continue
        ordered = [v for _, v in sorted(scans, key=lambda tv: tv[0])]
        pairs.append(ordered[:2])
    if not pairs:
        return {}
    m = np.asarray(pairs, float)
    return {"estimate_retest_icc": icc_2_1(m),
            "retest_drift_mps": round(float(np.mean(m[:, 1] - m[:, 0])),
                                      3),
            "n_retest_pairs": int(m.shape[0])}


def _leave_one_out(rows: list, surviving: list, axis: str,
                   base: float) -> dict:
    """Leave-one-<axis>-out folds, PARTICIPANT-disjoint: a participant
    with recordings at several sites/devices must not appear on both
    sides of their own fold (review finding)."""
    groups = sorted({r.get(axis) for r in rows if r.get(axis)})
    per = {}
    worst = None
    for g in groups:
        te = [r for r in rows if r.get(axis) == g]
        te_pids = {str(r["participant_id"]) for r in te}
        tr = [r for r in rows if r.get(axis) != g
              and str(r["participant_id"]) not in te_pids]
        if len(tr) < 4 or len(te) < 2:
            per[str(g)] = {"n": len(te), "note": "too small to score"}
            continue
        Htr = _matrix(tr, lambda r: _head_vec(r, surviving))
        Hte = _matrix(te, lambda r: _head_vec(r, surviving))
        p = _ridge(Htr, np.asarray([r["cfpwv"] for r in tr], float), Hte)
        rmse = _rmse([r["cfpwv"] for r in te], p)
        per[str(g)] = {"n": len(te), "rmse": round(rmse, 3)}
        ratio = rmse / max(base, 1e-9)
        worst = ratio if worst is None else max(worst, ratio)
    out = {"n_" + ("sites" if axis == "site" else "devices"): len(groups),
           ("per_site" if axis == "site" else "per_device"): per}
    if worst is not None:
        key = ("worst_site_rmse_ratio" if axis == "site"
               else "worst_device_rmse_ratio")
        out[key] = round(float(worst), 3)
    return out


def site_generalization(records: list, surviving: list, *,
                        seed: int = 20260831) -> dict:
    """Leave-one-site-out AND leave-one-device-out (V3 evidence): train
    on the other groups, score the held-out group; worst RMSE relative
    to the pooled battery head RMSE."""
    rows = [r for r in records if r.get("cfpwv") is not None]
    sites = sorted({r.get("site") for r in rows if r.get("site")})
    devices = sorted({r.get("device_label") for r in rows
                      if r.get("device_label")})
    out = {"n_sites": len(sites), "n_devices": len(devices)}
    if not surviving:
        return out
    pooled = fit_battery(rows, surviving, seed=seed)
    if not pooled.get("available"):
        return out
    base = pooled["models"]["head"]["rmse"]
    if len(sites) >= 2:
        out.update(_leave_one_out(rows, surviving, "site", base))
    if len(devices) >= 2:
        out.update(_leave_one_out(rows, surviving, "device_label", base))
    return out


def _axis_fairness(all_scans: list, test_rows: list, test_pred: list,
                   scan_key: str, row_key: str) -> dict:
    cov: dict = {}
    for s in all_scans:
        g = s.get(scan_key)
        if g is None:
            continue
        c = cov.setdefault(str(g), {"attempted": 0, "usable": 0})
        c["attempted"] += 1
        c["usable"] += int(bool(s.get("usable")))
    err: dict = {}
    for r, p in zip(test_rows, test_pred):
        g = r.get(row_key)
        if g is None:
            continue
        err.setdefault(str(g), []).append(abs(float(p) - r["cfpwv"]))
    out: dict = {"per_group": {}}
    ratios, rmses = [], {}
    for g, c in sorted(cov.items()):
        rate = c["usable"] / c["attempted"] if c["attempted"] else None
        cell = {"coverage": (round(rate, 3)
                             if rate is not None else None), **c}
        if g in err and len(err[g]) >= 2:
            rmses[g] = float(np.sqrt(np.mean(np.square(err[g]))))
            cell["rmse"] = round(rmses[g], 3)
            cell["n_test"] = len(err[g])
        out["per_group"][g] = cell
        if rate is not None:
            ratios.append(rate)
    if len(ratios) >= 2 and max(ratios) > 0:
        out["coverage_ratio_worst"] = round(min(ratios) / max(ratios), 3)
    if len(rmses) >= 2 and min(rmses.values()) > 0:
        out["rmse_ratio_worst"] = round(
            max(rmses.values()) / min(rmses.values()), 3)
    return out


def fairness_tables(all_scans: list, test_rows: list,
                    test_pred: list) -> dict:
    """Coverage (features obtained / scans attempted) and error parity
    for EVERY subgroup axis the V4 gate declares — fitzpatrick_group and
    device (review finding: a declared axis with no table was silently
    green-adjacent; the evaluator now names any missing axis). The
    `all_scans` list includes excluded scans so coverage is honest about
    who the pipeline drops."""
    return {"per_axis": {
        "fitzpatrick_group": _axis_fairness(
            all_scans, test_rows, test_pred,
            "fitzpatrick_group", "fitzpatrick_group"),
        "device": _axis_fairness(
            all_scans, test_rows, test_pred,
            "device_label", "device_label"),
    }}


# ------------------------------------------------------------ harness
def evaluate_vascular_dataset(dataset_dir, out_dir=None, *,
                              signal_domain: str = "synthetic",
                              runs_root=None, gates_path=None,
                              config=None, surviving_override=None
                              ) -> dict:
    """End-to-end: production-path features + cfPWV references ->
    baseline battery, retest, sites, fairness -> vascular-gate verdict,
    run dir + scoreboard, report. `surviving_override` exists for the
    machinery's own tests; the real path takes the V0-surviving set from
    the recorded fidelity study (config-driven, fail-closed)."""
    from configs import load_config, config_hash
    from datasets.io import load_recording
    from datasets.schema import vascular_from_dict
    from evaluation.vascular_gates import (append_scoreboard,
                                           evaluate_vascular_gates,
                                           latest_scoreboard_entry,
                                           load_vascular_gates,
                                           surviving_features,
                                           DEFAULT_RUNS)
    from research.vascular import WATERMARK
    from research.vascular.features import facial_session_features

    d = pathlib.Path(dataset_dir)
    manifests = sorted(d.glob("*.recording.json"))
    if not manifests:
        raise VascularHarnessError(
            f"{d} contains no *.recording.json manifests")
    rows, excluded, all_scans = [], [], []
    for mp in manifests:
        rid = mp.name[:-len(".recording.json")]
        try:
            rec = load_recording(str(mp))
            mdict = json.loads(mp.read_text())
        except (ValueError, KeyError) as e:
            excluded.append({"recording_id": rid,
                             "reasons": [f"invalid manifest: {e}"]})
            continue
        pwv_p = d / f"{rid}.pwv.json"
        if not pwv_p.exists():
            excluded.append({"recording_id": rid,
                             "reasons": ["no cfPWV reference "
                                         "(<id>.pwv.json) — unlabeled "
                                         "for stiffness"]})
            continue
        try:
            ref = vascular_from_dict(json.loads(pwv_p.read_text()))
        except ValueError as e:
            excluded.append({"recording_id": rid,
                             "reasons": [f"invalid cfPWV reference: "
                                         f"{e}"]})
            continue
        scan = {"recording_id": rid,
                "fitzpatrick_group": ref.fitzpatrick_group,
                "device_label": ref.device_label,
                "usable": False}
        facial = facial_session_features(str(d / rec.video_path),
                                         manifest=mdict, config=config)
        if not facial.get("available"):
            all_scans.append(scan)
            excluded.append({"recording_id": rid,
                             "reasons": list(facial.get("reasons")
                                             or [])})
            continue
        scan["usable"] = True
        all_scans.append(scan)
        rows.append({"recording_id": rid,
                     "participant_id": rec.participant_id,
                     "session_id": rec.session_id,
                     "video_start_utc": rec.video_start_utc,
                     "site": (ref.site_id or rec.site_id),
                     "device_label": ref.device_label,
                     "fitzpatrick_group": ref.fitzpatrick_group,
                     "age": ref.age_years, "sex": ref.sex,
                     "sbp": ref.brachial_sbp_mmhg,
                     "dbp": ref.brachial_dbp_mmhg,
                     "hr": ref.hr_at_measurement_bpm,
                     "features": facial["features"],
                     "cfpwv": ref.cfpwv_mps})

    if not rows:
        raise VascularHarnessError(
            "no usable labeled recordings — every recording was "
            "excluded: "
            + "; ".join(f"{e['recording_id']}: {e['reasons'][0]}"
                        for e in excluded[:5]))
    surviving = (list(surviving_override)
                 if surviving_override is not None
                 else surviving_features(runs_root=runs_root,
                                         gates_path=gates_path))
    battery = fit_battery(rows, surviving)
    report_battery = battery_report(battery)
    v1 = {}
    v2: dict = {}
    if battery.get("available"):
        v1 = {"rmse_improvement_vs_b3_mps":
              battery["rmse_improvement_vs_b3_mps"],
              "rmse_improvement_ci95":
              battery["rmse_improvement_ci95"],
              "added_r2": battery["added_r2"],
              "added_r_ci95": battery["added_r_ci95"],
              "n_test_participants": battery["n_test_participants"]}
        v2 = estimate_retest(battery["_test_rows"],
                             battery["_test_pred"]["head"])
    v3 = site_generalization(rows, surviving)
    v4 = fairness_tables(all_scans,
                         battery.get("_test_rows") or [],
                         (battery.get("_test_pred") or {}).get("head")
                         or [])
    gcfg = load_vascular_gates(gates_path)
    fid_entry = latest_scoreboard_entry("fidelity", runs_root=runs_root)
    evidence = {"data": {"signal_domain": signal_domain,
                         "participant_disjoint": True,
                         "session_disjoint": True,
                         "production_path":
                             signal_domain == "facial_rppg",
                         "n_recordings": len(rows),
                         "dataset_dir": str(d)},
                "v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": {}}
    if fid_entry:
        evidence["v0"] = (fid_entry.get("evidence") or {}).get("v0")
        evidence["fidelity_data"] = \
            (fid_entry.get("evidence") or {}).get("data")
    verdict = evaluate_vascular_gates(gcfg, evidence)

    run_id = "vasc-" + hashlib.sha256(
        (str(d) + signal_domain
         + ",".join(sorted(r["recording_id"] for r in rows))
         ).encode()).hexdigest()[:12]
    run_dir = pathlib.Path(runs_root or DEFAULT_RUNS) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "vascular_record.json").write_text(json.dumps(
        {"WATERMARK": WATERMARK, "run_id": run_id,
         "track": "vascular_stiffness", "kind": "evaluation",
         "n_recordings": len(rows), "n_excluded": len(excluded),
         "signal_domain": signal_domain,
         "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    (run_dir / "gate_results.json").write_text(json.dumps(
        {"WATERMARK": WATERMARK, "run_id": run_id,
         "evidence": evidence, **verdict}, indent=1, default=str))
    (run_dir / "model.json").write_text(json.dumps(
        {"WATERMARK": WATERMARK,
         **(battery.get("head_artifact")
            or {"kind": "none", "reason":
                battery.get("reason", "battery unavailable")})},
        indent=1))
    append_scoreboard({"kind": "evaluation", "run_id": run_id,
                       "evaluated_at":
                           time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "evidence": evidence,
                       "all_gates_green": verdict["all_gates_green"],
                       "promotion_open": verdict["promotion_open"]},
                      runs_root=runs_root)

    cfg = load_config() if config is None else config
    doc = {"WATERMARK": WATERMARK, "run_id": run_id,
           "provenance": {"config_hash": config_hash(cfg),
                          "dataset_dir": str(d),
                          "signal_domain": signal_domain,
                          "notice": (SYNTHETIC_NOTICE
                                     if signal_domain != "facial_rppg"
                                     else None)},
           "excluded": excluded,
           "n_included": len(rows),
           "surviving_features": surviving,
           "battery": report_battery,
           "estimate_retest": v2, "site_generalization": v3,
           "fairness": v4,
           "gates": {"verdict": verdict,
                     "promotion": ("OPEN" if verdict["promotion_open"]
                                   else "BLOCKED")}}
    out = pathlib.Path(out_dir or (d / "report_vascular"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_report.json").write_text(
        json.dumps(doc, indent=2, default=str))
    (out / "evaluation_report.md").write_text(_render_markdown(doc))
    return {"report_json": str(out / "evaluation_report.json"),
            "report_md": str(out / "evaluation_report.md"),
            "run_dir": str(run_dir), "report": doc}


def _render_markdown(doc: dict) -> str:
    from research.vascular import WATERMARK
    lines = [f"> {WATERMARK}", "",
             f"# Vascular evaluation — run {doc['run_id']}", ""]
    if doc["provenance"].get("notice"):
        lines += [f"**{doc['provenance']['notice']}**", ""]
    b = doc.get("battery") or {}
    if b.get("available"):
        lines += ["| model | RMSE | MAE | R2 | bias |", "|---|---|---|---|---|"]
        for name in list(BASELINES) + ["head"]:
            m = b["models"][name]
            lines.append(f"| {name} | {m['rmse']} | {m['mae']} | "
                         f"{m['r2']} | {m.get('bias')} |")
        lines += ["",
                  f"delta vs B3 (age+sex+BP): RMSE improvement "
                  f"{b['rmse_improvement_vs_b3_mps']} m/s "
                  f"(CI95 {b['rmse_improvement_ci95']}), added R2 "
                  f"{b['added_r2']} (signed-r CI95 "
                  f"{b['added_r_ci95']})", ""]
    else:
        lines += [f"battery unavailable: {b.get('reason')}", ""]
    lines += [f"surviving features: "
              f"{', '.join(doc['surviving_features']) or 'NONE'}", "",
              f"promotion: **{doc['gates']['promotion']}**", "",
              f"> {WATERMARK}"]
    return "\n".join(lines)
