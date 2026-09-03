"""Vascular-track gate machinery (v0.4 arterial-stiffness track).

The `vascular:` block of configs/gates.yaml is the pre-registered
contract; this module only APPLIES it. Pure stdlib + configs so it is
importable from anywhere (unlike the quarantined research/ package) and
must fail CLOSED on any error: no gates file, a broken gates file, no
evidence, a partial run — all of it reads as RED and render-forbidden.

Evidence arrives from TWO run kinds on the same scoreboard: a fidelity
study ("fidelity", gate V0 — research/vascular/fidelity.py) and a model
evaluation ("evaluation", gates V1-V5 — evaluation/vascular_metrics.py).
Gate status merges the latest entry of each kind; each kind carries its
own qualification data, so a facial-domain evaluation cannot borrow a
synthetic fidelity study's standing or vice versa.
"""
from __future__ import annotations

import json
import os
import pathlib

from configs import parse_yaml_subset
from evaluation import _gate_common

_REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_GATES = _REPO / "configs" / "gates.yaml"
DEFAULT_RUNS = pathlib.Path(
    os.environ.get("AVATARX_VASCULAR_RUNS",
                   _REPO / "evaluation" / "vascular_runs"))

TRACK_NOTE = ("VASCULAR STIFFNESS INFERENCE — GATED — "
              "NOT VALIDATED, NOT A MEASUREMENT")

GATE_TITLES = {
    "v0": "V0 — signal fidelity (facial vs contact reference)",
    "v1": "V1 — incremental validity vs age+sex+BP (PIVOTAL, T2 defense)",
    "v2": "V2 — repeatability of the estimate",
    "v3": "V3 — generalization (leave-one-site/device-out)",
    "v4": "V4 — fairness (coverage and error parity)",
    "v5": "V5 — claim mapping (owner + clinical advisor)",
}


def load_vascular_gates(path=None) -> dict:
    doc = parse_yaml_subset(
        pathlib.Path(path or DEFAULT_GATES).read_text())
    blk = doc.get("vascular")
    if not isinstance(blk, dict):
        raise ValueError("configs/gates.yaml has no vascular block")
    return blk


# shared mechanics (v0.5 refactor): contracts stay here, machinery is
# evaluation/_gate_common — a third track reuses instead of forking
_unset = _gate_common.unset
_req = _gate_common.make_req("vascular")
_qualification = _gate_common.qualification


def surviving_features(*, runs_root=None, gates_path=None) -> list:
    """The V0-surviving feature set, decided by config (the explicit
    fidelity thresholds in gates.yaml) applied to the latest recorded
    fidelity study — never by hand-editing a list. No study on record,
    a DISQUALIFIED study (surrogate domain, non-disjoint, underpowered
    cohort), or any error, means NO feature survives (fail closed) and
    no downstream model may train or predict."""
    try:
        gcfg = load_vascular_gates(gates_path)
        t0 = gcfg.get("v0_signal_fidelity") or {}
        icc_min = float(_req(t0, "feature_icc_min"))
        rt_min = float(_req(t0, "retest_icc_min"))
        n_min = int(_req(t0, "min_paired_participants"))
        entry = latest_scoreboard_entry("fidelity", runs_root=runs_root)
        ev = (entry or {}).get("evidence") or {}
        # review findings: a synthetic/underpowered fidelity study must
        # not silently decide the feature set for facial evaluations
        if _qualification(gcfg, ev.get("data")):
            return []
        v0 = ev.get("v0") or {}
        if int(v0.get("n_paired_participants") or 0) < n_min:
            return []
        per = v0.get("per_feature") or {}
        out = []
        for name in sorted(per):
            row = per[name] or {}
            icc = row.get("icc")
            rt = row.get("retest_icc")
            if icc is None or float(icc) < icc_min:
                continue
            # retest evidence absent counts as failed (fail closed)
            if rt is None or float(rt) < rt_min:
                continue
            out.append(name)
        return out
    except Exception:
        return []


def evaluate_vascular_gates(gcfg: dict, evidence: dict) -> dict:
    """Apply the pre-registered gates to recorded evidence. Missing
    evidence for a gate is a RED gate with its own reason, never a
    skipped row."""
    ev = evidence or {}
    disq_eval = _qualification(gcfg, ev.get("data"))
    disq_fid = _qualification(gcfg, ev.get("fidelity_data"))
    rows: list = []

    def row(key, numeric_pass, reasons, metrics, thresholds, disq):
        reasons = list(disq) + reasons
        status = "GREEN" if numeric_pass and not reasons else "RED"
        rows.append({"gate": key, "title": GATE_TITLES[key],
                     "status": status, "reasons": reasons,
                     "metrics": metrics, "thresholds": thresholds})

    # V0 — signal fidelity ------------------------------------------------
    t0 = gcfg.get("v0_signal_fidelity") or {}
    v0 = ev.get("v0") or {}
    reasons = []
    icc_min = float(_req(t0, "feature_icc_min"))
    rt_min = float(_req(t0, "retest_icc_min"))
    n_min = int(_req(t0, "min_paired_participants"))
    s_min = int(_req(t0, "min_surviving_features"))
    if not v0:
        reasons.append("no signal-fidelity study on record")
    else:
        n = int(v0.get("n_paired_participants") or 0)
        if n < n_min:
            reasons.append(f"{n} paired participants < {n_min}")
        per = v0.get("per_feature") or {}
        surv = [f for f, r in per.items()
                if (r or {}).get("icc") is not None
                and float(r["icc"]) >= icc_min
                and (r or {}).get("retest_icc") is not None
                and float(r["retest_icc"]) >= rt_min]
        if len(surv) < s_min:
            reasons.append(
                f"only {len(surv)} feature(s) met ICC >= {icc_min} and "
                f"retest ICC >= {rt_min} (< {s_min}) — no morphology "
                "signal survives the camera at fidelity grade")
    row("v0", not reasons, reasons, v0, t0, disq_fid)

    # V1 — incremental validity vs B3 (PIVOTAL) ---------------------------
    t1 = gcfg.get("v1_incremental_validity") or {}
    v1 = ev.get("v1") or {}
    reasons = []
    if not v1:
        reasons.append("no baseline-battery evaluation on record")
    else:
        imp = v1.get("rmse_improvement_vs_b3_mps")
        need = float(_req(t1, "rmse_improvement_vs_b3_mps_min"))
        if imp is None or float(imp) < need:
            reasons.append(f"RMSE improvement over B3 {imp} < {need} m/s "
                           "— indistinguishable from an age+sex+BP "
                           "regression")
        ci = v1.get("rmse_improvement_ci95")
        if bool(_req(t1, "ci_must_exclude_zero")) and \
                (not ci or ci[0] is None or float(ci[0]) <= 0.0):
            reasons.append("improvement CI does not exclude zero")
        ar2 = v1.get("added_r2")
        ar2_min = float(_req(t1, "added_r2_min"))
        if ar2 is None or float(ar2) < ar2_min:
            reasons.append(f"added variance explained {ar2} < {ar2_min}")
        nt = int(v1.get("n_test_participants") or 0)
        nt_min = int(_req(t1, "min_test_participants"))
        if nt < nt_min:
            reasons.append(f"{nt} held-out participants < {nt_min}")
    row("v1", not reasons, reasons, v1, t1, disq_eval)

    # V2 — repeatability of the estimate ----------------------------------
    t2 = gcfg.get("v2_estimate_repeatability") or {}
    v2 = ev.get("v2") or {}
    reasons = []
    if not v2:
        reasons.append("no test-retest study of the estimate on record")
    else:
        icc = v2.get("estimate_retest_icc")
        need = float(_req(t2, "estimate_retest_icc_min"))
        if icc is None or float(icc) < need:
            reasons.append(f"estimate retest ICC {icc} < {need}")
        drift = v2.get("retest_drift_mps")
        dmax = float(_req(t2, "retest_drift_mps_max"))
        if drift is None or abs(float(drift)) > dmax:
            reasons.append(f"retest drift {drift} outside +/-{dmax} m/s")
        npair = int(v2.get("n_retest_pairs") or 0)
        pmin = int(_req(t2, "min_retest_pairs"))
        if npair < pmin:
            reasons.append(f"{npair} retest pairs < {pmin}")
    row("v2", not reasons, reasons, v2, t2, disq_eval)

    # V3 — generalization (site AND device held out) ----------------------
    t3 = gcfg.get("v3_generalization") or {}
    v3 = ev.get("v3") or {}
    reasons = []
    if not v3:
        reasons.append("no leave-one-site/device-out study on record")
    else:
        rmax = float(_req(t3, "held_out_site_rmse_ratio_max"))
        dmax = float(_req(t3, "held_out_device_rmse_ratio_max"))
        for n_key, min_key, r_key, cap, label in (
                ("n_sites", "min_sites", "worst_site_rmse_ratio",
                 rmax, "site"),
                ("n_devices", "min_devices", "worst_device_rmse_ratio",
                 dmax, "device")):
            n = int(v3.get(n_key) or 0)
            nmin = int(_req(t3, min_key))
            if n < nmin:
                reasons.append(f"{n} {label}s < {nmin}")
            ratio = v3.get(r_key)
            if ratio is None or float(ratio) > cap:
                reasons.append(f"worst held-out-{label} RMSE ratio "
                               f"{ratio} > {cap}")
    row("v3", not reasons, reasons, v3, t3, disq_eval)

    # V4 — fairness: EVERY declared subgroup axis must have tables ------
    t4 = gcfg.get("v4_fairness") or {}
    v4 = ev.get("v4") or {}
    reasons = []
    axes = list(_req(t4, "subgroups"))
    cmin = float(_req(t4, "coverage_ratio_min"))
    emax = float(_req(t4, "rmse_ratio_max"))
    per_axis = v4.get("per_axis") or {}
    if not v4:
        reasons.append("no subgroup coverage/error-parity tables on "
                       "record")
    else:
        for axis in axes:
            a = per_axis.get(axis)
            if not a:
                reasons.append(f"no {axis} subgroup table on record — "
                               "declared axes cannot be silently "
                               "skipped")
                continue
            cov = a.get("coverage_ratio_worst")
            if cov is None or float(cov) < cmin:
                reasons.append(f"{axis}: worst-group coverage ratio "
                               f"{cov} < {cmin}")
            er = a.get("rmse_ratio_worst")
            if er is None or float(er) > emax:
                reasons.append(f"{axis}: worst-group RMSE ratio {er} > "
                               f"{emax}")
    row("v4", not reasons, reasons, v4, t4, disq_eval)

    # V5 — claim mapping: the signed signoff block IS the record ----------
    # (review finding: an evidence field no sanctioned tool could produce
    # made V5 permanently unclearable; the claim map is a human contract
    # fact, so it lives in gates.yaml signoff, nowhere else)
    t5 = gcfg.get("v5_claim_mapping") or {}
    v5 = ev.get("v5") or {}
    reasons = []
    permissible = str(_req(t5, "permissible_first_claim"))
    sign = gcfg.get("signoff") or {}
    signed = sign.get("owner_confirmed") is True and \
        not _unset(sign.get("clinical_advisor"))
    if not signed:
        reasons.append("claim mapping with owner + clinical advisor not "
                       "on record — the first permissible surface is a "
                       "longitudinal wellness comparison against the "
                       "user's own baseline, nothing absolute")
    if _unset(sign.get("claim_scope")) or \
            str(sign.get("claim_scope")) != permissible:
        reasons.append(f"signoff claim_scope is not "
                       f"{permissible!r} — nothing may render")
    row("v5", not reasons, reasons, v5, t5, disq_eval)

    all_green = all(r["status"] == "GREEN" for r in rows)
    return {"gates": rows, "all_gates_green": all_green,
            "clinical_signoff": signed,
            "promotion_open": bool(all_green and signed)}


# ------------------------------------------------------------ scoreboard
def append_scoreboard(entry: dict, runs_root=None) -> pathlib.Path:
    return _gate_common.append_scoreboard(entry, runs_root, DEFAULT_RUNS)


def latest_scoreboard_entry(kind=None, runs_root=None):
    return _gate_common.latest_scoreboard_entry(kind, runs_root,
                                                DEFAULT_RUNS)


def vascular_gate_status(*, runs_root=None, gates_path=None) -> dict:
    gcfg = load_vascular_gates(gates_path)
    fid = latest_scoreboard_entry("fidelity", runs_root=runs_root)
    ev_run = latest_scoreboard_entry("evaluation", runs_root=runs_root)
    evidence: dict = {}
    if fid:
        evidence["v0"] = (fid.get("evidence") or {}).get("v0") or {}
        evidence["fidelity_data"] = (fid.get("evidence") or {}).get("data")
    if ev_run:
        e = ev_run.get("evidence") or {}
        for k in ("v1", "v2", "v3", "v4", "v5"):
            if e.get(k) is not None:
                evidence[k] = e[k]
        evidence["data"] = e.get("data")
    verdict = evaluate_vascular_gates(gcfg, evidence)
    return {"track": "vascular", "note": TRACK_NOTE,
            "gates_version": gcfg.get("gates_version"),
            "requires_clinical_signoff":
                gcfg.get("requires_clinical_signoff"),
            "signoff": gcfg.get("signoff"),
            "fidelity_run": (fid or {}).get("run_id"),
            "evaluation_run": (ev_run or {}).get("run_id"),
            "surviving_features": surviving_features(
                runs_root=runs_root, gates_path=gates_path),
            **verdict,
            "promotion": ("OPEN" if verdict["promotion_open"]
                          else "BLOCKED")}


def vascular_render_allowed(*, runs_root=None, gates_path=None) -> bool:
    """The V-a rendering invariant, fail-CLOSED: any error means no."""
    try:
        return bool(vascular_gate_status(
            runs_root=runs_root, gates_path=gates_path)["promotion_open"])
    except Exception:
        return False


def render_vascular_status_html(doc: dict) -> str:
    banner = ("<div style='background:#7c2d12;color:#fed7aa;padding:8px "
              "12px;border-radius:8px;margin:10px 0;font-weight:600'>"
              f"{TRACK_NOTE}</div>")
    rows = []
    for g in doc.get("gates", []):
        color = "#166534" if g["status"] == "GREEN" else "#92400e"
        items = "".join(f"<li>{r}</li>" for r in g["reasons"]) or \
            "<li>pass</li>"
        rows.append(
            f"<h3>{g['title']}</h3>"
            f"<p style='color:{color};font-weight:700'>{g['status']}</p>"
            f"<ul>{items}</ul>"
            f"<pre>{json.dumps(g.get('metrics') or {}, indent=1)}</pre>")
    sign = "yes" if doc.get("clinical_signoff") else "PENDING"
    return ("<html><body style='font-family:system-ui;max-width:760px;"
            "margin:2em auto'>" + banner +
            "<h2>Arterial-stiffness track promotion scoreboard</h2>"
            f"<p>gates {doc.get('gates_version')} · fidelity run: "
            f"{doc.get('fidelity_run') or 'none'} · evaluation run: "
            f"{doc.get('evaluation_run') or 'none'} · promotion: "
            f"<b>{doc.get('promotion')}</b> · clinical signoff: {sign}"
            "</p>" + "".join(rows) + banner + "</body></html>")
