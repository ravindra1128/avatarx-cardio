"""§W gate machinery (v0.5 vasotone track).

The `vasotone:` block of configs/gates.yaml is the pre-registered
contract; this module only APPLIES it, on the SHARED gate mechanics
(evaluation/_gate_common — reused, not forked, per the v0.5 build
directive). Pure stdlib + configs; fails CLOSED on any error.

One run kind ("evaluation", evaluation/vasotone_metrics.py) carries all
§W evidence: the provocation study (W0/W2/W3/W4), the two null arms
(W1 + the surviving-feature set) and the qualification data. The claim
map (W5) lives in the signed signoff block, nowhere else.
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
    os.environ.get("AVATARX_VASOTONE_RUNS",
                   _REPO / "evaluation" / "vasotone_runs"))

TRACK_NOTE = ("VASOMOTOR REACTIVITY — §W-GATED — "
              "NOT VALIDATED, NOT A MEASUREMENT")

GATE_TITLES = {
    "w0": "W0 — reference tracking (facial vs contact-PI response)",
    "w1": "W1 — optics invariance (null_optics arm, T1 defense)",
    "w2": "W2 — beats HR/respiration baselines (PIVOTAL, T2 defense)",
    "w3": "W3 — dose/consistency (retest + graded ordering)",
    "w4": "W4 — fairness (detection parity and coverage)",
    "w5": "W5 — claim mapping (owner + clinical advisor)",
}

_unset = _gate_common.unset
_req = _gate_common.make_req("vasotone")
_qualification = _gate_common.qualification


def load_vasotone_gates(path=None) -> dict:
    doc = parse_yaml_subset(
        pathlib.Path(path or DEFAULT_GATES).read_text())
    blk = doc.get("vasotone")
    if not isinstance(blk, dict):
        raise ValueError("configs/gates.yaml has no vasotone (§W) block")
    return blk


def append_scoreboard(entry: dict, runs_root=None) -> pathlib.Path:
    return _gate_common.append_scoreboard(entry, runs_root, DEFAULT_RUNS)


def latest_scoreboard_entry(kind=None, runs_root=None):
    return _gate_common.latest_scoreboard_entry(kind, runs_root,
                                                DEFAULT_RUNS)


def survivors_from_study(w1: dict, gcfg: dict) -> list:
    """The ONE survival rule (config thresholds applied to a null-arm
    study): a feature responds to lamps — absolutely, or relative to
    natural drift — or lacks null-arm evidence, and it is dropped.
    Underpowered studies produce NO survivors (fail closed). Used both
    by the in-run harness and the scoreboard-reading path below."""
    t1 = (gcfg or {}).get("w1_optics_invariance") or {}
    p95_max = float(_req(t1, "null_optics_abs_delta_norm_p95_max"))
    ratio_max = float(_req(t1, "null_rest_ratio_max"))
    n_opt_min = int(_req(t1, "min_null_optics_sessions"))
    n_rest_min = int(_req(t1, "min_null_rest_sessions"))
    w1 = w1 or {}
    if int(w1.get("n_null_optics_sessions") or 0) < n_opt_min:
        return []
    if int(w1.get("n_null_rest_sessions") or 0) < n_rest_min:
        return []
    out = []
    for name in sorted(w1.get("per_feature") or {}):
        row = (w1["per_feature"].get(name) or {})
        po = row.get("null_optics_p95")
        pr = row.get("null_rest_p95")
        if po is None or pr is None:
            continue                      # no evidence = no survival
        # per-feature value floors: a p95 resting on a couple of values
        # must not ride on the arm-level session counts (review finding)
        if "n_null_optics_values" in row and \
                int(row.get("n_null_optics_values") or 0) < n_opt_min:
            continue
        if "n_null_rest_values" in row and \
                int(row.get("n_null_rest_values") or 0) < n_rest_min:
            continue
        if float(po) > p95_max:
            continue                      # responds to lamps
        if float(po) > ratio_max * max(float(pr), 1e-9):
            continue                      # responds to lamps vs drift
        out.append(name)
    return out


def surviving_tone_features(*, runs_root=None, gates_path=None) -> list:
    """The W1-surviving feature set from the LATEST recorded null-arm
    study — automatically, by config, never by hand-editing. No study,
    a disqualified/underpowered study, or any error means NOTHING
    survives (fail closed)."""
    try:
        gcfg = load_vasotone_gates(gates_path)
        entry = latest_scoreboard_entry("evaluation",
                                        runs_root=runs_root)
        ev = (entry or {}).get("evidence") or {}
        if _qualification(gcfg, ev.get("data")):
            return []
        return survivors_from_study(ev.get("w1") or {}, gcfg)
    except Exception:
        return []


def evaluate_vasotone_gates(gcfg: dict, evidence: dict) -> dict:
    """Apply §W to recorded evidence. Missing evidence for a gate is a
    RED gate with its own reason, never a skipped row."""
    ev = evidence or {}
    disq = _qualification(gcfg, ev.get("data"))
    rows: list = []

    def row(key, numeric_pass, reasons, metrics, thresholds):
        reasons = list(disq) + reasons
        status = "GREEN" if numeric_pass and not reasons else "RED"
        rows.append({"gate": key, "title": GATE_TITLES[key],
                     "status": status, "reasons": reasons,
                     "metrics": metrics, "thresholds": thresholds})

    # W0 — reference tracking ---------------------------------------------
    t0 = gcfg.get("w0_reference_tracking") or {}
    w0 = ev.get("w0") or {}
    reasons = []
    if not w0:
        reasons.append("no provocation study with a contact reference "
                       "on record")
    else:
        da = w0.get("direction_agreement")
        da_min = float(_req(t0, "direction_agreement_min"))
        if da is None or float(da) < da_min:
            reasons.append(f"direction agreement {da} < {da_min}")
        r = w0.get("magnitude_r")
        r_min = float(_req(t0, "magnitude_r_min"))
        if r is None or float(r) < r_min:
            reasons.append(f"response-magnitude correlation {r} < "
                           f"{r_min}")
        n = int(w0.get("n_provocations") or 0)
        n_min = int(_req(t0, "min_provocations"))
        if n < n_min:
            reasons.append(f"{n} provocations < {n_min}")
    row("w0", not reasons, reasons, w0, t0)

    # W1 — optics invariance (the T1 defense) -----------------------------
    t1 = gcfg.get("w1_optics_invariance") or {}
    w1 = ev.get("w1") or {}
    reasons = []
    p95_max = float(_req(t1, "null_optics_abs_delta_norm_p95_max"))
    ratio_max = float(_req(t1, "null_rest_ratio_max"))
    n_opt_min = int(_req(t1, "min_null_optics_sessions"))
    n_rest_min = int(_req(t1, "min_null_rest_sessions"))
    s_min = int(_req(t1, "min_surviving_features"))
    if not w1:
        reasons.append("no null-arm study on record — without the "
                       "optics placebo this track has no defense "
                       "against lamps")
    else:
        n_opt = int(w1.get("n_null_optics_sessions") or 0)
        n_rest = int(w1.get("n_null_rest_sessions") or 0)
        if n_opt < n_opt_min:
            reasons.append(f"{n_opt} null_optics sessions < {n_opt_min}")
        if n_rest < n_rest_min:
            reasons.append(f"{n_rest} null_rest sessions < {n_rest_min}")
        # the ONE survival rule (review finding: an inline twin of
        # survivors_from_study could drift)
        surv = survivors_from_study(w1, gcfg)
        if len(surv) < s_min:
            reasons.append(
                f"only {len(surv)} feature(s) below the optics-response "
                f"thresholds (p95 <= {p95_max} and <= {ratio_max}x "
                f"natural drift; < {s_min}) — the surviving set is "
                "empty, every candidate is an optics detector")
    row("w1", not reasons, reasons, w1, t1)

    # W2 — beats the HR/respiration baselines (PIVOTAL, T2) ---------------
    t2 = gcfg.get("w2_beats_baselines") or {}
    w2 = ev.get("w2") or {}
    reasons = []
    if not w2:
        reasons.append("no baseline-battery evaluation on record")
    else:
        ar2 = w2.get("added_r2_over_b3")
        ar2_min = float(_req(t2, "added_r2_over_b3_min"))
        if ar2 is None or float(ar2) < ar2_min:
            reasons.append(f"added variance over the HR+respiration "
                           f"baseline {ar2} < {ar2_min} — "
                           "indistinguishable from an HR/breathing "
                           "response detector")
        ci = w2.get("added_r_ci95")
        if bool(_req(t2, "ci_must_exclude_zero")) and \
                (not ci or ci[0] is None or float(ci[0]) <= 0.0):
            reasons.append("added-information CI does not exclude zero")
        nt = int(w2.get("n_test_participants") or 0)
        nt_min = int(_req(t2, "min_test_participants"))
        if nt < nt_min:
            reasons.append(f"{nt} held-out participants < {nt_min}")
    row("w2", not reasons, reasons, w2, t2)

    # W3 — dose/consistency -----------------------------------------------
    t3 = gcfg.get("w3_dose_consistency") or {}
    w3 = ev.get("w3") or {}
    reasons = []
    if not w3:
        reasons.append("no retest/graded-provocation study on record")
    else:
        icc = w3.get("retest_icc")
        icc_min = float(_req(t3, "retest_icc_min"))
        if icc is None or float(icc) < icc_min:
            reasons.append(f"response retest ICC {icc} < {icc_min}")
        npair = int(w3.get("n_retest_pairs") or 0)
        p_min = int(_req(t3, "min_retest_pairs"))
        if npair < p_min:
            reasons.append(f"{npair} retest pairs < {p_min}")
        of = w3.get("ordered_fraction")
        of_min = float(_req(t3, "ordered_fraction_min"))
        if of is None or float(of) < of_min:
            reasons.append(f"graded-provocation ordering {of} < "
                           f"{of_min}")
    row("w3", not reasons, reasons, w3, t3)

    # W4 — fairness --------------------------------------------------------
    t4 = gcfg.get("w4_fairness") or {}
    w4 = ev.get("w4") or {}
    reasons = []
    axes = list(_req(t4, "subgroups"))
    det_min = float(_req(t4, "detection_parity_ratio_min"))
    cov_min = float(_req(t4, "coverage_ratio_min"))
    if not w4:
        reasons.append("no subgroup parity tables on record")
    else:
        per_axis = w4.get("per_axis") or {}
        for axis in axes:
            a = per_axis.get(axis)
            if not a:
                reasons.append(f"no {axis} subgroup table on record — "
                               "declared axes cannot be silently "
                               "skipped")
                continue
            det = a.get("detection_ratio_worst")
            if det is None or float(det) < det_min:
                reasons.append(f"{axis}: worst-group response-detection "
                               f"ratio {det} < {det_min}")
            cov = a.get("coverage_ratio_worst")
            if cov is None or float(cov) < cov_min:
                reasons.append(f"{axis}: worst-group coverage ratio "
                               f"{cov} < {cov_min}")
    row("w4", not reasons, reasons, w4, t4)

    # W5 — claim mapping (the signed signoff IS the record) ---------------
    t5 = gcfg.get("w5_claim_mapping") or {}
    w5 = ev.get("w5") or {}
    reasons = []
    permissible = str(_req(t5, "permissible_first_claim"))
    sign = gcfg.get("signoff") or {}
    # owner_confirmed must be the parsed boolean True — a yaml null
    # spelling or arbitrary string must fail closed (review finding)
    signed = sign.get("owner_confirmed") is True and \
        not _unset(sign.get("clinical_advisor"))
    if not signed:
        reasons.append("claim mapping with owner + clinical advisor "
                       "not on record — the first permissible surface "
                       "is a wellness reactivity comparison against "
                       "the user's own baseline, explicitly "
                       "non-medical")
    if _unset(sign.get("claim_scope")) or \
            str(sign.get("claim_scope")) != permissible:
        reasons.append(f"signoff claim_scope is not {permissible!r} — "
                       "nothing may render")
    row("w5", not reasons, reasons, w5, t5)

    all_green = all(r["status"] == "GREEN" for r in rows)
    return {"gates": rows, "all_gates_green": all_green,
            "clinical_signoff": signed,
            "promotion_open": bool(all_green and signed)}


def vasotone_gate_status(*, runs_root=None, gates_path=None) -> dict:
    gcfg = load_vasotone_gates(gates_path)
    entry = latest_scoreboard_entry("evaluation", runs_root=runs_root)
    evidence = (entry or {}).get("evidence") or {}
    verdict = evaluate_vasotone_gates(gcfg, evidence)
    return {"track": "vasotone", "note": TRACK_NOTE,
            "gates_version": gcfg.get("gates_version"),
            "requires_clinical_signoff":
                gcfg.get("requires_clinical_signoff"),
            "signoff": gcfg.get("signoff"),
            "evaluation_run": (entry or {}).get("run_id"),
            "surviving_features": surviving_tone_features(
                runs_root=runs_root, gates_path=gates_path),
            **verdict,
            "promotion": ("OPEN" if verdict["promotion_open"]
                          else "BLOCKED")}


def vasotone_render_allowed(*, runs_root=None, gates_path=None) -> bool:
    """The W-a rendering invariant, fail-CLOSED: any error means no."""
    try:
        return bool(vasotone_gate_status(
            runs_root=runs_root,
            gates_path=gates_path)["promotion_open"])
    except Exception:
        return False


def render_vasotone_status_html(doc: dict) -> str:
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
            "<h2>Vasomotor-reactivity track promotion scoreboard "
            "(§W)</h2>"
            f"<p>gates {doc.get('gates_version')} · evaluation run: "
            f"{doc.get('evaluation_run') or 'none'} · promotion: "
            f"<b>{doc.get('promotion')}</b> · clinical signoff: {sign}"
            "</p>" + "".join(rows) + banner + "</body></html>")
