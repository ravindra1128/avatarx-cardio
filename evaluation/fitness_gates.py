"""
§V gate machinery (v0.4 T4) — the vo2 block of configs/gates.yaml is the
pre-registered contract; this module only APPLIES it. Pure stdlib on
purpose: the report layer imports `fitness_render_allowed()` to enforce
the INFERRED_FITNESS rendering invariant, so this must be importable
from app/ (unlike the quarantined §G module) and must fail CLOSED on any
error.

Two layers, mirroring §G: numeric verdicts vs evidence qualification
(facial rPPG through the production path, participant- AND
session-disjoint). Surrogate runs exercise the machinery and are
recorded; only qualifying data can turn a gate green, and promotion
additionally requires the vo2 clinical-signoff block.
"""
from __future__ import annotations

import json
import os
import pathlib

from configs import parse_yaml_subset

_REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_GATES = _REPO / "configs" / "gates.yaml"
DEFAULT_RUNS = pathlib.Path(
    os.environ.get("AVATARX_FITNESS_RUNS",
                   _REPO / "evaluation" / "fitness_runs"))

TRACK_NOTE = ("FITNESS INFERENCE — §V-GATED — NOT VALIDATED, "
              "NOT A MEASUREMENT")

GATE_TITLES = {
    "v1": "V1 — recovery-HR fidelity vs reference ECG (per Monk band)",
    "v2": "V2 — protocol repeatability + accepted-rate + no-read parity",
    "v3": "V3 — incremental value beyond demographics+activity (PIVOTAL)",
    "v4": "V4 — subgroup honesty (bias/LoA tables; medication routing)",
    "v5": "V5 — trend validity (head_trend beyond direction-only)",
}


def load_vo2_gates(path=None) -> dict:
    doc = parse_yaml_subset(pathlib.Path(path or DEFAULT_GATES).read_text())
    vo2 = doc.get("vo2")
    if not isinstance(vo2, dict):
        raise ValueError("configs/gates.yaml has no vo2 (§V) block")
    return vo2


def _req(section, key):
    """§V thresholds must be EXPLICIT in gates.yaml — a silently applied
    coded default would let the pre-registered contract drift without a
    signed changelog entry (v0.4 review finding)."""
    if not isinstance(section, dict) or key not in section:
        raise ValueError(f"configs/gates.yaml vo2 block missing required "
                         f"threshold {key!r} — the pre-registered "
                         "contract must be explicit")
    return section[key]


def _qualification(gcfg: dict, data) -> list:
    req = gcfg.get("evidence_requirements") or {}
    if data is None:
        return ["no evaluation run on record"]
    out = []
    dom = data.get("signal_domain")
    if dom != req.get("signal_domain", "facial_rppg"):
        out.append(f"evidence domain {dom!r} — §V gates are judged on "
                   "facial rPPG through the production path (this run is "
                   "machinery evidence only)")
    for k, label in (("participant_disjoint", "participant-disjoint"),
                     ("session_disjoint", "session-disjoint"),
                     ("production_path", "production-path")):
        if req.get(k, True) and not data.get(k):
            out.append(f"evidence is not {label}")
    return out


def evaluate_vo2_gates(gcfg: dict, evidence: dict) -> dict:
    ev = evidence or {}
    disq = _qualification(gcfg, ev.get("data"))
    rows = []

    def row(key, numeric_pass, reasons, metrics, thresholds):
        reasons = list(disq) + reasons
        status = "GREEN" if numeric_pass and not reasons else "RED"
        rows.append({"gate": key, "title": GATE_TITLES[key],
                     "status": status, "reasons": reasons,
                     "metrics": metrics, "thresholds": thresholds})

    # V1 — recovery-HR fidelity
    t1 = gcfg.get("v1_recovery_hr_fidelity") or {}
    v1 = ev.get("v1") or {}
    reasons = []
    bands = v1.get("hrr60_loa_bpm_by_band")
    lim = float(_req(t1, "hrr60_loa_bpm_max"))
    kill = float(_req(t1, "per_band_kill_bpm"))
    if not bands:
        reasons.append("no recovery-HR fidelity study on record")
    else:
        for band, loa in bands.items():
            if loa is None:
                reasons.append(f"Monk band {band}: LoA unavailable")
            elif loa > lim:
                reasons.append(f"Monk band {band}: HRR60 95% LoA "
                               f"±{loa} bpm exceeds ±{lim:.0f}")
            if loa is not None and loa > kill:
                reasons.append(f"Monk band {band}: beyond the "
                               f"±{kill:.0f} bpm kill rule — this band "
                               "must return NO_RESULT for recovery "
                               "metrics in code")
        if v1.get("per_window_rmse_by_recovery_time") is None:
            reasons.append("per-window HR RMSE by recovery time not "
                           "reported")
    row("v1", not reasons, reasons, v1, t1)

    # V2 — repeatability
    t2 = gcfg.get("v2_protocol_repeatability") or {}
    v2 = ev.get("v2") or {}
    reasons = []
    te = v2.get("hrr60_typical_error_bpm")
    if te is None:
        reasons.append("no test-retest study on record")
    elif te > float(_req(t2, "hrr60_typical_error_bpm_max")):
        reasons.append(f"HRR60 typical error {te} bpm exceeds the "
                       f"{t2.get('hrr60_typical_error_bpm_max')} bpm gate")
    ar = v2.get("accepted_scan_rate")
    if ar is None:
        reasons.append("accepted-scan rate (ages 20-75) unavailable")
    elif ar < float(_req(t2, "accepted_scan_rate_min")):
        reasons.append(f"accepted-scan rate {ar:.2f} below "
                       f"{t2.get('accepted_scan_rate_min')}")
    ab = v2.get("abstention_by_band")
    par = _req(t2, "noread_parity")
    if ab:
        vals = [x for x in ab.values() if x is not None]
        if vals:
            # same semantics as the sanctioned no_read_report: a band
            # fails when it exceeds BOTH margins via max(); a
            # zero-abstention best band must not fail-OPEN the gate
            # (v0.4 review finding: `best > 0 and ...` disabled it)
            best = min(vals)
            thr = max(best * float(_req(par, "max_ratio")),
                      best + float(_req(par, "max_abs_pts")) / 100.0)
            for band, x in ab.items():
                if x is not None and x > thr:
                    reasons.append(f"no-read parity FAIL: Monk band "
                                   f"{band} abstention {x:.2f} vs best "
                                   f"{best:.2f}")
    else:
        reasons.append("per-band abstention rates unavailable")
    row("v2", not reasons, reasons, v2, t2)

    # V3 — incremental value (pivotal)
    t3 = gcfg.get("v3_incremental_value") or {}
    v3 = ev.get("v3") or {}
    reasons = []
    dsee, ci = v3.get("delta_see_mlkgmin"), v3.get("delta_see_ci95")
    if dsee is None:
        reasons.append("no held-out CPET evaluation on record")
    else:
        if dsee < float(_req(t3, "delta_see_mlkgmin_min")):
            reasons.append(f"ΔSEE {dsee} mL/kg/min below the "
                           f"{t3.get('delta_see_mlkgmin_min')} gate — "
                           "recovery physiology adds too little beyond "
                           "demographics+activity")
        if not ci or ci[0] <= 0:
            reasons.append("ΔSEE confidence interval does not exclude "
                           "zero")
        gain, gci = (v3.get("category_accuracy_gain_pts"),
                     v3.get("category_gain_ci95"))
        if gain is None or gain < float(
                _req(t3, "category_accuracy_gain_pts_min")):
            reasons.append(f"category-accuracy gain {gain} pts below the "
                           f"{t3.get('category_accuracy_gain_pts_min')}-pt"
                           " gate")
        elif not gci or gci[0] <= 0:
            reasons.append("category-gain confidence interval does not "
                           "exclude zero")
        if _req(t3, "must_beat_static_face_negative_control") and \
                not v3.get("beats_static_face_control"):
            reasons.append("does not beat the static-face-image negative "
                           "control — the model may be reading "
                           "appearance, not physiology")
    row("v3", not reasons, reasons, v3, t3)

    # V4 — subgroup honesty
    t4 = gcfg.get("v4_subgroup_honesty") or {}
    v4 = ev.get("v4") or {}
    reasons = []
    if not v4.get("subgroup_tables_present"):
        reasons.append("per-subgroup bias/LoA tables not on record "
                       f"(required: {t4.get('subgroups')})")
    bias = v4.get("beta_blocker_bias_met")
    routed = bool(v4.get("medicated_hard_routed_to_trend"))
    lim4 = float(_req(t4, "beta_blocker_bias_met_max"))
    if not routed and (bias is None or bias > lim4):
        reasons.append(f"beta-blocker bias {bias} MET not shown ≤ "
                       f"{lim4:.0f} and medicated users are not "
                       "hard-routed to trend-only in code")
    row("v4", not reasons, reasons, v4, t4)

    # V5 — trend validity
    t5 = gcfg.get("v5_trend_validity") or {}
    v5 = ev.get("v5") or {}
    reasons = []
    r_ = v5.get("change_score_r")
    au = v5.get("auroc_1met")
    if r_ is None or au is None:
        reasons.append("no training-response study on record — "
                       "head_trend stays direction-only")
    else:
        if r_ < float(_req(t5, "change_score_r_min")):
            reasons.append(f"change-score r {r_} below "
                           f"{t5.get('change_score_r_min')}")
        if au < float(_req(t5, "auroc_1met_min")):
            reasons.append(f"AUROC {au} for a ≥1-MET change below "
                           f"{t5.get('auroc_1met_min')}")
    row("v5", not reasons, reasons, v5, t5)

    all_green = all(r["status"] == "GREEN" for r in rows)
    sign = gcfg.get("signoff") or {}
    signed = bool(sign.get("owner_confirmed")) and \
        sign.get("clinical_advisor") not in (None, "", "null")
    return {"gates": rows, "all_gates_green": all_green,
            "clinical_signoff": signed,
            "promotion_open": bool(all_green and signed)}


# ------------------------------------------------- scoreboard + status
def append_scoreboard(entry: dict, runs_root=None) -> pathlib.Path:
    root = pathlib.Path(runs_root or DEFAULT_RUNS)
    root.mkdir(parents=True, exist_ok=True)
    p = root / "scoreboard.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return p


def latest_scoreboard_entry(runs_root=None):
    p = pathlib.Path(runs_root or DEFAULT_RUNS) / "scoreboard.jsonl"
    if not p.exists():
        return None
    lines = [x for x in p.read_text().splitlines() if x.strip()]
    return json.loads(lines[-1]) if lines else None


def vo2_gate_status(*, runs_root=None, gates_path=None) -> dict:
    gcfg = load_vo2_gates(gates_path)
    entry = latest_scoreboard_entry(runs_root)
    verdict = evaluate_vo2_gates(gcfg, (entry or {}).get("evidence") or {})
    return {"track": "vo2", "note": TRACK_NOTE,
            "gates_version": gcfg.get("gates_version"),
            "requires_clinical_signoff":
                bool(gcfg.get("requires_clinical_signoff", True)),
            "signoff": gcfg.get("signoff"),
            "evidence_run": (entry or {}).get("run_id"),
            "evidence_data": ((entry or {}).get("evidence") or {})
            .get("data"),
            **verdict,
            "promotion": "OPEN" if verdict["promotion_open"] else "BLOCKED"}


def fitness_render_allowed(*, runs_root=None, gates_path=None) -> bool:
    """THE rendering invariant for INFERRED_FITNESS (invariant 9, v0.4
    extension): true only when every §V gate is green AND the clinical
    signoff stands. Fails CLOSED on any error — a broken gates file
    means nothing renders."""
    try:
        return bool(vo2_gate_status(runs_root=runs_root,
                                    gates_path=gates_path)
                    ["promotion_open"])
    except Exception:
        return False


def render_vo2_status_html(doc: dict) -> str:
    rows = []
    for g in doc["gates"]:
        colour = "#166534" if g["status"] == "GREEN" else "#92400e"
        reasons = "".join(f"<li>{r}</li>" for r in g["reasons"])
        rows.append(
            f"<tr><td>{g['title']}</td>"
            f"<td style='color:{colour};font-weight:bold'>{g['status']}"
            f"</td><td><ul>{reasons or '<li>pass</li>'}</ul>"
            f"<pre>{json.dumps(g['metrics'], indent=1)}</pre></td></tr>")
    return ("<!doctype html><meta charset='utf-8'>"
            f"<div style='background:#92400e;color:#fff;padding:10px;"
            f"font:bold 14px sans-serif'>{TRACK_NOTE}</div>"
            "<h2 style='font-family:sans-serif'>VO2/CRF fitness-track "
            "promotion scoreboard (§V)</h2>"
            f"<p style='font-family:sans-serif'>gates "
            f"{doc['gates_version']} · evidence run: "
            f"{doc.get('evidence_run') or 'none'} · promotion: "
            f"<b>{doc['promotion']}</b> · clinical signoff: "
            f"{'yes' if doc['clinical_signoff'] else 'PENDING'}</p>"
            "<table border='1' cellspacing='0' cellpadding='6' "
            "style='font:12px sans-serif;border-collapse:collapse'>"
            "<tr><th>gate</th><th>status</th><th>evidence</th></tr>"
            + "".join(rows) + "</table>"
            f"<div style='background:#92400e;color:#fff;padding:10px;"
            f"font:bold 14px sans-serif'>{TRACK_NOTE}</div>")
