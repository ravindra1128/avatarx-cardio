"""§F gate machinery (v0.6 atrial-flutter track).

The `flutter:` block of configs/gates.yaml is the pre-registered
contract; this module only APPLIES it, on the SHARED gate mechanics
(evaluation/_gate_common — reused, not forked). Pure stdlib + configs;
fails CLOSED on any error, so app/ can consult it directly.

What is being gated is NOT a flutter detector. The pulse cannot name a
rhythm, so §F gates a regular-tachyarrhythmia PATTERN FLAG that routes
to an ECG. Two consequences shape the gates:

  * F1's binding constraint is SPECIFICITY, not sensitivity, because a
    regular fast pulse is usually benign — and the flag must beat B3,
    the interpretable rate+dispersion+coupling rule. If it cannot, B3
    ships and the scoreboard says so, which is why f1 records both.
  * F2 exists so the KNOWN MISSES are counted rather than asserted
    (invariant F-c): slow fixed-block flutter at ~75 bpm is expected to
    be indistinguishable from normal sinus rhythm by pulse timing, and
    that expectation must be quantified per conduction ratio.

One run kind ("evaluation", evaluation/flutter_metrics.py) carries all
§F evidence. The claim map (F5) lives in the signed signoff block,
nowhere else.
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
    os.environ.get("AVATARX_FLUTTER_RUNS",
                   _REPO / "evaluation" / "flutter_runs"))

TRACK_NOTE = ("REGULAR-TACHYARRHYTHMIA PATTERN FLAG — §F-GATED — "
              "NOT VALIDATED, NAMES NO RHYTHM, NOT A DIAGNOSIS")

GATE_TITLES = {
    "f0": "F0 — pulse-rate accuracy during arrhythmia",
    "f1": "F1 — flag performance (sensitivity, specificity, vs B3)",
    "f2": "F2 — per-ratio reporting incl. the quantified known miss",
    "f3": "F3 — serial signature (integer-ratio steps, one atrial rate)",
    "f4": "F4 — fairness and coverage parity",
    "f5": "F5 — claim mapping (owner + clinical advisor)",
}

_unset = _gate_common.unset
_req = _gate_common.make_req("flutter")
_qualification = _gate_common.qualification


def load_flutter_gates(path=None) -> dict:
    doc = parse_yaml_subset(
        pathlib.Path(path or DEFAULT_GATES).read_text())
    blk = doc.get("flutter")
    if not isinstance(blk, dict):
        raise ValueError("configs/gates.yaml has no flutter (§F) block")
    return blk


def append_scoreboard(entry: dict, runs_root=None) -> pathlib.Path:
    return _gate_common.append_scoreboard(entry, runs_root, DEFAULT_RUNS)


def latest_scoreboard_entry(kind=None, runs_root=None):
    return _gate_common.latest_scoreboard_entry(kind, runs_root,
                                                DEFAULT_RUNS)


def _as_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None            # NaN is not a value


def evaluate_flutter_gates(gcfg: dict, evidence: dict) -> dict:
    """Apply §F to recorded evidence. Missing evidence for a gate is a
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

    # F0 — rate accuracy IN arrhythmia ------------------------------------
    t0 = gcfg.get("f0_rate_accuracy") or {}
    f0 = ev.get("f0") or {}
    reasons = []
    if not f0:
        reasons.append("no rate-accuracy evaluation on record — a "
                       "rate-based flag means nothing until the rate "
                       "itself is right during arrhythmia")
    else:
        bias_max = float(_req(t0, "max_abs_bias_bpm"))
        within_min = float(_req(t0, "min_within_5bpm_fraction"))
        n_min = int(_req(t0, "min_arrhythmia_scans"))
        bias = _as_float(f0.get("bias_bpm"))
        within = _as_float(f0.get("within_5bpm_fraction"))
        n = int(f0.get("n_arrhythmia_scans") or 0)
        if bias is None:
            reasons.append("no pulse-rate bias vs ECG on record")
        elif abs(bias) > bias_max:
            reasons.append(f"pulse-rate bias {bias:+.2f} bpm during "
                           f"arrhythmia exceeds +/-{bias_max} bpm")
        if within is None:
            reasons.append("no within-5-bpm fraction on record")
        elif within < within_min:
            reasons.append(f"only {within:.1%} of arrhythmia readings "
                           f"within 5 bpm of ECG (< {within_min:.0%})")
        if n < n_min:
            reasons.append(f"{n} arrhythmia scans with a paired rate "
                           f"reference (< {n_min})")
    row("f0", not reasons, reasons, f0, t0)

    # F1 — flag performance, and the B3 competitor ------------------------
    t1 = gcfg.get("f1_flag_performance") or {}
    f1 = ev.get("f1") or {}
    reasons = []
    if not f1:
        reasons.append("no flag-performance evaluation on record")
    else:
        sens_min = float(_req(t1, "min_sensitivity_2to1"))
        spec_min = float(_req(t1, "min_specificity_battery"))
        must_beat = bool(_req(t1, "must_beat_b3"))
        margin = float(_req(t1, "min_b3_margin"))
        n_pos_min = int(_req(t1, "min_flutter_scans"))
        n_neg_min = int(_req(t1, "min_negative_scans"))
        dominant = str(_req(t1, "require_dominant_negative"))
        sens = _as_float(f1.get("sensitivity_2to1"))
        spec = _as_float(f1.get("specificity_battery"))
        if sens is None:
            reasons.append("no 2:1-band sensitivity on record")
        elif sens < sens_min:
            reasons.append(f"2:1 sensitivity {sens:.3f} < {sens_min}")
        if spec is None:
            reasons.append("no battery specificity on record")
        elif spec < spec_min:
            reasons.append(f"specificity against the hard-negative "
                           f"battery {spec:.3f} < {spec_min}")
        n_pos = int(f1.get("n_flutter_scans") or 0)
        n_neg = int(f1.get("n_negative_scans") or 0)
        if n_pos < n_pos_min:
            reasons.append(f"{n_pos} ECG-confirmed flutter scans "
                           f"(< {n_pos_min})")
        if n_neg < n_neg_min:
            reasons.append(f"{n_neg} negative scans (< {n_neg_min})")
        # the battery must be dominated by the confounder that actually
        # presents; a specificity earned on easy negatives is not one
        share = _as_float((f1.get("negative_class_share") or {})
                          .get(dominant))
        if share is None:
            reasons.append(f"no {dominant} share recorded for the "
                           "negative battery")
        elif share < 0.5:
            reasons.append(f"{dominant} is only {share:.0%} of the "
                           "negative battery — specificity must be "
                           "earned against the dominant confounder")
        if must_beat:
            b3 = _as_float(f1.get("b3_specificity_at_matched_sens"))
            head = _as_float(f1.get("head_specificity_at_matched_sens"))
            if b3 is None or head is None:
                reasons.append("no head-vs-B3 comparison at matched "
                               "sensitivity on record — B3 is the "
                               "interpretable competitor and must be "
                               "beaten or shipped")
            elif head < b3 + margin:
                reasons.append(
                    f"head specificity {head:.3f} does not beat B3 "
                    f"{b3:.3f} by {margin} at matched sensitivity — "
                    "SHIP B3 (a transparent rule) and say so")
    row("f1", not reasons, reasons, f1, t1)

    # F2 — per-ratio reporting, misses quantified -------------------------
    t2 = gcfg.get("f2_per_ratio_reporting") or {}
    f2 = ev.get("f2") or {}
    reasons = []
    per_ratio = (f2.get("per_ratio") or {}) if f2 else {}
    if not f2:
        reasons.append("no per-ratio breakdown on record")
        required = []
    else:
        required = list(_req(t2, "required_ratios") or [])
        n_min = int(_req(t2, "min_scans_per_ratio"))
        for ratio in required:
            rr = per_ratio.get(ratio)
            if not rr:
                reasons.append(f"conduction ratio {ratio} not reported")
                continue
            if int(rr.get("n") or 0) < n_min:
                reasons.append(f"{ratio}: {int(rr.get('n') or 0)} scans "
                               f"(< {n_min})")
            if _as_float(rr.get("sensitivity")) is None:
                reasons.append(f"{ratio}: no sensitivity reported")
        if bool(_req(t2, "slow_block_miss_must_be_quantified")):
            slow = per_ratio.get("FOUR_TO_ONE") or {}
            if _as_float(slow.get("sensitivity")) is None:
                reasons.append(
                    "slow fixed-block (4:1) sensitivity is not "
                    "quantified — F-c requires the known miss to be "
                    "MEASURED, not asserted")
    row("f2", not reasons, reasons, f2, t2)

    # F3 — serial signature ------------------------------------------------
    t3 = gcfg.get("f3_serial_signature") or {}
    f3 = ev.get("f3") or {}
    reasons = []
    if not f3:
        reasons.append("no serial-signature evaluation on record")
    else:
        auc_min = float(_req(t3, "min_series_auc"))
        pos_min = int(_req(t3, "min_flutter_series"))
        neg_min = int(_req(t3, "min_nonflutter_series"))
        scans_min = int(_req(t3, "min_scans_per_series"))
        auc = _as_float(f3.get("series_auc"))
        if auc is None:
            reasons.append("no series AUC on record")
        elif auc < auc_min:
            reasons.append(f"serial-signature AUC {auc:.3f} < {auc_min}")
        n_pos = int(f3.get("n_flutter_series") or 0)
        n_neg = int(f3.get("n_nonflutter_series") or 0)
        if n_pos < pos_min:
            reasons.append(f"{n_pos} flutter series (< {pos_min})")
        if n_neg < neg_min:
            reasons.append(f"{n_neg} non-flutter series (< {neg_min})")
        worst = int(f3.get("min_scans_per_series") or 0)
        if worst < scans_min:
            reasons.append(f"shortest series has {worst} scans "
                           f"(< {scans_min}) — the signature is "
                           "undefined below that")
    row("f3", not reasons, reasons, f3, t3)

    # F4 — fairness --------------------------------------------------------
    t4 = gcfg.get("f4_fairness") or {}
    f4 = ev.get("f4") or {}
    reasons = []
    if not f4:
        reasons.append("no fairness evaluation on record")
    else:
        par_min = float(_req(t4, "detection_parity_ratio_min"))
        cov_min = float(_req(t4, "coverage_ratio_min"))
        min_group = int(_req(t4, "min_group_participants"))
        need_dark = bool(_req(t4, "require_darkest_band_present"))
        for axis in list(_req(t4, "subgroups") or []):
            a = (f4.get(axis) or {})
            if not a:
                reasons.append(f"{axis}: no subgroup table")
                continue
            par = _as_float(a.get("detection_parity_ratio_worst"))
            cov = _as_float(a.get("coverage_ratio_worst"))
            if par is None:
                reasons.append(f"{axis}: worst-group detection parity "
                               "is unrated (a group below the "
                               f"{min_group}-participant floor)")
            elif par < par_min:
                reasons.append(f"{axis}: worst-group detection parity "
                               f"{par:.3f} < {par_min}")
            if cov is None:
                reasons.append(f"{axis}: worst-group coverage unrated")
            elif cov < cov_min:
                reasons.append(f"{axis}: worst-group coverage ratio "
                               f"{cov:.3f} < {cov_min}")
            if need_dark and not a.get("darkest_band_present"):
                reasons.append(
                    f"{axis}: the darkest skin-tone band is absent from "
                    "the cohort — the reference study excluded its only "
                    "such patient and we do not get to do that")
    row("f4", not reasons, reasons, f4, t4)

    # F5 — claim mapping (the signed signoff IS the record) ---------------
    t5 = gcfg.get("f5_claim_mapping") or {}
    f5 = ev.get("f5") or {}
    reasons = []
    if not f5:
        reasons.append("no claim-mapping evidence on record")
    permissible = str(_req(t5, "permissible_first_claim"))
    sign = gcfg.get("signoff") or {}
    signed = sign.get("owner_confirmed") is True and \
        not _unset(sign.get("clinical_advisor"))
    if not signed:
        reasons.append("claim mapping with owner + clinical advisor not "
                       "on record — the first permissible surface is the "
                       "sanctioned regular-tachy sentence, which names "
                       "no rhythm and routes to an ECG")
    if _unset(sign.get("claim_scope")) or \
            str(sign.get("claim_scope")) != permissible:
        reasons.append(f"signoff claim_scope is not {permissible!r} — "
                       "nothing may render")
    if bool(_req(t5, "combined_endpoint_decision_required")) and \
            not (f5.get("combined_endpoint_decision") or "").strip():
        reasons.append("no combined-endpoint decision on record: AF "
                       "alone vs flag alone vs combined must be decided "
                       "on numbers and recorded (Task 6)")
    doc_rel = str(_req(t5, "limitations_doc_required"))
    if not (_REPO / doc_rel).exists():
        reasons.append(f"{doc_rel} is not published — the known-miss "
                       "registry is a shipped artifact, not a caveat "
                       "(F-c)")
    row("f5", not reasons, reasons, f5, t5)

    all_green = all(r["status"] == "GREEN" for r in rows)
    return {"gates": rows, "all_gates_green": all_green,
            "clinical_signoff": signed,
            "promotion_open": bool(all_green and signed)}


def flutter_gate_status(*, runs_root=None, gates_path=None) -> dict:
    gcfg = load_flutter_gates(gates_path)
    entry = latest_scoreboard_entry("evaluation", runs_root=runs_root)
    evidence = (entry or {}).get("evidence") or {}
    verdict = evaluate_flutter_gates(gcfg, evidence)
    return {"track": "flutter", "note": TRACK_NOTE,
            "gates_version": gcfg.get("gates_version"),
            "requires_clinical_signoff":
                gcfg.get("requires_clinical_signoff"),
            "signoff": gcfg.get("signoff"),
            "evaluation_run": (entry or {}).get("run_id"),
            "shipping_rule": shipping_rule(evidence,
                                           gates_path=gates_path),
            **verdict,
            "promotion": ("OPEN" if verdict["promotion_open"]
                          else "BLOCKED")}


def shipping_rule(evidence: dict, *, gates_path=None) -> str:
    """Which rule the evidence says should ship: the head, or B3.

    Pre-registered in the gates and reported on the scoreboard so the
    answer is visible before anyone argues about it. "undecided" until a
    head-vs-B3 comparison at matched sensitivity exists — and B3 is the
    default the moment anything is unclear, because B3 is the
    transparent rule and shipping the learned head is the claim that
    needs support.
    """
    f1 = (evidence or {}).get("f1") or {}
    b3 = _as_float(f1.get("b3_specificity_at_matched_sens"))
    head = _as_float(f1.get("head_specificity_at_matched_sens"))
    if b3 is None or head is None:
        return "undecided"
    try:
        gcfg = load_flutter_gates(gates_path)
        margin = float(_req(gcfg.get("f1_flag_performance") or {},
                            "min_b3_margin"))
    except Exception:
        # a missing or unreadable margin must not become a zero margin,
        # which would hand a tie to the head (review finding)
        return "undecided"
    return "head" if head >= b3 + margin else "b3"


def flutter_render_allowed(*, runs_root=None, gates_path=None) -> bool:
    """The F-a/F-b rendering invariant, fail-CLOSED: any error means no.
    This is the ONLY sanctioned source of `render_allowed` for
    heads.head_flutter.user_facing_text()."""
    try:
        return bool(flutter_gate_status(
            runs_root=runs_root,
            gates_path=gates_path)["promotion_open"])
    except Exception:
        return False


def render_flutter_status_html(doc: dict) -> str:
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
            "<h2>Regular-tachyarrhythmia flag promotion scoreboard "
            "(&sect;F)</h2>"
            f"<p>gates {doc.get('gates_version')} · evaluation run: "
            f"{doc.get('evaluation_run') or 'none'} · promotion: "
            f"<b>{doc.get('promotion')}</b> · clinical signoff: {sign} · "
            f"rule that would ship: <b>{doc.get('shipping_rule')}</b>"
            "</p>" + "".join(rows) + banner + "</body></html>")
