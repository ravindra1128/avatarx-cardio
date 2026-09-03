"""§R gate machinery (v0.7 regularity track), on the SHARED mechanics
(evaluation/_gate_common — reused, not forked). Pure stdlib + configs;
fails CLOSED on any error, so app/ can consult it directly.

Two run kinds carry §R evidence: "evaluation" (evaluation/
regularity_metrics.py — ceiling test, benign separation, baselines,
fairness) and "floor" (evaluation/regularity_floor.py — the jitter
budget, the MDI table, the beat-error study). R1 reads the floor run;
everything else reads the evaluation run. The claim map (R5) lives in
the signed signoff block, nowhere else.
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
    os.environ.get("AVATARX_REGULARITY_RUNS",
                   _REPO / "evaluation" / "regularity_runs"))

TRACK_NOTE = ("PULSE-REGULARITY INDEX — §R-GATED — NOT VALIDATED, "
              "NAMES NO RHYTHM, NOT A DIAGNOSIS")

GATE_TITLES = {
    "r0": "R0 — ceiling: camera vs ECG regularity agreement",
    "r1": "R1 — noise floor published (MDI per fps / SQI / skin tone)",
    "r2": "R2 — benign separation (RSA not flagged; age-stratified "
          "specificity)",
    "r3": "R3 — beats B3, B4, B5 subject-independent",
    "r4": "R4 — fairness (index bias, flag rate, coverage parity)",
    "r5": "R5 — claim mapping (owner + clinical advisor)",
}

_unset = _gate_common.unset
_req = _gate_common.make_req("regularity")
_qualification = _gate_common.qualification


def load_regularity_gates(path=None) -> dict:
    doc = parse_yaml_subset(
        pathlib.Path(path or DEFAULT_GATES).read_text())
    blk = doc.get("regularity")
    if not isinstance(blk, dict):
        raise ValueError("configs/gates.yaml has no regularity (§R) block")
    return blk


def append_scoreboard(entry: dict, runs_root=None) -> pathlib.Path:
    return _gate_common.append_scoreboard(entry, runs_root, DEFAULT_RUNS)


def latest_scoreboard_entry(kind=None, runs_root=None):
    return _gate_common.latest_scoreboard_entry(kind, runs_root,
                                                DEFAULT_RUNS)


def _as_float(v, lo=None, hi=None):
    """A finite number inside [lo, hi], or None. Booleans, NaN and the
    infinities are not measurements (review finding: -Infinity beat
    every margin and True read as kappa 1.0)."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    if lo is not None and f < lo:
        return None
    if hi is not None and f > hi:
        return None
    return f


def _as_count(v):
    if isinstance(v, bool):
        return 0
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 0
    return n if n >= 0 else 0


def _req_list(blk, key):
    """A pre-registered list that must be present AND non-empty: a null
    or empty list would make the gate vacuously green (review
    finding)."""
    v = _req(blk, key)
    if not isinstance(v, (list, tuple)) or not v:
        raise ValueError(f"configs/gates.yaml regularity.{key} must be a "
                         "non-empty list — an empty list would make the "
                         "gate vacuously green")
    return list(v)


_MDI_AXIS = {"fps": "fps", "sqi_grade": "sqi", "fitzpatrick_group": "fitz"}


def evaluate_regularity_gates(gcfg: dict, evidence: dict,
                              floor: dict = None) -> dict:
    """Apply §R to recorded evidence. Missing evidence for a gate is a
    RED gate with its own reason, never a skipped row."""
    ev = evidence or {}
    fl = floor or {}
    disq = _qualification(gcfg, ev.get("data"))
    rows: list = []

    def row(key, numeric_pass, reasons, metrics, thresholds):
        reasons = list(disq) + reasons
        status = "GREEN" if numeric_pass and not reasons else "RED"
        rows.append({"gate": key, "title": GATE_TITLES[key],
                     "status": status, "reasons": reasons,
                     "metrics": metrics, "thresholds": thresholds})

    # R0 — the ceiling ------------------------------------------------------
    t0 = gcfg.get("r0_ceiling") or {}
    r0 = ev.get("r0") or {}
    reasons = []
    if not r0:
        reasons.append("no camera-vs-ECG ceiling test on record")
    else:
        k_min = float(_req(t0, "kappa_min"))
        r_min = float(_req(t0, "index_r_min"))
        loa_max = float(_req(t0, "bland_altman_loa_max"))
        n_min = int(_req(t0, "min_paired_scans"))
        per_grade = bool(_req(t0, "per_sqi_grade"))
        kappa = _as_float(r0.get("kappa"), -1.0, 1.0)
        if kappa is None:
            reasons.append("no class agreement (kappa) on record")
        elif kappa < k_min:
            reasons.append(f"camera-vs-ECG kappa {kappa:.3f} < {k_min}")
        if per_grade:
            wk = _as_float(r0.get("worst_grade_kappa"), -1.0, 1.0)
            if wk is None:
                reasons.append("no per-SQI-grade kappa on record")
            elif wk < k_min:
                reasons.append(f"worst SQI-grade kappa {wk:.3f} < {k_min}")
        rr = _as_float(r0.get("index_r"), -1.0, 1.0)
        if rr is None:
            reasons.append("no index correlation on record")
        elif rr < r_min:
            reasons.append(f"index correlation {rr:.3f} < {r_min}")
        loa = _as_float(r0.get("loa_halfwidth"), 0.0, None)
        if loa is None:
            reasons.append("no Bland-Altman limits on record")
        elif loa > loa_max:
            reasons.append(f"Bland-Altman 95% limits +/-{loa:.4f} exceed "
                           f"{loa_max}")
        # the half-width alone lets a tight systematic offset through
        # (review finding): the bias is gated on its own
        bias_max = float(_req(t0, "bland_altman_bias_max"))
        bias = _as_float(r0.get("bias"))
        if bias is None:
            reasons.append("no Bland-Altman bias on record")
        elif abs(bias) > bias_max:
            reasons.append(f"camera-minus-ECG index bias {bias:+.4f} "
                           f"exceeds +/-{bias_max}")
        n = _as_count(r0.get("n_paired_scans"))
        if n < n_min:
            reasons.append(f"{n} paired camera+ECG scans (< {n_min})")
    row("r0", not reasons, reasons, r0, t0)

    # R1 — the noise floor, PUBLISHED -----------------------------------------
    t1 = gcfg.get("r1_noise_floor") or {}
    reasons = []
    if not fl:
        reasons.append("no noise-floor characterization on record "
                       "(cli.py regularity-floor)")
    else:
        need = _req_list(t1, "require_mdi_per")
        n_min = int(_req(t1, "min_regular_scans_per_cell"))
        fi_max = float(_req(t1, "max_false_irregular_rate_at_5pct_beat_error"))
        # the floor must have been computed at the PUBLISHED power and
        # confidence (review finding: the yaml values were never read)
        want_p = float(_req(t1, "mdi_power"))
        want_c = float(_req(t1, "mdi_confidence"))
        params = fl.get("parameters") or {}
        got_p, got_c = _as_float(params.get("mdi_power")), \
            _as_float(params.get("mdi_confidence"))
        if got_p is None or got_c is None:
            reasons.append("floor run does not record the MDI power and "
                           "confidence it was computed at")
        elif abs(got_p - want_p) > 1e-9 or abs(got_c - want_c) > 1e-9:
            reasons.append(f"floor run computed MDI at power {got_p} / "
                           f"confidence {got_c}, gates say {want_p} / "
                           f"{want_c}")
        if fl.get("signal_domain") is not None and \
                fl.get("signal_domain") != (ev.get("data") or {}).get(
                    "signal_domain"):
            reasons.append(f"floor run domain {fl.get('signal_domain')!r} "
                           "differs from the evaluation's — one cohort, "
                           "one floor")
        cells = fl.get("mdi_cells") or {}
        if not cells:
            reasons.append("no MDI cells on record")
        else:
            missing = [k for k, v in cells.items() if v is None]
            if missing:
                reasons.append(f"MDI not characterized in {len(missing)} of "
                               f"{len(cells)} cells ({', '.join(missing[:3])}"
                               f"{'...' if len(missing) > 3 else ''})")
            # every PRE-REGISTERED axis must be in every cell key (review
            # finding: the yaml list was read into an error string only)
            unknown = [a for a in need if a not in _MDI_AXIS]
            if unknown:
                reasons.append(f"require_mdi_per names axes the floor "
                               f"cannot key on: {unknown}")
            for k in cells:
                if not all(f"{_MDI_AXIS.get(a, a)}=" in k for a in need):
                    reasons.append(f"MDI cell {k!r} lacks a required axis "
                                   f"{need}")
                    break
            if any(_as_float(v, 0.0, None) is None for v in cells.values()
                   if v is not None):
                reasons.append("an MDI cell carries a value that is not a "
                               "finite non-negative number")
        gain = fl.get("interpolation_gain") or {}
        if not gain:
            reasons.append("no metronomic references — the interpolation "
                           "gain is unmeasured and the budget is theory")
        else:
            thin = [k for k, v in gain.items()
                    if int(v.get("n_metronomic_refs") or 0) < n_min]
            if thin:
                reasons.append(f"fewer than {n_min} metronomic references "
                               f"at fps {thin}")
        fi = fl.get("false_irregular_at_5pct") or {}
        if not fi:
            reasons.append("no beat-error study on record")
        else:
            for fps, rate in fi.items():
                r_ = _as_float(rate, 0.0, 1.0)
                if r_ is None:
                    reasons.append(f"false-irregular rate at 5% beat error "
                                   f"unmeasured at {fps} fps")
                elif r_ > fi_max:
                    reasons.append(f"false-irregular rate {r_:.3f} at 5% "
                                   f"beat error ({fps} fps) exceeds {fi_max}")
        doc_rel = "docs/regularity_track.md"
        p = _REPO / doc_rel
        if not p.exists() or "MDI" not in p.read_text():
            reasons.append(f"{doc_rel} does not publish the MDI table — "
                           "the noise floor is a product parameter, not an "
                           "internal number")
    row("r1", not reasons, reasons, fl, t1)

    # R2 — benign separation ---------------------------------------------------
    t2 = gcfg.get("r2_benign_separation") or {}
    r2 = ev.get("r2") or {}
    reasons = []
    if not r2:
        reasons.append("no benign-separation evaluation on record")
    else:
        rsa_max = float(_req(t2, "max_rsa_flag_rate"))
        bands = _req_list(t2, "age_bands")
        spec_min = float(_req(t2, "min_specificity_per_age_band"))
        n_min = int(_req(t2, "min_sessions_per_age_band"))
        rsa_n_min = int(_req(t2, "min_rsa_sessions"))
        nr_max = float(_req(t2, "max_no_read_rate_per_age_band"))
        rate = _as_float(r2.get("rsa_flag_rate"), 0.0, 1.0)
        n_rsa = _as_count(r2.get("n_rsa_sessions"))
        if rate is None:
            reasons.append("no RSA-session flag rate on record")
        elif rate > rsa_max:
            reasons.append(f"RSA-dominant sessions flagged as clinically "
                           f"irregular at {rate:.3f} > {rsa_max}")
        if n_rsa < rsa_n_min:
            reasons.append(f"{n_rsa} judged RSA sessions (< {rsa_n_min}) "
                           "— the RSA criterion needs a cohort")
        rsa_nr = _as_float(r2.get("rsa_no_read_rate"), 0.0, 1.0)
        if rsa_nr is None:
            reasons.append("RSA no-read rate unrecorded — abstention on "
                           "the sessions the head would get wrong buys "
                           "specificity with silence")
        elif rsa_nr > nr_max:
            reasons.append(f"head declined {rsa_nr:.3f} of RSA sessions "
                           f"(> {nr_max})")
        strata = r2.get("age_strata") or {}
        for b in bands:
            s = strata.get(b) or {}
            n = _as_count(s.get("n"))
            sp = _as_float(s.get("specificity"), 0.0, 1.0)
            nr = _as_float(s.get("no_read_rate"), 0.0, 1.0)
            if n < n_min:
                reasons.append(f"age band {b}: {n} judged sessions "
                               f"(< {n_min})")
            if nr is None:
                reasons.append(f"age band {b}: no-read rate unrecorded")
            elif nr > nr_max:
                reasons.append(f"age band {b}: head declined {nr:.3f} of "
                               f"benign sessions (> {nr_max})")
            if sp is None:
                reasons.append(f"age band {b}: specificity unrated")
            elif sp < spec_min:
                reasons.append(f"age band {b}: specificity {sp:.3f} < "
                               f"{spec_min} — threshold must be age-aware "
                               "or the claim narrowed")
    row("r2", not reasons, reasons, r2, t2)

    # R3 — baselines (PIVOTAL) --------------------------------------------------
    t3 = gcfg.get("r3_baselines") or {}
    r3 = ev.get("r3") or {}
    reasons = []
    if not r3:
        reasons.append("no baseline battery on record")
    else:
        m3 = float(_req(t3, "min_balanced_accuracy_margin_over_b3"))
        m4 = float(_req(t3, "min_balanced_accuracy_margin_over_b4"))
        m5 = float(_req(t3, "min_balanced_accuracy_margin_over_b5"))
        n_min = int(_req(t3, "min_test_participants"))
        nr_max = float(_req(t3, "max_head_no_read_rate"))
        head = _as_float(r3.get("head_balanced_accuracy"), 0.0, 1.0)
        if head is None:
            reasons.append("no held-out head balanced accuracy on record")
        # like with like: the baselines RE-SCORED on the rows the head
        # judged (review finding: the head was scored on an easier
        # subset than the baselines)
        for name, margin in (("b3", m3), ("b4", m4), ("b5", m5)):
            b = _as_float(r3.get(f"{name}_balanced_accuracy_on_judged"),
                          0.0, 1.0)
            if b is None:
                reasons.append(f"no held-out {name.upper()} on the head's "
                               "judged rows on record")
            elif head is not None and head < b + margin:
                reasons.append(f"head {head:.3f} does not beat "
                               f"{name.upper()} {b:.3f} (same rows) by "
                               f"{margin}")
        nr = _as_float(r3.get("head_no_read_rate"), 0.0, 1.0)
        if nr is None:
            reasons.append("head no-read rate unrecorded")
        elif nr > nr_max:
            reasons.append(f"head declined {nr:.3f} of held-out rows "
                           f"(> {nr_max}) — abstention is not accuracy")
        n = _as_count(r3.get("n_test_participants_judged",
                             r3.get("n_test_participants")))
        if n < n_min:
            reasons.append(f"{n} held-out participants judged (< {n_min})")
    row("r3", not reasons, reasons, r3, t3)

    # R4 — fairness -------------------------------------------------------------
    t4 = gcfg.get("r4_fairness") or {}
    r4 = ev.get("r4") or {}
    reasons = []
    if not r4:
        reasons.append("no fairness evaluation on record")
    else:
        bias_max = float(_req(t4, "max_index_bias"))
        par_min = float(_req(t4, "flag_rate_parity_ratio_min"))
        cov_min = float(_req(t4, "coverage_ratio_min"))
        need_dark = bool(_req(t4, "require_darkest_band_present"))
        for axis in _req_list(t4, "subgroups"):
            a = r4.get(axis) or {}
            if not a:
                reasons.append(f"{axis}: no subgroup table")
                continue
            b = _as_float(a.get("worst_index_bias"))
            if b is None:
                reasons.append(f"{axis}: index bias unrated")
            elif abs(b) > bias_max:
                reasons.append(f"{axis}: worst-group camera-minus-ECG index "
                               f"bias {b:+.4f} exceeds +/-{bias_max}")
            par = _as_float(a.get("flag_rate_parity_ratio_worst"), 0.0, 1.0)
            if par is None:
                reasons.append(f"{axis}: flag-rate parity unrated")
            elif par < par_min:
                reasons.append(f"{axis}: flag-rate parity {par:.3f} < "
                               f"{par_min}")
            cov = _as_float(a.get("coverage_ratio_worst"), 0.0, 1.0)
            if cov is None:
                reasons.append(f"{axis}: coverage parity unrated")
            elif cov < cov_min:
                reasons.append(f"{axis}: coverage ratio {cov:.3f} < "
                               f"{cov_min}")
            if need_dark and a.get("darkest_band_present") is not True:
                reasons.append(f"{axis}: the darkest skin-tone band is "
                               "absent from the cohort")
    row("r4", not reasons, reasons, r4, t4)

    # R5 — claim mapping ----------------------------------------------------------
    t5 = gcfg.get("r5_claim_mapping") or {}
    r5 = ev.get("r5") or {}
    reasons = []
    permissible = str(_req(t5, "permissible_first_claim"))
    sign = gcfg.get("signoff") or {}
    signed = sign.get("owner_confirmed") is True and \
        not _unset(sign.get("clinical_advisor"))
    if not signed:
        reasons.append("claim mapping with owner + clinical advisor not on "
                       "record — the first permissible surface is the "
                       "sanctioned irregular-rhythm notification, which "
                       "names no rhythm and offers the benign explanation "
                       "first")
    if _unset(sign.get("claim_scope")) or \
            str(sign.get("claim_scope")) != permissible:
        reasons.append(f"signoff claim_scope is not {permissible!r} — "
                       "nothing may render")
    if bool(_req(t5, "this_head_never_escalates")) and \
            r5.get("escalation_path_present"):
        reasons.append("an escalation path from this head to an AFib "
                       "sentence exists — escalation is head_afib's job")
    row("r5", not reasons, reasons, r5, t5)

    all_green = all(r["status"] == "GREEN" for r in rows)
    return {"gates": rows, "all_gates_green": all_green,
            "clinical_signoff": signed,
            "promotion_open": bool(all_green and signed)}


def _drift_reasons(gcfg: dict, entry, floor) -> list:
    """Evidence recorded under other gates is not evidence under these
    (review finding: a re-versioned block with a moved threshold opened
    on old numbers)."""
    out = []
    live = gcfg.get("gates_version")
    for label, row in (("evaluation", entry), ("floor", floor)):
        if not row:
            continue
        v = row.get("gates_version")
        if v != live:
            out.append(f"{label} run recorded under gates {v!r}; live "
                       f"gates are {live!r} — stale evidence")
    ref = (entry or {}).get("reference_label")
    if entry and ref != gcfg.get("reference_label"):
        out.append("evaluation run recorded under a different reference "
                   "label than the live one — stale evidence")
    return out


def regularity_gate_status(*, runs_root=None, gates_path=None) -> dict:
    gcfg = load_regularity_gates(gates_path)
    entry = latest_scoreboard_entry("evaluation", runs_root=runs_root)
    floor = latest_scoreboard_entry("floor", runs_root=runs_root)
    evidence = (entry or {}).get("evidence") or {}
    verdict = evaluate_regularity_gates(gcfg, evidence, floor or {})
    drift = _drift_reasons(gcfg, entry, floor)
    if drift:
        for g in verdict["gates"]:
            g["status"] = "RED"
            g["reasons"] = list(drift) + list(g.get("reasons") or [])
        verdict["all_gates_green"] = False
        verdict["promotion_open"] = False
    return {"track": "regularity", "note": TRACK_NOTE,
            "gates_version": gcfg.get("gates_version"),
            "requires_clinical_signoff":
                gcfg.get("requires_clinical_signoff"),
            "signoff": gcfg.get("signoff"),
            "reference_label": gcfg.get("reference_label"),
            "evaluation_run": (entry or {}).get("run_id"),
            "floor_run": (floor or {}).get("run_id"),
            **verdict,
            "promotion": ("OPEN" if verdict["promotion_open"]
                          else "BLOCKED")}


def regularity_render_allowed(*, runs_root=None, gates_path=None) -> bool:
    """The G-d rendering invariant, fail-CLOSED: any error means no."""
    try:
        return bool(regularity_gate_status(
            runs_root=runs_root,
            gates_path=gates_path)["promotion_open"])
    except Exception:
        return False


def render_regularity_status_html(doc: dict) -> str:
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
            f"<pre>{json.dumps(g.get('metrics') or {}, indent=1, default=str)}"
            "</pre>")
    sign = "yes" if doc.get("clinical_signoff") else "PENDING"
    return ("<html><body style='font-family:system-ui;max-width:760px;"
            "margin:2em auto'>" + banner +
            "<h2>Pulse-regularity track promotion scoreboard (&sect;R)</h2>"
            f"<p>gates {doc.get('gates_version')} · evaluation run: "
            f"{doc.get('evaluation_run') or 'none'} · floor run: "
            f"{doc.get('floor_run') or 'none'} · promotion: "
            f"<b>{doc.get('promotion')}</b> · clinical signoff: {sign}</p>"
            + "".join(rows) + banner + "</body></html>")
