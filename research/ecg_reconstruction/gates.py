"""
§G gate machinery (v0.3 T3/T4) — configs/gates.yaml is the pre-registered
contract; this module only APPLIES it. Every evaluation run appends its
evidence + verdicts to research/runs/scoreboard.jsonl; `gate_status()`
renders the current scoreboard; `models/registry.promote` reads a run's
gate_results.json and refuses while anything is red.

Two layers, kept honest:
  * numeric verdicts — do the numbers pass the thresholds on THIS run's
    data;
  * qualification — does the data itself count (facial rPPG through the
    production path, participant- AND session-disjoint). Synthetic and
    MIMIC finger-PPG runs exercise the machinery and are recorded, but a
    gate can only be GREEN on qualifying data; and promotion additionally
    requires the clinical signoff block in gates.yaml.
"""
from __future__ import annotations

import json
import os
import pathlib

from configs import parse_yaml_subset
from research.ecg_reconstruction import WATERMARK

_REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_GATES = _REPO / "configs" / "gates.yaml"
DEFAULT_RUNS = pathlib.Path(
    os.environ.get("AVATARX_RESEARCH_RUNS", _REPO / "research" / "runs"))

GATE_TITLES = {
    "g1": "G1 — beats the identity-template baseline",
    "g2": "G2 — interval fidelity (QT/PR/QRS MAE + RR-only baseline)",
    "g3": "G3 — abnormality preservation (blinded read)",
    "g4": "G4 — no confabulation (withheld-class challenge)",
    "g5": "G5 — detection non-inferiority vs the measured path",
}


def load_gates(path=None) -> dict:
    return parse_yaml_subset(pathlib.Path(path or DEFAULT_GATES).read_text())


def _qualification(gcfg: dict, data: dict) -> list:
    """Reasons the evidence CANNOT open gates (empty = qualifying)."""
    req = gcfg.get("evidence_requirements") or {}
    out = []
    if data is None:
        return ["no evaluation run on record"]
    dom = data.get("signal_domain")
    if dom != req.get("signal_domain", "facial_rppg"):
        out.append(f"evidence domain {dom!r} — gates are judged on facial "
                   "rPPG through the production path (this run is "
                   "machinery/pretraining evidence only)")
    for k, label in (("participant_disjoint", "participant-disjoint"),
                     ("session_disjoint", "session-disjoint"),
                     ("production_path", "production-path")):
        if req.get(k, True) and not data.get(k):
            out.append(f"evidence is not {label}")
    return out


def evaluate_gates(gcfg: dict, evidence: dict) -> dict:
    ev = evidence or {}
    data = ev.get("data")
    disq = _qualification(gcfg, data)
    agg = ev.get("fidelity") or {}
    g1 = ev.get("g1") or {}
    conf = ev.get("confabulation") or {}
    det = ev.get("detection") or {}
    reads = ev.get("blinded_reads")
    rows = []

    def row(key, numeric_pass, reasons, metrics, thresholds):
        reasons = list(disq) + reasons
        status = "GREEN" if numeric_pass and not reasons else "RED"
        rows.append({"gate": key, "title": GATE_TITLES[key],
                     "status": status, "reasons": reasons,
                     "metrics": metrics, "thresholds": thresholds})

    # G1
    ok = bool(g1.get("all_morphology_metrics_beat_identity"))
    reasons = [] if ok else (
        ["no fidelity evaluation on record"] if not g1 else
        [f"identity template not beaten on {k} "
         f"(decoder {v['decoder']} vs identity {v['identity']} ms)"
         for k, v in g1.items() if isinstance(v, dict)
         and not v.get("decoder_wins")])
    row("g1", ok, reasons, g1, gcfg.get("g1_identity_baseline") or {})

    # G2
    t2 = gcfg.get("g2_interval_fidelity") or {}
    reasons = []
    for k, lim_key in (("qt_mae_ms", "qt_mae_ms_max"),
                       ("pr_mae_ms", "pr_mae_ms_max"),
                       ("qrs_mae_ms", "qrs_mae_ms_max")):
        v = agg.get(f"decoder_{k}")
        lim = t2.get(lim_key)
        if v is None:
            reasons.append(f"{k} unavailable")
        elif lim is not None and v > lim:
            reasons.append(f"{k} {v} ms exceeds the {lim} ms gate")
    qt, rr_qt = agg.get("decoder_qt_mae_ms"), agg.get("rr_only_qt_mae_ms")
    if t2.get("must_beat_rr_only_qt_baseline", True):
        if qt is None or rr_qt is None:
            reasons.append("RR-only QT baseline comparison unavailable")
        elif qt >= rr_qt:
            reasons.append(f"QT MAE {qt} ms does not beat the RR-only "
                           f"regression baseline ({rr_qt} ms)")
    m2 = {k: agg.get(k) for k in ("decoder_qt_mae_ms", "decoder_pr_mae_ms",
                                  "decoder_qrs_mae_ms",
                                  "decoder_qt_mae_ms_ci95",
                                  "decoder_pr_mae_ms_ci95",
                                  "decoder_qrs_mae_ms_ci95",
                                  "rr_only_qt_mae_ms", "decoder_corr")}
    row("g2", not reasons, reasons, m2, t2)

    # G3
    t3 = gcfg.get("g3_blinded_read") or {}
    reasons = []
    if not reads or reads.get("n_scored", 0) == 0:
        reasons.append("no blinded cardiologist read on record")
        ok = False
    else:
        se, sp = reads.get("sensitivity"), reads.get("specificity")
        ok = (se is not None and sp is not None
              and se >= t3.get("sensitivity_min", 0.80)
              and sp >= t3.get("specificity_min", 0.80))
        if not ok:
            reasons.append(f"blinded read Se {se} / Sp {sp} below the "
                           f"{t3.get('sensitivity_min')}/"
                           f"{t3.get('specificity_min')} floors")
    row("g3", ok, reasons, reads or {}, t3)

    # G4
    t4 = gcfg.get("g4_no_confabulation") or {}
    reasons = []
    exc = conf.get("hallucinated_p_excess")
    lim = t4.get("max_hallucinated_p_prominence_excess", 0.10)
    if exc is None:
        reasons.append("withheld-class challenge not on record")
    elif exc > lim:
        reasons.append(f"decoder trained without AF hallucinates P waves "
                       f"into AF (prominence excess {exc} > {lim}: "
                       f"gen {conf.get('p_prominence_gen_on_withheld_af')} "
                       f"vs ref {conf.get('p_prominence_ref_on_af')})")
    if t4.get("blinded_reader_scored", True) and (
            not reads or reads.get("n_scored", 0) == 0):
        reasons.append("withheld-class blinded scoring not on record")
    row("g4", not reasons, reasons, conf, t4)

    # G5
    t5 = gcfg.get("g5_detection_noninferiority") or {}
    reasons = []
    deficit = det.get("auc_deficit")
    lim = t5.get("max_auc_deficit_vs_measured", 0.02)
    if deficit is None:
        reasons.append("detection endpoint comparison unavailable")
    elif deficit > lim:
        reasons.append(
            f"AF detection from the reconstruction loses "
            f"{deficit:.3f} AUC vs the measured path "
            f"(> {lim}): endpoints stay on the measured signal")
    row("g5", not reasons, reasons, det, t5)

    all_green = all(r["status"] == "GREEN" for r in rows)
    sign = gcfg.get("signoff") or {}
    signed = bool(sign.get("owner_confirmed")) and \
        sign.get("clinical_advisor") not in (None, "", "null")
    return {"gates": rows, "all_gates_green": all_green,
            "clinical_signoff": signed,
            "promotion_open": bool(all_green and signed)}


# ------------------------------------------------- scoreboard
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


def gate_status(*, runs_root=None, gates_path=None) -> dict:
    """The current scoreboard: latest run's evidence (if any) judged
    against the pre-registered gates. Operator-scored blinded reads, when
    they exist, live at <runs_root>/blinded_reads.json (the output of
    fidelity.score_blinded_reads)."""
    gcfg = load_gates(gates_path)
    entry = latest_scoreboard_entry(runs_root)
    evidence = (entry or {}).get("evidence")
    reads_p = pathlib.Path(runs_root or DEFAULT_RUNS) / "blinded_reads.json"
    if evidence is not None and reads_p.exists():
        with open(reads_p) as f:
            evidence = dict(evidence, blinded_reads=json.load(f))
    verdict = evaluate_gates(gcfg, evidence or {})
    return {"WATERMARK": WATERMARK,
            "gates_version": gcfg.get("gates_version"),
            "requires_clinical_signoff":
                bool(gcfg.get("requires_clinical_signoff", True)),
            "signoff": gcfg.get("signoff"),
            "evidence_run": (entry or {}).get("run_id"),
            "evidence_data": ((entry or {}).get("evidence") or {}).get("data"),
            **verdict,
            "promotion": "OPEN" if verdict["promotion_open"] else "BLOCKED"}


def render_gate_status_html(doc: dict) -> str:
    rows = []
    for g in doc["gates"]:
        colour = "#166534" if g["status"] == "GREEN" else "#92400e"
        reasons = "".join(f"<li>{r}</li>" for r in g["reasons"])
        rows.append(
            f"<tr><td>{g['title']}</td>"
            f"<td style='color:{colour};font-weight:bold'>{g['status']}</td>"
            f"<td><ul>{reasons or '<li>pass</li>'}</ul>"
            f"<pre>{json.dumps(g['metrics'], indent=1)}</pre></td></tr>")
    return ("<!doctype html><meta charset='utf-8'>"
            f"<div style='background:#b45309;color:#fff;padding:10px;"
            f"font:bold 14px sans-serif'>{WATERMARK}</div>"
            "<h2 style='font-family:sans-serif'>ECG-reconstruction "
            "promotion scoreboard (§G)</h2>"
            f"<p style='font-family:sans-serif'>gates {doc['gates_version']}"
            f" · evidence run: {doc.get('evidence_run') or 'none'} · "
            f"promotion: <b>{doc['promotion']}</b> · clinical signoff: "
            f"{'yes' if doc['clinical_signoff'] else 'PENDING'}</p>"
            "<table border='1' cellspacing='0' cellpadding='6' "
            "style='font:12px sans-serif;border-collapse:collapse'>"
            "<tr><th>gate</th><th>status</th><th>evidence</th></tr>"
            + "".join(rows) + "</table>"
            f"<div style='background:#b45309;color:#fff;padding:10px;"
            f"font:bold 14px sans-serif'>{WATERMARK}</div>")
