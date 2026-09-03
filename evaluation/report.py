"""
Evaluation report (T7): dataset directory -> Gate-1/1b + abstention +
leakage report, as JSON and markdown.

Layout expected in <dataset_dir>: for each recording id R the triple
  R.recording.json   (datasets.schema.Recording, strict loader)
  <video>            (path in the manifest, relative to the directory)
  <ecg>              (idem; R-peaks also live in the manifest)

RULES ENCODED HERE rather than left to discipline:
  * a recording failing the schema gate is EXCLUDED and listed with its
    reasons — it is never a data point, and never silently dropped;
  * every beat metric comes from `inference.pipeline.run_with_details`,
    i.e. THE production path (lesson P2);
  * beat metrics are reported PER RHYTHM ARM — pulse deficit hides in
    aggregates;
  * abstention (no-read) parity is audited across every available axis
    INCLUDING rhythm;
  * any serial-confirmation figure quotes its fp_persistent_share;
  * raw accuracy appears nowhere.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np

from configs import load_config, config_hash
from datasets.io import load_recording
from datasets.schema import Recording, Rhythm, ScanOutcome
from datasets.synchronization import ecg_to_video_clock
from datasets.splits import assert_no_leakage, split_composition, LeakageError
from beats.detector import flag_suspected_missed_beats
from beats.ibi import rmssd_from_runs
from evaluation.beat_metrics import (match_beats, ibi_agreement,
                                     GATE1_CONTROLLED, GATE1_ARRHYTHMIA,
                                     production_rmssd_error_ms,
                                     beat_confidence_calibration,
                                     missed_beat_flag_recall)
from evaluation.afib_metrics import (evaluate_binary, risk_coverage_curve,
                                     serial_confirmation, no_read_report)
from inference.pipeline import run_with_details

ECE_GATE = 0.10
FLAG_RECALL_GATE = 0.60
PRODUCTION_RMSSD_GATE_MS = 25.0


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _eval_one(rec: Recording, video: pathlib.Path) -> dict:
    result, det = run_with_details(
        str(video), manifest={"illuminance_lux":
                              rec.capture.illuminance_lux_mean},
        recording_id=rec.recording_id)
    row = {
        "recording_id": rec.recording_id,
        "participant_id": rec.participant_id,
        "arm": rec.protocol_id,
        "rhythm": "AF" if rec.recording_level_rhythm() is Rhythm.AFIB
        else "non-AF",
        "fps": rec.capture.measured_fps_mean or rec.capture.nominal_fps,
        "outcome": result.outcome.value,
        "predicted_class": result.predicted_class,
        "afib_probability": result.afib_probability,
        "sqi": result.signal_quality_index,
        "usable_beats": result.usable_beats,
        "no_read_reasons": result.no_read_reasons,
        "provenance": {"model_version": result.model_version,
                       "code_commit": result.code_commit,
                       "calibration_version": result.calibration_version,
                       "config_hash": result.config_hash},
    }
    if "series" not in det:
        return row

    # ECG onto the video clock via the sync record — the ONLY sanctioned way
    # (full model incl. drift; offset alone loses drift_ppm*t)
    ref = ecg_to_video_clock(np.asarray(rec.ecg_rpeaks_s, float), rec.sync)
    series = det["series"]                      # calibrated fused series
    m50 = match_beats(ref, series.times(), tolerance_ms=50)
    m100 = match_beats(ref, series.times(), tolerance_ms=100)
    ibi = ibi_agreement(ref, series.times(), m50)
    gate = GATE1_ARRHYTHMIA if row["rhythm"] == "AF" else GATE1_CONTROLLED
    ok, detail = gate.evaluate(m50, ibi)

    prod = production_rmssd_error_ms(ref, rmssd_from_runs(det["runset"]))
    ece = beat_confidence_calibration(m100, series.confidences())["ece"]
    flags = flag_suspected_missed_beats(series)
    recall = missed_beat_flag_recall(ref, series.times(), m100, flags)

    row.update({
        "f1_at_50ms": m50.f1, "sensitivity_50ms": m50.sensitivity,
        "ppv_50ms": m50.ppv, "ibi_mae_ms": ibi.ibi_mae_ms,
        "missed_beat_rate": m50.missed_beat_rate,
        "false_beat_rate": m50.false_beat_rate,
        "production_rmssd_error_ms": prod["error_ms"],
        "confidence_ece": ece,
        "missed_beat_flag_recall": recall["flag_recall"],
        "gate_name": gate.name, "gate_pass": bool(ok),
        "gate_detail": {k: {kk: (vv if not isinstance(vv, float)
                                 else float(vv))
                            for kk, vv in v.items()}
                        for k, v in detail.items()},
    })
    return row


def _arm_summary(rows: list) -> dict:
    return {
        "n_recordings": len(rows),
        "f1_at_50ms_mean": _mean([r.get("f1_at_50ms") for r in rows]),
        "ibi_mae_ms_mean": _mean([r.get("ibi_mae_ms") for r in rows]),
        "missed_beat_rate_mean": _mean([r.get("missed_beat_rate")
                                        for r in rows]),
        "false_beat_rate_mean": _mean([r.get("false_beat_rate")
                                       for r in rows]),
        "production_rmssd_error_ms_mean":
            _mean([r.get("production_rmssd_error_ms") for r in rows]),
        "confidence_ece_mean": _mean([r.get("confidence_ece") for r in rows]),
        "missed_beat_flag_recall_mean":
            _mean([r.get("missed_beat_flag_recall") for r in rows]),
    }


def evaluate_dataset(dataset_dir: str, out_dir: Optional[str] = None) -> dict:
    d = pathlib.Path(dataset_dir)
    manifests = sorted(d.glob("*.recording.json"))
    if not manifests:
        raise ValueError(f"no *.recording.json files in {d}")
    cfg = load_config()

    excluded, rows, included_recs = [], [], []
    for mp in manifests:
        rec = load_recording(str(mp))
        ok, why = rec.is_valid_for_beat_analysis()
        if not ok:
            excluded.append({"recording_id": rec.recording_id,
                             "reasons": why})
            continue
        video = d / rec.video_path
        if not video.exists():
            excluded.append({"recording_id": rec.recording_id,
                             "reasons": [f"video file missing: "
                                         f"{rec.video_path}"]})
            continue
        included_recs.append(rec)
        rows.append(_eval_one(rec, video))

    analysed = [r for r in rows if r["outcome"] == ScanOutcome.ACCEPT.value]

    # ---- classification (no raw accuracy, ever)
    y = np.array([1 if r["rhythm"] == "AF" else 0 for r in rows])
    pred = np.array([1 if r.get("predicted_class") == "AFIB_SUGGESTIVE" else 0
                     for r in rows])
    noread = np.array([r["outcome"] != ScanOutcome.ACCEPT.value for r in rows])
    pids = np.array([r["participant_id"] for r in rows])
    cls = evaluate_binary(y, pred, None, noread, pids, n_boot=500)
    cls_summary = cls.summary()

    # ---- risk-coverage over analysed scans, quality = SQI
    rc = []
    if analysed:
        rc = risk_coverage_curve(
            np.array([1 if r["rhythm"] == "AF" else 0 for r in analysed]),
            np.array([1 if r.get("predicted_class") == "AFIB_SUGGESTIVE"
                      else 0 for r in analysed]),
            np.array([r.get("sqi") or 0.0 for r in analysed]))

    # ---- abstention parity, every available axis INCLUDING rhythm
    groups = {"rhythm": [r["rhythm"] for r in rows],
              "arm": [r["arm"] for r in rows],
              "fps": [str(int(r["fps"])) for r in rows]}
    nr_ok, nr_rep = no_read_report(noread, groups)

    # ---- leakage audit (never silently absent)
    try:
        audit = {"pass": True, "report": assert_no_leakage(included_recs)}
    except LeakageError as e:
        audit = {"pass": False, "error": str(e)}

    # ---- serial confirmation, share ALWAYS quoted
    share = float(cfg["serial"]["fp_persistent_share_planning"])
    k, n = cfg["serial"]["rule"]
    se, sp = cls.sensitivity, cls.specificity
    if np.isfinite(se) and np.isfinite(sp):
        se_k, sp_k = serial_confirmation(se, sp, int(k), int(n),
                                         fp_persistent_share=share)
    else:
        se_k = sp_k = float("nan")
    serial = {"rule": f"{k}-of-{n}", "fp_persistent_share": share,
              "single_scan": {"sensitivity": se, "specificity": sp},
              "serial": {"sensitivity": se_k, "specificity": sp_k},
              "note": ("independence-model figures are upper bounds; the "
                       "quoted fp_persistent_share discounts persistent-"
                       "cause false positives which serial confirmation "
                       "cannot suppress")}

    af_rows = [r for r in rows if r["rhythm"] == "AF"]
    naf_rows = [r for r in rows if r["rhythm"] == "non-AF"]
    prov0 = rows[0]["provenance"] if rows else {}
    doc = {
        "provenance": {
            **prov0,
            "config_hash_resolved": config_hash(cfg),
            "dataset_dir": str(d), "n_recordings": len(manifests),
            "n_included": len(rows), "n_excluded": len(excluded),
            "synthetic_notice": ("all numbers in this report derive from "
                                 "synthetic recordings; none is a clinical "
                                 "performance claim"),
        },
        "excluded": excluded,
        "per_recording": rows,
        "beat_metrics_by_rhythm": {"AF": _arm_summary(af_rows),
                                   "non-AF": _arm_summary(naf_rows)},
        "gate1_non_af": {
            "gate": GATE1_CONTROLLED.name,
            "per_recording_pass": [{"recording_id": r["recording_id"],
                                    "pass": r.get("gate_pass")}
                                   for r in naf_rows],
            "extra_metrics": {
                "production_rmssd_error_ms": {
                    "mean": _mean([r.get("production_rmssd_error_ms")
                                   for r in naf_rows]),
                    "gate_abs_ms": PRODUCTION_RMSSD_GATE_MS},
                "confidence_ece": {
                    "mean": _mean([r.get("confidence_ece")
                                   for r in naf_rows]),
                    "gate": ECE_GATE}},
        },
        "gate1b_af": {
            "gate": GATE1_ARRHYTHMIA.name,
            "per_recording_pass": [{"recording_id": r["recording_id"],
                                    "pass": r.get("gate_pass")}
                                   for r in af_rows],
            "extra_metrics": {
                "missed_beat_flag_recall": {
                    "mean": _mean([r.get("missed_beat_flag_recall")
                                   for r in af_rows]),
                    "gate": FLAG_RECALL_GATE},
                "confidence_ece": {
                    "mean": _mean([r.get("confidence_ece")
                                   for r in af_rows]),
                    "gate": ECE_GATE}},
        },
        "classification": cls_summary,
        "risk_coverage": rc,
        "no_read_parity": {"pass": bool(nr_ok), **nr_rep},
        "leakage_audit": audit,
        "split_composition": split_composition(included_recs),
        "serial_confirmation": serial,
    }

    out = pathlib.Path(out_dir) if out_dir else d / "report"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "evaluation_report.json"
    with open(jp, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    mp_ = out / "evaluation_report.md"
    mp_.write_text(_render_markdown(doc))
    return {"report_json": str(jp), "report_md": str(mp_), "report": doc}


# ------------------------------------------------------------- markdown
def _f(x, nd=3):
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "NaN" if not np.isfinite(x) else f"{x:.{nd}f}"


def _render_markdown(doc: dict) -> str:
    L: list[str] = []
    p = doc["provenance"]
    L += ["# AvatarX AFib v0.1 — evaluation report", "",
          f"> {p['synthetic_notice']}", "",
          "## Provenance", "",
          f"- dataset: `{p['dataset_dir']}` — {p['n_included']} analysed, "
          f"{p['n_excluded']} EXCLUDED of {p['n_recordings']}",
          f"- code commit `{p.get('code_commit')}` · model "
          f"`{p.get('model_version')}` · calibration "
          f"`{p.get('calibration_version')}` · config "
          f"`{p.get('config_hash')}`", ""]

    L += ["## Excluded recordings (schema gate)", ""]
    if doc["excluded"]:
        for e in doc["excluded"]:
            L.append(f"- **{e['recording_id']}** — EXCLUDED: "
                     + "; ".join(e["reasons"]))
    else:
        L.append("- none")
    L.append("")

    L += ["## Beat metrics by rhythm (production path)", "",
          "| arm | n | F1@50ms | IBI MAE ms | missed | false | prod-RMSSD "
          "err ms | ECE |", "|---|---|---|---|---|---|---|---|"]
    for arm, s in doc["beat_metrics_by_rhythm"].items():
        L.append(f"| {arm} | {s['n_recordings']} | "
                 f"{_f(s['f1_at_50ms_mean'])} | "
                 f"{_f(s['ibi_mae_ms_mean'], 1)} | "
                 f"{_f(s['missed_beat_rate_mean'])} | "
                 f"{_f(s['false_beat_rate_mean'])} | "
                 f"{_f(s['production_rmssd_error_ms_mean'], 1)} | "
                 f"{_f(s['confidence_ece_mean'])} |")
    L.append("")

    for key, title in (("gate1_non_af", "Gate 1 (non-AF arm)"),
                       ("gate1b_af", "Gate 1b (AF arm)")):
        g = doc[key]
        n_pass = sum(1 for r in g["per_recording_pass"] if r["pass"])
        L += [f"## {title}", "",
              f"- {g['gate']}: {n_pass}/{len(g['per_recording_pass'])} "
              "recordings pass"]
        for mname, m in g["extra_metrics"].items():
            gate_v = m.get("gate_abs_ms", m.get("gate"))
            L.append(f"- {mname}: mean {_f(m['mean'])} (gate {gate_v})")
        L.append("")

    c = doc["classification"]
    L += ["## Classification (interim rule; window = recording)", "",
          f"- analysed {c['n_analysed']}, no-read {c['n_no_read']} "
          f"({_f(c['no_read_rate'])}), prevalence "
          f"{_f(c['prevalence_in_analysed'])}"]
    for k in ("sensitivity", "specificity", "ppv", "npv"):
        v = c.get(k)
        if v:
            L.append(f"- {k}: {_f(v['value'])} CI95 "
                     f"[{_f(v['ci95'][0])}, {_f(v['ci95'][1])}]"
                     if v.get("ci95") else f"- {k}: {_f(v['value'])}")
    L += [f"- NOTE: {c['NOTE']}", ""]

    L += ["## Risk-coverage (quality = SQI; AF retention shown)", "",
          "| coverage | n | sens | spec | AF retained |", "|---|---|---|---|---|"]
    for r in doc["risk_coverage"]:
        L.append(f"| {_f(r['coverage'], 2)} | {r['n']} | "
                 f"{_f(r['sensitivity'])} | {_f(r['specificity'])} | "
                 f"{r['n_afib_retained']} |")
    L.append("")

    nr = doc["no_read_parity"]
    L += ["## No-read parity (abstention audit, incl. rhythm)", "",
          f"- gate: {'PASS' if nr['pass'] else 'FAIL'}"]
    for dim, levels in nr["levels"].items():
        line = ", ".join(f"{k}: {_f(v['no_read_rate'])} (n={v['n']})"
                         for k, v in levels.items())
        L.append(f"- {dim}: {line}")
    for f_ in nr.get("failures", []):
        L.append(f"- FAIL: {f_}")
    L.append("")

    s = doc["serial_confirmation"]
    L += ["## Serial confirmation", "",
          f"- rule {s['rule']} at fp_persistent_share="
          f"{s['fp_persistent_share']}: Se "
          f"{_f(s['serial']['sensitivity'])} / Sp "
          f"{_f(s['serial']['specificity'])} (single-scan Se "
          f"{_f(s['single_scan']['sensitivity'])} / Sp "
          f"{_f(s['single_scan']['specificity'])})",
          f"- {s['note']}", ""]

    a = doc["leakage_audit"]
    L += ["## Leakage audit", "",
          f"- {'PASS' if a['pass'] else 'FAIL: ' + a.get('error', '')}"]
    L += ["", "## Split composition", "",
          "| split | participants | recordings | AF participants | "
          "hard negatives |", "|---|---|---|---|---|"]
    for split, v in doc["split_composition"].items():
        L.append(f"| {split} | {v['participants']} | {v['recordings']} | "
                 f"{v['af_participants']} | "
                 f"{v['hard_negative_participants']} |")
    L.append("")
    return "\n".join(L)
