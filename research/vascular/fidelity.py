"""V0 signal-fidelity study (v0.4 vascular track, T1 — run BEFORE any
modeling).

For every recording with a synchronized contact-PPG reference: extract
the IDENTICAL morphology feature set from the facial video (through the
gated production path — V-c) and from the contact waveform, then report
per feature: facial<->contact agreement (ICC(2,1) absolute agreement +
Bland-Altman bias/LoA), same-day test-retest repeatability (recordings
sharing participant+session), and strata by Fitzpatrick group and
capture device. Features that fail the EXPLICIT thresholds in the
vascular block of configs/gates.yaml are dropped from all downstream
modeling automatically (evaluation.vascular_gates.surviving_features
reads this study's record — by config, never by hand-editing).

Everything written here is a research artifact: WATERMARK-first JSON,
runs under $AVATARX_VASCULAR_RUNS, never a consumer surface.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import numpy as np

from research.vascular import WATERMARK, VascularError
from research.vascular.features import FEATURE_NAMES

MIN_RETEST_PAIRS_FOR_ICC = 3
MIN_STRATUM_N = 3


def icc_2_1(mat) -> float | None:
    """ICC(2,1): two-way random effects, absolute agreement, single
    measurement (Shrout & Fleiss). `mat` is (n_subjects, k_raters);
    rows with any non-finite value are dropped. None when fewer than
    3 complete rows remain (an ICC on 2 subjects is numerology)."""
    m = np.asarray(mat, float)
    if m.ndim != 2 or m.shape[1] < 2:
        return None
    m = m[np.all(np.isfinite(m), axis=1)]
    n, k = m.shape
    if n < 3:
        return None
    grand = m.mean()
    row_m = m.mean(axis=1)
    col_m = m.mean(axis=0)
    ss_rows = k * float(np.sum((row_m - grand) ** 2))
    ss_cols = n * float(np.sum((col_m - grand) ** 2))
    ss_total = float(np.sum((m - grand) ** 2))
    ss_err = ss_total - ss_rows - ss_cols
    ms_r = ss_rows / (n - 1)
    ms_c = ss_cols / (k - 1)
    ms_e = ss_err / max((n - 1) * (k - 1), 1)
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    if abs(denom) < 1e-12:
        return None
    return float((ms_r - ms_e) / denom)


def _pairs(rows: list, key: str) -> np.ndarray:
    """(n_participants, 2) facial/contact pairs for one feature —
    PARTICIPANT-level: a participant's recordings (retest replicates
    included) are averaged per arm first, so the agreement ICC's
    subject count is subjects, not pseudo-replicated recordings
    (review finding: 2 participants x 2 scans read as n=4)."""
    by_pid: dict = {}
    for r in rows:
        f = r["facial"]["features"].get(key)
        c = r["contact"]["features"].get(key)
        by_pid.setdefault(r["participant_id"], []).append(
            [np.nan if f is None else float(f),
             np.nan if c is None else float(c)])
    out = []
    for pid in sorted(by_pid):
        m = np.asarray(by_pid[pid], float)
        with np.errstate(invalid="ignore"):
            out.append([float(np.nanmean(m[:, 0])),
                        float(np.nanmean(m[:, 1]))])
    return np.asarray(out, float) if out else np.empty((0, 2))


def _scan_time(r: dict):
    """Sortable acquisition time; None when absent/unparseable."""
    t = r.get("video_start_utc")
    return t if isinstance(t, str) and t.strip() else None


def _retest_matrix(rows: list, key: str) -> np.ndarray:
    """(n_pairs, 2) facial feature values for same participant+session
    scan pairs, ordered by ACQUISITION TIME (manifest video_start_utc) —
    lexicographic manifest order would scramble drift signs (review
    finding). Pairs whose scans lack distinct timestamps are dropped,
    not guessed; None-valued scans are filtered before pairing."""
    by_visit: dict = {}
    for r in rows:
        by_visit.setdefault((r["participant_id"], r["session_id"]),
                            []).append(r)
    out = []
    for scans in by_visit.values():
        if len(scans) < 2:
            continue
        times = [_scan_time(r) for r in scans]
        if any(t is None for t in times) or len(set(times)) < len(times):
            continue                       # no honest temporal order
        ordered = [r for _, r in sorted(zip(times, scans),
                                        key=lambda p: p[0])]
        vals = [r["facial"]["features"].get(key) for r in ordered]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            out.append([float(vals[0]), float(vals[1])])
    return np.asarray(out, float) if out else np.empty((0, 2))


def fidelity_study(dataset_dir, *, signal_domain: str = "synthetic",
                   runs_root=None, gates_path=None, out_dir=None,
                   config=None) -> dict:
    from datasets.io import load_recording
    from datasets.schema import contact_ppg_from_dict, vascular_from_dict
    from evaluation.fitness_metrics import bland_altman
    from evaluation.vascular_gates import (append_scoreboard,
                                           evaluate_vascular_gates,
                                           load_vascular_gates, _req,
                                           DEFAULT_RUNS)
    from research.vascular.features import (contact_session_features,
                                            facial_session_features)

    d = pathlib.Path(dataset_dir)
    manifests = sorted(d.glob("*.recording.json"))
    if not manifests:
        raise VascularError(f"{d} contains no *.recording.json manifests")
    rows, excluded = [], []
    for mp in manifests:
        rid = mp.name[:-len(".recording.json")]
        try:
            rec = load_recording(str(mp))
            mdict = json.loads(mp.read_text())   # pipeline takes the dict
        except (ValueError, KeyError) as e:
            excluded.append({"recording_id": rid,
                             "reasons": [f"invalid manifest: {e}"]})
            continue
        ppg_p = d / f"{rid}.ppg.json"
        if not ppg_p.exists():
            excluded.append({"recording_id": rid,
                             "reasons": ["no contact-PPG reference — a "
                                         "fidelity pair needs both "
                                         "arms"]})
            continue
        try:
            ppg = contact_ppg_from_dict(json.loads(ppg_p.read_text()))
        except ValueError as e:
            excluded.append({"recording_id": rid,
                             "reasons": [f"invalid contact PPG: {e}"]})
            continue
        contact = contact_session_features(ppg)
        if not contact.get("available"):
            excluded.append({"recording_id": rid,
                             "reasons": ["contact arm unreadable"]
                             + list(contact.get("reasons") or [])})
            continue
        facial = facial_session_features(
            str(d / rec.video_path), manifest=mdict, config=config)
        if not facial.get("available"):
            excluded.append({"recording_id": rid,
                             "reasons": list(facial.get("reasons") or [])})
            continue
        strata = {"fitzpatrick_group": None, "device_label": None}
        pwv_p = d / f"{rid}.pwv.json"
        if pwv_p.exists():
            try:
                ref = vascular_from_dict(json.loads(pwv_p.read_text()))
                strata["fitzpatrick_group"] = ref.fitzpatrick_group
                strata["device_label"] = ref.device_label
            except ValueError:
                pass
        rows.append({"recording_id": rid,
                     "participant_id": rec.participant_id,
                     "session_id": rec.session_id,
                     "video_start_utc": rec.video_start_utc,
                     "facial": facial, "contact": contact, **strata})

    if not rows:
        raise VascularError(
            "no usable fidelity pairs — every recording was excluded: "
            + "; ".join(f"{e['recording_id']}: {e['reasons'][0]}"
                        for e in excluded[:5]))
    gcfg = load_vascular_gates(gates_path)
    t0 = gcfg.get("v0_signal_fidelity") or {}
    icc_min = float(_req(t0, "feature_icc_min"))
    rt_min = float(_req(t0, "retest_icc_min"))
    def _feature_table(subset: list) -> dict:
        table: dict = {}
        for name in FEATURE_NAMES:
            pairs = _pairs(subset, name)
            finite = pairs[np.all(np.isfinite(pairs), axis=1)] \
                if pairs.size else pairs
            entry: dict = {"n_participants": int(finite.shape[0]),
                           "n_recordings": len(subset),
                           "icc": icc_2_1(pairs)}
            if finite.shape[0] >= 3:
                entry.update(bland_altman(finite[:, 1], finite[:, 0]))
            rt = _retest_matrix(subset, name)
            entry["n_retest_pairs"] = int(rt.shape[0])
            entry["retest_icc"] = (
                icc_2_1(rt) if rt.shape[0] >= MIN_RETEST_PAIRS_FOR_ICC
                else None)
            table[name] = entry
        return table

    per_feature = _feature_table(rows)
    for name, entry in per_feature.items():
        entry["survives"] = bool(
            entry["icc"] is not None and entry["icc"] >= icc_min
            and entry["retest_icc"] is not None
            and entry["retest_icc"] >= rt_min)
    surviving = [n for n in FEATURE_NAMES if per_feature[n]["survives"]]

    # strata: the SAME per-feature agreement machinery per group —
    # coverage-only note for groups too small to support a statistic
    strata_out: dict = {}
    for axis in ("fitzpatrick_group", "device_label"):
        groups: dict = {}
        for r in rows:
            groups.setdefault(r.get(axis), []).append(r)
        table = {}
        for g, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
            n_pids = len({m["participant_id"] for m in members})
            cell: dict = {"n_recordings": len(members),
                          "n_participants": n_pids}
            if n_pids < MIN_STRATUM_N:
                cell["note"] = (f"under {MIN_STRATUM_N} participants — "
                                "reported for coverage, no agreement "
                                "statistic")
            else:
                cell["per_feature"] = _feature_table(members)
            table[str(g)] = cell
        strata_out[axis] = table

    pids = {r["participant_id"] for r in rows}
    evidence = {
        "fidelity_data": {"signal_domain": signal_domain,
                          "participant_disjoint": True,
                          "session_disjoint": True,
                          "production_path":
                              signal_domain == "facial_rppg",
                          "n_recordings": len(rows),
                          "dataset_dir": str(d)},
        "v0": {"n_paired_participants": len(pids),
               "per_feature": per_feature,
               "surviving": surviving,
               "strata": strata_out},
    }
    verdict = evaluate_vascular_gates(gcfg, evidence)

    run_id = "fid-" + hashlib.sha256(
        (str(d) + signal_domain
         + ",".join(sorted(r["recording_id"] for r in rows))
         ).encode()).hexdigest()[:12]
    run_dir = pathlib.Path(runs_root or DEFAULT_RUNS) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"WATERMARK": WATERMARK, "run_id": run_id,
              "kind": "fidelity", "track": "vascular_stiffness",
              "n_recordings": len(rows), "n_excluded": len(excluded),
              "n_paired_participants": len(pids),
              "signal_domain": signal_domain,
              "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (run_dir / "fidelity_record.json").write_text(
        json.dumps(record, indent=1))
    (run_dir / "gate_results.json").write_text(json.dumps(
        {"WATERMARK": WATERMARK, "run_id": run_id,
         "evidence": evidence, **verdict}, indent=1, default=str))
    append_scoreboard({"kind": "fidelity", "run_id": run_id,
                       "evaluated_at": record["evaluated_at"],
                       "evidence": {"v0": evidence["v0"],
                                    "data": evidence["fidelity_data"]},
                       "surviving": surviving,
                       "all_gates_green": verdict["all_gates_green"],
                       "promotion_open": verdict["promotion_open"]},
                      runs_root=runs_root)

    doc = {"WATERMARK": WATERMARK, "run_id": run_id,
           "signal_domain": signal_domain,
           "n_recordings": len(rows), "excluded": excluded,
           "per_feature": per_feature, "surviving_features": surviving,
           "strata": strata_out,
           "gates": {"verdict": verdict,
                     "promotion": ("OPEN" if verdict["promotion_open"]
                                   else "BLOCKED")},
           "note": ("surrogate-domain studies exercise the machinery "
                    "and can never open a gate"
                    if signal_domain != "facial_rppg" else None)}
    out = pathlib.Path(out_dir or (d / "report_vascular_fidelity"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "fidelity_report.json").write_text(
        json.dumps(doc, indent=2, default=str))
    (out / "fidelity_report.md").write_text(_render_markdown(doc))
    return {"report_json": str(out / "fidelity_report.json"),
            "report_md": str(out / "fidelity_report.md"),
            "run_dir": str(run_dir), "report": doc}


def _render_markdown(doc: dict) -> str:
    lines = [f"> {WATERMARK}", "",
             f"# Vascular signal-fidelity study — run {doc['run_id']}",
             "",
             f"domain: `{doc['signal_domain']}` · recordings: "
             f"{doc['n_recordings']} · excluded: {len(doc['excluded'])}",
             "", "| feature | ICC | bias | LoA | retest ICC | survives |",
             "|---|---|---|---|---|---|"]
    for name, e in doc["per_feature"].items():
        loa = ("" if e.get("loa_lower") is None
               else f"[{e['loa_lower']}, {e['loa_upper']}]")
        lines.append(
            f"| {name} | {e.get('icc')} | {e.get('bias', '')} | {loa} "
            f"| {e.get('retest_icc')} | "
            f"{'YES' if e['survives'] else 'no'} |")
    lines += ["", f"surviving features: "
              f"{', '.join(doc['surviving_features']) or 'NONE'}",
              "", f"promotion: **{doc['gates']['promotion']}**", "",
              f"> {WATERMARK}"]
    return "\n".join(lines)
