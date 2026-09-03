"""Vasotone evaluation harness (v0.5, T3/T4): the two null arms, the
provocation study, and the T2 baseline battery.

Per recording: production pipeline → tone_session_features (V-c/W-c/W-d
enforced there) → segregated by maneuver. The null arms define the
track's signature defenses: null_rest gives every feature its
natural-drift distribution; null_optics gives its lamp-response
distribution; the explicit gates.yaml thresholds turn those into the
surviving set (evaluation.vasotone_gates.survivors_from_study — one
rule, shared with the scoreboard path). Provocations are then scored
against the contact-PI reference (W0), the HR/respiration/motion
baseline battery (W2 — a "tone" response that an HR+respiration model
explains is not a tone response), retest/graded consistency (W3), and
Fitzpatrick parity (W4).

The facial PRIMARY reactivity response is the delta_norm of
norm_pulse_amplitude — the perfusion-index analog — and exists only
when that feature survives the null arms. Everything is within-session
(W-c); uncontrolled-optics sessions are excluded from gate evidence
with their reason (W-d).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import numpy as np

from evaluation.fitness_metrics import _ridge, _split
from evaluation.vascular_metrics import _partial_r, _partial_r2

SYNTHETIC_NOTICE = ("SYNTHETIC / SURROGATE DATA — machinery evidence "
                    "only; never quote as performance")
PRIMARY_FEATURE = "norm_pulse_amplitude"
BASELINES = ("b1_hr_response", "b2_resp_response", "b3_hr_resp",
             "b4_motion_artifact")
MIN_COVARIATE_COVERAGE = 0.6       # W2 refuses when a baseline column is
                                   # mostly missing (mean-imputation
                                   # would silently erase that baseline)


class VasotoneHarnessError(ValueError):
    pass


def _p95(vals: list):
    v = [abs(float(x)) for x in vals if x is not None]
    if len(v) < 2:
        return None
    return round(float(np.percentile(v, 95)), 5)


def null_arm_study(null_optics_rows: list, null_rest_rows: list,
                   feature_names) -> dict:
    """W1 evidence: per-feature |delta_norm| p95 under deliberate optics
    perturbation vs under quiet rest, with per-feature VALUE counts —
    a p95 resting on 2 values must not ride on 20-session arm counts
    (review finding) — and per-Fitzpatrick natural-drift floors for the
    W4 parity scoring."""
    per = {}
    for name in feature_names:
        opt_vals = [r["features"][name]["delta_norm"]
                    for r in null_optics_rows]
        rest_vals = [r["features"][name]["delta_norm"]
                     for r in null_rest_rows]
        per[name] = {
            "null_optics_p95": _p95(opt_vals),
            "null_rest_p95": _p95(rest_vals),
            "n_null_optics_values": sum(v is not None
                                        for v in opt_vals),
            "n_null_rest_values": sum(v is not None
                                      for v in rest_vals),
        }
    by_group: dict = {}
    for r in null_rest_rows:
        g = r.get("fitzpatrick_group")
        if g is not None:
            by_group.setdefault(str(g), []).append(
                r["features"].get(PRIMARY_FEATURE, {}).get("delta_norm"))
    return {"n_null_optics_sessions": len(null_optics_rows),
            "n_null_rest_sessions": len(null_rest_rows),
            "per_feature": per,
            "per_group_null_rest_p95": {g: _p95(v)
                                        for g, v in by_group.items()}}


def _hr_in_window(beat_t, span):
    t0, t1 = float(span[0]), float(span[1])
    bt = [t for t in beat_t if t0 <= t < t1]
    if len(bt) < 4:
        return None
    ibis = np.diff(np.asarray(bt))
    ibis = ibis[(ibis > 0.3) & (ibis < 2.0)]
    if ibis.size < 3:
        return None
    return float(60.0 / np.median(ibis))


def _torso_series(video_path: str):
    """(t, brightness) of the torso band (rows below 0.55) — an
    INDEPENDENT physical channel for the respiration baseline. The
    facial green trace also builds the outcome feature, so deriving B2
    from it was circular (review finding)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t, y = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h = frame.shape[0]
        band = frame[int(0.55 * h):, :, 1]
        y.append(float(np.mean(band)))
        t.append(i / fps)
        i += 1
    cap.release()
    return np.asarray(t, float), np.asarray(y, float), float(fps)


def covariate_responses(det, provocation, video_path=None) -> dict:
    """Per-session confound deltas for the baseline battery: HR response
    (B1), respiration response (B2, from the TORSO band of a second
    decode — never the facial trace the outcome is built on), and a
    motion-artifact response (B4, cardiac-band-stopped so it cannot
    smuggle the pulse amplitude) — each a within-session delta."""
    from preprocessing.roi import ROI_NAMES
    from rppg._filters import bandpass
    from rppg.respiration import respiratory_rate_from_motion
    lattice = det["lattice"]
    ing = det["ingest"]
    ts = np.asarray(ing.timestamps_s, float)
    fps = float(ing.meta.measured_fps_mean)
    beat_t = list(np.asarray(lattice.beat_t_s, float))
    base = provocation.phase_marks["baseline"]
    resp = provocation.phase_marks["stimulus"]
    hr_b, hr_r = _hr_in_window(beat_t, base), _hr_in_window(beat_t, resp)
    green = np.mean(np.stack(
        [np.asarray(ing.traces[r][:, 1], float) for r in ROI_NAMES]), 0)
    # motion proxy with the cardiac band REMOVED (review finding: raw
    # mean|diff| of the facial trace is a pulse-amplitude proxy)
    cardiac = bandpass(green - float(np.mean(green)), fps, 0.7,
                       min(4.0, 0.45 * fps))
    motion_sig = (green - float(np.mean(green))) - cardiac

    rr_b = rr_r = None
    if video_path is not None:
        try:
            tt, ty, tfps = _torso_series(video_path)

            def rr_in(span):
                m = (tt >= float(span[0])) & (tt < float(span[1]))
                if int(m.sum()) < 2 or \
                        float(tt[m][-1] - tt[m][0]) < 20.0:
                    return None
                rate, _ = respiratory_rate_from_motion(tt[m], ty[m],
                                                       tfps)
                return rate

            rr_b, rr_r = rr_in(base), rr_in(resp)
        except Exception:
            rr_b = rr_r = None

    def motion_in(span):
        m = (ts >= float(span[0])) & (ts < float(span[1]))
        if int(m.sum()) < 10:
            return None
        return float(np.mean(np.abs(np.diff(motion_sig[m]))))

    mo_b, mo_r = motion_in(base), motion_in(resp)
    return {
        "dhr_bpm": (None if hr_b is None or hr_r is None
                    else round(hr_r - hr_b, 3)),
        "dresp_brpm": (None if rr_b is None or rr_r is None
                       else round(rr_r - rr_b, 3)),
        "dmotion": (None if mo_b is None or mo_r is None or mo_b <= 0
                    else round((mo_r - mo_b) / mo_b, 5)),
    }


def _baseline_vec(name, r):
    c = r["covariates"]
    if name == "b1_hr_response":
        return [c.get("dhr_bpm")]
    if name == "b2_resp_response":
        return [c.get("dresp_brpm")]
    if name == "b3_hr_resp":
        return [c.get("dhr_bpm"), c.get("dresp_brpm")]
    if name == "b4_motion_artifact":
        return [c.get("dmotion")]
    raise VasotoneHarnessError(f"unknown baseline {name!r}")


def _matrix(rows, fn):
    return np.asarray([[np.nan if v is None else float(v)
                        for v in fn(r)] for r in rows], float)


MIN_MANEUVER_PAIRS_FOR_R = 5


def reference_tracking(prov_rows: list, detection_floor,
                       pi_floor: float) -> dict:
    """W0: facial primary response vs contact-PI response across
    provocations. Rows where either side is below its detection floor
    count as NON-agreement (conservative, fail closed). The magnitude
    correlation is WITHIN-MANEUVER (Fisher-z combined, weighted n-3) —
    a pooled r across maneuvers with different characteristic
    magnitudes would reward maneuver identity, not tracking (review
    finding)."""
    n = len(prov_rows)
    if n == 0:
        return {}
    agree = 0
    by_man: dict = {}
    floor = detection_floor if detection_floor is not None else np.inf
    for r in prov_rows:
        f = r.get("facial_response")
        p = (r.get("pi_response") or {}).get("delta_norm")
        if f is not None and p is not None:
            by_man.setdefault(r.get("maneuver"), []).append(
                (float(f), float(p)))
            if abs(f) >= floor and abs(p) >= pi_floor and \
                    np.sign(f) == np.sign(p):
                agree += 1
    out = {"n_provocations": n,
           "direction_agreement": round(agree / n, 3)}
    per_man = {}
    zs, ws = [], []
    for man, pairs in by_man.items():
        if len(pairs) < MIN_MANEUVER_PAIRS_FOR_R:
            per_man[man] = {"n": len(pairs),
                            "note": "below the per-maneuver floor"}
            continue
        fx = np.asarray([a for a, _ in pairs])
        px = np.asarray([b for _, b in pairs])
        if float(np.std(fx)) < 1e-9 or float(np.std(px)) < 1e-9:
            per_man[man] = {"n": len(pairs), "note": "degenerate"}
            continue
        r_m = float(np.clip(np.corrcoef(fx, px)[0, 1],
                            -0.999999, 0.999999))
        per_man[man] = {"n": len(pairs), "r": round(r_m, 3)}
        zs.append(np.arctanh(r_m) * (len(pairs) - 3))
        ws.append(len(pairs) - 3)
    out["per_maneuver_r"] = per_man
    if ws:
        out["magnitude_r"] = round(float(np.tanh(sum(zs) / sum(ws))), 3)
    return out


def battery(prov_rows: list, *, seed: int = 20260831,
            n_boot: int = 200) -> dict:
    """W2: does the facial response add information about the CONTACT
    reference's response beyond HR+respiration (B3)? Added information
    is a held-out partial-correlation statistic (no combined model);
    CIs are participant-cluster bootstrap with draw multiplicity."""
    rows = [r for r in prov_rows
            if (r.get("pi_response") or {}).get("delta_norm") is not None
            and r.get("facial_response") is not None]
    if len(rows) < 6:
        return {"available": False,
                "reason": f"only {len(rows)} scored provocations"}
    tr, te = _split(rows, seed)
    if len(tr) < 4 or len(te) < 2:
        return {"available": False,
                "reason": f"split too small (train {len(tr)}, "
                          f"test {len(te)})"}
    y_tr = np.asarray([r["pi_response"]["delta_norm"] for r in tr])
    y_te = np.asarray([r["pi_response"]["delta_norm"] for r in te])
    # fail closed on covariate starvation: a mostly-missing baseline
    # column would be silently mean-imputed away, handicapping the very
    # baseline the gate must beat (review finding)
    for name in BASELINES:
        X = _matrix(rows, lambda r, n=name: _baseline_vec(n, r))
        cov = float(np.mean(np.isfinite(X).all(axis=1)))
        if cov < MIN_COVARIATE_COVERAGE:
            return {"available": False,
                    "reason": f"covariate coverage for {name} is "
                              f"{cov:.2f} < {MIN_COVARIATE_COVERAGE} "
                              "— the baseline battery cannot be "
                              "honestly fielded"}
    preds = {}
    table = {}
    for name in BASELINES:
        p = _ridge(_matrix(tr, lambda r, n=name: _baseline_vec(n, r)),
                   y_tr,
                   _matrix(te, lambda r, n=name: _baseline_vec(n, r)))
        preds[name] = p
        table[name] = {"rmse": round(float(np.sqrt(np.mean(
            (p - y_te) ** 2))), 4)}
    facial = np.asarray([r["facial_response"] for r in te], float)
    added = _partial_r2(y_te, facial, preds["b3_hr_resp"])
    by_pid: dict = {}
    for i, r in enumerate(te):
        by_pid.setdefault(str(r["participant_id"]), []).append(i)
    pids = sorted(by_pid)
    ci = None
    if len(pids) >= 4:
        rng = np.random.default_rng(seed + 1)
        vals = []
        for _ in range(n_boot):
            draw = rng.choice(pids, size=len(pids), replace=True)
            idx = np.asarray([i for p in draw for i in by_pid[p]], int)
            if idx.size < 3:
                continue
            # SIGNED partial correlation: a CI over the square is
            # non-negative by construction and its exclude-zero test
            # does no statistical work (review finding, ~14%% null
            # false-pass at the config-minimum cohort)
            v = _partial_r(y_te[idx], facial[idx],
                           preds["b3_hr_resp"][idx])
            if np.isfinite(v):
                vals.append(v)
        if len(vals) >= 10:
            ci = [round(float(np.percentile(vals, 2.5)), 4),
                  round(float(np.percentile(vals, 97.5)), 4)]
    return {"available": True, "models": table,
            "added_r2_over_b3": round(float(added), 4),
            "added_r_ci95": ci,
            "n_test_participants": len(pids),
            "n_train": len(tr), "n_test": len(te)}


def _timestamp(r):
    v = r.get("video_start_utc")
    return v if isinstance(v, str) and v.strip() else None


def dose_consistency(prov_rows: list) -> dict:
    """W3: retest ICC — LIKE-FOR-LIKE pairs (same participant, maneuver
    AND intensity; different sessions ordered by acquisition time, pairs
    without distinct timestamps dropped, never guessed) with the ICC
    computed PER MANEUVER and the worst reported (pooling maneuvers
    with different characteristic magnitudes inflated the gated ICC —
    review finding); plus graded-intensity ordering."""
    from research.vascular.fidelity import icc_2_1
    by_key: dict = {}
    by_man_int: dict = {}
    for r in prov_rows:
        if r.get("facial_response") is None:
            continue
        by_key.setdefault((r["participant_id"], r["maneuver"],
                           r.get("intensity")), []).append(r)
        if r.get("intensity") is not None:
            by_man_int.setdefault(
                (r["participant_id"], r["maneuver"]), {}).setdefault(
                int(r["intensity"]), []).append(
                abs(float(r["facial_response"])))
    pairs_by_man: dict = {}
    for (pid, man, inten), scans in by_key.items():
        sessions: dict = {}
        for r in scans:
            sessions.setdefault(r["session_id"], []).append(r)
        if len(sessions) < 2:
            continue
        per_sess = []
        ok = True
        for sid, rows_ in sessions.items():
            stamps = [_timestamp(s) for s in rows_]
            if any(s is None for s in stamps):
                ok = False
                break
            per_sess.append((min(stamps), float(np.median(
                [s["facial_response"] for s in rows_]))))
        if not ok:
            continue
        per_sess.sort()
        if per_sess[0][0] == per_sess[1][0]:
            continue                     # no honest temporal order
        pairs_by_man.setdefault(man, []).append(
            [per_sess[0][1], per_sess[1][1]])
    ordered_hits, ordered_total = 0, 0
    for by_int in by_man_int.values():
        levels = sorted(by_int)
        for a, b in zip(levels, levels[1:]):
            ordered_total += 1
            if float(np.median(by_int[b])) >= float(np.median(by_int[a])):
                ordered_hits += 1
    out: dict = {}
    if pairs_by_man:
        per_man = {}
        worst = None
        total = 0
        for man, pairs in pairs_by_man.items():
            total += len(pairs)
            icc = icc_2_1(np.asarray(pairs, float))
            per_man[man] = {"n_pairs": len(pairs), "icc": icc}
            if icc is not None:
                worst = icc if worst is None else min(worst, icc)
        out["n_retest_pairs"] = total
        out["per_maneuver_icc"] = per_man
        out["retest_icc"] = worst
    if ordered_total:
        out["ordered_fraction"] = round(ordered_hits / ordered_total, 3)
        out["n_ordered_comparisons"] = ordered_total
    return out


def fairness(all_scans: list, prov_rows: list, detection_floor,
             group_floors=None, min_group_participants: int = 1) -> dict:
    """W4: per-Fitzpatrick response-detection and coverage parity —
    PARTICIPANT-level rates (a prolific participant must not lift a
    group), each group scored against max(pooled floor, its OWN
    null_rest drift floor) so group-dependent noise cannot register as
    detections, and groups below the participant floor surface as
    None-rated (RED at the gate), never silently dropped (review
    findings)."""
    pooled = detection_floor if detection_floor is not None else np.inf
    gf = group_floors or {}
    cov_by_pid: dict = {}
    for s in all_scans:
        g = s.get("fitzpatrick_group")
        pid = s.get("participant_id") or s.get("recording_id")
        if g is None:
            continue
        cov_by_pid.setdefault(str(g), {}).setdefault(pid, []).append(
            bool(s.get("usable")))
    det_by_pid: dict = {}
    for r in prov_rows:
        g = r.get("fitzpatrick_group")
        if g is None or r.get("facial_response") is None:
            continue
        floor_g = pooled
        if gf.get(str(g)) is not None:
            floor_g = max(pooled, float(gf[str(g)]))
        det_by_pid.setdefault(str(g), {}).setdefault(
            r["participant_id"], []).append(
            abs(float(r["facial_response"])) >= floor_g)
    per = {}
    cov_rates, det_rates = [], []
    for g in sorted(set(cov_by_pid) | set(det_by_pid)):
        cell: dict = {}
        cpid = cov_by_pid.get(g, {})
        cell["n_participants"] = len(cpid)
        cell["attempted"] = sum(len(v) for v in cpid.values())
        if cpid and len(cpid) >= min_group_participants:
            rates = [float(np.mean(v)) for v in cpid.values()]
            cell["coverage"] = round(float(np.mean(rates)), 3)
            cov_rates.append(cell["coverage"])
        else:
            cell["coverage"] = None
        dpid = det_by_pid.get(g, {})
        cell["scored_participants"] = len(dpid)
        if dpid and len(dpid) >= min_group_participants and \
                (gf.get(g) is not None or not gf):
            rates = [float(np.mean(v)) for v in dpid.values()]
            cell["detection"] = round(float(np.mean(rates)), 3)
            det_rates.append(cell["detection"])
        else:
            cell["detection"] = None
        cell["group_drift_floor"] = gf.get(g)
        per[g] = cell
    axis = {"per_group": per}
    groups_rated_cov = [g for g in per if per[g]["coverage"] is not None]
    if len(groups_rated_cov) == len(per) and len(cov_rates) >= 2 \
            and max(cov_rates) > 0:
        axis["coverage_ratio_worst"] = round(
            min(cov_rates) / max(cov_rates), 3)
    groups_rated_det = [g for g in per
                        if per[g]["detection"] is not None]
    if len(groups_rated_det) == len(per) and len(det_rates) >= 2 \
            and max(det_rates) > 0:
        axis["detection_ratio_worst"] = round(
            min(det_rates) / max(det_rates), 3)
    return {"per_axis": {"fitzpatrick_group": axis}}


# ------------------------------------------------------------ harness
def evaluate_vasotone_dataset(dataset_dir, out_dir=None, *,
                              signal_domain: str = "synthetic",
                              runs_root=None, gates_path=None,
                              config=None, surviving_override=None
                              ) -> dict:
    from datasets.io import load_recording
    from datasets.schema import pi_from_dict, provocation_from_dict
    from configs import load_config, config_hash
    from evaluation.vasotone_gates import (DEFAULT_RUNS,
                                           append_scoreboard,
                                           evaluate_vasotone_gates,
                                           load_vasotone_gates,
                                           survivors_from_study)
    from research.vascular import WATERMARK
    from research.vascular.tone_features import (TONE_FEATURES,
                                                 pi_response,
                                                 tone_session_features)
    from inference.pipeline import run_with_details

    d = pathlib.Path(dataset_dir)
    manifests = sorted(d.glob("*.recording.json"))
    if not manifests:
        raise VasotoneHarnessError(
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
        pp = d / f"{rid}.provocation.json"
        if not pp.exists():
            excluded.append({"recording_id": rid,
                             "reasons": ["no provocation record — a "
                                         "tone session without its "
                                         "protocol is unusable"]})
            continue
        try:
            prov = provocation_from_dict(json.loads(pp.read_text()))
        except ValueError as e:
            excluded.append({"recording_id": rid,
                             "reasons": [f"invalid provocation: {e}"]})
            continue
        scan = {"recording_id": rid,
                "participant_id": rec.participant_id,
                "fitzpatrick_group": prov.fitzpatrick_group,
                "usable": False}
        result, det = run_with_details(str(d / rec.video_path),
                                       manifest=mdict, config=config)
        tf = tone_session_features(result, det, prov,
                                   capture=mdict.get("capture") or {})
        if tf.get("uncontrolled_optics"):
            all_scans.append(scan)
            excluded.append({"recording_id": rid,
                             "reasons": ["W-d: uncontrolled optics — "
                                         "excluded from gate "
                                         "evaluations"]})
            continue
        if not tf.get("available"):
            all_scans.append(scan)
            excluded.append({"recording_id": rid,
                             "reasons": list(tf.get("reasons") or [])})
            continue
        scan["usable"] = True
        all_scans.append(scan)
        row = {"recording_id": rid,
               "participant_id": rec.participant_id,
               "session_id": rec.session_id,
               "video_start_utc": rec.video_start_utc,
               "maneuver": prov.maneuver,
               "intensity": prov.intensity,
               "fitzpatrick_group": prov.fitzpatrick_group,
               "features": tf["features"],
               "covariates": covariate_responses(
                   det, prov, video_path=str(d / rec.video_path))}
        if prov.maneuver not in ("null_optics", "null_rest"):
            pip = d / f"{rid}.pi.json"
            if not pip.exists():
                excluded.append({"recording_id": rid,
                                 "reasons": ["no contact-PI reference "
                                             "for a provocation arm — "
                                             "W0 cannot score it"]})
                continue
            try:
                pi = pi_from_dict(json.loads(pip.read_text()))
            except ValueError as e:
                excluded.append({"recording_id": rid,
                                 "reasons": [f"invalid PI trace: {e}"]})
                continue
            row["pi_response"] = pi_response(
                pi, prov.phase_marks["baseline"],
                prov.phase_marks["stimulus"])
        rows.append(row)
    if not rows:
        raise VasotoneHarnessError(
            "no usable tone sessions — every recording was excluded: "
            + "; ".join(f"{e['recording_id']}: {e['reasons'][0]}"
                        for e in excluded[:5]))

    null_optics_rows = [r for r in rows if r["maneuver"] == "null_optics"]
    null_rest_rows = [r for r in rows if r["maneuver"] == "null_rest"]
    prov_rows = [r for r in rows
                 if r["maneuver"] not in ("null_optics", "null_rest")]
    gcfg = load_vasotone_gates(gates_path)
    from evaluation.vasotone_gates import _req as _wreq
    t0cfg = gcfg.get("w0_reference_tracking") or {}
    t4cfg = gcfg.get("w4_fairness") or {}
    pi_floor = float(_wreq(t0cfg, "pi_detection_floor_delta_norm"))
    min_group = int(_wreq(t4cfg, "min_group_participants"))
    w1 = null_arm_study(null_optics_rows, null_rest_rows, TONE_FEATURES)
    surviving = (list(surviving_override)
                 if surviving_override is not None
                 else survivors_from_study(w1, gcfg))
    # the primary facial response exists only when the amplitude analog
    # survives the null arms (fail closed — never an optics detector)
    floor = None
    if PRIMARY_FEATURE in surviving:
        floor = (w1["per_feature"].get(PRIMARY_FEATURE)
                 or {}).get("null_rest_p95")
        for r in prov_rows:
            r["facial_response"] = \
                r["features"][PRIMARY_FEATURE]["delta_norm"]
    else:
        for r in prov_rows:
            r["facial_response"] = None
    w0 = reference_tracking(prov_rows, floor, pi_floor)
    w2 = battery(prov_rows)
    w2_ev = ({k: w2[k] for k in ("added_r2_over_b3", "added_r_ci95",
                                 "n_test_participants")}
             if w2.get("available") else {})
    w3 = dose_consistency(prov_rows)
    w4 = fairness(all_scans, prov_rows, floor,
                  group_floors=w1.get("per_group_null_rest_p95"),
                  min_group_participants=min_group)
    # disjointness flags are COMPUTED, not asserted (review finding):
    # the within-subject design shares participants across arms by
    # construction; the disjointness the gates require is carried by
    # the W2 battery's held-out participant split, which only exists
    # when the battery actually ran
    disjoint = bool(w2.get("available"))
    evidence = {"data": {"signal_domain": signal_domain,
                         "participant_disjoint": disjoint,
                         "session_disjoint": disjoint,
                         "production_path":
                             signal_domain == "facial_rppg",
                         "n_recordings": len(rows),
                         "dataset_dir": str(d)},
                "w0": w0, "w1": w1, "w2": w2_ev, "w3": w3, "w4": w4,
                "w5": {}}
    verdict = evaluate_vasotone_gates(gcfg, evidence)

    run_id = "vaso-" + hashlib.sha256(
        (str(d) + signal_domain
         + ",".join(sorted(r["recording_id"] for r in rows))
         ).encode()).hexdigest()[:12]
    run_dir = pathlib.Path(runs_root or DEFAULT_RUNS) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "vasotone_record.json").write_text(json.dumps(
        {"WATERMARK": WATERMARK, "run_id": run_id,
         "track": "vasomotor_reactivity", "kind": "evaluation",
         "n_recordings": len(rows), "n_excluded": len(excluded),
         "signal_domain": signal_domain,
         "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    (run_dir / "gate_results.json").write_text(json.dumps(
        {"WATERMARK": WATERMARK, "run_id": run_id,
         "evidence": evidence, **verdict}, indent=1, default=str))
    (run_dir / "model.json").write_text(json.dumps(
        {"WATERMARK": WATERMARK, "kind": "reactivity_primary",
         "primary_feature": PRIMARY_FEATURE,
         "surviving_features": surviving,
         "detection_floor_delta_norm": floor,
         "n_provocations": len(prov_rows)}, indent=1))
    append_scoreboard({"kind": "evaluation", "run_id": run_id,
                       "evaluated_at":
                           time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "evidence": evidence,
                       "surviving": surviving,
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
           "excluded": excluded, "n_included": len(rows),
           "arms": {"provocations": len(prov_rows),
                    "null_optics": len(null_optics_rows),
                    "null_rest": len(null_rest_rows)},
           "surviving_features": surviving,
           "null_arm_study": w1,
           "reference_tracking": w0,
           "battery": w2, "dose_consistency": w3, "fairness": w4,
           "gates": {"verdict": verdict,
                     "promotion": ("OPEN" if verdict["promotion_open"]
                                   else "BLOCKED")}}
    out = pathlib.Path(out_dir or (d / "report_vasotone"))
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
             f"# Vasotone evaluation — run {doc['run_id']}", ""]
    if doc["provenance"].get("notice"):
        lines += [f"**{doc['provenance']['notice']}**", ""]
    arms = doc["arms"]
    lines += [f"arms: {arms['provocations']} provocations · "
              f"{arms['null_optics']} null_optics · "
              f"{arms['null_rest']} null_rest · excluded: "
              f"{len(doc['excluded'])}", "",
              "| feature | null_optics p95 | null_rest p95 | survives |",
              "|---|---|---|---|"]
    surv = set(doc["surviving_features"])
    for name, e in doc["null_arm_study"]["per_feature"].items():
        lines.append(f"| {name} | {e['null_optics_p95']} | "
                     f"{e['null_rest_p95']} | "
                     f"{'YES' if name in surv else 'no'} |")
    b = doc.get("battery") or {}
    if b.get("available"):
        lines += ["", f"added information over the HR+respiration "
                  f"baseline: {b['added_r2_over_b3']} "
                  f"(signed-r CI95 {b['added_r_ci95']})"]
    else:
        lines += ["", f"battery unavailable: {b.get('reason')}"]
    lines += ["", f"promotion: **{doc['gates']['promotion']}**", "",
              f"> {WATERMARK}"]
    return "\n".join(lines)
