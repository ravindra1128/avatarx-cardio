"""
Fidelity evaluator (v0.3 T3) — every §G metric, one module.

Interval measurements here are AUTOMATED fiducial surrogates (slope-
threshold QRS onset/offset, tangent-method T end, P-peak-to-R for PR),
applied IDENTICALLY to reference and reconstruction so their difference
is meaningful. When adjudicated morphology labels exist
(datasets/reference.py labels[].morphology), those take precedence as the
reference-side truth; until then the surrogate nature of these
measurements is one more reason the gates stay red pending clinical
signoff. Correlation is computed but may be reported only ALONGSIDE the
interval metrics (G2), never instead of them.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

# lab reuse (research -> evaluation imports are fine; the quarantine is
# one-way: app/ and inference/ must reach NEITHER package)
from evaluation.inferred_ecg import report as _lab
from evaluation.inferred_ecg.decoder import FS
from evaluation.inferred_ecg.fiducials import _smooth, fiducials, rpeaks
from research.ecg_reconstruction import WATERMARK
from research.ecg_reconstruction.decoder import reconstruct, train
from training.runs import rank_auc

MATCH_TOL_S = 0.08
ENROLL_FRACTION = 0.6            # identity baseline's template period


# ------------------------------------------------- interval measurement
def beat_measurements(ecg: np.ndarray, fs: float,
                      r_times=None) -> list:
    """Per-beat automated measurements: QRS onset/offset (slope
    threshold), T end (tangent method), P/T peaks and prominences (lab
    fiducials) -> pr_ms, qrs_ms, qt_ms, rr_ms."""
    ecg = np.asarray(ecg, float)
    if r_times is None:
        r_times = rpeaks(ecg, fs)
    r_times = np.asarray(r_times, float)
    x = _smooth(ecg, fs)
    d = np.gradient(x) * fs
    fid = {round(f["r_t"], 3): f for f in fiducials(ecg, fs, r_times)}
    out = []
    prev_r = None
    for r in r_times:
        ri = int(round(r * fs))
        w = int(0.07 * fs)
        if ri - int(0.35 * fs) < 0 or ri + int(0.45 * fs) >= x.size:
            prev_r = r
            continue
        seg = np.abs(d[ri - w:ri + w + 1])
        thr = 0.10 * (float(np.max(seg)) or 1.0)
        onset_i = ri - w
        for i in range(ri, ri - w, -1):
            if abs(d[i]) < thr and abs(d[i - 1]) < thr:
                onset_i = i
                break
        offset_i = ri + w
        for i in range(ri, ri + w):
            if abs(d[i]) < thr and abs(d[i + 1]) < thr:
                offset_i = i
                break
        f = fid.get(round(float(r), 3))
        t_end_t = None
        if f is not None:
            ti = int(round(f["t_t"] * fs))
            j0, j1 = ti, min(x.size - 2, ti + int(0.15 * fs))
            if j1 > j0:
                j = j0 + int(np.argmin(d[j0:j1]))
                base_seg = x[ri - int(0.32 * fs):ri - int(0.28 * fs)]
                base = float(np.median(base_seg)) if base_seg.size \
                    else float(x[ri - w])
                if d[j] < 0:                     # tangent through max downslope
                    t_end_t = j / fs + (base - x[j]) / d[j]
                else:
                    t_end_t = f["t_t"] + 0.08    # flat T: fixed fallback
                t_end_t = min(max(t_end_t, f["t_t"]), f["t_t"] + 0.20)
        row = {"r_t": float(r),
               "rr_ms": (None if prev_r is None
                         else (r - prev_r) * 1000.0),
               "qrs_ms": (offset_i - onset_i) / fs * 1000.0,
               "pr_ms": f["pr_ms"] if f else None,
               "p_prom": f["p_prom"] if f else None,
               "t_prom": f["t_prom"] if f else None,
               "qt_ms": ((t_end_t - onset_i / fs) * 1000.0
                         if t_end_t is not None else None)}
        out.append(row)
        prev_r = r
    return out


def _med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return round(float(np.median(v)), 2) if v else None


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(a.size, b.size)
    if n == 0 or np.std(a[:n]) == 0 or np.std(b[:n]) == 0:
        return 0.0
    return round(float(np.corrcoef(a[:n], b[:n])[0, 1]), 3)


def interval_mae(ref_ecg: np.ndarray, gen_ecg: np.ndarray, fs: float
                 ) -> dict:
    """Matched-beat |Δinterval| medians (ms) of a generated ECG against
    the reference, same automated measure on both sides."""
    mr = beat_measurements(ref_ecg, fs)
    mg = beat_measurements(gen_ecg, fs)
    ref_t = np.array([m["r_t"] for m in mr]) if mr else np.array([])
    dq, dp, ds = [], [], []
    for g in mg:
        if not ref_t.size:
            break
        j = int(np.argmin(np.abs(ref_t - g["r_t"])))
        if abs(ref_t[j] - g["r_t"]) > MATCH_TOL_S:
            continue
        m = mr[j]
        for acc, k in ((dq, "qt_ms"), (dp, "pr_ms"), (ds, "qrs_ms")):
            if g[k] is not None and m[k] is not None:
                acc.append(abs(g[k] - m[k]))
    return {"qt_mae_ms": _med(dq), "pr_mae_ms": _med(dp),
            "qrs_mae_ms": _med(ds), "n_matched_beats": len(dq),
            "corr": _corr(gen_ecg, ref_ecg)}


# ------------------------------------------------- RR-only QT baseline
def rr_only_qt_fit(train_subjects: list, fs: float = FS) -> dict:
    """Linear QT ~ RR on the TRAIN subjects' reference measurements: the
    zero-morphology baseline G2 demands the decoder beat."""
    rr, qt = [], []
    for _, _, _, ecg in train_subjects:
        for m in beat_measurements(ecg, fs):
            if m["rr_ms"] is not None and m["qt_ms"] is not None:
                rr.append(m["rr_ms"]); qt.append(m["qt_ms"])
    if len(rr) < 10:
        return {"a": 350.0, "b": 0.0, "n_beats": len(rr)}
    A = np.stack([np.ones(len(rr)), np.asarray(rr)], 1)
    coef, *_ = np.linalg.lstsq(A, np.asarray(qt), rcond=None)
    return {"a": round(float(coef[0]), 3), "b": round(float(coef[1]), 5),
            "n_beats": len(rr)}


def rr_only_qt_mae(fit: dict, ref_ecg: np.ndarray, fs: float = FS):
    errs = []
    for m in beat_measurements(ref_ecg, fs):
        if m["rr_ms"] is not None and m["qt_ms"] is not None:
            errs.append(abs(fit["a"] + fit["b"] * m["rr_ms"] - m["qt_ms"]))
    return _med(errs)


# ------------------------------------------------- CIs
def bootstrap_ci(per_subject_values: list, *, n: int = 200,
                 seed: int = 11) -> list:
    v = [x for x in per_subject_values if x is not None and np.isfinite(x)]
    if len(v) < 3:
        return None
    rng = np.random.default_rng(seed)
    meds = [float(np.median(rng.choice(v, size=len(v), replace=True)))
            for _ in range(n)]
    return [round(float(np.percentile(meds, 2.5)), 2),
            round(float(np.percentile(meds, 97.5)), 2)]


# ------------------------------------------------- per-subject evaluation
def eval_subject(subject, artifact: dict, qt_fit: dict,
                 fs: float = FS) -> dict:
    """Decoder vs identity-template baseline vs RR-only QT baseline on one
    held-out subject; identity gets the same enrollment convention as the
    v0.2 lab (own first-60% template, re-timed to pulse beats)."""
    sid, is_af, ppg, ecg = subject
    cut = int(ENROLL_FRACTION * ppg.size)
    tpl_ppg, tpl_ecg = ppg[:cut], ecg[:cut]
    ev_ppg, ev_ecg = ppg[cut:], ecg[cut:]

    r_tpl = rpeaks(tpl_ecg, fs)
    template, r_off = _lab._identity_template(tpl_ecg, r_tpl)
    ppg_beats_tpl = _lab._ppg_beats(tpl_ppg)
    ptt = 0.25
    if ppg_beats_tpl.size and r_tpl.size:
        ds = [float(np.min(np.abs(ppg_beats_tpl - r)))
              for r in r_tpl if np.min(np.abs(ppg_beats_tpl - r)) < 0.6]
        if ds:
            ptt = float(np.median(ds))
    beats_ev = _lab._ppg_beats(ev_ppg) - ptt
    identity = _lab._place_template(template, r_off,
                                    beats_ev[beats_ev > 0], ev_ecg.size)
    generated = reconstruct(artifact, ev_ppg)
    return {"subject": sid, "is_af": int(is_af),
            "decoder": interval_mae(ev_ecg, generated, fs),
            "identity": interval_mae(ev_ecg, identity, fs),
            "rr_only_qt_mae_ms": rr_only_qt_mae(qt_fit, ev_ecg, fs)}


MORPHOLOGY_METRICS = ("qt_mae_ms", "pr_mae_ms", "qrs_mae_ms")


def aggregate(per_subject: list) -> dict:
    out = {}
    for side in ("decoder", "identity"):
        for k in MORPHOLOGY_METRICS + ("corr",):
            vals = [r[side][k] for r in per_subject]
            out[f"{side}_{k}"] = _med(vals)
            if k in MORPHOLOGY_METRICS:
                out[f"{side}_{k}_ci95"] = bootstrap_ci(vals)
    out["rr_only_qt_mae_ms"] = _med([r["rr_only_qt_mae_ms"]
                                     for r in per_subject])
    out["n_test_subjects"] = len(per_subject)
    return out


def g1_comparison(agg: dict) -> dict:
    """G1: the decoder must OUTPERFORM the identity template on every
    morphology metric; a tie or a missing number fails."""
    rows = {}
    ok_all = True
    for k in MORPHOLOGY_METRICS:
        d, i = agg.get(f"decoder_{k}"), agg.get(f"identity_{k}")
        ok = d is not None and i is not None and d < i
        rows[k] = {"decoder": d, "identity": i, "decoder_wins": bool(ok)}
        ok_all &= ok
    rows["all_morphology_metrics_beat_identity"] = bool(ok_all)
    return rows


# ------------------------------------------------- G4 confabulation
def confabulation_eval(train_subjects: list, test_subjects: list, *,
                       architecture: str = "windowed_mlp", seed: int = 7,
                       steps: int = 400, fs: float = FS) -> dict:
    """Withheld-class challenge: a decoder trained WITHOUT AF must not
    paint AF-typical-free morphology onto AF — measured as hallucinated P
    prominence on withheld AF vs the reference's own."""
    non_af = [s for s in train_subjects if not s[1]] or train_subjects
    art = train(non_af, architecture=architecture, seed=seed, steps=steps)
    p_gen, p_ref = [], []
    for s in test_subjects:
        if not s[1]:
            continue
        gen = reconstruct(art, s[2])
        mg = beat_measurements(gen, fs)
        mr = beat_measurements(s[3], fs)
        p_gen.append(_med([m["p_prom"] for m in mg]))
        p_ref.append(_med([m["p_prom"] for m in mr]))
    g, r = _med(p_gen), _med(p_ref)
    return {"withheld_class": "AFIB",
            "n_af_test_subjects": len(p_gen),
            "p_prominence_gen_on_withheld_af": g,
            "p_prominence_ref_on_af": r,
            "hallucinated_p_excess": (round(g - r, 3)
                                      if g is not None and r is not None
                                      else None)}


# ------------------------------------------------- G5 detection endpoints
def _irregularity(beat_times: np.ndarray):
    rr = np.diff(np.asarray(beat_times, float))
    rr = rr[(rr > 0.25) & (rr < 2.2)]
    if rr.size < 8:
        return float("nan")
    return float(np.median(np.abs(np.diff(rr))) / np.median(rr))


def detection_endpoints(test_subjects: list, artifact: dict,
                        fs: float = FS) -> dict:
    """AF endpoint (interval-irregularity score) computed from the
    MEASURED pulse path vs from the reconstruction — G5 asks whether the
    detour through a generated waveform loses detection performance."""
    y, s_meas, s_recon = [], [], []
    for sid, is_af, ppg, ecg in test_subjects:
        y.append(int(is_af))
        s_meas.append(_irregularity(_lab._ppg_beats(ppg)))
        s_recon.append(_irregularity(rpeaks(reconstruct(artifact, ppg), fs)))
    auc_m = rank_auc(np.array(y), np.array(s_meas))
    auc_r = rank_auc(np.array(y), np.array(s_recon))
    fin = lambda v: None if not np.isfinite(v) else round(float(v), 3)
    return {"endpoint": "afib_irregularity", "n_subjects": len(y),
            "auc_measured_path": fin(auc_m),
            "auc_from_reconstruction": fin(auc_r),
            "auc_deficit": (round(float(auc_m - auc_r), 3)
                            if np.isfinite(auc_m) and np.isfinite(auc_r)
                            else None)}


# ------------------------------------------------- G3 blinded-read kit
def generate_blinded_set(test_subjects: list, artifact: dict, out_dir, *,
                         seed: int = 23, fs: float = FS) -> dict:
    """Randomized, de-identified reconstruction strips for blinded
    cardiologist reads; the answer key is escrowed OUTSIDE the case
    folder. Every strip carries the watermark — blinding hides the
    answer, never the synthetic provenance."""
    out = pathlib.Path(out_dir)
    cases_dir = out / "cases"
    escrow = out / "escrow"
    cases_dir.mkdir(parents=True, exist_ok=True)
    escrow.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(test_subjects))
    key, cases = {}, []
    for k, idx in enumerate(order):
        sid, is_af, ppg, ecg = test_subjects[int(idx)]
        case_id = hashlib.sha256(f"{seed}|{k}|{sid}".encode()).hexdigest()[:8]
        gen = reconstruct(artifact, ppg)
        _strip_svg(gen, cases_dir / f"{case_id}.svg", fs)
        key[case_id] = {"subject": sid, "is_af": int(is_af)}
        cases.append(case_id)
    (cases_dir / "manifest.json").write_text(json.dumps(
        {"WATERMARK": WATERMARK, "instructions":
         "Blinded read: for each case, record is_af (0/1) and any "
         "morphology abnormality seen. Cases are reconstructions — "
         "the question is whether the truth survives them.",
         "cases": cases, "response_format": {"<case_id>": {"is_af": 0}}},
        indent=1))
    (escrow / "answer_key.json").write_text(json.dumps(
        {"WATERMARK": WATERMARK, "seed": seed, "key": key}, indent=1))
    return {"cases_dir": str(cases_dir),
            "answer_key": str(escrow / "answer_key.json"),
            "n_cases": len(cases)}


def score_blinded_reads(answer_key_path, responses_path) -> dict:
    with open(answer_key_path) as f:
        key = json.load(f)["key"]
    with open(responses_path) as f:
        resp = json.load(f)
    tp = fp = tn = fn = 0
    for cid, truth in key.items():
        r = resp.get(cid)
        if r is None:
            continue
        call, real = int(r.get("is_af", 0)), int(truth["is_af"])
        tp += call and real; fp += call and not real
        tn += (not call) and (not real); fn += (not call) and real
    se = tp / (tp + fn) if (tp + fn) else None
    sp = tn / (tn + fp) if (tn + fp) else None
    return {"n_scored": tp + fp + tn + fn,
            "sensitivity": None if se is None else round(se, 3),
            "specificity": None if sp is None else round(sp, 3)}


def _strip_svg(sig: np.ndarray, out_path, fs: float, seconds: float = 8.0):
    n = min(int(seconds * fs), sig.size)
    W, H = 900, 180
    s = np.asarray(sig[:n], float)
    rngv = float(np.max(np.abs(s))) or 1.0
    xs = np.linspace(20, W - 10, n)
    ys = 110 - s / rngv * 40
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
           f'<rect width="{W}" height="{H}" fill="white"/>'
           f'<text x="{W/2}" y="18" font-size="13" fill="#b45309" '
           f'text-anchor="middle" font-family="sans-serif" '
           f'font-weight="bold">{WATERMARK}</text>'
           f'<polyline points="{pts}" fill="none" stroke="#333" '
           f'stroke-width="1"/>'
           f'<text x="{W/2}" y="{H - 8}" font-size="11" fill="#b45309" '
           f'text-anchor="middle" font-family="sans-serif">{WATERMARK}'
           f'</text></svg>')
    pathlib.Path(out_path).write_text(svg)


# ------------------------------------------------- one-call evaluation
def evaluate_fidelity(train_subjects: list, test_subjects: list,
                      artifact: dict, *, fs: float = FS) -> dict:
    qt_fit = rr_only_qt_fit(train_subjects, fs)
    per_subject = [eval_subject(s, artifact, qt_fit, fs)
                   for s in test_subjects]
    agg = aggregate(per_subject)
    agg["rr_only_qt_fit"] = qt_fit
    return {"per_subject": per_subject, "aggregate": agg,
            "g1": g1_comparison(agg),
            "detection": detection_endpoints(test_subjects, artifact, fs)}
