"""
The falsification report (M4.14) — the three tests that matter, as one
automated, watermarked artifact under `evaluation/falsification_runs/`:

  (a) identity-template baseline: each held-out subject's own
      template-period average ECG beat, re-timed to PPG-detected beats —
      a model with ZERO learned morphology. If a trained decoder cannot
      beat it, the decoder's morphology is decoration.
  (b) interval-level error: fiducial/interval errors in ms against the
      reference ECG (correlation reported only alongside).
  (c) cross-rhythm morphology challenge: a decoder trained WITHOUT AF
      reconstructing AF segments (and vice versa) — hallucinated P waves
      made visible and quantified.

Every artifact carries the watermark (invariant 10). This module is
research-only and unreachable from app/ or the production pipeline.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
from scipy.signal import find_peaks

from evaluation.inferred_ecg import WATERMARK
from evaluation.inferred_ecg.decoder import (FS, decode, perform_subjects,
                                             split_subjects, synth_pairs,
                                             train_decoder)
from evaluation.inferred_ecg.fiducials import interval_errors, rpeaks

_REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO / "evaluation" / "falsification_runs"


def _ppg_beats(ppg: np.ndarray) -> np.ndarray:
    idx, _ = find_peaks(np.asarray(ppg, float), distance=int(0.35 * FS),
                        prominence=0.5)
    return idx / FS


def _identity_template(ref_ecg: np.ndarray, r_times: np.ndarray) -> tuple:
    """Average beat (window -0.30..+0.45 s around R) from the template
    period."""
    a, b = int(0.30 * FS), int(0.45 * FS)
    segs = []
    for r in r_times:
        ri = int(round(r * FS))
        if a <= ri < ref_ecg.size - b:
            segs.append(ref_ecg[ri - a:ri + b])
    if not segs:
        return np.zeros(a + b), a
    return np.mean(np.stack(segs), 0), a


def _place_template(template: np.ndarray, r_off: int, beat_times: np.ndarray,
                    n: int) -> np.ndarray:
    out = np.zeros(n)
    for bt in beat_times:
        ri = int(round(bt * FS))
        s0 = ri - r_off
        s1 = s0 + template.size
        if s0 >= 0 and s1 <= n:
            out[s0:s1] += template
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _eval_subject(subject, artifact) -> dict:
    sid, is_af, ppg, ecg = subject
    cut = int(0.6 * ppg.size)
    tpl_ppg, tpl_ecg = ppg[:cut], ecg[:cut]
    ev_ppg, ev_ecg = ppg[cut:], ecg[cut:]

    # identity baseline gets the SAME information as the decoder at eval
    # time: eval-period PPG only (plus its own template-period ECG).
    r_tpl = rpeaks(tpl_ecg, FS)
    template, r_off = _identity_template(tpl_ecg, r_tpl)
    ppg_beats_tpl = _ppg_beats(tpl_ppg)
    ptt = 0.25
    if ppg_beats_tpl.size and r_tpl.size:
        d = [float(np.min(np.abs(ppg_beats_tpl - r)))
             for r in r_tpl if np.min(np.abs(ppg_beats_tpl - r)) < 0.6]
        if d:
            ptt = float(np.median(d))
    beats_ev = _ppg_beats(ev_ppg) - ptt
    identity = _place_template(template, r_off,
                               beats_ev[beats_ev > 0], ev_ecg.size)
    generated = decode(artifact, ev_ppg)

    return {"subject": sid, "is_af": int(is_af),
            "decoder": {"corr": round(_corr(generated, ev_ecg), 3),
                        **interval_errors(ev_ecg, generated, FS)},
            "identity_baseline": {"corr": round(_corr(identity, ev_ecg), 3),
                                  **interval_errors(ev_ecg, identity, FS)},
            "_arrays": (ev_ecg, generated, identity)}


def _svg_strip(ref, gen, idn, out_path):
    W, H = 900, 260
    n = min(int(8 * FS), ref.size)
    xs = np.linspace(20, W - 10, n)

    def row(sig, y0, colour, label):
        s = np.asarray(sig[:n], float)
        rng = float(np.max(np.abs(s))) or 1.0
        ys = y0 - s / rng * 26
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        return (f'<text x="20" y="{y0 - 34}" font-size="11" fill="#555" '
                f'font-family="sans-serif">{label}</text>'
                f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                f'stroke-width="1"/>')

    svg = (f'<svg viewBox="0 0 {W} {H}" '
           f'xmlns="http://www.w3.org/2000/svg">'
           f'<rect width="{W}" height="{H}" fill="white"/>'
           + row(ref, 70, "#111", "reference ECG (measured)")
           + row(gen, 150, "#c22", "decoder output — " + WATERMARK)
           + row(idn, 230, "#26a", "identity-template baseline — "
                 + WATERMARK)
           + f'<text x="{W/2}" y="16" font-size="13" fill="#c22" '
           f'text-anchor="middle" font-family="sans-serif" '
           f'font-weight="bold">{WATERMARK}</text></svg>')
    pathlib.Path(out_path).write_text(svg)


def run_falsification(*, source: str = "synthetic", limit: int = 0,
                      steps: int = 400, seed: int = 7,
                      out_root=None) -> dict:
    if source == "mimic":
        subjects = perform_subjects(limit_per_class=limit)
        if not subjects:
            raise RuntimeError("MIMIC PERform cache not present — run with "
                               "--source synthetic or fetch the cache "
                               "(scripts/e6_degradation.py)")
    else:
        subjects = synth_pairs(max(limit, 6) if limit else 6, seed=seed)

    train, test = split_subjects(subjects, seed=seed)
    if not train or not test:
        raise RuntimeError(f"split produced empty side "
                           f"(train {len(train)}, test {len(test)})")

    art_all = train_decoder(train, seed=seed, steps=steps)
    art_no_af = train_decoder([s for s in train if not s[1]] or train,
                              seed=seed, steps=steps)
    art_af_only = train_decoder([s for s in train if s[1]] or train,
                                seed=seed, steps=steps)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(out_root or DEFAULT_OUT) / ts
    out.mkdir(parents=True, exist_ok=True)

    per_subject = []
    strip_done = False
    for s in test:
        r = _eval_subject(s, art_all)
        ref, gen, idn = r.pop("_arrays")
        if not strip_done:
            _svg_strip(ref, gen, idn, out / f"strip_{s[0]}.svg")
            strip_done = True
        per_subject.append(r)

    challenge = {"decoder_without_af_on_af": [],
                 "decoder_af_only_on_sinus": []}
    for s in test:
        if s[1]:
            r = _eval_subject(s, art_no_af)
            r.pop("_arrays")
            challenge["decoder_without_af_on_af"].append(r["decoder"])
        else:
            r = _eval_subject(s, art_af_only)
            r.pop("_arrays")
            challenge["decoder_af_only_on_sinus"].append(r["decoder"])

    def med(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return round(float(np.median(v)), 3) if v else None

    summary = {
        "n_train": len(train), "n_test": len(test),
        "decoder_median_corr": med([r["decoder"]["corr"]
                                    for r in per_subject]),
        "identity_median_corr": med([r["identity_baseline"]["corr"]
                                     for r in per_subject]),
        "decoder_pr_mae_ms": med([r["decoder"]["pr_interval_mae_ms"]
                                  for r in per_subject]),
        "identity_pr_mae_ms": med([r["identity_baseline"]
                                   ["pr_interval_mae_ms"]
                                   for r in per_subject]),
        "decoder_rt_mae_ms": med([r["decoder"]["rt_interval_mae_ms"]
                                  for r in per_subject]),
        "identity_rt_mae_ms": med([r["identity_baseline"]
                                   ["rt_interval_mae_ms"]
                                   for r in per_subject]),
        "hallucinated_p_on_af": med(
            [c["p_prominence_gen"]
             for c in challenge["decoder_without_af_on_af"]]),
        "reference_p_on_af": med(
            [c["p_prominence_ref"]
             for c in challenge["decoder_without_af_on_af"]]),
    }
    doc = {"WATERMARK": WATERMARK,
           "purpose": "falsification: why generated ECG waveforms do not "
                      "ship (spec B.15, invariant 10)",
           "source": source, "seed": seed, "steps": steps,
           "participant_disjoint": True,
           "summary": summary, "per_subject": per_subject,
           "cross_rhythm_challenge": challenge}
    with open(out / "report.json", "w") as f:
        json.dump(doc, f, indent=1)

    html = ["<!doctype html><meta charset='utf-8'>",
            f"<div style='background:#c22;color:#fff;padding:10px;"
            f"font:bold 14px sans-serif'>{WATERMARK}</div>",
            "<h2 style='font-family:sans-serif'>Inferred-ECG "
            "falsification report</h2>",
            f"<pre>{json.dumps({'summary': summary}, indent=1)}</pre>"]
    for svg in sorted(out.glob("*.svg")):
        html.append(svg.read_text())
    html.append(f"<div style='background:#c22;color:#fff;padding:10px;"
                f"font:bold 14px sans-serif'>{WATERMARK}</div>")
    (out / "report.html").write_text("\n".join(html))
    return {"out_dir": str(out), "summary": summary}
