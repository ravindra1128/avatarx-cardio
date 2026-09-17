"""Which waveform-rhythm estimator agrees with the SDK's heart rate on the
retained phone clips? (2026-09-17)

On the four 2026-09-17 staging scans (SDK rate 84-90 bpm) our
`spectral_pulse` read 45-48 bpm on 3 of 4 regions - not the subharmonic
branch, the strongest in-band peak itself: a ~0.8 Hz component dominates the
forehead and cheek waveforms while only the nose shows the true 1.4 Hz. The
resolver then folded a right-ish count to the wrong rate (fitness 87/100 at
45 bpm) and the ShenAI route lost its corroboration on every one of them.

This runs the production ingest + extraction on every retained clip that has
a sidecar and scores estimator VARIANTS against the SDK's heart_rate_bpm
(independent frames, no shared clock): fraction of clips within 10 %, and
median |error|. Read-only; nothing here changes the pipeline. Variants:
  current   inference.evidence._fundamental as shipped
  harm      harmonic-sum: score(f) = P(f) + 0.5 P(2f), argmax strictly inside
            the band (a pulse has a 2nd harmonic; drift and sway do not)
  harm_lm   harm, restricted to local maxima of P (never a monotone shoulder)
  roi_vote  per-ROI fundamentals (current), the median of those within the
            band's interior - one dominated region cannot carry the vote
Run: python3 scripts/spectral_variants.py [clips...]
"""
import sys, pathlib, glob, json, shutil, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

import app.measure_overrides  # noqa: F401  (launch overrides, e.g. collapsed fraction)
from app.measure_prep import trim_tail, downscale
from capture.ingest import ingest_video
from configs import load_config
from inference.evidence import (extract_and_detect, _welch_psd, _fundamental,
                                SPECTRAL_BAND_HZ)
from preprocessing.roi import ROI_NAMES

REPO = pathlib.Path(__file__).resolve().parents[1]
LO, HI = SPECTRAL_BAND_HZ


def _interior(f, p):
    m = (f >= LO) & (f <= HI)
    fb, pb = f[m], p[m]
    return (fb[1:-1], pb[1:-1]) if fb.size > 2 else (fb, pb)


def v_current(psds, fb):
    f0, _ = _fundamental(np.concatenate([[LO - 0.05], fb, [HI + 0.05]]),
                         np.concatenate([[0.0], np.mean(psds, axis=0), [0.0]]))
    return None if f0 is None else f0 * 60.0


def _harm_score(fb, pb):
    score = np.array(pb, float)
    for i, f in enumerate(fb):
        j = np.argmin(np.abs(fb - 2 * f))
        if abs(fb[j] - 2 * f) <= 0.06 and 2 * f <= HI:
            score[i] += 0.5 * pb[j]
    return score


def v_harm(psds, fb):
    pb = np.mean(psds, axis=0)
    return float(fb[int(np.argmax(_harm_score(fb, pb)))]) * 60.0


def v_harm_lm(psds, fb):
    pb = np.mean(psds, axis=0)
    sc = _harm_score(fb, pb)
    lm = np.zeros(pb.size, bool)
    lm[1:-1] = (pb[1:-1] > pb[:-2]) & (pb[1:-1] > pb[2:])
    if not lm.any():
        return v_harm(psds, fb)
    return float(fb[int(np.argmax(np.where(lm, sc, -np.inf)))]) * 60.0


def v_roi_vote(roi_f0s):
    v = [x for x in roi_f0s if x is not None]
    return None if len(v) < 2 else float(np.median(v)) * 60.0


def analyse(clip: pathlib.Path, cfg) -> dict:
    side = json.loads((pathlib.Path(str(clip) + ".shenai.json")).read_text())
    ref = (side.get("reference") or {}).get("heart_rate_bpm")
    tmp = tempfile.mkdtemp(prefix="specvar_")
    try:
        work = str(pathlib.Path(tmp) / clip.name)
        shutil.copy(str(clip), work)
        trim_tail(work, 45.0)
        d = downscale(work, "640x480")
        if d.get("applied") and d.get("path"):
            work = d["path"]
        ing = ingest_video(work, capture_profile="consumer")
        if not ing.ok:
            return {"clip": clip.name, "ref": ref, "error": "; ".join(ing.reasons)}
        fps = ing.meta.measured_fps_mean
        ts = np.asarray(ing.timestamps_s, float)
        raw, _, _ = extract_and_detect(ing.traces, ts, fps, cfg)
        psds, fb_ref, roi_f0 = [], None, []
        for roi in ROI_NAMES:
            f, p = _welch_psd(raw[roi], fps)
            if f is None:
                continue
            fb, pb = _interior(f, p)
            tot = float(np.sum(pb))
            if tot <= 0:
                continue
            if fb_ref is None or fb.shape == fb_ref.shape:
                fb_ref = fb
                psds.append(pb / tot)
            f0, _ = _fundamental(f, p)
            roi_f0.append(f0)
        if not psds:
            return {"clip": clip.name, "ref": ref, "error": "no spectra"}
        fused = np.mean(psds, axis=0)
        return {"clip": clip.name, "ref": ref,
                "fb": [round(float(x), 3) for x in fb_ref],
                "fused_psd": [float(x) for x in fused],
                "roi_psd": [[float(x) for x in q] for q in psds],
                "current": v_current(psds, fb_ref), "harm": v_harm(psds, fb_ref),
                "harm_lm": v_harm_lm(psds, fb_ref), "roi_vote": v_roi_vote(roi_f0),
                "roi_bpm": [None if x is None else round(x * 60, 1) for x in roi_f0]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv):
    cfg = load_config()
    clips = [pathlib.Path(a) for a in argv] or sorted(
        pathlib.Path(p) for p in glob.glob(str(REPO / "data/eval_corpus/2026*.webm"))
        if pathlib.Path(p + ".shenai.json").exists())
    rows = []
    for c in clips:
        r = analyse(c, cfg)
        rows.append(r)
        print(json.dumps(r), flush=True)
    variants = ("current", "harm", "harm_lm", "roi_vote")
    print("\nvariant     within10%   median|err| bpm   n")
    for v in variants:
        errs = [abs(r[v] - r["ref"]) for r in rows
                if r.get(v) is not None and r.get("ref")]
        rel = [abs(r[v] - r["ref"]) / r["ref"] <= 0.10 for r in rows
               if r.get(v) is not None and r.get("ref")]
        if errs:
            print(f"{v:10s}  {np.mean(rel):8.2f}   {np.median(errs):10.1f}   {len(errs)}")
    out = REPO / "data" / "eval_cache" / "spectral_variants.json"
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
