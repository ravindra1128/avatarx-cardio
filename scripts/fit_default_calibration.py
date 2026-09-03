"""
Fit the DEFAULT beat-confidence calibration artifact (T5 wiring of T2).

The pipeline needs a versioned calibration to interpret `min_conf:
CALIBRATED`. v0.1 ships one fitted on the SYNTHETIC multi-ROI suite (seeds
below, disjoint from the test held-out seeds) — adequate for an interface
proof, clearly labelled as such. When real DEV recordings exist, refit on
them (through `assert_preprocessing_is_split_safe`) and bump the artifact;
the content-hash version string in every ScanResult keeps old results
traceable to this synthetic fit.

Run:  python3 scripts/fit_default_calibration.py
Writes: inference/confidence_calibration_v01.json
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from beats.detector import detect_beats_single_roi, fuse_multi_roi
from beats.confidence import fit_confidence_calibration
from evaluation.beat_metrics import match_beats, beat_confidence_calibration

FIT_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
CHECK_SEEDS = (40, 41)
NOISE = 0.13
FPS = 60.0
OUT = pathlib.Path(__file__).resolve().parents[1] / "inference" / \
    "confidence_calibration_v01.json"


def _pulse_template(n):
    t = np.linspace(0, 1, n, endpoint=False)
    return np.exp(-((t - 0.18) ** 2) / (2 * 0.075 ** 2))


def make_fused(rr_s, seed):
    rng = np.random.default_rng(seed)
    onsets = np.concatenate([[0.0], np.cumsum(rr_s)])
    n = int(onsets[-1] * FPS) + 1
    clean = np.zeros(n)
    peaks = []
    for a, b in zip(onsets, onsets[1:]):
        i0, i1 = int(a * FPS), int(b * FPS)
        if i1 - i0 > 2:
            clean[i0:i1] = _pulse_template(i1 - i0)
            peaks.append(a + 0.18 * (b - a))
    per_roi = {}
    for roi in ("forehead", "cheek_l", "cheek_r", "nose"):
        nz = NOISE * (1.0 if roi == "forehead" else 1.4 if roi == "nose" else 1.15)
        per_roi[roi] = detect_beats_single_roi(clean + rng.normal(0, nz, n),
                                               FPS, roi)
    return fuse_multi_roi(per_roi, FPS, float(onsets[-1]), min_rois=2), \
        np.array(peaks)


def build_suite(seeds):
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        for kind in ("sinus", "af"):
            rr = (np.clip(rng.normal(0.85, 0.03, 106), 0.5, 1.4)
                  if kind == "sinus" else
                  np.clip(rng.normal(0.62, 0.17, 145), 0.28, 1.35))
            series, truth = make_fused(rr, seed=s * 7 + 1)
            m = match_beats(truth, series.times(), tolerance_ms=100)
            out.append((m, series.confidences()))
    return out


def main():
    suite = build_suite(FIT_SEEDS)
    cal = fit_confidence_calibration([m for m, _ in suite],
                                     [c for _, c in suite])
    OUT.parent.mkdir(exist_ok=True)
    cal.save(str(OUT))
    print(f"wrote {OUT}")
    print(f"version {cal.version}  fitted on {cal.n_fit_beats} beats "
          f"from {cal.n_fit_recordings} synthetic recordings")
    for m, raw in build_suite(CHECK_SEEDS):
        e_raw = beat_confidence_calibration(m, raw)["ece"]
        e_cal = beat_confidence_calibration(m, cal.apply_values(raw))["ece"]
        print(f"  check: ECE raw {e_raw:.3f} -> calibrated {e_cal:.3f}")


if __name__ == "__main__":
    main()
