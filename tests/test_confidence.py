"""
T2 — beat-confidence calibration.

Two properties are under test here:

1. RANKING (detector): the fused confidence must rank real beats above the
   chance-coincidence 2-ROI false beats that dominate the false population.
   The v3 formula used the cluster time RANGE, which grows with cluster
   size, so full-agreement real beats scored BELOW tight 2-ROI noise pairs —
   measured AUC 0.32-0.45, i.e. inverted. A monotone calibrator cannot
   repair an inverted ranking, so this is a hard precondition of T2 and is
   pinned by a regression test.

2. SCALE (calibrator): even ranked correctly, the confidences are
   compressed (demo ECE 0.42 vs the 0.10 gate). A monotone isotonic map
   fitted on ECG-matched labels must bring HELD-OUT ECE within the gate
   without reordering beats and without mutating the input series.

The fitting and held-out suites use DISJOINT seeds — fitting on the
evaluation suite is exactly the leak `assert_preprocessing_is_split_safe`
exists to prevent.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import functools

import numpy as np
import pytest

from beats.detector import (Beat, BeatSeries, detect_beats_single_roi,
                            fuse_multi_roi)
from beats.ibi import clean_runs
from beats.confidence import Calibrator, fit_confidence_calibration
from evaluation.beat_metrics import match_beats, beat_confidence_calibration
from evaluation.afib_metrics import _auroc


# ---------------------------------------------------------------- generator
def _pulse_template(n):
    t = np.linspace(0, 1, n, endpoint=False)
    return np.exp(-((t - 0.18) ** 2) / (2 * 0.075 ** 2))


NOISE = 0.13         # controlled-capture regime (demo uses 0.12 forehead)


def make_fused(rr_s, seed, fps=60.0, noise=NOISE, spike_times=()):
    """Multi-ROI synthetic recording -> (fused BeatSeries, true peak times).

    Ambient false beats arise naturally as chance coincidences of
    suprathreshold noise peaks across >=2 ROIs — the mechanism the fusion
    stage actually faces. `spike_times` additionally INJECT the controlled
    false-beat population for the T2 exclusion criterion: narrow (1-2 frame)
    suprathreshold transients hitting two ROIs at once, the signature of
    specular glints and sensor spikes.
    """
    rng = np.random.default_rng(seed)
    onsets = np.concatenate([[0.0], np.cumsum(rr_s)])
    n = int(onsets[-1] * fps) + 1
    peaks = []
    clean = np.zeros(n)
    for a, b in zip(onsets, onsets[1:]):
        i0, i1 = int(a * fps), int(b * fps)
        if i1 - i0 > 2:
            clean[i0:i1] = _pulse_template(i1 - i0)
            peaks.append(a + 0.18 * (b - a))
    per_roi = {}
    for roi in ("forehead", "cheek_l", "cheek_r", "nose"):
        nz = noise * (1.0 if roi == "forehead" else 1.4 if roi == "nose" else 1.15)
        sig = clean + rng.normal(0, nz, n)
        if roi in ("cheek_l", "cheek_r"):
            for ts in spike_times:
                i = int(ts * fps)
                if 2 < i < n - 3:
                    sig[i:i + 2] += 1.6
        per_roi[roi] = detect_beats_single_roi(sig, fps, roi)
    dur = float(onsets[-1])
    return fuse_multi_roi(per_roi, fps, dur, min_rois=2), np.array(peaks)


def sinus_rr(rng, n=106):
    return np.clip(rng.normal(0.85, 0.03, n), 0.5, 1.4)


def af_rr(rng, n=145):
    return np.clip(rng.normal(0.62, 0.17, n), 0.28, 1.35)


def build_suite(seeds):
    """(match, raw_confidences, series, truth) per synthetic recording."""
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        for kind in ("sinus", "af"):
            rr = sinus_rr(rng) if kind == "sinus" else af_rr(rng)
            series, truth = make_fused(rr, seed=s * 7 + 1)
            m = match_beats(truth, series.times(), tolerance_ms=100)
            out.append((m, series.confidences(), series, truth))
    return out


FIT_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
HELDOUT_SEEDS = (20, 21, 22)


@functools.lru_cache(maxsize=1)
def fitted_calibrator():
    suite = build_suite(FIT_SEEDS)
    return fit_confidence_calibration([m for m, _, _, _ in suite],
                                      [c for _, c, _, _ in suite])


# ------------------------------------------------- detector ranking (fix)
def test_fused_confidence_ranks_real_above_false():
    """Regression for the v0.1 measured defect: range-based spread inverted
    the ranking (AUC 0.32-0.45). The corrected formula must rank real beats
    above chance-coincidence false beats decisively."""
    ys, cs = [], []
    for m, conf, _, _ in build_suite(HELDOUT_SEEDS):
        lab = np.zeros(m.n_detected)
        lab[m.matched_det_idx] = 1
        ys.append(lab); cs.append(conf)
    y = np.concatenate(ys).astype(int)
    c = np.concatenate(cs)
    assert 0 < y.sum() < y.size, "suite must contain both real and false beats"
    assert _auroc(y, c) > 0.85, f"AUC {_auroc(y, c):.3f}"


# ---------------------------------------------------------------- behaviour
def test_apply_returns_new_series_and_never_mutates():
    cal = fitted_calibrator()
    series, _ = make_fused(sinus_rr(np.random.default_rng(50)), seed=51)
    before = series.confidences().copy()
    out = cal.apply(series)
    assert out is not series
    assert np.array_equal(series.confidences(), before)   # untouched
    assert len(out.beats) == len(series.beats)
    assert np.array_equal(out.times(), series.times())    # timing untouched


def test_calibration_is_monotone():
    cal = fitted_calibrator()
    grid = np.linspace(0, 1, 101)
    mapped = cal.apply_values(grid)
    assert np.all(np.diff(mapped) >= -1e-12)
    assert np.all((mapped >= 0) & (mapped <= 1))


def test_nan_confidence_calibrates_to_zero():
    cal = fitted_calibrator()
    assert cal.apply_values(np.array([np.nan]))[0] == 0.0  # fail closed


def test_heldout_ece_within_gate():
    """THE T2 acceptance: ECE <= 0.10 on a suite with seeds disjoint from
    fitting, and calibration must not make any recording worse."""
    cal = fitted_calibrator()
    for m, raw, series, _ in build_suite(HELDOUT_SEEDS):
        raw_ece = beat_confidence_calibration(m, raw)["ece"]
        cal_ece = beat_confidence_calibration(m, cal.apply_values(raw))["ece"]
        assert cal_ece <= 0.10, (cal_ece, raw_ece)
        assert cal_ece <= raw_ece + 0.02                   # never much worse


def test_clean_runs_at_calibrated_half_keeps_true_drops_injected_false():
    """T2 acceptance, controlled injection form. After calibration the 0.5
    threshold means P(real)=0.5: clean sinus keeps >=80% of true beats
    through `clean_runs(min_conf=0.5)`, and INJECTED false beats (2-ROI
    coincident narrow transients pushed through the real detector path) are
    >=80% excluded.

    MEASURED LIMITATION, on the record: the ambient unmatched population
    also contains broad 2-ROI clusters — double detections of real pulses —
    whose exclusion by confidence alone plateaus at ~33-59%. Those carry
    genuine pulse morphology; separating them is interval-domain work (the
    spec's short-pair splitter task, outside v0.1) and MUST NOT be forced
    into the confidence channel, which would push weak real AF beats out
    with them."""
    cal = fitted_calibrator()
    n_injected_total = 0
    for s in HELDOUT_SEEDS:
        rng = np.random.default_rng(s)
        # retention: clean sinus, nothing injected
        series, truth = make_fused(sinus_rr(rng), seed=s * 7 + 3)
        calibrated = cal.apply(series)
        rs = clean_runs(calibrated, min_conf=0.5, min_run_beats=4)
        assert rs.kept_beats >= 0.8 * truth.size, (rs.kept_beats, truth.size)
        m = match_beats(truth, series.times(), tolerance_ms=100)
        conf = cal.apply_values(series.confidences())
        real = np.zeros(len(series.beats), bool)
        real[m.matched_det_idx] = True
        assert np.mean(conf[real] >= 0.5) >= 0.8
        # exclusion: same recording with injected mid-diastole spike pairs
        rr = sinus_rr(np.random.default_rng(s))
        beats_t = np.concatenate([[0.0], np.cumsum(rr)])
        ii = np.arange(8, rr.size - 8, 6)
        inj = beats_t[ii] + 0.5 * rr[ii]                  # mid-diastole
        series2, truth2 = make_fused(sinus_rr(np.random.default_rng(s)),
                                     seed=s * 7 + 3, spike_times=inj)
        conf2 = cal.apply_values(series2.confidences())
        t2 = series2.times()
        # END-TO-END exclusion: an injected event is excluded unless it
        # survives every layer — fuses into a beat AND clears the calibrated
        # 0.5 threshold that admits it into clean runs. Most injections die
        # upstream (fusion membership, width discount); that is the system
        # working, and the measurement must credit it.
        survived = 0
        for ti in inj:
            if t2.size == 0:
                continue
            j = int(np.argmin(np.abs(t2 - ti)))
            is_injected_beat = abs(t2[j] - ti) < 0.06 and \
                np.min(np.abs(truth2 - t2[j])) > 0.12
            if is_injected_beat and conf2[j] >= 0.5:
                survived += 1
        n_injected_total += len(inj)
        assert survived <= 0.2 * len(inj), \
            f"{survived}/{len(inj)} injected false beats entered the run set"
    assert n_injected_total >= 10, "too few injected beats to test"


# ---------------------------------------------------------------- artefact
def test_serialise_roundtrip_preserves_mapping_and_version():
    cal = fitted_calibrator()
    blob = cal.to_json()
    back = Calibrator.from_json(blob)
    grid = np.linspace(0, 1, 57)
    assert np.allclose(back.apply_values(grid), cal.apply_values(grid))
    assert back.version == cal.version
    assert cal.version.startswith("cal-")
    assert len(cal.version) > 8


def test_version_changes_when_fit_changes():
    cal_a = fitted_calibrator()
    suite = build_suite((30, 31))
    cal_b = fit_confidence_calibration([m for m, _, _, _ in suite],
                                       [c for _, c, _, _ in suite])
    assert cal_a.version != cal_b.version


def test_save_and_load_file(tmp_path):
    cal = fitted_calibrator()
    p = tmp_path / "cal.json"
    cal.save(str(p))
    back = Calibrator.load(str(p))
    assert back.version == cal.version


def test_fit_refuses_mismatched_lengths():
    suite = build_suite((40,))
    m, c, _, _ = suite[0]
    with pytest.raises(ValueError):
        fit_confidence_calibration([m], [c[:-2]])
