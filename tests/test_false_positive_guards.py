"""
False-positive guards (v0.1.2) — written against a REAL false positive.

A user's live scan read AFIB_SUGGESTIVE while an Apple Watch check did not
detect AF. Reproduced on the lossless recording; the mechanism was NOT
rhythm irregularity but DETECTION-ERROR irregularity on a weak, incoherent
signal:

  * forehead + nose saw a regular ~1090 ms rhythm (~55 bpm); the cheeks
    were noise and "detected" 36–41 spurious peaks;
  * fusion built 22/25 beats from 2-ROI chance coincidences (cross-ROI
    coherence 0.06) — 7 half-intervals and 2 double-intervals: the
    HARMONIC signature of false/missed beats, not of AF;
  * the synthetic-fitted calibrator rated those 0.83, clean_runs had no
    short-interval splitter, SQI 0.40 / coverage 0.50 / 17 intervals sat
    exactly at floors, and the interim AF rule fired.

These tests pin the layered protection: (1) clean_runs breaks runs at the
short-pair false-beat signature (spec T5) and reports the split burden;
(2) an AF call requires VERIFIED beats — cross-ROI coherence, enough
intervals and coverage, and irregularity that is not explained by
detection-error harmonics — otherwise REPEAT_SCAN, never a class;
(3) the decision is explainable: a rationale records every gate and rule.

Reference disclaimer: an Apple Watch and this prototype are NOT
interchangeable references; ECG-confirmed rhythm is the only ground truth
for AF performance. The Watch reading here is a red flag, not a label.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from beats.detector import Beat, BeatSeries
from beats.ibi import clean_runs, rmssd_from_runs
from configs import load_config
from datasets.schema import ScanOutcome
from features.rhythm import RhythmFeatures, compute_rhythm_features_from_runs
from inference.decision_logic import decide, decide_with_rationale, \
    beat_evidence_from_series

REAL_SCANS = [pathlib.Path(p) for p in (
    "/tmp/avatarx_live/a9127ddbcde6/scan_a9127ddbcde6.avi",
    "/tmp/avatarx_live/bbfafba28faa/scan_bbfafba28faa.avi")]


# --------------------------------------------------------------- helpers
def _series(times_s, conf=0.9, agree=1.0, fps=30.0):
    beats = [Beat(t_s=float(t), confidence=conf, roi_agreement=agree,
                  signal_quality=0.9, amplitude=1.0, prominence=1.0)
             for t in times_s]
    return BeatSeries(beats, fps, float(times_s[-1]) + 1.0)


def regular_with_errors(seed=0, n=28, ibi=1.09, n_false=4, n_missed=2):
    """~55 bpm regular rhythm + false half-beats + missed beats: the exact
    detection-error signature measured on the real recording."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.full(n, ibi) + rng.normal(0, 0.012, n))
    keep = np.ones(n, bool)
    keep[rng.choice(np.arange(3, n - 3), n_missed, replace=False)] = False
    t = t[keep]
    extra = []
    for j in rng.choice(np.arange(2, t.size - 3), n_false, replace=False):
        extra.append(t[j] + 0.5 * (t[j + 1] - t[j]) + rng.normal(0, 0.03))
    return np.sort(np.concatenate([t, extra]))


def af_like(seed=13, n=45):
    rng = np.random.default_rng(seed)
    rr = np.clip(rng.normal(0.62, 0.17, n), 0.28, 1.35)
    return np.cumsum(rr)


# ------------------------------------------ (1) short-pair splitter in ibi
def test_short_pair_splitter_removes_false_half_beats():
    """A false beat splits one interval into two halves that sum to the
    local median. clean_runs must break the run there (no repair, no
    interpolation) and report the burden."""
    t = regular_with_errors(seed=1, n_false=4, n_missed=0)
    rs = clean_runs(_series(t), min_conf=0.5, min_run_beats=4)
    assert rs.n_false_pair_splits >= 3, rs.n_false_pair_splits
    ivl = rs.all_intervals()
    assert np.all(ivl > 0.75 * 1090), np.round(ivl).astype(int).tolist()
    assert rmssd_from_runs(rs) < 40                 # sinus-level again


def test_short_pair_splitter_preserves_af():
    """THE self-consistency check (spec T5): AF must survive the splitter —
    it must not recreate the open-rppg clean_rr failure."""
    t = af_like()
    rs = clean_runs(_series(t), min_conf=0.5)
    true_rmssd = float(np.sqrt(np.mean(np.diff(np.diff(t) * 1000) ** 2)))
    assert true_rmssd > 150
    assert rmssd_from_runs(rs) > 0.6 * true_rmssd
    assert rs.kept_beats / rs.total_beats > 0.75
    assert rs.split_fraction < 0.12                 # rare in genuine AF


def test_existing_af_preservation_test_still_green():
    from tests.test_pressure_fixes import test_clean_runs_do_not_recreate_the_clean_rr_failure
    test_clean_runs_do_not_recreate_the_clean_rr_failure()


# ------------------------------------------ (2) beat evidence + AF gates
def _feats_from(times, agree):
    s = _series(times, agree=agree)
    rs = clean_runs(s, min_conf=0.5)
    f = compute_rhythm_features_from_runs(rs.runs, rs.run_confidences,
                                          rs.dropout_rate)
    ev = beat_evidence_from_series(s, rs, sqi_components={
        "cross_roi_coherence": float(np.clip((agree - 0.5) / 0.5, 0, 1))})
    return f, ev, s


def test_detection_error_irregularity_is_repeat_scan_not_afib():
    """The real-FP signature: 2-ROI-only beats (coherence ~0), harmonic
    half/double intervals, ~17 usable intervals, coverage ~0.5. Must be
    REPEAT_SCAN with explanatory reasons — never AFIB_SUGGESTIVE, and not
    OTHER_IRREGULAR either (artifact vs arrhythmia is undecidable)."""
    cfg = load_config()
    for seed in range(6):
        t = regular_with_errors(seed=seed, n_false=6, n_missed=3)
        f, ev, s = _feats_from(t, agree=0.5)
        r, why = decide_with_rationale(f, sqi=0.40, coverage=0.50, config=cfg,
                                       evidence=ev)
        assert r.outcome is not ScanOutcome.ACCEPT, (seed, why)
        assert r.predicted_class is None
        assert r.no_read_reasons
        assert why["gates_failed"], why


def test_low_coherence_alone_blocks_any_class():
    """Beats not seen consistently across the face are not verified beats:
    no class may be emitted, however tidy the intervals look."""
    cfg = load_config()
    t = np.cumsum(np.full(30, 0.85))
    f, ev, s = _feats_from(t, agree=0.5)         # perfect regular sinus, 2-ROI
    r = decide(f, 0.6, 0.9, cfg, evidence=ev)
    assert r.outcome is ScanOutcome.REPEAT_SCAN
    assert any("across the face" in w or "coheren" in w.lower()
               for w in r.no_read_reasons), r.no_read_reasons


def test_afib_call_requires_enough_verified_intervals_and_coverage():
    cfg = load_config()
    t = af_like(n=45)
    f, ev, s = _feats_from(t, agree=1.0)
    # good coverage + count: AF stands
    assert decide(f, 0.9, 0.9, cfg, evidence=ev).predicted_class == \
        "AFIB_SUGGESTIVE"
    # coverage below the any-class floor -> abstain
    r = decide(f, 0.9, 0.5, cfg, evidence=ev)
    assert r.outcome is not ScanOutcome.ACCEPT
    # too few intervals for a positive call -> abstain, with the reason
    t_short = af_like(n=14)
    f2, ev2, _ = _feats_from(t_short, agree=1.0)
    r2 = decide(f2, 0.9, 0.9, cfg, evidence=ev2)
    assert r2.predicted_class != "AFIB_SUGGESTIVE"
    assert r2.outcome is not ScanOutcome.ACCEPT
    assert any("interval" in w for w in r2.no_read_reasons)


def test_heavy_detection_error_burden_abstains_even_if_rhythm_looks_regular():
    """Coherent beats, but 40% of intervals are half/double detection
    errors. The T5 splitter recovers a regular residue — yet a recording
    that error-ridden supports NO rhythm statement: REPEAT_SCAN, with the
    detection burden named."""
    cfg = load_config()
    t = regular_with_errors(seed=3, n=40, n_false=8, n_missed=4)
    f, ev, s = _feats_from(t, agree=1.0)
    assert ev["harmonic_fraction"] > 0.15 and ev["split_fraction"] > 0.30
    r, why = decide_with_rationale(f, 0.8, 0.9, cfg, evidence=ev)
    assert r.predicted_class is None
    assert r.outcome is ScanOutcome.REPEAT_SCAN
    assert any("detection" in w for w in r.no_read_reasons), r.no_read_reasons


def test_harmonic_gate_blocks_afib_when_rule_fires_on_harmonics():
    """Direct test of the AF-call harmonic gate: features that trip the AF
    rule, evidence saying the irregularity is ½x/2x structured -> abstain."""
    cfg = load_config()
    f = RhythmFeatures({"n_intervals": 24.0, "median_abs_succ_diff": 120.0,
                        "pnn50": 0.7, "median_ibi": 1000.0, "mean_ibi": 1000.0,
                        "dropout_rate": 0.1, "irregularity_index": 0.12},
                       24, 0.9, [])
    ev = {"cross_roi_coherence": 0.9, "harmonic_fraction": 0.35,
          "split_fraction": 0.05, "n_beats": 30, "n_intervals": 24}
    r, why = decide_with_rationale(f, 0.8, 0.9, cfg, evidence=ev)
    assert r.predicted_class is None and r.outcome is ScanOutcome.REPEAT_SCAN
    assert "afib_harmonic_fraction" in why["gates_failed"]
    assert any("harmonic" in w for w in r.no_read_reasons)


def test_two_region_verified_evidence_yields_result():
    """v0.1.4.2 (real user report, screenshot 2026-08-25): decent home
    lighting often gives exactly TWO strong ROIs; the >=3-ROI coherence
    component then reads 0.00 BY CONSTRUCTION — indistinguishable from
    garbage. Two regions that place the same beats within the AF-grade
    timing budget (median |dt| <= 30 ms, matched >= 0.75) ARE independent
    verification; the real FP scans failed exactly there (36-49 ms,
    matched 0.63-0.75). Clean sinus + 2-ROI-verified evidence => result."""
    cfg = load_config()
    f = RhythmFeatures({"n_intervals": 24.0, "median_abs_succ_diff": 18.0,
                        "pnn50": 0.05, "median_ibi": 850.0, "mean_ibi": 850.0,
                        "dropout_rate": 0.05, "irregularity_index": 0.02},
                       24, 0.9, [])
    ev = {"cross_roi_coherence": 0.0, "harmonic_fraction": 0.05,
          "split_fraction": 0.05, "n_beats": 26, "n_intervals": 24,
          "timing_precision_ms": 27.0, "timing_matched_fraction": 0.85}
    r, why = decide_with_rationale(f, 0.40, 0.9, cfg, evidence=ev)
    assert r.outcome is ScanOutcome.ACCEPT, r.no_read_reasons
    assert r.predicted_class == "SINUS"
    assert any(g["name"] == "two_region_verification" and g["pass"]
               for g in why["gates"])


def test_fp_measured_profiles_still_refused_under_two_region_mode():
    """REGRESSION: the two real false-positive recordings' measured
    evidence (coherence 0.06/0.12, timing 36/49 ms, matched 0.75/0.63 —
    on record in spec B.10 / docs/READINESS.md) must STILL abstain and
    must never produce an AF call: the two-region mode demands the
    AF-grade timing budget they failed."""
    cfg = load_config()
    for coh, tp, tm in ((0.06, 36.0, 0.75), (0.12, 49.0, 0.63)):
        f = RhythmFeatures({"n_intervals": 22.0, "median_abs_succ_diff": 95.0,
                            "pnn50": 0.6, "median_ibi": 900.0,
                            "mean_ibi": 900.0, "dropout_rate": 0.2,
                            "irregularity_index": 0.2}, 22, 0.9, [])
        ev = {"cross_roi_coherence": coh, "harmonic_fraction": 0.1,
              "split_fraction": 0.1, "n_beats": 24, "n_intervals": 22,
              "timing_precision_ms": tp, "timing_matched_fraction": tm}
        r, why = decide_with_rationale(f, 0.5, 0.9, cfg, evidence=ev)
        assert r.predicted_class != "AFIB_SUGGESTIVE", (coh, tp, tm)
        assert r.outcome is not ScanOutcome.ACCEPT, (coh, tp, tm)


def test_af_callable_on_two_region_verified_evidence():
    """An AF pattern whose beats two regions place within the AF timing
    budget is verified irregularity — refusing it forever would make the
    system AF-blind exactly for users in ordinary home lighting (the P8
    abstention-funnelling failure mode, applied to capture conditions)."""
    cfg = load_config()
    f = RhythmFeatures({"n_intervals": 24.0, "median_abs_succ_diff": 95.0,
                        "pnn50": 0.6, "median_ibi": 900.0, "mean_ibi": 900.0,
                        "dropout_rate": 0.1, "irregularity_index": 0.2},
                       24, 0.9, [])
    ev = {"cross_roi_coherence": 0.0, "harmonic_fraction": 0.05,
          "split_fraction": 0.05, "n_beats": 26, "n_intervals": 24,
          "timing_precision_ms": 25.0, "timing_matched_fraction": 0.85}
    r, why = decide_with_rationale(f, 0.5, 0.9, cfg, evidence=ev)
    assert r.outcome is ScanOutcome.ACCEPT, r.no_read_reasons
    assert r.predicted_class == "AFIB_SUGGESTIVE"


def test_deficit_rule_unmoved_by_two_region_verification():
    """The pulse-deficit inference (dropout as AF evidence) still needs
    >= 3-region coherence 0.50 — timing-verified 2-ROI evidence must NOT
    re-enable it: deficit is inferred from ABSENT beats, which two
    regions cannot cross-verify."""
    cfg = load_config()
    f = RhythmFeatures({"n_intervals": 30.0, "median_abs_succ_diff": 45.0,
                        "pnn50": 0.30, "median_ibi": 900.0, "mean_ibi": 900.0,
                        "dropout_rate": 0.35, "irregularity_index": 0.1},
                       30, 0.9, [])
    ev = {"cross_roi_coherence": 0.0, "harmonic_fraction": 0.05,
          "split_fraction": 0.05, "n_beats": 32, "n_intervals": 30,
          "timing_precision_ms": 25.0, "timing_matched_fraction": 0.85}
    r, why = decide_with_rationale(f, 0.5, 0.9, cfg, evidence=ev)
    assert r.predicted_class != "AFIB_SUGGESTIVE"
    assert why["rule"].get("deficit_rule_suppressed") or \
        not why["rule"].get("afib_deficit_rule")


def test_deficit_rule_needs_high_coherence():
    """dropout_rate is pulse-deficit evidence ONLY under good signal; with
    weak coherence it is detection failure and must not upgrade to AF."""
    cfg = load_config()
    v = {"n_intervals": 30.0, "median_abs_succ_diff": 45.0, "pnn50": 0.35,
         "median_ibi": 850.0, "mean_ibi": 850.0, "dropout_rate": 0.4,
         "irregularity_index": 0.05}
    f = RhythmFeatures(v, 30, 0.9, [])
    ev_bad = {"cross_roi_coherence": 0.3, "harmonic_fraction": 0.05,
              "n_beats": 40, "n_intervals": 30, "split_fraction": 0.0}
    ev_good = dict(ev_bad, cross_roi_coherence=0.9)
    assert decide(f, 0.8, 0.9, cfg, evidence=ev_bad).predicted_class != \
        "AFIB_SUGGESTIVE"
    assert decide(f, 0.8, 0.9, cfg, evidence=ev_good).predicted_class == \
        "AFIB_SUGGESTIVE"


def test_genuine_af_with_verified_beats_still_reads_afib():
    """The guards must not kill true positives: coherent, well-covered AF
    with plenty of intervals is still AFIB_SUGGESTIVE."""
    cfg = load_config()
    for seed in (13, 21, 34):
        f, ev, s = _feats_from(af_like(seed=seed, n=45), agree=1.0)
        r = decide(f, 0.85, 0.85, cfg, evidence=ev)
        assert r.predicted_class == "AFIB_SUGGESTIVE", (seed, r.no_read_reasons)


def test_clean_sinus_with_verified_beats_reads_sinus():
    cfg = load_config()
    rng = np.random.default_rng(5)
    t = np.cumsum(np.clip(rng.normal(0.85, 0.03, 34), 0.5, 1.4))
    f, ev, s = _feats_from(t, agree=1.0)
    assert decide(f, 0.85, 0.9, cfg, evidence=ev).predicted_class == "SINUS"


# ------------------------------------------ (3) explainability
def test_rationale_traces_every_gate_and_the_rule():
    cfg = load_config()
    f, ev, s = _feats_from(af_like(n=45), agree=1.0)
    r, why = decide_with_rationale(f, 0.85, 0.85, cfg, evidence=ev)
    assert why["outcome"] == "ACCEPT" and why["predicted_class"] == "AFIB_SUGGESTIVE"
    names = {g["name"] for g in why["gates"]}
    for need in ("sqi", "coverage", "n_intervals", "cross_roi_coherence"):
        assert any(need in n for n in names), names
    assert all("pass" in g and "value" in g and "threshold" in g for g in why["gates"])
    assert why["rule"]["fired"] == "AFIB_SUGGESTIVE"
    assert "features" in why and "median_abs_succ_diff" in why["features"]
    assert "evidence" in why


# ------------------------------------------ (4) the actual recordings
@pytest.mark.skipif(not all(p.exists() for p in REAL_SCANS),
                    reason="the user's live recordings are not present")
def test_real_false_positive_recordings_now_abstain():
    """Regression on the two lossless live scans that produced the false
    positive. The pipeline must NOT emit a class on them: REPEAT_SCAN with
    reasons that name the mechanism."""
    from inference.pipeline import run_with_details
    for p in REAL_SCANS:
        res, det = run_with_details(str(p), manifest={
            "capture_profile": "consumer", "assume_rig_locks": False})
        assert res.predicted_class is None, (p.name, res)
        assert res.outcome in (ScanOutcome.REPEAT_SCAN, ScanOutcome.NO_RESULT)
        assert res.no_read_reasons, p.name
        why = det["rationale"]
        assert why["gates_failed"], why


# ------------------------------------------ (5) result provenance is single-path
def test_demo_result_comes_only_from_the_recorded_scan(tmp_path):
    """The engine's result must be re-derivable from the file it recorded:
    same outcome/class/reasons/rationale gates when the production pipeline
    is re-run on that file. No cached, mock or default result path exists."""
    import time
    from app.scan_engine import ScanSession, SessionState
    from capture.video_reader import iter_frames
    from scripts.make_synth_video import synth_video
    from inference.pipeline import run_with_details
    p = str(tmp_path / "s.avi")
    synth_video(p, kind="sinus", fps=30.0, duration_s=34.0, seed=44)
    frames = [(t, f) for t, f in iter_frames(p)]
    s = ScanSession("prov", str(tmp_path), fps_hint=30.0, scan_seconds=16.0)
    for i in range(0, 420, 15):
        b = frames[i:i + 15]; s.push_frames([f for _, f in b], [t for t, _ in b])
    assert s.status()["feedback"]["disposition"] == "READY"
    s.start_scan()
    for i in range(420, 900, 15):
        b = frames[i:i + 15]; s.push_frames([f for _, f in b], [t for t, _ in b])
    s.finish_scan()
    for _ in range(900):
        if s.state in (SessionState.DONE, SessionState.FAILED):
            break
        time.sleep(0.1)
    assert s.state is SessionState.DONE, s.error
    r = s.result
    assert r["recording_id"] == "live-prov"
    res2, det2 = run_with_details(s.video_path, manifest={
        "capture_profile": "consumer", "assume_rig_locks": False},
        recording_id="live-prov")
    assert res2.outcome.value == r["outcome"]
    assert res2.predicted_class == r["predicted_class"]
    assert list(res2.no_read_reasons) == r["no_read_reasons"]
    assert [g["name"] for g in det2["rationale"]["gates"]] == \
        [g["name"] for g in r["rationale"]["gates"]]
    assert [g["pass"] for g in det2["rationale"]["gates"]] == \
        [g["pass"] for g in r["rationale"]["gates"]]


# ------------------------------------------ (6) audit-confirmed detector/clock defects
def test_detector_prefers_the_higher_peak_inside_the_refractory_window():
    """A small pre-peak (dicrotic/noise) 150 ms before the true systolic
    peak must not win by arriving first: within the refractory window the
    detector keeps the HIGHER peak (audit finding, measured on real
    webcam waveforms: strong-ROI IBI IQR 199 -> 123 ms)."""
    from beats.detector import detect_beats_single_roi
    fps = 60.0
    t = np.arange(0, 20, 1 / fps)
    x = np.zeros_like(t)
    truth = []
    for k in range(1, 22):
        c = k * 0.9
        truth.append(c)
        x += 1.0 * np.exp(-((t - c) ** 2) / (2 * 0.06 ** 2))          # systole
        x += 0.45 * np.exp(-((t - (c - 0.15)) ** 2) / (2 * 0.03 ** 2))  # pre-peak
    x += np.random.default_rng(0).normal(0, 0.03, x.size)
    b = detect_beats_single_roi(x, fps, "r")
    det = np.array([q.t_s for q in b])
    truth = np.array(truth)
    err = np.array([np.min(np.abs(det - c)) for c in truth]) * 1000
    assert np.median(err) < 25, np.round(err).astype(int).tolist()


def test_cross_roi_polarity_is_consistent():
    """POS output has ONE physical polarity across skin ROIs; a weak ROI's
    own (noise-level) skewness must not flip it relative to the others."""
    from rppg.pos import orient_rois_consistently
    rng = np.random.default_rng(1)
    t = np.arange(0, 20, 1 / 30)
    pulse = np.exp(-((np.mod(t, 0.9) / 0.9 - 0.18) ** 2) / (2 * 0.075 ** 2))
    strong = pulse + rng.normal(0, 0.1, t.size)
    weak_flipped = -0.05 * pulse + rng.normal(0, 0.3, t.size)   # inverted, noisy
    out = orient_rois_consistently({"forehead": strong, "cheek_r": weak_flipped})
    assert np.corrcoef(out["forehead"], strong)[0, 1] > 0.99
    assert np.corrcoef(out["cheek_r"], pulse)[0, 1] > 0            # flipped back


def test_beat_times_follow_the_capture_clock_not_frame_index(tmp_path):
    """The sidecar is the honest clock: with a 1 s hole in capture time,
    beats after the hole must move by 1 s (index/fps would not)."""
    import json
    from scripts.make_synth_video import synth_video
    from inference.pipeline import run_with_details
    from capture.video_reader import probe_video
    p = str(tmp_path / "s.avi")
    synth_video(p, kind="sinus", fps=30.0, duration_s=16.0, seed=61)
    n = probe_video(p).n_frames
    ts = np.arange(n) / 30.0
    ts[n // 2:] += 1.0                                # 1 s capture hole
    with open(p + ".timestamps.json", "w") as f:
        json.dump({"timestamps_s": ts.tolist()}, f)
    _, det = run_with_details(p, manifest={"capture_profile": "consumer",
                                           "assume_rig_locks": False})
    beats = det["fused"].times()
    late = beats[beats > ts[n // 2] - 0.5]
    early = beats[beats < ts[n // 2] - 1.5]
    assert late.size and early.size
    # the hole appears as a real ~1.85 s gap between the last early beat and
    # the first late beat, and the run splitter breaks there
    assert late.min() - early.max() > 1.5
    assert det["runset"].n_missed_splits >= 1 or det["runset"].n_runs >= 2


# ------------------------------------------ (7) timing-precision proxy (Gate 1 without ECG)
def test_timing_precision_evidence_from_per_roi_trains():
    from inference.decision_logic import timing_precision_from_trains
    rng = np.random.default_rng(3)
    base = np.cumsum(np.full(30, 1.07))
    tight = {"forehead": base + rng.normal(0, 0.003, 30),
             "nose": base + rng.normal(0, 0.003, 30),
             "cheek_l": base[::2] + rng.normal(0, 0.05, 15),
             "cheek_r": np.array([])}
    tp = timing_precision_from_trains(tight)
    assert tp["timing_precision_ms"] < 10 and tp["timing_matched_fraction"] > 0.9
    loose = {"forehead": base + rng.normal(0, 0.06, 30),
             "nose": base + rng.normal(0, 0.06, 30),
             "cheek_l": base + rng.normal(0, 0.09, 30)}
    tp2 = timing_precision_from_trains(loose)
    assert tp2["timing_precision_ms"] > 40


def test_jittered_beats_cannot_produce_an_afib_call():
    """Real-recording residual: coherent-enough beats whose per-beat timing
    scatters ±150 ms trip the AF rule (med|d| 159, pNN50 0.85) with no
    harmonics. Timing precision must gate the call: REPEAT_SCAN."""
    cfg = load_config()
    f = RhythmFeatures({"n_intervals": 27.0, "median_abs_succ_diff": 159.0,
                        "pnn50": 0.85, "median_ibi": 1072.0, "mean_ibi": 1072.0,
                        "dropout_rate": 0.0, "irregularity_index": 0.15},
                       27, 0.9, [])
    ev = {"cross_roi_coherence": 0.45, "harmonic_fraction": 0.04,
          "split_fraction": 0.0, "n_beats": 28, "n_intervals": 27,
          "timing_precision_ms": 36.0, "timing_matched_fraction": 0.75}
    r, why = decide_with_rationale(f, 0.5, 0.94, cfg, evidence=ev)
    assert r.predicted_class is None and r.outcome is ScanOutcome.REPEAT_SCAN
    assert "afib_timing_precision" in why["gates_failed"]
    # precise beats with the same features: the AF call stands
    ev2 = dict(ev, timing_precision_ms=8.0, timing_matched_fraction=0.98)
    assert decide(f, 0.5, 0.94, cfg, evidence=ev2).predicted_class == "AFIB_SUGGESTIVE"
    # very poor timing blocks ANY class
    ev3 = dict(ev, timing_precision_ms=55.0)
    f_sinus = RhythmFeatures({"n_intervals": 27.0, "median_abs_succ_diff": 20.0,
                              "pnn50": 0.1, "median_ibi": 900.0, "mean_ibi": 900.0,
                              "dropout_rate": 0.0, "irregularity_index": 0.02},
                             27, 0.9, [])
    r3 = decide(f_sinus, 0.6, 0.9, cfg, evidence=ev3)
    assert r3.predicted_class is None and r3.outcome is ScanOutcome.REPEAT_SCAN
