"""
Regression tests for the v2 pressure-test fixes (findings P1-P8).

Each test encodes a defect that was CONFIRMED empirically during the pressure
test, so a regression re-introduces a known, demonstrated failure mode.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from datasets.schema import (Recording, CaptureConfig, SyncRecord, SyncMethod,
                             RhythmAnnotation, Rhythm, Split)
from datasets.splits import make_participant_splits, split_composition
from beats.detector import Beat, BeatSeries
from beats.ibi import clean_runs, rmssd_from_runs
from features.rhythm import compute_rhythm_features_from_runs
from evaluation.beat_metrics import (match_beats, production_rmssd_error_ms,
                                     beat_confidence_calibration,
                                     missed_beat_flag_recall)
from evaluation.afib_metrics import serial_confirmation, no_read_report


# ------------------------------------------------------------------ fixtures
def rec(pid, rhythms, site="siteA"):
    return Recording(
        recording_id=f"r_{pid}", participant_id=pid, session_id=f"{pid}-s1",
        site_id=site, video_path="v", ecg_path="e",
        video_start_utc="t", video_end_utc="t", ecg_start_utc="t",
        ecg_end_utc="t", duration_s=90,
        capture=CaptureConfig("Pixel", "15", "front", 1920, 1080, 60,
                              codec="ffv1", exposure_locked=True,
                              awb_locked=True, illuminance_lux_mean=500),
        sync=SyncRecord(SyncMethod.LED_FLASH_MARKER, 0.0, 2.0,
                        verified_at_end=True, n_marker_events=24),
        rhythm_annotations=[RhythmAnnotation(0, 90, r, "c1", adjudicated=True)
                            for r in rhythms],
        ecg_rpeaks_s=list(np.arange(0, 90, 0.85)))


def sinus_series(n=106, mean_rr=0.85, sd=0.02, conf=0.9, seed=3) -> BeatSeries:
    rng = np.random.default_rng(seed)
    rr = np.clip(rng.normal(mean_rr, sd, n), 0.5, 1.4)
    t = np.cumsum(rr)
    beats = [Beat(t_s=float(ti), confidence=conf, roi_agreement=1.0,
                  signal_quality=0.9, amplitude=1.0, prominence=1.0)
             for ti in t]
    return BeatSeries(beats, 60.0, float(t[-1])), rr


# ------------------------------------------------- P7: split label-flip bug
def test_label_flip_does_not_move_participant():
    """A paroxysmal-AF patient whose AF is first captured on visit 3 must NOT
    migrate splits. v1 moved 9/120 (2 of them TRAIN->TEST) on this exact
    scenario; v2 hashes participant identity alone."""
    before = [rec(f"p{i}", [Rhythm.SINUS]) for i in range(120)]
    after = [rec(f"p{i}", [Rhythm.SINUS, Rhythm.AFIB] if i % 4 == 0
                 else [Rhythm.SINUS]) for i in range(120)]
    a1 = make_participant_splits(before).assignment()
    a2 = make_participant_splits(after).assignment()
    moved = [p for p in a1 if a2[p] is not a1[p]]
    assert moved == [], f"label evolution moved {len(moved)} participants"


def test_growth_still_moves_nobody():
    base = [rec(f"p{i}", [Rhythm.AFIB] if i % 2 else [Rhythm.PAC])
            for i in range(40)]
    grown = base + [rec(f"p{i}", [Rhythm.SINUS]) for i in range(40, 70)]
    a1 = make_participant_splits(base).assignment()
    a2 = make_participant_splits(grown).assignment()
    assert all(a2[p] is a1[p] for p in a1)


def test_composition_is_reported():
    recs = [rec(f"p{i}", [Rhythm.AFIB] if i % 3 == 0 else [Rhythm.SINUS])
            for i in range(30)]
    a = make_participant_splits(recs).assignment()
    for r in recs:
        r.split = a[r.participant_id]
    comp = split_composition(recs)
    assert sum(c["participants"] for c in comp.values()) == 30
    assert any(c["af_participants"] > 0 for c in comp.values())


# ---------------------------------------- P1/P6: clean runs rescue features
def test_raw_series_rmssd_is_destroyed_by_detection_errors():
    """Reproduce P1: false + missed beats push raw sinus RMSSD past the AF
    boundary. This test documents the failure the clean-run path fixes."""
    series, rr = sinus_series()
    t = series.times().tolist()
    # inject 6 false beats (low confidence) and delete 3 true beats
    rng = np.random.default_rng(5)
    beats = list(series.beats)
    for j in sorted(rng.choice(len(beats) - 2, 6, replace=False), reverse=True):
        mid = (beats[j].t_s + beats[j + 1].t_s) / 2
        beats.insert(j + 1, Beat(t_s=mid, confidence=0.25, roi_agreement=0.25,
                                 signal_quality=0.4, amplitude=0.3, prominence=0.2))
    for j in sorted(rng.choice(len(beats) - 2, 3, replace=False), reverse=True):
        del beats[j]
    corrupted = BeatSeries(sorted(beats, key=lambda b: b.t_s), 60.0, series.duration_s)

    raw_rmssd = float(np.sqrt(np.mean(np.diff(corrupted.ibi_ms()) ** 2)))
    true_rmssd = float(np.sqrt(np.mean(np.diff(rr * 1000) ** 2)))
    assert true_rmssd < 45
    assert raw_rmssd > 100, "expected raw-series RMSSD past the AF boundary"

    rs = clean_runs(corrupted, min_conf=0.5, min_run_beats=4)
    clean_rmssd = rmssd_from_runs(rs)
    assert abs(clean_rmssd - true_rmssd) < 15, (
        f"clean-run RMSSD {clean_rmssd:.1f} vs true {true_rmssd:.1f}")


def test_production_gate_metric_ties_paths_together():
    series, rr = sinus_series(seed=11)
    ref = np.cumsum(rr)
    rs = clean_runs(series, min_conf=0.5)
    out = production_rmssd_error_ms(ref, rmssd_from_runs(rs))
    assert abs(out["error_ms"]) < 10


def test_run_features_never_cross_boundaries():
    """Two clean runs separated by a gap: successive diffs must not include
    the cross-gap interval (which would manufacture irregularity)."""
    run1 = [Beat(t_s=0.85 * i, confidence=0.9, roi_agreement=1, signal_quality=0.9,
                 amplitude=1, prominence=1) for i in range(1, 30)]
    run2 = [Beat(t_s=30.0 + 0.85 * i, confidence=0.9, roi_agreement=1,
                 signal_quality=0.9, amplitude=1, prominence=1) for i in range(1, 30)]
    gapbeat = Beat(t_s=27.0, confidence=0.1, roi_agreement=0.25,
                   signal_quality=0.2, amplitude=0.2, prominence=0.1)
    series = BeatSeries(run1 + [gapbeat] + run2, 60.0, 60.0)
    rs = clean_runs(series, min_conf=0.5, min_run_beats=4)
    assert rs.n_runs == 2
    d = rs.within_run_diffs()
    assert np.all(np.abs(d) < 50), "a cross-run diff leaked into the statistics"
    f = compute_rhythm_features_from_runs(rs.runs, rs.run_confidences,
                                          rs.dropout_rate)
    assert f.values["rmssd"] < 20
    assert f.values["n_runs"] == 2


def test_fragmented_series_suppresses_sequence_features():
    runs = [np.full(20, 850.0) + np.random.default_rng(1).normal(0, 10, 20)
            for _ in range(4)]                     # 80 intervals, longest run 20
    f = compute_rhythm_features_from_runs(runs)
    assert np.isnan(f.values["sample_entropy"])    # needs 60 CONTIGUOUS
    assert any("longest clean run" in w for w in f.estimator_warnings)


def test_clean_runs_do_not_recreate_the_clean_rr_failure():
    """THE self-consistency check: the prior report condemned open-rppg's
    clean_rr for deleting 48% of AF intervals and collapsing RMSSD 230->66 ms.
    Our run-splitter must NOT do the same. AF RMSSD after cleaning must retain
    the bulk of the true irregularity and stay far above the sinus boundary."""
    rng = np.random.default_rng(13)
    rr = np.clip(rng.normal(0.62, 0.17, 145), 0.28, 1.35)      # AF-like
    t = np.cumsum(rr)
    beats = [Beat(t_s=float(ti), confidence=0.9, roi_agreement=1.0,
                  signal_quality=0.9, amplitude=1.0, prominence=1.0) for ti in t]
    series = BeatSeries(beats, 60.0, float(t[-1]))
    true_rmssd = float(np.sqrt(np.mean(np.diff(rr * 1000) ** 2)))
    rs = clean_runs(series, min_conf=0.5)
    clean = rmssd_from_runs(rs)
    assert true_rmssd > 150
    assert clean > 0.6 * true_rmssd, f"over-cleaning: {clean:.0f} vs {true_rmssd:.0f}"
    assert clean > 100, "cleaned AF must remain past the AF decision region"
    assert rs.kept_beats / rs.total_beats > 0.75, "must not discard the recording"


# ------------------------------------------------- P3: serial confirmation
def test_serial_confirmation_persistent_share_limits():
    se, sp = 0.85, 0.94
    se0, sp0 = serial_confirmation(se, sp, 2, 3, fp_persistent_share=0.0)
    se1, sp1 = serial_confirmation(se, sp, 2, 3, fp_persistent_share=1.0)
    assert sp0 > 0.985                             # optimistic independence
    assert abs(sp1 - sp) < 1e-9, "fully persistent FPs: confirmation must do nothing"
    assert abs(se1 - se0) < 1e-9                   # sensitivity model unchanged
    _, sp_half = serial_confirmation(se, sp, 2, 3, fp_persistent_share=0.5)
    assert sp < sp_half < sp0                      # monotone in between


# ------------------------------------------------- P4/P5: capture and sync
def test_single_flash_sync_is_rejected():
    s = SyncRecord(SyncMethod.LED_FLASH_MARKER, 0.0, 2.0,
                   verified_at_end=True, n_marker_events=1)
    ok, why = s.is_valid_for_beat_analysis()
    assert not ok and any("marker" in w for w in why)


def test_cross_correlation_sync_is_rejected():
    s = SyncRecord(SyncMethod.CROSS_CORRELATION, 0.0, 2.0, verified_at_end=True)
    ok, why = s.is_valid_for_beat_analysis()
    assert not ok and any("circular" in w for w in why)


def test_hardware_bitrate_capture_accepted_without_crf():
    c = CaptureConfig("iPhone 15", "18", "front", 1920, 1080, 60,
                      codec="hevc", crf=None, bitrate_mbps=50,
                      exposure_locked=True, awb_locked=True,
                      illuminance_lux_mean=500)
    ok, why = c.is_valid_for_beat_analysis()
    assert ok, why


def test_low_bitrate_capture_rejected():
    c = CaptureConfig("BudgetPhone", "14", "front", 1920, 1080, 60,
                      codec="h264", crf=None, bitrate_mbps=6,
                      exposure_locked=True, awb_locked=True,
                      illuminance_lux_mean=500)
    ok, why = c.is_valid_for_beat_analysis()
    assert not ok and any("bpp" in w for w in why)


def test_lossy_with_nothing_recorded_rejected():
    c = CaptureConfig("X", "1", "front", 1920, 1080, 60, codec="h264",
                      exposure_locked=True, awb_locked=True,
                      illuminance_lux_mean=500)
    ok, why = c.is_valid_for_beat_analysis()
    assert not ok and any("neither CRF nor bitrate" in w for w in why)


# ------------------------------------------------- P8: abstention parity
def test_no_read_parity_flags_selective_abstention():
    n = 300
    band = np.array(["1-4"] * 100 + ["5-7"] * 100 + ["8-10"] * 100)
    nr = np.zeros(n, bool)
    nr[200:] = np.random.default_rng(2).random(100) < 0.45   # dark band abstains
    nr[:200] = np.random.default_rng(3).random(200) < 0.08
    ok, rep = no_read_report(nr, {"monk_band": band})
    assert not ok
    assert any("8-10" in f for f in rep["failures"])


def test_no_read_parity_passes_when_uniform():
    n = 300
    band = np.array(["1-4", "5-7", "8-10"])[np.random.default_rng(4).integers(0, 3, n)]
    nr = np.random.default_rng(5).random(n) < 0.15
    ok, rep = no_read_report(nr, {"monk_band": band})
    assert ok, rep["failures"]


def test_af_conditional_no_read_is_visible():
    rhythm = np.array(["AFIB"] * 80 + ["SINUS"] * 220)
    nr = np.concatenate([np.random.default_rng(6).random(80) < 0.40,
                         np.random.default_rng(7).random(220) < 0.10])
    ok, rep = no_read_report(nr, {"rhythm": rhythm})
    assert not ok, "pulse-deficit-driven AF abstention must be flagged"


# ------------------------------------------------- P12: confidence honesty
def test_beat_confidence_calibration_separates_honest_from_dishonest():
    rng = np.random.default_rng(8)
    ref = np.cumsum(np.full(80, 0.85))
    det = np.concatenate([ref + 0.2, ref[:20] + 0.2 + 0.3])   # 20 false beats
    det = np.sort(det)
    m = match_beats(ref, det, tolerance_ms=60)
    is_false = np.zeros(det.size, bool)
    is_false[np.isin(np.round(det, 4),
                     np.round(ref[:20] + 0.5, 4))] = True
    honest = np.where(is_false, 0.15, 0.92) + rng.normal(0, 0.02, det.size)
    dishonest = np.full(det.size, 0.9)
    cal_h = beat_confidence_calibration(m, np.clip(honest, 0, 1))
    cal_d = beat_confidence_calibration(m, dishonest)
    assert cal_h["ece"] < cal_d["ece"] - 0.05


def test_missed_beat_flag_recall():
    ref = np.cumsum(np.full(60, 0.85))
    keep = np.ones(60, bool); keep[[20, 40]] = False           # two missed beats
    det = ref[keep] + 0.2
    m = match_beats(ref, det, tolerance_ms=60)
    # the merged intervals are at detected indices 19 and 38 (after removals)
    flagged = [19, 38]
    out = missed_beat_flag_recall(ref, det, m, flagged)
    assert out["n_missed"] == 2
    assert out["flag_recall"] == 1.0
    out_none = missed_beat_flag_recall(ref, det, m, [])
    assert out_none["flag_recall"] == 0.0
