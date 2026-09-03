"""
Integrity tests. These are evidence, not hygiene -- they run in CI on every
commit and their output is attached to evaluation reports.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from datasets.schema import (Recording, CaptureConfig, SyncRecord, SyncMethod,
                             RhythmAnnotation, Rhythm, Split, ScanResult,
                             ScanOutcome, MonkTone)
from datasets.splits import (make_participant_splits, assert_no_leakage,
                             LeakageError, assert_preprocessing_is_split_safe)
from evaluation.beat_metrics import (match_beats, ibi_agreement, estimate_ptt_ms,
                                     quantisation_floor_ms, decompose_rmssd_error,
                                     GATE1_CONTROLLED)


# ---------------------------------------------------------------- fixtures
def make_rec(pid, sid, rid, rhythms, split=Split.UNASSIGNED, crf=None,
             codec="ffv1", fps=60.0, lux=400.0, sync_ms=2.0, dur=90.0):
    return Recording(
        recording_id=rid, participant_id=pid, session_id=f"{pid}-s1", site_id=sid,
        video_path=f"/d/{rid}.mkv", ecg_path=f"/d/{rid}.edf",
        video_start_utc="2026-08-14T10:00:00Z", video_end_utc="2026-08-14T10:01:30Z",
        ecg_start_utc="2026-08-14T10:00:00Z", ecg_end_utc="2026-08-14T10:01:30Z",
        duration_s=dur,
        capture=CaptureConfig(phone_model="Pixel 9", os_version="15", camera="front",
                              width=1920, height=1080, nominal_fps=fps,
                              measured_fps_mean=fps, codec=codec, crf=crf,
                              exposure_locked=True, awb_locked=True, gain_locked=True,
                              beautification_disabled=True, illuminance_lux_mean=lux),
        sync=SyncRecord(method=SyncMethod.LED_FLASH_MARKER, offset_ms=0.0,
                        sync_uncertainty_ms=sync_ms, verified_at_end=True,
                        n_marker_events=24),
        rhythm_annotations=[
            RhythmAnnotation(0.0, dur, r, annotator_id="cardio_1", adjudicated=True)
            for r in rhythms],
        ecg_rpeaks_s=list(np.arange(0, dur, 0.85)),
        split=split,
    )


# ---------------------------------------------------------------- schema gates
def test_capture_rejects_high_crf():
    c = CaptureConfig("iPhone 15", "18", "front", 1920, 1080, 60, codec="h264",
                      crf=23, exposure_locked=True, awb_locked=True,
                      illuminance_lux_mean=400)
    ok, why = c.is_valid_for_beat_analysis()
    assert not ok and any("CRF" in w for w in why)


def test_capture_rejects_unlocked_exposure_and_low_light():
    c = CaptureConfig("iPhone 15", "18", "front", 1920, 1080, 60, codec="ffv1",
                      exposure_locked=False, awb_locked=True,
                      illuminance_lux_mean=40)
    ok, why = c.is_valid_for_beat_analysis()
    assert not ok
    assert any("white-balance" in w or "exposure" in w for w in why)
    assert any("illuminance" in w for w in why)


def test_capture_accepts_compliant_config():
    ok, why = CaptureConfig("Pixel 9", "15", "front", 1920, 1080, 60, codec="ffv1",
                            exposure_locked=True, awb_locked=True, gain_locked=True,
                            illuminance_lux_mean=500).is_valid_for_beat_analysis()
    assert ok, why


def test_sync_uncertainty_gate():
    s = SyncRecord(SyncMethod.NTP_SOFTWARE, 0.0, sync_uncertainty_ms=40.0,
                   verified_at_end=True)
    ok, why = s.is_valid_for_beat_analysis()
    assert not ok and any("uncertainty" in w for w in why)


def test_recording_level_label_is_derived_not_stored():
    r = make_rec("p1", "siteA", "r1", [Rhythm.SINUS, Rhythm.AFIB])
    assert r.recording_level_rhythm() is Rhythm.AFIB   # AF takes precedence


def test_user_facing_text_never_diagnoses():
    # Ban diagnostic ASSERTIONS. The word "diagnosis" is required in the
    # disclaimer, so a naive substring ban would forbid the safe phrasing --
    # the check must target the claim, not the vocabulary.
    banned_assertions = ["you have", "you are in", "diagnosed with",
                         "confirmed afib", "confirms", "is atrial fibrillation"]
    for cls in ["SINUS", "AFIB_SUGGESTIVE", "OTHER_IRREGULAR", "HIGH_RATE", None]:
        for outcome in ScanOutcome:
            txt = ScanResult("r", outcome, predicted_class=cls).user_facing_text().lower()
            assert not any(b in txt for b in banned_assertions), txt

    pos = ScanResult("r", ScanOutcome.ACCEPT,
                     predicted_class="AFIB_SUGGESTIVE").user_facing_text().lower()
    assert "not a diagnosis" in pos            # disclaimer is mandatory
    assert "clinician" in pos                  # escalation path is mandatory

    neg = ScanResult("r", ScanOutcome.ACCEPT,
                     predicted_class="SINUS").user_facing_text().lower()
    assert "cannot rule out" in neg            # no exclusion claim


# ---------------------------------------------------------------- leakage
def test_splits_are_participant_disjoint():
    recs = []
    for i in range(60):
        rh = [Rhythm.AFIB] if i % 3 == 0 else (
             [Rhythm.PAC_FREQUENT] if i % 3 == 1 else [Rhythm.SINUS])
        for k in range(3):                       # 3 recordings each
            recs.append(make_rec(f"p{i}", "siteA", f"r{i}_{k}", rh))
    plan = make_participant_splits(recs)
    a = plan.assignment()
    for r in recs:
        r.split = a[r.participant_id]
    rep = assert_no_leakage(recs)
    assert sum(rep["participants_per_split"].values()) == 60
    assert not (plan.train & plan.dev) and not (plan.train & plan.internal_test)


def test_leakage_detected_when_participant_spans_splits():
    recs = [make_rec("p1", "siteA", "r1", [Rhythm.AFIB], Split.TRAIN),
            make_rec("p1", "siteA", "r2", [Rhythm.AFIB], Split.INTERNAL_TEST),
            make_rec("p2", "siteA", "r3", [Rhythm.PAC], Split.INTERNAL_TEST)]
    with pytest.raises(LeakageError, match="both"):
        assert_no_leakage(recs)


def test_external_test_must_be_site_disjoint():
    recs = [make_rec("p1", "siteA", "r1", [Rhythm.AFIB], Split.TRAIN),
            make_rec("p2", "siteA", "r2", [Rhythm.AFIB], Split.EXTERNAL_TEST),
            make_rec("p3", "siteA", "r3", [Rhythm.PAC], Split.EXTERNAL_TEST)]
    with pytest.raises(LeakageError, match="site"):
        assert_no_leakage(recs)


def test_test_split_without_hard_negatives_is_rejected():
    recs = [make_rec("p1", "siteA", "r1", [Rhythm.AFIB], Split.TRAIN),
            make_rec("p2", "siteA", "r2", [Rhythm.AFIB], Split.INTERNAL_TEST),
            make_rec("p3", "siteA", "r3", [Rhythm.SINUS], Split.INTERNAL_TEST)]
    with pytest.raises(LeakageError, match="hard-negative"):
        assert_no_leakage(recs)


def test_preprocessing_fitted_outside_train_is_rejected():
    recs = [make_rec("p1", "siteA", "r1", [Rhythm.AFIB], Split.TRAIN),
            make_rec("p2", "siteA", "r2", [Rhythm.AFIB], Split.DEV)]
    with pytest.raises(LeakageError, match="fitted"):
        assert_preprocessing_is_split_safe(recs)


def test_split_assignment_is_stable_as_dataset_grows():
    base = [make_rec(f"p{i}", "siteA", f"r{i}",
                     [Rhythm.AFIB] if i % 2 else [Rhythm.PAC]) for i in range(40)]
    grown = base + [make_rec(f"p{i}", "siteA", f"r{i}",
                             [Rhythm.AFIB] if i % 2 else [Rhythm.PAC])
                    for i in range(40, 60)]
    a1 = make_participant_splits(base).assignment()
    a2 = make_participant_splits(grown).assignment()
    moved = [p for p in a1 if a2[p] is not a1[p]]
    # Some drift is unavoidable with proportional strata; demand it be small.
    assert len(moved) == 0, f"{len(moved)}/{len(a1)} participants moved when dataset grew"


# ---------------------------------------------------------------- beat metrics
def _synth(rr, ptt_ms=200.0, jitter_ms=0.0, drop=(), extra=(), seed=0):
    rng = np.random.default_rng(seed)
    ref = np.cumsum(rr)
    det = ref + ptt_ms / 1000.0
    if jitter_ms:
        det = det + rng.normal(0, jitter_ms / 1000.0, det.shape)
    keep = [i for i in range(len(det)) if i not in set(drop)]
    det = det[keep]
    if len(extra):
        det = np.sort(np.concatenate([det, np.asarray(extra)]))
    return ref, det


def test_perfect_detection_scores_perfectly():
    rr = np.full(60, 0.85)
    ref, det = _synth(rr)
    m = match_beats(ref, det, tolerance_ms=50)
    assert m.n_matched == len(ref)
    assert m.sensitivity == 1.0 and m.ppv == 1.0
    assert abs(m.ptt_estimate_ms - 200.0) < 5.0
    ibi = ibi_agreement(ref, det, m)
    assert ibi.ibi_mae_ms < 1e-6


def test_missed_beats_are_counted_and_excluded_from_ibi():
    rr = np.full(60, 0.85)
    ref, det = _synth(rr, drop=(10, 11, 30))
    m = match_beats(ref, det, tolerance_ms=50)
    assert m.n_matched == len(ref) - 3
    assert abs(m.missed_beat_rate - 3 / 60) < 1e-9
    ibi = ibi_agreement(ref, det, m)
    assert ibi.n_excluded_spanning_gap >= 2      # gaps must not pollute MAE
    assert ibi.ibi_mae_ms < 1e-6                 # surviving intervals are exact


def test_false_beats_reduce_ppv_not_sensitivity():
    rr = np.full(40, 0.85)
    ref, det = _synth(rr)
    det = np.sort(np.concatenate([det, det[:5] + 0.30]))   # 5 spurious beats
    m = match_beats(ref, det, tolerance_ms=50)
    assert m.sensitivity == 1.0
    assert m.ppv < 1.0 and abs(m.false_beat_rate - 5 / 45) < 1e-9


def test_matching_is_one_to_one():
    ref = np.array([1.0, 1.02, 1.04])
    det = np.array([1.0])                      # one detection, three refs nearby
    m = match_beats(ref, det, tolerance_ms=100, ptt_ms=0.0, auto_ptt=False)
    assert m.n_matched == 1                    # must not match the same beat 3x


def test_ptt_is_estimated_and_removed():
    rr = np.full(50, 0.8)
    ref, det = _synth(rr, ptt_ms=260.0)
    est, iqr = estimate_ptt_ms(ref, det)
    assert abs(est - 260.0) < 6.0
    m = match_beats(ref, det, tolerance_ms=50)
    assert m.n_matched == len(ref)             # would be 0 without PTT removal


def test_af_like_irregularity_is_preserved_not_smoothed():
    rng = np.random.default_rng(3)
    rr = np.clip(rng.normal(0.60, 0.16, 90), 0.30, 1.30)
    ref, det = _synth(rr, jitter_ms=8.0, seed=1)
    m = match_beats(ref, det, tolerance_ms=100)
    ibi = ibi_agreement(ref, det, m)
    assert ibi.ref_rmssd_ms > 100                       # genuinely AF-like
    assert abs(ibi.rmssd_error_ms) < 40                 # not collapsed toward sinus
    assert ibi.est_rmssd_ms > 100                       # still reads as irregular


def test_quantisation_floor_matches_closed_form():
    q30, q60 = quantisation_floor_ms(30), quantisation_floor_ms(60)
    assert abs(q30["sd_peak_ms"] - (1000 / 30) / np.sqrt(12)) < 1e-9
    assert abs(q30["sd_dibi_ms"] / q30["sd_ibi_ms"] - np.sqrt(3)) < 1e-9
    assert abs(q60["sd_dibi_ms"] * 2 - q30["sd_dibi_ms"]) < 1e-9   # halves with fps


def test_rmssd_decomposition_attributes_most_error_to_extractor():
    d = decompose_rmssd_error(true_rmssd_ms=33.8, measured_rmssd_ms=65.9, fps=30)
    assert d["extractor_ms"] > d["quantisation_ms"]
    assert 0.75 < d["extractor_share_of_variance"] < 0.90


def test_gate_blocks_a_bad_extractor():
    rr = np.full(60, 0.85)
    ref, det = _synth(rr, jitter_ms=60.0, seed=5)          # deliberately poor
    m50 = match_beats(ref, det, tolerance_ms=50)
    ibi = ibi_agreement(ref, det, m50)
    ok, detail = GATE1_CONTROLLED.evaluate(m50, ibi)
    assert not ok
    assert any(not v["pass"] for v in detail.values())


def test_gate_passes_a_good_extractor():
    rr = np.full(60, 0.85) + np.random.default_rng(9).normal(0, 0.03, 60)
    ref, det = _synth(rr, jitter_ms=6.0, seed=9)
    m50 = match_beats(ref, det, tolerance_ms=50)
    ibi = ibi_agreement(ref, det, m50)
    ok, detail = GATE1_CONTROLLED.evaluate(m50, ibi)
    assert ok, detail
