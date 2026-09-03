"""v0.6 flutter T4 — the ground-truth label schema. Flutter cannot be
named from a pulse, so the LABEL carries the whole naming burden:
12-lead, EP-level adjudication, atrial rate, conduction ratio and type.
This file pins that the bar is real and fails closed, and that the
sanctioned regular-tachy sentence names no rhythm."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dataclasses

import pytest

from datasets.schema import (ConductionRatio, FlutterType, MeasurementClass,
                             REGULAR_TACHY_SENTENCES, Rhythm,
                             RhythmAnnotation, flutter_label_problems)


def _ann(**over):
    d = dict(t_start_s=0.0, t_end_s=40.0, rhythm=Rhythm.ATRIAL_FLUTTER,
             annotator_id="reader1", adjudicated=True,
             atrial_rate_bpm=300.0,
             conduction_ratio=ConductionRatio.TWO_TO_ONE,
             flutter_type=FlutterType.TYPICAL,
             adjudicator_id="ep-7", adjudication_leads=12)
    d.update(over)
    return RhythmAnnotation(**d)


def test_conduction_ratio_divisors_and_variable_has_none():
    assert ConductionRatio.TWO_TO_ONE.divisor == 2.0
    assert ConductionRatio.THREE_TO_ONE.divisor == 3.0
    assert ConductionRatio.FOUR_TO_ONE.divisor == 4.0
    # VARIABLE block has no single divisor — inventing one would
    # manufacture an atrial rate the data cannot carry
    assert ConductionRatio.VARIABLE.divisor is None
    for r in ConductionRatio:
        d = r.divisor
        if d is not None:
            assert 300.0 / d == pytest.approx({2.0: 150.0, 3.0: 100.0,
                                               4.0: 75.0}[d])


def test_a_complete_flutter_label_is_usable():
    assert flutter_label_problems(_ann()) == []


@pytest.mark.parametrize("over,needle", [
    ({"adjudicated": False}, "not adjudicated"),
    ({"adjudicator_id": "  "}, "adjudicator identity"),
    ({"adjudication_leads": 1}, "1 lead(s)"),
    ({"adjudication_leads": 6}, "6 lead(s)"),
    ({"adjudication_leads": None}, "lead count not recorded"),
    ({"conduction_ratio": None}, "conduction ratio"),
    ({"flutter_type": None}, "flutter type"),
    ({"atrial_rate_bpm": None}, "no atrial rate"),
    ({"atrial_rate_bpm": 120.0}, "outside the plausible window"),
    ({"atrial_rate_bpm": 900.0}, "outside the plausible window"),
    ({"atrial_rate_bpm": "fast"}, "not a number"),
])
def test_incomplete_flutter_labels_fail_closed(over, needle):
    bad = flutter_label_problems(_ann(**over))
    assert bad, over
    assert any(needle in b for b in bad), (over, bad)


def test_single_lead_is_fine_for_a_negative_but_not_for_a_positive():
    # the 12-lead rule governs the POSITIVE label only: a single-lead
    # recording stays a perfectly good rate/regularity comparator
    sinus = RhythmAnnotation(0.0, 40.0, Rhythm.SINUS_TACHYCARDIA,
                             annotator_id="reader1", adjudicated=True)
    assert flutter_label_problems(sinus) == ["not an ATRIAL_FLUTTER "
                                             "annotation"]
    # lead count falls back to the recording's ECG when the reader did
    # not record their own — and a 1-lead recording then fails
    partial = _ann(adjudication_leads=None)
    assert flutter_label_problems(partial, ecg_leads=12) == []
    assert any("1 lead(s)" in b for b in
               flutter_label_problems(partial, ecg_leads=1))


def test_recording_level_helper_reports_the_best_annotation():
    from scripts.make_synth_flutter import _recording_manifest
    from datasets.schema import CaptureConfig, SyncMethod, SyncRecord

    def _rec(anns, leads):
        cap = CaptureConfig(
            phone_model="x", os_version="n/a", camera="front", width=320,
            height=240, nominal_fps=30.0, measured_fps_mean=30.0,
            measured_fps_jitter_ms=0.1, codec="ffv1", crf=None,
            exposure_locked=True, awb_locked=True, gain_locked=True,
            beautification_disabled=True, illuminance_lux_mean=500.0,
            mount="tripod")
        from datasets.schema import Recording, Split
        return Recording(
            recording_id="r", participant_id="p", session_id="s",
            site_id="siteA", video_path="v.avi", ecg_path="e.json",
            video_start_utc="2026-09-01T09:00:00Z",
            video_end_utc="2026-09-01T09:00:40Z",
            ecg_start_utc="2026-09-01T09:00:00Z",
            ecg_end_utc="2026-09-01T09:00:40Z", duration_s=40.0,
            capture=cap,
            sync=SyncRecord(SyncMethod.LED_FLASH_MARKER, offset_ms=0.0,
                            sync_uncertainty_ms=1.0, drift_ppm=1.0,
                            verified_at_end=True, n_marker_events=32),
            ecg_leads=leads, rhythm_annotations=anns,
            split=Split.UNASSIGNED)

    # no flutter annotation: says so plainly — "no problems" must never
    # be readable as "confirmed flutter"
    sinus = RhythmAnnotation(0.0, 40.0, Rhythm.SINUS, annotator_id="r",
                             adjudicated=True)
    assert _rec([sinus], 12).flutter_label_problems() == [
        "no ATRIAL_FLUTTER annotation on this recording"]
    assert _rec([sinus, _ann()], 12).flutter_label_problems() == []
    # the least-broken annotation is the one reported, with its index
    bad = _rec([_ann(adjudicated=False, adjudicator_id=None),
                _ann(adjudicated=False)], 12).flutter_label_problems()
    assert bad and all(b.startswith("annotation 1:") for b in bad)
    assert len(bad) == 1 and "not adjudicated" in bad[0]
    # a 1-lead recording cannot carry a positive even when adjudicated
    assert any("lead" in b for b in
               _rec([_ann(adjudication_leads=None)], 1)
               .flutter_label_problems())


def test_annotation_fields_are_additive_and_default_empty():
    """Every pre-v0.6 annotation stays constructible and reads None."""
    a = RhythmAnnotation(0.0, 10.0, Rhythm.SINUS, annotator_id="r")
    for f in ("atrial_rate_bpm", "conduction_ratio", "flutter_type",
              "adjudicator_id", "adjudication_leads"):
        assert getattr(a, f) is None, f
    names = {f.name for f in dataclasses.fields(RhythmAnnotation)}
    assert {"t_start_s", "t_end_s", "rhythm", "annotator_id"} <= names


def test_flutter_stays_an_afib_hard_negative():
    """The track detects flutter; it does NOT stop being the thing an
    AF detector must not call AF."""
    assert Rhythm.ATRIAL_FLUTTER in Rhythm.hard_negatives()
    assert Rhythm.SVT in Rhythm.hard_negatives()


def test_research_rhythm_class_exists_and_is_a_research_class():
    assert MeasurementClass.RESEARCH_RHYTHM.value == "RESEARCH_RHYTHM"
    # the payload filter keys off the RESEARCH_ prefix — that is what
    # makes an ungated flutter output unrenderable by construction
    assert MeasurementClass.RESEARCH_RHYTHM.value.startswith("RESEARCH_")
    from datasets.schema import public_head_results
    rows = [{"head": "afib", "measurement_class": "INFERRED_RHYTHM"},
            {"head": "flutter", "measurement_class": "RESEARCH_RHYTHM"}]
    assert [r["head"] for r in public_head_results(rows)] == ["afib"]


def test_sanctioned_regular_tachy_sentence_names_no_rhythm():
    s = REGULAR_TACHY_SENTENCES["regular_tachy"]
    assert set(REGULAR_TACHY_SENTENCES) == {"regular_tachy"}
    low = s.lower()
    for banned in ("flutter", "svt", "atrial tachycardia", "fibrillation",
                   "2:1", "conduction", "diagnos"):
        assert banned not in low, banned
    # it must route to the one instrument that CAN name the rhythm
    assert "ecg" in low
    assert "regular" in low and "fast" in low
