"""v0.4 T0 — the additive schema rev: INFERRED_FITNESS measurement class
(§V-gated rendering), SessionResult (classified, fenced, migratable),
challenge/participant_context manifest blocks, CPET label fail-closed
reader. A VO2 number must never appear in any sanctioned string."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dataclasses
import json
import re

import pytest

from datasets.io import recording_from_json
from datasets.schema import (ChallengeRecord, CpetLabel, Medications,
                             MeasurementClass, ParticipantContext,
                             ScanOutcome, SESSION_SENTENCES, SessionResult,
                             cpet_from_dict, session_result_from_dict)


def test_inferred_fitness_member_and_gating_docstring():
    assert MeasurementClass.INFERRED_FITNESS.value == "INFERRED_FITNESS"
    doc = MeasurementClass.__doc__
    assert "§V" in doc and "signed" in doc      # the rendering invariant


def test_every_sessionresult_field_is_classified():
    assert set(SessionResult.FIELD_CLASSES) == \
        set(SessionResult.__dataclass_fields__)
    for name, cls in SessionResult.FIELD_CLASSES.items():
        assert isinstance(cls, MeasurementClass), name
        assert not cls.value.startswith("RESEARCH_"), name
        assert cls is not MeasurementClass.RESEARCH_SYNTHETIC, name
        assert cls is not MeasurementClass.RESEARCH_VASCULAR, name
        assert cls is not MeasurementClass.RESEARCH_RHYTHM, name
    fitness = {n for n, c in SessionResult.FIELD_CLASSES.items()
               if c is MeasurementClass.INFERRED_FITNESS}
    assert fitness == {"fitness_category", "trend"}


def test_fitness_fields_default_none_and_stay_none_by_default():
    r = SessionResult("s1", ScanOutcome.ACCEPT, activity_performed=True)
    assert r.fitness_category is None and r.trend is None


def test_session_round_trip_and_fail_closed():
    r = SessionResult("s1", ScanOutcome.ACCEPT, activity_performed=True,
                      hrr60_bpm=24.5, confidence_stars=4,
                      protocol_id="sts_1min")
    d = json.loads(json.dumps(dataclasses.asdict(r), default=str))
    r2 = session_result_from_dict(d)
    assert r2.hrr60_bpm == 24.5 and r2.outcome is ScanOutcome.ACCEPT
    assert r2.schema_version == 1
    with pytest.raises(ValueError, match="newer"):
        session_result_from_dict(dict(d, schema_version=2))
    with pytest.raises(ValueError, match="unknown"):
        session_result_from_dict(dict(d, vo2max=42))


def test_session_sentences_fenced_and_number_free():
    for key, s in SESSION_SENTENCES.items():
        low = s.lower()
        for b in ("your ecg", "ecg waveform", "inferred ecg",
                  "we generated", "synthesized ecg", "synthetic ecg"):
            assert b not in low, key
        assert "diagnos" not in low or "not a" in low, key
        assert not re.search(r"\d", s), (key, "sanctioned session "
                                         "sentences carry no numbers")
        assert not re.search(r"m[lL]\s*/\s*kg\s*/\s*min", s), key
        assert "fitness rating" not in low or "not a fitness rating" in low


def test_user_facing_text_state_machine():
    def txt(**kw):
        return SessionResult("s", kw.pop("outcome", ScanOutcome.ACCEPT),
                             **kw).user_facing_text()
    assert txt(activity_performed=False) == SESSION_SENTENCES["resting_only"]
    assert txt(safety_blocked=True) == SESSION_SENTENCES["safety_blocked"]
    assert txt(activity_performed=True) == \
        SESSION_SENTENCES["recovery_complete"]
    assert txt(activity_performed=True,
               outcome=ScanOutcome.REPEAT_SCAN) == \
        SESSION_SENTENCES["repeat_protocol"]
    assert txt(activity_performed=True,
               outcome=ScanOutcome.NO_RESULT) == \
        SESSION_SENTENCES["no_result_protocol"]
    withstars = SessionResult("s", ScanOutcome.ACCEPT,
                              activity_performed=True,
                              confidence_stars=4).user_facing_text()
    assert withstars.endswith("Confidence: 4 of 5.")


def _manifest(**over):
    d = {"recording_id": "R1", "participant_id": "P1", "session_id": "S1",
         "site_id": "lab", "video_path": "R1.avi", "ecg_path": "R1.ecg",
         "video_start_utc": "2026-08-30T10:00:00+00:00",
         "video_end_utc": "2026-08-30T10:01:00+00:00",
         "ecg_start_utc": "2026-08-30T10:00:00+00:00",
         "ecg_end_utc": "2026-08-30T10:01:00+00:00",
         "duration_s": 60.0,
         "capture": {"phone_model": "x", "os_version": "1",
                     "camera": "front", "width": 640, "height": 480,
                     "nominal_fps": 30.0},
         "sync": {"method": "LED_FLASH_MARKER", "offset_ms": 1.0,
                  "sync_uncertainty_ms": 2.0}}
    d.update(over)
    return d


def test_manifest_challenge_and_context_blocks_round_trip():
    d = _manifest(
        challenge={"protocol_id": "sts_1min", "cadence_prescribed": 20.0,
                   "cadence_achieved": 19.2, "reps": 19,
                   "transition_s": 4.1},
        participant_context={"age": 44, "sex": "male",
                             "measured_weight_kg": 82.0,
                             "height_cm": 178.0,
                             "meds": {"beta_blocker": True},
                             "activity_ipaq": "moderate"})
    rec = recording_from_json(json.dumps(d))
    assert rec.challenge.protocol_id == "sts_1min"
    assert rec.challenge.transition_s == 4.1
    assert rec.participant_context.meds.beta_blocker is True
    assert rec.participant_context.meds.rate_limiting is True
    assert rec.participant_context.meds.stimulant is False
    # old rhythm-track manifests (no blocks) still load
    plain = recording_from_json(json.dumps(_manifest()))
    assert plain.challenge is None and plain.participant_context is None


def test_manifest_blocks_fail_closed_on_unknown_fields():
    bad1 = _manifest(challenge={"protocol_id": "sts_1min",
                                "cadence_prescribed": 20.0,
                                "estimated_vo2": 40.0})
    with pytest.raises(ValueError, match="ChallengeRecord"):
        recording_from_json(json.dumps(bad1))
    bad2 = _manifest(participant_context={"age": 44,
                                          "face_weight_kg": 80.0})
    with pytest.raises(ValueError, match="ParticipantContext"):
        recording_from_json(json.dumps(bad2))
    bad3 = _manifest(participant_context={"meds": {"beta_bloker": True}})
    with pytest.raises(ValueError, match="Medications"):
        recording_from_json(json.dumps(bad3))


def test_cpet_label_fail_closed():
    good = {"vo2peak_mlkgmin": 38.4, "modality": "treadmill",
            "protocol": "bruce", "rer_peak": 1.12, "hr_peak": 182,
            "effort_criteria": ["rer>1.10", "plateau"],
            "avg_window_s": 30, "cart": "cosmed", "lab": "L1",
            "test_date": "2026-07-01"}
    lab = cpet_from_dict(good)
    assert isinstance(lab, CpetLabel) and lab.vo2peak_mlkgmin == 38.4
    for bad, match in (
        (dict(good, vo2peak_mlkgmin=200.0), "plausible"),
        (dict(good, rer_peak=2.5), "rer_peak"),
        (dict(good, hr_peak=20), "hr_peak"),
        (dict(good, extra_field=1), "unknown"),
        ({k: v for k, v in good.items() if k != "modality"}, "required"),
        (dict(good, effort_criteria="rer"), "list"),
    ):
        with pytest.raises(ValueError, match=match):
            cpet_from_dict(bad)
