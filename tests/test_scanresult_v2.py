"""M1.5 + invariants 9 and 11 — ScanResult v2, measurement-class
labeling, v1-reader migration, and the ECG-wording fence on every
user-facing string surface."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dataclasses
import json
import re

import pytest

from datasets.schema import (ScanResult, ScanOutcome, MeasurementClass,
                             RATE_FLAG_SENTENCES, scan_result_from_dict)

_APP_HTML = pathlib.Path(__file__).resolve().parents[1] / "app" / "static" / \
    "index.html"


# ------------------------------------------------------- invariant 9
def test_every_scanresult_field_is_classified_and_none_synthetic():
    fields = set(ScanResult.__dataclass_fields__)
    table = ScanResult.FIELD_CLASSES
    assert set(table) == fields, (
        "FIELD_CLASSES must cover every ScanResult field exactly: "
        f"missing={fields - set(table)}, stale={set(table) - fields}")
    for name, cls in table.items():
        assert isinstance(cls, MeasurementClass), name
        # app/ renders ScanResult — nothing in it may ever be a research
        # quantity: not synthetic, not vascular, and (v0.6) not the
        # flutter research class either
        assert not cls.value.startswith("RESEARCH_"), name
        assert cls is not MeasurementClass.RESEARCH_SYNTHETIC, name
        assert cls is not MeasurementClass.RESEARCH_VASCULAR, name
        assert cls is not MeasurementClass.RESEARCH_RHYTHM, name


def test_head_results_items_carry_their_own_class():
    from heads.base import HeadResult
    hr = HeadResult(head="x", version="1",
                    measurement_class=MeasurementClass.MEASURED, value={})
    r = ScanResult("r", ScanOutcome.ACCEPT, head_results=[hr.to_dict()])
    assert r.schema_version == 2
    assert r.head_results[0]["measurement_class"] == "MEASURED"


# ------------------------------------------------------- migration shim
def test_v1_record_reads_into_v2_object():
    v1 = {"recording_id": "old", "outcome": "ACCEPT",
          "predicted_class": "SINUS", "signal_quality_index": 0.8,
          "no_read_reasons": [], "model_version": "interim-rules-v0.1.2",
          "code_commit": "abc1234", "calibration_version": "cal-x",
          "config_hash": "h"}                          # no schema_version
    r = scan_result_from_dict(v1)
    assert r.schema_version == 2
    assert r.head_results == [] and r.capture_meta is None
    assert r.outcome is ScanOutcome.ACCEPT
    assert r.user_facing_text()                        # text gate intact


def test_v2_round_trip_and_future_versions_fail_closed():
    r = ScanResult("rt", ScanOutcome.REPEAT_SCAN, confidence_stars=2,
                   confidence_limiting_factor="sqi")
    d = json.loads(json.dumps(dataclasses.asdict(r), default=str))
    r2 = scan_result_from_dict(d)
    assert r2.confidence_stars == 2 and r2.outcome is ScanOutcome.REPEAT_SCAN
    with pytest.raises(ValueError):
        scan_result_from_dict({"recording_id": "x", "outcome": "ACCEPT",
                               "schema_version": 3})
    with pytest.raises(ValueError):
        scan_result_from_dict({"recording_id": "x", "outcome": "ACCEPT",
                               "bogus_field": 1})


# ------------------------------------------------------- invariant 11
def _all_sanctioned_strings():
    from datasets.schema import SESSION_SENTENCES, SessionResult
    out = list(RATE_FLAG_SENTENCES.values())
    for cls in ["SINUS", "AFIB_SUGGESTIVE", "OTHER_IRREGULAR",
                "HIGH_RATE", None]:
        for outcome in ScanOutcome:
            for stars in (None, 1, 2, 3, 4, 5):
                out.append(ScanResult(
                    "r", outcome, predicted_class=cls,
                    confidence_stars=stars,
                    confidence_limiting_factor="lighting").user_facing_text())
    # v0.4: the session-level table + every SessionResult state rides the
    # same invariant-11 fence, as do the §V-gated fitness/trend tables
    # (unrenderable today, audited from day one)
    from datasets.schema import (FITNESS_CATEGORY_SENTENCES,
                                 TREND_DIRECTION_SENTENCES)
    out += list(SESSION_SENTENCES.values())
    out += list(FITNESS_CATEGORY_SENTENCES.values())
    out += list(TREND_DIRECTION_SENTENCES.values())
    # v0.6: the regular-tachy sentence rides the same fence. It was the
    # only sanctioned table exempt from this audit, and exempt is
    # exactly where an unaudited sentence would hide (review finding).
    from datasets.schema import REGULAR_TACHY_SENTENCES
    out += list(REGULAR_TACHY_SENTENCES.values())
    # v0.7: the regularity head's sanctioned sentence rides the same fence
    from datasets.schema import REGULARITY_SENTENCES
    out += list(REGULARITY_SENTENCES.values())
    for outcome in ScanOutcome:
        for activity in (False, True):
            for blocked in (False, True):
                for stars in (None, 1, 3, 5):
                    out.append(SessionResult(
                        "s", outcome, activity_performed=activity,
                        safety_blocked=blocked,
                        confidence_stars=stars).user_facing_text())
    return out


def test_ecg_appears_only_in_referral_context():
    """Invariant 11 (interpretation on record, spec B.15): every 'ECG'
    occurrence in a user-facing string must sit inside a clinician
    referral / ground-truth deferral sentence; phrases implying the app
    shows or produces an ECG are banned outright."""
    banned = ["your ecg", "ecg waveform", "inferred ecg", "we generated",
              "synthesized ecg", "synthetic ecg"]
    # "please have an ECG" IS a referral — the v0.6 sanctioned sentence
    # routes to the instrument that can name a rhythm without naming one
    # itself, which is the behaviour this invariant exists to require
    # ... and v0.7's "an ECG can tell you why" is the same referral: the
    # sentence declines to name the rhythm and points at what can
    ok_context = re.compile(
        r"(clinician|ground truth|read by a clinician|have an ECG"
        r"|an ECG can tell you why)", re.I)
    for s in _all_sanctioned_strings():
        low = s.lower()
        for b in banned:
            assert b not in low, s
        for m in re.finditer(r"\bECG\b", s):
            sentence = s[max(0, s.rfind(".", 0, m.start()) + 1):
                         s.find(".", m.end()) + 1 or len(s)]
            assert ok_context.search(sentence), (
                f"'ECG' outside referral context: {sentence!r}")


def test_app_page_never_claims_an_ecg():
    """v0.2.1: the consumer surface is the Cardiac Rhythm Scan Report
    (named by the renderer itself, title-tested in
    tests/test_report_forbidden.py); the page may mention ECG only in
    clinician-deferral context."""
    html = _APP_HTML.read_text()
    for b in ("your ECG", "ECG waveform", "Inferred ECG", "inferred ECG",
              "synthesized ECG", "synthetic ECG"):
        assert b not in html, b
    for m in re.finditer(r"\bECG\b", html):
        window = html[max(0, m.start() - 220):m.end() + 220].lower()
        assert "clinician" in window or "ground truth" in window, \
            html[max(0, m.start() - 80):m.end() + 80]
