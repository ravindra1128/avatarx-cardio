"""Source comparison is observable but cannot select the published rhythm."""
import copy
import json

import pytest

from app import measure_api, result_sheet
from app.rhythm_diagnostics import compare_sources
from app.scan_evidence import shenai_evidence_summary
from inference import shenai_route
from test_shenai_route import _video, _sidecar, _regular, _afib


def compare(doc, det, raw, enabled=True):
    doc.setdefault('debug', {})['shenai_assessment'] = shenai_evidence_summary(raw, det)
    return compare_sources(doc, det, raw, enabled)


@pytest.mark.parametrize('kwargs', [{}, {'outcome': 'ACCEPT', 'gates_failed': ()},
    {'gates_failed': ('afib_harmonic_fraction', 'afib_timing_matched')}])
def test_candidate_runs_without_changing_publication_or_inputs(kwargs):
    doc, det = _video(**kwargs)
    raw = _sidecar(_regular())
    doc['debug']['shenai_assessment'] = shenai_evidence_summary(raw, det)
    before_doc, before_raw = copy.deepcopy(doc), copy.deepcopy(raw)
    before_evidence, before_cfg = copy.deepcopy(det['evidence']), copy.deepcopy(det['config'])
    out = compare_sources(doc, det, raw, True)
    assert out['shenai_train']['state'] == 'assessed'
    assert out['shenai_train']['result'] == 'AFIB_NOT_DETECTED'
    assert out['contributes_to_published_result'] is False
    assert doc == before_doc and raw == before_raw
    assert det['evidence'] == before_evidence and det['config'] == before_cfg
    assert out['window_alignment'] == 'unknown' and out['verified_overlap_s'] is None
    assert 'heartbeats' not in json.dumps(out, allow_nan=False)
    if kwargs:
        assert out['shenai_train']['publication_policy_eligible'] is False
        assert out['shenai_train']['publication_blocker']


def test_selected_sdk_preserves_video_and_matches_diagnostic():
    doc, det = _video()
    raw = _sidecar(_regular())
    shenai_route.evaluate(doc, det, raw)
    out = compare(doc, det, raw)
    assert doc['rhythm_source'] == 'shenai_train'
    assert out['video']['result'] == 'INCONCLUSIVE'
    assert out['shenai_train']['result'] == 'AFIB_NOT_DETECTED'
    assert out['decision_agreement'] == 'disagree'


@pytest.mark.parametrize('kwargs', [{'sqi': .1}, {'coherence': .05, 'matched': .1}, {'tp': 70}])
def test_diagnostic_keeps_scan_quality_gates(kwargs):
    doc, det = _video(**kwargs)
    out = compare(doc, det, _sidecar(_regular()))
    assert out['shenai_train']['result'] == 'INCONCLUSIVE'
    assert out['shenai_train']['gates_failed']


def test_irregular_positive_control_is_not_smoothed_to_negative():
    doc, det = _video(coherence=.7, matched=.95, tp=10, stars=3,
                      spectral=75, lattice=75, sqi=.8)
    out = compare(doc, det, _sidecar(_afib()))
    assert out['shenai_train']['result'] == 'AFIB_DETECTED'


@pytest.mark.parametrize('raw,state', [(None, 'unavailable'), ({}, 'not_evaluated'),
    ({'heartbeats': []}, 'not_evaluated'), (_sidecar(_regular(), quality=.1), 'not_evaluated')])
def test_missing_or_rejected_train_has_no_fabricated_result(raw, state):
    doc, det = _video()
    out = compare(doc, det, raw)
    assert out['shenai_train']['state'] == state
    assert out['shenai_train']['result'] is None
    assert out['shenai_train']['reason']
    assert out['decision_agreement'] == 'not_comparable'


def test_original_order_is_not_silently_repaired():
    doc, det = _video()
    raw = _sidecar(_regular())
    raw['heartbeats'][5], raw['heartbeats'][6] = raw['heartbeats'][6], raw['heartbeats'][5]
    out = compare(doc, det, raw)
    assert out['shenai_train']['state'] == 'not_evaluated'
    assert out['shenai_train']['result'] is None


def test_disabled_route_can_be_observed_without_being_published():
    doc, det = _video()
    out = compare(doc, det, _sidecar(_regular()), enabled=False)
    assert out['shenai_train']['result'] == 'AFIB_NOT_DETECTED'
    assert out['shenai_train']['publication_policy_eligible'] is False
    assert 'disabled' in out['shenai_train']['publication_blocker']
    assert doc['outcome'] == 'REPEAT_SCAN'


def test_diagnostic_failure_does_not_fail_published_scan(monkeypatch):
    doc, det = _video()
    raw = _sidecar(_regular())
    def fail(*args):
        raise RuntimeError('diagnostic failure')
    monkeypatch.setattr('app.rhythm_diagnostics.compare_sources', fail)
    measure_api._apply_shenai_route(doc, det, 'diagnostic-test',
        attachment={'signal_delivery': {'state': 'attached'}, 'shenai_signals': raw})
    assert doc['outcome'] == 'ACCEPT'
    assert doc['rhythm_source'] == 'shenai_train'
    assert doc['debug']['rhythm_comparison']['state'] == 'assessment_failed'


def test_api_and_sheet_expose_additive_summary():
    doc, det = _video()
    measure_api._apply_shenai_route(doc, det, 'diagnostic-test',
        attachment={'signal_delivery': {'state': 'attached'}, 'shenai_signals': _sidecar(_regular())})
    row = result_sheet.row_from_doc(doc)
    assert row['Video Diagnostic Result'] == 'INCONCLUSIVE'
    assert row['SDK Diagnostic Result'] == 'AFIB_NOT_DETECTED'
    assert row['Source Window Alignment'] == 'unknown'
    assert len(result_sheet.COLUMNS) == len(set(result_sheet.COLUMNS))
    old = result_sheet.row_from_doc({})
    assert old['SDK Diagnostic Result'] == ''
    json.dumps(doc['debug']['rhythm_comparison'], allow_nan=False)


def test_nonfinite_video_features_do_not_erase_other_source():
    doc, det = _video()
    det['rationale']['features']['rmssd'] = float('nan')
    out = compare(doc, det, _sidecar(_regular()))
    assert out['video']['features']['rmssd'] is None
    assert out['shenai_train']['state'] == 'assessed'
    json.dumps(out, allow_nan=False)


def test_candidate_exception_keeps_video_report(monkeypatch):
    doc, det = _video()
    def fail(*args):
        raise RuntimeError('test')
    monkeypatch.setattr(shenai_route, '_evaluate_train', fail)
    out = compare(doc, det, _sidecar(_regular()))
    assert out['video']['result'] == 'INCONCLUSIVE'
    assert out['shenai_train']['state'] == 'assessment_failed'
    assert out['shenai_train']['result'] is None


def test_window_gaps_are_reported_without_assuming_clock_alignment():
    doc, det = _video()
    raw = _sidecar(_regular(), drop_at=20)
    out = compare(doc, det, raw)
    integrity = out['shenai_train']['integrity']
    assert integrity['gap_boundaries_n'] > 0
    assert integrity['first_beat_start_s'] == raw['heartbeats'][0]['start_location_sec']
    assert integrity['last_beat_end_s'] == raw['heartbeats'][-1]['end_location_sec']
    assert out['verified_overlap_s'] is None
