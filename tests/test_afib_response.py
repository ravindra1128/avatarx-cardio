from copy import deepcopy
import pytest
from app.afib_response import finalize_afib_response, capture_diagnostic_summary


@pytest.mark.parametrize('outcome,cls,expected', [
    ('ACCEPT', 'AFIB_SUGGESTIVE', 'AFIB_DETECTED'),
    ('ACCEPT', 'SINUS', 'AFIB_NOT_DETECTED'),
    ('ACCEPT', 'HIGH_RATE', 'AFIB_NOT_DETECTED'),
    ('ACCEPT', 'OTHER_IRREGULAR', 'INCONCLUSIVE'),
    ('REPEAT_SCAN', None, 'INCONCLUSIVE'), ('NO_RESULT', None, 'INCONCLUSIVE'),
])
def test_all_consumers_receive_the_selected_final_decision(outcome, cls, expected):
    old_head = {'head': 'afib', 'value': {'outcome': 'REPEAT_SCAN', 'predicted_class': None}, 'reasons': ['old']}
    other = {'head': 'resting_rate', 'value': 72}
    doc = {'outcome': outcome, 'predicted_class': cls, 'rhythm_source': 'shenai_train',
           'no_read_reasons': ['insufficient evidence'] if outcome != 'ACCEPT' else [],
           'head_results': [deepcopy(old_head), deepcopy(other)]}
    finalize_afib_response(doc)
    assert doc['afib_result'] == expected
    assert doc['head_results'][0]['value']['afib_result'] == expected
    assert doc['head_results'][0]['value']['outcome'] == outcome
    assert doc['head_results'][0]['value']['predicted_class'] == cls
    assert doc['head_results'][0]['reasons'] == doc['no_read_reasons']
    assert doc['head_results'][1] == other
    assert doc['debug']['video_afib_head'] == old_head
    before = deepcopy(doc); finalize_afib_response(doc); assert doc == before


def test_operational_failure_is_not_a_physiological_inconclusive():
    doc = {'error': 'worker failed', 'error_code': 'analysis_failed'}
    assert finalize_afib_response(doc) == doc
    assert 'afib_result' not in doc
    with pytest.raises(ValueError): finalize_afib_response({})


def test_client_capture_report_cannot_echo_recordings_or_arbitrary_data():
    raw = {'capture_diagnostics': {'version': 1, 'sdk_version': 'test', 'signal': [1, 2],
           'settings': {'deviceId': 'private', 'width': 640},
           'frames': {'adjacent_frame_steps': {'n': 10, 'p99_ms': 50, 'raw': [1, 2]},
                      'observations': 11, 'media_span_s': float('nan')},
           'quality': {'mean': float('inf'), 'min': .5}}}
    result = capture_diagnostic_summary(raw)
    assert result['state'] == 'reported'
    assert result['frames']['adjacent_frame_steps']['p99_ms'] == 50
    assert 'deviceId' not in result['settings']
    assert 'signal' not in result
    assert 'media_span_s' not in result['frames']
    assert 'mean' not in result['quality']
    assert capture_diagnostic_summary(None)['state'] == 'unavailable'


def test_http_job_finalizes_the_sdk_selected_result(monkeypatch, tmp_path):
    import runpy
    from pathlib import Path
    from app import measure_api as api
    fx = runpy.run_path(str(Path.cwd() / 'tests/test_shenai_route.py'))
    raw = fx['_sidecar'](fx['_regular']())
    doc, det = fx['_video'](stars=3)
    doc['head_results'] = [{'head': 'afib', 'value': {'outcome': 'REPEAT_SCAN', 'predicted_class': None}}]
    monkeypatch.setattr(api, 'measure_video_details', lambda *a, **kw: (doc, det))
    monkeypatch.setattr(api, '_pair_shenai', lambda *a, **kw: None)
    monkeypatch.setattr(api, 'SHENAI_ROUTE_ON', True)
    result = api.MeasureHandler._run_assembled(str(tmp_path / 'scan.webm'), {
        'signal_delivery': {'version': 1, 'state': 'attached'}, 'shenai_signals': raw})
    assert result['rhythm_source'] == 'shenai_train'
    assert result['afib_result'] == 'AFIB_NOT_DETECTED'
    assert result['head_results'][0]['value']['predicted_class'] == result['predicted_class'] == 'SINUS'
