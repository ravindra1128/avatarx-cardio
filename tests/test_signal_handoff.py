"""One job accepts both inputs; retaining recordings is independent of receipt."""
import copy
import io
import json
import struct
from concurrent.futures import Future

import pytest
from app import measure_api as api, result_sheet
from inference import shenai_route


UID = 'scan-input-123'
SIGNALS = {'upload_id': UID, 'schema_version': 1,
           'ppg': {'signal': [0.1, None, 0.2], 'n': 999, 'fs_source': 'unknown'},
           'heartbeats': [{'start_location_sec': 0.0, 'end_location_sec': 0.8}],
           'reference': {'average_signal_quality': 0.8, 'bad_signal_seconds': None,
                         'heart_rate_bpm': 75.0}}
ATTACHMENT = {'signal_delivery': {'version': 1, 'state': 'attached'}, 'shenai_signals': SIGNALS}


def handler(body=b''):
    h = api.MeasureHandler.__new__(api.MeasureHandler)
    h.headers = {'Content-Length': str(len(body)), 'User-Agent': 'test'}
    h.rfile = io.BytesIO(body)
    h.close_connection = False
    h.sent = []
    h._json = lambda code, doc: h.sent.append((code, doc))
    return h


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    from app.result_store import ResultStore
    from collections import OrderedDict
    monkeypatch.setattr(api, '_RESULTS', OrderedDict())
    monkeypatch.setattr(api, '_RESULT_STORE', ResultStore(tmp_path / 'results.sqlite', api.RESULT_TTL_S))
    monkeypatch.setenv('AFIB_KEEP_UPLOADS', '0')
    monkeypatch.delenv('AFIB_CLIPS_TOKEN', raising=False)
    monkeypatch.setattr(api, 'WORK_DIR', tmp_path / 'work')
    monkeypatch.setattr(api, 'UPLOAD_DIR', tmp_path / 'parts')
    monkeypatch.setattr(api, 'CLIPS_DIR', tmp_path / 'clips')
    monkeypatch.setattr(api, '_SIGNALS', {})
    monkeypatch.setattr(api, '_STARTED', {})
    monkeypatch.setattr(api, 'SHENAI_ROUTE_ON', True)
    monkeypatch.setattr(api, 'SHENAI_WAIT_S', 0)
    monkeypatch.setattr(result_sheet, 'schedule_append', lambda *a, **kw: None)


def test_envelope_carries_signals_and_preserves_query_metadata():
    hdr = json.dumps(ATTACHMENT).encode()
    wire = struct.pack('<I', len(hdr)) + hdr + b'video bytes'
    video, header = api._parse_envelope(wire, 'application/x-afib-measure', {
        'upload_id': [UID], 'session': ['person'], 'window_s': ['70'],
        'capture_profile': ['consumer'], 'ref_hr': ['75'], 'awb_locked': ['true']})
    assert video == b'video bytes'
    assert header['shenai_signals'] == SIGNALS
    assert header['upload_id'] == UID and header['window_s'] == 70
    assert header['reference']['ref_hr'] == 75 and header['manifest']['awb_locked'] is True


@pytest.mark.parametrize('state,payload', [('attached', None), ('missing_snapshot', SIGNALS),
                                          ('attached', {'upload_id': 'wrong-scan'})])
def test_invalid_identity_or_state_is_rejected_before_assembly(state, payload, monkeypatch):
    body = json.dumps({'signal_delivery': {'version': 1, 'state': state}, 'shenai_signals': payload}).encode()
    monkeypatch.setattr(api, '_assemble_parts', lambda *a: pytest.fail('consumed the video'))
    h = handler(body); h._start_job({'upload_id': [UID]})
    assert h.sent[-1][0] == 400
    assert h.rfile.tell() == len(body)


def test_start_attaches_inputs_to_the_accepted_job_without_a_sidecar(monkeypatch):
    queue = []
    class Pool:
        def submit(self, fn):
            queue.append(fn)
            return Future()
    monkeypatch.setattr(api, '_POOL', Pool())
    p = handler(b'video'); p._upload_part({'upload_id': [UID], 'index': ['0'], 'total': ['1']})
    seen = []
    def run(path, header):
        seen.append(header)
        assert header['shenai_signals'] == SIGNALS
        assert not (api._part_dir(UID) / api.SHENAI_PART_NAME).exists()
        return {'outcome': 'REPEAT_SCAN', 'debug': {}}
    monkeypatch.setattr(api.MeasureHandler, '_run_assembled', staticmethod(run))
    body = json.dumps(ATTACHMENT).encode()
    h = handler(body); h._start_job({'upload_id': [UID]})
    assert h.sent[-1][0] == 202 and len(queue) == 1 and not seen
    # A lost acknowledgement retries the same id/body, never queues twice.
    retry = handler(body); retry._start_job({'upload_id': [UID]})
    assert retry.sent[-1][0] == 202 and len(queue) == 1
    assert retry.rfile.tell() == len(body)
    queue[0]()
    assert len(seen) == 1 and not api._part_dir(UID).exists()
    assert api._peek_signals(UID) is None and not api.CLIPS_DIR.exists()


@pytest.mark.parametrize('retention', ['0', '1'])
def test_summary_receipt_does_not_depend_on_retention(retention, monkeypatch):
    monkeypatch.setenv('AFIB_KEEP_UPLOADS', retention)
    monkeypatch.setattr(shenai_route, 'wait_for', lambda *a: pytest.fail('waited for an attached input'))
    seen = []
    monkeypatch.setattr(shenai_route, 'evaluate', lambda doc, det, raw, **kw: (
        seen.append(raw) or {'used': False, 'reason': 'test'}))
    doc = {'outcome': 'ACCEPT', 'predicted_class': 'SINUS'}
    api._apply_shenai_route(doc, {}, UID, attachment=ATTACHMENT)
    info = doc['debug']['shenai_input']
    assert seen == [SIGNALS] and info['received'] is True
    assert info['ppg_n'] == 3 and info['ppg_missing_n'] == 1
    assert info['sdk_quality'] == 0.8 and info['sdk_bad_signal_s'] is None
    assert info['transport'] == 'job_request' and info['waited_s'] == 0
    row = result_sheet.row_from_doc(doc)
    assert row['Signals Received'] == 'TRUE' and row['Input Beats N'] == 1
    assert row['SDK Bad Signal s'] == '' and row['ShenAI Sidecar'] == ''
    assert 'heartbeats' not in json.dumps(doc) and 'start_location_sec' not in json.dumps(doc)
    assert doc['predicted_class'] == 'SINUS'


def test_explicit_missing_snapshot_skips_legacy_wait_and_stays_missing(monkeypatch):
    monkeypatch.setattr(shenai_route, 'wait_for', lambda *a: pytest.fail('unnecessary wait'))
    doc = {'outcome': 'REPEAT_SCAN'}
    api._apply_shenai_route(doc, {}, UID, attachment={
        'signal_delivery': {'version': 1, 'state': 'missing_snapshot'}, 'shenai_signals': None})
    assert doc['debug']['shenai_input']['received'] is False
    assert doc['debug']['shenai_input']['state'] == 'missing_snapshot'
    assert doc['outcome'] == 'REPEAT_SCAN'


def test_summary_failure_never_skips_the_route(monkeypatch):
    monkeypatch.setattr(api, '_signal_input_summary', lambda *a: (_ for _ in ()).throw(ValueError()))
    seen = []
    monkeypatch.setattr(shenai_route, 'evaluate', lambda doc, det, raw, **kw: seen.append(raw) or {'used': False})
    doc = {}
    api._apply_shenai_route(doc, {}, UID, attachment=ATTACHMENT)
    assert seen == [SIGNALS] and doc['debug']['shenai_input']['state'] == 'summary_failed'


def test_both_video_transports_use_the_same_route_and_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr(api, 'measure_video_details', lambda *a, **kw: (
        {'outcome': 'REPEAT_SCAN', 'biomarkers': {'items': []}}, {}))
    received = []
    def evaluate(doc, det, raw, **kw):
        received.append(raw)
        return {'used': False, 'reason': 'test'}
    monkeypatch.setattr(shenai_route, 'evaluate', evaluate)
    header = {**copy.deepcopy(ATTACHMENT), 'upload_id': UID, 'session': 'person'}
    doc_single = api.MeasureHandler._run(b'video', header)
    path = tmp_path / UID / 'scan.webm'; path.parent.mkdir(); path.write_bytes(b'video')
    doc_parts = api.MeasureHandler._run_assembled(str(path), header)
    assert received == [SIGNALS, SIGNALS]
    # Wall-clock telemetry differs; source evidence and decisions must match.
    for doc in (doc_parts, doc_single):
        assert doc['debug']['rhythm_comparison'].pop('elapsed_ms') >= 0
    assert doc_parts['debug'] == doc_single['debug']
    assert not list(api.WORK_DIR.rglob('scan.*'))
    assert not api.CLIPS_DIR.exists()


def test_legacy_sidecar_still_works_and_reports_receipt():
    api._hold_signals(UID, SIGNALS)
    doc = {'outcome': 'ACCEPT', 'predicted_class': 'SINUS'}
    api._apply_shenai_route(doc, {}, UID)
    assert doc['debug']['shenai_input']['transport'] == 'sidecar'
    assert doc['debug']['shenai_input']['received'] is True
    assert doc['predicted_class'] == 'SINUS'


@pytest.mark.parametrize('delivery', [{'version': True, 'state': 'attached'},
                                     {'version': 1, 'state': []},
                                     {'version': 1, 'state': 'unsupported'}])
def test_malformed_delivery_metadata_is_a_structured_400(delivery):
    h = handler(json.dumps({'signal_delivery': delivery, 'shenai_signals': SIGNALS}).encode())
    h._start_job({'upload_id': [UID]})
    assert h.sent[-1][0] == 400


def test_incomplete_start_body_is_not_accepted_even_if_json_parses():
    body = json.dumps(ATTACHMENT).encode()
    h = handler(body); h.headers['Content-Length'] = str(len(body) + 20)
    h._start_job({'upload_id': [UID]})
    assert h.sent[-1] == (400, {'error': 'incomplete start input'})
    assert h.close_connection is True


def test_combined_start_keeps_profile_and_signal_attachment(tmp_path, monkeypatch):
    profile = {'age': 35, 'sex': 'male', 'height_cm': 178, 'weight_kg': 76, 'activity_level': 3}
    body = json.dumps({**copy.deepcopy(ATTACHMENT), 'participant': profile}).encode()
    captured = {}
    monkeypatch.setattr(api._POOL, 'submit', lambda fn: fn())
    monkeypatch.setattr(api.MeasureHandler, '_run_assembled', staticmethod(
        lambda path, header: captured.update(header=header) or {'outcome': 'NO_RESULT'}))
    uploader = handler(b'video')
    uploader._upload_part({'upload_id': [UID], 'index': ['0'], 'total': ['1']})
    starter = handler(body)
    starter._start_job({'upload_id': [UID]})
    assert starter.sent[-1][0] == 202
    assert captured['header']['participant'] == profile
    assert captured['header']['shenai_signals'] == SIGNALS
    assert captured['header']['signal_delivery']['state'] == 'attached'
