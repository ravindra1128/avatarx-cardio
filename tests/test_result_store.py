from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from app.result_store import ResultStore


def test_completed_result_survives_process_object_and_memory_cache_loss(tmp_path):
    path = tmp_path / 'results.sqlite'
    doc = {'scan_id': 'unique-a', 'outcome': 'ACCEPT', 'afib_result': 'AFIB_NOT_DETECTED',
           'biomarkers': {'items': [{'key': 'fitness', 'value': 50}]}}
    assert ResultStore(path).put('unique-a', doc)
    assert ResultStore(path).get('unique-a') == doc
    assert ResultStore(path).get('unique-b') is None
    assert path.stat().st_mode & 0o777 == 0o600


def test_retention_expires_from_completion_not_last_read(tmp_path):
    now = [1000]
    store = ResultStore(tmp_path / 'results.sqlite', ttl_s=15, clock=lambda: now[0])
    store.put('a', {'outcome': 'REPEAT_SCAN'})
    now[0] = 1014; assert store.get('a') is not None
    now[0] = 1015; assert store.get('a') is None
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM results').fetchone()[0] == 0


def test_more_than_32_results_do_not_evict_unexpired_scans(tmp_path):
    store = ResultStore(tmp_path / 'results.sqlite')
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(lambda i: store.put(str(i), {'scan_id': str(i)}), range(40)))
    assert store.get('0') == {'scan_id': '0'}
    assert store.get('39') == {'scan_id': '39'}


def test_never_persists_raw_inputs_and_outputs_valid_json(tmp_path):
    store = ResultStore(tmp_path / 'results.sqlite')
    store.put('a', {'outcome': 'REPEAT_SCAN', 'signal_quality_index': float('nan'),
                    'shenai_signals': {'heartbeats': [1, 2]},
                    'debug': {'signal': [1, 2], 'timing_ms': 40}})
    result = store.get('a')
    assert result == {'outcome': 'REPEAT_SCAN', 'signal_quality_index': None, 'debug': {'timing_ms': 40}}
    json.dumps(result, allow_nan=False)


def test_storage_failure_is_reported_without_raising_into_analysis(tmp_path):
    path = tmp_path / 'blocked'; path.mkdir()
    store = ResultStore(path)
    assert not store.put('a', {'outcome': 'REPEAT_SCAN'})
    assert store.status()['last_error'] is not None
    assert store.get('a') is None
