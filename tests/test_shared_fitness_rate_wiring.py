"""Exercise the actual morphology-to-fitness call with controlled scan inputs.

These are wiring controls, not validation of physiological accuracy.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from features.hemodynamics import resting_hemodynamics, resting_rate_index
from heads.head_rate_flags import RateFlagsHead
from preprocessing.roi import ROI_NAMES


def _scan(intervals, spectral, regions, *, sqi=0.8):
    fps = 60.0
    ts = np.arange(0, 30, 1 / fps)
    phase = (ts % 1.0)
    pulse = np.exp(-((phase - 0.18) / 0.09) ** 2)
    pulse += 0.3 * np.exp(-((phase - 0.5) / 0.12) ** 2)
    rgb = np.column_stack((100 + pulse, 80 + 3 * pulse, 60 + 0.4 * pulse))
    beats = np.arange(1., 29.)
    finite = intervals[np.isfinite(intervals) & (intervals > 0)]
    count = 60000 / np.median(finite)
    reg = SimpleNamespace(runs=[intervals], n_intervals=len(finite),
                          values={'median_ibi': np.median(finite)},
                          dispersion={'rmssd_ms': 20., 'sdnn_ms': 25.})
    ev = {'pulse_lattice_bpm': count, 'pulse_spectral_bpm': spectral,
          'pulse_spectral_roi_agree': regions, 'pulse_spectral_snr': 7.,
          'pulse_agreement': abs(count - spectral) / spectral if spectral else None,
          'harmonic_fraction': 0., 'cross_roi_coherence': 0.8,
          'timing_precision_ms': 12., 'timing_matched_fraction': 0.95}
    return {'ingest': SimpleNamespace(timestamps_s=ts, traces={r: rgb.copy() for r in ROI_NAMES},
                                     meta=SimpleNamespace(measured_fps_mean=fps),
                                     track=SimpleNamespace(stability=0.99)),
            'lattice': SimpleNamespace(beat_t_s=beats, beat_confidence=np.ones(len(beats))),
            'sqi': SimpleNamespace(sqi=sqi), 'regularity': reg, 'evidence': ev}


@pytest.mark.parametrize('intervals,spectral,regions,sqi', [
    (np.full(24, 60000 / 117), 45., 1, 0.8),
    (np.full(24, 60000 / 101), 66., 3, 0.8),
    (np.full(24, 60000 / 70), 71., 3, 0.8),
    (np.full(24, 60000 / 70), 71., 3, 0.1),
    (np.full(8, 60000 / 70), 71., 3, 0.8),
    (np.r_[np.full(10, 700.), np.full(10, 1100.), np.nan, -1., 0.], 70., 3, 0.8),
])
def test_full_card_call_uses_rate_head_decision(intervals, spectral, regions, sqi):
    det = _scan(intervals, spectral, regions, sqi=sqi)
    out = resting_hemodynamics(det, outcome='REPEAT_SCAN',
                               capture={'exposure_locked': True, 'awb_locked': True})
    assert out['available'], out
    card = out['cardiorespiratory_fitness']
    head = RateFlagsHead().run(det['lattice'], det).to_dict()['value']
    assert card['resting_hr_bpm'] == head['median_bpm']
    assert card['rate_resolution']['confidence'] == head['rate_confidence']
    if head['median_bpm'] is None:
        assert card['available'] is False and card['score'] is None
        assert card['reason_code'] == 'resting_rate_unverified'
    else:
        assert card['score'] == round(100 * resting_rate_index(head['median_bpm']), 1)
        if sqi < 0.3 or head['rate_confidence'] != 'verified':
            assert card['tier'] == 'provisional'
            assert card['tier_reasons']


def test_merged_scan_preserves_staging_live_reference_fitness_without_profile():
    # Codex cannot resolve this video's rate. Staging option A still uses the
    # same request's live reference, without inventing profile inputs or VO2 units.
    det = _scan(np.full(24, 60000 / 117), 45., 1)
    out = resting_hemodynamics(det, outcome='REPEAT_SCAN',
                               reference={'ref_hr': 74., 'ref_source': 'shenai'})
    card = out['cardiorespiratory_fitness']
    assert card['available'] is True
    assert card['resting_hr_bpm'] == 74.
    assert card['score'] == round(100 * resting_rate_index(74.), 1)
    assert card['oxygen_uptake_estimate'] is None
    assert card['tier'] == 'provisional'
