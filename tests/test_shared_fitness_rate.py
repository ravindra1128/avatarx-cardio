"""Prevent production fitness scores from contradicting the shared rate decision."""
from types import SimpleNamespace

import numpy as np
import pytest

from features.rate_guard import resolve_resting_rate
from heads.head_rate_flags import RateFlagsHead
from features.hemodynamics import cardiorespiratory_indices, resting_rate_index


@pytest.mark.parametrize('count,spec,regions,harmonics,n', [
    (65., 66., 3, .05, 25),
    (101., 66., 3, .05, 25),
    (101., 66., 2, .05, 25),
    (117., 45., 1, .20, 33),
    (90., 70., 1, .05, 20),
    (130., 65., 3, .30, 20),
    (130., 65., 2, .30, 20),
    (101., 69., 2, .35, 20),
    (70., 71., 3, .05, 5),
    (70., None, 0, .05, 20),
])
def test_fitness_consumes_the_same_rate_decision_as_the_head(count, spec, regions, harmonics, n):
    ev = {'pulse_lattice_bpm': count, 'pulse_spectral_bpm': spec,
          'pulse_spectral_roi_agree': regions, 'pulse_spectral_snr': 7.0,
          'harmonic_fraction': harmonics,
          'pulse_agreement': abs(count - spec) / spec if spec else None}
    reg = SimpleNamespace(runs=[np.full(n, 60000 / count)],
                          dispersion={'rmssd_ms': 15., 'sdnn_ms': 20.})
    head = RateFlagsHead().run(None, {'regularity': reg, 'evidence': ev}).to_dict()
    rr = resolve_resting_rate(count if n >= 15 else None, n, ev, min_intervals=15)
    card = cardiorespiratory_indices(reg, count, rate_method='clean_interval_median',
                                     rate_intervals=n, rate_resolution=rr)
    assert card['resting_hr_bpm'] == head['value']['median_bpm']
    assert card['rate_resolution']['confidence'] == head['value']['rate_confidence']
    if head['value']['median_bpm'] is None:
        assert card['score'] is None and card['available'] is False
        assert card['tier'] is None and card['reason_code'] == 'resting_rate_unverified'
    else:
        assert card['score'] == round(100 * resting_rate_index(head['value']['median_bpm']), 1)
        if rr['confidence'] != 'verified' or n < 15:
            assert card['tier'] == 'provisional' and card['tier_reasons']


def test_uncertain_resolution_cannot_fall_back_to_a_numeric_count():
    rr = {'bpm': None, 'source': None, 'confidence': 'uncertain',
          'guard': {}, 'n_intervals': 33, 'reasons': ['regions do not agree']}
    card = cardiorespiratory_indices(None, 117., rate_method='clean_interval_median',
                                     rate_intervals=33, rate_resolution=rr)
    assert card['resting_hr_bpm'] is None
    assert card['estimate'] is None and card['score'] is None
    assert 'regions do not agree' in card['reason']


def test_unverified_legacy_rate_resolution_does_not_become_measured():
    rr = resolve_resting_rate(70., 25, {}, min_intervals=15)
    card = cardiorespiratory_indices(None, 70., rate_method='clean_interval_median',
                                     rate_intervals=25, rate_resolution=rr)
    assert card['resting_hr_bpm'] == 70.
    assert card['tier'] == 'provisional'
    assert any('unverified' in reason for reason in card['tier_reasons'])


def test_loss_of_region_support_cannot_publish_disputed_count_as_a_new_score():
    cards=[]
    for spec,regions in [(66.,3),(65.,2)]:
        ev={'pulse_lattice_bpm':101.3,'pulse_spectral_bpm':spec,
            'pulse_spectral_roi_agree':regions,'pulse_spectral_snr':7.,
            'harmonic_fraction':0.,'pulse_agreement':abs(101.3-spec)/spec}
        rr=resolve_resting_rate(101.3,25,ev,min_intervals=15)
        cards.append(cardiorespiratory_indices(None,101.3,rate_method='clean_interval_median',
                     rate_intervals=25,rate_resolution=rr))
    assert cards[0]['resting_hr_bpm']==66.
    assert cards[1]['resting_hr_bpm'] is None
    assert cards[1]['score'] is None and cards[1]['reason']


def test_lost_spectrum_cannot_reenable_hidden_count_fallback():
    ev={'pulse_lattice_bpm':90.,'pulse_spectral_bpm':None,'pulse_spectral_roi_agree':0,
        'harmonic_fraction':0.,'pulse_agreement':None}
    rr=resolve_resting_rate(90.,25,ev,min_intervals=15)
    card=cardiorespiratory_indices(None,90.,rate_method='clean_interval_median',
                                 rate_intervals=25,rate_resolution=rr)
    assert card['score'] is None and card['rate_resolution']['confidence']=='uncertain'
