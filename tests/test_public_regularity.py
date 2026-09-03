"""The public surrogate's plumbing, without the corpus: segment-wise
detection on the real clock, NO_RESULT windows."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from datasets.public_regularity import (finite_segments, ppg_runs,
                                        rpeaks_segmented)


def test_finite_segments_split_at_nan_blocks():
    x = np.array([np.nan, 1, 2, 3, np.nan, np.nan, 4, 5, 6, np.nan])
    assert finite_segments(x) == [(1, 4), (6, 9)]
    assert finite_segments(x, min_samples=3) == [(1, 4), (6, 9)]
    assert finite_segments(x, min_samples=4) == []
    assert finite_segments(np.arange(5.0)) == [(0, 5)]


def test_rpeaks_after_a_nan_block_stay_on_the_real_clock():
    """Review finding: dropping NaN samples spliced the record and every
    later R-peak came out earlier by the dropped duration."""
    fs = 125.0
    t = np.arange(0, 60.0, 1.0 / fs)
    ecg = np.zeros(t.size)
    truth = np.arange(0.5, 59.5, 0.8)
    for tt in truth:
        ecg += np.exp(-0.5 * ((t - tt) / 0.012) ** 2)
    ecg += 0.02 * np.random.default_rng(0).normal(size=t.size)
    ecg[int(20.0 * fs):int(20.5 * fs)] = np.nan          # a 0.5 s hole
    rp = rpeaks_segmented(ecg, fs)
    late = rp[rp > 25.0]
    assert late.size > 30
    # each detected peak after the hole lands on a true beat
    err = np.min(np.abs(late[:, None] - truth[None, :]), axis=1)
    assert float(np.max(err)) < 0.03


def test_ppg_window_with_a_nan_block_yields_runs_on_both_sides():
    fs = 125.0
    t = np.arange(0, 90.0, 1.0 / fs)
    ppg = np.sin(2 * np.pi * (75.0 / 60.0) * t) + \
        0.3 * np.sin(4 * np.pi * (75.0 / 60.0) * t)
    ppg[int(40.0 * fs):int(40.3 * fs)] = np.nan
    rs = ppg_runs(ppg, fps=fs)
    assert len(rs.runs) >= 2
    assert sum(r.size for r in rs.runs) > 80
    for tr in rs.run_times:
        assert not np.any((np.asarray(tr) > 40.0) & (np.asarray(tr) < 40.3))


def test_reference_rpeaks_are_refined_below_the_sample_grid():
    """Integer-sample peaks at 125 Hz make every R-R difference a
    multiple of 8 ms; a regular heart then reads a median |dRR| of
    exactly 0. Refined times are not on the grid."""
    fs = 125.0
    t = np.arange(0, 60.0, 1.0 / fs)
    ecg = np.zeros(t.size)
    truth = np.cumsum(np.full(70, 0.8123)) + 0.3
    truth = truth[truth < 59.5]
    for tt in truth:
        ecg += np.exp(-0.5 * ((t - tt) / 0.012) ** 2)
    rp = rpeaks_segmented(ecg, fs)
    assert rp.size == truth.size
    err = rp - truth
    assert float(np.max(np.abs(err))) < 0.004
    frac = (rp * fs) % 1.0
    assert np.mean((frac < 0.02) | (frac > 0.98)) < 0.5
