"""v0.4 T2 — recovery HR tracker: short-window trimmed-median HR with a
physiological slew bound, robust monotone trend, back-extrapolated
end-exercise proxy, HRR30/60/120. NOT the rhythm feature path — recovery
HR is non-stationary by construction (falls 20-40 bpm/min) and
clean_runs/rhythm features assume stationarity.

Acceptance (prompt v0.4 T2): HR(t) MAE <= 2 bpm and HRR60 error <= 3 bpm
on decay fixtures INCLUDING degradations; property tests pin the slew
bound and the back-extrapolation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from features.recovery import (MAX_SLEW_BPM_S, WINDOW_S, recovery_metrics,
                               slew_bound, window_hr_series)


def _hr(t, hr_rest=82.0, hr0=142.0, tau=45.0):
    return hr_rest + (hr0 - hr_rest) * np.exp(-np.asarray(t, float) / tau)


def _beats(duration_s=150.0, seed=3, *, jitter_ms=8.0, drop=0.0,
           hr_kw=None, conf_low_spans=()):
    """Beat times integrated from the true HR(t) decay + timing jitter;
    `drop` removes a random fraction (dropouts); `conf_low_spans` marks
    motion-burst/talking spans with low beat confidence."""
    rng = np.random.default_rng(seed)
    hr_kw = hr_kw or {}
    t, beats = 0.0, []
    while t < duration_s:
        t += 60.0 / float(_hr(t, **hr_kw))
        beats.append(t + rng.normal(0, jitter_ms / 1000.0))
    beats = np.asarray(sorted(beats))
    conf = np.clip(rng.normal(0.85, 0.05, beats.size), 0.0, 1.0)
    for a, b in conf_low_spans:
        conf[(beats >= a) & (beats <= b)] = 0.15
    if drop:
        keep = rng.random(beats.size) > drop
        beats, conf = beats[keep], conf[keep]
    return beats, conf


def test_hr_series_tracks_the_decay_within_2bpm():
    beats, conf = _beats()
    series = window_hr_series(beats, conf, duration_s=150.0)
    assert len(series) >= 30                     # 50% overlap coverage
    errs = [abs(hr - _hr(t)) for t, hr, c in series if c is not None]
    assert float(np.mean(errs)) <= 2.0, np.mean(errs)


def test_hrr_and_end_proxy_within_3bpm_clean_and_degraded():
    truth = {"hr_end": _hr(0.0), "hrr30": _hr(0.0) - _hr(30.0),
             "hrr60": _hr(0.0) - _hr(60.0), "hrr120": _hr(0.0) - _hr(120.0)}
    for name, kw in (
        ("clean", {}),
        ("dropouts", {"drop": 0.10}),
        ("motion+talking", {"conf_low_spans": [(35.0, 41.0), (80.0, 86.0)]}),
        ("jittery", {"jitter_ms": 20.0}),
    ):
        beats, conf = _beats(seed=5, **kw)
        m = recovery_metrics(beats, conf, duration_s=150.0)
        assert m["hrr60"] is not None, name
        assert abs(m["hr_end_proxy"] - truth["hr_end"]) <= 3.0, \
            (name, m["hr_end_proxy"], truth["hr_end"])
        for k in ("hrr30", "hrr60", "hrr120"):
            assert abs(m[k] - truth[k]) <= 3.0, (name, k, m[k], truth[k])
        assert m["recovery_slope_bpm_min"] < -5.0, name   # it IS a decay


def test_slew_bound_is_a_hard_property():
    """An artifact burst (detector doubling: 300 ms IBIs mid-decay) must
    not let the estimated series move faster than 3 bpm/s."""
    beats, conf = _beats(seed=9)
    burst = np.arange(50.0, 53.0, 0.3)           # 200 bpm artifact spray
    beats = np.sort(np.concatenate([beats, burst]))
    conf = np.full(beats.size, 0.8)
    series = window_hr_series(beats, conf, duration_s=150.0)
    ts = [t for t, hr, c in series]
    hrs = [hr for t, hr, c in series]
    for i in range(1, len(hrs)):
        dt = ts[i] - ts[i - 1]
        assert abs(hrs[i] - hrs[i - 1]) <= MAX_SLEW_BPM_S * dt + 1e-6


def test_slew_bound_function_directly():
    t = np.arange(0.0, 40.0, 4.0)
    hr = np.full(t.size, 100.0)
    hr[5] = 160.0                                 # single-window spike
    out = slew_bound(t, hr)
    assert float(np.max(np.abs(np.diff(out)))) <= MAX_SLEW_BPM_S * 4.0 + 1e-9
    smooth = 100.0 - 0.5 * t                      # within bound: untouched
    assert np.allclose(slew_bound(t, smooth), smooth)


def test_back_extrapolation_beats_first_window_value():
    """The first window center sits ~WINDOW_S/2 into recovery, where HR
    has already fallen; the 0-15 s fit must recover HR at t=0 better than
    just reading the first window."""
    beats, conf = _beats(seed=11, hr_kw={"tau": 30.0})   # fast decay
    m = recovery_metrics(beats, conf, duration_s=150.0)
    first_window = m["hr_series"][0][1]
    true0 = _hr(0.0, tau=30.0)
    assert abs(m["hr_end_proxy"] - true0) < abs(first_window - true0)
    assert m["hr_end_proxy"] > first_window       # decay: t=0 is higher


def test_monotone_trend_and_tau_are_research_tagged():
    beats, conf = _beats(seed=2)
    m = recovery_metrics(beats, conf, duration_s=150.0)
    trend = m["trend"]
    assert all(trend[i][1] >= trend[i + 1][1] - 1e-9
               for i in range(len(trend) - 1))    # non-increasing
    # exponential tau exists but ONLY under the research-telemetry key
    # (field CV 25-35% — never a user-facing number)
    assert "tau_s" in m["research_only"]
    assert 15.0 < m["research_only"]["tau_s"] < 120.0
    assert "tau_s" not in {k for k in m if k != "research_only"}


def test_fail_closed_on_insufficient_or_garbage_beats():
    m = recovery_metrics(np.array([1.0, 2.0, 3.0]), np.array([0.9] * 3),
                        duration_s=150.0)
    assert m["hrr60"] is None and m["hr_end_proxy"] is None
    assert any("insufficient" in r for r in m["reasons"])
    beats, conf = _beats(seed=4)
    m2 = recovery_metrics(beats, np.zeros(beats.size), duration_s=150.0)
    assert m2["hrr60"] is None                    # all low confidence
    m3 = recovery_metrics(beats, conf, duration_s=100.0)
    assert m3["hrr60"] is not None and m3["hrr120"] is None
    assert any("120" in r for r in m3["reasons"])


def test_quality_vector_reflects_bad_spans():
    beats, conf = _beats(seed=7, conf_low_spans=[(60.0, 75.0)])
    series = window_hr_series(beats, conf, duration_s=150.0)
    bad = [c for t, hr, c in series if 62.0 <= t <= 72.0]
    good = [c for t, hr, c in series if t <= 40.0]
    assert all(c is None or c < 0.4 for c in bad)
    assert np.median([c for c in good if c is not None]) > 0.6
