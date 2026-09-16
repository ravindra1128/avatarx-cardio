"""The alignment null and the arm-B tautology, as regression tests.

WHY THESE EXIST. scripts/compare_shenai_signal.py compares ShenAI's PPG
against our four-ROI lattice, and until 2026-09-11 its "unalignable" guard
picked the best of ~1000 candidate offsets and then declared the winner
trustworthy at matched/min(n) >= 0.5. MEASURED against that shipped rule: two
beat trains at the same rate with a deliberately destroyed timebase relation
were called ALIGNABLE in 54/60 = 90% of trials (median |dt| 36 ms, under the
40 ms ceiling the harness printed two lines later), and 3/40 pairs of pure
noise aligned as well. The permutation null took both to 0/60 and 0/40 with
power unchanged at 30/30.

These are the same experiments at a smaller draw count so they run in CI
seconds. The full versions, with the before/after comparison printed side by
side, are `python scripts/compare_shenai_signal.py --self-test`.
"""
import numpy as np
import pytest

from scripts.compare_shenai_signal import (
    _align, _detect_wave, _fit_fs, _legacy_alignable, _start_fs, _synth_train,
    _synth_wave)


DRAWS = 60          # 200 in production; enough here for a 95th percentile
TRIALS = 20


def _opt(steps=40):
    import argparse
    return argparse.Namespace(fs=None, fs_search=[0.98, 1.02], fs_steps=steps)


def _rate(gen, seed, rule):
    rng = np.random.default_rng(seed)
    return sum(rule(*gen(rng)) for _ in range(TRIALS)) / TRIALS


def _new(our, shen):
    return not _align(our, shen, 0.0, 100.0, n_null=DRAWS)["unalignable"]


def _old(our, shen):
    return _legacy_alignable(our, shen, 0.0, 100.0)[0]


def _same_rate(rng):
    """Same rate, no timebase relation — the case the old rule failed on."""
    return (_synth_train(rng, 57, 0.85, 0.030, rng.uniform(0, 1)),
            _synth_train(rng, 57, 0.85, 0.030,
                         rng.uniform(0, 1) + rng.uniform(-2, 2)))


def _noise(rng):
    """Independently drawn rates: nothing to find."""
    return (_synth_train(rng, 57, 60.0 / rng.uniform(50, 100), 0.030,
                         rng.uniform(0, 1)),
            _synth_train(rng, 57, 60.0 / rng.uniform(50, 100), 0.030,
                         rng.uniform(0, 1)))


def _aligned(rng):
    """Truly aligned: their beat ONSET vs our systolic PEAK, 2 s clock offset,
    15% of our beats missed — a realistic detector, not a perfect one."""
    base = _synth_train(rng, 60, 0.85, 0.045, rng.uniform(0, 1))
    shen = base - 0.15 + rng.normal(0, 0.020, 60)
    keep = rng.random(60) > 0.15
    return base[keep] + 2.0 + rng.normal(0, 0.030, int(keep.sum())), shen


def test_the_old_rule_really_did_align_noise():
    """The baseline the fix is measured against. If this ever stops failing,
    the null experiment has drifted and the other assertions mean nothing."""
    assert _rate(_same_rate, 20260911, _old) >= 0.5


@pytest.mark.parametrize("gen, seed", [(_same_rate, 20260911), (_noise, 424242)])
def test_no_offset_survives_the_null_when_there_is_nothing_to_find(gen, seed):
    assert _rate(gen, seed, _new) == 0.0


def test_a_real_alignment_still_passes():
    """A null that refuses everything would pass the test above and answer no
    question. Power is the other half of the claim."""
    assert _rate(_aligned, 11, _new) >= 0.9


def test_a_discontinuous_capture_clock_is_refused_outright():
    rng = np.random.default_rng(3)
    our, shen = _aligned(rng)
    out = _align(our, shen, 0.0, 100.0, refuse="capture clock is "
                 "discontinuous, 2 segments", n_null=DRAWS)
    assert out["unalignable"] and "discontinuous" in out["reason"]
    assert not np.isfinite(out["matched_frac"])      # nothing is computed


def test_arm_b_heart_rate_cannot_disagree_with_the_train_it_was_fitted_to():
    """The label on the row is a measurement, not an opinion: the waveform's
    TRUE sample rate is varied over a factor of 8 and arm B's bpm does not
    move, because fs is derived from — and refined against — the same train."""
    bpms = []
    for true_fs in (15.0, 30.0, 60.0, 120.0):
        sig, beats = _synth_wave(true_fs, 72.0, 48.0)
        doc = {"ppg": {"signal": [float(v) for v in sig]},
               "heartbeats": [{"start_location_sec": float(b),
                               "end_location_sec": float(b + 0.35)}
                              for b in beats],
               "measurement": {"recorder_duration_ms": 48000.0}}
        fs0, src = _start_fs(doc, sig)
        fsi = _fit_fs(doc, sig, beats, fs0, src, _opt())
        det = np.sort(np.array([b.t_s for b in _detect_wave(sig, fsi["fs"])]))
        bpms.append(60.0 / float(np.median(np.diff(det))))
    assert max(bpms) - min(bpms) < 1.0               # measured: 0.0 bpm spread
    assert all(abs(b - 72.0) <= 5.0 for b in bpms)   # "passes" every time


def test_a_void_drift_verdict_is_re_fitted_wide_before_it_is_reported():
    """A train covering a sub-span biases fs0 out of the +/-2% grid: measured
    2026-09-11, fs0 32.591 for a true 30.000 with the narrow grid stuck at
    31.940 and -64.7 ms/s. The wide re-fit must reach the true rate."""
    sig, beats = _synth_wave(30.0, 72.0, 48.0)
    train = beats[(beats >= 4.0) & (beats <= 44.0)]
    doc = {"ppg": {"signal": [float(v) for v in sig]},
           "heartbeats": [{"start_location_sec": float(b),
                           "end_location_sec": float(b + 0.35)}
                          for b in train],
           "measurement": {"recorder_duration_ms": 48000.0}}
    fs0, src = _start_fs(doc, sig)
    fsi = _fit_fs(doc, sig, train, fs0, src, _opt(60))
    assert fs0 > 32.0                                 # the bias is real
    assert fsi["narrow"]["fs"] > 31.5                 # the narrow grid cannot
    assert abs(fsi["fs"] - 30.0) < 0.3                # the wide one can
    # fs0/duration IS the sub-span (train ends at ~44 s of 48 s); the fitted
    # rate agrees with the estimate that never saw the train.
    assert abs(fsi["fs_ratio"] - 1.0) < 0.02
    assert fsi["fs0_ratio"] > 1.05
