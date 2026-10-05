"""features/pulse_harmonics.py: the recorded-only 2nd-harmonic shape descriptor.

Pins (1) that the estimator reads a known shape and refuses a shapeless one,
(2) the invariances that make it usable on the SDK's waveform (derived sample
rate, unknown time origin), (3) that client-supplied input can never raise,
and (4) that recording it changes NOTHING a user or a decision can see.
"""
import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from features import pulse_harmonics as ph


def pulse(fs=30.0, dur=50.0, hr=72.0, r2=0.40, phi2=80.0, snr_db=10.0, seed=1, hrv=0.03):
    """cos(th) + r2 cos(2 th + phi2) on a rate that breathes (+/-hrv at 0.25 Hz), plus white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur, 1.0 / fs)
    f = hr / 60.0 * (1.0 + hrv * np.sin(2 * np.pi * 0.25 * t))
    th = 2 * np.pi * np.cumsum(f) / fs
    x = np.cos(th) + r2 * np.cos(2 * th + math.radians(phi2))
    n = rng.standard_normal(t.size)
    return x + n * x.std() / n.std() / (10 ** (snr_db / 20.0))


def wrap(d):
    return abs((d + 180.0) % 360.0 - 180.0)


# ------------------------------------------------------------------ the estimator
@pytest.mark.parametrize("r2,phi2", [(0.40, 80.0), (0.25, 30.0), (0.50, -120.0)])
def test_reads_a_known_shape(r2, phi2):
    out = ph.estimate(pulse(r2=r2, phi2=phi2), 30.0, 72.0 / 60.0)
    assert out["state"] == "ok" and out["detectable"] is True
    assert wrap(out["phi2_deg"] - phi2) < 5.0
    assert abs(out["r2"] - r2) < 0.05
    assert out["f0_source"] == "reference_rate"


def test_a_shapeless_pulse_is_not_detectable():
    # A pure sinusoid at the phone clips' SNR: the deployed contour path reports a
    # dicrotic notch here on 90 of 90 inputs. This estimator must say "nothing there".
    ps = [ph.estimate(pulse(r2=0.0, snr_db=-5.0, seed=s), 30.0, 1.2)["p_value"] for s in range(12)]
    assert all(p is not None for p in ps)
    assert sum(p < ph.DETECTABLE_P for p in ps) <= 1          # p is ~uniform when nothing is there
    assert 0.2 < float(np.median(ps)) < 0.8


@pytest.mark.parametrize("hrv", [0.0, 0.03])
def test_a_metronomic_heart_is_still_detectable(hrv):
    # The first draft scored detectability against circular shifts of the fundamental.
    # That only breaks the phase lock when the rate VARIES, so a perfectly regular heart
    # read a clean, strongly shaped pulse as absent - biased against the low-HRV people
    # a stiffness question is about. The block test must not care about rate variability.
    out = ph.estimate(pulse(hrv=hrv, snr_db=5.0), 30.0, 1.2)
    assert out["detectable"] is True and out["p_value"] < 1e-3 and out["blocks"] >= ph.MIN_BLOCKS
    assert wrap(out["phi2_deg"] - 80.0) < 10.0               # one draw at +5 dB: SD ~2.6 deg


def test_a_real_shape_at_the_clips_noise_level_is_found():
    # Power, not only specificity: -5 dB is what the compressed phone clips measure.
    found = [ph.estimate(pulse(snr_db=-5.0, seed=s), 30.0, 1.2)["detectable"] for s in range(12)]
    assert sum(found) >= 9


def test_survives_the_live_frame_band_and_noise():
    # The SDK band-passes ~0.7-4 Hz; the 2nd harmonic (2.4 Hz at 72 bpm) sits inside it.
    from scipy.signal import butter, filtfilt
    x = pulse(snr_db=5.0)
    b, a = butter(3, [0.7 / 15.0, 4.0 / 15.0], btype="band")
    out = ph.estimate(filtfilt(b, a, x), 30.0, 1.2)
    assert out["detectable"] and wrap(out["phi2_deg"] - 80.0) < 6.0


def test_invariant_to_a_wrong_sample_rate():
    # The SDK's fs is DERIVED by the client (samples / last beat end); t0 is unknown.
    x = pulse()
    base = ph.estimate(x, 30.0, 1.2)
    for scale in (0.97, 1.03):
        out = ph.estimate(x, 30.0 * scale, 1.2)            # the hint is the TRUE rate
        assert out["state"] == "ok"
        assert wrap(out["phi2_deg"] - base["phi2_deg"]) < 1.0
        assert abs(out["r2"] - base["r2"]) < 0.01


def test_invariant_to_a_time_shift_and_gain():
    x = pulse()
    base = ph.estimate(x, 30.0, 1.2)
    out = ph.estimate(7.5 * x[11:] + 3.0, 30.0, 1.2)       # exposure/gain/offset, unknown origin
    assert wrap(out["phi2_deg"] - base["phi2_deg"]) < 1.5
    assert abs(out["r2"] - base["r2"]) < 0.01


def test_polarity_flips_phi2_by_half_a_turn():
    x = pulse()
    a, b = ph.estimate(x, 30.0, 1.2), ph.estimate(-x, 30.0, 1.2)
    assert wrap(wrap(a["phi2_deg"] - b["phi2_deg"]) - 180.0) < 1.0
    assert abs(a["r2"] - b["r2"]) < 1e-6


def test_deterministic():
    x = pulse()
    assert ph.estimate(x, 30.0, 1.2) == ph.estimate(x.copy(), 30.0, 1.2)


def test_without_a_rate_hint_it_finds_the_fundamental():
    out = ph.estimate(pulse(), 30.0, None)
    assert out["state"] == "ok" and out["f0_source"] == "spectral_peak"
    assert abs(out["f0_hz"] - 1.2) < 0.06


def test_halves_of_one_scan_agree_on_a_stable_shape():
    out = ph.estimate(pulse(dur=60.0), 30.0, 1.2)
    assert out["half_split_dphi2_deg"] is not None and out["half_split_dphi2_deg"] < 8.0


def test_crest_fraction_is_the_readable_form():
    assert ph.crest_fraction_2h(0.0, 0.0) == pytest.approx(0.5, abs=0.01)      # a sinusoid: T/2
    assert ph.crest_fraction_2h(0.37, 82.0) == pytest.approx(0.354, abs=0.01)  # the 2026-09-16 scans


# ------------------------------------------------------------------ boundary: never raises
@pytest.mark.parametrize("fs", [None, float("nan"), 0.0, -30.0, 5.0, 1000.0, "30", True])
def test_bad_sample_rates_are_a_state(fs):
    assert ph.estimate(pulse(), fs, 1.2)["state"] == "invalid_sample_rate"


def test_refusals_are_states_not_exceptions():
    x = pulse()
    assert ph.estimate(x[:300], 30.0, 1.2)["state"] == "too_short"
    assert ph.estimate(np.zeros(1500), 30.0, 1.2)["state"] == "flat_signal"
    holes = x.copy()
    holes[::10] = np.nan                                                       # 10 % missing
    assert ph.estimate(holes, 30.0, 1.2)["state"] == "too_many_missing_samples"
    few = x.copy()
    few[[40, 400, 900]] = np.nan                                               # interpolated
    assert ph.estimate(few, 30.0, 1.2)["state"] == "ok"
    assert ph.estimate(pulse(fs=12.0, hr=170.0), 12.0, 170 / 60.0)["state"] == "second_harmonic_above_nyquist"


@pytest.mark.parametrize("raw,state", [
    (None, "not_received"), ([], "not_received"), ({}, "no_waveform"),
    ({"ppg": "x"}, "no_waveform"), ({"ppg": {"signal": []}}, "no_waveform"),
    ({"ppg": {"signal": [1.0] * 900}}, "invalid_sample_rate"),
    ({"ppg": {"signal": [1.0] * 900, "fs_hz": "30"}}, "invalid_sample_rate"),
    ({"ppg": {"signal": [None, "a", True, {}] * 400, "fs_hz": 30.0}}, "too_many_missing_samples"),
])
def test_a_hostile_snapshot_is_a_state(raw, state):
    assert ph.from_sdk_snapshot(raw)["state"] == state


def test_the_snapshot_path_reads_the_waveform_and_the_sdk_rate():
    x = pulse(fs=31.0)
    sig = [round(float(v), 5) for v in x]
    sig[50] = None                                                             # the client may send a hole
    raw = {"ppg": {"signal": sig, "fs_hz": 31.0, "fs_source": "derived_from_beats"},
           "reference": {"heart_rate_bpm": 72.0}}
    out = ph.describe("sdk", raw)
    assert out["state"] == "ok" and wrap(out["phi2_deg"] - 80.0) < 5.0
    assert out["polarity"] == "as_delivered" and out["fs_source"] == "derived_from_beats"
    assert (out["mode"], out["contributes_to_published_result"], out["source"]) == ("recorded_only", False, "sdk")


def test_an_oversized_snapshot_is_bounded():
    raw = {"ppg": {"signal": [float(v) for v in pulse(dur=400.0)], "fs_hz": 30.0}}
    assert ph.from_sdk_snapshot(raw, 1.2)["span_s"] <= ph.MAX_SPAN_S


def test_every_record_is_strict_json():
    records = [ph.describe("sdk", None), ph.describe("clip", None), ph.describe("traces", None),
               ph.describe("sdk", {"ppg": {"signal": [float(v) for v in pulse()], "fs_hz": 30.0}})]
    for rec in records:
        json.dumps(rec, allow_nan=False)
    with pytest.raises(ValueError):
        ph.describe("finger", None)


# ------------------------------------------------------------------ our own signal
def fake_ingest(dur=50.0, fs=30.0, hole_at=None, ok=True):
    """Four regions carrying one pulse in the POS direction, on a capture clock."""
    from preprocessing.roi import ROI_NAMES
    rng = np.random.default_rng(3)
    p = pulse(fs=fs, dur=dur, snr_db=40.0)
    ts = np.arange(p.size) / fs
    if hole_at is not None:
        ts[ts >= hole_at] += 2.0                                               # a recorder hole
    traces = {}
    for k, roi in enumerate(ROI_NAMES):
        base = np.array([150.0, 110.0, 90.0])
        gain = np.array([0.33, 0.77, 0.53]) * 0.01 * (1.0 + 0.2 * k)           # the pulsatile RGB direction
        traces[roi] = base * (1.0 + np.outer(p, gain)) + rng.standard_normal((p.size, 3)) * 0.15
    return SimpleNamespace(ok=ok, traces=traces, timestamps_s=ts)


def test_our_own_regions_are_fused_into_one_descriptor():
    from rppg._filters import orient_peaks_up
    out = ph.describe("traces", fake_ingest(), 72.0)
    assert out["state"] == "ok" and out["detectable"] and out["regions"] == 4
    assert out["polarity"] == "skewness_oriented"
    # rppg.pos orients each region peaks-up by skewness, which may invert this shape:
    # the expected phase follows that rule, it is not assumed.
    clean = pulse(snr_db=80.0)
    expected = 80.0 if np.array_equal(orient_peaks_up(clean), clean) else 80.0 - 180.0
    assert wrap(out["phi2_deg"] - expected) < 10.0


def test_a_recorder_hole_is_not_bridged():
    out = ph.from_ingest(fake_ingest(dur=70.0, hole_at=45.0), 1.2)
    assert out["state"] == "ok" and out["span_s"] < 46.0                        # the longest gap-free stretch


@pytest.mark.parametrize("ing", [None, SimpleNamespace(ok=False, traces={}, timestamps_s=[]),
                                 SimpleNamespace(ok=True, traces={}, timestamps_s=[])])
def test_no_ingest_is_a_state(ing):
    assert ph.from_ingest(ing)["state"] == "not_available"


def test_a_short_stretch_is_a_state():
    assert ph.from_ingest(fake_ingest(dur=12.0))["state"] == "too_short"


# ------------------------------------------------------------------ recorded only
def test_recording_changes_nothing_but_its_own_debug_key():
    from app import measure_api as api
    doc = {"outcome": "REPEAT_SCAN", "afib_result": "INCONCLUSIVE", "mean_pulse_rate_bpm": 71.0,
           "biomarkers": {"items": [{"key": "arterial_stiffness", "value": 66.0, "tier": "provisional"}]},
           "debug": {"shenai_input": {"sdk_hr_bpm": 72.0}, "evidence": {"n_beats": 40}}}
    before = copy.deepcopy(doc)
    raw = {"ppg": {"signal": [float(v) for v in pulse()], "fs_hz": 30.0}}
    api._record_pulse_harmonics(doc, "sdk", raw)
    api._record_pulse_harmonics(doc, "clip", None)
    added = {k: doc["debug"].pop(k) for k in ("pulse_harmonics_sdk", "pulse_harmonics_clip")}
    assert doc == before                                                       # cards, result, evidence: untouched
    assert added["pulse_harmonics_sdk"]["state"] == "ok"
    assert added["pulse_harmonics_sdk"]["f0_source"] == "reference_rate"       # the SDK's own rate was the hint
    assert added["pulse_harmonics_clip"]["state"] == "not_available"
    assert all(r["contributes_to_published_result"] is False for r in added.values())


def test_a_failure_inside_the_recorder_never_reaches_the_scan(monkeypatch):
    from app import measure_api as api

    def boom(*a, **k):
        raise RuntimeError("estimator exploded")
    monkeypatch.setattr(ph, "describe", boom)
    doc = {"outcome": "ACCEPT"}
    api._record_pulse_harmonics(doc, "sdk", {"ppg": {}})
    assert doc["outcome"] == "ACCEPT"
    assert doc["debug"]["pulse_harmonics_sdk"]["state"] == "assessment_failed"


def test_the_sheet_carries_the_record_and_stays_blank_without_it():
    from app import result_sheet as rs
    cols = ["H2 SDK r2", "H2 SDK phi2", "H2 SDK p", "H2 Traces r2", "H2 Traces phi2", "H2 Traces p",
            "H2 Clip phi2", "H2 Clip p", "H2 Note"]
    assert all(c in rs.COLUMNS for c in cols) and len(rs.COLUMNS) == len(set(rs.COLUMNS))
    legacy = rs.row_from_doc({"outcome": "NO_RESULT"})
    assert all(legacy[c] == "" for c in cols)                                  # a doc from before this build
    doc = {"outcome": "REPEAT_SCAN", "debug": {
        "pulse_harmonics_sdk": ph.describe("sdk", {"ppg": {"signal": [float(v) for v in pulse()], "fs_hz": 30.0},
                                                   "reference": {"heart_rate_bpm": 72.0}}),
        "pulse_harmonics_clip": ph.describe("clip", None)}}
    row = rs.row_from_doc(doc)
    assert all(k in rs.COLUMNS for k in row)
    assert abs(float(row["H2 SDK phi2"]) - 80.0) < 5.0 and float(row["H2 SDK p"]) < 0.01
    assert row["H2 Clip phi2"] == "" and row["H2 Traces r2"] == ""              # a refusal is never a number
    assert "sdk ok" in row["H2 Note"] and "clip not_available" in row["H2 Note"]
