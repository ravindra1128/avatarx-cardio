"""v0.4-vascular fixtures — the pulse-shape hook stays bit-for-bit
backward compatible, the stiffness couplings are monotone and
physiologically signed, and the dataset generator writes a complete,
schema-valid paired campaign (video + contact PPG + cfPWV reference)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

from scripts.make_synth_video import (DEFAULT_PULSE_SHAPE, beat_waveform,
                                      pulse_series)
from scripts.make_synth_vascular import (contact_ppg_sidecar,
                                         make_vascular_dataset,
                                         shape_for_stiffness)


def test_pulse_shape_default_is_bit_for_bit_backward_compatible():
    rr = np.full(12, 0.85)
    t = np.linspace(0, 1, 400, endpoint=False)
    legacy = np.exp(-((t - 0.18) ** 2) / (2 * 0.075 ** 2))
    assert np.allclose(beat_waveform(t), legacy)
    a, pk_a, on_a = pulse_series(rr, 30.0)
    b, pk_b, on_b = pulse_series(rr, 30.0, shape=None)
    assert np.array_equal(a, b) and np.array_equal(pk_a, pk_b)
    # explicit default dict is the same waveform as None
    c, _, _ = pulse_series(rr, 30.0, shape=dict(DEFAULT_PULSE_SHAPE))
    assert np.array_equal(a, c)


def test_stiffness_couplings_monotone_and_physiologic():
    lo, mid, hi = (shape_for_stiffness(s) for s in (5.0, 8.0, 12.0))
    # stiffer -> larger, earlier reflection; shallower, earlier notch;
    # faster systolic rise
    assert lo["refl_amp"] < mid["refl_amp"] < hi["refl_amp"]
    assert lo["refl_frac"] > mid["refl_frac"] > hi["refl_frac"]
    assert lo["notch_depth"] > mid["notch_depth"] > hi["notch_depth"]
    assert lo["notch_frac"] > mid["notch_frac"] > hi["notch_frac"]
    assert lo["peak_frac"] > hi["peak_frac"]
    # all params stay inside their clamps at extreme stiffness
    for s in (3.0, 20.0):
        sh = shape_for_stiffness(s)
        assert 0.0 < sh["refl_amp"] <= 0.75
        assert 0.0 < sh["notch_depth"] <= 0.32


def test_compliant_waveform_has_notch_between_systole_and_reflection():
    t = np.linspace(0, 1, 2000, endpoint=False)
    w = beat_waveform(t, shape_for_stiffness(5.0))
    sh = shape_for_stiffness(5.0)
    i_sys = int(sh["peak_frac"] * 2000)
    i_refl = int(sh["refl_frac"] * 2000)
    notch_region = w[i_sys:i_refl]
    i_notch = i_sys + int(np.argmin(notch_region))
    # a real local minimum strictly between the two waves
    assert w[i_notch] < w[i_sys] and w[i_notch] < w[i_refl]
    # and at high stiffness the notch is shallower (relative to systole)
    w_hi = beat_waveform(t, shape_for_stiffness(12.0))
    sh_hi = shape_for_stiffness(12.0)
    rel_lo = 1.0 - w[i_notch] / w[i_sys]
    j_sys = int(sh_hi["peak_frac"] * 2000)
    j_refl = int(sh_hi["refl_frac"] * 2000)
    j_notch = j_sys + int(np.argmin(w_hi[j_sys:j_refl]))
    rel_hi = 1.0 - w_hi[j_notch] / w_hi[j_sys]
    assert rel_hi < rel_lo


def test_contact_ppg_sidecar_shape(tmp_path):
    rr = np.full(10, 0.8)
    doc = contact_ppg_sidecar(str(tmp_path / "x.ppg.json"), rr,
                              shape_for_stiffness(8.0), seed=3)
    from datasets.schema import contact_ppg_from_dict
    p = contact_ppg_from_dict(json.loads(
        (tmp_path / "x.ppg.json").read_text()))
    assert p.fs_hz == 250.0
    assert abs(len(p.samples) - 8.0 * 250.0) <= 2
    # the clean contact waveform correlates near-perfectly with the
    # noiseless morphology it wraps
    clean, _, _ = pulse_series(rr, 250.0, shape=shape_for_stiffness(8.0))
    r = np.corrcoef(np.asarray(p.samples), clean)[0, 1]
    assert r > 0.99


def test_make_vascular_dataset_is_complete_and_schema_valid(tmp_path):
    pytest.importorskip("cv2")
    from datasets.io import load_recording
    from datasets.schema import contact_ppg_from_dict, vascular_from_dict
    m = make_vascular_dataset(tmp_path, n_participants=4, seed=13,
                              duration_s=6.0, retest_participants=1)
    parts = m["participants"]
    assert len(parts) == 4 and m["note"].startswith("synthetic")
    # retest participant carries two recordings in ONE session
    assert len(parts[0]["recordings"]) == 2
    # the last participant has no reference -> exclusion path fuel
    assert parts[-1]["has_pwv"] is False
    for p in parts:
        for rid in p["recordings"]:
            assert (tmp_path / f"{rid}.avi").exists()
            assert (tmp_path / f"{rid}.ppg.json").exists()
            rec = load_recording(str(tmp_path / f"{rid}.recording.json"))
            assert rec.participant_id == p["participant_id"]
            ok, reasons = rec.is_valid_for_beat_analysis()
            assert ok, reasons
            contact_ppg_from_dict(json.loads(
                (tmp_path / f"{rid}.ppg.json").read_text()))
            pwv = tmp_path / f"{rid}.pwv.json"
            assert pwv.exists() == p["has_pwv"]
            if p["has_pwv"]:
                ref = vascular_from_dict(json.loads(pwv.read_text()))
                assert abs(ref.cfpwv_mps - p["cfpwv_label_mps"]) < 1e-6
    # retest pair shares session_id and carries a repeat reference read
    r1, r2 = parts[0]["recordings"]
    a = load_recording(str(tmp_path / f"{r1}.recording.json"))
    b = load_recording(str(tmp_path / f"{r2}.recording.json"))
    assert a.session_id == b.session_id
    ref = vascular_from_dict(json.loads(
        (tmp_path / f"{r1}.pwv.json").read_text()))
    assert ref.cfpwv_mps_repeat is not None
