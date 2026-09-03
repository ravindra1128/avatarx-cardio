"""v0.5 vasotone fixtures — envelope hooks stay backward compatible,
the tone envelope carries the injected response and vasomotion, the
gamma sweep perturbs frames nonlinearly, and the dataset generator
writes a complete schema-valid provocation campaign with both null arms,
a retest visit, and the uncontrolled-optics W-d fuel."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

from scripts.make_synth_vasotone import (DURATION_S, PHASES_S,
                                         gamma_sweep, pi_sidecar,
                                         tone_envelope)


def test_tone_envelope_shape_and_vasomotion():
    fps = 30.0
    n = int(DURATION_S * fps)
    env = tone_envelope(n, fps, 0.3, seed=1)
    t = np.arange(n) / fps
    base = env[(t >= 5) & (t < 20)]
    stim = env[(t >= 35) & (t < 48)]
    assert abs(float(np.mean(base)) - 1.0) < 0.02
    assert abs(float(np.mean(stim)) - 0.7) < 0.02
    # vasomotion wobble present at ~0.08 Hz (peak-to-peak ~2*depth)
    assert 0.04 < float(np.ptp(base)) < 0.15
    # zero response = flat (plus wobble only)
    flat = tone_envelope(n, fps, 0.0, seed=1)
    assert abs(float(np.mean(flat[(t >= 35) & (t < 48)])) - 1.0) < 0.02


def test_gamma_sweep_only_inside_stimulus():
    fps = 30.0
    n = int(DURATION_S * fps)
    g = gamma_sweep(n, fps, seed=0)
    t = np.arange(n) / fps
    b0, b1 = PHASES_S["stimulus"]
    assert np.all(g[(t < b0) | (t >= b1)] == 1.0)
    assert float(np.ptp(g[(t >= b0) & (t < b1)])) > 0.1


def test_envelope_hooks_validate_and_stay_optional(tmp_path):
    pytest.importorskip("cv2")
    from scripts.make_synth_video import synth_video
    with pytest.raises(ValueError, match="amplitude_envelope"):
        synth_video(str(tmp_path / "x.avi"), duration_s=2.0,
                    amplitude_envelope=[1.0, 2.0])       # wrong length
    with pytest.raises(ValueError, match="optics_gamma"):
        synth_video(str(tmp_path / "y.avi"), duration_s=2.0,
                    optics_gamma_envelope=[0.0] * 60)    # non-positive
    t = synth_video(str(tmp_path / "z.avi"), duration_s=2.0, seed=3)
    assert t["amplitude_envelope"] is None
    assert t["optics_gamma_envelope"] is None


def test_pi_sidecar_tracks_the_response(tmp_path):
    from datasets.schema import pi_from_dict
    doc = pi_sidecar(str(tmp_path / "p.pi.json"), 0.35, seed=2)
    pi = pi_from_dict(json.loads((tmp_path / "p.pi.json").read_text()))
    v = np.asarray(pi.values)
    t = np.arange(v.size) / pi.fs_hz
    base = float(np.median(v[(t >= 5) & (t < 20)]))
    stim = float(np.median(v[(t >= 35) & (t < 48)]))
    assert (base - stim) / base > 0.25            # ~35% constriction


def test_make_vasotone_dataset_complete_and_schema_valid(tmp_path):
    pytest.importorskip("cv2")
    from datasets.io import load_recording
    from datasets.schema import pi_from_dict, provocation_from_dict
    from scripts.make_synth_vasotone import make_vasotone_dataset
    m = make_vasotone_dataset(tmp_path, n_participants=2, seed=23,
                              retest_participants=1)
    parts = m["participants"]
    assert len(parts) == 2 and m["note"].startswith("synthetic")
    p0 = parts[0]
    # participant 0: cold_pressor + paced_breathing + both nulls + retest
    assert len(p0["recordings"]) == 5
    arms = set()
    for rid in p0["recordings"]:
        rec = load_recording(str(tmp_path / f"{rid}.recording.json"))
        ok, why = rec.is_valid_for_beat_analysis()
        assert ok, why
        prov = provocation_from_dict(json.loads(
            (tmp_path / f"{rid}.provocation.json").read_text()))
        arms.add(prov.maneuver)
        pi_from_dict(json.loads(
            (tmp_path / f"{rid}.pi.json").read_text()))
        if prov.maneuver == "null_optics":
            assert prov.optics_log
    assert arms == {"cold_pressor", "paced_breathing", "null_optics",
                    "null_rest"}
    # retest = same maneuver, DIFFERENT session/visit
    r1 = load_recording(str(tmp_path / f"{p0['recordings'][0]}"
                            ".recording.json"))
    rt = load_recording(str(tmp_path / f"{p0['recordings'][-1]}"
                            ".recording.json"))
    assert r1.session_id != rt.session_id
    # the W-d fuel: uncontrolled-optics manifest really is unlocked
    uc = m["uncontrolled"][0]
    mdict = json.loads((tmp_path / f"{uc}.recording.json").read_text())
    assert mdict["capture"]["exposure_locked"] is False
