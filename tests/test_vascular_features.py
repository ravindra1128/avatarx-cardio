"""v0.4-vascular T1 (features) — the morphology extractor reads known
analytic waveforms correctly, tracks shape changes in the physiologic
direction, computes the identical set on the contact arm, refuses
non-ACCEPT scans (V-c), and — the miniature T1 experiment — recovers a
stiffness-signed feature contrast through the REAL production video
path."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

from research.vascular.features import (FEATURE_NAMES, beat_morphology,
                                        contact_session_features,
                                        session_median)
from scripts.make_synth_video import beat_waveform
from scripts.make_synth_vascular import shape_for_stiffness

FS = 240.0


def _beat(shape, rr=0.85, fs=FS):
    t = np.linspace(0, 1, int(rr * fs), endpoint=False)
    return beat_waveform(t, shape)


def test_beat_morphology_reads_known_fiducials():
    sh = shape_for_stiffness(6.0)
    f = beat_morphology(_beat(sh), FS)
    assert f is not None
    # systolic peak location is known exactly
    assert abs(f["rise_time_s"] - sh["peak_frac"] * 0.85) < 0.04
    assert f["notch_present"] == 1.0
    assert abs(f["notch_time_frac"] - sh["notch_frac"]) < 0.06
    assert f["norm_upstroke_slope"] and f["norm_upstroke_slope"] > 0
    assert f["pulse_width50_s"] and 0.05 < f["pulse_width50_s"] < 0.6
    assert set(f) == set(FEATURE_NAMES)


def test_features_track_shape_in_the_physiologic_direction():
    lo = beat_morphology(_beat(shape_for_stiffness(5.0)), FS)
    hi = beat_morphology(_beat(shape_for_stiffness(12.0)), FS)
    # stiffer: bigger reflected wave, shallower notch (higher rel amp),
    # faster rise
    assert hi["reflection_index"] > lo["reflection_index"]
    assert hi["notch_rel_amp"] > lo["notch_rel_amp"]
    assert hi["rise_time_s"] < lo["rise_time_s"]


def test_beat_morphology_fails_closed():
    assert beat_morphology(np.ones(4), FS) is None          # too short
    assert beat_morphology(np.full(200, np.nan), FS) is None
    flat = np.zeros(int(0.85 * FS))
    assert beat_morphology(flat, FS) is None                # no amplitude
    # a beat at implausible duration is refused
    assert beat_morphology(_beat(shape_for_stiffness(8.0), rr=3.0), FS) \
        is None


def test_sdppg_gated_by_native_sampling_rate():
    sh = shape_for_stiffness(7.0)
    ref = beat_morphology(_beat(sh), FS, native_fs=250.0)
    cam = beat_morphology(_beat(sh), FS, native_fs=30.0)
    assert ref["sdppg_b_over_a"] is not None
    # resampling must not conjure derivative bandwidth from a 30 fps clip
    assert cam["sdppg_b_over_a"] is None and cam["sdppg_d_over_a"] is None


def test_session_median_and_notch_fraction():
    sh = shape_for_stiffness(6.0)
    rows = [beat_morphology(_beat(sh, rr=r), FS)
            for r in (0.78, 0.82, 0.86, 0.9, 0.84, 0.8, 0.88, 0.83)]
    rows.append(None)                                # a refused beat
    no_notch = dict(rows[0])
    no_notch.update(notch_present=0.0, notch_rel_amp=None,
                    notch_time_frac=None, reflection_index=None)
    rows.append(no_notch)
    s = session_median(rows)
    assert s["n_beats_used"] == 9
    assert 0.8 < s["features"]["notch_present"] < 1.0
    assert s["features"]["rise_time_s"] is not None
    assert any(k.endswith("_iqr") for k in s["quality"])


def test_contact_arm_computes_the_identical_feature_set(tmp_path):
    from datasets.schema import contact_ppg_from_dict
    from scripts.make_synth_vascular import contact_ppg_sidecar
    rng = np.random.default_rng(2)
    rr = np.clip(rng.normal(0.85, 0.03, 24), 0.5, 1.4)
    sh = shape_for_stiffness(6.0)
    contact_ppg_sidecar(str(tmp_path / "c.ppg.json"), rr, sh, seed=4)
    ppg = contact_ppg_from_dict(json.loads(
        (tmp_path / "c.ppg.json").read_text()))
    out = contact_session_features(ppg)
    assert out["available"] and out["n_beats_used"] >= 8
    assert set(out["features"]) == set(FEATURE_NAMES)
    # band-limited rise differs from the analytic peak fraction by a
    # smoothing offset; the cross-stiffness ORDERING below is the claim
    assert abs(out["features"]["rise_time_s"]
               - sh["peak_frac"] * float(np.mean(rr))) < 0.08
    assert out["features"]["notch_present"] > 0.7
    assert out["features"]["sdppg_b_over_a"] is not None
    # and the contact arm separates stiffness levels cleanly
    contact_ppg_sidecar(str(tmp_path / "h.ppg.json"), rr,
                        shape_for_stiffness(12.0), seed=4)
    hi = contact_session_features(contact_ppg_from_dict(json.loads(
        (tmp_path / "h.ppg.json").read_text())))
    assert hi["features"]["reflection_index"] > \
        out["features"]["reflection_index"]


# ----------------------------------------------- through the video path
@pytest.fixture(scope="module")
def vascular_clips(tmp_path_factory):
    pytest.importorskip("cv2")
    from scripts.make_synth_video import synth_video
    d = tmp_path_factory.mktemp("vasc_clips")
    rng = np.random.default_rng(5)
    out = {}
    for name, s in (("lo", 5.0), ("hi", 12.0)):
        rr = np.clip(rng.normal(0.85, 0.03, 30), 0.5, 1.4)
        synth_video(str(d / f"{name}.avi"), kind="vascular_sinus",
                    fps=60.0, duration_s=20.0, seed=50 + int(s),
                    rr_override=rr, pulse_shape=shape_for_stiffness(s))
        out[name] = str(d / f"{name}.avi")
    # too short for ACCEPT -> the V-c refusal fuel
    synth_video(str(d / "short.avi"), kind="sinus", fps=30.0,
                duration_s=6.0, seed=77)
    out["short"] = str(d / "short.avi")
    return out


def test_vc_no_features_on_non_accept_scans(vascular_clips):
    from research.vascular.features import facial_session_features
    out = facial_session_features(vascular_clips["short"])
    assert out["available"] is False
    assert any("V-c" in r for r in out["reasons"])
    assert "features" not in out


def test_morphology_survives_the_production_video_path(vascular_clips):
    """The miniature T1 experiment: two clips whose ONLY physiologic
    difference is latent stiffness must come out of the full production
    path (ingest -> tracking -> rPPG -> gates) with (a) features at all
    and (b) the stiffness-signed contrast preserved."""
    from research.vascular.features import facial_session_features
    lo = facial_session_features(vascular_clips["lo"])
    hi = facial_session_features(vascular_clips["hi"])
    assert lo["available"], lo
    assert hi["available"], hi
    assert lo["n_beats_used"] >= 8 and hi["n_beats_used"] >= 8
    fl, fh = lo["features"], hi["features"]
    # reflection up, rise faster with stiffness — through the camera
    assert fh["reflection_index"] > fl["reflection_index"]
    assert fh["rise_time_s"] < fl["rise_time_s"]
    # 60 fps carries SDPPG; the band is the wide morphology band
    assert lo["sdppg_derivable"] is True
    assert lo["band_hz"][1] <= 10.0
