"""v0.5 vasotone T1 — the tone-feature extractor: W-c emission shape
(never absolute-only), windowed statistics on constructed series, the
contact PI response, W-d optics gating, V-c refusal, and the miniature
T1 experiment: an injected constriction survives the production video
path as a within-session amplitude delta while a null-rest clip stays
flat."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from research.vascular.tone_features import (AMPLITUDE_FEATURES,
                                             TONE_FEATURES, _delta,
                                             lf_power_fraction,
                                             pi_response,
                                             windowed_median,
                                             windowed_trend)


def test_delta_is_the_only_emission_shape():
    d = _delta(2.0, 1.5)
    assert set(d) == {"baseline", "response", "delta", "delta_norm"}
    assert d["delta"] == -0.5 and abs(d["delta_norm"] + 0.25) < 1e-9
    n = _delta(None, 1.0)
    assert n["delta"] is None and n["delta_norm"] is None


def test_windowed_stats_on_constructed_series():
    rng = np.random.default_rng(0)
    series = [(t, 1.0 + 0.01 * rng.standard_normal())
              for t in np.arange(0, 25, 0.8)]
    series += [(t, 0.7 + 0.01 * rng.standard_normal())
               for t in np.arange(25, 50, 0.8)]
    b, nb = windowed_median(series, (0, 25))
    r, nr = windowed_median(series, (25, 50))
    assert abs(b - 1.0) < 0.02 and abs(r - 0.7) < 0.02
    assert nb > 20 and nr > 20
    # too few beats in a window -> None, honest count
    v, n = windowed_median(series[:3], (0, 25))
    assert v is None and n == 3
    # a within-window ramp is a trend (fraction of median per minute)
    ramp = [(t, 1.0 + 0.002 * t) for t in np.arange(0, 25, 0.8)]
    tr = windowed_trend(ramp, (0, 25))
    assert tr is not None and abs(tr - 0.002 * 60.0) < 0.02
    assert abs(windowed_trend(series, (0, 25))) < 0.05


def test_lf_power_detects_vasomotion_and_fails_closed():
    t = np.arange(0, 25, 0.8)
    wobble = [(ti, 1.0 + 0.05 * np.sin(2 * np.pi * 0.09 * ti))
              for ti in t]
    rng = np.random.default_rng(1)
    flat = [(ti, 1.0 + 0.005 * rng.standard_normal()) for ti in t]
    p_wobble = lf_power_fraction(wobble, (0, 25))
    p_flat = lf_power_fraction(flat, (0, 25))
    assert p_wobble is not None and p_flat is not None
    assert p_wobble > 0.6 and p_wobble > 2 * p_flat
    # a window too short to resolve any of the LF band refuses
    assert lf_power_fraction(wobble[:8], (0, 6)) is None
    # review findings: coverage honesty — beats confined to a tail
    # sub-range of the span refuse (no clamp-extrapolation) ...
    tail = [(ti, 1.0) for ti in np.arange(18, 25, 0.8)]
    assert lf_power_fraction(tail, (0, 25)) is None
    # ... and a long interior gap refuses (no linear bridge)
    gappy = [(ti, 1.0 + 0.05 * np.sin(2 * np.pi * 0.09 * ti))
             for ti in np.arange(0, 25, 0.8) if not (8 < ti < 16)]
    assert lf_power_fraction(gappy, (0, 25)) is None


def test_pi_response_shape_and_value():
    from datasets.schema import PiTrace
    v = [2.0] * 25 + [1.4] * 25 + [2.0] * 10
    pi = PiTrace(fs_hz=1.0, values=[float(x) for x in v])
    r = pi_response(pi, (0, 25), (25, 50))
    assert set(r) == {"baseline", "response", "delta", "delta_norm"}
    assert abs(r["delta_norm"] + 0.3) < 0.01


# ---------------------------------------------- through the video path
@pytest.fixture(scope="module")
def tone_clips(tmp_path_factory):
    pytest.importorskip("cv2")
    from scripts.make_synth_vasotone import tone_envelope
    from scripts.make_synth_video import synth_video
    d = tmp_path_factory.mktemp("tone_clips")
    rng = np.random.default_rng(9)
    n = int(60.0 * 30.0)
    out = {}
    for name, resp in (("cp", 0.30), ("rest", 0.0)):
        rr = np.clip(rng.normal(0.82, 0.03, 100), 0.5, 1.4)
        synth_video(str(d / f"{name}.avi"), kind="vasotone",
                    fps=30.0, duration_s=60.0, seed=90 + int(resp * 10),
                    rr_override=rr,
                    amplitude_envelope=tone_envelope(n, 30.0, resp,
                                                     seed=7))
        out[name] = str(d / f"{name}.avi")
    synth_video(str(d / "short.avi"), kind="sinus", fps=30.0,
                duration_s=6.0, seed=91)
    out["short"] = str(d / "short.avi")
    return out


def _prov():
    from datasets.schema import provocation_from_dict
    return provocation_from_dict(
        {"maneuver": "cold_pressor",
         "phase_marks": {"baseline": [0.0, 25.0],
                         "stimulus": [25.0, 50.0],
                         "recovery": [50.0, 60.0]},
         "intensity": 2})


_LOCKED = {"exposure_locked": True, "awb_locked": True}


def _run(path):
    from inference.pipeline import run_with_details
    return run_with_details(path)


def test_constriction_survives_the_production_video_path(tone_clips):
    from research.vascular.tone_features import tone_session_features
    res, det = _run(tone_clips["cp"])
    out = tone_session_features(res, det, _prov(), capture=_LOCKED)
    assert out["available"], out
    # W-c: every feature is the 4-key delta dict, nothing absolute-only
    assert set(out["features"]) == set(TONE_FEATURES)
    for name, v in out["features"].items():
        assert set(v) == {"baseline", "response", "delta",
                          "delta_norm"}, name
    namp = out["features"]["norm_pulse_amplitude"]
    assert namp["delta_norm"] is not None
    assert -0.45 < namp["delta_norm"] < -0.18, namp
    # the null-rest clip stays flat through the same machinery
    res0, det0 = _run(tone_clips["rest"])
    out0 = tone_session_features(res0, det0, _prov(), capture=_LOCKED)
    flat = out0["features"]["norm_pulse_amplitude"]
    assert flat["delta_norm"] is not None
    assert abs(flat["delta_norm"]) < 0.10, flat


def test_wd_uncontrolled_optics_withholds_amplitude(tone_clips):
    from research.vascular.tone_features import tone_session_features
    res, det = _run(tone_clips["cp"])
    out = tone_session_features(res, det, _prov(),
                                capture={"exposure_locked": False,
                                         "awb_locked": True})
    assert out["uncontrolled_optics"] is True
    for k in AMPLITUDE_FEATURES:
        assert out["features"][k]["delta_norm"] is None, k
    assert any("W-d" in r for r in out["reasons"])


def test_vc_no_tone_features_on_non_accept(tone_clips):
    from research.vascular.tone_features import tone_session_features
    res, det = _run(tone_clips["short"])
    out = tone_session_features(res, det, _prov(), capture=_LOCKED)
    assert out["available"] is False
    assert any("V-c" in r for r in out["reasons"])
    assert "features" not in out


def test_null_optics_gamma_actually_reaches_the_features(tmp_path):
    """Review finding: without this, W1 could pass vacuously on
    synthetic — nothing proved the rig's perturbation perturbs the
    extractor. Two clips, identical physiology and seed; one gets the
    gamma sweep. The amplitude analog must move materially under gamma
    and stay quiet without it (probed: 0.022 vs 0.0015)."""
    pytest.importorskip("cv2")
    from datasets.schema import provocation_from_dict
    from inference.pipeline import run_with_details
    from research.vascular.tone_features import tone_session_features
    from scripts.make_synth_vasotone import gamma_sweep
    from scripts.make_synth_video import synth_video
    rng = np.random.default_rng(3)
    rr = np.clip(rng.normal(0.82, 0.03, 100), 0.5, 1.4)
    n = int(60 * 30)
    outs = {}
    prov = provocation_from_dict(
        {"maneuver": "null_optics",
         "phase_marks": {"baseline": [0, 25], "stimulus": [25, 50]},
         "optics_log": "gamma sweep (test)"})
    for name, gam in (("plain", None),
                      ("gamma", gamma_sweep(n, 30.0, seed=4))):
        synth_video(str(tmp_path / f"{name}.avi"), kind="v", fps=30.0,
                    duration_s=60.0, seed=77, rr_override=rr,
                    optics_gamma_envelope=gam)
        res, det = run_with_details(str(tmp_path / f"{name}.avi"))
        outs[name] = tone_session_features(
            res, det, prov, capture={"exposure_locked": True,
                                     "awb_locked": True})
    plain = outs["plain"]["features"]["norm_pulse_amplitude"]
    gamma = outs["gamma"]["features"]["norm_pulse_amplitude"]
    assert abs(plain["delta_norm"]) < 0.005, plain
    assert abs(gamma["delta_norm"]) > 0.01, gamma
    assert abs(gamma["delta_norm"]) > 3 * abs(plain["delta_norm"])
