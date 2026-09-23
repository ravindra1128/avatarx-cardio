"""Facial perfusion index (features/facial_perfusion.py): the Vascular Tone
card's measurement since 2026-09-23.

Synthetic trace documents in the client's schema (t_s, traces, bbox), built
in LINEAR light and encoded to sRGB code values the way a camera would, with
a known pulsatile fraction. Each test pins one property the estimator must
have to be a measurement rather than a noise index."""
from __future__ import annotations

import math

import numpy as np
import pytest

from features import facial_perfusion as fpm

ROIS = fpm.ROI_NAMES
PBV = np.array([0.33, 0.77, 0.53]) / 0.77        # blood-volume pulse signature, green = 1
DC = {"forehead": (0.55, 0.40, 0.30), "cheek_l": (0.45, 0.30, 0.22),
      "cheek_r": (0.50, 0.34, 0.25), "nose": (0.48, 0.32, 0.24)}   # linear-light reflectance
REL = {"forehead": 0.8, "cheek_l": 1.2, "cheek_r": 1.0, "nose": 1.1}  # regional perfusion


def _encode(lin):
    lin = np.clip(lin, 0.0, 1.0)
    v = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.round(v * 255.0, 2)


def _pulse(t, hr_bpm, rng, hrv=0.03):
    """Unit-amplitude (peak-to-trough 1) PPG-like wave with beat-to-beat HRV."""
    rr = 60.0 / hr_bpm * (1.0 + hrv * rng.standard_normal(400))
    beats = np.cumsum(np.r_[0.0, rr])
    k = np.clip(np.searchsorted(beats, t, side="right") - 1, 0, beats.size - 2)
    ph = (t - beats[k]) / (beats[k + 1] - beats[k])
    w = np.exp(-((ph - 0.25) / 0.10) ** 2) + 0.3 * np.exp(-((ph - 0.6) / 0.12) ** 2)
    return w / (w.max() - w.min())


def make_doc(pi_pct=0.5, hr=72.0, seconds=50.0, fps=30.0, noise=0.0008, gain=1.0,
             gain_step=None, motion=None, seed=1, pulse_on=True, bbox=True):
    """noise: per-frame SD of each channel's relative fluctuation (linear)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 1.0 / fps)
    p = _pulse(t, hr, rng) if pulse_on else np.zeros_like(t)
    p = p - p.mean()
    g = np.full(t.size, gain)
    if gain_step is not None:                    # exposure jump mid-scan
        g[t >= gain_step[0]] *= gain_step[1]
    traces = {}
    for r in ROIS:
        ac = pi_pct / 100.0 * REL[r]
        rel = 1.0 + np.outer(ac * p, PBV) + noise * rng.standard_normal((t.size, 3))
        if motion is not None:                   # common-mode shading artefact
            rel = rel * (1.0 + motion(t))[:, None]
        lin = np.asarray(DC[r])[None, :] * g[:, None] * rel
        traces[r] = _encode(lin).tolist()
    doc = {"schema_version": 1, "t_s": t.round(4).tolist(), "traces": traces}
    if bbox:
        doc["bbox"] = [[0.35, 0.2, 0.3, 0.45]] * t.size
    return doc


def _expected_fundamental_pi(pi_pct):
    """2*A1 of the synthetic unit wave, x the regions' pairwise geometric scale
    (the estimator's index is 2*sqrt(median over region pairs of A_k*A_j))."""
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 50.0, 1 / 30.0)
    p = _pulse(t, 72.0, rng, hrv=0.0)
    f = np.fft.rfft(p - p.mean())
    k = int(np.argmax(np.abs(f[1:]))) + 1
    a1 = 2 * np.abs(f[k]) / p.size
    rel = list(REL.values())
    pairs = [rel[i] * rel[j] for i in range(len(rel)) for j in range(i + 1, len(rel))]
    return 2 * a1 * pi_pct * math.sqrt(float(np.median(pairs)))


def test_recovers_known_index_without_bias():
    """Averaged over scans the estimate sits on the truth (bias), and at
    typical facial levels (0.5-1 %) every single scan is within 8 % (measured
    per-scan spread ~3 % SD)."""
    for pi in (0.2, 0.5, 1.0):
        want = _expected_fundamental_pi(pi)
        got = [fpm.facial_perfusion(make_doc(pi_pct=pi, seed=s), hr_hint_bpm=72.0) for s in range(6)]
        assert all(g["available"] for g in got), [g.get("reason") for g in got]
        vals = [g["pi_percent"] for g in got]
        assert float(np.mean(vals)) == pytest.approx(want, rel=0.05), (pi, vals, want)
        if pi >= 0.5:
            assert all(v == pytest.approx(want, rel=0.08) for v in vals), (pi, vals, want)


def test_index_is_linear_light_so_brightness_does_not_move_it():
    """Same skin, same pulse, the camera 40 % darker: sRGB-encoded AC/DC would
    change with the operating point on the tone curve; linear AC/DC must not."""
    a = fpm.facial_perfusion(make_doc(gain=1.0), hr_hint_bpm=72.0)["pi_percent"]
    b = fpm.facial_perfusion(make_doc(gain=0.6), hr_hint_bpm=72.0)["pi_percent"]
    assert b == pytest.approx(a, rel=0.05)


def test_exposure_jump_mid_scan_does_not_move_it():
    a = fpm.facial_perfusion(make_doc(), hr_hint_bpm=72.0)["pi_percent"]
    b = fpm.facial_perfusion(make_doc(gain_step=(25.0, 0.7)), hr_hint_bpm=72.0)["pi_percent"]
    assert b == pytest.approx(a, rel=0.08)


def test_noise_only_is_not_a_perfusion_value():
    """A trace with no pulse must not produce an index built from noise: the
    region-coherence floor sits above every value pure noise produced."""
    for seed in range(10):
        for noise in (0.0008, 0.002, 0.005):
            fp = fpm.facial_perfusion(make_doc(pulse_on=False, noise=noise, seed=seed + 100),
                                      hr_hint_bpm=72.0)
            assert fp["available"] is False, (seed, noise, fp.get("per_region"))
            assert "coherence" in fp["reason"] or "heart rate" in fp["reason"]


def test_weak_pulse_is_not_read_low_and_marginal_values_are_provisional():
    """Detected pulses are unbiased (a phase-reference fit read 26-52 % of a
    known 0.2 % index at these noise levels; cross-region products with their
    noise floor removed read it within 12 %). A weaker pulse is either not
    measured or marked provisional, never reported as a confident low value."""
    want = _expected_fundamental_pi(0.2)
    got = [fpm.facial_perfusion(make_doc(pi_pct=0.2, noise=0.0012, seed=s), hr_hint_bpm=72.0)
           for s in range(4)]
    assert all(g["available"] for g in got)
    assert float(np.mean([g["pi_percent"] for g in got])) == pytest.approx(want, rel=0.12)
    for s in range(6):
        fp = fpm.facial_perfusion(make_doc(pi_pct=0.2, noise=0.0024, seed=s), hr_hint_bpm=72.0)
        if fp["available"]:
            assert fpm.vascular_tone_card(fp)["tier"] == "provisional"


def test_heart_rate_hint_error_does_not_move_it():
    doc = make_doc(hr=78.0)
    base = fpm.facial_perfusion(doc, hr_hint_bpm=78.0)["pi_percent"]
    for hint in (66.0, 90.0, None):
        assert fpm.facial_perfusion(doc, hr_hint_bpm=hint)["pi_percent"] == pytest.approx(base, rel=0.05)


def test_gross_motion_windows_are_dropped():
    """A 6 s burst of head motion at the pulse band (shading x10 the pulse):
    the median over low-motion windows keeps the value."""
    base = fpm.facial_perfusion(make_doc(), hr_hint_bpm=72.0)["pi_percent"]
    burst = lambda t: np.where((t > 5) & (t < 11), 0.03 * np.sin(2 * math.pi * 1.1 * t), 0.0)
    doc = make_doc(motion=burst)
    n = len(doc["t_s"])
    t = np.asarray(doc["t_s"])
    jitter = np.where((t > 5) & (t < 11), 0.02 * np.sin(2 * math.pi * 1.7 * t), 0.0)
    doc["bbox"] = [[0.35 + float(j), 0.2, 0.3, 0.45] for j in jitter]
    fp = fpm.facial_perfusion(doc, hr_hint_bpm=72.0)
    assert fp["windows"]["low_motion"] < fp["windows"]["total"]
    assert fp["pi_percent"] == pytest.approx(base, rel=0.15)
    assert n == len(doc["bbox"])


def test_missing_or_malformed_document_fails_closed_with_a_reason():
    for doc in (None, {}, {"t_s": [0, 1]}, {"t_s": "x", "traces": {}}):
        fp = fpm.facial_perfusion(doc)
        assert fp["available"] is False and fp["reason"]


def test_short_trace_is_not_measured():
    fp = fpm.facial_perfusion(make_doc(seconds=8.0), hr_hint_bpm=72.0)
    assert fp["available"] is False


def test_region_gaps_split_but_do_not_invent_signal():
    doc = make_doc(seconds=60.0)
    for r in ROIS:                                # face lost for 3 s mid-scan
        for i in range(900, 990):
            doc["traces"][r][i] = None
    fp = fpm.facial_perfusion(doc, hr_hint_bpm=72.0)
    assert fp["available"]
    assert fp["pi_percent"] == pytest.approx(_expected_fundamental_pi(0.5), rel=0.10)


def test_score_is_an_inverted_t_score_of_the_log_index():
    """50 at the published reference's geometric mean, 10 points per
    between-person SD of ln(PI), lower index -> higher score, clipped 0-100."""
    assert fpm.tone_score_from_pi(fpm.PI_REF_PERCENT) == 50.0
    one_sd_up = fpm.PI_REF_PERCENT * math.exp(fpm.PI_REF_LN_SD)
    assert fpm.tone_score_from_pi(one_sd_up) == pytest.approx(40.0, abs=0.1)
    assert fpm.tone_score_from_pi(fpm.PI_AT_SCORE_100) == pytest.approx(100.0, abs=0.1)
    assert fpm.tone_score_from_pi(fpm.PI_AT_SCORE_0) == pytest.approx(0.0, abs=0.1)
    assert fpm.tone_score_from_pi(0.01) == 100.0 and fpm.tone_score_from_pi(50.0) == 0.0
    s = [fpm.tone_score_from_pi(x) for x in (0.2, 0.3, 0.45, 0.7, 1.0)]
    assert s == sorted(s, reverse=True)
    # 0.55 +/- 0.16 % (the reference) lands inside the typical band
    for pi in (0.40, 0.55, 0.70):
        assert fpm.TONE_TYPICAL[0] - 0.5 <= fpm.tone_score_from_pi(pi) <= fpm.TONE_TYPICAL[1] + 0.5
    for bad in (None, 0.0, -1.0, float("nan"), "x"):
        assert fpm.tone_score_from_pi(bad) is None


def test_card_shape_and_labels():
    fp = fpm.facial_perfusion(make_doc(), hr_hint_bpm=72.0)
    card = fpm.vascular_tone_card(fp, legacy={"available": True, "amplitude_cv": 0.41,
                                              "score": 41.0, "n_beats": 9, "tier": "provisional"})
    assert card["available"] and card["tier"] == "measured"
    assert card["estimate"]["unit"] == "/100" and 0 <= card["score"] <= 100
    assert card["raw_name"] == "facial_perfusion_index_percent" and card["raw_unit"] == "%"
    assert card["score"] == fpm.tone_score_from_pi(fp["pi_percent"])
    assert 1 <= card["confidence"]["confidence_stars"] <= 5
    assert card["legacy_clip_cv"]["amplitude_cv"] == 0.41
    assert card["calibrated"] is False
    empty = fpm.vascular_tone_card({"available": False, "reason": "x"})
    assert empty["available"] is False and empty["reason"] == "x"


def test_deterministic():
    doc = make_doc()
    assert fpm.facial_perfusion(doc, hr_hint_bpm=72.0) == fpm.facial_perfusion(doc, hr_hint_bpm=72.0)


def test_colourless_shading_is_not_read_as_pulse():
    """Head motion or shading under side light changes every colour channel
    alike. The pulse is measured along the blood-volume colour signature with
    that direction removed: shading alone yields no value, and shading on top
    of a pulse leaves the index where it was. Green alone read shading 3x the
    pulse as ~9x the index."""
    shade = lambda t: 0.004 * np.sin(2 * math.pi * 1.2 * t + 0.3)      # colourless, at 72 bpm
    alone = fpm.facial_perfusion(make_doc(pulse_on=False, noise=0.0005, motion=shade), hr_hint_bpm=72.0)
    assert alone["available"] is False
    clean = fpm.facial_perfusion(make_doc(), hr_hint_bpm=72.0)
    shaded = fpm.facial_perfusion(make_doc(motion=shade), hr_hint_bpm=72.0)
    assert shaded["pi_percent"] == pytest.approx(clean["pi_percent"], rel=0.05)
    assert shaded["colour_signature"]["green_only_pi_percent"] > 2 * clean["pi_percent"]
    sig = clean["colour_signature"]
    assert sig["looks_like"] == "blood_volume"
    assert sig["r_over_g"] == pytest.approx(PBV[0], abs=0.08)
    assert sig["b_over_g"] == pytest.approx(PBV[2], abs=0.08)
