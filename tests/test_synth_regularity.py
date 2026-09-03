"""v0.7 fixtures — every cohort carries the physiology it claims, with
the truth written beside the clip."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

from scripts.make_synth_regularity import (COHORTS, LADDER_RMSSD_MS,
                                           bigeminy_rr, pac_isolated_rr,
                                           regular_rr, rsa_rr,
                                           trigeminy_rr)


def _rmssd(rr):
    return float(np.sqrt(np.mean(np.diff(rr) ** 2))) * 1000.0


def test_regular_rr_hits_the_requested_true_rmssd():
    for want in (0.0, 8.0, 16.0, 32.0):
        rr = regular_rr(300.0, rmssd_ms=want, seed=1)
        assert _rmssd(rr) == pytest.approx(want, abs=max(1.5, 0.12 * want))
        assert 60.0 / np.median(rr) == pytest.approx(70.0, abs=1.5)


def test_ectopy_patterns_have_their_signature():
    bg = bigeminy_rr(60.0, seed=2)
    # short/long alternation: successive intervals differ by ~76% of RR
    d = np.diff(bg)
    assert np.mean(np.sign(d[:-1]) != np.sign(d[1:])) > 0.95
    # coupled beat + compensatory pause sum to ~2 RR
    assert np.median(bg[:-1:2] + bg[1::2]) == pytest.approx(
        2 * 60.0 / 72.0, abs=0.03)
    tg = trigeminy_rr(60.0, seed=2)
    x = tg - tg.mean()
    lag3 = float(np.dot(x[:-3], x[3:]) / np.dot(x, x))
    lag2 = float(np.dot(x[:-2], x[2:]) / np.dot(x, x))
    assert lag3 > 0.8 and lag3 > lag2
    pac = pac_isolated_rr(120.0, seed=2)
    # sparse: most intervals are ordinary sinus
    med = np.median(pac)
    assert np.mean(np.abs(pac - med) / med > 0.15) < 0.35


def test_rsa_is_modulated_at_the_breathing_rate():
    rr = rsa_rr(120.0, depth=0.16, resp_brpm=15.0, noise_ms=1.0, seed=3)
    t = np.cumsum(rr)
    grid = np.arange(t[0], t[-1], 0.25)
    x = np.interp(grid, t, rr) - np.mean(rr)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    fr = np.fft.rfftfreq(x.size, 0.25)
    assert float(fr[np.argmax(spec)]) == pytest.approx(0.25, abs=0.03)


def test_every_cohort_declares_class_and_explanation():
    for name, c in COHORTS.items():
        assert c["expect"] in ("regular", "irregular"), name
        assert c["benign"] in (None, "respiration_coupled", "ectopy_pattern",
                               "chaotic"), name
        lo, hi = c["age"]
        assert 18 <= lo < hi <= 90, name
        rr = c["build"](40.0, 5, 15.0)
        assert rr.size > 8 and np.all(rr > 0.25) and np.all(rr < 2.2), name
    # the young cohort is the RSA cohort, by design
    assert COHORTS["rsa_young"]["age"][1] < 35
    assert COHORTS["rsa_young"]["benign"] == "respiration_coupled"


def test_ladder_rungs_are_pre_registered_and_start_at_zero():
    assert LADDER_RMSSD_MS[0] == 0.0
    assert list(LADDER_RMSSD_MS) == sorted(LADDER_RMSSD_MS)


def test_write_scan_writes_every_sidecar_with_the_truth(tmp_path):
    pytest.importorskip("cv2")
    from scripts.make_synth_regularity import write_scan
    row = write_scan(tmp_path, "r_a", "rsa_young", pid="pa",
                     session_id="pa-s1", seed=4, duration_s=12.0,
                     fitzpatrick=5)
    for suffix in (".avi", ".ecg.json", ".recording.json",
                   ".participant.json", ".avi.truth.json"):
        assert (tmp_path / f"r_a{suffix}").exists(), suffix
    part = json.loads((tmp_path / "r_a.participant.json").read_text())
    assert part["fitzpatrick_group"] == 5 and 20 <= part["age_years"] < 35
    man = json.loads((tmp_path / "r_a.recording.json").read_text())
    assert man["rhythm_annotations"][0]["rhythm"] == \
        "RESPIRATORY_SINUS_ARRHYTHMIA"
    assert man["ecg_rpeaks_s"]
    assert row["true_rmssd_ms"] > 50.0 and row["cohort"] == "rsa_young"
    lad = write_scan(tmp_path, "r_l", None, pid="pl", session_id="pl-s1",
                     seed=4, duration_s=12.0, rmssd_ms=16.0)
    assert lad["cohort"] == "ladder_16ms"
    assert lad["true_rmssd_ms"] == pytest.approx(16.0, abs=6.0)


def test_skin_tone_label_has_an_optical_counterpart(tmp_path):
    """Review finding: the Fitzpatrick label was a JSON field over
    pixel-identical skin. Now the clip's skin base colour follows the
    group, and the default group reproduces the historical generator."""
    pytest.importorskip("cv2")
    import cv2
    from scripts.make_synth_regularity import (FITZPATRICK_SKIN_RGB,
                                               write_scan)
    from scripts.make_synth_video import SKIN_RGB
    assert tuple(FITZPATRICK_SKIN_RGB[3]) == tuple(float(v) for v in SKIN_RGB)
    means = {}
    for fitz in (1, 6):
        write_scan(tmp_path, f"s{fitz}", "regular", pid=f"p{fitz}",
                   session_id=f"p{fitz}-s1", seed=4, duration_s=4.0,
                   fitzpatrick=fitz)
        truth = json.loads((tmp_path / f"s{fitz}.avi.truth.json").read_text())
        assert truth["skin_rgb"] == list(FITZPATRICK_SKIN_RGB[fitz])
        cap = cv2.VideoCapture(str(tmp_path / f"s{fitz}.avi"))
        ok, frame = cap.read()
        cap.release()
        assert ok
        h, w = frame.shape[:2]
        means[fitz] = frame[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5].mean()
    assert means[1] > means[6] + 60


def test_cohort_axes_are_not_confounded(tmp_path):
    """Review finding: with fps = n % 2 and group = n % 6, odd groups were
    always 30 fps. Every group must see both frame rates."""
    from scripts.make_synth_regularity import make_regularity_dataset
    pytest.importorskip("cv2")
    m = make_regularity_dataset(tmp_path, duration_s=2.0, per_cohort=2,
                                cohorts=["regular", "af", "bigeminy",
                                         "trigeminy", "pac_isolated",
                                         "rsa_mid"])
    by = {}
    for row in m["scans"]:
        by.setdefault(row["fitzpatrick_group"], set()).add(row["fps"])
    assert all(len(v) == 2 for v in by.values()), by
