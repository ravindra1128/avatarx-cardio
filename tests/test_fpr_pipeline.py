"""Facial Perfusion Response (FPR) pilot harness — B7 correctness, the PI
processing math, the DC-preserving extraction, the immutable-raw contract,
and the module's isolation from production.

These are the tests that CAN pass without hardware. The hardware stages
(B0-B6, B8) are exercised here only through synthetic stand-in recordings to
prove the ANALYZERS are correct; a real PASS still needs real rig data.
"""
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from research.facial_perfusion_pilot import (
    synth, pi as P, extract, qualify, reference as ref, WATERMARK, FprError,
    SIGMA_NULL_PASS, QUANT_HEADROOM_MIN, ROI_MIN_PX)
from research.facial_perfusion_pilot.stages import (
    run_B7, run_B4, run_B6, quantisation_headroom)
from research.facial_perfusion_pilot.contract import new_session, FprSession

_PKG = pathlib.Path(__file__).resolve().parents[1] / \
    "research" / "facial_perfusion_pilot"


# --------------------------------------------------------------------------
# Isolation from production (the research quarantine, at the FPR level)
# --------------------------------------------------------------------------
def test_fpr_never_imports_app_or_inference():
    offenders = []
    for py in _PKG.glob("*.py"):
        src = py.read_text()
        for bad in ("import app", "from app", "import inference",
                    "from inference"):
            if bad in src:
                offenders.append((py.name, bad))
    assert not offenders, f"FPR must not import production code: {offenders}"


def test_fpr_does_not_use_pos_for_the_signal():
    # POS discards the DC that PI needs; the harness must not extract with it.
    for py in _PKG.glob("*.py"):
        assert "pos_pulse" not in py.read_text(), \
            f"{py.name} uses POS, which projects out the DC term"


# --------------------------------------------------------------------------
# B0 quantisation-headroom math
# --------------------------------------------------------------------------
def test_b0_headroom_8bit_150px_passes():
    m = quantisation_headroom(8, ROI_MIN_PX * ROI_MIN_PX, 0.001)
    assert m["headroom_x"] > QUANT_HEADROOM_MIN
    assert 35 < m["headroom_x"] < 42            # ~38x for 8-bit / 150^2 / 0.1%


def test_b0_headroom_grows_with_pixels_and_bits():
    small = quantisation_headroom(8, 40 * 40, 0.001)["headroom_x"]
    big = quantisation_headroom(8, 300 * 300, 0.001)["headroom_x"]
    deeper = quantisation_headroom(10, 40 * 40, 0.001)["headroom_x"]
    assert big > small
    assert deeper > small


# --------------------------------------------------------------------------
# PI processing math
# --------------------------------------------------------------------------
@pytest.mark.parametrize("m", [0.002, 0.005, 0.01])
def test_pi_recovers_2m(m):
    tr = synth.synth_green_trace(fps=30.0, duration_s=40.0, hr_hz=1.2,
                                 modulation=m, dc=140.0)
    r = P.process_pi(tr.green, tr.t, tr.fps)
    assert r.baseline_pi == pytest.approx(synth.expected_pi(m), rel=0.03)


def test_pi_is_monotonic_in_modulation():
    def pi_of(m):
        tr = synth.synth_green_trace(modulation=m, duration_s=40.0)
        return P.process_pi(tr.green, tr.t, tr.fps).baseline_pi
    assert pi_of(0.002) < pi_of(0.005) < pi_of(0.01)


def test_clean_sinusoid_sigma_null_is_small():
    tr = synth.synth_green_trace(modulation=0.005, duration_s=60.0)
    r = P.process_pi(tr.green, tr.t, tr.fps)
    assert r.sigma_null_pct < SIGMA_NULL_PASS
    assert r.beat_yield > 0.8
    assert r.reject_fraction < 0.05


def test_delta_pi_step_recovery():
    tr, b_end, exp = synth.synth_step_trace(
        m_baseline=0.005, m_response=0.007)
    r = P.process_pi(tr.green, tr.t, tr.fps, baseline_end_s=b_end)
    from research.facial_perfusion_pilot import PI_WINDOW_S
    mask = (r.win_centers_s >= b_end + PI_WINDOW_S) & \
        np.isfinite(r.delta_pi_pct)
    rec = float(np.median(r.delta_pi_pct[mask]))
    assert rec == pytest.approx(exp, rel=0.05)


# --------------------------------------------------------------------------
# Regression: local rejection must PRESERVE a real change but CATCH a spike
# --------------------------------------------------------------------------
def test_local_rejection_preserves_a_real_step():
    # A genuine within-subject step is the effect we want to detect; a global
    # outlier filter would erase it. Reject fraction must stay low.
    tr, b_end, _ = synth.synth_step_trace(m_baseline=0.005, m_response=0.009)
    r = P.process_pi(tr.green, tr.t, tr.fps, baseline_end_s=b_end)
    assert r.reject_fraction < 0.10
    # both regimes retain beats
    acc = [c for c in r.cycles if c.accepted]
    assert sum(c.t_peak < b_end for c in acc) > 20
    assert sum(c.t_peak > b_end for c in acc) > 20


def test_local_rejection_catches_a_lone_artefact_beat():
    tr = synth.synth_green_trace(modulation=0.005, duration_s=40.0, dc=140.0)
    g = tr.green.copy()
    fps = tr.fps
    i0 = int(20 * fps)                               # corrupt one beat at ~20 s
    i1 = i0 + int(0.8 * fps)
    g[i0:i1] += 8.0 * 0.005 * 140.0 * np.hanning(i1 - i0)  # big amplitude spike
    r = P.process_pi(g, tr.t, fps)
    reasons = [c.reject_reason for c in r.cycles if not c.accepted]
    assert any(reason == "amplitude outlier" for reason in reasons)
    assert r.reject_fraction < 0.30                  # only the artefact, not all


# --------------------------------------------------------------------------
# B7 — the software-only gate
# --------------------------------------------------------------------------
def test_b7_passes():
    r = run_B7()
    assert r.status == "PASS", r.reason
    assert r.metrics["pi_rel_error"] < 0.05
    assert r.metrics["delta_rel_error"] < 0.10


# --------------------------------------------------------------------------
# DC-preserving extraction end to end (through 8-bit lossless video)
# --------------------------------------------------------------------------
def test_extraction_preserves_dc_and_recovers_pi(tmp_path):
    vp = str(tmp_path / "synth.avi")
    synth.write_synth_video(vp, fps=30.0, duration_s=20.0, hr_hz=1.2,
                            modulation=0.02, dc_luma=140.0, size=200,
                            noise_lsb=1.0)
    ex = extract.extract_roi_trace(vp, (10, 10, 190, 190))
    # DC is preserved (POS would have removed it): mean green ~ dc_luma
    assert ex.channel_means()["green"] == pytest.approx(140.0, abs=2.0)
    r = P.process_pi(ex.green, ex.t, ex.fps)
    assert r.baseline_pi == pytest.approx(synth.expected_pi(0.02), rel=0.05)


def test_extract_enforces_roi_floor(tmp_path):
    vp = str(tmp_path / "small.avi")
    synth.write_synth_video(vp, duration_s=3.0, size=120)
    with pytest.raises(FprError):
        extract.extract_roi_trace(vp, (0, 0, 100, 100))     # < 150 px


# --------------------------------------------------------------------------
# Immutable-raw data contract
# --------------------------------------------------------------------------
def test_raw_is_write_once(tmp_path):
    s = new_session(str(tmp_path / "sess"))
    s.write_bytes("raw/session.avi", b"first")
    with pytest.raises(FprError):
        s.write_bytes("raw/session.avi", b"second")     # immutable
    # derived is rewritable
    s.save_json("derived/pi.json", {"a": 1})
    s.save_json("derived/pi.json", {"a": 2})
    assert s.load_json("derived/pi.json")["a"] == 2


# --------------------------------------------------------------------------
# Stage analyzers via synthetic stand-in recordings
# --------------------------------------------------------------------------
def test_b4_flags_nonuniform_flat_field(tmp_path):
    s = new_session(str(tmp_path / "sess"))
    s.save_json("roi/roi.json", {"roi": [10, 10, 190, 190]})
    synth.write_flat_field_video(s.path("calib/flat.avi"), duration_s=2.0,
                                 size=200, gradient=0.25, noise_lsb=1.0)
    assert run_B4(s).status == "FAIL"


def test_b6_phantom_flat_passes_drift_fails(tmp_path):
    flat = new_session(str(tmp_path / "flat"))
    flat.save_json("roi/roi.json", {"roi": [10, 10, 190, 190]})
    synth.write_phantom_video(flat.path("controls/phantom.avi"),
                              duration_s=20.0, size=200, dc_drift_frac=0.0)
    r_flat = run_B6(flat)
    assert r_flat.status == "PASS", r_flat.reason

    drift = new_session(str(tmp_path / "drift"))
    drift.save_json("roi/roi.json", {"roi": [10, 10, 190, 190]})
    synth.write_phantom_video(drift.path("controls/phantom.avi"),
                              duration_s=20.0, size=200, dc_drift_frac=0.02)
    r_drift = run_B6(drift)
    assert r_drift.status == "FAIL"
    assert r_drift.hard_stop is True


# --------------------------------------------------------------------------
# Full sequence on an empty session -> INCONCLUSIVE, B7 PASS
# --------------------------------------------------------------------------
def test_empty_session_is_inconclusive_with_b7_pass(tmp_path):
    s = new_session(str(tmp_path / "empty"))
    q = qualify.qualify_session(s, save=True)
    assert q.verdict == "INCONCLUSIVE"
    by_stage = {st["stage"]: st["status"] for st in q.stages}
    assert by_stage["B7"] == "PASS"
    assert by_stage["B8"] == "INCONCLUSIVE"
    assert "B8" in q.missing_inputs and "session.avi" in q.missing_inputs["B8"]
    # the verdict artefact carries the non-measurement watermark
    assert s.load_json("report/qualification.json")["watermark"] == WATERMARK


# --------------------------------------------------------------------------
# Finger contact-PPG reference (the stimulus positive control)
# --------------------------------------------------------------------------
def _finger_waveform(m_base, m_resp):
    tr, b_end, _ = synth.synth_step_trace(
        fps=100.0, baseline_s=60.0, response_s=60.0, hr_hz=1.2,
        m_baseline=m_base, m_response=m_resp, dc=180.0)
    return tr


def test_read_finger_ppg_ok_and_fail_closed():
    t, ppg = ref.read_finger_ppg("t_s,ppg\n0.0,1.0\n0.01,1.2\n0.02,0.9\n")
    assert t.size == 3 and ppg.size == 3
    with pytest.raises(FprError):
        ref.read_finger_ppg("weird,cols\n1,2\n")        # unknown header
    with pytest.raises(FprError):
        ref.read_finger_ppg("t_s,ppg\n0,1\n0,1\n")      # time not increasing


def test_finger_pi_export_reuses_repo_parser():
    csv = "t_s,pi_percent\n" + "\n".join(f"{i},{2.0}" for i in range(0, 40, 2))
    pt, pv = ref.read_finger_pi_export(csv)
    assert pt.size == pv.size >= 2
    f = ref.analyze_finger(pi_t=pt, pi_pct=pv)
    assert f.source == "pi_export"


def test_finger_waveform_yields_hr_and_pi():
    tr = _finger_waveform(0.08, 0.08)                   # steady
    f = ref.analyze_finger(ppg=tr.green, t=tr.t)
    assert f.source == "waveform"
    assert f.hr_bpm == pytest.approx(72.0, abs=3.0)     # 1.2 Hz
    assert f.snr_db > 6.0
    assert np.nanmedian(f.pi_pct) == pytest.approx(16.0, rel=0.1)  # 2*0.08


def test_stimulus_response_detects_vasoconstriction():
    tr = _finger_waveform(0.08, 0.04)                   # perfusion halves
    f = ref.analyze_finger(ppg=tr.green, t=tr.t)
    v = ref.stimulus_response(f, (0, 60), (60, 120))
    assert v.verdict == "responded"
    assert v.direction == "drop"
    assert v.delta_pct == pytest.approx(-50.0, abs=5.0)


def test_stimulus_no_response_flags_problem():
    tr = _finger_waveform(0.08, 0.08)                   # no change
    f = ref.analyze_finger(ppg=tr.green, t=tr.t)
    v = ref.stimulus_response(f, (0, 60), (60, 120))
    assert v.verdict == "no_response"


def test_attribution_uses_finger_as_control():
    responded = ref.StimulusVerdict("responded", -50.0, 16.0, 8.0, "drop", 10.0)
    flat = ref.StimulusVerdict("no_response", 0.0, 16.0, 16.0, "rise", 10.0)
    assert ref.attribute(-3.0, responded)["category"] == \
        ref.FACIAL_SIGNAL_QUESTION
    assert ref.attribute(-40.0, responded)["category"] == ref.BOTH_RESPONDED
    assert ref.attribute(-40.0, flat)["category"] == ref.STIMULUS_PROBLEM


def test_sham_check():
    assert ref.sham_check(-30.0, -25.0)["category"] == ref.EXPECTANCY_OR_ARTIFACT
    assert ref.sham_check(-30.0, -3.0)["category"] == ref.REAL_EXCEEDS_SHAM


def test_events_to_windows():
    ev = ref.read_events("t_s,event\n60,immersion_in\n120,immersion_out\n")
    w = ref.windows_from_events(ev, total_s=180.0)
    assert w["baseline"] == (0.0, 60.0)
    assert w["stimulus"] == (60.0, 120.0)
    with pytest.raises(FprError):
        ref.windows_from_events([(1.0, "movement")])    # no immersion markers


def test_session_loads_finger_and_events(tmp_path):
    s = new_session(str(tmp_path / "sess"))
    tr = _finger_waveform(0.08, 0.04)
    rows = "t_s,ppg\n" + "\n".join(f"{ti:.5f},{v:.5f}"
                                   for ti, v in zip(tr.t, tr.green))
    s.write_bytes("controls/finger_ppg.csv", rows.encode())
    s.write_bytes("controls/events.csv",
                  b"t_s,event\n60,immersion_in\n120,immersion_out\n")
    finger = s.load_finger()
    events = s.load_events()
    w = ref.windows_from_events(events, total_s=float(tr.t[-1]))
    v = ref.stimulus_response(finger, w["baseline"], w["stimulus"])
    assert v.verdict == "responded" and v.direction == "drop"


# --------------------------------------------------------------------------
# Continuous reference channels: HR, temperature, and the thermal confound
# --------------------------------------------------------------------------
def test_read_channels_hr_and_temperature():
    t, ch = ref.read_channels(
        "t_s,forehead_c,hand_c,water_c\n0,34,33,4\n1,34.1,32.5,4\n")
    assert list(ch) == ["forehead_c", "hand_c", "water_c"] and t.size == 2
    ht, hv = ref.read_hr("t_s,hr_bpm\n0,72\n1,74\n2,76\n")
    assert hv.tolist() == [72.0, 74.0, 76.0]
    with pytest.raises(FprError):
        ref.read_channels("nope,x\n1,2\n")               # no time header
    with pytest.raises(FprError):
        ref.read_channels("t_s,a,b\n0,1\n")              # ragged row


def _rising_pi_face(m0=0.004, m1=0.009, dur=120.0, fps=30.0):
    n = int(dur * fps)
    env = np.linspace(m0, m1, n)
    tr = synth.synth_green_trace(fps=fps, duration_s=dur, hr_hz=1.2,
                                 modulation=env, dc=140.0)
    return P.process_pi(tr.green, tr.t, tr.fps)


def test_thermal_confound_flags_when_pi_tracks_temperature():
    face = _rising_pi_face()                              # PI rises over time
    temp_t = np.linspace(0, 120, 60)
    temp_rising = 33.0 + 2.0 * (temp_t / 120.0)          # temp rises too
    c = ref.thermal_confound_check(face, temp_t, temp_rising)
    assert c["flagged"] and c["category"] == ref.THERMAL_CONFOUND
    assert abs(c["r"]) >= 0.7


def test_thermal_confound_not_flagged_when_pi_is_flat():
    flat = synth.synth_green_trace(modulation=0.005, duration_s=120.0)
    face = P.process_pi(flat.green, flat.t, flat.fps)    # PI ~ constant
    temp_t = np.linspace(0, 120, 60)
    temp_rising = 33.0 + 2.0 * (temp_t / 120.0)
    c = ref.thermal_confound_check(face, temp_t, temp_rising)
    assert not c["flagged"] and c["category"] == ref.NO_THERMAL_CONFOUND


def test_reference_overlay_renders_all_channels_from_session(tmp_path):
    from research.facial_perfusion_pilot import diagnostics
    s = new_session(str(tmp_path / "sess"))
    s.save_json("roi/roi.json", {"roi": [10, 10, 190, 190]})
    synth.write_synth_video(s.path("raw/session.avi"), fps=30.0,
                            duration_s=20.0, hr_hz=1.2, modulation=0.01,
                            dc_luma=140.0, size=200)
    ft = _finger_waveform(0.08, 0.04)
    s.write_bytes("controls/finger_ppg.csv",
                  ("t_s,ppg\n" + "\n".join(f"{a:.4f},{b:.4f}"
                   for a, b in zip(ft.t, ft.green))).encode())
    s.write_bytes("controls/heart_rate.csv",
                  ("t_s,hr_bpm\n" + "\n".join(f"{i},{72.0}"
                   for i in range(0, 20))).encode())
    s.write_bytes("controls/temperature.csv",
                  ("t_s,forehead_c,hand_c,water_c\n" + "\n".join(
                   f"{i},34.0,33.0,4.0" for i in range(0, 20))).encode())
    s.write_bytes("controls/events.csv",
                  b"t_s,event\n10,immersion_in\n15,immersion_out\n")
    out = str(tmp_path / "overlay.png")
    p = diagnostics.render_reference_overlay(s, out)
    assert os.path.exists(p) and os.path.getsize(p) > 1000


# --------------------------------------------------------------------------
# Sham-vs-real comparison (the expectancy / apparatus / artifact control)
# --------------------------------------------------------------------------
from research.facial_perfusion_pilot import sham_real as SR   # noqa: E402

_WIN = {"baseline": (0.0, 40.0), "stimulus": (40.0, 70.0),
        "recovery": (70.0, 90.0)}


def _face_trial(m_base, m_resp, in_s=40, out_s=70, dur=90, fps=30):
    n = int(dur * fps)
    env = np.full(n, m_base)
    env[int(in_s * fps):int(out_s * fps)] = m_resp
    tr = synth.synth_green_trace(fps=fps, duration_s=dur, hr_hz=1.2,
                                 modulation=env, dc=140.0)
    return P.process_pi(tr.green, tr.t, tr.fps)


def _finger_trial(step, in_s=40, dur=90):
    if step:
        ft, _, _ = synth.synth_step_trace(fps=50.0, baseline_s=in_s,
            response_s=dur - in_s, hr_hz=1.2, m_baseline=0.08,
            m_response=0.04, dc=180.0)
    else:
        ft = synth.synth_green_trace(fps=50.0, duration_s=dur, hr_hz=1.2,
                                     modulation=0.08, dc=180.0)
    return ref.analyze_finger(ppg=ft.green, t=ft.t)


def _trial(cond, m_base, m_resp, finger_step):
    return SR.analyze_trial(_face_trial(m_base, m_resp), _WIN, cond,
                            finger=_finger_trial(finger_step))


def test_trial_windows_named_and_fallback():
    ev = [(0, "null_in"), (60, "sham_in"), (150, "sham_out"),
          (300, "real_in"), (390, "real_out")]
    w = ref.trial_windows(ev, "real", baseline_s=120)
    assert w["stimulus"] == (300, 390) and w["baseline"] == (180, 300)
    single = ref.trial_windows([(60, "immersion_in"), (150, "immersion_out")],
                               "real")
    assert single["stimulus"] == (60, 150)           # fallback


def test_compare_real_exceeds_sham():
    real = _trial("real", 0.006, 0.012, finger_step=True)
    sham = _trial("sham", 0.006, 0.006, finger_step=False)
    cmp = SR.compare_sham_real(real, sham)
    assert cmp.category == ref.REAL_EXCEEDS_SHAM
    assert real.face_delta_pct > 50 and abs(sham.face_delta_pct) < 15
    assert real.finger.verdict == "responded" and real.finger.direction == "drop"
    assert sham.finger.verdict == "no_response"
    assert cmp.caveats == []


def test_compare_flags_expectancy_when_sham_matches_real():
    real = _trial("real", 0.006, 0.012, finger_step=True)
    sham = _trial("sham", 0.006, 0.012, finger_step=False)   # sham face ALSO steps
    cmp = SR.compare_sham_real(real, sham)
    assert cmp.category == ref.EXPECTANCY_OR_ARTIFACT


def test_compare_caveats_when_sham_finger_also_responds():
    real = _trial("real", 0.006, 0.012, finger_step=True)
    sham = _trial("sham", 0.006, 0.006, finger_step=True)    # sham finger DROPS
    cmp = SR.compare_sham_real(real, sham)
    assert cmp.category == ref.REAL_EXCEEDS_SHAM
    assert any("SHAM finger" in c for c in cmp.caveats)


def test_comparison_is_json_serializable():
    cmp = SR.compare_sham_real(_trial("real", 0.006, 0.012, True),
                               _trial("sham", 0.006, 0.006, False))
    json.dumps(cmp.to_dict())                         # must not raise


def test_sham_real_overlay_renders():
    from research.facial_perfusion_pilot import diagnostics
    cmp = SR.compare_sham_real(_trial("real", 0.006, 0.012, True),
                               _trial("sham", 0.006, 0.006, False))
    import tempfile
    out = os.path.join(tempfile.mkdtemp(), "sr.png")
    diagnostics.sham_real_overlay(cmp, out)
    assert os.path.exists(out) and os.path.getsize(out) > 1000


def _sham_real_session(tmp_path, name, face_menv, finger_step, in_s=25,
                       out_s=40, dur=60):
    s = new_session(str(tmp_path / name))
    s.save_json("roi/roi.json", {"roi": [10, 10, 190, 190]})
    synth.write_synth_video(s.path("raw/session.avi"), fps=30.0, duration_s=dur,
                            hr_hz=1.2, modulation=face_menv, dc_luma=140.0,
                            size=200)
    if finger_step:
        ft, _, _ = synth.synth_step_trace(fps=50.0, baseline_s=in_s,
            response_s=dur - in_s, hr_hz=1.2, m_baseline=0.08,
            m_response=0.04, dc=180.0)
    else:
        ft = synth.synth_green_trace(fps=50.0, duration_s=dur, hr_hz=1.2,
                                     modulation=0.08, dc=180.0)
    s.write_bytes("controls/finger_ppg.csv",
                  ("t_s,ppg\n" + "\n".join(f"{a:.4f},{b:.4f}"
                   for a, b in zip(ft.t, ft.green))).encode())
    s.write_bytes("controls/events.csv",
                  f"t_s,event\n{in_s},immersion_in\n{out_s},immersion_out\n"
                  .encode())
    return s


def test_sham_real_end_to_end_from_two_sessions(tmp_path):
    n = int(60 * 30)
    real_env = np.full(n, 0.008)
    real_env[int(25 * 30):int(40 * 30)] = 0.016         # +100% facial dPI
    real = _sham_real_session(tmp_path, "real", real_env, finger_step=True)
    sham = _sham_real_session(tmp_path, "sham", np.full(n, 0.008),
                              finger_step=False)
    cmp = SR.compare_sessions(real, sham)
    assert cmp.category == ref.REAL_EXCEEDS_SHAM
    assert cmp.real.face_delta_pct > 40
    assert cmp.real.finger.verdict == "responded"
