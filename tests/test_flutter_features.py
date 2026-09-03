"""v0.6 flutter T1 — the three feature families, their refusals, and the
miniature experiment that decides whether the track has anything: does
RSA coupling survive the real video path and separate metronomic 150
from sinus tachycardia at 150, where dispersion cannot?"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from beats.lattice import BeatLattice, LATTICE_VERSION
from features.flutter import (age_band, DEFAULT_RMSSD_FLOOR_MS,
                              flutter_features, hyper_regularity,
                              latent_atrial_fit, rate_fingerprint,
                              respiratory_coupling, tachogram,
                              variable_block_features)


def _lat(rr_s, *, t0=0.0, runs=None, fps=240.0):
    """A lattice from an RR series (seconds), with interval times.

    fps defaults HIGH so unit tests exercise the PHYSIOLOGICAL floor;
    the camera's quantization floor dominates at consumer frame rates
    and is pinned separately.
    """
    rr = np.asarray(rr_s, float)
    t = t0 + np.cumsum(rr)
    ibi = rr * 1000.0
    if runs is None:
        runs, times = [ibi], [t]
    else:
        runs_, times = [], []
        i = 0
        for k in runs:
            runs_.append(ibi[i:i + k])
            times.append(t[i:i + k])
            i += k
        runs = runs_
    return BeatLattice(
        version=LATTICE_VERSION, fps=fps,
        duration_s=float(t[-1] + 1.0), beat_t_s=np.concatenate(
            [np.array([t0]), t]),
        beat_confidence=np.ones(t.size + 1),
        beat_agreement=np.ones(t.size + 1), runs=runs,
        run_confidences=[np.ones(r.size) for r in runs],
        n_intervals=int(sum(r.size for r in runs)), dropout_rate=0.0,
        split_fraction=0.0, per_roi_times={}, segments=[],
        run_times=times)


def _flutter(n=100, base=0.400, jitter=0.003, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(base + rng.normal(0, jitter, n), 0.25, 1.5)


def _sinus(bpm=150.0, depth=0.10, brpm=15.0, dur=60.0, noise=0.006,
           seed=0):
    rng = np.random.default_rng(seed)
    f, base, out, t = brpm / 60.0, 60.0 / bpm, [], 0.0
    while t <= dur:
        out.append(max(base * (1 + depth * np.sin(2 * np.pi * f * t))
                       + rng.normal(0, noise), 0.25))
        t += out[-1]
    return np.asarray(out)


def _resp(brpm=15.0, dur=60.0, fs=30.0, quality=0.9, phase=0.0):
    t = np.arange(0, dur, 1.0 / fs)
    return {"t": t, "y": np.sin(2 * np.pi * (brpm / 60.0) * t + phase),
            "rate_brpm": brpm, "quality": quality}


# ------------------------------------------------- family 1: rate
def test_rate_fingerprint_finds_the_conduction_band():
    r = rate_fingerprint(_lat(_flutter()))
    assert r["median_bpm"] == pytest.approx(150.0, abs=1.5)
    assert r["band"] == "2:1" and r["in_band"] is True
    assert r["band_fraction"] > 0.9
    lo, hi = r["bpm_ci95"]
    assert lo <= r["median_bpm"] <= hi and (hi - lo) < 5.0
    assert r["sustained_s"] > 30.0
    # 3:1 and 4:1 are computed too — they are where ordinary rhythms
    # live, which is the point: an ordinary 67 bpm resting pulse sits
    # INSIDE the 4:1 band (4:1 of 250-330 is 63-83), and only the fact
    # that the flag fires on 2:1 alone keeps that from mattering
    assert rate_fingerprint(_lat(_flutter(base=0.600)))["band"] == "3:1"
    resting = rate_fingerprint(_lat(_flutter(base=0.900)))
    assert resting["band"] == "4:1" and resting["in_band"] is True
    # a rate BETWEEN the bands belongs to none of them
    between = rate_fingerprint(_lat(_flutter(base=0.520)))   # ~115 bpm
    assert between["in_band"] is False


def test_rate_fingerprint_refuses_below_the_interval_floor():
    r = rate_fingerprint(_lat(_flutter(n=8)))
    assert r["median_bpm"] is None and r["in_band"] is False
    assert "clean intervals" in r["reason"]


def test_an_unsustained_rate_is_not_in_band():
    """Half the scan at 150 and half at 75 is not a sustained 150."""
    rr = np.concatenate([_flutter(n=50), _flutter(n=25, base=0.800)])
    r = rate_fingerprint(_lat(rr))
    assert r["band_fraction"] < 0.7
    assert r["in_band"] is False


# ------------------------------------------ family 2: hyper-regularity
def test_age_bands_and_unknown_age_takes_the_strictest_floor():
    assert age_band(30) == "<40" and age_band(45) == "40-59"
    assert age_band(70) == "60-74" and age_band(80) == ">=75"
    for bad in (None, "", "old", float("nan")):
        assert age_band(bad) == "unknown"
    floors = DEFAULT_RMSSD_FLOOR_MS
    assert floors["unknown"] == min(floors.values())
    # a low-variability older sinus rhythm must not trip the floor that
    # a 30-year-old would (RMSSD ~12 ms: under 18, over 10)
    lat = _lat(_flutter(jitter=0.0085, seed=1))
    assert 10.0 < hyper_regularity(lat, age_years=30)["rmssd_ms"] < 18.0
    assert hyper_regularity(lat, age_years=30)["below_floor"] is True
    assert hyper_regularity(lat, age_years=80)["below_floor"] is False


def test_the_binding_floor_is_the_camera_not_the_patient():
    """Measured: a metronomic source reads RMSSD 15.0 ms at 30 fps and
    8.2 ms at 60 fps — 0.45 of a frame period, i.e. quantization. So at
    consumer frame rates the physiological floor is masked, and the
    feature must SAY so rather than quietly report a heart finding."""
    from features.flutter import measurement_floor_ms
    assert measurement_floor_ms(30.0) == pytest.approx(25.0, abs=0.1)
    assert measurement_floor_ms(60.0) == pytest.approx(12.5, abs=0.1)
    assert measurement_floor_ms(30.0) > 2 * measurement_floor_ms(60.0) - 1
    for bad in (None, 0.0, -5.0, "fast", float("nan")):
        assert measurement_floor_ms(bad) is None
    rr = _flutter(jitter=0.006, seed=2)          # RMSSD ~8.5 ms
    slow = hyper_regularity(_lat(rr, fps=30.0), age_years=30)
    fast = hyper_regularity(_lat(rr, fps=240.0), age_years=30)
    assert slow["measurement_limited"] is True
    assert slow["effective_floor_ms"] == slow["measurement_floor_ms"]
    assert fast["measurement_limited"] is False
    assert fast["effective_floor_ms"] == fast["rmssd_floor_ms"]
    # an unknown frame rate is not a permissive floor: no verdict at all
    lat = _lat(rr, fps=240.0)
    object.__setattr__(lat, "fps", float("nan"))
    unk = hyper_regularity(lat, age_years=30)
    assert unk["below_floor"] is False
    assert unk["unknown_measurement_floor"] is True
    assert unk["rmssd_ms"] is not None           # still reported


def test_flutter_is_below_the_floor_and_sinus_tach_is_not():
    fl = hyper_regularity(_lat(_flutter(seed=2)), age_years=60)
    st = hyper_regularity(_lat(_sinus(seed=2)), age_years=60)
    assert fl["below_floor"] is True and st["below_floor"] is False
    assert fl["rmssd_ms"] < st["rmssd_ms"]


def test_diffs_never_cross_a_run_boundary():
    """A fabricated jump between runs must not enter the dispersion."""
    rr = np.concatenate([_flutter(n=30), _flutter(n=30, base=0.800)])
    one = _lat(rr)                       # one run: the seam counts
    two = _lat(rr, runs=[30, 30])        # two runs: it cannot
    assert hyper_regularity(one, age_years=60)["rmssd_ms"] > \
        hyper_regularity(two, age_years=60)["rmssd_ms"]
    assert hyper_regularity(two, age_years=60)["rmssd_ms"] < 8.0


def test_tachogram_refuses_gaps_and_short_spans():
    assert tachogram(_lat(_flutter(n=10)))[0] is None       # too few
    # 40 intervals at 150 bpm is only 16 s — under five breaths, so the
    # respiratory band is unresolvable and the series is refused
    assert tachogram(_lat(_flutter(n=40)))[0] is None
    assert tachogram(_lat(_flutter(n=100)))[0] is not None
    # a 20 s hole: no clean intervals there, so no interpolation over it
    rr = _flutter(n=40)
    lat = _lat(rr, runs=[20, 20])
    lat.run_times[1] = lat.run_times[1] + 20.0
    assert tachogram(lat)[0] is None


def test_rsa_coupling_separates_flutter_from_sinus_tach():
    """The discriminator. Same rate, and here the DISPERSION is matched
    too, so only the coupling can tell them apart."""
    resp = _resp()
    st = respiratory_coupling(_lat(_sinus(seed=3)), resp)
    fl = respiratory_coupling(_lat(_flutter(seed=3)), resp)
    assert st["available"] and fl["available"]
    assert st["tachogram_resp_fraction"] > 0.5
    assert fl["tachogram_resp_fraction"] < 0.15
    assert st["phase_locking"] > 0.9
    # jitter-matched metronomic: dispersion cannot separate it from
    # sinus tach, coupling can (the measured 0.03 vs 0.91 case)
    jit = _flutter(jitter=0.013, seed=4)
    sin = _sinus(seed=4)
    rmssd = lambda x: float(np.sqrt(np.mean(np.diff(x) ** 2))) * 1000.0
    assert abs(rmssd(jit) - rmssd(sin)) < 8.0
    j = respiratory_coupling(_lat(jit), resp)
    s = respiratory_coupling(_lat(sin), resp)
    assert j["tachogram_resp_fraction"] < 0.20
    assert s["tachogram_resp_fraction"] > 3 * j["tachogram_resp_fraction"]


def test_coupling_is_unavailable_not_zero_without_a_respiration_channel():
    """An absent measurement is NOT evidence of absent coupling — this
    is the difference between abstaining and accusing."""
    lat = _lat(_flutter())
    for bad in (None, {}, {"rate_brpm": None}, {"rate_brpm": 40.0},
                {"rate_brpm": 3.0}):
        c = respiratory_coupling(lat, bad)
        assert c["available"] is False
        assert c["tachogram_resp_fraction"] is None
        assert c["reason"]
    # rate present but no waveform: the band fraction still computes,
    # phase locking honestly reports None
    c = respiratory_coupling(lat, {"rate_brpm": 15.0})
    assert c["available"] is True
    assert c["tachogram_resp_fraction"] is not None
    assert c["phase_locking"] is None


def test_phase_locking_needs_a_stable_relationship():
    lat = _lat(_sinus(seed=6))
    good = respiratory_coupling(lat, _resp())
    assert good["phase_locking"] > 0.9
    # a respiration channel at a DIFFERENT rate than the modulation is
    # now refused outright rather than read as "decoupled"
    off = respiratory_coupling(lat, _resp(brpm=22.0))
    assert off["available"] is False
    assert off["tachogram_resp_fraction"] is None


# ------------------------------------------------ family 3: the series
def test_latent_atrial_fit_recovers_one_atrial_rate_from_ratio_steps():
    f = latent_atrial_fit([150.0, 100.0, 75.0])
    assert f["score"] > 0.9
    assert f["atrial_bpm"] == pytest.approx(300.0, abs=3.0)
    assert sorted(set(f["ratios"])) == [2, 3, 4]
    assert f["residual_bpm"] < 1.0


def test_a_series_at_one_rate_scores_zero():
    """Three repeat scans of the SAME stable rhythm are explained by any
    atrial rate that divides it — that must not read as a conduction
    pattern (the guard that keeps the differentiator honest)."""
    f = latent_atrial_fit([150.0, 149.0, 151.0])
    assert f["score"] == 0.0
    assert "two separated" in f["reason"]
    # near-identical clusters are also refused
    assert latent_atrial_fit([150.0, 148.0, 138.0])["score"] == 0.0


def test_series_needs_three_scans_and_reports_why():
    f = latent_atrial_fit([150.0, 75.0])
    assert f["score"] == 0.0 and f["atrial_bpm"] is None
    assert "2 usable scans" in f["reason"]
    assert latent_atrial_fit(None)["n_scans"] == 0
    assert latent_atrial_fit([150.0, None, 75.0])["n_scans"] == 2


def test_non_flutter_series_does_not_fit_a_common_atrial_rate():
    # exercise recovery: rates drift smoothly, no integer structure
    f = latent_atrial_fit([132.0, 118.0, 104.0, 96.0])
    assert f["score"] < 0.6, f
    # ... while a real conduction series scores high on the same scale
    assert latent_atrial_fit([140.0, 93.3, 70.0])["score"] > 0.9


# ------------------------------------------------- variable block
def test_variable_block_lands_on_one_atrial_cycle_and_af_does_not():
    rng = np.random.default_rng(8)
    cycle = 60.0 / 300.0
    vb = np.asarray([cycle * int(rng.choice([2, 3, 4]))
                     + rng.normal(0, 0.004) for _ in range(60)])
    af = np.clip(rng.normal(0.62, 0.17, 60), 0.28, 1.35)
    v = variable_block_features(_lat(vb))
    a = variable_block_features(_lat(af))
    assert v["lattice_fraction"] > 0.9
    assert v["atrial_bpm"] == pytest.approx(300.0, abs=8.0)
    assert set(v["ratios_used"]) <= {2, 3, 4, 5}
    assert a["lattice_fraction"] < v["lattice_fraction"]


# ------------------------------------------------------ the session
class _Res:
    def __init__(self, outcome="ACCEPT"):
        self.outcome = type("O", (), {"value": outcome})()


def test_session_features_refuse_non_accept_scans():
    for bad in ("REPEAT_SCAN", "NO_RESULT"):
        out = flutter_features(_Res(bad), _lat(_flutter()))
        assert out["available"] is False
        assert "not ACCEPT" in out["reasons"][0]
        assert "rate" not in out


def test_session_features_carry_every_family_and_name_gaps():
    out = flutter_features(_Res(), _lat(_flutter(seed=9)),
                           respiration=_resp(), age_years=60,
                           series_rates_bpm=[150.0, 100.0, 75.0])
    assert out["available"] is True
    assert set(out) >= {"rate", "regularity", "variable_block", "series"}
    assert out["rate"]["in_band"] is True
    assert out["regularity"]["below_floor"] is True
    assert out["regularity"]["coupling"]["available"] is True
    assert out["series"]["score"] > 0.9
    assert out["reasons"] == []
    # without respiration the gap is NAMED, not silently zero
    out2 = flutter_features(_Res(), _lat(_flutter(seed=9)), age_years=60)
    assert out2["available"] is True
    assert any("respiratory coupling unavailable" in r
               for r in out2["reasons"])
    assert out2["regularity"]["coupling"]["tachogram_resp_fraction"] is None


def test_config_can_move_the_bands_and_floors():
    cfg = {"flutter": {"bands": {"2:1": (100.0, 120.0)},
                       "rmssd_floor_ms": {"unknown": 0.5},
                       "min_intervals": 15}}
    out = flutter_features(_Res(), _lat(_flutter(seed=9)), cfg=cfg)
    assert out["rate"]["in_band"] is False        # 150 is outside now
    assert out["regularity"]["below_floor"] is False   # floor moved down
    assert out["regularity"]["rmssd_floor_ms"] == 0.5


def test_a_lattice_without_interval_times_yields_nothing_invented():
    """Stored v1 lattices carry no run_times. The features must read
    that as "no times", never as times at zero."""
    lat = _lat(_flutter())
    bare = BeatLattice(
        version=LATTICE_VERSION, fps=lat.fps, duration_s=lat.duration_s,
        beat_t_s=lat.beat_t_s, beat_confidence=lat.beat_confidence,
        beat_agreement=lat.beat_agreement, runs=lat.runs,
        run_confidences=lat.run_confidences, n_intervals=lat.n_intervals,
        dropout_rate=0.0, split_fraction=0.0, per_roi_times={},
        segments=[])
    assert bare.run_times == []
    assert rate_fingerprint(bare)["median_bpm"] is None
    assert tachogram(bare)[0] is None
    r = respiratory_coupling(bare, _resp())
    assert r["available"] is False and r["reason"]
    # dispersion needs no times and still works — the run structure is
    # all it depends on
    assert hyper_regularity(bare, age_years=60)["rmssd_ms"] is not None


# ------------------------------------------------- through the video
@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    pytest.importorskip("cv2")
    from scripts.make_synth_flutter import flutter_rr, sinus_rr
    from scripts.make_synth_video import synth_video
    d = tmp_path_factory.mktemp("flutter_clips")
    out = {}
    for name, rr in (("flutter", flutter_rr(60.0, ratio=2, seed=5)),
                     # SHALLOW rsa: the hard confound. Deep RSA swings
                     # the rate out of the band on its own, so a
                     # shallow-RSA sinus tach is the honest test of
                     # whether coupling (not rate) does the separating.
                     ("sinus_tach", sinus_rr(60.0, bpm=152.0,
                                             rsa_depth=0.05, seed=5)),
                     ("sinus_tach_deep", sinus_rr(60.0, bpm=150.0,
                                                  rsa_depth=0.10, seed=5)),
                     ("flutter_4to1", flutter_rr(60.0, ratio=4, seed=5))):
        p = str(d / f"{name}.avi")
        synth_video(p, kind=f"flutter_track_{name}", fps=30.0,
                    duration_s=60.0, seed=5, rr_override=rr,
                    torso_respiration={"brpm": 15.0, "bob_px": 6.0})
        out[name] = p
    return out


def _run(path):
    from inference.pipeline import run_with_details
    from rppg.respiration import (respiratory_rate_from_motion,
                                  torso_motion_series)
    res, det = run_with_details(path)
    t, y, fps = torso_motion_series(path)
    rate, conc = respiratory_rate_from_motion(t, y, fps)
    from rppg._filters import moving_average_detrend
    resp = {"t": t, "y": moving_average_detrend(y, int(12.0 * fps)),
            "rate_brpm": rate, "quality": conc}
    return res, det, resp


def test_rsa_coupling_survives_the_production_video_path(clips):
    """THE miniature experiment for this track. Two clips at the same
    rate through the real pipeline: the camera must see the breathing,
    and the coupling must separate them where dispersion does not."""
    res_f, det_f, resp_f = _run(clips["flutter"])
    res_s, det_s, resp_s = _run(clips["sinus_tach"])
    assert resp_f["rate_brpm"] == pytest.approx(15.0, abs=1.5)
    assert resp_s["rate_brpm"] == pytest.approx(15.0, abs=1.5)
    f = flutter_features(res_f, det_f["lattice"], respiration=resp_f,
                         age_years=60)
    s = flutter_features(res_s, det_s["lattice"], respiration=resp_s,
                         age_years=60)
    assert f["available"] and s["available"]
    # BOTH sustain the same rate band: rate cannot separate them
    assert f["rate"]["band"] == s["rate"]["band"] == "2:1"
    assert f["rate"]["in_band"] and s["rate"]["in_band"]
    # ... but the coupling does
    cf = f["regularity"]["coupling"]["tachogram_resp_fraction"]
    cs = s["regularity"]["coupling"]["tachogram_resp_fraction"]
    assert cf is not None and cs is not None
    assert cs > 0.4, cs
    assert cf < 0.2, cf
    assert cs > 3 * cf
    # and the dispersion floor agrees on the same clips — at 30 fps that
    # floor is the CAMERA's, which the feature declares
    assert f["regularity"]["measurement_limited"] is True
    assert f["regularity"]["below_floor"] is True
    assert s["regularity"]["below_floor"] is False


def test_widening_the_band_costs_the_second_exit(clips):
    """A measured TRADE-OFF, recorded rather than hidden. Widening the
    2:1 band to 125-165 (so it covers the 250-300/min atrial range the
    limitations doc states) means deep-RSA sinus tachycardia now SUSTAINS
    the band — it no longer swings out of it. The band is therefore no
    longer an independent exit, and the coupling test plus the
    dispersion floor carry the whole discrimination for this
    confounder."""
    from heads import get_head
    res, det, resp = _run(clips["sinus_tach_deep"])
    out = flutter_features(res, det["lattice"], respiration=resp,
                           age_years=60)
    assert out["rate"]["band"] == "2:1"
    assert out["rate"]["median_bpm"] == pytest.approx(150.0, abs=4.0)
    assert out["rate"]["in_band"] is True          # the cost
    assert out["rate"]["band_fraction"] > 0.80
    # ... and it is still not flagged, by the other two criteria
    assert out["regularity"]["below_floor"] is False
    assert out["regularity"]["coupling"]["tachogram_resp_fraction"] > 0.5
    flag = get_head("flutter").run(
        det["lattice"], {"scan_outcome": res.outcome.value, "cfg": {},
                         "respiration": resp,
                         "age_years": 60}).value["regular_tachy_flag"]
    assert flag is False


def test_slow_fixed_block_is_invisible_through_the_video_path(clips):
    """F-c, measured rather than asserted: 4:1 flutter at ~75 bpm is
    metronomic and normal-rated, so the rate band excludes it and the
    product must never claim otherwise."""
    res, det, resp = _run(clips["flutter_4to1"])
    out = flutter_features(res, det["lattice"], respiration=resp,
                           age_years=60)
    assert out["rate"]["median_bpm"] == pytest.approx(75.0, abs=4.0)
    assert out["rate"]["band"] == "4:1"
    # it IS hyper-regular — the regularity family sees it fine ...
    assert out["regularity"]["below_floor"] is True
    # ... and it is STILL not a 2:1-band finding. That is the miss.
    assert out["rate"]["band"] != "2:1"


def test_run_times_survive_serialization_and_the_real_pipeline(clips):
    """The additive lattice field, both directions: a stored v1 lattice
    (no run_times key) still loads and reads as "no times", and the
    production path actually populates it, parallel to runs."""
    from beats.lattice import BeatLattice
    v1 = {"version": LATTICE_VERSION, "fps": 30.0, "duration_s": 30.0,
          "beat_t_s": [1.0, 2.0], "beat_confidence": [1.0, 1.0],
          "beat_agreement": [1.0, 1.0], "runs": [[1000.0]],
          "run_confidences": [[1.0]], "n_intervals": 1,
          "dropout_rate": 0.0, "split_fraction": 0.0,
          "per_roi_times": {}, "segments": []}
    lat = BeatLattice.from_dict(v1)
    assert lat.run_times == []                  # absent, not invented
    assert "run_times" in lat.to_dict()
    assert BeatLattice.from_dict(lat.to_dict()).run_times == []
    # ... and a real scan carries one time per interval, in order
    from inference.pipeline import run_with_details
    _, det = run_with_details(clips["flutter"])
    lat = det["lattice"]
    assert lat.run_times and len(lat.run_times) == len(lat.runs)
    assert [len(r) for r in lat.runs] == [len(t) for t in lat.run_times]
    for t in lat.run_times:
        assert np.all(np.diff(t) > 0)
        assert 0.0 <= t[0] and t[-1] <= lat.duration_s + 1.0
    # a round trip through the dict preserves them
    back = BeatLattice.from_dict(lat.to_dict())
    assert all(np.allclose(a, b) for a, b in zip(back.run_times,
                                                 lat.run_times))


def test_a_zero_score_fit_names_no_atrial_rate():
    """A refused fit must not hand back a number only the score reveals
    as meaningless."""
    for rates in ([400.0, 200.0, 100.0],      # needs a 1:1 that is not
                                              # in the ratio set
                  [132.0, 118.0, 104.0, 96.0],   # smooth recovery drift
                  [150.0, 149.0, 151.0]):         # one cluster
        f = latent_atrial_fit(rates)
        assert f["score"] == 0.0, rates
        assert f["atrial_bpm"] is None, (rates, f)
        assert f["ratios"] is None, rates
        assert f["reason"], rates
    # and a real fit still reports both
    good = latent_atrial_fit([150.0, 100.0, 75.0])
    assert good["atrial_bpm"] == pytest.approx(300.0, abs=3.0)
    assert good["ratios"] and good["reason"] is None


def test_coupling_refuses_a_respiration_rate_the_tachogram_contradicts():
    """The discriminator fails OPEN if a wrong breathing rate is taken
    on trust: the coupling test is a narrow band around the REPORTED
    rate, so a 2 br/min error read a fully coupled sinus tachycardia as
    'modulation absent' — i.e. as a flag. Measured before the fix:
    1.00 at 15/min, 0.17 at 17/min, 0.00 at 18/min."""
    lat = _lat(_sinus(seed=3))                # true RSA at 15/min
    ok = respiratory_coupling(lat, _resp(15.0))
    assert ok["available"] is True
    assert ok["tachogram_resp_fraction"] > 0.9
    for wrong in (17.0, 18.0, 22.0, 12.0):
        c = respiratory_coupling(lat, _resp(wrong))
        assert c["available"] is False, wrong
        assert c["tachogram_resp_fraction"] is None, wrong
        assert "cannot be trusted" in c["reason"], wrong
        assert c["tachogram_peak_hz"] == pytest.approx(0.25, abs=0.02)
    # a low-quality respiration channel is refused before it is used
    poor = respiratory_coupling(lat, _resp(15.0, quality=0.1))
    assert poor["available"] is False
    assert "quality" in poor["reason"]
    # and a genuinely unmodulated series is still DECOUPLED, not
    # refused: its strongest respiratory-band bin is noise, and
    # refusing on that would make the flag unreachable
    fl = respiratory_coupling(_lat(_flutter(seed=3)), _resp(15.0))
    assert fl["available"] is True
    assert fl["tachogram_resp_fraction"] < 0.15


def test_sustained_is_a_fraction_of_time_not_of_intervals():
    """At 150 bpm an in-band interval is 2.5x shorter than one at 60, so
    counting intervals over-weights the fast stretch: half a scan in
    band passed a 70% 'sustained' test."""
    # 40 s at 150 bpm (100 intervals) then 40 s at 60 bpm (40 intervals)
    rr = np.concatenate([np.full(100, 0.4), np.full(40, 1.0)])
    r = rate_fingerprint(_lat(rr))
    assert r["band_interval_fraction"] > 0.70      # what it used to read
    assert r["band_fraction"] == pytest.approx(0.5, abs=0.02)
    assert r["in_band"] is False
    assert r["sustained_s"] == pytest.approx(40.0, abs=1.0)


def test_the_2to1_band_covers_the_atrial_range_the_docs_claim():
    """F-c: the product must not contradict its own limitations doc,
    which states atria fire at ~250-300/min. 2:1 of that is 125-150, and
    a band starting at 140 silently missed most of it."""
    for atrial in (250.0, 260.0, 280.0, 300.0, 330.0):
        rr = np.full(200, 2 * 60.0 / atrial)
        r = rate_fingerprint(_lat(rr))
        assert r["band"] == "2:1", atrial
        assert r["in_band"] is True, (atrial, r["median_bpm"])
    doc = (pathlib.Path(__file__).resolve().parents[1] / "docs" /
           "flutter_limitations.md").read_text()
    assert "250" in doc and "300" in doc
