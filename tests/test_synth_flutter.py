"""v0.6 flutter T7 — the fixture generator. A fixture that does not
carry the physiology it claims makes every downstream gate vacuous, so
the interval series, the conduction arithmetic, the breathing channel
and the RSA coupling are each pinned here, and the legacy synth path is
proven bit-for-bit unchanged."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

from datasets.schema import ConductionRatio, Rhythm, flutter_label_problems
from scripts.make_synth_flutter import (COHORTS, DEFAULT_ATRIAL_BPM,
                                        af_rr, flutter_rr, paced_rr,
                                        sinus_rr, svt_rr,
                                        switching_flutter_rr,
                                        variable_block_rr)

D = 40.0


def _bpm(rr):
    return 60.0 / np.median(rr)


def test_fixed_ratio_flutter_lands_on_the_conducted_rate():
    """The whole point of the track: one atrial rate, three pulses."""
    for ratio, want in ((2, 150.0), (3, 100.0), (4, 75.0)):
        rr = flutter_rr(D, atrial_bpm=DEFAULT_ATRIAL_BPM, ratio=ratio,
                        seed=3)
        assert _bpm(rr) == pytest.approx(want, abs=1.5), ratio
        # metronomic: dispersion far below any sinus rhythm
        assert np.std(rr) * 1000.0 < 6.0, ratio
        assert np.sum(rr) <= D


def test_flutter_is_hyper_regular_and_sinus_is_not():
    fl = flutter_rr(D, ratio=2, seed=5)
    st = sinus_rr(D, bpm=150.0, seed=5)
    rmssd = lambda x: float(np.sqrt(np.mean(np.diff(x) ** 2))) * 1000.0
    assert rmssd(fl) < 6.0
    assert rmssd(st) > 3 * rmssd(fl)
    # ... and at the SAME rate, so rate alone cannot separate them
    assert abs(_bpm(fl) - _bpm(st)) < 8.0


def test_conduction_ratio_step_inside_one_scan():
    rr = switching_flutter_rr(D, ratios=(2, 4), seed=7)
    # split at the DECLARED step time (the fixture switches on the
    # clock); half the kept duration is a different instant, because
    # truncation drops the last partial interval
    step_at = D / 2.0
    starts = np.cumsum(rr) - rr          # an interval belongs to the
    first = rr[starts < step_at]         # segment it STARTED in; exactly
    second = rr[starts >= step_at]       # one interval straddles the step
    assert first.size > 5 and second.size > 5
    r = float(np.median(second) / np.median(first))
    assert r == pytest.approx(2.0, abs=0.15)     # 2:1 -> 4:1 halves the rate
    # each half is itself metronomic — the STEP is the signature, not
    # any change in regularity
    for half_rr in (first, second):
        assert np.std(half_rr) * 1000.0 < 6.0


def test_variable_block_is_irregular_and_lands_in_af_space():
    vb = variable_block_rr(D, seed=11)
    af = af_rr(D, seed=11)
    rmssd = lambda x: float(np.sqrt(np.mean(np.diff(x) ** 2))) * 1000.0
    assert rmssd(vb) > 40.0                      # genuinely irregular
    assert rmssd(af) > 40.0
    # every interval is an integer multiple of ONE atrial cycle
    cycle = 60.0 / DEFAULT_ATRIAL_BPM
    k = vb / cycle
    assert np.all(np.abs(k - np.round(k)) < 0.10)
    assert set(np.round(k).astype(int)) <= {2, 3, 4}


def test_negative_cohort_rates_sit_where_the_battery_needs_them():
    assert _bpm(svt_rr(D, seed=2)) == pytest.approx(165.0, abs=4.0)
    assert _bpm(paced_rr(D, seed=2)) == pytest.approx(70.0, abs=2.0)
    assert _bpm(sinus_rr(D, bpm=62.0, rsa_depth=0.02, seed=2)) == \
        pytest.approx(62.0, abs=3.0)
    # beta-blocked sinus is hyper-regular at a NORMAL rate: only the
    # rate band keeps it out of the flag
    bb = sinus_rr(D, bpm=62.0, rsa_depth=0.02, noise_ms=4.0, seed=2)
    assert float(np.sqrt(np.mean(np.diff(bb) ** 2))) * 1000.0 < 25.0


def test_rsa_depth_shows_up_at_the_breathing_frequency():
    """Sinus intervals must actually be modulated at resp_brpm — the
    hyper-regularity family is measured against exactly this."""
    for depth in (0.05, 0.10):
        rr = sinus_rr(120.0, bpm=150.0, rsa_depth=depth, resp_brpm=15.0,
                      noise_ms=1.0, seed=4)
        t = np.cumsum(rr)
        grid = np.arange(t[0], t[-1], 0.25)
        x = np.interp(grid, t, rr) - np.mean(rr)
        spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        fr = np.fft.rfftfreq(x.size, 0.25)
        peak = float(fr[np.argmax(spec)])
        assert peak == pytest.approx(0.25, abs=0.03), depth
    # flutter has no such peak: its power sits in the noise floor
    fl = flutter_rr(120.0, ratio=2, seed=4)
    t = np.cumsum(fl)
    grid = np.arange(t[0], t[-1], 0.25)
    x = np.interp(grid, t, fl) - np.mean(fl)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    fr = np.fft.rfftfreq(x.size, 0.25)
    band = (np.abs(fr - 0.25) <= 0.03)
    wide = (fr >= 0.04) & (fr <= 1.0)
    assert float(spec[band].sum() / spec[wide].sum()) < 0.20


def test_every_cohort_declares_what_the_flag_should_do():
    for name, c in COHORTS.items():
        assert isinstance(c["flag_expected"], bool), name
        assert isinstance(c["rhythm"], Rhythm), name
        rr = c["build"](D, 17, 15.0)
        assert rr.size > 8, name
        assert np.all(rr > 0.2) and np.all(rr < 2.0), name
    # the KNOWN MISS is declared in the fixture table, not just prose
    assert COHORTS["flutter_4to1"]["flag_expected"] is False
    assert COHORTS["flutter_2to1"]["flag_expected"] is True
    # SVT is covered BY DESIGN — the flag cannot separate it from 2:1
    assert COHORTS["svt"]["flag_expected"] is True
    assert COHORTS["sinus_tach"]["flag_expected"] is False


# ------------------------------------------------------- the video path
def test_legacy_synth_output_is_unchanged_without_the_breathing_hook():
    """torso_respiration=None must reproduce the historical video and
    truth bit for bit — every pre-v0.6 fixture depends on it."""
    pytest.importorskip("cv2")
    import hashlib
    import tempfile
    from scripts.make_synth_video import synth_video
    d = pathlib.Path(tempfile.mkdtemp())
    a = synth_video(str(d / "a.avi"), kind="sinus", fps=30.0,
                    duration_s=6.0, seed=3)
    b = synth_video(str(d / "b.avi"), kind="sinus", fps=30.0,
                    duration_s=6.0, seed=3, torso_respiration=None)
    assert hashlib.sha256((d / "a.avi").read_bytes()).hexdigest() == \
        hashlib.sha256((d / "b.avi").read_bytes()).hexdigest()
    assert a["torso_respiration"] is None
    assert {k: v for k, v in a.items() if k != "torso_respiration"} == \
        {k: v for k, v in b.items() if k != "torso_respiration"}


def test_breathing_hook_is_validated_and_recorded():
    pytest.importorskip("cv2")
    import tempfile
    from scripts.make_synth_video import synth_video
    d = pathlib.Path(tempfile.mkdtemp())
    for bad in ({"brpm": 40.0}, {"brpm": 2.0}, {"brpm": 15.0,
                                                "bob_px": 0.0}):
        with pytest.raises(ValueError):
            synth_video(str(d / "x.avi"), kind="sinus", fps=30.0,
                        duration_s=4.0, seed=1, torso_respiration=bad)
    t = synth_video(str(d / "ok.avi"), kind="sinus", fps=30.0,
                    duration_s=4.0, seed=1,
                    torso_respiration={"brpm": 15.0, "bob_px": 6.0})
    assert t["torso_respiration"] == {"brpm": 15.0, "bob_px": 6.0}


def test_breathing_is_readable_and_leaves_the_pulse_path_alone():
    """The bar must produce a real respiration read (the RSA family is
    undefined without one) while changing nothing about the face path:
    it is neutral grey, so the skin-chromaticity tracker cannot see it,
    and it sits below the ellipse, so no ROI can."""
    pytest.importorskip("cv2")
    import tempfile
    from inference.pipeline import run_with_details
    from rppg.respiration import (respiratory_rate_from_motion,
                                  torso_motion_series)
    from scripts.make_synth_video import synth_video
    d = pathlib.Path(tempfile.mkdtemp())
    rr = flutter_rr(60.0, ratio=2, seed=5)
    paths = {}
    for name, resp in (("breath", {"brpm": 15.0, "bob_px": 6.0}),
                       ("plain", None)):
        paths[name] = str(d / f"{name}.avi")
        synth_video(paths[name], kind="flutter_track_test", fps=30.0,
                    duration_s=60.0, seed=5, rr_override=rr,
                    torso_respiration=resp)
    t, y, fps = torso_motion_series(paths["breath"])
    rate, conc = respiratory_rate_from_motion(t, y, fps)
    assert rate == pytest.approx(15.0, abs=1.0)
    assert conc > 0.5
    # no bar -> no breathing to read: fail-closed, as before
    t0, y0, fps0 = torso_motion_series(paths["plain"])
    assert respiratory_rate_from_motion(t0, y0, fps0)[0] is None
    # the pulse path is untouched by the bar
    a, det_a = run_with_details(paths["breath"])
    b, det_b = run_with_details(paths["plain"])
    assert a.mean_pulse_rate_bpm == pytest.approx(b.mean_pulse_rate_bpm,
                                                  abs=1.0)
    assert det_a["lattice"].n_intervals == det_b["lattice"].n_intervals


def test_dataset_writes_labeled_scans_and_a_serial_subprotocol(tmp_path):
    pytest.importorskip("cv2")
    from scripts.make_synth_flutter import make_flutter_dataset
    m = make_flutter_dataset(tmp_path, duration_s=12.0,
                             cohorts=["flutter_2to1", "sinus_tach"],
                             serial_participants=1)
    assert len(m["scans"]) == 2 + 3
    assert m["series"] and m["series"][0]["flutter_series"] is True
    assert len(m["series"][0]["recordings"]) == 3     # >= 3: the F3 floor
    for row in m["scans"]:
        p = tmp_path / f"{row['recording_id']}.recording.json"
        doc = json.loads(p.read_text())
        assert doc["participant_id"] == row["participant_id"]
        anns = doc["rhythm_annotations"]
        assert len(anns) == 1
        if row["rhythm"] == Rhythm.ATRIAL_FLUTTER.value:
            # a positive carries the full naming burden
            assert doc["ecg_leads"] == 12
            assert anns[0]["adjudicator_id"]
            assert anns[0]["atrial_rate_bpm"] == DEFAULT_ATRIAL_BPM
            assert anns[0]["conduction_ratio"]
        else:
            assert doc["ecg_leads"] == 1
            assert anns[0]["conduction_ratio"] is None
    # the serial participant's three scans step one atrial rate
    ids = m["series"][0]["recordings"]
    rates = [r["median_bpm"] for r in m["scans"]
             if r["recording_id"] in ids]
    assert len(rates) == 3
    for r in rates:
        k = DEFAULT_ATRIAL_BPM / r
        assert abs(k - round(k)) < 0.12, rates
    assert len({round(DEFAULT_ATRIAL_BPM / r) for r in rates}) == 3


def test_generated_flutter_labels_pass_the_label_bar(tmp_path):
    """The generator must not emit a positive its own schema rejects."""
    pytest.importorskip("cv2")
    from scripts.make_synth_flutter import write_scan
    row = write_scan(tmp_path, "r_x", "flutter_2to1", pid="p1",
                     session_id="p1-s1", seed=2, duration_s=8.0)
    doc = json.loads((tmp_path / "r_x.recording.json").read_text())
    a = doc["rhythm_annotations"][0]
    from datasets.schema import FlutterType, RhythmAnnotation
    ann = RhythmAnnotation(
        a["t_start_s"], a["t_end_s"], Rhythm(a["rhythm"]),
        annotator_id=a["annotator_id"], adjudicated=a["adjudicated"],
        atrial_rate_bpm=a["atrial_rate_bpm"],
        conduction_ratio=ConductionRatio(a["conduction_ratio"]),
        flutter_type=FlutterType(a["flutter_type"]),
        adjudicator_id=a["adjudicator_id"],
        adjudication_leads=a["adjudication_leads"])
    assert flutter_label_problems(ann) == []
    assert row["flag_expected"] is True


def test_flag_expected_is_an_ORACLE_not_a_label():
    """Every cohort's declared flag_expected must match what the REAL
    head does with it. Asserting only that the field is a bool would
    pass for any implementation — and did, while the SVT fixture sat on
    the band edge and contradicted its own declaration."""
    import numpy as np
    from beats.lattice import BeatLattice, LATTICE_VERSION
    from heads import get_head
    from scripts.make_synth_flutter import DEFAULT_RESP_BRPM
    head = get_head("flutter")
    wrong = {}
    for name, c in COHORTS.items():
        rr = c["build"](90.0, 31, DEFAULT_RESP_BRPM)
        t = np.cumsum(rr)
        lat = BeatLattice(
            version=LATTICE_VERSION, fps=240.0,
            duration_s=float(t[-1] + 1),
            beat_t_s=np.concatenate([[0.0], t]),
            beat_confidence=np.ones(t.size + 1),
            beat_agreement=np.ones(t.size + 1), runs=[rr * 1000.0],
            run_confidences=[np.ones(rr.size)], n_intervals=int(rr.size),
            dropout_rate=0.0, split_fraction=0.0, per_roi_times={},
            segments=[], run_times=[t])
        tt = np.arange(0, float(t[-1]) + 5, 1 / 30.0)
        resp = {"t": tt,
                "y": np.sin(2 * np.pi * (DEFAULT_RESP_BRPM / 60.0) * tt),
                "rate_brpm": DEFAULT_RESP_BRPM, "quality": 0.9}
        got = head.run(lat, {"scan_outcome": "ACCEPT", "cfg": {},
                             "respiration": resp,
                             "age_years": 60}).value["regular_tachy_flag"]
        if got is not c["flag_expected"]:
            wrong[name] = (c["flag_expected"], got)
    assert not wrong, wrong
