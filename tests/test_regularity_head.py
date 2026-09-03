"""v0.7 Task 4 — head_regularity: graded not binary (G-b), benign
evidence on every irregular call (G-c), no rhythm naming (G-d), and
abstention that is distinguishable from a negative."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ast
import json

import numpy as np
import pytest

from beats.lattice import BeatLattice, LATTICE_VERSION
from configs import load_config
from datasets.schema import MeasurementClass, REGULARITY_SENTENCES
from heads import available_heads, enabled_heads, get_head
from heads.head_regularity import (BENIGN_EVIDENCE, user_facing_text,
                                   WATERMARK)
from scripts.make_synth_regularity import COHORTS, regular_rr

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _lat(rr, fps=30.0):
    rr = np.asarray(rr, float)
    t = np.cumsum(rr)
    return BeatLattice(
        version=LATTICE_VERSION, fps=fps, duration_s=float(t[-1] + 1),
        beat_t_s=np.concatenate([[0.0], t]),
        beat_confidence=np.ones(t.size + 1),
        beat_agreement=np.ones(t.size + 1), runs=[rr * 1000.0],
        run_confidences=[np.ones(rr.size)], n_intervals=int(rr.size),
        dropout_rate=0.0, split_fraction=0.0, per_roi_times={},
        segments=[], run_times=[t])


def _resp(dur=60.0, brpm=15.0, quality=0.9):
    t = np.arange(0, dur + 5, 1 / 30.0)
    return {"t": t, "y": np.sin(2 * np.pi * brpm / 60.0 * t),
            "rate_brpm": brpm, "quality": quality}


def _ctx(**over):
    d = {"scan_outcome": "ACCEPT", "cfg": {}, "respiration": _resp()}
    d.update(over)
    return d


def test_registered_research_only_never_default_enabled():
    h = get_head("regularity")
    assert h.research_only is True and "regularity" in available_heads()
    assert "regularity" not in [x.name for x in enabled_heads(load_config())]
    r = h.run(_lat(regular_rr(60.0)), _ctx())
    assert r.measurement_class is MeasurementClass.RESEARCH_RHYTHM
    from datasets.schema import public_head_results
    assert public_head_results([r.to_dict()]) == []


def test_every_cohort_gets_class_and_explanation_as_declared():
    """The fixture table is an ORACLE: class AND benign explanation for
    every cohort, two seeds each, through the real head."""
    h = get_head("regularity")
    wrong = {}
    for name, c in COHORTS.items():
        for seed in (5, 11):
            v = h.run(_lat(c["build"](60.0, seed, 15.0)), _ctx()).value
            got = (v["class"], (v["benign_pattern_evidence"] or {})
                   .get("evidence"))
            want = (c["expect"], c["benign"] or "not_applicable")
            if got != want:
                wrong[f"{name}/s{seed}"] = {"got": got, "want": want,
                                           "index": v["index"]["value"]}
    assert not wrong, wrong


def test_graded_not_binary_the_index_and_ci_always_ride_with_a_class():
    """G-b: any path producing a class produces the index and CI."""
    h = get_head("regularity")
    for c in COHORTS.values():
        v = h.run(_lat(c["build"](60.0, 5, 15.0)), _ctx()).value
        if v["class"] is not None:
            assert v["index"]["value"] is not None
            lo, hi = v["index"]["ci95"]
            assert lo <= v["index"]["value"] <= hi
            assert v["index"]["n_diffs"] >= 15
            assert v["threshold"] == pytest.approx(0.06)
    # ... and the measurement stands even when the head abstains
    v = h.run(_lat(regular_rr(60.0)), _ctx(respiration=None)).value
    assert v["class"] is None and v["abstained"] is True
    assert v["index"]["value"] is not None and v["index"]["ci95"] is not None
    assert v["jitter"]["rmssd_floor_ms"] is not None


def test_benign_evidence_is_mandatory_on_every_irregular_call():
    """G-c: an irregular class ALWAYS carries benign_pattern_evidence
    from the closed vocabulary; indeterminate is allowed, omission is
    not."""
    h = get_head("regularity")
    seen = set()
    for c in COHORTS.values():
        for seed in (5, 11, 17):
            v = h.run(_lat(c["build"](60.0, seed, 15.0)), _ctx()).value
            if v["class"] == "irregular":
                ev = v["benign_pattern_evidence"]
                assert ev is not None and ev["evidence"] in BENIGN_EVIDENCE
                assert "coupling_available" in ev
                seen.add(ev["evidence"])
    assert {"respiration_coupled", "ectopy_pattern", "chaotic"} <= seen
    # the closed vocabulary IS the G-c vocabulary
    assert set(BENIGN_EVIDENCE) == {"respiration_coupled", "ectopy_pattern",
                                    "chaotic", "indeterminate"}


def test_indeterminate_is_explicit_never_silent():
    """A series that is irregular but neither coupled, periodic nor
    pervasively scattered: the evidence says so, in so many words."""
    h = get_head("regularity")
    rng = np.random.default_rng(4)
    # moderate irregularity with a few isolated but unpaired long beats
    rr = 0.85 + rng.normal(0, 0.012, 70)
    rr[::9] += 0.09
    v = h.run(_lat(rr), _ctx(respiration=_resp(brpm=22.0))).value
    if v["class"] == "irregular":
        assert v["benign_pattern_evidence"]["evidence"] in BENIGN_EVIDENCE
    assert v["benign_pattern_evidence"] is not None


def test_abstains_on_missing_channel_but_judges_non_breathing_periodicity():
    h = get_head("regularity")
    lat = _lat(COHORTS["trigeminy"]["build"](60.0, 5, 15.0))
    # no channel: abstain, with the reason named
    r = h.run(lat, _ctx(respiration=None))
    assert r.value["class"] is None and r.value["abstained"] is True
    assert any("breathing channel" in x for x in r.reasons)
    # poor channel: abstain
    r = h.run(lat, _ctx(respiration=_resp(quality=0.1)))
    assert r.value["class"] is None
    # a channel whose reported rate the tachogram contradicts because the
    # series is periodic at a NON-breathing rate (trigeminy's 3-beat
    # cycle sits inside the respiratory band): that is a finding, not
    # an absence — the head judges it and explains it as ectopy
    r = h.run(lat, _ctx())
    assert r.value["class"] == "irregular"
    assert r.value["benign_pattern_evidence"]["evidence"] == "ectopy_pattern"
    # too few clean intervals: abstain
    r = h.run(_lat(regular_rr(9.0)), _ctx())
    assert r.value["class"] is None
    assert any("clean intervals" in x for x in r.reasons)
    # non-ACCEPT: abstain
    for bad in ("REPEAT_SCAN", "NO_RESULT", None):
        r = h.run(lat, _ctx(scan_outcome=bad))
        assert r.value["class"] is None


def test_no_rhythm_is_named_and_wording_leaves_only_through_the_gate():
    """G-d."""
    h = get_head("regularity")
    banned = ("afib", "fibrillation", "flutter", "ectopy", "premature",
              "bigeminy", "svt", "sinus")
    for c in COHORTS.values():
        v = h.run(_lat(c["build"](60.0, 5, 15.0)), _ctx()).value
        blob = json.dumps(v).lower()
        # the internal evidence label "ectopy_pattern" is the G-c
        # vocabulary the prompt mandates; it is research payload
        # (RESEARCH_RHYTHM) and never a sentence. Everything else that
        # names a rhythm is banned outright.
        blob_ext = blob.replace("ectopy_pattern", "")
        for b in banned:
            assert b not in blob_ext, (b, blob[:200])
        assert v["user_facing"] is None
        text = user_facing_text(v, render_allowed=True)
        if v["class"] == "irregular":
            assert text == REGULARITY_SENTENCES["irregular"]
            low = text.lower()
            for b in banned:
                assert b not in low, b
            assert "breathing" in low and "ecg" in low
        else:
            assert text is None
        assert user_facing_text(v) is None                # default: closed
        assert user_facing_text(v, render_allowed=False) is None
    assert "NOT A DIAGNOSIS" in WATERMARK


def test_head_never_escalates_to_an_afib_sentence():
    """R5: escalation is head_afib's job. The head's module never
    references the AF sentence path, and its only sentence key maps to
    the one sanctioned table."""
    src = (_ROOT / "heads" / "head_regularity.py").read_text()
    assert "AFIB_SUGGESTIVE" not in src
    assert "afib_probability" not in src
    assert set(REGULARITY_SENTENCES) == {"irregular"}
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [a.name for a in node.names]
            assert not mod.startswith(("research", "inference.decision")), mod
            assert "decide_with_rationale" not in names


def test_pipeline_publishes_a_research_row_that_abstains_and_is_stripped(tmp_path):
    pytest.importorskip("cv2")
    from datasets.schema import public_head_results
    from inference.pipeline import run_with_details
    from scripts.make_synth_regularity import write_scan
    row = write_scan(tmp_path, "r_x", "rsa_young", pid="p1",
                     session_id="p1-s1", seed=3, duration_s=40.0)
    res, det = run_with_details(str(tmp_path / row["video"]),
                                heads=["afib", "regularity"])
    hr = [h for h in res.head_results if h["head"] == "regularity"][0]
    assert hr["measurement_class"] == "RESEARCH_RHYTHM"
    # the production path has no respiration channel: abstain, index kept
    assert hr["value"]["class"] is None
    assert hr["value"]["index"]["value"] is not None
    assert [h["head"] for h in public_head_results(res.head_results)] == \
        ["afib"]
    # the representation the head read is the one the pipeline built
    assert det["regularity"].index["value"] == hr["value"]["index"]["value"]


# ------------------------------------------ the benign rule, adversarially
def _resp_wave(dur, brpm, phase=0.0, fs=30.0):
    t = np.arange(0, dur, 1.0 / fs)
    return {"t": t, "y": np.sin(2 * np.pi * brpm / 60.0 * t + phase),
            "rate_brpm": brpm, "quality": 0.7}


def _reg_from_rr(rr, resp):
    from features.regularity import regularity_from_runs
    t = np.cumsum(rr)
    return regularity_from_runs([np.diff(t) * 1000.0], run_times=[t[1:]],
                                fps=30.0, respiration=resp)


def _evidence(rr, resp):
    from heads.head_regularity import benign_pattern_evidence
    return benign_pattern_evidence(_reg_from_rr(rr, resp), threshold=0.06)


def test_rsa_is_explained_only_when_the_residual_is_regular():
    from scripts.make_synth_regularity import rsa_rr
    rr = rsa_rr(45.0, depth=0.16, resp_brpm=15.0, noise_ms=2.0, seed=1)
    ev = _evidence(rr, _resp_wave(45.0, 15.0))
    assert ev["evidence"] == "respiration_coupled"
    assert ev["coupling_route"] in ("reported_rate", "peak_near_reported_rate")
    assert ev["residual_index_estimate"] < 0.06
    assert ev["phase_locking"] > 0.95


def test_a_breathing_rate_error_does_not_turn_rsa_into_ectopy():
    """Review finding: a 1.5-2 br/min error in the reported rate eroded
    the band fraction and the alternation index took over. Phase locking
    is rate-error-robust and the waveform exposes the true rate."""
    from scripts.make_synth_regularity import rsa_rr
    rr = rsa_rr(45.0, depth=0.16, resp_brpm=20.0, noise_ms=2.0, seed=1)
    # ~3.4 beats per breath: lag-3 autocorrelation is high (ectopy-like)
    for reported in (18.5, 20.0, 21.5):
        ev = _evidence(rr, _resp_wave(45.0, 20.0) | {"rate_brpm": reported})
        assert ev["evidence"] == "respiration_coupled", (reported, ev)
    rr = rsa_rr(45.0, depth=0.16, resp_brpm=15.0, noise_ms=2.0, seed=1)
    ev = _evidence(rr, _resp_wave(45.0, 15.0) | {"rate_brpm": 17.2})
    assert ev["evidence"] == "respiration_coupled", ev
    assert ev["coupling_route"] in ("peak_near_reported_rate",
                                    "waveform_at_peak")
    # a larger error: the reported rate is no longer corroborated at
    # all and only the waveform can expose it
    ev = _evidence(rr, _resp_wave(45.0, 15.0) | {"rate_brpm": 18.5})
    assert ev["evidence"] == "respiration_coupled", ev
    assert ev["coupling_route"] == "waveform_at_peak"


def test_ectopy_with_a_pattern_period_in_the_breathing_band_stays_ectopy():
    """Review finding: trigeminy at 72 bpm has a 24/min pattern period,
    inside the respiratory band. Unless the breath is at that very rate
    AND locked to it, the tachogram is not moving with the breath."""
    from scripts.make_synth_regularity import bigeminy_rr, trigeminy_rr
    rr = trigeminy_rr(45.0, seed=2)
    for brpm in (12.0, 15.0, 20.0, 21.0):
        ev = _evidence(rr, _resp_wave(45.0, brpm, phase=1.0))
        assert ev["evidence"] == "ectopy_pattern", (brpm, ev)
    ev = _evidence(bigeminy_rr(45.0, seed=2), _resp_wave(45.0, 15.0, phase=1.0))
    assert ev["evidence"] == "ectopy_pattern", ev


def test_known_limitation_ectopy_exactly_at_the_breathing_rate_is_coupled():
    """DOCUMENTED LIMITATION (docs/regularity_track.md): an ectopic
    pattern whose period equals the breathing period within the window's
    resolution, with a stable phase, is indistinguishable from RSA by
    any of the three signatures. Pinned so the limitation cannot drift
    unnoticed."""
    from scripts.make_synth_regularity import trigeminy_rr
    rr = trigeminy_rr(45.0, seed=2)                 # 72 bpm -> 24/min
    ev = _evidence(rr, _resp_wave(45.0, 24.0, phase=1.0))
    assert ev["evidence"] == "respiration_coupled"


def test_a_chaotic_series_with_respiratory_modulation_is_not_explained():
    """Review finding: AF-like scatter with 20 % modulation at the
    breathing rate had > 50 % of its power in the band and read as
    respiration_coupled. What is left after the breath is removed must
    itself be regular, and here it is not."""
    from scripts.make_synth_flutter import af_rr
    rr = af_rr(45.0, seed=5)
    t = np.cumsum(rr)
    mod = rr * (1.0 + 0.20 * np.sin(2 * np.pi * 15.0 / 60.0 * t))
    ev = _evidence(mod, _resp_wave(45.0, 15.0))
    assert ev["evidence"] == "chaotic", ev
    assert ev["explained_fraction"] is not None and \
        ev["explained_fraction"] >= 0.5
    assert ev["residual_index_estimate"] >= 0.06
    ev = _evidence(rr, _resp_wave(45.0, 15.0))
    assert ev["evidence"] == "chaotic", ev


def test_evidence_detail_always_states_what_it_used():
    from scripts.make_synth_regularity import regular_rr
    ev = _evidence(regular_rr(45.0, rmssd_ms=8.0, seed=3),
                   _resp_wave(45.0, 15.0))
    for k in ("explained_fraction", "coupling_route", "phase_locking",
              "residual_index_estimate", "residual_threshold",
              "alternation_index", "outlier_topology"):
        assert k in ev, k
    assert ev["residual_threshold"] == 0.06
