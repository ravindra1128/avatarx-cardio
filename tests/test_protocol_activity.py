"""v0.4 T3 (part 1) — challenges, safety screen, workload context, and
the cadence counter (acceptance: within ±1 rep on scripted fixtures;
workload never touches imagery; tracker recorded in provenance)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import re

import pytest

from activity.workload import workload_context
from protocol.challenges import (CHALLENGES, ChallengeError,
                                 DEFAULT_CHALLENGE, get_challenge)
from protocol.safety import (NON_BLOCKING, SCREEN_QUESTIONS, STOP_RULES,
                             challenge_allowed, evaluate_screen)


def _answers(**over):
    a = {q: False for q in SCREEN_QUESTIONS}
    a.update(over)
    return a


def test_challenge_registry_prescriptions():
    assert DEFAULT_CHALLENGE == "sts_1min"
    assert set(CHALLENGES) == {"sts_1min", "step_3min", "march_2min"}
    sts = get_challenge("sts_1min")
    assert sts.cadence_per_min == 20.0 and sts.duration_s == 60.0
    assert "measured_weight_kg" in sts.workload_inputs
    assert get_challenge("step_3min").step_height_m == 0.20
    with pytest.raises(ChallengeError, match="unknown"):
        get_challenge("sprint_40yd")
    for c in CHALLENGES.values():
        assert set(c.contraindication_ids) <= set(SCREEN_QUESTIONS)


def test_safety_screen_fails_closed():
    ok = evaluate_screen(_answers())
    assert ok.passed and not ok.blockers and not ok.meds_flagged
    blocked = evaluate_screen(_answers(chest_pain_activity=True))
    assert not blocked.passed and blocked.blockers == \
        ("chest_pain_activity",)
    # unanswered questions BLOCK (fail-closed), meds flag routes not blocks
    partial = evaluate_screen({"chest_pain_activity": False})
    assert not partial.passed and len(partial.unanswered) == 6
    meds = evaluate_screen(_answers(bp_heart_meds=True))
    assert meds.passed and meds.meds_flagged
    assert "bp_heart_meds" in NON_BLOCKING
    with pytest.raises(ValueError, match="unknown"):
        evaluate_screen({"favourite_colour": True})
    joint = evaluate_screen(_answers(bone_joint_problem=True))
    assert not challenge_allowed(joint, get_challenge("sts_1min"))


def test_safety_text_is_fixed_and_never_diagnoses():
    for s in list(SCREEN_QUESTIONS.values()) + list(STOP_RULES):
        low = s.lower()
        assert "atrial" not in low and "fibrillation" not in low
        assert not re.search(r"\byou have\b(?!.*\?)", low)  # no assertions
        assert "?" in s or s in STOP_RULES


def test_workload_context_from_user_inputs_only():
    sts = get_challenge("sts_1min")
    ctx = {"measured_weight_kg": 80.0, "height_cm": 175.0,
           "age": 44, "sex": "other"}
    w = workload_context(sts, ctx)
    assert w["available"] and "never" in w["source"]        # provenance
    # m*g*0.27*h = 80*9.81*0.4725 ~= 370.8 J; 20 reps / 60 s
    assert abs(w["work_per_rep_j"] - 370.8) < 1.0
    assert abs(w["avg_power_w"] - 123.6) < 0.5
    assert 5.0 < w["est_mets"] < 8.0
    missing = workload_context(sts, {"age": 44, "sex": "f"})
    assert missing["available"] is False
    assert any("measured_weight_kg" in r for r in missing["reasons"])
    march = workload_context(get_challenge("march_2min"),
                             {"measured_weight_kg": 70.0, "age": 30,
                              "sex": "f"})
    assert march["est_mets"] == 4.0 and march["avg_power_w"] is None


@pytest.fixture(scope="module")
def activity_clips(tmp_path_factory):
    cv2 = pytest.importorskip("cv2")
    from scripts.make_synth_recovery import activity_video
    d = tmp_path_factory.mktemp("act")
    truths = {}
    truths["on_pace"] = (str(d / "on.avi"), activity_video(
        str(d / "on.avi"), duration_s=60.0, cadence_per_min=20.0, seed=2))
    truths["slow"] = (str(d / "slow.avi"), activity_video(
        str(d / "slow.avi"), duration_s=60.0, cadence_per_min=14.0,
        seed=3))
    return truths


def test_cadence_counter_within_one_rep(activity_clips):
    from activity.pose_cadence import count_reps
    for name, (path, truth) in activity_clips.items():
        out = count_reps(path)
        assert out["tracker"] == "motion_energy"      # no pose model env
        assert out["reps"] is not None, (name, out)
        assert abs(out["reps"] - truth["reps"]) <= 1, (name, out, truth)
        assert abs(out["cadence_per_min"]
                   - truth["cadence_per_min"]) <= 2.0, (name, out)
        assert out["confidence"] > 0.3


def test_cadence_counter_survives_small_amplitude_marching(tmp_path):
    """Review finding: zero-padded 'same' detrending created edge ramps
    ~baseline/2 that buried small rep amplitudes under the 0.3*std peak
    threshold — a 4 px march (76:1 baseline/amplitude) counted 2 of 20."""
    cv2 = pytest.importorskip("cv2")
    from scripts.make_synth_recovery import activity_video
    from activity.pose_cadence import count_reps
    p = str(tmp_path / "small.avi")
    truth = activity_video(p, duration_s=60.0, cadence_per_min=20.0,
                           bob_px=4.0, seed=6)
    out = count_reps(p)
    assert out["reps"] is not None
    assert abs(out["reps"] - truth["reps"]) <= 1, out


def test_cadence_counter_fails_closed_on_short_clip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from scripts.make_synth_recovery import activity_video
    from activity.pose_cadence import count_reps
    p = str(tmp_path / "short.avi")
    activity_video(p, duration_s=6.0, cadence_per_min=20.0, seed=4)
    out = count_reps(p)
    assert out["reps"] is None and "short" in out["note"]
