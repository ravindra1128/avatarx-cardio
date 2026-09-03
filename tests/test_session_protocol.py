"""v0.4 T3b — the session state machine end-to-end on T1 fixtures:
compliance contract, safety gating, vitals-still-render carve-out, and
the CLI surface. Forbidden-output check at the boundary: no VO2 number,
no fitness category, ever, in the session JSON while §V is red."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re
import subprocess

import pytest

cv2 = pytest.importorskip("cv2")

from datasets.schema import ScanOutcome
from protocol.challenges import get_challenge
from protocol.session import (TRANSITION_TIMEOUT_S, evaluate_compliance,
                              load_session_manifest, run_session)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------- compliance contract
def test_compliance_bands():
    sts = get_challenge("sts_1min")
    ok = evaluate_compliance(sts, reps=19, cadence_per_min=19.0,
                             transition_s=4.0)
    assert ok["verdict"] == "compliant" and not ok["reasons"]
    mid = evaluate_compliance(sts, reps=17, cadence_per_min=17.0,
                              transition_s=4.0)
    assert mid["verdict"] == "repeat"
    bad = evaluate_compliance(sts, reps=13, cadence_per_min=13.0,
                              transition_s=4.0)
    assert bad["verdict"] == "no_result"
    late = evaluate_compliance(sts, reps=20, cadence_per_min=20.0,
                               transition_s=TRANSITION_TIMEOUT_S + 2.0)
    assert late["verdict"] == "no_result"
    assert any("transition" in r for r in late["reasons"])
    unverified = evaluate_compliance(sts, reps=None, cadence_per_min=None,
                                     transition_s=3.0)
    assert unverified["verdict"] == "no_result"
    # an UNRECORDED transition fails closed too (review finding: the
    # None case used to skip the transition gate entirely)
    no_t = evaluate_compliance(sts, reps=20, cadence_per_min=20.0,
                               transition_s=None)
    assert no_t["verdict"] == "no_result"
    assert any("not recorded" in r for r in no_t["reasons"])


# ---------------------------------------------------- fixtures (shared)
@pytest.fixture(scope="module")
def fixture_sessions(tmp_path_factory):
    from scripts.make_synth_recovery import make_recovery_session
    d = tmp_path_factory.mktemp("sess")
    truths = {
        "clean": make_recovery_session(d, "clean", seed=5, rest_s=40.0,
                                       recovery_s=100.0),
        "slow": make_recovery_session(d, "slow", seed=6, rest_s=40.0,
                                      recovery_s=100.0,
                                      cadence_scale=0.7),
        "resting": make_recovery_session(d, "resting", seed=7,
                                         rest_s=40.0,
                                         include_activity=False),
    }
    return d, truths


def test_manifest_loader_fails_closed(fixture_sessions, tmp_path):
    d, _ = fixture_sessions
    m = load_session_manifest(d / "clean.session.json")
    assert m["protocol_id"] == "sts_1min"
    assert pathlib.Path(m["phases"]["rest"]["video"]).exists()
    bad = dict(json.loads((d / "clean.session.json").read_text()))
    bad["protocol_id"] = "sprint_40yd"
    p = tmp_path / "bad.session.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(Exception, match="unknown"):
        load_session_manifest(p)


def test_clean_session_accepts_with_measured_recovery(fixture_sessions,
                                                      tmp_path,
                                                      monkeypatch):
    d, truths = fixture_sessions
    store_path = tmp_path / "trend.json"
    monkeypatch.setenv("AVATARX_TREND_STORE", str(store_path))
    result, det = run_session(d / "clean.session.json")
    assert result.outcome is ScanOutcome.ACCEPT, result.no_read_reasons
    assert result.activity_performed and not result.safety_blocked
    assert result.compliance["verdict"] == "compliant"
    assert result.activity_tracker == "motion_energy"
    assert abs(result.hrr60_bpm - truths["clean"]["hrr60_true"]) <= 3.0
    assert result.hrr120_bpm is None                # 100 s recovery scan
    assert 60.0 < result.hr_rest_bpm < 90.0
    assert result.user_facing_text().startswith("Your session is complete")
    # never a fitness claim while §V is red — the heads run, classified,
    # but the INFERRED_FITNESS fields stay None
    assert result.fitness_category is None and result.trend is None
    assert [h["head"] for h in result.head_results] == \
        ["recovery", "fitness", "trend"]
    classes = {h["head"]: h["measurement_class"]
               for h in result.head_results}
    assert classes == {"recovery": "MEASURED",
                       "fitness": "INFERRED_FITNESS",
                       "trend": "INFERRED_FITNESS"}
    assert result.head_results[0]["value"]["hrr60_bpm"] == \
        result.hrr60_bpm
    assert det["phases_run"] == ["REST_SCAN", "GUIDED_ACTIVITY",
                                 "TRANSITION", "RECOVERY_SCAN", "GATES",
                                 "HEADS"]
    # the accepted session landed in the local trend store
    from trend.store import TrendStore
    rows = TrendStore(store_path).sessions()
    assert len(rows) == 1 and rows[0]["hrr60_bpm"] == result.hrr60_bpm


def test_noncompliant_pace_is_no_result_with_vitals(fixture_sessions):
    d, _ = fixture_sessions
    result, det = run_session(d / "slow.session.json")
    assert result.outcome is ScanOutcome.NO_RESULT
    assert result.compliance["verdict"] == "no_result"
    assert result.hrr60_bpm is None                 # non-comparable
    assert result.hr_rest_bpm is not None           # vitals still render
    assert "We couldn't verify the activity" in result.user_facing_text()


def test_resting_only_session(fixture_sessions):
    d, _ = fixture_sessions
    result, det = run_session(d / "resting.session.json")
    assert not result.activity_performed
    assert result.hr_rest_bpm is not None
    assert result.hrr60_bpm is None
    assert "require the short guided activity" in result.user_facing_text()


def test_safety_blocked_session(fixture_sessions, tmp_path):
    d, _ = fixture_sessions
    m = json.loads((d / "clean.session.json").read_text())
    m["safety_screen"]["answers"]["chest_pain_activity"] = True
    p = d / "blocked.session.json"                  # keep video paths
    p.write_text(json.dumps(m))
    result, det = run_session(p)
    assert result.safety_blocked
    assert not result.activity_performed
    assert "GUIDED_ACTIVITY" not in det["phases_run"]
    assert result.hr_rest_bpm is not None           # resting scan ran
    assert "skip the activity portion" in result.user_facing_text()


def test_cli_session_json_and_forbidden_outputs(fixture_sessions):
    d, _ = fixture_sessions
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"), "session",
                        "--manifest", str(d / "clean.session.json")],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)                      # stdout is pure JSON
    assert doc["schema_version"] == 1
    assert doc["outcome"] == "ACCEPT"
    assert doc["user_facing_text"]
    assert doc["fitness_category"] is None and doc["trend"] is None
    # forbidden outputs at the boundary: no VO2 number, no category word
    low = r.stdout.lower()
    assert not re.search(r"m[ll]\s*/\s*kg\s*/\s*min", low)
    assert "vo2" not in low and "vo₂" not in low
    for word in ("below average", "above average", "excellent fitness",
                 "poor fitness"):
        assert word not in low
