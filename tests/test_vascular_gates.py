"""v0.4-vascular T5 — the pre-registered vascular gates: all red with no
evidence, explicit thresholds that provably bind, config-driven feature
survival (never hand-edited), fail-closed render guard, a green+signed
path that CAN open, promote refusal/success, and the CLI track."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import os
import subprocess

import pytest

from evaluation.vascular_gates import (TRACK_NOTE, evaluate_vascular_gates,
                                       load_vascular_gates,
                                       surviving_features,
                                       vascular_gate_status,
                                       vascular_render_allowed)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GATES = _ROOT / "configs" / "gates.yaml"
_SIGNOFF = ("signoff: {owner_confirmed: false, clinical_advisor: null, "
            "date: null, claim_scope: null}")


def _signed_gates(tmp_path, edits=()):
    text = _GATES.read_text()
    text = text.replace(
        _SIGNOFF,
        "signoff: {owner_confirmed: true, clinical_advisor: dr-vasc, "
        "date: 2026-08-31, claim_scope: longitudinal_wellness_trend}")
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    p = tmp_path / "gates.yaml"
    p.write_text(text)
    return p


def _facial_data():
    return {"signal_domain": "facial_rppg", "participant_disjoint": True,
            "session_disjoint": True, "production_path": True}


def _green_evidence():
    per = {"reflection_index": {"icc": 0.90, "retest_icc": 0.84},
           "rise_time_s": {"icc": 0.81, "retest_icc": 0.77},
           "notch_rel_amp": {"icc": 0.40, "retest_icc": 0.80}}
    return {
        "data": _facial_data(), "fidelity_data": _facial_data(),
        "v0": {"n_paired_participants": 80, "per_feature": per},
        "v1": {"rmse_improvement_vs_b3_mps": 0.8,
               "rmse_improvement_ci95": [0.2, 1.4], "added_r2": 0.09,
               "n_test_participants": 55},
        "v2": {"estimate_retest_icc": 0.90, "retest_drift_mps": 0.1,
               "n_retest_pairs": 25},
        "v3": {"n_sites": 3, "worst_site_rmse_ratio": 1.1,
               "n_devices": 2, "worst_device_rmse_ratio": 1.15},
        "v4": {"per_axis": {
            "fitzpatrick_group": {"coverage_ratio_worst": 0.92,
                                  "rmse_ratio_worst": 1.1},
            "device": {"coverage_ratio_worst": 0.95,
                       "rmse_ratio_worst": 1.05}}},
        "v5": {},
    }


def test_all_red_with_no_evidence(tmp_path):
    doc = vascular_gate_status(runs_root=tmp_path)
    assert [g["gate"] for g in doc["gates"]] == \
        ["v0", "v1", "v2", "v3", "v4", "v5"]
    assert all(g["status"] == "RED" for g in doc["gates"])
    assert doc["promotion"] == "BLOCKED" and doc["note"] == TRACK_NOTE
    assert doc["surviving_features"] == []
    # every gate speaks its own missing-study reason
    joined = json.dumps(doc["gates"])
    for phrase in ("no signal-fidelity study", "no baseline-battery",
                   "no test-retest study", "no leave-one-site",
                   "no subgroup", "claim mapping"):
        assert phrase in joined, phrase


def test_green_signed_path_can_open(tmp_path):
    gcfg = load_vascular_gates(_signed_gates(tmp_path))
    v = evaluate_vascular_gates(gcfg, _green_evidence())
    assert all(g["status"] == "GREEN" for g in v["gates"]), v["gates"]
    assert v["clinical_signoff"] and v["promotion_open"]
    # unsigned repo contract: identical evidence stays closed
    v2 = evaluate_vascular_gates(load_vascular_gates(), _green_evidence())
    assert v2["promotion_open"] is False
    assert any("claim_scope" in r for g in v2["gates"]
               for r in g["reasons"])


def test_surrogate_domain_can_never_open_a_gate(tmp_path):
    gcfg = load_vascular_gates(_signed_gates(tmp_path))
    ev = _green_evidence()
    ev["data"] = dict(ev["data"], signal_domain="synthetic")
    ev["fidelity_data"] = dict(ev["fidelity_data"],
                               signal_domain="synthetic")
    v = evaluate_vascular_gates(gcfg, ev)
    assert all(g["status"] == "RED" for g in v["gates"])
    assert any("machinery evidence only" in r
               for g in v["gates"] for r in g["reasons"])


def test_thresholds_must_be_explicit_and_actually_bind(tmp_path):
    # deleting a threshold is a loud contract violation, not a default
    p = _signed_gates(tmp_path, edits=[
        ("    feature_icc_min: 0.75           # facial vs contact "
         "reference, per feature\n", "")])
    with pytest.raises(ValueError, match="feature_icc_min"):
        evaluate_vascular_gates(load_vascular_gates(p), _green_evidence())
    # tightening provably binds: the pivotal V1 flips red
    p2 = _signed_gates(tmp_path, edits=[
        ("rmse_improvement_vs_b3_mps_min: 0.5",
         "rmse_improvement_vs_b3_mps_min: 5.0")])
    v = evaluate_vascular_gates(load_vascular_gates(p2),
                                _green_evidence())
    v1 = next(g for g in v["gates"] if g["gate"] == "v1")
    assert v1["status"] == "RED"
    assert v["promotion_open"] is False


def test_surviving_features_are_config_driven_not_hand_edited(tmp_path):
    from evaluation.vascular_gates import append_scoreboard
    per = {"reflection_index": {"icc": 0.90, "retest_icc": 0.84},
           "rise_time_s": {"icc": 0.78, "retest_icc": 0.72},
           "notch_rel_amp": {"icc": 0.86, "retest_icc": 0.55},
           "sdppg_b_over_a": {"icc": 0.30, "retest_icc": 0.90},
           "pulse_width50_s": {"icc": None, "retest_icc": None}}
    append_scoreboard({"kind": "fidelity", "run_id": "fid-test",
                       "evidence": {"v0": {"per_feature": per,
                                           "n_paired_participants": 80},
                                    "data": _facial_data()}},
                      runs_root=tmp_path)
    surv = surviving_features(runs_root=tmp_path)
    # dropped: low ICC, low retest ICC, missing evidence — automatically
    assert surv == ["reflection_index", "rise_time_s"]
    # tightening the config drops more — no hand-edited list anywhere
    tighter = _signed_gates(tmp_path, edits=[
        ("feature_icc_min: 0.75", "feature_icc_min: 0.85")])
    assert surviving_features(runs_root=tmp_path,
                              gates_path=tighter) == ["reflection_index"]
    # no fidelity study on record -> nothing survives (fail closed)
    assert surviving_features(runs_root=tmp_path / "empty") == []
    # a SURROGATE-domain or underpowered study must not decide the
    # feature set for facial evaluations (review finding)
    synth = tmp_path / "synth_runs"
    append_scoreboard({"kind": "fidelity", "run_id": "fid-synth",
                       "evidence": {"v0": {"per_feature": per,
                                           "n_paired_participants": 80},
                                    "data": {"signal_domain": "synthetic",
                                             "participant_disjoint": True,
                                             "session_disjoint": True,
                                             "production_path": False}}},
                      runs_root=synth)
    assert surviving_features(runs_root=synth) == []
    small = tmp_path / "small_runs"
    append_scoreboard({"kind": "fidelity", "run_id": "fid-small",
                       "evidence": {"v0": {"per_feature": per,
                                           "n_paired_participants": 6},
                                    "data": _facial_data()}},
                      runs_root=small)
    assert surviving_features(runs_root=small) == []


def test_render_guard_fails_closed_on_any_error(tmp_path):
    assert vascular_render_allowed(runs_root=tmp_path) is False
    broken = tmp_path / "broken.yaml"
    broken.write_text("nonsense: [unclosed")
    assert vascular_render_allowed(gates_path=broken) is False
    assert vascular_render_allowed(
        gates_path=tmp_path / "missing.yaml") is False
    # a valid-JSON non-object scoreboard line fails cleanly, not with an
    # uncaught AttributeError (review finding)
    bad = tmp_path / "badrows"
    bad.mkdir()
    (bad / "scoreboard.jsonl").write_text('"just a string"\n[1,2]\n')
    assert vascular_render_allowed(runs_root=bad) is False
    assert surviving_features(runs_root=bad) == []
    with pytest.raises(ValueError, match="not a JSON object"):
        from evaluation.vascular_gates import latest_scoreboard_entry
        latest_scoreboard_entry("fidelity", runs_root=bad)


def test_null_spellings_never_count_as_signed(tmp_path):
    # 'None', 'NULL' and '~' are unset, not an advisor (review finding)
    for spelling in ("None", "NULL", "~"):
        p = _signed_gates(tmp_path)
        text = p.read_text().replace("clinical_advisor: dr-vasc",
                                     f"clinical_advisor: {spelling}")
        q = tmp_path / f"g_{spelling.strip('~') or 'tilde'}.yaml"
        q.write_text(text)
        v = evaluate_vascular_gates(load_vascular_gates(q),
                                    _green_evidence())
        assert v["clinical_signoff"] is False
        assert v["promotion_open"] is False


def test_promote_refuses_red_and_accepts_green_signed(tmp_path):
    from models.registry import promote, PromotionRefused
    run = tmp_path / "vasc-run"
    run.mkdir()
    (run / "vascular_record.json").write_text(json.dumps(
        {"run_id": "vasc-test", "track": "vascular_stiffness"}))
    (run / "model.json").write_text(json.dumps({"kind": "ridge"}))
    red = evaluate_vascular_gates(load_vascular_gates(), {})
    (run / "gate_results.json").write_text(json.dumps(red))
    with pytest.raises(PromotionRefused, match="vascular"):
        promote(run, registry_path=tmp_path / "m.jsonl",
                spec_path=tmp_path / "s.md")
    # green + signed promotes, appends registry row + spec line
    green = evaluate_vascular_gates(
        load_vascular_gates(_signed_gates(tmp_path)), _green_evidence())
    (run / "gate_results.json").write_text(json.dumps(green))
    (tmp_path / "s.md").write_text("# spec\n")
    row = promote(run, registry_path=tmp_path / "m.jsonl",
                  spec_path=tmp_path / "s.md")
    assert row["track"] == "vascular_stiffness"
    assert "Vascular promotion" in (tmp_path / "s.md").read_text()


def test_cli_gate_status_track_vascular(tmp_path):
    env = dict(os.environ, AVATARX_VASCULAR_RUNS=str(tmp_path))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                       "gate-status", "--track", "vascular"],
                      capture_output=True, text=True, timeout=120,
                      env=env)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert len(doc["gates"]) == 6 and doc["promotion"] == "BLOCKED"
    assert doc["note"] == TRACK_NOTE


def test_gate_status_merges_fidelity_and_evaluation_entries(tmp_path):
    """Review finding: the two-kind scoreboard merge had no direct
    coverage. Green merge, latest-of-each-kind, and cross-kind
    non-borrowing are the three contracts."""
    from evaluation.vascular_gates import append_scoreboard
    runs = tmp_path / "runs"
    ev = _green_evidence()
    append_scoreboard({"kind": "fidelity", "run_id": "fid-old",
                       "evidence": {"v0": {}, "data": _facial_data()}},
                      runs_root=runs)
    append_scoreboard({"kind": "fidelity", "run_id": "fid-new",
                       "evidence": {"v0": ev["v0"],
                                    "data": _facial_data()}},
                      runs_root=runs)
    append_scoreboard({"kind": "evaluation", "run_id": "ev-new",
                       "evidence": {"v1": ev["v1"], "v2": ev["v2"],
                                    "v3": ev["v3"], "v4": ev["v4"],
                                    "v5": {}, "data": _facial_data()}},
                      runs_root=runs)
    doc = vascular_gate_status(runs_root=runs,
                               gates_path=_signed_gates(tmp_path))
    # latest of each kind wins; the merged verdict is fully green+signed
    assert doc["fidelity_run"] == "fid-new"
    assert doc["evaluation_run"] == "ev-new"
    assert doc["promotion"] == "OPEN"
    assert doc["surviving_features"] == ["reflection_index",
                                        "rise_time_s"]
    # cross-kind non-borrowing: a synthetic fidelity study reddens V0
    # (and kills the survivors) while the evaluation side stays judged
    # on its own qualified evidence
    runs2 = tmp_path / "runs2"
    append_scoreboard({"kind": "fidelity", "run_id": "fid-syn",
                       "evidence": {"v0": ev["v0"],
                                    "data": dict(_facial_data(),
                                                 signal_domain="synthetic")}},
                      runs_root=runs2)
    append_scoreboard({"kind": "evaluation", "run_id": "ev-2",
                       "evidence": {"v1": ev["v1"], "v2": ev["v2"],
                                    "v3": ev["v3"], "v4": ev["v4"],
                                    "v5": {}, "data": _facial_data()}},
                      runs_root=runs2)
    doc2 = vascular_gate_status(runs_root=runs2,
                                gates_path=_signed_gates(tmp_path))
    by_gate = {g["gate"]: g for g in doc2["gates"]}
    assert by_gate["v0"]["status"] == "RED"
    assert by_gate["v1"]["status"] == "GREEN"
    assert doc2["promotion"] == "BLOCKED"
    assert doc2["surviving_features"] == []
