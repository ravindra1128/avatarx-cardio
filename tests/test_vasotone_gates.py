"""v0.5 vasotone T5 — the §W gates: all red with no evidence, explicit
thresholds that bind, the config-driven optics-survival set (a feature
that responds to lamps is dropped automatically), fail-closed render
guard, a green+signed path that CAN open, promote refusal/success, and
the CLI track."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import os
import subprocess

import pytest

from evaluation.vasotone_gates import (TRACK_NOTE,
                                       evaluate_vasotone_gates,
                                       load_vasotone_gates,
                                       surviving_tone_features,
                                       vasotone_gate_status,
                                       vasotone_render_allowed)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GATES = _ROOT / "configs" / "gates.yaml"


def _signed_gates(tmp_path, edits=()):
    text = _GATES.read_text()
    # sign ONLY the vasotone block (its signoff line is unique via the
    # claim_scope + position after the vasotone banner)
    head, _, tail = text.partition("vasotone:")
    tail = tail.replace(
        "signoff: {owner_confirmed: false, clinical_advisor: null, "
        "date: null, claim_scope: null}",
        "signoff: {owner_confirmed: true, clinical_advisor: dr-tone, "
        "date: 2026-08-31, claim_scope: vasomotor_reactivity_trend}", 1)
    text = head + "vasotone:" + tail
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    p = tmp_path / "gates.yaml"
    p.write_text(text)
    return p


def _facial_data():
    return {"signal_domain": "facial_rppg", "participant_disjoint": True,
            "session_disjoint": True, "production_path": True}


def _w1_per_feature():
    return {"norm_pulse_amplitude": {"null_optics_p95": 0.04,
                                     "null_rest_p95": 0.03},
            "vasomotor_lf_power": {"null_optics_p95": 0.30,
                                   "null_rest_p95": 0.05},
            "notch_rel_amp": {"null_optics_p95": 0.03,
                              "null_rest_p95": 0.04},
            "reflection_index": {"null_optics_p95": None,
                                 "null_rest_p95": 0.02}}


def _green_evidence():
    return {
        "data": _facial_data(),
        "w0": {"direction_agreement": 0.9, "magnitude_r": 0.75,
               "n_provocations": 80},
        "w1": {"n_null_optics_sessions": 30, "n_null_rest_sessions": 30,
               "per_feature": _w1_per_feature()},
        "w2": {"added_r2_over_b3": 0.2, "added_r_ci95": [0.2, 0.6],
               "n_test_participants": 30},
        "w3": {"retest_icc": 0.7, "n_retest_pairs": 15,
               "ordered_fraction": 0.8},
        "w4": {"per_axis": {"fitzpatrick_group":
                            {"detection_ratio_worst": 0.85,
                             "coverage_ratio_worst": 0.9}}},
        "w5": {},
    }


def test_all_red_with_no_evidence(tmp_path):
    doc = vasotone_gate_status(runs_root=tmp_path)
    assert [g["gate"] for g in doc["gates"]] == \
        ["w0", "w1", "w2", "w3", "w4", "w5"]
    assert all(g["status"] == "RED" for g in doc["gates"])
    assert doc["promotion"] == "BLOCKED" and doc["note"] == TRACK_NOTE
    assert doc["surviving_features"] == []
    joined = json.dumps(doc["gates"])
    for phrase in ("no provocation study", "no null-arm study",
                   "no baseline-battery", "no retest/graded",
                   "no subgroup parity", "claim mapping"):
        assert phrase in joined, phrase


def test_green_signed_path_can_open(tmp_path):
    gcfg = load_vasotone_gates(_signed_gates(tmp_path))
    v = evaluate_vasotone_gates(gcfg, _green_evidence())
    assert all(g["status"] == "GREEN" for g in v["gates"]), v["gates"]
    assert v["clinical_signoff"] and v["promotion_open"]
    # unsigned repo contract: identical evidence stays closed
    v2 = evaluate_vasotone_gates(load_vasotone_gates(),
                                 _green_evidence())
    assert v2["promotion_open"] is False
    assert any("claim_scope" in r for g in v2["gates"]
               for r in g["reasons"])


def test_surrogate_domain_can_never_open_a_gate(tmp_path):
    gcfg = load_vasotone_gates(_signed_gates(tmp_path))
    ev = _green_evidence()
    ev["data"] = dict(ev["data"], signal_domain="synthetic")
    v = evaluate_vasotone_gates(gcfg, ev)
    assert all(g["status"] == "RED" for g in v["gates"])
    assert any("machinery evidence only" in r
               for g in v["gates"] for r in g["reasons"])


def test_thresholds_must_be_explicit_and_actually_bind(tmp_path):
    p = _signed_gates(tmp_path, edits=[
        ("    null_optics_abs_delta_norm_p95_max: 0.10\n", "")])
    with pytest.raises(ValueError, match="null_optics_abs"):
        evaluate_vasotone_gates(load_vasotone_gates(p),
                                _green_evidence())
    # tightening binds: the pivotal W2 flips red
    p2 = _signed_gates(tmp_path, edits=[
        ("added_r2_over_b3_min: 0.10", "added_r2_over_b3_min: 0.9")])
    v = evaluate_vasotone_gates(load_vasotone_gates(p2),
                                _green_evidence())
    w2 = next(g for g in v["gates"] if g["gate"] == "w2")
    assert w2["status"] == "RED" and v["promotion_open"] is False


def test_surviving_features_drop_optics_detectors_by_config(tmp_path):
    from evaluation.vasotone_gates import append_scoreboard
    runs = tmp_path / "runs"
    append_scoreboard(
        {"kind": "evaluation", "run_id": "vaso-x",
         "evidence": {"data": _facial_data(),
                      "w1": {"n_null_optics_sessions": 30,
                             "n_null_rest_sessions": 30,
                             "per_feature": _w1_per_feature()}}},
        runs_root=runs)
    surv = surviving_tone_features(runs_root=runs)
    # lf_power responds to lamps (0.30 > 0.10 cap); reflection_index has
    # no optics evidence -> both dropped, automatically
    assert surv == ["norm_pulse_amplitude", "notch_rel_amp"]
    # tightening the ratio drops the amplitude too (0.04 > 1.0x0.03)
    tighter = _signed_gates(tmp_path, edits=[
        ("null_rest_ratio_max: 2.0", "null_rest_ratio_max: 1.0")])
    assert surviving_tone_features(
        runs_root=runs, gates_path=tighter) == ["notch_rel_amp"]
    # no study on record / surrogate domain / underpowered -> nothing
    assert surviving_tone_features(runs_root=tmp_path / "none") == []
    runs2 = tmp_path / "runs2"
    append_scoreboard(
        {"kind": "evaluation", "run_id": "vaso-syn",
         "evidence": {"data": dict(_facial_data(),
                                   signal_domain="synthetic"),
                      "w1": {"n_null_optics_sessions": 30,
                             "n_null_rest_sessions": 30,
                             "per_feature": _w1_per_feature()}}},
        runs_root=runs2)
    assert surviving_tone_features(runs_root=runs2) == []
    runs3 = tmp_path / "runs3"
    append_scoreboard(
        {"kind": "evaluation", "run_id": "vaso-small",
         "evidence": {"data": _facial_data(),
                      "w1": {"n_null_optics_sessions": 3,
                             "n_null_rest_sessions": 30,
                             "per_feature": _w1_per_feature()}}},
        runs_root=runs3)
    assert surviving_tone_features(runs_root=runs3) == []


def test_render_guard_and_null_spellings(tmp_path):
    assert vasotone_render_allowed(runs_root=tmp_path) is False
    broken = tmp_path / "broken.yaml"
    broken.write_text("nonsense: [unclosed")
    assert vasotone_render_allowed(gates_path=broken) is False
    for spelling in ("None", "NULL", "~"):
        p = _signed_gates(tmp_path)
        q = tmp_path / f"g_{spelling.strip('~') or 'tilde'}.yaml"
        q.write_text(p.read_text().replace(
            "clinical_advisor: dr-tone",
            f"clinical_advisor: {spelling}"))
        v = evaluate_vasotone_gates(load_vasotone_gates(q),
                                    _green_evidence())
        assert v["clinical_signoff"] is False
        assert v["promotion_open"] is False


def test_promote_refuses_red_and_accepts_green_signed(tmp_path):
    from models.registry import promote, PromotionRefused
    run = tmp_path / "vaso-run"
    run.mkdir()
    (run / "vasotone_record.json").write_text(json.dumps(
        {"run_id": "vaso-test", "track": "vasomotor_reactivity"}))
    (run / "model.json").write_text(json.dumps({"kind": "reactivity"}))
    red = evaluate_vasotone_gates(load_vasotone_gates(), {})
    (run / "gate_results.json").write_text(json.dumps(red))
    with pytest.raises(PromotionRefused, match="§W"):
        promote(run, registry_path=tmp_path / "m.jsonl",
                spec_path=tmp_path / "s.md")
    # a hand-crafted partial gate list is refused too (completeness)
    partial = dict(red)
    partial["gates"] = [g for g in red["gates"] if g["gate"] != "w3"]
    (run / "gate_results.json").write_text(json.dumps(partial))
    with pytest.raises(PromotionRefused, match="w3 missing"):
        promote(run, registry_path=tmp_path / "m.jsonl",
                spec_path=tmp_path / "s.md")
    green = evaluate_vasotone_gates(
        load_vasotone_gates(_signed_gates(tmp_path)), _green_evidence())
    (run / "gate_results.json").write_text(json.dumps(green))
    (tmp_path / "s.md").write_text("# spec\n")
    row = promote(run, registry_path=tmp_path / "m.jsonl",
                  spec_path=tmp_path / "s.md")
    assert row["track"] == "vasomotor_reactivity"
    assert "Vasotone promotion" in (tmp_path / "s.md").read_text()


def test_gate_status_reads_latest_evaluation_entry(tmp_path):
    from evaluation.vasotone_gates import append_scoreboard
    runs = tmp_path / "runs"
    append_scoreboard({"kind": "evaluation", "run_id": "vaso-old",
                       "evidence": {}}, runs_root=runs)
    append_scoreboard({"kind": "evaluation", "run_id": "vaso-new",
                       "evidence": _green_evidence()}, runs_root=runs)
    doc = vasotone_gate_status(runs_root=runs,
                               gates_path=_signed_gates(tmp_path))
    assert doc["evaluation_run"] == "vaso-new"
    assert doc["promotion"] == "OPEN"
    assert doc["surviving_features"] == ["norm_pulse_amplitude",
                                        "notch_rel_amp"]


def test_cli_gate_status_track_vasotone(tmp_path):
    env = dict(os.environ, AVATARX_VASOTONE_RUNS=str(tmp_path))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "gate-status", "--track", "vasotone"],
                       capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert len(doc["gates"]) == 6 and doc["promotion"] == "BLOCKED"
    assert doc["note"] == TRACK_NOTE
