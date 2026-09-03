"""v0.4 T4 — §V gate machinery: pre-registered thresholds applied
honestly, surrogate evidence can never open a gate, signoff blocks
promotion even when all green, promote refuses head_fitness on any red,
and the INFERRED_FITNESS rendering invariant fails closed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import copy
import json
import os
import subprocess

import pytest

from evaluation.fitness_gates import (TRACK_NOTE, evaluate_vo2_gates,
                                      fitness_render_allowed,
                                      load_vo2_gates,
                                      render_vo2_status_html,
                                      vo2_gate_status)
from models.registry import promote, PromotionRefused

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _perfect_evidence():
    return {
        "data": {"signal_domain": "facial_rppg",
                 "participant_disjoint": True, "session_disjoint": True,
                 "production_path": True, "n_test_participants": 60},
        "v1": {"hrr60_loa_bpm_by_band": {"1-4": 3.8, "5-7": 4.2,
                                         "8-10": 4.6},
               "per_window_rmse_by_recovery_time": {"0-30": 2.1,
                                                    "30-60": 1.8}},
        "v2": {"hrr60_typical_error_bpm": 3.1, "accepted_scan_rate": 0.91,
               "abstention_by_band": {"1-4": 0.06, "5-7": 0.08,
                                      "8-10": 0.09}},
        "v3": {"delta_see_mlkgmin": 0.8, "delta_see_ci95": [0.2, 1.4],
               "category_accuracy_gain_pts": 7.0,
               "category_gain_ci95": [2.0, 12.0],
               "beats_static_face_control": True},
        "v4": {"subgroup_tables_present": True,
               "beta_blocker_bias_met": 0.6,
               "medicated_hard_routed_to_trend": True},
        "v5": {"change_score_r": 0.7, "auroc_1met": 0.8},
    }


def test_all_red_with_no_evidence():
    v = evaluate_vo2_gates(load_vo2_gates(), {})
    assert [g["gate"] for g in v["gates"]] == ["v1", "v2", "v3", "v4",
                                               "v5"]
    assert all(g["status"] == "RED" for g in v["gates"])
    assert not v["promotion_open"]
    assert any("no evaluation run on record" in r
               for g in v["gates"] for r in g["reasons"])


def test_surrogate_domain_can_never_open_a_gate():
    ev = _perfect_evidence()
    ev["data"]["signal_domain"] = "synthetic"
    v = evaluate_vo2_gates(load_vo2_gates(), ev)
    assert all(g["status"] == "RED" for g in v["gates"])
    assert any("facial rPPG" in r for r in v["gates"][0]["reasons"])


def test_signoff_blocks_promotion_even_when_all_green():
    gcfg = load_vo2_gates()
    v = evaluate_vo2_gates(gcfg, _perfect_evidence())
    assert all(g["status"] == "GREEN" for g in v["gates"]), \
        [(g["gate"], g["reasons"]) for g in v["gates"]]
    assert v["all_gates_green"] and not v["clinical_signoff"]
    assert not v["promotion_open"]
    signed = copy.deepcopy(gcfg)
    signed["signoff"] = {"owner_confirmed": True,
                         "clinical_advisor": "Dr. Y", "date": "2027-06-01"}
    assert evaluate_vo2_gates(signed, _perfect_evidence())["promotion_open"]


def test_each_gate_fails_on_its_own_criterion():
    gcfg = load_vo2_gates()
    for mutate, gate, needle in (
        (lambda e: e["v1"]["hrr60_loa_bpm_by_band"].update({"8-10": 6.4}),
         "v1", "8-10"),
        (lambda e: e["v1"]["hrr60_loa_bpm_by_band"].update({"8-10": 9.0}),
         "v1", "kill rule"),
        (lambda e: e["v2"].update(accepted_scan_rate=0.70), "v2",
         "accepted-scan rate"),
        (lambda e: e["v2"].update(abstention_by_band={"1-4": 0.05,
                                                      "8-10": 0.30}),
         "v2", "parity FAIL"),
        (lambda e: e["v3"].update(delta_see_mlkgmin=0.2), "v3",
         "adds too little"),
        (lambda e: e["v3"].update(delta_see_ci95=[-0.1, 1.2]), "v3",
         "exclude zero"),
        (lambda e: e["v3"].update(beats_static_face_control=False), "v3",
         "static-face"),
        (lambda e: e["v4"].update(medicated_hard_routed_to_trend=False,
                                  beta_blocker_bias_met=1.6), "v4",
         "hard-routed"),
        (lambda e: e["v5"].update(change_score_r=0.4), "v5",
         "change-score"),
    ):
        ev = _perfect_evidence()
        mutate(ev)
        v = evaluate_vo2_gates(gcfg, ev)
        row = next(g for g in v["gates"] if g["gate"] == gate)
        assert row["status"] == "RED", (gate, needle, row)
        assert any(needle in r for r in row["reasons"]), (gate, row)


def test_v2_parity_fails_closed_at_zero_abstention_best_band():
    """Review finding: `best > 0 and ...` disabled the parity gate when
    the best band abstained on nobody — exactly when a 30% band should
    scream."""
    ev = _perfect_evidence()
    ev["v2"]["abstention_by_band"] = {"1-4": 0.0, "5-7": 0.04,
                                      "8-10": 0.30}
    v = evaluate_vo2_gates(load_vo2_gates(), ev)
    row = next(g for g in v["gates"] if g["gate"] == "v2")
    assert row["status"] == "RED"
    assert any("parity FAIL" in r and "8-10" in r for r in row["reasons"])


def test_thresholds_must_be_explicit_and_actually_bind():
    """Review finding: every threshold read fell back to a coded default
    equal to today's yaml — the config could silently stop binding."""
    gcfg = copy.deepcopy(load_vo2_gates())
    del gcfg["v3_incremental_value"]["delta_see_mlkgmin_min"]
    with pytest.raises(ValueError, match="delta_see_mlkgmin_min"):
        evaluate_vo2_gates(gcfg, _perfect_evidence())
    tight = copy.deepcopy(load_vo2_gates())
    tight["v3_incremental_value"]["delta_see_mlkgmin_min"] = 2.0
    v = evaluate_vo2_gates(tight, _perfect_evidence())
    row = next(g for g in v["gates"] if g["gate"] == "v3")
    assert row["status"] == "RED"                # tightening binds


def test_promote_fitness_success_path_appends_registry_and_spec(tmp_path):
    """The green+signed path must actually work when its day comes."""
    from evaluation.fitness_gates import evaluate_vo2_gates as ev
    gcfg = copy.deepcopy(load_vo2_gates())
    gcfg["signoff"] = {"owner_confirmed": True,
                       "clinical_advisor": "Dr. Y", "date": "2027-06-01"}
    verdict = ev(gcfg, _perfect_evidence())
    assert verdict["promotion_open"]
    run = tmp_path / "fit-green"
    run.mkdir()
    (run / "fitness_record.json").write_text(json.dumps(
        {"run_id": "fit-green-001", "track": "vo2_fitness"}))
    (run / "model.json").write_text("{}")
    (run / "gate_results.json").write_text(json.dumps(
        {"run_id": "fit-green-001", **verdict}))
    spec = tmp_path / "spec.md"
    spec.write_text("# spec\n")
    row = promote(run, registry_path=tmp_path / "models.jsonl",
                  spec_path=spec)
    assert row["track"] == "vo2_fitness"
    assert all(g["status"] == "GREEN" for g in row["gates"])
    reg = (tmp_path / "models.jsonl").read_text()
    assert "vo2_fitness" in reg
    assert "Fitness promotion" in spec.read_text()


def test_v4_code_routing_substitutes_for_missing_bias_study():
    ev = _perfect_evidence()
    ev["v4"] = {"subgroup_tables_present": True,
                "beta_blocker_bias_met": None,
                "medicated_hard_routed_to_trend": True}
    v = evaluate_vo2_gates(load_vo2_gates(), ev)
    assert next(g for g in v["gates"]
                if g["gate"] == "v4")["status"] == "GREEN"


def test_gate_status_and_render_invariant_fail_closed(tmp_path):
    doc = vo2_gate_status(runs_root=tmp_path / "nowhere")
    assert doc["promotion"] == "BLOCKED" and doc["evidence_run"] is None
    assert doc["track"] == "vo2" and doc["note"] == TRACK_NOTE
    html = render_vo2_status_html(doc)
    assert html.count(TRACK_NOTE) >= 2 and "BLOCKED" in html
    assert fitness_render_allowed(runs_root=tmp_path / "nowhere") is False
    # broken gates file -> render stays closed, never raises
    bad = tmp_path / "broken.yaml"
    bad.write_text("nonsense: [unclosed")
    assert fitness_render_allowed(gates_path=bad) is False


def test_promote_refuses_fitness_while_v_gates_red(tmp_path):
    run = tmp_path / "fit-run"
    run.mkdir()
    (run / "fitness_record.json").write_text(json.dumps(
        {"run_id": "fit-000", "track": "vo2_fitness"}))
    (run / "model.json").write_text("{}")
    verdict = evaluate_vo2_gates(load_vo2_gates(), {})
    (run / "gate_results.json").write_text(json.dumps(
        {"run_id": "fit-000", **verdict}))
    with pytest.raises(PromotionRefused) as ei:
        promote(run, registry_path=tmp_path / "models.jsonl",
                spec_path=tmp_path / "spec.md")
    msg = str(ei.value)
    assert "§V" in msg and "clinical signoff" in msg
    assert msg.count("V") >= 5                    # every red gate listed
    assert not (tmp_path / "models.jsonl").exists()
    # and with no gate evaluation at all: refused, not trusted
    (run / "gate_results.json").unlink()
    with pytest.raises(PromotionRefused, match="no §V evaluation"):
        promote(run, registry_path=tmp_path / "models.jsonl",
                spec_path=tmp_path / "spec.md")


def test_cli_gate_status_track_vo2(tmp_path):
    env = dict(os.environ, AVATARX_FITNESS_RUNS=str(tmp_path / "runs"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "gate-status", "--track", "vo2",
                        "--html", str(tmp_path / "v.html")],
                       capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["track"] == "vo2" and doc["promotion"] == "BLOCKED"
    assert len(doc["gates"]) == 5
    assert TRACK_NOTE in (tmp_path / "v.html").read_text()
    # the default track is untouched §G
    r2 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "gate-status"], capture_output=True, text=True,
                        timeout=120)
    assert r2.returncode == 0
    assert json.loads(r2.stdout)["gates"][0]["gate"] == "g1"
