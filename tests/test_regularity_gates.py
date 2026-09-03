"""v0.7 T6 — the §R gates: every threshold pre-registered in
configs/gates.yaml, every gate red until its evidence is on record,
numbers necessary and never sufficient, promotion refused at the door
with every reason listed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import pytest

from evaluation import regularity_gates as rg
from evaluation.regularity_gates import (evaluate_regularity_gates,
                                         regularity_gate_status,
                                         regularity_render_allowed,
                                         GATE_TITLES,
                                         load_regularity_gates,
                                         render_regularity_status_html)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _green_evidence():
    return {
        "data": {"signal_domain": "facial_rppg",
                 "participant_disjoint": True, "session_disjoint": True,
                 "production_path": True},
        "r0": {"kappa": 0.91, "worst_grade_kappa": 0.84, "index_r": 0.93,
               "loa_halfwidth": 0.021, "bias": 0.004, "n_paired_scans": 140},
        "r2": {"rsa_flag_rate": 0.04, "n_rsa_sessions": 48,
               "rsa_no_read_rate": 0.08,
               "age_strata": {"<35": {"n": 40, "specificity": 0.95,
                                      "no_read_rate": 0.1},
                              "35-59": {"n": 50, "specificity": 0.97,
                                        "no_read_rate": 0.05},
                              ">=60": {"n": 30, "specificity": 0.96,
                                       "no_read_rate": 0.12}}},
        "r3": {"head_balanced_accuracy": 0.93,
               "b3_balanced_accuracy": 0.89, "b4_balanced_accuracy": 0.60,
               "b5_balanced_accuracy": 0.55,
               "b3_balanced_accuracy_on_judged": 0.90,
               "b4_balanced_accuracy_on_judged": 0.62,
               "b5_balanced_accuracy_on_judged": 0.55,
               "head_no_read_rate": 0.08,
               "n_test_participants": 33, "n_test_participants_judged": 31},
        "r4": {"fitzpatrick_group": {"worst_index_bias": -0.004,
                                     "flag_rate_parity_ratio_worst": 0.82,
                                     "coverage_ratio_worst": 0.88,
                                     "darkest_band_present": True}},
        "r5": {"escalation_path_present": False},
    }


def _r2(*, rsa_flag_rate=0.04, n_rsa=48, rsa_no_read=0.08, young_n=40,
        young_spec=0.95, young_no_read=0.1):
    return {"rsa_flag_rate": rsa_flag_rate, "n_rsa_sessions": n_rsa,
            "rsa_no_read_rate": rsa_no_read,
            "age_strata": {"<35": {"n": young_n, "specificity": young_spec,
                                   "no_read_rate": young_no_read},
                           "35-59": {"n": 50, "specificity": 0.97,
                                     "no_read_rate": 0.05},
                           ">=60": {"n": 30, "specificity": 0.96,
                                    "no_read_rate": 0.12}}}


def _r3(*, head=0.93, b3=0.90, b4=0.62, b5=0.55, no_read=0.08,
        n_judged=31):
    return {"head_balanced_accuracy": head,
            "b3_balanced_accuracy": b3 - 0.01,
            "b4_balanced_accuracy": b4 - 0.02,
            "b5_balanced_accuracy": b5,
            "b3_balanced_accuracy_on_judged": b3,
            "b4_balanced_accuracy_on_judged": b4,
            "b5_balanced_accuracy_on_judged": b5,
            "head_no_read_rate": no_read,
            "n_test_participants": n_judged + 2,
            "n_test_participants_judged": n_judged}


def _green_floor():
    return {"parameters": {"mdi_power": 0.80, "mdi_confidence": 0.95},
            "signal_domain": "facial_rppg",
            "mdi_cells": {"fps=30.0|sqi=A|fitz=3": 6.0,
                          "fps=60.0|sqi=A|fitz=5": 4.0},
            "interpolation_gain": {"30.0": {"n_metronomic_refs": 12},
                                   "60.0": {"n_metronomic_refs": 11}},
            "false_irregular_at_5pct": {"30.0": 0.0, "60.0": 0.0}}


@pytest.fixture
def mdi_doc(tmp_path, monkeypatch):
    """R1 reads docs/regularity_track.md for the published MDI table;
    point the module at a repo copy so the test does not depend on the
    doc's current wording."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "regularity_track.md").write_text("## MDI table\n")
    monkeypatch.setattr(rg, "_REPO", tmp_path)
    return tmp_path


def _gcfg():
    return load_regularity_gates()


# ------------------------------------------------------------ the block
def test_block_is_pre_registered_with_every_threshold_present():
    g = _gcfg()
    assert g["gates_version"].startswith("R-")
    assert g["requires_clinical_signoff"] is True
    assert g["evidence_requirements"]["signal_domain"] == "facial_rppg"
    for key in ("reference_label", "r0_ceiling", "r1_noise_floor",
                "r2_benign_separation", "r3_baselines", "r4_fairness",
                "r5_claim_mapping"):
        assert isinstance(g.get(key), dict), key
        assert g[key].get("requires_clinical_signoff") is True, key
    assert g["reference_label"]["index"] == "irregularity_index"
    assert g["reference_label"]["irregular_if_index_at_least"] == 0.06
    assert g["r5_claim_mapping"]["permissible_first_claim"] == \
        "irregular_rhythm_notification"
    assert g["r5_claim_mapping"]["this_head_never_escalates"] is True
    assert g["signoff"]["owner_confirmed"] is False


def test_all_green_evidence_still_blocks_without_a_signature(mdi_doc):
    v = evaluate_regularity_gates(_gcfg(), _green_evidence(),
                                  _green_floor())
    reds = {g["gate"] for g in v["gates"] if g["status"] == "RED"}
    assert reds == {"r5"}, [(g["gate"], g["reasons"]) for g in v["gates"]]
    assert v["clinical_signoff"] is False and v["promotion_open"] is False
    r5 = [g for g in v["gates"] if g["gate"] == "r5"][0]
    assert any("owner + clinical advisor" in r for r in r5["reasons"])
    assert any("claim_scope" in r for r in r5["reasons"])


def test_signed_and_green_opens_promotion(mdi_doc):
    g = _gcfg()
    g["signoff"] = {"owner_confirmed": True, "clinical_advisor": "Dr X",
                    "date": "2026-09-01",
                    "claim_scope": "irregular_rhythm_notification"}
    v = evaluate_regularity_gates(g, _green_evidence(), _green_floor())
    assert v["all_gates_green"] is True and v["promotion_open"] is True
    for bad in (None, "None", "null", "~", "", "yes", 1):
        g["signoff"]["owner_confirmed"] = bad
        assert evaluate_regularity_gates(
            g, _green_evidence(), _green_floor())["promotion_open"] is \
            False, bad
    # the wrong claim scope closes it too
    g["signoff"]["owner_confirmed"] = True
    g["signoff"]["claim_scope"] = "afib_detection"
    v = evaluate_regularity_gates(g, _green_evidence(), _green_floor())
    assert v["promotion_open"] is False


def test_missing_evidence_reds_every_gate_with_its_own_reason():
    v = evaluate_regularity_gates(_gcfg(), {}, {})
    assert len(v["gates"]) == 6
    assert {g["gate"] for g in v["gates"]} == set(GATE_TITLES)
    for g in v["gates"]:
        assert g["status"] == "RED", g["gate"]
        assert g["reasons"], g["gate"]
    assert v["promotion_open"] is False


def test_surrogate_domain_can_never_open_a_gate(mdi_doc):
    for dom in ("synthetic", "public_ppg"):
        ev = _green_evidence()
        ev["data"]["signal_domain"] = dom
        v = evaluate_regularity_gates(_gcfg(), ev, _green_floor())
        assert all(g["status"] == "RED" for g in v["gates"])
        assert all(any("machinery evidence only" in r
                       for r in g["reasons"]) for g in v["gates"])


@pytest.mark.parametrize("patch,gate,needle", [
    ({"r0": {"kappa": 0.79, "worst_grade_kappa": 0.84, "index_r": 0.93,
             "loa_halfwidth": 0.02, "bias": 0.0, "n_paired_scans": 140}}, "r0",
     "kappa 0.790"),
    ({"r0": {"kappa": 0.91, "worst_grade_kappa": 0.70, "index_r": 0.93,
             "loa_halfwidth": 0.02, "bias": 0.0, "n_paired_scans": 140}}, "r0",
     "worst SQI-grade kappa"),
    ({"r0": {"kappa": 0.91, "worst_grade_kappa": 0.84, "index_r": 0.80,
             "loa_halfwidth": 0.02, "bias": 0.0, "n_paired_scans": 140}}, "r0",
     "index correlation"),
    ({"r0": {"kappa": 0.91, "worst_grade_kappa": 0.84, "index_r": 0.93,
             "loa_halfwidth": 0.05, "bias": 0.0, "n_paired_scans": 140}}, "r0",
     "Bland-Altman"),
    ({"r0": {"kappa": 0.91, "worst_grade_kappa": 0.84, "index_r": 0.93,
             "loa_halfwidth": 0.02, "bias": 0.0, "n_paired_scans": 99}}, "r0",
     "paired camera+ECG scans"),
    ({"r0": {"kappa": 0.91, "worst_grade_kappa": 0.84, "index_r": 0.93,
             "loa_halfwidth": 0.01, "bias": 0.05, "n_paired_scans": 140}},
     "r0", "index bias +0.0500"),
    ({"r0": {"kappa": 0.91, "worst_grade_kappa": 0.84, "index_r": 0.93,
             "loa_halfwidth": 0.01, "n_paired_scans": 140}},
     "r0", "no Bland-Altman bias"),
    ({"r2": _r2(rsa_flag_rate=0.2)}, "r2", "RSA-dominant"),
    ({"r2": _r2(young_spec=0.80)}, "r2", "age band <35: specificity 0.800"),
    ({"r2": _r2(young_n=4)}, "r2", "age band <35: 4 judged sessions"),
    ({"r2": _r2(n_rsa=5)}, "r2", "judged RSA sessions (< 20)"),
    ({"r2": _r2(rsa_no_read=0.5)}, "r2", "declined 0.500 of RSA"),
    ({"r2": _r2(young_no_read=0.6)}, "r2",
     "age band <35: head declined 0.600"),
    ({"r3": _r3(b3=0.92)}, "r3", "does not beat B3"),
    ({"r3": _r3(b4=0.90)}, "r3", "does not beat B4"),
    ({"r3": _r3(n_judged=12)}, "r3", "held-out participants judged"),
    ({"r3": _r3(no_read=0.4)}, "r3", "abstention is not accuracy"),
    ({"r4": {"fitzpatrick_group": {"worst_index_bias": 0.02,
                                   "flag_rate_parity_ratio_worst": 0.82,
                                   "coverage_ratio_worst": 0.88,
                                   "darkest_band_present": True}}}, "r4",
     "index bias"),
    ({"r4": {"fitzpatrick_group": {"worst_index_bias": 0.0,
                                   "flag_rate_parity_ratio_worst": 0.5,
                                   "coverage_ratio_worst": 0.88,
                                   "darkest_band_present": True}}}, "r4",
     "flag-rate parity"),
    ({"r4": {"fitzpatrick_group": {"worst_index_bias": 0.0,
                                   "flag_rate_parity_ratio_worst": 0.82,
                                   "coverage_ratio_worst": 0.88,
                                   "darkest_band_present": False}}}, "r4",
     "darkest skin-tone band"),
    ({"r5": {"escalation_path_present": True}}, "r5",
     "escalation is head_afib's job"),
])
def test_each_numeric_criterion_reds_its_own_gate(patch, gate, needle,
                                                  mdi_doc):
    ev = _green_evidence()
    ev.update(patch)
    v = evaluate_regularity_gates(_gcfg(), ev, _green_floor())
    row = [g for g in v["gates"] if g["gate"] == gate][0]
    assert row["status"] == "RED"
    assert any(needle in r for r in row["reasons"]), row["reasons"]
    others = [g for g in v["gates"] if g["gate"] not in (gate, "r5")]
    assert all(g["status"] == "GREEN" for g in others), \
        [(g["gate"], g["reasons"]) for g in others]


@pytest.mark.parametrize("patch,needle", [
    ({"mdi_cells": {}}, "no MDI cells"),
    ({"mdi_cells": {"fps=30.0|sqi=A|fitz=3": None}}, "not characterized"),
    ({"mdi_cells": {"fps=30.0|sqi=A": 5.0}}, "lacks a required axis"),
    ({"interpolation_gain": {}}, "interpolation gain is unmeasured"),
    ({"interpolation_gain": {"30.0": {"n_metronomic_refs": 3}}},
     "fewer than 10 metronomic references"),
    ({"false_irregular_at_5pct": {}}, "no beat-error study"),
    ({"false_irregular_at_5pct": {"30.0": 0.2}}, "exceeds 0.05"),
])
def test_r1_reads_the_floor_run_and_reds_each_defect(patch, needle,
                                                     mdi_doc):
    fl = _green_floor()
    fl.update(patch)
    v = evaluate_regularity_gates(_gcfg(), _green_evidence(), fl)
    r1 = [g for g in v["gates"] if g["gate"] == "r1"][0]
    assert r1["status"] == "RED"
    assert any(needle in r for r in r1["reasons"]), r1["reasons"]


def test_r1_requires_the_mdi_table_to_be_published(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(rg, "_REPO", tmp_path)
    v = evaluate_regularity_gates(_gcfg(), _green_evidence(),
                                  _green_floor())
    r1 = [g for g in v["gates"] if g["gate"] == "r1"][0]
    assert r1["status"] == "RED"
    assert any("does not publish the MDI table" in r for r in r1["reasons"])


def test_a_threshold_missing_from_the_yaml_fails_loudly(mdi_doc):
    g = _gcfg()
    del g["r2_benign_separation"]["max_rsa_flag_rate"]
    with pytest.raises(ValueError, match="max_rsa_flag_rate"):
        evaluate_regularity_gates(g, _green_evidence(), _green_floor())


def test_status_and_render_are_fail_closed(tmp_path):
    doc = regularity_gate_status(runs_root=tmp_path)
    assert doc["track"] == "regularity"
    assert doc["promotion"] == "BLOCKED"
    assert len(doc["gates"]) == 6
    assert doc["evaluation_run"] is None and doc["floor_run"] is None
    assert regularity_render_allowed(runs_root=tmp_path) is False
    html = render_regularity_status_html(doc)
    assert "BLOCKED" in html and "NOT VALIDATED" in html
    bad = tmp_path / "bad.yaml"
    bad.write_text("regularity: not-a-block\n")
    assert regularity_render_allowed(runs_root=tmp_path,
                                     gates_path=bad) is False
    assert regularity_render_allowed(runs_root=tmp_path,
                                     gates_path=tmp_path / "x.yaml") is False


def test_promote_refuses_and_lists_every_red_reason(tmp_path):
    from models.registry import promote, PromotionRefused
    run = tmp_path / "reg-1"
    run.mkdir()
    v = evaluate_regularity_gates(_gcfg(), {}, {})
    (run / "gate_results.json").write_text(json.dumps(v))
    (run / "regularity_record.json").write_text(json.dumps(
        {"run_id": "reg-1"}))
    (run / "model.json").write_text("{}")
    with pytest.raises(PromotionRefused) as e:
        promote(run, registry_path=tmp_path / "reg.jsonl",
                spec_path=tmp_path / "spec.md")
    msg = str(e.value)
    assert "§R" in msg
    for gate in GATE_TITLES.values():
        assert gate in msg, gate
    assert "clinical signoff pending" in msg
    assert not (tmp_path / "reg.jsonl").exists()
    # a hand-crafted file listing only green gates is refused too
    (run / "gate_results.json").write_text(json.dumps(
        {"gates": [{"gate": "r0", "status": "GREEN", "title": "R0"}],
         "all_gates_green": True, "clinical_signoff": True,
         "promotion_open": True}))
    with pytest.raises(PromotionRefused) as e2:
        promote(run, registry_path=tmp_path / "reg.jsonl",
                spec_path=tmp_path / "spec.md")
    assert "r1" in str(e2.value) and "r5" in str(e2.value)


def test_promote_records_the_track_when_everything_stands(tmp_path,
                                                          mdi_doc,
                                                          monkeypatch):
    """The door opens only when the recorded evidence ALSO opens under
    the live, signed gates.yaml — simulated here with a signed copy."""
    from models.registry import promote
    signed = (_ROOT / "configs" / "gates.yaml").read_text().replace(
        "signoff: {owner_confirmed: false, clinical_advisor: null, "
        "date: null, claim_scope: null}\n\n  evidence_requirements:\n"
        "    participant_disjoint: true\n    session_disjoint: true\n"
        "    production_path: true\n    signal_domain: facial_rppg\n\n"
        "  reference_label:",
        "signoff: {owner_confirmed: true, clinical_advisor: Dr X, "
        "date: 2026-09-01, claim_scope: irregular_rhythm_notification}"
        "\n\n  evidence_requirements:\n"
        "    participant_disjoint: true\n    session_disjoint: true\n"
        "    production_path: true\n    signal_domain: facial_rppg\n\n"
        "  reference_label:")
    assert "owner_confirmed: true" in signed, "signoff line not found"
    gp = tmp_path / "gates_signed.yaml"
    gp.write_text(signed)
    monkeypatch.setattr(rg, "DEFAULT_GATES", gp)
    g = load_regularity_gates(gp)
    assert g["signoff"]["owner_confirmed"] is True
    v = evaluate_regularity_gates(g, _green_evidence(), _green_floor())
    assert v["promotion_open"] is True
    run = tmp_path / "reg-2"
    run.mkdir()
    (run / "gate_results.json").write_text(json.dumps(v))
    (run / "evidence.json").write_text(json.dumps(
        {"evidence": _green_evidence(), "floor": _green_floor(),
         "gates_version": g["gates_version"]}))
    (run / "regularity_record.json").write_text(json.dumps(
        {"run_id": "reg-2"}))
    (run / "model.json").write_text("{}")
    spec = tmp_path / "spec.md"
    spec.write_text("# spec\n")
    row = promote(run, registry_path=tmp_path / "reg.jsonl", spec_path=spec)
    assert row["track"] == "pulse_regularity"
    assert {x["gate"] for x in row["gates"]} == set(GATE_TITLES)
    assert "Regularity-track promotion" in spec.read_text()


def test_no_gate_results_file_is_refused(tmp_path):
    from models.registry import promote, PromotionRefused
    run = tmp_path / "reg-3"
    run.mkdir()
    (run / "regularity_record.json").write_text(json.dumps(
        {"run_id": "reg-3"}))
    with pytest.raises(PromotionRefused, match="no §R evaluation"):
        promote(run, registry_path=tmp_path / "reg.jsonl",
                spec_path=tmp_path / "spec.md")


def test_cli_gate_status_regularity_track(tmp_path):
    import os
    import subprocess
    env = dict(os.environ, AVATARX_REGULARITY_RUNS=str(tmp_path / "runs"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "gate-status", "--track", "regularity",
                        "--html", str(tmp_path / "s.html")],
                       capture_output=True, text=True, timeout=300,
                       env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    doc = json.loads(r.stdout)
    assert doc["track"] == "regularity"
    assert doc["promotion"] == "BLOCKED"
    assert len(doc["gates"]) == 6
    assert (tmp_path / "s.html").exists()


def test_r1_checks_the_floor_was_computed_at_the_published_parameters(
        mdi_doc):
    fl = _green_floor()
    fl["parameters"] = {"mdi_power": 0.5, "mdi_confidence": 0.95}
    v = evaluate_regularity_gates(_gcfg(), _green_evidence(), fl)
    r1 = [g for g in v["gates"] if g["gate"] == "r1"][0]
    assert any("gates say" in r for r in r1["reasons"]), r1["reasons"]
    fl = _green_floor()
    fl.pop("parameters", None)
    v = evaluate_regularity_gates(_gcfg(), _green_evidence(), fl)
    r1 = [g for g in v["gates"] if g["gate"] == "r1"][0]
    assert any("does not record the MDI power" in r for r in r1["reasons"])
    # a floor from another domain than the evaluation is refused
    fl = _green_floor()
    fl["signal_domain"] = "synthetic"
    v = evaluate_regularity_gates(_gcfg(), _green_evidence(), fl)
    r1 = [g for g in v["gates"] if g["gate"] == "r1"][0]
    assert any("one cohort, one floor" in r for r in r1["reasons"])


def test_gate_status_reds_everything_on_gates_version_drift(tmp_path,
                                                            mdi_doc):
    """Review finding: evidence recorded under one gates version opened
    gates under another with a moved threshold."""
    from evaluation.regularity_gates import append_scoreboard
    g = _gcfg()
    append_scoreboard({"kind": "floor", "run_id": "f1",
                       "gates_version": g["gates_version"],
                       **_green_floor()}, runs_root=tmp_path)
    append_scoreboard({"kind": "evaluation", "run_id": "e1",
                       "gates_version": "R-1999-01-01-9",
                       "reference_label": g["reference_label"],
                       "evidence": _green_evidence()}, runs_root=tmp_path)
    doc = regularity_gate_status(runs_root=tmp_path)
    assert doc["promotion"] == "BLOCKED"
    assert all(any("stale evidence" in r for r in gt["reasons"])
               for gt in doc["gates"])


def test_numbers_that_are_not_measurements_never_pass(mdi_doc):
    """Review finding: -Infinity beat every margin, True read as a
    kappa of 1.0, an empty pre-registered list made a gate vacuous."""
    ev = _green_evidence()
    ev["r3"]["b3_balanced_accuracy_on_judged"] = float("-inf")
    v = evaluate_regularity_gates(_gcfg(), ev, _green_floor())
    r3 = [g for g in v["gates"] if g["gate"] == "r3"][0]
    assert r3["status"] == "RED"
    ev = _green_evidence()
    ev["r0"]["worst_grade_kappa"] = True
    v = evaluate_regularity_gates(_gcfg(), ev, _green_floor())
    assert [g for g in v["gates"] if g["gate"] == "r0"][0]["status"] == "RED"
    ev = _green_evidence()
    ev["data"]["production_path"] = "false"
    v = evaluate_regularity_gates(_gcfg(), ev, _green_floor())
    assert all(g["status"] == "RED" for g in v["gates"])
    ev = _green_evidence()
    ev["r4"]["fitzpatrick_group"]["darkest_band_present"] = "false"
    v = evaluate_regularity_gates(_gcfg(), ev, _green_floor())
    assert [g for g in v["gates"] if g["gate"] == "r4"][0]["status"] == "RED"
    g = _gcfg()
    g["r4_fairness"]["subgroups"] = []
    with pytest.raises(ValueError, match="non-empty list"):
        evaluate_regularity_gates(g, _green_evidence(), _green_floor())


def test_promote_re_evaluates_the_evidence_under_the_live_gates(tmp_path,
                                                                 mdi_doc):
    """Review finding: a forged gate_results.json promoted while the
    live yaml was unsigned. The door recomputes from evidence.json."""
    import json as _json
    from models.registry import promote, PromotionRefused
    run = tmp_path / "reg-forged"
    run.mkdir()
    g = _gcfg()
    g["signoff"] = {"owner_confirmed": True, "clinical_advisor": "Dr X",
                    "date": "2026-09-01",
                    "claim_scope": "irregular_rhythm_notification"}
    v = evaluate_regularity_gates(g, _green_evidence(), _green_floor())
    assert v["promotion_open"]
    (run / "gate_results.json").write_text(_json.dumps(v))
    (run / "regularity_record.json").write_text(_json.dumps(
        {"run_id": "reg-forged"}))
    (run / "model.json").write_text("{}")
    # no evidence.json at all: refused
    with pytest.raises(PromotionRefused, match="no evidence.json"):
        promote(run, registry_path=tmp_path / "reg.jsonl",
                spec_path=tmp_path / "spec.md")
    # evidence present, but the LIVE yaml is unsigned: refused
    (run / "evidence.json").write_text(_json.dumps(
        {"evidence": _green_evidence(), "floor": _green_floor(),
         "gates_version": g["gates_version"]}))
    with pytest.raises(PromotionRefused) as e:
        promote(run, registry_path=tmp_path / "reg.jsonl",
                spec_path=tmp_path / "spec.md")
    assert "live re-evaluation" in str(e.value)
    assert not (tmp_path / "reg.jsonl").exists()
