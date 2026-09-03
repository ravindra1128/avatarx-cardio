"""v0.6 flutter T5 — the §F gates: every threshold pre-registered in
configs/gates.yaml, missing evidence is a RED gate with its own reason,
and promotion is refused while anything is red.

The gate this track lives or dies on is F1's specificity against the
hard-negative battery, because a regular fast pulse is usually benign.
Second is F2, which forces the known misses to be COUNTED (F-c)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import pytest

from evaluation.flutter_gates import (evaluate_flutter_gates,
                                      flutter_gate_status,
                                      flutter_render_allowed,
                                      GATE_TITLES, load_flutter_gates,
                                      render_flutter_status_html,
                                      shipping_rule, TRACK_NOTE)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _green_evidence():
    """Evidence that clears every numeric criterion, so the tests can
    show exactly which single change reds each gate."""
    return {
        "data": {"signal_domain": "facial_rppg",
                 "participant_disjoint": True, "session_disjoint": True,
                 "production_path": True},
        "f0": {"bias_bpm": -1.2, "within_5bpm_fraction": 0.88,
               "n_arrhythmia_scans": 44},
        "f1": {"sensitivity_2to1": 0.86, "specificity_battery": 0.962,
               "n_flutter_scans": 52, "n_negative_scans": 240,
               "negative_class_share": {"sinus_tachycardia": 0.62},
               "b3_specificity_at_matched_sens": 0.93,
               "head_specificity_at_matched_sens": 0.962},
        "f2": {"per_ratio": {
            "TWO_TO_ONE": {"n": 28, "sensitivity": 0.89},
            "THREE_TO_ONE": {"n": 12, "sensitivity": 0.25},
            "FOUR_TO_ONE": {"n": 12, "sensitivity": 0.0},
            "VARIABLE": {"n": 11, "sensitivity": 0.18}}},
        "f3": {"series_auc": 0.90, "n_flutter_series": 12,
               "n_nonflutter_series": 25, "min_scans_per_series": 3},
        "f4": {"fitzpatrick_group": {
            "detection_parity_ratio_worst": 0.81,
            "coverage_ratio_worst": 0.86,
            "darkest_band_present": True}},
        "f5": {"combined_endpoint_decision": "combined AF-or-flutter"},
    }


def _gcfg():
    return load_flutter_gates()


def test_block_is_pre_registered_with_every_threshold_present():
    g = _gcfg()
    assert g["gates_version"] == "F-2026-09-01-1"
    assert g["requires_clinical_signoff"] is True
    for key in ("f0_rate_accuracy", "f1_flag_performance",
                "f2_per_ratio_reporting", "f3_serial_signature",
                "f4_fairness", "f5_claim_mapping"):
        assert isinstance(g[key], dict), key
    # the Cramer floor is what F0 must beat, and by a real margin
    assert g["f0_rate_accuracy"]["max_abs_bias_bpm"] < 7.45
    assert g["f0_rate_accuracy"]["min_within_5bpm_fraction"] > 0.486
    # specificity is the binding constraint, above sensitivity
    f1 = g["f1_flag_performance"]
    assert f1["min_specificity_battery"] > f1["min_sensitivity_2to1"]
    assert f1["must_beat_b3"] is True
    assert f1["require_dominant_negative"] == "sinus_tachycardia"
    # F2 must cover every ratio incl. the known miss
    assert set(g["f2_per_ratio_reporting"]["required_ratios"]) == {
        "TWO_TO_ONE", "THREE_TO_ONE", "FOUR_TO_ONE", "VARIABLE"}
    assert g["f2_per_ratio_reporting"][
        "slow_block_miss_must_be_quantified"] is True
    assert g["f4_fairness"]["require_darkest_band_present"] is True
    assert g["f5_claim_mapping"]["permissible_first_claim"] == \
        "regular_tachy_ecg_referral"


def test_all_green_evidence_still_blocks_without_a_signature():
    """Numbers are necessary, never sufficient: F5 needs the owner and a
    clinical advisor on record."""
    v = evaluate_flutter_gates(_gcfg(), _green_evidence())
    reds = {g["gate"] for g in v["gates"] if g["status"] == "RED"}
    assert reds == {"f5"}
    assert v["clinical_signoff"] is False
    assert v["promotion_open"] is False
    f5 = [g for g in v["gates"] if g["gate"] == "f5"][0]
    assert any("owner + clinical advisor" in r for r in f5["reasons"])
    assert any("claim_scope" in r for r in f5["reasons"])


def test_signed_and_green_opens_promotion():
    g = _gcfg()
    g["signoff"] = {"owner_confirmed": True, "clinical_advisor": "Dr X",
                    "date": "2026-09-01",
                    "claim_scope": "regular_tachy_ecg_referral"}
    v = evaluate_flutter_gates(g, _green_evidence())
    assert v["all_gates_green"] is True
    assert v["promotion_open"] is True
    # ... and every yaml-null spelling of owner_confirmed closes it again
    for bad in (None, "None", "null", "~", "", "yes", 1):
        g["signoff"]["owner_confirmed"] = bad
        assert evaluate_flutter_gates(g, _green_evidence())[
            "promotion_open"] is False, bad


def test_missing_evidence_reds_every_gate_with_its_own_reason():
    v = evaluate_flutter_gates(_gcfg(), {})
    assert len(v["gates"]) == 6
    assert {g["gate"] for g in v["gates"]} == set(GATE_TITLES)
    for g in v["gates"]:
        assert g["status"] == "RED", g["gate"]
        assert g["reasons"], g["gate"]
    assert v["promotion_open"] is False


def test_surrogate_domain_can_never_open_a_gate():
    ev = _green_evidence()
    ev["data"]["signal_domain"] = "synthetic"
    v = evaluate_flutter_gates(_gcfg(), ev)
    assert all(g["status"] == "RED" for g in v["gates"])
    assert all(any("machinery evidence only" in r for r in g["reasons"])
               for g in v["gates"])


@pytest.mark.parametrize("patch,gate,needle", [
    ({"f0": {"bias_bpm": -7.45, "within_5bpm_fraction": 0.88,
             "n_arrhythmia_scans": 44}}, "f0", "bias"),
    ({"f0": {"bias_bpm": -1.0, "within_5bpm_fraction": 0.486,
             "n_arrhythmia_scans": 44}}, "f0", "within 5 bpm"),
    ({"f0": {"bias_bpm": -1.0, "within_5bpm_fraction": 0.9,
             "n_arrhythmia_scans": 12}}, "f0", "arrhythmia scans"),
    ({"f3": {"series_auc": 0.5, "n_flutter_series": 12,
             "n_nonflutter_series": 25,
             "min_scans_per_series": 3}}, "f3", "AUC"),
    ({"f3": {"series_auc": 0.9, "n_flutter_series": 12,
             "n_nonflutter_series": 25,
             "min_scans_per_series": 2}}, "f3", "shortest series"),
])
def test_each_numeric_criterion_reds_its_own_gate(patch, gate, needle):
    ev = _green_evidence()
    ev.update(patch)
    v = evaluate_flutter_gates(_gcfg(), ev)
    row = [g for g in v["gates"] if g["gate"] == gate][0]
    assert row["status"] == "RED"
    assert any(needle in r for r in row["reasons"]), row["reasons"]


def test_f1_refuses_a_specificity_earned_on_easy_negatives():
    """A battery dominated by clean sinus rhythm proves nothing: the
    confounder that actually presents is sinus TACHYCARDIA."""
    ev = _green_evidence()
    ev["f1"]["negative_class_share"] = {"sinus_tachycardia": 0.2,
                                        "sinus": 0.8}
    row = [g for g in evaluate_flutter_gates(_gcfg(), ev)["gates"]
           if g["gate"] == "f1"][0]
    assert row["status"] == "RED"
    assert any("dominant confounder" in r for r in row["reasons"])
    # ... and an unreported share is not a pass either
    ev["f1"].pop("negative_class_share")
    row = [g for g in evaluate_flutter_gates(_gcfg(), ev)["gates"]
           if g["gate"] == "f1"][0]
    assert any("no sinus_tachycardia share" in r for r in row["reasons"])


def test_f1_says_ship_b3_when_the_head_does_not_beat_it():
    ev = _green_evidence()
    ev["f1"]["head_specificity_at_matched_sens"] = 0.931   # +0.001
    v = evaluate_flutter_gates(_gcfg(), ev)
    row = [g for g in v["gates"] if g["gate"] == "f1"][0]
    assert row["status"] == "RED"
    assert any("SHIP B3" in r for r in row["reasons"])
    assert shipping_rule(ev) == "b3"
    # beating it by the pre-registered margin is what counts
    ev["f1"]["head_specificity_at_matched_sens"] = 0.955
    assert shipping_rule(ev) == "head"
    assert [g for g in evaluate_flutter_gates(_gcfg(), ev)["gates"]
            if g["gate"] == "f1"][0]["status"] == "GREEN"
    # no comparison at all: undecided, and the gate stays red
    ev["f1"].pop("b3_specificity_at_matched_sens")
    assert shipping_rule(ev) == "undecided"
    assert any("must be\n                               beaten or shipped"
               .replace("\n                               ", " ") in r
               for r in [g for g in evaluate_flutter_gates(_gcfg(), ev)
                         ["gates"] if g["gate"] == "f1"][0]["reasons"])


def test_f2_requires_the_known_miss_to_be_measured_not_asserted():
    ev = _green_evidence()
    # a zero sensitivity for 4:1 is FINE — that is the honest expected
    # result and it passes, because it was measured
    assert [g for g in evaluate_flutter_gates(_gcfg(), ev)["gates"]
            if g["gate"] == "f2"][0]["status"] == "GREEN"
    # omitting it is not
    ev["f2"]["per_ratio"]["FOUR_TO_ONE"] = {"n": 12}
    row = [g for g in evaluate_flutter_gates(_gcfg(), ev)["gates"]
           if g["gate"] == "f2"][0]
    assert row["status"] == "RED"
    assert any("MEASURED, not asserted" in r for r in row["reasons"])
    # dropping the ratio entirely is caught by name
    del ev["f2"]["per_ratio"]["VARIABLE"]
    row = [g for g in evaluate_flutter_gates(_gcfg(), ev)["gates"]
           if g["gate"] == "f2"][0]
    assert any("VARIABLE not reported" in r for r in row["reasons"])


def test_f4_reds_when_the_darkest_band_is_absent():
    """The reference study excluded its only Fitzpatrick-VI patient. An
    absent band is a RED gate here, not a footnote."""
    ev = _green_evidence()
    ev["f4"]["fitzpatrick_group"]["darkest_band_present"] = False
    row = [g for g in evaluate_flutter_gates(_gcfg(), ev)["gates"]
           if g["gate"] == "f4"][0]
    assert row["status"] == "RED"
    assert any("darkest skin-tone band is absent" in r
               for r in row["reasons"])
    # an unrated (too-small) group is named, never silently dropped
    ev["f4"]["fitzpatrick_group"] = {
        "detection_parity_ratio_worst": None,
        "coverage_ratio_worst": None, "darkest_band_present": True}
    row = [g for g in evaluate_flutter_gates(_gcfg(), ev)["gates"]
           if g["gate"] == "f4"][0]
    assert any("unrated" in r for r in row["reasons"])


def test_f5_requires_the_limitations_doc_and_the_endpoint_decision():
    g = _gcfg()
    g["signoff"] = {"owner_confirmed": True, "clinical_advisor": "Dr X",
                    "date": "2026-09-01",
                    "claim_scope": "regular_tachy_ecg_referral"}
    ev = _green_evidence()
    ev["f5"] = {}
    row = [x for x in evaluate_flutter_gates(g, ev)["gates"]
           if x["gate"] == "f5"][0]
    assert row["status"] == "RED"
    assert any("combined-endpoint decision" in r for r in row["reasons"])
    # the limitations doc is a SHIPPED artifact (F-c): its absence reds
    # the gate, and its presence is checked on disk
    doc = _ROOT / g["f5_claim_mapping"]["limitations_doc_required"]
    ev["f5"] = {"combined_endpoint_decision": "combined"}
    row = [x for x in evaluate_flutter_gates(g, ev)["gates"]
           if x["gate"] == "f5"][0]
    if doc.exists():
        assert row["status"] == "GREEN"
    else:
        assert any("known-miss registry is a shipped artifact" in r
                   for r in row["reasons"])


def test_a_threshold_missing_from_the_yaml_fails_loudly():
    """A silently-applied coded default would let the pre-registered
    contract drift."""
    g = _gcfg()
    del g["f1_flag_performance"]["min_specificity_battery"]
    with pytest.raises(ValueError, match="min_specificity_battery"):
        evaluate_flutter_gates(g, _green_evidence())


def test_status_and_render_are_fail_closed(tmp_path):
    doc = flutter_gate_status(runs_root=tmp_path)
    assert doc["track"] == "flutter"
    assert doc["promotion"] == "BLOCKED"
    assert doc["evaluation_run"] is None
    assert doc["shipping_rule"] == "undecided"
    assert flutter_render_allowed(runs_root=tmp_path) is False
    # a broken gates file must not open anything
    bad = tmp_path / "bad.yaml"
    bad.write_text("flutter: not-a-block\n")
    assert flutter_render_allowed(runs_root=tmp_path,
                                  gates_path=bad) is False
    assert flutter_render_allowed(runs_root=tmp_path,
                                  gates_path=tmp_path / "nope") is False
    html = render_flutter_status_html(doc)
    assert TRACK_NOTE in html and "BLOCKED" in html
    assert "NAMES NO RHYTHM" in TRACK_NOTE


def test_promote_refuses_and_lists_every_red_reason(tmp_path):
    from models.registry import promote, PromotionRefused
    run = tmp_path / "flut-1"
    run.mkdir()
    v = evaluate_flutter_gates(_gcfg(), {})
    (run / "gate_results.json").write_text(json.dumps(v))
    (run / "flutter_record.json").write_text(json.dumps(
        {"run_id": "flut-1", "shipping_rule": "undecided"}))
    (run / "model.json").write_text("{}")
    with pytest.raises(PromotionRefused) as e:
        promote(run, registry_path=tmp_path / "reg.jsonl",
                spec_path=tmp_path / "spec.md")
    msg = str(e.value)
    assert "§F" in msg
    for gate in GATE_TITLES.values():
        assert gate in msg, gate
    assert "clinical signoff pending" in msg
    # a hand-crafted file listing only green gates is refused too
    (run / "gate_results.json").write_text(json.dumps(
        {"gates": [{"gate": "f0", "status": "GREEN", "title": "F0"}],
         "all_gates_green": True, "clinical_signoff": True,
         "promotion_open": True}))
    with pytest.raises(PromotionRefused) as e2:
        promote(run, registry_path=tmp_path / "reg.jsonl",
                spec_path=tmp_path / "spec.md")
    assert "f1" in str(e2.value) and "f5" in str(e2.value)


def test_no_gate_results_file_is_refused(tmp_path):
    from models.registry import promote, PromotionRefused
    run = tmp_path / "flut-2"
    run.mkdir()
    (run / "flutter_record.json").write_text(json.dumps(
        {"run_id": "flut-2"}))
    with pytest.raises(PromotionRefused, match="no §F evaluation"):
        promote(run, registry_path=tmp_path / "reg.jsonl",
                spec_path=tmp_path / "spec.md")


def test_cli_gate_status_flutter_track(tmp_path):
    import os
    import subprocess
    env = dict(os.environ, AVATARX_FLUTTER_RUNS=str(tmp_path / "runs"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "gate-status", "--track", "flutter",
                        "--html", str(tmp_path / "s.html")],
                       capture_output=True, text=True, timeout=300,
                       env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    doc = json.loads(r.stdout)
    assert doc["track"] == "flutter"
    assert doc["promotion"] == "BLOCKED"
    assert len(doc["gates"]) == 6
    assert (tmp_path / "s.html").exists()
