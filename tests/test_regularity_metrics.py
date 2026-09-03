"""v0.7 Task 3/5 — agreement statistics, the ceiling test, benign
separation, baselines and fairness on fabricated rows, pinned BOTH
ways where a harness could otherwise only give one answer."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from evaluation.regularity_gates import (evaluate_regularity_gates,
                                         load_regularity_gates)

from evaluation.regularity_metrics import (baselines, benign_separation,
                                           bland_altman, ceiling_test,
                                           clinically_irregular,
                                           cohens_kappa, fairness,
                                           pearson_r)


def test_kappa_and_bland_altman():
    a = ["regular"] * 8 + ["irregular"] * 8
    assert cohens_kappa(a, a)["kappa"] == 1.0
    b = ["regular"] * 4 + ["irregular"] * 4 + ["regular"] * 4 + \
        ["irregular"] * 4
    assert cohens_kappa(a, b)["kappa"] == 0.0
    # indeterminate rows are excluded from kappa, and counted apart
    k = cohens_kappa(a + ["indeterminate"], a + ["regular"])
    assert k["n"] == 16 and k["kappa"] == 1.0
    assert cohens_kappa([], [])["kappa"] is None
    ba = bland_altman([0.10, 0.20, 0.30], [0.09, 0.21, 0.29])
    assert abs(ba["bias"]) < 0.01 and ba["loa"][0] < 0 < ba["loa"][1]
    assert pearson_r([1, 2, 3, 4], [2, 4, 6, 8])["r"] == 1.0
    assert pearson_r([1, 1, 1], [1, 2, 3])["r"] is None


def _row(pid, cam, ecg, *, grade="A", ds="d", rhythm="SINUS", age=50,
         fitz=3, head_class="auto", evidence=None, rmssd=None,
         shannon=None, ibi=850.0, sqi=0.8, sex="F"):
    """head_class None = the head abstained; "auto" = it agreed with the
    camera index."""
    cc = ("indeterminate" if cam is None else
          "irregular" if cam >= 0.06 else "regular")
    ec = ("indeterminate" if ecg is None else
          "irregular" if ecg >= 0.06 else "regular")
    return {"recording_id": f"{pid}-{cam}", "participant_id": pid,
            "session_id": f"{pid}-s", "dataset": ds, "rhythm": rhythm,
            "sqi_grade": grade, "sqi": sqi, "fitzpatrick_group": fitz,
            "age_years": age, "sex": sex,
            "age_band": ("<35" if age < 35 else "35-59" if age < 60
                         else ">=60"),
            "camera_index": cam, "camera_class": cc, "ecg_index": ecg,
            "ecg_class": ec,
            "camera_features": {"values": {
                "rmssd": rmssd if rmssd is not None else
                (cam or 0.0) * 800.0,
                "shannon_entropy": shannon if shannon is not None else
                min(0.9, (cam or 0.0) * 3.0), "median_ibi": ibi}},
            "head": {"class": cc if head_class == "auto" else head_class,
                     "benign_pattern_evidence": (
                         {"evidence": evidence} if evidence else
                         ({"evidence": "not_applicable"} if cc == "regular"
                          else {"evidence": "indeterminate"}))}}


def test_ceiling_test_reports_agreement_per_grade_and_dataset():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        e = 0.02 + 0.2 * (i % 2)
        rows.append(_row(f"p{i}", e + rng.normal(0, 0.004), e,
                         grade="A" if i < 10 else "B",
                         ds="synthetic" if i % 3 else "public"))
    c = ceiling_test(rows)
    assert c["kappa"] == 1.0 and c["index_r"] > 0.99
    assert c["loa_halfwidth"] < 0.02
    assert set(c["per_sqi_grade"]) == {"A", "B"}
    assert c["worst_grade_kappa"] == 1.0
    assert c["n_paired_scans"] == 20
    # a camera that reads everything regular has kappa 0, not "fine"
    bad = [dict(r, camera_index=0.02, camera_class="regular") for r in rows]
    assert ceiling_test(bad)["kappa"] == 0.0


def test_benign_separation_counts_unexplained_irregularity_only():
    rows = []
    # RSA sessions the head explained: not clinically irregular
    for i in range(10):
        rows.append(_row(f"r{i}", 0.12, 0.12, rhythm="RESPIRATORY_SINUS_"
                         "ARRHYTHMIA", age=25, evidence="respiration_coupled"))
    # RSA sessions the head could NOT explain: flagged
    for i in range(2):
        rows.append(_row(f"u{i}", 0.12, 0.12, rhythm="RESPIRATORY_SINUS_"
                         "ARRHYTHMIA", age=28, evidence="indeterminate"))
    # older regular sinus, and one abstention
    for i in range(6):
        rows.append(_row(f"s{i}", 0.02, 0.02, rhythm="SINUS", age=65))
    rows.append(_row("a0", 0.02, 0.02, rhythm="SINUS", age=66,
                     head_class=None))
    b = benign_separation(rows)
    assert b["n_rsa_sessions"] == 12
    assert b["rsa_flag_rate"] == pytest.approx(2 / 12, abs=1e-4)
    assert b["rsa_explained_rate"] == pytest.approx(10 / 12, abs=1e-4)
    assert b["age_strata"]["<35"]["specificity"] == pytest.approx(10 / 12,
                                                                  abs=1e-4)
    assert b["age_strata"][">=60"]["specificity"] == 1.0
    assert b["age_strata"]["35-59"]["specificity"] is None
    assert b["n_abstained"] == 1
    assert clinically_irregular({"class": None}) is None
    assert clinically_irregular({"class": "regular"}) is False
    assert clinically_irregular({"class": "irregular",
                                 "benign_pattern_evidence":
                                 {"evidence": "chaotic"}}) is True


def test_baselines_are_scored_held_out_and_can_lose():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(40):
        irregular = i % 2 == 1
        e = 0.20 if irregular else 0.02
        rows.append(_row(f"p{i:02d}", e + rng.normal(0, 0.005), e,
                         age=(30 if irregular else 60) + int(rng.integers(
                             0, 5)), sqi=0.8))
    b = baselines(rows)
    assert b["n_train_rows"] > 0 and b["n_test_rows"] > 0
    assert b["head_balanced_accuracy"] == 1.0
    assert b["b1_balanced_accuracy"] == 1.0            # RMSSD tracks the index
    assert b["b3_balanced_accuracy"] is not None
    # demographics leak the label here BY CONSTRUCTION (age correlates
    # with irregularity in this fixture): B4 must expose that, not hide it
    assert b["b4_balanced_accuracy"] is not None and \
        b["b4_balanced_accuracy"] > 0.8
    # SQI carries nothing: B5 sits near chance
    assert b["b5_balanced_accuracy"] is not None and \
        b["b5_balanced_accuracy"] < 0.75
    assert b["head_no_read_rate"] == 0.0
    # a head that always says regular loses to every baseline
    dumb = [dict(r, head={"class": "regular",
                          "benign_pattern_evidence": {"evidence":
                                                      "not_applicable"}})
            for r in rows]
    assert baselines(dumb)["head_balanced_accuracy"] == 0.5


def test_fairness_is_participant_level_and_names_the_darkest_band():
    rows = []
    for i in range(6):
        for fitz in (2, 6):
            rows.append(_row(f"p{i}f{fitz}", 0.02, 0.02, fitz=fitz))
    f = fairness(rows)["fitzpatrick_group"]
    assert f["darkest_band_present"] is True
    assert f["worst_index_bias"] == 0.0
    assert f["coverage_ratio_worst"] == 1.0
    assert f["unrated_groups"] == []
    light = fairness([r for r in rows if r["fitzpatrick_group"] == 2])
    assert light["fitzpatrick_group"]["darkest_band_present"] is False
    assert light["fitzpatrick_group"]["coverage_ratio_worst"] is None
    rows.append(_row("solo", 0.02, 0.02, fitz=4))
    assert fairness(rows)["fitzpatrick_group"]["unrated_groups"] == ["4"]


def test_kappa_is_undefined_on_a_single_class_cohort_not_one():
    """Review finding: a cohort with no irregular scan on either side
    has expected agreement 1 and kappa undefined; it must read as no
    agreement on record, never as a perfect 1.0."""
    k = cohens_kappa(["regular"] * 30, ["regular"] * 30)
    assert k["kappa"] is None and k["single_class"] is True
    assert k["observed_agreement"] == 1.0
    rows = [_row(f"p{i}", 0.02, 0.02) for i in range(30)]
    c = ceiling_test(rows)
    assert c["kappa"] is None and c["worst_grade_kappa"] is None
    v = evaluate_regularity_gates(load_regularity_gates(), {
        "data": {"signal_domain": "facial_rppg", "participant_disjoint":
                 True, "session_disjoint": True, "production_path": True},
        "r0": c}, {})
    r0 = [g for g in v["gates"] if g["gate"] == "r0"][0]
    assert any("no class agreement" in r for r in r0["reasons"])


def test_a_pair_is_a_scan_judged_on_both_sides():
    """Review finding: 99 no-read scans plus one agreeing pair must not
    open R0 on n = 100."""
    rows = [_row(f"p{i}", 0.02, 0.02) for i in range(1)]
    for i in range(99):
        r = _row(f"q{i}", 0.02, 0.02)
        r["camera_class"] = "indeterminate"        # non-ACCEPT scan
        rows.append(r)
    c = ceiling_test(rows)
    assert c["n_paired_scans"] == 1
    assert c["n_indeterminate_either_side"] == 99
    assert c["n_scans"] == 100


def test_worst_grade_kappa_is_unrated_when_any_grade_is_undefined():
    rows = []
    for i in range(10):
        rows.append(_row(f"a{i}", 0.02 if i % 2 else 0.2,
                         0.02 if i % 2 else 0.2, grade="A"))
        rows.append(_row(f"c{i}", 0.02, 0.02, grade="C"))   # single class
    c = ceiling_test(rows)
    assert c["per_sqi_grade"]["A"]["kappa"]["kappa"] == 1.0
    assert c["per_sqi_grade"]["C"]["kappa"]["kappa"] is None
    assert c["worst_grade_kappa"] is None


def test_bland_altman_bias_is_reported_for_the_gate():
    rows = [_row(f"p{i}", e + 0.05, e) for i, e in
            enumerate([0.02, 0.03, 0.2, 0.25, 0.02, 0.21])]
    c = ceiling_test(rows)
    assert c["bias"] == pytest.approx(0.05, abs=1e-6)
    assert c["loa_halfwidth"] == pytest.approx(0.0, abs=1e-6)
