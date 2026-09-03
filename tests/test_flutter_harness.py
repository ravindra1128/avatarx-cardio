"""v0.6 flutter T3/T6 — the hard-negative battery, the baselines, and
the combined endpoint.

The battery is pinned BOTH ways: in a world where the flag genuinely
separates the target from the confounders it must show that, and in a
world where the flag is really just a rate threshold it must show THAT
too. A harness that can only produce one answer measures nothing.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

from evaluation.flutter_metrics import (b1_rate_only,
                                        b2_rate_and_naive_dispersion,
                                        b3_interpretable,
                                        combined_endpoint,
                                        combined_referral,
                                        evaluate_flutter_dataset,
                                        fairness, flag_performance,
                                        hard_negative_battery,
                                        per_ratio_report, rate_accuracy,
                                        serial_signature)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _row(pid, cohort, *, bpm, rmssd, coupling, rhythm, ratio=None,
         flag=None, fitz=3, age=60, ecg_bpm=None, afib_flag=False,
         band=None, in_band=None, fps=240.0, series=None, sess=None,
         cv=None):
    band = band or ("2:1" if 140 <= bpm <= 165 else
                    "3:1" if 95 <= bpm <= 110 else
                    "4:1" if 70 <= bpm <= 80 else "2:1")
    return {
        "recording_id": f"r_{pid}_{cohort}_{bpm:.0f}",
        "participant_id": pid, "session_id": sess or f"{pid}-s1",
        "series_id": series or pid, "cohort": cohort, "rhythm": rhythm,
        "conduction_ratio": ratio, "fitzpatrick_group": fitz,
        "age_years": age, "fps": fps,
        "ecg_rate_bpm": bpm if ecg_bpm is None else ecg_bpm,
        "head_flag": flag, "afib_flag": afib_flag,
        "features": {
            "available": True,
            "rate": {"median_bpm": bpm, "band": band,
                     "in_band": (band == "2:1" and 140 <= bpm <= 165)
                     if in_band is None else in_band,
                     "band_fraction": 0.95, "n_intervals": 90},
            "regularity": {"rmssd_ms": rmssd,
                           # SDNN/mean: iid beat jitter gives
                           # sdnn ~ rmssd/sqrt2, while a coherent
                           # respiratory oscillation gives sdnn > rmssd
                           "cv": (cv if cv is not None else
                                  (0.707 if (coupling or 0) < 0.2
                                   else 1.15) * rmssd
                                  / (60000.0 / bpm)),
                           "coupling": {
                               "available": coupling is not None,
                               "tachogram_resp_fraction": coupling}},
        }}


def _cohort(*, informative=True, n=8):
    """A labeled cohort. `informative=False` is the world where the flag
    is really just "fast": the confounders become indistinguishable from
    the target on every criterion except rate."""
    rows = []
    for i in range(n):
        # the target: 2:1 flutter, metronomic, uncoupled
        rows.append(_row(f"p{i}a", "flutter_2to1", bpm=150.0, rmssd=4.0,
                         coupling=0.03, rhythm="ATRIAL_FLUTTER",
                         ratio="TWO_TO_ONE", flag=True))
        # the dominant confounder: sinus tach at the same rate
        st_coupling = 0.03 if not informative else 0.85
        rows.append(_row(f"p{i}b", "sinus_tachycardia", bpm=151.0,
                         rmssd=4.0 if not informative else 26.0,
                         coupling=st_coupling,
                         rhythm="SINUS_TACHYCARDIA",
                         flag=not informative))
        # SVT: covered by design, counts as a negative for specificity
        rows.append(_row(f"p{i}c", "svt", bpm=158.0, rmssd=5.0,
                         coupling=0.04, rhythm="SVT", flag=True))
        # slow fixed block: the known miss
        rows.append(_row(f"p{i}d", "flutter_4to1", bpm=75.0, rmssd=4.0,
                         coupling=0.03, rhythm="ATRIAL_FLUTTER",
                         ratio="FOUR_TO_ONE", flag=False))
        # variable block: irregular, AF's space
        rows.append(_row(f"p{i}e", "flutter_variable", bpm=110.0,
                         rmssd=90.0, coupling=0.05,
                         rhythm="ATRIAL_FLUTTER", ratio="VARIABLE",
                         flag=False))
        # 3:1
        rows.append(_row(f"p{i}f", "flutter_3to1", bpm=100.0, rmssd=4.0,
                         coupling=0.03, rhythm="ATRIAL_FLUTTER",
                         ratio="THREE_TO_ONE", flag=False))
        # benign negatives
        rows.append(_row(f"p{i}g", "sinus", bpm=72.0, rmssd=60.0,
                         coupling=0.7, rhythm="SINUS", flag=False))
        rows.append(_row(f"p{i}h", "paced", bpm=70.0, rmssd=2.0,
                         coupling=0.02, rhythm="PACED", flag=False))
        rows.append(_row(f"p{i}i", "af", bpm=97.0, rmssd=180.0,
                         coupling=0.06, rhythm="AFIB", flag=False,
                         afib_flag=True))
        # a scan the head could not judge (no respiration channel)
        rows.append(_row(f"p{i}j", "sinus_tachycardia", bpm=150.0,
                         rmssd=20.0, coupling=None,
                         rhythm="SINUS_TACHYCARDIA", flag=None))
    return rows


# ------------------------------------------------------- F0
def test_rate_accuracy_excludes_sinus_and_reports_the_bias():
    rows = [_row("p1", "flutter_2to1", bpm=147.0, rmssd=4.0,
                 coupling=0.03, rhythm="ATRIAL_FLUTTER",
                 ratio="TWO_TO_ONE", ecg_bpm=150.0),
            _row("p2", "flutter_2to1", bpm=148.0, rmssd=4.0,
                 coupling=0.03, rhythm="ATRIAL_FLUTTER",
                 ratio="TWO_TO_ONE", ecg_bpm=150.0),
            # a sinus scan with a huge error must NOT dilute the number:
            # the reference bias is an ARRHYTHMIA number
            _row("p3", "sinus", bpm=72.0, rmssd=60.0, coupling=0.7,
                 rhythm="SINUS", ecg_bpm=72.0)]
    r = rate_accuracy(rows)
    assert r["n_arrhythmia_scans"] == 2
    assert r["bias_bpm"] == pytest.approx(-2.5, abs=0.01)
    assert r["within_5bpm_fraction"] == 1.0
    assert r["within_5bpm_ci95"][0] < 1.0        # a CI, not a point
    # no paired reference at all is reported, not defaulted
    bare = rate_accuracy([_row("p4", "flutter_2to1", bpm=150.0,
                               rmssd=4.0, coupling=0.03,
                               rhythm="ATRIAL_FLUTTER",
                               ratio="TWO_TO_ONE", ecg_bpm=None)])
    bare_rows = [dict(r_, ecg_rate_bpm=None) for r_ in rows]
    assert rate_accuracy(bare_rows)["n_arrhythmia_scans"] == 0
    assert "no arrhythmia scan" in rate_accuracy(bare_rows)["reason"]


# ------------------------------------------------------- the battery
def test_every_confounder_gets_its_own_false_positive_rate():
    b = hard_negative_battery(_cohort())
    per = b["per_class"]
    assert set(per) >= {"sinus_tachycardia", "svt", "sinus", "paced",
                        "af"}
    assert per["sinus_tachycardia"]["false_positive_rate"] == 0.0
    assert per["svt"]["false_positive_rate"] == 1.0      # by design
    assert per["sinus"]["false_positive_rate"] == 0.0
    # abstentions are excluded from the rate and reported separately
    assert per["sinus_tachycardia"]["n_abstained"] == 8
    assert per["sinus_tachycardia"]["abstention_rate"] == 0.5
    # flutter rows never appear among the negatives, whatever the ratio
    assert not any(k.startswith("flutter") for k in per)
    # shares are reported so a battery of easy negatives is visible
    assert abs(sum(b["negative_class_share"].values()) - 1.0) < 1e-3


def test_the_battery_measures_a_flag_that_is_really_just_rate():
    """The pinned negative control: when the confounders are identical
    to the target except in name, the battery must SAY so."""
    good = flag_performance(_cohort(informative=True))
    bad = flag_performance(_cohort(informative=False))
    assert good["sensitivity_2to1"] == 1.0
    assert bad["sensitivity_2to1"] == 1.0
    # specificity is what separates the two worlds
    assert good["specificity_battery"] > bad["specificity_battery"]
    assert bad["per_negative_class"]["sinus_tachycardia"][
        "false_positive_rate"] == 1.0
    assert good["per_negative_class"]["sinus_tachycardia"][
        "false_positive_rate"] == 0.0


def test_b1_b2_b3_are_real_rules_that_can_disagree():
    target = _row("p1", "flutter_2to1", bpm=150.0, rmssd=4.0,
                  coupling=0.03, rhythm="ATRIAL_FLUTTER",
                  ratio="TWO_TO_ONE")
    sinus_tach = _row("p2", "sinus_tachycardia", bpm=150.0, rmssd=26.0,
                      coupling=0.85, rhythm="SINUS_TACHYCARDIA")
    # B1 cannot tell them apart at all — that is its purpose
    assert b1_rate_only(target) is True
    assert b1_rate_only(sinus_tach) is True
    # B2 uses dispersion and gets it right here
    assert b2_rate_and_naive_dispersion(target) is True
    assert b2_rate_and_naive_dispersion(sinus_tach) is False
    # B3 uses the coupling too
    assert b3_interpretable(target) is True
    assert b3_interpretable(sinus_tach) is False
    # B3 NEVER abstains: a missing coupling reads as "not coupled",
    # the optimistic assumption the head refuses to make
    no_resp = _row("p3", "sinus_tachycardia", bpm=150.0, rmssd=4.0,
                   coupling=None, rhythm="SINUS_TACHYCARDIA")
    assert b3_interpretable(no_resp) is True
    from evaluation.flutter_metrics import head_flag
    assert head_flag(no_resp) is None


def test_head_vs_b3_uses_the_heads_own_verdict():
    """The criterion that decides which rule ships must actually see the
    head. Scoring the head with B3's rule compared B3 against itself and
    reported whatever the abstentions happened to produce."""
    rows = _cohort()
    f = flag_performance(rows, target_sensitivity=0.8)
    # SVT flags by design and counts as a negative, so specificity is
    # below 1.0 — and it matches specificity_battery, which is the same
    # denominator (targets + the negative battery, never non-2:1
    # flutter, which is neither)
    assert f["head_specificity_at_matched_sens"] == 0.8667
    assert f["specificity_battery"] == 0.8
    assert f["b3_specificity_at_matched_sens"] is not None
    assert f["comparison_is_held_out"] is True
    assert f["head_sensitivity_at_match"] >= 0.8
    # INVERT every head verdict: the head's number must move, or the
    # comparison is not reading the head at all
    flipped = [dict(r, head_flag=(None if r["head_flag"] is None
                                  else not r["head_flag"])) for r in rows]
    g = flag_performance(flipped, target_sensitivity=0.8)
    assert g["head_specificity_at_matched_sens"] != \
        f["head_specificity_at_matched_sens"]
    assert g["head_specificity_at_matched_sens"] < 0.2
    # ... while B3, untouched, reports the same number in both runs
    assert g["b3_specificity_at_matched_sens"] == \
        f["b3_specificity_at_matched_sens"]


def test_b3_is_matched_on_the_rows_the_head_actually_judged():
    """Otherwise the head 'wins' by declining exactly the rows B3 gets
    wrong. Construct that trap: sinus tach the head abstains on and B3
    flags."""
    rows = []
    for i in range(10):
        rows.append(_row(f"t{i}", "flutter_2to1", bpm=150.0, rmssd=4.0,
                         coupling=0.03, rhythm="ATRIAL_FLUTTER",
                         ratio="TWO_TO_ONE", flag=True))
    for i in range(20):
        # metronomic-looking sinus tach with NO respiration channel:
        # the head abstains, B3 reads "not coupled" and flags
        rows.append(_row(f"n{i}", "sinus_tachycardia", bpm=150.0,
                         rmssd=4.0, coupling=None,
                         rhythm="SINUS_TACHYCARDIA", flag=None))
    f = flag_performance(rows, target_sensitivity=0.8)
    # like-for-like: on the rows the head judged there are no negatives
    # at all, so B3's specificity there is undefined — NOT a free win
    # the held-out half carries only targets the head judged, so there
    # are no negatives to score B3 against — NOT a free win
    assert f["comparison_rows"] > 0
    assert f["b3_specificity_at_matched_sens"] is None
    # and B3 judging everything is reported, showing what was declined
    assert f["b3_specificity_all_rows"] == 0.0
    assert f["no_read_rate"] > 0
    # the gate cannot read this as "head beats B3"
    from evaluation.flutter_gates import shipping_rule
    assert shipping_rule({"f1": f}) == "undecided"


def test_b4_is_fit_on_a_participant_disjoint_half():
    f = flag_performance(_cohort(), target_sensitivity=0.8)
    assert "b4_specificity_at_matched_sens" in f


# --------------------------------------------------- F2 known misses
def test_per_ratio_reports_the_known_miss_as_a_measured_zero():
    r = per_ratio_report(_cohort())
    per = r["per_ratio"]
    assert set(per) == {"TWO_TO_ONE", "THREE_TO_ONE", "FOUR_TO_ONE",
                        "VARIABLE"}
    assert per["TWO_TO_ONE"]["sensitivity"] == 1.0
    assert per["FOUR_TO_ONE"]["sensitivity"] == 0.0      # measured
    assert per["FOUR_TO_ONE"]["n"] == 8
    assert r["slow_block_sensitivity"] == 0.0
    assert r["slow_block_n"] == 8
    assert "indistinguishable" in r["note"]
    # unrecorded ratios are named rather than folded into a bucket
    rows = _cohort() + [_row("pz", "flutter_x", bpm=150.0, rmssd=4.0,
                             coupling=0.03, rhythm="ATRIAL_FLUTTER",
                             ratio=None, flag=True)]
    assert "UNRECORDED" in per_ratio_report(rows)["per_ratio"]


# --------------------------------------------------- F3 serial
def test_serial_signature_separates_stepping_series_from_drifting():
    rows = []
    for k in range(6):     # flutter series: 150 -> 100 -> 75
        for j, bpm in enumerate((150.0, 100.0, 75.0)):
            rows.append(_row(f"fs{k}", "flutter_2to1", bpm=bpm,
                             rmssd=4.0, coupling=0.03,
                             rhythm="ATRIAL_FLUTTER",
                             ratio="TWO_TO_ONE", series=f"fs{k}",
                             sess=f"fs{k}-s{j}"))
    for k in range(6):     # recovery series: smooth drift, no structure
        for j, bpm in enumerate((132.0, 118.0, 104.0)):
            rows.append(_row(f"ns{k}", "sinus_tachycardia", bpm=bpm,
                             rmssd=30.0, coupling=0.8,
                             rhythm="SINUS_TACHYCARDIA",
                             series=f"ns{k}", sess=f"ns{k}-s{j}"))
    s = serial_signature(rows)
    assert s["n_flutter_series"] == 6 and s["n_nonflutter_series"] == 6
    assert s["series_auc"] is not None and s["series_auc"] > 0.9
    assert s["min_scans_per_series"] == 3
    # a two-scan series is excluded and COUNTED, never scored as zero
    rows += [_row("short1", "flutter_2to1", bpm=150.0, rmssd=4.0,
                  coupling=0.03, rhythm="ATRIAL_FLUTTER",
                  ratio="TWO_TO_ONE", series="short1")] * 2
    s2 = serial_signature(rows)
    assert s2["n_series_too_short"] >= 1
    assert s2["n_flutter_series"] == 6


# --------------------------------------------------- F4 fairness
def test_fairness_is_participant_level_and_names_unrated_groups():
    rows = []
    for i in range(6):
        for fitz in (2, 6):
            rows.append(_row(f"p{i}f{fitz}", "flutter_2to1", bpm=150.0,
                             rmssd=4.0, coupling=0.03,
                             rhythm="ATRIAL_FLUTTER",
                             ratio="TWO_TO_ONE", flag=True, fitz=fitz))
    f = fairness(rows)["fitzpatrick_group"]
    assert f["detection_parity_ratio_worst"] == 1.0
    assert f["darkest_band_present"] is True
    assert f["unrated_groups"] == []
    # a group below the floor is None-rated BY NAME, not dropped
    rows += [_row("solo", "flutter_2to1", bpm=150.0, rmssd=4.0,
                  coupling=0.03, rhythm="ATRIAL_FLUTTER",
                  ratio="TWO_TO_ONE", flag=False, fitz=4)]
    f2 = fairness(rows)["fitzpatrick_group"]
    assert f2["unrated_groups"] == ["4"]
    assert f2["detection_parity_ratio_worst"] is None
    # a cohort with no dark-skin participants is visible as such
    light = [r for r in rows if r["fitzpatrick_group"] != 6]
    assert fairness(light)["fitzpatrick_group"][
        "darkest_band_present"] is False


# ------------------------------------------- Task 6 combined endpoint
def test_combined_referral_is_a_pure_fusion_that_names_nothing():
    assert combined_referral(True, False)["trigger"] == "irregular"
    assert combined_referral(True, True)["trigger"] == "both"
    assert combined_referral(False, True)["trigger"] == "regular_tachy"
    assert combined_referral(False, True)["sentence_key"] == \
        "regular_tachy"
    assert combined_referral(True, True)["sentence_key"] == "afib"
    assert combined_referral(False, False)["refer"] is False
    # one head abstaining does not veto the other's positive
    assert combined_referral(None, True)["refer"] is True
    assert combined_referral(True, None)["refer"] is True
    # both abstaining is a no-read, not a negative
    both = combined_referral(None, None)
    assert both["refer"] is None and "neither head" in both["reason"]
    assert combined_referral(None, False)["refer"] is False
    # the forbidden-token regex over the UNMODIFIED output: stripping
    # the token before checking for it is not a test
    from tests.test_flutter_forbidden import FLUT_RE
    for v in (combined_referral(True, True), combined_referral(False, True),
              combined_referral(None, True), combined_referral(False, False),
              combined_referral(None, None)):
        assert not FLUT_RE.search(json.dumps(v)), v


def test_combined_endpoint_ranks_the_three_candidates_with_numbers():
    rows = _cohort()
    out = combined_endpoint(rows)
    c = out["candidates"]
    assert set(c) == {"afib_alone", "flutter_alone", "combined"}
    for name, d in c.items():
        assert d["sensitivity"] is not None, name
        assert d["at_deployment_prevalence"]["ppv"] is not None, name
        assert d["at_deployment_prevalence"]["prevalence"] == 0.033
    # the combined endpoint catches what either head alone misses — but
    # that is an IDENTITY (it is an OR of the two), not a finding, which
    # is exactly why it cannot be the ranking criterion
    assert c["combined"]["sensitivity"] >= c["afib_alone"]["sensitivity"]
    assert c["combined"]["sensitivity"] >= \
        c["flutter_alone"]["sensitivity"]
    assert out["ranked_by"].startswith("false_alarms_per_true_case")
    # below the pre-registered sensitivity floor nothing is recommended,
    # and the harness says why rather than crowning the OR by default
    assert out["recommended"] is None
    assert "cannot be made on the numbers" in \
        out["no_recommendation_reason"]
    assert "not shippable" in out["note"]


def test_the_endpoint_ranking_can_refuse_to_crown_the_or_fusion():
    """A sensitivity-first rank always picks `combined`, because it is
    an OR of the other two. The burden criterion must be able to pick
    something else when the OR buys its extra catch with false alarms."""
    rows = _cohort()
    # make the flag noisy: it now flags every sinus-tach scan too, so
    # combining it with the AF head costs far more than it catches
    noisy = [dict(r, head_flag=(True if r["cohort"] == "sinus_tachycardia"
                                and r["head_flag"] is not None
                                else r["head_flag"])) for r in rows]
    out = combined_endpoint(noisy, min_sensitivity=0.05)
    c = out["candidates"]
    assert c["combined"]["sensitivity"] >= c["afib_alone"]["sensitivity"]
    burden = {n: d["at_deployment_prevalence"][
        "false_alarms_per_true_case"] for n, d in c.items()}
    assert out["recommended"] == min(burden, key=burden.get)
    assert burden[out["recommended"]] < burden["combined"] or \
        out["recommended"] == "combined"
    # PPV must be projected, never the cohort's own
    ppv = c["combined"]["at_deployment_prevalence"]["ppv"]
    assert 0.0 < ppv < 0.5, ppv


# ------------------------------------------------- the whole evaluation
def test_evaluation_writes_a_run_and_blocks_on_thin_evidence(tmp_path):
    out = evaluate_flutter_dataset(_cohort(), runs_root=tmp_path,
                                   signal_domain="synthetic")
    assert out["run_id"].startswith("flut-")
    run = pathlib.Path(out["run_dir"])
    for f in ("gate_results.json", "flutter_record.json", "model.json"):
        assert (run / f).exists(), f
    v = out["verdict"]
    assert v["promotion_open"] is False          # synthetic + thin
    reds = {g["gate"] for g in v["gates"] if g["status"] == "RED"}
    assert reds == set(g["gate"] for g in v["gates"])
    # the scoreboard entry is readable by the gate module
    from evaluation.flutter_gates import (flutter_gate_status,
                                          latest_scoreboard_entry)
    e = latest_scoreboard_entry("evaluation", runs_root=tmp_path)
    assert e["run_id"] == out["run_id"]
    doc = flutter_gate_status(runs_root=tmp_path)
    assert doc["evaluation_run"] == out["run_id"]
    assert doc["promotion"] == "BLOCKED"
    # ... and every §F evidence block is present, so the reds are the
    # gates' own criteria and not missing data
    for k in ("f0", "f1", "f2", "f3", "f4", "f5"):
        assert out["evidence"][k], k


def test_evaluation_refuses_an_empty_cohort(tmp_path):
    with pytest.raises(ValueError, match="not an evaluation"):
        evaluate_flutter_dataset([], runs_root=tmp_path,
                                 signal_domain="synthetic")


def test_disjointness_flags_are_computed_not_asserted(tmp_path):
    one = [_row("solo", "flutter_2to1", bpm=150.0, rmssd=4.0,
                coupling=0.03, rhythm="ATRIAL_FLUTTER",
                ratio="TWO_TO_ONE", flag=True)]
    out = evaluate_flutter_dataset(one, runs_root=tmp_path,
                                   signal_domain="synthetic")
    d = out["evidence"]["data"]
    assert d["participant_disjoint"] is False
    assert d["session_disjoint"] is False
    assert d["n_participants"] == 1


# --------------------------------------------------- end to end
def test_end_to_end_on_synthetic_fixtures(tmp_path):
    """The whole chain on REAL video: generate a labeled cohort, run the
    production pipeline, build rows through the actual head, evaluate,
    and land a BLOCKED verdict for the right reasons."""
    pytest.importorskip("cv2")
    from evaluation.flutter_metrics import rows_from_dataset
    from scripts.make_synth_flutter import make_flutter_dataset
    d = tmp_path / "ds"
    make_flutter_dataset(d, duration_s=40.0, serial_participants=1,
                         cohorts=["flutter_2to1", "sinus_tach",
                                  "flutter_4to1"])
    rows = rows_from_dataset(d)
    assert len(rows) == 6                       # 3 cohorts + 3 serial
    by_cohort = {r["recording_id"]: r for r in rows}
    assert all(r["scan_outcome"] == "ACCEPT" for r in rows), \
        {r["recording_id"]: r["scan_outcome"] for r in rows}
    # the ECG label rode through, including the v0.6 ratio field
    flut = [r for r in rows if r["rhythm"] == "ATRIAL_FLUTTER"]
    assert flut and all(r["conduction_ratio"] for r in flut)
    # the rate came out right during arrhythmia
    acc = rate_accuracy(rows)
    assert acc["n_arrhythmia_scans"] >= 3
    assert abs(acc["bias_bpm"]) < 3.0, acc
    # respiration reached the head, so it did NOT have to abstain — and
    # each cohort's verdict is asserted BY NAME. "at least one row was
    # judged" would pass for a head that flagged everything or nothing.
    got = {r["recording_id"]: r["head_flag"] for r in rows}
    for rid, flag in got.items():
        assert flag is not None, (rid, got)
    for rid, want in got.items():
        if "flutter_2to1" in rid:
            assert got[rid] is True, (rid, got)     # the target
        elif "sinus_tach" in rid:
            assert got[rid] is False, (rid, got)    # the confounder
        elif "flutter_4to1" in rid:
            assert got[rid] is False, (rid, got)    # the known miss
    out = evaluate_flutter_dataset(rows, runs_root=tmp_path / "runs",
                                   signal_domain="synthetic",
                                   production_path=True)
    assert out["verdict"]["promotion_open"] is False
    # every gate red, and the surrogate-domain disqualifier is named
    for g in out["verdict"]["gates"]:
        assert g["status"] == "RED"
        assert any("machinery evidence only" in r for r in g["reasons"])
    # the per-ratio table carries the known miss as a measured number
    per = out["evidence"]["f2"]["per_ratio"]
    assert "TWO_TO_ONE" in per and "FOUR_TO_ONE" in per


def test_cli_evaluate_flutter(tmp_path):
    pytest.importorskip("cv2")
    import os
    import subprocess
    from datasets.registry import register_dataset
    from scripts.make_synth_flutter import make_flutter_dataset
    d = tmp_path / "ds"
    # 40 s: below ~20 s of analysable span the respiration reader and
    # the tachogram both refuse, and every scan abstains
    make_flutter_dataset(d, duration_s=40.0, serial_participants=0,
                         cohorts=["flutter_2to1", "sinus_tach"])
    reg = tmp_path / "registry.jsonl"
    env = dict(os.environ, AVATARX_REGISTRY=str(reg),
               AVATARX_FLUTTER_RUNS=str(tmp_path / "runs"))
    register_dataset(d, license_class="internal_consented",
                     consent_class="research_v1", registry_path=reg)
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "evaluate-flutter", str(d),
                        "--domain", "synthetic"],
                       capture_output=True, text=True, timeout=900,
                       env=env)
    assert r.returncode == 0, r.stderr[-2500:]
    doc = json.loads(r.stdout)
    assert doc["promotion"] == "BLOCKED"
    assert doc["n_scans"] == 2
    assert "RESEARCH ARTIFACT" in doc["WATERMARK"]
    assert doc["shipping_rule"] in ("head", "b3", "undecided")
    # an unregistered directory is refused before any compute
    r2 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "evaluate-flutter", str(tmp_path / "nope")],
                        capture_output=True, text=True, timeout=300,
                        env=env)
    assert r2.returncode != 0


def test_a_cohort_the_flag_declined_entirely_is_a_result_not_a_crash():
    """Every scan abstained (no respiration channel anywhere). The
    evaluation must complete and report that, not raise: a detector
    that declined the whole cohort is informative."""
    rows = [_row(f"p{i}", "sinus_tachycardia", bpm=150.0, rmssd=20.0,
                 coupling=None, rhythm="SINUS_TACHYCARDIA", flag=None)
            for i in range(6)]
    f = flag_performance(rows)
    assert f["no_read_rate"] == 1.0
    assert f["sensitivity_2to1"] is None
    assert f["specificity_battery"] is None
    b = hard_negative_battery(rows)
    assert b["per_class"]["sinus_tachycardia"]["false_positive_rate"] \
        is None
    assert b["per_class"]["sinus_tachycardia"]["abstention_rate"] == 1.0


def test_battery_classes_key_on_the_adjudicated_rhythm_by_default():
    """Through the real path there is no fixture 'cohort' field — the
    class must come from the ECG label, and its spelling must match the
    gates' pre-registered dominant-negative name or the gate would red
    forever with 'no share recorded' and look like missing data."""
    rows = [dict(r) for r in _cohort()]
    for r in rows:
        r.pop("cohort")                       # as rows_from_dataset gives
    b = hard_negative_battery(rows)
    assert "sinus_tachycardia" in b["per_class"]
    assert "svt" in b["per_class"] and "paced" in b["per_class"]
    assert "afib" in b["per_class"]
    from evaluation.flutter_gates import load_flutter_gates
    dominant = load_flutter_gates()["f1_flag_performance"][
        "require_dominant_negative"]
    assert dominant in b["negative_class_share"], (
        dominant, sorted(b["negative_class_share"]))
    # the gate reads that share and must not report it as missing
    f1 = flag_performance(rows)
    row = None
    from evaluation.flutter_gates import evaluate_flutter_gates
    ev = {"data": {"signal_domain": "facial_rppg",
                   "participant_disjoint": True, "session_disjoint": True,
                   "production_path": True}, "f1": f1}
    for g in evaluate_flutter_gates(load_flutter_gates(), ev)["gates"]:
        if g["gate"] == "f1":
            row = g
    assert not any("no sinus_tachycardia share" in r
                   for r in row["reasons"]), row["reasons"]


def test_the_battery_at_the_frame_rate_the_product_actually_uses():
    """The harness fixtures default to a high fps so the PHYSIOLOGICAL
    floor is exercised. Production is 30 fps, where the camera's own
    quantization floor (25 ms) binds instead — the regime the whole
    disclosure story is about, and it must be measured too."""
    fast = flag_performance(_cohort())
    slow = flag_performance([dict(r, fps=30.0) for r in _cohort()])
    # the head's verdicts are fixture-set, so its numbers do not move;
    # B3 recomputes the floor from fps and DOES move
    assert slow["b3_specificity"] is not None
    assert slow["b3_specificity"] <= fast["b3_specificity"], (
        slow["b3_specificity"], fast["b3_specificity"])
    # at 30 fps a 20 ms sinus-tach dispersion now reads "below floor",
    # so the coupling term is the only thing left holding B3 up
    from evaluation.flutter_metrics import b3_interpretable
    borderline = _row("x", "sinus_tachycardia", bpm=150.0, rmssd=20.0,
                      coupling=0.85, rhythm="SINUS_TACHYCARDIA",
                      fps=30.0)
    assert b3_interpretable(borderline) is False        # coupling saves it
    no_resp = dict(borderline)
    no_resp["features"]["regularity"]["coupling"][
        "tachogram_resp_fraction"] = None
    assert b3_interpretable(no_resp) is True            # ... and without
                                                        # it, B3 flags


def test_b4_uses_demographics_that_actually_vary():
    """A baseline called demographics+rate must have demographics: with
    a constant age and no sex it is B1 refit under another name."""
    rows = []
    for i in range(12):
        rows.append(_row(f"d{i}", "flutter_2to1", bpm=150.0, rmssd=4.0,
                         coupling=0.03, rhythm="ATRIAL_FLUTTER",
                         ratio="TWO_TO_ONE", flag=True,
                         age=40 + 3 * i))
        rows[-1]["sex"] = "F" if i % 2 else "M"
        rows.append(_row(f"e{i}", "sinus_tachycardia", bpm=151.0,
                         rmssd=26.0, coupling=0.85,
                         rhythm="SINUS_TACHYCARDIA", flag=False,
                         age=30 + 4 * i))
        rows[-1]["sex"] = "M" if i % 2 else "F"
    ages = {r["age_years"] for r in rows}
    sexes = {r.get("sex") for r in rows}
    assert len(ages) > 5 and sexes == {"M", "F"}
    f = flag_performance(rows, target_sensitivity=0.8)
    assert f.get("b4_specificity_at_matched_sens") is not None, f.get(
        "b4_reason")
    assert 0.0 <= f["b4_specificity_at_matched_sens"] <= 1.0
    assert f.get("b4_n_test_rows", 0) > 0


def test_participant_metadata_reaches_the_rows_so_f4_can_be_rated(tmp_path):
    """Skin-tone group, age and sex are PARTICIPANT data, not Recording
    data. Without a sidecar carrying them the fairness gate had no
    subgroup to rate and every generated cohort read 'unrated' — which
    looks identical to a cohort that genuinely lacks diversity."""
    pytest.importorskip("cv2")
    from evaluation.flutter_metrics import rows_from_dataset
    from scripts.make_synth_flutter import write_scan
    for i, fitz in enumerate((2, 6)):
        write_scan(tmp_path, f"r_p{i}", "flutter_2to1", pid=f"p{i}",
                   session_id=f"p{i}-s1", seed=5 + i, duration_s=40.0,
                   fitzpatrick=fitz)
    rows = rows_from_dataset(tmp_path)
    assert len(rows) == 2
    assert sorted(r["fitzpatrick_group"] for r in rows) == [2, 6]
    assert all(r["age_years"] is not None for r in rows)
    assert all(r.get("sex") in ("F", "M") for r in rows)
    f = fairness(rows, min_group_participants=1)["fitzpatrick_group"]
    assert f["darkest_band_present"] is True
    assert f["n_rated_groups"] == 2
    assert f["detection_parity_ratio_worst"] is not None
