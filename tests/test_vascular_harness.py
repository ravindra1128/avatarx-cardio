"""v0.4-vascular T3/T4 — the baseline battery and the T2 defense: on a
world where morphology carries age/BP-independent stiffness the head
beats B3; on the age-shortcut world it must NOT; V-d is structural; the
auxiliary evidence (retest, sites, fairness) computes honestly; and the
end-to-end dataset evaluation lands a BLOCKED verdict with a scoreboard
entry and a full report."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

from evaluation.vascular_metrics import (BASELINES, VascularHarnessError,
                                         battery_report,
                                         estimate_retest,
                                         fairness_tables, fit_battery,
                                         site_generalization)

SURV = ["reflection_index", "rise_time_s", "notch_rel_amp"]


def _records(n=40, seed=1, informative=True):
    """Tabular world mirror of make_synth_vascular: cfPWV = age part +
    independent part; BP tracks only the age part; morphology carries
    the full stiffness when informative, else only the age part."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        age = float(rng.uniform(22, 78))
        s_age = 4.6 + 0.075 * (age - 20)
        s_extra = float(rng.normal(0, 1.1))
        s = s_age + s_extra
        sbp = 102 + 0.45 * (age - 20) + 2.2 * (s_age - 6) \
            + rng.normal(0, 6)
        morph_s = s if informative else s_age
        feats = {"reflection_index":
                 0.32 + 0.05 * (morph_s - 8) + rng.normal(0, 0.02),
                 "rise_time_s":
                 0.20 - 0.006 * (morph_s - 8) + rng.normal(0, 0.004),
                 "notch_rel_amp":
                 0.15 + 0.03 * (morph_s - 8) + rng.normal(0, 0.01)}
        rows.append({"recording_id": f"r{i:03d}",
                     "participant_id": f"tp{i:03d}",
                     "session_id": f"tp{i:03d}-s1",
                     "video_start_utc": f"2026-08-31T{8 + i // 60:02d}:"
                                        f"{i % 60:02d}:00Z",
                     "site": "siteA" if i % 2 else "siteB",
                     "device_label": "rig-1" if i % 2 else "rig-2",
                     "fitzpatrick_group": 2 + (i % 4),
                     "age": age, "sex": "female" if i % 2 else "male",
                     "sbp": float(sbp),
                     "dbp": float(0.6 * sbp + rng.normal(0, 4)),
                     "hr": float(rng.uniform(56, 84)),
                     "features": feats,
                     "cfpwv": float(s + rng.normal(0, 0.25))})
    return rows


def test_head_beats_b3_when_morphology_carries_real_signal():
    b = fit_battery(_records(informative=True), SURV)
    assert b["available"]
    assert set(b["models"]) == set(BASELINES) | {"head"}
    assert b["rmse_improvement_vs_b3_mps"] > 0.2, b
    assert b["rmse_improvement_ci95"] and \
        b["rmse_improvement_ci95"][0] > 0
    assert b["added_r2"] > 0.05


def test_age_shortcut_world_head_does_not_beat_b3():
    """THE T2 defense: when the face carries only age-explainable
    morphology, the harness must show no incremental value — a head
    that scored here would be a covert age predictor."""
    b = fit_battery(_records(informative=False, seed=3), SURV)
    assert b["available"]
    assert b["rmse_improvement_vs_b3_mps"] < 0.2, b
    assert (b["rmse_improvement_ci95"] is None
            or b["rmse_improvement_ci95"][0] <= 0)


def test_no_surviving_features_blocks_all_modeling():
    b = fit_battery(_records(), [])
    assert b["available"] is False and "fail closed" in b["reason"]
    rep = battery_report(b)
    assert rep["available"] is False


def test_vd_is_structural_not_editorial():
    b = fit_battery(_records(), SURV)
    broken = dict(b)
    broken["models"] = {k: v for k, v in b["models"].items()
                        if k != "b3_age_sex_bp"}
    with pytest.raises(VascularHarnessError, match="V-d"):
        battery_report(broken)
    broken2 = {k: v for k, v in b.items()
               if k != "rmse_improvement_vs_b3_mps"}
    with pytest.raises(VascularHarnessError, match="B3 delta"):
        battery_report(broken2)
    # the head artifact NEVER contains demographics (T2 invariant)
    art = b["head_artifact"]
    assert set(art["features"]) == set(SURV)
    for banned in ("age", "sex", "sbp", "dbp"):
        assert banned not in art["features"]


def test_estimate_retest_orders_by_acquisition_time():
    rows = _records(n=8, seed=7)
    # retest scans: SECOND in time but FIRST lexicographically ("_a2")
    # — a manifest-order pairing would flip the drift sign
    firsts, seconds = [], []
    for r in rows[:5]:
        firsts.append(dict(r, recording_id=r["recording_id"] + "_z1"))
        seconds.append(dict(r, recording_id=r["recording_id"] + "_a2",
                            video_start_utc=r["video_start_utc"]
                            .replace(":00Z", ":30Z")))
    v2 = estimate_retest(seconds + firsts,
                         [r["cfpwv"] + 0.3 for r in rows[:5]]
                         + [r["cfpwv"] for r in rows[:5]])
    assert v2["n_retest_pairs"] == 5
    # drift is second-scan minus first-scan IN TIME: exactly +0.3
    assert v2["retest_drift_mps"] == pytest.approx(0.3, abs=1e-6)
    assert v2["estimate_retest_icc"] is not None \
        and v2["estimate_retest_icc"] > 0.9
    # a visit without distinct timestamps is dropped, not guessed
    same_t = [dict(r) for r in rows[:1]] + [
        dict(rows[0], recording_id="x2")]
    assert estimate_retest(same_t, [8.0, 8.5]) == {}


def test_site_and_fairness_evidence_values():
    rows = _records(n=30, seed=7)
    v3 = site_generalization(rows, SURV)
    assert v3["n_sites"] == 2 and "worst_site_rmse_ratio" in v3
    # device-out runs too (review finding: the gate declares devices)
    assert v3["n_devices"] == 2 and "worst_device_rmse_ratio" in v3
    scans = [{"recording_id": r["recording_id"],
              "fitzpatrick_group": r["fitzpatrick_group"],
              "device_label": r["device_label"],
              "usable": i % 7 != 0} for i, r in enumerate(rows)]
    b = fit_battery(rows, SURV)
    v4 = fairness_tables(scans, b["_test_rows"], b["_test_pred"]["head"])
    fz = v4["per_axis"]["fitzpatrick_group"]
    assert "coverage_ratio_worst" in fz and fz["per_group"]
    assert "device" in v4["per_axis"]
    # pinned values: 30 scans, groups 2..5 of size 8/8/7/7; the i%7
    # unusable scans fall 2/1/1/1 across groups
    assert fz["per_group"]["2"]["attempted"] == 8
    assert fz["per_group"]["2"]["coverage"] == pytest.approx(0.75)
    cov = [c["coverage"] for c in fz["per_group"].values()]
    assert fz["coverage_ratio_worst"] == pytest.approx(
        min(cov) / max(cov), abs=1e-3)


def test_fairness_rmse_ratio_deterministic():
    # hand-built errors: group 2 -> |err| 1.0, group 3 -> |err| 2.0
    test_rows = [{"fitzpatrick_group": 2, "device_label": "rig-1",
                  "cfpwv": 8.0}] * 2 + \
                [{"fitzpatrick_group": 3, "device_label": "rig-1",
                  "cfpwv": 8.0}] * 2
    preds = [9.0, 7.0, 10.0, 6.0]
    scans = [{"fitzpatrick_group": g, "device_label": "rig-1",
              "usable": True} for g in (2, 2, 3, 3)]
    v4 = fairness_tables(scans, test_rows, preds)
    fz = v4["per_axis"]["fitzpatrick_group"]
    assert fz["per_group"]["2"]["rmse"] == pytest.approx(1.0)
    assert fz["per_group"]["3"]["rmse"] == pytest.approx(2.0)
    assert fz["rmse_ratio_worst"] == pytest.approx(2.0)


def test_leave_one_out_folds_are_participant_disjoint():
    # a participant recorded at BOTH sites must never train the fold
    # that scores their own held-out site (review finding)
    rows = _records(n=20, seed=9)
    straddler = dict(rows[0], recording_id="r_straddle",
                     site="siteA" if rows[0]["site"] == "siteB"
                     else "siteB")
    from evaluation.vascular_metrics import _leave_one_out
    import evaluation.vascular_metrics as vm
    seen = {}
    orig = vm._ridge

    def spy(Xtr, ytr, Xte, *a, **kw):
        seen["n_tr"] = len(ytr)
        return orig(Xtr, ytr, Xte, *a, **kw)

    vm._ridge = spy
    try:
        out = _leave_one_out(rows + [straddler], SURV,
                             "site", base=1.0)
    finally:
        vm._ridge = orig
    assert "worst_site_rmse_ratio" in out
    # the straddler's OTHER-site row is excluded from training when
    # their own site is held out: train sizes are below the naive count
    assert seen["n_tr"] < len(rows) + 1


@pytest.fixture(scope="module")
def vasc_ds(tmp_path_factory):
    pytest.importorskip("cv2")
    from scripts.make_synth_vascular import make_vascular_dataset
    d = tmp_path_factory.mktemp("vasc_eval")
    # 8 participants (7 labeled) so the participant-disjoint split has
    # a real train/test side and the battery is genuinely available —
    # review finding: the old 5-participant fixture made every battery
    # assertion dead code behind an `if available`
    manifest = make_vascular_dataset(d, n_participants=8, seed=17,
                                     duration_s=20.0,
                                     retest_participants=2)
    return d, manifest


def test_evaluate_vascular_dataset_end_to_end(vasc_ds, tmp_path):
    from evaluation.vascular_metrics import evaluate_vascular_dataset
    from models.registry import promote, PromotionRefused
    d, manifest = vasc_ds
    out = evaluate_vascular_dataset(
        d, out_dir=tmp_path / "rep", runs_root=tmp_path / "runs",
        surviving_override=SURV)
    doc = out["report"]
    # the unlabeled participant is EXCLUDED with its reason
    no_pwv = [p for p in manifest["participants"] if not p["has_pwv"]][0]
    assert any(e["recording_id"] == no_pwv["recordings"][0]
               and "cfPWV" in e["reasons"][0] for e in doc["excluded"])
    assert doc["n_included"] >= 4
    # verdict: synthetic domain -> BLOCKED, every gate RED
    assert doc["gates"]["promotion"] == "BLOCKED"
    assert all(g["status"] == "RED"
               for g in doc["gates"]["verdict"]["gates"])
    # V-d: the battery is AVAILABLE on this fixture and complete —
    # unconditional (review finding: conditional asserts were dead code)
    b = doc["battery"]
    assert b.get("available"), b
    assert set(b["models"]) == set(BASELINES) | {"head"}
    assert "rmse_improvement_vs_b3_mps" in b and "added_r2" in b
    # the written report never leaks private per-participant fields
    assert "_test_rows" not in b and "_test_pred" not in b
    assert "head_artifact" not in b
    report_text = pathlib.Path(out["report_json"]).read_text()
    assert "_test_rows" not in report_text
    md = pathlib.Path(out["report_md"]).read_text()
    assert "b3_age_sex_bp" in md
    # run dir + scoreboard + promote refusal close the loop
    run = pathlib.Path(out["run_dir"])
    assert (run / "vascular_record.json").exists()
    sb = (tmp_path / "runs" / "scoreboard.jsonl").read_text().splitlines()
    assert json.loads(sb[-1])["promotion_open"] is False
    with pytest.raises(PromotionRefused, match="vascular"):
        promote(run, registry_path=tmp_path / "m.jsonl",
                spec_path=tmp_path / "s.md")


def test_evaluate_vascular_dataset_config_driven_path(vasc_ds, tmp_path):
    """The REAL surviving-features path (no override): an empty runs
    root means nothing survives and the battery is blocked; a seeded
    QUALIFIED facial fidelity entry hands the harness its feature set —
    config-driven end to end (review finding: only the override path
    was tested)."""
    from evaluation.vascular_gates import append_scoreboard
    from evaluation.vascular_metrics import evaluate_vascular_dataset
    d, _ = vasc_ds
    runs = tmp_path / "runs_empty"
    out = evaluate_vascular_dataset(d, out_dir=tmp_path / "rep0",
                                    runs_root=runs)
    b = out["report"]["battery"]
    assert b["available"] is False and "fail closed" in b["reason"]
    assert out["report"]["surviving_features"] == []
    runs2 = tmp_path / "runs_seeded"
    per = {f: {"icc": 0.9, "retest_icc": 0.8}
           for f in ("reflection_index", "rise_time_s")}
    append_scoreboard(
        {"kind": "fidelity", "run_id": "fid-seed",
         "evidence": {"v0": {"per_feature": per,
                             "n_paired_participants": 80},
                      "data": {"signal_domain": "facial_rppg",
                               "participant_disjoint": True,
                               "session_disjoint": True,
                               "production_path": True}}},
        runs_root=runs2)
    out2 = evaluate_vascular_dataset(d, out_dir=tmp_path / "rep1",
                                     runs_root=runs2)
    assert out2["report"]["surviving_features"] ==         ["reflection_index", "rise_time_s"]
    assert out2["report"]["battery"]["available"] is True
    # the eval run's own gate_results carries the merged v0 evidence
    gr = json.loads((pathlib.Path(out2["run_dir"])
                     / "gate_results.json").read_text())
    assert gr["evidence"]["v0"]["n_paired_participants"] == 80


def test_cli_evaluate_vascular_requires_registration(tmp_path):
    import os
    import subprocess
    _ROOT = pathlib.Path(__file__).resolve().parents[1]
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "x.recording.json").write_text("{}")
    (ds / "x.pwv.json").write_text("{}")
    env = dict(os.environ,
               AVATARX_REGISTRY=str(tmp_path / "reg.jsonl"),
               AVATARX_VASCULAR_RUNS=str(tmp_path / "runs"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "evaluate-vascular", str(ds)],
                       capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode == 2 and "register" in r.stderr.lower()
    r2 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "register-dataset", str(ds)],
                        capture_output=True, text=True, timeout=120,
                        env=env)
    assert r2.returncode == 0, r2.stderr
    r3 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "evaluate-vascular", str(ds)],
                        capture_output=True, text=True, timeout=120,
                        env=env)
    assert r3.returncode == 2
    assert "evaluate-vascular failed" in r3.stderr
