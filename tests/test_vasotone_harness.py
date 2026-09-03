"""v0.5 vasotone T3/T4 — the null-arm study (a lamp-responding feature
is dropped by config), W0 reference tracking, the T2 baseline battery
both ways (a real tone response adds information beyond HR+respiration;
an HR-shortcut world does not), dose/consistency, fairness, and the
end-to-end dataset evaluation with an honest BLOCKED verdict."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

from evaluation.vasotone_metrics import (BASELINES, PRIMARY_FEATURE,
                                         battery, dose_consistency,
                                         fairness, null_arm_study,
                                         reference_tracking)

FEATS = ("norm_pulse_amplitude", "vasomotor_lf_power", "notch_rel_amp")


def _row(pid, maneuver, dn, *, pi=None, dhr=None, dresp=None,
         dmotion=0.0, sess="s1", start="2026-09-01T09:00:00Z",
         intensity=None, fitz=3):
    feats = {name: {"baseline": 1.0, "response": 1.0 + (dn or 0.0),
                    "delta": dn, "delta_norm": dn}
             for name in FEATS}
    r = {"recording_id": f"{pid}-{maneuver}-{sess}",
         "participant_id": pid, "session_id": f"{pid}-{sess}",
         "video_start_utc": start, "maneuver": maneuver,
         "intensity": intensity, "fitzpatrick_group": fitz,
         "features": feats,
         "covariates": {"dhr_bpm": dhr, "dresp_brpm": dresp,
                        "dmotion": dmotion}}
    if pi is not None:
        r["pi_response"] = {"baseline": 2.0, "response": 2.0 * (1 + pi),
                            "delta": 2.0 * pi, "delta_norm": pi}
    if maneuver not in ("null_optics", "null_rest"):
        r["facial_response"] = dn
    return r


def test_null_arm_study_p95s_and_survival_rule():
    from evaluation.vasotone_gates import (load_vasotone_gates,
                                           survivors_from_study)
    rng = np.random.default_rng(0)
    # amplitude quiet under lamps; lf_power responds to lamps hard
    nopt = [_row(f"p{i}", "null_optics",
                 float(rng.normal(0, 0.02))) for i in range(25)]
    for r in nopt:
        r["features"]["vasomotor_lf_power"]["delta_norm"] = \
            float(rng.normal(0.4, 0.1))
    nrest = [_row(f"p{i}", "null_rest",
                  float(rng.normal(0, 0.02))) for i in range(25)]
    w1 = null_arm_study(nopt, nrest, FEATS)
    assert w1["n_null_optics_sessions"] == 25
    per = w1["per_feature"]
    assert per["vasomotor_lf_power"]["null_optics_p95"] > 0.2
    assert per[PRIMARY_FEATURE]["null_optics_p95"] < 0.08
    surv = survivors_from_study(w1, load_vasotone_gates())
    assert PRIMARY_FEATURE in surv
    assert "vasomotor_lf_power" not in surv       # the lamp detector


def test_reference_tracking_agreement_and_floors():
    rows = [_row(f"p{i}", "cold_pressor", -0.3 - 0.02 * i, pi=-0.28)
            for i in range(8)]
    rows.append(_row("p9", "cold_pressor", 0.2, pi=-0.3))  # disagrees
    rows.append(_row("p10", "cold_pressor", -0.01, pi=-0.3))  # below floor
    w0 = reference_tracking(rows, 0.05, 0.05)
    assert w0["n_provocations"] == 10
    assert abs(w0["direction_agreement"] - 0.8) < 1e-9
    # magnitude r is WITHIN-maneuver (all rows are cold_pressor here)
    assert w0["magnitude_r"] is not None
    assert w0["per_maneuver_r"]["cold_pressor"]["n"] == 10


def test_magnitude_r_is_within_maneuver_not_between():
    """Review finding: maneuver-mean separation alone must not satisfy
    W0 — rows with ZERO within-maneuver tracking but well-separated
    maneuver means must not produce a high magnitude_r."""
    rng = np.random.default_rng(5)
    rows = []
    for i in range(12):     # cold_pressor: big responses, no tracking
        rows.append(_row(f"p{i}", "cold_pressor",
                         -0.4 + float(rng.normal(0, 0.05)),
                         pi=-0.4 + float(rng.normal(0, 0.05))))
    for i in range(12):     # paced_breathing: small, no tracking
        rows.append(_row(f"q{i}", "paced_breathing",
                         -0.1 + float(rng.normal(0, 0.05)),
                         pi=-0.1 + float(rng.normal(0, 0.05))))
    w0 = reference_tracking(rows, 0.05, 0.05)
    # pooled-across-maneuvers r would be ~0.8 from the mean gap alone;
    # the within-maneuver combined r stays near zero
    assert abs(w0["magnitude_r"]) < 0.45, w0


def test_battery_informative_vs_hr_shortcut_worlds():
    rng = np.random.default_rng(2)
    # informative: PI response has an HR-independent component that the
    # facial response tracks
    rows = []
    for i in range(40):
        dhr = float(rng.normal(8, 3))
        extra = float(rng.normal(0, 0.1))
        pi = -0.02 * dhr + extra + float(rng.normal(0, 0.02))
        face = 0.9 * extra - 0.02 * dhr + float(rng.normal(0, 0.03))
        rows.append(_row(f"p{i:02d}", "cold_pressor", face, pi=pi,
                         dhr=dhr, dresp=float(rng.normal(2, 1))))
    b = battery(rows)
    assert b["available"] and set(b["models"]) == set(BASELINES)
    assert b["added_r2_over_b3"] > 0.3
    # the CI is over the SIGNED partial correlation (review finding:
    # a squared statistic's CI can never straddle zero)
    assert b["added_r_ci95"] and b["added_r_ci95"][0] > 0
    # HR-shortcut world: the face carries ONLY the HR response — no
    # information beyond B3 may be claimed
    rows2 = []
    for i in range(40):
        dhr = float(rng.normal(8, 3))
        pi = -0.02 * dhr + float(rng.normal(0, 0.1))
        face = -0.02 * dhr + float(rng.normal(0, 0.03))
        rows2.append(_row(f"q{i:02d}", "cold_pressor", face, pi=pi,
                          dhr=dhr, dresp=float(rng.normal(2, 1))))
    b2 = battery(rows2)
    assert b2["available"]
    assert b2["added_r2_over_b3"] < 0.15, b2
    # the signed CI must NOT exclude zero in the shortcut world — this
    # is the null-calibration check the squared CI could never fail
    assert (b2["added_r_ci95"] is None or b2["added_r_ci95"][0] <= 0.0)


def test_battery_refuses_covariate_starvation():
    rng = np.random.default_rng(8)
    rows = [_row(f"p{i:02d}", "cold_pressor",
                 float(rng.normal(-0.2, 0.05)),
                 pi=float(rng.normal(-0.2, 0.05)),
                 dhr=float(rng.normal(8, 3)), dresp=None)
            for i in range(30)]
    b = battery(rows)
    assert b["available"] is False
    assert "covariate coverage" in b["reason"]


def test_dose_consistency_retest_and_ordering():
    rows = []
    for i in range(6):
        tau = -0.25 - 0.02 * i
        rows.append(_row(f"p{i}", "cold_pressor", tau, sess="s1",
                         start="2026-09-01T09:00:00Z", intensity=2))
        rows.append(_row(f"p{i}", "cold_pressor", tau + 0.02, sess="s2",
                         start="2026-09-02T09:00:00Z", intensity=2))
        # graded: intensity 1 weaker than intensity 3
        rows.append(_row(f"p{i}", "paced_breathing", tau * 0.4,
                         sess="s1", intensity=1))
        rows.append(_row(f"p{i}", "paced_breathing", tau * 0.9,
                         sess="s1", intensity=3))
    w3 = dose_consistency(rows)
    assert w3["n_retest_pairs"] == 6
    assert w3["retest_icc"] is not None and w3["retest_icc"] > 0.8
    assert w3["ordered_fraction"] == 1.0


def test_fairness_detection_and_coverage_parity():
    scans = [{"recording_id": f"r{i}", "fitzpatrick_group": 2 + i % 2,
              "usable": i % 5 != 0} for i in range(20)]
    rows = [_row(f"p{i}", "cold_pressor",
                 -0.3 if i % 2 == 0 else -0.01, fitz=2 + i % 2)
            for i in range(12)]
    w4 = fairness(scans, rows, 0.05)
    axis = w4["per_axis"]["fitzpatrick_group"]
    assert "detection_ratio_worst" in axis
    assert axis["per_group"]["2"]["detection"] == 1.0
    assert axis["per_group"]["3"]["detection"] == 0.0
    assert axis["detection_ratio_worst"] == 0.0


@pytest.fixture(scope="module")
def vaso_ds(tmp_path_factory):
    pytest.importorskip("cv2")
    from scripts.make_synth_vasotone import make_vasotone_dataset
    d = tmp_path_factory.mktemp("vaso_eval")
    manifest = make_vasotone_dataset(d, n_participants=3, seed=23,
                                     retest_participants=1)
    return d, manifest


def test_evaluate_vasotone_dataset_end_to_end(vaso_ds, tmp_path):
    from evaluation.vasotone_metrics import evaluate_vasotone_dataset
    from models.registry import promote, PromotionRefused
    d, manifest = vaso_ds
    out = evaluate_vasotone_dataset(
        d, out_dir=tmp_path / "rep", runs_root=tmp_path / "runs",
        surviving_override=[PRIMARY_FEATURE, "notch_rel_amp"])
    doc = out["report"]
    # every arm is present and the uncontrolled-optics session is
    # excluded with its W-d reason
    assert doc["arms"]["null_optics"] == 3
    assert doc["arms"]["null_rest"] == 3
    assert doc["arms"]["provocations"] >= 4
    uc = manifest["uncontrolled"][0]
    assert any(e["recording_id"] == uc and "W-d" in e["reasons"][0]
               for e in doc["excluded"])
    # the null-arm study measured both distributions per feature
    per = doc["null_arm_study"]["per_feature"]
    assert per[PRIMARY_FEATURE]["null_optics_p95"] is not None
    assert per[PRIMARY_FEATURE]["null_rest_p95"] is not None
    # W0 scored the provocations against the contact reference
    assert doc["reference_tracking"]["n_provocations"] >= 4
    # verdict: synthetic domain -> BLOCKED, every gate RED
    assert doc["gates"]["promotion"] == "BLOCKED"
    assert all(g["status"] == "RED"
               for g in doc["gates"]["verdict"]["gates"])
    # run artifacts + scoreboard + promote refusal close the loop
    run = pathlib.Path(out["run_dir"])
    rec = json.loads((run / "vasotone_record.json").read_text())
    assert list(rec)[0] == "WATERMARK"
    sb = (tmp_path / "runs" / "scoreboard.jsonl").read_text().splitlines()
    assert json.loads(sb[-1])["promotion_open"] is False
    with pytest.raises(PromotionRefused, match="§W"):
        promote(run, registry_path=tmp_path / "m.jsonl",
                spec_path=tmp_path / "s.md")
    md = pathlib.Path(out["report_md"]).read_text()
    assert "null_optics" in md and "survives" in md
    # W-c: nothing in the report carries an absolute tone level — every
    # feature entry is the 4-key delta shape
    for e in doc["null_arm_study"]["per_feature"].values():
        assert set(e) == {"null_optics_p95", "null_rest_p95",
                          "n_null_optics_values", "n_null_rest_values"}


def test_config_driven_survival_blocks_scoring_on_tiny_null_arms(
        vaso_ds, tmp_path):
    """The REAL path (no override): 3 null sessions < the 20-session
    floor, so nothing survives, the primary response never exists, and
    W0/W2 are blocked fail-closed — the v0.4 chain, tone edition."""
    from evaluation.vasotone_metrics import evaluate_vasotone_dataset
    d, _ = vaso_ds
    out = evaluate_vasotone_dataset(d, out_dir=tmp_path / "rep0",
                                    runs_root=tmp_path / "runs0")
    doc = out["report"]
    assert doc["surviving_features"] == []
    assert doc["reference_tracking"].get("direction_agreement") in (
        None, 0.0)
    assert doc["battery"]["available"] is False
    v = {g["gate"]: g for g in doc["gates"]["verdict"]["gates"]}
    assert any("optics-response thresholds" in r or
               "null_optics sessions" in r for r in v["w1"]["reasons"])


def test_cli_evaluate_vasotone_requires_registration(tmp_path):
    import os
    import subprocess
    _ROOT = pathlib.Path(__file__).resolve().parents[1]
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "x.recording.json").write_text("{}")
    (ds / "x.provocation.json").write_text("{}")
    env = dict(os.environ,
               AVATARX_REGISTRY=str(tmp_path / "reg.jsonl"),
               AVATARX_VASOTONE_RUNS=str(tmp_path / "runs"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "evaluate-vasotone", str(ds)],
                       capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode == 2 and "register" in r.stderr.lower()
    r2 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "register-dataset", str(ds)],
                        capture_output=True, text=True, timeout=120,
                        env=env)
    assert r2.returncode == 0, r2.stderr
    r3 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "evaluate-vasotone", str(ds)],
                        capture_output=True, text=True, timeout=120,
                        env=env)
    assert r3.returncode == 2
    assert "evaluate-vasotone failed" in r3.stderr


def test_config_driven_survival_on_the_real_pipeline_path(vaso_ds,
                                                          tmp_path):
    """Review finding: the survival rule was only exercised with
    hand-fed evidence or an override. With the session floors lowered
    to what the fixture can supply, the REAL measured null-arm p95s
    must decide the surviving set — and the downstream scoring must be
    CONSISTENT with that decision either way."""
    from evaluation.vasotone_gates import (load_vasotone_gates,
                                           survivors_from_study)
    from evaluation.vasotone_metrics import evaluate_vasotone_dataset
    d, _ = vaso_ds
    gates = tmp_path / "gates.yaml"
    text = (pathlib.Path(__file__).resolve().parents[1] / "configs"
            / "gates.yaml").read_text()
    text = text.replace("min_null_optics_sessions: 20",
                        "min_null_optics_sessions: 3")
    text = text.replace("min_null_rest_sessions: 20",
                        "min_null_rest_sessions: 3")
    gates.write_text(text)
    out = evaluate_vasotone_dataset(d, out_dir=tmp_path / "rep",
                                    runs_root=tmp_path / "runs",
                                    gates_path=gates)
    doc = out["report"]
    w1 = doc["null_arm_study"]
    # the rule and the harness agree exactly on the measured data
    expected = survivors_from_study(w1, load_vasotone_gates(gates))
    assert doc["surviving_features"] == expected
    # measured p95s exist for the primary on both arms
    per = w1["per_feature"]
    assert per[PRIMARY_FEATURE]["null_optics_p95"] is not None
    assert per[PRIMARY_FEATURE]["null_rest_p95"] is not None
    # downstream consistency: primary survived -> provocations scored;
    # primary dropped -> no facial response, W0 honestly empty
    rt = doc["reference_tracking"]
    if PRIMARY_FEATURE in expected:
        assert rt.get("n_provocations", 0) > 0
        assert (tmp_path / "runs").exists()
        run = pathlib.Path(out["run_dir"])
        model = json.loads((run / "model.json").read_text())
        assert model["detection_floor_delta_norm"] == \
            per[PRIMARY_FEATURE]["null_rest_p95"]
    else:
        assert rt.get("direction_agreement") in (None, 0.0)
