"""v0.4 T6 — the §6 fitness harness: agreement metrics, the mandatory
baseline ladder on identical participant-disjoint splits, the
falsification probes, respiration, and the end-to-end CPET-labeled
evaluation whose §V verdict lands on the fitness scoreboard."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess

import numpy as np
import pytest

from evaluation.fitness_metrics import (LADDER, bland_altman, fit_ladder,
                                        see, shuffled_workload_test,
                                        transition_sensitivity,
                                        vo2_category)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_agreement_metrics():
    y = [30.0, 40.0, 50.0]
    assert see(y, y) == 0.0
    assert see(y, [32.0, 42.0, 52.0]) == 2.0
    ba = bland_altman(y, [32.0, 41.0, 49.0])
    assert abs(ba["bias"] - 2 / 3) < 1e-3     # 3-dp rounding in the dict
    assert ba["loa_lower"] < ba["bias"] < ba["loa_upper"]
    assert vo2_category(50.0, 25) == "above"
    assert vo2_category(20.0, 25) == "below"
    assert vo2_category(30.0, 65) == "typical"


def _records(n=36, seed=1, informative=True):
    """Synthetic ladder records: vo2 driven by age + workload + HRR60
    (when informative) so recovery physiology genuinely adds value;
    static-face features are pure appearance noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        age = float(rng.integers(22, 70))
        mets = float(rng.uniform(3.0, 9.0))
        # the incremental recovery signal must be INDEPENDENT of the
        # demographics rungs, or the ladder has nothing real to find
        # (an age-collinear HRR adds ~nothing beyond age — which is
        # precisely the §V3 null hypothesis, not this test's premise)
        hrr60 = float(np.clip(30.0 + rng.normal(0, 8), 5, 55))
        vo2 = 58.0 - 0.30 * age + 1.2 * mets \
            + (0.35 * hrr60 if informative else 0.0) + rng.normal(0, 0.8)
        rows.append({"participant_id": f"L{i:03d}", "age": age,
                     "sex": "female" if i % 2 else "male",
                     "weight_kg": 70.0 + rng.normal(0, 8),
                     "height_cm": 172.0, "activity_ipaq": "moderate",
                     "est_mets": mets, "hr_rest_bpm": 60.0 + 0.2 * age,
                     "rr_rest_brpm": 14.0,
                     "hr_end_proxy_bpm": 190.0 - 0.7 * age,
                     "hrr30_bpm": hrr60 * 0.55, "hrr60_bpm": hrr60,
                     "recovery_slope_bpm_min": -hrr60,
                     "static_face": list(rng.normal(0, 1, 6)),
                     "vo2": float(np.clip(vo2, 15, 65))})
    return rows


def test_ladder_finds_real_recovery_signal_and_static_control_loses():
    lad = fit_ladder(_records())
    assert lad["available"]
    assert set(lad["rungs"]) == {name for name, _ in LADDER}
    assert lad["delta_see_mlkgmin"] > 0.3, lad["delta_see_mlkgmin"]
    assert lad["delta_see_ci95"] is not None
    assert lad["beats_static_face_control"] is True
    # every rung on the SAME disjoint split
    assert lad["n_train"] + lad["n_test"] == 36


def test_ladder_reports_nothing_when_signal_absent():
    lad = fit_ladder(_records(informative=False, seed=3))
    assert lad["available"]
    assert lad["delta_see_mlkgmin"] < 0.3      # nothing beyond demo+activity


def test_shuffled_workload_probe_degrades():
    out = shuffled_workload_test(_records(seed=2))
    assert out["available"] and out["degrades"] is True
    assert out["see_full_shuffled_workload"] > out["see_full"]
    # review finding: an UNCHANGED SEE must read as NOT degraded — a
    # model that never used the workload is what the probe exists for
    same = [dict(r, est_mets=5.0) for r in _records(seed=4)]
    out2 = shuffled_workload_test(same)         # constant workload:
    assert out2["degrades"] is False            # shuffle changes nothing


def test_ridge_is_unbiased_on_perfectly_predictable_labels():
    """Review finding: the old ridge penalized the intercept against
    uncentred VO2, shrinking every prediction by ~mean(y)/(n+1)."""
    from evaluation.fitness_metrics import _ridge
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, (8, 2))
    Xte = rng.normal(0, 1, (5, 2))
    y = 40.0 + 5.0 * Xtr[:, 0]
    pred = _ridge(Xtr, y, Xte)
    true = 40.0 + 5.0 * Xte[:, 0]
    assert float(np.mean(np.abs(pred - true))) < 1.0
    assert abs(float(np.mean(pred - true))) < 0.5     # no shrinkage bias


def test_transition_sensitivity_shrinks_hrr60():
    from tests.test_recovery_features import _beats
    beats, conf = _beats(seed=5)
    sweep = transition_sensitivity(beats, conf, 150.0)
    assert sweep["+0s"] is not None
    assert sweep["+10s"] < sweep["+0s"] - 3.0   # a slow transition costs
    vals = [sweep[k] for k in ("+0s", "+2s", "+5s", "+10s")]
    assert all(vals[i] >= vals[i + 1] - 0.5 for i in range(len(vals) - 1))


def test_respiration_rest_only_band_and_fail_closed(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from rppg.respiration import (resting_respiratory_rate,
                                  respiratory_rate_from_motion)
    from scripts.make_synth_recovery import activity_video
    # a 15/min gentle bob = breathing-scale motion
    p = str(tmp_path / "breathing.avi")
    activity_video(p, duration_s=40.0, cadence_per_min=15.0, bob_px=4.0,
                   seed=8)
    rr = resting_respiratory_rate(p)
    assert rr is not None and abs(rr - 15.0) <= 1.5, rr
    # review finding: slow breathers with ordinary fast-inhale/slow-
    # exhale asymmetry were reported at DOUBLE their rate because the
    # old 3 s detrend suppressed the low half of the band. 9/min with a
    # 34% 2nd harmonic must read as ~9, not 18.
    from rppg.respiration import respiratory_rate_from_motion
    fps, dur = 30.0, 60.0
    t = np.arange(0, dur, 1.0 / fps)
    f0 = 9.0 / 60.0
    y = (np.sin(2 * np.pi * f0 * t)
         + 0.34 * np.sin(4 * np.pi * f0 * t)
         + np.random.default_rng(1).normal(0, 0.05, t.size)) + 200.0
    rate, conf = respiratory_rate_from_motion(t, y, fps)
    assert rate is not None and abs(rate - 9.0) <= 1.0, (rate, conf)
    # cadence-scale motion (60/min) is OUTSIDE the validated band -> None
    p2 = str(tmp_path / "fast.avi")
    activity_video(p2, duration_s=40.0, cadence_per_min=60.0, seed=9)
    assert resting_respiratory_rate(p2) is None
    # motionless synthetic noise -> None
    rng = np.random.default_rng(0)
    rate, conf = respiratory_rate_from_motion(
        np.arange(0, 40, 1 / 30.0), rng.normal(0, 0.05, 1200), 30.0)
    assert rate is None


@pytest.fixture(scope="module")
def fitness_ds(tmp_path_factory):
    cv2 = pytest.importorskip("cv2")
    from scripts.make_synth_recovery import make_fitness_dataset
    d = tmp_path_factory.mktemp("fitds")
    manifest = make_fitness_dataset(d, n_participants=6, seed=9)
    return d, manifest


def test_evaluate_fitness_end_to_end(fitness_ds, tmp_path):
    from evaluation.fitness_metrics import evaluate_fitness_dataset
    from models.registry import promote, PromotionRefused
    d, manifest = fitness_ds
    out = evaluate_fitness_dataset(d, out_dir=tmp_path / "rep",
                                   runs_root=tmp_path / "runs")
    doc = out["report"]
    # EXCLUDED-with-reasons: the deliberately unlabeled participant
    no_cpet = [p for p in manifest["participants"] if not p["has_cpet"]]
    assert any(e["session_id"] == no_cpet[0]["participant_id"]
               and "CPET" in e["reasons"][0] for e in doc["excluded"])
    assert doc["n_included"] >= 4
    assert doc["baseline_ladder"]["available"], doc["baseline_ladder"]
    # §V verdict: synthetic domain -> every gate RED, promotion BLOCKED
    assert doc["gates"]["promotion"] == "BLOCKED"
    assert all(g["status"] == "RED"
               for g in doc["gates"]["verdict"]["gates"])
    assert pathlib.Path(out["report_json"]).exists()
    md = pathlib.Path(out["report_md"]).read_text()
    assert "SYNTHETIC" in md and "baseline ladder" in md.lower()
    # the scoreboard entry + promote refusal close the loop
    sb = (tmp_path / "runs" / "scoreboard.jsonl").read_text().splitlines()
    assert json.loads(sb[-1])["promotion_open"] is False
    with pytest.raises(PromotionRefused, match="§V"):
        promote(out["run_dir"], registry_path=tmp_path / "m.jsonl",
                spec_path=tmp_path / "s.md")


def test_cli_evaluate_fitness_requires_registration(tmp_path):
    import os
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "x.session.json").write_text("{}")            # invalid manifest
    (ds / "x.cpet.json").write_text(json.dumps(
        {"vo2peak_mlkgmin": 40.0, "modality": "treadmill",
         "protocol": "ramp"}))
    env = dict(os.environ,
               AVATARX_REGISTRY=str(tmp_path / "reg.jsonl"),
               AVATARX_FITNESS_RUNS=str(tmp_path / "runs"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "evaluate-fitness", str(ds)],
                       capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode == 2
    assert "register" in r.stderr.lower()
    # session datasets ARE registrable (v0.4 registry extension) — after
    # registration the precondition passes and the next gate speaks
    r2 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "register-dataset", str(ds)],
                        capture_output=True, text=True, timeout=120,
                        env=env)
    assert r2.returncode == 0, r2.stderr
    r3 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "evaluate-fitness", str(ds)],
                        capture_output=True, text=True, timeout=120,
                        env=env)
    assert r3.returncode == 2
    assert "register" not in r3.stderr.lower()
    assert "evaluate-fitness failed" in r3.stderr
