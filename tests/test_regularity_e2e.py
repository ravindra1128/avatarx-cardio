"""v0.7 end to end — real video through the production path, the ECG
reference, the head, the floor, the evaluation, the gates, the CLI, and
the public corpus (contact PPG vs ECG on real hearts, as a surrogate)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import os
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    pytest.importorskip("cv2")
    from datasets.registry import register_dataset
    from scripts.make_synth_regularity import (make_dispersion_ladder,
                                               make_regularity_dataset)
    d = tmp_path_factory.mktemp("regds")
    ds = d / "ds"
    make_regularity_dataset(ds, duration_s=40.0,
                            cohorts=["regular", "rsa_young", "af",
                                     "bigeminy"], per_cohort=1,
                            fps_cycle=(30.0,))
    make_dispersion_ladder(ds, duration_s=40.0, rungs=(0.0, 24.0),
                           fps_list=(30.0,), reps=2, fitz_groups=(3,))
    reg = d / "registry.jsonl"
    register_dataset(ds, license_class="internal_consented",
                     consent_class="research_v1", registry_path=reg)
    return ds, reg, d


@pytest.fixture(scope="module")
def rows(dataset):
    from evaluation.regularity_metrics import rows_from_dataset
    return rows_from_dataset(dataset[0])


def test_rows_ceiling_head_and_evaluation_on_real_video(dataset, rows):
    ds, reg, d = dataset
    from evaluation.regularity_metrics import (ceiling_test,
                                               evaluate_regularity_dataset)
    assert len(rows) == 8
    by = {r["recording_id"]: r for r in rows}
    # the reference came from the ECG sidecar and the camera path
    # reproduced its verdict on every cohort clip
    for r in rows:
        assert r["ecg_class"] in ("regular", "irregular"), r["recording_id"]
        assert r["camera_class"] == r["ecg_class"], (
            r["recording_id"], r["camera_index"], r["ecg_index"])
        assert r["resp_rate_brpm"] == pytest.approx(15.0, abs=1.5)
        assert r["fitzpatrick_group"] is not None
    c = ceiling_test(rows)
    assert c["kappa"] == 1.0
    assert c["index_r"] > 0.95
    # the head judged every clip (respiration was available) and
    # explained the RSA and the ectopy
    heads = {r["recording_id"]: r["head"] for r in rows}
    for rid, hv in heads.items():
        assert hv["class"] in ("regular", "irregular"), (rid, hv)
        if "rsa_young" in rid:
            assert hv["benign_pattern_evidence"]["evidence"] == \
                "respiration_coupled"
        if "bigeminy" in rid:
            assert hv["benign_pattern_evidence"]["evidence"] == \
                "ectopy_pattern"
        if "_af" in rid.split("_")[-1] or rid.endswith("af"):
            assert hv["class"] == "irregular"
    out = evaluate_regularity_dataset(rows, signal_domain="synthetic",
                                      production_path=True,
                                      runs_root=d / "runs")
    assert out["verdict"]["promotion_open"] is False
    for g in out["verdict"]["gates"]:
        assert g["status"] == "RED"
        assert any("machinery evidence only" in x for x in g["reasons"])
    assert out["evidence"]["r0"]["kappa"] == 1.0


def test_floor_report_on_the_ladder(dataset, rows):
    ds, reg, d = dataset
    from evaluation.regularity_floor import regularity_floor_report
    doc = regularity_floor_report(rows, runs_root=d / "runs",
                                  beat_error_trials=6)
    assert doc["n_paired_scans"] == 8
    g = doc["interpolation_gain"]["30.0"]
    # two metronomic rungs at 30 fps: the measured gain is what v0.6 saw
    assert g["n_metronomic_refs"] == 2
    assert 0.45 < g["interpolation_gain_measured"] < 0.95
    assert doc["mdi"]["cells"]
    assert (pathlib.Path(doc["run_dir"]) / "floor.json").exists()
    from evaluation.regularity_gates import regularity_gate_status
    st = regularity_gate_status(runs_root=d / "runs")
    assert st["floor_run"] == doc["run_id"]
    assert st["promotion"] == "BLOCKED"


def test_cli_verbs(dataset):
    ds, reg, d = dataset
    env = dict(os.environ, AVATARX_REGISTRY=str(reg),
               AVATARX_REGULARITY_RUNS=str(d / "cli_runs"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "regularity-floor", str(ds), "--trials", "4"],
                       capture_output=True, text=True, timeout=1800, env=env)
    assert r.returncode == 0, r.stderr[-2500:]
    doc = json.loads(r.stdout)
    assert "RESEARCH ARTIFACT" in doc["WATERMARK"]
    assert doc["n_paired_scans"] == 8
    r2 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "evaluate-regularity", str(ds)],
                        capture_output=True, text=True, timeout=1800,
                        env=env)
    assert r2.returncode == 0, r2.stderr[-2500:]
    doc2 = json.loads(r2.stdout)
    assert doc2["promotion"] == "BLOCKED"
    assert doc2["ceiling_kappa"] == 1.0
    r3 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "gate-status", "--track", "regularity",
                         "--html", str(d / "s.html")],
                        capture_output=True, text=True, timeout=300, env=env)
    assert r3.returncode == 0, r3.stderr[-1500:]
    doc3 = json.loads(r3.stdout)
    assert doc3["track"] == "regularity" and len(doc3["gates"]) == 6
    assert doc3["evaluation_run"] == doc2["run_id"]
    assert (d / "s.html").exists()
    # an unregistered directory is refused before any compute
    r4 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "evaluate-regularity", str(d / "nope")],
                        capture_output=True, text=True, timeout=300,
                        env=env)
    assert r4.returncode != 0


def test_public_corpus_ceiling_on_real_hearts():
    """MIMIC PERform AF: contact PPG vs ECG through the same code — a
    surrogate (public_ppg) that can never open a gate, but the first
    ceiling number on people."""
    from datasets.public_regularity import available, mimic_perform_rows
    if not available():
        pytest.skip("public corpus not cached")
    from evaluation.regularity_metrics import ceiling_test
    rows = mimic_perform_rows(max_subjects=2, max_windows_per_subject=2)
    assert len(rows) == 8
    assert {r["dataset"] for r in rows} == {"mimic_perform_af_csv",
                                            "mimic_perform_non_af_csv"}
    assert all(r["surrogate"] == "public_ppg" for r in rows)
    for r in rows:
        assert r["ecg_n_intervals"] > 30 and r["camera_n_intervals"] > 0
    c = ceiling_test(rows)
    assert c["n_paired_scans"] == 8
    assert c["kappa"] is not None
