"""
T7 — evaluation report over a directory of (recording.json, video, ecg)
triples.

The dataset fixture contains sinus / AF / AF-with-pulse-deficit / RSA
(hard negative) arms at 30 and 60 fps, a CRF-28 recording that must be
EXCLUDED by the schema gate (not silently analysed), and hash-assigned
participant splits whose composition the leakage audit checks.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    from scripts.make_synth_dataset import make_dataset
    d = tmp_path_factory.mktemp("synthds")
    manifest = make_dataset(str(d), n_participants=12, duration_s=20.0,
                            seed=11)
    return d, manifest


@pytest.fixture(scope="session")
def report(dataset):
    from evaluation.report import evaluate_dataset
    d, _ = dataset
    out = evaluate_dataset(str(d))
    with open(out["report_json"]) as f:
        doc = json.load(f)
    md = pathlib.Path(out["report_md"]).read_text()
    return out, doc, md


# ------------------------------------------------------------- generation
def test_dataset_has_triples_and_arms(dataset):
    d, manifest = dataset
    recs = sorted(d.glob("*.recording.json"))
    assert len(recs) >= 13                     # 12 analysable + 1 CRF-28
    arms = {m["arm"] for m in manifest["recordings"]}
    assert {"sinus", "af", "af_deficit", "rsa"} <= arms
    for r in recs:
        rid = r.name.replace(".recording.json", "")
        assert (d / f"{rid}.avi").exists()
        assert (d / f"{rid}.ecg.json").exists()


# ---------------------------------------------------------------- report
def test_report_renders_files(report):
    out, doc, md = report
    assert pathlib.Path(out["report_json"]).exists()
    assert pathlib.Path(out["report_md"]).exists()
    for section in ("Provenance", "Gate 1", "Gate 1b", "Risk-coverage",
                    "No-read parity", "Leakage audit", "Serial confirmation"):
        assert section in md, f"missing section {section}"


def test_crf28_recording_is_excluded_not_analysed(report):
    _, doc, md = report
    exc = doc["excluded"]
    assert len(exc) == 1
    assert any("CRF" in r for r in exc[0]["reasons"])
    assert exc[0]["recording_id"] not in [r["recording_id"]
                                          for r in doc["per_recording"]]
    assert "EXCLUDED" in md


def test_gate_tables_include_production_rmssd_and_ece(report):
    _, doc, _ = report
    g1 = doc["gate1_non_af"]
    assert "production_rmssd_error_ms" in g1["extra_metrics"]
    assert "confidence_ece" in g1["extra_metrics"]
    assert isinstance(g1["per_recording_pass"], list)
    g1b = doc["gate1b_af"]
    assert "missed_beat_flag_recall" in g1b["extra_metrics"]


def test_beat_metrics_split_by_rhythm(report):
    _, doc, _ = report
    bm = doc["beat_metrics_by_rhythm"]
    assert "AF" in bm and "non-AF" in bm
    for arm in bm.values():
        assert "f1_at_50ms_mean" in arm and arm["n_recordings"] > 0


def test_risk_coverage_reports_af_retention(report):
    _, doc, _ = report
    rows = doc["risk_coverage"]
    assert [round(r["coverage"], 2) for r in rows] == \
        sorted([round(r["coverage"], 2) for r in rows], reverse=True)
    assert all("n_afib_retained" in r for r in rows)


def test_no_read_parity_includes_rhythm_axis(report):
    _, doc, _ = report
    assert "rhythm" in doc["no_read_parity"]["levels"]


def test_leakage_audit_and_split_composition_attached(report):
    _, doc, _ = report
    assert doc["leakage_audit"]["pass"] is True, doc["leakage_audit"]
    comp = doc["split_composition"]
    assert any(v.get("af_participants", 0) > 0 for v in comp.values())


def test_serial_confirmation_quotes_persistent_share(report):
    _, doc, md = report
    sc = doc["serial_confirmation"]
    assert "fp_persistent_share" in sc
    assert "fp_persistent_share" in md          # printed, not just stored


def test_provenance_complete(report):
    _, doc, _ = report
    p = doc["provenance"]
    for k in ("config_hash", "code_commit", "calibration_version",
              "model_version", "dataset_dir", "n_recordings"):
        assert p.get(k) not in (None, "", "unknown"), k


def test_classification_never_uses_raw_accuracy_headline(report):
    _, doc, _ = report
    s = doc["classification"]
    assert "sensitivity" in s and "specificity" in s
    assert "accuracy" not in s


# ------------------------------------------------------------------- CLI
def test_cli_evaluate_end_to_end(tmp_path):
    from scripts.make_synth_dataset import make_dataset
    d = tmp_path / "ds"
    d.mkdir()
    make_dataset(str(d), n_participants=6, duration_s=15.0, seed=3)
    # v0.2 M2.9: evaluate consumes registered datasets only
    reg = tmp_path / "registry.jsonl"
    import os
    env = dict(os.environ, AVATARX_REGISTRY=str(reg))
    r0 = subprocess.run([sys.executable, str(REPO / "cli.py"), "evaluate",
                         str(d)], capture_output=True, text=True, cwd=REPO,
                        env=env)
    assert r0.returncode == 2 and "not a registered dataset" in r0.stderr
    rr = subprocess.run([sys.executable, str(REPO / "cli.py"),
                         "register-dataset", str(d)], capture_output=True,
                        text=True, cwd=REPO, env=env)
    assert rr.returncode == 0, rr.stderr
    p = subprocess.run([sys.executable, str(REPO / "cli.py"), "evaluate",
                        str(d)], capture_output=True, text=True, cwd=REPO,
                       env=env)
    assert p.returncode == 0, p.stderr
    doc = json.loads(p.stdout)
    assert pathlib.Path(doc["report_md"]).exists()
