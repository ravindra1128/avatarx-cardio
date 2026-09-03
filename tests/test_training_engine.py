"""M3 — training engine: config-driven runs over registered datasets,
production-path feature extraction, mandatory baselines, promotion gate.
The synthetic-corpus caveat applies to every number here (interface
proof, not performance)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from datasets.registry import register_dataset
from models.registry import promote, PromotionRefused, load_model_registry
from training.baselines import mandatory_baseline_report, promotion_gate
from training.runs import run_training, rank_auc, TrainingError

_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One registered synthetic dataset + one completed training run,
    shared by every M3 test (the pipeline runs once per recording)."""
    base = tmp_path_factory.mktemp("m3")
    ds = base / "ds"
    ds.mkdir()
    from scripts.make_synth_dataset import make_dataset
    make_dataset(str(ds), n_participants=10, duration_s=12.0, seed=11)
    reg = base / "registry.jsonl"
    row = register_dataset(ds, license_class="internal_consented",
                           consent_class="research_v1", registry_path=reg)
    cfg = base / "train_demo.yaml"
    cfg.write_text(f"""train:
  run_name: m3-demo
  model: logreg
  seed: 7
  split_seed: 20260814
  dataset_ids: [{row['id']}]
""")
    out = run_training(cfg, registry_path=reg, out_root=base / "runs")
    return {"base": base, "reg": reg, "ds_row": row, "cfg": cfg,
            "run": out}


def test_rank_auc_basics():
    assert rank_auc(np.array([0, 0, 1, 1]),
                    np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert abs(rank_auc(np.array([0, 1, 0, 1]),
                        np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9
    assert rank_auc(np.array([0, 1]), np.array([0.9, np.nan])) == 0.0


def test_run_record_carries_lineage_and_dev_metrics(trained):
    rec = trained["run"]["record"]
    assert rec["run_id"].startswith("run-")
    assert rec["datasets"][0]["id"] == trained["ds_row"]["id"]
    assert rec["git_commit"] and rec["config_hash"]
    assert rec["n_train"] >= 4
    run_dir = pathlib.Path(trained["run"]["run_dir"])
    assert (run_dir / "model.json").exists()
    assert (run_dir / "features.jsonl").exists()
    feats = [json.loads(x) for x in
             (run_dir / "features.jsonl").read_text().splitlines()]
    assert all(f["split"] in ("TRAIN", "DEV", "INTERNAL_TEST")
               for f in feats)


def test_training_is_deterministic_by_config(trained):
    out2 = run_training(trained["cfg"], registry_path=trained["reg"],
                        out_root=trained["base"] / "runs2")
    assert out2["run_id"] == trained["run"]["run_id"]
    a = json.loads((pathlib.Path(trained["run"]["run_dir"])
                    / "model.json").read_text())
    b = json.loads((pathlib.Path(out2["run_dir"]) / "model.json").read_text())
    assert a["model_json"] == b["model_json"]        # bit-identical weights


def test_mandatory_baselines_and_detectors(trained):
    rep = mandatory_baseline_report(trained["run"]["run_dir"])
    assert set(rep["baselines"]) == {"tachogram_stats", "hr_only",
                                     "participant_history", "device_site"}
    # honest disjoint splits: the memorization/leak detectors sit at chance
    assert rep["baselines"]["participant_history"] <= 0.60
    assert rep["baselines"]["device_site"] <= 0.60
    assert rep["leak_flags"] == []
    assert rep["candidate_auc"] >= rep["baselines"]["hr_only"]
    assert "parity" in rep and rep.get("parity_note")   # fitz unrecorded


def test_promotion_gate_and_registry(trained, tmp_path):
    mreg = tmp_path / "models.jsonl"
    spec = tmp_path / "SPEC.md"
    spec.write_text("# spec\n")
    row = promote(trained["run"]["run_dir"], registry_path=mreg,
                  spec_path=spec)
    assert row["semver"] == "0.1.0"
    assert load_model_registry(mreg)[0]["run_id"] == \
        trained["run"]["run_id"]
    assert "Model promotion 0.1.0" in spec.read_text()

    # refusal paths: doctored eval reports, every reason listed
    good = mandatory_baseline_report(trained["run"]["run_dir"])
    leaky = dict(good)
    leaky["baselines"] = dict(good["baselines"],
                              participant_history=0.85)
    leaky["leak_flags"] = ["participant_history AUC 0.85 > 0.6: leak"]
    with pytest.raises(PromotionRefused, match="leak"):
        promote(trained["run"]["run_dir"], registry_path=mreg,
                spec_path=spec, eval_report=leaky)
    weak = dict(good)
    weak["candidate_auc"] = good["baselines"]["tachogram_stats"] - 0.3
    ok, reasons = promotion_gate(weak, fairness_cfg={"noread_max_ratio": 1.5})
    assert not ok and len(reasons) >= 2               # ALL reasons listed


def test_unusable_config_fails_closed(trained, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("train: {run_name: x, model: forest, seed: 1, "
                   "dataset_ids: [nope]}\n")
    with pytest.raises(TrainingError):
        run_training(bad, registry_path=trained["reg"],
                     out_root=tmp_path / "r")


def test_cli_train_and_promote(trained, tmp_path):
    import os
    env = dict(os.environ,
               AVATARX_REGISTRY=str(trained["reg"]),
               AVATARX_MODEL_REGISTRY=str(tmp_path / "models.jsonl"),
               AVATARX_SPEC_PATH=str(tmp_path / "SPEC.md"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"), "train",
                        str(trained["cfg"])], capture_output=True,
                       text=True, timeout=600, env=env, cwd=_ROOT)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["run_id"] == trained["run"]["run_id"]
    r2 = subprocess.run([sys.executable, str(_ROOT / "cli.py"), "promote",
                         doc["run_dir"]], capture_output=True, text=True,
                        timeout=300, env=env, cwd=_ROOT)
    assert r2.returncode == 0, r2.stderr
    assert json.loads(r2.stdout)["semver"]
