"""v0.3 T4 — the paired-data flywheel: registered facial dataset whose
manifests name ingested reference sessions -> production-path rPPG paired
with reference ECG on a common clock -> a facial-source training run
whose evidence is domain-qualified (and whose gates stay honestly red)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from datasets.reference import detect_rpeaks, ingest_reference
from datasets.registry import register_dataset
from research.ecg_reconstruction.train import (facial_pairs,
                                               run_reconstruction_training)

ECG_FS = 250.0


def _ecg_from_rpeaks(rt: np.ndarray, duration_s: float, with_p: bool,
                     seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / ECG_FS)
    mv = np.zeros_like(t)
    for r in rt:
        mv += 1.2 * np.exp(-0.5 * ((t - r) / 0.012) ** 2)
        mv += 0.25 * np.exp(-0.5 * ((t - r - 0.25) / 0.05) ** 2)
        if with_p:
            mv += 0.25 * np.exp(-0.5 * ((t - r + 0.16) / 0.04) ** 2)
    return t, mv + rng.normal(0, 0.02, t.size)


def _reference_session(sdir: pathlib.Path, truth: dict, *, rhythm: str,
                       seed: int):
    sdir.mkdir(parents=True, exist_ok=True)
    rt = np.asarray(truth["rpeaks_s"], float)
    rt = rt[(rt > 0.2) & (rt < truth["duration_s"] - 0.2)]
    t, mv = _ecg_from_rpeaks(rt, truth["duration_s"], rhythm == "SINUS",
                             seed)
    (sdir / "ecg_export.csv").write_text(
        "t_s,mv\n" + "\n".join(f"{a:.6f},{b:.6f}" for a, b in zip(t, mv)))
    ev = np.linspace(1.0, truth["duration_s"] - 1.0, 12)
    (sdir / "sync_events.json").write_text(json.dumps(
        {"video_events_s": ev.tolist(), "ecg_events_s": ev.tolist()}))
    morph = {"conduction_pattern": "normal", "qrs_ms": 90, "qt_ms": 380,
             "pr_ms": 160 if rhythm == "SINUS" else None}
    (sdir / "labels.json").write_text(json.dumps(
        [{"t_start_s": 0.0, "t_end_s": float(t[-1]), "rhythm": rhythm,
          "annotator_initials": "PG", "date": "2026-08-29",
          "esc_definition_confirmed": True, "morphology": morph}]))
    ingest_reference(sdir)


@pytest.fixture(scope="module")
def flywheel(tmp_path_factory):
    from scripts.make_synth_video import synth_video
    base = tmp_path_factory.mktemp("flywheel")
    ds = base / "ds"
    ds.mkdir()
    for rid, pid, kind, rhythm in (("R1", "P001", "sinus", "SINUS"),
                                   ("R2", "P002", "af", "AFIB")):
        truth = synth_video(str(ds / f"{rid}.avi"), kind=kind, fps=30.0,
                            duration_s=20.0, seed=11)
        _reference_session(ds / f"{rid}_ref", truth, rhythm=rhythm,
                           seed=13)
        (ds / f"{rid}.recording.json").write_text(json.dumps(
            {"recording_id": rid, "participant_id": pid,
             "session_id": f"{pid}-s1", "video_path": f"{rid}.avi",
             "reference_dir": f"{rid}_ref"}))
    reg = base / "registry.jsonl"
    row = register_dataset(ds, license_class="internal_consented",
                           consent_class="research_v1", registry_path=reg)
    return {"base": base, "reg": reg, "row": row}


def test_facial_pairs_come_from_the_production_path(flywheel):
    pairs = facial_pairs([flywheel["row"]["id"],],
                         registry_path=flywheel["reg"])
    assert sorted(p[0] for p in pairs) == ["P001", "P002"]
    assert {p[0]: p[1] for p in pairs} == {"P001": 0, "P002": 1}
    for pid, is_af, ppg, ecg in pairs:
        assert ppg.shape == ecg.shape and ppg.size > 15 * 125
        # the ECG side must still carry the video-truth beats
        det = detect_rpeaks(np.arange(ecg.size) / 125.0, ecg)
        assert det.size >= 15
        # the pulse side must be pulsatile at a plausible beat count
        assert float(np.std(ppg)) > 0


def test_facial_training_run_is_domain_qualified_but_still_red(flywheel):
    cfg = flywheel["base"] / "train_facial.yaml"
    cfg.write_text(f"""reconstruction:
  run_name: flywheel-demo
  architecture: windowed_mlp
  source: facial
  facial_dataset_ids: [{flywheel['row']['id']}]
  seed: 7
  split_seed: 7
  test_fraction: 0.5
  steps: 150
  confabulation_steps: 150
  blinded_set: false
""")
    out = run_reconstruction_training(cfg,
                                      runs_root=flywheel["base"] / "runs",
                                      registry_path=flywheel["reg"])
    gr = out["gate_results"]
    data = gr["evidence"]["data"]
    assert data["signal_domain"] == "facial_rppg"
    assert data["production_path"] and data["participant_disjoint"]
    # domain qualifies, so no disqualifier reasons — the gates must now be
    # red on their own criteria (no blinded read; interval fidelity)
    g3 = next(g for g in gr["gates"] if g["gate"] == "g3")
    assert g3["status"] == "RED"
    assert any("no blinded" in r for r in g3["reasons"])
    assert not any("facial rPPG" in r for g in gr["gates"]
                   for r in g["reasons"])
    assert not gr["promotion_open"]
