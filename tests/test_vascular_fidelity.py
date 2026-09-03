"""v0.4-vascular T1 (fidelity) — ICC(2,1) against the published Shrout &
Fleiss example, retest guards, and the end-to-end V0 study on a paired
synthetic dataset: per-feature scoreboard, exclusions with reasons,
watermarked run artifacts, honest BLOCKED verdict, and the fail-closed
surviving set."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess

import numpy as np
import pytest

from research.vascular.fidelity import icc_2_1

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_icc_2_1_matches_shrout_fleiss_published_value():
    # Shrout & Fleiss (1979), table with 6 targets x 4 judges:
    # ICC(2,1) = 0.29.
    m = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
         [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    assert abs(icc_2_1(m) - 0.29) < 0.005


def test_icc_2_1_limits_and_guards():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 40)
    assert icc_2_1(np.stack([x, x], 1)) > 0.999          # identity
    noise = rng.normal(0, 1, (40, 2))
    assert abs(icc_2_1(noise)) < 0.35                    # no agreement
    # absolute agreement penalizes a systematic offset
    off = np.stack([x, x + 2.0], 1)
    assert icc_2_1(off) < 0.45
    # fail-closed guards
    assert icc_2_1(np.ones((2, 2))) is None              # too few subjects
    bad = np.stack([x, x], 1)
    bad[5, 1] = np.nan                                   # NaN rows dropped
    assert icc_2_1(bad) > 0.999
    assert icc_2_1(np.ones((5, 1))) is None              # one rater


def test_agreement_icc_is_participant_level_not_recording_level():
    # review finding: 2 participants x 2 scans each must NOT sneak past
    # the 3-subject ICC floor as four pseudo-replicated rows
    from research.vascular.fidelity import _pairs
    def row(pid, sid, f, c):
        return {"participant_id": pid, "session_id": sid,
                "video_start_utc": None,
                "facial": {"features": {"reflection_index": f}},
                "contact": {"features": {"reflection_index": c}}}
    rows = [row("p1", "p1-s1", 5.0, 5.1), row("p1", "p1-s1", 5.2, 5.0),
            row("p2", "p2-s1", 9.0, 9.2), row("p2", "p2-s1", 9.1, 9.0)]
    pairs = _pairs(rows, "reflection_index")
    assert pairs.shape == (2, 2)                # participants, not scans
    assert icc_2_1(pairs) is None               # 2 subjects -> refused


def test_retest_pairs_are_time_ordered_and_none_filtered():
    from research.vascular.fidelity import _retest_matrix
    def row(rid, t, v):
        return {"participant_id": "p1", "session_id": "p1-s1",
                "recording_id": rid, "video_start_utc": t,
                "facial": {"features": {"rise_time_s": v}},
                "contact": {"features": {}}}
    # lexicographic id order (scan10 < scan9) must NOT decide the pair
    m = _retest_matrix([row("scan10", "2026-08-31T11:00:00Z", 0.30),
                        row("scan9", "2026-08-31T10:00:00Z", 0.20)],
                       "rise_time_s")
    assert m.shape == (1, 2) and list(m[0]) == [0.20, 0.30]
    # a None-valued first scan does not discard the visit
    m2 = _retest_matrix([row("a", "2026-08-31T10:00:00Z", None),
                         row("b", "2026-08-31T10:10:00Z", 0.2),
                         row("c", "2026-08-31T10:20:00Z", 0.25)],
                        "rise_time_s")
    assert m2.shape == (1, 2) and list(m2[0]) == [0.2, 0.25]
    # missing or duplicate timestamps -> no honest order -> dropped
    m3 = _retest_matrix([row("a", None, 0.2),
                         row("b", "2026-08-31T10:10:00Z", 0.25)],
                        "rise_time_s")
    assert m3.shape[0] == 0
    m4 = _retest_matrix([row("a", "2026-08-31T10:00:00Z", 0.2),
                         row("b", "2026-08-31T10:00:00Z", 0.25)],
                        "rise_time_s")
    assert m4.shape[0] == 0


@pytest.fixture(scope="module")
def fidelity_ds(tmp_path_factory):
    pytest.importorskip("cv2")
    from scripts.make_synth_vascular import make_vascular_dataset
    d = tmp_path_factory.mktemp("vasc_fid")
    manifest = make_vascular_dataset(d, n_participants=5, seed=13,
                                     duration_s=20.0,
                                     retest_participants=2)
    return d, manifest


def test_fidelity_study_end_to_end(fidelity_ds, tmp_path):
    from evaluation.vascular_gates import surviving_features
    from research.vascular import WATERMARK
    from research.vascular.fidelity import fidelity_study
    d, manifest = fidelity_ds
    # break one contact arm to prove the exclusion path speaks
    victim = manifest["participants"][2]["recordings"][0]
    (d / f"{victim}.ppg.json").rename(d / f"{victim}.ppg.hidden")
    try:
        out = fidelity_study(d, runs_root=tmp_path / "runs",
                             out_dir=tmp_path / "rep")
    finally:
        (d / f"{victim}.ppg.hidden").rename(d / f"{victim}.ppg.json")
    doc = out["report"]
    assert doc["WATERMARK"] == WATERMARK
    assert any(e["recording_id"] == victim
               and "contact-PPG" in e["reasons"][0]
               for e in doc["excluded"])
    # per-feature table complete, agreement measured on the synthetic rig
    per = doc["per_feature"]
    assert set(per) == set(
        __import__("research.vascular.features",
                   fromlist=["FEATURE_NAMES"]).FEATURE_NAMES)
    ri = per["reflection_index"]
    assert ri["icc"] is not None and ri["n_participants"] >= 3
    assert ri["n_recordings"] > ri["n_participants"]   # retest replicates
    # deterministic fixture: exactly 2 retest visits -> retest ICC is
    # REFUSED (numerology floor), so nothing survives -> fail-closed
    assert "bias" in ri and ri["n_retest_pairs"] == 2
    assert ri["retest_icc"] is None
    # verdict: synthetic domain + tiny cohort -> BLOCKED, V0 red
    assert doc["gates"]["promotion"] == "BLOCKED"
    v0 = next(g for g in doc["gates"]["verdict"]["gates"]
              if g["gate"] == "v0")
    assert v0["status"] == "RED"
    assert any("machinery evidence only" in r for r in v0["reasons"])
    # run artifacts: watermark-first record + scoreboard entry
    run = pathlib.Path(out["run_dir"])
    rec = json.loads((run / "fidelity_record.json").read_text())
    assert list(rec)[0] == "WATERMARK"
    assert (run.parent / "scoreboard.jsonl").exists()
    # the config-driven surviving set reads THIS scoreboard, fail-closed
    surv = surviving_features(runs_root=tmp_path / "runs")
    assert surv == doc["surviving_features"]
    assert doc["surviving_features"] == []
    md = pathlib.Path(out["report_md"]).read_text()
    assert WATERMARK in md and "survives" in md
    # strata tables are present and honest about small groups
    assert "fitzpatrick_group" in doc["strata"]
    for cell in doc["strata"]["fitzpatrick_group"].values():
        assert "n_participants" in cell
        # a stratum big enough to score carries the SAME per-feature
        # machinery; smaller ones carry the honest coverage note
        assert ("per_feature" in cell) or ("note" in cell)


def test_cli_vascular_fidelity_requires_registration(tmp_path):
    import os
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "x.recording.json").write_text("{}")
    (ds / "x.ppg.json").write_text("{}")
    env = dict(os.environ,
               AVATARX_REGISTRY=str(tmp_path / "reg.jsonl"),
               AVATARX_VASCULAR_RUNS=str(tmp_path / "runs"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "vascular-fidelity", str(ds)],
                       capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode == 2
    assert "register" in r.stderr.lower()
    r2 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "register-dataset", str(ds)],
                        capture_output=True, text=True, timeout=120,
                        env=env)
    assert r2.returncode == 0, r2.stderr
    r3 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "vascular-fidelity", str(ds)],
                        capture_output=True, text=True, timeout=120,
                        env=env)
    assert r3.returncode == 2
    assert "vascular-fidelity failed" in r3.stderr
    assert "register" not in r3.stderr.lower()
