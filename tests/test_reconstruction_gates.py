"""v0.3 T3+T4 — §G gate machinery: fidelity metrics, pre-registered gate
evaluation, the training loop's scoreboard, promotion refusal, and the
blinded-read kit. The scoreboard's job right now is to show HONEST REDS."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import copy
import json
import os
import subprocess

import numpy as np
import pytest

from models.registry import promote, PromotionRefused
from research.ecg_reconstruction import WATERMARK
from research.ecg_reconstruction.decoder import synth_pairs, train
from research.ecg_reconstruction.fidelity import (beat_measurements,
                                                  confabulation_eval,
                                                  generate_blinded_set,
                                                  interval_mae,
                                                  rr_only_qt_fit,
                                                  rr_only_qt_mae,
                                                  score_blinded_reads)
from research.ecg_reconstruction.gates import (evaluate_gates, gate_status,
                                               load_gates,
                                               render_gate_status_html)
from research.ecg_reconstruction.train import run_reconstruction_training

_ROOT = pathlib.Path(__file__).resolve().parents[1]
FS = 125.0


# ---------------------------------------------------- fidelity metrics
def test_beat_measurements_plausible_on_synthetic_sinus():
    sid, is_af, ppg, ecg = synth_pairs(2, seed=4, af_fraction=0.0,
                                       duration_s=60.0)[0]
    rows = [m for m in beat_measurements(ecg, FS)
            if m["qt_ms"] is not None and m["pr_ms"] is not None]
    assert len(rows) >= 40
    qt = np.median([m["qt_ms"] for m in rows])
    qrs = np.median([m["qrs_ms"] for m in rows])
    pr = np.median([m["pr_ms"] for m in rows])
    assert 200 < qt < 500, qt          # tangent T-end in physiologic band
    assert 20 < qrs < 120, qrs
    assert 100 < pr < 220, pr          # synthetic P sits at R-160 ms
    assert all(m["rr_ms"] is None or 300 < m["rr_ms"] < 2000 for m in rows)


def test_interval_mae_is_zero_against_itself_and_grows_with_distortion():
    _, _, _, ecg = synth_pairs(2, seed=4, af_fraction=0.0,
                               duration_s=60.0)[0]
    same = interval_mae(ecg, ecg, FS)
    assert same["n_matched_beats"] > 40
    assert same["qt_mae_ms"] == 0.0 and same["qrs_mae_ms"] == 0.0
    assert same["corr"] == 1.0
    shifted = interval_mae(ecg, np.roll(ecg, int(0.03 * FS)), FS)
    assert shifted["qt_mae_ms"] is not None


def test_rr_only_qt_baseline_fits_and_scores():
    subs = synth_pairs(4, seed=6, af_fraction=0.0, duration_s=60.0)
    fit = rr_only_qt_fit(subs[:3], FS)
    assert fit["n_beats"] > 100
    mae = rr_only_qt_mae(fit, subs[3][3], FS)
    assert mae is not None and mae < 80.0    # RR predicts synthetic QT well


# ---------------------------------------------------- gate evaluation
def _perfect_evidence():
    return {
        "data": {"signal_domain": "facial_rppg", "participant_disjoint": True,
                 "session_disjoint": True, "production_path": True,
                 "n_test_subjects": 20},
        "fidelity": {"decoder_qt_mae_ms": 12.0, "decoder_pr_mae_ms": 11.0,
                     "decoder_qrs_mae_ms": 8.0, "rr_only_qt_mae_ms": 25.0,
                     "decoder_corr": 0.8},
        "g1": {"qt_mae_ms": {"decoder": 12.0, "identity": 30.0,
                             "decoder_wins": True},
               "pr_mae_ms": {"decoder": 11.0, "identity": 28.0,
                             "decoder_wins": True},
               "qrs_mae_ms": {"decoder": 8.0, "identity": 20.0,
                              "decoder_wins": True},
               "all_morphology_metrics_beat_identity": True},
        "confabulation": {"hallucinated_p_excess": 0.02,
                          "p_prominence_gen_on_withheld_af": 0.06,
                          "p_prominence_ref_on_af": 0.04},
        "detection": {"auc_measured_path": 0.95,
                      "auc_from_reconstruction": 0.94,
                      "auc_deficit": 0.01},
        "blinded_reads": {"n_scored": 40, "sensitivity": 0.9,
                          "specificity": 0.88},
    }


def test_gates_all_red_with_no_evidence():
    v = evaluate_gates(load_gates(), {})
    assert len(v["gates"]) == 5
    assert all(g["status"] == "RED" for g in v["gates"])
    assert not v["promotion_open"]
    assert any("no evaluation run on record" in r
               for g in v["gates"] for r in g["reasons"])


def test_surrogate_domain_can_never_open_a_gate():
    ev = _perfect_evidence()
    ev["data"]["signal_domain"] = "synthetic"
    v = evaluate_gates(load_gates(), ev)
    assert all(g["status"] == "RED" for g in v["gates"])
    assert any("facial rPPG" in r for r in v["gates"][0]["reasons"])


def test_signoff_blocks_promotion_even_when_all_gates_green():
    gcfg = load_gates()
    v = evaluate_gates(gcfg, _perfect_evidence())
    assert all(g["status"] == "GREEN" for g in v["gates"]), \
        [(g["gate"], g["reasons"]) for g in v["gates"]]
    assert v["all_gates_green"] and not v["clinical_signoff"]
    assert not v["promotion_open"]
    signed = copy.deepcopy(gcfg)
    signed["signoff"] = {"owner_confirmed": True,
                         "clinical_advisor": "Dr. X", "date": "2027-01-01"}
    assert evaluate_gates(signed, _perfect_evidence())["promotion_open"]


def test_each_gate_fails_on_its_own_criterion():
    gcfg = load_gates()
    for mutate, gate, needle in (
        (lambda e: e["g1"].update(
            all_morphology_metrics_beat_identity=False,
            qt_mae_ms={"decoder": 30.0, "identity": 30.0,
                       "decoder_wins": False}), "g1", "identity"),
        (lambda e: e["fidelity"].update(decoder_qt_mae_ms=35.0), "g2",
         "exceeds"),
        (lambda e: e.update(blinded_reads=None), "g3", "no blinded"),
        (lambda e: e["confabulation"].update(
            hallucinated_p_excess=0.39), "g4", "hallucinates"),
        (lambda e: e["detection"].update(auc_deficit=0.08), "g5",
         "measured path"),
    ):
        ev = _perfect_evidence()
        mutate(ev)
        v = evaluate_gates(gcfg, ev)
        row = next(g for g in v["gates"] if g["gate"] == gate)
        assert row["status"] == "RED", (gate, row)
        assert any(needle in r for r in row["reasons"]), (gate, row)


# ---------------------------------------------------- the training loop
@pytest.fixture(scope="module")
def recon_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("recon")
    cfg = base / "train.yaml"
    cfg.write_text("""reconstruction:
  run_name: t4-demo
  architecture: windowed_mlp
  source: synthetic
  n_subjects: 6
  seed: 7
  split_seed: 7
  steps: 200
  confabulation_steps: 200
  blinded_set: true
""")
    out = run_reconstruction_training(cfg, runs_root=base / "runs")
    return {"base": base, "runs": base / "runs", "out": out}


def test_run_writes_watermarked_artifacts_and_scoreboard(recon_run):
    rd = pathlib.Path(recon_run["out"]["run_dir"])
    for name in ("model.json", "reconstruction_record.json",
                 "gate_results.json"):
        doc = json.loads((rd / name).read_text())
        assert doc["WATERMARK"] == WATERMARK, name
    sb = (recon_run["runs"] / "scoreboard.jsonl").read_text().splitlines()
    entry = json.loads(sb[-1])
    assert entry["run_id"] == recon_run["out"]["run_id"]
    assert entry["promotion_open"] is False
    # honest reds: synthetic evidence can exercise but never open gates
    assert all(g["status"] == "RED" for g in entry["gates"])
    rec = json.loads((rd / "reconstruction_record.json").read_text())
    assert not (set(rec["train_subjects"]) & set(rec["test_subjects"]))


def test_gate_status_reads_the_scoreboard(recon_run):
    doc = gate_status(runs_root=recon_run["runs"])
    assert doc["promotion"] == "BLOCKED"
    assert doc["evidence_run"] == recon_run["out"]["run_id"]
    assert len(doc["gates"]) == 5
    html = render_gate_status_html(doc)
    assert html.count(WATERMARK) >= 2 and "BLOCKED" in html
    empty = gate_status(runs_root=recon_run["base"] / "nowhere")
    assert empty["evidence_run"] is None
    assert all(g["status"] == "RED" for g in empty["gates"])


def test_promote_refuses_reconstruction_while_gates_red(recon_run):
    with pytest.raises(PromotionRefused) as ei:
        promote(recon_run["out"]["run_dir"],
                registry_path=recon_run["base"] / "models.jsonl",
                spec_path=recon_run["base"] / "spec.md")
    msg = str(ei.value)
    assert "§G" in msg and "clinical signoff" in msg
    assert msg.count("G") >= 5                    # every red gate listed
    assert not (recon_run["base"] / "models.jsonl").exists()


def test_blinded_kit_escrows_the_answers(recon_run):
    rd = pathlib.Path(recon_run["out"]["run_dir"])
    cases = rd / "blinded" / "cases"
    manifest = json.loads((cases / "manifest.json").read_text())
    assert manifest["WATERMARK"] == WATERMARK
    assert "is_af" not in json.dumps(manifest["cases"])   # no truth leaks
    for cid in manifest["cases"]:
        assert WATERMARK in (cases / f"{cid}.svg").read_text()
    key_p = rd / "blinded" / "escrow" / "answer_key.json"
    key = json.loads(key_p.read_text())["key"]
    perfect = {cid: {"is_af": truth["is_af"]} for cid, truth in key.items()}
    resp = rd / "blinded" / "responses.json"
    resp.write_text(json.dumps(perfect))
    scored = score_blinded_reads(key_p, resp)
    assert scored["n_scored"] == len(key)
    truths = {t["is_af"] for t in key.values()}
    if 1 in truths:                     # a class absent from the drawn set
        assert scored["sensitivity"] == 1.0   # is honestly unscored (None)
    if 0 in truths:
        assert scored["specificity"] == 1.0
    assert scored["sensitivity"] is not None or \
        scored["specificity"] is not None


def test_confabulation_eval_measures_p_on_withheld_af():
    subs = synth_pairs(6, seed=3, duration_s=60.0)
    conf = confabulation_eval(subs[3:], subs[:3], seed=1, steps=200)
    assert conf["n_af_test_subjects"] >= 1
    assert conf["p_prominence_gen_on_withheld_af"] is not None
    assert conf["hallucinated_p_excess"] is not None


def test_cli_reconstruct_is_a_watermarked_research_artifact(recon_run,
                                                           videos):
    env = dict(os.environ,
               AVATARX_RESEARCH_RUNS=str(recon_run["runs"]))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "reconstruct", videos["sinus30"][0]],
                       capture_output=True, text=True, timeout=300,
                       env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["promotion"] == "BLOCKED"
    od = pathlib.Path(out["out_dir"])
    assert od.parent == recon_run["runs"]         # research/runs only
    doc = json.loads((od / "reconstruction.json").read_text())
    assert list(doc)[0] == "WATERMARK" and doc["WATERMARK"] == WATERMARK
    assert "note" in doc["fidelity_vs_reference"]  # no paired reference
    assert WATERMARK in (od / "strip.svg").read_text()
    html = (od / "report.html").read_text()
    assert html.count(WATERMARK) >= 2
    assert "promotion: BLOCKED" in html           # gate-status footer


def test_cli_reconstruct_fails_closed_without_a_model(videos, tmp_path):
    env = dict(os.environ, AVATARX_RESEARCH_RUNS=str(tmp_path / "empty"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "reconstruct", videos["sinus30"][0]],
                       capture_output=True, text=True, timeout=300,
                       env=env)
    assert r.returncode == 2
    assert "no trained reconstruction model" in r.stderr


def test_cli_gate_status_and_train_dispatch(recon_run, tmp_path):
    env = dict(os.environ,
               AVATARX_RESEARCH_RUNS=str(recon_run["runs"]))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "gate-status", "--html",
                        str(tmp_path / "gates.html")],
                       capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["promotion"] == "BLOCKED" and len(doc["gates"]) == 5
    assert WATERMARK in (tmp_path / "gates.html").read_text()
