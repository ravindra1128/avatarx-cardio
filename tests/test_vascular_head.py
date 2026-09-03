"""v0.4-vascular T3 — head_vascular: registered but never default-
enabled, quarantine-clean by construction, inert and number-free on
every surface the pipeline can reach (V-a), estimating only when the
research surface supplies model + features, and refusing demographic
or stale artifacts (T2)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re

import pytest

from configs import load_config
from datasets.schema import MeasurementClass
from heads import enabled_heads, get_head

_ROOT = pathlib.Path(__file__).resolve().parents[1]
VASC_RE = re.compile(
    r"vascular[\s_-]+age|artery[\s_-]+age|PWV|pulse[\s_-]*wave[\s_-]*velocity"
    r"|\d+(?:\.\d+)?\s*m/s", re.I)


def _artifact(**over):
    d = {"kind": "ridge_morphology",
         "features": ["reflection_index", "rise_time_s"],
         "mu": [0.3, 0.2], "sd": [0.05, 0.01], "coef": [0.8, -0.4],
         "intercept": 8.0, "residual_sd": 0.9, "n_train": 30}
    d.update(over)
    return d


def _feats(**over):
    d = {"available": True,
         "features": {"reflection_index": 0.35, "rise_time_s": 0.19},
         "quality": {"per_beat_scatter": {}}, "n_beats_used": 18}
    d.update(over)
    return d


def _ctx(**over):
    d = {"vascular_research_surface": True,
         "vascular_features": _feats(),
         "vascular_model": _artifact(),
         "surviving_features": ["reflection_index", "rise_time_s"]}
    d.update(over)
    return d


def test_registered_research_only_never_default_enabled():
    h = get_head("vascular")
    assert h.research_only is True
    assert h.version.endswith("-research")
    assert "vascular" not in [x.name for x in
                              enabled_heads(load_config())]


def test_head_module_never_imports_research_code():
    # AST-based, not substring-based (review finding: __import__ or an
    # aliased importlib call would slip past a text grep)
    import ast
    tree = ast.parse((_ROOT / "heads" / "head_vascular.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith(("research", "importlib")), \
                    a.name
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(
                ("research", "importlib")), node.module
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", getattr(fn, "attr", ""))
            assert name not in ("__import__", "import_module"), name
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                assert not node.value.startswith("research."), node.value


def test_watermark_literal_matches_the_track_constant():
    from heads.head_vascular import WATERMARK as head_wm
    from research.vascular import WATERMARK as track_wm
    assert head_wm == track_wm


def test_inert_and_number_free_on_every_pipeline_surface():
    h = get_head("vascular")
    for ctx in ({}, {"cfg": {}, "features": None},
                {"vascular_research_surface": "yes"},   # not True
                {"vascular_research_surface": False}):
        r = h.run(None, ctx)
        assert r.measurement_class is MeasurementClass.RESEARCH_VASCULAR
        assert r.value == {"available": False}
        blob = json.dumps(r.to_dict())
        assert not VASC_RE.search(blob), blob
        assert not re.search(r"\d", json.dumps(r.value))
        assert any("V-a" in x for x in r.reasons)


def test_research_surface_estimates_with_ci_and_watermark():
    r = get_head("vascular").run(None, _ctx())
    v = r.value
    assert v["available"] is True
    # z = [(0.35-0.3)/0.05, (0.19-0.2)/0.01] = [1, -1]
    # est = 8.0 + 1*0.8 + (-1)*(-0.4) = 9.2
    assert abs(v["estimate_cfpwv_mps"] - 9.2) < 1e-6
    assert v["ci95_mps"][0] < 9.2 < v["ci95_mps"][1]
    assert "RESEARCH ARTIFACT" in v["watermark"]
    assert v["features_used"] == ["reflection_index", "rise_time_s"]


def test_fail_closed_refusals():
    h = get_head("vascular")
    no_model = h.run(None, _ctx(vascular_model={}))
    assert no_model.value["available"] is False
    assert any("no trained vascular model" in x for x in no_model.reasons)
    no_surv = h.run(None, _ctx(surviving_features=[]))
    assert any("fail closed" in x for x in no_surv.reasons)
    stale = h.run(None, _ctx(surviving_features=["rise_time_s"]))
    assert any("stale artifact" in x for x in stale.reasons)
    gated = h.run(None, _ctx(vascular_features={
        "available": False, "reasons": ["V-c"]}))
    assert any("V-c" in x for x in gated.reasons)
    # T2: an artifact smuggling demographics is refused loudly
    demo = h.run(None, _ctx(vascular_model=_artifact(
        features=["reflection_index", "age"]),
        surviving_features=["reflection_index", "age"]))
    assert demo.value["available"] is False
    assert any("T2" in x for x in demo.reasons)
    # none of the refusal paths carries an estimate
    for r in (no_model, no_surv, stale, gated, demo):
        assert "estimate_cfpwv_mps" not in r.value


def test_session_head_chain_is_unchanged():
    # the three-phase session runs exactly recovery/fitness/trend —
    # head_vascular is not wired there (research surfaces only)
    src = (_ROOT / "protocol" / "session.py").read_text()
    assert 'get_head("vascular")' not in src


def test_cli_process_keeps_vascular_out_of_the_consumer_result(tmp_path):
    pytest.importorskip("cv2")
    import os
    import subprocess
    from scripts.make_synth_video import synth_video
    v = tmp_path / "scan.avi"
    synth_video(str(v), kind="sinus", fps=30.0, duration_s=20.0, seed=21)
    env = dict(os.environ,
               AVATARX_VASCULAR_RUNS=str(tmp_path / "runs"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "process", str(v), "--heads",
                        "afib,rate_flags,vascular"],
                       capture_output=True, text=True, timeout=600,
                       env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    doc = json.loads(r.stdout)              # stdout stays pure JSON
    blob = json.dumps(doc)
    assert not VASC_RE.search(blob), "vascular token in consumer result"
    assert "vascular" not in [h.get("head") for h in
                              (doc.get("head_results") or [])]
    assert "vascular research report" in r.stderr
    m = re.search(r"measurement\): (\S+)", r.stderr)
    rep = json.loads(pathlib.Path(m.group(1)).read_text())
    assert list(rep)[0] == "WATERMARK"
    assert rep["head_result"]["head"] == "vascular"
    # no model on record -> the research report itself is fail-closed
    assert rep["head_result"]["value"]["available"] is False
    assert rep["gates"]["promotion"] == "BLOCKED"


def test_run_research_head_success_and_fallbacks(tmp_path, monkeypatch):
    """Direct coverage of the research-surface glue (review finding: its
    success path was unreachable in the only indirect test)."""
    import research.vascular.head_runner as hr_mod
    from evaluation.vascular_gates import append_scoreboard

    class _Res:
        recording_id = "rid-x"

    feats = {"available": True,
             "features": {"reflection_index": 0.35, "rise_time_s": 0.19},
             "quality": {}, "n_beats_used": 20}
    monkeypatch.setattr(hr_mod, "features_from_details",
                        lambda result, det: feats)
    runs = tmp_path / "runs"
    per = {f: {"icc": 0.9, "retest_icc": 0.8}
           for f in ("reflection_index", "rise_time_s")}
    append_scoreboard(
        {"kind": "fidelity", "run_id": "fid-1",
         "evidence": {"v0": {"per_feature": per,
                             "n_paired_participants": 80},
                      "data": {"signal_domain": "facial_rppg",
                               "participant_disjoint": True,
                               "session_disjoint": True,
                               "production_path": True}}},
        runs_root=runs)
    append_scoreboard({"kind": "evaluation", "run_id": "ev-1",
                       "evidence": {}}, runs_root=runs)
    (runs / "ev-1").mkdir(parents=True)
    (runs / "ev-1" / "model.json").write_text(json.dumps(_artifact()))
    out = hr_mod.run_research_head(_Res(), {"lattice": None},
                                   runs_root=runs)
    v = out["head_result"].value
    assert v["available"] is True and "estimate_cfpwv_mps" in v
    rep = json.loads(pathlib.Path(out["report_path"]).read_text())
    assert list(rep)[0] == "WATERMARK"
    assert rep["gates"]["promotion"] == "BLOCKED"
    # fallbacks: no scoreboard / missing model.json / corrupt / coef-less
    # / incoherent array lengths all yield {} -> available False
    assert hr_mod._latest_model_artifact(tmp_path / "none") == {}
    runs2 = tmp_path / "runs2"
    append_scoreboard({"kind": "evaluation", "run_id": "ev-9",
                       "evidence": {}}, runs_root=runs2)
    assert hr_mod._latest_model_artifact(runs2) == {}
    (runs2 / "ev-9").mkdir(parents=True)
    (runs2 / "ev-9" / "model.json").write_text("{not json")
    assert hr_mod._latest_model_artifact(runs2) == {}
    (runs2 / "ev-9" / "model.json").write_text(json.dumps(
        {"kind": "none"}))
    assert hr_mod._latest_model_artifact(runs2) == {}
    (runs2 / "ev-9" / "model.json").write_text(json.dumps(
        _artifact(mu=[0.3])))                     # length mismatch
    assert hr_mod._latest_model_artifact(runs2) == {}
    # gate-status failure -> fail-closed gates line, report still written
    monkeypatch.setattr(hr_mod, "run_research_head",
                        hr_mod.run_research_head)
    import evaluation.vascular_gates as vg

    def _boom(**kw):
        raise RuntimeError("gates unreadable")

    monkeypatch.setattr(vg, "vascular_gate_status", _boom)
    out2 = hr_mod.run_research_head(_Res(), {"lattice": None},
                                    runs_root=runs)
    rep2 = json.loads(pathlib.Path(out2["report_path"]).read_text())
    assert rep2["gates"]["gates_version"].startswith("unreadable")


def test_corrupt_artifact_lengths_refused_by_the_head():
    r = get_head("vascular").run(None, _ctx(vascular_model=_artifact(
        mu=[0.3])))
    assert r.value["available"] is False
    assert any("corrupt artifact" in x for x in r.reasons)


def test_public_head_results_filter_drops_research_classes():
    """Invariant-9 boundary filter (review finding: a config-enabled
    research head would ride into the browser payload as a stub)."""
    from datasets.schema import public_head_results
    rows = [{"head": "afib", "measurement_class": "INFERRED_RHYTHM"},
            {"head": "vascular",
             "measurement_class": "RESEARCH_VASCULAR"},
            {"head": "recovery", "measurement_class": "MEASURED"},
            {"head": "fitness",
             "measurement_class": "INFERRED_FITNESS"}]
    kept = [r["head"] for r in public_head_results(rows)]
    assert kept == ["afib", "recovery", "fitness"]
    assert public_head_results(None) == []
