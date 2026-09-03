"""v0.5 vasotone T4 — head_vasotone: registered but never default-
enabled, quarantine-clean (AST-pinned), inert and number-free on every
pipeline-reachable surface (W-a), reading only when the research surface
supplies the recorded rule + surviving features, refusing stale/
uncontrolled/floorless inputs, and structurally unable to emit an
absolute tone level (W-c)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re

import pytest

from configs import load_config
from datasets.schema import MeasurementClass
from heads import enabled_heads, get_head

_ROOT = pathlib.Path(__file__).resolve().parents[1]
TONE_RE = re.compile(
    r"vascular[\s_-]+tone|vasoconstrict|vasodilat|perfusion[\s_-]+index"
    r"|endothelial", re.I)


def _feats(**over):
    d = {"available": True, "uncontrolled_optics": False,
         "maneuver": "cold_pressor",
         "features": {"norm_pulse_amplitude":
                      {"baseline": 0.02, "response": 0.014,
                       "delta": -0.006, "delta_norm": -0.3},
                      "notch_rel_amp":
                      {"baseline": 0.1, "response": 0.12,
                       "delta": 0.02, "delta_norm": 0.2}},
         "n_beats": {"baseline": 25, "response": 25}}
    d.update(over)
    return d


def _rule(**over):
    d = {"kind": "reactivity_primary",
         "primary_feature": "norm_pulse_amplitude",
         "surviving_features": ["norm_pulse_amplitude", "notch_rel_amp"],
         "detection_floor_delta_norm": 0.05, "n_provocations": 80}
    d.update(over)
    return d


def _ctx(**over):
    d = {"vasotone_research_surface": True,
         "vasotone_features": _feats(),
         "vasotone_model": _rule(),
         "surviving_features": ["norm_pulse_amplitude", "notch_rel_amp"],
         "pi_response": {"baseline": 2.0, "response": 1.4,
                         "delta": -0.6, "delta_norm": -0.3}}
    d.update(over)
    return d


def test_registered_research_only_never_default_enabled():
    h = get_head("vasotone")
    assert h.research_only is True
    assert "vasotone" not in [x.name for x in
                              enabled_heads(load_config())]


def test_head_module_never_imports_research_code():
    import ast
    tree = ast.parse((_ROOT / "heads" / "head_vasotone.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith(("research", "importlib"))
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(
                ("research", "importlib"))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", getattr(fn, "attr", ""))
            assert name not in ("__import__", "import_module")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                assert not node.value.startswith("research.")


def test_watermark_matches_track_constant():
    from heads.head_vasotone import WATERMARK as head_wm
    from research.vascular import WATERMARK as track_wm
    assert head_wm == track_wm


def test_inert_and_number_free_on_every_pipeline_surface():
    h = get_head("vasotone")
    for ctx in ({}, {"cfg": {}},
                {"vasotone_research_surface": "yes"},
                {"vasotone_research_surface": False}):
        r = h.run(None, ctx)
        assert r.measurement_class is MeasurementClass.RESEARCH_VASCULAR
        assert r.value == {"available": False}
        blob = json.dumps(r.to_dict())
        assert not TONE_RE.search(blob), blob
        assert not re.search(r"\d", json.dumps(r.value))
        assert any("W-a" in x for x in r.reasons)


def test_research_surface_reading_is_delta_only():
    r = get_head("vasotone").run(None, _ctx())
    v = r.value
    assert v["available"] is True
    assert v["response_delta_norm"] == -0.3
    assert v["detected"] is True and v["direction"] == "decrease"
    assert v["magnitude_vs_drift"] == 6.0
    assert v["contact_reference"]["direction_agrees"] is True
    # W-c structural: no absolute tone level anywhere in the value —
    # a RECURSIVE key walk (review finding: a top-level substring check
    # would miss a re-emitted baseline/response absolute deeper down)
    assert "watermark" in v and "RESEARCH ARTIFACT" in v["watermark"]

    def _keys(obj):
        out = set()
        if isinstance(obj, dict):
            for k, val in obj.items():
                out.add(k)
                out |= _keys(val)
        elif isinstance(obj, list):
            for val in obj:
                out |= _keys(val)
        return out

    banned = {"baseline", "response", "tone_level", "tone_score",
              "absolute"}
    assert not (_keys(v) & banned), _keys(v) & banned
    for val in v["surviving_deltas"].values():
        assert val is None or isinstance(val, float)


def test_fail_closed_refusals():
    h = get_head("vasotone")
    cases = {
        "no_rule": _ctx(vasotone_model={}),
        "no_surv": _ctx(surviving_features=[]),
        "stale": _ctx(vasotone_model=_rule(
            primary_feature="vasomotor_lf_power")),
        "uncontrolled": _ctx(vasotone_features=_feats(
            uncontrolled_optics=True)),
        "no_floor": _ctx(vasotone_model=_rule(
            detection_floor_delta_norm=None)),
        "gated": _ctx(vasotone_features={"available": False,
                                         "reasons": ["V-c"]}),
    }
    expect = {"no_rule": "no recorded reactivity rule",
              "no_surv": "fail closed",
              "stale": "stale rule",
              "uncontrolled": "W-d",
              "no_floor": "detection floor",
              "gated": "V-c"}
    for name, ctx in cases.items():
        r = h.run(None, ctx)
        assert r.value["available"] is False, name
        assert any(expect[name] in x for x in r.reasons), (name,
                                                           r.reasons)
        assert "response_delta_norm" not in r.value, name


def test_run_research_tone_head_success_and_fallbacks(tmp_path,
                                                      monkeypatch):
    import research.vascular.tone_runner as tr_mod
    from evaluation.vasotone_gates import append_scoreboard

    class _Res:
        recording_id = "rid-t"

    monkeypatch.setattr(tr_mod, "tone_session_features",
                        lambda result, det, prov, capture=None: _feats())
    runs = tmp_path / "runs"
    per = {"norm_pulse_amplitude": {"null_optics_p95": 0.03,
                                    "null_rest_p95": 0.03},
           "notch_rel_amp": {"null_optics_p95": 0.03,
                             "null_rest_p95": 0.03}}
    append_scoreboard(
        {"kind": "evaluation", "run_id": "vaso-1",
         "evidence": {"data": {"signal_domain": "facial_rppg",
                               "participant_disjoint": True,
                               "session_disjoint": True,
                               "production_path": True},
                      "w1": {"n_null_optics_sessions": 30,
                             "n_null_rest_sessions": 30,
                             "per_feature": per}}}, runs_root=runs)
    (runs / "vaso-1").mkdir(parents=True)
    (runs / "vaso-1" / "model.json").write_text(json.dumps(_rule()))
    from datasets.schema import provocation_from_dict
    prov = provocation_from_dict(
        {"maneuver": "cold_pressor",
         "phase_marks": {"baseline": [0, 25], "stimulus": [25, 50]}})
    from datasets.schema import PiTrace
    from research.vascular.tone_features import pi_response as _pr
    pi = PiTrace(fs_hz=1.0, values=[2.0] * 25 + [1.4] * 35)
    out = tr_mod.run_research_tone_head(_Res(), {"lattice": None}, prov,
                                        pi=pi, runs_root=runs)
    v = out["head_result"].value
    assert v["available"] is True and v["detected"] is True
    # the contact reference rides through, value-pinned (a baseline/
    # stimulus swap would flip the sign — review finding)
    expect = _pr(pi, prov.phase_marks["baseline"],
                 prov.phase_marks["stimulus"])["delta_norm"]
    assert v["contact_reference"]["delta_norm"] == expect
    assert v["contact_reference"]["direction_agrees"] is True
    rep = json.loads(pathlib.Path(out["report_path"]).read_text())
    assert list(rep)[0] == "WATERMARK"
    assert rep["gates"]["promotion"] == "BLOCKED"
    # fallbacks: no scoreboard / wrong-kind artifact -> {} -> refusal
    assert tr_mod._latest_rule(tmp_path / "none") == {}
    runs2 = tmp_path / "runs2"
    append_scoreboard({"kind": "evaluation", "run_id": "vaso-9",
                       "evidence": {}}, runs_root=runs2)
    (runs2 / "vaso-9").mkdir(parents=True)
    (runs2 / "vaso-9" / "model.json").write_text(json.dumps(
        {"kind": "ridge_morphology"}))
    assert tr_mod._latest_rule(runs2) == {}


def test_cli_process_keeps_vasotone_out_of_the_consumer_result(tmp_path):
    pytest.importorskip("cv2")
    import os
    import subprocess
    from scripts.make_synth_video import synth_video
    v = tmp_path / "scan.avi"
    synth_video(str(v), kind="sinus", fps=30.0, duration_s=20.0, seed=31)
    prov = tmp_path / "p.provocation.json"
    prov.write_text(json.dumps(
        {"maneuver": "cold_pressor",
         "phase_marks": {"baseline": [0, 10], "stimulus": [10, 20]}}))
    env = dict(os.environ,
               AVATARX_VASOTONE_RUNS=str(tmp_path / "runs"))
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "process", str(v), "--heads", "afib,vasotone",
                        "--provocation", str(prov)],
                       capture_output=True, text=True, timeout=600,
                       env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    doc = json.loads(r.stdout)
    blob = json.dumps(doc)
    assert not TONE_RE.search(blob)
    assert "vasotone" not in [h.get("head") for h in
                              (doc.get("head_results") or [])]
    assert "vasotone research report" in r.stderr
    m = re.search(r"measurement\): (\S+)", r.stderr)
    rep = json.loads(pathlib.Path(m.group(1)).read_text())
    assert list(rep)[0] == "WATERMARK"
    # no recorded rule -> the research report itself is fail-closed
    assert rep["head_result"]["value"]["available"] is False
    # and forgetting --provocation fails softly on stderr, result intact
    r2 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "process", str(v), "--heads", "afib,vasotone"],
                        capture_output=True, text=True, timeout=600,
                        env=env)
    assert r2.returncode == 0
    json.loads(r2.stdout)
    assert "vasotone research head failed" in r2.stderr
    # with a LOCKED manifest and a --pi sidecar, the report reaches the
    # head with capture evidence and the contact reference (review
    # finding: the --pi path and the manifest lock route were untested)
    man = tmp_path / "m.recording.json"
    man.write_text(json.dumps({"capture": {"exposure_locked": True,
                                           "awb_locked": True}}))
    pi_f = tmp_path / "p.pi.json"
    pi_f.write_text(json.dumps({"fs_hz": 1.0,
                                "values": [2.0] * 15 + [1.5] * 20,
                                "t0_video_s": 0.0,
                                "device": "test"}))
    r3 = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                         "process", str(v), "--heads", "afib,vasotone",
                         "--provocation", str(prov),
                         "--pi", str(pi_f), "--manifest", str(man)],
                        capture_output=True, text=True, timeout=600,
                        env=env)
    assert r3.returncode == 0, r3.stderr[-1500:]
    m3 = re.search(r"measurement\): (\S+)", r3.stderr)
    rep3 = json.loads(pathlib.Path(m3.group(1)).read_text())
    # still fail-closed (no recorded rule) but W-d is NOT among the
    # reasons — the lock evidence arrived via the manifest
    assert rep3["head_result"]["value"]["available"] is False
    assert not any("W-d" in x
                   for x in rep3["head_result"]["reasons"])
