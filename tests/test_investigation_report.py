"""v0.8 — the Research & Investigation Report: every gated head's raw
output for one scan, watermarked, beside its track's live gate status,
QUARANTINED from every participant surface."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ast
import json
import os
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]

TRACK_KEYS = {"rhythm_regularity", "atrial_flutter", "arterial_stiffness",
              "vascular_tone", "cardiorespiratory_fitness"}


# ------------------------------------------------------------ quarantine
def test_no_participant_surface_can_reach_the_investigation_report():
    """app/ and inference/ never import research.investigation; the
    server has no route to it; the consumer renderer does not know it."""
    for d in ("app", "inference", "heads", "datasets"):
        for py in (_ROOT / d).rglob("*.py"):
            tree = ast.parse(py.read_text())
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                for m in mods:
                    assert not m.startswith("research"), (py, m)
    server = (_ROOT / "app" / "server.py").read_text().lower()
    assert "investigation" not in server and "research-report" not in server
    renderer = (_ROOT / "app" / "report_render.py").read_text().lower()
    assert "investigation" not in renderer
    # and the only producer is the explicit CLI verb
    cli = (_ROOT / "cli.py").read_text()
    assert 'add_parser("research-report"' in cli
    body = cli.split("def cmd_process")[1]
    body = body[:body.index("\ndef ")]          # cmd_process alone
    assert "research.investigation" not in body
    body = cli.split("def cmd_session")[1]
    body = body[:body.index("\ndef ")]
    assert "research.investigation" not in body


def test_track_table_covers_the_five_named_tracks():
    from research.investigation import TRACKS, WATERMARK
    assert {t[0] for t in TRACKS} == TRACK_KEYS
    for word in ("RESEARCH", "INVESTIGATION", "NOT VALIDATED",
                 "NOT A DIAGNOSIS", "NOT FOR PARTICIPANT"):
        assert word in WATERMARK


# ------------------------------------------------------------ the document
@pytest.fixture(scope="module", autouse=True)
def _research_roots(tmp_path_factory):
    """The vascular and vasotone research runners write their own
    per-scan reports under their tracks' runs roots; tests must not
    litter the repository's live roots (gitignored, but live evidence
    roots all the same). Redirect both for the module, in-process and
    for the CLI subprocesses via the env the modules honour."""
    root = tmp_path_factory.mktemp("research_roots")
    mp = pytest.MonkeyPatch()
    import evaluation.vascular_gates as vg
    import evaluation.vasotone_gates as wg
    mp.setattr(vg, "DEFAULT_RUNS", root / "vascular")
    mp.setattr(wg, "DEFAULT_RUNS", root / "vasotone")
    mp.setenv("AVATARX_VASCULAR_RUNS", str(root / "vascular"))
    mp.setenv("AVATARX_VASOTONE_RUNS", str(root / "vasotone"))
    yield root
    mp.undo()


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    pytest.importorskip("cv2")
    from scripts.make_synth_video import synth_video
    d = tmp_path_factory.mktemp("inv")
    p = d / "rsa30.avi"
    synth_video(str(p), kind="rsa", fps=30.0, duration_s=40.0, seed=11,
                torso_respiration={"brpm": 15.0, "bob_px": 6.0})
    return p


@pytest.fixture(scope="module")
def doc(clip, tmp_path_factory):
    from research.investigation.report import build_investigation_report
    return build_investigation_report(str(clip), age_years=58,
                                      recording_id="inv-test")


def test_watermark_first_and_every_track_present(doc):
    from research.investigation import WATERMARK
    assert list(doc)[0] == "WATERMARK" and doc["WATERMARK"] == WATERMARK
    assert set(doc["tracks"]) == TRACK_KEYS
    for key, t in doc["tracks"].items():
        assert t["status"] in ("value", "abstained", "not_run",
                               "not_applicable"), key
        assert t["research_only"] is True, key
        assert "gates" in t and t["gates"]["promotion"] in ("BLOCKED",
                                                            "OPEN"), key
        assert t["gates"]["promotion"] == "BLOCKED", key   # today
        assert t["watermark"] == WATERMARK, key
        assert t["title"] and t["head"], key


def test_single_scan_tracks_carry_values_and_reasons(doc):
    reg = doc["tracks"]["rhythm_regularity"]
    assert reg["status"] in ("value", "abstained")
    assert reg["head_result"]["measurement_class"] == "RESEARCH_RHYTHM"
    v = reg["head_result"]["value"]
    assert v["index"] is not None and v["index"]["ci95"] is not None
    if reg["status"] == "value":
        assert v["class"] in ("regular", "irregular")
        assert v["benign_pattern_evidence"]["evidence"] in (
            "respiration_coupled", "ectopy_pattern", "chaotic",
            "indeterminate", "not_applicable")
    fl = doc["tracks"]["atrial_flutter"]
    assert fl["head_result"]["measurement_class"] == "RESEARCH_RHYTHM"
    assert fl["status"] in ("value", "abstained")
    assert fl["head_result"]["reasons"]
    va = doc["tracks"]["arterial_stiffness"]
    assert va["head_result"]["measurement_class"] == "RESEARCH_VASCULAR"
    assert va["status"] in ("value", "abstained")
    # the raw morphology features ride with the section on an ACCEPT
    # scan, whatever the head decided
    assert isinstance(va["morphology_features"], dict)
    assert va["side_effect"]
    if doc["consumer_result"]["outcome"] == "ACCEPT":
        assert va["morphology_features"].get("available", True) is not False


def test_tracks_that_need_a_protocol_say_what_they_need(doc):
    vt = doc["tracks"]["vascular_tone"]
    assert vt["status"] == "not_applicable"
    assert "provocation" in vt["needs"].lower()
    vo = doc["tracks"]["cardiorespiratory_fitness"]
    assert vo["status"] == "not_applicable"
    assert "session" in vo["needs"].lower()


def test_consumer_context_is_quoted_never_authored(doc):
    from datasets.schema import ScanResult
    c = doc["consumer_result"]
    assert c["outcome"] in ("ACCEPT", "REPEAT_SCAN", "NO_RESULT")
    assert c["report_title"] == "AvatarX Cardiac Rhythm Scan Report"
    # the sentence is the one the participant saw, verbatim
    assert isinstance(c["user_facing_text"], str) and c["user_facing_text"]


def test_html_carries_the_watermark_on_every_section(doc):
    from research.investigation import WATERMARK
    from research.investigation.report import render_investigation_html
    import html as _h
    html = render_investigation_html(doc)
    assert html.count(_h.escape(WATERMARK)) >= 1 + len(TRACK_KEYS)
    assert "AvatarX Research &amp; Investigation Report" in html
    for t in doc["tracks"].values():
        assert t["title"] in html or t["title"].replace("&", "&amp;") in html
        assert "BLOCKED" in html
    assert "NOT the consumer scan report" in html
    # no waveform, no ECG-paper mimicry, nothing red
    for banned in ("<svg", "mm/s", "mV", "#ff0000", "red;"):
        assert banned not in html, banned


def test_consumer_report_on_the_same_scan_is_untouched(clip):
    """The participant's document knows nothing of this one."""
    from inference.pipeline import run_with_details
    from app.report_render import render_report
    from app.report_data import report_capture_meta
    from configs import load_config
    from research.investigation import WATERMARK
    res, det = run_with_details(str(clip), manifest={}, recording_id="c")
    html = render_report(res, report_capture_meta(res, det, load_config()))
    assert WATERMARK not in html and "Investigation" not in html
    from datasets.schema import public_head_results
    for r in public_head_results(det.get("head_results")):
        assert not str(r.get("measurement_class")).startswith("RESEARCH_")


def test_written_files_and_cli_verb(clip, tmp_path):
    from research.investigation import WATERMARK
    env = dict(os.environ)
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "research-report", str(clip), "--age", "58",
                        "--out", str(tmp_path / "inv")],
                       capture_output=True, text=True, timeout=900, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout)
    assert list(out)[0] == "WATERMARK" and out["WATERMARK"] == WATERMARK
    assert set(out["tracks"]) == TRACK_KEYS
    paths = out["written"]
    assert pathlib.Path(paths["json_path"]).exists()
    assert pathlib.Path(paths["html_path"]).exists()
    import html as _h
    assert pathlib.Path(paths["html_path"]).read_text().count(
        _h.escape(WATERMARK)) >= 6
    # the consumer verb on the same clip never mentions it
    r2 = subprocess.run([sys.executable, str(_ROOT / "cli.py"), "process",
                         str(clip)], capture_output=True, text=True,
                        timeout=900, env=env)
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert "Investigation" not in r2.stdout and WATERMARK not in r2.stdout
    assert "research-report" not in r2.stdout


def test_session_manifest_runs_the_fitness_track(tmp_path):
    """The §V track needs the three-phase session; given one, the
    document carries the fitness head's raw result and the session
    summary, still watermarked and BLOCKED."""
    pytest.importorskip("cv2")
    from scripts.make_synth_recovery import make_recovery_session
    from research.investigation.report import build_investigation_report
    d = tmp_path / "sess"
    make_recovery_session(d, "s1", seed=12)
    manifest = d / "s1.session.json"
    assert manifest.exists()
    doc = build_investigation_report(str(d / "s1_rest.avi"), age_years=44,
                                     session_manifest=str(manifest),
                                     recording_id="s1-rest")
    vo = doc["tracks"]["cardiorespiratory_fitness"]
    assert vo["status"] in ("value", "abstained"), vo.get("note")
    assert vo["gates"]["promotion"] == "BLOCKED"
    hr = vo["head_result"]
    assert hr["measurement_class"] == "INFERRED_FITNESS"
    assert "session" in vo and vo["session"]["outcome"] in (
        "ACCEPT", "REPEAT_SCAN", "NO_RESULT")
    assert "mL/kg/min" not in json.dumps(doc)
    import re
    assert not re.search(r"m[lL]\s*/\s*kg\s*/\s*min", json.dumps(doc))


def test_gate_status_fails_closed_and_tolerates_a_bare_status_doc(monkeypatch):
    """A track whose status function raises, or returns a document
    without a gate list, reads BLOCKED with the error recorded — never
    OPEN, never a missing section."""
    import research.investigation.report as rep
    import types, sys as _sys
    mod = types.ModuleType("_inv_fake_gates")

    def boom():
        raise RuntimeError("scoreboard unreadable")

    def bare():
        return {"promotion": "OPEN"}          # no gates, no version

    mod.boom, mod.bare = boom, bare
    monkeypatch.setitem(_sys.modules, "_inv_fake_gates", mod)
    g = rep._gate_status("_inv_fake_gates:boom")
    assert g["promotion"] == "BLOCKED" and "fail closed" in g["error"]
    assert g["gates"] == [] and g["clinical_signoff"] is False
    g2 = rep._gate_status("_inv_fake_gates:bare")
    assert g2["promotion"] == "OPEN" and g2["gates"] == []
    assert g2["clinical_signoff"] is False
    # and the sanctioned five all answer with a gate list today
    from research.investigation import TRACKS
    for _, _, _, _, spec in TRACKS:
        g3 = rep._gate_status(spec)
        assert g3["promotion"] == "BLOCKED" and len(g3["gates"]) >= 5, spec
        assert "error" not in g3, spec


def test_a_scan_without_a_lattice_marks_the_rhythm_heads_not_run(tmp_path):
    """A NO_RESULT scan never ran the rhythm heads: the sections say so
    with the scan's reasons, and are not dressed up as abstentions."""
    pytest.importorskip("cv2")
    from scripts.make_synth_video import synth_video
    from research.investigation.report import build_investigation_report
    p = tmp_path / "dark.avi"
    synth_video(str(p), kind="sinus", fps=30.0, duration_s=20.0, seed=3,
                lux_scale=0.15)
    doc = build_investigation_report(str(p), recording_id="dark")
    assert doc["consumer_result"]["outcome"] == "NO_RESULT"
    for key in ("rhythm_regularity", "atrial_flutter"):
        t = doc["tracks"][key]
        assert t["status"] == "not_run", (key, t)
        assert t["head_result"]["value"] is None
        assert t["head_result"]["reasons"] == doc["consumer_result"][
            "no_read_reasons"]


def test_fitness_abstention_is_labelled_abstained():
    from research.investigation.report import _status_of
    assert _status_of({"head": "fitness", "value": {"category": None,
                                                    "routed": None}}) == \
        "abstained"
    assert _status_of({"head": "fitness", "value": {"category": "typical",
                                                    "routed": None}}) == \
        "value"
    assert _status_of({"head": "vascular", "value": {"available": False}}) \
        == "abstained"
    assert _status_of({"head": "regularity", "value": {"abstained": True}}) \
        == "abstained"


def test_cli_rejects_invalid_arguments_with_exit_2(clip, tmp_path):
    env = dict(os.environ)

    def run(*extra):
        return subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                               "research-report", *extra],
                              capture_output=True, text=True, timeout=600,
                              env=env)
    bad = tmp_path / "nope.json"
    assert run(str(tmp_path / "missing.avi")).returncode == 2
    r = run(str(clip), "--session", str(bad))
    assert r.returncode == 2 and "session manifest not found" in r.stderr
    (tmp_path / "list.json").write_text("[1, 2]")
    r = run(str(clip), "--session", str(tmp_path / "list.json"))
    assert r.returncode == 2 and "JSON object" in r.stderr
    r = run(str(clip), "--videos", "a.avi")
    assert r.returncode == 2 and "--videos needs --session" in r.stderr
    r = run(str(clip), "--pi", str(bad))
    assert r.returncode == 2 and "--pi needs --provocation" in r.stderr
    (tmp_path / "prov.json").write_text('{"phase_marks": null}')
    r = run(str(clip), "--provocation", str(tmp_path / "prov.json"))
    assert r.returncode == 2 and "invalid provocation" in r.stderr
    assert "Traceback" not in r.stderr
    r = run(str(clip), "--age", "-5")
    assert r.returncode == 2 and "plausible age" in r.stderr
    r = run(str(clip), "--manifest", str(tmp_path / "list.json"))
    assert r.returncode == 2 and "invalid manifest" in r.stderr
