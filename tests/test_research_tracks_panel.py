"""v0.8: the five GATED research tracks on the results screen's
RESEARCH / DEBUG METRICS panel — flag-gated, screen-only, and fenced
off from every consumer surface."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re

import pytest

from app.research_tracks import (PANEL_NOTE, WATERMARK,
                                 research_tracks_block,
                                 research_tracks_enabled)
from configs import load_config

_ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ["rhythm_regularity", "atrial_flutter", "arterial_stiffness",
        "vascular_tone", "cardiorespiratory_fitness"]


# ------------------------------------------------------------ the flag
def test_flag_is_on_by_default_and_env_wins_both_ways(monkeypatch):
    """Owner-directed default (spec B.24 #15): the tracks are in the
    research panel unless an operator turns them off. Red gates do not
    suppress them — the gate status is printed beside each value."""
    cfg = load_config()
    assert cfg["report"]["research_tracks"] is True
    monkeypatch.delenv("AVATARX_RESEARCH_TRACKS", raising=False)
    assert research_tracks_enabled(cfg) is True
    # "0" means OFF even when the config says on (the v0.4 lesson)
    monkeypatch.setenv("AVATARX_RESEARCH_TRACKS", "0")
    assert research_tracks_enabled(cfg) is False
    monkeypatch.setenv("AVATARX_RESEARCH_TRACKS", "1")
    off = dict(cfg, report=dict(cfg["report"], research_tracks=False))
    assert research_tracks_enabled(off) is True
    monkeypatch.delenv("AVATARX_RESEARCH_TRACKS")
    assert research_tracks_enabled(off) is False


def test_block_is_empty_only_when_explicitly_turned_off(monkeypatch):
    monkeypatch.setenv("AVATARX_RESEARCH_TRACKS", "0")
    assert research_tracks_block(None, {}, load_config()) == {}


# ------------------------------------------------------------ content
@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    pytest.importorskip("cv2")
    from inference.pipeline import run_with_details
    from scripts.make_synth_video import synth_video
    p = tmp_path_factory.mktemp("panel") / "rsa.avi"
    synth_video(str(p), kind="rsa", fps=30.0, duration_s=40.0, seed=11,
                torso_respiration={"brpm": 15.0, "bob_px": 6.0})
    res, det = run_with_details(str(p), manifest={}, recording_id="panel")
    return res, det, str(p)


@pytest.fixture(scope="module")
def block(scan, tmp_path_factory):
    res, det, path = scan
    return research_tracks_block(res, det, load_config(), video_path=path)


def test_every_track_is_present_with_its_live_gate_status(block):
    assert block["watermark"] == WATERMARK and block["note"] == PANEL_NOTE
    assert [t["key"] for t in block["tracks"]] == KEYS
    for t in block["tracks"]:
        g = t["gates"]
        assert t["watermark"] == WATERMARK and t["title"] and t["head"]
        assert t["status"] in ("value", "abstained", "not_run",
                               "not_applicable", "not_available_here")
        assert g["promotion"] == "BLOCKED"          # all five, today
        assert g["clinical_signoff"] is False
        assert len(g["gates"]) >= 5 and "error" not in g
        assert all(x["status"] == "RED" for x in g["gates"])
        if t["status"] in ("not_applicable", "not_available_here"):
            assert t["needs"]


def test_the_two_single_scan_tracks_carry_real_values(block):
    by = {t["key"]: t for t in block["tracks"]}
    reg = by["rhythm_regularity"]
    assert reg["status"] == "value"
    assert reg["value"]["index"]["value"] is not None
    assert reg["value"]["index"]["ci95"] is not None
    assert reg["value"]["class"] in ("regular", "irregular")
    assert reg["value"]["user_facing"] is None      # never a sentence
    fl = by["atrial_flutter"]
    assert fl["status"] in ("value", "abstained")
    assert fl["value"]["user_facing"] is None
    assert block["respiration_channel"]["rate_brpm"] == pytest.approx(
        15.0, abs=2.0)


# ------------------------------------------------------ the four fences
def test_the_consumer_payload_boundary_is_untouched(scan):
    """head_results keeps being filtered; the block rides its own key."""
    res, det, _ = scan
    from datasets.schema import public_head_results
    public = public_head_results(det.get("head_results"))
    for r in public:
        assert not str(r["measurement_class"]).startswith("RESEARCH_")
    src = (_ROOT / "app" / "scan_engine.py").read_text()
    assert 'report["research_tracks"] = blk' in src
    assert 'report["head_results"] = public_head_results(' in src
    src = (_ROOT / "app" / "session_flow.py").read_text()
    assert 'doc["research_tracks"] = blk' in src
    assert 'doc["head_results"] = public_head_results(' in src


def test_the_printed_report_never_carries_it(scan):
    """The Cardiac Rhythm Scan Report is the participant's document."""
    res, det, _ = scan
    from app.report_data import report_capture_meta
    from app.report_render import render_report
    html = render_report(res, report_capture_meta(res, det, load_config()))
    assert WATERMARK not in html
    for token in ("GATED RESEARCH", "research_tracks", "promotion",
                  "BLOCKED"):
        assert token not in html, token
    # and the panel that does carry it is hidden in print
    page = (_ROOT / "app" / "static" / "index.html").read_text()
    assert "@media print { .debug { display:none; } }" in page


def test_the_client_page_names_no_track_and_no_rhythm():
    """The renderer is generic: every label arrives in the payload, so
    the static page still asserts nothing about a rhythm (the forbidden
    -token tests scan this file and stay green)."""
    page = (_ROOT / "app" / "static" / "index.html").read_text()
    assert "researchTracksHtml" in page and "GATED RESEARCH TRACKS" in page
    # the track-naming tokens the §F/vascular/§W fences ban: zero before
    # this change and zero after (the titles arrive in the payload)
    for banned in ("flutter", r"\bSVT\b", "conduction", "vascular",
                   "vasotone", "arterial", "stiffness", "atrial", "PWV",
                   "vasodilat", "vasoconstrict", "perfusion index",
                   "artery age", "vascular age"):
        assert not re.search(banned, page, re.I), banned
    # "irregularity index" is the v0.1 MEASURED HRV row and stays as it
    # was — this change adds no new occurrence
    assert len(re.findall(r"irregular", page, re.I)) == 2


def test_no_sentence_about_the_person_anywhere_in_the_block(block):
    """Numbers, labels and the heads' own technical reasons only. The
    fence is that nothing here ADDRESSES the person or repeats a
    sanctioned consumer sentence; statements about the waveform (for
    instance that no dicrotic notch was detected) are exactly what a
    research surface is for."""
    from datasets.schema import (RATE_FLAG_SENTENCES,
                                 REGULAR_TACHY_SENTENCES,
                                 REGULARITY_SENTENCES, SESSION_SENTENCES)
    blob = json.dumps(block)
    low = blob.lower()
    # no second person anywhere
    for banned in ("your ", "you ", "you'", "yours"):
        assert banned not in low, banned
    # and no sanctioned consumer sentence, verbatim or in part
    sanctioned = (list(RATE_FLAG_SENTENCES.values())
                  + list(REGULAR_TACHY_SENTENCES.values())
                  + list(REGULARITY_SENTENCES.values())
                  + list(SESSION_SENTENCES.values()))
    for sentence in sanctioned:
        assert sentence.lower() not in low, sentence[:40]
        head = sentence.split(".")[0].lower()
        if len(head) > 25:
            assert head not in low, head[:40]
    for t in block["tracks"]:
        v = t.get("value") or {}
        assert v.get("user_facing") is None
        assert v.get("sentence") is None


# ------------------------------------------------------------ fail-soft
def test_gate_status_fails_closed(monkeypatch):
    import app.research_tracks as rt
    assert rt._gates("no.such.module:nope")["promotion"] == "BLOCKED"
    assert "fail closed" in rt._gates("no.such.module:nope")["error"]
    assert rt._gates("no.such.module:nope")["gates"] == []


def test_a_broken_builder_never_breaks_the_results_screen(monkeypatch):
    import app.research_tracks as rt

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(rt, "_build", boom)
    blk = rt.research_tracks_block(None, {}, load_config())
    assert blk["tracks"] == [] and "kaboom" in blk["error"]
    assert blk["watermark"] == WATERMARK


def test_a_scan_with_no_lattice_marks_the_rhythm_heads_not_run():
    import types
    shim = types.SimpleNamespace(
        outcome=types.SimpleNamespace(value="NO_RESULT"),
        no_read_reasons=["no face detected in the video"],
        recording_id="dark")
    blk = research_tracks_block(shim, {}, load_config())
    by = {t["key"]: t for t in blk["tracks"]}
    for k in ("rhythm_regularity", "atrial_flutter"):
        assert by[k]["status"] == "not_run"
        assert by[k]["reasons"] == ["no face detected in the video"]
        assert "never ran" in by[k]["note"]
