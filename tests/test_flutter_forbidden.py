"""Invariant F-b (v0.6) — while §F is red, the strings "flutter",
"SVT", "atrial tachycardia", "2:1" and "conduction ratio" are forbidden
on every user-facing surface. The sanctioned regular-tachy sentence
(F-a) is deliberately clean of all of them: it describes what was seen
(fast, unusually regular) and routes to an ECG, because the pulse
cannot name the rhythm.

Operator docs (README/RUNBOOK/docs/) and research artifacts name these
constructs freely and are out of scope — they are not participant
surfaces.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re

from tests.test_scanresult_v2 import _all_sanctioned_strings

_ROOT = pathlib.Path(__file__).resolve().parents[1]
FLUT_RE = re.compile(
    r"flutter|\bSVT\b|atrial[\s_-]+tachycardia"
    r"|\b\d\s*[:：]\s*1\b|conduction[\s_-]+rati", re.I)


def test_the_regex_itself_catches_each_forbidden_form():
    for bad in ("atrial flutter", "Flutter suspected", "SVT",
                "atrial tachycardia", "atrial-tachycardia",
                "2:1 conduction", "2 : 1", "4:1", "conduction ratio",
                "conduction-ratio step"):
        assert FLUT_RE.search(bad), bad
    # ... and does not fire on the vocabulary the product legitimately
    # uses, including the sanctioned sentence's own words
    for ok in ("a sustained fast, unusually regular pulse",
               "please have an ECG", "150 beats per minute",
               "your pulse was faster than usual", "svtable",
               "regular rhythm", "scan started 9:15"):
        assert not FLUT_RE.search(ok), ok
    # a bare "1:1" is caught too, deliberately: any n:1 statement about
    # the heart is a conduction claim this product must not make
    assert FLUT_RE.search("1:1 conduction")


def test_every_sanctioned_string_is_clean():
    from datasets.schema import REGULAR_TACHY_SENTENCES
    for s in _all_sanctioned_strings():
        assert not FLUT_RE.search(s), s
    # including the v0.6 sentence itself — it must survive its own rule
    for s in REGULAR_TACHY_SENTENCES.values():
        assert not FLUT_RE.search(s), s


def test_client_page_is_clean():
    html = (_ROOT / "app" / "static" / "index.html").read_text()
    assert not FLUT_RE.search(html)


def test_rendered_reports_are_clean():
    from tests.test_report_render import _accept_result, _meta
    from app.report_render import render_report
    for mode in ("findings_only", "full"):
        meta = _meta()
        meta["report_mode"] = mode
        assert not FLUT_RE.search(render_report(_accept_result(), meta)), \
            mode


def test_consumer_payload_is_clean_even_with_the_head_enabled(tmp_path):
    """The head carries a conduction band internally (that is its job);
    the payload boundary is what must never show it."""
    import pytest
    pytest.importorskip("cv2")
    from datasets.schema import public_head_results
    from inference.pipeline import run_with_details
    from scripts.make_synth_flutter import flutter_rr
    from scripts.make_synth_video import synth_video
    v = str(tmp_path / "f.avi")
    synth_video(v, kind="flutter_track_2to1", fps=30.0, duration_s=40.0,
                seed=5, rr_override=flutter_rr(40.0, ratio=2, seed=5))
    res, _ = run_with_details(v, heads=["afib", "flutter"])
    public = public_head_results(res.head_results)
    assert not FLUT_RE.search(json.dumps(public))
    assert not FLUT_RE.search(res.user_facing_text())
    # the head's own record is where the band lives, and it is stripped
    raw = [h for h in res.head_results if h["head"] == "flutter"]
    assert raw and raw[0]["measurement_class"] == "RESEARCH_RHYTHM"
    assert public == [h for h in res.head_results
                      if h["head"] != "flutter"]


def test_a_flagging_head_result_still_renders_no_forbidden_token():
    """Force the flag on, then confirm the only thing that can reach a
    human is the sanctioned sentence."""
    import numpy as np
    from beats.lattice import BeatLattice, LATTICE_VERSION
    from datasets.schema import REGULAR_TACHY_SENTENCES
    from heads import get_head
    from heads.head_flutter import user_facing_text
    rng = np.random.default_rng(0)
    rr = np.clip(0.4 + rng.normal(0, 0.003, 100), 0.25, 1.5)
    t = np.cumsum(rr)
    lat = BeatLattice(
        version=LATTICE_VERSION, fps=240.0, duration_s=float(t[-1] + 1),
        beat_t_s=np.concatenate([[0.0], t]),
        beat_confidence=np.ones(t.size + 1),
        beat_agreement=np.ones(t.size + 1), runs=[rr * 1000.0],
        run_confidences=[np.ones(rr.size)], n_intervals=int(rr.size),
        dropout_rate=0.0, split_fraction=0.0, per_roi_times={},
        segments=[], run_times=[t])
    tt = np.arange(0, 45, 1 / 30)
    resp = {"t": tt, "y": np.sin(2 * np.pi * 0.25 * tt),
            "rate_brpm": 15.0, "quality": 0.9}
    r = get_head("flutter").run(lat, {"scan_outcome": "ACCEPT",
                                      "cfg": {}, "respiration": resp,
                                      "age_years": 60})
    assert r.value["regular_tachy_flag"] is True
    assert FLUT_RE.search(json.dumps(r.value))        # internally: yes
    text = user_facing_text(r.value, render_allowed=True)
    assert text == REGULAR_TACHY_SENTENCES["regular_tachy"]
    assert not FLUT_RE.search(text)                   # to a human: no
    assert "ECG" in text


# ------------------------------------------------ F-c: the known misses
def test_the_known_miss_registry_is_a_shipped_artifact():
    """F-c: docs/flutter_limitations.md must exist and must state the
    permanent misses. A gate (F5) depends on this file being on disk."""
    doc = _ROOT / "docs" / "flutter_limitations.md"
    assert doc.exists()
    text = doc.read_text()
    low = text.lower()
    # the headline miss, stated explicitly
    assert "4:1" in text and "~75" in text
    assert "indistinguishable from normal sinus rhythm" in low
    assert "permanent miss" in low
    # naming impossibility and the SVT overlap
    assert "f wave" in low or "f waves" in low
    assert "svt" in low and "covered by design" in low
    # the dominant confounder
    assert "sinus tachycardia" in low
    assert "specificity" in low
    # the measured capture limitation
    assert "15.0 ms at 30 fps" in text and "8.2 ms at 60 fps" in text
    # and the negative-scan statement
    assert "does not exclude atrial flutter" in low


def test_the_product_does_not_contradict_the_known_miss_registry():
    """The doc says a 4:1-rate metronomic pulse is not distinguishable
    from sinus by timing. The head must behave that way."""
    import numpy as np
    from beats.lattice import BeatLattice, LATTICE_VERSION
    from heads import get_head
    rng = np.random.default_rng(1)
    out = {}
    for name, base in (("four_to_one", 0.800), ("two_to_one", 0.400)):
        rr = np.clip(base + rng.normal(0, 0.003, 100), 0.25, 1.5)
        t = np.cumsum(rr)
        lat = BeatLattice(
            version=LATTICE_VERSION, fps=240.0,
            duration_s=float(t[-1] + 1),
            beat_t_s=np.concatenate([[0.0], t]),
            beat_confidence=np.ones(t.size + 1),
            beat_agreement=np.ones(t.size + 1), runs=[rr * 1000.0],
            run_confidences=[np.ones(rr.size)], n_intervals=int(rr.size),
            dropout_rate=0.0, split_fraction=0.0, per_roi_times={},
            segments=[], run_times=[t])
        tt = np.arange(0, float(t[-1]) + 5, 1 / 30)
        resp = {"t": tt, "y": np.sin(2 * np.pi * 0.25 * tt),
                "rate_brpm": 15.0, "quality": 0.9}
        out[name] = get_head("flutter").run(
            lat, {"scan_outcome": "ACCEPT", "cfg": {},
                  "respiration": resp, "age_years": 60}).value
    # the documented miss: hyper-regular, and still not flagged
    assert out["four_to_one"]["regular_tachy_flag"] is False
    assert out["four_to_one"]["evidence"]["regularity"][
        "below_floor"] is True
    # while the documented target is
    assert out["two_to_one"]["regular_tachy_flag"] is True


def test_no_surface_claims_a_negative_scan_excludes_anything():
    from datasets.schema import REGULAR_TACHY_SENTENCES
    for s in _all_sanctioned_strings():
        low = s.lower()
        for claim in ("no flutter", "rules out", "ruled out",
                      "rhythm is normal", "your heart is fine"):
            assert claim not in low, (claim, s)
    assert "rule out" not in \
        REGULAR_TACHY_SENTENCES["regular_tachy"].lower()


def test_the_cli_consumer_json_strips_the_research_head(tmp_path):
    """The app boundary filters RESEARCH_* head results; the CLI must
    too. head_flutter runs ON the measured path by design (unlike the
    vascular/vasotone heads, which are stripped from the pipeline), so
    without this filter its payload — including its sentence key — was
    printed to stdout while §F is red."""
    import os
    import subprocess
    import sys as _sys
    import pytest
    pytest.importorskip("cv2")
    from scripts.make_synth_flutter import flutter_rr
    from scripts.make_synth_video import synth_video
    v = str(tmp_path / "f.avi")
    synth_video(v, kind="flutter_track_2to1", fps=30.0, duration_s=30.0,
                seed=5, rr_override=flutter_rr(30.0, ratio=2, seed=5),
                torso_respiration={"brpm": 15.0, "bob_px": 6.0})
    r = subprocess.run([_sys.executable, str(_ROOT / "cli.py"),
                        "process", v, "--heads", "afib,flutter"],
                       capture_output=True, text=True, timeout=900,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-1500:]
    doc = json.loads(r.stdout)
    heads = [h.get("head") for h in (doc.get("head_results") or [])]
    assert "flutter" not in heads, heads
    assert "afib" in heads
    assert not FLUT_RE.search(json.dumps(doc))
    assert "RESEARCH_RHYTHM" not in r.stdout
    # the sanctioned AF-path sentence is still printed verbatim
    assert doc["user_facing_text"]
