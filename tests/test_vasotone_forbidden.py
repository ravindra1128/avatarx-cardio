"""Invariant W-b (v0.5) — the strings "vascular tone",
"vasoconstriction" (and vasodilation), "perfusion index" and
"endothelial" are forbidden on every user-facing surface while §W is
red; W-c bans absolute tone values structurally (see the head and
extractor tests). Operator docs (README/RUNBOOK/docs/) legitimately
name these constructs and are deliberately out of scope — they are not
participant surfaces."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re

from tests.test_scanresult_v2 import _all_sanctioned_strings

_ROOT = pathlib.Path(__file__).resolve().parents[1]
TONE_RE = re.compile(
    r"vascular[\s_-]+tone|vasoconstrict|vasodilat|perfusion[\s_-]+index"
    r"|endothelial", re.I)


def test_the_regex_itself_catches_each_forbidden_form():
    for bad in ("your vascular tone", "vasoconstriction detected",
                "vasodilation", "Perfusion Index 2.1", "endothelial",
                "perfusion-index 2.1", "vascular-tone reading"):
        assert TONE_RE.search(bad), bad
    for ok in ("vasomotor reactivity", "71 bpm", "recovery scan",
               "pulse rhythm"):
        assert not TONE_RE.search(ok), ok


def test_every_sanctioned_string_is_clean():
    for s in _all_sanctioned_strings():
        assert not TONE_RE.search(s), s


def test_client_page_is_clean():
    html = (_ROOT / "app" / "static" / "index.html").read_text()
    assert not TONE_RE.search(html)


def test_rendered_reports_are_clean():
    from tests.test_report_render import _accept_result, _meta
    from app.report_render import render_report
    for mode in ("findings_only", "full"):
        meta = _meta()
        meta["report_mode"] = mode
        assert not TONE_RE.search(render_report(_accept_result(),
                                                meta)), mode


def test_session_fragment_clean_even_with_fitness_branch_open():
    from app.report_session import render_session_fragment
    from datasets.schema import ScanOutcome, SessionResult
    r = SessionResult("s1", ScanOutcome.ACCEPT,
                      protocol_id="sts_1min", activity_performed=True,
                      hr_rest_bpm=64.0, hrr60_bpm=22.0,
                      confidence_stars=4, fitness_category="typical",
                      trend={"direction": "stable",
                             "magnitude_class": None, "n_sessions": 3})
    assert not TONE_RE.search(
        render_session_fragment(r, _render_allowed=lambda: True))


def test_pipeline_reachable_head_output_is_clean():
    from heads import get_head
    r = get_head("vasotone").run(None, {})
    assert not TONE_RE.search(json.dumps(r.to_dict()))


def test_track_note_and_gate_titles_stay_token_clean():
    # the scoreboard renders in operator terminals users can glance at
    from evaluation.vasotone_gates import GATE_TITLES, TRACK_NOTE
    assert not TONE_RE.search(TRACK_NOTE)
    for t in GATE_TITLES.values():
        assert not TONE_RE.search(t), t
