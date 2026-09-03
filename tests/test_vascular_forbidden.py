"""Invariant V-b (v0.4 vascular) — the strings "vascular age",
"artery age", "PWV"/"pulse wave velocity" and any m/s value are
forbidden on every user-facing surface while the vascular gates are
red. Swept: the full sanctioned-string enumeration (every
user_facing_text state + every sentence table), the client page, the
rendered rhythm report in both modes, the session fragment WITH the
fitness branch forced open, and the head's pipeline-reachable output.
Operator docs (README/RUNBOOK/docs/) legitimately name PWV and are
deliberately out of scope — they are not participant surfaces."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re

from tests.test_scanresult_v2 import _all_sanctioned_strings

_ROOT = pathlib.Path(__file__).resolve().parents[1]
VASC_RE = re.compile(
    r"vascular[\s_-]+age|artery[\s_-]+age|PWV|pulse[\s_-]*wave[\s_-]*velocity"
    r"|\d+(?:\.\d+)?\s*m/s", re.I)


def test_the_regex_itself_catches_each_forbidden_form():
    for bad in ("your vascular age is", "artery age: 62", "cfPWV",
                "PWV 8", "pulse wave velocity", "pulse-wave velocity",
                "vascular-age", "8.4 m/s", "12 m/s"):
        assert VASC_RE.search(bad), bad
    # units already in the app must not false-positive
    for ok in ("71 bpm", "30.0 fps", "2 s", "-3 bpm/min", "25 mm/s"):
        assert not VASC_RE.search(ok), ok


def test_every_sanctioned_string_is_clean():
    for s in _all_sanctioned_strings():
        assert not VASC_RE.search(s), s


def test_client_page_is_clean():
    html = (_ROOT / "app" / "static" / "index.html").read_text()
    assert not VASC_RE.search(html)


def test_rendered_reports_are_clean():
    from tests.test_report_render import _accept_result, _meta
    from app.report_render import render_report
    for mode in ("findings_only", "full"):
        meta = _meta()
        meta["report_mode"] = mode
        doc = render_report(_accept_result(), meta)
        assert not VASC_RE.search(doc), mode


def test_session_fragment_clean_even_with_fitness_branch_open():
    from app.report_session import render_session_fragment
    from datasets.schema import SessionResult
    from datasets.schema import ScanOutcome
    r = SessionResult("s1", ScanOutcome.ACCEPT,
                      protocol_id="sts_1min", activity_performed=True,
                      hr_rest_bpm=64.0, hrr60_bpm=22.0,
                      confidence_stars=4,
                      fitness_category="typical",
                      trend={"direction": "stable",
                             "magnitude_class": None, "n_sessions": 3})
    html = render_session_fragment(r, _render_allowed=lambda: True)
    assert not VASC_RE.search(html)


def test_pipeline_reachable_head_output_is_clean():
    from heads import get_head
    r = get_head("vascular").run(None, {})
    assert not VASC_RE.search(json.dumps(r.to_dict()))


def test_vascular_status_note_never_says_pwv():
    # even the research scoreboard's banner note stays token-clean —
    # it is quoted in operator terminals that users can glance at
    from evaluation.vascular_gates import TRACK_NOTE
    assert not VASC_RE.search(TRACK_NOTE)
