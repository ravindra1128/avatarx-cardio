"""v0.2.1 Task 5 — forbidden-content tests: the report borrows LAYOUT
familiarity, never ECG identity. These tests are the visual-language
extension of the never-diagnose invariant."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ast
import math
import re

from app.report_render import (render_report, render_report_fragment,
                               REPORT_TITLE, STRIP_LABEL,
                               REFERRAL_SENTENCE, STRIP_SECONDS)
from datasets.schema import ScanResult, ScanOutcome
from tests.test_report_render import (_accept_result, _fixture_signal,
                                      _meta)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_HTML_PAGE = (_ROOT / "app" / "static" / "index.html").read_text()

FORBIDDEN_TOKENS = ["aVR", "aVL", "aVF", "mm/s",
                    "Lead I", "Lead II", "Lead III",
                    "paper speed", "25 mm", "50 mm"]
FORBIDDEN_WORDS = [rf"\bV{i}\b" for i in range(1, 7)] + [r"\bmV\b"]


def _n_rows(seconds=30.0):
    return int(math.ceil(seconds / STRIP_SECONDS))


def test_not_an_ecg_once_per_strip_row():
    frag = render_report_fragment(_accept_result(), _meta())
    assert frag.count("NOT AN ECG") == _n_rows() == 3


def test_ecg_substring_count_is_exactly_accounted_for():
    """R4 (reconciled with R5 per spec B.16): 'ECG' = one per strip label
    + one referral + any occurrences REQUIRED by the sanctioned sentence
    itself. Nothing else — not title, metadata, axes, or alt text."""
    for cls in ("SINUS", "AFIB_SUGGESTIVE"):
        r = _accept_result()
        r.predicted_class = cls
        frag = render_report_fragment(r, _meta())
        expected = _n_rows() + 1 + r.user_facing_text().count("ECG")
        assert frag.count("ECG") == expected, (cls, frag.count("ECG"),
                                               expected)
    assert "inferred ECG" not in render_report(_accept_result(), _meta())
    assert "Inferred ECG" not in _HTML_PAGE and "inferred ECG" not in _HTML_PAGE


def test_title_exact_and_free_of_ecg():
    doc = render_report(_accept_result(), _meta())
    m = re.search(r"<title>(.*?)</title>", doc)
    assert m and m.group(1) == REPORT_TITLE
    assert "ECG" not in REPORT_TITLE
    frag = render_report_fragment(_accept_result(), _meta())
    assert frag.count(f"<h1>{REPORT_TITLE}</h1>") == 1


def test_no_ecg_paper_mimicry_tokens():
    for outcome_cls in (("ACCEPT", "SINUS"), ("NO_RESULT", None)):
        r = _accept_result()
        r.outcome = ScanOutcome(outcome_cls[0])
        r.predicted_class = outcome_cls[1]
        doc = render_report(r, _meta())
        for tok in FORBIDDEN_TOKENS:
            assert tok not in doc, tok
        for pat in FORBIDDEN_WORDS:
            assert not re.search(pat, doc), pat
        for word in ("red", "pink", "crimson", "salmon"):
            assert not re.search(rf"\b{word}\b", doc), word


def test_all_report_colours_are_non_red():
    """No pink/red ECG-paper styling anywhere: every hex colour in the
    rendered document must not be red-dominant."""
    doc = render_report(_accept_result(), _meta())
    for hexcol in set(re.findall(r"#([0-9a-fA-F]{6})\b", doc)):
        r = int(hexcol[0:2], 16)
        g = int(hexcol[2:4], 16)
        b = int(hexcol[4:6], 16)
        assert r <= max(g, b) + 40, f"red-dominant colour #{hexcol}"


def test_axes_are_seconds_and_normalized_au():
    frag = render_report_fragment(_accept_result(), _meta())
    assert "normalized a.u." in frag
    assert re.search(r"\d+ s</text>", frag)          # seconds axis labels
    assert re.search(r"\d+ ms</text>", frag)         # tachogram in ms


def test_no_synthesized_waveform_code_path_in_app():
    """Import audit: nothing under app/ (report renderer included) may
    reach the quarantined generator, and the renderer never calls a
    decode/synthesise routine."""
    for py in (_ROOT / "app").rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert not m.startswith("evaluation.inferred_ecg"), (py, m)
    # identifier-level audit of the renderer (prose/docstrings exempt):
    # no call or name may reference a generator/decoder routine
    tree = ast.parse((_ROOT / "app" / "report_render.py").read_text())
    banned_names = {"decode", "train_decoder", "generate_ecg",
                    "synthesize", "synthesise"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in banned_names, node.id
        if isinstance(node, ast.Attribute):
            assert node.attr not in banned_names, node.attr


def test_no_result_renders_full_report_with_reasons():
    r = ScanResult("nr", ScanOutcome.NO_RESULT,
                   no_read_reasons=["only 3 clean intervals — too few"],
                   confidence_stars=1, confidence_limiting_factor="sqi")
    frag = render_report_fragment(r, _meta(sig={}))
    assert REPORT_TITLE in frag
    assert "Why no result" in frag and "clean intervals" in frag
    assert "insufficient quality" in frag
    assert "MEASUREMENTS" in frag and "FINDINGS" in frag
    assert REFERRAL_SENTENCE in frag


def test_referral_sentence_verbatim_and_label_verbatim():
    frag = render_report_fragment(_accept_result(), _meta())
    assert REFERRAL_SENTENCE == ("This screening result is not a diagnosis. "
                                 "A positive or uncertain result should be "
                                 "confirmed with an ECG.")
    assert STRIP_LABEL == "FACIAL PULSE WAVEFORM (rPPG) — NOT AN ECG"
    assert STRIP_LABEL in frag and REFERRAL_SENTENCE in frag
