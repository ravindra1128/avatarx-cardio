"""v0.2.1 Tasks 2-3 — the Cardiac Rhythm Scan Report renderer: pure
stdlib, deterministic, print-ready, honest under NO_RESULT, exported by
cli.py process --report."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ast
import json
import math
import subprocess

import pytest

from app.report_render import (render_report, render_report_fragment,
                               REPORT_TITLE, STRIP_LABEL,
                               REFERRAL_SENTENCE, STRIP_SECONDS)
from datasets.schema import ScanResult, ScanOutcome

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _fixture_signal(seconds=30.0, fps=30.0):
    """Deterministic MEASURED-shaped payload (no pipeline needed)."""
    n = int(seconds * fps)
    t = [round(i / fps, 3) for i in range(n)]
    vals = [round(math.sin(2 * math.pi * 1.2 * x)
                  + 0.3 * math.sin(2 * math.pi * 2.4 * x), 3) for x in t]
    beats = [{"t": round(b, 3), "conf": 0.9 if i % 4 else 0.55}
             for i, b in enumerate(
                 [0.4 + k * 0.83 for k in range(int(seconds / 0.83) - 1)])]
    intervals = [{"t": beats[i]["t"], "ms": 830.0, "conf": 0.8}
                 for i in range(1, len(beats))]
    return {"t_s": t, "values": vals, "fps": fps, "beats": beats,
            "intervals": intervals, "segments": [[0.0, seconds]]}


def _accept_result():
    return ScanResult("fix-accept", ScanOutcome.ACCEPT,
                      predicted_class="SINUS", signal_quality_index=0.82,
                      usable_beats=34, analysed_seconds=27.0,
                      mean_pulse_rate_bpm=72.0, confidence_stars=4,
                      model_version="interim-rules-v0.1.2",
                      code_commit="deadbee", calibration_version="cal-x",
                      config_hash="h")


def _meta(sig=None, **over):
    m = {"session_id": "fix-accept", "datetime": "2026-08-25 21:00",
         "device": "yunet", "measured_fps": 30.0,
         "lighting": "~300 lux (proxy)", "scan_seconds": 30.0,
         "caveats": ["camera exposure/white-balance not locked (automatic): "
                     "consumer capture — not research-grade"],
         "report_mode": "full",       # these tests pin the v0.2.1 layout;
                                      # the v0.3 product default is
                                      # findings_only (test_findings_only)
         "show_tachogram": True,
         "signal": sig if sig is not None else _fixture_signal()}
    m.update(over)
    return m


def test_renderer_is_pure_stdlib():
    tree = ast.parse((_ROOT / "app" / "report_render.py").read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert not imports & {"numpy", "scipy", "cv2", "mediapipe"}, imports


def test_deterministic_snapshot():
    a = render_report(_accept_result(), _meta())
    b = render_report(_accept_result(), _meta())
    assert a == b
    assert a.startswith("<!doctype html>")
    assert f"<title>{REPORT_TITLE}</title>" in a


def test_layout_blocks_and_strip_rows():
    frag = render_report_fragment(_accept_result(), _meta())
    assert frag.count(REPORT_TITLE) == 1
    n_rows = int(math.ceil(30.0 / STRIP_SECONDS))
    assert frag.count("NOT AN ECG") == n_rows == 3
    assert REFERRAL_SENTENCE in frag
    assert "MEASUREMENTS" in frag and "FINDINGS" in frag
    assert "normalized a.u." in frag
    assert "BEAT-INTERVAL TREND" in frag
    assert "72 bpm" in frag                      # measurements populated
    assert "★★★★☆" in frag                       # confidence stars
    assert "@media print" in frag and "@page" in frag
    assert frag.count("<polyline") >= 3          # a trace in every strip
    # sanctioned sentence verbatim (R5)
    assert _accept_result().user_facing_text() in frag.replace("&#x27;", "'")


def test_tachogram_flag_off_removes_the_trend():
    frag = render_report_fragment(_accept_result(),
                                  _meta(show_tachogram=False))
    assert "BEAT-INTERVAL TREND" not in frag


def test_no_result_is_a_first_class_report():
    r = ScanResult("fix-dark", ScanOutcome.NO_RESULT,
                   no_read_reasons=["illuminance 69 lux below floor 100",
                                    "no face detected in the video"],
                   confidence_stars=1, confidence_limiting_factor="lighting",
                   model_version="interim-rules-v0.1")
    frag = render_report_fragment(r, _meta(sig={}, scan_seconds=None))
    assert frag.count(REPORT_TITLE) == 1
    assert "Why no result" in frag and "illuminance" in frag
    assert "insufficient quality" in frag        # R6 overlay
    assert frag.count("NOT AN ECG") >= 1         # strips still labeled
    assert REFERRAL_SENTENCE in frag
    assert r.user_facing_text() in frag.replace("&#x27;", "'")


def test_dynamic_strings_are_escaped():
    r = _accept_result()
    r.recording_id = '<script>alert(1)</script>'
    frag = render_report_fragment(r, _meta(session_id=r.recording_id))
    assert "<script>alert" not in frag
    assert "&lt;script&gt;" in frag


def test_cli_report_export_on_fixtures(videos, tmp_path):
    """The CLI export goes through the PRODUCT config, which since v0.3
    defaults to findings_only: no waveform of any kind on the page."""
    for name, expect in (("sinus30", "SINUS"), ("dark30", None)):
        out = tmp_path / f"{name}.html"
        r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                            "process", videos[name][0],
                            "--report", str(out)],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        doc = json.loads(r.stdout)
        assert doc["predicted_class"] == expect
        htmlpage = out.read_text()
        assert htmlpage.startswith("<!doctype html>")
        assert REPORT_TITLE in htmlpage
        assert "<polyline" not in htmlpage and "<svg" not in htmlpage
        assert "MEASUREMENTS" in htmlpage and "FINDINGS" in htmlpage
        assert REFERRAL_SENTENCE.split(". ")[0] in htmlpage
        if expect is None:
            assert "Why no result" in htmlpage
