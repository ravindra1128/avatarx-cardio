"""v0.3 T5 — the consumer surface: findings-only by default (zero
waveform elements of any kind), zero synthetic-ECG elements in ANY mode
while gates are red, and no import path from app/ or inference/ into the
quarantined research/ package."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ast

from app.report_data import report_capture_meta
from app.report_render import (REFERRAL_SENTENCE, REPORT_TITLE, STRIP_LABEL,
                               render_report, render_report_fragment)
from configs import load_config
from datasets.schema import ScanResult, ScanOutcome
from tests.test_report_render import _accept_result, _fixture_signal, _meta

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# the v0.3 product surface: meta WITHOUT a report_mode key must fail
# closed to findings-only, and the config default must say so explicitly


def _findings_meta(**over):
    m = _meta(**over)
    m.pop("report_mode")
    return m


def test_default_config_is_findings_only():
    assert load_config()["report"]["mode"] == "findings_only"
    r = _accept_result()
    cm = report_capture_meta(r, {}, load_config())
    assert cm["report_mode"] == "findings_only"


def test_findings_only_has_zero_waveform_elements():
    frag = render_report_fragment(_accept_result(), _findings_meta())
    for tok in ("<svg", "<polyline", "<canvas", "<path",
                STRIP_LABEL, "BEAT-INTERVAL TREND", "NOT AN ECG",
                "normalized a.u."):
        assert tok not in frag, tok
    # ...while the findings surface itself is complete: sentence, stars,
    # pulse rate, caveats, referral
    assert REPORT_TITLE in frag
    assert "MEASUREMENTS" in frag and "FINDINGS" in frag
    assert "72 bpm" in frag and "★★★★☆" in frag
    assert "consumer capture" in frag              # caveat rendered
    assert REFERRAL_SENTENCE in frag
    r = _accept_result()
    assert r.user_facing_text() in frag.replace("&#x27;", "'")


def test_findings_only_ecg_accounting():
    """R4 under findings-only: 'ECG' = the referral sentence's one
    occurrence + whatever the sanctioned sentence itself requires."""
    for cls in ("SINUS", "AFIB_SUGGESTIVE"):
        r = _accept_result()
        r.predicted_class = cls
        frag = render_report_fragment(r, _findings_meta())
        expected = 1 + r.user_facing_text().count("ECG")
        assert frag.count("ECG") == expected, (cls, frag.count("ECG"))


def test_findings_only_no_result_keeps_reasons():
    r = ScanResult("nr", ScanOutcome.NO_RESULT,
                   no_read_reasons=["only 3 clean intervals — too few"],
                   confidence_stars=1, confidence_limiting_factor="sqi")
    frag = render_report_fragment(r, _findings_meta(sig={}))
    assert "Why no result" in frag and "clean intervals" in frag
    assert "<svg" not in frag
    assert REFERRAL_SENTENCE in frag


def test_zero_synthetic_ecg_elements_in_any_mode():
    from research.ecg_reconstruction import WATERMARK
    for meta in (_findings_meta(), _meta()):           # default AND full
        doc = render_report(_accept_result(), meta)
        assert WATERMARK not in doc
        assert "SYNTHETIC" not in doc and "synthetic ECG" not in doc
        assert "reconstruct" not in doc.lower()
    page = (_ROOT / "app" / "static" / "index.html").read_text()
    assert "SYNTHETIC" not in page and "reconstruct" not in page.lower()


def test_quarantine_no_path_from_app_or_inference_into_research():
    """Transitive import audit, stricter than the v0.2 walker: EVERY
    module under app/ and inference/ (not just the pipeline) must be
    unable to reach research/ (or the v0.2 lab)."""
    top = {}
    for py in _ROOT.rglob("*.py"):
        rel = py.relative_to(_ROOT)
        if rel.parts[0] in ("tests", "scratch_unused") or \
                "falsification_runs" in rel.parts or "runs" in rel.parts:
            continue
        top[str(rel).replace("/", ".")[:-3]] = py

    def imports(py):
        out = set()
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                out |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                out.add(node.module)
            elif isinstance(node, ast.Call):
                # dynamic imports with a literal module name count too —
                # __import__("research.x") / import_module("research.x")
                # must not slip past the walk (v0.4-vascular review)
                fn = node.func
                name = getattr(fn, "id", getattr(fn, "attr", ""))
                if name in ("__import__", "import_module") and \
                        node.args and \
                        isinstance(node.args[0], ast.Constant) and \
                        isinstance(node.args[0].value, str):
                    out.add(node.args[0].value)
        return out

    def resolve(name):
        parts = name.split(".")
        while parts:
            cand = ".".join(parts)
            if cand in top:
                return cand
            if cand + ".__init__" in top:
                return cand + ".__init__"
            parts = parts[:-1]
        return None

    seeds = [m for m in top
             if m.startswith("app.") or m.startswith("inference.")]
    assert seeds
    seen, frontier = set(), list(seeds)
    while frontier:
        mod = frontier.pop()
        if mod in seen:
            continue
        seen.add(mod)
        for imp in imports(top[mod]):
            r = resolve(imp)
            if r and r not in seen:
                frontier.append(r)
    banned = [m for m in seen if m.startswith("research")
              or m.startswith("evaluation.inferred_ecg")]
    assert not banned, f"quarantine breached via imports: {banned}"
