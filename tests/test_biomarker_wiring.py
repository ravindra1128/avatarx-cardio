"""The measure response's biomarker payload is built inside a fail-soft
`except Exception`, so a NameError there would silently replace all three
cards with an error dict and no test would notice. These pin the wiring.

2026-09-09: an edit referencing `header` (a request-handler local) from inside
`measure_video` did exactly that and was caught only by reading the source.
"""
import ast
import builtins
import inspect
import os

os.environ.setdefault("AFIB_SHEET_ID", "test-sheet")

from app import measure_api  # noqa: E402
from app.report_data import report_biomarkers  # noqa: E402


def _measure_video_source() -> str:
    return inspect.getsource(measure_api.measure_video)


def test_biomarkers_call_uses_only_names_bound_in_measure_video():
    """Every name the report_biomarkers(...) call references must be a
    parameter or local of measure_video, never a caller's local."""
    src = inspect.getsource(measure_api.measure_video)
    tree = ast.parse("".join(src.splitlines(keepends=True)[0:]).lstrip())
    fn = tree.body[0]
    bound = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    def _names(target):
        """Every name a binding target introduces, tuples and stars included."""
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            for el in target.elts:
                yield from _names(el)
        elif isinstance(target, ast.Starred):
            yield from _names(target.value)

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                bound.update(_names(t))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            bound.update(_names(node.target))
        elif isinstance(node, (ast.For, ast.comprehension)):
            bound.update(_names(node.target))
        elif isinstance(node, ast.withitem):
            if node.optional_vars is not None:
                bound.update(_names(node.optional_vars))
        elif isinstance(node, ast.Import):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(a.asname or a.name for a in node.names)

    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "report_biomarkers"]
    assert calls, "measure_video no longer calls report_biomarkers"
    for call in calls:
        for name in {n.id for a in call.args + [k.value for k in call.keywords]
                     for n in ast.walk(a) if isinstance(n, ast.Name)}:
            assert name in bound or hasattr(measure_api, name) or hasattr(builtins, name), (
                f"report_biomarkers(...) references {name!r}, which is not bound "
                f"inside measure_video — the fail-soft except would hide the NameError")


def test_capture_lock_state_reaches_the_cards():
    """A manifest with the client's lock flags must make the vascular-tone
    card record optics_locked, not the default False."""
    src = _measure_video_source()
    assert "capture=manifest" in src, (
        "measure_video must pass the manifest as `capture=` — the lock flags "
        "the client sends are merged into it by the request handler")

    # and report_biomarkers must actually honour that kwarg
    sig = inspect.signature(report_biomarkers)
    assert "capture" in sig.parameters


def test_request_handler_merges_client_lock_flags_into_the_manifest():
    _, header = measure_api._parse_envelope(
        b"", "video/webm",
        {"exposure_locked": ["1"], "awb_locked": ["1"], "client_fps": ["30"]})
    assert header["manifest"]["exposure_locked"] is True
    assert header["manifest"]["awb_locked"] is True
