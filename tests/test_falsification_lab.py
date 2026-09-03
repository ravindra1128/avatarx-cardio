"""M4 — the falsification lab: quarantined (invariant 10), watermarked,
and structurally honest (identity baseline, interval-level error,
cross-rhythm challenge)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ast
import json

import numpy as np
import pytest

from evaluation.inferred_ecg import WATERMARK
from evaluation.inferred_ecg.decoder import (synth_pairs, split_subjects,
                                             train_decoder, decode)
from evaluation.inferred_ecg.report import run_falsification

_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------- invariant 10
def _module_of(path: pathlib.Path) -> str:
    return str(path.relative_to(_ROOT)).replace("/", ".")[:-3]


def _repo_imports(py: pathlib.Path) -> set:
    tree = ast.parse(py.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_quarantine_no_import_path_from_app_or_pipeline():
    """Walk the repo-internal import graph from app/ and
    inference/pipeline.py transitively; evaluation.inferred_ecg must be
    unreachable."""
    top = {}
    for py in _ROOT.rglob("*.py"):
        rel = py.relative_to(_ROOT)
        if rel.parts[0] in ("tests", "path", "scratch_unused") or \
                "falsification_runs" in rel.parts:
            continue
        top[_module_of(py)] = py

    def resolve(name):
        for cand in (name, name + ".__init__"):
            if cand in top:
                return cand
        parts = name.split(".")
        while parts:
            cand = ".".join(parts)
            if cand in top:
                return cand
            parts = parts[:-1]
        return None

    seeds = [m for m in top if m.startswith("app.")
             or m == "inference.pipeline"]
    seen = set()
    frontier = list(seeds)
    while frontier:
        mod = frontier.pop()
        if mod in seen:
            continue
        seen.add(mod)
        for imp in _repo_imports(top[mod]):
            r = resolve(imp)
            if r and r not in seen:
                frontier.append(r)
    banned = [m for m in seen if m.startswith("evaluation.inferred_ecg")]
    assert not banned, f"quarantine breached via imports: {banned}"


# ---------------------------------------------------- decoder mechanics
def test_decoder_learns_something_on_train_subject():
    subs = synth_pairs(4, seed=5, duration_s=60.0)
    art = train_decoder(subs, seed=1, steps=200)
    sid, is_af, ppg, ecg = subs[0]
    gen = decode(art, ppg)
    r = float(np.corrcoef(gen[500:-500], ecg[500:-500])[0, 1])
    assert r > 0.2, r                       # learned more than noise
    assert art["decoder_version"].startswith("ppg2ecg")
    assert sorted(art["train_subjects"]) == sorted(s[0] for s in subs)


def test_split_is_participant_disjoint_and_stable():
    subs = synth_pairs(10, seed=2, duration_s=30.0)
    tr, te = split_subjects(subs, seed=3)
    assert tr and te
    assert not ({s[0] for s in tr} & {s[0] for s in te})
    tr2, te2 = split_subjects(subs, seed=3)
    assert [s[0] for s in tr2] == [s[0] for s in tr]


# ---------------------------------------------------- the report
@pytest.fixture(scope="module")
def falsified(tmp_path_factory):
    out = tmp_path_factory.mktemp("falsify")
    return run_falsification(source="synthetic", steps=250, seed=7,
                             out_root=out), out


def test_report_written_only_under_out_root_and_watermarked(falsified):
    res, out_root = falsified
    run_dir = pathlib.Path(res["out_dir"])
    assert run_dir.parent == out_root
    doc = json.loads((run_dir / "report.json").read_text())
    assert list(doc)[0] == "WATERMARK" and doc["WATERMARK"] == WATERMARK
    html = (run_dir / "report.html").read_text()
    assert html.count(WATERMARK) >= 2               # banner top and bottom
    svgs = list(run_dir.glob("*.svg"))
    assert svgs and all(WATERMARK in s.read_text() for s in svgs)
    assert doc["participant_disjoint"] is True


def test_three_tests_present_with_interval_metrics(falsified):
    res, _ = falsified
    doc = json.loads((pathlib.Path(res["out_dir"]) / "report.json")
                     .read_text())
    s = doc["summary"]
    for k in ("decoder_median_corr", "identity_median_corr",
              "decoder_pr_mae_ms", "identity_pr_mae_ms",
              "hallucinated_p_on_af", "reference_p_on_af"):
        assert k in s, k
    assert doc["per_subject"], "no held-out subjects evaluated"
    for r in doc["per_subject"]:
        assert r["decoder"]["n_matched_beats"] is not None
        assert "pr_interval_mae_ms" in r["decoder"]
        assert "pr_interval_mae_ms" in r["identity_baseline"]


def test_cross_rhythm_challenge_shows_hallucinated_p_waves(falsified):
    """The structural point of the lab: a decoder that never saw AF draws
    P waves into AF segments (its training prior), while the reference AF
    has essentially none. On the synthetic corpus this asymmetry must be
    visible."""
    res, _ = falsified
    s = res["summary"]
    assert s["hallucinated_p_on_af"] is not None
    assert s["reference_p_on_af"] is not None
    assert s["hallucinated_p_on_af"] > s["reference_p_on_af"], s
