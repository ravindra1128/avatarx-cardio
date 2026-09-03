"""v0.7 invariant G-a — ONE representation.

`head_afib`, `head_flutter` and `head_regularity` consume
RegularityFeatures from features/regularity.py. A second implementation
of an interval statistic anywhere in inference/ or heads/ is a bug: two
paths that can disagree will eventually disagree in production (lesson
P2). This file is the import/duplication audit, plus the contract of
the representation itself.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ast
import math

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# numpy calls that constitute an interval DISPERSION/DISTRIBUTION
# statistic when applied to beat intervals. Location statistics (mean,
# median of a rate) are not on this list: a head may still ask "what is
# the median rate", it may not re-derive variability.
_DISPERSION_CALLS = {"diff", "ediff1d", "gradient", "std", "nanstd",
                     "var", "nanvar", "corrcoef", "cov", "histogram",
                     "percentile", "nanpercentile", "quantile",
                     "nanquantile", "log", "log2", "fft", "rfft"}
# Detection-error EVIDENCE lives in decision_logic by design (v0.1.2):
# harmonic_fraction reads the RAW fused series to find half/double
# intervals, timing_precision matches per-ROI beat trains. Neither is a
# regularity statistic and neither may move — they are allowlisted BY
# NAME with a BUDGET (the exact number of each banned call the function
# is known to make), so a new call cannot hide inside an allowlisted
# function (review finding: the old allowlist exempted the whole of
# run_with_details, the one place the runs are in hand).
_ALLOWLIST = {}
_BUDGET_FILE = pathlib.Path(__file__).resolve().parent / "fixtures" / \
    "regularity_audit_budget.json"


def _load_budget():
    import json
    return {tuple(k.split("::")): v for k, v in
            json.loads(_BUDGET_FILE.read_text()).items()}


def test_the_allowlist_names_only_functions_that_exist():
    """A stale allowlist entry is a hole a new duplicate could hide in."""
    for (rel, func), _ in sorted(_load_budget().items()):
        tree = ast.parse((_ROOT / rel).read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert func in names, (rel, func)


def _numpy_aliases(tree):
    """Every local name bound to numpy or to one of its members:
    `import numpy as X`, `from numpy import diff as d`, `from numpy
    import fft` (review finding: the old audit only knew `np`)."""
    mods, members = {"np", "numpy", "_np"}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "numpy":
                    mods.add(a.asname or "numpy")
                elif a.name.startswith("numpy."):
                    mods.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom) and node.module and \
                node.module.split(".")[0] == "numpy":
            for a in node.names:
                members[a.asname or a.name] = a.name
    return mods, members


def _calls(tree):
    """Yield (INNERMOST enclosing function, statistic name) for every
    numpy dispersion call and for every hand-written successive
    difference (`x[1:] - x[:-1]`) in the module."""
    mods, members = _numpy_aliases(tree)
    stack = []

    def walk(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.append(node.name)
            for child in ast.iter_child_nodes(node):
                yield from walk(child)
            stack.pop()
            return
        here = stack[-1] if stack else "<module>"
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                base = fn.value
                if isinstance(base, ast.Name) and base.id in mods:
                    yield here, fn.attr
                elif isinstance(base, ast.Attribute) and \
                        isinstance(base.value, ast.Name) and \
                        base.value.id in mods:
                    yield here, base.attr            # np.fft.rfft
                    yield here, fn.attr
            elif isinstance(fn, ast.Name) and fn.id in members:
                yield here, members[fn.id].split(".")[-1]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) \
                and isinstance(node.left, ast.Subscript) \
                and isinstance(node.right, ast.Subscript) \
                and isinstance(node.left.slice, ast.Slice) \
                and isinstance(node.right.slice, ast.Slice):
            yield here, "slice_difference"
        for child in ast.iter_child_nodes(node):
            yield from walk(child)

    yield from walk(tree)


def _audit():
    """{(file, function): {statistic: count}} over heads/ + inference/."""
    found = {}
    for rel in sorted(list((_ROOT / "heads").glob("*.py"))
                      + list((_ROOT / "inference").glob("*.py"))):
        relpath = str(rel.relative_to(_ROOT))
        tree = ast.parse(rel.read_text())
        for func, attr in _calls(tree):
            if attr in _DISPERSION_CALLS or attr == "slice_difference":
                found.setdefault((relpath, func), {})
                found[(relpath, func)][attr] = \
                    found[(relpath, func)].get(attr, 0) + 1
    return found


def test_no_second_interval_statistics_implementation_in_heads_or_inference():
    """Every banned call in heads/ and inference/ is on the budget, at
    exactly the count the budget records — no more (a new statistic),
    no fewer (a stale budget is a hole)."""
    found = _audit()
    budget = _load_budget()
    problems = []
    for key, counts in sorted(found.items()):
        if key not in budget:
            problems.append(f"{key[0]}::{key[1]} uses {counts} — not "
                            "allowlisted")
        elif counts != budget[key]:
            problems.append(f"{key[0]}::{key[1]} uses {counts}, budget "
                            f"says {budget[key]}")
    for key in sorted(set(budget) - set(found)):
        problems.append(f"budget entry {key} matches nothing — stale")
    assert not problems, "\n".join(problems)


def test_every_interval_consuming_head_reads_the_representation():
    """The heads that need dispersion declare it and read it from
    context['regularity']; none of them touches lattice.runs for a
    statistic."""
    src = {n: (_ROOT / "heads" / f"head_{n}.py").read_text()
           for n in ("afib", "irregularity", "rate_flags", "flutter",
                     "regularity")}
    for n, s in src.items():
        # the substring "regularity" is not evidence (review finding:
        # head_flutter matched on its extractor's sub-dict); the head
        # must read the object the pipeline published
        assert 'get("regularity")' in s or "['regularity']" in s or \
            '["regularity"]' in s, n
    # the decision head derives its legacy view FROM the representation
    assert "as_rhythm_features()" in src["afib"]
    # the two rhythm stubs build nothing of their own
    for n in ("irregularity", "rate_flags"):
        assert "np.diff" not in src[n] and "np.std" not in src[n], n
        assert "regularity_from_runs" in src[n], n
    # the flutter extractor imports, never redefines, the machinery
    fl = (_ROOT / "features" / "flutter.py").read_text()
    for name in ("def _clean", "def _run_diffs", "def tachogram",
                 "def respiratory_coupling", "def _phase_locking"):
        assert name not in fl, name
    assert "from features.regularity import" in fl


def test_the_v1_raw_series_path_is_gone():
    """features/rhythm.py's raw-series compute_rhythm_features had no
    callers and different (weighted, ddof=0) formulas — exactly the
    second implementation G-a bans."""
    import features.rhythm as rhythm
    assert not hasattr(rhythm, "compute_rhythm_features")
    assert hasattr(rhythm, "compute_rhythm_features_from_runs")
    # the estimators are re-exported, not re-implemented
    import features.regularity as reg
    for name in ("sample_entropy", "shannon_entropy", "spectral_entropy",
                 "turning_point_ratio", "markov_surprise"):
        assert getattr(rhythm, name) is getattr(reg, name), name


def test_representation_is_versioned_and_the_legacy_view_is_the_same_dict():
    from features.regularity import (REGULARITY_FEATURES_VERSION,
                                     regularity_from_runs)
    rng = np.random.default_rng(0)
    runs = [850.0 + rng.normal(0, 25, 40), 860.0 + rng.normal(0, 25, 30)]
    reg = regularity_from_runs(runs, [np.ones(40), np.ones(30)],
                               run_times=[np.cumsum(runs[0]) / 1000.0,
                                          40.0 + np.cumsum(runs[1]) / 1000.0],
                               fps=30.0)
    assert reg.version == REGULARITY_FEATURES_VERSION == \
        "regularity-features-v1"
    view = reg.as_rhythm_features()
    assert view.values is reg.values                 # the SAME object
    assert view.n_intervals == reg.n_intervals == 70
    # families are additive and complete
    assert set(reg.dispersion) >= {"rmssd_ms", "sdnn_ms", "cv",
                                   "median_abs_succ_diff_ms", "pnn50",
                                   "pnn80", "rel_mad", "irregularity_index"}
    assert set(reg.distribution) >= {"shannon_entropy", "sample_entropy",
                                     "poincare_sd1", "poincare_sd2",
                                     "poincare_ratio", "spectral_entropy"}
    assert set(reg.structure) == {"coupling", "periodicity", "outliers"}
    assert set(reg.confidence) >= {"n_runs", "run_lengths", "longest_run",
                                   "beat_confidence", "jitter"}
    assert reg.index["name"] == "irregularity_index"
    assert reg.index["value"] == pytest.approx(
        reg.values["irregularity_index"], abs=1e-5)
    lo, hi = reg.index["ci95"]
    assert lo <= reg.index["value"] <= hi
    # the legacy key set is EXACTLY the pre-v0.7 vector
    assert set(reg.values) == {
        "n_runs", "n_intervals", "longest_run", "dropout_rate", "mean_sqi",
        "mean_ibi", "median_ibi", "pnn50", "pnn20", "median_abs_succ_diff",
        "irregularity_index", "sdnn", "cv_ibi", "rmssd", "sdsd",
        "poincare_sd1", "poincare_sd2", "poincare_ratio", "sample_entropy",
        "shannon_entropy", "spectral_entropy", "turning_point_ratio",
        "markov_surprise", "autocorr_lag1"}
    # JSON-safe serialisation, NaN honestly null
    d = reg.to_dict()
    assert d["version"] == reg.version
    assert d["values"]["sample_entropy"] is None       # 40 < 60 floor


def test_successive_difference_statistics_never_cross_a_run_break():
    from features.regularity import regularity_from_runs
    a = 400.0 + np.zeros(30)
    b = 800.0 + np.zeros(30)
    one = regularity_from_runs([np.concatenate([a, b])])
    two = regularity_from_runs([a, b])
    # one run: the seam is a 400 ms jump and inflates every diff stat
    assert one.values["rmssd"] > 50.0
    assert one.dispersion["pnn80"] > 0.0
    # two runs: nothing crosses the break
    assert two.values["rmssd"] == 0.0
    assert two.dispersion["pnn80"] == 0.0
    # pooled distribution statistics DO see both runs — that is what
    # they are for — and the representation says which is which
    assert two.values["sdnn"] > 100.0
    assert two.dispersion["n_within_run_diffs"] == 58


def test_fewer_than_five_intervals_yields_the_structural_keys_only():
    from features.regularity import regularity_from_runs
    reg = regularity_from_runs([np.array([800.0, 810.0, 790.0])])
    assert reg.n_intervals == 3
    assert set(reg.values) == {"n_runs", "n_intervals", "longest_run",
                               "dropout_rate", "mean_sqi"}
    assert reg.index["value"] is None and reg.index["ci95"] is None
    assert any("fewer than 5" in w for w in reg.estimator_warnings)
    assert reg.structure["coupling"]["available"] is False


# ------------------------------------------- standing invariant (v0.1 #6)
def test_sqi_never_scores_periodicity_and_cannot_reach_the_substrate():
    """SQI must never score periodicity: the anti-periodicity test in
    tests/test_signal_quality.py is the behavioural guard; this is the
    structural one — the quality index has no path to the interval
    representation, so it cannot grow one quietly."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "rppg" /
           "signal_quality.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for n in names:
            assert not n.startswith(("features.regularity",
                                     "features.rhythm", "features.flutter",
                                     "heads")), n
    # identifiers actually USED (calls, attributes, names) — docstrings
    # may describe the rule; code may not implement the statistic
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    for banned in ("regularity_from_runs", "compute_rhythm_features",
                   "irregularity_index", "rmssd", "autocorr_lag1",
                   "corrcoef", "phase_locking", "short_run_periodicity"):
        assert banned not in used, banned
    # and the behavioural guard is still present, by name
    guard = (pathlib.Path(__file__).resolve().parent /
             "test_signal_quality.py").read_text()
    assert "test_clean_af_scores_at_least_clean_sinus_minus_005" in guard
