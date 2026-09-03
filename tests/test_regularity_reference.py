"""v0.7 Task 3 — the ECG-derived reference label: computable, published,
applied by the SAME code to both paths."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from datasets.regularity_reference import (classify_index,
                                           load_reference_definition,
                                           reference_from_rpeaks,
                                           runs_from_rpeaks)
from scripts.make_synth_regularity import (COHORTS, bigeminy_rr,
                                           regular_rr, rsa_rr)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_definition_is_read_from_gates_yaml_and_is_explicit():
    d = load_reference_definition()
    assert d["index"] == "irregularity_index"
    assert 0.0 < float(d["irregular_if_index_at_least"]) < 1.0
    assert int(d["min_intervals"]) >= 10
    assert float(d["ecg_beat_confidence"]) == 1.0
    # a definition missing a field is refused, never defaulted
    bad = _ROOT / "tests" / "fixtures"
    p = bad / "_gates_missing.yaml"
    p.write_text("regularity:\n  reference_label: {index: irregularity_index}\n")
    with pytest.raises(ValueError, match="must be explicit"):
        load_reference_definition(p)
    p.unlink()


def test_classify_is_the_one_rule():
    d = {"index": "irregularity_index", "irregular_if_index_at_least": 0.06,
         "min_intervals": 15, "ecg_beat_confidence": 1.0}
    assert classify_index(0.02, 40, d) == "regular"
    assert classify_index(0.06, 40, d) == "irregular"
    assert classify_index(0.30, 40, d) == "irregular"
    assert classify_index(0.30, 10, d) == "indeterminate"
    assert classify_index(None, 40, d) == "indeterminate"
    assert classify_index(float("nan"), 40, d) == "indeterminate"


def test_reference_runs_apply_the_pipeline_run_discipline():
    """ECG beats are trusted beat by beat, but the physiologic-range and
    missed-beat splitters apply exactly as they do to camera beats."""
    t = np.cumsum(regular_rr(60.0, rmssd_ms=8.0, seed=1))
    rs = runs_from_rpeaks(t)
    assert len(rs.runs) == 1 and rs.n_intervals == t.size - 1
    # a missed R peak (a 2x interval) splits the run rather than being
    # bridged or interpolated — no repair
    t2 = np.delete(t, 30)
    rs2 = runs_from_rpeaks(t2)
    assert len(rs2.runs) == 2
    # exactly the 2x interval is dropped: nothing is bridged, nothing
    # else is lost
    assert sum(r.size for r in rs2.runs) == t2.size - 2


def test_cohorts_get_the_published_verdict():
    d = load_reference_definition()
    for name, c in COHORTS.items():
        for seed in (3, 9):
            rr = c["build"](60.0, seed, 15.0)
            ref = reference_from_rpeaks(np.cumsum(rr), definition=d)
            assert ref["class"] == c["expect"], (name, seed, ref["index"])
            assert ref["ci95"] is not None
            lo, hi = ref["ci95"]
            assert lo <= ref["index"] <= hi


def test_reference_and_camera_share_one_definition_object():
    """Same code, same threshold: the label the head applies to camera
    intervals IS the label the reference applies to ECG intervals."""
    from heads.head_regularity import _definition
    assert _definition({}) == load_reference_definition()
    # deep RSA crosses the threshold by the published rule on BOTH sides
    rr = rsa_rr(60.0, depth=0.16, seed=2)
    ref = reference_from_rpeaks(np.cumsum(rr))
    assert ref["class"] == "irregular"
    assert ref["index"] > float(ref["definition"]["irregular_if_index_at_least"])


def test_indeterminate_is_a_result_not_an_error():
    ref = reference_from_rpeaks(np.arange(0, 8.0, 0.8))     # 9 intervals
    assert ref["class"] == "indeterminate"
    assert ref["n_intervals"] < 15
    ref0 = reference_from_rpeaks([])
    assert ref0["class"] == "indeterminate" and ref0["index"] is None
    assert reference_from_rpeaks(np.array([1.0, 2.0]))["class"] == \
        "indeterminate"


def test_index_ci_coverage_is_measured_and_pinned():
    """The CI is labelled by its MEASURED coverage: on iid-jittered
    regular series the percentile bootstrap of a median under-covers at
    small n (0.88 at n = 15 for a nominal 0.95). Pinned so the label
    cannot drift from the behaviour."""
    from features.regularity import (_index_ci, INDEX_CI_COVERAGE_MEASURED,
                                     regularity_from_runs)
    rng = np.random.default_rng(1)
    big = np.clip(0.85 + rng.normal(0, 0.008 / np.sqrt(2), 200000),
                  0.3, 2.0) * 1000
    pop = float(np.median(np.abs(np.diff(big))) / np.median(big))
    # floors sit below the measured values by a sampling margin (200
    # trials): they guard against DRIFT, not the third decimal
    for n, floor in ((15, 0.80), (70, 0.85)):
        hits, T = 0, 200
        for t in range(T):
            rr = np.clip(0.85 + rng.normal(0, 0.008 / np.sqrt(2), n + 1),
                         0.3, 2.0) * 1000
            lo, hi = _index_ci(np.diff(rr), float(np.median(rr)), seed=t)
            hits += (lo <= pop <= hi)
        cov = hits / T
        assert cov >= floor, (n, cov)
        assert cov <= 0.99, (n, cov)                 # not vacuously wide
    reg = regularity_from_runs([np.full(30, 850.0) + np.arange(30) % 3],
                               fps=30.0)
    assert reg.index["ci_method"].startswith("moving-block")
    assert reg.index["ci_coverage_measured"] == INDEX_CI_COVERAGE_MEASURED
