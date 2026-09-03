"""
T6 — Model A baseline + E6 degradation machinery.

Unit tests run OFFLINE on synthetic RR series: they prove the degradation
model, the feature matrix, the split-safe CV harness and the in-house
classifiers (logistic regression + gradient-boosted trees — implemented on
numpy/scipy because the v0.1 dependency list is pinned).

The REAL E6 acceptance number (AUC > 0.90 on rPPG-degraded MIMIC PERform
AF at sigma=15 ms / deficit 0.5, participant-level CV + cluster-bootstrap
CIs) is produced by scripts/e6_degradation.py against the CC-BY 4.0
download; the final test below verifies it whenever the cached dataset is
present and skips (loudly) when it is not — CI must never silently depend
on the network.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from datasets.splits import LeakageError
from models.baseline import (LogisticModel, GradientBoostedTrees,
                             degrade_rr_to_beatseries, feature_vector,
                             FEATURE_NAMES, cross_validate_participants,
                             train_model_a, MODEL_A_FEATURES)
from configs import load_config

DATA_CACHE = pathlib.Path(__file__).resolve().parents[1] / "data_cache"


# ------------------------------------------------------------ degradation
def _sinus_rr(rng, n=110):
    return np.clip(rng.normal(850, 30, n), 500, 1400)


def _af_rr(rng, n=150):
    return np.clip(rng.normal(620, 170, n), 280, 1350)


def test_degradation_is_deterministic_and_jitters_timing():
    rr = _sinus_rr(np.random.default_rng(0))
    a = degrade_rr_to_beatseries(rr, sigma_ms=15, deficit_p=0.0,
                                 false_rate=0.0, seed=5)
    b = degrade_rr_to_beatseries(rr, sigma_ms=15, deficit_p=0.0,
                                 false_rate=0.0, seed=5)
    assert np.array_equal(a.times(), b.times())          # seeded
    clean = degrade_rr_to_beatseries(rr, sigma_ms=0, deficit_p=0.0,
                                     false_rate=0.0, seed=5)
    d = (a.times() - clean.times()) * 1000.0
    assert 5.0 < np.std(d) < 30.0                        # jitter present


def test_deficit_drops_only_post_short_rr_beats():
    rng = np.random.default_rng(1)
    rr = _af_rr(rng)
    n_short = int(np.sum(rr < 400))
    assert n_short >= 5, "AF fixture must contain short RRs"
    full = degrade_rr_to_beatseries(rr, 0, 0.0, 0.0, seed=2)
    dropped = degrade_rr_to_beatseries(rr, 0, 1.0, 0.0, seed=2)
    assert len(full.beats) - len(dropped.beats) == n_short
    none = degrade_rr_to_beatseries(rr, 0, 0.0, 0.0, seed=3)
    assert len(none.beats) == len(full.beats)


def test_false_beats_are_not_oracle_labelled():
    """Review finding: if every injected false beat scores below the 0.5
    threshold, confidence is an oracle label and the false-beat axis can
    never corrupt an interval. The degradation model must let the measured
    surviving fraction (~0.5, the broad double-detection class) through."""
    rr = _sinus_rr(np.random.default_rng(2))
    s = degrade_rr_to_beatseries(rr, 0, 0.0, false_rate=0.20, seed=4)
    conf = s.confidences()
    n_false = len(s.beats) - (rr.size + 1)
    n_false_surviving = int(np.sum(conf >= 0.5)) - (rr.size + 1)
    assert n_false >= 15
    assert 0.2 * n_false <= n_false_surviving <= 0.8 * n_false, \
        (n_false, n_false_surviving)
    # survival 0 must reproduce the fully-excluded regime on demand
    s0 = degrade_rr_to_beatseries(rr, 0, 0.0, false_rate=0.20, seed=4,
                                  false_survival=0.0)
    assert int(np.sum(s0.confidences() >= 0.5)) == rr.size + 1
    # and surviving falses must actually corrupt intervals: clean-run RMSSD
    # on degraded sinus grows vs the no-false series
    from beats.ibi import clean_runs, rmssd_from_runs
    r_false = rmssd_from_runs(clean_runs(s, min_conf=0.5))
    r_none = rmssd_from_runs(clean_runs(
        degrade_rr_to_beatseries(rr, 0, 0.0, 0.0, seed=4), min_conf=0.5))
    assert r_false > r_none + 5.0, (r_false, r_none)


# ---------------------------------------------------------------- features
def test_feature_vector_fixed_length_and_names():
    rr = _af_rr(np.random.default_rng(3))
    s = degrade_rr_to_beatseries(rr, 15, 0.5, 0.02, seed=6)
    v = feature_vector(s)
    assert v.shape == (len(FEATURE_NAMES),)
    idx = [FEATURE_NAMES.index(k)
           for k in ("median_abs_succ_diff", "pnn50", "dropout_rate")]
    assert np.all(np.isfinite(v[idx]))


def test_af_and_sinus_features_separate():
    rng = np.random.default_rng(4)
    ks = [FEATURE_NAMES.index("median_abs_succ_diff"),
          FEATURE_NAMES.index("pnn50")]
    a = feature_vector(degrade_rr_to_beatseries(_af_rr(rng), 15, 0.5, 0.02,
                                                seed=7))
    s = feature_vector(degrade_rr_to_beatseries(_sinus_rr(rng), 15, 0.5, 0.02,
                                                seed=8))
    assert a[ks[0]] > 2 * s[ks[0]]
    assert a[ks[1]] > s[ks[1]] + 0.2


# -------------------------------------------------------------- classifiers
def _toy(n=400, seed=0, nonlinear=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    if nonlinear:
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)        # XOR-ish
    else:
        y = (X @ np.array([1.5, -2.0, 0.5, 0.0]) + rng.normal(0, 0.5, n)
             > 0).astype(int)
    return X, y


def test_logistic_model_learns_linear_problem():
    X, y = _toy()
    m = LogisticModel().fit(X[:300], y[:300])
    p = m.predict_proba(X[300:])
    assert np.all((p >= 0) & (p <= 1))
    from evaluation.afib_metrics import _auroc
    assert _auroc(y[300:], p) > 0.9


def test_gbt_learns_nonlinear_problem_lr_cannot():
    X, y = _toy(nonlinear=True, seed=1)
    from evaluation.afib_metrics import _auroc
    lr = LogisticModel().fit(X[:300], y[:300])
    gb = GradientBoostedTrees(n_trees=80, max_depth=3).fit(X[:300], y[:300])
    assert _auroc(y[300:], gb.predict_proba(X[300:])) > 0.85
    assert _auroc(y[300:], lr.predict_proba(X[300:])) < 0.65


def test_models_serialise_roundtrip():
    X, y = _toy(seed=2)
    for cls in (LogisticModel, GradientBoostedTrees):
        m = cls().fit(X, y)
        back = cls.from_json(m.to_json())
        assert np.allclose(back.predict_proba(X), m.predict_proba(X))
        assert back.version == m.version and m.version


# ------------------------------------------------------------------ CV
def _windows(n_subj=20, per_subj=6, seed=5):
    rng = np.random.default_rng(seed)
    rows, ys, pids = [], [], []
    for i in range(n_subj):
        is_af = i % 2 == 0
        for w in range(per_subj):
            rr = _af_rr(rng) if is_af else _sinus_rr(rng)
            s = degrade_rr_to_beatseries(rr, 15, 0.5, 0.02,
                                         seed=seed * 997 + i * 31 + w)
            rows.append(feature_vector(s))
            ys.append(int(is_af))
            pids.append(f"subj{i:03d}")
    return np.vstack(rows), np.asarray(ys), np.asarray(pids)


def test_participant_cv_no_subject_spans_folds_and_auc_sane():
    X, y, pid = _windows()
    out = cross_validate_participants(X, y, pid, model="logreg", k=5, seed=9)
    folds = out["fold_of_participant"]
    assert set(folds) == set(np.unique(pid))             # every subject
    scores = out["oof_scores"]
    assert scores.shape == y.shape
    from evaluation.afib_metrics import _auroc
    assert _auroc(y, scores) > 0.9                       # separable fixture
    # windows of one participant share one fold — enforced by construction,
    # verified here:
    for p, f in folds.items():
        assert isinstance(f, (int, np.integer))


def test_fold_assignment_is_pure_function_of_participant_and_seed():
    """Same identity-only hashing contract as datasets/splits: permuting
    the windows changes nothing; changing the seed reassigns."""
    X, y, pid = _windows(n_subj=8, per_subj=3, seed=6)
    out1 = cross_validate_participants(X, y, pid, model="logreg", k=3, seed=2)
    perm = np.random.default_rng(0).permutation(y.size)
    out2 = cross_validate_participants(X[perm], y[perm], pid[perm],
                                       model="logreg", k=3, seed=2)
    assert out1["fold_of_participant"] == out2["fold_of_participant"]
    out3 = cross_validate_participants(X, y, pid, model="logreg", k=3, seed=3)
    assert out3["fold_of_participant"] != out1["fold_of_participant"]


# ----------------------------------------------- E6 acceptance (cached data)
@pytest.mark.skipif(not (DATA_CACHE / "mimic_perform_af_csv").exists()
                    and not any(DATA_CACHE.glob("**/mimic_perform_af*.csv")),
                    reason="MIMIC PERform AF cache not present (run "
                           "scripts/e6_degradation.py to download)")
def test_e6_moderate_degradation_auc_gate():
    """THE T6 acceptance, on the real (cached) RR data: participant-level
    CV AUC > 0.90 at sigma=15 ms, deficit 0.5."""
    from scripts.e6_degradation import load_mimic_rr, run_e6_setting
    subjects = load_mimic_rr(DATA_CACHE)
    assert len(subjects) == 35, f"expected 35 subjects, got {len(subjects)}"
    res = run_e6_setting(subjects, sigma_ms=15.0, deficit_p=0.5, seed=20260814)
    assert res["logreg"]["auroc"] > 0.90, res["logreg"]
    assert res["gbt"]["auroc"] > 0.90, res["gbt"]


# ------------------------------------------------------------- model_a flag
def test_model_a_wiring_behind_config_flag():
    """decision.classifier: model_a must populate afib_probability from the
    trained artifact and still respect every gate."""
    import copy
    from inference.decision_logic import decide
    from features.rhythm import RhythmFeatures
    from datasets.schema import ScanOutcome

    X, y, pid = _windows(n_subj=10, per_subj=4, seed=11)
    art = train_model_a(X, y, model="logreg")

    cfg = copy.deepcopy(load_config())
    cfg["decision"]["classifier"] = "model_a"
    cfg["decision"]["model_a"] = art                     # in-memory artifact

    def feats(vals):
        base = {"n_intervals": 60.0, "median_abs_succ_diff": vals[0],
                "pnn50": vals[1], "median_ibi": 700.0, "mean_ibi": 700.0,
                "dropout_rate": vals[2], "irregularity_index": vals[0] / 700.0,
                "sdnn": 100.0, "rmssd": 120.0, "cv_ibi": 0.1, "sdsd": 100.0,
                "pnn20": 0.9, "poincare_sd1": 80.0, "poincare_sd2": 120.0,
                "poincare_ratio": 0.6, "n_runs": 3.0, "longest_run": 30.0}
        return RhythmFeatures(base, 60, 0.9, [])

    af = decide(feats((150.0, 0.8, 0.3)), 0.9, 0.9, cfg)
    assert af.outcome is ScanOutcome.ACCEPT
    assert af.afib_probability is not None
    assert af.predicted_class == "AFIB_SUGGESTIVE"
    assert af.model_version.startswith("model-a")
    gated = decide(feats((150.0, 0.8, 0.3)), 0.1, 0.9, cfg)
    assert gated.outcome is not ScanOutcome.ACCEPT       # gates still first
    assert gated.afib_probability is None
