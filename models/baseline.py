"""
Model A (T6): interval-feature classifiers on the PRODUCTION feature path.

E6 is the experiment this module exists for: take REAL RR series (MIMIC
PERform AF, CC-BY 4.0), degrade them to measured rPPG conditions — timing
jitter, pulse-deficit dropouts, false-beat insertions — push them through
`clean_runs` -> `compute_rhythm_features_from_runs` (the SAME code the
product runs, lesson P2), and ask whether AF still separates. The
degradation budget IS the result; tuning on test folds would destroy it.

The classifiers are implemented in-house on numpy/scipy because the v0.1
dependency list is pinned (no scikit-learn/xgboost). They are small,
inspected, and unit-tested against problems with known answers.

INVARIANTS HONOURED HERE:
  * features come from clean runs only — never a raw beat train;
  * no demographic feature can enter: FEATURE_NAMES is the closed list;
  * cross-validation is PARTICIPANT-level, fold assignment a pure function
    of (participant_id, seed) via the same hashing as datasets/splits;
  * every fitted transform (imputation medians, standardisation) is fitted
    inside model.fit, i.e. on the training fold only, and the fit set is
    additionally screened by assert_preprocessing_is_split_safe.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional, Sequence

import numpy as np

from beats.detector import Beat, BeatSeries
from beats.ibi import clean_runs
from features.rhythm import compute_rhythm_features_from_runs
from datasets.splits import (_stable_unit_interval, LeakageError,
                             assert_preprocessing_is_split_safe,
                             windows_are_participant_grouped)
from datasets.schema import Recording, CaptureConfig, SyncRecord, SyncMethod, \
    Split

# Closed feature list — timing/interval features plus run structure. No
# demographics, no identifiers, nothing recording-level.
FEATURE_NAMES = (
    "n_intervals", "n_runs", "longest_run", "dropout_rate",
    "mean_ibi", "median_ibi", "sdnn", "cv_ibi",
    "pnn50", "pnn20", "median_abs_succ_diff", "irregularity_index",
    "rmssd", "sdsd", "poincare_sd1", "poincare_sd2", "poincare_ratio",
    "autocorr_lag1", "turning_point_ratio", "markov_surprise",
    "spectral_entropy",
)

MODEL_A_FEATURES = FEATURE_NAMES
SHORT_RR_DEFICIT_MS = 400.0


# ---------------------------------------------------------------- E6 model
# Fraction of inserted false beats whose calibrated confidence SURVIVES the
# clean_runs threshold. Measured basis (T2, on the record in the spec
# changelog): narrow-transient false beats are >= 80% excluded, but the
# broad 2-ROI double-detection class is only ~33-59% excluded — i.e. ~half
# of that class enters the run set and corrupts intervals, and no v0.1
# stage removes it (the short-pair splitter is out of scope). A degradation
# model whose false beats are excluded with CERTAINTY would make the
# false-beat axis a no-op and inflate the E6 result (review finding).
FALSE_BEAT_SURVIVAL = 0.5


def degrade_rr_to_beatseries(rr_ms: np.ndarray, sigma_ms: float,
                             deficit_p: float, false_rate: float,
                             seed: int, fps: float = 60.0,
                             false_survival: float = FALSE_BEAT_SURVIVAL
                             ) -> BeatSeries:
    """Real RR series -> rPPG-degraded BeatSeries.

    Degradations, applied in the physiologically meaningful order:
      1. pulse deficit — the beat ENDING an RR < 400 ms is dropped with
         probability `deficit_p` (insufficient diastolic filling: the beat
         happened on ECG but ejected too little volume to be seen);
      2. timing jitter — N(0, sigma_ms) on every surviving beat time;
      3. false insertions — `false_rate` x n extra beats at uniform times.
         A `false_survival` fraction carries confidence ABOVE the 0.5
         threshold (the measured broad-double-detection class that the
         confidence channel does NOT remove) and therefore lands inside
         clean runs, splitting genuine intervals; the rest score low and
         only fragment runs.
    """
    rng = np.random.default_rng(seed)
    rr = np.asarray(rr_ms, float)
    t = np.concatenate([[0.0], np.cumsum(rr)]) / 1000.0   # n+1 beat times

    keep = np.ones(t.size, bool)
    short = np.flatnonzero(rr < SHORT_RR_DEFICIT_MS) + 1  # beat ending rr[i]
    if short.size:
        keep[short] = rng.random(short.size) >= deficit_p

    jitter = rng.normal(0.0, sigma_ms / 1000.0, t.size)
    times = t + jitter
    beats = [Beat(t_s=float(ti), confidence=0.9, roi_agreement=1.0,
                  signal_quality=0.9, amplitude=1.0, prominence=1.0)
             for ti, k in zip(times, keep) if k]

    n_false = int(round(false_rate * t.size))
    if n_false:
        ft = rng.uniform(0.0, float(t[-1]), n_false)
        survive = rng.random(n_false) < false_survival
        for x, sv in zip(ft, survive):
            conf = float(rng.uniform(0.55, 0.9)) if sv \
                else float(np.clip(rng.normal(0.2, 0.05), 0.01, 0.45))
            beats.append(Beat(t_s=float(x), confidence=conf,
                              roi_agreement=0.5, signal_quality=0.4,
                              amplitude=0.3, prominence=0.2))
    beats.sort(key=lambda b: b.t_s)
    duration = float(t[-1] + np.mean(rr) / 1000.0)
    return BeatSeries(beats, fps, duration)


def feature_vector(series: BeatSeries, min_conf: float = 0.5) -> np.ndarray:
    """PRODUCTION path: clean_runs -> run features -> fixed vector.

    Missing/underpowered estimators stay NaN here; imputation is a fitted
    transform and happens inside model.fit on the training fold only.
    """
    rs = clean_runs(series, min_conf=min_conf, min_run_beats=4)
    f = compute_rhythm_features_from_runs(rs.runs, rs.run_confidences,
                                          rs.dropout_rate)
    return np.array([f.values.get(k, float("nan")) for k in FEATURE_NAMES])


# ------------------------------------------------------------- classifiers
def _version_of(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class LogisticModel:
    """L2 logistic regression with in-fit NaN-median imputation and
    standardisation (both therefore split-safe by construction)."""

    def __init__(self, l2: float = 1e-2):
        self.l2 = float(l2)
        self.medians = self.mu = self.sd = self.w = None
        self.b = 0.0
        self.version = ""

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float).copy()
        for j in range(X.shape[1]):
            bad = ~np.isfinite(X[:, j])
            X[bad, j] = self.medians[j]
        # winsorise at ±8 SD: a near-constant column with one extreme value
        # otherwise produces huge z-scores and optimizer overflow
        return np.clip((X - self.mu) / self.sd, -8.0, 8.0)

    def fit(self, X, y) -> "LogisticModel":
        from scipy.optimize import minimize
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.medians = np.where(np.isfinite(med), med, 0.0)
        Xi = X.copy()
        for j in range(X.shape[1]):
            Xi[~np.isfinite(Xi[:, j]), j] = self.medians[j]
        self.mu = Xi.mean(axis=0)
        self.sd = Xi.std(axis=0) + 1e-9
        Z = np.clip((Xi - self.mu) / self.sd, -8.0, 8.0)

        def obj(wb):
            w, b = wb[:-1], wb[-1]
            z = Z @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            eps = 1e-12
            nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            nll += 0.5 * self.l2 * np.dot(w, w)
            g = Z.T @ (p - y) / y.size + self.l2 * w
            gb = np.mean(p - y)
            return nll, np.concatenate([g, [gb]])

        res = minimize(obj, np.zeros(Z.shape[1] + 1), jac=True,
                       method="L-BFGS-B", options={"maxiter": 500})
        self.w, self.b = res.x[:-1], float(res.x[-1])
        self.version = _version_of(self._payload())
        return self

    def predict_proba(self, X) -> np.ndarray:
        Z = self._prep(np.asarray(X, float))
        z = Z @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _payload(self) -> dict:
        return {"type": "logreg", "l2": self.l2,
                "medians": np.round(self.medians, 10).tolist(),
                "mu": np.round(self.mu, 10).tolist(),
                "sd": np.round(self.sd, 10).tolist(),
                "w": np.round(self.w, 10).tolist(), "b": round(self.b, 10)}

    def to_json(self) -> str:
        return json.dumps({**self._payload(), "version": self.version})

    @classmethod
    def from_json(cls, blob: str) -> "LogisticModel":
        d = json.loads(blob)
        m = cls(l2=d["l2"])
        m.medians = np.asarray(d["medians"])
        m.mu = np.asarray(d["mu"]); m.sd = np.asarray(d["sd"])
        m.w = np.asarray(d["w"]); m.b = float(d["b"])
        m.version = d.get("version") or _version_of(m._payload())
        return m


class GradientBoostedTrees:
    """Small XGBoost-style gradient boosting (logloss, Newton leaves)."""

    def __init__(self, n_trees: int = 120, max_depth: int = 3,
                 learning_rate: float = 0.1, min_leaf: int = 5,
                 reg_lambda: float = 1.0):
        self.n_trees = n_trees; self.max_depth = max_depth
        self.lr = learning_rate; self.min_leaf = min_leaf
        self.lam = reg_lambda
        self.medians = None
        self.f0 = 0.0
        self.trees: list = []
        self.version = ""

    # ---- tree building
    def _fit_tree(self, X, g, h, depth):
        G, H = g.sum(), h.sum()
        leaf = {"leaf": float(G / (H + self.lam))}
        if depth == 0 or g.size < 2 * self.min_leaf:
            return leaf
        best = None
        base = G * G / (H + self.lam)
        for j in range(X.shape[1]):
            col = X[:, j]
            qs = np.unique(np.quantile(col, np.linspace(0.1, 0.9, 12)))
            for thr in qs:
                m = col <= thr
                nl = int(m.sum())
                if nl < self.min_leaf or g.size - nl < self.min_leaf:
                    continue
                Gl, Hl = g[m].sum(), h[m].sum()
                Gr, Hr = G - Gl, H - Hl
                gain = Gl * Gl / (Hl + self.lam) + Gr * Gr / (Hr + self.lam) \
                    - base
                if best is None or gain > best[0]:
                    best = (gain, j, float(thr), m)
        if best is None or best[0] <= 1e-9:
            return leaf
        _, j, thr, m = best
        return {"feat": j, "thr": thr,
                "left": self._fit_tree(X[m], g[m], h[m], depth - 1),
                "right": self._fit_tree(X[~m], g[~m], h[~m], depth - 1)}

    @staticmethod
    def _tree_predict(node, X):
        if "leaf" in node:
            return np.full(X.shape[0], node["leaf"])
        m = X[:, node["feat"]] <= node["thr"]
        out = np.empty(X.shape[0])
        out[m] = GradientBoostedTrees._tree_predict(node["left"], X[m])
        out[~m] = GradientBoostedTrees._tree_predict(node["right"], X[~m])
        return out

    def _impute(self, X):
        X = np.asarray(X, float).copy()
        for j in range(X.shape[1]):
            X[~np.isfinite(X[:, j]), j] = self.medians[j]
        return X

    def fit(self, X, y) -> "GradientBoostedTrees":
        X = np.asarray(X, float); y = np.asarray(y, float)
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.medians = np.where(np.isfinite(med), med, 0.0)
        Xi = self._impute(X)
        p0 = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        self.f0 = float(np.log(p0 / (1 - p0)))
        F = np.full(y.size, self.f0)
        self.trees = []
        for _ in range(self.n_trees):
            p = 1.0 / (1.0 + np.exp(-F))
            g = y - p
            h = p * (1 - p) + 1e-12
            tree = self._fit_tree(Xi, g, h, self.max_depth)
            self.trees.append(tree)
            F = F + self.lr * self._tree_predict(tree, Xi)
        self.version = _version_of({"f0": self.f0, "n": len(self.trees),
                                    "trees": self.trees})
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xi = self._impute(X)
        F = np.full(Xi.shape[0], self.f0)
        for tree in self.trees:
            F += self.lr * self._tree_predict(tree, Xi)
        return 1.0 / (1.0 + np.exp(-F))

    def to_json(self) -> str:
        return json.dumps({"type": "gbt", "n_trees": self.n_trees,
                           "max_depth": self.max_depth, "lr": self.lr,
                           "min_leaf": self.min_leaf, "lam": self.lam,
                           "medians": self.medians.tolist(),
                           "f0": self.f0, "trees": self.trees,
                           "version": self.version})

    @classmethod
    def from_json(cls, blob: str) -> "GradientBoostedTrees":
        d = json.loads(blob)
        m = cls(n_trees=d["n_trees"], max_depth=d["max_depth"],
                learning_rate=d["lr"], min_leaf=d["min_leaf"],
                reg_lambda=d["lam"])
        m.medians = np.asarray(d["medians"])
        m.f0 = float(d["f0"]); m.trees = d["trees"]
        m.version = d.get("version", "")
        return m


_MODELS = {"logreg": LogisticModel, "gbt": GradientBoostedTrees}


# ---------------------------------------------------------------- CV
def _stub_recording(pid: str, split: Split) -> Recording:
    """Minimal Recording so the CANONICAL split-safety guard can screen the
    fit set — the guard's contract is Recordings, and re-implementing it
    here would fork the leakage policy."""
    return Recording(
        recording_id=f"stub-{pid}", participant_id=pid, session_id="s",
        site_id="site", video_path="", ecg_path="", video_start_utc="",
        video_end_utc="", ecg_start_utc="", ecg_end_utc="", duration_s=0.0,
        capture=CaptureConfig("stub", "0", "front", 1, 1, 30.0),
        sync=SyncRecord(SyncMethod.NONE, 0.0, 0.0), split=split)


def cross_validate_participants(X, y, participant_ids, *, model: str,
                                k: int = 5, seed: int = 20260814) -> dict:
    """Out-of-fold scores with PARTICIPANT-level folds.

    Fold assignment is a pure function of (participant_id, seed) — the same
    identity-only hashing contract as datasets/splits, so label evolution
    or window reordering can never migrate a subject between folds.
    """
    X = np.asarray(X, float); y = np.asarray(y, int)
    pid = np.asarray(participant_ids)
    uniq = sorted(set(pid.tolist()))
    fold_of = {p: int(_stable_unit_interval(str(p), seed) * k) for p in uniq}

    oof = np.full(y.size, np.nan)
    per_fold = []
    for f in range(k):
        test_p = {p for p in uniq if fold_of[p] == f}
        if not test_p:
            continue
        train_mask = np.array([p not in test_p for p in pid])
        if len(np.unique(y[train_mask])) < 2:
            raise ValueError(f"fold {f}: training data single-class; "
                             "need more participants")
        split_map = {p: (Split.DEV if p in test_p else Split.TRAIN)
                     for p in uniq}
        windows_are_participant_grouped(pid.tolist(), split_map)
        assert_preprocessing_is_split_safe(
            [_stub_recording(p, split_map[p])
             for p in sorted({q for q, m in zip(pid.tolist(), train_mask)
                              if m})])
        m = _MODELS[model]().fit(X[train_mask], y[train_mask])
        oof[~train_mask] = m.predict_proba(X[~train_mask])
        per_fold.append({"fold": f, "n_test": int((~train_mask).sum()),
                         "n_test_participants": len(test_p)})
    if np.any(~np.isfinite(oof)):
        raise RuntimeError("some windows never received an out-of-fold score")
    return {"oof_scores": oof, "fold_of_participant": fold_of,
            "per_fold": per_fold}


def train_model_a(X, y, *, model: str = "logreg") -> dict:
    """Fit the shippable Model A artifact on the full provided data.

    Evaluation numbers must NEVER come from this artifact — they come from
    cross_validate_participants. This is the deployment fit only.
    """
    m = _MODELS[model]().fit(np.asarray(X, float), np.asarray(y, int))
    return {"type": model, "model_json": m.to_json(),
            "version": f"model-a-{model}-{m.version}",
            "feature_names": list(FEATURE_NAMES), "threshold": 0.5}


def load_model_a(artifact: dict):
    m = _MODELS[artifact["type"]].from_json(artifact["model_json"])
    return m, artifact
