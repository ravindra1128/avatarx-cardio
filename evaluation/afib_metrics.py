"""
AFib-level evaluation: confidence intervals, subgroups, risk-coverage.

Three rules encoded here rather than left to discipline:

1.  NO RAW ACCURACY as a headline. At 2% prevalence a model that always says
    "no AF" scores 98% accurate and is worthless. `summary()` refuses to
    return accuracy unless prevalence is reported beside it.

2.  NO-READ RATE IS A PRIMARY RESULT. Quality gating is legitimate and
    necessary, but a system reporting Se 95 / Sp 99 on 60% of scans is a
    different product from one reporting Se 88 / Sp 96 on 95%. Prior AvatarX
    research found that no paper in this field reports accuracy as a function
    of rejection rate -- every gating paper gives error at a threshold and
    omits how much data survived. `risk_coverage_curve` produces the missing
    curve.

3.  PARTICIPANT-LEVEL AND RECORDING-LEVEL ARE DIFFERENT NUMBERS. Both are
    reported. Confidence intervals at the participant level use a cluster
    bootstrap that resamples PARTICIPANTS, because recordings from one person
    are not independent observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional, Callable
import numpy as np


# --------------------------------------------------------------------------
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval. Correct near 0 and 1, unlike the normal approx."""
    if n == 0:
        return (float("nan"), float("nan"))
    from math import sqrt
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(alpha)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _z(alpha: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha / 2)


@dataclass
class ClassificationResult:
    n: int
    tp: int; fp: int; tn: int; fn: int
    sensitivity: float; sens_ci: tuple[float, float]
    specificity: float; spec_ci: tuple[float, float]
    ppv: float; ppv_ci: tuple[float, float]
    npv: float; npv_ci: tuple[float, float]
    f1: float
    prevalence: float
    auroc: Optional[float] = None
    auroc_ci: Optional[tuple[float, float]] = None
    no_read_rate: float = 0.0
    n_no_read: int = 0

    def summary(self) -> dict:
        return {
            "n_analysed": self.n, "n_no_read": self.n_no_read,
            "no_read_rate": round(self.no_read_rate, 4),
            "prevalence_in_analysed": round(self.prevalence, 4),
            "sensitivity": _fmt(self.sensitivity, self.sens_ci),
            "specificity": _fmt(self.specificity, self.spec_ci),
            "ppv": _fmt(self.ppv, self.ppv_ci),
            "npv": _fmt(self.npv, self.npv_ci),
            "f1": round(self.f1, 4),
            "auroc": _fmt(self.auroc, self.auroc_ci) if self.auroc is not None else None,
            "false_positives_per_1000_analysed":
                round(1000 * self.fp / self.n, 1) if self.n else None,
            "NOTE": ("Raw accuracy is deliberately omitted. Interpret PPV only "
                     "against the deployment prevalence, not this cohort's."),
        }


def _fmt(v, ci):
    if v is None or not np.isfinite(v):
        return None
    return {"value": round(float(v), 4),
            "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)] if ci else None}


def evaluate_binary(y_true: Sequence[int], y_pred: Sequence[int],
                    y_score: Optional[Sequence[float]] = None,
                    no_read_mask: Optional[Sequence[bool]] = None,
                    participant_ids: Optional[Sequence[str]] = None,
                    n_boot: int = 2000, seed: int = 20260814) -> ClassificationResult:
    """Binary AF evaluation with no-read accounting and cluster bootstrap CIs."""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    nr = np.zeros(y_true.size, bool) if no_read_mask is None else np.asarray(no_read_mask, bool)

    keep = ~nr
    yt, yp = y_true[keep], y_pred[keep]
    ys = np.asarray(y_score, float)[keep] if y_score is not None else None
    pid = np.asarray(participant_ids)[keep] if participant_ids is not None else None

    tp = int(np.sum((yt == 1) & (yp == 1))); fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0))); fn = int(np.sum((yt == 1) & (yp == 0)))
    n = int(keep.sum())
    se = tp / (tp + fn) if (tp + fn) else float("nan")
    sp = tn / (tn + fp) if (tn + fp) else float("nan")
    pv = tp / (tp + fp) if (tp + fp) else float("nan")
    nv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")

    if n == 0:
        # every scan was a no-read: there is nothing to bound, and the
        # cluster bootstrap would resample an empty participant set.
        # A cohort the detector declined entirely is a legitimate (and
        # informative) result, not a crash.
        nan_ci = (float("nan"), float("nan"))
        se_ci = sp_ci = pv_ci = nv_ci = nan_ci
    elif pid is None:
        se_ci = wilson_ci(tp, tp + fn); sp_ci = wilson_ci(tn, tn + fp)
        pv_ci = wilson_ci(tp, tp + fp); nv_ci = wilson_ci(tn, tn + fn)
    else:
        se_ci, sp_ci, pv_ci, nv_ci = _cluster_bootstrap_ci(yt, yp, pid, n_boot, seed)

    auroc = auroc_ci = None
    if ys is not None and len(np.unique(yt)) == 2:
        auroc = _auroc(yt, ys)
        auroc_ci = _auroc_boot_ci(yt, ys, pid, n_boot, seed)

    return ClassificationResult(
        n=n, tp=tp, fp=fp, tn=tn, fn=fn,
        sensitivity=se, sens_ci=se_ci, specificity=sp, spec_ci=sp_ci,
        ppv=pv, ppv_ci=pv_ci, npv=nv, npv_ci=nv_ci, f1=f1,
        prevalence=float(np.mean(yt)) if n else float("nan"),
        auroc=auroc, auroc_ci=auroc_ci,
        no_read_rate=float(np.mean(nr)), n_no_read=int(nr.sum()))


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    o = np.argsort(s, kind="mergesort")
    r = np.empty_like(o, float); r[o] = np.arange(1, s.size + 1)
    # average ranks for ties
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        sums = np.zeros(u.size); np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    n1 = float(np.sum(y == 1)); n0 = float(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((np.sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def _cluster_bootstrap_ci(yt, yp, pid, n_boot, seed):
    """Resample PARTICIPANTS, not recordings. Recordings within a person are
    correlated; resampling them independently understates every interval."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(pid)
    idx_of = {p: np.flatnonzero(pid == p) for p in uniq}
    out = {k: [] for k in ("se", "sp", "pv", "nv")}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_of[p] for p in pick])
        t, p_ = yt[idx], yp[idx]
        tp = np.sum((t == 1) & (p_ == 1)); fp = np.sum((t == 0) & (p_ == 1))
        tn = np.sum((t == 0) & (p_ == 0)); fn = np.sum((t == 1) & (p_ == 0))
        if (tp + fn): out["se"].append(tp / (tp + fn))
        if (tn + fp): out["sp"].append(tn / (tn + fp))
        if (tp + fp): out["pv"].append(tp / (tp + fp))
        if (tn + fn): out["nv"].append(tn / (tn + fn))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) \
        if len(a) > 10 else (float("nan"), float("nan"))
    return q(out["se"]), q(out["sp"]), q(out["pv"]), q(out["nv"])


def _auroc_boot_ci(yt, ys, pid, n_boot, seed):
    rng = np.random.default_rng(seed + 1)
    vals = []
    if pid is None:
        for _ in range(n_boot):
            i = rng.integers(0, yt.size, yt.size)
            if len(np.unique(yt[i])) == 2:
                vals.append(_auroc(yt[i], ys[i]))
    else:
        uniq = np.unique(pid); idx_of = {p: np.flatnonzero(pid == p) for p in uniq}
        for _ in range(n_boot):
            pick = rng.choice(uniq, uniq.size, replace=True)
            i = np.concatenate([idx_of[p] for p in pick])
            if len(np.unique(yt[i])) == 2:
                vals.append(_auroc(yt[i], ys[i]))
    if len(vals) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------
def risk_coverage_curve(y_true: Sequence[int], y_pred: Sequence[int],
                        quality: Sequence[float],
                        coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.75, 0.6, 0.5)
                        ) -> list[dict]:
    """Accuracy as a function of retained fraction -- the missing curve.

    Sort by signal quality, keep the top `c` fraction, recompute. This is the
    ONLY defensible way to choose a gating threshold, and it must be reported
    alongside any headline operating point.
    """
    y_true = np.asarray(y_true, int); y_pred = np.asarray(y_pred, int)
    q = np.asarray(quality, float)
    order = np.argsort(-q)
    rows = []
    for c in coverages:
        k = max(int(round(c * q.size)), 1)
        idx = order[:k]
        t, p = y_true[idx], y_pred[idx]
        tp = np.sum((t == 1) & (p == 1)); fp = np.sum((t == 0) & (p == 1))
        tn = np.sum((t == 0) & (p == 0)); fn = np.sum((t == 1) & (p == 0))
        rows.append({
            "coverage": round(float(k / q.size), 3),
            "n": int(k),
            "quality_threshold": float(q[order[k - 1]]),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "ppv_in_cohort": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
            "n_afib_retained": int(np.sum(t == 1)),
        })
    return rows


def projected_ppv(sensitivity: float, specificity: float,
                  prevalences: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.20)
                  ) -> list[dict]:
    """Project PPV/NPV into deployment prevalences. Cohort PPV is not shippable."""
    out = []
    for p in prevalences:
        tp = p * sensitivity; fn = p * (1 - sensitivity)
        fp = (1 - p) * (1 - specificity); tn = (1 - p) * specificity
        out.append({
            "prevalence": p,
            "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
            "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
            "false_positives_per_1000_scans": 1000 * fp,
            "true_positives_per_1000_scans": 1000 * tp,
            "false_alarms_per_true_case": fp / tp if tp else float("inf"),
        })
    return out


def serial_confirmation(sensitivity: float, specificity: float,
                        k: int, n: int,
                        fp_persistent_share: float = 0.0) -> tuple[float, float]:
    """Effective Se/Sp when k of n scans must be positive.

    v2 (pressure-test finding P3): the independence model is WRONG for this
    problem's dominant false-positive source. Ectopy, RSA and face/skin-tone
    interactions are PERSISTENT — a subject who false-positives once will
    false-positive on every repeat, so serial confirmation does nothing
    against them. It suppresses only TRANSIENT (artifact/motion) FPs.

    `fp_persistent_share` = fraction of single-scan false positives from
    persistent causes. Evidence anchors: 100% of DoubleCheck-AF's PPG false
    positives came from the frequent-ectopy arm; 40% of Apple's false
    notifications had another arrhythmia on the patch. Planning value: 0.5-0.8.
    At Se .85/Sp .94, 2-of-3, prevalence 2%: PPV is 0.65 at share=0 but 0.28
    at share=0.8. Default 0.0 preserves the (optimistic) v1 behaviour —
    ALWAYS quote the share used.

    Sensitivity model assumes AF is present throughout all n scans, i.e.
    scans are taken within one sitting (minutes apart). Spacing scans across
    days turns paroxysmal Se into a burden-sampling problem instead — a
    different mechanism with opposite behaviour; do not mix the two.
    """
    from math import comb
    kofn = lambda p: sum(comb(n, i) * p ** i * (1 - p) ** (n - i)
                         for i in range(k, n + 1))
    f = 1.0 - specificity
    rho = fp_persistent_share * f                 # subjects who fire every scan
    q = ((1.0 - fp_persistent_share) * f) / (1.0 - rho) if rho < 1.0 else 0.0
    se_k = kofn(sensitivity)
    sp_k = 1.0 - (rho + (1.0 - rho) * kofn(q))
    return se_k, sp_k


def no_read_report(no_read_mask: Sequence[bool],
                   groups: dict[str, Sequence],
                   max_ratio: float = 1.5,
                   max_abs_gap: float = 0.10) -> tuple[bool, dict]:
    """No-read (abstention) parity audit — v2 (pressure-test finding P8).

    The fairness gate on Se/Sp among ANALYSED scans is gameable by abstention:
    a system can 'pass' on dark skin by refusing to read dark-skinned faces,
    and can 'detect AF well' by abstaining on AF (pulse deficit lowers beat
    confidence, so quality gating preferentially funnels AF scans into
    no-read). Parity of the abstention rate itself must therefore be audited
    across every axis — including RHYTHM: pass groups={"rhythm": [...]} and
    inspect P(no-read | AF) vs P(no-read | non-AF).

    Fails when a level's no-read rate exceeds the best level's by more than
    `max_ratio` AND by more than `max_abs_gap` absolute (the AND avoids
    flagging 1% vs 2%).
    """
    nr = np.asarray(no_read_mask, bool)
    report: dict = {}
    fails: list[str] = []
    for name, vals in groups.items():
        v = np.asarray(vals)
        rates = {}
        for lev in sorted(set(v.tolist())):
            m = v == lev
            rates[str(lev)] = {"n": int(m.sum()),
                               "no_read_rate": float(np.mean(nr[m])) if m.sum() else float("nan")}
        report[name] = rates
        powered = {k: r["no_read_rate"] for k, r in rates.items()
                   if r["n"] >= 30 and np.isfinite(r["no_read_rate"])}
        if len(powered) < 2:
            continue
        best = min(powered.values())
        for lev, rate in powered.items():
            if rate > max(best * max_ratio, best + max_abs_gap):
                fails.append(f"{name}={lev}: no-read {rate:.1%} vs best {best:.1%} "
                             f"(ratio {rate / best if best > 0 else float('inf'):.2f})")
    return (len(fails) == 0), {"levels": report, "failures": fails}


def subgroup_report(y_true, y_pred, groups: dict[str, Sequence],
                    min_n: int = 30) -> dict:
    """Stratified performance. Never ship an aggregate number alone.

    Strata below `min_n` are reported as UNDERPOWERED with their counts rather
    than given a point estimate -- a sensitivity of 1.00 on n=4 is not a
    result, and printing it invites exactly the wrong conclusion.
    """
    y_true = np.asarray(y_true, int); y_pred = np.asarray(y_pred, int)
    rep: dict = {}
    for name, vals in groups.items():
        v = np.asarray(vals)
        rep[name] = {}
        for lev in sorted(set(v.tolist())):
            m = v == lev
            t, p = y_true[m], y_pred[m]
            n_af = int(np.sum(t == 1))
            if m.sum() < min_n or n_af < 5:
                rep[name][str(lev)] = {"status": "UNDERPOWERED",
                                       "n": int(m.sum()), "n_afib": n_af}
                continue
            tp = np.sum((t == 1) & (p == 1)); fp = np.sum((t == 0) & (p == 1))
            tn = np.sum((t == 0) & (p == 0)); fn = np.sum((t == 1) & (p == 0))
            rep[name][str(lev)] = {
                "status": "OK", "n": int(m.sum()), "n_afib": n_af,
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                "sens_ci95": wilson_ci(int(tp), int(tp + fn)),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
                "spec_ci95": wilson_ci(int(tn), int(tn + fp)),
            }
    return rep


def fairness_gate(report: dict, max_gap: float = 0.10) -> tuple[bool, list[str]]:
    """No powered subgroup may fall more than `max_gap` below the best.

    Applied to sensitivity AND specificity. A model that is sensitive but
    unspecific in dark skin fails just as surely as the reverse -- the
    published camera-AF operating point for darkest skin (Se 97 / Sp 81)
    is precisely that failure shape.
    """
    fails: list[str] = []
    for dim, levels in report.items():
        for metric in ("sensitivity", "specificity"):
            vals = {k: v[metric] for k, v in levels.items()
                    if v.get("status") == "OK" and np.isfinite(v.get(metric, np.nan))}
            if len(vals) < 2:
                continue
            best = max(vals.values())
            for lev, val in vals.items():
                if best - val > max_gap:
                    fails.append(f"{dim}={lev}: {metric} {val:.3f} is "
                                 f"{best - val:.3f} below best ({best:.3f})")
    return (len(fails) == 0), fails
