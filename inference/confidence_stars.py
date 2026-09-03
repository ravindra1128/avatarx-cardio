"""
Confidence stars (v0.1.5): a 1-5 star rating of a scan attempt.

The blocking readiness gate is replaced (in `mode: advisory`) by this
rating: the scan always starts unless you are not pointing a working
camera at a face; poor conditions LOWER THE STARS, they do not prevent
the attempt. A refused scan teaches nothing; a 1-star scan carries its
own explanation.

THE RULE THAT MUST NOT BE LOST — lowering the bar to start must not
lower the bar to call AF. Stars are the user-visible expression of the
v0.1.2 evidence gates, never their replacement, enforced by two hard
overrides on the weighted arithmetic:

  * any BLOCKING readiness check fails        -> stars = 1
  * any AF EVIDENCE GATE in decision.evidence
    fails (evaluated with the same semantics
    the decision applies, incl. the v0.1.4.2
    two-region waiver)                        -> stars = min(stars, 2)

so "a scan below 3 stars can never be AFIB_SUGGESTIVE" is arithmetic,
not policy — and the decision layer additionally downgrades any AF call
carrying < 3 stars (inference/decision_logic.py).

Star meanings (docs/READINESS.md):
  5  every advisory check passes with margin
  4  all advisory checks pass
  3  minimum at which a rhythm call is supportable — AF evidence gates met
  2  pulse detected, rhythm call not supportable
  1  conditions too poor, or a blocking check failed

Uses only what `window_evidence()` already produces — no new signal
processing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_DEFAULT_WEIGHTS = {
    "cross_roi_coherence": 0.30,
    "beat_timing": 0.20,
    "signal_snr": 0.15,
    "sqi": 0.15,
    "prelim_beats": 0.10,
    "tracking": 0.05,
    "motion": 0.05,
}
_DEFAULT_CUTS = [0.30, 0.50, 0.68, 0.85]

_DEFAULT_EVID = {
    "coherence_floor": 0.20, "afib_min_coherence": 0.35,
    "afib_min_intervals": 20, "afib_max_harmonic_fraction": 0.20,
    "afib_max_split_fraction": 0.15, "afib_max_timing_precision_ms": 30.0,
    "afib_min_timing_matched": 0.75,
}


@dataclass(frozen=True)
class ConfidenceStars:
    stars: int                  # 1..5
    score: float                # 0..1, continuous weighted margin
    limiting_factor: str        # name of the lowest-margin check
    hint: str                   # that check's existing actionable hint
    per_check: dict             # name -> {value, threshold, margin, weight}


def _margin(check: dict) -> float:
    """Per-check margin in [0, 1] from the value/threshold pair the
    readiness checklist already carries. >= checks: value/threshold;
    <= checks: threshold/value; boolean checks: 0/1. Missing or
    non-finite values fail closed to 0."""
    v, thr, op = check.get("value"), check.get("threshold"), check.get("op")
    if op == "is" or isinstance(v, bool):
        return 1.0 if check.get("pass") else 0.0
    try:
        v = float(v)
        thr = float(thr)
    except (TypeError, ValueError):
        return 1.0 if check.get("pass") else 0.0
    if not np.isfinite(v):
        return 0.0
    if op == ">=":
        if thr <= 0:
            return 1.0 if check.get("pass") else 0.0
        return float(np.clip(v / thr, 0.0, 1.0))
    if op == "<=":
        if v <= 0:
            return 1.0
        return float(np.clip(thr / v, 0.0, 1.0))
    return 1.0 if check.get("pass") else 0.0


def _af_gates_fail(ev: dict, ec: dict, *, final: bool) -> bool:
    """Would any AF evidence gate refuse this evidence? SAME semantics as
    decision_logic (incl. the two-region waiver for the coherence quota);
    the interval-count gate is length-dependent and applies only to the
    final recording (`final=True`), not to a live 8 s window."""
    tp = ev.get("timing_precision_ms")
    tm = ev.get("timing_matched_fraction")
    tp_ok = tp is not None and np.isfinite(tp) and \
        float(tp) <= float(ec["afib_max_timing_precision_ms"])
    tm_ok = tm is not None and np.isfinite(tm) and \
        float(tm) >= float(ec["afib_min_timing_matched"])
    two_region = tp_ok and tm_ok
    coh = ev.get("cross_roi_coherence")
    coh_ok = (coh is not None and np.isfinite(coh) and
              float(coh) >= float(ec["afib_min_coherence"])) or two_region
    hf = ev.get("harmonic_fraction", 0.0) or 0.0
    sf = ev.get("split_fraction", 0.0) or 0.0
    fails = (not coh_ok) or (not tp_ok) or (not tm_ok) or \
        (float(hf) > float(ec["afib_max_harmonic_fraction"])) or \
        (float(sf) > float(ec["afib_max_split_fraction"]))
    if final:
        n_int = ev.get("n_intervals", 0) or 0
        fails = fails or (float(n_int) < float(ec["afib_min_intervals"]))
    return bool(fails)


def confidence_stars(ev: dict, readiness: dict, cfg: dict, *,
                     final: bool = True) -> ConfidenceStars:
    """Rate the attempt 1-5 from the evidence and the readiness checklist.

    `final=True` (a completed recording) applies every AF evidence gate to
    the star ceiling, including the length-dependent interval count;
    `final=False` (the live window meter) skips only that count — an 8 s
    window can never hold 20 intervals and would otherwise pin the live
    meter at 2 stars for everyone.
    """
    dc = cfg.get("decision", {}) if cfg else {}
    cc = dc.get("confidence") or {}
    ec = {**_DEFAULT_EVID, **(dc.get("evidence") or {})}
    weights = {**_DEFAULT_WEIGHTS, **(cc.get("weights") or {})}
    cuts = list(cc.get("star_cuts") or _DEFAULT_CUTS)
    checks = readiness.get("checks") or {}

    per_check = {}
    total_w = 0.0
    acc = 0.0
    worst_name, worst_margin = "insufficient", -1.0
    for name, w in weights.items():
        c = checks.get(name)
        if c is None:
            continue
        m = _margin(c)
        per_check[name] = {"value": c.get("value"),
                           "threshold": c.get("threshold"),
                           "margin": round(m, 3), "weight": w}
        acc += w * m
        total_w += w
        if worst_margin < 0 or m < worst_margin:
            worst_name, worst_margin = name, m
    score = float(acc / total_w) if total_w > 0 else 0.0

    stars = 1 + int(np.searchsorted(np.asarray(cuts, float), score,
                                    side="right"))
    stars = int(np.clip(stars, 1, 5))

    # hard overrides — the safety coupling, not cosmetics
    if not readiness.get("blocking_pass", True):
        stars = 1
        worst_name = next((k for k in readiness.get("failing", [])
                           if k in (readiness.get("checks") or {})),
                          worst_name)
    elif _af_gates_fail(ev, ec, final=final):
        stars = min(stars, 2)

    hint = ""
    if worst_name in checks:
        hint = checks[worst_name].get("hint") or ""
    return ConfidenceStars(stars=stars, score=round(score, 4),
                           limiting_factor=worst_name, hint=hint,
                           per_check=per_check)
