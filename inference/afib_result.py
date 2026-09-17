"""ONE AFib result per completed scan: AFIB_DETECTED | AFIB_NOT_DETECTED |
INCONCLUSIVE (owner goal, 2026-09-17).

This is a CONTRACT layer, not a classifier. It reads the decision the
pipeline already made (ScanOutcome, predicted_class, the classifier's
probability when a probabilistic classifier ran, the gates that failed) and
names exactly one of three results, with a structured basis that says which
evidence carried it and, for INCONCLUSIVE, which of three things was missing:

  capture   the scan itself did not yield a readable pulse (face, light,
            frame rate, tracking) - NO_RESULT before any beats were read;
  signal    beats were read but the scan-level evidence did not support a
            rhythm statement (quality, cross-region coherence, timing, too
            few clean intervals) - the pipeline's REPEAT_SCAN;
  rhythm    the rhythm WAS assessable and the evidence sits between the two
            decision thresholds: neither "AF pattern present" nor "AF pattern
            absent" is supported at the targeted error rates.

THE DECISION BAND. With the RR classifier (decision.classifier: model_a) the
scan carries afib_probability. Two thresholds tau_lo < tau_hi come from the
model artifact (`decision_band`, set by scripts/e6_window45.py on MIMIC
PERform AF out-of-fold scores at phone-level timing noise so that decided
windows clear the targeted sensitivity AND specificity; everything between
is inconclusive). They are never tuned on a scan.
  p >= tau_hi  and the pipeline's AF-call gates passed   -> AFIB_DETECTED
  p >= tau_hi  but an AF-call gate failed (unverified)    -> INCONCLUSIVE (rhythm)
  p <= tau_lo  on an ACCEPT                               -> AFIB_NOT_DETECTED
  tau_lo < p < tau_hi                                     -> INCONCLUSIVE (rhythm)
Without a probability (interim rules) the class decides: AFIB_SUGGESTIVE ->
DETECTED; SINUS / HIGH_RATE -> NOT_DETECTED; OTHER_IRREGULAR -> INCONCLUSIVE
(irregular beats that did not meet the AF pattern are exactly the case a
three-way contract must not force).

NEVER-DIAGNOSE. The result is a category the product asked for; the only
prose stays `user_facing_text` (datasets/schema.py, 21 CFR 870.2790). This
module emits no sentence.
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Optional

AFIB_DETECTED = "AFIB_DETECTED"
AFIB_NOT_DETECTED = "AFIB_NOT_DETECTED"
INCONCLUSIVE = "INCONCLUSIVE"
RESULTS = (AFIB_DETECTED, AFIB_NOT_DETECTED, INCONCLUSIVE)

# Fallback band when the artifact carries none: the classifier's own 0.5 cut
# with a symmetric abstention margin. Deliberately WIDE - a band that was
# never measured must abstain more, not less.
DEFAULT_BAND = {"tau_lo": 0.25, "tau_hi": 0.75, "source": "default (unmeasured)"}

_BAND_CACHE: dict = {}


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def decision_band(config: Optional[dict]) -> dict:
    """The (tau_lo, tau_hi) the configured model artifact carries, or the
    default band. Cached per artifact path."""
    dc = (config or {}).get("decision") or {}
    art = dc.get("model_a")
    if isinstance(art, dict) and isinstance(art.get("decision_band"), dict):
        b = art["decision_band"]
        return {"tau_lo": float(b["tau_lo"]), "tau_hi": float(b["tau_hi"]),
                "source": art.get("version", "artifact")}
    p = pathlib.Path(dc.get("model_a_path", "models/model_a_v01.json"))
    if not p.is_absolute():
        p = pathlib.Path(__file__).resolve().parents[1] / p
    key = str(p)
    if key in _BAND_CACHE:
        return _BAND_CACHE[key]
    band = dict(DEFAULT_BAND)
    try:
        with open(p) as f:
            art = json.load(f)
        b = art.get("decision_band")
        if isinstance(b, dict) and _f(b.get("tau_lo")) is not None and _f(b.get("tau_hi")) is not None:
            band = {"tau_lo": float(b["tau_lo"]), "tau_hi": float(b["tau_hi"]),
                    "source": art.get("version", p.name)}
    except (OSError, ValueError, TypeError):
        pass
    _BAND_CACHE[key] = band
    return band


def afib_result(doc: dict, config: Optional[dict] = None) -> dict:
    """{result, basis} for one finished response doc (after any second
    interval source has been applied). Pure; never raises."""
    try:
        return _afib_result(doc, config)
    except Exception as e:                                     # noqa: BLE001
        return {"result": INCONCLUSIVE,
                "basis": {"category": "signal", "why": f"result layer failed safely: "
                                                       f"{type(e).__name__}: {e}"}}


def _afib_result(doc: dict, config: Optional[dict]) -> dict:
    outcome = str(doc.get("outcome") or "")
    cls = doc.get("predicted_class")
    p = _f(doc.get("afib_probability"))
    dbg = doc.get("debug") or {}
    rationale = dbg.get("rationale") or {}
    failed = list(rationale.get("gates_failed") or [])
    rule = (rationale.get("rule") or {}) if isinstance(rationale, dict) else {}
    source = doc.get("rhythm_source") or "video"
    band = decision_band(config)
    basis = {"source": source, "outcome": outcome, "predicted_class": cls,
             "afib_probability": p, "band": band, "gates_failed": failed,
             "rule_fired": rule.get("fired"), "stars": doc.get("confidence_stars")}

    if outcome == "NO_RESULT":
        reasons = [str(r) for r in (doc.get("no_read_reasons") or [])]
        capture_like = (not rationale.get("gates")) or any(
            k in " ".join(reasons).lower() for k in ("face", "fps", "frame", "light",
                                                     "illumin", "outside the frame",
                                                     "too close", "too far"))
        basis.update(category="capture" if capture_like else "signal",
                     why=(reasons[0] if reasons else "no readable pulse"))
        return {"result": INCONCLUSIVE, "basis": basis}

    if outcome != "ACCEPT":
        reasons = [str(r) for r in (doc.get("no_read_reasons") or [])]
        # An irregular pattern that the AF-call gates would not verify is a
        # RHYTHM-level inconclusive: the rhythm was assessed and not settled.
        rhythm_level = str(rule.get("fired") or "").startswith("ABSTAIN (irregular") or \
            str(rule.get("fired") or "").startswith("ABSTAIN (confidence")
        basis.update(category="rhythm" if rhythm_level else "signal",
                     why=(reasons[0] if reasons else "rhythm not assessable"))
        return {"result": INCONCLUSIVE, "basis": basis}

    # ---- ACCEPT: a rhythm statement was made ------------------------------
    if cls == "AFIB_SUGGESTIVE":
        # The pipeline only reaches this class through its AF-call gates and
        # the 3-star floor; with a probability it must also clear tau_hi.
        if p is not None and p < band["tau_hi"]:
            basis.update(category="rhythm",
                         why=f"AF pattern called at p={p:.2f}, below the detection "
                             f"threshold {band['tau_hi']:.2f}")
            return {"result": INCONCLUSIVE, "basis": basis}
        basis.update(category=None, why="AF pattern present and verified across regions")
        return {"result": AFIB_DETECTED, "basis": basis}
    if p is not None:
        if p <= band["tau_lo"]:
            basis.update(category=None, why=f"AF probability {p:.2f} <= {band['tau_lo']:.2f}")
            return {"result": AFIB_NOT_DETECTED, "basis": basis}
        basis.update(category="rhythm",
                     why=f"AF probability {p:.2f} between {band['tau_lo']:.2f} and "
                         f"{band['tau_hi']:.2f}: neither result supported")
        return {"result": INCONCLUSIVE, "basis": basis}
    # No probability: the interim rule's class decides, conservatively.
    if cls in ("SINUS", "HIGH_RATE"):
        basis.update(category=None, why=f"regular intervals ({cls}), no AF pattern")
        return {"result": AFIB_NOT_DETECTED, "basis": basis}
    if cls == "OTHER_IRREGULAR":
        basis.update(category="rhythm",
                     why="irregular beats without the AF pattern: not forced either way")
        return {"result": INCONCLUSIVE, "basis": basis}
    basis.update(category="rhythm", why="pulse measured, rhythm not classified")
    return {"result": INCONCLUSIVE, "basis": basis}
