"""Sham-vs-real comparison — the expectancy / apparatus / artifact control.

The protocol runs, per subject, a SHAM trial (thermoneutral ~33 C immersion:
same apparatus, same expectation, no thermal stimulus) and a REAL trial (cold
~4 C immersion: the actual cold-pressor). The sham exists to answer one
question (owner brief section 14):

    if the facial change under sham is comparable to real, the apparent
    effect is EXPECTANCY / APPARATUS / ARTIFACT, not the provocation.

This module computes, for each condition, the facial dPI (magnitude compared;
direction reported, not assumed -- the facial response to cold is not a priori
a rise or a fall) and the FINGER control dPI (expected to DROP under real cold
and NOT under sham). It then compares them and returns a verdict, gated by the
finger cross-check: a real>>sham result is only trustworthy if the finger
confirms the cold reached the periphery in real and not in sham.

Two entry points:
  * `compare_sessions(real, sham)`  -- two separate single-trial sessions.
  * `trial_from_session(session, condition)` + `compare_sham_real(...)` --
    named phases (sham_*/real_*) within one continuous session.

No score, no physiological claim: this is an instrument/design control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import FprError
from . import pi as _pi
from . import reference as _ref


@dataclass
class TrialResult:
    condition: str                                   # "sham" | "real" | "null"
    face_baseline_pi_pct: float
    face_stimulus_pi_pct: float
    face_delta_pct: float                            # (stim/base - 1)*100
    windows: dict
    n_face_windows: int = 0
    finger: Optional[_ref.StimulusVerdict] = None    # finger control verdict
    face_pi: Optional[_pi.PiResult] = None           # for plotting (not JSON)
    finger_pi: Optional[_ref.FingerResult] = None    # for plotting (not JSON)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"condition": self.condition,
                "face_baseline_pi_pct": self.face_baseline_pi_pct,
                "face_stimulus_pi_pct": self.face_stimulus_pi_pct,
                "face_delta_pct": self.face_delta_pct,
                "n_face_windows": self.n_face_windows,
                "windows": {k: list(v) for k, v in self.windows.items()},
                "finger": self.finger.to_dict() if self.finger else None,
                "notes": list(self.notes)}


@dataclass
class ShamRealComparison:
    real: TrialResult
    sham: TrialResult
    category: str
    verdict: str
    ratio: float
    caveats: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"category": self.category, "verdict": self.verdict,
                "ratio": self.ratio, "caveats": list(self.caveats),
                "real": self.real.to_dict(), "sham": self.sham.to_dict()}


# --------------------------------------------------------------------------
# Build a trial
# --------------------------------------------------------------------------
def analyze_trial(face: _pi.PiResult, windows: dict, condition: str, *,
                  finger: Optional[_ref.FingerResult] = None) -> TrialResult:
    """Facial dPI (and, if present, the finger control) for one trial's
    baseline/stimulus windows."""
    base, stim, delta = _pi.pi_change(face.win_centers_s, face.win_pi,
                                      windows["baseline"], windows["stimulus"])
    fv = None
    if finger is not None:
        fv = _ref.stimulus_response(finger, windows["baseline"],
                                    windows["stimulus"])
    n_win = int(np.isfinite(face.win_pi).sum()) if face.win_pi.size else 0
    notes = []
    if not np.isfinite(delta):
        notes.append("facial dPI not computable (no PI windows in one of the "
                     "trial windows)")
    return TrialResult(
        condition=condition, face_baseline_pi_pct=base * 100.0,
        face_stimulus_pi_pct=stim * 100.0, face_delta_pct=delta,
        windows=windows, n_face_windows=n_win, finger=fv, face_pi=face,
        finger_pi=finger, notes=notes)


def trial_from_session(session, condition: str, *, baseline_s: float = 120.0,
                       recovery_s: float = 180.0) -> TrialResult:
    """Extract the facial PI from a session's raw/session.avi, derive the
    trial windows from its events, and analyze the trial (with the finger
    control if present)."""
    from .extract import extract_roi_trace
    roi = session.load_roi()
    ex = extract_roi_trace(session.path("raw/session.avi"), roi)
    face = _pi.process_pi(ex.green, ex.t, ex.fps)     # PI series (baseline=whole)
    events = session.load_events()
    if not events:
        raise FprError(f"{condition}: session has no controls/events.csv")
    windows = _ref.trial_windows(events, condition, baseline_s=baseline_s,
                                 recovery_s=recovery_s)
    finger = None
    if session.exists("controls/finger_ppg.csv") or \
            session.exists("controls/finger_pi.csv"):
        finger = session.load_finger()
    return analyze_trial(face, windows, condition, finger=finger)


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------
def compare_sham_real(real: TrialResult, sham: TrialResult, *,
                      ratio: float = 0.7) -> ShamRealComparison:
    """Compare facial dPI magnitudes and gate the verdict on the finger
    control. `ratio`: sham is "comparable" to real when |sham| >= ratio*|real|.
    """
    sc = _ref.sham_check(real.face_delta_pct, sham.face_delta_pct, ratio=ratio)
    category = sc["category"]
    caveats: list = []

    real_finger_responded = (real.finger is not None and
                             real.finger.verdict == "responded")
    sham_finger_neutral = (sham.finger is None or
                           sham.finger.verdict != "responded")

    if category == _ref.EXPECTANCY_OR_ARTIFACT:
        verdict = ("sham reproduces the real facial change "
                   f"(sham dPI {sham.face_delta_pct:+.1f}% vs real "
                   f"{real.face_delta_pct:+.1f}%): the apparent effect is "
                   "expectancy / apparatus / artifact, NOT the cold "
                   "provocation")
    elif category == _ref.REAL_EXCEEDS_SHAM:
        verdict = ("the real facial change exceeds sham "
                   f"(real dPI {real.face_delta_pct:+.1f}% vs sham "
                   f"{sham.face_delta_pct:+.1f}%): consistent with an effect "
                   "specific to the cold provocation")
        if real.finger is not None and not real_finger_responded:
            caveats.append("the REAL finger control did not confirm a "
                           "peripheral response -- the cold may not have "
                           "reached the periphery; treat the facial result "
                           "with caution")
        if not sham_finger_neutral:
            caveats.append("the SHAM finger control ALSO responded -- the "
                           "sham was not thermally neutral (or a systemic "
                           "artifact is present); the specificity claim is "
                           "weakened")
    else:                                             # UNRESOLVED
        verdict = ("sham-vs-real is unresolved: one facial dPI was not "
                   "computable -- check capture/SNR before interpreting")

    # finger direction sanity (cold pressor is expected to LOWER finger PI)
    if real_finger_responded and real.finger.direction == "rise":
        caveats.append("the real finger perfusion ROSE rather than fell under "
                       "cold -- unexpected direction; verify the control")

    return ShamRealComparison(real=real, sham=sham, category=category,
                              verdict=verdict, ratio=ratio, caveats=caveats)


def compare_sessions(real_session, sham_session, *, baseline_s: float = 120.0,
                     recovery_s: float = 180.0, ratio: float = 0.7
                     ) -> ShamRealComparison:
    """End-to-end comparison of two separate single-trial sessions."""
    real = trial_from_session(real_session, "real", baseline_s=baseline_s,
                              recovery_s=recovery_s)
    sham = trial_from_session(sham_session, "sham", baseline_s=baseline_s,
                              recovery_s=recovery_s)
    return compare_sham_real(real, sham, ratio=ratio)


def format_comparison(cmp: ShamRealComparison) -> str:
    """Human-readable sham-vs-real summary."""
    from . import WATERMARK
    lines = [WATERMARK, "", f"SHAM vs REAL: {cmp.category}", "",
             f"  {cmp.verdict}", ""]
    for tr in (cmp.real, cmp.sham):
        fv = tr.finger
        fstr = (f"finger {fv.verdict} ({fv.delta_pct:+.1f}% {fv.direction})"
                if fv else "no finger control")
        lines.append(f"  {tr.condition:>4}: facial dPI "
                     f"{tr.face_delta_pct:+.1f}%  |  {fstr}")
    for c in cmp.caveats:
        lines.append(f"  ! {c}")
    return "\n".join(lines)
