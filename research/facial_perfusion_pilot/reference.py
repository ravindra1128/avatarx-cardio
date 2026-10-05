"""Finger contact-PPG reference channel — the STIMULUS POSITIVE CONTROL.

The finger PPG answers the question the facial signal cannot answer about
itself: did the provocation actually reach the periphery? A cold-pressor
drives sympathetic vasoconstriction, so a contralateral-hand contact PPG
should show its perfusion FALL during immersion. If it does not, the problem
is the stimulus, not the face (owner brief section 14).

Two input formats are accepted:

  * raw waveform  `t_s,ppg`         -- the brief's primary control (section 8).
    HR and a finger perfusion index (AC/DC) are derived here with the SAME
    pipeline as the face (`pi.process_pi`), so face and finger PI are directly
    comparable.
  * device export `t_s,pi_percent`  -- when the oximeter already outputs PI.
    Parsed by the repo's existing `datasets.reference.read_pi_csv` (lazy-
    imported), so a file that feeds the vasotone track feeds this one too.

This module DERIVES a control readout; it is not a physiological measurement
and produces no score. The finger PI is used only to decide whether the
stimulus worked and, with the facial dPI, to attribute a null result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union
import os

import numpy as np

from . import FprError, PULSE_BAND_HZ
from . import pi as _pi

# Rule-of-thumb minimum |dPI| (percent) to call a change a real response.
# Indicative only (report Gate D is indicative), not a validated threshold;
# configurable at every call site.
RESPONSE_MIN_ABS_PCT = 10.0

# Failure-attribution categories (owner brief section 14).
STIMULUS_PROBLEM = "stimulus_problem"
FACIAL_SIGNAL_QUESTION = "facial_signal_or_physiology_question"
BOTH_RESPONDED = "both_responded"
UNRESOLVED = "unresolved"
EXPECTANCY_OR_ARTIFACT = "expectancy_or_artifact"
REAL_EXCEEDS_SHAM = "real_exceeds_sham"
THERMAL_CONFOUND = "thermal_confound"
NO_THERMAL_CONFOUND = "no_thermal_confound"

# |r| between facial PI and skin temperature at/above which a facial change is
# suspected to be thermal, not perfusional (indicative, not validated).
THERMAL_R_FLAG = 0.7


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
def _as_text(text_or_path: str) -> str:
    if "\n" in text_or_path or "," in text_or_path:
        return text_or_path
    if os.path.exists(text_or_path):
        with open(text_or_path) as f:
            return f.read()
    return text_or_path


def read_finger_ppg(text_or_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw contact PPG: two columns `t_s,ppg`, strictly increasing time.

    Fail-closed on an unknown header (the posture of `read_pi_csv`): a
    silently mis-parsed control is worse than a rejected one.
    """
    text = _as_text(text_or_path)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise FprError("finger PPG: empty input")
    head = lines[0].lower().replace(" ", "")
    start = 0
    if head in ("t_s,ppg", "t_s,value", "time,ppg", "t,ppg"):
        start = 1
    elif not _looks_numeric(lines[0]):
        raise FprError("finger PPG: first line must be numeric or the header "
                       "'t_s,ppg' — fail closed on unknown layouts")
    t, v = [], []
    for ln in lines[start:]:
        a, _, b = ln.partition(",")
        try:
            ti, vi = float(a), float(b)
        except ValueError:
            raise FprError(f"finger PPG: non-numeric row {ln!r}")
        if t and ti <= t[-1]:
            raise FprError("finger PPG: time must be strictly increasing")
        t.append(ti)
        v.append(vi)
    if len(t) < 2:
        raise FprError("finger PPG: fewer than 2 samples")
    return np.asarray(t, float), np.asarray(v, float)


def read_finger_pi_export(text_or_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Oximeter PI export `t_s,pi_percent`, via the repo's sanctioned parser
    (`datasets.reference.read_pi_csv`), lazy-imported so the core FPR path
    carries no dependency on it."""
    from datasets.reference import read_pi_csv
    t, v = read_pi_csv(_as_text(text_or_path))
    return np.asarray(t, float), np.asarray(v, float)


def _looks_numeric(line: str) -> bool:
    a, _, b = line.partition(",")
    try:
        float(a), float(b)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
@dataclass
class FingerResult:
    source: str                                      # "waveform" | "pi_export"
    fps: float
    pi_t: np.ndarray                                 # PI series time (s)
    pi_pct: np.ndarray                               # PI series (percent)
    hr_bpm: float = float("nan")
    snr_db: float = float("nan")
    ppg: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    pi_result: Optional[_pi.PiResult] = None
    notes: list = field(default_factory=list)

    def pi_in_window(self, t0: float, t1: float) -> float:
        """Median finger PI (%) over [t0, t1); NaN if no samples there."""
        m = np.isfinite(self.pi_pct) & (self.pi_t >= t0) & (self.pi_t < t1)
        return float(np.median(self.pi_pct[m])) if m.any() else float("nan")

    def summary(self) -> dict:
        return {"source": self.source, "fps": self.fps, "hr_bpm": self.hr_bpm,
                "snr_db": self.snr_db, "n_pi_samples": int(self.pi_pct.size),
                "median_pi_pct": float(np.nanmedian(self.pi_pct))
                if self.pi_pct.size else float("nan"), "notes": list(self.notes)}


def analyze_finger(*, ppg: Optional[np.ndarray] = None,
                   t: Optional[np.ndarray] = None,
                   pi_t: Optional[np.ndarray] = None,
                   pi_pct: Optional[np.ndarray] = None) -> FingerResult:
    """Build a FingerResult from EITHER a raw waveform (ppg, t) OR a device
    PI export (pi_t, pi_pct)."""
    if ppg is not None and t is not None:
        t = np.asarray(t, float)
        ppg = np.asarray(ppg, float)
        if t.shape != ppg.shape or t.size < 2:
            raise FprError("finger waveform: t and ppg must match, len >= 2")
        fps = float(1.0 / np.median(np.diff(t)))
        res = _pi.process_pi(ppg, t, fps)            # baseline = whole record
        return FingerResult(
            source="waveform", fps=fps, pi_t=res.win_centers_s,
            pi_pct=res.win_pi * 100.0, hr_bpm=res.median_hr_bpm,
            snr_db=res.snr_db, ppg=ppg, t=t, pi_result=res)
    if pi_t is not None and pi_pct is not None:
        pt = np.asarray(pi_t, float)
        pv = np.asarray(pi_pct, float)
        fps = float(1.0 / np.median(np.diff(pt))) if pt.size > 1 else \
            float("nan")
        return FingerResult(source="pi_export", fps=fps, pi_t=pt, pi_pct=pv,
                            notes=["PI from device export; HR/SNR not derived"])
    raise FprError("analyze_finger needs (ppg, t) or (pi_t, pi_pct)")


# --------------------------------------------------------------------------
# Stimulus positive control + failure attribution
# --------------------------------------------------------------------------
@dataclass
class StimulusVerdict:
    verdict: str                                     # responded/no_response/unresolved
    delta_pct: float                                 # (stim/base - 1)*100
    baseline_pi_pct: float
    stimulus_pi_pct: float
    direction: str                                   # "drop"/"rise"/"n/a"
    threshold_pct: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def stimulus_response(finger: FingerResult, baseline_window: tuple,
                      stimulus_window: tuple, *,
                      min_abs_pct: float = RESPONSE_MIN_ABS_PCT
                      ) -> StimulusVerdict:
    """Did the finger's perfusion change between the baseline and the stimulus
    window? A cold-pressor is EXPECTED to lower finger PI; direction is
    reported, not required, so the control also works for other provocations.
    """
    base = finger.pi_in_window(*baseline_window)
    stim = finger.pi_in_window(*stimulus_window)
    if not (np.isfinite(base) and base > 0 and np.isfinite(stim)):
        return StimulusVerdict("unresolved", float("nan"), base, stim, "n/a",
                               min_abs_pct)
    delta = (stim / base - 1.0) * 100.0
    direction = "drop" if delta < 0 else "rise"
    verdict = "responded" if abs(delta) >= min_abs_pct else "no_response"
    return StimulusVerdict(verdict, float(delta), float(base), float(stim),
                           direction, min_abs_pct)


def attribute(face_delta_pct: float, finger: StimulusVerdict, *,
              face_min_abs_pct: float = RESPONSE_MIN_ABS_PCT) -> dict:
    """Attribute a facial result using the finger control (owner brief §14).

      finger no_response          -> the STIMULUS is the problem
      finger responded, face flat -> FACIAL signal/SNR/physiology question
      finger responded, face too  -> both responded
      finger unresolved           -> unresolved (control failed)
    """
    if finger.verdict == "unresolved":
        cat, why = UNRESOLVED, ("the finger control itself was not "
                                "resolvable; fix the reference before "
                                "interpreting the face")
    elif finger.verdict == "no_response":
        cat, why = STIMULUS_PROBLEM, ("the finger perfusion did not change: "
                                      "the provocation did not reach the "
                                      "periphery -- a stimulus/compliance "
                                      "problem, not a facial one")
    elif abs(face_delta_pct) < face_min_abs_pct:
        cat, why = FACIAL_SIGNAL_QUESTION, ("the finger confirms the stimulus "
                                            "worked but the face did not "
                                            "follow: a facial signal / SNR / "
                                            "physiology question -- the pilot's "
                                            "actual open question")
    else:
        cat, why = BOTH_RESPONDED, ("both the finger control and the face "
                                    "changed with the stimulus")
    return {"category": cat, "explanation": why,
            "face_delta_pct": float(face_delta_pct),
            "finger": finger.to_dict()}


def sham_check(real_face_delta_pct: float, sham_face_delta_pct: float, *,
               ratio: float = 0.7) -> dict:
    """Sham vs real: if the sham response is a large fraction of the real one,
    the apparent effect is expectancy / apparatus / artifact, not the
    provocation (owner brief section 14)."""
    r, sh = abs(real_face_delta_pct), abs(sham_face_delta_pct)
    if r <= 1e-9:
        cat = EXPECTANCY_OR_ARTIFACT if sh > 1e-9 else UNRESOLVED
    elif sh >= ratio * r:
        cat = EXPECTANCY_OR_ARTIFACT
    else:
        cat = REAL_EXCEEDS_SHAM
    return {"category": cat, "real_delta_pct": float(real_face_delta_pct),
            "sham_delta_pct": float(sham_face_delta_pct), "ratio": ratio}


# --------------------------------------------------------------------------
# Event markers -> protocol windows
# --------------------------------------------------------------------------
def read_events(text_or_path: str) -> list:
    """Event markers `t_s,event`. Recognised events include immersion_in,
    immersion_out, movement, operator. Returns [(t_s, event), ...]."""
    text = _as_text(text_or_path)
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.lower().replace(" ", "").startswith("t_s,"):
            continue
        a, _, b = ln.partition(",")
        try:
            out.append((float(a), b.strip()))
        except ValueError:
            raise FprError(f"events: non-numeric time in row {ln!r}")
    return out


def windows_from_events(events: list, *, total_s: Optional[float] = None
                        ) -> dict:
    """Baseline / stimulus / recovery windows from immersion_in/out markers."""
    ein = next((t for t, e in events if e == "immersion_in"), None)
    eout = next((t for t, e in events if e == "immersion_out"), None)
    if ein is None or eout is None:
        raise FprError("events must contain immersion_in and immersion_out")
    end = total_s if total_s is not None else eout + (eout - ein)
    return {"baseline": (0.0, ein), "stimulus": (ein, eout),
            "recovery": (eout, end)}


def trial_windows(events: list, condition: str, *, baseline_s: float = 120.0,
                  recovery_s: float = 180.0) -> dict:
    """Windows for one NAMED provocation trial within a session.

    `condition` in {"null", "sham", "real"} matches `<condition>_in` /
    `<condition>_out` markers (the protocol's null -> sham -> real run in one
    session). Falls back to the generic `immersion_in`/`immersion_out` for a
    single-trial session (two-session sham-vs-real mode). Baseline is the
    `baseline_s` before immersion (protocol: 2 min); recovery the `recovery_s`
    after (protocol: 3 min).
    """
    ein = next((t for t, e in events
                if e in (f"{condition}_in", "immersion_in")), None)
    eout = next((t for t, e in events
                 if e in (f"{condition}_out", "immersion_out")), None)
    if ein is None or eout is None:
        raise FprError(f"events must contain {condition}_in/{condition}_out "
                       "(or immersion_in/out for a single-trial session)")
    return {"baseline": (max(0.0, ein - baseline_s), ein),
            "stimulus": (ein, eout),
            "recovery": (eout, eout + recovery_s)}


# --------------------------------------------------------------------------
# Continuous reference channels: HR, temperature, humidity
# --------------------------------------------------------------------------
def read_channels(text_or_path: str) -> tuple[np.ndarray, dict]:
    """Multi-column time series `t_s,<name>,<name>,...` -> (t, {name: values}).

    Handles the single-channel files (heart_rate.csv, humidity.csv) and the
    multi-channel temperature.csv (forehead/hand/ambient/water) uniformly.
    Fail-closed on a missing time header or ragged rows.
    """
    text = _as_text(text_or_path)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise FprError("channels: empty input")
    header = [c.strip().lower() for c in lines[0].split(",")]
    if header[0] not in ("t_s", "time", "t"):
        raise FprError("channels: first column must be a time column with a "
                       "header row 't_s,...' — fail closed on unknown layouts")
    names = header[1:]
    if not names:
        raise FprError("channels: no value columns")
    t: list = []
    cols: dict = {n: [] for n in names}
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) != len(header):
            raise FprError(f"channels: row {ln!r} has {len(parts)} columns, "
                           f"expected {len(header)}")
        try:
            ti = float(parts[0])
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            raise FprError(f"channels: non-numeric value in row {ln!r}")
        if t and ti <= t[-1]:
            raise FprError("channels: time must be strictly increasing")
        t.append(ti)
        for n, v in zip(names, vals):
            cols[n].append(v)
    if len(t) < 2:
        raise FprError("channels: fewer than 2 samples")
    return np.asarray(t, float), {n: np.asarray(v, float)
                                  for n, v in cols.items()}


def read_hr(text_or_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Continuous HR `t_s,hr_bpm` -> (t, hr). Picks an hr/bpm column, or the
    only value column if unnamed."""
    t, chans = read_channels(text_or_path)
    key = next((k for k in chans if "hr" in k or "bpm" in k or "heart" in k),
               None)
    if key is None:
        if len(chans) == 1:
            key = next(iter(chans))
        else:
            raise FprError("HR: no hr/bpm column found among "
                           f"{list(chans)}")
    return t, chans[key]


def read_temperature(text_or_path: str) -> tuple[np.ndarray, dict]:
    """Temperature `t_s,forehead_c,hand_c,ambient_c,water_c` -> (t, {chan})."""
    return read_channels(text_or_path)


def thermal_confound_check(face_res, temp_t, temp_vals, *,
                           channel: str = "forehead",
                           r_flag: float = THERMAL_R_FLAG) -> dict:
    """Does facial PI track skin temperature? (owner brief section 14).

    A high correlation means the apparent perfusion change may be a
    temperature-driven colour/emissivity artifact, not perfusion. Correlates
    the facial windowed PI against the temperature interpolated onto the PI
    window clock.
    """
    c = np.asarray(face_res.win_centers_s, float)
    v = np.asarray(face_res.win_pi, float)
    m = np.isfinite(v)
    tt = np.asarray(temp_t, float)
    tv = np.asarray(temp_vals, float)
    if int(m.sum()) < 4 or tt.size < 2:
        return {"category": UNRESOLVED, "r": float("nan"), "flagged": False,
                "channel": channel,
                "explanation": "insufficient data to test a thermal confound"}
    temp_on_pi = np.interp(c[m], tt, tv)
    pv = v[m]
    if np.std(temp_on_pi) < 1e-9 or np.std(pv) < 1e-12:
        r = 0.0
    else:
        r = float(np.corrcoef(temp_on_pi, pv)[0, 1])
    flagged = abs(r) >= r_flag
    if flagged:
        expl = (f"facial PI tracks the {channel} temperature (r={r:+.2f}): a "
                f"possible THERMAL CONFOUND -- a skin-temperature-driven "
                f"colour/emissivity change, not perfusion")
    else:
        expl = f"facial PI does not track {channel} temperature (r={r:+.2f})"
    return {"category": THERMAL_CONFOUND if flagged else NO_THERMAL_CONFOUND,
            "r": r, "flagged": flagged, "channel": channel,
            "explanation": expl}
