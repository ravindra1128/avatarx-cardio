"""FPR perfusion-index (PI) processing.

The experimental definition (owner brief section 11; report section 11):

    PI = AC amplitude / DC

computed the pulse-oximeter way. For each detected cardiac cycle:

    AC = peak-to-trough amplitude of the RAW green ROI-mean within the cycle
    DC = mean of the RAW green ROI-mean within the cycle
    PI = AC / DC                      (dimensionless; ~0.001-0.02 = 0.1-2%)

The DC term is essential and is why POS is NOT used upstream (POS divides
each channel by its temporal mean and so discards DC). This module consumes
the DC-bearing spatial-mean green trace from `extract.py` (or a synthetic
trace from `synth.py`).

WITHIN-SUBJECT CHANGE, NOT ABSOLUTE PI
--------------------------------------
The pilot's question is about a *repeatable within-subject change*, so the
reported quantities are relative to a baseline:

    dPI(t)      = (PI(t) / baseline_PI - 1) * 100      [percent change]
    sigma_base  = SD of dPI over the baseline window   [percent]
    sigma_null  = SD of dPI over a no-stimulus recording  [percent]  <-- primary

sigma_null is the primary pilot readout: the noise floor of the change we
are trying to detect, in the same units as the effect (Rasche reported a
~5.8% change). PASS <= 2%, hard stop > 4% (see __init__ thresholds).

Nothing here is hidden behind a model. Every quantity is a documented
arithmetic transform of the ROI-mean trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rppg._filters import bandpass, orient_peaks_up
from . import (PULSE_BAND_HZ, PI_WINDOW_S, PI_OVERLAP_S, BASELINE_TAIL_S,
               FprError)

try:
    from scipy.signal import find_peaks
    _HAVE_SCIPY = True
except ImportError:                                  # pragma: no cover
    _HAVE_SCIPY = False

# A physiologically plausible resting/active inter-beat interval, used both
# to seed peak spacing and to reject implausible cycles.
MIN_IBI_S = 0.30                                     # 200 bpm ceiling
MAX_IBI_S = 1.80                                     # 33 bpm floor
ENSEMBLE_LEN = 64                                    # samples per aligned beat
AMP_RATIO = 2.0                                      # reject a beat outside
                                                     # [1/2, 2]x its LOCAL
                                                     # median amplitude
AMP_LOCAL_HALF = 7                                   # +/-7 beats (~15-beat
                                                     # rolling neighbourhood)
TEMPLATE_CORR_MIN = 0.60                             # cycle vs ensemble corr


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class Cycle:
    i_start: int
    i_peak: int
    i_end: int
    t_peak: float
    ibi_s: float
    ac: float
    dc: float
    pi: float
    accepted: bool = True
    reject_reason: Optional[str] = None


@dataclass
class PiResult:
    fps: float
    n_frames: int
    duration_s: float
    cycles: list = field(default_factory=list)       # list[Cycle]
    n_detected: int = 0
    n_accepted: int = 0
    beat_yield: float = 0.0
    reject_fraction: float = 0.0
    snr_db: float = float("nan")
    median_hr_bpm: float = float("nan")
    # windowed PI
    win_centers_s: np.ndarray = field(default_factory=lambda: np.array([]))
    win_pi: np.ndarray = field(default_factory=lambda: np.array([]))
    # baseline / change
    baseline_pi: float = float("nan")
    sigma_baseline_pct: float = float("nan")
    delta_pi_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_null_pct: float = float("nan")
    ensemble: Optional[np.ndarray] = None
    ensemble_pi: float = float("nan")
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "fps": self.fps, "duration_s": self.duration_s,
            "n_detected": self.n_detected, "n_accepted": self.n_accepted,
            "beat_yield": self.beat_yield,
            "reject_fraction": self.reject_fraction,
            "snr_db": self.snr_db, "median_hr_bpm": self.median_hr_bpm,
            "baseline_pi": self.baseline_pi,
            "baseline_pi_pct": self.baseline_pi * 100.0,
            "sigma_baseline_pct": self.sigma_baseline_pct,
            "sigma_null_pct": self.sigma_null_pct,
            "ensemble_pi": self.ensemble_pi,
            "n_windows": int(self.win_pi.size),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# Signal steps
# --------------------------------------------------------------------------
def pulse_ac(green: np.ndarray, fps: float,
             band: tuple[float, float] = PULSE_BAND_HZ) -> np.ndarray:
    """Band-passed, peaks-up pulsatile component of the green ROI-mean.

    Used ONLY to locate beats and to measure SNR. AC amplitude for PI is
    taken from the RAW green within each cycle, never from this filtered
    trace, so the DC never touches a normalising filter.
    """
    g = np.asarray(green, float)
    ac = bandpass(g, fps, band[0], band[1])
    return orient_peaks_up(ac)


def detect_beats(ac: np.ndarray, fps: float) -> np.ndarray:
    """Systolic peak indices in the pulsatile trace."""
    ac = np.asarray(ac, float)
    if ac.size < int(2 * fps):
        return np.array([], dtype=int)
    z = ac - np.mean(ac)
    sd = float(np.std(z))
    if sd <= 1e-12:
        return np.array([], dtype=int)
    min_dist = max(int(MIN_IBI_S * fps), 1)
    if _HAVE_SCIPY:
        peaks, _ = find_peaks(z, distance=min_dist, prominence=0.3 * sd)
        return peaks.astype(int)
    # Fallback: local maxima above 0.3 SD with a refractory window.
    peaks = []
    last = -min_dist
    thr = 0.3 * sd
    for i in range(1, ac.size - 1):
        if z[i] >= thr and z[i] >= z[i - 1] and z[i] > z[i + 1] \
                and i - last >= min_dist:
            peaks.append(i)
            last = i
    return np.asarray(peaks, dtype=int)


def _trough_between(green: np.ndarray, a: int, b: int) -> int:
    """Index of the minimum of the raw green in [a, b] (the cycle foot)."""
    a = max(a, 0)
    b = min(b, green.size)
    if b <= a:
        return a
    return int(a + np.argmin(green[a:b]))


def segment_cycles(green: np.ndarray, peaks: np.ndarray, t: np.ndarray,
                   fps: float) -> list:
    """Foot-to-foot cycles around each detected systolic peak.

    AC = raw green (peak) - raw green (foot); DC = mean raw green over the
    cycle; PI = AC / DC.
    """
    green = np.asarray(green, float)
    cycles: list = []
    if peaks.size < 2:
        return cycles
    for k in range(peaks.size - 1):
        p0, p1 = int(peaks[k]), int(peaks[k + 1])
        foot0 = _trough_between(green, p0, p1)
        # previous foot: min between the prior peak and this peak
        prev_p = int(peaks[k - 1]) if k > 0 else max(p0 - int(fps), 0)
        foot_prev = _trough_between(green, prev_p, p0)
        seg = green[foot_prev:foot0 + 1]
        if seg.size < 3:
            continue
        dc = float(np.mean(seg))
        ac = float(np.max(seg) - np.min(seg))
        if dc <= 0 or not np.isfinite(dc) or not np.isfinite(ac):
            continue
        ibi = float(t[p1] - t[p0]) if p1 < t.size and p0 < t.size else \
            float((p1 - p0) / fps)
        cycles.append(Cycle(
            i_start=foot_prev, i_peak=p0, i_end=foot0,
            t_peak=float(t[p0]) if p0 < t.size else float(p0 / fps),
            ibi_s=ibi, ac=ac, dc=dc, pi=ac / dc))
    return cycles


def reject_cycles(cycles: list, green: np.ndarray) -> tuple[list, dict]:
    """Flag implausible cycles. Returns (cycles, stats).

    A cycle is rejected when its inter-beat interval is non-physiological,
    its amplitude is a gross outlier RELATIVE TO ITS LOCAL NEIGHBOURHOOD, or
    its shape correlates poorly with the ensemble template. Rejection is
    FLAGGED, never silently dropped: every cycle keeps its reason so the
    diagnostics can show what was removed.

    The amplitude test is deliberately LOCAL and scale-free (outside
    [1/2, 2]x a rolling median). A GLOBAL outlier filter would reject a
    genuine slow change in PI -- exactly the within-subject effect the pilot
    exists to detect -- as an "outlier"; that would be measuring our own
    filter, not perfusion.
    """
    if not cycles:
        return cycles, {"n_detected": 0, "n_accepted": 0,
                        "reject_fraction": 0.0}
    for c in cycles:
        if not (MIN_IBI_S <= c.ibi_s <= MAX_IBI_S):
            c.accepted, c.reject_reason = False, "implausible interval"
    pis = np.array([c.pi for c in cycles], float)
    for i, c in enumerate(cycles):
        if not c.accepted:
            continue
        lo = max(0, i - AMP_LOCAL_HALF)
        hi = min(len(cycles), i + AMP_LOCAL_HALF + 1)
        local = float(np.median(pis[lo:hi]))
        if local <= 0:
            continue
        ratio = c.pi / local
        if ratio > AMP_RATIO or ratio < 1.0 / AMP_RATIO:
            c.accepted, c.reject_reason = False, "amplitude outlier"
    # template correlation on the interval-plausible, amplitude-plausible set
    ens = ensemble_template([c for c in cycles if c.accepted], green)
    if ens is not None:
        for c in cycles:
            if not c.accepted:
                continue
            w = _resample_cycle(green, c, ENSEMBLE_LEN)
            if w is None:
                continue
            cc = float(np.corrcoef(w, ens)[0, 1]) if np.std(w) > 1e-12 else 0.0
            if cc < TEMPLATE_CORR_MIN:
                c.accepted, c.reject_reason = False, "poor cycle shape"
    n_det = len(cycles)
    n_acc = sum(c.accepted for c in cycles)
    return cycles, {"n_detected": n_det, "n_accepted": n_acc,
                    "reject_fraction": float((n_det - n_acc) / max(n_det, 1))}


def _resample_cycle(green: np.ndarray, c: Cycle, n: int) -> Optional[np.ndarray]:
    seg = np.asarray(green[c.i_start:c.i_end + 1], float)
    if seg.size < 3:
        return None
    xp = np.linspace(0.0, 1.0, seg.size)
    x = np.linspace(0.0, 1.0, n)
    w = np.interp(x, xp, seg)
    w = w - w.mean()
    return w


def ensemble_template(cycles: list, green: np.ndarray) -> Optional[np.ndarray]:
    """Mean aligned cycle (zero-mean), or None if too few beats."""
    segs = [_resample_cycle(green, c, ENSEMBLE_LEN) for c in cycles]
    segs = [s for s in segs if s is not None]
    if len(segs) < 3:
        return None
    return np.mean(np.vstack(segs), axis=0)


def pulse_snr_db(ac: np.ndarray, fps: float, hr_bpm: float,
                 band: tuple[float, float] = PULSE_BAND_HZ) -> float:
    """Ratio (dB) of spectral power in a narrow band around the pulse
    fundamental (+/- first harmonic) to the rest of the physiological band.
    """
    ac = np.asarray(ac, float)
    if ac.size < int(4 * fps) or not np.isfinite(hr_bpm) or hr_bpm <= 0:
        return float("nan")
    x = ac - ac.mean()
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / fps)
    f0 = hr_bpm / 60.0
    tol = 0.15                                       # +/-0.15 Hz around lines
    sig_mask = np.zeros_like(f, bool)
    for h in (1, 2):
        sig_mask |= np.abs(f - h * f0) <= tol
    band_mask = (f >= band[0]) & (f <= band[1])
    noise_mask = band_mask & ~sig_mask
    sig = float(spec[sig_mask].sum())
    noise = float(spec[noise_mask].sum())
    if noise <= 0 or sig <= 0:
        return float("nan")
    return float(10.0 * np.log10(sig / noise))


def windowed_pi(cycles: list, duration_s: float,
                window_s: float = PI_WINDOW_S,
                overlap_s: float = PI_OVERLAP_S) -> tuple[np.ndarray, np.ndarray]:
    """Median accepted-beat PI in each sliding window.

    Windows with fewer than 3 accepted beats yield NaN (kept in place so the
    time base stays uniform).
    """
    step = max(window_s - overlap_s, 1e-3)
    accepted = [c for c in cycles if c.accepted]
    if duration_s <= 0:
        return np.array([]), np.array([])
    starts = np.arange(0.0, max(duration_s - window_s, 0.0) + 1e-9, step)
    if starts.size == 0:
        starts = np.array([0.0])
    centers, vals = [], []
    for s in starts:
        e = s + window_s
        pis = [c.pi for c in accepted if s <= c.t_peak < e]
        centers.append(s + window_s / 2.0)
        vals.append(float(np.median(pis)) if len(pis) >= 3 else np.nan)
    return np.asarray(centers), np.asarray(vals)


def pi_change(win_centers: np.ndarray, win_pi: np.ndarray,
              baseline_window: tuple, stimulus_window: tuple
              ) -> tuple[float, float, float]:
    """(baseline_pi, stimulus_pi, delta_pct) between two time windows of a
    windowed-PI series. delta_pct = (stimulus/baseline - 1)*100. Shared by the
    facial and finger channels so a trial is measured the same way on both.
    """
    c = np.asarray(win_centers, float)
    v = np.asarray(win_pi, float)

    def _med(w):
        m = np.isfinite(v) & (c >= w[0]) & (c < w[1])
        return float(np.median(v[m])) if m.any() else float("nan")

    base = _med(baseline_window)
    stim = _med(stimulus_window)
    if not (np.isfinite(base) and base > 0 and np.isfinite(stim)):
        return base, stim, float("nan")
    return base, stim, (stim / base - 1.0) * 100.0


def baseline_and_change(win_centers: np.ndarray, win_pi: np.ndarray,
                        baseline_end_s: Optional[float],
                        tail_s: float = BASELINE_TAIL_S
                        ) -> tuple[float, float, np.ndarray, float]:
    """(baseline_pi, sigma_baseline_pct, delta_pi_pct, sigma_null_pct).

    baseline_pi = mean window-PI over the final `tail_s` of the baseline
    period (the whole recording when `baseline_end_s` is None, i.e. a null
    recording). sigma_* are the SD of the percent-change series -- the
    coefficient of variation of PI expressed as a percent.
    """
    finite = np.isfinite(win_pi)
    if not finite.any():
        return float("nan"), float("nan"), np.array([]), float("nan")
    c = np.asarray(win_centers, float)
    v = np.asarray(win_pi, float)
    if baseline_end_s is not None:
        # a baseline WINDOW must lie fully inside the baseline period, else it
        # straddles the stimulus onset and mixes the two levels into the
        # reference (the bias that makes a real dPI read low).
        b_end = float(baseline_end_s)
        b_hi = b_end - PI_WINDOW_S / 2.0
    else:
        b_end = float(c[finite].max())
        b_hi = b_end
    b_mask = finite & (c <= b_hi) & (c >= b_end - tail_s)
    if not b_mask.any():                             # fall back to all baseline
        b_mask = finite & (c <= b_end)
    baseline = float(np.mean(v[b_mask]))
    if not np.isfinite(baseline) or baseline <= 0:
        return baseline, float("nan"), np.array([]), float("nan")
    delta = (v / baseline - 1.0) * 100.0
    sigma_base = float(np.nanstd(delta[b_mask]))
    # sigma_null: SD of the percent-change over the WHOLE finite series
    sigma_null = float(np.nanstd(delta[finite]))
    return baseline, sigma_base, delta, sigma_null


# --------------------------------------------------------------------------
# Top-level driver
# --------------------------------------------------------------------------
def process_pi(green: np.ndarray, t: np.ndarray, fps: float, *,
               baseline_end_s: Optional[float] = None) -> PiResult:
    """Full PI pipeline over one green ROI-mean trace.

    `green` : raw spatially-averaged green ROI-mean per frame (DC-bearing).
    `t`     : per-frame timestamps in seconds (honest clock).
    `baseline_end_s` : end of the baseline period; None for a null recording
                       (baseline = whole recording).
    """
    green = np.asarray(green, float)
    t = np.asarray(t, float)
    if green.ndim != 1 or t.shape != green.shape:
        raise FprError("green and t must be 1-D arrays of equal length")
    if green.size < int(4 * fps):
        raise FprError("trace too short for PI (need >= 4 s)")
    duration = float(t[-1] - t[0]) if t.size else 0.0

    ac = pulse_ac(green, fps)
    peaks = detect_beats(ac, fps)
    cycles = segment_cycles(green, peaks, t, fps)
    cycles, stats = reject_cycles(cycles, green)
    accepted = [c for c in cycles if c.accepted]

    ibis = np.array([c.ibi_s for c in accepted], float)
    median_hr = float(60.0 / np.median(ibis)) if ibis.size else float("nan")
    # detection yield: fraction of the recording covered by accepted beats
    covered = float(np.sum(ibis)) if ibis.size else 0.0
    yield_ = float(min(covered / duration, 1.0)) if duration > 0 else 0.0

    ens = ensemble_template(accepted, green)
    ens_pi = float("nan")
    if ens is not None:
        dc_ref = float(np.mean([c.dc for c in accepted]))
        ens_pi = float((ens.max() - ens.min()) / dc_ref) if dc_ref > 0 else \
            float("nan")

    snr = pulse_snr_db(ac, fps, median_hr)
    win_c, win_pi = windowed_pi(cycles, duration)
    baseline, sig_base, delta, sig_null = baseline_and_change(
        win_c, win_pi, baseline_end_s)

    res = PiResult(
        fps=fps, n_frames=int(green.size), duration_s=duration,
        cycles=cycles, n_detected=stats["n_detected"],
        n_accepted=stats["n_accepted"], beat_yield=yield_,
        reject_fraction=stats["reject_fraction"], snr_db=snr,
        median_hr_bpm=median_hr, win_centers_s=win_c, win_pi=win_pi,
        baseline_pi=baseline, sigma_baseline_pct=sig_base,
        delta_pi_pct=delta, sigma_null_pct=sig_null,
        ensemble=ens, ensemble_pi=ens_pi)
    return res
