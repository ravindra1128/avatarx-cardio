"""
Beat detection with per-beat confidence and multi-ROI consensus.

The contract this module exists to satisfy: DO NOT RETURN A BPM NUMBER.
Return a list of beats, each carrying enough information for the downstream
rhythm engine to decide how much to trust it.

The physiological reason confidence is mandatory rather than nice-to-have:
in AF a short RR interval gives insufficient diastolic filling, the stroke
volume is too small to produce a detectable peripheral pulse, and the beat
vanishes from the optical signal. This is pulse deficit. It is not an
algorithm defect and it cannot be engineered away -- but the system must
distinguish "no beat occurred" from "a beat occurred and I could not see it",
because the two have opposite implications for an irregularity statistic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional
import numpy as np


@dataclass
class Beat:
    t_s: float                       # beat time on the video clock
    confidence: float                # 0-1, calibrated
    roi_agreement: float             # 0-1, fraction of ROIs that saw this beat
    signal_quality: float            # 0-1, local SQI at this beat
    amplitude: float                 # relative pulse amplitude (arbitrary units)
    prominence: float
    source_rois: list[str] = field(default_factory=list)
    interpolated: bool = False       # inserted to bridge a suspected missed beat


@dataclass
class BeatSeries:
    beats: list[Beat]
    fps: float
    duration_s: float

    def times(self) -> np.ndarray:
        return np.array([b.t_s for b in self.beats])

    def confidences(self) -> np.ndarray:
        return np.array([b.confidence for b in self.beats])

    def ibi_ms(self) -> np.ndarray:
        return np.diff(self.times()) * 1000.0

    def ibi_confidence(self) -> np.ndarray:
        """An interval is only as trustworthy as its weaker endpoint."""
        c = self.confidences()
        return np.minimum(c[:-1], c[1:]) if c.size > 1 else np.array([])

    def usable_beats(self, threshold: float = 0.5) -> int:
        return int(np.sum(self.confidences() >= threshold))

    def coverage(self) -> float:
        """Fraction of the recording spanned by confidently-detected beats."""
        if len(self.beats) < 2 or self.duration_s <= 0:
            return 0.0
        return float((self.times()[-1] - self.times()[0]) / self.duration_s)


# A genuine systolic peak is broad: half-prominence width stays >= ~100 ms
# even at the 250 ms RR floor. Peaks narrower than this are progressively
# discounted as noise spikes. 80 ms leaves margin below the physiologic floor.
MIN_PLAUSIBLE_PULSE_WIDTH_MS = 80.0


# --------------------------------------------------------------------------
def _refractory_ms(recent_ibi_ms: Optional[np.ndarray]) -> float:
    """Dynamic refractory period.

    A fixed refractory period is wrong for AF: it must be short enough to
    admit genuine short-coupled beats (down to ~250 ms in rapid AF) but long
    enough to reject dicrotic notches. Adapt to the recent rhythm, floor at
    240 ms (250 bpm ceiling), and widen when intervals are long.
    """
    if recent_ibi_ms is None or recent_ibi_ms.size == 0:
        return 300.0
    med = float(np.median(recent_ibi_ms))
    return float(np.clip(0.45 * med, 240.0, 500.0))


def detect_beats_single_roi(signal: np.ndarray, fps: float, roi: str = "roi",
                            sqi_series: Optional[np.ndarray] = None) -> list[Beat]:
    """Adaptive peak detection on one ROI's pulse waveform.

    Deliberately simple and inspectable: an adaptive-threshold local-maximum
    detector with a dynamic refractory period and parabolic sub-sample peak
    interpolation. It is the CORRECTNESS FLOOR and the regression baseline
    against which any learned detector must prove itself, not the final
    detector.

    Sub-sample interpolation is not an optimisation -- at 30 fps, naive
    peak-picking quantises each peak with SD = (1000/30)/sqrt(12) = 9.6 ms,
    which propagates to 23.6 ms on successive differences and inflates a true
    30 ms sinus RMSSD by ~27%. That inflation pushes sinus rhythm toward the
    AF decision boundary and manufactures false positives.
    """
    x = np.asarray(signal, float)
    if x.size < int(2 * fps):
        return []
    x = (x - np.mean(x)) / (np.std(x) + 1e-9)

    beats: list[Beat] = []
    recent: list[float] = []
    i, n = 1, x.size - 1
    last_t = -np.inf
    last_amp = -np.inf

    # Adaptive amplitude threshold over a trailing window.
    win = max(int(3 * fps), 30)
    while i < n:
        lo = max(0, i - win)
        local = x[lo:i + 1]
        thr = np.median(local) + 0.5 * (np.percentile(local, 90) - np.median(local)) \
            if local.size > 5 else 0.0

        if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > thr:
            t = i / fps
            refr = _refractory_ms(np.array(recent[-8:]) if recent else None) / 1000.0
            # v0.1.2 (audit-confirmed defect): a supra-threshold pre-peak
            # (dicrotic remnant / noise) that arrives FIRST used to win, and
            # the refractory test then suppressed the true, higher systolic
            # peak 100-400 ms later — mis-timing the beat and, on weak
            # waveforms, seeding half/double intervals. Inside the
            # refractory window a HIGHER local maximum now REPLACES the
            # earlier one (standard keep-the-max refractory), so the beat
            # is placed on the systolic peak. Measured on real webcam
            # waveforms: strong-ROI IBI IQR 199 -> 123 ms.
            y0, y1, y2 = x[i - 1], x[i], x[i + 1]
            left = x[max(0, i - int(0.2 * fps)):i]
            right = x[i + 1:i + 1 + int(0.2 * fps)]
            prom = float(y1 - max(left.min() if left.size else y1,
                                  right.min() if right.size else y1))
            # Replace only a low-PROMINENCE predecessor (a shoulder on the
            # upstroke), never a genuine small beat: AF's short-coupled beats
            # are small in amplitude but still prominent, and the permanent
            # anti-periodicity test forbids penalising them.
            replace = bool((t - last_t < refr) and beats and (x[i] > last_amp)
                           and prom > 1.5 * max(beats[-1].prominence, 1e-6))
            if t - last_t >= refr or replace:
                # Parabolic sub-sample refinement.
                denom = (y0 - 2 * y1 + y2)
                delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                delta = float(np.clip(delta, -0.5, 0.5))
                t_ref = (i + delta) / fps

                q = 1.0
                if sqi_series is not None and sqi_series.size == x.size:
                    q = float(np.clip(sqi_series[i], 0.0, 1.0))
                # v0.1 measured extension (T2): peak WIDTH at half prominence.
                # Threshold-crossing noise spikes are 1-2 frames wide (median
                # 17 ms measured); genuine pulse peaks are ~100-150 ms even
                # for weak short-RR AF beats (p10 67 ms) — so width separates
                # noise from pulse WITHOUT penalising pulse deficit the way an
                # amplitude prior would. Folded into the local quality term.
                level = y1 - 0.5 * max(prom, 1e-6)
                l_ = i
                while l_ > 0 and x[l_ - 1] > level:
                    l_ -= 1
                r_ = i
                while r_ < x.size - 1 and x[r_ + 1] > level:
                    r_ += 1
                width_ms = (r_ - l_ + 1) / fps * 1000.0
                q *= float(np.clip(width_ms / MIN_PLAUSIBLE_PULSE_WIDTH_MS,
                                   0.0, 1.0))

                nb = Beat(t_s=t_ref, confidence=0.0, roi_agreement=0.0,
                          signal_quality=q, amplitude=float(y1),
                          prominence=prom, source_rois=[roi])
                if replace:
                    prev_t = beats[-1].t_s
                    beats[-1] = nb
                    if recent:                       # stretch the last IBI
                        recent[-1] += (t_ref - prev_t) * 1000.0
                else:
                    beats.append(nb)
                    if last_t > -np.inf:
                        recent.append((t_ref - last_t) * 1000.0)
                last_t = t_ref
                last_amp = float(y1)
        i += 1
    return beats


def fuse_multi_roi(per_roi: dict[str, list[Beat]], fps: float, duration_s: float,
                   tolerance_ms: float = 60.0,
                   min_rois: int = 2) -> BeatSeries:
    """Consensus fusion across facial ROIs.

    Premise: a true cardiac pulse appears across forehead, cheeks and nose
    with small, physiologically-bounded phase offsets. A motion or
    illumination artifact generally does not -- it is spatially local, or it
    is global but with the wrong temporal signature.

    Honest caveat carried into the confidence value rather than the docstring:
    ROI signals are NOT statistically independent. They share the illuminant,
    the face tracker and the camera's auto-exposure. Cross-ROI agreement is
    therefore a strong ARTIFACT VETO and a weaker BEAT CONFIRMER, and the
    weighting below reflects that asymmetry -- agreement raises confidence
    sub-linearly, disagreement lowers it steeply.
    """
    rois = sorted(per_roi)
    if not rois:
        return BeatSeries([], fps, duration_s)

    events: list[tuple[float, str, Beat]] = [
        (b.t_s, r, b) for r in rois for b in per_roi[r]]
    events.sort(key=lambda e: e[0])

    tol = tolerance_ms / 1000.0

    # Two-pass clustering. A single greedy pass that compares each event to the
    # FIRST member of the open cluster fragments a genuine beat into two
    # clusters whenever ROI detections straddle the tolerance width -- which
    # produces MORE fused beats than any single ROI reported, and inflates
    # every downstream irregularity statistic. Pass 1 grows clusters against a
    # running centroid; pass 2 merges neighbours that ended up within tolerance.
    clusters: list[list[tuple[float, str, Beat]]] = []
    centroid: float | None = None
    for e in events:
        if clusters and centroid is not None and abs(e[0] - centroid) <= tol:
            clusters[-1].append(e)
            centroid = float(np.mean([x[0] for x in clusters[-1]]))
        else:
            clusters.append([e])
            centroid = e[0]

    merged: list[list[tuple[float, str, Beat]]] = []
    for cl in clusters:
        if merged:
            c_prev = float(np.mean([x[0] for x in merged[-1]]))
            c_cur = float(np.mean([x[0] for x in cl]))
            prev_rois = {r for _, r, _ in merged[-1]}
            cur_rois = {r for _, r, _ in cl}
            # Merge only when the two clusters are close AND describe different
            # ROIs -- two detections from the SAME ROI that close together are
            # a double-detection, not one beat seen twice.
            if abs(c_cur - c_prev) <= tol and not (prev_rois & cur_rois):
                merged[-1].extend(cl)
                continue
        merged.append(cl)
    clusters = merged

    fused: list[Beat] = []
    n_rois = len(rois)
    for cl in clusters:
        seen = {r for _, r, _ in cl}
        if len(seen) < min_rois and n_rois >= min_rois:
            continue                                  # rejected as artifact
        # Quality-weighted time estimate.
        w = np.array([b.signal_quality * max(b.prominence, 1e-6) for _, _, b in cl])
        t = np.array([e[0] for e in cl])
        t_fused = float(np.sum(w * t) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(t))

        agree = len(seen) / n_rois
        mad_ms = float(np.mean(np.abs(t - np.mean(t))) * 1000.0)
        # v0.1 measured correction (T2): v1 used the cluster time RANGE with
        # a sqrt agreement reward. Range grows with cluster SIZE, so
        # full-agreement real beats were penalised harder than the tight
        # 2-ROI chance coincidences that dominate the false-beat population —
        # measured ranking AUC 0.32-0.45 (INVERTED) on the controlled-capture
        # synthetic suite, which no monotone calibration can repair.
        # Mean-absolute-deviation is size-normalised, and the 1.5 agreement
        # exponent makes membership dominate: AUC 0.90 sinus / 0.96 AF at the
        # same conditions (regression: tests/test_confidence.py).
        conf = (agree ** 1.5) * float(np.exp(-mad_ms / tolerance_ms))
        conf *= float(np.mean([b.signal_quality for _, _, b in cl]))

        fused.append(Beat(
            t_s=t_fused, confidence=float(np.clip(conf, 0.0, 1.0)),
            roi_agreement=agree,
            signal_quality=float(np.mean([b.signal_quality for _, _, b in cl])),
            amplitude=float(np.mean([b.amplitude for _, _, b in cl])),
            prominence=float(np.mean([b.prominence for _, _, b in cl])),
            source_rois=sorted(seen)))

    fused.sort(key=lambda b: b.t_s)
    return BeatSeries(fused, fps, duration_s)


def flag_suspected_missed_beats(series: BeatSeries,
                                ratio: float = 1.75) -> list[int]:
    """Indices of intervals that are suspiciously long relative to neighbours.

    Returns INDICES ONLY -- it deliberately does not insert beats. In AF a
    long interval is frequently real, and silently interpolating one would
    erase the very irregularity being measured. The rhythm engine decides
    what to do; this function only raises its hand.
    """
    ibi = series.ibi_ms()
    if ibi.size < 4:
        return []
    out = []
    for k in range(ibi.size):
        lo = max(0, k - 4); hi = min(ibi.size, k + 5)
        neigh = np.delete(ibi[lo:hi], min(k - lo, hi - lo - 1))
        if neigh.size and ibi[k] > ratio * float(np.median(neigh)):
            out.append(k)
    return out
