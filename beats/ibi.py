"""
Clean-run interval extraction — the production feature path.

WHY THIS MODULE IS LOAD-BEARING (pressure-test findings P1/P2/P6)
-----------------------------------------------------------------
Interval-dispersion statistics are catastrophically fragile to beat-detection
errors: on a synthetic 100-interval sinus series (RMSSD 27 ms), ONE undetected
missed beat drives RMSSD to 125 ms — past the AF boundary — and ONE false beat
to 65 ms. At exactly Gate-1-limit detection quality (10% missed + 10% false),
raw-series sinus RMSSD medians ~430 ms: every sinus subject would read as AF.

The evaluation harness never saw this because it computes RMSSD on ECG-MATCHED
consecutive pairs, which silently excludes the very errors that corrupt the
production path. The harness and the product were measuring different
quantities — a gate could pass while the product failed.

The fix has three parts:
1.  Features are computed on CLEAN RUNS: maximal stretches of consecutive
    beats whose confidence clears a threshold. Successive-difference
    statistics (RMSSD, pNN, Poincaré SD1) NEVER cross a run boundary — a run
    break is exactly where a missed/false beat is suspected.
2.  Bounded statistics (pNN50/pNN20, median-based indices) lead the feature
    hierarchy; unbounded ones (RMSSD, SDNN) are computed run-wise only.
3.  A new gate metric ties the two paths: production-path RMSSD (this module)
    versus ECG RMSSD must agree, on the same recordings the beat harness
    scores. See evaluation.beat_metrics.production_rmssd_error_ms.

The run structure is also physiologically meaningful: the DROPOUT RATE under
good local signal quality is the optical signature of pulse deficit, which is
itself AF evidence (clinicians palpate exactly this). It is exposed here as a
feature input rather than discarded as noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .detector import BeatSeries


@dataclass
class RunSet:
    """Intervals grouped into confidence-clean runs.

    `runs` holds interval arrays in ms; intervals within a run come from
    consecutive beats that all cleared `min_conf`. Nothing is ever computed
    across a run boundary.
    """
    runs: list[np.ndarray] = field(default_factory=list)
    run_confidences: list[np.ndarray] = field(default_factory=list)
    run_amplitudes: list[np.ndarray] = field(default_factory=list)
    # v0.6: the END-BEAT time (s) of every interval in every run, so an
    # interval can be placed on the clock. Needed to test the interval
    # series against a time-varying covariate (respiration): an interval
    # without its time cannot be correlated with anything. Parallel to
    # `runs` in shape; nothing about run construction changes.
    run_times: list[np.ndarray] = field(default_factory=list)
    total_beats: int = 0
    kept_beats: int = 0
    min_conf: float = 0.5
    # v0.1.2 detection-error accounting (spec T5 short-pair splitter)
    n_missed_splits: int = 0          # run breaks at suspected MISSED beats
    n_false_pair_splits: int = 0      # false beats removed by the pair rule
    n_candidate_intervals: int = 0    # intervals examined by the splitters

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def split_fraction(self) -> float:
        """Fraction of examined intervals broken by the missed/false-beat
        splitters — the DETECTION-ERROR burden of the recording. Genuine AF
        keeps this low (its irregularity is distributed, not harmonic);
        a weak signal with double/missed detections drives it up."""
        n = self.n_candidate_intervals
        return ((self.n_missed_splits + 2 * self.n_false_pair_splits) / n
                if n else 0.0)

    @property
    def n_intervals(self) -> int:
        return int(sum(r.size for r in self.runs))

    @property
    def longest_run(self) -> int:
        return int(max((r.size for r in self.runs), default=0))

    @property
    def dropout_rate(self) -> float:
        """Fraction of detected beats excluded by the confidence filter.

        Under good global SQI, a high dropout rate is evidence of pulse
        deficit (short-RR beats too weakly perfused to see) — an AF feature,
        not merely a quality problem.
        """
        return 1.0 - self.kept_beats / self.total_beats if self.total_beats else 0.0

    def all_intervals(self) -> np.ndarray:
        return np.concatenate(self.runs) if self.runs else np.array([])

    def within_run_diffs(self) -> np.ndarray:
        """Successive differences that never span a run boundary."""
        ds = [np.diff(r) for r in self.runs if r.size >= 2]
        return np.concatenate(ds) if ds else np.array([])

    def within_run_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """(x_i, x_{i+1}) pairs within runs — for Poincaré / lag-1 statistics."""
        a = [r[:-1] for r in self.runs if r.size >= 2]
        b = [r[1:] for r in self.runs if r.size >= 2]
        if not a:
            return np.array([]), np.array([])
        return np.concatenate(a), np.concatenate(b)


def _suspicious_interval_indices(ibi_ms: np.ndarray, ratio: float = 1.75,
                                 window: int = 4) -> list[int]:
    """Indices of intervals suspiciously LONG relative to their local median —
    the signature of a missed beat between two confident detections.

    Confidence filtering is structurally blind to this error: both endpoint
    beats are real and confident; it is the beat BETWEEN them that vanished.
    (Pressure test: one such merge multiplies sinus RMSSD 4.6x.)
    """
    out = []
    n = ibi_ms.size
    for k in range(n):
        lo, hi = max(0, k - window), min(n, k + window + 1)
        neigh = np.delete(ibi_ms[lo:hi], k - lo)
        if neigh.size >= 3 and ibi_ms[k] > ratio * float(np.median(neigh)):
            out.append(k)
    return out


def _false_pair_indices(ibi_ms: np.ndarray, short_ratio: float = 0.75,
                        sum_tol: float = 0.20, window: int = 4) -> list[int]:
    """Indices k such that intervals k and k+1 are BOTH short and SUM to the
    local median — the split signature of a FALSE beat between two real ones
    (spec T5). Neither half is repaired: the run is broken on both sides so
    the false beat and both sub-intervals leave the feature path.

    In genuine AF two consecutive short intervals summing to the local median
    are rare (measured ~1-3% of intervals on AF-like series); in a weak
    signal with double detections they are the dominant error mode. This is
    the interval-domain complement of the confidence channel, which the
    real-recording false positive showed cannot be relied on alone.
    """
    out = []
    n = ibi_ms.size
    for k in range(n - 1):
        lo, hi = max(0, k - window), min(n, k + 2 + window)
        neigh = np.delete(ibi_ms[lo:hi], [k - lo, k + 1 - lo])
        if neigh.size < 3:
            continue
        med = float(np.median(neigh))
        a, b = float(ibi_ms[k]), float(ibi_ms[k + 1])
        if a < short_ratio * med and b < short_ratio * med and \
                abs((a + b) - med) <= sum_tol * med:
            out.append(k)
    return out


def clean_runs(series: BeatSeries, min_conf: float = 0.5,
               min_run_beats: int = 4,
               max_physiologic_ibi_ms: float = 2200.0,
               min_physiologic_ibi_ms: float = 250.0,
               missed_beat_ratio: float = 1.75) -> RunSet:
    """Split a beat series into clean runs along FOUR error channels.

    4. SHORT-PAIR (v0.1.2, spec T5): two adjacent short intervals that sum
       to the local median mark a FALSE beat between real ones; both
       sub-intervals are excluded (run broken on both sides). No repair.

    1. CONFIDENCE: a beat below `min_conf` breaks the run — catches FALSE
       beats, which the fusion stage scores low.
    2. PHYSIOLOGIC RANGE: an interval outside [250, 2200] ms breaks the run —
       catches gross residual merges/splits.
    3. LOCAL-RATIO: an interval > `missed_beat_ratio` x its local median
       breaks the run — catches a MISSED beat between two confident
       detections, which channels 1-2 cannot see (both endpoints are real).

    Channel 3 is a deliberate, bounded trade against AF sensitivity: in AF a
    locally-extreme long interval is sometimes genuine, and breaking there
    discards that one difference. The direction of the error is conservative
    (measured irregularity can only be reduced, never manufactured), which is
    correct because specificity is the binding constraint at screening
    prevalence. This is NOT the `open-rppg clean_rr` failure mode — that
    filter deleted outliers against a global band and collapsed AF RMSSD from
    230 to 66 ms; this one only refuses to bridge a single locally-anomalous
    interval, and the regression suite asserts AF RMSSD survives it.

    Deliberately NO interpolation and NO outlier re-insertion: in AF a long
    interval is frequently real, and repairing it would erase the signal
    being measured.
    """
    beats = series.beats
    out = RunSet(min_conf=min_conf, total_beats=len(beats))
    if len(beats) < 2:
        return out

    # Pass 1: confidence + physiologic-range segmentation into candidate runs.
    candidates: list[list] = []
    cur: list = []
    prev = None
    for b in beats:
        ok = b.confidence >= min_conf
        if ok and prev is not None and cur:
            gap = (b.t_s - prev.t_s) * 1000.0
            if not (min_physiologic_ibi_ms <= gap <= max_physiologic_ibi_ms):
                candidates.append(cur); cur = []
        if ok:
            cur.append(b)
        else:
            if cur:
                candidates.append(cur); cur = []
        prev = b if ok else None
    if cur:
        candidates.append(cur)

    # Pass 2: split each candidate at suspected-missed-beat intervals.
    def emit(chunk: list) -> None:
        if len(chunk) < min_run_beats:
            return
        t = np.array([b.t_s for b in chunk])
        ibi = np.diff(t) * 1000.0
        out.runs.append(ibi)
        out.run_times.append(t[1:])                # each interval's end beat
        c = np.array([b.confidence for b in chunk])
        out.run_confidences.append(np.minimum(c[:-1], c[1:]))
        out.run_amplitudes.append(np.array([b.amplitude for b in chunk]))
        out.kept_beats += len(chunk)

    for cand in candidates:
        if len(cand) < 2:
            continue
        ibi = np.asarray(np.diff([b.t_s for b in cand]) * 1000.0)
        out.n_candidate_intervals += int(ibi.size)
        missed = set(_suspicious_interval_indices(ibi, ratio=missed_beat_ratio))
        pairs = _false_pair_indices(ibi)
        out.n_missed_splits += len(missed)
        out.n_false_pair_splits += len(pairs)
        cuts = set(missed)
        for k in pairs:                            # exclude BOTH halves
            cuts.add(k); cuts.add(k + 1)
        chunk: list = [cand[0]]
        for k in range(1, len(cand)):
            if (k - 1) in cuts:                    # interval k-1 is suspicious:
                emit(chunk)                        # end run BEFORE bridging it
                chunk = [cand[k]]
            else:
                chunk.append(cand[k])
        emit(chunk)
    return out


def rmssd_from_runs(rs: RunSet) -> float:
    d = rs.within_run_diffs()
    return float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")


def sdnn_from_runs(rs: RunSet) -> float:
    x = rs.all_intervals()
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
