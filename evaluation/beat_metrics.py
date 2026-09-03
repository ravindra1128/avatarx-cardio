"""
Beat-level validation of remote pulse extraction against ECG R-peaks.

WHY THIS MODULE EXISTS
----------------------
Prior AvatarX research established that no published rPPG paper reports
inter-beat-interval error in milliseconds or peak-detection F1 against ECG
R-peaks, and that the field's reference benchmark (rPPG-Toolbox) cannot
measure it -- its "peak detection" mode computes
    hr = 60 / (mean(diff(peaks)) / fs)
which averages away every beat-to-beat interval. A pipeline can therefore
score excellently on every published rPPG metric while being unusable for
rhythm analysis.

This module implements the missing measurement. It is Gate 1 of the AvatarX
programme: if beat timing is unreliable, AFib classification is not worth
optimising, and the honest response is to fix extraction or stop.

DESIGN NOTES
------------
* Matching is one-to-one and greedy-by-proximity within a tolerance window.
  Naive nearest-neighbour matching double-counts and flatters recall.
* Pulse arrival at the face LAGS the ECG R-peak by the pulse transit time
  (~150-300 ms). PTT is estimated and removed before matching, otherwise
  every beat is "missed". PTT is not constant -- in AF it varies with
  diastolic filling -- so it is estimated robustly (median) and its
  variability is reported as a diagnostic rather than assumed away.
* Metrics are reported SEPARATELY for AF and non-AF segments. Aggregate beat
  metrics hide the failure that matters: pulse deficit removes short-RR beats
  precisely during AF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np


# --------------------------------------------------------------------------
@dataclass
class BeatMatchResult:
    n_reference: int
    n_detected: int
    n_matched: int
    tolerance_ms: float
    ptt_estimate_ms: float
    ptt_iqr_ms: float
    timing_errors_ms: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    matched_ref_idx: np.ndarray = field(repr=False, default_factory=lambda: np.array([], int))
    matched_det_idx: np.ndarray = field(repr=False, default_factory=lambda: np.array([], int))

    # -------- detection quality
    @property
    def sensitivity(self) -> float:
        """Fraction of true beats recovered (a.k.a. recall)."""
        return self.n_matched / self.n_reference if self.n_reference else float("nan")

    @property
    def ppv(self) -> float:
        """Fraction of detected beats that are real (a.k.a. precision)."""
        return self.n_matched / self.n_detected if self.n_detected else float("nan")

    @property
    def f1(self) -> float:
        se, pv = self.sensitivity, self.ppv
        return 2 * se * pv / (se + pv) if (se + pv) else float("nan")

    @property
    def missed_beat_rate(self) -> float:
        return 1.0 - self.sensitivity if self.n_reference else float("nan")

    @property
    def false_beat_rate(self) -> float:
        return 1.0 - self.ppv if self.n_detected else float("nan")

    # -------- timing quality
    @property
    def timing_mae_ms(self) -> float:
        return float(np.mean(np.abs(self.timing_errors_ms))) if self.timing_errors_ms.size else float("nan")

    @property
    def timing_rmse_ms(self) -> float:
        return float(np.sqrt(np.mean(self.timing_errors_ms ** 2))) if self.timing_errors_ms.size else float("nan")

    @property
    def timing_bias_ms(self) -> float:
        return float(np.mean(self.timing_errors_ms)) if self.timing_errors_ms.size else float("nan")

    def timing_percentile_ms(self, q: float) -> float:
        return float(np.percentile(np.abs(self.timing_errors_ms), q)) if self.timing_errors_ms.size else float("nan")


@dataclass
class IBIResult:
    """Interval-level agreement, computed only on CONSECUTIVE matched pairs.

    Intervals spanning a missed beat are excluded from MAE and counted
    separately -- folding them in produces a bimodal error distribution whose
    mean describes nothing.
    """
    n_intervals: int
    n_excluded_spanning_gap: int
    ibi_mae_ms: float
    ibi_rmse_ms: float
    ibi_bias_ms: float
    loa_lower_ms: float
    loa_upper_ms: float
    pct_within_25ms: float
    pct_within_50ms: float
    pct_within_100ms: float
    ref_rmssd_ms: float
    est_rmssd_ms: float
    rmssd_error_ms: float
    ref_sdnn_ms: float
    est_sdnn_ms: float
    sdnn_error_ms: float


# --------------------------------------------------------------------------
def estimate_ptt_ms(reference_s: Sequence[float],
                    detected_s: Sequence[float],
                    search_ms: tuple[float, float] = (0.0, 500.0)) -> tuple[float, float]:
    """Robustly estimate pulse transit time by cross-correlating beat trains.

    Returns (median_ptt_ms, iqr_ms). The IQR is a diagnostic: a wide IQR means
    PTT is unstable, which is itself informative during AF.
    """
    ref = np.asarray(reference_s, float)
    det = np.asarray(detected_s, float)
    if ref.size < 3 or det.size < 3:
        return float("nan"), float("nan")

    lo, hi = search_ms
    # Coarse grid search on total absolute nearest-neighbour distance.
    grid = np.arange(lo, hi + 1.0, 2.0)
    costs = []
    for g in grid:
        shifted = det - g / 1000.0
        idx = np.searchsorted(ref, shifted)
        idx = np.clip(idx, 1, ref.size - 1)
        d = np.minimum(np.abs(shifted - ref[idx - 1]), np.abs(shifted - ref[idx]))
        costs.append(np.median(d))
    best = float(grid[int(np.argmin(costs))])

    # Refine: per-beat lag at the best coarse offset.
    shifted = det - best / 1000.0
    idx = np.searchsorted(ref, shifted)
    idx = np.clip(idx, 1, ref.size - 1)
    left, right = ref[idx - 1], ref[idx]
    nearest = np.where(np.abs(shifted - left) < np.abs(shifted - right), left, right)
    lags_ms = (det - nearest) * 1000.0
    lags_ms = lags_ms[np.abs(lags_ms - best) < 200.0]      # reject gross outliers
    if lags_ms.size < 3:
        return best, float("nan")
    q1, q3 = np.percentile(lags_ms, [25, 75])
    return float(np.median(lags_ms)), float(q3 - q1)


def match_beats(reference_s: Sequence[float],
                detected_s: Sequence[float],
                tolerance_ms: float = 100.0,
                ptt_ms: Optional[float] = None,
                auto_ptt: bool = True) -> BeatMatchResult:
    """One-to-one greedy-by-proximity beat matching.

    Args:
        reference_s: ECG R-peak times (seconds, ECG clock, already offset-
            corrected onto the video clock by SyncRecord.offset_ms).
        detected_s:  rPPG pulse-peak times (seconds, same clock).
        tolerance_ms: match window. Report at 50 ms AND 100 ms -- 50 ms is
            the physiologically meaningful bar (Blok 2021 used it for contact
            PPG and found only 89.2% of beats within it on cardiac patients);
            100 ms is the permissive bar used for coverage accounting.
        ptt_ms: fixed pulse transit time to subtract. If None and auto_ptt,
            it is estimated.

    Returns BeatMatchResult. Timing error is signed: detected minus reference
    after PTT removal, so positive means the pulse was detected late.
    """
    ref = np.sort(np.asarray(reference_s, float))
    det = np.sort(np.asarray(detected_s, float))

    ptt_iqr = float("nan")
    if ptt_ms is None:
        if auto_ptt:
            ptt_ms, ptt_iqr = estimate_ptt_ms(ref, det)
            if not np.isfinite(ptt_ms):
                ptt_ms = 0.0
        else:
            ptt_ms = 0.0
    det_c = det - ptt_ms / 1000.0

    tol = tolerance_ms / 1000.0
    # Build all candidate pairs within tolerance, then take them greedily in
    # ascending distance so each beat is used at most once.
    cand: list[tuple[float, int, int]] = []
    for j, t in enumerate(det_c):
        lo = np.searchsorted(ref, t - tol, side="left")
        hi = np.searchsorted(ref, t + tol, side="right")
        for i in range(lo, hi):
            cand.append((abs(t - ref[i]), i, j))
    cand.sort()

    used_ref: set[int] = set()
    used_det: set[int] = set()
    mr: list[int] = []
    md: list[int] = []
    for _, i, j in cand:
        if i in used_ref or j in used_det:
            continue
        used_ref.add(i); used_det.add(j)
        mr.append(i); md.append(j)

    order = np.argsort(mr)
    mr_a = np.asarray(mr, int)[order]
    md_a = np.asarray(md, int)[order]
    errs = (det_c[md_a] - ref[mr_a]) * 1000.0 if mr_a.size else np.array([])

    return BeatMatchResult(
        n_reference=ref.size, n_detected=det.size, n_matched=mr_a.size,
        tolerance_ms=tolerance_ms, ptt_estimate_ms=float(ptt_ms),
        ptt_iqr_ms=ptt_iqr, timing_errors_ms=errs,
        matched_ref_idx=mr_a, matched_det_idx=md_a,
    )


def ibi_agreement(reference_s: Sequence[float],
                  detected_s: Sequence[float],
                  match: BeatMatchResult) -> IBIResult:
    """Interval-level agreement on consecutive matched beat pairs only."""
    ref = np.sort(np.asarray(reference_s, float))
    det = np.sort(np.asarray(detected_s, float))
    mr, md = match.matched_ref_idx, match.matched_det_idx

    ref_ibi, est_ibi, n_gap = [], [], 0
    for k in range(len(mr) - 1):
        if mr[k + 1] == mr[k] + 1 and md[k + 1] == md[k] + 1:
            ref_ibi.append((ref[mr[k + 1]] - ref[mr[k]]) * 1000.0)
            est_ibi.append((det[md[k + 1]] - det[md[k]]) * 1000.0)
        else:
            n_gap += 1

    ref_ibi = np.asarray(ref_ibi); est_ibi = np.asarray(est_ibi)
    if ref_ibi.size == 0:
        nan = float("nan")
        return IBIResult(0, n_gap, nan, nan, nan, nan, nan, nan, nan, nan,
                         nan, nan, nan, nan, nan, nan)

    d = est_ibi - ref_ibi
    rmssd = lambda x: float(np.sqrt(np.mean(np.diff(x) ** 2))) if x.size > 1 else float("nan")
    sdnn = lambda x: float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
    r_rmssd, e_rmssd = rmssd(ref_ibi), rmssd(est_ibi)
    r_sdnn, e_sdnn = sdnn(ref_ibi), sdnn(est_ibi)

    return IBIResult(
        n_intervals=int(ref_ibi.size),
        n_excluded_spanning_gap=n_gap,
        ibi_mae_ms=float(np.mean(np.abs(d))),
        ibi_rmse_ms=float(np.sqrt(np.mean(d ** 2))),
        ibi_bias_ms=float(np.mean(d)),
        loa_lower_ms=float(np.mean(d) - 1.96 * np.std(d, ddof=1)) if d.size > 1 else float("nan"),
        loa_upper_ms=float(np.mean(d) + 1.96 * np.std(d, ddof=1)) if d.size > 1 else float("nan"),
        pct_within_25ms=float(np.mean(np.abs(d) <= 25) * 100),
        pct_within_50ms=float(np.mean(np.abs(d) <= 50) * 100),
        pct_within_100ms=float(np.mean(np.abs(d) <= 100) * 100),
        ref_rmssd_ms=r_rmssd, est_rmssd_ms=e_rmssd,
        rmssd_error_ms=e_rmssd - r_rmssd,
        ref_sdnn_ms=r_sdnn, est_sdnn_ms=e_sdnn,
        sdnn_error_ms=e_sdnn - r_sdnn,
    )


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
@dataclass
class BeatGate:
    """Gate 1 acceptance thresholds.

    Rationale for each number is given in the implementation plan; they are
    reproduced here so the code is self-documenting and so that changing a
    gate is a reviewable, traceable event.
    """
    name: str
    max_ibi_mae_ms: float
    min_beat_f1_at_50ms: float
    max_missed_beat_rate: float
    max_false_beat_rate: float
    max_rmssd_error_ms: float

    def evaluate(self, m50: BeatMatchResult, ibi: IBIResult) -> tuple[bool, dict]:
        checks = {
            "ibi_mae_ms":        (ibi.ibi_mae_ms,        self.max_ibi_mae_ms,        "<="),
            "beat_f1@50ms":      (m50.f1,                self.min_beat_f1_at_50ms,   ">="),
            "missed_beat_rate":  (m50.missed_beat_rate,  self.max_missed_beat_rate,  "<="),
            "false_beat_rate":   (m50.false_beat_rate,   self.max_false_beat_rate,   "<="),
            "abs_rmssd_error_ms":(abs(ibi.rmssd_error_ms), self.max_rmssd_error_ms,  "<="),
        }
        out, ok_all = {}, True
        for k, (val, thr, op) in checks.items():
            ok = (val <= thr) if op == "<=" else (val >= thr)
            ok = bool(ok) and np.isfinite(val)
            out[k] = {"value": val, "threshold": thr, "op": op, "pass": ok}
            ok_all &= ok
        return ok_all, out


# Gate 1 -- controlled capture, still subject, adequate light. Must pass before
# any AFib classifier work is prioritised.
GATE1_CONTROLLED = BeatGate(
    name="Gate 1 -- controlled capture",
    max_ibi_mae_ms=30.0,        # comfortably below the ~60 ms AF/sinus RMSSD margin
    min_beat_f1_at_50ms=0.90,   # contact wrist PPG on cardiac patients achieves
                                # 89.2% of beats within 50 ms; parity is the bar
    max_missed_beat_rate=0.10,
    max_false_beat_rate=0.10,
    max_rmssd_error_ms=20.0,
)

# Gate 1b -- the same subject population but recorded DURING arrhythmia.
# Deliberately looser on missed beats because pulse deficit is physiological,
# not an algorithm defect -- but the algorithm must KNOW it missed them, which
# is enforced by the beat-confidence calibration test, not here.
GATE1_ARRHYTHMIA = BeatGate(
    name="Gate 1b -- during arrhythmia",
    max_ibi_mae_ms=40.0,
    min_beat_f1_at_50ms=0.85,
    max_missed_beat_rate=0.15,
    max_false_beat_rate=0.10,
    max_rmssd_error_ms=35.0,
)

# Gate 1c -- real-world self-administered capture.
GATE1_REALWORLD = BeatGate(
    name="Gate 1c -- real-world self-scan",
    max_ibi_mae_ms=50.0,
    min_beat_f1_at_50ms=0.80,
    max_missed_beat_rate=0.20,
    max_false_beat_rate=0.15,
    max_rmssd_error_ms=45.0,
)


def quantisation_floor_ms(fps: float) -> dict:
    """Irreducible timing noise from frame quantisation alone.

    SD(peak) = (1000/fps)/sqrt(12); an IBI is a first difference (x sqrt(2));
    a successive difference (what RMSSD is built from) is a second difference
    (x sqrt(6)). Use this to separate 'the camera cannot do better' from
    'our extractor is bad' -- they demand different engineering.
    """
    sd_peak = (1000.0 / fps) / np.sqrt(12)
    return {"fps": fps,
            "sd_peak_ms": sd_peak,
            "sd_ibi_ms": np.sqrt(2) * sd_peak,
            "sd_dibi_ms": np.sqrt(6) * sd_peak}


def decompose_rmssd_error(true_rmssd_ms: float, measured_rmssd_ms: float,
                          fps: float) -> dict:
    """Split observed RMSSD inflation into quantisation vs extractor error.

    Assumes independent additive timing noise, so observed RMSSD ~
    sqrt(true^2 + noise^2). Returns the extractor residual, which is the
    engineerable term and therefore the one to optimise.
    """
    q = quantisation_floor_ms(fps)["sd_dibi_ms"]
    total = np.sqrt(max(measured_rmssd_ms ** 2 - true_rmssd_ms ** 2, 0.0))
    extractor = np.sqrt(max(total ** 2 - q ** 2, 0.0))
    return {
        "true_rmssd_ms": true_rmssd_ms,
        "measured_rmssd_ms": measured_rmssd_ms,
        "total_noise_ms": float(total),
        "quantisation_ms": float(q),
        "extractor_ms": float(extractor),
        "quantisation_share_of_variance": float(q ** 2 / total ** 2) if total > 0 else float("nan"),
        "extractor_share_of_variance": float(extractor ** 2 / total ** 2) if total > 0 else float("nan"),
    }


# --------------------------------------------------------------------------
# v2 additions (pressure-test findings P1/P6/P12)
# --------------------------------------------------------------------------
def production_rmssd_error_ms(ref_rpeaks_s: Sequence[float],
                              production_rmssd_ms: float) -> dict:
    """Gate metric tying the PRODUCTION feature path to ECG ground truth.

    The matched-pair harness silently excludes detection errors, so it can
    pass while the production path fails (measured: sinus true 31 ms, matched
    58 ms, raw-series production 338 ms). This metric compares what the
    feature engine ACTUALLY computed (clean-run RMSSD from beats/ibi.py)
    against the ECG-derived RMSSD, on the same recording. Gate 1 v2 requires
    |error| <= 25 ms in sinus rhythm.
    """
    ref = np.sort(np.asarray(ref_rpeaks_s, float))
    if ref.size < 3:
        return {"ref_rmssd_ms": float("nan"),
                "production_rmssd_ms": production_rmssd_ms,
                "error_ms": float("nan")}
    ibi = np.diff(ref) * 1000.0
    ref_rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    return {"ref_rmssd_ms": ref_rmssd,
            "production_rmssd_ms": float(production_rmssd_ms),
            "error_ms": float(production_rmssd_ms - ref_rmssd)}


def beat_confidence_calibration(match: BeatMatchResult,
                                det_confidences: Sequence[float],
                                n_bins: int = 5) -> dict:
    """Is the detector's per-beat confidence honest?

    Clean runs (the production feature path) stand or fall on confidence being
    informative: among beats asserted at confidence c, roughly a fraction c
    should be real (ECG-matched). Reports per-bin (mean confidence, fraction
    matched, n) and an expected-calibration-error style summary. The plan
    referenced this test without defining it — now it exists.
    """
    conf = np.asarray(det_confidences, float)
    if conf.size != match.n_detected:
        raise ValueError("det_confidences length must equal n_detected")
    matched = np.zeros(match.n_detected, bool)
    matched[match.matched_det_idx] = True

    order = np.argsort(conf)
    bins = np.array_split(order, n_bins)
    rows, ece = [], 0.0
    for b in bins:
        if b.size == 0:
            continue
        mc = float(np.mean(conf[b])); fm = float(np.mean(matched[b]))
        rows.append({"mean_confidence": mc, "fraction_matched": fm, "n": int(b.size)})
        ece += (b.size / conf.size) * abs(mc - fm)
    return {"ece": float(ece), "bins": rows,
            "overall_ppv": float(np.mean(matched)) if conf.size else float("nan")}


def missed_beat_flag_recall(reference_s: Sequence[float],
                            detected_s: Sequence[float],
                            match: BeatMatchResult,
                            flagged_interval_idx: Sequence[int]) -> dict:
    """Of the true beats the detector missed, what fraction fall inside
    intervals it FLAGGED as suspiciously long?

    Gate 1b tolerates missed beats during AF because pulse deficit is
    physiology — but only if the system KNOWS where its gaps are. High flag
    recall means downstream logic can treat those regions as uncertain rather
    than as evidence.
    """
    ref = np.sort(np.asarray(reference_s, float))
    det = np.sort(np.asarray(detected_s, float))
    unmatched = sorted(set(range(ref.size)) - set(match.matched_ref_idx.tolist()))
    if not unmatched:
        return {"n_missed": 0, "n_flag_covered": 0, "flag_recall": float("nan")}
    ptt = match.ptt_estimate_ms / 1000.0 if np.isfinite(match.ptt_estimate_ms) else 0.0
    covered = 0
    flagged = set(int(i) for i in flagged_interval_idx)
    for i in unmatched:
        t = ref[i] + ptt                          # where the pulse would appear
        k = int(np.searchsorted(det, t)) - 1      # interval index k: (det[k], det[k+1])
        if k in flagged:
            covered += 1
    return {"n_missed": len(unmatched), "n_flag_covered": covered,
            "flag_recall": covered / len(unmatched)}
