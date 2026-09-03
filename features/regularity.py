"""
The canonical regularity representation (v0.7, invariant G-a).

AFib and flutter are SPECIALIZATIONS of regularity: AFib is irregular
with a chaotic signature, flutter is suspiciously regular at a fast
rate. Both heads, and the regularity head itself, consume the ONE
representation built here — never a parallel feature path. The repo's
P2 lesson is the reason: a matched-pair harness once passed while the
production path read sinus as AF, because two implementations of the
same statistics could disagree, and eventually did.

Four families, all from ACCEPT-grade clean runs, nothing computed
across a run break, no interval repair of any kind (a repaired interval
erases the very evidence this module exists to measure):

  dispersion    RMSSD / SDNN / CV / median absolute successive
                difference / pNN / relative MAD — successive-difference
                statistics pooled over WITHIN-run differences (each run
                contributes its diffs, so the pooling is run-length
                weighted); distribution statistics pooled over intervals.
  distribution  Shannon and sample entropy, Poincaré SD1/SD2, spectral
                entropy, Markov surprise, turning-point ratio — the
                classic AF descriptors and the interpretable competitor
                to any learned model. Sequence-order estimators run on
                the LONGEST run only: concatenating runs would inject
                fake transitions at the seams.
  structure     the benign/pathological discriminator: respiratory
                coupling of the interval series to the camera-derived
                breathing channel (RSA is phase-locked to breathing, AF
                is not), short-run periodicity of the interval pattern
                (bigeminy/trigeminy alternation => ectopy), and outlier
                topology (isolated short/long pairs => ectopy with a
                compensatory pause; pervasive scatter => chaotic).
  confidence    run structure, the per-beat confidence distribution,
                and the propagated timing-jitter budget — what the
                camera itself contributes to every number above.

The `values` dict is the pre-v0.7 production feature vector, key for
key and bit for bit (features/rhythm.py's RhythmFeatures is now a view
of it); the decision logic, Model A and the training engine read it
unchanged. The new families are additive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math

import numpy as np

REGULARITY_FEATURES_VERSION = "regularity-features-v1"

# Structure-family constants (see respiratory_coupling below).
TACHOGRAM_FS_HZ = 4.0
MAX_TACHOGRAM_GAP_S = 3.0
RESP_BAND_HZ = (8.0 / 60.0, 25.0 / 60.0)
RESP_HALF_WIDTH_HZ = 0.03
MIN_INTERVALS_FOR_TACHOGRAM = 15
# Below this the respiration channel is not trustworthy enough to test
# coupling against (rppg.respiration's own concentration floor is 0.25).
MIN_RESP_QUALITY = 0.25
# A "decoupled" verdict this low must be corroborated by the tachogram's
# own spectrum before it counts ...
UNCORROBORATED_FRACTION = 0.20
# ... and a peak elsewhere only DISCREDITS the reported rate if the
# tachogram is DOMINATED by it (more than half the respiratory-band
# power). A metronomic series' strongest bin is noise, and a
# variable-block series wanders broadbandly — measured 0.42 at a
# spurious 8/min — so a lower bar refused those instead of the case it
# is for: a genuinely modulated series whose reported rate is wrong
# (measured 0.98).
CORROBORATED_ELSEWHERE_FRACTION = 0.50

# Timing-jitter budget (Task 2). Beat times come from a sampled
# waveform: naive peak picking quantises each beat with SD
# (1000/fps)/sqrt(12); parabolic sub-sample refinement recovers part of
# that. MEASURED through the production path on metronomic clips (v0.6):
# RMSSD 15.0 ms at 30 fps and 8.2 ms at 60 fps against a raw-quantization
# prediction of sqrt(6)*9.62 = 23.6 and 11.8 ms — an interpolation gain
# of 0.64-0.69. PLANNING VALUES, recomputed by `cli.py regularity-floor`.
DEFAULT_INTERPOLATION_GAIN = 0.65
# Low-confidence beats are placed less precisely. Planning value: a beat
# at confidence 0 carries twice the timing error of a certain one.
DEFAULT_CONFIDENCE_PENALTY = 1.0

# Outlier topology: an interval further than this many MADs from the
# run median is an outlier; a short outlier followed within two beats by
# a long one whose sum lands near 2x the median is a coupled beat plus
# its compensatory pause — the ectopy signature.
OUTLIER_MADS = 3.0
COMPENSATORY_SUM_TOL = 0.15
# "Pervasive" scatter is judged against the MEDIAN INTERVAL, not the
# spread: a chaotic series has a huge MAD and therefore almost no
# MAD-outliers. An interval more than this fraction from its run median
# is scattered; when most intervals are, the scatter is pervasive.
SCATTER_REL_DEV = 0.10
PERVASIVE_SCATTER_FRACTION = 0.50


# ------------------------------------------------------------ estimators
# Moved verbatim from features/rhythm.py (v0.1) — the sample-size floors
# are load-bearing: refusing to return a number below them is the point.
def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """SampEn(m, r*SD). Returns NaN below the sample-size floor.

    Refusing to return a number is the point. SampEn on ~30 intervals -- what
    a 30 s scan at 60 bpm yields -- is dominated by estimator variance, and a
    classifier trained on such values learns the noise. N>=100 is the
    conventional floor for m=2; 60 is admitted with a warning.
    """
    x = np.asarray(x, float)
    N = x.size
    if N < 60:
        return float("nan")
    r = r_frac * np.std(x, ddof=1)
    if r <= 0:
        return float("nan")

    def _phi(mm: int) -> float:
        emb = np.array([x[i:i + mm] for i in range(N - mm)])
        if emb.shape[0] < 2:
            return float("nan")
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        np.fill_diagonal(d, np.inf)
        return float(np.sum(d <= r))

    A, B = _phi(m + 1), _phi(m)
    if not np.isfinite(A) or not np.isfinite(B) or A == 0 or B == 0:
        return float("nan")
    return float(-np.log(A / B))


def shannon_entropy(x: np.ndarray, bins: int = 16) -> float:
    """Normalised Shannon entropy of the interval histogram.

    Histogram-based, therefore the most sample-hungry of the entropy family.
    With 30 intervals across 16 bins the average occupancy is under 2 and the
    estimate is essentially unusable; the floor below reflects that.
    """
    x = np.asarray(x, float)
    if x.size < 50:
        return float("nan")
    h, _ = np.histogram(x, bins=bins)
    p = h[h > 0] / h.sum()
    return float(-np.sum(p * np.log(p)) / np.log(bins))


def spectral_entropy(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.size < 32:
        return float("nan")
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    p = p[1:]
    if p.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(p.size))


def turning_point_ratio(x: np.ndarray) -> float:
    """Fraction of points that are local extrema. ~2/3 for a random series."""
    x = np.asarray(x, float)
    if x.size < 20:
        return float("nan")
    t = np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) |
               (x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))
    return float(t / (x.size - 2))


def markov_surprise(x: np.ndarray, n_states: int = 3) -> float:
    """Mean surprise of a symbolised first-order interval-transition model.

    Symbolise each interval as shorter / similar / longer than its
    predecessor, then measure the entropy of the transition matrix. AF's
    irregular-irregularity produces a near-uniform transition matrix; ectopy
    produces a characteristic short-long (coupling-pause) structure, which is
    exactly the discriminator that pure dispersion statistics like RMSSD miss.
    """
    x = np.asarray(x, float)
    if x.size < 40:
        return float("nan")
    d = np.diff(x)
    thr = 0.05 * np.median(x)
    sym = np.where(d < -thr, 0, np.where(d > thr, 2, 1))
    T = np.zeros((n_states, n_states))
    for a, b in zip(sym[:-1], sym[1:]):
        T[a, b] += 1
    row = T.sum(axis=1, keepdims=True)
    P = np.divide(T, row, out=np.zeros_like(T), where=row > 0)
    ent, tot = 0.0, 0.0
    for i in range(n_states):
        if row[i, 0] > 0:
            p = P[i][P[i] > 0]
            ent += row[i, 0] * (-np.sum(p * np.log(p)))
            tot += row[i, 0]
    return float(ent / tot / np.log(n_states)) if tot > 0 else float("nan")


# ------------------------------------------------------- run accessors
def clean_intervals(obj):
    """(intervals ms, end-beat times s) from clean runs only, in time
    order, from any object carrying `runs` and `run_times`. Returns empty
    arrays when the object carries no interval times — never a guess,
    and never times synthesized from an index."""
    runs = list(getattr(obj, "runs", []) or [])
    times = list(getattr(obj, "run_times", []) or [])
    if not runs or len(times) != len(runs):
        return np.array([]), np.array([])
    ibi, t = [], []
    for r, rt in zip(runs, times):
        r = np.asarray(r, float)
        rt = np.asarray(rt, float)
        if r.size == 0 or rt.size != r.size:
            continue
        ok = np.isfinite(r) & (r > 0) & np.isfinite(rt)
        ibi.append(r[ok])
        t.append(rt[ok])
    if not ibi:
        return np.array([]), np.array([])
    ibi = np.concatenate(ibi)
    t = np.concatenate(t)
    order = np.argsort(t)
    return ibi[order], t[order]


def run_diffs(obj) -> np.ndarray:
    """Successive differences WITHIN runs only — never across a break."""
    out = []
    for r in (getattr(obj, "runs", []) or []):
        r = np.asarray(r, float)
        r = r[np.isfinite(r) & (r > 0)]
        if r.size >= 2:
            out.append(np.diff(r))
    return np.concatenate(out) if out else np.array([])


# ---------------------------------------------------- the representation
@dataclass
class RegularityFeatures:
    """The one representation. `values` is the pre-v0.7 production
    feature vector (bit-identical); the families are additive."""
    version: str
    values: dict
    n_intervals: int
    mean_confidence: float
    estimator_warnings: list
    dispersion: dict = field(default_factory=dict)
    distribution: dict = field(default_factory=dict)
    structure: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)
    run_confidences: Optional[list] = None
    run_times: list = field(default_factory=list)
    fps: Optional[float] = None

    def as_rhythm_features(self):
        """The legacy view the decision logic, Model A and the training
        engine read — the SAME dict object, so nothing can drift."""
        from features.rhythm import RhythmFeatures
        return RhythmFeatures(self.values, self.n_intervals,
                              self.mean_confidence,
                              list(self.estimator_warnings))

    def to_dict(self) -> dict:
        def _j(v):
            if isinstance(v, dict):
                return {k: _j(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_j(x) for x in v]
            if isinstance(v, np.ndarray):
                return [_j(x) for x in v.tolist()]
            if isinstance(v, (np.floating, float)):
                f = float(v)
                return None if not np.isfinite(f) else f
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            return v
        return {"version": self.version, "values": _j(self.values),
                "n_intervals": int(self.n_intervals),
                "mean_confidence": _j(self.mean_confidence),
                "estimator_warnings": list(self.estimator_warnings),
                "dispersion": _j(self.dispersion),
                "distribution": _j(self.distribution),
                "structure": _j(self.structure),
                "confidence": _j(self.confidence),
                "index": _j(self.index), "fps": _j(self.fps)}


# -------------------------------------------------- the legacy vector
def _legacy_values(runs, run_confidences, dropout_rate, mean_sqi):
    """THE PRODUCTION FEATURE PATH (v2, after pressure-test findings
    P1/P6), moved VERBATIM from features/rhythm.py so the values stay
    bit-identical. Rationale (kept with the code it governs):

      * BOUNDED statistics (pNN50/pNN20, median irregularity) are
        primary — a single residual detection error moves them ~2
        points, not 4.6x.
      * RMSSD/SDNN are computed run-wise and are secondary.
      * Sequence-order features (sample entropy, Shannon, Markov, TPR)
        are computed on the LONGEST run only — concatenating runs would
        inject fake transition patterns at the seams. If the longest run
        is below an estimator's floor, that estimator returns NaN with a
        warning: fragmentation IS low quality and must surface.
      * dropout_rate joins the feature set: under good SQI it is the
        optical pulse-deficit signature, which is AF evidence.
    """
    warn: list = []
    v: dict = {}
    runs = [np.asarray(r, float) for r in runs if len(r) > 0]
    all_int = np.concatenate(runs) if runs else np.array([])
    n = int(all_int.size)

    v["n_runs"] = float(len(runs))
    v["n_intervals"] = float(n)
    v["longest_run"] = float(max((r.size for r in runs), default=0))
    v["dropout_rate"] = float(dropout_rate)
    v["mean_sqi"] = float(mean_sqi)

    if n < 5:
        return v, n, float("nan"), ["fewer than 5 clean intervals -- no "
                                    "rhythm features computed; return "
                                    "NO_RESULT upstream"], runs, all_int, \
            np.array([]), np.array([])

    diffs_list = [np.diff(r) for r in runs if r.size >= 2]
    d = np.concatenate(diffs_list) if diffs_list else np.array([])
    if d.size < 4:
        warn.append("fewer than 4 within-run successive differences; "
                    "diff-based features suppressed")

    # ---- primary: bounded / robust (survive residual detection errors)
    v["mean_ibi"] = float(np.mean(all_int))
    v["median_ibi"] = float(np.median(all_int))
    if d.size >= 4:
        v["pnn50"] = float(np.mean(np.abs(d) > 50))
        v["pnn20"] = float(np.mean(np.abs(d) > 20))
        v["median_abs_succ_diff"] = float(np.median(np.abs(d)))
        v["irregularity_index"] = (v["median_abs_succ_diff"] / v["median_ibi"]
                                   if v["median_ibi"] > 0 else float("nan"))

    # ---- secondary: unbounded dispersion, run-wise only
    v["sdnn"] = float(np.std(all_int, ddof=1)) if n > 1 else float("nan")
    v["cv_ibi"] = v["sdnn"] / v["mean_ibi"] if v.get("mean_ibi") else float("nan")
    if d.size >= 4:
        v["rmssd"] = float(np.sqrt(np.mean(d ** 2)))
        v["sdsd"] = float(np.std(d, ddof=1))
        v["poincare_sd1"] = float(np.std(d, ddof=1) / np.sqrt(2))
        sd = float(np.std(all_int, ddof=1))
        v["poincare_sd2"] = float(np.sqrt(max(2 * sd ** 2 - v["poincare_sd1"] ** 2, 0.0)))
        v["poincare_ratio"] = (v["poincare_sd1"] / v["poincare_sd2"]
                               if v["poincare_sd2"] > 0 else float("nan"))

    # ---- sequence-order features: longest run only
    longest = max(runs, key=lambda r: r.size) if runs else np.array([])
    L = longest.size
    if L < 60:
        warn.append(f"longest clean run has {L} intervals: sample entropy "
                    "suppressed (needs >=60 contiguous)")
    if L < 50:
        warn.append(f"longest clean run has {L} intervals: Shannon entropy "
                    "suppressed (needs >=50 contiguous)")
    v["sample_entropy"] = sample_entropy(longest) if L >= 60 else float("nan")
    v["shannon_entropy"] = shannon_entropy(longest) if L >= 50 else float("nan")
    v["spectral_entropy"] = spectral_entropy(longest)
    v["turning_point_ratio"] = turning_point_ratio(longest)
    v["markov_surprise"] = markov_surprise(longest)
    # a constant run has zero variance: corrcoef is NaN there (unchanged
    # value), and the warning numpy raises for it is noise, not a finding
    with np.errstate(invalid="ignore", divide="ignore"):
        v["autocorr_lag1"] = (float(np.corrcoef(longest[:-1],
                                                longest[1:])[0, 1])
                              if L > 5 else float("nan"))

    mean_conf = float("nan")
    if run_confidences:
        cc = np.concatenate([np.asarray(c) for c in run_confidences if len(c)])
        mean_conf = float(np.mean(cc)) if cc.size else float("nan")
    return v, n, mean_conf, warn, runs, all_int, d, longest


# --------------------------------------------------- timing jitter budget
def timing_jitter_budget(fps, *, mean_beat_confidence=None,
                         interpolation_gain: float = DEFAULT_INTERPOLATION_GAIN,
                         confidence_penalty: float = DEFAULT_CONFIDENCE_PENALTY
                         ) -> dict:
    """What the camera itself contributes to interval dispersion, per
    session, RECOMPUTED from first principles rather than hard-coded:

      beat-time quantization   sigma_t = (1000/fps) / sqrt(12)   [uniform]
      after sub-sample interp   x interpolation_gain (measured, 0.65)
      after beat confidence     x (1 + penalty * (1 - mean_conf))
      interval error            sigma_int = sqrt(2) * sigma_t
      RMSSD floor on metronome  sqrt(6) * sigma_t  (a successive
                                difference spans three beats)

    Every field is None when fps is unknown: an unknown frame rate must
    never become a permissive floor.
    """
    try:
        f = float(fps)
    except (TypeError, ValueError):
        f = float("nan")
    if not np.isfinite(f) or f <= 0:
        return {"fps": None, "frame_ms": None, "sigma_beat_raw_ms": None,
                "interpolation_gain": interpolation_gain,
                "confidence_gain": None, "sigma_beat_ms": None,
                "sigma_interval_ms": None, "rmssd_floor_ms": None,
                "reason": "frame rate unknown — no jitter budget"}
    frame_ms = 1000.0 / f
    sigma_raw = frame_ms / np.sqrt(12.0)
    if mean_beat_confidence is None or not np.isfinite(
            float(mean_beat_confidence)):
        conf_gain = None
        sigma_beat = sigma_raw * float(interpolation_gain)
    else:
        c = float(np.clip(mean_beat_confidence, 0.0, 1.0))
        conf_gain = 1.0 + float(confidence_penalty) * (1.0 - c)
        sigma_beat = sigma_raw * float(interpolation_gain) * conf_gain
    return {"fps": f, "frame_ms": round(frame_ms, 4),
            "sigma_beat_raw_ms": round(float(sigma_raw), 4),
            "interpolation_gain": float(interpolation_gain),
            "confidence_gain": (None if conf_gain is None
                                else round(conf_gain, 4)),
            "sigma_beat_ms": round(float(sigma_beat), 4),
            "sigma_interval_ms": round(float(np.sqrt(2.0) * sigma_beat), 4),
            "rmssd_floor_ms": round(float(np.sqrt(6.0) * sigma_beat), 4),
            "reason": None}


# ------------------------------------------------------ structure family
def longest_timed_run(obj):
    """(intervals_ms, end-beat times_s) of the longest clean run that
    carries times, or (empty, empty)."""
    runs = list(getattr(obj, "runs", []) or [])
    times = list(getattr(obj, "run_times", []) or [])
    best = (np.zeros(0), np.zeros(0))
    for i, r in enumerate(runs):
        r = np.asarray(r, float)
        t = (np.asarray(times[i], float) if i < len(times) and
             times[i] is not None else np.zeros(0))
        if r.size and t.size == r.size and r.size > best[0].size:
            best = (r, t)
    return best


def tachogram(obj, fs_hz: float = TACHOGRAM_FS_HZ,
              max_gap_s: float = MAX_TACHOGRAM_GAP_S):
    """(grid_t, interval series on a uniform grid) or (None, None).

    Built from the LONGEST clean run only. A run boundary is a stretch
    the pipeline refused (a rejected beat, a missed beat, a false pair),
    and interpolating across it — however short — invents interval
    values inside the very seam the run discipline exists to exclude
    (review finding on v0.7: every single-beat break at ordinary heart
    rates is under the old 3 s bridge). Within the run the series is
    contiguous by construction; the gap check remains as a guard.
    """
    ibi, t = longest_timed_run(obj)
    if ibi.size < MIN_INTERVALS_FOR_TACHOGRAM or t.size != ibi.size:
        return None, None
    span = float(t[-1] - t[0])
    if span < 20.0:                 # < 5 breaths at 15/min: unresolvable
        return None, None
    if float(np.max(np.diff(t))) > max_gap_s:
        return None, None
    grid = np.arange(t[0], t[-1], 1.0 / fs_hz)
    if grid.size < 32:
        return None, None
    return grid, np.interp(grid, t, ibi)


def dominant_frequency(x, fs_hz, band):
    """The strongest frequency of `x` inside `band`, or None."""
    x = np.asarray(x, float)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    fr = np.fft.rfftfreq(x.size, 1.0 / fs_hz)
    m = (fr >= band[0]) & (fr <= band[1])
    if not m.any() or float(spec[m].sum()) <= 0:
        return None
    return float(fr[m][np.argmax(spec[m])])


def band_fraction(x, fs_hz, f_center, half=RESP_HALF_WIDTH_HZ,
                  wide=(0.04, 1.0)) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    fr = np.fft.rfftfreq(x.size, 1.0 / fs_hz)
    w = (fr >= wide[0]) & (fr <= wide[1])
    tot = float(spec[w].sum())
    if tot <= 0:
        return float("nan")
    b = w & (np.abs(fr - f_center) <= half)
    return float(spec[b].sum() / tot)


def phase_locking(x, y, fs_hz, f, n_seg: int = 3):
    """Phase-locking value between the f-components of two signals over
    n_seg segments. Coupled signals hold a stable phase difference;
    coincidence does not. None when a segment is too short to carry two
    cycles of f."""
    n = min(x.size, y.size)
    seg = n // n_seg
    if f <= 0 or seg < int(2.0 * fs_hz / f):
        return None
    phases = []
    for s in range(n_seg):
        xs = np.asarray(x[s * seg:(s + 1) * seg], float)
        ys = np.asarray(y[s * seg:(s + 1) * seg], float)
        tt = np.arange(xs.size) / fs_hz
        ref = np.exp(-2j * np.pi * f * tt)
        cx = np.sum((xs - xs.mean()) * ref)
        cy = np.sum((ys - ys.mean()) * ref)
        if abs(cx) < 1e-12 or abs(cy) < 1e-12:
            return None
        phases.append(np.angle(cx) - np.angle(cy))
    return round(float(abs(np.mean(np.exp(1j * np.asarray(phases))))), 4)


def respiratory_coupling(obj, respiration, *,
                         fs_hz: float = TACHOGRAM_FS_HZ) -> dict:
    """How much of the interval series moves with the breath.

    `respiration` is the camera-derived respiration channel:
    {"t": times_s, "y": signal, "rate_brpm": float, "quality": float}
    — the torso-motion second decode (rppg/respiration.py), NOT the
    facial trace the intervals come from. Returns None-valued fields
    whenever the respiration channel is missing or its rate is
    unresolved: an absent coupling measurement is NOT evidence of absent
    coupling, and a head must abstain rather than flag on it.
    """
    out = {"available": False, "resp_rate_brpm": None,
           "resp_quality": None, "tachogram_resp_fraction": None,
           "phase_locking": None, "phase_locking_at_peak": None,
           "resp_waveform_fraction_at_peak": None,
           "tachogram_peak_fraction": None, "tachogram_rel_std": None,
           "reason": None,
           # v0.7: WHY it is unavailable matters. A missing or poor
           # channel is an absence (heads abstain); a tachogram
           # dominated by a non-breathing periodicity is a FINDING
           # (ectopy patterns do exactly that), so the head can still
           # judge, with the at-reported-rate fraction below.
           "status": "no_tachogram",
           "tachogram_resp_fraction_at_reported_rate": None}
    grid, tach = tachogram(obj, fs_hz)
    if grid is None:
        out["reason"] = ("no gap-free interval series long enough to "
                         "resolve the respiratory band")
        return out
    r = respiration or {}
    rate = r.get("rate_brpm")
    if rate is None:
        out["status"] = "no_channel"
        out["reason"] = "no camera-derived respiration rate available"
        return out
    f_r = float(rate) / 60.0
    if not (RESP_BAND_HZ[0] <= f_r <= RESP_BAND_HZ[1]):
        out["status"] = "no_channel"
        out["reason"] = f"respiration rate {rate} outside the validated band"
        return out
    quality = None if r.get("quality") is None else float(r["quality"])
    out.update({"resp_rate_brpm": float(rate), "resp_quality": quality})
    if quality is not None and quality < MIN_RESP_QUALITY:
        out["status"] = "poor_quality"
        out["reason"] = (f"respiration channel quality {quality:.2f} "
                         f"below {MIN_RESP_QUALITY} — the breathing rate "
                         "is not trustworthy enough to test coupling "
                         "against")
        return out
    # The coupling test is a narrow band around the REPORTED breathing
    # rate, so an error in that rate reads as absent modulation. Measured
    # (v0.6 review finding): a 2 br/min error turned a fully coupled
    # sinus tachycardia from 1.00 to 0.17, i.e. into a flag. So the
    # reported rate must be CORROBORATED by the tachogram itself before
    # its absence of power there means anything.
    frac = band_fraction(tach, fs_hz, f_r)
    peak = dominant_frequency(tach, fs_hz, RESP_BAND_HZ)
    out["tachogram_peak_hz"] = (None if peak is None else round(peak, 4))
    out["tachogram_resp_fraction_at_reported_rate"] = (
        None if not np.isfinite(frac) else round(frac, 4))
    if frac <= UNCORROBORATED_FRACTION and peak is not None and \
            abs(peak - f_r) > RESP_HALF_WIDTH_HZ:
        peak_frac = band_fraction(tach, fs_hz, peak)
        out["tachogram_peak_fraction"] = round(peak_frac, 4)
        if peak_frac >= CORROBORATED_ELSEWHERE_FRACTION:
            out["status"] = "rate_uncorroborated"
            out["reason"] = (
                f"tachogram is modulated at {peak * 60:.1f}/min "
                f"({peak_frac:.2f} of its power) but the camera "
                f"reported {float(rate):.1f}/min — the coupling test "
                "cannot be trusted at either rate")
            med = float(np.median(tach))
            out["tachogram_rel_std"] = (round(float(np.std(tach)) / med, 5)
                                        if med > 0 else None)
            ry = np.asarray(r.get("y") if r.get("y") is not None else [],
                            float)
            rt = np.asarray(r.get("t") if r.get("t") is not None else [],
                            float)
            if ry.size and rt.size == ry.size:
                # is the breath WAVEFORM itself at the tachogram's peak
                # (the reported rate was wrong), and are the two locked
                # there?
                resp_on_grid = np.interp(grid, rt, ry)
                out["resp_waveform_fraction_at_peak"] = round(
                    band_fraction(resp_on_grid, fs_hz, peak), 4)
                out["phase_locking_at_peak"] = phase_locking(
                    tach, resp_on_grid, fs_hz, peak)
            return out
    out["tachogram_resp_fraction"] = round(frac, 4)
    # what the breath could EXPLAIN: the tachogram's relative spread,
    # and the fraction of its power at its own respiratory-band peak.
    # A consumer that has independent evidence the tachogram moves with
    # the breath (phase locking) may trust the peak fraction where the
    # reported-rate fraction was eroded by a rate-estimate error; the
    # residual it implies is what decides "explained".
    med = float(np.median(tach))
    out["tachogram_rel_std"] = (round(float(np.std(tach)) / med, 5)
                                if med > 0 else None)
    if peak is not None:
        out["tachogram_peak_fraction"] = round(band_fraction(tach, fs_hz,
                                                             peak), 4)
    ry = np.asarray(r.get("y") if r.get("y") is not None else [], float)
    rt = np.asarray(r.get("t") if r.get("t") is not None else [], float)
    if ry.size and rt.size == ry.size:
        resp_on_grid = np.interp(grid, rt, ry)
        out["phase_locking"] = phase_locking(tach, resp_on_grid, fs_hz,
                                             f_r)
        if peak is not None and abs(peak - f_r) > RESP_HALF_WIDTH_HZ:
            # the tachogram's own peak sits away from the reported rate:
            # is the breath waveform ITSELF at that peak (a rate-estimate
            # error), and are the two locked there?
            out["resp_waveform_fraction_at_peak"] = round(
                band_fraction(resp_on_grid, fs_hz, peak), 4)
            out["phase_locking_at_peak"] = phase_locking(
                tach, resp_on_grid, fs_hz, peak)
    out["available"] = bool(out["tachogram_resp_fraction"] is not None
                            and np.isfinite(
                                out["tachogram_resp_fraction"]))
    out["status"] = "ok" if out["available"] else "no_spectrum"
    if not out["available"]:
        out["reason"] = "interval series carries no resolvable spectrum"
    return out


def short_run_periodicity(longest_run) -> dict:
    """Bigeminy/trigeminy-style alternation: autocorrelation of the
    interval series at lags 2 and 3 within the longest run. A coupled
    ectopic beat plus its pause repeats every 2 (bigeminy) or 3
    (trigeminy) intervals, so the series is strongly periodic at that
    lag; AF is not periodic at any lag; sinus with RSA is periodic only
    at the breathing period (many beats)."""
    x = np.asarray(longest_run, float)
    out = {"lag2": None, "lag3": None, "alternation_index": None,
           "n": int(x.size)}
    if x.size < 12:
        return out
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return out
    for lag in (2, 3):
        r = float(np.dot(x[:-lag], x[lag:]) / denom)
        out[f"lag{lag}"] = round(r, 4)
    out["alternation_index"] = round(max(out["lag2"], out["lag3"]), 4)
    return out


def outlier_topology(runs) -> dict:
    """Isolated short/long pairs (ectopy with a compensatory pause) vs
    pervasive scatter (chaotic). Computed WITHIN runs: an outlier is an
    interval more than OUTLIER_MADS robust SDs from its run's median; a
    short outlier followed within two intervals by a long one whose sum
    is near twice the median is a coupled-beat pair."""
    n_total = n_out = n_paired = n_scatter = 0
    for r in (runs or []):
        r = np.asarray(r, float)
        r = r[np.isfinite(r) & (r > 0)]
        if r.size < 6:
            n_total += int(r.size)
            continue
        med = float(np.median(r))
        n_scatter += int(np.sum(np.abs(r - med) / med > SCATTER_REL_DEV))
        mad = float(np.median(np.abs(r - med))) * 1.4826
        if mad <= 0:
            n_total += int(r.size)
            continue
        z = (r - med) / mad
        out_idx = np.flatnonzero(np.abs(z) > OUTLIER_MADS)
        n_total += int(r.size)
        n_out += int(out_idx.size)
        seen = set()
        for i in out_idx:
            if i in seen or z[i] >= 0:
                continue
            for j in (i + 1, i + 2):
                if j < r.size and z[j] > OUTLIER_MADS and \
                        abs((r[i] + r[j]) - 2.0 * med) <= \
                        COMPENSATORY_SUM_TOL * 2.0 * med:
                    n_paired += 2
                    seen.add(i)
                    seen.add(j)
                    break
    frac = (n_out / n_total) if n_total else None
    pair_frac = (n_paired / n_out) if n_out else None
    scatter = (n_scatter / n_total) if n_total else None
    if frac is None:
        topology = "unavailable"
    elif scatter is not None and scatter >= PERVASIVE_SCATTER_FRACTION:
        topology = "pervasive"
    elif frac < 0.05:
        topology = "none"
    elif pair_frac is not None and pair_frac >= 0.5:
        topology = "paired"
    else:
        topology = "scattered"
    return {"outlier_fraction": (None if frac is None else round(frac, 4)),
            "paired_fraction": (None if pair_frac is None
                                else round(pair_frac, 4)),
            "scatter_fraction": (None if scatter is None
                                 else round(scatter, 4)),
            "n_outliers": int(n_out), "topology": topology}


# ------------------------------------------------------------- the index
INDEX_CI_METHOD = "moving-block percentile bootstrap, B=400, nominal 95%"
INDEX_CI_COVERAGE_MEASURED = {"n=15": 0.88, "n=30": 0.90, "n=70": 0.94}


def _index_ci(d, med_ibi, *, n_boot: int = 400, seed: int = 20260901):
    """Bootstrap CI of the index (median |successive diff| / median IBI)
    by MOVING-BLOCK resampling of the within-run successive differences
    (adjacent differences share a beat, so the sequence is 1-dependent;
    blocks of ~n^(1/3) keep that dependence inside the block).

    MEASURED, not assumed (review finding, then re-measured after the
    fix): on iid-jittered regular series the percentile interval of a
    median under-covers at small n whatever the resampling unit —
    coverage 0.88 at n = 15, 0.90 at n = 30, 0.94 at n = 70 for a
    nominal 0.95, identical for the block and the iid bootstrap. The
    interval is therefore labelled by its measured coverage
    (INDEX_CI_COVERAGE_MEASURED) and pinned by test; it is approximate
    (the median IBI is held fixed). Deterministic (fixed seed): the CI
    is a property of the data, not of the run."""
    d = np.asarray(d, float)
    n = int(d.size)
    if n < 4 or not np.isfinite(med_ibi) or med_ibi <= 0:
        return None
    rng = np.random.default_rng(seed)
    b = int(max(2, min(n, round(n ** (1.0 / 3.0)))))
    k = int(math.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(
        n_boot, -1)[:, :n]
    boots = np.median(np.abs(d[idx]), axis=1) / med_ibi
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)]


# --------------------------------------------------------- the entry
def regularity_from_runs(runs, run_confidences=None, dropout_rate: float = 0.0,
                         mean_sqi: float = float("nan"), *,
                         run_times=None, fps=None, respiration=None,
                         interpolation_gain: float = DEFAULT_INTERPOLATION_GAIN
                         ) -> RegularityFeatures:
    """Build the canonical representation from clean runs.

    `run_times` (each interval's end-beat time, parallel to `runs`) and
    `fps` come from the BeatLattice; `respiration` is the optional
    camera-derived breathing channel for the structure family.
    """
    v, n, mean_conf, warn, runs_f, all_int, d, longest = _legacy_values(
        runs, run_confidences, dropout_rate, mean_sqi)
    reg = RegularityFeatures(
        version=REGULARITY_FEATURES_VERSION, values=v, n_intervals=n,
        mean_confidence=mean_conf, estimator_warnings=list(warn),
        run_confidences=(None if run_confidences is None else
                         [np.asarray(c, float).ravel().tolist()
                          for c in run_confidences]),
        runs=[np.asarray(r, float) for r in runs_f],
        run_times=[np.asarray(t, float) for t in (run_times or [])],
        fps=(None if fps is None else float(fps)))

    # dispersion family — the legacy numbers under their own names plus
    # the two relative measures head_irregularity used to compute itself
    med = v.get("median_ibi")
    rel_mad = (float(np.median(np.abs(all_int - med)) / med)
               if n >= 1 and med and med > 0 else None)
    reg.dispersion = {
        "rmssd_ms": v.get("rmssd"), "sdnn_ms": v.get("sdnn"),
        "sdsd_ms": v.get("sdsd"), "cv": v.get("cv_ibi"),
        "median_abs_succ_diff_ms": v.get("median_abs_succ_diff"),
        "irregularity_index": v.get("irregularity_index"),
        "pnn50": v.get("pnn50"), "pnn20": v.get("pnn20"),
        "pnn80": (float(np.mean(np.abs(d) > 80.0)) if d.size else None),
        "rel_mad": rel_mad,
        "n_within_run_diffs": int(d.size),
    }
    reg.distribution = {
        "shannon_entropy": v.get("shannon_entropy"),
        "sample_entropy": v.get("sample_entropy"),
        "spectral_entropy": v.get("spectral_entropy"),
        "poincare_sd1": v.get("poincare_sd1"),
        "poincare_sd2": v.get("poincare_sd2"),
        "poincare_ratio": v.get("poincare_ratio"),
        "markov_surprise": v.get("markov_surprise"),
        "turning_point_ratio": v.get("turning_point_ratio"),
        "autocorr_lag1": v.get("autocorr_lag1"),
    }
    reg.structure = {
        "coupling": respiratory_coupling(reg, respiration),
        "periodicity": short_run_periodicity(longest),
        "outliers": outlier_topology(runs_f),
    }
    conf_vals = None
    if run_confidences:
        cc = [np.asarray(c, float) for c in run_confidences if len(c)]
        if cc:
            conf_vals = np.concatenate(cc)
    reg.confidence = {
        "n_runs": int(len(runs_f)),
        "run_lengths": [int(r.size) for r in runs_f],
        "longest_run": int(longest.size) if hasattr(longest, "size") else 0,
        "n_intervals": int(n),
        "beat_confidence": (None if conf_vals is None or not conf_vals.size
                            else {"mean": round(float(np.mean(conf_vals)), 4),
                                  "p10": round(float(np.percentile(conf_vals,
                                                                   10)), 4),
                                  "min": round(float(np.min(conf_vals)), 4)}),
        "jitter": timing_jitter_budget(
            fps, mean_beat_confidence=(
                None if conf_vals is None or not conf_vals.size
                else float(np.mean(conf_vals))),
            interpolation_gain=interpolation_gain),
    }
    idx = v.get("irregularity_index")
    reg.index = {
        "name": "irregularity_index",
        "definition": "median |successive difference| / median interval, "
                      "within-run differences only",
        "value": (None if idx is None or not np.isfinite(idx)
                  else float(idx)),   # the statistic the definition names, unrounded
        "ci95": _index_ci(d, med if med is not None else float("nan")),
        "ci_method": INDEX_CI_METHOD,
        "ci_coverage_measured": dict(INDEX_CI_COVERAGE_MEASURED),
        "n_diffs": int(d.size),
    }
    return reg
