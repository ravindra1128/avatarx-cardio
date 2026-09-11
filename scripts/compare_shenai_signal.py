"""
!!! DO NOT DECIDE ANYTHING FROM THIS SCRIPT YET (2026-09-11) !!!

The CAPTURE side of route 6 is sound and deployed. This COMPARISON is not, and
an adversarial review killed it twice in opposite directions:

  round 1, no permutation null: two beat trains with the same rate and a
    DELIBERATELY DESTROYED timebase relation were declared "alignable" in
    51/60 = 85% of trials, median |dt| 35 ms - under the 40 ms ceiling this
    script printed two lines later as if it were evidence. Pure noise aligned
    3 times in 40.

  round 2, permutation null added: the null gates on a matched FRACTION, which
    is bounded at 1.0, so at ordinary HRV a permuted train still matches
    everything inside the tolerance and p95 saturates - making the test
    mathematically unsatisfiable. Measured power on TRULY ALIGNED trains at the
    production 48-beat window: SDNN 50ms=100%, 35ms=93%, 25ms=73%, 20ms=40%,
    15ms=13%, 10ms=3%. Our own reference scans read RMSSD 25-45 ms, i.e.
    SDNN ~18-32 ms - exactly where the power collapses.

  And the decisive one: simulating the world where ShenAI genuinely IS better,
    with HRV drawn from our own stated reference range, four independent 6-clip
    cohorts all FAILED clause 1 of the pre-registered rule. The rule says
    REJECT in the adopt-worthy world. Because it also fails when ShenAI
    over-smooths, it fails in BOTH worlds and carries zero discriminating
    information.

ROOT CAUSE, and why tuning will not fix it: beat-by-beat comparison needs an
offset between ShenAI's measurement clock and the clip's container clock, and
that offset may simply not be recoverable. Every gate either lets chance
through or rejects the truth.

THE REWRITE: drop the alignment-dependent arm entirely. The comparisons that
answer the actual question - is ShenAI's single dense signal a more
self-consistent basis than four regions that disagree by 15-23 bpm? - need no
shared timebase at all: per-waveform SNR, our cross-ROI disagreement vs their
single signal, each system's rate against the reference HR (a scalar),
intervals surviving clean_runs, and RMSSD/SDNN to catch over-smoothing.

Kept in the tree because the extraction, arm plumbing and TOO-GOOD detectors
are reusable. The alignment block and the pre-registered rule are not.
"""

"""ShenAI's dense PPG + beat train vs our four-ROI POS lattice, offline.

WHY THIS EXISTS. The measured bottleneck is cross-ROI beat FUSION, not the
detector. At the phone's compression level the four face regions disagree on
the pulse by 15-23 bpm and cross-region waveform correlation is 0.161 (lab rig
0.330; measured 2026-09-10, scripts/compare_capture.py). That shatters the
interval series: coverage sits at 0.47 against the 0.50 floor and ACCEPT lands
on 1 scan in 49. ShenAI builds a 3D face model and extracts ONE dense signal,
so its waveform and its beat train are a SECOND INDEPENDENT OPINION on the
same beats — free to collect (route 6 ships them in a sidecar) and cheap to
judge. If they are more self-consistent than ours, the beat lattice gets a
better front end; if not, we learned it without a study.

FIVE ARMS per clip, because the question is only separable with controls:
  A   ours, PRODUCTION-CONFIGURED (trim_tail DEFAULT_WINDOW_S then downscale
      DEFAULT_SCALE on a COPY, then ingest -> extract_and_detect ->
      fuse_multi_roi min_rois=2 -> Calibrator -> clean_runs). The defaults are
      production's, not "off": a 2026-09-11 review found arm A analysing
      209.9 s at 640x480 — five times production's window at 1.8x its linear
      resolution — while its SNR was read against a bench measured at the
      phone's compression level. The effective window and scale are printed on
      every clip header so the label can be checked, not trusted.
  An  ours, the SAME four-ROI path at the clip's NATIVE resolution (same trim,
      no downscale). A - An is the DOWNSCALE term, see THE CODEC CONFOUND.
  A'  CONTROL: our forehead ROI alone through the SINGLE-waveform path.
      Without it arm B "wins" merely by not paying the four-ROI fusion tax —
      on one train fused confidence collapses to the beat's own peak-width
      factor (beats/detector.py:255-271), so clean_runs filters far less.
  B   ShenAI's waveform through OUR detector.
  C   ShenAI's own beat train. Its confidence is SYNTHESISED (the SDK exposes
      none), so clean_runs' confidence channel is inert on that arm.

THE CODEC CONFOUND — read before any row. ARMS A/An/A' AND ARMS B/C ARE NOT
EXTRACTING FROM THE SAME DATA. ShenAI owns its own camera and runs on the LIVE
frames; our arms run on a 5 Mbps VP8/H.264 MediaRecorder clip that the service
then downscales. Arm B's waveform NEVER passed through a codec and ours always
did — and compression is the NAMED cause of the bottleneck this harness exists
to attack (cross-ROI r 0.330 lab -> 0.161 phone; four ROIs disagreeing 15-23
bpm). So if arm B wins, "their extractor is better" and "they never saw the
compression" are NOT distinguishable from this table, and the two have OPPOSITE
remedies: adopt their extractor, versus extract on-device before encoding /
raise the bitrate. Arm An makes the part we CAN vary separable (A - An is the
downscale term). The encode itself cannot be undone offline — there is no
uncompressed copy of a phone scan in this repo — so A - An LOWER-BOUNDS the
codec+resolution term and never removes it.

NO AGREEMENT NUMBER WITHOUT A NULL. Putting ShenAI's train on the clip's clock
means searching ~1000 candidate offsets, and the winner of a search is not
evidence until it beats what the search finds in noise. MEASURED against the
rule this file shipped with (best offset, then matched/min(n) >= 0.5): two
trains at the same rate with a DESTROYED timebase relation were called
alignable in 54/60 = 90% of trials at a median |dt| of 36 ms, and 3/40 pure-
noise pairs aligned too. With the permutation null in _align: 0/60 and 0/40,
while a truly aligned pair (15% of our beats missed, 30 ms detector jitter)
still passes 30/30. Every agreement line now prints the null percentile it
beat; a clip that does not beat it prints "unalignable" and no numbers. Re-run
it any time with --self-test, or tests/test_shenai_alignment_null.py.

WHICH CARDS EACH ARM COULD EVER SUPPORT. decision_logic.py:253-273 and :365-372
let an AF call through only when cross_roi_coherence clears the floor (0.20 for
any rhythm statement, afib_min_coherence 0.35 for the AF call) OR
two_region_verified, and two_region_verified is beat-timing agreement across
INDEPENDENT skin regions (afib_max_timing_precision_ms 30 ms at
afib_min_timing_matched 0.75). ShenAI ships ONE dense signal and ONE train, so
arms B and C have no second region and CANNOT satisfy either branch: adopting
ShenAI as the front end AS-IS would downgrade every irregular series to
REPEAT_SCAN and the AFib card would never fire. Arms B/C can speak to Pulse,
and to Fitness so far as it rests on rate and interval statistics. Coverage and
interval counts on those rows are therefore evidence about a FRONT END feeding
our lattice, never evidence that the verification gate is satisfiable.

PRE-REGISTERED DECISION RULE (fixed 2026-09-11, BEFORE any sidecar existed —
without this the summary is a wall of numbers in which several arms are
inflated by construction and the reader picks a winner post hoc):

  ADOPT ShenAI's waveform as a front end (arm B) only if, over >= 10 clips with
  sidecars, ALL of:
    1. arm B's alignment passes the permutation null on >= 80% of clips (an
       agreement number that does not beat its own null is not evidence);
    2. arm B is not VOID (|drift| <= 2.0 ms/s) on >= 80% of clips;
    3. median coverage(B) >= median coverage(A') + 0.10 AND median
       n_intervals(B) >= median n_intervals(A') + 5 — A', the single-train
       control, is THE comparator; beating A (four-ROI, calibrated) is not
       evidence because B does not pay the fusion tax either;
    4. our beat count on their waveform is within 10% of their own beat count
       on >= 80% of clips (the only rate quantity on arm B that is not
       tautological, see NEAR-TAUTOLOGY below);
    5. NOT flagged TOO-GOOD (median RMSSD >= 20 ms and median SDNN >= 30 ms).
  REJECT if any of: (1) fails; or median RMSSD(B) < 20 ms (a waveform
  regularised by the SDK's 3D face model raises every metric here while
  destroying AFib sensitivity — that is the failure that would cost the most
  and is the hardest to see); or B's coverage advantage over A' is under 0.05.
  ANY OTHER OUTCOME IS "NOT SETTLED — KEEP COLLECTING", explicitly including
  the likeliest one: B ahead of A' on coverage and intervals while its RMSSD
  sits near the TOO-GOOD bar. Nothing in this rule licenses a decision about
  the AFib card; see WHICH CARDS ABOVE.

NEAR-TAUTOLOGY ON ARM B's RATE. fs starts at signal.length / last end_location
and is then REFINED to minimise disagreement with the SAME beat train, so
bpm_B is that train's rate re-expressed and carries no independent rate
information. MEASURED 2026-09-11 by --self-test (experiment 2): with the train
held fixed and only the waveform's TRUE sample rate changed to 15/30/60/120 Hz,
fs0 tracked it (15.153/30.305/60.610/121.221) and arm B read 72.0 bpm every
time against a 72 bpm reference — four different truths, one answer, all
"within +/-5". d_hr is therefore labelled NEAR-TAUTOLOGICAL on the row and
EXCLUDED from the summary's |dHR|<=5 column; the independent quantity, OUR
detected beat count on THEIR waveform over their own beat count (cntR), is
printed in its place — it moved (1.00/1.00/1.02/1.02) where bpm could not.

WHAT THIS IS NOT. There is no ECG, no cuff and no CPET in this repo
(CLAUDE.md:20-40). ShenAI sees the SAME camera under the SAME illuminant, so
its errors are correlated with ours — agreement can be two systems making one
mistake. Precedent on record: 2026-09-09, a phone scan read 100 bpm on the
day's BEST cross-ROI coherence against a reference of 55, because at ~0 dB
per-ROI SNR every region locked onto the same sub-multiple (inference/
evidence.py:69-84). Columns are therefore named shenai/agreement, never
reference/error, and arm C's agreement with reference.heart_rate_bpm is a
TAUTOLOGY (both come from the SDK) and is labelled as one.

THE SAMPLE RATE IS DERIVED, NOT MEASURED. getFullPpgSignal() returns a bare
number[] — no rate, no t0 (webapp/avatarxvitals/index.d.ts:374-377). The
client records how it derived fs in ppg.fs_source; this harness re-derives and
then REFINES by grid search. The drift test is mandatory, not optional:
detect_beats_single_roi hard-codes t_s = index/fps (beats/detector.py:145-150),
so a wrong fs drifts LINEARLY — 1% over 40 s is 400 ms, more than a whole IBI —
and presents as "ShenAI disagrees with its own waveform", a conclusion about us
rather than about them. Every arm-B number is VOID above 2 ms/s of residual
slope. Phase (t0) stays unresolved: a residual MEDIAN of 100-200 ms is EXPECTED
(start_location_sec is a beat ONSET, our detector places the systolic PEAK) —
the median is a convention offset, the IQR is the evidence.

A VOID verdict USED to be terminal, because the +/-2% grid is anchored to a
fs0 that the same sub-span bias already moved: measured 2026-09-11
(--self-test experiment 3), a synthetic 30.000 Hz / 72 bpm / 48 s signal whose
train covered only 4-44 s gave fs0 32.591 -> refined 31.940 at -64.7 ms/s
(true 30.000) — the true rate was OUTSIDE the search, and the wide re-fit lands
at 30.050 with -1.7 ms/s. VOID now re-fits automatically over
FS_WIDE_SEARCH and reports both attempts, and every clip prints the
INDEPENDENT estimate signal.length / (recorder_duration_ms / 1000) beside the
fitted one with their ratio: the ratio IS the sub-span (a fitted/duration ratio
stable at ~1.20 across clips says the train covers ~83% of the waveform, not
that the SDK's clock drifts).

  python scripts/compare_shenai_signal.py                    # the eval corpus
  python scripts/compare_shenai_signal.py clip.webm -v
  python scripts/compare_shenai_signal.py data/eval_corpus --json /tmp/r.json
  python scripts/compare_shenai_signal.py --self-test        # no clips needed

Sidecars are pulled next to their clips by scripts/pull_scan_clips.py.

READ ONLY: never writes to, trims, or deletes any input.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def _skill_common():
    """The optimizer skill's _common, loaded by path (.claude is not a package).

    apply_pipeline_env() must run BEFORE the pipeline is imported: it sets
    AFIB_MAX_COLLAPSED_FRACTION=0.05, production's launch override. Without it
    this harness scores a stricter pipeline than the service runs."""
    p = _REPO / ".claude/skills/cardio-pipeline-optimizer/scripts/_common.py"
    try:
        spec = importlib.util.spec_from_file_location("_cardio_opt_common", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                       # type: ignore[union-attr]
        mod.apply_pipeline_env()
        return mod
    except Exception:                                      # noqa: BLE001
        os.environ.setdefault("AFIB_MAX_COLLAPSED_FRACTION", "0.05")
        return None


_COMMON = _skill_common()

from configs import load_config                                       # noqa: E402
from capture.ingest import ingest_video                               # noqa: E402
from preprocessing.roi import ROI_NAMES                               # noqa: E402
from rppg.pos import pos_pulse                                        # noqa: E402
from rppg.signal_quality import compute_sqi                           # noqa: E402
from beats.detector import Beat, BeatSeries, detect_beats_single_roi, \
    fuse_multi_roi                                                    # noqa: E402
from beats.ibi import clean_runs, rmssd_from_runs, sdnn_from_runs     # noqa: E402
from inference.evidence import extract_and_detect, _welch_psd, \
    _fundamental                                                      # noqa: E402
from inference.decision_logic import harmonic_fraction, \
    timing_precision_from_trains                                      # noqa: E402
from evaluation.beat_metrics import match_beats                       # noqa: E402
from inference.pipeline import CALIBRATED_MIN_CONF, _load_calibrator  # noqa: E402
from scripts.compare_capture import _peak                             # noqa: E402
from app.measure_prep import DEFAULT_SCALE, DEFAULT_WINDOW_S          # noqa: E402

ARMS = ("A", "An", "Ap", "B", "C")
ARM_LABEL = {"A": "A  ours 4-ROI", "An": "An ours 4-ROI nat",
             "Ap": "A' ours forehead",
             "B": "B  shenai wave", "C": "C  shenai train"}
OURS = ("A", "An", "Ap")
# Standing reference points, so a new row can be placed rather than admired.
BENCH = {
    "xROI r, lab rig (demo_*.avi)": "0.330   measured 2026-09-10",
    "xROI r, phone (webapp clips)": "0.161   measured 2026-09-10",
    "coverage_floor (hard gate f2)": "0.50",
    "min_coverage_any / min_intervals_any": "0.60 / 15",
    "max_timing_precision_ms_any": "40 ms   (refs 1-5 ms; real FP scans 36-49)"
                                   " — INDEPENDENT ROI PAIRS only",
    "pulse agreement bar (CLAUDE.md)": "+/-5 bpm on >=80% of scans",
    "ACCEPT rate on record": "1 scan in 49",
}
MAX_SLOPE_MS_PER_S = 2.0      # above this arm B is drift, not signal
FS_WIDE_SEARCH = (0.80, 1.25)  # the VOID re-fit: a train over a sub-span moves
                               # fs0 by the sub-span RATIO, which the +/-2%
                               # grid cannot reach (measured 2026-09-11: fs0
                               # 32.493 for a true 30.000, a +8.3% bias)
# TOO-GOOD is the detector for the failure that would cost the most: an SDK
# waveform regularised by its 3D face model raises coverage, intervals and
# split-fraction here while destroying AFib sensitivity. The old bars (10 / 15
# ms) sat FAR below normal short-term sinus HRV (RMSSD 20-50 ms, SDNN 30-60 ms;
# Nunan 2010 normative review, and this repo's own reference scans read RMSSD
# 25-45 ms), so moderate smoothing — halving a 40 ms RMSSD to 20 — passed
# unflagged. Raised 2026-09-11 to the FLOOR of the normal range: below it, a
# resting adult's beat-to-beat variability has been removed by something. This
# is a caveat flag, not a gate; it will occasionally fire on a genuinely
# low-HRV subject, which is the correct trade for catching smoothing at all.
TOO_GOOD_RMSSD_MS = 20.0
TOO_GOOD_SDNN_MS = 30.0
# Alignment permutation null (see _align).
NULL_DRAWS = 200
NULL_PCTL = 95.0
NULL_SEED = 20260911


# ------------------------------------------------------------------ sidecar
def _sidecar_for(clip: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(clip) + ".shenai.json")


def _ppg(doc: dict) -> np.ndarray:
    """The waveform with nulls as NaN — a null is a GAP, never a zero."""
    sig = ((doc.get("ppg") or {}).get("signal")) or []
    return np.array([np.nan if v is None else float(v) for v in sig], float)


def _beat_starts(doc: dict) -> np.ndarray:
    hb = doc.get("heartbeats") or []
    t = [b.get("start_location_sec") for b in hb if isinstance(b, dict)]
    return np.sort(np.array([float(x) for x in t if x is not None], float))


def _beat_span_s(doc: dict) -> float:
    hb = doc.get("heartbeats") or []
    if len(hb) < 2:
        return float("nan")
    try:
        return float(hb[-1]["end_location_sec"]) - float(hb[0]["start_location_sec"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _start_fs(doc: dict, sig: np.ndarray) -> tuple[float, str]:
    """(fs, source) before refinement. Mirrors the client's rules so a sidecar
    written by an older client still yields the same starting point."""
    ppg = doc.get("ppg") or {}
    fs = ppg.get("fs_hz")
    try:
        fs = float(fs)
    except (TypeError, ValueError):
        fs = float("nan")
    if np.isfinite(fs) and 1.0 < fs < 1000.0:
        return fs, str(ppg.get("fs_source") or "sdk")
    hb = doc.get("heartbeats") or []
    if sig.size and len(hb) >= 2:
        try:
            end = float(hb[-1]["end_location_sec"])
            if end >= 5.0:
                return sig.size / end, "derived_from_beats (recomputed)"
        except (KeyError, TypeError, ValueError):
            pass
    dur = (doc.get("measurement") or {}).get("recorder_duration_ms")
    try:
        dur = float(dur) / 1000.0
    except (TypeError, ValueError):
        dur = 0.0
    if sig.size and dur > 5.0:
        return sig.size / dur, "derived_from_duration (recomputed)"
    return float("nan"), "unknown"


# ------------------------------------------------------- fs + drift (arm B)
def _finite_runs(sig: np.ndarray, fs: float) -> list[tuple[int, int]]:
    """Contiguous finite stretches long enough to detect on (2*fs samples,
    the detector's own floor). Nulls split the signal exactly as a capture
    hole splits ours (inference/evidence.capture_segments)."""
    if not np.isfinite(fs) or fs <= 0 or sig.size == 0:
        return []
    ok = np.isfinite(sig)
    out, i, n = [], 0, sig.size
    need = int(2 * fs)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= max(need, 4):
            out.append((i, j))
        i = j
    return out


def _shen_span(doc: dict, fs: float) -> float:
    """ShenAI's own HOLE-FREE span: the waveform's finite stretches when the
    rate is known, else the beat train's span. Every coverage number is printed
    against this AND against our capture span, because the two are not the same
    footing — the retained clip can carry a real hole (the recorder keeps a
    rolling 48 s window by dropping MIDDLE chunks, MediaRecorderProcessor.js:
    184-204) that capture_segments splits on and ShenAI never saw."""
    runs = _finite_runs(_ppg(doc), fs)
    if runs:
        return float(sum((b - a) / fs for a, b in runs))
    return _beat_span_s(doc)


def _detect_wave(sig: np.ndarray, fs: float) -> list[Beat]:
    """Our detector on one dense waveform, gap-aware, times in seconds."""
    beats: list[Beat] = []
    for a, b in _finite_runs(sig, fs):
        for bt in detect_beats_single_roi(sig[a:b], fs, "shenai"):
            bt.t_s = float(bt.t_s + a / fs)
            beats.append(bt)
    return beats


def _residuals(det_t: np.ndarray, ref_t: np.ndarray) -> np.ndarray:
    """Signed detected-minus-nearest-reference seconds."""
    if det_t.size == 0 or ref_t.size < 2:
        return np.array([])
    idx = np.clip(np.searchsorted(ref_t, det_t), 1, ref_t.size - 1)
    left, right = ref_t[idx - 1], ref_t[idx]
    near = np.where(np.abs(det_t - left) <= np.abs(det_t - right), left, right)
    return det_t - near


def _unwrap(d: np.ndarray, ref_t: np.ndarray) -> np.ndarray:
    """Residuals with the nearest-neighbour WRAP removed.

    Measured on a phase-locked fixture (2026-09-11): nearest-neighbour
    residuals saw-tooth once the drift passes half an IBI, so a 10%-wrong fs
    fitted a slope of 0.74 ms/s (well under the 2 ms/s VOID bar) while a
    CONSISTENT fixture fitted -3.09 ms/s. The drift test was reading noise in
    both directions. Unwrapping at the beat period restores the ramp: the same
    two cases then read -98.1 ms/s (VOID, correctly) and +0.16 ms/s."""
    if d.size < 3 or ref_t.size < 3:
        return d
    period = float(np.median(np.diff(ref_t)))
    if not np.isfinite(period) or period <= 0:
        return d
    return np.unwrap(d, period=period)


def _refine_fs(sig: np.ndarray, fs0: float, ref_t: np.ndarray,
               lo: float, hi: float, steps: int) -> dict:
    """Grid-search fs minimising the SPREAD of |our beat - nearest ShenAI beat|.

    Deviation from the contract's "minimise the median", with its reason: the
    median IS the onset-vs-peak convention offset (100-200 ms), not error, so
    minimising it optimises the wrong quantity — on the phase-locked fixture it
    picked 27.603 Hz for a true 27.5 and manufactured a -3.1 ms/s slope, while
    minimising the deviation ABOUT the median lands on 27.503. The median is
    still reported; it is simply not what the search chases.

    Two-stage when the direct grid would be too expensive: the detector is a
    per-sample Python loop, so 200 passes over a 200k-sample signal is tens of
    millions of iterations. Coarse-then-fine reaches the same resolution in ~42
    passes; which one ran is printed, because a coarser search is a weaker
    claim about fs."""
    out = {"fs": fs0, "search": "none", "resid_med_ms": float("nan"),
           "resid_iqr_ms": float("nan"), "slope_ms_per_s": float("nan"),
           "n_det": 0}
    if not np.isfinite(fs0) or fs0 <= 0 or ref_t.size < 5 or sig.size == 0:
        return out

    def cost(fs: float) -> tuple[float, np.ndarray]:
        t = np.sort(np.array([b.t_s for b in _detect_wave(sig, fs)], float))
        d = _unwrap(_residuals(t, ref_t), ref_t)
        if d.size < 5:
            return float("inf"), t
        return float(np.median(np.abs(d - np.median(d)))), t

    budget = 4_000_000
    if sig.size * max(steps, 1) <= budget:
        grid = np.linspace(fs0 * lo, fs0 * hi, max(int(steps), 2))
        mode = f"grid {len(grid)}"
    else:
        coarse = np.linspace(fs0 * lo, fs0 * hi, 21)
        best_c = min(coarse, key=lambda f: cost(f)[0])
        half = (fs0 * hi - fs0 * lo) / 20.0
        grid = np.linspace(best_c - half, best_c + half, 21)
        mode = "grid 21+21 (coarse-to-fine; signal too long for a full sweep)"

    best_fs, best_cost, best_t = fs0, float("inf"), np.array([])
    for fs in grid:
        c, t = cost(float(fs))
        if c < best_cost:
            best_fs, best_cost, best_t = float(fs), c, t
    out["fs"], out["search"], out["n_det"] = best_fs, mode, int(best_t.size)

    t_b = np.sort(best_t)
    d = _unwrap(_residuals(t_b, ref_t), ref_t)
    if d.size >= 5:
        # The median is the CONVENTION offset (their beat onset vs our systolic
        # peak); the detrended IQR is the evidence about timing.
        out["resid_med_ms"] = float(np.median(d)) * 1000.0
        slope, icept = np.polyfit(t_b, d * 1000.0, 1)
        resid = d * 1000.0 - (slope * t_b + icept)
        q1, q3 = np.percentile(resid, [25, 75])
        out["resid_iqr_ms"] = float(q3 - q1)
        out["slope_ms_per_s"] = float(slope)
    return out


# ----------------------------------------------------------------- metrics
def _spectral(sig, fs) -> tuple[float, float]:
    try:
        f, p = _welch_psd(sig, fs)
        if f is None:
            return float("nan"), float("nan")
        f0, snr = _fundamental(f, p)
        if f0 is None:
            return float("nan"), float("nan")
        return float(f0 * 60.0), float(snr)
    except Exception:                                      # noqa: BLE001
        return float("nan"), float("nan")


def _row(arm, rs, *, wave=None, fps=float("nan"), n_raw=0, n_fused=0,
         cap_ours=float("nan"), cap_shen=float("nan"), ref_hr=None,
         notes=()) -> dict:
    """One (clip, arm) record. Coverage reproduces inference/pipeline.py:194-201
    exactly; BOTH denominators are carried because ours is the gap-split capture
    span while ShenAI's is its own hole-free span — different footings."""
    ibi = rs.all_intervals()
    analysed_s = float(sum(float(np.sum(r)) for r in rs.runs) / 1000.0)
    snr_db, _ = _peak(wave, fps) if wave is not None and np.isfinite(fps) \
        else (float("nan"), float("nan"))
    sbpm, ssnr = _spectral(wave, fps) if wave is not None and np.isfinite(fps) \
        else (float("nan"), float("nan"))
    bpm = 60000.0 / float(np.median(ibi)) if ibi.size else float("nan")
    rmssd, sdnn = rmssd_from_runs(rs), sdnn_from_runs(rs)
    flags = list(notes)
    if np.isfinite(rmssd) and rmssd < TOO_GOOD_RMSSD_MS:
        flags.append(f"TOO-GOOD rmssd {rmssd:.0f}ms (suspected smoothing)")
    if np.isfinite(sdnn) and sdnn < TOO_GOOD_SDNN_MS:
        flags.append(f"TOO-GOOD sdnn {sdnn:.0f}ms (suspected smoothing)")
    gates = []
    cov = analysed_s / cap_ours if np.isfinite(cap_ours) and cap_ours > 0 \
        else float("nan")
    cov_s = analysed_s / cap_shen if np.isfinite(cap_shen) and cap_shen > 0 \
        else float("nan")
    best_cov = cov_s if (arm in ("B", "C") and np.isfinite(cov_s)) else cov
    if np.isfinite(best_cov) and best_cov < 0.50:
        gates.append("coverage<0.50")
    elif np.isfinite(best_cov) and best_cov < 0.60:
        gates.append("coverage<0.60")
    if rs.n_intervals < 15:
        gates.append("intervals<15")
    d_hr = float("nan")
    if ref_hr is not None and np.isfinite(bpm):
        try:
            d_hr = bpm - float(ref_hr)
        except (TypeError, ValueError):
            d_hr = float("nan")
    return {
        "arm": arm, "ok": True, "reason": "",
        "snr_db": snr_db, "spec_bpm": sbpm, "spec_snr": ssnr,
        "n_raw": int(n_raw), "n_fused": int(n_fused), "kept": int(rs.kept_beats),
        "bpm": bpm, "d_hr": d_hr,
        "d_hr_ok": bool(np.isfinite(d_hr) and abs(d_hr) <= 5.0),
        "n_intervals": rs.n_intervals, "n_runs": rs.n_runs,
        "longest_run": rs.longest_run, "dropout": rs.dropout_rate,
        "split": rs.split_fraction, "analysed_s": analysed_s,
        "cov_ours": cov, "cov_shen": cov_s,
        "harmonic": harmonic_fraction(ibi) if ibi.size else float("nan"),
        "rmssd_ms": rmssd, "sdnn_ms": sdnn,
        "gates_failed": gates, "flags": flags,
    }


def _blank(arm: str, reason: str) -> dict:
    return {"arm": arm, "ok": False, "reason": reason, "gates_failed": [],
            "flags": []}


# -------------------------------------------------------------------- arms
def _ours(clip: str, cfg: dict, cal, runs_kw: dict, profile: str, *,
          ref_hr=None, cap_shen: float = float("nan"),
          arm_a: str = "A", arm_ap: str = "Ap", a_note: str = "") -> dict:
    """Arms A and A' share one ingest+extraction: A' must differ from A ONLY in
    the fusion it pays for, or it is not a control. Calls ingest_video, never
    measure_video — measure_video trims and downscales IN PLACE and unlinks its
    input (a 2026-09-09 diagnostic destroyed eight corpus recordings that way).

    Run twice per clip with different arm names: once production-configured
    (arm A) and once at the clip's native resolution (arm An). See THE CODEC
    CONFOUND in the module docstring for why the pair is not optional."""
    ing = ingest_video(clip, capture_profile=profile)
    if not ing.ok:
        why = "; ".join(ing.reasons) or "ingest failed"
        return {arm_a: _blank(arm_a, why), arm_ap: _blank(arm_ap, why),
                "ctx": {}}
    fps = ing.meta.measured_fps_mean
    ts = np.asarray(ing.timestamps_s, float)
    raw, per_roi_beats, segments = extract_and_detect(ing.traces, ts, fps, cfg)
    cap = float(sum(ts[b - 1] - ts[a] for a, b in segments)) if segments \
        else float(ing.meta.duration_s)
    cap = max(cap, 1e-6)

    # Filtered per-ROI waveforms for the SNR column ONLY, built exactly as
    # scripts/compare_capture.py builds them, so the dB sits beside the
    # 0.330 / 0.161 bench instead of on a private scale.
    waves = {}
    for r in ROI_NAMES:
        a = np.asarray(ing.traces.get(r, []), float)
        if a.ndim == 2 and a.shape[0] >= 256:
            waves[r] = pos_pulse(a, fps)
    snr_med = float(np.median([s for s in (_peak(w, fps)[0] for w in waves.values())
                               if np.isfinite(s)] or [np.nan]))

    fused = fuse_multi_roi(per_roi_beats, fps, ing.meta.duration_s, min_rois=2)
    series = cal.apply(fused)
    rs = clean_runs(series, **runs_kw)
    sqi = compute_sqi(raw, fps, fused, tracking_stability=ing.track.stability)
    a_row = _row(arm_a, rs, wave=None, fps=fps,
                 n_raw=sum(len(v) for v in per_roi_beats.values()),
                 n_fused=len(fused.beats), cap_ours=cap, cap_shen=cap_shen,
                 ref_hr=ref_hr)
    a_row["snr_db"] = snr_med
    sb, ss = _spectral(waves.get("forehead"), fps)
    a_row["spec_bpm"], a_row["spec_snr"] = sb, ss
    a_row["coherence"] = float(sqi.components.get("cross_roi_coherence", np.nan))
    a_row["sqi"] = float(sqi.sqi)
    if a_row["coherence"] < 0.20:
        a_row["gates_failed"].append("coherence<0.20")
    a_row["flags"].append(a_note or "calibrated confidence (production path)")

    fh = per_roi_beats.get("forehead", [])
    fused_fh = fuse_multi_roi({"forehead": fh}, fps, ing.meta.duration_s,
                              min_rois=1)
    rs_fh = clean_runs(fused_fh, **runs_kw)
    ap_row = _row(arm_ap, rs_fh, wave=waves.get("forehead"), fps=fps,
                  n_raw=len(fh), n_fused=len(fused_fh.beats), cap_ours=cap,
                  cap_shen=cap_shen, ref_hr=ref_hr)
    ap_row["flags"].append("UNCALIBRATED single-train confidence "
                           "(= peak width alone); coherence unevaluable")
    # Inter-segment holes, in seconds. The recorder keeps chunks[0] plus the
    # last (maxChunks-1) and DROPS THE MIDDLE (MediaRecorderProcessor.js:
    # 184-204); staging runs maxDurationMs 48000 / timesliceMs 3000 => 16
    # chunks, so ANY scan over ~48 s lands a clip whose container clock has a
    # hole of minutes (a real scan recorded 277 s). capture_segments splits on
    # it; alignment must refuse it rather than fit one scalar offset across it.
    gaps = [float(ts[segments[k + 1][0]] - ts[segments[k][1] - 1])
            for k in range(len(segments) - 1)]
    ctx = {"fps": fps, "cap_ours": cap, "clock": "container (retained clips "
           "carry no .timestamps.json)", "our_times": series.times(),
           "n_segments": len(segments), "seg_gaps_s": gaps,
           "duration_s": float(ing.meta.duration_s),
           "res": f"{ing.meta.width}x{ing.meta.height}",
           "caveats": list(ing.capture_caveats or [])}
    return {arm_a: a_row, arm_ap: ap_row, "ctx": ctx}


def _shenai(doc: dict, fsinfo: dict, runs_kw: dict, cap_ours: float) -> dict:
    """Arms B and C from the sidecar. Both are uncalibrated by construction and
    say so: the calibration map was fitted on FOUR-ROI fused confidences and is
    out of domain on a single train."""
    sig, fs = _ppg(doc), fsinfo["fs"]
    ref_t = _beat_starts(doc)
    ref_hr = (doc.get("reference") or {}).get("heart_rate_bpm")
    runs = _finite_runs(sig, fs)
    span_b = _shen_span(doc, fs)
    out = {}

    if sig.size == 0:
        out["B"] = _blank("B", "sidecar carries no waveform")
    elif not np.isfinite(fs):
        out["B"] = _blank("B", "no plausible sample rate (fs_source unknown)")
    elif not runs:
        out["B"] = _blank("B", f"no finite stretch >= 2/fs ({sig.size} samples)")
    else:
        beats = _detect_wave(sig, fs)
        fused = fuse_multi_roi({"shenai": beats}, fs, span_b, min_rois=1)
        rs = clean_runs(fused, **runs_kw)
        notes = ["UNCALIBRATED single-train confidence; coherence and "
                 "cross-ROI timing unevaluable",
                 "SNR not comparable to arm A: the SDK's waveform arrives "
                 "band-limited, so out-of-band energy was removed for it"]
        slope = fsinfo.get("slope_ms_per_s", float("nan"))
        if np.isfinite(slope) and abs(slope) > MAX_SLOPE_MS_PER_S:
            notes.append(f"VOID: residual drifts {slope:+.1f} ms/s — fs is "
                         "wrong (or the beat train covers a sub-span)")
        notes.append("d_hr is NEAR-TAUTOLOGICAL: fs was derived from this "
                     "train and then REFINED against it, so bpm carries no "
                     "independent rate information (--self-test experiment 2, "
                     "2026-09-11: true fs 15/30/60/120 Hz ALL read 72.0 bpm "
                     "against a 72 bpm reference). Excluded from the summary's "
                     "|dHR|<=5 column; cntR is the independent quantity.")
        out["B"] = _row("B", rs, wave=sig[runs[0][0]:runs[0][1]], fps=fs,
                        n_raw=len(beats), n_fused=len(fused.beats),
                        cap_ours=cap_ours, cap_shen=span_b, ref_hr=ref_hr,
                        notes=notes)
        out["B"]["void"] = bool(np.isfinite(slope) and
                                abs(slope) > MAX_SLOPE_MS_PER_S)
        out["B"]["times"] = fused.times()
        # THE independent rate quantity on arm B: how many beats OUR detector
        # finds in THEIR waveform against how many THEY declare. Unlike bpm it
        # is not re-expressed through the fitted fs — it is a count against a
        # count — so it can disagree, which is the whole point.
        out["B"]["count_ratio"] = (len(beats) / ref_t.size
                                   if ref_t.size else float("nan"))
        out["B"]["count_ratio_ok"] = bool(
            ref_t.size and abs(len(beats) / ref_t.size - 1.0) <= 0.10)
        out["B"]["d_hr_ok"] = False      # never counted: see the note above

    if ref_t.size < 2:
        out["C"] = _blank("C", "sidecar carries no beat train")
    else:
        span_c = _beat_span_s(doc)
        beats = [Beat(t_s=float(t), confidence=1.0, roi_agreement=1.0,
                      signal_quality=1.0, amplitude=float("nan"),
                      prominence=float("nan"), source_rois=["shenai"])
                 for t in ref_t]
        rs = clean_runs(BeatSeries(beats, fs if np.isfinite(fs) else 0.0,
                                   span_c), **runs_kw)
        out["C"] = _row("C", rs, wave=None, fps=float("nan"),
                        n_raw=len(beats), n_fused=len(beats),
                        cap_ours=cap_ours, cap_shen=span_c, ref_hr=ref_hr,
                        notes=["confidence SYNTHESISED (the SDK exposes none) "
                               "— clean_runs' confidence channel is inert here; "
                               "coherence and cross-ROI timing unevaluable",
                               "agreement with reference.heart_rate_bpm is a "
                               "TAUTOLOGY: both come from ShenAI — excluded "
                               "from the summary's |dHR|<=5 column"])
        out["C"]["times"] = ref_t
        out["C"]["d_hr_ok"] = False      # tautology; counting it would inflate
    return out


# --------------------------------------------------------------- alignment
def _match_curve(our_t: np.ndarray, shen_t: np.ndarray, grid: np.ndarray,
                 tol_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(matched, denom) per candidate offset, vectorised over the whole grid.

    matched[k] counts ShenAI beats with one of OUR beats within tol of
    shen + grid[k]; denom[k] is how many ShenAI beats lie INSIDE the overlap at
    that offset. Overlap-normalised, because a far offset that slides the
    trains apart otherwise looks "unmatchable" for a reason that has nothing to
    do with timing, which would deflate the null and flatter the observation.
    The fraction is "what share of THEIR beats we placed within tol" — it
    cannot exceed 1, and a detector that fires constantly scores well on it,
    which is precisely what the null below is for.

    Nearest-neighbour, one count per ShenAI beat. match_beats' greedy
    one-to-one matching is slightly stricter, but the printed agreement
    numbers are the only place that precision matters; here the SAME
    estimator must run on the observed offset and on every null draw or the
    comparison is not like-for-like."""
    our_t = np.sort(np.asarray(our_t, float))
    shen_t = np.sort(np.asarray(shen_t, float))
    g = np.asarray(grid, float)
    if our_t.size == 0 or shen_t.size == 0 or g.size == 0:
        return np.zeros(g.size, int), np.zeros(g.size, int)
    matched = np.zeros(g.size, int)
    for t in shen_t:                       # loop the SHORTER train (~57 beats)
        x = t + g
        j = np.searchsorted(our_t, x)
        lo = our_t[np.clip(j - 1, 0, our_t.size - 1)]
        hi = our_t[np.clip(j, 0, our_t.size - 1)]
        matched += (np.minimum(np.abs(x - lo), np.abs(x - hi)) <= tol_s)
    # Overlap on ShenAI's clock, widened by tol at both ends: a beat that sits
    # one tolerance outside the shared span can still match, and counting the
    # match while excluding it from the denominator puts the fraction over 1.
    lo_t = np.maximum(our_t[0] - g, shen_t[0]) - tol_s
    hi_t = np.minimum(our_t[-1] - g, shen_t[-1]) + tol_s
    n_shen = (np.searchsorted(shen_t, hi_t, "right")
              - np.searchsorted(shen_t, lo_t, "left"))
    return matched, np.maximum(n_shen, 0)


def _null_p95(our_t: np.ndarray, shen_t: np.ndarray, off: float, tol_s: float,
              n_draws: int, seed: int) -> dict:
    """The matched fraction a WRONG offset reaches on these two trains.

    WHY THIS GUARD EXISTS. The old rule — best of ~1000 candidate offsets,
    then "alignable" at matched/min(n) >= 0.5 — has no null and does not
    survive one. With ~57 beats at ~0.85 s a +/-100 ms window covers ~24% of
    the cycle BY CHANCE, and maximising over 1000 offsets reliably clears 0.5.
    MEASURED against the shipped function 2026-09-11: two trains at the same
    rate with a deliberately destroyed timebase relation were declared
    ALIGNABLE in 54/60 = 90% of trials at a median |dt| of 36 ms — under the
    40 ms ceiling printed two lines later — and 3/40 PURE-NOISE pairs
    (independently drawn rates) aligned too.

    Two nulls, both of the SAME statistic (max over an equal-width offset
    window), because a max must be compared against a max — and the observed
    value is taken at the ONE selected offset, which is strictly conservative:
      perm  shuffle ShenAI's OWN interval series and keep its start. Rate and
            interval distribution survive; the timebase relation does not.
            THIS IS THE GATE: beating it is the claim "these two trains share a
            timebase", which is exactly what the old rule asserted without
            evidence.
      far   the observed curve at offsets outside the plausible window, in
            windows of the same +/-5 s width. This is a CAVEAT, not a gate: on
            a quasi-periodic train a whole-beat alias (offset +/- k*IBI) fits
            nearly as well — measured 2026-09-11, a fixture whose ShenAI train
            WAS our train shifted by 2 s (with 15% of our beats dropped and
            30 ms of detector jitter) passed the perm gate 30/30 but failed the
            far null in 12/30. Gating on it would throw away 40% of genuine
            alignments to restate a known property of periodic signals, so it
            is printed as "offset identified only up to +/-k beats" instead."""
    out = {"perm_p95": float("nan"), "far_p95": float("nan"),
           "p95": float("nan"), "n_perm": 0, "n_far": 0}
    our_t = np.sort(np.asarray(our_t, float))
    shen_t = np.sort(np.asarray(shen_t, float))
    if our_t.size < 5 or shen_t.size < 5:
        return out
    rng = np.random.default_rng(seed)
    min_den = max(10, int(0.25 * min(our_t.size, shen_t.size)))
    grid = np.arange(off - 5.0, off + 5.0 + 1e-9, 0.010)

    def best(matched, denom, mask=None) -> float:
        ok = denom >= min_den
        if mask is not None:
            ok &= mask
        if not np.any(ok):
            return float("nan")
        return float(np.max(matched[ok] / denom[ok]))

    ibi = np.diff(shen_t)
    if ibi.size >= 4:
        vals = []
        for _ in range(n_draws):
            perm = shen_t[0] + np.cumsum(rng.permutation(ibi))
            vals.append(best(*_match_curve(our_t, perm, grid, tol_s)))
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out["perm_p95"] = float(np.percentile(vals, NULL_PCTL))
            out["n_perm"] = len(vals)

    # Far window: every offset with real overlap, minus the plausible +/-5 s.
    g0 = shen_t[0] - our_t[-1]
    g1 = shen_t[-1] - our_t[0]
    full = np.arange(g0, g1 + 1e-9, 0.010)
    if full.size > 1200:                       # else there is no "elsewhere"
        matched, denom = _match_curve(our_t, shen_t, full, tol_s)
        far = (np.abs(full - off) > 5.0) & (denom >= min_den)
        idx = np.flatnonzero(far)
        if idx.size > 1000:
            vals = []
            for _ in range(n_draws):
                c = int(rng.choice(idx))
                a, b = max(c - 500, 0), min(c + 501, full.size)
                v = best(matched[a:b], denom[a:b], far[a:b])
                if np.isfinite(v):
                    vals.append(v)
            if vals:
                out["far_p95"] = float(np.percentile(vals, NULL_PCTL))
                out["n_far"] = len(vals)
    out["p95"] = out["perm_p95"]           # the gate; far_p95 is the caveat
    return out


def _align(our_t: np.ndarray, shen_t: np.ndarray, offset0: float,
           tol_ms: float, *, refuse: str = "", n_null: int = NULL_DRAWS,
           seed: int = NULL_SEED) -> dict:
    """ShenAI's clock against the clip's container clock, null test included.

    estimate_ptt_ms's default search window is (0, 500) ms — far too narrow for
    an offset measured in seconds — so the coarse offset is found here (+/-5 s
    at 10 ms) and handed to match_beats as an explicit ptt_ms. Nothing is
    reported unless the matched fraction at that offset beats the 95th
    percentile of _null_p95: an agreement number that a wrong offset also
    reaches is not evidence about timing, and printing it invites a timing
    conclusion the data cannot carry."""
    out = {"offset_s": float("nan"), "n_matched": 0, "unalignable": True,
           "matched_frac": float("nan"), "null_p95": float("nan"),
           "null_perm_p95": float("nan"), "null_far_p95": float("nan"),
           "n_null": 0}
    if refuse:
        out["reason"] = refuse
        return out
    our_t = np.sort(np.asarray(our_t, float))
    shen_t = np.sort(np.asarray(shen_t, float))
    if our_t.size < 5 or shen_t.size < 5:
        out["reason"] = "fewer than 5 beats on one side"
        return out
    tol_s = float(tol_ms) / 1000.0
    grid = np.arange(offset0 - 5.0, offset0 + 5.0 + 1e-9, 0.010)
    costs = [np.median(np.abs(_residuals(our_t - g, shen_t))) for g in grid]
    off = float(grid[int(np.argmin(costs))])
    out["offset_s"] = off

    m_obs, d_obs = _match_curve(our_t, shen_t, np.array([off]), tol_s)
    denom = int(d_obs[0])
    frac = float(m_obs[0]) / denom if denom > 0 else float("nan")
    out["matched_frac"], out["overlap_beats"] = frac, denom
    nul = _null_p95(our_t, shen_t, off, tol_s, n_null, seed)
    out["null_p95"], out["null_perm_p95"] = nul["p95"], nul["perm_p95"]
    out["null_far_p95"], out["n_null"] = nul["far_p95"], nul["n_perm"]
    out["alias_ambiguous"] = bool(np.isfinite(nul["far_p95"]) and
                                  np.isfinite(frac) and frac <= nul["far_p95"])

    for tol in sorted({50.0, float(tol_ms)}):
        m = match_beats(shen_t, our_t, tolerance_ms=tol, ptt_ms=off * 1000.0)
        errs = np.abs(m.timing_errors_ms)
        key = f"t{int(tol)}"
        out[key] = {
            "n_matched": int(m.n_matched), "sensitivity": float(m.sensitivity),
            "ppv": float(m.ppv),
            "median_abs_dt_ms": float(np.median(errs)) if errs.size else float("nan"),
            "iqr_ms": float(np.percentile(errs, 75) - np.percentile(errs, 25))
            if errs.size else float("nan"),
        }
        if tol == float(tol_ms):
            out["n_matched"] = int(m.n_matched)
    if denom < 10:
        out["reason"] = f"only {denom} beats overlap at the best offset"
    elif not np.isfinite(frac) or frac < 0.5:
        out["reason"] = (f"matched {frac:.2f} of the overlap, under half"
                         if np.isfinite(frac) else "no overlap")
    elif not np.isfinite(nul["p95"]):
        out["reason"] = "the alignment null could not be built (too few beats)"
    elif frac <= nul["p95"]:
        out["reason"] = (f"matched {frac:.2f} does not beat its own "
                         f"permutation null p95 {nul['p95']:.2f} over "
                         f"{nul['n_perm']} draws — a train with the same rate "
                         f"and no timebase relation reaches the same agreement")
    else:
        out["unalignable"] = False
    return out


def _fit_fs(doc: dict, sig: np.ndarray, ref_t: np.ndarray, fs0: float,
            fs_src: str, opt) -> dict:
    """The fitted sample rate, its drift verdict, and the INDEPENDENT
    duration-derived rate beside it. Split out of measure() so --self-test can
    drive it on a synthetic signal whose true rate is known."""
    if getattr(opt, 'fs', None):
        # A forced rate still gets the drift test — MORE so, since forcing the
        # wrong one is exactly how arm B turns into a linear ramp. Only the
        # search is skipped (lo = hi = 1.0 evaluates the single point).
        fsinfo = _refine_fs(sig, float(opt.fs), ref_t, 1.0, 1.0, 2)
        fsinfo["search"], fs_src = "forced", "forced"
    else:
        fsinfo = _refine_fs(sig, fs0, ref_t, opt.fs_search[0], opt.fs_search[1],
                            opt.fs_steps)
        # A VOID verdict from a +/-2% grid anchored to a biased fs0 is not a
        # finding, it is a search that could not reach the answer: measured
        # 2026-09-11 (--self-test experiment 3) on a synthetic 30.000 Hz /
        # 72 bpm / 48 s signal whose train covered 4-44 s, fs0 came out 32.591
        # and the grid's own edge (31.940) was still 6% high at -64.7 ms/s.
        # Re-fit WIDE before reporting VOID; keep the narrow attempt so the
        # two can be compared — a wide fit that does NOT improve the slope is
        # evidence the drift is real and the narrow verdict stands.
        slope = fsinfo.get("slope_ms_per_s", float("nan"))
        if np.isfinite(slope) and abs(slope) > MAX_SLOPE_MS_PER_S:
            wide = _refine_fs(sig, fs0, ref_t, FS_WIDE_SEARCH[0],
                              FS_WIDE_SEARCH[1], max(int(opt.fs_steps), 40))
            fsinfo["narrow"] = {k: fsinfo.get(k) for k in
                                ("fs", "slope_ms_per_s", "resid_iqr_ms",
                                 "search")}
            w_slope = wide.get("slope_ms_per_s", float("nan"))
            better = (np.isfinite(w_slope) and
                      (not np.isfinite(slope) or abs(w_slope) < abs(slope)))
            if better:
                wide["search"] = (f"{wide['search']} WIDENED "
                                  f"{FS_WIDE_SEARCH[0]:.2f}-"
                                  f"{FS_WIDE_SEARCH[1]:.2f} after VOID")
                fsinfo = {**wide, "narrow": fsinfo["narrow"]}
            else:
                fsinfo["search"] += (f" (+wide {FS_WIDE_SEARCH[0]:.2f}-"
                                     f"{FS_WIDE_SEARCH[1]:.2f} re-fit: no "
                                     f"improvement, slope {w_slope:+.1f} ms/s)")
    # The INDEPENDENT rate estimate — samples over the RECORDER's duration,
    # which knows nothing about the beat train. Until 2026-09-11 it was only a
    # last-resort fallback inside _start_fs and never a check. Two ratios, two
    # different jobs:
    #   fitted/duration   VALIDATOR. ~1.00 means the grid search landed on the
    #                     rate an independent estimate also gives; far from
    #                     1.00 means one of the two is wrong and arm B is not
    #                     safe to read.
    #   fs0/duration      THE SUB-SPAN. fs0 is samples over the TRAIN's end, so
    #                     this ratio IS the fraction of the waveform the train
    #                     covers, inverted (1.09 => the train ends at ~92% of
    #                     the recording). Stable across clips => a structural
    #                     sub-span in the SDK's output, not a drifting clock.
    dur_ms = (doc.get("measurement") or {}).get("recorder_duration_ms")
    try:
        fs_dur = sig.size / (float(dur_ms) / 1000.0)
    except (TypeError, ValueError, ZeroDivisionError):
        fs_dur = float("nan")
    fsinfo["fs_duration"] = float(fs_dur)
    ok_dur = bool(np.isfinite(fs_dur) and fs_dur > 0)
    fsinfo["fs_ratio"] = (float(fsinfo["fs"]) / fs_dur if ok_dur and
                          np.isfinite(fsinfo.get("fs", float("nan")))
                          else float("nan"))
    fsinfo["fs0_ratio"] = (float(fs0) / fs_dur if ok_dur and np.isfinite(fs0)
                           else float("nan"))
    fsinfo["fs0"], fsinfo["fs_source"] = fs0, fs_src
    return fsinfo


# ------------------------------------------------------------------ driver
def measure(clip: pathlib.Path, sidecar: pathlib.Path, cfg, cal, opt) -> dict:
    doc = json.loads(sidecar.read_text())
    if not isinstance(doc, dict):
        raise ValueError("sidecar is not a JSON object")
    sig = _ppg(doc)
    ref_t = _beat_starts(doc)
    fs0, fs_src = _start_fs(doc, sig)
    fsinfo = _fit_fs(doc, sig, ref_t, fs0, fs_src, opt)

    rec = {"clip": clip.name, "sidecar": sidecar.name,
           "schema_version": doc.get("schema_version"),
           "upload_id": doc.get("upload_id"), "session": doc.get("session"),
           "captured_at": doc.get("captured_at"),
           "ppg": {k: (doc.get("ppg") or {}).get(k)
                   for k in ("n", "fs_hz", "fs_source", "truncated",
                             "dropped_head_samples")},
           "n_samples": int(sig.size), "n_beats": int(ref_t.size),
           "n_null_samples": int(np.sum(~np.isfinite(sig))),
           "measurement": doc.get("measurement") or {}, "fs": fsinfo, "arms": {}}

    ref_hr = (doc.get("reference") or {}).get("heart_rate_bpm")
    cap_shen = _shen_span(doc, fsinfo["fs"])

    def run(scale: str, window_s: float, *, arm_a: str, arm_ap: str,
            note: str, prov: str) -> dict:
        """One prep+ingest+extraction pass on a COPY. measure_prep rewrites
        IN PLACE and measure_video unlinks its input, so neither is ever
        pointed at `clip` itself."""
        work, tmp = str(clip), None
        try:
            if scale != "off" or window_s > 0:
                tmp = tempfile.mkdtemp(prefix="shenai_cmp_")
                work = str(pathlib.Path(tmp) / clip.name)
                shutil.copy(str(clip), work)
                from app.measure_prep import downscale, trim_tail
                if window_s > 0:
                    rec[prov + "trim"] = trim_tail(work, float(window_s))
                if scale != "off":
                    rec[prov + "downscale"] = downscale(work, scale)
            return _ours(work, cfg, cal, opt.runs_kw, opt.profile,
                         ref_hr=ref_hr, cap_shen=cap_shen, arm_a=arm_a,
                         arm_ap=arm_ap, a_note=note)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    ours = run(opt.scale, opt.window_s, arm_a="A", arm_ap="Ap",
               note=f"calibrated confidence; PRODUCTION CONFIG "
                    f"(window {opt.window_s:.0f} s, scale {opt.scale})",
               prov="")
    rec["effective"] = {"window_s": float(opt.window_s), "scale": opt.scale}
    ctx = ours["ctx"]
    rec["arms"] = {"A": ours["A"], "Ap": ours["Ap"]}
    if not getattr(opt, "skip_native", False):
        # Arm An: identical pipeline, identical window, NO downscale. A - An is
        # the downscale term; the encode itself cannot be undone offline, so it
        # bounds rather than removes the codec confound (module docstring).
        nat = run("off", opt.window_s, arm_a="An", arm_ap="Anp",
                  note=f"calibrated confidence; NATIVE resolution "
                       f"(window {opt.window_s:.0f} s, no downscale) — the "
                       f"A/An gap is the downscale term, NOT the codec term",
                  prov="native_")
        rec["arms"]["An"] = nat["An"]
        rec["native_res"] = (nat.get("ctx") or {}).get("res")

    rec["capture"] = {k: ctx.get(k) for k in
                      ("fps", "res", "cap_ours", "clock", "n_segments",
                       "seg_gaps_s", "duration_s", "caveats")}
    shen = _shenai(doc, fsinfo, opt.runs_kw,
                   ctx.get("cap_ours", float("nan")))
    rec["arms"].update(shen)

    # Offset prior: trim_tail's own cut when we emulated it, else the clip's
    # span minus ShenAI's — the recording ENDS when the measurement completes
    # (app/measure_prep.py:238), so ShenAI's window sits at the tail. Without
    # this prior a +/-5 s search never reaches a 210 s clip's tail and every
    # such clip reads "unalignable".
    offset0 = float((rec.get("trim") or {}).get("trimmed_from_s") or 0.0)
    if not offset0:
        span_shen = _beat_span_s(doc)
        dur = float(ctx.get("duration_s") or 0.0)
        if np.isfinite(span_shen) and dur > span_shen:
            offset0 = dur - span_shen
    # A discontinuous capture clock cannot carry ONE scalar offset, and the
    # prior above assumes ShenAI's window is a contiguous TAIL of the clip —
    # which the recorder's middle-drop makes false (in a 2026-09-11 run that
    # prior came out at 165.0 s, a number with no defensible meaning). Refuse.
    n_seg = int(ctx.get("n_segments") or 0)
    gaps = list(ctx.get("seg_gaps_s") or [])
    beat_period = float(np.median(np.diff(ref_t))) if ref_t.size >= 3 else 1.0
    refuse = ""
    if n_seg > 1:
        refuse = (f"capture clock is discontinuous, {n_seg} segments "
                  f"(gaps {', '.join(f'{g:.1f}' for g in gaps[:4])} s; one "
                  f"beat period is {beat_period:.2f} s) — a single scalar "
                  f"offset cannot span a hole the recorder cut out of the "
                  f"middle of the clip")
    elif gaps and max(gaps) > beat_period:
        refuse = (f"inter-segment gap {max(gaps):.2f} s exceeds one beat "
                  f"period {beat_period:.2f} s")
    rec["alignment"] = _align(ctx.get("our_times", np.array([])), ref_t,
                              offset0, opt.tolerance_ms, refuse=refuse,
                              n_null=getattr(opt, "null_draws", NULL_DRAWS))
    rec["alignment"]["offset_prior_s"] = offset0
    trains = {}
    if shen.get("B", {}).get("ok") and "times" in shen["B"]:
        # Remove the convention offset FIRST. timing_precision_from_trains
        # matches within 100 ms, and their beat ONSET sits 100-200 ms before
        # our systolic PEAK — left in, the check reports the offset and
        # matches almost nothing (measured 0.11 matched on the 2026-09-11
        # fixture; 1.00 once removed). Precision is the residual AFTER it.
        off = fsinfo.get("resid_med_ms", 0.0)
        off = off / 1000.0 if np.isfinite(off) else 0.0
        trains["shenai_wave"] = np.asarray(shen["B"]["times"], float) - off
    if shen.get("C", {}).get("ok") and "times" in shen["C"]:
        trains["shenai_train"] = shen["C"]["times"]
    rec["shenai_self_timing"] = timing_precision_from_trains(trains) \
        if len(trains) == 2 else {"timing_precision_ms": float("nan"),
                                  "timing_matched_fraction": 0.0,
                                  "timing_pair": None}
    for a in ARMS:                                   # times are for the table
        rec["arms"].get(a, {}).pop("times", None)
    return rec


# ----------------------------------------------------------------- printing
def _c(v, w=6, p=2) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-".rjust(w)
    return f"{f:{w}.{p}f}" if np.isfinite(f) else "-".rjust(w)


HEAD = (f"{'arm':17} {'SNRdB':>6} {'spcbpm':>7} {'spcSNR':>7} {'raw':>5} "
        f"{'fused':>5} {'kept':>5} {'bpm':>6} {'dHR':>7} {'cntR':>5} "
        f"{'ints':>5} "
        f"{'runs':>5} {'long':>5} {'drop':>5} {'split':>6} {'covOur':>7} "
        f"{'covShn':>7} {'harm':>5} {'RMSSD':>6} {'SDNN':>6}")


PREAMBLE = """
WHICH CARDS EACH ARM COULD SUPPORT — read before the table:
  A / An / A'  four-ROI (A, An) or single-ROI (A') on OUR compressed clip:
               Pulse, Fitness, AFib. Only these arms can ever reach the AFib
               card, because only they carry independent skin regions.
  B / C        ShenAI's ONE dense signal and ONE train: Pulse, and Fitness so
               far as it rests on rate/intervals. NOT AFib — decision_logic
               needs cross_roi_coherence >= 0.35 OR two-region verification
               (30 ms at matched 0.75 across INDEPENDENT regions), and a single
               train satisfies neither. Adopting ShenAI as-is would downgrade
               every irregular series to REPEAT_SCAN.
  CONFOUND     arms B/C never passed through the MediaRecorder codec; A/An/A'
               always did. A - An isolates the DOWNSCALE only; the encode
               cannot be undone offline, so a B win is not attributable."""


def _print_clip(rec: dict, verbose: bool) -> None:
    cap = rec.get("capture") or {}
    fs = rec["fs"]
    eff = rec.get("effective") or {}
    nseg = cap.get("n_segments")
    gaps = list(cap.get("seg_gaps_s") or [])
    print(f"\n{rec['clip']}   {cap.get('res') or '?'}  "
          f"{_c(cap.get('duration_s'), 5, 1).strip()} s @ "
          f"{_c(cap.get('fps'), 4, 1).strip()} fps   clock: {cap.get('clock')}")
    # The effective window/scale are printed because the arm label claims
    # "production path" and a 2026-09-11 review caught that claim being false
    # (209.9 s at 640x480 against production's 40 s at 480x360).
    print(f"  effective config: window {eff.get('window_s')} s, scale "
          f"{eff.get('scale')}  (arm An at native "
          f"{rec.get('native_res') or 'n/a'})   capture segments: {nseg}"
          + (f"  gaps {', '.join(f'{g:.1f}' for g in gaps[:4])} s" if gaps
             else ""))
    if isinstance(nseg, int) and nseg > 1:
        print("     WARNING: the container clock has a hole. The recorder "
              "keeps chunks[0] + the last 15 and DROPS THE MIDDLE, so a scan "
              "over ~48 s lands a clip whose clock jumps by minutes.")
    for k in ("trim", "downscale", "native_trim", "native_downscale"):
        # measure_prep's own rule: a scaled/trimmed run must never be mistaken
        # for a native one. Both ran on a COPY; the input was not touched.
        if rec.get(k):
            i = rec[k]
            print(f"  {k} (on a temp COPY): applied={i.get('applied')} "
                  f"{i.get('reason') or ''} "
                  f"{'from ' + str(i.get('trimmed_from_s')) + ' s' if i.get('trimmed_from_s') else ''}"
                  .rstrip())
    print(f"  sidecar: {rec['n_samples']} samples "
          f"({rec['n_null_samples']} null), {rec['n_beats']} beats, "
          f"upload_id {str(rec.get('upload_id'))[:28]!r}")
    print(f"  fs: {_c(fs['fs'], 6, 3).strip()} Hz  "
          f"(start {_c(fs['fs0'], 6, 3).strip()} Hz, {fs['fs_source']}, "
          f"{fs['search']})   residual med "
          f"{_c(fs['resid_med_ms'], 5, 0).strip()} ms  IQR "
          f"{_c(fs['resid_iqr_ms'], 5, 0).strip()} ms  slope "
          f"{_c(fs['slope_ms_per_s'], 5, 2).strip()} ms/s")
    # The duration-derived rate knows nothing about the beat train, so the
    # ratio separates "the SDK's clock drifts" from "the train covers a
    # sub-span of the waveform" — the two look identical in the slope alone.
    print(f"     fs cross-check: fitted {_c(fs['fs'], 6, 3).strip()} Hz vs "
          f"duration-derived {_c(fs.get('fs_duration'), 6, 3).strip()} Hz "
          f"(samples / recorder_duration_ms)   fitted/duration "
          f"{_c(fs.get('fs_ratio'), 5, 3).strip()} (validator, want ~1.000)"
          f"   fs0/duration {_c(fs.get('fs0_ratio'), 5, 3).strip()} "
          f"(the train's sub-span: stable across clips => structural, "
          f"not drift)")
    if fs.get("narrow"):
        nw = fs["narrow"]
        print(f"     fs re-fit after VOID: narrow grid gave "
              f"{_c(nw.get('fs'), 6, 3).strip()} Hz at "
              f"{_c(nw.get('slope_ms_per_s'), 5, 2).strip()} ms/s; wide grid "
              f"{FS_WIDE_SEARCH[0]:.2f}-{FS_WIDE_SEARCH[1]:.2f} kept")
    print("  " + HEAD)
    for a in ARMS:
        r = rec["arms"].get(a)
        if not r:
            continue
        if not r.get("ok"):
            print(f"  {ARM_LABEL[a]:17} (no row) {r['reason']}")
            continue
        # dHR is suffixed on the arms where it cannot fail: T = tautology
        # (arm C, both numbers are ShenAI's), t = near-tautology (arm B, fs was
        # fitted to this train). cntR is arm B's independent replacement.
        mark = {"B": "t", "C": "T"}.get(a, " ")
        print(f"  {ARM_LABEL[a]:17} {_c(r['snr_db'])} {_c(r['spec_bpm'],7,1)} "
              f"{_c(r['spec_snr'],7,1)} {r['n_raw']:5d} {r['n_fused']:5d} "
              f"{r['kept']:5d} {_c(r['bpm'],6,1)} {_c(r['d_hr'],6,1)}{mark} "
              f"{_c(r.get('count_ratio'),5,2)} "
              f"{r['n_intervals']:5d} {r['n_runs']:5d} {r['longest_run']:5d} "
              f"{_c(r['dropout'],5,2)} {_c(r['split'],6,2)} "
              f"{_c(r['cov_ours'],7,2)} {_c(r['cov_shen'],7,2)} "
              f"{_c(r['harmonic'],5,2)} {_c(r['rmssd_ms'],6,1)} "
              f"{_c(r['sdnn_ms'],6,1)}"
              + ("  VOID" if r.get("void") else ""))
    for a in ARMS:
        r = rec["arms"].get(a) or {}
        for g in r.get("gates_failed", []):
            print(f"     {a}: gate {g}")
        for f in (r.get("flags", []) if (verbose or a in ("B", "C")) else []):
            print(f"     {a}: {f}")
    al = rec["alignment"]
    nul = (f"matched {_c(al.get('matched_frac'),4,2).strip()} vs null p95 "
           f"{_c(al.get('null_p95'),4,2).strip()} "
           f"(perm {_c(al.get('null_perm_p95'),4,2).strip()} over "
           f"{al.get('n_null', 0)} draws, far "
           f"{_c(al.get('null_far_p95'),4,2).strip()})")
    if al.get("unalignable"):
        why = al.get("reason") or "no offset survived the null"
        print(f"     alignment: unalignable ({why}; offset prior "
              f"{_c(al.get('offset_prior_s'), 5, 1).strip()} s) — no agreement "
              "numbers, because a bogus offset would read as a timing finding")
        if np.isfinite(al.get("matched_frac", float("nan"))):
            print(f"     alignment null: {nul}")
    else:
        for key in sorted(k for k in al if k.startswith("t") and
                          isinstance(al[k], dict)):
            m = al[key]
            print(f"     shenai/agreement @{key[1:]}ms: offset "
                  f"{al['offset_s']:+.3f} s  matched {m['n_matched']}  "
                  f"sens {m['sensitivity']:.2f}  ppv {m['ppv']:.2f}  "
                  f"|dt| med {m['median_abs_dt_ms']:.0f} ms  IQR "
                  f"{m['iqr_ms']:.0f} ms")
        print(f"     alignment null PASSED: {nul}")
        if al.get("alias_ambiguous"):
            print("     caveat: a whole-beat alias (offset +/- k*IBI) matches "
                  "as well as this one — the offset is identified only up to "
                  "+/-k beats, so beat IDENTITY is not established even though "
                  "the timebase relation is.")
        print("     (a 100-200 ms median is EXPECTED: their beat ONSET vs our "
              "systolic PEAK. The IQR is the evidence.)")
    st = rec["shenai_self_timing"]
    # NOT the two-region check. 40 ms (max_timing_precision_ms_any) is the
    # Gate-1 proxy on INDEPENDENT ROI pairs; putting ShenAI's waveform against
    # ShenAI's own train in that slot would invite the reader to conclude the
    # verification gate is satisfiable by a single-signal front end. It is not
    # — and fs was fitted to minimise exactly this residual, so the number is
    # an internal-consistency check on the sample rate, nothing more.
    print(f"     ShenAI internal consistency (their waveform vs their own "
          f"train, convention offset removed; NOT a two-region verification, "
          f"and fs was fitted to minimise this residual): "
          f"{_c(st['timing_precision_ms'],5,1).strip()} ms at matched "
          f"{st['timing_matched_fraction']:.2f}")


def _summary(recs: list[dict], min_n: int) -> None:
    n = len(recs)
    if n < min_n:
        print(f"\nSUMMARY withheld: {n} clip(s) with a sidecar, minimum "
              f"{min_n}. A verdict from a handful of clips is a hypothesis — "
              f"CLIPS_KEEP is 12 on an ephemeral disk, so pull often "
              f"(scripts/pull_scan_clips.py).")
        return
    n_align = sum(1 for r in recs if not (r.get("alignment") or {}).get(
        "unalignable", True))
    print(f"\nSUMMARY over {n} clips — median per arm")
    print("  " + f"{'arm':17} {'covOur':>7} {'covShn':>7} {'ints':>6} "
          f"{'bpm':>6} {'|dHR|<=5':>9} {'cntR+-10%':>10} {'harm':>6} "
          f"{'RMSSD':>7} {'split':>6}")
    for a in ARMS:
        rows = [r["arms"][a] for r in recs
                if r["arms"].get(a, {}).get("ok") and not r["arms"][a].get("void")]
        if not rows:
            print(f"  {ARM_LABEL[a]:17} no evaluable rows")
            continue
        med = lambda k: float(np.nanmedian([r.get(k, np.nan)   # noqa: E731
                                            for r in rows]))
        # Arms B and C are EXCLUDED from the |dHR|<=5 column, not scored 0 on
        # it: arm C's reference and its beats are both ShenAI's, and arm B's
        # fs was fitted to arm C's train. Printing them there put a structural
        # identity next to the CLAUDE.md +/-5 bpm bar and read as an accuracy
        # result. Arm B's replacement is cntR: OUR beat count on THEIR
        # waveform against THEIR beat count, within 10%.
        if a in ("B", "C"):
            hit_s = "  excluded"
        else:
            hit_s = f"{sum(1 for r in rows if r.get('d_hr_ok')):4d}/" \
                    f"{len(rows):<4d}"
        cnt = [r for r in rows if "count_ratio" in r]
        cnt_s = (f"{sum(1 for r in cnt if r.get('count_ratio_ok')):4d}/"
                 f"{len(cnt):<5d}" if cnt else "         -")
        print(f"  {ARM_LABEL[a]:17} {_c(med('cov_ours'),7,2)} "
              f"{_c(med('cov_shen'),7,2)} {_c(med('n_intervals'),6,1)} "
              f"{_c(med('bpm'),6,1)} {hit_s:>9} {cnt_s:>10} "
              f"{_c(med('harmonic'),6,2)} {_c(med('rmssd_ms'),7,1)} "
              f"{_c(med('split'),6,2)}")
    # Every clause of the pre-registered rule must be checkable from THIS
    # block, or the rule is decorative: alignment null, VOID rate, coverage and
    # intervals against A', the count ratio, and the TOO-GOOD medians.
    n_void = sum(1 for r in recs if r["arms"].get("B", {}).get("void"))
    n_b = sum(1 for r in recs if r["arms"].get("B", {}).get("ok"))
    print(f"  arm B VOID (|drift| > {MAX_SLOPE_MS_PER_S:.0f} ms/s) on "
          f"{n_void}/{n_b} clips that produced a B row — VOID rows are "
          f"excluded from the medians above.")
    print(f"  alignment null passed on {n_align}/{n} clips — every agreement "
          f"number above rests on that; a clip that fails it has no timing "
          f"evidence at all, only a best-of-1000-offsets coincidence.")
    print("  Arms A'/B/C are UNCALIBRATED single-train paths: clean_runs' "
          "confidence channel is much weaker there, which inflates their "
          "intervals and coverage. A' is the only like-for-like comparator "
          "for B; A is the production number; An - A is the downscale term.")
    print("  THE CODEC CONFOUND STANDS ON EVERY ROW: arms B/C never saw the "
          "MediaRecorder encode, arms A/An/A' always did. If B leads, "
          "'their extractor is better' and 'they never paid the compression' "
          "are NOT separable here, and the remedies are opposite (adopt their "
          "extractor vs. extract on-device / raise the bitrate).")
    print("  Neither B nor C can support the AFib card at all: one signal, one "
          "train, no independent region — see WHICH CARDS above.")
    print("  Decide with the PRE-REGISTERED rule in the module docstring "
          "(ADOPT/REJECT/NOT SETTLED), not by picking a winner from this "
          "table.")


# ----------------------------------------------------------- self-test
# The two experiments that broke this harness on 2026-09-11, kept runnable so
# a later change has to face them again. No clips, no sidecars, no network.
def _legacy_alignable(our_t, shen_t, offset0, tol_ms) -> tuple[bool, float]:
    """The rule this file shipped with until 2026-09-11, kept ONLY as the null
    experiment's baseline and never called on real data: take the best of ~1000
    candidate offsets, then declare the result trustworthy when
    n_matched / min(n) >= 0.5. There is no null anywhere in it."""
    our_t = np.sort(np.asarray(our_t, float))
    shen_t = np.sort(np.asarray(shen_t, float))
    if our_t.size < 5 or shen_t.size < 5:
        return False, float("nan")
    grid = np.arange(offset0 - 5.0, offset0 + 5.0 + 1e-9, 0.010)
    costs = [np.median(np.abs(_residuals(our_t - g, shen_t))) for g in grid]
    off = float(grid[int(np.argmin(costs))])
    m = match_beats(shen_t, our_t, tolerance_ms=tol_ms, ptt_ms=off * 1000.0)
    errs = np.abs(m.timing_errors_ms)
    frac = m.n_matched / max(min(shen_t.size, our_t.size), 1)
    return bool(frac >= 0.5), (float(np.median(errs)) if errs.size
                               else float("nan"))


def _synth_train(rng, n: int, ibi_s: float, jitter_s: float, t0: float):
    return t0 + np.cumsum(rng.normal(ibi_s, jitter_s, n))


def _synth_wave(fs: float, bpm: float, dur_s: float, seed: int = 3):
    """A pulse train sampled at a KNOWN rate: peaks at the beat times, so the
    detector's answer can be checked against a truth the fit cannot see."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * fs)) / fs
    beats = np.arange(0.5, dur_s, 60.0 / bpm)
    sig = np.zeros(t.size)
    for b in beats:
        sig += np.exp(-((t - b) / 0.055) ** 2)
    sig -= 0.45 * np.exp(-((t[:, None] - (beats + 0.22)[None, :]) / 0.075)
                         ** 2).sum(1)          # dicrotic notch, for realism
    return sig + rng.normal(0, 0.01, t.size), beats


def _null_experiment(n_draws: int) -> list[str]:
    out = []
    print("\nEXPERIMENT 1 — the alignment null. Two beat trains, NO shared "
          "timebase.\n  Each row: how often each rule called the pair "
          "ALIGNABLE. Anything above ~5% is\n  a harness that prints "
          "confident timing numbers about noise.")

    def trial(name, trials, gen, seed):
        rng = np.random.default_rng(seed)
        old_ok, new_ok, old_dt, new_dt = 0, 0, [], []
        for _ in range(trials):
            our, shen = gen(rng)
            ok, dt = _legacy_alignable(our, shen, 0.0, 100.0)
            old_ok += ok
            old_dt.append(dt if ok else np.nan)
            a = _align(our, shen, 0.0, 100.0, n_null=n_draws)
            if not a["unalignable"]:
                new_ok += 1
                new_dt.append((a.get("t100") or {}).get("median_abs_dt_ms",
                                                        np.nan))
        f = lambda d: (f"{np.nanmedian(d):.0f} ms"       # noqa: E731
                       if np.any(np.isfinite(d)) else "-")
        line = (f"  {name:34} BEFORE {old_ok:3d}/{trials:<3d} "
                f"({old_ok/trials:4.0%}, median |dt| {f(old_dt):>7})"
                f"   AFTER {new_ok:3d}/{trials:<3d} ({new_ok/trials:4.0%}, "
                f"median |dt| {f(new_dt):>7})")
        print(line)
        return line

    def same_rate(rng):      # ~57 beats at ~0.85 s, timebase relation destroyed
        return (_synth_train(rng, 57, 0.85, 0.030, rng.uniform(0, 1)),
                _synth_train(rng, 57, 0.85, 0.030,
                             rng.uniform(0, 1) + rng.uniform(-2, 2)))

    def indep_rate(rng):     # pure noise: rates drawn independently
        return (_synth_train(rng, 57, 60.0 / rng.uniform(50, 100), 0.030,
                             rng.uniform(0, 1)),
                _synth_train(rng, 57, 60.0 / rng.uniform(50, 100), 0.030,
                             rng.uniform(0, 1)))

    def truly_aligned(rng):  # POWER: the same beats, onset-vs-peak + 2 s clock
        base = _synth_train(rng, 57, 0.85, 0.045, rng.uniform(0, 1))
        return base + 2.0, base - 0.15 + rng.normal(0, 0.020, 57)

    def aligned_lossy(rng):  # POWER with a realistic detector: 15% missed
        base = _synth_train(rng, 60, 0.85, 0.045, rng.uniform(0, 1))
        shen = base - 0.15 + rng.normal(0, 0.020, 60)
        keep = rng.random(60) > 0.15
        return base[keep] + 2.0 + rng.normal(0, 0.030, int(keep.sum())), shen

    out.append(trial("NULL same rate, no timebase", 60, same_rate, 20260911))
    out.append(trial("NULL independent rates (noise)", 40, indep_rate, 424242))
    out.append(trial("POWER truly aligned", 30, truly_aligned, 7))
    out.append(trial("POWER aligned, 15% beats missed", 30, aligned_lossy, 11))
    print("  A null-experiment rate near 0% with power near 100% is the "
          "only acceptable pair:\n  refusing everything would pass the first "
          "test and answer no question.")
    return out


def _tautology_experiment() -> list[str]:
    print("\nEXPERIMENT 2 — is arm B's heart rate independent of ShenAI's own "
          "train?\n  The train is HELD FIXED at 72 bpm; only the waveform's "
          "TRUE sample rate changes.\n  A rate that carries information must "
          "move when the truth moves.")
    out = []
    dur, bpm = 48.0, 72.0
    for true_fs in (15.0, 30.0, 60.0, 120.0):
        sig, beats = _synth_wave(true_fs, bpm, dur)
        train = beats + 0.0
        doc = {"ppg": {"signal": [float(v) for v in sig]},
               "heartbeats": [{"start_location_sec": float(b),
                               "end_location_sec": float(b + 0.35)}
                              for b in train],
               "measurement": {"recorder_duration_ms": dur * 1000.0}}
        fs0, src = _start_fs(doc, sig)
        opt = argparse.Namespace(fs=None, fs_search=[0.98, 1.02], fs_steps=60)
        fsi = _fit_fs(doc, sig, train, fs0, src, opt)
        det = np.sort(np.array([b.t_s for b in _detect_wave(sig, fsi["fs"])]))
        ibi = np.diff(det)
        arm_b_bpm = 60.0 / float(np.median(ibi)) if ibi.size else float("nan")
        ratio = det.size / max(train.size, 1)
        line = (f"  true fs {true_fs:6.1f} Hz -> fs0 {fs0:7.3f} -> fitted "
                f"{fsi['fs']:7.3f}   arm B bpm {arm_b_bpm:5.1f} "
                f"(ref {bpm:.0f}, |dHR| {abs(arm_b_bpm - bpm):4.1f}) "
                f"   cntR {ratio:4.2f}")
        print(line)
        out.append(line)
    print("  bpm is the SAME for four different truths because fs was fitted "
          "to the train it is\n  being compared against — that is why d_hr is "
          "labelled NEAR-TAUTOLOGICAL on the\n  row and excluded from the "
          "summary's |dHR|<=5 column. cntR (our beat count on\n  their "
          "waveform / their beat count) is a count against a count and CAN "
          "disagree.")
    return out


def _void_experiment() -> list[str]:
    print("\nEXPERIMENT 3 — a VOID drift verdict when the train covers a "
          "SUB-SPAN.\n  True 30.000 Hz, 72 bpm, 48 s; the train is trimmed to "
          "4-44 s, which biases fs0.")
    out = []
    dur, fs_true = 48.0, 30.0
    sig, beats = _synth_wave(fs_true, 72.0, dur)
    for lo, hi, tag in ((4.0, 44.0, "train 4-44 s"), (0.0, 47.0, "train ~full")):
        train = beats[(beats >= lo) & (beats <= hi)]
        doc = {"ppg": {"signal": [float(v) for v in sig]},
               "heartbeats": [{"start_location_sec": float(b),
                               "end_location_sec": float(b + 0.35)}
                              for b in train],
               "measurement": {"recorder_duration_ms": dur * 1000.0}}
        fs0, src = _start_fs(doc, sig)
        opt = argparse.Namespace(fs=None, fs_search=[0.98, 1.02], fs_steps=60)
        fsi = _fit_fs(doc, sig, train, fs0, src, opt)
        nw = fsi.get("narrow") or {}
        line = (f"  {tag:14} fs0 {fs0:7.3f} -> narrow "
                f"{nw.get('fs', fsi['fs']):7.3f} "
                f"(slope {nw.get('slope_ms_per_s', fsi['slope_ms_per_s']):+7.2f}"
                f" ms/s) -> reported {fsi['fs']:7.3f} "
                f"(slope {fsi['slope_ms_per_s']:+7.2f} ms/s)"
                f"   fitted/duration {fsi['fs_ratio']:5.3f} "
                f"(validator)  fs0/duration {fsi['fs0_ratio']:5.3f} "
                f"(the sub-span; true fs {fs_true:.3f})")
        print(line)
        out.append(line)
    print("  The wide re-fit reaches the true rate the +/-2% grid cannot "
          "(and still reports\n  VOID when the residual stays above 2 ms/s). "
          "fitted/duration ~1.00 says the fit\n  agrees with an estimate that "
          "never saw the train; fs0/duration is the sub-span\n  itself — "
          "1.086 here for a train that ends at 44 s of a 48 s recording.")
    return out


def _self_test() -> int:
    print("compare_shenai_signal --self-test   (2026-09-11; synthetic only, "
          "no clip is read or written)")
    _null_experiment(NULL_DRAWS)
    _tautology_experiment()
    _void_experiment()
    print("\nself-test complete.")
    return 0


# ---------------------------------------------------------------------- CLI
def _corpus() -> pathlib.Path:
    if _COMMON is not None:
        return _COMMON.corpus_dir()
    d = os.getenv("EVAL_CORPUS_DIR") or str(_REPO / "data" / "eval_corpus")
    return pathlib.Path(d).expanduser().resolve()


def _clips(paths: list[str]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for p in paths:
        q = pathlib.Path(p)
        if q.is_dir():
            found = sorted(set(list(q.glob("*.webm")) + list(q.glob("*.mp4")) +
                               [pathlib.Path(str(s)[:-len(".shenai.json")])
                                for s in q.glob("*.shenai.json")]))
            out.extend(f for f in found if f.exists())
        else:
            out.append(q)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", metavar="PATH")
    ap.add_argument("--sidecar", help="explicit sidecar JSON (single clip)")
    ap.add_argument("--fs", type=float, help="force the PPG sample rate (Hz)")
    ap.add_argument("--fs-search", nargs=2, type=float, default=[0.98, 1.02],
                    metavar=("LO", "HI"), help="grid bounds as multipliers")
    ap.add_argument("--fs-steps", type=int, default=200)
    # Production's own defaults, not "off": arm A is labelled "ours,
    # production path" and must BE that. Measured 2026-09-11, the old defaults
    # had arm A analysing 209.9 s at 640x480 against production's 40 s at
    # 480x360 while its SNR was read on a bench measured at the phone's
    # compression level. Pass --scale off --window-s 0 for the old behaviour.
    ap.add_argument("--scale", default=DEFAULT_SCALE, metavar="WxH|off",
                    help=f"re-encode a COPY before ingest "
                         f"(default: {DEFAULT_SCALE}, production's)")
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S,
                    help=f"emulate trim_tail on a COPY; 0 = keep the clip's "
                         f"own clock (default: {DEFAULT_WINDOW_S:.0f} s, "
                         f"production's)")
    ap.add_argument("--skip-native", action="store_true",
                    help="skip arm An (the native-resolution control) — "
                         "halves the runtime and forfeits the only separable "
                         "part of the codec confound")
    ap.add_argument("--null-draws", type=int, default=NULL_DRAWS,
                    help=f"permutation draws for the alignment null "
                         f"(default {NULL_DRAWS}; fewer is a weaker null)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the null and tautology experiments on synthetic "
                         "trains — no clips or sidecars needed")
    ap.add_argument("--min-conf", type=float, default=None)
    ap.add_argument("--tolerance-ms", type=float, default=100.0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--profile", default="consumer",
                    help="capture profile (production sends 'consumer')")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return _self_test()

    if a.json_out:
        out = pathlib.Path(a.json_out).resolve()
        corpus = _corpus()
        if corpus == out or corpus in out.parents:
            print(f"refusing to write inside the corpus ({corpus}) — it is "
                  f"gitignored, precious, and this harness is read-only.",
                  file=sys.stderr)
            return 2

    cfg = load_config()
    rc = cfg["runs"]
    lo_ms, hi_ms = rc["ibi_physiologic_ms"]
    min_conf = a.min_conf if a.min_conf is not None else (
        CALIBRATED_MIN_CONF if rc["min_conf"] == "CALIBRATED"
        else float(rc["min_conf"]))
    a.runs_kw = {"min_conf": float(min_conf),
                 "min_run_beats": int(rc["min_run_beats"]),
                 "max_physiologic_ibi_ms": float(hi_ms),
                 "min_physiologic_ibi_ms": float(lo_ms),
                 "missed_beat_ratio": float(rc["missed_ratio"])}
    try:
        cal = _load_calibrator(cfg)
    except RuntimeError as e:
        print(f"{e}", file=sys.stderr)
        return 2

    clips = _clips(a.paths or [str(_corpus())])
    if not clips:
        print(__doc__)
        return 1
    wanted = [x.strip() for x in a.arms.split(",") if x.strip()]
    # An costs a second full ingest+extraction pass; don't pay for it when the
    # caller has already excluded it from --arms.
    a.skip_native = bool(a.skip_native or "An" not in wanted)
    print(PREAMBLE)

    recs, skipped = [], []
    for clip in clips:
        if not clip.exists():
            print(f"missing: {clip}")
            continue
        side = pathlib.Path(a.sidecar) if a.sidecar and len(clips) == 1 \
            else _sidecar_for(clip)
        if not side.exists():
            skipped.append(clip.name)
            continue
        try:
            rec = measure(clip, side, cfg, cal, a)
        except Exception as e:                                 # noqa: BLE001
            print(f"{clip.name}: FAILED {type(e).__name__}: {e}")
            continue
        rec["arms"] = {k: v for k, v in rec["arms"].items() if k in wanted}
        recs.append(rec)
        _print_clip(rec, a.verbose)

    if skipped:
        print(f"\nskipped (no sidecar beside the clip): {len(skipped)} — "
              + ", ".join(skipped[:6]) + (" ..." if len(skipped) > 6 else ""))
        print("   pull them with: python scripts/pull_scan_clips.py --get all")
    if not recs:
        print("\nno clip carried a ShenAI sidecar. Only scans recorded after "
              "route 6 shipped have one, and nothing is retained unless the "
              "staging service has AFIB_KEEP_UPLOADS=1 and AFIB_CLIPS_TOKEN "
              "set — verify with GET /api/clips before concluding anything.")
        return 1

    _summary(recs, a.min_n)
    print("\nstanding reference points:")
    for k, v in BENCH.items():
        print(f"   {k:38} {v}")
    print("\nShenAI is a SECOND OPINION, not ground truth: same camera, same "
          "illuminant, correlated errors (inference/evidence.py:69-84).")
    if a.json_out:
        # NaN/Infinity are not JSON: json.dumps emits them anyway and anything
        # but Python's own parser rejects the file. A missing metric is null —
        # the same convention the sidecar itself uses for an absent value.
        def _clean(o):
            if isinstance(o, float):
                return o if np.isfinite(o) else None
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple, np.ndarray)):
                return [_clean(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return _clean(o.item())
            return o
        pathlib.Path(a.json_out).write_text(
            json.dumps(_clean(recs), indent=2, default=str, allow_nan=False))
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
