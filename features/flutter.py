"""
Flutter-signature features (v0.6 T1) — the inverse of the AFib feature
path.

AFib is found because its pulse is IRREGULAR. Flutter conducts a fixed
fraction of a ~250-300/min atrial circuit, so its pulse is
METRONOMICALLY REGULAR (2:1 ~150, 3:1 ~100, 4:1 ~75) and an
irregularity-based detector reads it as normal sinus rhythm. Nothing
here tries to NAME the rhythm: F waves are atrial morphology, they are
not in the interval channel, and no amount of interval processing
recovers them. What is measurable is the ventricular response, in three
families:

  1. RATE FINGERPRINT — a sustained median rate with a tight CI inside a
     conduction band (2:1 is the specific one; 3:1 and 4:1 are computed
     and expected to be non-specific), and how much of the scan holds it.
  2. HYPER-REGULARITY — "too regular" for the rate: dispersion below an
     age-adjusted floor, tachogram spectral concentration, and ABSENT
     RESPIRATORY MODULATION. The last is the load-bearing one: sinus
     rhythm at any rate retains some respiratory sinus arrhythmia, while
     fixed conduction decouples the ventricle from the respiratory
     drive. Measured on the v0.6 fixtures, a jitter-matched metronomic
     clip and a true sinus tachycardia are indistinguishable by
     dispersion (RMSSD ~25 ms both) while their tachogram respiratory-
     band fractions are 0.03 vs 0.91.
  3. SERIAL CONDUCTION-RATIO STEPS — across a registered scan series,
     rates clustering at integer-ratio steps of ONE latent atrial rate
     (150 -> 100 -> 75 implies ~300). This is the track's differentiator
     because nothing else in cardiology produces it.

Plus VARIABLE-BLOCK features, deliberately: variable conduction is
irregular and overlaps AF's feature space, so the confusion has to be
measurable rather than hidden.

Discipline inherited from the AFib path and not negotiable here:
ACCEPT-grade clean runs only, nothing computed across a run boundary
(a run break is where a missed/false beat is suspected), no interval
repair of any kind — flutter's conduction changes are SIGNAL, and a
filter that smoothed them would erase the thing being measured.
"""
from __future__ import annotations

import numpy as np

# The conduction bands, derived from the atrial rate they imply rather
# than picked: typical flutter runs 250-300/min (docs/flutter_track.md),
# so 2:1 conduction produces 125-150 and the band must cover it or the
# product silently misses most of its own declared target (review
# finding: a 140 floor covered only 280-330 atrial). The upper edge is
# kept at 165 to admit the faster circuits. 2:1 is the only band with a
# specific claim; the others are computed so the F2 per-ratio table can
# report what they do, which is expected to be "collide with ordinary
# rhythms" — 3:1 of 250-330 is 83-110 and 4:1 is 63-83, both squarely
# inside ordinary human heart rates.
DEFAULT_BANDS = {"2:1": (125.0, 165.0), "3:1": (83.0, 110.0),
                 "4:1": (63.0, 83.0)}
# The coupling constants (MIN_RESP_QUALITY, UNCORROBORATED_FRACTION,
# CORROBORATED_ELSEWHERE_FRACTION) and their rationale live in
# features/regularity.py and are imported below — a local copy here was
# dead and misleading (review finding).
# Age-adjusted dispersion floor (ms): below this, at a banded rate, the
# pulse is "too regular for sinus". HRV falls with age, so the floor
# must fall too or every older sinus patient trips it. PLANNING VALUES —
# pre-registered in configs, revised only from data (§F, spec B.22).
DEFAULT_RMSSD_FLOOR_MS = {"<40": 18.0, "40-59": 14.0, "60-74": 10.0,
                          ">=75": 8.0, "unknown": 8.0}
# ...and the floor the CAMERA imposes, which at consumer frame rates is
# the binding one. Beat times are estimated on a sampled waveform, so a
# perfectly metronomic source still reads a non-zero RMSSD. Measured on
# v0.6 fixtures through the production path: a zero-jitter 150 bpm clip
# reads 15.0 ms at 30 fps and 8.2 ms at 60 fps — 0.45 of a frame period
# in both cases, i.e. quantization, not physiology. The effective floor
# is therefore max(age floor, this), and when this one wins the scan is
# marked measurement_limited: the PHYSIOLOGICAL claim was superseded by
# the camera and no consumer should read it as a heart finding.
# PLANNING VALUE with margin (0.45 measured x ~1.67); at 30 fps it masks
# every age floor, which is exactly the disclosure docs/
# flutter_limitations.md has to carry.
DEFAULT_MEASUREMENT_FLOOR_FRAMES = 0.75
# v0.7 (G-a): the interval-series machinery lives in features/
# regularity.py — imported, never duplicated. The names below are kept
# for the flutter head and its tests.
from features.regularity import (  # noqa: E402
    CORROBORATED_ELSEWHERE_FRACTION, MIN_RESP_QUALITY, RESP_BAND_HZ,
    RESP_HALF_WIDTH_HZ, TACHOGRAM_FS_HZ, MAX_TACHOGRAM_GAP_S,
    UNCORROBORATED_FRACTION, band_fraction as _band_fraction,
    clean_intervals as _clean, dominant_frequency as _dominant_frequency,
    phase_locking as _phase_locking, regularity_from_runs,
    respiratory_coupling, run_diffs as _run_diffs, tachogram)
MIN_INTERVALS = 15
MIN_SUSTAINED_FRACTION = 0.70
SERIAL_MIN_SCANS = 3
SERIAL_ATRIAL_GRID_BPM = (180.0, 400.0)
SERIAL_RATIOS = (2, 3, 4, 5)


def age_band(age_years) -> str:
    """The band whose dispersion floor applies. Unknown age takes the
    STRICTEST (lowest) floor: without an age we must not accuse an
    older, low-variability sinus rhythm of being too regular."""
    try:
        a = float(age_years)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(a):
        return "unknown"
    if a < 40:
        return "<40"
    if a < 60:
        return "40-59"
    if a < 75:
        return "60-74"
    return ">=75"


# ------------------------------------------------------ family 1: rate
def rate_fingerprint(lattice, *, bands=None, min_intervals=MIN_INTERVALS,
                     sustained_fraction=MIN_SUSTAINED_FRACTION) -> dict:
    """Sustained median rate, its bootstrap-free CI, and which
    conduction band (if any) holds it for most of the scan."""
    bands = dict(bands or DEFAULT_BANDS)
    ibi, t = _clean(lattice)
    out = {"n_intervals": int(ibi.size), "median_bpm": None,
           "bpm_ci95": None, "band": None, "band_fraction": None,
           "sustained_s": None, "in_band": False}
    if ibi.size < min_intervals:
        out["reason"] = (f"only {int(ibi.size)} clean intervals "
                         f"(< {min_intervals})")
        return out
    bpm = 60000.0 / ibi
    med = float(np.median(bpm))
    # distribution-free CI of the median (order statistics, normal
    # approximation) — an interval estimate, never a point claim
    n = bpm.size
    k = 1.96 * np.sqrt(n) / 2.0
    lo_i = max(int(np.floor(n / 2.0 - k)), 0)
    hi_i = min(int(np.ceil(n / 2.0 + k)), n - 1)
    s = np.sort(bpm)
    out.update({"median_bpm": round(med, 2),
                "bpm_ci95": [round(float(s[lo_i]), 2),
                             round(float(s[hi_i]), 2)]})
    best = None
    for name, (lo, hi) in bands.items():
        frac = float(np.mean((bpm >= lo) & (bpm <= hi)))
        if best is None or frac > best[1]:
            best = (name, frac)
    name, frac = best
    lo, hi = bands[name]
    inb = (bpm >= lo) & (bpm <= hi)
    # "sustained" must be a fraction of TIME, not of intervals: at 150
    # bpm an in-band interval is 2.5x shorter than one at 60, so
    # counting intervals over-weights the fast stretch and half a scan
    # in band could pass a 70% test (review finding).
    total_s = float(np.sum(ibi)) / 1000.0
    in_band_s = float(np.sum(ibi[inb])) / 1000.0
    time_frac = in_band_s / total_s if total_s > 0 else 0.0
    out["band"] = name
    out["band_fraction"] = round(time_frac, 4)
    out["band_interval_fraction"] = round(frac, 4)
    out["sustained_s"] = round(in_band_s, 2)
    out["in_band"] = bool(lo <= med <= hi
                          and time_frac >= sustained_fraction)
    return out


# ---------------------------------------------- family 2: regularity
def measurement_floor_ms(fps, frames=DEFAULT_MEASUREMENT_FLOOR_FRAMES):
    """The RMSSD a metronomic source reads through this pipeline at
    `fps`, with margin. None when fps is unknown — and an unknown fps
    must NOT be treated as a permissive floor."""
    try:
        f = float(fps)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f) or f <= 0:
        return None
    return float(frames) * 1000.0 / f


def hyper_regularity(lattice, *, age_years=None, respiration=None,
                     rmssd_floors=None, fs_hz: float = TACHOGRAM_FS_HZ,
                     measurement_floor_frames=None, regularity=None) -> dict:
    """"Too regular" indices against the effective floor: the larger of
    the age-adjusted physiological floor and the camera's own
    quantization floor."""
    floors = dict(rmssd_floors or DEFAULT_RMSSD_FLOOR_MS)
    band = age_band(age_years)
    floor = float(floors.get(band, floors["unknown"]))
    m_floor = measurement_floor_ms(
        getattr(lattice, "fps", None),
        DEFAULT_MEASUREMENT_FLOOR_FRAMES if measurement_floor_frames
        is None else measurement_floor_frames)
    effective = floor if m_floor is None else max(floor, m_floor)
    # v0.7 (G-a): dispersion comes from the ONE representation — the
    # same pooled-interval SDNN/CV and within-run RMSSD the AF decision
    # reads — never recomputed here. A caller may hand one in; otherwise
    # it is built from the lattice by the same module.
    reg = regularity if regularity is not None else regularity_from_runs(
        list(getattr(lattice, "runs", []) or []),
        run_times=list(getattr(lattice, "run_times", []) or []),
        fps=getattr(lattice, "fps", None), respiration=respiration)
    ibi, _ = _clean(lattice)
    out = {"age_band": band, "rmssd_floor_ms": floor,
           "measurement_floor_ms": (None if m_floor is None
                                    else round(m_floor, 3)),
           "effective_floor_ms": round(effective, 3),
           "measurement_limited": bool(m_floor is not None
                                       and m_floor > floor),
           "rmssd_ms": None, "sdnn_ms": None, "cv": None,
           "below_floor": False, "spectral_concentration": None,
           "coupling": None}
    if m_floor is None:
        # no frame rate on the lattice: the camera's contribution is
        # unknown, so a "too regular" verdict is not available at all
        out["unknown_measurement_floor"] = True
    if ibi.size >= 2 and reg.dispersion.get("sdnn_ms") is not None:
        out["sdnn_ms"] = round(float(reg.dispersion["sdnn_ms"]), 3)
        out["cv"] = round(float(reg.dispersion["cv"]), 5)
    if reg.dispersion.get("rmssd_ms") is not None:
        rmssd = float(reg.dispersion["rmssd_ms"])
        out["rmssd_ms"] = round(rmssd, 3)
        out["below_floor"] = bool(m_floor is not None and rmssd < effective)
    grid, tach = tachogram(lattice, fs_hz)
    if grid is not None:
        # concentration of the tachogram's own spectrum: a metronomic
        # series has no structure to concentrate, so this reads LOW for
        # flutter and high for a rhythmically modulated series. Reported
        # for the record; the flag leans on dispersion + coupling.
        # Composed from the ONE spectral machinery (G-a): the fraction
        # of wide-band power within +/- RESP_HALF_WIDTH_HZ of the
        # dominant frequency.
        f_dom = _dominant_frequency(tach, fs_hz, (0.04, 1.0))
        if f_dom is not None:
            out["spectral_concentration"] = round(
                float(_band_fraction(tach, fs_hz, f_dom)), 4)
    # the representation handed in may have been built WITHOUT a
    # respiration channel (the pipeline builds it that way); a caller
    # that supplies one now gets the coupling measured against it,
    # never the stale "no channel" verdict. Latent since 09dc0b0 (no
    # caller handed both in); first reached by the v0.8 research
    # report. A representation with no coupling entry at all still
    # falls back to measuring (a no-channel dict when there is none).
    coup = reg.structure.get("coupling")
    if coup is None or (respiration is not None and not coup.get("available")
                        and coup.get("status") in (None, "no_channel",
                                                   "no_tachogram")):
        coup = respiratory_coupling(lattice, respiration, fs_hz=fs_hz)
    out["coupling"] = coup
    return out


# ------------------------------------------ family 3: serial signature
def latent_atrial_fit(rates_bpm, *, ratios=SERIAL_RATIOS,
                      grid_bpm=SERIAL_ATRIAL_GRID_BPM,
                      tol_bpm: float = 5.0, step: float = 0.5,
                      min_scans: int = SERIAL_MIN_SCANS,
                      min_separation_bpm: float = 15.0) -> dict:
    """Fit ONE latent atrial rate whose integer divisors explain a
    series of ventricular rates.

    Scores 0 unless at least TWO DISTINCT divisors are occupied and the
    occupied rate clusters are genuinely separated: a series that sits
    at one rate is explained by any atrial rate that divides it, so
    calling that a conduction-ratio pattern would be a free pass. This
    is the guard that keeps the track's differentiator from firing on
    three repeat scans of the same stable rhythm.
    """
    r = np.asarray([x for x in (rates_bpm or []) if x is not None], float)
    r = r[np.isfinite(r) & (r > 0)]
    out = {"n_scans": int(r.size), "score": 0.0, "atrial_bpm": None,
           "ratios": None, "residual_bpm": None, "reason": None}
    if r.size < min_scans:
        out["reason"] = (f"{int(r.size)} usable scans "
                         f"(< {min_scans}) — no serial signature")
        return out
    grid = np.arange(grid_bpm[0], grid_bpm[1] + step, step)
    best = None
    for a in grid:
        cand = a / np.asarray(ratios, float)
        idx = np.argmin(np.abs(r[:, None] - cand[None, :]), axis=1)
        resid = float(np.mean(np.abs(r - cand[idx])))
        occupied = sorted({int(ratios[i]) for i in idx})
        if len(occupied) < 2:
            continue
        cluster = sorted(float(np.mean(r[idx == ratios.index(k)]))
                         for k in occupied)
        if min(np.diff(cluster)) < min_separation_bpm:
            continue
        if best is None or resid < best[0]:
            best = (resid, float(a), [int(ratios[i]) for i in idx])
    if best is None:
        out["reason"] = ("rates do not occupy two separated integer-"
                         "ratio steps of any single atrial rate")
        return out
    resid, a, occ = best
    score = float(np.clip(1.0 - resid / tol_bpm, 0.0, 1.0))
    out["score"] = round(score, 4)
    out["residual_bpm"] = round(resid, 3)
    if score <= 0.0:
        # the best available fit is no fit: reporting its atrial rate
        # anyway would hand a consumer a fabricated number that only
        # the score reveals as meaningless
        out["reason"] = (f"best integer-ratio fit misses by "
                         f"{resid:.1f} bpm (tolerance {tol_bpm}) — no "
                         "common atrial rate explains this series")
        return out
    out.update({"atrial_bpm": round(a, 1), "ratios": occ})
    return out


# ------------------------------------------------- variable-block block
def variable_block_features(lattice, *,
                            atrial_grid_bpm=SERIAL_ATRIAL_GRID_BPM,
                            ratios=SERIAL_RATIOS, step: float = 0.5,
                            tol_frac: float = 0.08) -> dict:
    """Within ONE scan: are the intervals integer multiples of a common
    atrial cycle? Variable-block flutter says yes while looking
    irregular; AF says no. Emitted so the AF/flutter confusion is
    measurable rather than hidden — it is NOT part of the flag."""
    ibi, _ = _clean(lattice)
    out = {"n_intervals": int(ibi.size), "lattice_fraction": None,
           "atrial_bpm": None, "ratios_used": None}
    if ibi.size < MIN_INTERVALS:
        return out
    best = None
    for a in np.arange(atrial_grid_bpm[0], atrial_grid_bpm[1] + step, step):
        cycle = 60000.0 / a                        # ms
        k = ibi / cycle
        near = np.abs(k - np.round(k))
        used = np.round(k).astype(int)
        ok = (near <= tol_frac) & np.isin(used, np.asarray(ratios))
        frac = float(np.mean(ok))
        if best is None or frac > best[0]:
            best = (frac, float(a), sorted(set(used[ok].tolist())))
    frac, a, used = best
    out.update({"lattice_fraction": round(frac, 4),
                "atrial_bpm": round(a, 1), "ratios_used": used})
    return out


# --------------------------------------------------------- the session
def flutter_features(result, lattice, *, respiration=None,
                     age_years=None, cfg=None,
                     series_rates_bpm=None, regularity=None) -> dict:
    """Every family for one scan, on an ACCEPT-grade result only.

    A non-ACCEPT scan produces no features at all (not zeros): the
    quality/evidence gates already decided the beats are not
    trustworthy, and a flag computed on them would launder that.
    """
    fc = ((cfg or {}).get("flutter") or {})
    outcome = getattr(getattr(result, "outcome", None), "value",
                      getattr(result, "outcome", None))
    if outcome != "ACCEPT":
        return {"available": False,
                "reasons": [f"scan outcome {outcome!r} is not ACCEPT — "
                            "flutter features are computed on clean, "
                            "gate-passing runs only"]}
    bands = fc.get("bands") or DEFAULT_BANDS
    floors = fc.get("rmssd_floor_ms") or DEFAULT_RMSSD_FLOOR_MS
    rate = rate_fingerprint(
        lattice, bands=bands,
        min_intervals=int(fc.get("min_intervals", MIN_INTERVALS)),
        sustained_fraction=float(fc.get("min_sustained_fraction",
                                        MIN_SUSTAINED_FRACTION)))
    reg = hyper_regularity(
        lattice, age_years=age_years, respiration=respiration,
        rmssd_floors=floors,
        measurement_floor_frames=fc.get("measurement_floor_frames"),
        regularity=regularity)
    out = {"available": True, "reasons": [],
           "rate": rate, "regularity": reg,
           "variable_block": variable_block_features(lattice),
           "series": latent_atrial_fit(
               series_rates_bpm,
               min_scans=int(fc.get("series_min_scans",
                                    SERIAL_MIN_SCANS)))}
    if rate.get("reason"):
        out["reasons"].append(rate["reason"])
    if not reg["coupling"]["available"]:
        out["reasons"].append(
            "respiratory coupling unavailable: "
            f"{reg['coupling']['reason']}")
    return out
