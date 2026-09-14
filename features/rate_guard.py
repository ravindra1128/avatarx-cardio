"""Guard the resting pulse against beat-doubling / halving.

The resting-rate feature the fitness card rides on is the median of the clean
inter-beat intervals (features/hemodynamics.py). When the beat detector splits
one beat into two it produces half-length intervals; when it merges two it
produces double-length ones. Either drags the interval median off toward a
factor of 2x or 0.5x, and a resting scan reads 100+ bpm when the true pulse is
~65 (seen live: 101 vs a 65 reference, 35 % half-length intervals).

The spectral rate (inference/evidence.spectral_pulse) is computed with a
subharmonic check, so it resists that error. When the interval rate shows a
doubling signature AGAINST a trustworthy spectral rate, the spectral rate is
the safer resting rate.

This is a PURE detector. It decides whether the interval rate looks doubled
against a spectral anchor it can defend (finite, positive, and — when the SNR
is given — above a floor) and, if so, hands back the spectral rate with the
reason. It never invents a rate and never fires without that anchor, so on a
clean scan it is a no-op. It does NOT touch the clinical rhythm/flutter head
(features/flutter.py), whose harmonic handling is its own; this is the
resting-rate / fitness path only.
"""
from __future__ import annotations

import math

# The interval rate is a clean "double"/"half" when it sits within this
# fraction of 2x / 0.5x the spectral rate — a detector split/merge, not
# physiology (no resting adult sits at exactly twice a trustworthy rhythm).
HARMONIC_RATIO_TOL = 0.15
# A split can contaminate only SOME intervals, lifting the median part-way
# rather than cleanly to 2x (the 101-vs-65 case sat at ratio ~1.5). Above this
# share of half-length intervals the median is not trustworthy even then.
HARMONIC_FRACTION_HI = 0.30
# Below this ratio the interval rate is not even "inflated"; the split branch
# needs the rate above the anchor before it will act.
INFLATED_RATIO_LO = 1.0 + HARMONIC_RATIO_TOL
# The spectral anchor must clear this in-band SNR to stand in as the truth.
SPECTRAL_SNR_FLOOR = 1.5


def _finite(*xs) -> bool:
    try:
        return all(math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def guard_resting_pulse(lattice_bpm, spectral_bpm, *, harmonic_fraction=0.0,
                        spectral_snr=None, ref_bpm=None) -> dict:
    """Decide whether `lattice_bpm` (interval-median resting rate) is a
    doubling / halving error against `spectral_bpm`, and if so return the
    spectral rate to use instead.

    `ref_bpm` (an external reference such as ShenAI's HR) is recorded for
    telemetry only and never required — inference stays camera-only and
    replayable. Returns a dict with `guarded`, `reported_bpm` (spectral when
    guarded, else the lattice rate), `signature` ('double' | 'half' |
    'split_inflation' | None), `ratio`, and a one-line `reason`.
    """
    out = {"guarded": False, "reported_bpm": lattice_bpm,
           "raw_lattice_bpm": lattice_bpm, "spectral_bpm": spectral_bpm,
           "ratio": None, "harmonic_fraction": float(harmonic_fraction or 0.0),
           "spectral_snr": spectral_snr, "ref_bpm": ref_bpm,
           "signature": None, "reason": None}
    # A trustworthy spectral anchor and a finite interval rate are required.
    if not (_finite(lattice_bpm, spectral_bpm) and float(spectral_bpm) > 0
            and float(lattice_bpm) > 0):
        return out
    if spectral_snr is not None and _finite(spectral_snr) \
            and float(spectral_snr) < SPECTRAL_SNR_FLOOR:
        return out

    lat, spec = float(lattice_bpm), float(spectral_bpm)
    ratio = lat / spec
    out["ratio"] = round(ratio, 4)
    hf = float(harmonic_fraction or 0.0)

    def near(x, target):
        return abs(x - target) <= HARMONIC_RATIO_TOL * target

    if near(ratio, 2.0):
        sig = "double"
    elif near(ratio, 0.5):
        sig = "half"
    elif hf >= HARMONIC_FRACTION_HI and ratio >= INFLATED_RATIO_LO:
        sig = "split_inflation"
    else:
        return out

    tail = (f", {hf * 100:.0f}% half-length intervals"
            if sig == "split_inflation" else "")
    out.update(
        guarded=True, reported_bpm=round(spec, 2), signature=sig,
        reason=(f"resting rate {lat:.0f} bpm looks {sig.replace('_', ' ')} "
                f"against the waveform's dominant rhythm {spec:.0f} bpm "
                f"(ratio {ratio:.2f}{tail}); using the rhythm rate"))
    return out
