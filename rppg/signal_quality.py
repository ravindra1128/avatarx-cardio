"""
Composite signal-quality index (T4).

THE DESIGN RULE, in force forever: the SQI must never measure PERIODICITY.
AF is aperiodic by definition; an autocorrelation-peak or
spectral-peak-sharpness quality score blocks exactly the patients being
screened (measured on the open-rppg SQI: clean AF 0.28 vs clean sinus 0.84
— the AF scan is discarded as "bad signal"). Every component below measures
signal PRESENCE or MEASUREMENT integrity instead:

  snr_in_band            in-band vs out-of-band energy ratio of the raw
                         (unfiltered) extractor output — pulse energy lives
                         in 0.7-4 Hz wherever the rhythm wanders inside it
  skewness               pulse waveforms are systolic-peak asymmetric;
                         noise and motion residue are symmetric
  spectral_concentration BROADBAND fraction of total energy inside the
                         physiological band — explicitly not peak sharpness
  cross_roi_coherence    beats seen by >2 ROIs (chance coincidences pair
                         two ROIs; real perfusion covers the face)
  tracking_stability     from the face tracker

Weights live in configs/default.yaml (sqi.weights). NaN anywhere scores
that component 0 — a corrupt input can only lower quality, never raise it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from configs import load_config

SQI_COMPONENTS = ("snr_in_band", "skewness", "spectral_concentration",
                  "cross_roi_coherence", "tracking_stability")


@dataclass
class SQIResult:
    sqi: float
    components: dict = field(default_factory=dict)
    per_roi: dict = field(default_factory=dict)


def _band_energies(x: np.ndarray, fps: float, lo: float, hi: float
                   ) -> tuple[float, float]:
    z = x - np.mean(x)
    spec = np.abs(np.fft.rfft(z)) ** 2
    f = np.fft.rfftfreq(z.size, 1.0 / fps)
    keep = f > 0.05                       # ignore DC/very-slow detrend residue
    inb = float(np.sum(spec[keep & (f >= lo) & (f <= hi)]))
    tot = float(np.sum(spec[keep]))
    return inb, tot


def _snr_component(x: np.ndarray, fps: float, lo: float, hi: float) -> float:
    inb, tot = _band_energies(x, fps, lo, hi)
    out = tot - inb
    if tot <= 0:
        return 0.0
    snr_db = 10.0 * np.log10((inb + 1e-12) / (out + 1e-12))
    # logistic centred at +3 dB: comfortably above parity reads as good
    return float(1.0 / (1.0 + np.exp(-(snr_db - 3.0) / 2.0)))


def _concentration_component(x: np.ndarray, fps: float, lo: float, hi: float
                             ) -> float:
    inb, tot = _band_energies(x, fps, lo, hi)
    return float(inb / tot) if tot > 0 else 0.0


def _skew_component(x: np.ndarray, fps: float, lo: float, hi: float) -> float:
    from ._filters import bandpass
    z = bandpass(x, fps, lo, hi)
    sd = z.std()
    if sd <= 1e-12:
        return 0.0
    skew = float(np.mean(((z - z.mean()) / sd) ** 3))
    return float(np.clip(skew / 0.8, 0.0, 1.0))


def _coherence_component(fused) -> float:
    """Mean cross-ROI membership beyond the 2-ROI chance floor."""
    if fused is None or not getattr(fused, "beats", None):
        return 0.0
    agree = np.array([b.roi_agreement for b in fused.beats], float)
    if agree.size == 0:
        return 0.0
    return float(np.mean(np.clip((agree - 0.5) / 0.5, 0.0, 1.0)))


def compute_sqi(waveforms: dict, fps: float, fused=None, *,
                tracking_stability: float = 1.0,
                weights: Optional[dict] = None,
                band_hz: Optional[tuple] = None,
                energy_band_hz: Optional[tuple] = None) -> SQIResult:
    """Composite SQI in [0, 1] from per-ROI UNFILTERED extractor outputs.

    `waveforms` maps roi -> 1-D waveform (extractor output BEFORE
    bandpassing — in-band fractions are meaningless on a bandpassed
    signal). `fused` is the beat-level fusion result for coherence.
    """
    cfg = load_config()
    if weights is None:
        weights = cfg["sqi"]["weights"]
    if band_hz is None:
        band_hz = tuple(cfg["sqi"]["band_hz"])
    lo, hi = float(band_hz[0]), float(band_hz[1])
    # Energy fractions use a rate-neutral band that includes the first
    # harmonics of the fastest physiological rate; the (narrower) filter band
    # is kept for the skewness component. Explicit band_hz overrides both.
    if energy_band_hz is None:
        eb = cfg["sqi"].get("energy_band_hz")
        energy_band_hz = tuple(eb) if (eb and band_hz == tuple(cfg["sqi"]["band_hz"])) \
            else band_hz
    elo, ehi = float(energy_band_hz[0]), float(energy_band_hz[1])

    per_roi: dict = {}
    snrs, skews, concs = [], [], []
    for roi, w in (waveforms or {}).items():
        w = np.asarray(w, float)
        if w.size < 32 or not np.all(np.isfinite(w)):
            per_roi[roi] = {"snr_in_band": 0.0, "skewness": 0.0,
                            "spectral_concentration": 0.0,
                            "note": "short or non-finite waveform"}
            snrs.append(0.0); skews.append(0.0); concs.append(0.0)
            continue
        s = _snr_component(w, fps, elo, ehi)
        k = _skew_component(w, fps, lo, hi)
        c = _concentration_component(w, fps, elo, ehi)
        per_roi[roi] = {"snr_in_band": s, "skewness": k,
                        "spectral_concentration": c}
        snrs.append(s); skews.append(k); concs.append(c)

    # A single region is commonly covered by glasses, hair, a hand, or a hard
    # shadow.  The old arithmetic mean let that one region veto three clean
    # independent skin regions.  The median is robust to one contaminated ROI;
    # cross-ROI beat membership still requires independent confirmation, so
    # this cannot turn a lone periodic artifact into accepted evidence.
    comp = {
        "snr_in_band": float(np.median(snrs)) if snrs else 0.0,
        "skewness": float(np.median(skews)) if skews else 0.0,
        "spectral_concentration": float(np.median(concs)) if concs else 0.0,
        "cross_roi_coherence": _coherence_component(fused),
        "tracking_stability": float(np.clip(tracking_stability, 0.0, 1.0))
        if np.isfinite(tracking_stability) else 0.0,
    }
    wsum = float(sum(weights[k] for k in SQI_COMPONENTS))
    sqi = sum(float(weights[k]) * comp[k] for k in SQI_COMPONENTS) / wsum \
        if wsum > 0 else 0.0
    if not waveforms:
        sqi = 0.0
    return SQIResult(sqi=float(np.clip(sqi, 0.0, 1.0)),
                     components=comp, per_roi=per_roi)
