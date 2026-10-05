"""Phase-locked 2nd harmonic of the pulse: a RECORDED-ONLY shape descriptor.

Owner decision 2026-09-20 ("option 1, record only"). This module publishes
nothing, gates nothing and feeds no card. It writes a few scalars per scan to
`debug.pulse_harmonics_*` and the tracking sheet so ONE question can be
answered from real scans: does the facial pulse SHAPE differ between people,
or is it the same for everyone (imposed by a producer's processing)?

Why this descriptor, and only this one (scripts/stiffness_evidence.py):

* The Arterial Stiffness card's markers are not measurements at a phone
  scan's SNR. Fed a shapeless sinusoid plus noise, the deployed contour path
  "finds a dicrotic notch" on 90 of 90 inputs with the same reflection-index
  distribution real scans give, and its crest time reads ~0.25 x RR whatever
  the true upstroke is.
* On the COMPRESSED clip the pulse's 2nd harmonic is found on 0 of 11 testable
  scans (this test finds 12 of 12 synthetic ones at the clips' noise level) and
  its phase across scans is at chance (R 0.27; random phases give 0.27). On
  the LIVE-FRAME signal of the same scans it is found on 16 of 16 and repeats
  within ~3 degrees (phi2 82 +/- 4, r2 0.368 +/- 0.014, one person).
* The 3rd harmonic of the SDK's waveform is an artefact of its ~0.7-4 Hz
  band-pass (correlation -0.8 with heart rate) and is deliberately not read.

The estimator is fiducial-free. With z_k the analytic signal in the band
around k*f0 and u1 = z1/|z1| the fundamental's unit phasor,

    c2 = < |z1| z2 conj(u1)^2 > / < |z1|^2 >

is the 2nd harmonic relative to the fundamental's OWN phase: r2 = |c2| is its
relative amplitude and phi2 = arg c2 its relative phase. No beat is detected,
no foot or peak is picked and nothing is aligned, so there is no alignment
jitter. Noise in the 2nd band is uncorrelated with u1^2 and therefore adds
variance, not bias. A uniform error in the sample rate rescales f0 and 2*f0
together and leaves (r2, phi2) unchanged, which matters because the SDK's
sample rate is DERIVED by the client and its time origin is unknown.
Inverting the waveform shifts phi2 by 180 degrees.

What this is NOT: a stiffness marker. On fingertip PPG (MIMIC PERform, 15
people in sinus rhythm) the index tracks true crest fraction at r = +0.66,
but heart rate alone explains 57 % of it and with heart rate removed the
correlation is +0.31, 95 % CI -0.28..+0.93. It failed the validity bar set
before that number was seen (>= 0.5). It collapses in atrial fibrillation
(ICC 0.40), so rows with an irregular rhythm must be read separately.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 1
HALF_BW_HZ = 0.35              # either side of k*f0: wide enough for resting HRV
MIN_SPAN_S = 20.0
MAX_SPAN_S = 120.0             # bounds the work on a client-supplied array
FS_RANGE_HZ = (10.0, 120.0)
F0_RANGE_HZ = (0.6, 3.5)       # 36-210 bpm
F0_HINT_TOL = 0.10
MAX_MISSING_FRACTION = 0.02
MAX_GAP_S = 0.2                # a longer step in a capture clock is a hole, not wander
EDGE_S = 1.5                   # filter/Hilbert edges, dropped
BLOCK_S = 6.0                  # >> the bands' correlation time (~1.4 s), so blocks are independent
MIN_BLOCKS, MAX_BLOCKS = 5, 12
DETECTABLE_P = 0.01
SOURCES = ("sdk", "traces", "clip")


def _finite_number(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _analytic_band(x, fs, lo, hi):
    from scipy.signal import butter, filtfilt, hilbert
    nyq = fs / 2.0
    b, a = butter(3, [max(lo, 0.05) / nyq, min(hi, 0.99 * nyq) / nyq], btype="band")
    return hilbert(filtfilt(b, a, x))


def _spectrum(x, fs):
    from scipy.signal import welch
    n = int(min(x.size, fs * 20))
    return welch(x - x.mean(), fs=fs, nperseg=n, noverlap=int(0.75 * n))


def _fundamental(x, fs, hint_hz):
    """(f0, source): the spectral peak near the hint, else the strongest pulse-band peak."""
    f, p = _spectrum(x, fs)
    if hint_hz is not None and F0_RANGE_HZ[0] <= hint_hz <= F0_RANGE_HZ[1]:
        m = (f >= (1 - F0_HINT_TOL) * hint_hz) & (f <= (1 + F0_HINT_TOL) * hint_hz)
        if m.any():
            return float(f[m][np.argmax(p[m])]), "reference_rate"
    m = (f >= 0.7) & (f <= 3.0)
    if not m.any():
        return None, None
    return float(f[m][np.argmax(p[m])]), "spectral_peak"


def _locked_c2(x, fs, f0):
    """(c2, p, blocks, se). p is the probability, under "no harmonic is locked to the
    fundamental", of block estimates as consistent as these: the scan is cut into
    `blocks` stretches, each gives its own complex c2, and a locked harmonic makes
    them all point the same way while noise points them anywhere. With Re and Im of
    the block values iid normal under the null, m |mean|^2 / (2 s^2) ~ F(2, 2(m-1)).
    `se` is (SE of Re c2, SE of Im c2) from the same block values, or None.

    Not a surrogate test, deliberately. Shifting the fundamental against the harmonic
    only breaks the lock when the heart rate VARIES: on a metronomic heart it breaks
    nothing and reads a perfect pulse as absent - a bias against exactly the low-HRV
    people (older, diabetic) a stiffness question is about. This test does not depend
    on rate variability, has an exact null, and uses no random numbers."""
    edge = int(EDGE_S * fs)
    sl = slice(edge, x.size - edge)
    z1 = _analytic_band(x, fs, f0 - HALF_BW_HZ, f0 + HALF_BW_HZ)[sl]
    z2 = _analytic_band(x, fs, 2 * (f0 - HALF_BW_HZ), 2 * (f0 + HALF_BW_HZ))[sl]
    a1 = np.abs(z1)
    w = a1 * a1
    if float(w.sum()) <= 1e-24:
        return None, None, 0, None
    q = a1 * np.conj(z1 / np.maximum(a1, 1e-12)) ** 2 * z2      # filtered once: no per-block edges
    c2 = complex(q.sum() / w.sum())
    m = int(min(MAX_BLOCKS, (q.size / fs) // BLOCK_S))
    if m < MIN_BLOCKS:
        return c2, None, m, None
    cm = np.array([q[i].sum() / max(float(w[i].sum()), 1e-24)
                   for i in np.array_split(np.arange(q.size), m)])
    se = (float(np.std(cm.real, ddof=1) / math.sqrt(m)), float(np.std(cm.imag, ddof=1) / math.sqrt(m)))
    s2 = float((np.abs(cm - cm.mean()) ** 2).sum() / (2 * (m - 1)))
    if s2 <= 1e-24:
        return c2, 0.0, m, se
    from scipy.stats import f as f_dist
    return c2, float(f_dist.sf(m * abs(cm.mean()) ** 2 / (2.0 * s2), 2, 2 * (m - 1))), m, se


def crest_fraction_2h(r2: float, phi2_deg: float, n: int = 720) -> float:
    """(foot -> peak) / cycle of cos(th) + r2 cos(2 th + phi2): the descriptor
    in its readable form. A function of (r2, phi2) only; adds no information."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    y = np.cos(th) + r2 * np.cos(2.0 * th + math.radians(phi2_deg))
    return int(np.argmax(np.roll(y, -int(np.argmin(y))))) / float(n)


def _wrap_deg(d: float) -> float:
    return abs((d + 180.0) % 360.0 - 180.0)


def estimate(x, fs: float, f0_hint_hz=None) -> dict:
    """The descriptor of ONE uniformly sampled pulse waveform. Never raises on
    bad numbers: every refusal is a `state`."""
    fs = _finite_number(fs)
    if fs is None or not (FS_RANGE_HZ[0] <= fs <= FS_RANGE_HZ[1]):
        return {"state": "invalid_sample_rate"}
    x = np.asarray(x, float).ravel()
    x = x[-int(MAX_SPAN_S * fs):]
    if x.size < MIN_SPAN_S * fs:
        return {"state": "too_short", "span_s": round(x.size / fs, 2)}
    bad = ~np.isfinite(x)
    if bad.mean() > MAX_MISSING_FRACTION:
        return {"state": "too_many_missing_samples", "missing_fraction": round(float(bad.mean()), 4)}
    if bad.any():
        i = np.arange(x.size)
        x = np.interp(i, i[~bad], x[~bad])
    if float(np.std(x)) <= 1e-12:
        return {"state": "flat_signal"}
    f0, f0_source = _fundamental(x, fs, _finite_number(f0_hint_hz))
    if f0 is None or not (F0_RANGE_HZ[0] <= f0 <= F0_RANGE_HZ[1]):
        return {"state": "no_fundamental"}
    if 2.0 * (f0 + HALF_BW_HZ) >= 0.47 * fs:
        return {"state": "second_harmonic_above_nyquist", "f0_hz": round(f0, 4), "fs_hz": round(fs, 3)}
    c2, p_value, blocks, se = _locked_c2(x, fs, f0)
    if c2 is None:
        return {"state": "no_fundamental"}
    r2, phi2 = abs(c2), math.degrees(math.atan2(c2.imag, c2.real))
    f, p = _spectrum(x, fs)
    band = (f >= 0.5) & (f <= min(6.0, 0.45 * fs))
    sig = np.abs(f - f0) <= 0.12
    snr1 = (10.0 * math.log10(p[sig].sum() / max(p[band & ~sig].sum(), 1e-30))
            if sig.any() and (band & ~sig).any() else None)
    # Internal consistency: the same descriptor from each half of the scan.
    half_d = half_da = None
    h = x.size // 2
    if h >= 0.5 * MIN_SPAN_S * fs:
        halves = []
        for part in (x[:h], x[h:]):
            fh, _ = _fundamental(part, fs, f0)
            halves.append(_locked_c2(part, fs, fh)[0] if fh else None)
        if None not in halves:
            half_d = _wrap_deg(math.degrees(math.atan2(halves[0].imag, halves[0].real))
                               - math.degrees(math.atan2(halves[1].imag, halves[1].real)))
            half_da = abs(halves[0].imag - halves[1].imag)
    return {
        "state": "ok",
        "r2": round(r2, 4), "phi2_deg": round(phi2, 2),
        # Im c2 = r2 sin(phi2): how much faster the pulse rises than it falls (the
        # asymmetry); Re c2 = r2 cos(phi2): how peaked it is. Each with the standard
        # error of the block estimates, so a reader can see the scan's own precision.
        "asym": round(c2.imag, 4), "asym_se": None if se is None else round(se[1], 4),
        "peak": round(c2.real, 4), "peak_se": None if se is None else round(se[0], 4),
        "p_value": None if p_value is None else float(f"{p_value:.3g}"),
        "blocks": blocks,
        "detectable": bool(p_value is not None and p_value < DETECTABLE_P),
        "crest_fraction_2h": round(crest_fraction_2h(r2, phi2), 4),
        "half_split_dphi2_deg": None if half_d is None else round(half_d, 2),
        "half_split_dasym": None if half_da is None else round(half_da, 4),
        "f0_hz": round(f0, 4), "f0_source": f0_source,
        "fundamental_snr_db": None if snr1 is None else round(snr1, 2),
        "fs_hz": round(fs, 3), "span_s": round(x.size / fs, 2),
    }


def from_sdk_snapshot(raw, f0_hint_hz=None) -> dict:
    """The SDK's live-frame waveform, exactly as delivered (its polarity, its
    band-pass). Client-supplied: every field is checked before it is used."""
    if not isinstance(raw, dict):
        return {"state": "not_received"}
    ppg = raw.get("ppg") if isinstance(raw.get("ppg"), dict) else {}
    sig = ppg.get("signal")
    if not isinstance(sig, list) or not sig:
        return {"state": "no_waveform"}
    fs = _finite_number(ppg.get("fs_hz"))
    if fs is None or not (FS_RANGE_HZ[0] <= fs <= FS_RANGE_HZ[1]):
        return {"state": "invalid_sample_rate"}
    tail = sig[-int(MAX_SPAN_S * fs):]
    x = np.array([v if _finite_number(v) is not None else np.nan for v in tail], float)
    if f0_hint_hz is None:
        ref = raw.get("reference") if isinstance(raw.get("reference"), dict) else {}
        bpm = _finite_number(ref.get("heart_rate_bpm"))
        f0_hint_hz = None if bpm is None else bpm / 60.0
    out = estimate(x, fs, f0_hint_hz)
    out["polarity"] = "as_delivered"
    out["fs_source"] = str(ppg.get("fs_source") or "")[:40]
    return out


def from_ingest(ing, f0_hint_hz=None) -> dict:
    """OUR signal: wide-band POS per facial region on the longest gap-free
    stretch, fused by each region's fundamental SNR. `ing` is an IngestResult -
    the compressed clip's, or the live-frame trace document's."""
    if ing is None or not getattr(ing, "ok", False) or not getattr(ing, "traces", None):
        return {"state": "not_available"}
    from preprocessing.roi import ROI_NAMES
    from rppg.pos import pos_pulse
    ts = np.asarray(ing.timestamps_s, float)
    if ts.size < 3:
        return {"state": "too_short"}
    cuts = np.where(np.diff(ts) > MAX_GAP_S)[0]
    edges = np.concatenate([[0], cuts + 1, [ts.size]])
    a, e = max(zip(edges[:-1], edges[1:]), key=lambda ab: ab[1] - ab[0])
    span = float(ts[e - 1] - ts[a])
    if e - a < 3 or span < MIN_SPAN_S:
        return {"state": "too_short", "span_s": round(span, 2)}
    fs = (e - a - 1) / span
    if not (FS_RANGE_HZ[0] <= fs <= FS_RANGE_HZ[1]):
        return {"state": "invalid_sample_rate"}
    waves = {}
    for roi in ROI_NAMES:
        rgb = np.asarray(ing.traces.get(roi), float)[a:e] if roi in ing.traces else None
        if rgb is None or rgb.ndim != 2 or not np.all(np.isfinite(rgb)):
            continue
        w = pos_pulse(rgb, fs, band=(0.5, min(10.0, 0.45 * fs)))
        if float(np.std(w)) > 1e-12:
            waves[roi] = w / np.std(w)
    if not waves:
        return {"state": "no_usable_region"}
    hint = _finite_number(f0_hint_hz)
    f0, _ = _fundamental(np.mean(list(waves.values()), axis=0), fs, hint)
    if f0 is None:
        return {"state": "no_fundamental"}
    fund = {r: np.real(_analytic_band(w, fs, f0 - HALF_BW_HZ, f0 + HALF_BW_HZ)) for r, w in waves.items()}
    snr = {r: float(np.var(fund[r]) / max(np.var(waves[r] - fund[r]), 1e-12)) for r in waves}
    ref = max(snr, key=snr.get)
    fused = np.zeros_like(waves[ref])
    for r, w in waves.items():
        sign = -1.0 if (r != ref and float(np.corrcoef(fund[r], fund[ref])[0, 1]) < 0) else 1.0
        fused += sign * snr[r] * w
    # The REFERENCE rate again, not our refined one: two +/-10 % searches in a row can
    # walk ~20 % off the pulse onto a clip's strongest artefact (seen: 72 vs 84 bpm).
    out = estimate(fused, fs, hint if hint is not None else f0)
    out["polarity"] = "skewness_oriented"          # rppg.pos orients each region peaks-up
    out["regions"] = len(waves)
    return out


def describe(source: str, payload, f0_hint_bpm=None) -> dict:
    """The record for one source, as it is written to debug.pulse_harmonics_<source>."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    bpm = _finite_number(f0_hint_bpm)
    hint = None if bpm is None else bpm / 60.0
    body = from_sdk_snapshot(payload, hint) if source == "sdk" else from_ingest(payload, hint)
    return dict({"version": VERSION, "mode": "recorded_only",
                 "contributes_to_published_result": False, "source": source}, **body)
