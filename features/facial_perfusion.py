"""Facial pulsatile perfusion index from live-frame ROI traces (2026-09-23).

WHAT THE VASCULAR TONE CARD MEASURES NOW, AND WHY

A camera cannot measure vascular tone (smooth-muscle constriction of the
vessel wall) directly, and no facial-camera measure of tone has been
validated anywhere. The closest quantity a face camera can measure is the
PERFUSION INDEX: the pulsatile fraction of the light the skin returns, AC/DC,
the camera analogue of the pulse-oximeter index clinicians read as a
peripheral vasomotor-tone proxy (it falls when skin vessels constrict and
rises when they relax or when the pulse pressure rises). The card reports it
for the facial skin, inverted onto the card's 0-100 scale (higher score =
weaker facial pulsatile perfusion), and prints the index beside the score.

What it is NOT, from the literature (research/vascular_tone_audit/): facial
skin barely constricts reflexively (cold-pressor effect on facial camera
amplitude -0.30 SD vs -1.30 at the finger; ear -2 % vs finger -48 %), and
facial AC/DC also moves with local heat, pulse pressure, posture, blood
volume and even mental stress. So it is a relative, uncalibrated index of
facial skin pulse strength, read against the same person's own scans on the
same device and light - not vascular resistance, endothelial function or a
diagnosis. Skin tone, glare and lighting angle change the absolute level.

WHY NOT THE PREVIOUS FORMULA. Until 2026-09-23 the card was the coefficient
of variation of per-beat pulse amplitude measured on the compressed clip.
Through that exact computation a pulse with ZERO amplitude variation plus
camera noise read 16-20, and a true 0 % vs 20 % variation read the same
(19.4 vs 19.1) at the phone clip's per-region SNR; the same moment of one
scan read 45 from the clip and 100 from the live-frame traces. It measured
the noise of the capture, not the person (research/vascular_tone_audit/).

HOW (arithmetic on the client's own per-frame region averages):
  1. Each region's mean RGB on a uniform 30 Hz clock inside contiguous
     stretches (a hole over 0.5 s splits; nothing is interpolated across a
     frame without a face). A region in view for under 60 % of the scan is
     left out instead of vetoing the others' frames.
  2. sRGB code values -> linear light (IEC 61966-2-1), so AC/DC is a ratio
     of light, not of gamma-encoded code values: the same pulse reads the
     same whether the camera exposes the face bright or dark.
  3. c(t) = L(t) / DC(t) - 1 with DC a zero-phase 0.25 Hz low-pass: slow
     exposure or lighting changes cancel in the ratio. (A 2 s moving
     average, tried first, kept ~12 % of a 72 bpm pulse inside "DC" and
     divided that much of the amplitude away.)
  4. The heart rate: the traces' own pulse peak, searched within +/- 15 bpm
     of the live-frame reference rate when the client sent one.
  5. Every region's pulse signal - its channels projected onto the blood-
     volume colour signature with the colourless (motion/shading) direction
     removed, in green-equivalent units (PULSE_SIGNATURE) - band-passed to the
     heart rate +/- 15 bpm, as an analytic signal z_k(t); per 10 s window (2 s hop) the regions'
     cross-products C_kj = <z_k conj(z_j)>, averaged over the windows without
     gross face motion or artefact-level band power (the pulse's products are
     the same in every window; independent noise averages toward zero).
  6. The pulse is common to the regions and their noise is not, so for
     k != j, |C_kj| = A_k A_j: noise sits on the diagonal and never enters.
     A region counts when its median coherence with the others clears
     MIN_REGION_COHERENCE (set above every value pure noise produced). The
     scan's index is 2 * sqrt(median |C_kj| over counting pairs), in %: the
     peak-to-trough of the heart-rate fundamental of the regions' geometric
     mean pulse. A fit against a noisy phase reference was tried first and
     read 26-88 % of a known amplitude at realistic SNR; this reads 98-106 %.

Precision comes from ~50 beats of cross-region agreement on uncompressed
frames - never from smoothing across scans, clipping, or a previous score.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

VERSION = "fpi-2026-09-23.2"
ROI_NAMES = ("forehead", "cheek_l", "cheek_r", "nose")

FS_HZ = 30.0                   # uniform analysis clock
MAX_GAP_S = 0.5                # a longer hole splits the trace
MIN_REGION_PRESENCE = 0.6      # a region in view for less of the scan is left out
DC_LOWPASS_HZ = 0.25           # the DC term: zero-phase low-pass far below any heart rate
PULSE_BAND_HZ = (0.6, 4.0)     # 36-240 bpm
HR_PLAUSIBLE_BPM = (40.0, 180.0)
HINT_SEARCH_HZ = 0.25          # +/- 15 bpm around the live-frame reference rate
PEAK_MIN_RATIO = 3.0           # a pulse peak must stand this far above the band's median power
HALF_BAND_HZ = 0.25            # regions compared in the heart rate +/- 15 bpm
# The blood-volume pulse's colour signature (relative AC/DC of R, G, B with
# green = 1; de Haan & van Leest, Physiol Meas 2014;35:1913). The pulse is
# measured along it after removing the COLOURLESS direction (1,1,1): head
# motion and shading under side light change every channel alike and, on a
# 2026-09-23 scan in one-sided light, outweighed the pulse ~7x in green
# alone. PULSE_WEIGHTS . (1,1,1) = 0 and PULSE_WEIGHTS . signature = 1, so
# the output is the GREEN-equivalent pulsatile fraction for a pulse with this
# signature; a camera/light whose true signature differs reads a constant
# factor off (R/G 0.47, B/G 0.64 -> x0.94), the same scan after scan.
PULSE_SIGNATURE = (0.43, 1.0, 0.69)
WINDOW_S = 10.0
HOP_S = 2.0
MAX_FAST_MOTION = 0.02         # RMS 0.5-4 Hz face-box motion / face width, per window
MIN_WINDOWS = 3                # low-motion windows for any estimate
ARTEFACT_POWER_RATIO = 4.0     # a window with more heart-rate-band power than this x the median is an artefact
# Region coherence is computed on noise-floor-corrected cross-products
# (_debiased). Pure noise (480 synthetic regions, 3 noise levels, 40 seeds)
# then gives a median of 0.000, a 99th percentile of 0.264 and a maximum of
# 0.281; the bar sits above every one of them.
MIN_REGION_COHERENCE = 0.30
MEASURED_MIN_REGIONS = 3
# A "measured" value also needs the counting regions to agree comfortably
# above the noise bar: just over it, the scans that clear it are the ones
# noise pushed up (synthetic: a 0.2 % pulse detected in 1 of 6 read 1.76x).
MEASURED_MIN_COHERENCE = 0.45
# A pulse whose red amplitude is this close to its green one is more motion or
# shading than blood (blood-volume pulse R/G ~0.43-0.6; live scans 0.47-0.57).
MOTION_LIKE_R_OVER_G = 0.85
MEASURED_MIN_SECONDS = 20.0
BOOTSTRAP_N = 100

# The 0-100 score is a T-score of the log index against the one published
# healthy-adult reference for facial camera AC/DC (Rasche et al., Sci Rep
# 2020;10:16464: forehead, n = 31, 0.55 +/- 0.16 %), INVERTED so that lower
# facial pulsatile perfusion reads higher on the tone scale: 50 at the
# reference's geometric mean, 10 points per between-person SD of log PI,
# i.e. ~25 points per halving/doubling. "Typical" 40-60 is the reference's
# middle ~68 %. Fixed from the literature before any repeat-scan data was
# scored. Absolute AC/DC is camera- and scene-specific (the reference used a
# 12-bit industrial camera), so the score is only as comparable across
# devices as the index is; repeat scans of one person on one device are the
# comparison it supports.
PI_REF_PERCENT = 0.528         # geometric mean of 0.55 +/- 0.16 %
PI_REF_LN_SD = 0.285           # between-person SD of ln(PI) for CV 29 %
POINTS_PER_SD = 10.0
TONE_TYPICAL = (40.0, 60.0)    # +/- 1 between-person SD (index 0.70-0.40 %)
PI_AT_SCORE_100 = round(PI_REF_PERCENT * math.exp(-5 * PI_REF_LN_SD), 4)   # 0.127 %
PI_AT_SCORE_0 = round(PI_REF_PERCENT * math.exp(5 * PI_REF_LN_SD), 4)      # 2.196 %


def _pulse_weights() -> np.ndarray:
    sig = np.asarray(PULSE_SIGNATURE, float)
    d = sig - sig.mean()
    return d / float(d @ d)


PULSE_WEIGHTS = _pulse_weights()


def srgb_to_linear(v) -> np.ndarray:
    """sRGB code values (0-255) -> linear light (0-1), IEC 61966-2-1."""
    x = np.clip(np.asarray(v, float) / 255.0, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def tone_score_from_pi(pi_percent) -> Optional[float]:
    """Facial perfusion index (%) -> the card's 0-100 tone score (inverted
    T-score against the published reference; see PI_REF_*)."""
    try:
        v = float(pi_percent)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    s = 50.0 - POINTS_PER_SD * math.log(v / PI_REF_PERCENT) / PI_REF_LN_SD
    return round(float(min(100.0, max(0.0, s))), 1)


def _bandpass(x, lo, hi, order=2):
    sos = butter(order, [lo, hi], "bandpass", fs=FS_HZ, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _lowpass(x, hi, order=2):
    sos = butter(order, hi, "lowpass", fs=FS_HZ, output="sos")
    return sosfiltfilt(sos, x, axis=0, padtype="even")


def _pos(cn: np.ndarray) -> np.ndarray:
    """POS projection (Wang 2017) of DC-normalised channels; used only to
    find the heart rate, never for the amplitude (its scale is data-driven)."""
    s1 = cn[:, 1] - cn[:, 2]
    s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]
    sd2 = float(np.std(s2))
    return s1 + (float(np.std(s1)) / sd2) * s2 if sd2 > 1e-12 else s1


def _rows(v, n) -> np.ndarray:
    out = np.full((n, 3), np.nan)
    for i, r in enumerate((v or [])[:n]):
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            try:
                out[i] = [float(r[0]), float(r[1]), float(r[2])]
            except (TypeError, ValueError):
                pass
    return out


def parse_document(doc):
    """(t, {roi: (n,3)}, bbox (n,4) or None) from a trace document, or None."""
    if not isinstance(doc, dict):
        return None
    try:
        t = np.asarray([float(x) for x in doc.get("t_s") or []], float)
    except (TypeError, ValueError):
        return None
    n = t.size
    tr = doc.get("traces") if isinstance(doc.get("traces"), dict) else {}
    rgbs = {r: _rows(tr.get(r), n) for r in ROI_NAMES if isinstance(tr.get(r), list)}
    bbox = None
    if isinstance(doc.get("bbox"), list) and len(doc["bbox"]) == n:
        bbox = np.full((n, 4), np.nan)
        for i, b in enumerate(doc["bbox"]):
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                try:
                    bbox[i] = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                except (TypeError, ValueError):
                    pass
    if n < 10 or not rgbs:
        return None
    return t, rgbs, bbox


def _stretches(t, rgbs, bbox):
    """Uniform-clock stretches (>= one window long) where every kept region
    has a value; the face box rides along when it is complete."""
    ok = np.isfinite(t)
    for a in rgbs.values():
        ok &= np.all(np.isfinite(a), axis=1)
    idx = np.flatnonzero(ok)
    if idx.size < 10:
        return []
    tt = t[idx]
    order = np.argsort(tt, kind="stable")
    idx, tt = idx[order], tt[order]
    keep = np.r_[True, np.diff(tt) > 0]
    idx, tt = idx[keep], tt[keep]
    cut = np.flatnonzero(np.diff(tt) > MAX_GAP_S) + 1
    out = []
    for a, b in zip([0] + cut.tolist(), cut.tolist() + [idx.size]):
        if b - a < 2 or tt[b - 1] - tt[a] < WINDOW_S:
            continue
        g = np.arange(tt[a], tt[b - 1], 1.0 / FS_HZ)
        seg = {r: np.stack([np.interp(g, tt[a:b], rgbs[r][idx[a:b], k]) for k in range(3)], 1)
               for r in rgbs}
        bb = None
        if bbox is not None:
            bi = bbox[idx[a:b]]
            if np.all(np.isfinite(bi)):
                bb = np.stack([np.interp(g, tt[a:b], bi[:, k]) for k in range(4)], 1)
        out.append((g, seg, bb))
    return out


def _lock_frequency(pulses, hint_hz):
    """(f0_hz, source): the traces' own pulse peak, searched within
    +/- HINT_SEARCH_HZ of the live-frame reference rate when there is one."""
    x = np.concatenate(pulses) if pulses else np.array([])
    if x.size < int(8 * FS_HZ):
        return (hint_hz, "live_frame_reference_rate") if hint_hz else (None, None)
    f, p = welch(x - x.mean(), FS_HZ, nperseg=min(x.size, int(16 * FS_HZ)), nfft=int(64 * FS_HZ))
    lo_hz, hi_hz = HR_PLAUSIBLE_BPM[0] / 60.0, HR_PLAUSIBLE_BPM[1] / 60.0
    band = (f >= PULSE_BAND_HZ[0]) & (f <= PULSE_BAND_HZ[1])
    floor = float(np.median(p[band])) if band.any() else 0.0
    if hint_hz:
        near = (f >= max(lo_hz, hint_hz - HINT_SEARCH_HZ)) & (f <= min(hi_hz, hint_hz + HINT_SEARCH_HZ))
        if near.any():
            k = int(np.argmax(np.where(near, p, -np.inf)))
            if p[k] >= PEAK_MIN_RATIO * floor:
                return float(f[k]), "trace_peak_near_reference_rate"
        return hint_hz, "live_frame_reference_rate"
    plaus = (f >= lo_hz) & (f <= hi_hz)
    k = int(np.argmax(np.where(plaus, p, -np.inf)))
    return (float(f[k]), "trace_spectral_peak") if p[k] >= PEAK_MIN_RATIO * floor else (None, None)


def _window_matrices(g, cn, bb, rois, f0, channel=None):
    """Per window: the regions' cross-product matrix of the pulse signal's
    analytic signal in f0 +/- HALF_BAND_HZ, and the window's fast face
    motion. `channel` None = the colour-projected pulse (PULSE_WEIGHTS), an
    int = one raw channel (0 R, 1 G, 2 B), or a weight vector."""
    if channel is None:
        channel = PULSE_WEIGHTS
    lo, hi = max(0.5, f0 - HALF_BAND_HZ), f0 + HALF_BAND_HZ
    pick = (lambda c: c[:, channel]) if isinstance(channel, int) else (lambda c: c @ channel)
    Z = np.stack([hilbert(_bandpass(pick(cn[r]), lo, hi, order=3)) for r in rois])
    motion = None
    if bb is not None:
        mx = _bandpass(bb[:, 0] + bb[:, 2] / 2.0, 0.5, 4.0)
        my = _bandpass(bb[:, 1] + bb[:, 3] / 2.0, 0.5, 4.0)
        motion = (mx, my, bb[:, 2])
    n, w, hop = len(g), int(WINDOW_S * FS_HZ), int(HOP_S * FS_HZ)
    out = []
    for s in range(0, n - w + 1, hop):
        sl = slice(s, s + w)
        mot = None
        if motion is not None:
            width = float(np.mean(motion[2][sl]))
            if width > 0:
                mot = float(np.hypot(np.std(motion[0][sl]), np.std(motion[1][sl])) / width)
        zz = Z[:, sl]
        # Accelerate's BLAS (macOS) can raise spurious floating-point flags
        # in this product; the result is checked for finiteness instead.
        with np.errstate(all="ignore"):
            C = (zz @ zz.conj().T) / zz.shape[1]
        if np.all(np.isfinite(C)):
            out.append({"t0": float(g[s]), "motion": mot, "C": C})
    return out


def _robust_matrix(wins) -> np.ndarray:
    """Mean cross-product matrix over the windows, after dropping any window
    whose total heart-rate-band power exceeds ARTEFACT_POWER_RATIO x the
    scan's median (head motion at the start of a scan measured ~100x). The
    trim looks at band POWER only, never at the index it would produce. A
    complex MEAN, not an element-wise median: the median of real and
    imaginary parts is not rotation-invariant and read a scan whose
    inter-region pulse phase wandered at 0.26 % while its two halves read
    0.39 and 0.31 %; the mean read 0.35 %."""
    stack = np.stack([w["C"] for w in wins])
    power = np.real(np.einsum("wkk->w", stack))
    keep = power <= ARTEFACT_POWER_RATIO * float(np.median(power))
    return stack[keep].mean(axis=0)


def _pair_index(C: np.ndarray, used) -> Optional[float]:
    """2 * sqrt(median |C_kj|) over the counting pairs, in % (None if not positive)."""
    pairs = [abs(C[k, j]) for i, k in enumerate(used) for j in used[i + 1:]]
    m = float(np.median(pairs)) if pairs else 0.0
    return 200.0 * math.sqrt(m) if m > 0 else None


def _debiased(C: np.ndarray, n_eff: float) -> np.ndarray:
    """|C_kj| with its noise floor removed: the bias-corrected coherence
    estimator applied to the cross-product, |C|^2 -> (N|C|^2 - C_kk C_jj)/(N-1)
    (unbiased for the squared coherence of Gaussian data; ~0 for pure noise,
    ~|C|^2 for a strong pulse - a plain |C|^2 - C_kk C_jj / N also took the
    pulse's own power off and read a known index 5-12 % low)."""
    d = np.maximum(np.real(np.diag(C)), 1e-30)
    n = max(float(n_eff), 2.0)
    return np.sqrt(np.maximum((n * np.abs(C) ** 2 - np.outer(d, d)) / (n - 1.0), 0.0))


def _n_eff(wins) -> float:
    """Independent samples behind a mean over these windows: the seconds
    they cover (overlaps counted once) x the analysis bandwidth."""
    spans = sorted((w["t0"], w["t0"] + WINDOW_S) for w in wins)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in spans:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return max(1.0, total * 2.0 * HALF_BAND_HZ)


def _estimate(wins, rois, n_eff: float) -> dict:
    """Scan index, per-region coherence and amplitude from the windows.

    |C_kj| of an averaged cross-product carries a noise floor: for noise-only
    regions its square averages C_kk C_jj / n_eff (n_eff = time x bandwidth,
    the independent samples behind the average). Near the detection limit
    that floor rode on the pulse and a scan that only just cleared the
    coherence bar read up to 1.6x its true index. It is removed from every
    pair before anything else (_debiased)."""
    C = _robust_matrix(wins)
    d = np.maximum(np.real(np.diag(C)), 1e-30)
    n = len(rois)
    mag = _debiased(C, n_eff)
    coh = {r: float(np.median([mag[k, j] / math.sqrt(d[k] * d[j]) for j in range(n) if j != k]))
           for k, r in enumerate(rois)}
    used = [k for k, r in enumerate(rois) if coh[r] >= MIN_REGION_COHERENCE]
    # Per-region amplitude (diagnostic only): median over the counting
    # regions' triads of |C_kj||C_kl|/|C_jl|. The scan index never divides.
    amp = {}
    for k in used:
        est = [mag[k, j] * mag[k, l] / mag[j, l] for j in used for l in used
               if len({k, j, l}) == 3 and j < l and mag[j, l] > 0]
        if est:
            amp[rois[k]] = 200.0 * math.sqrt(float(np.median(est)))
    return {"pi": _pair_index(mag, used), "coherence": coh,
            "used": [rois[k] for k in used], "used_idx": used, "region_pi": amp}


def _raw_channel_index(prepared, rois, used_idx, f0, channel):
    wins = []
    for g, cn, _, bb in prepared:
        wins += _window_matrices(g, cn, bb, rois, f0, channel=channel)
    wins = [w for w in wins if w["motion"] is None or w["motion"] <= MAX_FAST_MOTION]
    if len(wins) < MIN_WINDOWS:
        return None
    return _pair_index(_debiased(_robust_matrix(wins), _n_eff(wins)), used_idx)


def _colour_signature(prepared, rois, used_idx, f0) -> dict:
    """The raw channels' cross-region pulse amplitudes: green alone (the
    classic single-channel index, reported for comparison), and the red/green
    and blue/green ratios. Blood absorbs green far more than red, so a
    blood-volume pulse reads R/G ~0.4-0.6 (de Haan & van Leest 2014: 0.43);
    head motion and shading change every channel alike (R/G ~1). A validity
    check on WHAT pulses, never an input to the value."""
    green = _raw_channel_index(prepared, rois, used_idx, f0, 1)
    out = {"green_only_pi_percent": None if green is None else round(green, 4)}
    for ch, name in ((0, "r_over_g"), (2, "b_over_g")):
        v = _raw_channel_index(prepared, rois, used_idx, f0, ch)
        out[name] = round(v / green, 3) if v and green else None
    r = out.get("r_over_g")
    out["looks_like"] = (None if r is None else
                         "motion_or_shading" if r >= MOTION_LIKE_R_OVER_G else "blood_volume")
    return out


def facial_perfusion(doc, *, hr_hint_bpm: Optional[float] = None) -> dict:
    """The scan's facial pulsatile perfusion index from a live-frame trace
    document (schema_version 1: t_s, traces{forehead,cheek_l,cheek_r,nose},
    optional bbox), as the client posts it to /api/scan-traces.

    Returns `available`, `pi_percent`, `per_region`, window counts,
    `uncertainty`, the heart-rate source, and `reason` when no value could be
    measured. Never raises on a malformed document."""
    out = {"version": VERSION, "available": False, "pi_percent": None,
           "per_region": {}, "windows": {"total": 0, "low_motion": 0},
           "seconds_used": 0.0, "f0_hz": None, "f0_source": None}
    parsed = parse_document(doc)
    if parsed is None:
        out["reason"] = "no usable live-frame trace document"
        return out
    t, rgbs, bbox = parsed
    # Presence is judged over the frames where the face was in view at all:
    # the sampler also records the seconds before the scan starts and after
    # it ends (no face), and counting those vetoed a scan with 30 s of clean
    # face in an 83 s document.
    present = {r: np.all(np.isfinite(a), axis=1) for r, a in rgbs.items()}
    face = np.any(np.stack(list(present.values())), axis=0)
    n_face = int(face.sum())
    rgbs = {r: a for r, a in rgbs.items()
            if n_face and float(present[r][face].mean()) >= MIN_REGION_PRESENCE}
    rois = [r for r in ROI_NAMES if r in rgbs]
    out["regions_present"] = rois
    if len(rois) < 2:
        out["reason"] = "fewer than two face regions were in view for most of the scan"
        return out
    stretches = _stretches(t, rgbs, bbox)
    if not stretches:
        out["reason"] = (f"no continuous stretch of at least {WINDOW_S:.0f} s with the "
                         "face regions in view")
        return out
    prepared = []
    for g, seg, bb in stretches:
        cn, pulses = {}, {}
        for r in rois:
            lin = srgb_to_linear(seg[r])
            c = _bandpass(lin / np.maximum(_lowpass(lin, DC_LOWPASS_HZ), 1e-9) - 1.0,
                          *PULSE_BAND_HZ)
            cn[r] = c
            p = _pos(c)
            pulses[r] = p / max(float(np.std(p)), 1e-12)
        prepared.append((g, cn, pulses, bb))
    hint = None
    try:
        if hr_hint_bpm is not None and HR_PLAUSIBLE_BPM[0] <= float(hr_hint_bpm) <= HR_PLAUSIBLE_BPM[1]:
            hint = float(hr_hint_bpm) / 60.0
    except (TypeError, ValueError):
        hint = None
    f0, src = _lock_frequency([sum(p.values()) for _, _, p, _ in prepared], hint)
    if f0 is None:
        out["reason"] = "no heart rate to lock the pulse to"
        return out
    out["f0_hz"], out["f0_source"] = round(f0, 4), src
    out["reference_rate_bpm"] = None if hint is None else round(hint * 60.0, 1)
    windows = []
    for g, cn, _, bb in prepared:
        windows += _window_matrices(g, cn, bb, rois, f0)
    kept = [w for w in windows if w["motion"] is None or w["motion"] <= MAX_FAST_MOTION]
    out["windows"] = {"total": len(windows), "low_motion": len(kept)}
    out["seconds_used"] = round(sum(len(g) for g, *_ in prepared) / FS_HZ, 1)
    if len(kept) < MIN_WINDOWS:
        out["reason"] = (f"{len(kept)} low-motion {WINDOW_S:.0f} s windows "
                         f"(need {MIN_WINDOWS})")
        return out
    n_eff = _n_eff(kept)
    out["n_eff"] = round(n_eff, 1)
    est = _estimate(kept, rois, n_eff)
    out["per_region"] = {r: {"coherence": round(est["coherence"][r], 3),
                             "counts": r in est["used"],
                             "pi_percent": (round(est["region_pi"][r], 4)
                                            if r in est["region_pi"] else None)}
                         for r in rois}
    if est["pi"] is None:
        out["reason"] = ("the pulse was not common to two or more face regions above the "
                         f"noise (region coherence below {MIN_REGION_COHERENCE:.2f})")
        return out
    pi = float(est["pi"])
    out["available"] = True
    out["pi_percent"] = round(pi, 4)
    out["regions_used"] = est["used"]
    out["coherence_median"] = round(float(np.median([est["coherence"][r] for r in est["used"]])), 3)
    out["colour_signature"] = _colour_signature(prepared, rois, est["used_idx"], f0)
    # Precision: resample windows with replacement and repeat the estimate
    # with the same counting regions. Overlapping windows are correlated, so
    # this is optimistic; the repeat-scan study is the real yardstick.
    rng = np.random.default_rng(0)
    reps = []
    for _ in range(BOOTSTRAP_N):
        v = _pair_index(_debiased(_robust_matrix([kept[i] for i in rng.integers(0, len(kept), len(kept))]),
                                  n_eff),
                        est["used_idx"])
        if v:
            reps.append(math.log(v))
    if len(reps) >= BOOTSTRAP_N // 2:
        se = float(np.std(reps))
        out["uncertainty"] = {"log_se": round(se, 4),
                              "ci95_percent": [round(pi * math.exp(-1.96 * se), 4),
                                               round(pi * math.exp(1.96 * se), 4)],
                              "method": "window bootstrap (overlapping windows: optimistic)"}
    return out


def _stars_from_log_se(se: Optional[float], n_regions: int) -> int:
    if se is None:
        return 1
    stars = 5 if se <= 0.05 else 4 if se <= 0.10 else 3 if se <= 0.20 else 2 if se <= 0.35 else 1
    return min(stars, 3) if n_regions < MEASURED_MIN_REGIONS else stars


def vascular_tone_card(fp: dict, *, legacy: Optional[dict] = None) -> dict:
    """The Vascular Tone endpoint in the shape features/hemodynamics.py gives
    every card (available, tier, tier_reasons, score, estimate, raw_*,
    confidence), from facial_perfusion()'s result. `legacy` is the previous
    clip-based card, kept under `legacy_clip_cv` for the audit trail only."""
    card = {
        "available": False, "calibrated": False, "source": "live_frame_traces",
        "definition": ("facial pulsatile perfusion index: peak-to-trough of the heart-rate "
                       "fundamental as a share of the green light the facial skin returns "
                       "(AC/DC, linear light), from the pulse common to the face regions over "
                       "the low-motion part of the scan. Score = inverted T-score of the log "
                       "index against published healthy-adult facial camera values (Rasche "
                       "2020: 0.55 +/- 0.16 %): 50 at 0.53 %, 10 points per between-person SD; "
                       "a LOWER index reads as a HIGHER score."),
        "interpretation": ("relative, uncalibrated index of facial skin pulse strength (facial "
                           "pulsatile perfusion), shown inverted as a tone score: it falls when "
                           "skin vessels constrict or the pulse pressure drops and rises with "
                           "dilation, warmth or a stronger pulse. Facial skin constricts little "
                           "reflexively, so this is a proxy, not a measurement of vascular "
                           "tone, vascular resistance or endothelial function. Compare it with "
                           "your own scans on the same device, in the same light."),
        "not_reactivity": ("vasomotor REACTIVITY is a response to a timed provocation and "
                           "is not derivable from one resting scan"),
        "facial_perfusion": fp,
    }
    if legacy is not None:
        card["legacy_clip_cv"] = {k: legacy.get(k) for k in
                                  ("available", "amplitude_cv", "score", "n_beats", "tier")}
    if not fp.get("available"):
        card["reason"] = fp.get("reason") or "facial perfusion could not be measured"
        return card
    pi = float(fp["pi_percent"])
    score = tone_score_from_pi(pi)
    n_reg = len(fp.get("regions_used") or [])
    tier_reasons = []
    if n_reg < MEASURED_MIN_REGIONS:
        tier_reasons.append(f"the pulse was common to {n_reg} face regions "
                            f"(a measured value needs {MEASURED_MIN_REGIONS})")
    coh = fp.get("coherence_median")
    if coh is not None and coh < MEASURED_MIN_COHERENCE:
        tier_reasons.append(f"the face regions agree on the pulse only weakly (coherence {coh:.2f}; "
                            f"a measured value needs {MEASURED_MIN_COHERENCE:.2f}), so this value "
                            "may read high")
    if (fp.get("colour_signature") or {}).get("looks_like") == "motion_or_shading":
        tier_reasons.append("the pulse's colour pattern looks more like head motion or "
                            "shading than blood (red nearly as strong as green)")
    if float(fp.get("seconds_used") or 0.0) < MEASURED_MIN_SECONDS:
        tier_reasons.append(f"{fp.get('seconds_used')} s of continuous face trace "
                            f"(a measured value needs {MEASURED_MIN_SECONDS:.0f})")
    se = (fp.get("uncertainty") or {}).get("log_se")
    card.update({
        "available": True,
        "tier": "provisional" if tier_reasons else "measured",
        "tier_reasons": tier_reasons,
        "score": score,
        "score_typical_range": list(TONE_TYPICAL),
        "raw_name": "facial_perfusion_index_percent",
        "raw_value": round(pi, 2),
        "raw_unit": "%",
        "estimate": {"label": "Research Estimate / Prototype", "name": "vascular_tone_index",
                     "value": score, "unit": "/100",
                     "method": ("0-100 inverse log scale of the facial pulsatile perfusion "
                                "index measured on the live camera frames; uncalibrated")},
        "confidence": {"kind": "signal_evidence_not_endpoint_accuracy",
                       "confidence_stars": _stars_from_log_se(se, n_reg),
                       "log_se": se, "regions_used": fp.get("regions_used"),
                       "windows": fp.get("windows"), "f0_source": fp.get("f0_source"),
                       "colour_signature": fp.get("colour_signature")},
    })
    return card
