"""
PRBS LED-marker synchronisation (T1).

The collection rig flashes an LED in a known pseudo-random binary sequence
(PRBS) at the start AND end of every recording, and logs each flash onset to
the ECG event channel via a shared hardware trigger. This module recovers the
video/ECG clock mapping from the two observations:

    video_t = ecg_t * (1 + drift_ppm * 1e-6) + offset_ms / 1000

PHYSICS (v2 correction, encoded in datasets/schema.py): a single flash is
bounded by ±half a frame period, so it can never meet the 5 ms budget at
consumer frame rates. A >=10-event sequence reaches ~frame/sqrt(12*K). The
uncertainty this module reports is floored at that quantisation bound — it
must never claim better than physics allows, no matter how small the fit
residuals are.

Sub-frame recovery is possible because a camera INTEGRATES light over each
frame's exposure: the one frame that straddles a flash onset has a brightness
proportional to the fraction of the frame the LED was on, which localises the
edge to well under a frame. Cross-correlation against the ECG *signal* is
deliberately absent — the schema rejects it as circular.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .schema import SyncRecord, SyncMethod

CALIBRATION_NOISE_FLOOR = 1e-12

# Maximal-length LFSR tap positions (1-indexed, Fibonacci form).
_LFSR_TAPS = {3: (3, 2), 4: (4, 3), 5: (5, 3), 6: (6, 5),
              7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5)}


def prbs_chips(order: int = 6) -> np.ndarray:
    """Maximal-length PRBS chip sequence of period 2**order - 1 (0/1 ints).

    The same generator runs on the capture rig and in the tests, so the
    template is a shared, versioned artifact rather than a magic array.
    """
    if order not in _LFSR_TAPS:
        raise ValueError(f"unsupported LFSR order {order}")
    taps = _LFSR_TAPS[order]
    state = [1] * order
    out = []
    for _ in range(2 ** order - 1):
        out.append(state[-1])
        fb = 0
        for t in taps:
            fb ^= state[t - 1]
        state = [fb] + state[:-1]
    return np.asarray(out, int)


def chip_onset_times(chips: np.ndarray, chip_s: float) -> np.ndarray:
    """Times (s, relative to burst start) of every 0->1 chip transition."""
    c = np.asarray(chips, int)
    prev = np.concatenate([[0], c[:-1]])
    return np.flatnonzero((c == 1) & (prev == 0)) * float(chip_s)


def chip_edge_times(chips: np.ndarray, chip_s: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(times, direction +1/-1) of EVERY chip transition, both polarities.

    Used internally to anchor each burst's lag: the brightness lo/hi level
    estimates are shared across a burst, and a level error shifts every
    RISING edge one way and every FALLING edge the other — averaging both
    polarities cancels the correlated term that otherwise turns into a
    drift error between the start and end bursts (measured: up to ~20 ppm
    at 30 fps with rising-only estimation).
    """
    c = np.asarray(chips, int)
    prev = np.concatenate([[0], c[:-1]])
    rise = np.flatnonzero((c == 1) & (prev == 0))
    fall = np.flatnonzero((c == 0) & (prev == 1))
    t = np.concatenate([rise, fall]) * float(chip_s)
    d = np.concatenate([np.ones(rise.size), -np.ones(fall.size)])
    order = np.argsort(t)
    return t[order], d[order]


def _integrated_template(chips: np.ndarray, chip_s: float, fps: float,
                         sub: int = 16) -> np.ndarray:
    """Expected frame-mean brightness of one burst starting at frame 0."""
    chips = np.asarray(chips, float)
    n = int(np.ceil(len(chips) * chip_s * fps))
    starts = np.arange(n)[:, None] / fps
    t_fine = starts + np.arange(sub)[None, :] / (sub * fps)
    idx = np.floor(t_fine / chip_s).astype(int)
    inside = (idx >= 0) & (idx < len(chips))
    led = np.where(inside, chips[np.clip(idx, 0, len(chips) - 1)], 0.0)
    return led.mean(axis=1)


def detect_marker_events(brightness: np.ndarray, fps: float,
                         template: np.ndarray, chip_s: float = 0.1037,
                         min_corr: float = 0.55) -> np.ndarray:
    """Detect PRBS flash-onset times (s, video clock) in a brightness trace.

    `template` is the PRBS chip sequence (0/1 per chip). Every burst of the
    sequence present in the trace is found by normalised cross-correlation;
    each onset within a burst is then refined to sub-frame precision from the
    exposure-integrated brightness of the frame straddling the edge.

    RIG CONSTRAINT: the chip period must be INCOMMENSURATE with the frame
    period (default 103.7 ms, not a round 100 ms). At 30 fps a 100 ms chip is
    exactly 3 frames, so every edge in a burst lands at the same sub-frame
    phase and quantisation error becomes a coherent per-burst bias instead of
    dithering out — measured effect: drift error grows ~3x past its budget.

    Returns an empty array when no burst clears `min_corr` — fail closed,
    never guess.
    """
    x = np.asarray(brightness, float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return np.array([])
    w = _integrated_template(template, chip_s, fps)
    m = w.size
    if x.size < m + 2:
        return np.array([])

    xw = x - np.median(x)
    wz = w - w.mean()
    num = np.correlate(xw, wz, mode="valid")
    c1 = np.cumsum(np.concatenate([[0.0], xw]))
    c2 = np.cumsum(np.concatenate([[0.0], xw ** 2]))
    s = c1[m:] - c1[:-m]
    ss = c2[m:] - c2[:-m]
    var = np.maximum(ss / m - (s / m) ** 2, CALIBRATION_NOISE_FLOOR)
    corr = num / (np.sqrt(var * m) * (np.linalg.norm(wz) + 1e-12))

    # Non-maximum suppression: one peak per burst, bursts >= template apart.
    lags = []
    c = corr.copy()
    while True:
        k = int(np.argmax(c))
        if c[k] < min_corr:
            break
        lags.append(k)
        c[max(0, k - m):k + m] = -np.inf
    if not lags:
        return np.array([])

    onsets = chip_onset_times(template, chip_s)
    edge_t, edge_dir = chip_edge_times(template, chip_s)
    frame = 1.0 / fps
    events = []
    half_win = max(int(round(0.6 * chip_s * fps)), 2)
    for l0 in sorted(lags):
        seg = x[l0:l0 + m]
        lo = float(np.percentile(seg, 10))
        hi = float(np.percentile(seg, 90))
        if hi - lo <= 0:
            continue
        t0 = l0 * frame                              # burst start, video clock
        # Estimate EVERY chip transition (both polarities) individually,
        # then anchor the burst with the trimmed-mean lag. Rationale:
        # (a) a lo/hi level error shifts rising and falling edges in
        #     OPPOSITE directions, so mixing polarities cancels the
        #     burst-correlated error that regression would otherwise turn
        #     into drift (measured up to ~20 ppm at 30 fps rising-only);
        # (b) ~2x the observations halve the independent noise too.
        lags_obs = []
        for te, dr in zip(edge_t, edge_dir):
            j0 = int(np.floor((t0 + te) * fps))
            a = max(0, j0 - half_win)
            b = min(x.size - 1, j0 + half_win)
            if b - a < 2:
                continue
            f = (x[a:b + 1] - lo) / (hi - lo)
            cross = None
            for k in range(f.size - 1):
                if dr > 0 and f[k] < 0.5 <= f[k + 1]:
                    cross = k + 1
                    break
                if dr < 0 and f[k] >= 0.5 > f[k + 1]:
                    cross = k + 1
                    break
            if cross is None:
                continue
            # Exactly one frame straddles the edge and holds a FRACTIONAL
            # value (fraction of the exposure the LED was ON). For a rising
            # edge the LED was on for the LAST f of the frame; for a
            # falling edge, the FIRST f.
            fk = float(f[cross - 1])
            fc = float(f[cross])
            if dr > 0:
                if 0.05 < fc < 0.95:
                    tau = (a + cross + 1 - fc) * frame
                elif 0.05 < fk < 0.95:
                    tau = (a + cross - fk) * frame
                else:                            # edge on a frame boundary
                    tau = (a + cross) * frame
            else:
                if 0.05 < fk < 0.95:
                    tau = (a + cross - 1 + fk) * frame
                elif 0.05 < fc < 0.95:
                    tau = (a + cross + fc) * frame
                else:
                    tau = (a + cross) * frame
            lags_obs.append(tau - (t0 + te))
        if len(lags_obs) < max(4, edge_t.size // 4):
            continue                                 # too few edges resolved
        lags_obs = np.asarray(lags_obs)
        med = float(np.median(lags_obs))
        mad = float(np.median(np.abs(lags_obs - med))) + 1e-9
        keep = np.abs(lags_obs - med) <= 4.0 * mad + 1e-6
        lag = float(np.mean(lags_obs[keep]))
        # Emit the ONSET events (the schema's marker events) anchored to
        # the polarity-balanced burst lag.
        events.extend((t0 + onsets + lag).tolist())
    return np.sort(np.asarray(events))


def ecg_to_video_clock(times_s: np.ndarray, sync: SyncRecord) -> np.ndarray:
    """THE sanctioned ECG->video mapping: applies the FULL fitted model,
    video_t = ecg_t * (1 + drift_ppm*1e-6) + offset_ms/1000.

    Dropping the drift term is not a shortcut: a well-fitted 50 ppm drift
    passes the 5 ms sync gate (the fit residuals are small) while an
    offset-only mapping is off by 4.5 ms at the end of a 90 s recording —
    a systematic error the size of the entire sync budget.
    """
    t = np.asarray(times_s, float)
    drift = (sync.drift_ppm or 0.0) * 1e-6
    return t * (1.0 + drift) + sync.offset_ms / 1000.0


def _pair_events(video_s: np.ndarray, ecg_s: np.ndarray,
                 window_s: float = 0.030, tol_s: float = 0.050
                 ) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one pairing of video and ECG events via the densest offset.

    All pairwise differences are scanned for the `window_s`-wide band with
    the most support (the true offset; spurious pairs scatter); events are
    then greedily matched within `tol_s` of that offset.
    """
    if video_s.size == 0 or ecg_s.size == 0:
        return np.array([]), np.array([])
    d = np.sort((video_s[:, None] - ecg_s[None, :]).ravel())
    j = np.searchsorted(d, d + window_s, side="right")
    counts = j - np.arange(d.size)
    k = int(np.argmax(counts))
    coarse = float(np.median(d[k:j[k]]))

    cand = []
    for i, v in enumerate(video_s):
        for jj, e in enumerate(ecg_s):
            err = abs(v - e - coarse)
            if err <= tol_s:
                cand.append((err, i, jj))
    cand.sort()
    used_v, used_e, pv, pe = set(), set(), [], []
    for _, i, jj in cand:
        if i in used_v or jj in used_e:
            continue
        used_v.add(i); used_e.add(jj)
        pv.append(i); pe.append(jj)
    order = np.argsort(pe)
    return video_s[np.asarray(pv, int)[order]], ecg_s[np.asarray(pe, int)[order]]


def estimate_sync(video_events_s: np.ndarray, ecg_events_s: np.ndarray, *,
                  recording_duration_s: Optional[float] = None,
                  fps: Optional[float] = None) -> SyncRecord:
    """Fit offset + drift from paired marker events; report HONEST uncertainty.

    Always returns a SyncRecord rather than raising: an unusable sync is
    expressed as a record that fails `is_valid_for_beat_analysis()`, with the
    gate producing the reasons. NaN events are discarded before pairing, so
    corrupt input degrades the marker count instead of poisoning the fit.

    The reported uncertainty is max(fit standard error at the worst-covered
    end of the recording, frame/sqrt(12*K) quantisation floor). When `fps` is
    unknown the floor conservatively assumes 30 fps.
    """
    v = np.asarray(video_events_s, float)
    e = np.asarray(ecg_events_s, float)
    v = np.sort(v[np.isfinite(v)])
    e = np.sort(e[np.isfinite(e)])
    pv, pe = _pair_events(v, e)
    k = int(pv.size)

    frame_ms = 1000.0 / (fps if fps else 30.0)
    floor_ms = frame_ms / np.sqrt(12.0 * max(k, 1))

    if k == 0:
        return SyncRecord(SyncMethod.LED_FLASH_MARKER, offset_ms=float("nan"),
                          sync_uncertainty_ms=float("inf"), drift_ppm=None,
                          verified_at_end=False, n_marker_events=0)

    span = float(pe[-1] - pe[0]) if k > 1 else 0.0
    if k >= 6 and span >= 10.0:
        # video = a * ecg + b, least squares.
        A = np.vstack([pe, np.ones(k)]).T
        (a, b), res, _, _ = np.linalg.lstsq(A, pv, rcond=None)
        drift_ppm: Optional[float] = float((a - 1.0) * 1e6)
        offset_ms = float(b * 1000.0)
        r = pv - (a * pe + b)
        sigma = float(np.std(r, ddof=2)) if k > 2 else 0.0
        ebar = float(np.mean(pe))
        sxx = float(np.sum((pe - ebar) ** 2))
        se_ends = max(
            sigma * np.sqrt(1.0 / k + (t - ebar) ** 2 / sxx)
            for t in (float(pe[0]), float(pe[-1]))) if sxx > 0 else sigma
        unc_ms = max(se_ends * 1000.0, floor_ms)
    else:
        # Too few events / too short a span for a drift fit: offset only.
        diffs = pv - pe
        offset_ms = float(np.mean(diffs) * 1000.0)
        drift_ppm = None
        sigma = float(np.std(diffs, ddof=1)) if k > 1 else 0.0
        unc_ms = max(sigma / np.sqrt(k) * 1000.0, floor_ms)

    verified = False
    if k >= 2:
        if recording_duration_s:
            verified = (pv[0] <= 0.3 * recording_duration_s
                        and pv[-1] >= 0.7 * recording_duration_s)
        else:
            gaps = np.diff(pv)
            if gaps.size >= 3:
                g = float(np.max(gaps))
                verified = g > 5.0 * float(np.median(gaps)) and \
                    np.sum(pv < pv[0] + g) >= 2
    return SyncRecord(SyncMethod.LED_FLASH_MARKER, offset_ms=offset_ms,
                      sync_uncertainty_ms=float(unc_ms), drift_ppm=drift_ppm,
                      verified_at_end=bool(verified), n_marker_events=k)
