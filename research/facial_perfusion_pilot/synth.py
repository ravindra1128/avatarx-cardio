"""Synthetic generators for FPR pipeline validation.

Two kinds:

  * signal-level  -- a green ROI-mean trace with a KNOWN DC and a KNOWN
    fractional pulse modulation, so the recovered PI has a closed-form
    expected value. This is the authoritative input for B7 (processing
    correctness).

  * video-level   -- an FFV1 (lossless) clip of a flat patch modulated in
    green plus optional Gaussian sensor noise, written with a per-frame
    timestamp sidecar. This exercises the DC-preserving extraction path
    (spatial averaging + 8-bit quantisation) end to end.

Convention: for green(t) = DC * (1 + m * sin(2*pi*f*t)), the per-cycle
peak-to-trough AC is 2*m*DC and the cycle-mean DC is DC, so the recovered

    PI = AC / DC = 2 * m      (i.e. PI_percent = 200 * m).

Everything here is clearly SYNTHETIC and is never a substitute for a real
rig recording; it validates the PROCESSING, not the instrument.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

import numpy as np

try:
    import cv2
except ImportError:                                  # pragma: no cover
    cv2 = None


def expected_pi(modulation: float) -> float:
    """Closed-form PI for a sinusoid of fractional modulation `m`."""
    return 2.0 * float(modulation)


@dataclass
class SynthTrace:
    t: np.ndarray
    green: np.ndarray
    fps: float
    hr_hz: float
    modulation: float          # may be an array-like per-sample envelope
    dc: float


def synth_green_trace(*, fps: float = 30.0, duration_s: float = 30.0,
                      hr_hz: float = 1.2, modulation: float = 0.005,
                      dc: float = 140.0, noise_frac: float = 0.0,
                      seed: int = 0) -> SynthTrace:
    """Green ROI-mean trace: dc * (1 + m*sin(2*pi*hr*t)) + noise.

    `noise_frac` is Gaussian noise SD as a fraction of dc (measurement noise
    on the already-averaged ROI mean). `modulation` may be a scalar or a
    per-sample array (to inject a step change for the dPI test).
    """
    n = int(round(duration_s * fps))
    t = np.arange(n) / fps
    m = np.asarray(modulation, float)
    if m.ndim == 0:
        m = np.full(n, float(modulation))
    elif m.size != n:
        raise ValueError("modulation envelope must match sample count")
    green = dc * (1.0 + m * np.sin(2.0 * np.pi * hr_hz * t))
    if noise_frac > 0:
        rng = np.random.default_rng(seed)
        green = green + rng.normal(0.0, noise_frac * dc, n)
    return SynthTrace(t=t, green=green, fps=fps, hr_hz=hr_hz,
                      modulation=modulation, dc=dc)


def synth_step_trace(*, fps: float = 30.0, baseline_s: float = 60.0,
                     response_s: float = 60.0, hr_hz: float = 1.2,
                     m_baseline: float = 0.005, m_response: float = 0.007,
                     dc: float = 140.0, noise_frac: float = 0.0,
                     seed: int = 0) -> tuple[SynthTrace, float, float]:
    """Baseline then a step change in modulation. Returns (trace,
    baseline_end_s, expected_delta_pi_pct)."""
    n_b = int(round(baseline_s * fps))
    n_r = int(round(response_s * fps))
    env = np.concatenate([np.full(n_b, m_baseline),
                          np.full(n_r, m_response)])
    tr = synth_green_trace(fps=fps, duration_s=(n_b + n_r) / fps, hr_hz=hr_hz,
                           modulation=env, dc=dc, noise_frac=noise_frac,
                           seed=seed)
    expected_delta = (expected_pi(m_response) / expected_pi(m_baseline)
                      - 1.0) * 100.0
    return tr, float(baseline_s), float(expected_delta)


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("opencv (cv2) is required to write synthetic "
                           "video; install opencv-python-headless")


def write_synth_video(path: str, *, fps: float = 30.0, duration_s: float = 20.0,
                      hr_hz: float = 1.2, modulation: float = 0.01,
                      dc_luma: float = 140.0, size: int = 200,
                      noise_lsb: float = 1.0, seed: int = 0) -> str:
    """Write an FFV1 clip of a flat patch whose GREEN channel carries the
    modulation, plus a `<path>.timestamps.json` sidecar.

    `noise_lsb` adds per-pixel Gaussian noise (in 8-bit LSB) that DITHERS the
    sub-LSB modulation so spatial averaging can recover it -- this is exactly
    the quantisation regime B0/B1 characterise. Returns `path`.
    """
    _require_cv2()
    import json
    n = int(round(duration_s * fps))
    t = np.arange(n) / fps
    m = np.asarray(modulation, float)
    m_arr = np.full(n, float(modulation)) if m.ndim == 0 else m
    if m_arr.size != n:
        raise ValueError("modulation envelope must match the frame count")
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"),
                             float(fps), (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open FFV1 writer for {path}")
    try:
        for i in range(n):
            g = dc_luma * (1.0 + m_arr[i] * np.sin(2.0 * np.pi * hr_hz * t[i]))
            frame = np.empty((size, size, 3), np.float64)
            frame[..., 0] = dc_luma                  # B
            frame[..., 1] = g                        # G (BGR order)
            frame[..., 2] = dc_luma                  # R
            if noise_lsb > 0:
                frame = frame + rng.normal(0.0, noise_lsb, frame.shape)
            writer.write(np.clip(np.round(frame), 0, 255).astype(np.uint8))
    finally:
        writer.release()
    with open(path + ".timestamps.json", "w") as f:
        json.dump({"timestamps_s": t.tolist()}, f)
    return path


def write_flat_field_video(path: str, *, fps: float = 30.0,
                           duration_s: float = 2.0, size: int = 200,
                           luma: float = 140.0, gradient: float = 0.0,
                           noise_lsb: float = 1.0, seed: int = 1) -> str:
    """A uniform grey card, optionally with a corner-to-corner brightness
    `gradient` (fraction of `luma`) to exercise the B4 uniformity metric."""
    _require_cv2()
    import json
    n = int(round(duration_s * fps))
    rng = np.random.default_rng(seed)
    ramp = np.linspace(1.0 - gradient / 2.0, 1.0 + gradient / 2.0, size)
    field = (luma * ramp)[None, :] * np.ones((size, 1))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"),
                             float(fps), (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open FFV1 writer for {path}")
    try:
        for _ in range(n):
            frame = np.repeat(field[..., None], 3, axis=2)
            if noise_lsb > 0:
                frame = frame + rng.normal(0.0, noise_lsb, frame.shape)
            writer.write(np.clip(np.round(frame), 0, 255).astype(np.uint8))
    finally:
        writer.release()
    with open(path + ".timestamps.json", "w") as f:
        json.dump({"timestamps_s": (np.arange(n) / fps).tolist()}, f)
    return path


def write_phantom_video(path: str, *, fps: float = 30.0, duration_s: float = 30.0,
                        size: int = 200, luma: float = 140.0,
                        dc_drift_frac: float = 0.0, noise_lsb: float = 1.0,
                        seed: int = 2) -> str:
    """A non-perfused phantom: NO pulse modulation. `dc_drift_frac` is a slow
    linear DC drift over the clip (fraction of `luma`) to exercise the B6
    drift metric. A correct instrument yields a FLAT PI here."""
    _require_cv2()
    import json
    n = int(round(duration_s * fps))
    rng = np.random.default_rng(seed)
    drift = np.linspace(0.0, dc_drift_frac * luma, n)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"),
                             float(fps), (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open FFV1 writer for {path}")
    try:
        for i in range(n):
            frame = np.full((size, size, 3), luma + drift[i], np.float64)
            if noise_lsb > 0:
                frame = frame + rng.normal(0.0, noise_lsb, frame.shape)
            writer.write(np.clip(np.round(frame), 0, 255).astype(np.uint8))
    finally:
        writer.release()
    with open(path + ".timestamps.json", "w") as f:
        json.dump({"timestamps_s": (np.arange(n) / fps).tolist()}, f)
    return path
