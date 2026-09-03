"""
Video decoding + capture-provenance measurement (T3).

Reads a video FILE (the v0.1 offline pipeline; no camera control) and
produces the measured facts the schema's capture gate needs: fps mean and
jitter, codec, bitrate, and an illuminance proxy.

HONESTY NOTES
-------------
* Codec/bitrate come from container metadata and file size. An unknown lossy
  codec, or lossy with no measurable bitrate, FAILS the schema gate — that is
  correct behaviour, not an error to work around: unknown compression is
  indistinguishable from fatal compression.
* `lux_proxy` is a PLACEHOLDER mapping from mean luma (500 lux at luma 160,
  linear), calibrated for nothing. It exists so a clearly-dark file fails
  closed. Rig recordings carry a metered value in the manifest, which always
  overrides this proxy.
* AE/AWB/gain locks are not recoverable from a file. `build_capture_config`
  defaults them to the rig guarantee (locked); pass `assume_rig_locks=False`
  for footage of unknown origin, which will then fail the gate — again, the
  correct fail-closed behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional
import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from datasets.schema import CaptureConfig

LUX_PROXY_AT_LUMA_160 = 500.0
LOSSLESS_FOURCC = {"ffv1", "hfyu", "rawv", "y800", "i420"}


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("opencv (cv2) is required for video reading; "
                           "install opencv-python-headless")


def _fourcc_str(v: float) -> str:
    i = int(v)
    s = "".join(chr((i >> (8 * k)) & 0xFF) for k in range(4))
    return s.strip("\x00 ").lower()


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    n_frames: int
    nominal_fps: float
    measured_fps_mean: float
    measured_fps_jitter_ms: Optional[float]
    duration_s: float
    codec_fourcc: str
    file_bitrate_mbps: Optional[float]
    mean_luma: float
    lux_proxy: float
    duplicate_frame_fraction: float = 0.0
    max_duplicate_run: int = 0
    collapsed_interval_fraction: float = 0.0
    bright_clip_fraction: float = 0.0
    dark_clip_fraction: float = 0.0


def load_timestamp_sidecar(path: str) -> Optional[np.ndarray]:
    """Per-frame capture timestamps written beside a video by a live
    capture (`<video>.timestamps.json`, {"timestamps_s": [...]}). Browser or
    camera captures are written at a NOMINAL container fps; the sidecar is
    the honest clock and, when present, overrides the container."""
    import json
    p = path + ".timestamps.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    ts = np.asarray(d.get("timestamps_s", []), float)
    if ts.size == 0 or not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
        raise IOError(f"timestamp sidecar {p} is empty or non-monotone")
    return ts


def iter_frames(path: str) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_s, BGR frame). Timestamps come from the sidecar when
    present, else the container's clock (CAP_PROP_POS_MSEC), else index/fps."""
    _require_cv2()
    side = load_timestamp_sidecar(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    last_t = -1.0
    try:
        while True:
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, frame = cap.read()
            if not ok:
                break
            if side is not None:
                if i >= side.size:
                    raise IOError("timestamp sidecar shorter than video")
                t = float(side[i])
            else:
                t = t_ms / 1000.0
                if not np.isfinite(t) or t <= last_t:
                    t = i / fps
                if t <= last_t:                      # still degenerate
                    t = last_t + 1.0 / fps
            yield t, frame
            last_t = t
            i += 1
        if side is not None and i != side.size:
            raise IOError("timestamp sidecar longer than video")
    finally:
        cap.release()


def probe_video(path: str, luma_stride: int = 5) -> VideoMeta:
    """One decoding pass: frame count, measured fps stats, mean luma."""
    _require_cv2()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    nominal = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    ts, lumas, n = [], [], 0
    bright_fracs, dark_fracs = [], []
    duplicate_frames = duplicate_run = max_duplicate_run = 0
    previous_sample = None
    for t, frame in iter_frames(path):
        ts.append(t)
        # Exact repeated decoded content is a camera/transport delivery
        # failure, not extra physiological evidence.  Compare a spatial
        # sample to keep the probe streaming and cheap.  We intentionally do
        # not use a near-duplicate threshold: a still face has real sensor
        # noise and tiny perfusion changes that must not be discarded.
        sample = np.ascontiguousarray(frame[::8, ::8])
        if previous_sample is not None and sample.shape == previous_sample.shape \
                and np.array_equal(sample, previous_sample):
            duplicate_frames += 1
            duplicate_run += 1
            max_duplicate_run = max(max_duplicate_run, duplicate_run)
        else:
            duplicate_run = 0
        previous_sample = sample.copy()
        if n % luma_stride == 0:
            # ITU-R BT.601 luma from BGR
            b = frame[..., 0].astype(float)
            g = frame[..., 1].astype(float)
            r = frame[..., 2].astype(float)
            lum = 0.114 * b + 0.587 * g + 0.299 * r
            lumas.append(float(np.mean(lum)))
            bright_fracs.append(float(np.mean(lum >= 250.0)))
            dark_fracs.append(float(np.mean(lum <= 20.0)))
        n += 1
    if n < 2:
        raise IOError(f"video {path} has fewer than 2 decodable frames")

    dt = np.diff(np.asarray(ts))
    # The frame RATE is the typical frame period (median), so an intentional
    # pause or a dropped batch (a hole in the capture clock, handled as a
    # segment boundary downstream) does not masquerade as a slow camera.
    # Jitter is judged on the non-hole periods. Report to 0.01 fps so a
    # nominal-30 camera at 33.3334 ms is not judged "below 30" by float dust.
    med = float(np.median(dt))
    fps_mean = float(round(1.0 / med, 2)) if med > 0 else 0.0
    normal = dt[(dt >= 0.5 * med) & (dt < 2.5 * med)] if med > 0 else dt
    jitter_ms = float(np.std(normal) * 1000.0) if normal.size else 0.0
    duration = float(ts[-1] - ts[0] + med)
    size_bits = os.path.getsize(path) * 8
    bitrate = float(size_bits / duration / 1e6) if duration > 0 else None
    luma = float(np.mean(lumas)) if lumas else float("nan")

    return VideoMeta(
        path=path, width=width, height=height, n_frames=n,
        nominal_fps=nominal or fps_mean, measured_fps_mean=fps_mean,
        measured_fps_jitter_ms=jitter_ms, duration_s=duration,
        codec_fourcc=fourcc,
        file_bitrate_mbps=None if fourcc in LOSSLESS_FOURCC else bitrate,
        mean_luma=luma,
        lux_proxy=float(LUX_PROXY_AT_LUMA_160 * luma / 160.0),
        duplicate_frame_fraction=float(duplicate_frames / max(n - 1, 1)),
        max_duplicate_run=int(max_duplicate_run),
        collapsed_interval_fraction=float(np.mean(dt < 0.5 * med))
        if med > 0 else 1.0,
        bright_clip_fraction=float(np.mean(bright_fracs))
        if bright_fracs else 0.0,
        dark_clip_fraction=float(np.mean(dark_fracs))
        if dark_fracs else 0.0,
    )


def build_capture_config(meta: VideoMeta, *,
                         illuminance_lux: Optional[float] = None,
                         assume_rig_locks: bool = True,
                         exposure_locked: Optional[bool] = None,
                         awb_locked: Optional[bool] = None,
                         phone_model: str = "offline-file",
                         mount: str = "tripod") -> CaptureConfig:
    """CaptureConfig from measured file facts.

    `illuminance_lux` (a metered manifest value) overrides the luma proxy.
    `exposure_locked` / `awb_locked`, when given, are the camera's REPORTED
    states (e.g. from the browser track settings) and override the rig
    assumption — record what the device said, never what we hoped.
    """
    fourcc = meta.codec_fourcc
    codec = "ffv1" if fourcc in LOSSLESS_FOURCC else (
        "mjpeg" if fourcc == "mjpg" else (fourcc or "unknown"))
    return CaptureConfig(
        phone_model=phone_model, os_version="n/a", camera="file",
        width=meta.width, height=meta.height,
        nominal_fps=meta.nominal_fps,
        measured_fps_mean=meta.measured_fps_mean,
        measured_fps_jitter_ms=meta.measured_fps_jitter_ms,
        codec=codec, crf=None,
        bitrate_mbps=meta.file_bitrate_mbps,
        exposure_locked=(assume_rig_locks if exposure_locked is None
                         else bool(exposure_locked)),
        awb_locked=(assume_rig_locks if awb_locked is None
                    else bool(awb_locked)),
        gain_locked=assume_rig_locks, beautification_disabled=True,
        illuminance_lux_mean=(illuminance_lux if illuminance_lux is not None
                              else meta.lux_proxy),
        mount=mount,
    )
