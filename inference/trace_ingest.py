"""Live-frame ROI traces from the phone as pipeline input (2026-09-17).

WHY. Every consumer-camera AF result with clinical numbers reads the pulse
from LIVE, uncompressed frames; the floor for a usable rPPG signal is
~10 Mb/s (McDuff 2017) and Android Chrome caps our recorder near 7. Our own
harness measured the cost: cross-region correlation 0.33 on the rig, 0.16
on the phone clip, one beat in five surviving fusion, and on 2026-09-17 our
beat-timing gate (40-53 ms) blocked every scan the ShenAI train was right
on. The client can sample our four regions' mean RGB from the frames the
SDK already receives - a few hundred bytes per frame, no image, no codec -
and post them beside the clip. This module turns that document into the
same IngestResult capture/ingest.py produces from a video, so the ENTIRE
pipeline (extraction, fusion, calibration, gates, decision, cards) runs on
it unchanged.

HOW IT ENTERS THE PIPELINE WITHOUT TOUCHING IT. inference/pipeline.py is
hook-protected and opens every run with ingest_video(video_path). A
prebuilt IngestResult is REGISTERED under a sentinel path
("traces://<id>") and the module's ingest_video name is wrapped once: a
registered path returns its result and is forgotten; any other path goes
to the real reader. Concurrent video jobs are untouched.

WHAT IT IS, TODAY. A RECORDED path: the trace-path result travels under
debug.trace_path and the sheet's Trace columns beside the video path's and
the ShenAI train's, on every scan, so the three can be compared live for
as long as it takes to trust it. It does not decide afib_result until that
comparison says it should (owner decision).

Client document (schema_version 1, posted to /api/scan-traces):
  t_s        per-frame presentation time, seconds, monotone
  traces     {forehead, cheek_l, cheek_r, nose: [[r,g,b] | null, ...]}
             mean RGB (0-255) of the region, null when the face was absent
  bbox       optional [[x,y,w,h], ...] normalised face box per frame
  luma       optional per-frame face luma (0-255)
  width, height, fps_nominal, sampler ({"roi_px": ..., "trim": ...})
"""
from __future__ import annotations

import math
import threading
import uuid
from typing import Optional

import numpy as np

from capture.ingest import IngestResult, apply_capture_gate
from capture.face_tracking import TrackSummary
from capture.video_reader import VideoMeta, build_capture_config, LUX_PROXY_AT_LUMA_160
from preprocessing.roi import ROI_NAMES

SCHEME = "traces://"
MIN_SECONDS = 3.0
MAX_FRAMES = 5000                 # ~166 s at 30 fps; the recorder keeps 48 s
MIN_FINITE_FRACTION = 0.98        # same bar capture/ingest.py holds a video to

_REGISTRY: dict = {}
_LOCK = threading.Lock()
_PATCHED = False


# ------------------------------------------------------------- the document
def _rows(v, n: int) -> np.ndarray:
    out = np.full((n, 3), np.nan)
    if not isinstance(v, list):
        return out
    for i, row in enumerate(v[:n]):
        if isinstance(row, (list, tuple)) and len(row) == 3:
            try:
                r, g, b = float(row[0]), float(row[1]), float(row[2])
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(x) for x in (r, g, b)):
                out[i] = (r, g, b)
    return out


def ingest_traces(payload: dict, *, capture_profile: str = "consumer",
                  exposure_locked: Optional[bool] = None,
                  awb_locked: Optional[bool] = None,
                  assume_rig_locks: bool = False,
                  upload_id: str = "") -> IngestResult:
    """The client's trace document -> IngestResult, judged by the SAME capture
    gate a video is (fps floor, illuminance), with the codec term absent by
    construction (nothing was encoded). Never raises."""
    try:
        return _ingest(payload, capture_profile, exposure_locked, awb_locked,
                       assume_rig_locks, upload_id)
    except Exception as e:                                     # noqa: BLE001
        return IngestResult(ok=False, reasons=[f"trace document unusable: "
                                               f"{type(e).__name__}: {e}"],
                            capture_profile=capture_profile)


def _ingest(payload, profile, exposure_locked, awb_locked, rig, upload_id):
    if not isinstance(payload, dict):
        return IngestResult(ok=False, reasons=["trace document is not an object"],
                            capture_profile=profile)
    t = np.asarray(payload.get("t_s") or [], float)
    n = int(min(t.size, MAX_FRAMES))
    if n < 2:
        return IngestResult(ok=False, reasons=["trace document carries no frames"],
                            capture_profile=profile)
    t = t[:n]
    tr = payload.get("traces") if isinstance(payload.get("traces"), dict) else {}
    rois = {r: _rows(tr.get(r), n) for r in ROI_NAMES}
    found = np.ones(n, bool)
    for r in ROI_NAMES:
        found &= np.isfinite(rois[r]).all(axis=1)
    found &= np.isfinite(t)
    # monotone clock: a frame whose time does not advance is dropped, as the
    # video reader repairs the same fault
    keep = found.copy()
    last = -np.inf
    for i in range(n):
        if keep[i]:
            if t[i] <= last:
                keep[i] = False
            else:
                last = t[i]
    n_kept = int(keep.sum())
    fps_nom = float(payload.get("fps_nominal") or 30.0)
    width = int(payload.get("width") or 0)
    height = int(payload.get("height") or 0)

    # tracker summary: coverage and, when the client sent its face box, the
    # same jitter measure capture/face_tracking.py uses
    gap = longest = 0
    for f in found:
        gap = 0 if f else gap + 1
        longest = max(longest, gap)
    stability = 0.0
    bbox = payload.get("bbox")
    if n_kept >= 3:
        coverage = float(found.mean())
        jitter = 0.0
        if isinstance(bbox, list) and len(bbox) >= n:
            bb = np.array([[float(x) for x in b] if isinstance(b, (list, tuple)) and len(b) == 4
                           else [np.nan] * 4 for b in bbox[:n]], float)
            bb = bb[keep]
            bb = bb[np.isfinite(bb).all(axis=1)]
            if bb.shape[0] >= 3:
                cx, cy = bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2
                ax, ay = bb[:, 2] / 2, bb[:, 3] / 2
                scale = float(np.median(ax)) + 1e-9
                centre_jitter = float(np.median(np.hypot(np.diff(cx), np.diff(cy))))
                size_jitter = float(np.median(np.maximum(np.abs(np.diff(ax)), np.abs(np.diff(ay)))))
                jitter = max(centre_jitter / (0.05 * scale), size_jitter / (0.05 * scale))
        stability = float(np.clip(1.0 - jitter, 0.0, 1.0) * coverage)
    ts = t[keep]
    dt = np.diff(ts) if n_kept > 1 else np.array([])
    period = float(np.median(dt)) if dt.size else (1.0 / fps_nom)
    fps = 1.0 / period if period > 0 else fps_nom
    track = TrackSummary(n_frames=n, n_found=n_kept,
                         longest_gap_s=float(longest / fps) if fps else 0.0,
                         stability=stability, tracker="client-sdk-bbox")

    traces = {r: rois[r][keep] for r in ROI_NAMES}
    if n_kept < 2 or (ts[-1] - ts[0]) < MIN_SECONDS:
        return IngestResult(ok=False,
                            reasons=[f"only {n_kept} usable frames "
                                     f"({(ts[-1] - ts[0]) if n_kept > 1 else 0:.1f} s) in the "
                                     f"trace document (need {MIN_SECONDS:.0f} s)"],
                            track=track, capture_profile=profile)
    luma_all = np.mean([0.299 * traces[r][:, 0] + 0.587 * traces[r][:, 1] + 0.114 * traces[r][:, 2]
                        for r in ROI_NAMES], axis=0)
    face_luma = float(np.median(luma_all))
    face_lux = LUX_PROXY_AT_LUMA_160 * face_luma / 160.0
    dup = 0
    if n_kept > 1:
        same = np.ones(n_kept - 1, bool)
        for r in ROI_NAMES:
            same &= np.all(np.abs(np.diff(traces[r], axis=0)) < 1e-9, axis=1)
        dup = int(same.sum())
    meta = VideoMeta(path=f"{SCHEME}{upload_id or 'unknown'}", width=width, height=height,
                     n_frames=n, nominal_fps=fps_nom, measured_fps_mean=float(fps),
                     measured_fps_jitter_ms=float(np.std(dt) * 1000.0) if dt.size else None,
                     duration_s=float(ts[-1] - ts[0]), codec_fourcc="none",
                     file_bitrate_mbps=None, mean_luma=face_luma, lux_proxy=float(face_lux),
                     duplicate_frame_fraction=float(dup / max(n_kept - 1, 1)),
                     max_duplicate_run=0,
                     collapsed_interval_fraction=float(np.mean(dt < 0.5 * period)) if dt.size else 0.0)
    capture = build_capture_config(meta, illuminance_lux=face_lux, assume_rig_locks=rig,
                                   exposure_locked=exposure_locked, awb_locked=awb_locked,
                                   phone_model="client-live-frames", mount="handheld")
    ok, why, caveats = apply_capture_gate(capture, profile)
    caveats = list(caveats) + ["regions sampled on the device from the live camera frames "
                               "(no video codec in the path)"]
    if not ok:
        return IngestResult(ok=False, reasons=list(why), meta=meta, capture=capture,
                            track=track, capture_caveats=caveats, capture_profile=profile)
    uneven = np.zeros(n_kept, bool)
    per_roi = np.vstack([0.299 * traces[r][:, 0] + 0.587 * traces[r][:, 1] + 0.114 * traces[r][:, 2]
                         for r in ROI_NAMES])
    with np.errstate(invalid="ignore", divide="ignore"):
        uneven = (per_roi.max(axis=0) - per_roi.min(axis=0)) / np.maximum(per_roi.mean(axis=0), 1e-9) > 0.5
    clipped = float(np.mean(np.vstack([traces[r] for r in ROI_NAMES]).max(axis=1) >= 250.0))
    dark = float(np.mean(np.vstack([traces[r] for r in ROI_NAMES]).mean(axis=1) <= 10.0))
    photometric = {"n_observed_frames": n_kept,
                   "uneven_frame_fraction": float(uneven.mean()),
                   "median_usable_rois": float(len(ROI_NAMES)),
                   "max_clipped_fraction": clipped, "max_dark_fraction": dark}
    if photometric["uneven_frame_fraction"] > 0.25:
        caveats.append(f"uneven facial illumination in "
                       f"{photometric['uneven_frame_fraction']:.0%} of tracked frames")
    finite = min(float(np.isfinite(traces[r]).all(axis=1).mean()) for r in ROI_NAMES)
    if finite < MIN_FINITE_FRACTION:
        return IngestResult(ok=False, reasons=["trace document has non-finite region samples"],
                            meta=meta, capture=capture, track=track,
                            capture_caveats=caveats, capture_profile=profile)
    return IngestResult(ok=True, meta=meta, capture=capture, traces=traces,
                        timestamps_s=ts, track=track, capture_caveats=caveats,
                        capture_profile=profile, photometric=photometric)


# ------------------------------------------------------------- the registry
def _install():
    """Wrap inference.pipeline.ingest_video once: registered sentinel paths
    return their prebuilt result, everything else reaches the real reader."""
    global _PATCHED
    if _PATCHED:
        return
    import inference.pipeline as pl
    real = pl.ingest_video

    def dispatch(video_path, *args, **kwargs):
        with _LOCK:
            ing = _REGISTRY.pop(str(video_path), None)
        if ing is not None:
            return ing
        return real(video_path, *args, **kwargs)

    dispatch._trace_ingest_real = real                        # for tests
    pl.ingest_video = dispatch
    _PATCHED = True


def register(ing: IngestResult) -> str:
    """Park a prebuilt IngestResult and return the sentinel path to run the
    pipeline on. Consumed on first use."""
    _install()
    key = f"{SCHEME}{uuid.uuid4().hex}"
    with _LOCK:
        _REGISTRY[key] = ing
    return key


def run_on_traces(payload: dict, *, manifest: Optional[dict] = None, config=None,
                  upload_id: str = ""):
    """(ScanResult, det) for a trace document, through THE pipeline
    (run_with_details), or (None, {"ingest": IngestResult}) when the
    document did not pass the capture gate."""
    from inference.pipeline import run_with_details
    m = manifest or {}
    ing = ingest_traces(payload, capture_profile=m.get("capture_profile", "consumer"),
                        exposure_locked=m.get("exposure_locked"),
                        awb_locked=m.get("awb_locked"),
                        assume_rig_locks=bool(m.get("assume_rig_locks", False)),
                        upload_id=upload_id)
    key = register(ing)
    return run_with_details(key, manifest=m, config=config,
                            recording_id=f"traces-{upload_id or 'unknown'}")
