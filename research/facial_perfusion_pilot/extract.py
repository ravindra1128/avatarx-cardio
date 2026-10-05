"""DC-preserving ROI extraction for FPR.

Turns a recording into the raw spatially-averaged channel intensities the
perfusion index needs. Deliberately DIFFERENT from the production rPPG path:

  * It keeps the DC. It computes the plain spatial MEAN of each raw channel
    over a STATIC forehead ROI. It never divides by a temporal mean (POS) or
    band-limits before the mean, so PI = AC / DC is computable downstream.

  * The ROI is STATIC (owner brief section 8): a fixed pixel rectangle held
    constant for the whole clip, not a per-frame tracked box. Continuous
    tracking is intentionally avoided for the primary experiment -- a moving
    box injects motion into the DC. Place the ROI once (physical fiducial or
    the one-frame helper below), then hold it.

  * A 4x4 sub-region grid is extracted alongside, so a later analysis can
    tell a whole-instrument failure from a local dead zone from real signal.

All three channels are saved even though green is the analysis channel.
Reuses only `capture.video_reader` (decode + honest frame clock).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from capture.video_reader import iter_frames, probe_video
from . import FprError, ROI_MIN_PX

# ROI rectangle in (x0, y0, x1, y1) pixel coordinates.
Rect = tuple[int, int, int, int]


@dataclass
class ExtractResult:
    t: np.ndarray                                    # per-frame timestamp (s)
    green: np.ndarray                                # raw green ROI-mean
    red: np.ndarray
    blue: np.ndarray
    fps: float
    roi: Rect
    roi_grid: np.ndarray = field(                    # (n_frames, 4, 4) green
        default_factory=lambda: np.zeros((0, 4, 4)))
    n_frames: int = 0
    width: int = 0
    height: int = 0
    codec: str = ""
    notes: list = field(default_factory=list)

    def channel_means(self) -> dict:
        return {"green": float(np.mean(self.green)),
                "red": float(np.mean(self.red)),
                "blue": float(np.mean(self.blue))}


def _grid_means_green(patch_bgr: np.ndarray) -> np.ndarray:
    """4x4 grid of green means over an ROI patch (BGR)."""
    h, w = patch_bgr.shape[:2]
    out = np.full((4, 4), np.nan)
    ys = np.linspace(0, h, 5).astype(int)
    xs = np.linspace(0, w, 5).astype(int)
    g = patch_bgr[..., 1].astype(np.float64)
    for r in range(4):
        for c in range(4):
            sub = g[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            if sub.size:
                out[r, c] = float(np.mean(sub))
    return out


def extract_roi_trace(video_path: str, roi: Rect, *,
                      enforce_min_px: bool = True) -> ExtractResult:
    """Static-ROI raw channel means per frame, plus the 4x4 green grid.

    `roi` is (x0, y0, x1, y1) in pixels, HELD CONSTANT for every frame.
    """
    x0, y0, x1, y1 = (int(v) for v in roi)
    if x1 <= x0 or y1 <= y0:
        raise FprError(f"degenerate ROI {roi}")
    if enforce_min_px and (x1 - x0 < ROI_MIN_PX or y1 - y0 < ROI_MIN_PX):
        raise FprError(f"ROI {x1 - x0}x{y1 - y0}px is below the "
                       f"{ROI_MIN_PX}x{ROI_MIN_PX} floor (report section 7)")
    meta = probe_video(video_path)
    ts: list[float] = []
    g_list: list[float] = []
    r_list: list[float] = []
    b_list: list[float] = []
    grid_list: list[np.ndarray] = []
    for t, frame in iter_frames(video_path):
        h, w = frame.shape[:2]
        cx1, cy1 = min(x1, w), min(y1, h)
        if x0 >= cx1 or y0 >= cy1:
            raise FprError("ROI falls outside the frame; check the ROI "
                           "definition against the recording resolution")
        patch = frame[y0:cy1, x0:cx1, :].astype(np.float64)
        ts.append(t)
        b_list.append(float(np.mean(patch[..., 0])))
        g_list.append(float(np.mean(patch[..., 1])))
        r_list.append(float(np.mean(patch[..., 2])))
        grid_list.append(_grid_means_green(patch))
    if len(ts) < 2:
        raise FprError("fewer than 2 frames extracted")
    return ExtractResult(
        t=np.asarray(ts), green=np.asarray(g_list), red=np.asarray(r_list),
        blue=np.asarray(b_list), fps=meta.measured_fps_mean,
        roi=(x0, y0, x1, y1), roi_grid=np.asarray(grid_list),
        n_frames=len(ts), width=meta.width, height=meta.height,
        codec=meta.codec_fourcc)


def suggest_forehead_roi(video_path: str, *, hold_px: int = ROI_MIN_PX
                         ) -> Optional[Rect]:
    """Best-effort STATIC forehead rectangle from the first well-tracked
    frame, for convenience only. The primary experiment should use a fixed
    ROI from a physical fiducial (owner brief section 8), not this. Returns
    None if a face is not found. Face tracking is imported lazily so the
    core extraction path carries no model dependency.
    """
    try:
        from capture.face_tracking import FaceTracker
        from preprocessing.roi import roi_bounds_px
    except Exception as e:                           # noqa: BLE001
        raise FprError(f"forehead suggestion needs face tracking: {e}")
    tracker = FaceTracker()
    for _, frame in iter_frames(video_path):
        obs = tracker.process(frame)
        if not getattr(obs, "found", False):
            continue
        bounds = roi_bounds_px(obs, frame.shape)
        fx0, fx1, fy0, fy1 = bounds["forehead"]
        cx = (fx0 + fx1) // 2
        cy = (fy0 + fy1) // 2
        half = max(hold_px, (fx1 - fx0)) // 2
        h, w = frame.shape[:2]
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, x0 + 2 * half)
        y1 = min(h, y0 + 2 * half)
        return (int(x0), int(y0), int(x1), int(y1))
    return None
