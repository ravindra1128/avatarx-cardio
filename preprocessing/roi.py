"""
ROI geometry + per-ROI mean RGB traces (T3).

Four regions — forehead, left cheek, right cheek, nose — defined relative
to the tracked face oval, so the geometry is tracker-agnostic. ROIs stay
SEPARATE all the way to beat detection: fusion happens at the BEAT level in
beats/detector.py, never by averaging pixels across regions (averaging
would let one artifact-contaminated region silently poison the others, and
would destroy the cross-ROI agreement signal the fusion stage relies on).
"""
from __future__ import annotations

import numpy as np

from capture.face_tracking import FaceObservation

ROI_NAMES = ("forehead", "cheek_l", "cheek_r", "nose")

# (x0, x1, y0, y1) in face-oval units: x in semi-axis-ax units from centre,
# y in semi-axis-ay units (negative = up). Chosen to stay inside the oval.
_ROI_RECTS = {
    "forehead": (-0.50, 0.50, -0.75, -0.45),
    "cheek_l":  (-0.62, -0.18, 0.05, 0.45),
    "cheek_r":  (0.18, 0.62, 0.05, 0.45),
    "nose":     (-0.14, 0.14, -0.15, 0.25),
}


def _usable_landmarks(obs: FaceObservation) -> bool:
    """Whether five-point geometry is safe enough to drive ROI placement.

    Detectors occasionally return finite-looking but collapsed landmarks while
    a face is partly outside the frame or covered.  Letting that geometry reach
    ``int(round(...))`` either crashes the scan or produces tiny, wandering
    ROIs.  The oval fallback is less precise but stable and auditable.
    """
    lm = getattr(obs, "landmarks", None)
    if lm is None:
        return False
    a = np.asarray(lm, float)
    if a.shape != (5, 2) or not np.all(np.isfinite(a)):
        return False
    eye_span = float(np.hypot(*(a[1] - a[0])))
    face_scale = max(float(obs.ax), float(obs.ay), 1.0)
    return bool(eye_span >= max(4.0, 0.12 * face_scale) and
                a[0, 0] < a[1, 0] and
                float(np.mean(a[:2, 1])) < float(np.mean(a[3:, 1])))


def _landmark_rects(lm: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    """ROI rectangles anchored on 5 landmarks (eye_L, eye_R, nose, mouth_L,
    mouth_R), in units of the inter-ocular distance d and the eye-to-mouth
    distance m. Validated visually on a real portrait: forehead band above
    the brows, cheeks lateral to the nose between eye and mouth level."""
    eye_l, eye_r, nose, m_l, m_r = lm
    eye_mid = (eye_l + eye_r) / 2.0
    d = float(np.hypot(*(eye_r - eye_l))) + 1e-6
    mouth_y = float((m_l[1] + m_r[1]) / 2.0)
    m = max(mouth_y - float(eye_mid[1]), 0.5 * d)
    fx, fy = float(eye_mid[0]), float(eye_mid[1])
    nx, ny = float(nose[0]), float(nose[1])
    return {
        "forehead": (fx - 0.65 * d, fx + 0.65 * d, fy - 0.85 * d, fy - 0.40 * d),
        "cheek_l":  (nx - 0.95 * d, nx - 0.35 * d, fy + 0.30 * m, mouth_y - 0.05 * m),
        "cheek_r":  (nx + 0.35 * d, nx + 0.95 * d, fy + 0.30 * m, mouth_y - 0.05 * m),
        "nose":     (nx - 0.20 * d, nx + 0.20 * d, ny - 0.35 * m, ny + 0.10 * m),
    }


def roi_bounds_px(obs: FaceObservation, shape: tuple[int, int]
                  ) -> dict[str, tuple[int, int, int, int]]:
    """Pixel rectangles (x0, x1, y0, y1) for each ROI, clipped to frame.

    Landmark geometry when the tracker provides landmarks (real faces via
    YuNet/mediapipe); oval geometry otherwise (skin-segmentation fallback,
    synthetic faces)."""
    h, w = shape[:2]
    out = {}
    if _usable_landmarks(obs):
        rects = _landmark_rects(np.asarray(obs.landmarks, float))
        for name, (x0, x1, y0, y1) in rects.items():
            out[name] = (max(0, int(round(x0))), min(w, int(round(x1))),
                         max(0, int(round(y0))), min(h, int(round(y1))))
        return out
    for name, (u0, u1, v0, v1) in _ROI_RECTS.items():
        x0 = int(round(obs.cx + u0 * obs.ax))
        x1 = int(round(obs.cx + u1 * obs.ax))
        y0 = int(round(obs.cy + v0 * obs.ay))
        y1 = int(round(obs.cy + v1 * obs.ay))
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        out[name] = (x0, x1, y0, y1)
    return out


def roi_integrity(obs: FaceObservation, shape: tuple[int, int]
                  ) -> dict[str, float]:
    """Fraction of each ROI's NOMINAL area that lies inside the frame.

    `roi_bounds_px` silently clamps to the frame; a clamped ROI averages
    whatever pixels remain, which stops being the intended region. This
    is the honest framing criterion (v0.1.4.2): a close face is fine as
    long as every ROI keeps its pixels — the arbitrary face-width cap it
    replaces had no post-scan counterpart and blocked a real user whose
    tracking was 0.98 with beats present."""
    h, w = shape[:2]
    if _usable_landmarks(obs):
        rects = _landmark_rects(np.asarray(obs.landmarks, float))
    else:
        rects = {name: (obs.cx + u0 * obs.ax, obs.cx + u1 * obs.ax,
                        obs.cy + v0 * obs.ay, obs.cy + v1 * obs.ay)
                 for name, (u0, u1, v0, v1) in _ROI_RECTS.items()}
    out = {}
    for name, (x0, x1, y0, y1) in rects.items():
        nominal = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
        cw = max(min(w, x1) - max(0, x0), 0.0)
        ch = max(min(h, y1) - max(0, y0), 0.0)
        out[name] = float(cw * ch / nominal) if nominal > 0 else 0.0
    return out


def mean_rgb(frame_bgr: np.ndarray, obs: FaceObservation) -> dict[str, np.ndarray]:
    """Per-ROI robust spatial mean as RGB (OpenCV BGR is channel-flipped).

    Returns NaN triplets for degenerate (empty) regions — downstream must
    treat NaN as missing, never as zero signal.  The darkest 10% and brightest
    5% of pixels by *within-ROI* luminance are excluded.  That suppresses
    glasses rims, facial hair, hard shadows and specular highlights without a
    skin-colour threshold, so it does not encode a light-skin prior.  Uniform
    patches (including synthetic regression fixtures) fall back to all pixels.
    """
    out = {}
    for name, (x0, x1, y0, y1) in roi_bounds_px(obs, frame_bgr.shape).items():
        if x1 - x0 < 2 or y1 - y0 < 2:
            out[name] = np.full(3, np.nan)
            continue
        patch = frame_bgr[y0:y1, x0:x1, :].astype(float)
        pix = patch.reshape(-1, 3)
        lum = 0.114 * pix[:, 0] + 0.587 * pix[:, 1] + 0.299 * pix[:, 2]
        q10, q95 = np.quantile(lum, (0.10, 0.95))
        keep = (lum >= q10) & (lum <= q95)
        # Quantised/uniform phone frames can put every pixel on one boundary;
        # never turn that benign case into an empty ROI.
        if int(keep.sum()) < max(16, int(0.25 * pix.shape[0])):
            keep = np.ones(pix.shape[0], dtype=bool)
        out[name] = pix[keep].mean(axis=0)[::-1]          # BGR -> RGB
    return out


def roi_photometry(frame_bgr: np.ndarray, obs: FaceObservation) -> dict:
    """Measured per-ROI exposure/framing diagnostics for gating and logs.

    Metrics are colour-agnostic luminance statistics.  They intentionally do
    not attempt to infer skin tone.  A region is ``usable`` only when it is in
    frame, large enough, and not predominantly crushed or clipped.
    """
    integrity = roi_integrity(obs, frame_bgr.shape)
    per_roi = {}
    means = []
    for name, (x0, x1, y0, y1) in roi_bounds_px(obs, frame_bgr.shape).items():
        if x1 - x0 < 2 or y1 - y0 < 2:
            per_roi[name] = {"pixels": 0, "integrity": integrity[name],
                             "mean_luma": None, "dark_fraction": 1.0,
                             "clipped_fraction": 1.0, "usable": False}
            continue
        p = frame_bgr[y0:y1, x0:x1, :].astype(np.float32)
        lum = 0.114 * p[..., 0] + 0.587 * p[..., 1] + 0.299 * p[..., 2]
        mean = float(np.mean(lum))
        dark = float(np.mean(lum <= 20.0))
        clipped = float(np.mean(lum >= 250.0))
        usable = bool(integrity[name] >= 0.80 and lum.size >= 64 and
                      dark < 0.60 and clipped < 0.20)
        per_roi[name] = {
            "pixels": int(lum.size), "integrity": float(integrity[name]),
            "mean_luma": mean, "dark_fraction": dark,
            "clipped_fraction": clipped, "usable": usable,
        }
        if usable:
            means.append(mean)
    if means:
        med = float(np.median(means))
        imbalance = float((max(means) - min(means)) / max(med, 1.0))
    else:
        imbalance = float("inf")
    return {
        "per_roi": per_roi,
        "usable_rois": int(sum(bool(v["usable"]) for v in per_roi.values())),
        "max_clipped_fraction": float(max(
            (v["clipped_fraction"] for v in per_roi.values()), default=1.0)),
        "max_dark_fraction": float(max(
            (v["dark_fraction"] for v in per_roi.values()), default=1.0)),
        "luma_imbalance": imbalance,
    }
