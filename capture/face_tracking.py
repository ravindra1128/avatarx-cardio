"""
Face tracking (T3, extended for the live demo): per-frame face localisation
as an upright oval, plus 5 landmarks when the backend provides them.

Backend chain, first success locks (and is recorded in provenance):

1. "mediapipe"          — FaceLandmarker, ONLY if a local model file exists
                          (mediapipe >= 1.0 removed the bundled-model
                          solutions API; models are never downloaded here).
                          Point AVATARX_MEDIAPIPE_MODEL at a
                          face_landmarker.task file to enable.
2. "yunet"              — OpenCV FaceDetectorYN with the vendored OpenCV-zoo
                          model (capture/models/, Apache-2.0, 232 KB): box +
                          5 landmarks (eyes, nose tip, mouth corners), ~5 ms
                          per 640x480 frame. THE real-face detector.
3. "opencv_haar"        — Haar cascade, only when the OpenCV build ships it
                          (OpenCV 5 headless does not).
4. "skin_segmentation"  — skin-chromaticity mask -> largest component ->
                          moment ellipse. The only backend that can follow
                          the synthetic ellipse faces; on real footage it is
                          a last resort and its use is visible in provenance.

All backends emit the SAME observation (upright oval + score, landmarks
optional), so ROI geometry downstream is tracker-agnostic.

Temporal smoothing: detector boxes jitter frame to frame by a few pixels,
which shows up in the ROI means as noise. The geometry handed to ROI
extraction is EMA-smoothed; the RAW observations are kept and the
stability metric is judged on them (smoothing must never flatter quality).

Fail-closed rule (v0.1 prompt): no face for more than MAX_FACE_GAP_S — or
never — aborts the scan with a reason; nothing is extrapolated past the gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os
import pathlib

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

MAX_FACE_GAP_S = 2.0
YUNET_MODEL = pathlib.Path(__file__).resolve().parent / "models" / \
    "face_detection_yunet_2023mar.onnx"


def yunet_available() -> bool:
    return cv2 is not None and hasattr(cv2, "FaceDetectorYN") and \
        YUNET_MODEL.exists()


@dataclass
class FaceObservation:
    found: bool
    cx: float = 0.0
    cy: float = 0.0
    ax: float = 0.0          # semi-axis, x
    ay: float = 0.0          # semi-axis, y
    score: float = 0.0
    # optional 5 landmarks (image px): eye_L, eye_R (viewer left/right),
    # nose tip, mouth_L, mouth_R
    landmarks: Optional[np.ndarray] = None
    box: Optional[tuple] = None                  # (x, y, w, h) detector box


@dataclass
class TrackSummary:
    n_frames: int
    n_found: int
    longest_gap_s: float
    stability: float          # 1 - normalised centre/size jitter (RAW obs)
    tracker: str


# ------------------------------------------------------------- backends
class _MediapipeBackend:
    name = "mediapipe"

    def __init__(self):
        model = os.environ.get("AVATARX_MEDIAPIPE_MODEL", "")
        if not model or not os.path.exists(model):
            raise RuntimeError("no local mediapipe face model")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model),
            num_faces=1)
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, frame_bgr: np.ndarray) -> FaceObservation:
        mp = self._mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect(img)
        if not res.face_landmarks:
            return FaceObservation(False)
        h, w = frame_bgr.shape[:2]
        pts = res.face_landmarks[0]
        xs = np.array([p.x for p in pts]) * w
        ys = np.array([p.y for p in pts]) * h
        # canonical mesh indices: 33/263 eye outer corners, 1 nose tip,
        # 61/291 mouth corners
        lm = None
        if len(pts) >= 468:
            lm = np.array([[xs[33], ys[33]], [xs[263], ys[263]],
                           [xs[1], ys[1]], [xs[61], ys[61]],
                           [xs[291], ys[291]]])
            if lm[0, 0] > lm[1, 0]:
                lm[[0, 1]] = lm[[1, 0]]
                lm[[3, 4]] = lm[[4, 3]]
        return FaceObservation(True, float(xs.mean()), float(ys.mean()),
                               float((xs.max() - xs.min()) / 2),
                               float((ys.max() - ys.min()) / 2), 1.0,
                               landmarks=lm,
                               box=(float(xs.min()), float(ys.min()),
                                    float(xs.max() - xs.min()),
                                    float(ys.max() - ys.min())))


class _YuNetBackend:
    name = "yunet"

    def __init__(self):
        if not yunet_available():
            raise RuntimeError("YuNet model unavailable")
        self._det = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 240),
                                              0.6, 0.3, 500)
        self._size = (320, 240)

    def detect(self, frame_bgr: np.ndarray) -> FaceObservation:
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._size:
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._det.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return FaceObservation(False)
        f = max(faces, key=lambda r: r[2] * r[3])
        x, y, bw, bh = [float(v) for v in f[:4]]
        lm = f[4:14].reshape(5, 2).astype(float)
        if lm[0, 0] > lm[1, 0]:                          # enforce L,R order
            lm[[0, 1]] = lm[[1, 0]]
            lm[[3, 4]] = lm[[4, 3]]
        return FaceObservation(True, x + bw / 2, y + bh / 2,
                               bw * 0.5, bh * 0.55, float(f[14]),
                               landmarks=lm, box=(x, y, bw, bh))


class _HaarBackend:
    name = "opencv_haar"

    def __init__(self):
        if cv2 is None or not hasattr(cv2, "CascadeClassifier") or \
                not hasattr(cv2, "data"):
            raise RuntimeError("opencv haar cascades unavailable")
        path = os.path.join(cv2.data.haarcascades,
                            "haarcascade_frontalface_default.xml")
        if not os.path.exists(path):
            raise RuntimeError("haar cascade file missing")
        self._cc = cv2.CascadeClassifier(path)
        if self._cc.empty():
            raise RuntimeError("haar cascade failed to load")

    def detect(self, frame_bgr: np.ndarray) -> FaceObservation:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cc.detectMultiScale(gray, 1.15, 4, minSize=(40, 40))
        if len(faces) == 0:
            return FaceObservation(False)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return FaceObservation(True, x + w / 2, y + h / 2,
                               w * 0.55, h * 0.70, 0.9,
                               box=(float(x), float(y), float(w), float(h)))


class _SkinBackend:
    name = "skin_segmentation"
    MIN_AREA_FRAC = 0.01

    def detect(self, frame_bgr: np.ndarray) -> FaceObservation:
        f = frame_bgr.astype(np.float32)
        b, g, r = f[..., 0], f[..., 1], f[..., 2]
        # Luminance-independent chroma rule.  The old absolute R>60 mask
        # failed systematically on darker skin and under lower exposure.
        # YCrCb carries skin chroma across a substantially wider luminance
        # range; the relative RGB term rejects grey backgrounds.  This is a
        # fallback locator only -- biomarker extraction itself never uses a
        # skin-colour classifier.
        if cv2 is not None:
            ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = [ycc[..., i].astype(np.float32) for i in range(3)]
            chroma = (y >= 22) & (cr >= 128) & (cr <= 185) & \
                     (cb >= 70) & (cb <= 140)
        else:
            chroma = np.ones(r.shape, dtype=bool)
        relative = (r > 35) & (r > 1.04 * g) & (g > 1.02 * b) & \
                   ((r - b) > 10)
        mask = (chroma & relative).astype(np.uint8)
        if cv2 is not None:
            k = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
            nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            if nlab <= 1:
                return FaceObservation(False)
            lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = labels == lab
        else:
            mask = mask.astype(bool)
        n_px = int(mask.sum())
        if n_px < self.MIN_AREA_FRAC * mask.size:
            return FaceObservation(False)
        ys, xs = np.nonzero(mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        ax = 2.0 * float(xs.std())
        ay = 2.0 * float(ys.std())
        if ax < 5 or ay < 5:
            return FaceObservation(False)
        return FaceObservation(True, cx, cy, ax, ay, 0.7,
                               box=(cx - ax, cy - ay, 2 * ax, 2 * ay))


class FaceTracker:
    """Tries backends in order until one finds a face, then locks to it.

    `process()` returns the SMOOTHED observation for ROI geometry; the raw
    per-frame observations are kept in `observations` for the stability
    metric.
    """

    def __init__(self, smooth_alpha: float = 0.35):
        self._backends = []
        for cls in (_MediapipeBackend, _YuNetBackend, _HaarBackend,
                    _SkinBackend):
            try:
                self._backends.append(cls())
            except Exception:
                continue
        self._locked: Optional[int] = None
        self.alpha = float(smooth_alpha)
        self.observations: list[FaceObservation] = []
        self.smoothed_observations: list[FaceObservation] = []
        self._state: Optional[FaceObservation] = None
        self._locked_misses = 0
        self._smooth_misses = 0

    @property
    def tracker(self) -> str:
        if self._locked is None:
            return "none"
        return self._backends[self._locked].name

    def _smooth(self, obs: FaceObservation) -> FaceObservation:
        if not obs.found:
            self._smooth_misses += 1
            # Do not blend a reacquired face with geometry from before an
            # occlusion/head turn.  A short one-frame detector flicker keeps
            # the filter warm; a real gap resets it.
            if self._smooth_misses >= 3:
                self._state = None
            return obs
        self._smooth_misses = 0
        if self._state is None or not self._state.found:
            self._state = obs
            return obs
        a = self.alpha
        s = self._state
        lm = obs.landmarks
        if lm is not None and s.landmarks is not None:
            lm = a * lm + (1 - a) * s.landmarks
        box = obs.box
        if box is not None and s.box is not None:
            box = tuple(a * np.asarray(box) + (1 - a) * np.asarray(s.box))
        out = FaceObservation(True,
                              a * obs.cx + (1 - a) * s.cx,
                              a * obs.cy + (1 - a) * s.cy,
                              a * obs.ax + (1 - a) * s.ax,
                              a * obs.ay + (1 - a) * s.ay,
                              obs.score, landmarks=lm, box=box)
        self._state = out
        return out

    def _inject_for_test(self, obs: FaceObservation) -> FaceObservation:
        self.observations.append(obs)
        sm = self._smooth(obs)
        self.smoothed_observations.append(sm)
        return sm

    def process(self, frame_bgr: np.ndarray) -> FaceObservation:
        if self._locked is not None:
            obs = self._backends[self._locked].detect(frame_bgr)
            if obs.found:
                self._locked_misses = 0
            else:
                self._locked_misses += 1
                # A backend can lose a partially turned, masked, or strongly
                # lit face while another local detector can still track it.
                # Re-open the chain after persistent misses rather than
                # remaining permanently locked to a failed detector.
                if self._locked_misses >= 3:
                    for i, be in enumerate(self._backends):
                        if i == self._locked:
                            continue
                        o = be.detect(frame_bgr)
                        if o.found:
                            self._locked = i
                            self._locked_misses = 0
                            obs = o
                            break
        else:
            obs = FaceObservation(False)
            for i, be in enumerate(self._backends):
                o = be.detect(frame_bgr)
                if o.found:
                    self._locked = i
                    self._locked_misses = 0
                    obs = o
                    break
        self.observations.append(obs)
        sm = self._smooth(obs)
        self.smoothed_observations.append(sm)
        return sm

    def summary(self, fps: float) -> TrackSummary:
        obs = self.observations
        found = np.array([o.found for o in obs], bool)
        n = len(obs)
        gap = longest = 0
        for f in found:
            gap = 0 if f else gap + 1
            longest = max(longest, gap)
        stability = 0.0
        if found.sum() >= 3:
            kept = [o for o, f in zip(obs, found) if f]
            cx = np.array([o.cx for o in kept])
            cy = np.array([o.cy for o in kept])
            ax = np.array([o.ax for o in kept])
            ay = np.array([o.ay for o in kept])
            scale = float(np.median(ax)) + 1e-9
            centre_jitter = float(np.median(np.hypot(np.diff(cx), np.diff(cy))))
            size_jitter = float(np.median(
                np.maximum(np.abs(np.diff(ax)), np.abs(np.diff(ay)))))
            jitter = max(centre_jitter / (0.05 * scale),
                         size_jitter / (0.05 * scale))
            # Detection coverage matters: a tracker that returns a perfectly
            # still box on half the frames is not stable.  Long gaps are
            # already preserved separately for the hard face-loss gate.
            coverage = float(found.mean()) if n else 0.0
            stability = float(np.clip(1.0 - jitter, 0.0, 1.0) * coverage)
        return TrackSummary(n_frames=n, n_found=int(found.sum()),
                            longest_gap_s=float(longest / fps) if fps else 0.0,
                            stability=stability, tracker=self.tracker)
