"""
Live scan engine (v0.1.4): the state machine behind the consumer demo.

    PREVIEW ──READY held──► SCANNING ──good time reaches N s──► PROCESSING ──► DONE
       ▲   (readiness gate)      │ quality drops: PAUSE (timer & recording stop)   └► FAILED
       └───── restart ◄──────────┘ pause budget exceeded

THE PRE-SCAN EVIDENCE IS THE POST-SCAN EVIDENCE — the parity rule. Readiness is
computed by the SAME function the production pipeline uses
(`inference/evidence.py`) on a rolling window of the live ROI traces,
against the ANY-RESULT thresholds the decision itself applies (coverage,
coherence, beat timing, SQI — read from the same config keys), plus the
framing / face-region-lighting / exposure-step / motion / tracking checks
only the live view can make. In default advisory mode the countdown starts
once the basic face, framing and camera-delivery checks allow an attempt;
downstream pulse evidence continues maturing during the scan and governs
pause/recovery plus the final result. Blocking mode retains the stricter
all-checks hold. A check changes state only after `debounce_evals`
consecutive evaluations (single-eval flicker is forgiven), and every
evaluation is recorded in `readiness_log.jsonl` for debugging.

DURING THE SCAN the same monitoring continues every 0.5 s. The timer pauses
for capture faults that make frames unusable (face/framing, lighting,
exposure, motion, delivery, frame rate, timestamps, tracking). Pulse SNR,
coherence and beat evidence remain live guidance but do not deadlock the
recording: the complete scan is processed and the final quality gates decide
whether a result is supportable. A pause is an honest hole in the capture
clock, which the gap-aware pipeline treats as a segment boundary. If pauses exceed
`max_paused_s` or `max_pauses`, the session RESTARTS at readiness rather
than delivering a doomed recording.

Frames arrive as BGR arrays with the capture clock's timestamps. Recording
is lossless FFV1 with the timestamps in a sidecar. On finish the file goes
to `inference.pipeline.run_with_details` — the results report is built from
that ScanResult and nothing else. Every user-facing rhythm sentence is
`ScanResult.user_facing_text()` verbatim.
"""
from __future__ import annotations

import enum
import json
import pathlib
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:                                   # keep importable
    cv2 = None

from configs import load_config
from capture.face_tracking import FaceTracker, FaceObservation, MAX_FACE_GAP_S
from capture.video_reader import LUX_PROXY_AT_LUMA_160
from preprocessing.roi import ROI_NAMES, roi_bounds_px, mean_rgb, \
    roi_integrity, roi_photometry
from inference.evidence import window_evidence, readiness_from_evidence, \
    _DEFAULT_READINESS
from inference.confidence_stars import confidence_stars

DEFAULT_SCAN_SECONDS = 30.0
EVIDENCE_INTERVAL_S = 0.5         # readiness/quality recompute cadence
FACE_WIDTH_MIN = 0.20             # framing: face box width / frame width
                                  # (pixel floor: ~1200 px per ROI at 640 px)
ROI_INTEGRITY_MIN = 0.5           # framing veto: an ROI keeping less than
                                  # half its nominal pixels is not the
                                  # intended region (v0.1.4.2 — replaces the
                                  # arbitrary 0.70 face-width cap, which had
                                  # no post-scan counterpart and blocked a
                                  # real close-to-laptop user)
CENTRE_TOL = 0.22                 # framing: |centre offset| / frame size
FACE_LUMA_MIN = 60.0              # lighting: mean luma of the face region
MOTION_WARN = 0.035               # median per-frame centre move / face width
MOTION_ABORT = 0.09
ABORT_AFTER_S = 2.0               # sustained hard fault before a live abort
EXPOSURE_STEP = 0.12              # relative frame-luma step between 0.5 s bins
LIGHTING_IMBALANCE_MAX = 0.85     # max ROI luma range / median usable ROI
MAX_TIMESTAMP_ERROR_STREAK = 3
MAX_DUPLICATE_RUN = 2

READINESS_CHECKS = ("face", "framing", "lighting", "exposure", "motion",
                    "frame_delivery", "frame_rate", "timestamps", "tracking", "signal_snr",
                    "cross_roi_coherence", "beat_timing", "prelim_beats", "sqi")


class SessionState(str, enum.Enum):
    PREVIEW = "preview"
    SCANNING = "scanning"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


def _luma(frame_bgr: np.ndarray) -> float:
    f = frame_bgr.astype(np.float32)
    return float(np.mean(0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]))


class ScanSession:
    """One user's scan. Thread-safe for one producer + status readers."""

    def __init__(self, session_id: str, work_dir: str, *,
                 fps_hint: float = 30.0,
                 scan_seconds: float = DEFAULT_SCAN_SECONDS,
                 config: Optional[dict] = None):
        self.id = session_id
        self.work_dir = pathlib.Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.fps_hint = float(fps_hint)
        self.scan_seconds = float(scan_seconds)
        self.cfg = config or load_config()
        self.rc = {**_DEFAULT_READINESS,
                   **(self.cfg["decision"].get("readiness") or {})}
        # 'advisory' allows an API attempt after physical blocking checks,
        # while can_record still requires usable evidence; 'blocking' also
        # requires every check + hold before the attempt starts.
        self._advisory = str(self.rc.get("mode", "advisory")) == "advisory"
        self.state = SessionState.PREVIEW
        self.error: Optional[dict] = None
        self.result: Optional[dict] = None
        self.feedback: dict = {
            "face_found": False, "framing": "no_face", "lighting": "unknown",
            "motion": "ok", "signal_quality": None, "coherence": None,
            "coherence_ok": False, "ready": False, "progress": 0.0,
            "tracker": "none", "roi_boxes": None, "face_box": None,
            "readiness": {"ready": False, "checks": {}, "failing": [],
                          "hint": "Starting camera…", "hold_s": 0.0},
            "paused": False, "good_seconds": 0.0, "paused_seconds": 0.0,
            "restarted": False, "restart_reason": None,
            "disposition": "REPEAT_SCAN", "timestamp_errors": 0,
            "duplicate_frames": 0, "frame_delivery": "ok",
        }
        self.camera_settings: dict = {}
        self.client_dropped = 0
        self._lock = threading.Lock()
        self._tracker = FaceTracker()
        self._last_obs: Optional[FaceObservation] = None
        self._face_gap = 0.0
        self._motion_bad_s = 0.0
        self._dark_s = 0.0
        self._prev_t: Optional[float] = None
        self._prev_frame_sample: Optional[np.ndarray] = None
        self._timestamp_error_streak = 0
        self._duplicate_run = 0
        self._duplicate_frames = 0
        self._timestamp_errors = 0
        self._last_ev_t = -1e9
        self._ready_since: Optional[float] = None
        self._fail_streaks: dict = {}                 # check -> consecutive fails
        self._log_f = None                            # readiness_log.jsonl
        # rolling buffers (window_s) for evidence
        self._trace_t: deque = deque()
        self._trace: dict = {r: deque() for r in ROI_NAMES}
        self._lumas: deque = deque()                  # (t, frame luma)
        self._centres: deque = deque(maxlen=60)
        # scan recording + good-time accounting
        self._writer = None
        self.video_path: Optional[str] = None
        self._scan_ts: list[float] = []
        self._frame_luma: list[tuple[float, float]] = []
        self._size: Optional[tuple] = None
        self._good_s = 0.0
        self._paused = False
        self._paused_s = 0.0
        self._n_pauses = 0
        self._pause_t0: Optional[float] = None
        self._resume_ok_since: Optional[float] = None
        self._proc_thread: Optional[threading.Thread] = None
        self.created_at = time.time()

    # ------------------------------------------------------------ status
    def note_client_dropped(self, n: int) -> None:
        with self._lock:
            self.client_dropped = max(int(self.client_dropped), int(n))

    def status(self) -> dict:
        with self._lock:
            return {"id": self.id, "state": self.state.value,
                    "disposition": self.feedback.get("disposition"),
                    "feedback": dict(self.feedback),
                    "error": self.error, "result": self.result,
                    "scan_seconds": self.scan_seconds}

    # ------------------------------------------------------------ frames
    def push_frames(self, frames_bgr: list, timestamps_s: list) -> dict:
        with self._lock:
            if self.state not in (SessionState.PREVIEW, SessionState.SCANNING):
                return dict(self.feedback)
            if len(frames_bgr) != len(timestamps_s):
                raise ValueError("frame batch and timestamp batch lengths differ")
            self.feedback["restarted"] = False
            for f, t in zip(frames_bgr, timestamps_s):
                a = np.asarray(f)
                if a.ndim != 3 or a.shape[2] != 3 or a.size == 0:
                    raise ValueError("camera frame must be a non-empty HxWx3 array")
                self._one(a, float(t))
                if self.state is SessionState.FAILED:
                    break
            return dict(self.feedback)

    # ---------------------------------------------------- per-frame path
    def _one(self, frame: np.ndarray, t: float) -> None:
        # Keep delivery failures out of the physiological trace.  Isolated
        # duplicates/bad timestamps are skipped and recover automatically;
        # persistent timestamp failure restarts the attempt visibly.
        if not np.isfinite(t) or (self._prev_t is not None and t <= self._prev_t):
            self._timestamp_errors += 1
            self._timestamp_error_streak += 1
            self.feedback.update({"timestamp_errors": self._timestamp_errors,
                                  "frame_delivery": "invalid_timestamps",
                                  "ready": False, "disposition": "REPEAT_SCAN"})
            self.feedback["hint"] = "Camera timestamps are invalid — restart the camera"
            if self.state is SessionState.SCANNING:
                if self._timestamp_error_streak >= MAX_TIMESTAMP_ERROR_STREAK:
                    self._pause_delivery_frame(
                        self._timestamp_error_streak /
                        max(self.fps_hint, 1.0))
                    self._restart("Camera timing became invalid — restart the camera and try again.")
            return
        self._timestamp_error_streak = 0

        sample = np.ascontiguousarray(frame[::8, ::8])
        duplicate = (self._prev_frame_sample is not None and
                     sample.shape == self._prev_frame_sample.shape and
                     np.array_equal(sample, self._prev_frame_sample))
        self._prev_frame_sample = sample.copy()
        if duplicate:
            dt_dup = (t - self._prev_t) if self._prev_t is not None \
                else 1.0 / self.fps_hint
            self._prev_t = t
            self._duplicate_run += 1
            self._duplicate_frames += 1
            self.feedback.update({"duplicate_frames": self._duplicate_frames,
                                  "frame_delivery": "duplicated_frames",
                                  "ready": False, "disposition": "REPEAT_SCAN"})
            self.feedback["hint"] = "Camera frames are repeating — close other camera apps"
            if self.state is SessionState.SCANNING and \
                    self._duplicate_run > MAX_DUPLICATE_RUN:
                self._pause_delivery_frame(float(np.clip(dt_dup, 1e-3, 0.5)))
            return
        self._duplicate_run = 0
        self.feedback["frame_delivery"] = "ok"

        h, w = frame.shape[:2]
        if self._size is None:
            self._size = (w, h)
        dt = (t - self._prev_t) if self._prev_t is not None else 1.0 / self.fps_hint
        dt = float(np.clip(dt, 1e-3, 0.5))
        self._prev_t = t
        fb = self.feedback

        obs = self._tracker.process(frame)
        fb["tracker"] = self._tracker.tracker
        frame_luma = _luma(frame)

        # ---- face / framing
        if obs.found:
            self._face_gap = 0.0
            self._last_obs = obs
            box = obs.box or (obs.cx - obs.ax, obs.cy - obs.ay, 2 * obs.ax, 2 * obs.ay)
            bx, by, bw, bh = box
            fw = bw / w
            offx = (bx + bw / 2 - w / 2) / w
            offy = (by + bh / 2 - h / 2) / h
            integ = roi_integrity(obs, frame.shape)
            if fw < FACE_WIDTH_MIN:
                framing = "too_far"
            elif min(integ.values()) < ROI_INTEGRITY_MIN:
                # ROI geometry actually broken: pick the actionable hint —
                # a big face pushed a region out (move back) vs. a face at
                # the frame edge (re-centre)
                framing = "too_close" if fw > 0.55 else "off_centre"
            elif abs(offx) > CENTRE_TOL or abs(offy) > CENTRE_TOL:
                framing = "off_centre"
            else:
                framing = "ok"
            fb["face_found"] = True
            fb["framing"] = framing
            fb["roi_integrity"] = {k: round(v, 2) for k, v in integ.items()}
            fb["face_box"] = [bx / w, by / h, bw / w, bh / h]
            rb = roi_bounds_px(obs, frame.shape)
            fb["roi_boxes"] = {k: [v[0] / w, v[2] / h, (v[1] - v[0]) / w,
                                   (v[3] - v[2]) / h] for k, v in rb.items()}
            x0, y0 = int(max(0, bx)), int(max(0, by))
            x1, y1 = int(min(w, bx + bw)), int(min(h, by + bh))
            face_luma = _luma(frame[y0:y1, x0:x1]) if x1 > x0 and y1 > y0 else 0.0
            photo = roi_photometry(frame, obs)
            fb["photometric"] = {
                "usable_rois": photo["usable_rois"],
                "max_clipped_fraction": round(photo["max_clipped_fraction"], 3),
                "max_dark_fraction": round(photo["max_dark_fraction"], 3),
                "luma_imbalance": (round(photo["luma_imbalance"], 3)
                                   if np.isfinite(photo["luma_imbalance"])
                                   else None),
            }
            self._centres.append((t, obs.cx, obs.cy, bw))
        else:
            self._face_gap += dt
            fb["face_found"] = False
            fb["framing"] = "no_face"
            fb["face_box"] = None
            fb["roi_boxes"] = None
            face_luma = frame_luma
            photo = None
            self._centres.clear()

        # ---- lighting + exposure stability
        # v0.1.4: lux proxy from the FACE region when a face is present —
        # the photons that matter fall on the skin; a well-lit face in a
        # dark room is analysable (whole-frame mean punished the
        # background, not the subject). Mirrors ingest's deferred face-
        # region lux gate, so readiness and post-scan judge the same
        # quantity.
        lux_proxy = LUX_PROXY_AT_LUMA_160 * face_luma / 160.0
        dark = face_luma < FACE_LUMA_MIN or lux_proxy < float(
            self.cfg["capture"]["lux_floor"])
        overexposed = bool(obs.found and photo is not None and
                           (face_luma >= 245.0 or
                            sum(v["clipped_fraction"] >= 0.20 for v in
                                photo["per_roi"].values()) >= 3))
        uneven = bool(obs.found and photo is not None and
                      photo["usable_rois"] >= 2 and
                      photo["luma_imbalance"] > LIGHTING_IMBALANCE_MAX)
        lighting = ("dark" if dark else "overexposed" if overexposed
                    else "uneven" if uneven else "ok")
        fb["lighting"] = lighting
        fb["lux_proxy"] = round(lux_proxy, 1)
        self._dark_s = self._dark_s + dt if dark else 0.0
        self._lumas.append((t, frame_luma))
        while self._lumas and t - self._lumas[0][0] > self.rc["window_s"]:
            self._lumas.popleft()
        exposure_ok = not self._exposure_stepped_window()

        # ---- motion
        motion = "ok"
        if len(self._centres) >= 6:
            c = np.array([(x, y, bw_) for _, x, y, bw_ in self._centres])
            recent = c[-int(min(len(c), max(6, self.fps_hint))):]
            move = np.hypot(np.diff(recent[:, 0]), np.diff(recent[:, 1]))
            rel = float(np.median(move)) / (float(np.median(recent[:, 2])) + 1e-6)
            fb["motion_rel"] = round(rel, 4)
            if rel > MOTION_ABORT:
                motion = "moving"; self._motion_bad_s += dt
            elif rel > MOTION_WARN:
                motion = "moving"; self._motion_bad_s += dt * 0.5
            else:
                self._motion_bad_s = max(0.0, self._motion_bad_s - dt)
        fb["motion"] = motion

        # ---- rolling ROI traces (the evidence window)
        if obs.found and fb["framing"] == "ok" and photo is not None and \
                photo["usable_rois"] >= 2:
            vals = mean_rgb(frame, obs)
            self._trace_t.append(t)
            for r in ROI_NAMES:
                self._trace[r].append(vals[r])
            while self._trace_t and t - self._trace_t[0] > self.rc["window_s"]:
                self._trace_t.popleft()
                for r in ROI_NAMES:
                    self._trace[r].popleft()
        elif not obs.found and self._face_gap > MAX_FACE_GAP_S:
            self._trace_t.clear()
            for r in ROI_NAMES:
                self._trace[r].clear()

        # ---- readiness / quality (shared evidence code), every 0.5 s
        if (t - self._last_ev_t) >= EVIDENCE_INTERVAL_S:
            self._last_ev_t = t
            self._update_readiness(obs.found, fb["framing"], lighting == "ok",
                                   exposure_ok, motion == "ok",
                                   lighting_status=lighting,
                                   delivery_ok=(fb["frame_delivery"] == "ok"))
        rd = fb["readiness"]
        # Start permission and record permission are intentionally separate:
        # even an advisory-start attempt counts time only on usable evidence.
        good = bool(rd.get("can_record", rd.get("can_start"))) if self._advisory \
            else bool(rd["ready"])

        # ---- start availability (preview)
        if self.state is SessionState.PREVIEW:
            if self._advisory:
                # API start permission is available on the blocking checks;
                # disposition stays REPEAT_SCAN until all evidence is ready.
                rd["hold_s"] = 0.0
                fb["ready"] = bool(rd.get("can_start"))
                fb["disposition"] = "READY" if rd.get("ready") else "REPEAT_SCAN"
                fb["progress"] = 0.0
                return
            if good:
                if self._ready_since is None:
                    self._ready_since = t
                held = t - self._ready_since
            else:
                self._ready_since = None
                held = 0.0
            rd["hold_s"] = round(held, 2)
            fb["ready"] = bool(good and held >= float(self.rc["hold_s"]))
            fb["disposition"] = "READY" if fb["ready"] else "REPEAT_SCAN"
            if good and not fb["ready"]:
                rd["hint"] = (f"Pulse signal stabilizing… "
                              f"{held:.0f}/{self.rc['hold_s']:.0f} s")
            fb["progress"] = 0.0
            return

        # ---- scanning: good-time timer, pause/resume/restart, hard aborts
        if self.state is SessionState.SCANNING:
            hard = None
            if not self._advisory:
                # Blocking mode preserves the stricter immediate-abort path;
                # advisory mode pauses/restarts these recoverable conditions.
                if self._dark_s > ABORT_AFTER_S:
                    hard = ("lighting",
                            "The scene became too dark during the scan.")
                elif self._motion_bad_s > ABORT_AFTER_S * 2:
                    hard = ("movement", "Too much movement during the scan.")
            if not good:
                if not self._paused:
                    self._paused = True
                    self._n_pauses += 1
                    self._pause_t0 = t
                    self._resume_ok_since = None
                self._paused_s += dt
            else:
                if self._paused:
                    if self._resume_ok_since is None:
                        self._resume_ok_since = t
                    if t - self._resume_ok_since >= float(self.rc["resume_hold_s"]):
                        self._paused = False
                        self._pause_t0 = None
                    else:
                        self._paused_s += dt
                if not self._paused:
                    self._record(frame, t, frame_luma)
                    self._good_s += dt
            fb["paused"] = self._paused
            fb["disposition"] = "REPEAT_SCAN" if self._paused else "READY"
            fb["good_seconds"] = round(self._good_s, 2)
            fb["paused_seconds"] = round(self._paused_s, 2)
            fb["progress"] = float(np.clip(self._good_s / self.scan_seconds, 0.0, 1.0))
            fb["elapsed_s"] = round(self._good_s, 2)
            if hard is not None:
                self._fail(*hard)
                return
            if self._paused_s > float(self.rc["max_paused_s"]) or \
                    self._n_pauses > int(self.rc["max_pauses"]):
                self._restart("Signal quality dropped for too long — "
                              "let's re-check and start again.")

    def _exposure_stepped_window(self) -> bool:
        """AE STEP detector (v0.1.4): a > EXPOSURE_STEP jump between
        ADJACENT 0.5 s luma bins within the last 2 s — an exposure
        transient that corrupts the waveform right now. Slow drift is a
        < 0.1 Hz trend the 0.7 Hz high-pass removes; the old first-half/
        second-half test failed 8 s of readiness on a 1.5%/s AE ramp."""
        if len(self._lumas) < int(1.0 * self.fps_hint):
            return False
        arr = np.asarray(self._lumas)
        t1 = float(arr[-1, 0])
        recent = arr[arr[:, 0] >= t1 - 2.0]
        if recent.shape[0] < int(0.8 * self.fps_hint):
            return False
        bins = []
        for k in range(4):
            m = (recent[:, 0] >= t1 - 2.0 + 0.5 * k) & \
                (recent[:, 0] < t1 - 2.0 + 0.5 * (k + 1))
            if np.any(m):
                bins.append(float(np.mean(recent[m, 1])))
        if len(bins) < 2:
            return False
        b = np.asarray(bins)
        rel = np.abs(np.diff(b)) / (b[:-1] + 1e-6)
        return bool(np.max(rel) > EXPOSURE_STEP)

    # engine-observed conditions are already frame-level-smoothed (face
    # gap, dark seconds, motion medians, exposure bins) — during a SCAN
    # they pause immediately; the debounce protects against single-eval
    # flicker of the EVIDENCE estimators (and, in preview, the hold).
    _IMMEDIATE_IN_SCAN = ("face", "framing", "lighting", "exposure",
                          "motion", "frame_delivery", "tracking", "frame_rate")

    def _pause_delivery_frame(self, dt: float) -> None:
        """Account for a frame that cannot enter the physiological trace."""
        if not self._paused:
            self._paused = True
            self._n_pauses += 1
            self._pause_t0 = self._prev_t
            self._resume_ok_since = None
        self._paused_s += max(float(dt), 0.0)
        self.feedback.update({"paused": True,
                              "paused_seconds": round(self._paused_s, 2),
                              "disposition": "REPEAT_SCAN"})
        if self._paused_s > float(self.rc["max_paused_s"]) or \
                self._n_pauses > int(self.rc["max_pauses"]):
            self._restart("Camera frame delivery stayed unstable — restart the "
                          "camera and try again.")

    def _effective_failing(self, failing_now: list, immediate=()) -> list:
        """Debounce (v0.1.4): a check counts as failing only after
        `debounce_evals` CONSECUTIVE failing evaluations (default 2 =
        1 s), except checks named in `immediate`. Estimator flicker on a
        single 0.5 s window must not throw away 3 s of accumulated hold;
        persistent problems still do."""
        need = int(self.rc.get("debounce_evals", 2))
        streaks = self._fail_streaks
        self._fail_streaks = {c: streaks.get(c, 0) + 1 for c in failing_now}
        return [c for c in failing_now
                if c in immediate or self._fail_streaks[c] >= need]

    def _jlog(self, rec: dict) -> None:
        """Append one JSON line to the session's readiness log. A session
        that never becomes ready must still leave evidence of WHY."""
        def clean(v):
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            if isinstance(v, (bool, str)) or v is None:
                return v
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, (float, np.floating)):
                return round(float(v), 4) if np.isfinite(v) else None
            return str(v)
        try:
            if self._log_f is None:
                self._log_f = open(self.work_dir / "readiness_log.jsonl",
                                   "a", buffering=1)
            self._log_f.write(json.dumps(clean(rec)) + "\n")
        except OSError:
            pass

    def _biomarker_log(self, payload: dict) -> None:
        """Persist the complete calculation contract for paired-data QA.

        Unlike the readiness stream, this file has one record per completed
        calculation (or an explicit pipeline error).  It includes values,
        methods, signal evidence, and abstention reasons exactly as returned
        by the API; there is no second logging-only calculation.
        """
        rec = {"recording_id": f"live-{self.id}",
               "pipeline": ["scan", "facial_rppg", "beat_lattice",
                            "resting_hemodynamics", "api_state",
                            "report_render"],
               "biomarkers": payload}
        try:
            with open(self.work_dir / "biomarker_log.jsonl", "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass

    def _update_readiness(self, face_ok, framing, lighting_ok, exposure_ok,
                          motion_ok, *, lighting_status="ok",
                          delivery_ok=True) -> None:
        n = len(self._trace_t)
        stab = self._tracker.summary(self.fps_hint).stability \
            if len(self._tracker.observations) >= 3 else None
        tic = time.perf_counter()
        if n >= int(3.0 * self.fps_hint):
            ts = np.asarray(self._trace_t)
            traces = {r: np.vstack(self._trace[r]) for r in ROI_NAMES}
            fps = float(1.0 / np.median(np.diff(ts))) if n > 1 else self.fps_hint
            ev = window_evidence(traces, ts, fps, self.cfg,
                                 tracking_stability=stab if stab is not None else 1.0)
        else:
            ev = {"insufficient": True, "fps": self.fps_hint, "max_gap_ms": 0.0,
                  "jitter_ms": 0.0,
                  "insufficient_reason": "window shorter than 3 seconds"}
        eval_ms = (time.perf_counter() - tic) * 1000.0
        rd = readiness_from_evidence(ev, self.cfg, face_ok=face_ok,
                                     framing=framing, lighting_ok=lighting_ok,
                                     exposure_ok=exposure_ok, motion_ok=motion_ok,
                                     tracking_stability=stab,
                                     lighting_status=lighting_status,
                                     delivery_ok=delivery_ok)
        # debounce: transient single-eval failures do not break the hold
        failing_now = list(rd["failing"])
        eff = self._effective_failing(
            failing_now,
            immediate=(self._IMMEDIATE_IN_SCAN
                       if self.state is SessionState.SCANNING else ()))
        rd["failing_now"] = failing_now
        rd["failing"] = eff
        rd["ready"] = not eff
        if rd.get("mode") == "advisory":
            blocking = set(self.rc.get("blocking_checks",
                                       ["face", "framing",
                                        "frame_rate_floor"]))
            rd["blocking_pass"] = not any(c in blocking for c in eff)
            rd["advisory_pass"] = not any(c not in blocking for c in eff)
            rd["can_start"] = rd["blocking_pass"]
        # Starting remains permissive in advisory mode. Capture faults can
        # pause good-time accounting, but provisional pulse evidence cannot:
        # it needs the complete scan and final pipeline to decide whether an
        # output is supportable. This avoids making a user prove the result
        # before the system has recorded the evidence needed to compute it.
        record_checks = set(self.rc.get("recording_checks", READINESS_CHECKS))
        record_fail = [c for c in eff if c in record_checks]
        if ev.get("insufficient") and ev.get("insufficient_reason") == \
                "window shorter than 3 seconds":
            warmup_only = {"timestamps", "signal_snr", "cross_roi_coherence",
                           "beat_timing", "prelim_beats", "sqi"}
            record_fail = [c for c in record_fail if c not in warmup_only]
        rd["recording_failing"] = record_fail
        rd["can_record"] = not record_fail
        if eff:
            rd["hint"] = next((rd["checks"][k]["hint"] for k in rd["checks"]
                               if k in eff and rd["checks"][k]["hint"]),
                              rd["hint"])
        else:
            rd["hint"] = "Great — hold still"
        rd["hold_s"] = self.feedback["readiness"].get("hold_s", 0.0)
        rd["evidence"] = {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in ev.items()
                          if k in ("sqi", "cross_roi_coherence", "timing_precision_ms",
                                   "timing_matched_fraction", "n_beats",
                                   "frac_multi_roi", "fps", "max_gap_ms",
                                   "jitter_ms", "split_fraction", "seconds",
                                   "window_coverage", "timestamp_valid",
                                   "collapsed_interval_fraction")}
        rd["per_roi_snr"] = {k: round(v, 2) for k, v in
                             (ev.get("per_roi_snr") or {}).items()}
        fb = self.feedback
        fb["readiness"] = rd
        fb["signal_quality"] = ev.get("sqi") if not ev.get("insufficient") else None
        fb["coherence"] = ev.get("cross_roi_coherence") if not ev.get("insufficient") else None
        fb["coherence_ok"] = bool(rd["checks"].get("cross_roi_coherence", {}).get("pass"))
        fb["hint"] = rd["hint"]
        # live 1-5 star meter (final=False: the interval-count AF gate is
        # length-dependent and only applies to the finished recording)
        if not ev.get("insufficient"):
            cs = confidence_stars(ev, rd, self.cfg, final=False)
            fb["stars"] = {"stars": cs.stars, "score": cs.score,
                           "limiting_factor": cs.limiting_factor,
                           "hint": cs.hint}
        else:
            fb["stars"] = None
        self._jlog({"kind": "eval", "t": self._last_ev_t,
                    "state": self.state.value, "ready": rd["ready"],
                    "disposition": ("READY" if rd["ready"] else "REPEAT_SCAN"),
                    "failing": eff, "failing_now": failing_now,
                    "checks": {k: [v.get("value"), bool(v["pass"])]
                               for k, v in rd["checks"].items()},
                    "ev": rd["evidence"], "per_roi_snr": rd["per_roi_snr"],
                    "lux_proxy": fb.get("lux_proxy"),
                    "lighting": fb.get("lighting"),
                    "photometric": fb.get("photometric"),
                    "frame_delivery": fb.get("frame_delivery"),
                    "timestamp_errors": self._timestamp_errors,
                    "duplicate_frames": self._duplicate_frames,
                    "motion_rel": fb.get("motion_rel"),
                    "paused": fb.get("paused"), "hold_s": rd["hold_s"],
                    "stars": (fb.get("stars") or {}).get("stars"),
                    "eval_ms": eval_ms})

    # ------------------------------------------------------------- scan
    def start_scan(self, camera_settings: Optional[dict] = None,
                   force: bool = False) -> dict:
        with self._lock:
            if self.state is not SessionState.PREVIEW:
                raise RuntimeError(f"cannot start scan from state {self.state}")
            if not force and not self.feedback.get("ready"):
                raise RuntimeError("not ready: " + "; ".join(
                    self.feedback["readiness"].get("failing", [])) or
                    "readiness not established")
            if cv2 is None:
                import sys as _sys
                raise RuntimeError(
                    "opencv (cv2) is not importable in the Python running "
                    f"this server ({_sys.executable}). Stop the demo and "
                    "relaunch with an interpreter that has "
                    "opencv-python-headless installed — `cli.py demo` now "
                    "checks this at startup.")
            self.camera_settings = dict(camera_settings or {})
            self.video_path = str(self.work_dir / f"scan_{self.id}.avi")
            self._scan_ts = []
            self._frame_luma = []
            self._good_s = 0.0
            self._paused = False
            self._paused_s = 0.0
            self._n_pauses = 0
            self._motion_bad_s = 0.0
            self._dark_s = 0.0
            self._timestamp_error_streak = 0
            self._duplicate_run = 0
            self.state = SessionState.SCANNING
            self.feedback["progress"] = 0.0
            self.feedback["paused"] = False
            self._jlog({"kind": "event", "event": "start_scan",
                        "t": self._prev_t, "forced": bool(force),
                        "camera_settings": self.camera_settings})
            return {"video_path": self.video_path}

    def _restart(self, reason: str) -> None:
        """Back to readiness without a result: the recording so far is
        discarded (a doomed recording is not worth the user's wait)."""
        self._close_writer()
        try:
            if self.video_path and pathlib.Path(self.video_path).exists():
                pathlib.Path(self.video_path).unlink()
                sc = pathlib.Path(self.video_path + ".timestamps.json")
                if sc.exists():
                    sc.unlink()
        except OSError:
            pass
        self.state = SessionState.PREVIEW
        self._ready_since = None
        self._jlog({"kind": "event", "event": "restart", "reason": reason,
                    "t": self._prev_t, "good_s": self._good_s,
                    "paused_s": self._paused_s, "n_pauses": self._n_pauses})
        self.feedback.update({"ready": False, "progress": 0.0, "paused": False,
                              "restarted": True, "restart_reason": reason,
                              "disposition": "REPEAT_SCAN"})
        self.feedback["readiness"]["hold_s"] = 0.0
        self.feedback["readiness"]["hint"] = reason

    def _record(self, frame: np.ndarray, t: float, luma: float) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                self.video_path, cv2.VideoWriter_fourcc(*"FFV1"),
                self.fps_hint, (w, h))
            if not self._writer.isOpened():
                self._writer = cv2.VideoWriter(
                    self.video_path, cv2.VideoWriter_fourcc(*"MJPG"),
                    self.fps_hint, (w, h))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if self._scan_ts and t <= self._scan_ts[-1]:
            raise RuntimeError("non-monotone capture timestamp reached recorder")
        self._writer.write(np.ascontiguousarray(frame))
        self._scan_ts.append(t)
        self._frame_luma.append((t, luma))

    def _fail(self, code: str, message: str) -> None:
        self.state = SessionState.FAILED
        self.error = {"code": code, "message": message}
        self.feedback["disposition"] = "NO_RESULT"
        self._jlog({"kind": "event", "event": "fail", "code": code,
                    "message": message, "t": self._prev_t})
        self._close_writer()

    def abort(self, code: str = "cancelled", message: str = "Scan cancelled.") -> None:
        with self._lock:
            if self.state in (SessionState.PREVIEW, SessionState.SCANNING):
                self._fail(code, message)

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            if self.video_path and self._scan_ts:
                with open(self.video_path + ".timestamps.json", "w") as f:
                    json.dump({"timestamps_s": [t - self._scan_ts[0]
                                                for t in self._scan_ts],
                               "camera_settings": self.camera_settings,
                               "client_dropped_frames": int(self.client_dropped),
                               "duplicate_frames_skipped": int(self._duplicate_frames),
                               "timestamp_errors_skipped": int(self._timestamp_errors),
                               "paused_seconds": round(self._paused_s, 3),
                               "n_pauses": int(self._n_pauses),
                               "good_seconds": round(self._good_s, 3),
                               "capture_clock": "browser/camera timestamps, "
                                                "seconds from first frame; "
                                                "holes are pauses/dropped frames"}, f)

    def finish_scan(self) -> None:
        with self._lock:
            if self.state is not SessionState.SCANNING:
                raise RuntimeError(f"cannot finish from state {self.state}")
            # The browser normally calls finish only at 100%; enforce the
            # contract server-side so an interrupted/early request cannot be
            # mistaken for a completed scan.  Keep the session in SCANNING so
            # it can recover and finish, rather than silently processing a
            # short file.
            if self._good_s + 0.25 < self.scan_seconds:
                remaining = max(self.scan_seconds - self._good_s, 0.0)
                self.feedback.update({
                    "disposition": "REPEAT_SCAN", "paused": True,
                    "hint": f"Scan interrupted — collect {remaining:.1f} more seconds",
                })
                raise RuntimeError(
                    f"scan incomplete: {self._good_s:.1f} of "
                    f"{self.scan_seconds:.1f} good seconds captured")
            self._close_writer()
            self._jlog({"kind": "event", "event": "finish_scan",
                        "t": self._prev_t, "good_s": self._good_s,
                        "paused_s": self._paused_s})
            self.state = SessionState.PROCESSING
            self._proc_thread = threading.Thread(target=self._process,
                                                 daemon=True)
            self._proc_thread.start()

    # ------------------------------------------------------- processing
    def _process(self) -> None:
        try:
            result = self._run_pipeline()
            with self._lock:
                self.result = result
                self.state = SessionState.DONE
                self.feedback["disposition"] = (
                    "READY" if result.get("outcome") == "ACCEPT"
                    else result.get("outcome", "NO_RESULT"))
                self._jlog({"kind": "event", "event": "result",
                            "outcome": result.get("outcome"),
                            "predicted_class": result.get("predicted_class"),
                            "no_read_reasons": result.get("no_read_reasons")})
        except Exception as e:                       # surface, never hide
            with self._lock:
                self.state = SessionState.FAILED
                self.error = {"code": "processing_error",
                              "message": f"Processing failed: {e}"}
                self._biomarker_log({
                    "schema_version": 1, "complete": False,
                    "outcome": "PROCESSING_ERROR", "items": [],
                    "error": f"{type(e).__name__}: {e}"})
                self._jlog({"kind": "event", "event": "processing_error",
                            "message": str(e), "t": self._prev_t})

    def _run_pipeline(self) -> dict:
        from inference.pipeline import run_with_details
        cs = self.camera_settings or {}
        exp_locked = (cs.get("exposureMode") == "manual") if "exposureMode" in cs else None
        awb_locked = (cs.get("whiteBalanceMode") == "manual") \
            if "whiteBalanceMode" in cs else None
        manifest = {"capture_profile": "consumer", "assume_rig_locks": False,
                    "exposure_locked": exp_locked, "awb_locked": awb_locked}
        scan_result, det = run_with_details(self.video_path, manifest=manifest,
                                            recording_id=f"live-{self.id}")
        # v0.4: the multi-phase orchestrator consumes the lattice for
        # recovery metrics; kept only in memory, never serialized
        self.last_scan_result = scan_result
        self.last_det = det
        ing = det.get("ingest")
        report = {
            "recording_id": scan_result.recording_id,
            "outcome": scan_result.outcome.value,
            "disposition": ("READY" if scan_result.outcome.value == "ACCEPT"
                            else scan_result.outcome.value),
            "predicted_class": scan_result.predicted_class,
            "afib_probability": scan_result.afib_probability,
            "user_facing_text": scan_result.user_facing_text(),
            "pulse_bpm": scan_result.mean_pulse_rate_bpm,
            "signal_quality": scan_result.signal_quality_index,
            "usable_beats": scan_result.usable_beats,
            "analysed_seconds": scan_result.analysed_seconds,
            "no_read_reasons": list(scan_result.no_read_reasons),
            "confidence_stars": scan_result.confidence_stars,
            "confidence_limiting_factor":
                scan_result.confidence_limiting_factor,
            "provenance": {"model_version": scan_result.model_version,
                           "code_commit": scan_result.code_commit,
                           "calibration_version": scan_result.calibration_version,
                           "config_hash": scan_result.config_hash},
            "capture": {"profile": "consumer", "caveats": [],
                        "measured_fps": None, "fps_jitter_ms": None,
                        "lux_proxy": None, "exposure_locked": exp_locked,
                        "awb_locked": awb_locked, "tracker": None,
                        "tracking_stability": None,
                        "face_found_fraction": None,
                        "n_frames": len(self._scan_ts),
                        "client_dropped_frames": int(self.client_dropped),
                        "duplicate_frames_skipped": int(self._duplicate_frames),
                        "timestamp_errors_skipped": int(self._timestamp_errors),
                        "scan_seconds": (self._scan_ts[-1] - self._scan_ts[0])
                        if len(self._scan_ts) > 1 else 0.0},
            "scan_quality": {"good_seconds": round(self._good_s, 2),
                             "paused_seconds": round(self._paused_s, 2),
                             "n_pauses": int(self._n_pauses),
                             "disposition": ("READY" if scan_result.outcome.value == "ACCEPT"
                                             else scan_result.outcome.value)},
            "sqi_components": None,
        }
        if ing is not None:
            report["capture"]["caveats"] = list(getattr(ing, "capture_caveats", []))
            if self.client_dropped:
                report["capture"]["caveats"].append(
                    f"{self.client_dropped} camera frames were not delivered by "
                    "the browser (dropped); analysis skipped those gaps")
            if self._n_pauses:
                report["capture"]["caveats"].append(
                    f"scan paused {self._n_pauses}x ({self._paused_s:.0f} s) while "
                    "signal quality was insufficient; only good time was analysed")
            if ing.meta is not None:
                report["capture"]["measured_fps"] = ing.meta.measured_fps_mean
                report["capture"]["fps_jitter_ms"] = ing.meta.measured_fps_jitter_ms
                report["capture"]["lux_proxy"] = ing.meta.lux_proxy
                report["capture"]["duplicate_frame_fraction"] = \
                    ing.meta.duplicate_frame_fraction
                report["capture"]["collapsed_interval_fraction"] = \
                    ing.meta.collapsed_interval_fraction
            if ing.track is not None:
                report["capture"]["tracker"] = ing.track.tracker
                report["capture"]["tracking_stability"] = ing.track.stability
                report["capture"]["face_found_fraction"] = float(
                    ing.track.n_found / max(ing.track.n_frames, 1))
            report["capture"]["photometric"] = dict(
                getattr(ing, "photometric", {}) or {})
        if "sqi" in det:
            report["sqi_components"] = {k: round(float(v), 3)
                                        for k, v in det["sqi"].components.items()}
        report["rationale"] = det.get("rationale")
        report["evidence"] = det.get("evidence")
        # v0.2.1: per-head results ride as data; the RESULTS SCREEN is the
        # unified Cardiac Rhythm Scan Report (app/report_render.py).
        # RESEARCH_* head entries are filtered at this boundary — they
        # exist only in research artifacts, never in the browser payload
        from datasets.schema import public_head_results
        report["head_results"] = public_head_results(
            det.get("head_results"))
        # One canonical serializer connects the computation to both the API
        # state and the generated HTML report.  An overall rhythm abstention
        # does not erase endpoint-usable research features; a value remains
        # null only when that endpoint's own measured inputs are unavailable.
        # No UI fallback can manufacture a replacement.
        from app.report_data import report_capture_meta
        report_meta = report_capture_meta(
            scan_result, det, self.cfg,
            {"session_id": self.id,
             "datetime": time.strftime("%Y-%m-%d %H:%M"),
             "device": (self.camera_settings or {}).get("label")
             or (report["capture"] or {}).get("tracker"),
             "exposure_locked": exp_locked,
             "awb_locked": awb_locked})
        report["biomarkers"] = report_meta["biomarkers"]
        report["biomarker_debug"] = {
            "stages": ["scan", "facial_rppg", "beat_lattice",
                       "resting_hemodynamics", "api_state",
                       "report_render"],
            "log_file": "biomarker_log.jsonl",
        }
        # v0.8 (owner-directed, spec B.24): the five GATED research
        # tracks, in their OWN payload key so the consumer boundary
        # above is untouched, and only when the operator flag is on.
        # Screen-only: print CSS hides the panel that renders it.
        from app.research_tracks import research_tracks_block
        blk = research_tracks_block(
            scan_result, det, self.cfg, video_path=self.video_path,
            capture={"exposure_locked": exp_locked,
                     "awb_locked": awb_locked})
        if blk:
            report["research_tracks"] = blk
        from app.report_render import render_report_fragment
        report["report_html"] = render_report_fragment(
            scan_result, report_meta)
        self._biomarker_log(report["biomarkers"])
        self._jlog({
            "kind": "biomarkers", "t": self._prev_t,
            "complete": report["biomarkers"].get("complete"),
            "items": [{"key": x.get("key"), "status": x.get("status"),
                       "value": x.get("value"), "unit": x.get("unit"),
                       "reason": x.get("reason")}
                      for x in report["biomarkers"].get("items", [])],
        })
        return report
