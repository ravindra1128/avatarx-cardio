"""
Streaming video ingestion (T3): decode -> capture gate -> track -> ROI traces.

FAIL-CLOSED ORDERING. The capture gate (codec / fps / illuminance) is
evaluated BEFORE any tracking: a too-dark or over-compressed file must fail
with the capture reason, not with whatever downstream symptom the bad
capture happens to cause. Face tracking failures then abort with their own
reason. Nothing after a failed gate runs at all.

CAPTURE PROFILES (v0.1.1, live demo)
  "research"  — the schema gate verbatim: lossless/CRF/bpp, fps >= 30,
                lux >= 100, AE+AWB locked, no beautification. Default for
                `cli.py process` and every evaluation.
  "consumer"  — the same gate EXCEPT the AE/AWB lock requirement, which a
                laptop/phone webcam driven from a browser cannot honour.
                The lock state is still recorded truthfully in the capture
                record and surfaced as an explicit caveat on the result;
                every other gate still fails closed. This is a DOCUMENTED
                DEVIATION for the consumer scan experience (see spec B.8),
                not a silent weakening: it lives here, in one place, and
                the caveat travels with the result.

Single pass, frame at a time — nothing buffers whole videos in memory.
Tracking gaps are preserved as holes in the capture clock; stale face
geometry is never reused to manufacture ROI samples.  Longer gaps abort.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from datasets.schema import CaptureConfig
from capture.video_reader import VideoMeta, probe_video, iter_frames, \
    build_capture_config, LUX_PROXY_AT_LUMA_160
from capture.face_tracking import FaceTracker, TrackSummary, MAX_FACE_GAP_S
from preprocessing.roi import ROI_NAMES, mean_rgb, roi_integrity, roi_photometry

_LOCK_REASON_MARKER = "auto-exposure and/or auto-white-balance not locked"
_FPS_REASON_MARKER = "fps below"
_LUX_REASON_MARKER = "illuminance"
# Consumer webcams commonly report 29.97 fps or dip briefly under load; a
# strict <30 gate rejects genuinely-usable capture. The consumer profile
# accepts down to this floor and surfaces a caveat (beat timing is coarser
# below 30 fps — the quantisation-floor argument). Research profile keeps 30.
CONSUMER_FPS_FLOOR = 24.0

# Fraction of an ROI trace that must be finite for the trace to be usable.
#
# mean_rgb returns NaN for a degenerate (clipped/empty) ROI box, and its
# contract is explicit that "downstream must treat NaN as missing, never as
# zero signal". The rPPG extractors honour that: rppg/pos.py and
# rppg/chrom.py skip any window whose mean is non-finite and carry on.
#
# This gate used to demand np.all(np.isfinite(...)), which meant a single
# degenerate frame discarded a whole ROI and failed the scan before the
# extractors ever saw it. Measured on a real browser capture: 3 frames out
# of 2824 (0.106%) had a degenerate forehead box — ~1.6 s of signal out of
# 101 s — and that failed the entire recording.
#
# A coverage floor matches how the rest of the pipeline reasons (clean-
# interval coverage has a 0.5 floor) and still rejects a trace that is
# genuinely broken rather than momentarily clipped.
MIN_FINITE_FRACTION = 0.98
MAX_DUPLICATE_FRACTION = 0.05
MAX_DUPLICATE_RUN = 2
MAX_COLLAPSED_INTERVAL_FRACTION = 0.02
MAX_WHOLE_FRAME_CLIP_FRACTION = 0.20


@dataclass
class IngestResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    meta: Optional[VideoMeta] = None
    capture: Optional[CaptureConfig] = None
    traces: dict = field(default_factory=dict)      # roi -> (n, 3) RGB
    timestamps_s: np.ndarray = field(default_factory=lambda: np.array([]))
    track: Optional[TrackSummary] = None
    capture_caveats: list[str] = field(default_factory=list)
    capture_profile: str = "research"
    photometric: dict = field(default_factory=dict)


def apply_capture_gate(capture: CaptureConfig, profile: str
                       ) -> tuple[bool, list[str], list[str]]:
    """(ok, reasons, caveats) for the given profile."""
    ok, why = capture.is_valid_for_beat_analysis()
    if profile == "research":
        return ok, why, []
    if profile != "consumer":
        raise ValueError(f"unknown capture profile {profile!r}")
    fps = capture.measured_fps_mean or capture.nominal_fps or 0.0
    reasons, caveats = [], []
    for w in why:
        if _LOCK_REASON_MARKER in w:
            caveats.append("camera exposure/white-balance not locked "
                           "(automatic): consumer capture — not research-grade")
        elif _FPS_REASON_MARKER in w and fps >= CONSUMER_FPS_FLOOR:
            caveats.append(f"captured at {fps:.1f} fps (below the 30 fps "
                           "research floor): beat timing is coarser")
        else:
            reasons.append(w)
    return len(reasons) == 0, reasons, caveats


def ingest_video(path: str, *,
                 illuminance_lux: Optional[float] = None,
                 assume_rig_locks: bool = True,
                 exposure_locked: Optional[bool] = None,
                 awb_locked: Optional[bool] = None,
                 capture_profile: str = "research",
                 max_face_gap_s: float = MAX_FACE_GAP_S) -> IngestResult:
    """Video file -> per-ROI RGB traces, or a failure with reasons."""
    try:
        meta = probe_video(path)
    except (IOError, RuntimeError) as e:
        return IngestResult(ok=False, reasons=[f"unreadable video: {e}"],
                            capture_profile=capture_profile)

    delivery_reasons = []
    if meta.duplicate_frame_fraction > MAX_DUPLICATE_FRACTION or \
            meta.max_duplicate_run > MAX_DUPLICATE_RUN:
        delivery_reasons.append(
            f"duplicated video frames ({meta.duplicate_frame_fraction:.1%}; "
            f"longest repeated run {meta.max_duplicate_run}) exceed the "
            "capture limit — retry after closing other camera apps")
    if meta.collapsed_interval_fraction > MAX_COLLAPSED_INTERVAL_FRACTION:
        delivery_reasons.append(
            f"capture timestamps contain {meta.collapsed_interval_fraction:.1%} "
            "collapsed intervals — retry with a stable camera frame rate")
    if meta.bright_clip_fraction > MAX_WHOLE_FRAME_CLIP_FRACTION:
        delivery_reasons.append(
            f"video is overexposed ({meta.bright_clip_fraction:.1%} of sampled "
            "pixels clipped) — move away from direct light or lower exposure")
    if delivery_reasons:
        return IngestResult(ok=False, reasons=delivery_reasons, meta=meta,
                            capture_profile=capture_profile)

    capture = build_capture_config(meta, illuminance_lux=illuminance_lux,
                                   assume_rig_locks=assume_rig_locks,
                                   exposure_locked=exposure_locked,
                                   awb_locked=awb_locked)
    cap_ok, cap_why, caveats = apply_capture_gate(capture, capture_profile)
    # v0.1.4: when NO metered lux was supplied, the gate judged the
    # whole-frame PLACEHOLDER proxy — which punishes a dark BACKGROUND,
    # not a dark face (frame luma < 32/255 fails even with a well-lit
    # face). If the proxy lux is the ONLY blocker, defer the verdict:
    # track first, then judge illuminance on the FACE region, where the
    # photons that matter fall. Fail-closed ordering is preserved — a
    # genuinely dark scene still fails with the illuminance reason
    # (alone, or ahead of the tracking reason if no face is found).
    lux_deferred = False
    if not cap_ok and illuminance_lux is None and \
            all(_LUX_REASON_MARKER in w for w in cap_why):
        lux_deferred = True
    if not cap_ok and not lux_deferred:
        return IngestResult(ok=False, reasons=cap_why, meta=meta,
                            capture=capture, capture_caveats=caveats,
                            capture_profile=capture_profile)
    _dark_scene_reason = (cap_why[0] + " (whole-frame estimate)") \
        if lux_deferred else None

    tracker = FaceTracker()
    fps = meta.measured_fps_mean
    max_gap = max(int(max_face_gap_s * fps), 1)
    gap = 0
    ever_found = False
    unusable_gap = 0
    ts: list[float] = []
    rows: dict[str, list[np.ndarray]] = {r: [] for r in ROI_NAMES}
    face_lumas: list[float] = []
    photo_rows: list[dict] = []
    uneven_frames = 0

    for t, frame in iter_frames(path):
        obs = tracker.process(frame)
        if obs.found:
            ever_found = True
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                face_reason = (f"no face detected for more than "
                               f"{max_face_gap_s:.0f} s"
                               if ever_found else
                               "no face detected in the video")
                return IngestResult(
                    ok=False,
                    reasons=(([_dark_scene_reason] if _dark_scene_reason
                              else []) + [face_reason]),
                    meta=meta, capture=capture,
                    track=tracker.summary(fps), capture_caveats=caveats,
                    capture_profile=capture_profile)
        if not obs.found:
            # A missing observation is a real hole.  Reusing the last box
            # samples background/occluder pixels and can create false beats.
            continue

        integ = roi_integrity(obs, frame.shape)
        photo = roi_photometry(frame, obs)
        photo_rows.append(photo)
        uneven_frames += int(np.isfinite(photo["luma_imbalance"]) and
                             photo["luma_imbalance"] > 0.80)
        geometry_ok = min(integ.values()) >= 0.50
        exposure_ok = photo["usable_rois"] >= 2
        if not geometry_ok or not exposure_ok:
            unusable_gap += 1
            if unusable_gap > max_gap:
                if not geometry_ok:
                    why = ("face is partially outside the frame or too close "
                           "for stable skin regions — centre the full face and "
                           "move back slightly")
                else:
                    why = ("fewer than two facial skin regions have usable "
                           "exposure — use even light and remove occlusion")
                return IngestResult(
                    ok=False, reasons=[why], meta=meta, capture=capture,
                    track=tracker.summary(fps), capture_caveats=caveats,
                    capture_profile=capture_profile,
                    photometric={"last": photo})
            continue
        unusable_gap = 0

        # Always measure face-region light for provenance; it is the source
        # of truth when the whole-frame proxy is misleading.
        if obs.found:
            use = obs
            box = use.box or (use.cx - use.ax, use.cy - use.ay,
                              2 * use.ax, 2 * use.ay)
            x0 = int(max(0, box[0])); y0 = int(max(0, box[1]))
            x1 = int(min(frame.shape[1], box[0] + box[2]))
            y1 = int(min(frame.shape[0], box[1] + box[3]))
            if x1 > x0 and y1 > y0:
                crop = frame[y0:y1, x0:x1].astype(np.float32)
                face_lumas.append(float(np.mean(
                    0.114 * crop[..., 0] + 0.587 * crop[..., 1] +
                    0.299 * crop[..., 2])))
        ts.append(t)
        for roi, val in mean_rgb(frame, obs).items():
            rows[roi].append(val)

    if not ever_found:
        return IngestResult(ok=False,
                            reasons=(([_dark_scene_reason] if _dark_scene_reason
                                      else []) +
                                     ["no face detected in the video"]),
                            meta=meta, capture=capture,
                            track=tracker.summary(fps), capture_caveats=caveats,
                            capture_profile=capture_profile)

    if lux_deferred:
        face_luma = float(np.mean(face_lumas)) if face_lumas else 0.0
        face_lux = LUX_PROXY_AT_LUMA_160 * face_luma / 160.0
        capture = build_capture_config(meta, illuminance_lux=face_lux,
                                       assume_rig_locks=assume_rig_locks,
                                       exposure_locked=exposure_locked,
                                       awb_locked=awb_locked)
        cap_ok, cap_why, caveats = apply_capture_gate(capture, capture_profile)
        if not cap_ok:
            return IngestResult(ok=False, reasons=cap_why, meta=meta,
                                capture=capture,
                                track=tracker.summary(fps),
                                capture_caveats=caveats,
                                capture_profile=capture_profile)
        caveats = caveats + [
            f"illuminance judged on the FACE region (~{face_lux:.0f} lux "
            f"proxy); the whole-frame scene is darker "
            f"(~{meta.lux_proxy:.0f} lux proxy)"]

    n_photo = len(photo_rows)
    photometric = {
        "n_observed_frames": n_photo,
        "uneven_frame_fraction": float(uneven_frames / max(n_photo, 1)),
        "median_usable_rois": float(np.median(
            [p["usable_rois"] for p in photo_rows])) if photo_rows else 0.0,
        "max_clipped_fraction": float(max(
            (p["max_clipped_fraction"] for p in photo_rows), default=1.0)),
        "max_dark_fraction": float(max(
            (p["max_dark_fraction"] for p in photo_rows), default=1.0)),
    }
    if photometric["uneven_frame_fraction"] > 0.25:
        caveats = caveats + [
            f"uneven facial illumination in "
            f"{photometric['uneven_frame_fraction']:.0%} of tracked frames; "
            "per-region normalization and robust ROI fusion were applied"]

    traces = {roi: np.vstack(v) for roi, v in rows.items() if v}
    finite_frac = {roi: float(np.isfinite(tr).all(axis=1).mean())
                   for roi, tr in traces.items()}
    bad = [roi for roi, f in finite_frac.items() if f < MIN_FINITE_FRACTION]
    if bad:
        detail = ", ".join(f"{roi} {100.0 * finite_frac[roi]:.1f}% finite"
                           for roi in bad)
        return IngestResult(ok=False,
                            reasons=[f"non-finite ROI trace(s): {bad} "
                                     f"({detail}; floor "
                                     f"{100.0 * MIN_FINITE_FRACTION:.0f}%)"],
                            meta=meta, capture=capture,
                            track=tracker.summary(fps), capture_caveats=caveats,
                            capture_profile=capture_profile,
                            photometric=photometric)
    # Momentary clipping that passed the floor is still worth surfacing: the
    # extractors will skip those windows, so the scan is thinner than it looks.
    partial = sorted(roi for roi, f in finite_frac.items() if f < 1.0)
    if partial:
        caveats = caveats + [
            "brief ROI drop-out (" + ", ".join(
                f"{roi} {100.0 * (1.0 - finite_frac[roi]):.2f}% of frames"
                for roi in partial) + ") — those windows are skipped"]
    return IngestResult(ok=True, meta=meta, capture=capture, traces=traces,
                        timestamps_s=np.asarray(ts),
                        track=tracker.summary(fps), capture_caveats=caveats,
                        capture_profile=capture_profile,
                        photometric=photometric)
