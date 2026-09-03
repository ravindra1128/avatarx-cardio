"""
End-to-end inference orchestration (T5).

    video file ─► ingest (capture gate FIRST, then face tracking, ROIs)
              ─► per-ROI POS/CHROM ─► per-ROI beat detection
              ─► two-pass consensus fusion ─► T2 calibration
              ─► clean_runs (min_conf = CALIBRATED -> 0.5)
              ─► regularity_from_runs (features/regularity.py) — ONE
                 representation for every head; never the raw series
              ─► T4 composite SQI
              ─► decision logic (gates, then classifier)
              ─► ScanResult with full provenance

THIS IS THE PRODUCTION PATH. Any metric quoted anywhere must come through
this function (lesson P2: a matched-pair harness passed while the raw
production path read sinus as AF). The evaluation harness calls exactly
this `run`.

Every ScanResult carries model_version, code_commit, calibration_version
and config_hash — a result that cannot be traced to the code, calibration
and config that produced it is not evidence.
"""
from __future__ import annotations

import pathlib
import subprocess
from typing import Optional

import numpy as np

from configs import load_config, config_hash
from datasets.schema import ScanResult, ScanOutcome
from capture.ingest import ingest_video
from preprocessing.roi import ROI_NAMES
from rppg.pos import pos_pulse
from rppg.chrom import chrom_pulse
from inference.evidence import extract_and_detect, capture_segments, \
    readiness_from_evidence
from inference.confidence_stars import confidence_stars, ConfidenceStars
from beats.lattice import BeatLattice
from rppg.signal_quality import compute_sqi
from beats.detector import detect_beats_single_roi, fuse_multi_roi
from beats.confidence import Calibrator
from beats.ibi import clean_runs
from features.regularity import regularity_from_runs
from inference.decision_logic import decide_with_rationale, \
    beat_evidence_from_series

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EXTRACTORS = {"pos": pos_pulse, "chrom": chrom_pulse}
CALIBRATED_MIN_CONF = 0.5      # what "CALIBRATED" resolves to after T2


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=_REPO_ROOT, capture_output=True, text=True,
                             timeout=5)
        c = out.stdout.strip()
        return c if out.returncode == 0 and c else "no-git"
    except Exception:
        return "no-git"


def _load_calibrator(cfg: dict) -> Calibrator:
    p = pathlib.Path(cfg["decision"]["calibration_path"])
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.exists():
        raise RuntimeError(
            f"calibration artifact missing: {p} — run "
            "scripts/fit_default_calibration.py (results without a "
            "versioned calibration are untraceable; refusing to guess)")
    return Calibrator.load(str(p))


def run(video_path: str, manifest: Optional[dict] = None,
        config: Optional[dict] = None, *,
        recording_id: Optional[str] = None) -> ScanResult:
    """Process one video file into a ScanResult (the production entry)."""
    result, _ = run_with_details(video_path, manifest, config,
                                 recording_id=recording_id)
    return result


def run_with_details(video_path: str, manifest: Optional[dict] = None,
                     config: Optional[dict] = None, *,
                     heads: Optional[list] = None,
                     recording_id: Optional[str] = None
                     ) -> tuple[ScanResult, dict]:
    """As `run`, but also returns the intermediate stages the EVALUATION
    harness needs (fused series, calibrated series, run set, SQI). The
    ScanResult is IDENTICAL to `run`'s — the harness measures the
    production path itself, never a parallel reimplementation (lesson P2).

    Never raises for quality problems — those become NO_RESULT/REPEAT_SCAN
    with reasons; raises only for broken installations (missing calibration
    artifact, no cv2)."""
    cfg = config or load_config()
    chash = config_hash(cfg)
    commit = _git_commit()
    cal = _load_calibrator(cfg)
    rid = recording_id or pathlib.Path(video_path).stem
    manifest = manifest or {}

    def finish(r: ScanResult) -> ScanResult:
        r.code_commit = commit
        r.calibration_version = cal.version
        r.config_hash = chash
        return r

    # Manifest keys understood here: illuminance_lux (metered value beats
    # the luma proxy), capture_profile ("research" default | "consumer" for
    # the live demo — see capture/ingest.py), assume_rig_locks, and the
    # camera-REPORTED exposure_locked / awb_locked states.
    ing = ingest_video(video_path,
                       illuminance_lux=manifest.get("illuminance_lux"),
                       assume_rig_locks=bool(manifest.get("assume_rig_locks",
                                                          True)),
                       exposure_locked=manifest.get("exposure_locked"),
                       awb_locked=manifest.get("awb_locked"),
                       capture_profile=manifest.get("capture_profile",
                                                    "research"))
    if not ing.ok:
        # v0.1.5: even a capture-invalid attempt is GRADED, not just
        # refused — 1 star with the failing capture condition named
        reasons = list(ing.reasons)
        lim = ("lighting" if any("illuminance" in w for w in reasons)
               else "face" if any("face" in w.lower() for w in reasons)
               else "frame_rate" if any("fps" in w for w in reasons)
               else "sqi")
        res = ScanResult(recording_id=rid,
                         outcome=ScanOutcome.NO_RESULT,
                         no_read_reasons=reasons,
                         model_version="interim-rules-v0.1")
        res.confidence_stars = 1
        res.confidence_limiting_factor = lim
        meta = getattr(ing, "meta", None)
        track = getattr(ing, "track", None)
        res.capture_meta = {
            "capture_profile": getattr(ing, "capture_profile", None),
            "measured_fps": getattr(meta, "measured_fps_mean", None),
            "duplicate_frame_fraction": getattr(
                meta, "duplicate_frame_fraction", None),
            "collapsed_interval_fraction": getattr(
                meta, "collapsed_interval_fraction", None),
            "bright_clip_fraction": getattr(meta, "bright_clip_fraction", None),
            "tracking_stability": getattr(track, "stability", None),
            "face_found_fraction": (
                float(track.n_found / track.n_frames)
                if track is not None and track.n_frames else None),
            "tracker": getattr(track, "tracker", None),
            "photometric": dict(getattr(ing, "photometric", {}) or {}),
            "caveats": list(getattr(ing, "capture_caveats", []) or []),
        }
        return finish(res), {"ingest": ing}

    fps = ing.meta.measured_fps_mean
    extractor = _EXTRACTORS[cfg["decision"]["extractor"]]
    band = tuple(cfg["sqi"]["band_hz"])

    # v0.1.2/3: extraction + detection on the CAPTURE clock, split at
    # capture gaps, ROI polarity made consistent — via the SAME shared
    # function the live readiness gate uses (inference/evidence.py).
    ts = np.asarray(ing.timestamps_s, float)
    raw_waveforms, per_roi_beats, segments = extract_and_detect(
        ing.traces, ts, fps, cfg)
    n_gaps = max(len(segments) - 1, 0)

    fused = fuse_multi_roi(per_roi_beats, fps, ing.meta.duration_s, min_rois=2)
    series = cal.apply(fused)

    rc = cfg["runs"]
    min_conf = CALIBRATED_MIN_CONF if rc["min_conf"] == "CALIBRATED" \
        else float(rc["min_conf"])
    lo_ms, hi_ms = rc["ibi_physiologic_ms"]
    rs = clean_runs(series, min_conf=min_conf,
                    min_run_beats=int(rc["min_run_beats"]),
                    max_physiologic_ibi_ms=float(hi_ms),
                    min_physiologic_ibi_ms=float(lo_ms),
                    missed_beat_ratio=float(rc["missed_ratio"]))

    sqi = compute_sqi(raw_waveforms, fps, fused,
                      tracking_stability=ing.track.stability)
    # v0.7 (invariant G-a): the ONE regularity representation, computed
    # once; every head reads it (the afib decision through its legacy
    # RhythmFeatures view — bit-identical, pinned by the equivalence
    # test). Respiration is not on the production path, so the
    # structure family's coupling reads "unavailable" here.
    regularity = regularity_from_runs(
        rs.runs, rs.run_confidences, rs.dropout_rate, mean_sqi=sqi.sqi,
        run_times=rs.run_times, fps=fps)
    features = regularity.as_rhythm_features()
    analysed_s = float(sum(float(np.sum(r)) for r in rs.runs) / 1000.0)
    # coverage is judged against CAPTURED time (sum of contiguous capture
    # segments): an intentional pause of the live scan is a hole in the
    # capture clock, not scan time that failed to yield beats
    captured_s = float(sum(ts[b - 1] - ts[a] for a, b in segments)) \
        if segments else float(ing.meta.duration_s)
    captured_s = max(captured_s, 1e-6)
    coverage = analysed_s / captured_s

    # v0.1.2: beat EVIDENCE travels with the features so the decision can
    # verify the beats behind any irregularity (coherence, harmonic and
    # splitter burden, counts) — rhythm features alone cannot tell
    # detection error from arrhythmia.
    evidence = beat_evidence_from_series(
        series, rs, sqi.components,
        per_roi_trains={r: np.array([b.t_s for b in bs])
                        for r, bs in per_roi_beats.items()})
    evidence["capture_gaps"] = int(n_gaps)
    evidence["capture_segments"] = [(float(ts[a]), float(ts[b - 1]))
                                    for a, b in segments]
    evidence["captured_seconds"] = captured_s
    evidence["duplicate_frame_fraction"] = float(
        ing.meta.duplicate_frame_fraction)
    evidence["collapsed_interval_fraction"] = float(
        ing.meta.collapsed_interval_fraction)
    evidence["photometric"] = dict(ing.photometric or {})

    # v0.1.5: grade the whole recording 1-5 stars from the SAME evidence
    # the decision consumes (readiness checklist vocabulary), then let the
    # decision couple it to the AF call (< 3 stars can never be
    # AFIB_SUGGESTIVE)
    span = float(ts[-1] - ts[0]) if ts.size > 1 else 0.0
    dts = np.diff(ts) if ts.size > 1 else np.array([])
    rd_ev = {
        "insufficient": False,
        "sqi": float(sqi.sqi), "components": dict(sqi.components),
        "per_roi_snr": {r: float(v.get("snr_in_band", 0.0))
                        for r, v in sqi.per_roi.items()},
        "cross_roi_coherence": float(evidence["cross_roi_coherence"]),
        "timing_precision_ms": evidence.get("timing_precision_ms",
                                            float("nan")),
        "timing_matched_fraction": evidence.get("timing_matched_fraction",
                                                0.0),
        "n_beats": int(evidence["n_beats"]),
        "frac_multi_roi": float(evidence["frac_multi_roi"]),
        "split_fraction": float(evidence["split_fraction"]),
        "harmonic_fraction": float(evidence["harmonic_fraction"]),
        "n_intervals": int(evidence["n_intervals"]),
        "fps": float(fps),
        "max_gap_ms": float(np.max(dts) * 1000.0) if dts.size else 0.0,
        "jitter_ms": float(ing.meta.measured_fps_jitter_ms or 0.0),
        "window_coverage": float(captured_s / span) if span > 0 else 0.0,
        "seconds": span,
        "timestamp_valid": bool(ts.size > 1 and np.all(np.isfinite(ts)) and
                                np.all(np.diff(ts) > 0)),
        "collapsed_interval_fraction": float(
            ing.meta.collapsed_interval_fraction),
    }
    rd = readiness_from_evidence(rd_ev, cfg,
                                 tracking_stability=ing.track.stability)
    stars = confidence_stars(rd_ev, rd, cfg, final=True)

    # v0.2 (M1.2): the decision runs as endpoint head #1 over the formal
    # BeatLattice — one production path, now expressed as the mandatory
    # `afib` head; further heads are plug-ins over the same lattice
    # (invariant 14). Behaviour is regression-locked to v0.1.5.
    from heads import enabled_heads
    lattice = BeatLattice.from_pipeline(
        series, rs, {r: bs for r, bs in per_roi_beats.items()},
        fps, ing.meta.duration_s,
        segments=[(float(ts[a]), float(ts[b - 1])) for a, b in segments])
    context = {"cfg": cfg, "recording_id": rid, "features": features,
               "regularity": regularity,
               "sqi": float(sqi.sqi), "coverage": float(coverage),
               "evidence": evidence, "confidence": stars,
               "caveats": list(ing.capture_caveats)}
    head_list = enabled_heads(cfg, override=heads)
    if any(h.name == "rhythm_map" for h in head_list):
        context["waveform"] = _display_waveform(ing, ts, segments, cfg)
    head_results = []
    result = rationale = None
    for h in head_list:
        if h.name == "afib":
            hr, result, rationale = h.run(lattice, context)
            # v0.6: heads that must respect the gate need the gate's
            # verdict. Published only AFTER the decision head has run,
            # so a head ordered before it sees no outcome and must fail
            # closed rather than assume one.
            context["scan_outcome"] = result.outcome.value
        else:
            hr = h.run(lattice, context)
        head_results.append(hr.to_dict())
    rationale["confidence"] = {
        "stars": stars.stars, "score": stars.score,
        "limiting_factor": stars.limiting_factor,
        "per_check": stars.per_check}
    result.head_results = head_results
    result.capture_meta = {
        "capture_profile": ing.capture_profile,
        "measured_fps": float(fps),
        "width": ing.meta.width, "height": ing.meta.height,
        "codec": ing.meta.codec_fourcc,
        "tracker": ing.track.tracker,
        "tracking_stability": float(ing.track.stability),
        "face_found_fraction": float(ing.track.n_found /
                                     max(ing.track.n_frames, 1)),
        "duplicate_frame_fraction": float(ing.meta.duplicate_frame_fraction),
        "collapsed_interval_fraction": float(
            ing.meta.collapsed_interval_fraction),
        "photometric": dict(ing.photometric or {}),
        "caveats": list(ing.capture_caveats)}
    result.usable_beats = int(rs.kept_beats)
    result.analysed_seconds = analysed_s
    if result.signal_quality_index is None:
        result.signal_quality_index = float(sqi.sqi)
    rationale["sqi_components"] = {k: float(v) for k, v in sqi.components.items()}
    rationale["coverage"] = float(coverage)
    rationale["analysed_seconds"] = analysed_s
    return finish(result), {
        "ingest": ing, "fused": fused, "series": series, "runset": rs,
        "sqi": sqi, "features": features, "regularity": regularity,
        "coverage": coverage,
        "evidence": evidence, "rationale": rationale,
        "lattice": lattice, "head_results": head_results,
        # the RESOLVED per-beat confidence floor, so downstream research
        # consumers honor a config-tightened floor instead of a copy
        "min_conf": float(min_conf),
        "config": cfg,
    }

def _display_waveform(ing, ts, segments, cfg) -> dict:
    """Compact MEASURED waveform for the Rhythm Map: the last contiguous
    capture segment (<= 12 s) of the forehead POS pulse, on the capture
    clock — same construction as the live demo report."""
    import numpy as _np
    from rppg.pos import pos_pulse
    try:
        fps = ing.meta.measured_fps_mean
        segs = [(float(ts[a]), float(ts[b - 1])) for a, b in segments] or \
            [(float(ts[0]), float(ts[-1]))]
        a_t, b_t = segs[-1]
        keep = (ts >= max(a_t, b_t - 12.0)) & (ts <= b_t)
        if keep.sum() < int(2 * fps):
            return {}
        wave = pos_pulse(ing.traces["forehead"][keep], fps,
                         band=tuple(cfg["sqi"]["band_hz"]))
        sd = float(_np.std(wave)) or 1.0
        return {"values": [round(float(v / sd), 3) for v in wave],
                "t0_s": float(ts[keep][0]), "fps": float(fps)}
    except Exception:
        return {}
