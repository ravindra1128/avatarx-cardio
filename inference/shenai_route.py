"""The ShenAI beat train as a SECOND interval source for the rhythm decision.

WHY (owner decision 2026-09-16). On every phone scan the rhythm statement
died at one gate: 15 clean consecutive intervals at 60 % coverage. Our
four-region fusion keeps roughly one beat in five at the phone's compression
level (median 7 clean intervals over the nine retained staging scans of
2026-09-16), while ShenAI's own beat train - already shipped beside every
scan as a sidecar (/api/scan-signals) - held a median of 56 at coverage 0.98,
with RMSSD 30 ms / SDNN 43 ms (not a smoothed train) and, on every scan that
carried the SDK's own lnRMSSD, an RMSSD from the train equal to it to the
decimal. scripts/compare_shenai_signal.py holds the full comparison.

WHAT THIS IS NOT. It is not a second pipeline and it makes no call of its
own. It runs the SAME decision (inference/decision_logic.decide_with_rationale)
with the same gates, thresholds, never-diagnose text, star coupling and rate
resolver; only the interval series comes from the train. It never fuses the
train with our beats (the two clocks cannot be aligned - measured, see the
harness), never reads the SDK's waveform (no sample rate), and never runs
when the video path already ACCEPTed.

WHEN IT MAY RUN - three conditions, all fail-closed:
  1. The video path abstained ONLY on interval gates (coverage, interval
     count, split fraction, NaN features). Every SCAN gate our regions
     provide - signal quality, cross-region coherence (or two-region
     verification), beat-timing precision - passed on this scan. A scan whose
     pulse was not seen consistently across the face gets no rhythm statement
     from anyone.
  2. The train passes its own checks: enough beats over enough seconds,
     contiguous (a dropped beat is a segment break, never spanned), the SDK's
     quality and bad-signal figures when present, and - the regularisation
     control - its RMSSD agrees with the SDK's own lnRMSSD when present.
  3. Its rate is corroborated by OUR evidence on the same scan: within
     RATE_TOL of our resolved resting rate (verified/provisional) or of our
     waveform's dominant rhythm when >= MIN_SPECTRAL_ROI_AGREE regions back
     it. No corroboration, no route.

WHAT THE DECISION SEES. Our scan-level evidence (coherence, timing precision,
matched fraction, spectral pulse) unchanged; the train's interval evidence
(count, split fraction, harmonic fraction, dropout, lattice rate) in place of
ours; and mean_roi_agreement = 1/4 so the irregularity noise floor (audit #2)
treats the train as ONE unaveraged detector at the scan's measured timing
precision - the conservative direction: the floor is higher, not lower.

EVERYTHING is recorded under debug.shenai_route and the sheet's `Rhythm
Source` / `ShenAI Route` columns, used or not, with the reason.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

ROUTE_NAME = "shenai_train"

# Gates the train may answer for. Anything else that failed on the video
# path means the SCAN did not support a rhythm statement.
INTERVAL_GATES = frozenset({"coverage", "n_intervals_min", "coverage_any_class",
                            "n_intervals_any_class", "split_fraction_any_class"})
SCAN_GATES = frozenset({"sqi", "cross_roi_coherence", "timing_precision_any_class"})

MIN_TRAIN_BEATS = 15            # the any-call interval floor, as beats
MIN_TRAIN_SPAN_S = 20.0
MIN_SDK_QUALITY = 0.5           # average_signal_quality, when the sidecar has it
MAX_BAD_SIGNAL_FRACTION = 0.10  # bad_signal_seconds / span, when present
RATE_TOL = 0.15                 # same tolerance as features.hemodynamics.PULSE_AGREEMENT_TOL
MIN_SPECTRAL_ROI_AGREE = 2      # for the spectral rhythm to corroborate the train
RMSSD_CONSISTENCY_TOL = 0.30    # train RMSSD vs the SDK's own lnRMSSD
DROPPED_BEAT_GAP_S = 0.05       # start[i+1] - end[i] beyond this = a dropped beat
SINGLE_TRAIN_ROI_AGREEMENT = 0.25   # k = 4 * this = 1: one detector, no averaging


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ------------------------------------------------------------------ the train
def train_from_sidecar(raw: dict) -> dict:
    """Beat times, segment indices and the SDK's own figures from a sidecar
    document. Pure; never raises on a malformed document (returns beats=[])."""
    out = {"beats": [], "segments": [], "n_dropped": 0, "span_s": 0.0,
           "fs_hz": None, "quality": None, "bad_signal_s": None,
           "sdk_hr_bpm": None, "sdk_rmssd_ms": None, "sdk_sdnn_ms": None}
    if not isinstance(raw, dict):
        return out
    hb = raw.get("heartbeats")
    if not isinstance(hb, list):
        return out
    st, en = [], []
    for b in hb:
        if not isinstance(b, dict):
            continue
        s, e = _f(b.get("start_location_sec")), _f(b.get("end_location_sec"))
        if s is None or e is None or e <= s:
            continue
        st.append(s)
        en.append(e)
    if len(st) < 2:
        return out
    order = np.argsort(st)
    st = np.asarray(st, float)[order]
    en = np.asarray(en, float)[order]
    seg, segs = 0, [0]
    for i in range(1, st.size):
        if abs(st[i] - en[i - 1]) > DROPPED_BEAT_GAP_S or st[i] <= st[i - 1]:
            seg += 1
        segs.append(seg)
    out["beats"] = st.tolist()
    out["segments"] = segs
    out["n_dropped"] = int(seg)
    out["span_s"] = float(en[-1] - st[0])
    ppg = raw.get("ppg") if isinstance(raw.get("ppg"), dict) else {}
    out["fs_hz"] = _f(ppg.get("fs_hz"))
    ref = raw.get("reference") if isinstance(raw.get("reference"), dict) else {}
    out["quality"] = _f(ref.get("average_signal_quality"))
    out["bad_signal_s"] = _f(ref.get("bad_signal_seconds"))
    out["sdk_hr_bpm"] = _f(ref.get("heart_rate_bpm"))
    out["sdk_sdnn_ms"] = _f(ref.get("hrv_sdnn_ms"))
    ln = _f(ref.get("hrv_lnrmssd_ms"))
    # The SDK names it lnRMSSD and it IS the natural log: on the 2026-09-16
    # scans exp(value) matched RMSSD computed from the shipped train to the
    # decimal (14.8, 20.4, 23.7 ms). Anything above ~7 cannot be a log of a
    # millisecond RMSSD and is taken as already linear.
    out["sdk_rmssd_ms"] = (math.exp(ln) if ln is not None and ln < 7.0 else ln)
    return out


def train_series(train: dict, fs_hz: Optional[float]):
    """A BeatSeries from the train with SYNTHESISED confidence (the SDK
    exposes none): clean_runs' confidence channel is inert on it, its other
    three channels (physiologic bounds, missed-beat ratio, short-pair) and
    the segment breaks are what filter it."""
    from beats.detector import Beat, BeatSeries
    beats = [Beat(t_s=float(t), confidence=1.0, roi_agreement=1.0,
                  signal_quality=1.0, amplitude=float("nan"),
                  prominence=float("nan"), source_rois=["shenai"], segment=int(s))
             for t, s in zip(train["beats"], train["segments"])]
    fps = float(fs_hz) if fs_hz else 30.0
    return BeatSeries(beats, fps, float(train["span_s"]))


# ------------------------------------------------------------ preconditions
def video_path_allows_route(rationale: Optional[dict], outcome: Optional[str]) -> tuple:
    """(ok, reason). The route may answer for INTERVAL gates only."""
    if outcome == "ACCEPT":
        return False, "video path accepted; the route is not needed"
    if not isinstance(rationale, dict) or not rationale.get("gates"):
        return False, ("no decision rationale on the video path (the scan was "
                       "refused before any beats were read)")
    failed = [g.get("name") for g in rationale["gates"] if not g.get("pass")]
    if not failed:
        return False, f"video path abstained without a failed gate ({outcome})"
    blocking = [n for n in failed
                if not (n in INTERVAL_GATES or str(n).startswith("feature:"))]
    if blocking:
        return False, ("the scan itself did not support a rhythm statement: "
                       + ", ".join(str(b) for b in blocking))
    return True, "video path abstained on interval gates only: " + ", ".join(map(str, failed))


def train_checks(train: dict, rmssd_ms: Optional[float]) -> list:
    """Reasons the train is unusable; empty when it passes."""
    why = []
    n = len(train["beats"])
    if n < MIN_TRAIN_BEATS:
        why.append(f"only {n} beats in the train (< {MIN_TRAIN_BEATS})")
    if train["span_s"] < MIN_TRAIN_SPAN_S:
        why.append(f"train spans {train['span_s']:.1f} s (< {MIN_TRAIN_SPAN_S:.0f} s)")
    q = train.get("quality")
    if q is not None and q < MIN_SDK_QUALITY:
        why.append(f"SDK signal quality {q:.2f} (< {MIN_SDK_QUALITY})")
    bad = train.get("bad_signal_s")
    if bad is not None and train["span_s"] > 0 and bad / train["span_s"] > MAX_BAD_SIGNAL_FRACTION:
        why.append(f"SDK flagged {bad:.0f} s of {train['span_s']:.0f} s as bad signal")
    sdk_r = train.get("sdk_rmssd_ms")
    if sdk_r is not None and rmssd_ms is not None and sdk_r > 0:
        if abs(rmssd_ms - sdk_r) / sdk_r > RMSSD_CONSISTENCY_TOL:
            why.append(f"train RMSSD {rmssd_ms:.0f} ms disagrees with the SDK's own "
                       f"{sdk_r:.0f} ms: the shipped train is not the one the SDK "
                       "measured (regularised or truncated)")
    return why


def corroborate_rate(train_bpm: Optional[float], video_ev: dict,
                     min_intervals: int) -> dict:
    """Is the train's rate backed by OUR evidence on the same scan? Two
    independent checks, either suffices: our resolved resting rate (the same
    resolver every head uses) or our waveform's dominant rhythm when enough
    regions back it. Both come from the video; neither shares a clock with
    the train, so this needs no alignment."""
    from features.rate_guard import resolve_resting_rate
    out = {"train_bpm": train_bpm, "ours_bpm": None, "ours_confidence": None,
           "spectral_bpm": None, "spectral_roi_agree": None,
           "backed_by": [], "ok": False}
    if train_bpm is None or train_bpm <= 0:
        out["reason"] = "the train has no rate"
        return out
    ev = video_ev or {}
    try:
        rr = resolve_resting_rate(ev.get("pulse_lattice_bpm"),
                                  ev.get("pulse_lattice_n_intervals") or ev.get("n_intervals"),
                                  ev, min_intervals=int(min_intervals))
    except Exception:                                          # noqa: BLE001
        rr = {}
    ours = _f(rr.get("bpm"))
    out["ours_bpm"], out["ours_confidence"] = ours, rr.get("confidence")
    if ours is not None and rr.get("confidence") in ("verified", "provisional") \
            and abs(train_bpm - ours) / ours <= RATE_TOL:
        out["backed_by"].append(f"our resting rate {ours:.0f} bpm ({rr.get('confidence')})")
    spec, roia = _f(ev.get("pulse_spectral_bpm")), ev.get("pulse_spectral_roi_agree")
    try:
        roia = int(roia) if roia is not None else None
    except (TypeError, ValueError):
        roia = None
    out["spectral_bpm"], out["spectral_roi_agree"] = spec, roia
    if spec is not None and spec > 0 and roia is not None and roia >= MIN_SPECTRAL_ROI_AGREE \
            and abs(train_bpm - spec) / spec <= RATE_TOL:
        out["backed_by"].append(f"our waveform rhythm {spec:.0f} bpm ({roia} of 4 regions)")
    out["ok"] = bool(out["backed_by"])
    if not out["ok"]:
        parts = []
        if ours is not None:
            parts.append(f"our rate {ours:.0f} bpm ({rr.get('confidence')})")
        else:
            parts.append(f"our rate unresolved ({rr.get('confidence') or 'none'})")
        if spec is not None:
            parts.append(f"waveform {spec:.0f} bpm ({roia if roia is not None else '?'} of 4)")
        out["reason"] = (f"train rate {train_bpm:.0f} bpm is not corroborated by our "
                         f"evidence: " + ", ".join(parts))
    return out


# ------------------------------------------------------------------ evaluate
def evaluate(doc: dict, det: dict, raw: Optional[dict], *,
             waited_s: float = 0.0) -> dict:
    """Run the route on one finished scan. Mutates `doc` ONLY when the route
    is used; always returns the record, which the caller stores under
    debug.shenai_route. Never raises: a failure is a record with a reason."""
    rec = {"route": ROUTE_NAME, "attempted": False, "used": False,
           "waited_s": round(float(waited_s), 2), "reason": None,
           "video_outcome": doc.get("outcome"),
           "video_gates_failed": None, "train": None, "rate_check": None}
    try:
        return _evaluate(doc, det, raw, rec)
    except Exception as e:                                     # noqa: BLE001
        rec["reason"] = f"route failed safely: {type(e).__name__}: {e}"
        rec["used"] = False
        return rec


def _evaluate(doc: dict, det: dict, raw: Optional[dict], rec: dict) -> dict:
    rationale = (doc.get("debug") or {}).get("rationale") or det.get("rationale")
    rec["video_gates_failed"] = (list(rationale.get("gates_failed") or [])
                                 if isinstance(rationale, dict) else None)
    ok, why = video_path_allows_route(rationale, doc.get("outcome"))
    if not ok:
        rec["reason"] = why
        return rec
    return _evaluate_train(doc, det, raw, rec)


def _evaluate_train(doc: dict, det: dict, raw: Optional[dict], rec: dict) -> dict:
    """Shared train assessment. Publication eligibility is enforced by _evaluate.

    Diagnostic callers must supply isolated documents and never publish this output.
    All physiological checks remain here, shared by both callers.
    """
    from beats.ibi import clean_runs, rmssd_from_runs, sdnn_from_runs
    from features.regularity import regularity_from_runs
    from inference.confidence_stars import ConfidenceStars
    from inference.decision_logic import beat_evidence_from_series, decide_with_rationale
    from inference.evidence import pulse_agreement

    rationale = (doc.get("debug") or {}).get("rationale") or det.get("rationale")
    if not isinstance(raw, dict):
        rec["reason"] = ("no ShenAI sidecar arrived for this scan"
                         + (f" (waited {rec['waited_s']:.1f} s)" if rec["waited_s"] else ""))
        return rec
    rec["attempted"] = True

    train = train_from_sidecar(raw)
    cfg = det.get("config")
    if not cfg:
        from configs import load_config
        cfg = load_config()
    rc = cfg["runs"]
    lo_ms, hi_ms = rc["ibi_physiologic_ms"]
    min_conf = float(det.get("min_conf", 0.5))
    series = train_series(train, train.get("fs_hz"))
    rs = clean_runs(series, min_conf=min_conf,
                    min_run_beats=int(rc["min_run_beats"]),
                    max_physiologic_ibi_ms=float(hi_ms),
                    min_physiologic_ibi_ms=float(lo_ms),
                    missed_beat_ratio=float(rc["missed_ratio"])) \
        if train["beats"] else None
    rmssd = _f(rmssd_from_runs(rs)) if rs is not None and rs.runs else None
    sdnn = _f(sdnn_from_runs(rs)) if rs is not None and rs.runs else None
    ibi_clean = (np.concatenate([np.asarray(r, float).ravel() for r in rs.runs])
                 if rs is not None and rs.runs else np.array([]))
    ibi_clean = ibi_clean[np.isfinite(ibi_clean) & (ibi_clean > 0)]
    train_bpm = float(np.median(60000.0 / ibi_clean)) if ibi_clean.size >= 4 else None
    rec["train"] = {
        "beats_n": len(train["beats"]), "span_s": round(train["span_s"], 2),
        "dropped_beats": train["n_dropped"], "fs_hz": train.get("fs_hz"),
        "sdk_quality": train.get("quality"), "sdk_bad_signal_s": train.get("bad_signal_s"),
        "sdk_hr_bpm": train.get("sdk_hr_bpm"), "sdk_rmssd_ms": train.get("sdk_rmssd_ms"),
        "clean_intervals": int(rs.n_intervals) if rs is not None else 0,
        "kept_beats": int(rs.kept_beats) if rs is not None else 0,
        "bpm": (round(train_bpm, 1) if train_bpm is not None else None),
        "rmssd_ms": (round(rmssd, 1) if rmssd is not None else None),
        "sdnn_ms": (round(sdnn, 1) if sdnn is not None else None),
    }
    bad = train_checks(train, rmssd)
    if bad:
        rec["reason"] = "train refused: " + "; ".join(bad)
        return rec

    video_ev = dict(det.get("evidence") or (doc.get("debug") or {}).get("evidence") or {})
    ec_min_int = int(((cfg.get("decision") or {}).get("evidence") or {})
                     .get("min_intervals_any", 15))
    rate = corroborate_rate(train_bpm, video_ev, ec_min_int)
    rec["rate_check"] = rate
    if not rate["ok"]:
        rec["reason"] = rate.get("reason")
        return rec

    # ---- the decision, on the train's intervals with OUR scan evidence ------
    sqi_obj = det.get("sqi")
    sqi = _f(getattr(sqi_obj, "sqi", None))
    if sqi is None:
        sqi = _f(doc.get("signal_quality_index")) or float("nan")
    comps = dict(getattr(sqi_obj, "components", None) or
                 (rationale or {}).get("sqi_components") or {})
    fps = float(train.get("fs_hz") or 30.0)
    regularity = regularity_from_runs(rs.runs, rs.run_confidences, rs.dropout_rate,
                                      mean_sqi=sqi, run_times=rs.run_times, fps=fps)
    features = regularity.as_rhythm_features()
    analysed_s = float(sum(float(np.sum(r)) for r in rs.runs) / 1000.0)
    coverage = analysed_s / max(train["span_s"], 1e-6)

    ev = dict(video_ev)
    ev.update(beat_evidence_from_series(series, rs, comps))
    # Scan-level verification stays OURS: the train has one region.
    for k in ("cross_roi_coherence", "timing_precision_ms", "timing_matched_fraction",
              "timing_pair", "frac_multi_roi", "pulse_spectral_bpm",
              "pulse_spectral_roi_agree", "pulse_spectral_snr"):
        if k in video_ev:
            ev[k] = video_ev[k]
    ev["mean_roi_agreement"] = SINGLE_TRAIN_ROI_AGREEMENT
    ev["pulse_agreement"] = pulse_agreement(ev.get("pulse_lattice_bpm"),
                                            ev.get("pulse_spectral_bpm"))
    ev["rhythm_source"] = ROUTE_NAME
    ev["shenai_train"] = dict(rec["train"])

    conf = (rationale or {}).get("confidence") or {}
    stars = None
    if conf.get("stars") is not None:
        stars = ConfidenceStars(stars=int(conf["stars"]), score=float(conf.get("score") or 0.0),
                                limiting_factor=str(conf.get("limiting_factor") or ""),
                                hint="", per_check=dict(conf.get("per_check") or {}))
    rid = str(doc.get("recording_id") or "unspecified")
    res, why = decide_with_rationale(features, sqi, coverage, cfg, recording_id=rid,
                                     evidence=ev, confidence=stars)
    why["rhythm_source"] = ROUTE_NAME
    why["confidence"] = conf
    why["sqi_components"] = {k: float(v) for k, v in comps.items() if _f(v) is not None}
    why["coverage"] = float(coverage)
    why["analysed_seconds"] = analysed_s
    rec["route_outcome"] = res.outcome.value
    rec["route_class"] = res.predicted_class
    rec["route_reasons"] = list(res.no_read_reasons)
    rec["route_coverage"] = round(coverage, 3)
    rec["used"] = True
    rec["reason"] = (f"used: {rs.n_intervals} clean intervals at coverage "
                     f"{coverage:.2f}, rate {train_bpm:.0f} bpm backed by "
                     + " and ".join(rate["backed_by"]))

    # ---- apply to the response ---------------------------------------------
    dbg = doc.setdefault("debug", {})
    dbg["video_rationale"] = rationale
    dbg["video_outcome"] = {"outcome": doc.get("outcome"),
                            "predicted_class": doc.get("predicted_class"),
                            "no_read_reasons": list(doc.get("no_read_reasons") or []),
                            "usable_beats": doc.get("usable_beats"),
                            "analysed_seconds": doc.get("analysed_seconds"),
                            "mean_pulse_rate_bpm": doc.get("mean_pulse_rate_bpm")}
    dbg["rationale"] = why
    dbg["evidence"] = ev
    doc["outcome"] = res.outcome.value
    doc["afib_probability"] = res.afib_probability
    doc["predicted_class"] = res.predicted_class
    doc["no_read_reasons"] = list(res.no_read_reasons)
    doc["confidence_stars"] = res.confidence_stars
    doc["confidence_limiting_factor"] = res.confidence_limiting_factor
    doc["mean_pulse_rate_bpm"] = res.mean_pulse_rate_bpm
    doc["usable_beats"] = int(rs.kept_beats)
    doc["analysed_seconds"] = analysed_s
    doc["user_facing_text"] = res.user_facing_text()
    doc["rhythm_source"] = ROUTE_NAME
    return rec


# ---------------------------------------------------------------- waiting
def wait_for(getter: Callable[[], Optional[dict]], budget_s: float,
             poll_s: float = 0.25) -> tuple:
    """(document or None, seconds waited). The sidecar is posted right after
    /api/start is accepted, so it normally precedes the decision by the whole
    job; the budget covers a slow link, and an absent sidecar costs at most
    the budget."""
    t0 = time.perf_counter()
    doc = getter()
    while doc is None and (time.perf_counter() - t0) < budget_s:
        time.sleep(poll_s)
        doc = getter()
    return doc, time.perf_counter() - t0
