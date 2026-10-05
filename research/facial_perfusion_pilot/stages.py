"""FPR bench-qualification stages B0-B8.

Each `run_Bx(session)` returns a `StageResult` with one of three statuses:

  PASS          the gate is met, on real evidence
  FAIL          the gate is violated; `hard_stop=True` cancels the pilot
  INCONCLUSIVE  the required recording is absent, or the evidence is
                marginal -- the reason names exactly what is missing

Only B7 (synthetic processing correctness) can reach PASS/FAIL from software
alone. Every hardware stage returns INCONCLUSIVE until its recording exists.
Thresholds live in `__init__`; they are never relaxed here to force a PASS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from capture.video_reader import iter_frames
from . import (WATERMARK, QUANT_HEADROOM_MIN, PHANTOM_DC_DRIFT_MAX,
               UNIFORMITY_MAX, FLICKER_MAINS_SNR_DB, LOCK_ROI_SHIFT_MAX,
               LOCK_RATIO_SHIFT_MAX, SNR_MIN_DB, BEAT_YIELD_MIN,
               BEAT_REJECT_MAX, PI_BASELINE_MIN, PI_BASELINE_MAX,
               SIGMA_NULL_PASS, SIGMA_NULL_HARD_STOP, ROI_MIN_PX)
from .contract import FprSession, STAGE_INPUTS
from . import pi as _pi
from . import synth as _synth
from . import extract as _extract

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"


@dataclass
class StageResult:
    stage: str
    title: str
    status: str
    metrics: dict = field(default_factory=dict)
    reason: str = ""
    corrective_experiment: str = ""
    hard_stop: bool = False
    is_gate: bool = True
    watermark: str = WATERMARK

    def to_dict(self) -> dict:
        return {"stage": self.stage, "title": self.title,
                "status": self.status, "metrics": self.metrics,
                "reason": self.reason,
                "corrective_experiment": self.corrective_experiment,
                "hard_stop": self.hard_stop, "is_gate": self.is_gate}


def _inconclusive(stage: str, title: str, corrective: str) -> StageResult:
    rel = STAGE_INPUTS.get(stage)
    return StageResult(stage, title, INCONCLUSIVE,
                       reason=f"required input {rel!r} is absent",
                       corrective_experiment=corrective)


# --------------------------------------------------------------------------
# Pure analyzers (unit-testable without a session)
# --------------------------------------------------------------------------
def quantisation_headroom(bit_depth: int, n_pixels: int,
                          expected_ac_dc: float) -> dict:
    """Effective resolution = LSB / sqrt(N); headroom vs expected AC/DC.

    This is the report's B0 formula (section 9). It is the IDEAL, spatially
    independent case -- lossy/chroma-subsampled capture reduces the effective
    N (spatially correlated noise), which is why B1's measured noise, not
    this number, is the final word.
    """
    lsb_frac = 1.0 / (2 ** int(bit_depth) - 1)
    eff_res = lsb_frac / np.sqrt(max(int(n_pixels), 1))
    headroom = float(expected_ac_dc / eff_res) if eff_res > 0 else float("inf")
    return {"bit_depth": int(bit_depth), "n_pixels": int(n_pixels),
            "lsb_fraction": float(lsb_frac),
            "effective_resolution": float(eff_res),
            "expected_ac_dc": float(expected_ac_dc),
            "headroom_x": headroom, "headroom_min": QUANT_HEADROOM_MIN}


def _welford_roi(video_path: str, roi: Optional[tuple], channel: int = 1
                 ) -> dict:
    """One pass: per-pixel temporal mean/SD (Welford) + ROI-mean series for
    one channel over an ROI (whole frame if roi is None)."""
    n = 0
    mean = m2 = None
    series: list[float] = []
    ts: list[float] = []
    for t, frame in iter_frames(video_path):
        if roi is None:
            patch = frame[:, :, channel].astype(np.float64)
        else:
            x0, y0, x1, y1 = roi
            patch = frame[y0:y1, x0:x1, channel].astype(np.float64)
        ts.append(t)
        series.append(float(np.mean(patch)))
        if mean is None:
            mean = np.zeros_like(patch)
            m2 = np.zeros_like(patch)
        n += 1
        delta = patch - mean
        mean += delta / n
        m2 += delta * (patch - mean)
    if n < 2:
        raise _pi.FprError("need >= 2 frames for noise stats")
    per_pixel_sd = np.sqrt(m2 / (n - 1))
    return {"n_frames": n, "per_pixel_sd_mean": float(np.mean(per_pixel_sd)),
            "per_pixel_sd_max": float(np.max(per_pixel_sd)),
            "roi_mean_series": np.asarray(series),
            "roi_mean_sd": float(np.std(series)),
            "roi_mean_level": float(np.mean(series)),
            "t": np.asarray(ts)}


def _dc_drift(series: np.ndarray, fps: float, smooth_s: float = 2.0) -> float:
    """(max - min)/mean of a slow-smoothed level series -- fractional DC
    drift with pulse/noise averaged out."""
    x = np.asarray(series, float)
    k = max(int(smooth_s * fps), 1)
    if x.size <= k:
        sm = x
    else:
        kern = np.ones(k) / k
        sm = np.convolve(x, kern, mode="valid")
    lvl = float(np.mean(x)) or 1e-9
    return float((sm.max() - sm.min()) / lvl)


def mains_flicker_snr_db(series: np.ndarray, fps: float) -> dict:
    """Largest spectral peak near a mains-related line (50/100/60/120 Hz and
    their aliases into [0, Nyquist]) vs the local spectral floor, in dB."""
    x = np.asarray(series, float)
    x = x - x.mean()
    if x.size < int(4 * fps):
        return {"mains_snr_db": float("nan"), "peak_hz": None}
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / fps)
    nyq = fps / 2.0
    lines = []
    for base in (50.0, 100.0, 60.0, 120.0):
        a = base % fps
        for cand in (a, fps - a):
            if 0.5 < cand < nyq:
                lines.append(cand)
    best_db, best_hz = float("nan"), None
    floor = float(np.median(spec[1:])) or 1e-12
    for line in lines:
        mask = np.abs(f - line) <= 0.3
        if not mask.any():
            continue
        peak = float(spec[mask].max())
        db = 10.0 * np.log10(peak / floor)
        if not np.isfinite(best_db) or db > best_db:
            best_db, best_hz = db, line
    return {"mains_snr_db": best_db, "peak_hz": best_hz}


def uniformity_from_grid(grid_mean: np.ndarray) -> float:
    """(max - min)/mean across a 4x4 grid of ROI sub-means."""
    g = np.asarray(grid_mean, float)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return float("nan")
    return float((g.max() - g.min()) / (np.mean(g) or 1e-9))


# --------------------------------------------------------------------------
# Stage runners
# --------------------------------------------------------------------------
def run_B0(s: FprSession) -> StageResult:
    title = "B0 quantisation headroom"
    corr = ("provide spec/camera.json {bit_depth, roi_px_w, roi_px_h, "
            "expected_ac_dc_min} for the candidate camera; the calc takes "
            "an afternoon and precedes any purchase")
    if not s.exists("spec/camera.json"):
        return _inconclusive("B0", title, corr)
    spec = s.load_json("spec/camera.json")
    try:
        w = int(spec.get("roi_px_w") or spec.get("roi_px"))
        h = int(spec.get("roi_px_h") or spec.get("roi_px"))
        bits = int(spec["bit_depth"])
        exp = float(spec.get("expected_ac_dc_min", PI_BASELINE_MIN))
    except (KeyError, TypeError, ValueError) as e:
        return StageResult("B0", title, INCONCLUSIVE,
                           reason=f"spec/camera.json is incomplete: {e}",
                           corrective_experiment=corr)
    m = quantisation_headroom(bits, w * h, exp)
    m["roi_px_w"], m["roi_px_h"] = w, h
    roi_ok = w >= ROI_MIN_PX and h >= ROI_MIN_PX
    passed = m["headroom_x"] >= QUANT_HEADROOM_MIN and roi_ok
    reason = (f"effective resolution {m['effective_resolution']:.2e} of full "
              f"scale gives {m['headroom_x']:.1f}x headroom over the "
              f"{exp*100:.2f}% expected AC/DC "
              f"(floor {QUANT_HEADROOM_MIN:.0f}x)")
    if not roi_ok:
        reason += f"; ROI {w}x{h}px is below the {ROI_MIN_PX}px floor"
    return StageResult("B0", title, PASS if passed else FAIL, metrics=m,
                       reason=reason, hard_stop=not passed,
                       corrective_experiment=("" if passed else
                       "choose a higher-bit-depth camera or a larger ROI"))


def run_B1(s: FprSession) -> StageResult:
    title = "B1 dark-frame noise"
    corr = ("record calib/dark.avi: 60 s lens-capped at the session's exact "
            "camera settings")
    if not s.exists("calib/dark.avi"):
        return _inconclusive("B1", title, corr)
    roi = s.load_roi() if s.exists("roi/roi.json") else None
    st = _welford_roi(s.path("calib/dark.avi"), roi)
    roi_mean_sd_frac = st["roi_mean_sd"] / 255.0
    per_px_frac = st["per_pixel_sd_mean"] / 255.0
    spec = s.load_json("spec/camera.json") if s.exists("spec/camera.json") \
        else {}
    read_noise_lsb = spec.get("read_noise_lsb")
    metrics = {"per_pixel_sd_lsb": st["per_pixel_sd_mean"],
               "per_pixel_sd_fraction": per_px_frac,
               "roi_mean_sd_lsb": st["roi_mean_sd"],
               "roi_mean_sd_fraction": roi_mean_sd_frac,
               "read_noise_lsb_spec": read_noise_lsb}
    # the ROI-mean dark noise must sit well below the smallest AC/DC we mean
    # to resolve, or it fabricates/obscures a PI.
    passed = roi_mean_sd_frac < PI_BASELINE_MIN / QUANT_HEADROOM_MIN
    reason = (f"dark ROI-mean noise {roi_mean_sd_frac*100:.4f}% of full scale "
              f"(per-pixel {st['per_pixel_sd_mean']:.2f} LSB)")
    if read_noise_lsb is not None:
        exceeded = st["per_pixel_sd_mean"] > float(read_noise_lsb)
        reason += f"; datasheet read noise {read_noise_lsb} LSB " + \
            ("EXCEEDED" if exceeded else "met")
        if exceeded:
            return StageResult("B1", title, FAIL, metrics=metrics,
                               reason=reason, hard_stop=True,
                               corrective_experiment="hardware fault: check "
                               "sensor cooling/settings or RMA the camera")
    return StageResult("B1", title, PASS if passed else INCONCLUSIVE,
                       metrics=metrics, reason=reason,
                       corrective_experiment=("" if passed else
                       "compare per-pixel SD against the sensor datasheet "
                       "read noise (add read_noise_lsb to spec/camera.json)"))


def run_B2(s: FprSession) -> StageResult:
    title = "B2 exposure/WB/focus lock"
    corr = ("record controls/lock_challenge.avi: 60 s static, then a bright "
            "object introduced OUTSIDE the ROI; do NOT trust the API's lock "
            "claim")
    if not s.exists("controls/lock_challenge.avi") or not s.exists("roi/roi.json"):
        return _inconclusive("B2", title, corr)
    roi = s.load_roi()
    ex = _extract.extract_roi_trace(s.path("controls/lock_challenge.avi"), roi,
                                    enforce_min_px=False)
    # onset: the frame where whole-frame brightness jumps (object enters). Use
    # a robust split at the largest step in a smoothed whole-frame proxy --
    # here approximated by the ROI-independent blue level's derivative.
    g = ex.green
    n = g.size
    third = max(n // 3, 1)
    pre = slice(0, third)
    post = slice(n - third, n)
    roi_shift = abs(np.mean(g[post]) - np.mean(g[pre])) / 255.0
    def _ratio(chan):
        return np.mean(chan[post]) / (np.mean(g[post]) or 1e-9) - \
               np.mean(chan[pre]) / (np.mean(g[pre]) or 1e-9)
    rg_shift = abs(_ratio(ex.red))
    bg_shift = abs(_ratio(ex.blue))
    ratio_shift = float(max(rg_shift, bg_shift))
    metrics = {"roi_mean_shift_fraction": float(roi_shift),
               "channel_ratio_shift": ratio_shift,
               "roi_shift_max": LOCK_ROI_SHIFT_MAX,
               "ratio_shift_max": LOCK_RATIO_SHIFT_MAX}
    exposure_locked = roi_shift < LOCK_ROI_SHIFT_MAX
    wb_locked = ratio_shift < LOCK_RATIO_SHIFT_MAX
    passed = exposure_locked and wb_locked
    reason = (f"ROI mean moved {roi_shift*100:.3f}% "
              f"({'exposure locked' if exposure_locked else 'EXPOSURE NOT LOCKED'}); "
              f"channel ratio moved {ratio_shift*100:.3f}% "
              f"({'WB locked' if wb_locked else 'WHITE BALANCE NOT LOCKED'})")
    return StageResult("B2", title, PASS if passed else FAIL, metrics=metrics,
                       reason=reason, hard_stop=not passed,
                       corrective_experiment=("" if passed else
                       "the sensor is not honouring the lock the API reports; "
                       "fix manual control before proceeding"))


def run_B3(s: FprSession) -> StageResult:
    title = "B3 lighting flicker"
    corr = ("record controls/flicker.avi at 2-3x the analysis frame rate so "
            "50/100 or 60/120 Hz lines are inside Nyquist")
    if not s.exists("controls/flicker.avi"):
        return _inconclusive("B3", title, corr)
    roi = s.load_roi() if s.exists("roi/roi.json") else None
    st = _welford_roi(s.path("controls/flicker.avi"), roi)
    fps = 1.0 / np.median(np.diff(st["t"])) if st["t"].size > 1 else 30.0
    m = mains_flicker_snr_db(st["roi_mean_series"], float(fps))
    present = np.isfinite(m["mains_snr_db"]) and \
        m["mains_snr_db"] >= FLICKER_MAINS_SNR_DB
    reason = (f"strongest mains-related peak "
              f"{m['mains_snr_db']:.1f} dB at {m['peak_hz']} Hz "
              f"(floor {FLICKER_MAINS_SNR_DB:.0f} dB)")
    return StageResult("B3", title, FAIL if present else PASS,
                       metrics={**m, "fps": float(fps)}, reason=reason,
                       hard_stop=False,
                       corrective_experiment=("switch to DC-driven flicker-"
                       "free LED panels; check for AC dimmers" if present
                       else ""))


def run_B4(s: FprSession) -> StageResult:
    title = "B4 illumination uniformity"
    corr = ("record calib/flat.avi: 60 s of a uniform grey card filling the "
            "ROI under the session lighting")
    if not s.exists("calib/flat.avi") or not s.exists("roi/roi.json"):
        return _inconclusive("B4", title, corr)
    roi = s.load_roi()
    ex = _extract.extract_roi_trace(s.path("calib/flat.avi"), roi,
                                    enforce_min_px=False)
    grid_mean = np.nanmean(ex.roi_grid, axis=0)
    u = uniformity_from_grid(grid_mean)
    passed = np.isfinite(u) and u <= UNIFORMITY_MAX
    return StageResult("B4", title, PASS if passed else FAIL,
                       metrics={"uniformity": float(u),
                                "uniformity_max": UNIFORMITY_MAX,
                                "grid_mean": grid_mean.tolist()},
                       reason=f"corner-to-corner variation {u*100:.1f}% "
                              f"(max {UNIFORMITY_MAX*100:.0f}%)",
                       hard_stop=False,
                       corrective_experiment=("" if passed else
                       "reposition/diffuse the panels for even ROI illumination"))


def run_B5(s: FprSession) -> StageResult:
    title = "B5 warm-up drift"
    corr = ("record controls/warmup_phantom.avi: power the lamps cold and "
            "record the phantom for ~30 min")
    if not s.exists("controls/warmup_phantom.avi"):
        return _inconclusive("B5", title, corr)
    roi = s.load_roi() if s.exists("roi/roi.json") else None
    st = _welford_roi(s.path("controls/warmup_phantom.avi"), roi)
    t = st["t"]
    fps = 1.0 / np.median(np.diff(t)) if t.size > 1 else 30.0
    series = st["roi_mean_series"]
    dur = float(t[-1] - t[0]) if t.size else 0.0
    # residual drift over the LAST third (proxy for post-warm-up stability)
    n = series.size
    tail = series[2 * n // 3:]
    tail_drift = _dc_drift(tail, float(fps)) if tail.size > int(fps) else \
        float("nan")
    settled = np.isfinite(tail_drift) and tail_drift < PHANTOM_DC_DRIFT_MAX
    return StageResult("B5", title, PASS if settled else INCONCLUSIVE,
                       metrics={"duration_s": dur,
                                "tail_drift_fraction": tail_drift,
                                "drift_max": PHANTOM_DC_DRIFT_MAX},
                       reason=(f"post-warm-up DC drift {tail_drift*100:.3f}% "
                               f"over the final third of {dur/60:.0f} min"),
                       hard_stop=False,
                       corrective_experiment=("" if settled else
                       "extend warm-up beyond 20 min until tail drift < 0.5%"))


def run_B6(s: FprSession) -> StageResult:
    title = "B6 phantom stability (instrumental null)"
    corr = ("record controls/phantom.avi: 10 min of the skin-tone non-"
            "perfused phantom, lamps warm, run the full pipeline on it")
    if not s.exists("controls/phantom.avi") or not s.exists("roi/roi.json"):
        return _inconclusive("B6", title, corr)
    roi = s.load_roi()
    ex = _extract.extract_roi_trace(s.path("controls/phantom.avi"), roi,
                                    enforce_min_px=False)
    drift = _dc_drift(ex.green, ex.fps)
    res = _pi.process_pi(ex.green, ex.t, ex.fps)   # null: baseline = whole
    spurious_pulse = (np.isfinite(res.snr_db) and res.snr_db >= SNR_MIN_DB
                      and res.beat_yield >= BEAT_YIELD_MIN)
    metrics = {"dc_drift_fraction": float(drift),
               "dc_drift_max": PHANTOM_DC_DRIFT_MAX,
               "phantom_sigma_null_pct": res.sigma_null_pct,
               "phantom_snr_db": res.snr_db,
               "phantom_beat_yield": res.beat_yield}
    failed = drift >= PHANTOM_DC_DRIFT_MAX or spurious_pulse
    reason = f"phantom DC drift {drift*100:.3f}% (max {PHANTOM_DC_DRIFT_MAX*100:.1f}%)"
    if spurious_pulse:
        reason += (f"; a pulse-like signal appeared on a NON-perfused phantom "
                   f"(SNR {res.snr_db:.1f} dB) -- instrumental artifact")
    return StageResult("B6", title, FAIL if failed else PASS, metrics=metrics,
                       reason=reason, hard_stop=failed,
                       corrective_experiment=("" if not failed else
                       "an apparent PI on the phantom is instrumental: check "
                       "lighting drift, lock, and compression before subjects"))


def run_B7(s: Optional[FprSession] = None, *, pi_tol_rel: float = 0.05,
           delta_tol_rel: float = 0.10, sigma_max_pct: float = 1.0) -> StageResult:
    """Synthetic PI pipeline test -- PROCESSING correctness. Needs no
    hardware. Injects a known sinusoid on a known DC and checks the recovered
    PI, then a step change and checks the recovered dPI."""
    title = "B7 synthetic PI pipeline"
    m = 0.005
    tr = _synth.synth_green_trace(fps=30.0, duration_s=40.0, hr_hz=1.2,
                                  modulation=m, dc=140.0, noise_frac=0.0)
    res = _pi.process_pi(tr.green, tr.t, tr.fps)
    expected_pi = _synth.expected_pi(m)
    rec_pi = res.baseline_pi
    pi_err = abs(rec_pi - expected_pi) / expected_pi if expected_pi else \
        float("inf")
    # step (dPI) recovery
    step_tr, b_end, exp_delta = _synth.synth_step_trace(
        fps=30.0, baseline_s=60.0, response_s=60.0, hr_hz=1.2,
        m_baseline=0.005, m_response=0.007, dc=140.0)
    step_res = _pi.process_pi(step_tr.green, step_tr.t, step_tr.fps,
                              baseline_end_s=b_end)
    # measure the response on windows fully PAST the step (a 10 s window that
    # straddles an instantaneous step averages the two levels); a real
    # physiological change is gradual, so this is the honest read.
    from . import PI_WINDOW_S as _W
    resp_mask = (step_res.win_centers_s >= b_end + _W) & \
        np.isfinite(step_res.delta_pi_pct)
    rec_delta = float(np.median(step_res.delta_pi_pct[resp_mask])) \
        if resp_mask.any() else float("nan")
    delta_err = abs(rec_delta - exp_delta) / abs(exp_delta) if exp_delta else \
        float("inf")

    metrics = {
        "injected_modulation": m, "expected_pi": expected_pi,
        "recovered_pi": rec_pi, "recovered_pi_pct": rec_pi * 100.0,
        "pi_rel_error": float(pi_err),
        "clean_sigma_null_pct": res.sigma_null_pct,
        "expected_delta_pi_pct": exp_delta,
        "recovered_delta_pi_pct": rec_delta,
        "delta_rel_error": float(delta_err),
        "beat_yield": res.beat_yield, "reject_fraction": res.reject_fraction,
        "snr_db": res.snr_db}
    ok_pi = pi_err <= pi_tol_rel
    ok_sigma = np.isfinite(res.sigma_null_pct) and \
        res.sigma_null_pct <= sigma_max_pct
    ok_delta = delta_err <= delta_tol_rel
    passed = ok_pi and ok_sigma and ok_delta
    reason = (f"recovered PI {rec_pi*100:.3f}% vs injected {expected_pi*100:.3f}% "
              f"({pi_err*100:.1f}% error); clean sigma_null "
              f"{res.sigma_null_pct:.2f}%; recovered dPI {rec_delta:.1f}% vs "
              f"expected {exp_delta:.1f}% ({delta_err*100:.1f}% error)")
    return StageResult("B7", title, PASS if passed else FAIL, metrics=metrics,
                       reason=reason, hard_stop=not passed,
                       corrective_experiment=("" if passed else
                       "processing bug: fix before any subject recording"))


def run_B8(s: FprSession) -> StageResult:
    title = "B8 end-to-end null (sigma_null)"
    corr = ("record raw/session.avi: ~10 min of a stationary subject with no "
            "stimulation, and define roi/roi.json")
    if not s.exists("raw/session.avi") or not s.exists("roi/roi.json"):
        return _inconclusive("B8", title, corr)
    roi = s.load_roi()
    ex = _extract.extract_roi_trace(s.path("raw/session.avi"), roi)
    res = _pi.process_pi(ex.green, ex.t, ex.fps)     # null recording
    baseline_ok = (np.isfinite(res.baseline_pi) and
                   PI_BASELINE_MIN <= res.baseline_pi <= PI_BASELINE_MAX)
    signal_ok = (np.isfinite(res.snr_db) and res.snr_db >= SNR_MIN_DB and
                 res.beat_yield >= BEAT_YIELD_MIN and
                 res.reject_fraction < BEAT_REJECT_MAX)
    metrics = {**res.summary(), "sigma_null_pass": SIGMA_NULL_PASS,
               "sigma_null_hard_stop": SIGMA_NULL_HARD_STOP,
               "baseline_in_band": baseline_ok, "signal_gates_ok": signal_ok}
    sn = res.sigma_null_pct
    if not np.isfinite(sn):
        return StageResult("B8", title, INCONCLUSIVE, metrics=metrics,
                           reason="sigma_null not computable (too few "
                           "windows with accepted beats)",
                           corrective_experiment="check SNR/yield; the "
                           "capture may be too poor to compute PI")
    if sn > SIGMA_NULL_HARD_STOP:
        status, hard = FAIL, True
    elif sn <= SIGMA_NULL_PASS and signal_ok and baseline_ok:
        status, hard = PASS, False
    else:
        status, hard = INCONCLUSIVE, False
    reason = (f"sigma_null {sn:.2f}% (PASS <= {SIGMA_NULL_PASS}%, hard stop "
              f"> {SIGMA_NULL_HARD_STOP}%); SNR {res.snr_db:.1f} dB, yield "
              f"{res.beat_yield*100:.0f}%, reject {res.reject_fraction*100:.0f}%, "
              f"baseline PI {res.baseline_pi*100:.2f}%")
    return StageResult("B8", title, status, metrics=metrics, reason=reason,
                       hard_stop=hard,
                       corrective_experiment=("" if status == PASS else
                       "diagnose via the failure-attribution table before "
                       "tuning anything"))


STAGES: dict[str, Callable[[FprSession], StageResult]] = {
    "B0": run_B0, "B1": run_B1, "B2": run_B2, "B3": run_B3, "B4": run_B4,
    "B5": run_B5, "B6": run_B6, "B7": run_B7, "B8": run_B8,
}
