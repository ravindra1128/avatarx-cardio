"""FPR diagnostic outputs (owner brief section 12).

These exist to make failure MECHANISMS visually obvious BEFORE any
physiology interpretation. Each renderer draws whatever data is present and
skips the rest, so the same call works on a synthetic B7 run and on a full
rig session. matplotlib is imported lazily (Agg backend) so the core harness
carries no plotting dependency.

The renderers cover: raw RGB over time, phantom/colour-chart traces, the
frequency spectrum, raw AC snippets, the PI and dPI time series, the finger-
PPG/PI/HR/temperature overlay, the 4x4 ROI heatmap, per-window quality, and
sigma_null. Ones whose channels are absent are simply not produced.
"""
from __future__ import annotations

from typing import Optional
import os

import numpy as np

from . import PULSE_BAND_HZ


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:                            # noqa: BLE001
        raise RuntimeError(f"matplotlib unavailable: {e}")


def raw_rgb_over_time(ex, path: str) -> str:
    """Diagnostic 1: raw R/G/B ROI-mean intensity over time."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(ex.t, ex.red, color="tab:red", lw=0.7, label="R")
    ax.plot(ex.t, ex.green, color="tab:green", lw=0.7, label="G")
    ax.plot(ex.t, ex.blue, color="tab:blue", lw=0.7, label="B")
    ax.set(xlabel="time (s)", ylabel="raw ROI mean (LSB)",
           title="Raw RGB ROI intensity")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    return _save(fig, path)


def spectrum(green: np.ndarray, fps: float, path: str,
             band=PULSE_BAND_HZ) -> str:
    """Diagnostic 3: amplitude spectrum of the green ROI-mean."""
    plt = _mpl()
    x = np.asarray(green, float)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    f = np.fft.rfftfreq(x.size, 1.0 / fps)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(f, spec, lw=0.7)
    ax.axvspan(band[0], band[1], color="tab:green", alpha=0.12,
               label="pulse band")
    ax.set(xlim=(0, min(6.0, fps / 2)), xlabel="frequency (Hz)",
           ylabel="amplitude", title="Green ROI-mean spectrum")
    ax.legend(fontsize=8)
    return _save(fig, path)


def ac_snippet(res, green: np.ndarray, path: str, seconds: float = 6.0) -> str:
    """Diagnostic 4: a raw AC waveform snippet with detected beats."""
    plt = _mpl()
    from .pi import pulse_ac
    fps = res.fps
    n = min(int(seconds * fps), green.size)
    ac = pulse_ac(green[:n], fps)
    t = np.arange(n) / fps
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, ac, lw=0.8, color="tab:purple")
    peaks = [c.i_peak for c in res.cycles if c.accepted and c.i_peak < n]
    if peaks:
        ax.plot(np.array(peaks) / fps, ac[peaks], "r.", ms=6, label="beats")
        ax.legend(fontsize=8)
    ax.set(xlabel="time (s)", ylabel="AC (band-passed green)",
           title=f"Raw AC snippet (first {seconds:.0f} s)")
    return _save(fig, path)


def pi_series(res, path: str) -> str:
    """Diagnostic 5+6: PI and dPI time series."""
    plt = _mpl()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    a1.plot(res.win_centers_s, res.win_pi * 100.0, "o-", ms=3, lw=0.8)
    if np.isfinite(res.baseline_pi):
        a1.axhline(res.baseline_pi * 100.0, color="k", ls="--", lw=0.8,
                   label="baseline")
        a1.legend(fontsize=8)
    a1.set(ylabel="PI (%)", title="Perfusion index (windowed)")
    if res.delta_pi_pct.size:
        a2.plot(res.win_centers_s, res.delta_pi_pct, "o-", ms=3, lw=0.8,
                color="tab:orange")
    a2.axhline(0, color="k", lw=0.6)
    a2.set(xlabel="time (s)", ylabel="dPI (%)",
           title=f"dPI  (sigma_null = {res.sigma_null_pct:.2f}%)")
    return _save(fig, path)


def roi_heatmap(ex, path: str) -> str:
    """Diagnostic 8: 4x4 ROI PI/level heatmap (time-mean green per cell)."""
    plt = _mpl()
    grid = np.nanmean(ex.roi_grid, axis=0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(grid, cmap="viridis")
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax.text(c, r, f"{grid[r, c]:.0f}", ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean green (LSB)")
    ax.set(title="4x4 ROI sub-region green", xticks=[], yticks=[])
    return _save(fig, path)


def phantom_trace(ex_phantom, res_phantom, path: str) -> str:
    """Diagnostic 2: phantom green level + its (should-be-flat) PI."""
    plt = _mpl()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    a1.plot(ex_phantom.t, ex_phantom.green, lw=0.7, color="tab:brown")
    a1.set(ylabel="phantom green (LSB)", title="Phantom DC (must be flat)")
    if res_phantom.win_pi.size:
        a2.plot(res_phantom.win_centers_s, res_phantom.win_pi * 100.0,
                "o-", ms=3, lw=0.8, color="tab:brown")
    a2.set(xlabel="time (s)", ylabel="phantom PI (%)",
           title="Phantom PI (any response here is instrumental)")
    return _save(fig, path)


def finger_face_overlay(face_res, finger, path: str, *,
                        hr: Optional[tuple] = None,
                        temperature: Optional[tuple] = None,
                        stimulus_window: Optional[tuple] = None) -> str:
    """Diagnostic 7: facial PI + finger-control PI + HR + temperature on ONE
    clock, with the stimulus window shaded. Channels that are None are
    omitted. `hr` is (t, values); `temperature` is (t, {channel: values}).

    The comparison is the point: if the finger control moves and the face
    does not, the question is the face; if temperature tracks facial PI, the
    apparent change may be thermal, not perfusional (owner brief section 14).
    """
    plt = _mpl()
    plan = [("facial PI (%)", "face")]
    if finger is not None:
        plan.append(("finger PI (%)", "finger"))
    if hr is not None:
        plan.append(("HR (bpm)", "hr"))
    if temperature is not None:
        plan.append(("temp (C)", "temp"))
    fig, axes = plt.subplots(len(plan), 1, figsize=(9, 2.1 * len(plan)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (ylabel, kind) in zip(axes, plan):
        if kind == "face":
            ax.plot(face_res.win_centers_s, face_res.win_pi * 100.0, "o-",
                    ms=3, lw=0.8, color="tab:green", label="facial PI")
        elif kind == "finger":
            ax.plot(finger.pi_t, finger.pi_pct, "o-", ms=3, lw=0.8,
                    color="tab:red", label="finger PI (control)")
        elif kind == "hr":
            ax.plot(hr[0], hr[1], lw=0.9, color="tab:purple", label="HR")
        elif kind == "temp":
            for name, vals in temperature[1].items():
                ax.plot(temperature[0], vals, lw=0.9, label=name)
        ax.set(ylabel=ylabel)
        if stimulus_window is not None:
            ax.axvspan(stimulus_window[0], stimulus_window[1],
                       color="tab:blue", alpha=0.12, label="stimulus")
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set(xlabel="time (s)")
    title = "Reference-channel overlay"
    if finger is not None and np.isfinite(getattr(finger, "hr_bpm",
                                                  float("nan"))):
        title += f"  (finger HR {finger.hr_bpm:.0f} bpm)"
    fig.suptitle(title)
    return _save(fig, path)


def render_reference_overlay(session, path: str, *,
                             enforce_min_px: bool = True) -> str:
    """Build the full reference overlay from a session: extract facial PI from
    raw/session.avi, pull the finger control, HR and temperature, shade the
    stimulus window from the event markers, and render. Missing channels are
    simply omitted."""
    from .extract import extract_roi_trace
    from .pi import process_pi
    from . import reference as _ref
    roi = session.load_roi()
    ex = extract_roi_trace(session.path("raw/session.avi"), roi,
                           enforce_min_px=enforce_min_px)
    events = session.load_events()
    stim = None
    base_end = None
    if events:
        try:
            w = _ref.windows_from_events(events, total_s=float(ex.t[-1]))
            stim, base_end = w["stimulus"], w["stimulus"][0]
        except Exception:                            # noqa: BLE001 optional
            stim = base_end = None
    face = process_pi(ex.green, ex.t, ex.fps, baseline_end_s=base_end)
    finger = None
    if session.exists("controls/finger_ppg.csv") or \
            session.exists("controls/finger_pi.csv"):
        finger = session.load_finger()
    return finger_face_overlay(face, finger, path, hr=session.load_hr(),
                               temperature=session.load_temperature(),
                               stimulus_window=stim)


def _trial_face_delta(tr):
    f = getattr(tr, "face_pi", None)
    base = tr.face_baseline_pi_pct / 100.0
    if f is None or f.win_pi.size == 0 or base <= 0:
        return None
    ein = tr.windows["stimulus"][0]
    return f.win_centers_s - ein, (f.win_pi / base - 1.0) * 100.0


def _trial_finger_delta(tr):
    fp = getattr(tr, "finger_pi", None)
    if fp is None:
        return None
    base = fp.pi_in_window(*tr.windows["baseline"])
    if not (np.isfinite(base) and base > 0):
        return None
    ein = tr.windows["stimulus"][0]
    return fp.pi_t - ein, (fp.pi_pct / base - 1.0) * 100.0


def sham_real_overlay(cmp, path: str) -> str:
    """Facial and finger dPI for sham vs real, aligned to immersion onset
    (t=0), stimulus window shaded. If the sham traces sit on top of the real
    ones, the effect is expectancy/apparatus/artifact, not the provocation."""
    plt = _mpl()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    colors = {"real": "tab:red", "sham": "tab:blue"}
    stim_dur = 0.0
    for tr in (cmp.real, cmp.sham):
        col = colors.get(tr.condition, "tab:gray")
        stim_dur = max(stim_dur, tr.windows["stimulus"][1] -
                       tr.windows["stimulus"][0])
        fd = _trial_face_delta(tr)
        if fd is not None:
            a1.plot(fd[0], fd[1], "o-", ms=3, lw=1.0, color=col,
                    label=f"{tr.condition} (dPI {tr.face_delta_pct:+.1f}%)")
        gd = _trial_finger_delta(tr)
        if gd is not None:
            a2.plot(gd[0], gd[1], "o-", ms=3, lw=1.0, color=col,
                    label=tr.condition)
    for ax, ylab, title in ((a1, "facial dPI (%)", "Facial dPI: sham vs real"),
                            (a2, "finger dPI (%)",
                             "Finger control dPI: sham vs real")):
        ax.axvspan(0.0, stim_dur, color="tab:green", alpha=0.10,
                   label="stimulus")
        ax.axhline(0, color="k", lw=0.6)
        ax.set(ylabel=ylab, title=title)
        ax.legend(fontsize=8, loc="upper right")
    a2.set(xlabel="time from immersion onset (s)")
    fig.suptitle(f"SHAM vs REAL — {cmp.category}")
    return _save(fig, path)


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def render_b7(out_dir: str) -> list:
    """Render the B7 synthetic diagnostics (works with no hardware)."""
    from . import synth as _synth
    from .pi import process_pi
    os.makedirs(out_dir, exist_ok=True)
    tr = _synth.synth_green_trace(fps=30.0, duration_s=40.0, hr_hz=1.2,
                                  modulation=0.005, dc=140.0)
    res = process_pi(tr.green, tr.t, tr.fps)
    made = []
    try:
        made.append(spectrum(tr.green, tr.fps,
                             os.path.join(out_dir, "b7_spectrum.png")))
        made.append(ac_snippet(res, tr.green,
                              os.path.join(out_dir, "b7_ac_snippet.png")))
        made.append(pi_series(res, os.path.join(out_dir, "b7_pi_series.png")))
    except RuntimeError:
        return []
    return made
