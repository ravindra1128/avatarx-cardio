"""
Synthetic three-phase recovery sessions (v0.4 T1) — rest clip, guided-
activity clip (periodic vertical bobbing at a scripted cadence; the ONLY
thing the camera may extract from it is workload verification), and a
recovery clip whose HR(t) follows a known mono-exponential decay. Truth
JSON rides beside every artifact.

Everything produced here is an INTERFACE PROOF: it exercises the session
state machine, the cadence counter and the recovery tracker against known
truth. It is not evidence about accuracy on human skin, and no number
derived from these videos may be quoted as a performance claim (v0.1
synthetic-data honesty rule, verbatim).

Usage:
  python3 scripts/make_synth_recovery.py <out_dir>   # standard fixture set
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from protocol.safety import SCREEN_QUESTIONS                               # noqa: E402
from scripts.make_synth_video import (BACKGROUND, PIXEL_NOISE_SD,          # noqa: E402
                                      PULSE_GAIN_RGB, SIZE, SKIN_RGB,
                                      synth_video)

try:
    import cv2
except ImportError:                                   # keep importable
    cv2 = None

HONESTY_NOTE = ("synthetic interface-proof video; not evidence about "
                "humans")


def recovery_hr(t, *, hr_rest: float, hr0: float, tau: float):
    """The parametric truth: HR(t) = rest + (end-exercise - rest)*e^-t/tau."""
    return hr_rest + (hr0 - hr_rest) * np.exp(-np.asarray(t, float) / tau)


def recovery_rr(duration_s: float, *, hr_rest: float = 82.0,
                hr0: float = 142.0, tau: float = 45.0,
                jitter_ms: float = 8.0, seed: int = 0) -> np.ndarray:
    """RR series (s) integrated from the decay + beat-timing jitter."""
    rng = np.random.default_rng(seed)
    rr, t = [], 0.0
    while t < duration_s:
        ibi = 60.0 / float(recovery_hr(t, hr_rest=hr_rest, hr0=hr0,
                                       tau=tau))
        ibi = max(0.30, ibi + rng.normal(0, jitter_ms / 1000.0))
        rr.append(ibi)
        t += ibi
    return np.asarray(rr)


def rest_rr(duration_s: float, *, hr_rest: float = 74.0,
            seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s / 0.5)
    return np.clip(rng.normal(60.0 / hr_rest, 0.03, n), 0.5, 1.4)


def activity_video(path: str, *, fps: float = 30.0,
                   duration_s: float = 60.0,
                   cadence_per_min: float = 20.0,
                   bob_px: float = 26.0, seed: int = 0,
                   codec: str = "FFV1") -> dict:
    """The guided-activity clip: the same synthetic face bobbing
    vertically once per rep at the scripted cadence. NO pulse is painted
    — there is deliberately nothing physiological to extract, which is
    exactly the point (HR is never estimated during movement)."""
    if cv2 is None:
        raise RuntimeError("opencv (cv2) is required to write synthetic "
                           "video")
    rng = np.random.default_rng(seed + 3000)
    w, h = SIZE
    n = int(duration_s * fps)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps,
                             (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"),
                                 fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("no usable video codec (tried FFV1, MJPG)")
    f_hz = cadence_per_min / 60.0
    cx0, cy0, ax, ay = w // 2, h // 2, int(w * 0.22), int(h * 0.40)
    tgrid = np.arange(n) / fps
    bob = -bob_px * 0.5 * (1.0 - np.cos(2.0 * np.pi * f_hz * tgrid))
    reps = int(np.floor(f_hz * duration_s))
    for i in range(n):
        frame = np.full((h, w, 3), BACKGROUND, float)
        rgb = SKIN_RGB + rng.normal(0, 0.5, 3)
        cv2.ellipse(frame, (cx0, int(cy0 + bob[i])), (ax, ay), 0, 0, 360,
                    tuple(float(v) for v in rgb[::-1]), thickness=-1)
        frame += rng.normal(0, PIXEL_NOISE_SD, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()
    truth = {"fps": fps, "duration_s": duration_s,
             "cadence_per_min": cadence_per_min, "reps": reps,
             "bob_px": bob_px, "seed": seed, "note": HONESTY_NOTE}
    with open(path + ".truth.json", "w") as f:
        json.dump(truth, f, indent=2)
    return truth


def make_recovery_session(out_dir, name: str, *, seed: int = 5,
                          protocol_id: str = "sts_1min",
                          hr_rest: float = 82.0, hr0: float = 142.0,
                          tau: float = 45.0,
                          rest_s: float = 60.0, activity_s: float = 60.0,
                          recovery_s: float = 150.0,
                          cadence_per_min: float = 20.0,
                          cadence_scale: float = 1.0,
                          transition_s: float = 4.0,
                          include_activity: bool = True,
                          recovery_jitter_px: float = 0.0,
                          recovery_deficit_ms: float = 0.0,
                          recovery_lux: float = 1.0,
                          rr_jitter_ms: float = 8.0,
                          participant_id: str = None,
                          participant_age: int = 44) -> dict:
    """One complete fixture session: three clips + session manifest +
    truth. `cadence_scale` scripts a non-compliant activity (e.g. 0.7 =
    30% slow); `include_activity=False` scripts a resting-only session."""
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    rest_truth = synth_video(str(d / f"{name}_rest.avi"), kind="rest",
                             fps=30.0, duration_s=rest_s, seed=seed,
                             rr_override=rest_rr(rest_s, seed=seed))
    act_truth = None
    if include_activity:
        act_truth = activity_video(
            str(d / f"{name}_activity.avi"), fps=30.0,
            duration_s=activity_s,
            cadence_per_min=cadence_per_min * cadence_scale, seed=seed)
    rec_truth = synth_video(
        str(d / f"{name}_recovery.avi"), kind="recovery_decay", fps=30.0,
        duration_s=recovery_s, seed=seed,
        jitter_px=recovery_jitter_px,
        deficit_drop_short_ms=recovery_deficit_ms,
        lux_scale=recovery_lux,
        rr_override=recovery_rr(recovery_s, hr_rest=hr_rest, hr0=hr0,
                                tau=tau, jitter_ms=rr_jitter_ms,
                                seed=seed))

    session = {"session_id": name, "protocol_id": protocol_id,
               "participant_id": participant_id or name,
               "participant_context": {"age": participant_age,
                                       "sex": "other",
                                       "measured_weight_kg": 78.0,
                                       "height_cm": 175.0,
                                       "meds": {}},
               "safety_screen": {"passed": True,
                                 "answers": {q: False
                                             for q in SCREEN_QUESTIONS}},
               "phases": {"rest": {"video": f"{name}_rest.avi"}}}
    if include_activity:
        session["phases"]["activity"] = {"video": f"{name}_activity.avi"}
        session["phases"]["recovery"] = {"video": f"{name}_recovery.avi",
                                         "transition_s": transition_s}
    with open(d / f"{name}.session.json", "w") as f:
        json.dump(session, f, indent=2)

    hr0_true = float(recovery_hr(0.0, hr_rest=hr_rest, hr0=hr0, tau=tau))
    truth = {"name": name, "protocol_id": protocol_id,
             "hr_rest": hr_rest, "hr0": hr0, "tau": tau,
             "hr_end_true": hr0_true,
             "hrr30_true": hr0_true - float(recovery_hr(
                 30.0, hr_rest=hr_rest, hr0=hr0, tau=tau)),
             "hrr60_true": hr0_true - float(recovery_hr(
                 60.0, hr_rest=hr_rest, hr0=hr0, tau=tau)),
             "hrr120_true": hr0_true - float(recovery_hr(
                 120.0, hr_rest=hr_rest, hr0=hr0, tau=tau)),
             "cadence_prescribed": cadence_per_min,
             "cadence_scripted": cadence_per_min * cadence_scale,
             "reps_scripted": (act_truth or {}).get("reps"),
             "transition_s": transition_s,
             "include_activity": include_activity,
             "degradations": {"jitter_px": recovery_jitter_px,
                              "deficit_ms": recovery_deficit_ms,
                              "lux": recovery_lux},
             "note": HONESTY_NOTE}
    with open(d / f"{name}.truth.json", "w") as f:
        json.dump(truth, f, indent=2)
    return truth


def main(out_dir: str) -> None:
    specs = [
        ("clean", {}),
        ("dropouts", {"recovery_deficit_ms": 430.0}),
        ("motion", {"recovery_jitter_px": 2.0}),
        ("slowpace", {"cadence_scale": 0.7}),          # non-compliant
        ("resting_only", {"include_activity": False}),
    ]
    for name, kw in specs:
        t = make_recovery_session(out_dir, name, seed=5, **kw)
        print(f"wrote {name}: hrr60_true={t['hrr60_true']:.1f} bpm")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synth_recovery")


def make_fitness_dataset(out_dir, n_participants: int = 6,
                         seed: int = 9) -> dict:
    """CPET-labeled fixture dataset for the §6 harness: one three-phase
    session per participant + a `<id>.cpet.json` label synthesized from a
    KNOWN linear model of age + true HRR60 (so the ladder has real signal
    to find). The LAST participant deliberately lacks a CPET label — the
    EXCLUDED-with-reasons path must fire. Interface proof only."""
    rng = np.random.default_rng(seed)
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"participants": []}
    for i in range(n_participants):
        pid = f"fp{seed:02d}{i:02d}"
        age = 24 + 9 * i
        truth = make_recovery_session(
            d, pid, seed=seed * 100 + i, rest_s=30.0, activity_s=60.0,
            recovery_s=80.0, hr_rest=68.0 + 3.0 * i,
            hr0=152.0 - 3.0 * i, tau=34.0 + 3.0 * i,
            participant_id=pid, participant_age=age)
        vo2 = float(np.clip(52.0 - 0.28 * age
                            + 0.12 * truth["hrr60_true"]
                            + rng.normal(0, 1.0), 18.0, 60.0))
        if i < n_participants - 1:              # last one: no CPET label
            with open(d / f"{pid}.cpet.json", "w") as f:
                json.dump({"vo2peak_mlkgmin": round(vo2, 1),
                           "modality": "treadmill", "protocol": "ramp",
                           "rer_peak": 1.12, "hr_peak": 150.0 + i,
                           "effort_criteria": ["rer>1.10"],
                           "avg_window_s": 30.0, "lab": "synthetic"},
                          f, indent=2)
        manifest["participants"].append({"participant_id": pid,
                                         "age": age,
                                         "vo2_true": round(vo2, 1),
                                         "has_cpet":
                                             i < n_participants - 1})
    with open(d / "fitness_dataset.json", "w") as f:
        json.dump(dict(manifest, note=HONESTY_NOTE), f, indent=2)
    return manifest
