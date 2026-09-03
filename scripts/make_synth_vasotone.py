"""
Synthetic vasomotor-provocation fixture generator (v0.5 vasotone track).

A latent per-provocation TONE RESPONSE drives the optical pulse
amplitude within one continuous session (baseline → stimulus →
recovery) through the `amplitude_envelope` hook of synth_video, with a
0.08 Hz vasomotion wobble everywhere. The contact perfusion-index
sidecar tracks the SAME latent response (the clean reference arm for
gate W0). The two null arms are first-class fixtures:

  null_optics — physiology flat while a per-frame GAMMA sweep perturbs
                the frames nonlinearly during the "stimulus" window
                (a pure brightness scale would cancel in AC/DC; gamma
                does not — the W1 test needs a real optics confound);
  null_rest   — flat physiology, stable optics: the natural-drift arm.

Every recording gets a Recording manifest (AE/AWB locked, except the
deliberate `uncontrolled_optics` variant for the W-d exclusion path), a
provocation sidecar with phase marks, and a PI sidecar. Nothing here is
evidence about humans (HONESTY_NOTE travels with every artifact);
surrogate-domain runs can never open a gate.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.make_synth_vascular import (HONESTY_NOTE,
                                         _recording_manifest,
                                         shape_for_stiffness)
from scripts.make_synth_video import synth_video

PHASES_S = {"baseline": (0.0, 25.0), "stimulus": (25.0, 50.0),
            "recovery": (50.0, 60.0)}
DURATION_S = 60.0
VASOMOTION_HZ = 0.08
VASOMOTION_DEPTH = 0.04


def tone_envelope(n: int, fps: float, response: float, *,
                  ramp_s: float = 5.0, seed: int = 0) -> np.ndarray:
    """Per-frame amplitude multiplier: 1.0 at baseline, ramping to
    (1 - response) during the stimulus window (constriction shrinks the
    optical pulse), back in recovery; vasomotion wobble throughout."""
    t = np.arange(n) / fps
    env = np.ones(n)
    b0, b1 = PHASES_S["stimulus"]
    r1 = PHASES_S["recovery"][1]
    for i, ti in enumerate(t):
        if b0 <= ti < b0 + ramp_s:
            env[i] = 1.0 - response * (ti - b0) / ramp_s
        elif b0 + ramp_s <= ti < b1:
            env[i] = 1.0 - response
        elif b1 <= ti < min(b1 + ramp_s, r1):
            env[i] = 1.0 - response * (1.0 - (ti - b1) / ramp_s)
    rng = np.random.default_rng(seed + 41)
    env = env * (1.0 + VASOMOTION_DEPTH
                 * np.sin(2 * np.pi * VASOMOTION_HZ * t
                          + rng.uniform(0, 2 * np.pi)))
    return np.clip(env, 0.05, None)


def gamma_sweep(n: int, fps: float, *, depth: float = 0.12,
                seed: int = 0) -> np.ndarray:
    """null_optics per-frame gamma: 1.0 outside the stimulus window,
    slow randomized +/- sweeps inside it (the rig 'varies
    illumination/exposure within realistic bounds'). Each session draws
    its own phase/rate/depth — identical perturbations across sessions
    would make the null arm a single repeated sample (review finding) —
    and the sweep is mean-centered over the window so the perturbation
    tests optics response, not a baked-in brightness step."""
    rng = np.random.default_rng(seed + 13)
    t = np.arange(n) / fps
    g = np.ones(n)
    b0, b1 = PHASES_S["stimulus"]
    inside = (t >= b0) & (t < b1)
    freq = 0.05 * float(rng.uniform(0.7, 1.3))
    phase = float(rng.uniform(0, 2 * np.pi))
    depth_i = depth * float(rng.uniform(0.7, 1.0))
    sweep = np.sin(2 * np.pi * freq * (t[inside] - b0) + phase)
    sweep = sweep - float(np.mean(sweep))
    g[inside] = 1.0 + depth_i * sweep
    return np.clip(g, 0.5, 2.0)


def pi_sidecar(path: str, response: float, *, fs: float = 1.0,
               seed: int = 0, base_pi: float = 2.0) -> dict:
    """Contact perfusion-index trace tracking the latent response."""
    rng = np.random.default_rng(seed + 97)
    n = int(DURATION_S * fs)
    env = tone_envelope(n, fs, response, seed=seed)
    vals = np.clip(base_pi * env + rng.normal(0, 0.03, n), 0.05, None)
    doc = {"fs_hz": fs, "values": [round(float(v), 4) for v in vals],
           "t0_video_s": 0.0, "device": "synthetic-oximeter"}
    pathlib.Path(path).write_text(json.dumps(doc))
    return doc


def make_vasotone_dataset(out_dir, n_participants: int = 4, *,
                          seed: int = 23, fps: float = 30.0,
                          retest_participants: int = 1,
                          informative: bool = True,
                          include_uncontrolled: bool = True) -> dict:
    """Per participant: one cold_pressor provocation (+ a paced_breathing
    second provocation for the first participant), a null_optics arm and
    a null_rest arm; the first `retest_participants` repeat the
    cold_pressor at a second visit. `informative=False` breaks the
    face-side coupling (envelope flat) while the PI reference still
    responds — the world where the camera carries nothing."""
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_frames = int(DURATION_S * fps)
    participants = []
    marks = {k: list(v) for k, v in PHASES_S.items()}

    def _write(rid, pid, session, *, arm, response, gamma=None,
               start="2026-09-01T09:00:00Z", locked=True, extra=None):
        rr = np.clip(rng.normal(0.82, 0.03,
                                int(DURATION_S / 0.6)), 0.5, 1.4)
        stable = int(hashlib.sha256(rid.encode()).hexdigest()[:6], 16)
        env = (tone_envelope(n_frames, fps, response,
                             seed=seed + stable % 1000)
               if response is not None else None)
        truth = synth_video(
            str(d / f"{rid}.avi"), kind="vasotone_sinus", fps=fps,
            duration_s=DURATION_S, seed=seed * 100 + stable % 997,
            rr_override=rr, amplitude_envelope=env,
            optics_gamma_envelope=gamma,
            pulse_shape=shape_for_stiffness(8.0))
        (d / f"{rid}.ecg.json").write_text(json.dumps(
            {"rpeaks_s": truth["rpeaks_s"], "source": "synthetic"}))
        rec = _recording_manifest(rid, pid, session, "siteA", fps,
                                  DURATION_S, truth, f"{rid}.avi",
                                  f"{rid}.ecg.json", start_utc=start)
        mdict = json.loads(rec.to_json())
        if not locked:
            mdict["capture"]["exposure_locked"] = False
            mdict["capture"]["awb_locked"] = False
        (d / f"{rid}.recording.json").write_text(json.dumps(mdict))
        prov = {"maneuver": arm, "phase_marks": marks,
                "fitzpatrick_group": fitz}
        if extra:
            prov.update(extra)
        (d / f"{rid}.provocation.json").write_text(json.dumps(prov))
        return truth

    for i in range(n_participants):
        pid = f"vt{seed:02d}{i:02d}"
        fitz = 2 + (i % 4)
        tau = float(rng.uniform(0.18, 0.42))
        arms = []
        # provocation arm(s)
        face_tau = tau if informative else 0.0
        rid = f"r_{pid}_cp"
        _write(rid, pid, f"{pid}-s1", arm="cold_pressor",
               response=face_tau, extra={"intensity": 2})
        pi_sidecar(str(d / f"{rid}.pi.json"), tau, seed=seed + i)
        arms.append(rid)
        if i == 0:
            rid2 = f"r_{pid}_pb"
            _write(rid2, pid, f"{pid}-s1", arm="paced_breathing",
                   response=face_tau * 0.6,
                   start="2026-09-01T09:10:00Z",
                   extra={"intensity": 1})
            pi_sidecar(str(d / f"{rid2}.pi.json"), tau * 0.6,
                       seed=seed + 50 + i)
            arms.append(rid2)
        # null arms
        rid_no = f"r_{pid}_nullopt"
        _write(rid_no, pid, f"{pid}-s1", arm="null_optics", response=0.0,
               gamma=gamma_sweep(n_frames, fps, seed=seed + i),
               start="2026-09-01T09:20:00Z",
               extra={"optics_log": "gamma sweep +/-12% at 0.05 Hz"})
        pi_sidecar(str(d / f"{rid_no}.pi.json"), 0.0, seed=seed + 70 + i)
        arms.append(rid_no)
        rid_nr = f"r_{pid}_nullrest"
        _write(rid_nr, pid, f"{pid}-s1", arm="null_rest", response=0.0,
               start="2026-09-01T09:30:00Z")
        pi_sidecar(str(d / f"{rid_nr}.pi.json"), 0.0, seed=seed + 90 + i)
        arms.append(rid_nr)
        # retest visit: same maneuver, second session
        if i < retest_participants:
            rid_rt = f"r_{pid}_cp_v2"
            _write(rid_rt, pid, f"{pid}-s2", arm="cold_pressor",
                   response=(tau + float(rng.normal(0, 0.03))
                             if informative else 0.0),
                   start="2026-09-02T09:00:00Z",
                   extra={"intensity": 2})
            pi_sidecar(str(d / f"{rid_rt}.pi.json"), tau,
                       seed=seed + 110 + i)
            arms.append(rid_rt)
        participants.append({"participant_id": pid,
                             "fitzpatrick_group": fitz,
                             "tone_response_true": round(tau, 3),
                             "recordings": arms})
    fitz = 3
    extras = []
    if include_uncontrolled:
        # W-d fuel: an amplitude session WITHOUT AE/AWB lock
        rid_u = f"r_vt{seed:02d}_uncontrolled"
        _write(rid_u, f"vt{seed:02d}uc", f"vt{seed:02d}uc-s1",
               arm="cold_pressor", response=0.3, locked=False,
               extra={"intensity": 2})
        pi_sidecar(str(d / f"{rid_u}.pi.json"), 0.3, seed=seed + 130)
        extras.append(rid_u)
    manifest = {"participants": participants,
                "uncontrolled": extras, "seed": seed,
                "informative": informative, "fps": fps,
                "duration_s": DURATION_S, "phases": PHASES_S,
                "note": HONESTY_NOTE}
    (d / "vasotone_dataset.json").write_text(json.dumps(manifest,
                                                        indent=1))
    return manifest


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_vasotone"
    m = make_vasotone_dataset(out)
    print(json.dumps({"participants": len(m["participants"]),
                      "out": out, "note": HONESTY_NOTE}, indent=1))
