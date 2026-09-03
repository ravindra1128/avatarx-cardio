"""
Synthetic arterial-stiffness fixture generator (v0.4 vascular track).

A latent per-participant stiffness variable (cfPWV-like, m/s) drives the
pulse morphology the way stiffer arteries do — faster systolic rise,
earlier and larger reflected wave, shallower and earlier dicrotic notch —
via the `pulse_shape` hook of scripts/make_synth_video.py. Each recording
gets a synchronized contact-PPG sidecar carrying the SAME underlying
waveform at reference bandwidth, a tonometry reference block
(<rid>.pwv.json), a full Recording manifest and an ECG R-peak sidecar,
so the fidelity harness, the baselines and the gate machinery are
end-to-end testable BEFORE any clinical data exists.

Two epistemic regimes, both deliberate:
  informative=True   morphology carries an age/BP-INDEPENDENT stiffness
                     component (s_extra) — incremental value over the
                     age+sex+BP baseline is genuinely present;
  informative=False  morphology carries ONLY the age-explainable part —
                     the T2 "age shortcut" world, where a correct harness
                     must show the head does NOT beat B3.

Nothing here is evidence about humans (HONESTY_NOTE travels with every
artifact); surrogate-domain runs can never open a gate.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datasets.schema import (CaptureConfig, Recording, Rhythm,
                             RhythmAnnotation, Split, SyncMethod, SyncRecord)
from scripts.make_synth_video import pulse_series, synth_video

HONESTY_NOTE = "synthetic interface-proof video; not evidence about humans"
CONTACT_FS_HZ = 250.0


def shape_for_stiffness(s_mps: float) -> dict:
    """Monotone, physiologically-signed morphology couplings. Stiffer
    (higher cfPWV): faster upstroke, earlier + larger wave reflection,
    shallower + earlier dicrotic notch."""
    d = float(s_mps) - 8.0
    c = lambda v, lo, hi: float(np.clip(v, lo, hi))
    return {
        "peak_frac": c(0.20 - 0.006 * d, 0.12, 0.26),
        "sigma": c(0.075 - 0.0025 * d, 0.050, 0.095),
        "refl_amp": c(0.32 + 0.050 * d, 0.08, 0.75),
        "refl_frac": c(0.50 - 0.014 * d, 0.34, 0.62),
        "refl_sigma": 0.10,
        "notch_depth": c(0.18 - 0.014 * d, 0.02, 0.32),
        "notch_frac": c(0.44 - 0.008 * d, 0.32, 0.54),
        "notch_sigma": 0.05,
    }


def contact_ppg_sidecar(path: str, rr: np.ndarray, shape: dict, *,
                        seed: int, fs_hz: float = CONTACT_FS_HZ) -> dict:
    """Write <rid>.ppg.json: the same beat train and morphology sampled at
    contact-reference bandwidth with small sensor noise, on the video
    clock (t0 = 0: the synthetic rig is perfectly synchronized)."""
    rng = np.random.default_rng(seed + 7)
    pulse, _, _ = pulse_series(rr, fs_hz, shape=shape)
    samples = pulse + rng.normal(0.0, 0.01, pulse.size)
    doc = {"fs_hz": fs_hz,
           "samples": [round(float(x), 5) for x in samples],
           "t0_video_s": 0.0, "device": "synthetic-contact-ppg"}
    with open(path, "w") as f:
        json.dump(doc, f)
    return doc


def _recording_manifest(rid: str, pid: str, session_id: str, site: str,
                        fps: float, duration_s: float, truth: dict,
                        vname: str, ename: str,
                        start_utc: str = "2026-08-31T10:00:00Z"
                        ) -> Recording:
    cap = CaptureConfig(
        phone_model="synth-rig", os_version="n/a", camera="front",
        width=320, height=240, nominal_fps=fps, measured_fps_mean=fps,
        measured_fps_jitter_ms=0.1, codec="ffv1", crf=None,
        exposure_locked=True, awb_locked=True, gain_locked=True,
        beautification_disabled=True, illuminance_lux_mean=500.0,
        mount="tripod")
    sync = SyncRecord(SyncMethod.LED_FLASH_MARKER, offset_ms=0.0,
                      sync_uncertainty_ms=1.5, drift_ppm=2.0,
                      verified_at_end=True, n_marker_events=32)
    ann = RhythmAnnotation(
        0.0, duration_s, Rhythm.SINUS, annotator_id="synth",
        adjudicated=True,
        ventricular_rate_bpm=float(60.0 / np.mean(truth["rr_s"])))
    return Recording(
        recording_id=rid, participant_id=pid, session_id=session_id,
        site_id=site, video_path=vname, ecg_path=ename,
        video_start_utc=start_utc,
        video_end_utc=start_utc.replace(":00Z", ":20Z"),
        ecg_start_utc=start_utc,
        ecg_end_utc=start_utc.replace(":00Z", ":20Z"),
        duration_s=duration_s, capture=cap, sync=sync,
        ecg_sampling_hz=500.0, ecg_leads=1, ecg_device="synthetic",
        rhythm_annotations=[ann], ecg_rpeaks_s=truth["rpeaks_s"],
        protocol_id="vascular_rest", lighting_condition="controlled",
        motion_condition="stationary", operator="technician",
        split=Split.UNASSIGNED, dataset_version="synth-vascular-v0.1")


def make_vascular_dataset(out_dir, n_participants: int = 6, *,
                          seed: int = 13, duration_s: float = 20.0,
                          fps: float = 60.0, retest_participants: int = 1,
                          informative: bool = True) -> dict:
    """Paired fidelity/evaluation fixture dataset. The LAST participant
    gets no .pwv.json (the excluded-with-reasons path must fire)."""
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    participants = []
    for i in range(n_participants):
        pid = f"vp{seed:02d}{i:02d}"
        age = float(rng.uniform(22.0, 78.0))
        s_age = 4.6 + 0.075 * (age - 20.0)
        s_extra = float(np.clip(rng.normal(0.0, 1.1), -2.5, 2.5))
        s_true = float(np.clip(s_age + s_extra, 3.6, 16.0))
        # BP tracks age and the age-part of stiffness ONLY — s_extra is
        # the incremental information the face may or may not carry
        sbp = float(np.clip(102.0 + 0.45 * (age - 20.0)
                            + 2.2 * (s_age - 6.0) + rng.normal(0.0, 6.0),
                            92.0, 210.0))
        dbp = float(np.clip(0.60 * sbp + rng.normal(0.0, 4.0), 50.0, 120.0))
        hr = float(rng.uniform(56.0, 84.0))
        cfpwv = float(np.clip(s_true + rng.normal(0.0, 0.25), 3.1, 24.0))
        shape = shape_for_stiffness(s_true if informative else s_age)
        has_pwv = i < n_participants - 1
        n_recs = 2 if i < retest_participants else 1
        # ONE repeat reference read per visit — both scan sidecars must
        # carry the identical value (review finding: two draws
        # contradicted each other for the same visit)
        cfpwv_repeat = (round(float(np.clip(
            cfpwv + rng.normal(0.0, 0.3), 3.1, 24.0)), 2)
            if n_recs == 2 else None)
        site = "siteA" if i % 2 == 0 else "siteB"
        rec_ids = []
        for v in range(n_recs):
            rid = f"r_{pid}" + ("" if v == 0 else f"_v{v + 1}")
            vname, ename = f"{rid}.avi", f"{rid}.ecg.json"
            rr = np.clip(rng.normal(60.0 / hr, 0.035,
                                    int(duration_s * hr / 60.0) + 6),
                         0.5, 1.4)
            truth = synth_video(str(d / vname), kind="vascular_sinus",
                                fps=fps, duration_s=duration_s,
                                seed=seed * 1000 + i * 29 + v,
                                rr_override=rr, pulse_shape=shape)
            with open(d / ename, "w") as f:
                json.dump({"rpeaks_s": truth["rpeaks_s"],
                           "source": "synthetic"}, f)
            rec = _recording_manifest(
                rid, pid, f"{pid}-s1", site, fps, duration_s, truth,
                vname, ename,
                start_utc=f"2026-08-31T10:{10 * v:02d}:00Z")
            (d / f"{rid}.recording.json").write_text(rec.to_json())
            contact_ppg_sidecar(str(d / f"{rid}.ppg.json"),
                                np.asarray(truth["rr_s"], float), shape,
                                seed=seed * 1000 + i * 29 + v)
            if has_pwv:
                ref = {"cfpwv_mps": round(cfpwv, 2),
                       "device_model": "synthetic-tonometer",
                       "operator": "synth",
                       "age_years": round(age, 1),
                       "sex": "female" if i % 2 else "male",
                       "brachial_sbp_mmhg": round(sbp, 1),
                       "brachial_dbp_mmhg": round(dbp, 1),
                       "hr_at_measurement_bpm": round(hr, 1),
                       "meds_antihypertensive": bool(i % 4 == 3),
                       "fitzpatrick_group": 2 + (i % 4),
                       "site_id": site,
                       "device_label": f"rig-{1 if site == 'siteA' else 2}"}
                if cfpwv_repeat is not None:
                    ref["cfpwv_mps_repeat"] = cfpwv_repeat
                (d / f"{rid}.pwv.json").write_text(json.dumps(ref))
            rec_ids.append(rid)
        participants.append({
            "participant_id": pid, "age": round(age, 1),
            "s_true_mps": round(s_true, 3),
            "s_age_mps": round(s_age, 3), "s_extra_mps": round(s_extra, 3),
            "cfpwv_label_mps": round(cfpwv, 2), "sbp": round(sbp, 1),
            "has_pwv": has_pwv, "recordings": rec_ids, "site": site,
            "fitzpatrick_group": 2 + (i % 4)})
    manifest = {"participants": participants, "seed": seed,
                "informative": informative, "fps": fps,
                "duration_s": duration_s, "note": HONESTY_NOTE}
    (d / "vascular_dataset.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_vascular"
    m = make_vascular_dataset(out)
    print(json.dumps({"participants": len(m["participants"]),
                      "out": out, "note": HONESTY_NOTE}, indent=1))
