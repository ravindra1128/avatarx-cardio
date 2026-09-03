"""
Synthetic evaluation dataset generator (T7): (recording.json, video, ecg)
triples with hash-assigned participant splits.

Arms: sinus / AF / AF-with-pulse-deficit / RSA (the hard negative the
leakage audit demands in every test split), alternating 30 and 60 fps.
One extra recording carries a CRF-28 manifest — the capture gate must
EXCLUDE it; it exists to prove exclusion is visible, not silent.

Split assignment uses the SAME identity-only hash as
datasets/splits.make_participant_splits, and arms are assigned round-robin
WITHIN each split so every split contains AF and a hard negative.

Everything here is synthetic; no number computed from it is a performance
claim.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datasets.schema import (Recording, CaptureConfig, SyncRecord, SyncMethod,
                             RhythmAnnotation, Rhythm, Split)
from datasets.splits import _stable_unit_interval
from scripts.make_synth_video import synth_video

ARM_CYCLE = ("af", "rsa", "sinus", "af_deficit")
ARM_RHYTHM = {"af": Rhythm.AFIB, "af_deficit": Rhythm.AFIB,
              "sinus": Rhythm.SINUS,
              "rsa": Rhythm.RESPIRATORY_SINUS_ARRHYTHMIA}
FRACTIONS = (0.65, 0.15, 0.20)


def _split_of(pid: str, seed: int) -> Split:
    u = _stable_unit_interval(pid, seed)
    if u < FRACTIONS[0]:
        return Split.TRAIN
    if u < FRACTIONS[0] + FRACTIONS[1]:
        return Split.DEV
    return Split.INTERNAL_TEST


def _recording(rid: str, pid: str, arm: str, fps: float, duration_s: float,
               split: Split, truth: dict, video_name: str, ecg_name: str,
               crf_override=None, codec_override=None) -> Recording:
    cap = CaptureConfig(
        phone_model="synth-rig", os_version="n/a", camera="front",
        width=320, height=240, nominal_fps=fps, measured_fps_mean=fps,
        measured_fps_jitter_ms=0.1,
        codec=codec_override or "ffv1", crf=crf_override,
        exposure_locked=True, awb_locked=True, gain_locked=True,
        beautification_disabled=True, illuminance_lux_mean=500.0,
        mount="tripod")
    sync = SyncRecord(SyncMethod.LED_FLASH_MARKER, offset_ms=0.0,
                      sync_uncertainty_ms=1.5, drift_ppm=2.0,
                      verified_at_end=True, n_marker_events=32)
    ann = RhythmAnnotation(
        0.0, duration_s, ARM_RHYTHM[arm], annotator_id="synth",
        adjudicated=True,
        ventricular_rate_bpm=float(60.0 / np.mean(truth["rr_s"])))
    return Recording(
        recording_id=rid, participant_id=pid, session_id=f"{pid}-s1",
        site_id="siteA", video_path=video_name, ecg_path=ecg_name,
        video_start_utc="2026-08-14T10:00:00Z",
        video_end_utc="2026-08-14T10:00:20Z",
        ecg_start_utc="2026-08-14T10:00:00Z",
        ecg_end_utc="2026-08-14T10:00:20Z",
        duration_s=duration_s, capture=cap, sync=sync,
        ecg_sampling_hz=500.0, ecg_leads=1, ecg_device="synthetic",
        rhythm_annotations=[ann], ecg_rpeaks_s=truth["rpeaks_s"],
        protocol_id=arm, lighting_condition="controlled",
        motion_condition="stationary", operator="technician",
        split=split, dataset_version="synth-v0.1")


def make_dataset(out_dir: str, n_participants: int = 12,
                 duration_s: float = 20.0, seed: int = 11) -> dict:
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    pids = [f"p{seed:02d}{i:03d}" for i in range(n_participants)]
    by_split: dict = {}
    for p in pids:
        by_split.setdefault(_split_of(p, seed), []).append(p)

    manifest = {"recordings": []}
    recs = []
    for split, members in sorted(by_split.items(), key=lambda kv: kv[0].value):
        for j, pid in enumerate(sorted(members)):
            arm = ARM_CYCLE[j % len(ARM_CYCLE)]
            fps = 30.0 if j % 2 == 0 else 60.0
            rid = f"r_{pid}_{arm}"
            vname, ename = f"{rid}.avi", f"{rid}.ecg.json"
            truth = synth_video(
                str(d / vname),
                kind=("af" if arm == "af_deficit" else arm), fps=fps,
                duration_s=duration_s,
                deficit_drop_short_ms=400.0 if arm == "af_deficit" else 0.0,
                seed=seed * 1000 + j * 17 + len(recs))
            with open(d / ename, "w") as f:
                json.dump({"rpeaks_s": truth["rpeaks_s"],
                           "source": "synthetic"}, f)
            rec = _recording(rid, pid, arm, fps, duration_s, split, truth,
                             vname, ename)
            (d / f"{rid}.recording.json").write_text(rec.to_json())
            recs.append(rec)
            manifest["recordings"].append(
                {"recording_id": rid, "participant_id": pid, "arm": arm,
                 "fps": fps, "split": split.value})

    # the deliberately-invalid CRF-28 recording (schema gate must EXCLUDE it)
    pid, arm, fps = f"p{seed:02d}bad", "sinus", 30.0
    rid = f"r_{pid}_crf28"
    vname, ename = f"{rid}.avi", f"{rid}.ecg.json"
    truth = synth_video(str(d / vname), kind="sinus", fps=fps,
                        duration_s=duration_s, seed=seed * 1000 + 999)
    with open(d / ename, "w") as f:
        json.dump({"rpeaks_s": truth["rpeaks_s"], "source": "synthetic"}, f)
    rec = _recording(rid, pid, arm, fps, duration_s, _split_of(pid, seed),
                     truth, vname, ename, crf_override=28,
                     codec_override="h264")
    (d / f"{rid}.recording.json").write_text(rec.to_json())
    manifest["recordings"].append(
        {"recording_id": rid, "participant_id": pid, "arm": "sinus_crf28",
         "fps": fps, "split": rec.split.value})

    # composition sanity: every test split must hold AF + a hard negative
    for split, members in by_split.items():
        if split is Split.INTERNAL_TEST and len(members) >= 2:
            arms = {ARM_CYCLE[j % len(ARM_CYCLE)]
                    for j in range(len(members))}
            assert "af" in arms and "rsa" in arms, (
                "internal test split lacks AF or hard negative; "
                "change the seed")
    with open(d / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_dataset"
    m = make_dataset(out)
    print(f"wrote {len(m['recordings'])} recordings to {out}")
