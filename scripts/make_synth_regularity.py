"""
Synthetic regularity fixture generator (v0.7 regularity track).

The track's reference label is COMPUTABLE from the ECG's own R-R
series, so every fixture is built from an explicit RR series whose
truth is written beside the clip. Cohorts cover what the substrate head
has to tell apart:

  regular     sinus with iid jitter at a chosen true RMSSD;
  rsa_young   deep respiratory sinus arrhythmia (12-18% at the breathing
              rate) — REAL irregularity that is benign, the track's
              dominant false positive; ages 20-34 by design;
  rsa_mid     moderate RSA (6-9%), ages 35-59;
  af          chaotic (make_synth_flutter.af_rr);
  bigeminy    every other beat ectopic: intervals alternate short
              (coupling) / long (compensatory) — the ectopy pattern;
  trigeminy   every third beat;
  pac_isolated a coupled beat plus compensatory pause every ~8 beats;
  ladder_*    a dispersion LADDER of true RMSSD rungs for the noise
              floor (Task 2): the smallest rung the camera separates
              from the metronome IS the minimum detectable irregularity.

Every clip carries a breathing bar (the structure family needs a
respiration channel), an ECG R-peak sidecar, a Recording manifest and
a participant sidecar (age/sex/skin-tone group) for the age strata and
the fairness gate. Nothing here is evidence about humans.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datasets.schema import Rhythm, RhythmAnnotation
from scripts.make_synth_flutter import (_recording_manifest, af_rr,
                                        sinus_rr)
from scripts.make_synth_vascular import HONESTY_NOTE
from scripts.make_synth_video import synth_video

DEFAULT_RESP_BRPM = 15.0
LADDER_RMSSD_MS = (0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def _truncate(rr, duration_s):
    rr = np.asarray(rr, float)
    return rr[np.cumsum(rr) <= duration_s]


def regular_rr(duration_s, *, bpm=70.0, rmssd_ms=8.0, seed=0):
    """iid Gaussian jitter with sigma chosen so the TRUE RMSSD is
    `rmssd_ms` (RMSSD of iid noise = sqrt(2) sigma)."""
    rng = np.random.default_rng(seed + 101)
    base = 60.0 / float(bpm)
    sigma = float(rmssd_ms) / np.sqrt(2.0) / 1000.0
    n = int(duration_s / base) + 6
    return _truncate(np.clip(base + rng.normal(0.0, sigma, n), 0.3, 2.0),
                     duration_s)


def rsa_rr(duration_s, *, bpm=68.0, depth=0.15, resp_brpm=DEFAULT_RESP_BRPM,
           noise_ms=5.0, seed=0):
    return sinus_rr(duration_s, bpm=bpm, rsa_depth=depth,
                    resp_brpm=resp_brpm, noise_ms=noise_ms, seed=seed)


def bigeminy_rr(duration_s, *, bpm=72.0, coupling=0.62, jitter_ms=4.0,
                seed=0):
    """N V N V ...: the interval INTO the ectopic beat is short (coupling
    x RR), the one out of it long (compensatory: 2 RR - short)."""
    rng = np.random.default_rng(seed + 211)
    rr = 60.0 / float(bpm)
    s, l = coupling * rr, 2.0 * rr - coupling * rr
    out, t = [], 0.0
    while t <= duration_s:
        for v in (s, l):
            x = float(v + rng.normal(0.0, jitter_ms / 1000.0))
            out.append(x)
            t += x
    return _truncate(out, duration_s)


def trigeminy_rr(duration_s, *, bpm=72.0, coupling=0.62, jitter_ms=4.0,
                 seed=0):
    rng = np.random.default_rng(seed + 223)
    rr = 60.0 / float(bpm)
    s, l = coupling * rr, 2.0 * rr - coupling * rr
    out, t = [], 0.0
    while t <= duration_s:
        for v in (rr, s, l):
            x = float(v + rng.normal(0.0, jitter_ms / 1000.0))
            out.append(x)
            t += x
    return _truncate(out, duration_s)


def pac_isolated_rr(duration_s, *, bpm=70.0, every=8, coupling=0.70,
                    pause=1.25, rsa_depth=0.04, seed=0):
    """Sinus (with mild RSA) carrying an isolated premature beat every
    `every` beats: a short coupling interval followed by an incomplete
    compensatory pause."""
    rng = np.random.default_rng(seed + 307)
    base = sinus_rr(duration_s * 1.2, bpm=bpm, rsa_depth=rsa_depth,
                    noise_ms=5.0, seed=seed)
    out = []
    for i, v in enumerate(base):
        if i > 0 and i % every == 0:
            out.append(float(coupling * v))
            out.append(float(pause * v + rng.normal(0.0, 0.004)))
        else:
            out.append(float(v))
    return _truncate(out, duration_s)


# Fitzpatrick group -> synthetic skin base colour (RGB). Approximate
# remitted-light tones spanning I..VI; group III is the historical
# generator default. The label now has an OPTICAL counterpart in the
# clip (review finding: it was a JSON field over pixel-identical skin).
FITZPATRICK_SKIN_RGB = {1: (225.0, 190.0, 160.0), 2: (205.0, 165.0, 130.0),
                        3: (180.0, 140.0, 95.0), 4: (150.0, 110.0, 75.0),
                        5: (115.0, 80.0, 55.0), 6: (85.0, 58.0, 40.0)}

COHORTS = {
    # ages span the bands wherever physiology allows, so an age band is
    # not a cohort in disguise (review finding); RSA is the exception —
    # its depth IS age-dependent, so it is three cohorts
    "regular": dict(rhythm=Rhythm.SINUS, age=(18, 85), expect="regular",
                    benign=None,
                    build=lambda d, s, b: regular_rr(d, bpm=70.0,
                                                     rmssd_ms=8.0, seed=s)),
    "rsa_young": dict(rhythm=Rhythm.RESPIRATORY_SINUS_ARRHYTHMIA,
                      age=(20, 34), expect="irregular",
                      benign="respiration_coupled",
                      build=lambda d, s, b: rsa_rr(d, bpm=64.0, depth=0.16,
                                                   resp_brpm=b, seed=s)),
    "rsa_mid": dict(rhythm=Rhythm.RESPIRATORY_SINUS_ARRHYTHMIA,
                    age=(35, 59), expect="irregular",
                    benign="respiration_coupled",
                    build=lambda d, s, b: rsa_rr(d, bpm=70.0, depth=0.08,
                                                 resp_brpm=b, seed=s)),
    "rsa_old": dict(rhythm=Rhythm.RESPIRATORY_SINUS_ARRHYTHMIA,
                    age=(60, 82), expect="irregular",
                    benign="respiration_coupled",
                    build=lambda d, s, b: rsa_rr(d, bpm=66.0, depth=0.10,
                                                 resp_brpm=b, seed=s)),
    "af": dict(rhythm=Rhythm.AFIB, age=(40, 85), expect="irregular",
               benign="chaotic",
               build=lambda d, s, b: af_rr(d, seed=s)),
    "bigeminy": dict(rhythm=Rhythm.PVC_FREQUENT, age=(25, 80),
                     expect="irregular", benign="ectopy_pattern",
                     build=lambda d, s, b: bigeminy_rr(d, seed=s)),
    "trigeminy": dict(rhythm=Rhythm.PVC_FREQUENT, age=(25, 80),
                      expect="irregular", benign="ectopy_pattern",
                      build=lambda d, s, b: trigeminy_rr(d, seed=s)),
    # an isolated premature beat every ~8 beats does NOT cross the
    # published threshold (index ~0.04): sparse ectopy reads REGULAR by
    # definition, so no explanation is owed — recorded as a known
    # behaviour of the label, not a miss of the head
    "pac_isolated": dict(rhythm=Rhythm.PAC, age=(18, 85),
                         expect="regular", benign=None,
                         build=lambda d, s, b: pac_isolated_rr(d, seed=s)),
}


def write_scan(out_dir, rid, cohort, *, pid, session_id, seed,
               duration_s=45.0, fps=30.0, site="siteA", fitzpatrick=3,
               resp_brpm=DEFAULT_RESP_BRPM, age=None, sex=None,
               rmssd_ms=None,
               start_utc="2026-09-01T09:00:00Z") -> dict:
    """One labeled scan: video with breathing bar, ECG R-peak sidecar,
    Recording manifest, participant sidecar. `rmssd_ms` builds a
    dispersion-ladder rung instead of a cohort."""
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    stable = int(hashlib.sha256(rid.encode()).hexdigest()[:6], 16)
    if rmssd_ms is not None:
        rr = regular_rr(duration_s, bpm=70.0, rmssd_ms=float(rmssd_ms),
                        seed=seed + stable % 997)
        rhythm = Rhythm.SINUS
        cohort_name = f"ladder_{float(rmssd_ms):g}ms"
        age_range = (35, 70)
    else:
        c = COHORTS[cohort]
        rr = c["build"](duration_s, seed + stable % 997, resp_brpm)
        rhythm = c["rhythm"]
        cohort_name = cohort
        age_range = c["age"]
    truth = synth_video(str(d / f"{rid}.avi"),
                        kind=f"regularity_track_{cohort_name}", fps=fps,
                        duration_s=duration_s, seed=seed * 100 + stable % 991,
                        rr_override=rr,
                        torso_respiration={"brpm": resp_brpm, "bob_px": 6.0},
                        skin_rgb=FITZPATRICK_SKIN_RGB.get(
                            int(fitzpatrick), FITZPATRICK_SKIN_RGB[3]))
    (d / f"{rid}.ecg.json").write_text(json.dumps(
        {"rpeaks_s": truth["rpeaks_s"], "source": "synthetic"}))
    ann = RhythmAnnotation(0.0, duration_s, rhythm, annotator_id="synth",
                           adjudicated=True,
                           ventricular_rate_bpm=float(
                               60.0 / np.mean(truth["rr_s"])))
    rec = _recording_manifest(rid, pid, session_id, site, fps, duration_s,
                              truth, f"{rid}.avi", f"{rid}.ecg.json", ann,
                              start_utc=start_utc, ecg_leads=1)
    (d / f"{rid}.recording.json").write_text(rec.to_json())
    if age is None:
        lo, hi = age_range
        age = int(lo + (stable % max(hi - lo, 1)))
    if sex is None:
        sex = "F" if stable % 2 else "M"
    (d / f"{rid}.participant.json").write_text(json.dumps(
        {"participant_id": pid, "fitzpatrick_group": int(fitzpatrick),
         "age_years": int(age), "sex": sex}))
    rr_t = np.asarray(truth["rr_s"], float)
    return {"recording_id": rid, "participant_id": pid,
            "cohort": cohort_name, "rhythm": rhythm.value,
            "fps": fps, "fitzpatrick_group": int(fitzpatrick),
            "age_years": int(age), "sex": sex,
            "true_rmssd_ms": round(float(np.sqrt(np.mean(np.diff(rr_t) ** 2))
                                         * 1000.0), 3) if rr_t.size > 2
            else None,
            "true_median_bpm": round(float(60.0 / np.median(rr_t)), 2),
            "video": f"{rid}.avi", "start_utc": start_utc}


def make_regularity_dataset(out_dir, *, seed=71, duration_s=45.0,
                            cohorts=None, per_cohort=1,
                            fps_cycle=(30.0, 60.0)) -> dict:
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for cohort in list(cohorts or COHORTS):
        for i in range(per_cohort):
            pid = f"rg{seed:02d}{n:03d}"
            rows.append(write_scan(
                d, f"r_{pid}", cohort, pid=pid, session_id=f"{pid}-s1",
                seed=seed + n, duration_s=duration_s,
                # skin-tone group, frame rate and site must not be
                # confounded (review finding: with fps = n % 2 and
                # group = n % 6, odd groups were always 30 fps / siteB)
                fps=fps_cycle[(n + n // 6) % len(fps_cycle)],
                site="siteA" if (n // 3) % 2 else "siteB",
                fitzpatrick=1 + (n % 6),
                start_utc=f"2026-09-01T{9 + n % 10:02d}:00:00Z"))
            n += 1
    manifest = {"scans": rows, "seed": seed, "duration_s": duration_s,
                "note": HONESTY_NOTE}
    (d / "regularity_dataset.json").write_text(json.dumps(manifest,
                                                          indent=1))
    return manifest


def make_dispersion_ladder(out_dir, *, seed=73, duration_s=45.0,
                           rungs=LADDER_RMSSD_MS, fps_list=(30.0, 60.0),
                           reps=1, fitz_groups=(2, 5)) -> dict:
    """True-RMSSD rungs at each frame rate: the noise-floor fixture.
    `reps` clips per (fps, rung, skin-tone group) — the MDI cell needs
    several scans of ONE group inside one dispersion bin, which a
    rotating group assignment never supplied (review finding). Two
    groups by default: a light and a dark tone."""
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for fps in fps_list:
        for rung in rungs:
            for fitz in fitz_groups:
                for k in range(reps):
                    pid = f"ld{seed:02d}{n:03d}"
                    rows.append(write_scan(
                        d, f"r_{pid}", None, pid=pid,
                        session_id=f"{pid}-s1", seed=seed + n,
                        duration_s=duration_s, fps=fps,
                        fitzpatrick=int(fitz), rmssd_ms=rung,
                        start_utc=f"2026-09-02T{9 + n % 10:02d}:00:00Z"))
                    n += 1
    manifest = {"scans": rows, "seed": seed, "rungs_ms": list(rungs),
                "fps": list(fps_list), "fitz_groups": list(fitz_groups),
                "note": HONESTY_NOTE}
    (d / "regularity_ladder.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_regularity"
    m = make_regularity_dataset(out)
    print(json.dumps({"scans": len(m["scans"]), "out": out,
                      "note": HONESTY_NOTE}, indent=1))
