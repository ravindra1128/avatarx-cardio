"""
Synthetic atrial-flutter fixture generator (v0.6 flutter track).

The physiology this track inverts: AF is found because its pulse is
IRREGULAR; flutter conducts a fixed fraction of a ~250-300/min atrial
circuit and so produces a METRONOMIC pulse — 2:1 at ~150, 3:1 at ~100,
4:1 at ~75. Every fixture here is therefore built from an explicit
atrial rate and an explicit conduction ratio, so a fixture always knows
which of the two ventricular signatures it carries.

Two channels matter and both are generated:

  * the interval series — fixed-ratio flutter is metronomic, variable
    block is irregular (and overlaps AF's feature space by construction,
    so the confusion is measurable rather than hidden);
  * BREATHING — a grey torso bar (synth_video's `torso_respiration`
    hook). Sinus rhythms modulate their intervals at the SAME breathing
    frequency (respiratory sinus arrhythmia); flutter, SVT and paced
    rhythms do not, because fixed conduction decouples the ventricle
    from the respiratory drive. That coupling is the only thing that
    separates 2:1 flutter from sinus tachycardia at the same rate, since
    dispersion alone cannot (measured on these fixtures: a jitter-
    matched metronomic clip and true sinus tach both read RMSSD ~25 ms,
    while their tachogram respiratory-band fractions are 0.03 vs 0.91).

The hard negatives are first-class fixtures, not an afterthought: a
regular fast pulse is USUALLY benign, so specificity against sinus
tachycardia, SVT, beta-blocked sinus, paced rhythm and AF is the whole
product risk (spec §F, Task 3).

Nothing here is evidence about humans (HONESTY_NOTE travels with every
artifact); surrogate-domain runs can never open a gate.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datasets.schema import (CaptureConfig, ConductionRatio, FlutterType,
                             Recording, Rhythm, RhythmAnnotation, Split,
                             SyncMethod, SyncRecord)
from scripts.make_synth_vascular import HONESTY_NOTE
from scripts.make_synth_video import synth_video

DEFAULT_ATRIAL_BPM = 300.0
DEFAULT_RESP_BRPM = 15.0
RATIO_ENUM = {2: ConductionRatio.TWO_TO_ONE, 3: ConductionRatio.THREE_TO_ONE,
              4: ConductionRatio.FOUR_TO_ONE}


def _truncate(rr: list, duration_s: float) -> np.ndarray:
    rr = np.asarray(rr, float)
    return rr[np.cumsum(rr) <= duration_s]


def flutter_rr(duration_s: float, *, atrial_bpm: float = DEFAULT_ATRIAL_BPM,
               ratio: int = 2, jitter_ms: float = 3.0,
               seed: int = 0) -> np.ndarray:
    """Fixed-ratio flutter: every ventricular interval is `ratio` atrial
    cycles long. Real fixed-block conduction is not perfectly rigid, so a
    few ms of jitter rides on it — enough that a zero-variance detector
    would be cheating, small enough to stay far below sinus."""
    rng = np.random.default_rng(seed + 11)
    cycle = 60.0 / float(atrial_bpm)
    base = ratio * cycle
    n = int(duration_s / base) + 4
    return _truncate(base + rng.normal(0.0, jitter_ms / 1000.0, n),
                     duration_s)


def switching_flutter_rr(duration_s: float, *,
                         atrial_bpm: float = DEFAULT_ATRIAL_BPM,
                         ratios=(2, 4), jitter_ms: float = 3.0,
                         seed: int = 0) -> np.ndarray:
    """A conduction-ratio STEP inside one scan (2:1 -> 4:1): the same
    atrial circuit, a different divisor, so the pulse rate jumps by an
    integer factor with no change in regularity. Nothing else in
    cardiology does this."""
    rng = np.random.default_rng(seed + 23)
    cycle = 60.0 / float(atrial_bpm)
    span = duration_s / len(ratios)
    out: list = []
    t = 0.0
    for k, ratio in enumerate(ratios):
        # switch on the CLOCK, not on an interval count: each segment
        # occupies its own share of the scan, so the step is where the
        # fixture says it is
        end = (k + 1) * span
        base = ratio * cycle
        while t < end:
            rr = float(base + rng.normal(0.0, jitter_ms / 1000.0))
            out.append(rr)
            t += rr
    return _truncate(out, duration_s)


def variable_block_rr(duration_s: float, *,
                      atrial_bpm: float = DEFAULT_ATRIAL_BPM,
                      ratios=(2, 3, 4), p_stay: float = 0.6,
                      jitter_ms: float = 3.0, seed: int = 0) -> np.ndarray:
    """Variable block: the divisor changes beat to beat, so the pulse is
    IRREGULAR and lands squarely in AF's feature space. Included so the
    AF/flutter confusion is measured, not hidden — and so the F2 per-
    ratio table has a variable row to report."""
    rng = np.random.default_rng(seed + 37)
    cycle = 60.0 / float(atrial_bpm)
    ratios = list(ratios)
    k = int(rng.integers(len(ratios)))
    out, t = [], 0.0
    while t <= duration_s:
        if rng.random() > p_stay:
            k = int(rng.integers(len(ratios)))
        rr = ratios[k] * cycle + rng.normal(0.0, jitter_ms / 1000.0)
        out.append(rr)
        t += rr
    return _truncate(out, duration_s)


def sinus_rr(duration_s: float, *, bpm: float = 150.0,
             rsa_depth: float = 0.10,
             resp_brpm: float = DEFAULT_RESP_BRPM,
             noise_ms: float = 6.0, seed: int = 0) -> np.ndarray:
    """Sinus rhythm at any rate, WITH respiratory sinus arrhythmia at
    `resp_brpm`. Sinus rhythm retains some RSA at every rate — that is
    what the hyper-regularity family tests for, and its absence (not the
    rate) is what distinguishes fixed conduction."""
    rng = np.random.default_rng(seed + 53)
    f = float(resp_brpm) / 60.0
    base = 60.0 / float(bpm)
    out, t = [], 0.0
    while t <= duration_s:
        rr = base * (1.0 + rsa_depth * np.sin(2 * np.pi * f * t)) \
            + rng.normal(0.0, noise_ms / 1000.0)
        out.append(max(float(rr), 0.25))
        t += out[-1]
    return _truncate(out, duration_s)


def svt_rr(duration_s: float, *, bpm: float = 165.0,
           jitter_ms: float = 4.0, seed: int = 0) -> np.ndarray:
    """SVT / AVNRT: abrupt, regular, re-entrant. By PULSE it is
    indistinguishable from 2:1 flutter — deliberately so. The flag covers
    both by design and the sanctioned sentence names neither (F-a)."""
    rng = np.random.default_rng(seed + 67)
    base = 60.0 / float(bpm)
    n = int(duration_s / base) + 4
    return _truncate(base + rng.normal(0.0, jitter_ms / 1000.0, n),
                     duration_s)


def paced_rr(duration_s: float, *, bpm: float = 70.0,
             jitter_ms: float = 1.0, seed: int = 0) -> np.ndarray:
    """Pacemaker rhythm: metronomic by design, at a normal rate. The rate
    band is what keeps it out of the flag; it is reported separately."""
    rng = np.random.default_rng(seed + 71)
    base = 60.0 / float(bpm)
    n = int(duration_s / base) + 4
    return _truncate(base + rng.normal(0.0, jitter_ms / 1000.0, n),
                     duration_s)


def af_rr(duration_s: float, *, mean_bpm: float = 97.0,
          sd_ms: float = 170.0, seed: int = 0) -> np.ndarray:
    """AF: irregularly irregular. `sd_ms` low + a controlled rate gives
    the 'AF with regularized rate' confounder."""
    rng = np.random.default_rng(seed + 89)
    base = 60.0 / float(mean_bpm)
    n = int(duration_s / max(base - sd_ms / 1000.0, 0.28)) + 6
    rr = np.clip(rng.normal(base, sd_ms / 1000.0, n), 0.28, 1.35)
    return _truncate(rr, duration_s)


# --------------------------------------------------------------- cohorts
# Each cohort row: the RR builder, the ECG-truth label, and whether the
# INTERVALS are respiratory-coupled. `flag_expected` records what an
# honest §F flag should do — including the permanent misses (F-c).
COHORTS = {
    "flutter_2to1": dict(
        rhythm=Rhythm.ATRIAL_FLUTTER, ratio=2, coupled=False,
        flag_expected=True,
        build=lambda d, s, b: flutter_rr(d, ratio=2, seed=s)),
    "flutter_3to1": dict(
        rhythm=Rhythm.ATRIAL_FLUTTER, ratio=3, coupled=False,
        flag_expected=False,      # ~100 bpm: below the 2:1 band, reported
        build=lambda d, s, b: flutter_rr(d, ratio=3, seed=s)),
    "flutter_4to1": dict(
        rhythm=Rhythm.ATRIAL_FLUTTER, ratio=4, coupled=False,
        flag_expected=False,      # ~75 bpm: THE known miss (F-c)
        build=lambda d, s, b: flutter_rr(d, ratio=4, seed=s)),
    "flutter_variable": dict(
        rhythm=Rhythm.ATRIAL_FLUTTER, ratio="VARIABLE", coupled=False,
        flag_expected=False,      # irregular: AF's feature space
        build=lambda d, s, b: variable_block_rr(d, seed=s)),
    "flutter_switch": dict(
        rhythm=Rhythm.ATRIAL_FLUTTER, ratio=2, coupled=False,
        flag_expected=False,      # a mid-scan step is not a sustained rate
        build=lambda d, s, b: switching_flutter_rr(d, seed=s)),
    "sinus_tach": dict(
        rhythm=Rhythm.SINUS_TACHYCARDIA, ratio=None, coupled=True,
        flag_expected=False,      # THE dominant negative
        build=lambda d, s, b: sinus_rr(d, bpm=150.0, resp_brpm=b, seed=s)),
    "sinus_tach_shallow_rsa": dict(
        rhythm=Rhythm.SINUS_TACHYCARDIA, ratio=None, coupled=True,
        flag_expected=False,
        build=lambda d, s, b: sinus_rr(d, bpm=152.0, rsa_depth=0.05,
                                       resp_brpm=b, seed=s)),
    "svt": dict(
        rhythm=Rhythm.SVT, ratio=None, coupled=False,
        flag_expected=True,       # covered by design — the ECG names it
        # 158, not 165: at the band's upper EDGE half the intervals fall
        # outside it and the fixture would contradict its own
        # flag_expected (review finding)
        build=lambda d, s, b: svt_rr(d, bpm=158.0, seed=s)),
    "beta_blocked_sinus": dict(
        rhythm=Rhythm.SINUS, ratio=None, coupled=True,
        flag_expected=False,      # hyper-regular but normal rate
        build=lambda d, s, b: sinus_rr(d, bpm=62.0, rsa_depth=0.02,
                                       noise_ms=4.0, resp_brpm=b,
                                       seed=s)),
    "paced": dict(
        rhythm=Rhythm.PACED, ratio=None, coupled=False,
        flag_expected=False,      # metronomic by design, normal rate
        build=lambda d, s, b: paced_rr(d, seed=s)),
    "af": dict(
        rhythm=Rhythm.AFIB, ratio=None, coupled=False, flag_expected=False,
        build=lambda d, s, b: af_rr(d, seed=s)),
    "af_regularized": dict(
        rhythm=Rhythm.AFIB, ratio=None, coupled=False, flag_expected=False,
        build=lambda d, s, b: af_rr(d, mean_bpm=140.0, sd_ms=60.0, seed=s)),
    "sinus": dict(
        rhythm=Rhythm.SINUS, ratio=None, coupled=True, flag_expected=False,
        build=lambda d, s, b: sinus_rr(d, bpm=72.0, resp_brpm=b, seed=s)),
}


def _annotation(cohort: str, duration_s: float, rr: np.ndarray,
                *, atrial_bpm: float) -> RhythmAnnotation:
    c = COHORTS[cohort]
    vent = float(60.0 / np.mean(rr)) if rr.size else None
    ratio = c["ratio"]
    is_flutter = c["rhythm"] is Rhythm.ATRIAL_FLUTTER
    return RhythmAnnotation(
        0.0, duration_s, c["rhythm"], annotator_id="synth",
        adjudicated=True, ventricular_rate_bpm=vent,
        # a flutter POSITIVE carries the full naming burden (12-lead, EP
        # reader, atrial rate, ratio, type); everything else leaves the
        # v0.6 block empty, which is what a non-flutter label means
        atrial_rate_bpm=(atrial_bpm if is_flutter else None),
        conduction_ratio=((ConductionRatio.VARIABLE
                           if ratio == "VARIABLE" else RATIO_ENUM[ratio])
                          if is_flutter else None),
        flutter_type=(FlutterType.TYPICAL if is_flutter else None),
        adjudicator_id=("synth-ep" if is_flutter else None),
        adjudication_leads=(12 if is_flutter else None))


def _recording_manifest(rid, pid, session_id, site, fps, duration_s,
                        truth, vname, ename, ann, *, start_utc,
                        ecg_leads: int) -> Recording:
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
    return Recording(
        recording_id=rid, participant_id=pid, session_id=session_id,
        site_id=site, video_path=vname, ecg_path=ename,
        video_start_utc=start_utc,
        video_end_utc=start_utc.replace(":00Z", ":50Z"),
        ecg_start_utc=start_utc,
        ecg_end_utc=start_utc.replace(":00Z", ":50Z"),
        duration_s=duration_s, capture=cap, sync=sync,
        ecg_sampling_hz=500.0, ecg_leads=ecg_leads,
        ecg_device="synthetic-12lead" if ecg_leads >= 12 else "synthetic",
        rhythm_annotations=[ann], ecg_rpeaks_s=truth["rpeaks_s"],
        protocol_id="flutter_rest", lighting_condition="controlled",
        motion_condition="stationary", operator="technician",
        split=Split.UNASSIGNED, dataset_version="synth-flutter-v0.1")


def write_scan(out_dir, rid: str, cohort: str, *, pid: str,
               session_id: str, seed: int, duration_s: float = 40.0,
               fps: float = 30.0, site: str = "siteA",
               fitzpatrick: int = 3, resp_brpm: float = DEFAULT_RESP_BRPM,
               atrial_bpm: float = DEFAULT_ATRIAL_BPM,
               breathing: bool = True,
               start_utc: str = "2026-09-01T09:00:00Z") -> dict:
    """One labeled scan of `cohort`: video (+ breathing bar), ECG R-peak
    sidecar, Recording manifest with the v0.6 rhythm annotation."""
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    c = COHORTS[cohort]
    stable = int(hashlib.sha256(rid.encode()).hexdigest()[:6], 16)
    rr = c["build"](duration_s, seed + stable % 997, resp_brpm)
    truth = synth_video(
        str(d / f"{rid}.avi"), kind=f"flutter_track_{cohort}", fps=fps,
        duration_s=duration_s, seed=seed * 100 + stable % 991,
        rr_override=rr,
        torso_respiration=({"brpm": resp_brpm, "bob_px": 6.0}
                           if breathing else None))
    (d / f"{rid}.ecg.json").write_text(json.dumps(
        {"rpeaks_s": truth["rpeaks_s"], "source": "synthetic"}))
    ann = _annotation(cohort, duration_s, np.asarray(truth["rr_s"], float),
                      atrial_bpm=atrial_bpm)
    rec = _recording_manifest(
        rid, pid, session_id, site, fps, duration_s, truth,
        f"{rid}.avi", f"{rid}.ecg.json", ann, start_utc=start_utc,
        ecg_leads=12 if c["rhythm"] is Rhythm.ATRIAL_FLUTTER else 1)
    (d / f"{rid}.recording.json").write_text(rec.to_json())
    # participant-level metadata is Participant data, not Recording
    # data, so it rides in its own sidecar — without it the fairness
    # gate (F4) has no subgroup to rate and reads "unrated" on every
    # generated cohort
    (d / f"{rid}.participant.json").write_text(json.dumps(
        {"participant_id": pid, "fitzpatrick_group": fitzpatrick,
         "age_years": 40 + (stable % 45), "sex": "F" if stable % 2
         else "M"}))
    return {"recording_id": rid, "participant_id": pid, "cohort": cohort,
            "session_id": session_id, "fitzpatrick_group": fitzpatrick,
            "resp_brpm": resp_brpm if breathing else None,
            "rhythm": c["rhythm"].value,
            "flag_expected": c["flag_expected"],
            "median_bpm": round(float(60.0 / np.median(truth["rr_s"])), 1),
            "site_id": site, "video": f"{rid}.avi",
            "start_utc": start_utc}


def make_flutter_dataset(out_dir, *, seed: int = 61,
                         duration_s: float = 40.0, fps: float = 30.0,
                         cohorts=None, serial_participants: int = 2) -> dict:
    """A small end-to-end fixture cohort: one scan per cohort class plus
    the SERIAL sub-protocol — >= 3 scans per participant across a
    rate-varying window, where a flutter participant steps 2:1 -> 3:1 ->
    4:1 on ONE atrial rate and the control participant's rates wander
    without any common divisor."""
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    rows, series = [], []
    names = list(cohorts or COHORTS)
    for i, cohort in enumerate(names):
        pid = f"fl{seed:02d}{i:02d}"
        rows.append(write_scan(
            d, f"r_{pid}_{cohort}", cohort, pid=pid,
            session_id=f"{pid}-s1", seed=seed + i, duration_s=duration_s,
            fps=fps, site="siteA" if i % 2 == 0 else "siteB",
            fitzpatrick=2 + (i % 5),
            start_utc=f"2026-09-01T{9 + i % 8:02d}:00:00Z"))
    # serial sub-protocol: the track's differentiator needs >= 3 scans
    for k in range(serial_participants):
        flut = (k == 0)
        pid = f"fls{seed:02d}{k:02d}"
        steps = ["flutter_2to1", "flutter_3to1", "flutter_4to1"] if flut \
            else ["sinus_tach", "sinus", "beta_blocked_sinus"]
        ids = []
        for j, cohort in enumerate(steps):
            rid = f"r_{pid}_v{j + 1}"
            rows.append(write_scan(
                d, rid, cohort, pid=pid, session_id=f"{pid}-s{j + 1}",
                seed=seed + 300 + 7 * k + j, duration_s=duration_s,
                fps=fps, fitzpatrick=3 + k,
                start_utc=f"2026-09-0{2 + j}T09:00:00Z"))
            ids.append(rid)
        series.append({"participant_id": pid, "recordings": ids,
                       "flutter_series": flut,
                       "atrial_bpm_true": DEFAULT_ATRIAL_BPM if flut
                       else None})
    manifest = {"scans": rows, "series": series, "seed": seed,
                "fps": fps, "duration_s": duration_s,
                "atrial_bpm": DEFAULT_ATRIAL_BPM,
                "resp_brpm": DEFAULT_RESP_BRPM, "note": HONESTY_NOTE}
    (d / "flutter_dataset.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_flutter"
    m = make_flutter_dataset(out)
    print(json.dumps({"scans": len(m["scans"]), "series": len(m["series"]),
                      "out": out, "note": HONESTY_NOTE}, indent=1))
