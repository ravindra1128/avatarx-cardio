"""
Paired-ECG reference ingestion (v0.2 M2.8, L6) — invariant 12: reference
ECG flows ONE WAY, into training/evaluation. Nothing written here is ever
read by app/ or echoed into a consumer artifact.

A session directory contains:
  ecg_export.csv        two columns t_s,mv (header optional)  — OR —
  ecg.hea + ecg.dat     minimal WFDB (format 16, single- or multi-signal;
                        channel 0 is used)
  sync_events.json      {"video_events_s": [...], "ecg_events_s": [...]}
                        — the PRBS/LED marker events on each clock, from
                        the capture rig
  labels.json           episode-locked rhythm labels (list of rows):
                        {t_start_s, t_end_s (ECG clock), rhythm,
                         annotator_initials, date,
                         esc_definition_confirmed,
                         morphology?: {conduction_pattern,
                                       pr_ms, qrs_ms, qt_ms}}
                        — the optional morphology block (v0.3 T1) carries
                        the adjudicated conduction pattern and measured
                        intervals per segment; it is the reconstruction
                        track's G2/G3 ground truth and is validated hard
                        when present.

`ingest_reference()` verifies the marker events OVERLAP the ECG record,
fits offset + drift with honest uncertainty (datasets/synchronization),
FAILS CLOSED when uncertainty exceeds the spec threshold (config
sync.max_uncertainty_ms) or marker count is below the floor, detects
R-peaks in-house, maps them onto the video clock via the sanctioned
full-model mapping, validates every label row (rhythm vocabulary,
positive in-range span, adjudicator fields, ESC-definition checkbox) and
writes `reference.json` next to the inputs.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from configs import load_config
from datasets.schema import Rhythm
from datasets.synchronization import estimate_sync, ecg_to_video_clock

REQUIRED_LABEL_FIELDS = ("t_start_s", "t_end_s", "rhythm",
                         "annotator_initials", "date",
                         "esc_definition_confirmed")

# v0.3 T1 — adjudicated MORPHOLOGY per episode segment: the reconstruction
# track's ground truth (gates G2/G3 in configs/gates.yaml). Optional per
# row (rhythm-only campaigns stay valid) but validated hard when present.
CONDUCTION_PATTERNS = ("normal", "lbbb", "rbbb", "ivcd",
                       "preexcitation", "paced", "unknown")
_MORPH_INTERVAL_RANGES_MS = {"pr_ms": (40.0, 400.0),
                             "qrs_ms": (40.0, 250.0),
                             "qt_ms": (200.0, 700.0)}


def _validate_morphology(i: int, m: dict) -> dict:
    if not isinstance(m, dict):
        raise ReferenceError(f"label row {i}: morphology must be a mapping")
    unknown = set(m) - {"conduction_pattern"} - set(_MORPH_INTERVAL_RANGES_MS)
    if unknown:
        raise ReferenceError(f"label row {i}: unknown morphology fields "
                             f"{sorted(unknown)}")
    pat = m.get("conduction_pattern")
    if pat not in CONDUCTION_PATTERNS:
        raise ReferenceError(
            f"label row {i}: morphology.conduction_pattern {pat!r} not in "
            f"{CONDUCTION_PATTERNS}")
    out = {"conduction_pattern": str(pat)}
    for k, (lo, hi) in _MORPH_INTERVAL_RANGES_MS.items():
        v = m.get(k)
        if v is None:
            out[k] = None                 # legitimately unmeasurable (AF: PR)
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ReferenceError(f"label row {i}: morphology.{k} must be a "
                                 f"number in ms or null") from None
        if not (lo <= v <= hi):
            raise ReferenceError(f"label row {i}: morphology.{k} {v:.0f} ms "
                                 f"outside the plausible range "
                                 f"[{lo:.0f}, {hi:.0f}]")
        out[k] = v
    return out


class ReferenceError(ValueError):
    pass


# ------------------------------------------------------------- ECG readers
def read_ecg_csv(path) -> tuple:
    """(t_s, mv) from a two-column CSV; header line tolerated."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(";", ",").split(",")
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                if not rows:
                    continue                       # header
                raise ReferenceError(f"unparseable ECG CSV line: {line!r}")
    if len(rows) < 100:
        raise ReferenceError("ECG CSV too short to be a reference record")
    arr = np.asarray(rows, float)
    t, mv = arr[:, 0], arr[:, 1]
    if np.any(np.diff(t) <= 0):
        raise ReferenceError("ECG CSV timestamps must be strictly increasing")
    return t, mv


def read_wfdb(hea_path) -> tuple:
    """Minimal WFDB reader: format 16 (little-endian int16), channel 0.
    Enough for reference exports; anything else fails closed with the
    reason."""
    hea_path = pathlib.Path(hea_path)
    lines = [ln.strip() for ln in hea_path.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    head = lines[0].split()
    name, nsig, fs = head[0], int(head[1]), float(head[2])
    nsamp = int(head[3]) if len(head) > 3 else None
    sig = lines[1].split()
    fname, fmt = sig[0], sig[1]
    if fmt.split("+")[0] != "16":
        raise ReferenceError(f"WFDB format {fmt!r} unsupported (reader "
                             "handles format 16 only)")
    gain = float(sig[2].split("/")[0].split("(")[0]) if len(sig) > 2 else 200.0
    if gain == 0:
        gain = 200.0
    raw = np.fromfile(hea_path.parent / fname, dtype="<i2")
    if nsig > 1:
        raw = raw[: (raw.size // nsig) * nsig].reshape(-1, nsig)[:, 0]
    if nsamp:
        raw = raw[:nsamp]
    if raw.size < 100:
        raise ReferenceError("WFDB record too short")
    t = np.arange(raw.size) / fs
    return t, raw.astype(float) / gain


# ------------------------------------------------------------- R peaks
def detect_rpeaks(t_s: np.ndarray, mv: np.ndarray) -> np.ndarray:
    """In-house R-peak detection (Pan-Tompkins-style): 5-25 Hz bandpass,
    squared derivative, moving-window integration, adaptive threshold with
    a 250 ms refractory, then refinement to the local |signal| maximum."""
    t_s = np.asarray(t_s, float)
    mv = np.asarray(mv, float)
    fs = 1.0 / float(np.median(np.diff(t_s)))
    ny = fs / 2.0
    hi = min(25.0, 0.9 * ny)
    b, a = butter(2, [5.0 / ny, hi / ny], btype="band")
    x = filtfilt(b, a, mv - np.mean(mv))
    d = np.gradient(x) * fs
    e = d * d
    win = max(int(0.12 * fs), 1)
    integ = np.convolve(e, np.ones(win) / win, mode="same")
    thr = 0.25 * float(np.percentile(integ, 99))
    idx, _ = find_peaks(integ, height=thr, distance=int(0.25 * fs))
    refine = max(int(0.06 * fs), 1)
    out = []
    for i in idx:
        a0, b0 = max(0, i - refine), min(mv.size, i + refine + 1)
        j = a0 + int(np.argmax(np.abs(x[a0:b0])))
        out.append(t_s[j])
    return np.asarray(sorted(set(out)), float)


# ------------------------------------------------------------- labels
def validate_labels(rows: list, t_max_s: float) -> list:
    if not isinstance(rows, list) or not rows:
        raise ReferenceError("labels.json must be a non-empty list of "
                             "episode rows")
    out = []
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED_LABEL_FIELDS if k not in r]
        if missing:
            raise ReferenceError(f"label row {i}: missing {missing}")
        try:
            Rhythm(r["rhythm"])
        except ValueError:
            raise ReferenceError(f"label row {i}: unknown rhythm "
                                 f"{r['rhythm']!r}") from None
        a, b = float(r["t_start_s"]), float(r["t_end_s"])
        if not (0.0 <= a < b <= t_max_s + 1.0):
            raise ReferenceError(f"label row {i}: span [{a}, {b}] outside "
                                 f"the ECG record (0..{t_max_s:.1f} s)")
        if not r["esc_definition_confirmed"]:
            raise ReferenceError(f"label row {i}: ESC-definition checkbox "
                                 "not confirmed — episode labels must "
                                 "follow the guideline definition")
        if not str(r["annotator_initials"]).strip() or \
                not str(r["date"]).strip():
            raise ReferenceError(f"label row {i}: adjudicator fields empty")
        out.append({"t_start_s": a, "t_end_s": b,
                    "rhythm": str(Rhythm(r["rhythm"]).value),
                    "annotator_initials": str(r["annotator_initials"]),
                    "date": str(r["date"]),
                    "esc_definition_confirmed": True,
                    "adjudication_note": r.get("adjudication_note"),
                    "morphology": (_validate_morphology(i, r["morphology"])
                                   if r.get("morphology") is not None
                                   else None)})
    return out


# ------------------------------------------------------------- ingestion
def ingest_reference(session_dir, config: Optional[dict] = None) -> dict:
    sd = pathlib.Path(session_dir)
    cfg = config or load_config()
    max_unc = float(cfg["sync"]["max_uncertainty_ms"])
    min_ev = int(cfg["sync"]["min_marker_events"])

    csv = sd / "ecg_export.csv"
    hea = sd / "ecg.hea"
    if csv.exists():
        t, mv = read_ecg_csv(csv)
        src = csv
    elif hea.exists():
        t, mv = read_wfdb(hea)
        src = hea
    else:
        raise ReferenceError(f"no ECG export in {sd} (ecg_export.csv or "
                             "ecg.hea + ecg.dat)")

    sync_path = sd / "sync_events.json"
    if not sync_path.exists():
        raise ReferenceError("sync_events.json missing — a reference "
                             "without verified sync is unusable for "
                             "beat-level work")
    with open(sync_path) as f:
        se = json.load(f)
    v_ev = np.asarray(se.get("video_events_s") or [], float)
    e_ev = np.asarray(se.get("ecg_events_s") or [], float)
    if e_ev.size and (np.min(e_ev) < float(t[0]) - 1.0 or
                      np.max(e_ev) > float(t[-1]) + 1.0):
        raise ReferenceError(
            "sync markers do not overlap the ECG record "
            f"(markers {np.min(e_ev):.1f}..{np.max(e_ev):.1f} s vs ECG "
            f"{t[0]:.1f}..{t[-1]:.1f} s)")
    sync = estimate_sync(v_ev, e_ev,
                         recording_duration_s=float(t[-1] - t[0]))
    if sync.n_marker_events < min_ev:
        raise ReferenceError(
            f"only {sync.n_marker_events} paired marker events "
            f"(< {min_ev}): sync unverifiable — fail closed")
    if not np.isfinite(sync.sync_uncertainty_ms) or \
            sync.sync_uncertainty_ms > max_unc:
        raise ReferenceError(
            f"sync uncertainty {sync.sync_uncertainty_ms:.2f} ms exceeds "
            f"the {max_unc:.0f} ms spec threshold — fail closed")

    labels_path = sd / "labels.json"
    if not labels_path.exists():
        raise ReferenceError("labels.json missing — episode-locked rhythm "
                             "labels are what the reference is FOR")
    with open(labels_path) as f:
        labels = validate_labels(json.load(f), float(t[-1]))

    rpeaks = detect_rpeaks(t, mv)
    ref = {
        "reference_version": "reference-v1",
        "source_file": src.name,
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "ecg": {"fs_hz": round(1.0 / float(np.median(np.diff(t))), 3),
                "n_samples": int(mv.size),
                "t0_s": float(t[0]), "t1_s": float(t[-1])},
        "sync": {"offset_ms": float(sync.offset_ms),
                 "drift_ppm": (float(sync.drift_ppm)
                               if sync.drift_ppm is not None else None),
                 "uncertainty_ms": float(sync.sync_uncertainty_ms),
                 "n_marker_events": int(sync.n_marker_events)},
        "rpeaks_s_ecg": [round(float(x), 4) for x in rpeaks],
        "rpeaks_s_video": [round(float(x), 4)
                           for x in ecg_to_video_clock(rpeaks, sync)],
        "labels": labels,
        "flow": "training/evaluation only — never a consumer artifact "
                "(invariant 12)",
    }
    # v0.5 vasotone: optional pulse-oximeter perfusion-index trace
    # (pi_export.csv, ECG clock) — the W0 reference arm, mapped onto the
    # video clock with the SAME sanctioned sync model as the R-peaks
    pi_csv = sd / "pi_export.csv"
    if pi_csv.exists():
        pt, pv = read_pi_csv(pi_csv.read_text())
        pt_video = ecg_to_video_clock(np.asarray(pt, float), sync)
        ref["perfusion_index"] = {
            "t_s_video": [round(float(x), 3) for x in pt_video],
            "pi_percent": [round(float(x), 4) for x in pv],
            "n": len(pv), "source_file": pi_csv.name}
        # bridge to the vasotone harness's PiTrace shape (uniform 1 Hz
        # on the video clock, fail closed on coverage gaps) — without
        # this the ingested trace had no sanctioned consumer (v0.5
        # review finding)
        tv = np.asarray(pt_video, float)
        if tv.size >= 2 and float(np.max(np.diff(tv))) <= 2.0:
            grid = np.arange(float(tv[0]), float(tv[-1]), 1.0)
            if grid.size >= 30:
                vals = np.interp(grid, tv, np.asarray(pv, float))
                with open(sd / "pi_trace.json", "w") as f:
                    json.dump({"fs_hz": 1.0,
                               "values": [round(float(v), 4)
                                          for v in vals],
                               "t0_video_s": round(float(grid[0]), 3),
                               "device": "oximeter-export"}, f)
    with open(sd / "reference.json", "w") as f:
        json.dump(ref, f, indent=1)
    return ref


# --------------------------------------------------------------------------
# v0.4 vascular track — tonometry export parsing (SphygmoCor/Complior class)
# --------------------------------------------------------------------------

# Key aliases seen across SphygmoCor/Complior-class CSV exports. This
# parser handles the documented generic shape (one "key<sep>value" row per
# line, comma or semicolon separated); real device exports vary by
# firmware — extend the alias tables against real export fixtures, never
# by guessing.
_PWV_KEYS = ("pwv (m/s)", "cfpwv (m/s)", "carotid-femoral pwv", "cf-pwv",
             "pwv")
_FIELD_ALIASES = {
    "device": "device_model", "device model": "device_model",
    "operator": "operator",
    "sp (mmhg)": "brachial_sbp_mmhg", "sbp (mmhg)": "brachial_sbp_mmhg",
    "systolic (mmhg)": "brachial_sbp_mmhg",
    "dp (mmhg)": "brachial_dbp_mmhg", "dbp (mmhg)": "brachial_dbp_mmhg",
    "diastolic (mmhg)": "brachial_dbp_mmhg",
    "hr (bpm)": "hr_at_measurement_bpm", "hr": "hr_at_measurement_bpm",
    "cavi": "cavi", "date": "test_date", "test date": "test_date",
}


def parse_pwv_export(text: str) -> dict:
    """Parse a SphygmoCor/Complior-class key-value export into a dict for
    datasets.schema.vascular_from_dict. Fail-closed: no recognisable
    cfPWV row or no device row -> ValueError. Unrecognised rows are
    ignored (device exports carry dozens of derived indices this track
    deliberately does not ingest)."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        sep = ";" if (";" in line and "," not in line.split(";")[0]) else ","
        if sep not in line:
            continue
        key, _, val = line.partition(sep)
        key, val = key.strip().lower(), val.strip()
        if not val:
            continue
        if key in _PWV_KEYS:
            num = val.lower().replace("m/s", "").strip()
            if "cfpwv_mps" not in out:
                out["cfpwv_mps"] = float(num)
            elif "cfpwv_mps_repeat" not in out:
                # the retest sub-protocol's second read — never dropped
                # silently (review finding)
                out["cfpwv_mps_repeat"] = float(num)
            else:
                raise ValueError("tonometry export: more than two cfPWV "
                                 "rows — ambiguous, fail closed")
        elif key in _FIELD_ALIASES:
            field = _FIELD_ALIASES[key]
            if field in ("device_model", "operator", "test_date"):
                out[field] = val
            else:
                out[field] = float(val.split()[0])
    if "cfpwv_mps" not in out:
        raise ValueError("tonometry export: no cfPWV row recognised — "
                         "fail closed (extend the alias table against a "
                         "real export fixture, do not guess)")
    if "device_model" not in out:
        raise ValueError("tonometry export: no device row — an unnamed "
                         "reference device cannot enter the evidence")
    if "operator" not in out:
        raise ValueError("tonometry export: no operator row — an "
                         "unattributed tonometry read cannot enter the "
                         "evidence")
    return out


# --------------------------------------------------------------------------
# v0.5 vasotone track — perfusion-index reference trace ingestion
# --------------------------------------------------------------------------

def read_pi_csv(text: str):
    """Parse a pulse-oximeter perfusion-index export: strict two-column
    CSV `t_s,pi_percent` with strictly increasing time and positive PI.
    Real oximeter exports vary by vendor — extend against real fixtures,
    never by guessing (same posture as parse_pwv_export)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or lines[0].lower().replace(" ", "") != "t_s,pi_percent":
        raise ValueError("pi export: first line must be the header "
                         "'t_s,pi_percent' — fail closed on unknown "
                         "layouts")
    t, v = [], []
    for ln in lines[1:]:
        a, _, b = ln.partition(",")
        ti, vi = float(a), float(b)
        if vi <= 0:
            raise ValueError(f"pi export: non-positive PI value {vi}")
        if t and ti <= t[-1]:
            raise ValueError("pi export: time must be strictly "
                             "increasing")
        t.append(ti)
        v.append(vi)
    if len(t) < 2:
        raise ValueError("pi export: fewer than 2 samples")
    return t, v
