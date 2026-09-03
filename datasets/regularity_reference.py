"""
The ECG-derived regularity reference (v0.7 Task 3) — computable ground
truth, published, no adjudication.

The reference label for "regular vs irregular" is derived
DETERMINISTICALLY from the reference ECG's own R-R series using the
IDENTICAL statistics the camera path applies to its intervals — the
same features/regularity.py code, the same clean-run discipline, the
same index, the same pre-registered threshold (configs/gates.yaml,
`regularity.reference_label`, REQUIRES_CLINICAL_SIGNOFF). That makes
this the one head that can be validated at scale on ANY paired
video+ECG data — healthy volunteers, public corpora, every session
collected for the other tracks — with no cardiologist in the loop for
the base label. There is no hidden ground truth: the definition below
IS the label.

    reference class = irregular  iff  index >= irregular_if_index_at_least
    index = median |successive R-R difference| / median R-R
            (within clean runs only; the ECG beats are trusted beat by
            beat, but the physiologic-range and missed/false-beat
            splitters apply exactly as they do to camera beats, so the
            two paths see the same run structure)
    indeterminate                iff  fewer than min_intervals clean
                                      intervals

What this label IS: the ECG's own verdict on interval regularity by the
product's statistic. What it is NOT: a rhythm diagnosis. An irregular
reference may be AF, ectopy, or a healthy person breathing; sorting
those is the benign-pattern problem (gate R2), not the label's job.
"""
from __future__ import annotations

import pathlib

import numpy as np

from beats.detector import Beat, BeatSeries
from beats.ibi import clean_runs
from configs import load_config, parse_yaml_subset
from features.regularity import regularity_from_runs

_REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_GATES = _REPO / "configs" / "gates.yaml"
CLASSES = ("regular", "irregular", "indeterminate")


def load_reference_definition(gates_path=None) -> dict:
    """The pre-registered definition, read from gates.yaml — never a
    coded default, so the label cannot drift from what is published."""
    doc = parse_yaml_subset(
        pathlib.Path(gates_path or DEFAULT_GATES).read_text())
    blk = (doc.get("regularity") or {}).get("reference_label")
    if not isinstance(blk, dict):
        raise ValueError("configs/gates.yaml regularity block has no "
                         "reference_label definition")
    for k in ("index", "irregular_if_index_at_least", "min_intervals",
              "ecg_beat_confidence"):
        if k not in blk:
            raise ValueError(f"reference_label is missing {k!r} — the "
                             "published definition must be explicit")
    if str(blk["index"]) != "irregularity_index":
        raise ValueError("only the irregularity_index reference is "
                         "implemented; the definition names "
                         f"{blk['index']!r}")
    return blk


def classify_index(index, n_intervals, definition: dict) -> str:
    """The class from the index — the ONE rule, used for camera and ECG
    alike."""
    if n_intervals is None or int(n_intervals) < int(
            definition["min_intervals"]):
        return "indeterminate"
    if index is None or not np.isfinite(float(index)):
        return "indeterminate"
    thr = float(definition["irregular_if_index_at_least"])
    return "irregular" if float(index) >= thr else "regular"


def resolve_min_conf(cfg=None) -> float:
    """The ONE resolution of configs/default.yaml `runs.min_conf` that
    every interval path shares — the pipeline, the ECG reference, the
    public surrogate and the beat-error study (review finding: two of
    them hard-coded 0.5)."""
    rc = (cfg or load_config())["runs"]
    return 0.5 if rc["min_conf"] == "CALIBRATED" else float(rc["min_conf"])


def runs_from_rpeaks(rpeaks_s, *, beat_confidence: float = 1.0,
                     fps: float = 500.0, config=None):
    """Clean runs from a reference R-peak series under the SAME run
    discipline the pipeline applies to camera beats (configs/default.yaml
    `runs`). The beats are trusted (confidence 1.0 by default), so the
    confidence channel is inert; the physiologic-range and missed/
    false-beat splitters still apply — identically."""
    cfg = config or load_config()
    rc = cfg["runs"]
    seq = [] if rpeaks_s is None else list(np.asarray(rpeaks_s).ravel())
    min_conf = resolve_min_conf(cfg)
    rp = np.asarray([float(x) for x in seq if x is not None], float)
    rp = np.sort(rp[np.isfinite(rp)])
    beats = [Beat(t_s=float(t), confidence=float(beat_confidence),
                  roi_agreement=1.0, signal_quality=1.0, amplitude=1.0,
                  prominence=1.0, source_rois=["ecg"]) for t in rp]
    duration = float(rp[-1] - rp[0]) if rp.size > 1 else 0.0
    series = BeatSeries(beats, float(fps), duration)
    lo_ms, hi_ms = rc["ibi_physiologic_ms"]
    return clean_runs(series, min_conf=min_conf,
                      min_run_beats=int(rc["min_run_beats"]),
                      max_physiologic_ibi_ms=float(hi_ms),
                      min_physiologic_ibi_ms=float(lo_ms),
                      missed_beat_ratio=float(rc["missed_ratio"]))


def reference_from_rpeaks(rpeaks_s, *, definition=None, gates_path=None,
                          config=None, fps: float = 500.0) -> dict:
    """The reference verdict for one recording: the representation
    computed from the ECG R-R series by the same code, its index + CI,
    and the class under the published definition."""
    d = definition or load_reference_definition(gates_path)
    rs = runs_from_rpeaks(rpeaks_s,
                          beat_confidence=float(d["ecg_beat_confidence"]),
                          fps=fps, config=config)
    reg = regularity_from_runs(rs.runs, rs.run_confidences,
                               rs.dropout_rate, run_times=rs.run_times,
                               fps=fps)
    idx = reg.index.get("value")
    return {"features": reg, "index": idx, "ci95": reg.index.get("ci95"),
            "n_intervals": int(reg.n_intervals),
            "n_runs": int(len(rs.runs)),
            "rmssd_ms": reg.values.get("rmssd"),
            "class": classify_index(idx, reg.n_intervals, d),
            "definition": {"index": d["index"],
                           "irregular_if_index_at_least":
                               float(d["irregular_if_index_at_least"]),
                           "min_intervals": int(d["min_intervals"]),
                           "ecg_beat_confidence":
                               float(d["ecg_beat_confidence"])}}
