"""
Beat-confidence calibration (T2).

The two-pass fusion detector produces confidences that are HONEST (monotone
in the probability the beat is real) but COMPRESSED — measured demo ECE 0.42
against the 0.10 gate. That breaks the production feature path, because
`clean_runs` interprets confidence as P(beat is real) when it thresholds at
0.5: with compressed scores a fixed 0.5 either abstains on everything or
excludes nothing.

The fix is a MONOTONE map fitted against ECG-matched labels (isotonic
regression via pool-adjacent-violators). Monotone matters: a non-monotone
recalibration could reorder beats, silently changing which beats survive the
run filter for reasons unrelated to detection quality.

Governance: the fitted map is a versioned artifact. Its version string is a
content hash of the fitted parameters, carried into
`ScanResult.calibration_version`, so any result can be traced to the exact
calibration that produced it. Fitting data must come from TRAIN/DEV only —
wire `datasets.splits.assert_preprocessing_is_split_safe` in upstream of any
fit on real recordings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import hashlib
import json

import numpy as np

from .detector import Beat, BeatSeries


def _pool_adjacent_violators(y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted PAV: returns per-point fitted values (non-decreasing)."""
    vals: list[float] = []
    wts: list[float] = []
    counts: list[int] = []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); counts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1] + 1e-15:
            wt = wts[-1] + wts[-2]
            v = (vals[-1] * wts[-1] + vals[-2] * wts[-2]) / wt
            vals.pop(); wts.pop()
            c = counts.pop()
            vals[-1] = v; wts[-1] = wt; counts[-1] += c
    out = np.concatenate([np.full(c, v) for v, c in zip(vals, counts)])
    return out, np.asarray(counts, int)


@dataclass
class Calibrator:
    """Monotone raw-confidence -> P(beat is real) map. Immutable in use."""
    x: np.ndarray                       # breakpoint raw confidences, ascending
    y: np.ndarray                       # calibrated P(real) at breakpoints
    n_fit_beats: int = 0
    n_fit_recordings: int = 0
    method: str = "isotonic-pav-v1"
    version: str = field(default="", repr=True)

    def __post_init__(self):
        self.x = np.asarray(self.x, float)
        self.y = np.asarray(self.y, float)
        if not self.version:
            payload = json.dumps({"m": self.method,
                                  "x": np.round(self.x, 10).tolist(),
                                  "y": np.round(self.y, 10).tolist()},
                                 sort_keys=True)
            self.version = "cal-" + hashlib.sha256(payload.encode()).hexdigest()[:12]

    # ---------------------------------------------------------------- apply
    def apply_values(self, confidences: Sequence[float]) -> np.ndarray:
        """Map raw confidences to calibrated P(real). NaN -> 0 (fail closed)."""
        c = np.asarray(confidences, float)
        if self.x.size == 0:
            out = np.zeros_like(c)
        else:
            out = np.interp(c, self.x, self.y,
                            left=float(self.y[0]), right=float(self.y[-1]))
        out = np.clip(out, 0.0, 1.0)
        out[~np.isfinite(c)] = 0.0
        return out

    def apply(self, series: BeatSeries) -> BeatSeries:
        """Return a NEW BeatSeries with calibrated confidences.

        Everything else — timing, amplitudes, ROI provenance — is copied
        verbatim; calibration must never alter what was detected, only how
        much it is trusted.
        """
        mapped = self.apply_values([b.confidence for b in series.beats])
        beats = [Beat(t_s=b.t_s, confidence=float(m),
                      roi_agreement=b.roi_agreement,
                      signal_quality=b.signal_quality, amplitude=b.amplitude,
                      prominence=b.prominence,
                      source_rois=list(b.source_rois),
                      interpolated=b.interpolated)
                 for b, m in zip(series.beats, mapped)]
        return BeatSeries(beats, series.fps, series.duration_s)

    # ------------------------------------------------------------ artefact
    def to_json(self) -> str:
        return json.dumps({
            "method": self.method, "version": self.version,
            "x": self.x.tolist(), "y": self.y.tolist(),
            "n_fit_beats": self.n_fit_beats,
            "n_fit_recordings": self.n_fit_recordings}, indent=2)

    @classmethod
    def from_json(cls, blob: str) -> "Calibrator":
        d = json.loads(blob)
        return cls(x=np.asarray(d["x"], float), y=np.asarray(d["y"], float),
                   n_fit_beats=int(d.get("n_fit_beats", 0)),
                   n_fit_recordings=int(d.get("n_fit_recordings", 0)),
                   method=d.get("method", "isotonic-pav-v1"),
                   version=d.get("version", ""))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        with open(path) as f:
            return cls.from_json(f.read())


def fit_confidence_calibration(matches: Sequence, confidences: Sequence[np.ndarray]
                               ) -> Calibrator:
    """Fit an isotonic confidence calibration from ECG-matched labels.

    Args:
        matches: BeatMatchResult per recording (from evaluation.beat_metrics.
            match_beats against the recording's R-peaks).
        confidences: the detector's raw per-beat confidences per recording,
            aligned with each match's detected-beat indexing.

    Labels are matched/unmatched — the only ground truth for "this detected
    beat is real" that exists. Raises ValueError on any length mismatch
    rather than silently misaligning labels.
    """
    if len(matches) != len(confidences):
        raise ValueError("matches and confidences must pair one-to-one")
    xs, ys = [], []
    for m, c in zip(matches, confidences):
        c = np.asarray(c, float)
        if c.size != m.n_detected:
            raise ValueError(
                f"confidence array of size {c.size} does not match "
                f"{m.n_detected} detected beats")
        lab = np.zeros(m.n_detected, float)
        lab[m.matched_det_idx] = 1.0
        keep = np.isfinite(c)
        xs.append(c[keep]); ys.append(lab[keep])
    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])
    if x.size < 10:
        raise ValueError(f"only {x.size} labelled beats; refusing to fit")

    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    fitted, counts = _pool_adjacent_violators(y, np.ones_like(y))

    # Collapse to block breakpoints: (mean x of block, fitted value).
    bx, by = [], []
    i = 0
    for c in counts:
        bx.append(float(np.mean(x[i:i + c])))
        by.append(float(fitted[i]))
        i += c
    return Calibrator(x=np.asarray(bx), y=np.asarray(by),
                      n_fit_beats=int(x.size), n_fit_recordings=len(matches))
