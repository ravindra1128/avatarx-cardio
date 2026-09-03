"""
BeatLattice (v0.2, L3) — the versioned, formal interface between the
signal/beat layers and the endpoint heads (invariant 14: every head
consumes this; nothing may bypass L2/L3).

The lattice is a read-only snapshot of what the gated pipeline measured:
fused beats with their confidences, the clean runs (IBI ms) with run
confidences, detection-burden statistics, per-ROI beat trains and the
capture segments. It carries a schema version so stored lattices remain
readable as the platform grows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

LATTICE_VERSION = "beat-lattice-v1"


@dataclass(frozen=True)
class BeatLattice:
    version: str
    fps: float
    duration_s: float
    beat_t_s: np.ndarray            # fused, calibrated beat times
    beat_confidence: np.ndarray
    beat_agreement: np.ndarray      # cross-ROI membership per beat
    runs: list                      # list of clean-run IBI arrays (ms)
    run_confidences: list
    n_intervals: int
    dropout_rate: float
    split_fraction: float
    per_roi_times: dict             # roi -> beat-time array (capture clock)
    segments: list = field(default_factory=list)   # [(t0_s, t1_s)]
    # v0.6, ADDITIVE (the version string is unchanged deliberately: a
    # bump would make every stored v1 lattice unreadable by from_dict,
    # while an optional field with an empty default leaves them valid).
    # Each interval's end-beat time, parallel to `runs` — without it an
    # interval cannot be placed on the clock, and the flutter track's
    # respiratory-coupling family is undefined.
    run_times: list = field(default_factory=list)

    @classmethod
    def from_pipeline(cls, series, runset, per_roi_beats: dict, fps: float,
                      duration_s: float, segments: Optional[list] = None
                      ) -> "BeatLattice":
        beats = series.beats if series is not None else []
        return cls(
            version=LATTICE_VERSION,
            fps=float(fps), duration_s=float(duration_s),
            beat_t_s=np.array([b.t_s for b in beats], float),
            beat_confidence=np.array([b.confidence for b in beats], float),
            beat_agreement=np.array([b.roi_agreement for b in beats], float),
            runs=[np.asarray(r, float) for r in runset.runs],
            run_confidences=[np.asarray(c, float)
                             for c in runset.run_confidences],
            n_intervals=int(runset.n_intervals),
            dropout_rate=float(runset.dropout_rate),
            split_fraction=float(getattr(runset, "split_fraction", 0.0)),
            per_roi_times={r: np.array([b.t_s for b in bs], float)
                           for r, bs in (per_roi_beats or {}).items()},
            segments=[(float(a), float(b)) for a, b in (segments or [])],
            run_times=[np.asarray(t, float)
                       for t in getattr(runset, "run_times", [])],
        )

    def to_dict(self) -> dict:
        """JSON-safe, versioned serialisation."""
        return {
            "version": self.version, "fps": self.fps,
            "duration_s": self.duration_s,
            "beat_t_s": self.beat_t_s.tolist(),
            "beat_confidence": self.beat_confidence.tolist(),
            "beat_agreement": self.beat_agreement.tolist(),
            "runs": [r.tolist() for r in self.runs],
            "run_confidences": [np.asarray(c, float).tolist()
                                for c in self.run_confidences],
            "n_intervals": self.n_intervals,
            "dropout_rate": self.dropout_rate,
            "split_fraction": self.split_fraction,
            "per_roi_times": {k: v.tolist()
                              for k, v in self.per_roi_times.items()},
            "segments": [list(s) for s in self.segments],
            "run_times": [np.asarray(t, float).tolist()
                          for t in self.run_times],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeatLattice":
        if d.get("version") != LATTICE_VERSION:
            raise ValueError(f"unsupported lattice version {d.get('version')!r}"
                             f" (this build reads {LATTICE_VERSION})")
        return cls(
            version=d["version"], fps=float(d["fps"]),
            duration_s=float(d["duration_s"]),
            beat_t_s=np.asarray(d["beat_t_s"], float),
            beat_confidence=np.asarray(d["beat_confidence"], float),
            beat_agreement=np.asarray(d["beat_agreement"], float),
            runs=[np.asarray(r, float) for r in d["runs"]],
            run_confidences=[np.asarray(c, float)
                             for c in d["run_confidences"]],
            n_intervals=int(d["n_intervals"]),
            dropout_rate=float(d["dropout_rate"]),
            split_fraction=float(d["split_fraction"]),
            per_roi_times={k: np.asarray(v, float)
                           for k, v in d["per_roi_times"].items()},
            segments=[tuple(s) for s in d["segments"]],
            # absent in stored v1 lattices — an empty list is the honest
            # reading, and consumers must treat it as "no times", never
            # as "times at zero"
            run_times=[np.asarray(t, float)
                       for t in (d.get("run_times") or [])],
        )
