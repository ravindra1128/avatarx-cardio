"""
Rhythm feature engine.

Each feature carries a TRANSFERABILITY rating recording how well it survives
the move from ECG/contact-PPG to remote facial pulse:

  HIGH   depends only on beat TIMING -> survives
  MED    depends on relative pulse AMPLITUDE -> partially survives; rPPG
         amplitude is uncalibrated but its beat-to-beat VARIATION is real
  LOW    depends on sub-cycle waveform MORPHOLOGY -> does not survive

The LOW group is not a matter of better engineering. The dicrotic notch
occupies roughly 50-100 ms; at 30 fps that is 1-3 samples, at or below the
joint temporal-resolution and SNR floor. Every feature built on it --
augmentation index, stiffness index, second-derivative wave features,
pulse rise/fall time -- is inaccessible from consumer video. This matters
enormously for AFib specifically, because pulse rise/fall time is the
feature that discriminates PAC/PVC ectopy from AF in contact PPG. The
remote pipeline inherits the ectopy false-positive problem in full while
losing the standard mitigation, which is why the classifier needs an
explicit OTHER_IRREGULAR class rather than a binary AF/not-AF head.

All features accept an optional per-interval confidence weight so that
low-confidence intervals contribute proportionally rather than being either
silently included or hard-dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

TRANSFERABILITY = {
    "mean_ibi": "HIGH", "median_ibi": "HIGH", "sdnn": "HIGH", "rmssd": "HIGH",
    "sdsd": "HIGH", "cv_ibi": "HIGH", "pnn50": "HIGH", "pnn20": "HIGH",
    "sample_entropy": "HIGH", "shannon_entropy": "HIGH", "spectral_entropy": "HIGH",
    "poincare_sd1": "HIGH", "poincare_sd2": "HIGH", "poincare_ratio": "HIGH",
    "turning_point_ratio": "HIGH", "irregularity_index": "HIGH",
    "markov_surprise": "HIGH", "autocorr_lag1": "HIGH", "n_intervals": "HIGH",
    "amplitude_cv": "MED", "amplitude_rmssd": "MED", "amplitude_ibi_slope": "MED",
    "pulse_rise_time": "LOW", "dicrotic_notch_depth": "LOW",
    "augmentation_index": "LOW", "second_derivative_ratio": "LOW",
}


@dataclass
class RhythmFeatures:
    values: dict[str, float]
    n_intervals: int
    mean_confidence: float
    estimator_warnings: list[str]

    def usable(self) -> dict[str, float]:
        """Only features whose sample-size preconditions were met."""
        return {k: v for k, v in self.values.items() if np.isfinite(v)}


# --------------------------------------------------------------------------
# v0.7 (invariant G-a): every interval statistic lives in ONE module,
# features/regularity.py. This module keeps the RhythmFeatures VIEW the
# decision logic, Model A and the training engine read, plus the
# transferability table; the estimators are re-exported unchanged. The
# v1 raw-series path (compute_rhythm_features) was deleted: the pipeline
# never called it and a second implementation is exactly what G-a bans.
from features.regularity import (markov_surprise, regularity_from_runs,  # noqa: F401,E402
                                 sample_entropy, shannon_entropy,
                                 spectral_entropy, turning_point_ratio)


def compute_rhythm_features_from_runs(runs: list[np.ndarray],
                                      run_confidences: Optional[list[np.ndarray]] = None,
                                      dropout_rate: float = 0.0,
                                      mean_sqi: float = float("nan")
                                      ) -> RhythmFeatures:
    """THE PRODUCTION FEATURE PATH, as a view of the canonical
    representation (features/regularity.py). Bit-identical to the
    pre-v0.7 values: the arithmetic moved verbatim and is pinned by
    tests/test_regularity_equivalence.py."""
    return regularity_from_runs(runs, run_confidences, dropout_rate,
                                mean_sqi).as_rhythm_features()
