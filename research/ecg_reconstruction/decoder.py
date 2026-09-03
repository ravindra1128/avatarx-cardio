"""
Reconstruction model (v0.3 T2) — a config-swappable ARCHITECTURE REGISTRY
over permissively-licensed, CPU-trainable sequence models mapping the
measured facial-pulse representation to a single-lead ECG estimate.

The registry, not any single network, is the interface: training configs
name an architecture and the artifact records it, so a deeper model slots
in without touching the training loop, the fidelity evaluator or the
gates. v1 is the v0.2 falsification decoder (windowed MLP sliding a
+/-0.5 s pulse context per output sample — conv-scale, ~10k parameters,
manual Adam) promoted from one-off lab experiment to the track's baseline
model; reusing it rather than duplicating it is deliberate (v0.3 step 0.3).

Data utilities (MIMIC PERform cache, synthetic paired fixtures,
participant-disjoint quantile splits) are likewise reused from the lab.
"""
from __future__ import annotations

import numpy as np

from evaluation.inferred_ecg.decoder import (FS, decode as _mlpwin_decode,
                                             perform_subjects,      # noqa: F401 (re-exported)
                                             split_subjects,        # noqa: F401
                                             synth_pairs,           # noqa: F401
                                             train_decoder as _mlpwin_train)


class ReconstructionError(ValueError):
    pass


ARCHITECTURES = {
    "windowed_mlp": {
        "description": "one-hidden-layer regressor over a +/-0.5 s pulse "
                       "context per ECG sample (conv-scale; v0.2 lab "
                       "decoder, ppg2ecg-mlpwin-v1)",
        "train": _mlpwin_train,          # (subjects, *, seed, steps, batch, lr)
        "decode": _mlpwin_decode,        # (artifact, ppg) -> ecg estimate
    },
}


def get_architecture(name: str) -> dict:
    try:
        return ARCHITECTURES[name]
    except KeyError:
        raise ReconstructionError(
            f"unknown reconstruction architecture {name!r}; available: "
            f"{sorted(ARCHITECTURES)}") from None


def train(subjects: list, *, architecture: str = "windowed_mlp",
          seed: int = 7, steps: int = 400, batch: int = 256,
          lr: float = 3e-3) -> dict:
    """Deterministic: (subjects, architecture, seed, steps, batch, lr)
    fully determine the artifact. The artifact records its architecture so
    reconstruct() can dispatch without any side channel."""
    arch = get_architecture(architecture)
    art = arch["train"](subjects, seed=seed, steps=steps, batch=batch, lr=lr)
    art["architecture"] = architecture
    return art


def reconstruct(artifact: dict, ppg: np.ndarray) -> np.ndarray:
    """Pulse signal (z-normalised, FS Hz) -> synthetic ECG estimate.
    RESEARCH ARTIFACT — every rendering site carries the watermark."""
    arch = get_architecture(artifact.get("architecture", "windowed_mlp"))
    return arch["decode"](artifact, ppg)
