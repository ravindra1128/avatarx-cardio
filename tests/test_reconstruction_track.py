"""v0.3 T2 — the reconstruction decoder: architecture registry
(config-swappable, fail-closed), deterministic training, lineage in the
artifact, watermark string exact."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

from research.ecg_reconstruction import WATERMARK
from research.ecg_reconstruction.decoder import (ARCHITECTURES,
                                                 ReconstructionError,
                                                 get_architecture,
                                                 reconstruct, synth_pairs,
                                                 train)


def test_watermark_string_exact():
    assert WATERMARK == ("SYNTHETIC ECG — RESEARCH ARTIFACT — "
                        "NOT A MEASUREMENT")


def test_architecture_registry_fails_closed():
    with pytest.raises(ReconstructionError, match="windowed_mlp"):
        get_architecture("transformer_xl")
    assert "windowed_mlp" in ARCHITECTURES
    for spec in ARCHITECTURES.values():
        assert callable(spec["train"]) and callable(spec["decode"])


@pytest.fixture(scope="module")
def trained():
    subs = synth_pairs(4, seed=5, duration_s=60.0)
    return subs, train(subs, seed=1, steps=200)


def test_training_is_deterministic_and_records_lineage(trained):
    subs, art = trained
    art2 = train(subs, seed=1, steps=200)
    assert json.dumps(art, sort_keys=True) == json.dumps(art2, sort_keys=True)
    assert art["architecture"] == "windowed_mlp"
    assert art["seed"] == 1 and art["steps"] == 200
    assert sorted(art["train_subjects"]) == sorted(s[0] for s in subs)


def test_reconstruct_dispatches_by_artifact_architecture(trained):
    subs, art = trained
    sid, is_af, ppg, ecg = subs[0]
    gen = reconstruct(art, ppg)
    assert gen.shape == ppg.shape
    r = float(np.corrcoef(gen[500:-500], ecg[500:-500])[0, 1])
    assert r > 0.2, r                    # learned more than noise
    with pytest.raises(ReconstructionError):
        reconstruct(dict(art, architecture="nope"), ppg)
