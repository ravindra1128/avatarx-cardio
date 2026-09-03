"""M1.4 — head_rhythm_map: the Rhythm Map renders ONLY MEASURED data
(pulse waveform + beat ticks with confidence shading, tachogram strip,
Poincaré plot) as a static SVG. No generated ECG anywhere; even the word
does not appear in the artifact."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from beats.lattice import BeatLattice, LATTICE_VERSION
from heads.base import get_head
from inference.pipeline import run_with_details

_CONSUMER = {"capture_profile": "consumer", "assume_rig_locks": False}


def test_rhythm_map_from_production_pipeline(videos):
    r, det = run_with_details(videos["sinus30"][0], manifest=_CONSUMER,
                              recording_id="rm1")
    hr = next(h for h in det["head_results"] if h["head"] == "rhythm_map")
    assert hr["measurement_class"] == "MEASURED"
    svg = hr["value"]["svg"]
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert set(hr["value"]["panels"]) == {"waveform", "tachogram",
                                          "poincare"}
    for label in ("Measured pulse waveform", "Tachogram", "Poincar"):
        assert label in svg
    assert "ECG" not in svg and "ecg" not in svg     # honest naming
    assert svg.count("<line") >= 10                  # beat ticks present


def test_rhythm_map_degrades_gracefully_without_signal():
    lat = BeatLattice(
        version=LATTICE_VERSION, fps=30.0, duration_s=10.0,
        beat_t_s=np.array([]), beat_confidence=np.array([]),
        beat_agreement=np.array([]), runs=[], run_confidences=[],
        n_intervals=0, dropout_rate=1.0, split_fraction=0.0,
        per_roi_times={}, segments=[])
    hr = get_head("rhythm_map").run(lat, {})
    svg = hr.value["svg"]
    assert svg.startswith("<svg") and hr.value["panels"] == []
    assert "no waveform" in svg and "no clean intervals" in svg
