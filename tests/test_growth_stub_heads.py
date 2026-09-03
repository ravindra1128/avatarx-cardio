"""M5 — growth stubs: research/internal flags only, never enabled by
default, never user-facing. (The v0.2 flutter_suspicion stub was
superseded by the v0.6 head_flutter track; its coverage moved to
tests/test_flutter_head.py.)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from beats.lattice import BeatLattice, LATTICE_VERSION
from configs import load_config
from heads.base import get_head, enabled_heads


def _lat(ibi_ms):
    r = np.asarray(ibi_ms, float)
    return BeatLattice(version=LATTICE_VERSION, fps=30.0, duration_s=30.0,
                       beat_t_s=np.array([]), beat_confidence=np.array([]),
                       beat_agreement=np.array([]), runs=[r],
                       run_confidences=[np.ones(max(len(r) - 1, 1))],
                       n_intervals=len(r), dropout_rate=0.0,
                       split_fraction=0.0, per_roi_times={}, segments=[])


def test_stub_heads_registered_but_not_enabled_by_default():
    cfg = load_config()
    names = [h.name for h in enabled_heads(cfg)]
    assert "burden" not in names
    assert "irregularity" not in names
    assert get_head("burden").research_only
    assert get_head("irregularity").research_only


def test_irregularity_flag_is_rhythm_agnostic():
    """v0.3 T6: fires on sustained irregularity of ANY kind (AF-like or
    ectopy-like), stays quiet on regular rhythms, and never says a word."""
    h = get_head("irregularity")
    rng = np.random.default_rng(9)
    af_like = _lat(np.clip(rng.normal(600, 120, 40), 300, 1200))
    r = h.run(af_like, {})
    assert r.value["research_flag"] is True
    assert r.value["user_facing"] is None
    # bigeminy-like: alternating short/long — agnostic to the cause
    bigem = _lat(np.tile([500.0, 900.0], 20) + rng.normal(0, 8, 40))
    assert h.run(bigem, {}).value["research_flag"] is True
    sinus = _lat(850.0 + rng.normal(0, 20, 40))
    assert h.run(sinus, {}).value["research_flag"] is False
    flutter_like = _lat(400.0 + rng.normal(0, 4, 40))   # regular tachy
    assert h.run(flutter_like, {}).value["research_flag"] is False
    assert h.run(_lat([800.0] * 5), {}).value["research_flag"] is False
    assert "insufficient" in h.run(_lat([800.0] * 5),
                                   {}).value["evidence"]["note"]


def test_burden_is_abstention_aware():
    h = get_head("burden")
    series = [
        {"date": "2026-08-20", "outcome": "ACCEPT",
         "predicted_class": "AFIB_SUGGESTIVE"},
        {"date": "2026-08-20", "outcome": "ACCEPT",
         "predicted_class": "SINUS"},
        {"date": "2026-08-20", "outcome": "REPEAT_SCAN",
         "predicted_class": None},
        {"date": "2026-08-21", "outcome": "REPEAT_SCAN",
         "predicted_class": None},
        {"date": "2026-08-21", "outcome": "NO_RESULT",
         "predicted_class": None},
    ]
    r = h.run(None, {"scan_series": series})
    d20 = r.value["per_day"]["2026-08-20"]
    assert d20["af_suggestive_fraction"] == 0.5        # 1 of 2 ACCEPTED
    assert d20["abstention_rate"] == round(1 / 3, 3)
    d21 = r.value["per_day"]["2026-08-21"]
    assert d21["af_suggestive_fraction"] is None       # no accepted scans:
    assert d21["abstention_rate"] == 1.0               # NOT zero burden
    assert r.value["user_facing"] is None
