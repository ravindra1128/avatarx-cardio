"""v0.7 Task 2 — the noise floor: the jitter budget recomputed from
first principles, false irregularity contained by the clean-run
architecture, and the MDI that refuses to be a number without data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from evaluation.regularity_floor import (beat_error_study, budget_table,
                                         calibrate_interpolation_gain,
                                         inject_beat_errors, mdi_from_rows)
from features.regularity import timing_jitter_budget


def test_budget_is_recomputed_not_typed_in():
    b30 = timing_jitter_budget(30.0, interpolation_gain=1.0)
    b60 = timing_jitter_budget(60.0, interpolation_gain=1.0)
    # (1000/fps)/sqrt(12) per beat, sqrt(2) on intervals: 13.6 / 6.8 ms
    assert b30["sigma_interval_ms"] == pytest.approx(13.608, abs=0.01)
    assert b60["sigma_interval_ms"] == pytest.approx(6.804, abs=0.01)
    # sqrt(6) on successive differences: the 23.6 ms the detector's own
    # docstring predicts for naive peak picking at 30 fps
    assert b30["rmssd_floor_ms"] == pytest.approx(23.57, abs=0.01)
    assert b30["rmssd_floor_ms"] / b60["rmssd_floor_ms"] == \
        pytest.approx(2.0, abs=1e-6)
    # the calibrated gain reproduces what metronomic clips measured
    # through the pipeline in v0.6 (15.0 ms at 30 fps, 8.2 at 60)
    c30 = timing_jitter_budget(30.0)
    assert 14.0 < c30["rmssd_floor_ms"] < 16.5
    # confidence penalty: a less certain beat is placed less precisely
    lo = timing_jitter_budget(30.0, mean_beat_confidence=0.5)
    assert lo["sigma_beat_ms"] > c30["sigma_beat_ms"]
    # unknown fps: NO budget, never a permissive one
    for bad in (None, 0, -1, "x", float("nan")):
        assert timing_jitter_budget(bad)["rmssd_floor_ms"] is None
    tbl = budget_table((30.0, 60.0))
    assert [r["fps"] for r in tbl] == [30.0, 60.0]
    assert tbl[0]["raw_rmssd_floor_ms"] > tbl[0]["calibrated_rmssd_floor_ms"]


def test_clean_runs_contain_beat_errors_and_the_raw_series_does_not():
    """Beat errors inside a run inflate dispersion. The architecture is
    supposed to contain them; measure that it does — and measure
    honestly which statistic is protected by what."""
    out = beat_error_study(fps_list=(30.0,),
                           error_rates=(0.0, 0.05, 0.20, 0.40),
                           n_trials=12, duration_s=60.0)
    rows = {r["beat_error_rate"]: r for r in out["fps"]["30.0"]}
    # no errors: neither path reads irregular
    assert rows[0.0]["clean_runs"]["false_irregular_rate"] == 0.0
    assert rows[0.0]["raw_series"]["false_irregular_rate"] == 0.0
    # 5 % errors: the clean-run path stays honest on every statistic ...
    assert rows[0.05]["clean_runs"]["false_irregular_rate"] <= 0.1
    assert rows[0.05]["clean_runs"]["rmssd_p50_ms"] < \
        1.5 * rows[0.0]["clean_runs"]["rmssd_p50_ms"]
    # ... while the raw series' RMSSD is already wrecked by sparse errors
    assert rows[0.05]["raw_series"]["rmssd_p50_ms"] > \
        3.0 * rows[0.05]["clean_runs"]["rmssd_p50_ms"]
    # the index's median is robust to SPARSE errors on either path — a
    # design property of the statistic that must be stated, not hidden
    assert rows[0.05]["raw_series"]["false_irregular_rate"] <= 0.1
    # ... and only the clean-run path survives errors that are no longer
    # sparse
    assert rows[0.40]["raw_series"]["false_irregular_rate"] >= 0.5
    assert rows[0.40]["clean_runs"]["false_irregular_rate"] <= 0.25
    assert rows[0.40]["raw_series"]["index_p50"] > \
        5 * rows[0.40]["clean_runs"]["index_p50"]
    assert out["definition"]["irregular_if_index_at_least"] == 0.06


def test_injected_errors_are_the_errors_claimed():
    rng = np.random.default_rng(0)
    rr = np.full(70, 0.85)
    t0, _ = inject_beat_errors(rr, miss_rate=0.0, false_rate=0.0, fps=30.0,
                               jitter_sigma_ms=0.0, rng=rng)
    assert t0.size == 71
    tm, _ = inject_beat_errors(rr, miss_rate=0.2, false_rate=0.0, fps=30.0,
                               jitter_sigma_ms=0.0, rng=rng)
    assert tm.size < 71
    tf, cf = inject_beat_errors(rr, miss_rate=0.0, false_rate=0.2, fps=30.0,
                                jitter_sigma_ms=0.0, rng=rng)
    assert tf.size > 71 and cf.min() == 0.9 and np.all(np.diff(tf) > 0)


def _row(pid, fps, ecg_rmssd, cam_index, grade="A", fitz=3):
    return {"recording_id": f"{pid}-{ecg_rmssd}", "participant_id": pid,
            "fps": fps, "sqi_grade": grade, "fitzpatrick_group": fitz,
            "ecg_rmssd_ms": ecg_rmssd, "camera_index": cam_index,
            "camera_rmssd_ms": cam_index * 850.0}


def test_mdi_is_the_smallest_bin_the_camera_separates_from_the_metronome():
    rng = np.random.default_rng(1)
    rows = []
    # metronomic references at 30 fps read a quantization floor ~0.015
    for i in range(6):
        rows.append(_row(f"m{i}", 30.0, 1.0, 0.015 + rng.normal(0, 0.001)))
    # rungs: 7 ms still hides under the floor; 12 ms clears it
    for i in range(4):
        rows.append(_row(f"a{i}", 30.0, 7.0, 0.016 + rng.normal(0, 0.001)))
        rows.append(_row(f"b{i}", 30.0, 12.0, 0.022 + rng.normal(0, 0.001)))
        rows.append(_row(f"c{i}", 30.0, 25.0, 0.035 + rng.normal(0, 0.002)))
    out = mdi_from_rows(rows)
    cell = out["cells"]["fps=30|sqi=A|fitz=3"]
    assert cell["mdi_ms"] == pytest.approx(12.0, abs=0.5)
    assert cell["n_metronomic_refs"] == 6
    bins = {tuple(b["bin_ms"]): b for b in cell["bins"]}
    assert bins[(5.0, 10.0)]["detected_fraction"] < 0.8
    assert bins[(10.0, 15.0)]["detected_fraction"] >= 0.8
    # a frame rate with no metronomic references is NOT characterized
    rows60 = [_row("z1", 60.0, 12.0, 0.02), _row("z2", 60.0, 12.0, 0.02),
              _row("z3", 60.0, 12.0, 0.02)]
    out2 = mdi_from_rows(rows + rows60)
    assert out2["cells"]["fps=60|sqi=A|fitz=3"]["mdi_ms"] is None
    assert "floor unknown" in out2["cells"]["fps=60|sqi=A|fitz=3"]["reason"]
    # ... and a cell whose rungs never clear the floor is not a number
    flat = [_row(f"m{i}", 30.0, 1.0, 0.015) for i in range(6)] + \
        [_row(f"f{i}", 30.0, 12.0, 0.010) for i in range(4)]
    assert mdi_from_rows(flat)["cells"]["fps=30|sqi=A|fitz=3"]["mdi_ms"] \
        is None


def test_interpolation_gain_is_measured_from_metronomic_references():
    rows = [_row(f"m{i}", 30.0, 1.0, 0.015) for i in range(4)]
    for r in rows:
        r["camera_rmssd_ms"] = 15.0
    g = calibrate_interpolation_gain(rows)["30.0"]
    assert g["n_metronomic_refs"] == 4
    assert g["interpolation_gain_measured"] == pytest.approx(15.0 / 23.57,
                                                             abs=0.01)
    assert calibrate_interpolation_gain([]) == {}


# ------------------------------------------------ review-driven fixes
def _frow(rid, *, fps=30.0, ecg=2.0, cam=15.0, idx=0.02, grade="A",
         fitz=3, outcome="ACCEPT"):
    return {"recording_id": rid, "fps": fps, "ecg_rmssd_ms": ecg,
            "camera_rmssd_ms": cam, "camera_index": idx, "sqi_grade": grade,
            "fitzpatrick_group": fitz, "scan_outcome": outcome}


def test_measured_fps_falls_into_its_nominal_class():
    from evaluation.regularity_floor import nominal_fps
    assert nominal_fps(29.97) == 30.0 and nominal_fps(30.02) == 30.0
    assert nominal_fps(59.94) == 60.0 and nominal_fps(24.0) == 24.0
    assert nominal_fps(33.0) == 33.0                # no class: kept
    rows = [_frow(f"a{i}", fps=29.97 + 0.01 * (i % 3)) for i in range(12)]
    g = calibrate_interpolation_gain(rows)
    assert list(g) == ["30.0"] and g["30.0"]["n_metronomic_refs"] == 12


def test_gain_removes_the_references_own_dispersion_in_quadrature():
    import math
    rows = [_frow(f"a{i}", ecg=4.0, cam=math.sqrt(4.0 ** 2 + 15.0 ** 2))
            for i in range(12)]
    g = calibrate_interpolation_gain(rows)["30.0"]
    assert g["camera_only_rmssd_p50_ms"] == pytest.approx(15.0, abs=1e-6)
    assert g["interpolation_gain_measured"] == pytest.approx(
        15.0 / 23.5702, abs=1e-3)


def test_rejected_scans_are_neither_references_nor_detections():
    from evaluation.regularity_floor import mdi_from_rows
    rows = [_frow(f"m{i}", idx=0.01) for i in range(6)]
    rows += [_frow(f"r{i}", idx=0.5, outcome="REPEAT_SCAN") for i in range(6)]
    g = calibrate_interpolation_gain(rows)["30.0"]
    assert g["n_metronomic_refs"] == 6
    m = mdi_from_rows(rows, min_regular_refs=3)
    assert m["floors_per_fps"]["30.0"]["n"] == 6
    # a bin of rejected scans at 12 ms true dispersion: 0 detected of 4
    rows2 = rows + [_frow(f"n{i}", ecg=12.0, idx=None, outcome="NO_RESULT")
                    for i in range(4)]
    m2 = mdi_from_rows(rows2, min_regular_refs=3)
    cell = m2["cells"]["fps=30|sqi=A|fitz=3"]
    b = [x for x in cell["bins"] if x["bin_ms"] == [10.0, 15.0]][0]
    assert b["n"] == 4 and b["n_no_read"] == 4
    assert b["detected_fraction"] == 0.0


def test_mdi_requires_detection_to_be_monotone_in_true_dispersion():
    from evaluation.regularity_floor import mdi_from_rows
    rows = [_frow(f"m{i}", idx=0.01) for i in range(6)]
    rows += [_frow(f"b{i}", ecg=7.0, idx=0.05) for i in range(4)]    # passes
    rows += [_frow(f"c{i}", ecg=12.0, idx=0.005) for i in range(4)]  # fails
    rows += [_frow(f"d{i}", ecg=25.0, idx=0.08) for i in range(4)]   # passes
    cell = mdi_from_rows(rows, min_regular_refs=3)["cells"][
        "fps=30|sqi=A|fitz=3"]
    assert cell["mdi_ms"] is None
    assert "not monotone" in cell["reason"]
    rows = [r for r in rows if not r["recording_id"].startswith("c")]
    cell = mdi_from_rows(rows, min_regular_refs=3)["cells"][
        "fps=30|sqi=A|fitz=3"]
    assert cell["mdi_ms"] == 7.0


def test_floor_run_records_the_published_parameters(tmp_path):
    from evaluation.regularity_floor import regularity_floor_report
    from evaluation.regularity_gates import (latest_scoreboard_entry,
                                             load_regularity_gates)
    rows = [_frow(f"m{i}") for i in range(4)]
    doc = regularity_floor_report(rows, runs_root=tmp_path,
                                  beat_error_trials=2)
    g = load_regularity_gates()["r1_noise_floor"]
    assert doc["parameters"]["mdi_power"] == g["mdi_power"]
    assert doc["parameters"]["mdi_confidence"] == g["mdi_confidence"]
    entry = latest_scoreboard_entry("floor", runs_root=tmp_path)
    assert entry["parameters"] == doc["parameters"]
    assert entry["signal_domain"] == "synthetic"
