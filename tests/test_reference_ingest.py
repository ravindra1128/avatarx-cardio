"""M2.8 — paired-ECG reference ingestion: sync verified and fail-closed,
in-house R peaks, episode-locked labels with adjudicator fields."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess

import numpy as np
import pytest

from datasets.reference import (ingest_reference, detect_rpeaks,
                                read_wfdb, ReferenceError)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
FS = 250.0
OFFSET_MS, DRIFT_PPM = 123.0, 40.0


def _synth_ecg(duration_s=60.0, hr_bpm=72.0, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / FS)
    period = 60.0 / hr_bpm
    rtimes = np.arange(0.5, duration_s - 0.5, period)
    rtimes = rtimes + rng.normal(0, 0.01, rtimes.size)
    mv = 0.08 * np.sin(2 * np.pi * 0.25 * t)          # baseline wander
    for rt in rtimes:                                  # QRS complexes
        mv += 1.2 * np.exp(-0.5 * ((t - rt) / 0.012) ** 2)
        mv += 0.25 * np.exp(-0.5 * ((t - rt - 0.25) / 0.05) ** 2)  # T wave
    mv += rng.normal(0, 0.02, t.size)
    return t, mv, rtimes


def _session(tmp, *, n_events=16, event_jitter_ms=0.0, labels=None,
             ecg=None):
    t, mv, rtimes = ecg or _synth_ecg()
    rows = "\n".join(f"{a:.6f},{b:.6f}" for a, b in zip(t, mv))
    (tmp / "ecg_export.csv").write_text("t_s,mv\n" + rows)
    ev_e = np.linspace(2.0, t[-1] - 2.0, n_events)
    rng = np.random.default_rng(9)
    ev_v = (ev_e * (1 + DRIFT_PPM * 1e-6) + OFFSET_MS / 1000.0
            + rng.normal(0, event_jitter_ms / 1000.0, ev_e.size))
    (tmp / "sync_events.json").write_text(json.dumps(
        {"video_events_s": ev_v.tolist(), "ecg_events_s": ev_e.tolist()}))
    if labels is None:
        labels = [{"t_start_s": 0.0, "t_end_s": float(t[-1]),
                   "rhythm": "SINUS", "annotator_initials": "PG",
                   "date": "2026-08-25", "esc_definition_confirmed": True}]
    (tmp / "labels.json").write_text(json.dumps(labels))
    return t, mv, rtimes


def test_rpeak_detector_against_known_truth():
    t, mv, rtimes = _synth_ecg()
    det = detect_rpeaks(t, mv)
    matched = 0
    errs = []
    for rt in rtimes:
        d = np.min(np.abs(det - rt))
        if d < 0.05:
            matched += 1
            errs.append(d)
    assert matched / rtimes.size >= 0.98
    assert det.size <= rtimes.size * 1.02              # no false-peak spray
    assert np.median(errs) * 1000.0 < 10.0


def test_ingest_writes_reference_with_verified_sync(tmp_path):
    t, mv, rtimes = _session(tmp_path)
    ref = ingest_reference(tmp_path)
    assert (tmp_path / "reference.json").exists()
    assert abs(ref["sync"]["offset_ms"] - OFFSET_MS) < 3.0
    assert ref["sync"]["uncertainty_ms"] <= 5.0
    assert ref["sync"]["n_marker_events"] >= 10
    # R peaks land on the video clock through the sanctioned full model
    rv = np.asarray(ref["rpeaks_s_video"])
    expect = rtimes * (1 + DRIFT_PPM * 1e-6) + OFFSET_MS / 1000.0
    hits = sum(1 for e in expect if np.min(np.abs(rv - e)) < 0.05)
    assert hits / expect.size >= 0.97
    assert ref["labels"][0]["rhythm"] == "SINUS"
    assert "never a consumer artifact" in ref["flow"]


def test_fail_closed_on_bad_sync(tmp_path):
    _session(tmp_path, n_events=4)                     # below the 10 floor
    with pytest.raises(ReferenceError, match="marker events"):
        ingest_reference(tmp_path)
    for f in tmp_path.iterdir():
        f.unlink()
    _session(tmp_path, event_jitter_ms=40.0)           # blows the 5 ms budget
    with pytest.raises(ReferenceError, match="uncertainty"):
        ingest_reference(tmp_path)
    for f in tmp_path.iterdir():
        f.unlink()
    t, mv, _ = _synth_ecg()
    _session(tmp_path)
    (tmp_path / "sync_events.json").write_text(json.dumps(
        {"video_events_s": [200.0, 210.0], "ecg_events_s": [200.0, 210.0]}))
    with pytest.raises(ReferenceError, match="overlap"):
        ingest_reference(tmp_path)


def test_label_validation_fails_closed(tmp_path):
    bad_rows = [
        [{"t_start_s": 0, "t_end_s": 10, "rhythm": "WIBBLE",
          "annotator_initials": "PG", "date": "2026-08-25",
          "esc_definition_confirmed": True}],
        [{"t_start_s": 0, "t_end_s": 10, "rhythm": "AFIB",
          "annotator_initials": "PG", "date": "2026-08-25",
          "esc_definition_confirmed": False}],
        [{"t_start_s": 50, "t_end_s": 40, "rhythm": "AFIB",
          "annotator_initials": "PG", "date": "2026-08-25",
          "esc_definition_confirmed": True}],
        [{"t_start_s": 0, "t_end_s": 10, "rhythm": "AFIB",
          "date": "2026-08-25", "esc_definition_confirmed": True}],
    ]
    for rows in bad_rows:
        for f in tmp_path.iterdir():
            f.unlink()
        _session(tmp_path, labels=rows)
        with pytest.raises(ReferenceError):
            ingest_reference(tmp_path)


def test_morphology_labels_round_trip(tmp_path):
    """v0.3 T1: adjudicated morphology fields (conduction pattern + measured
    PR/QRS/QT per segment) ride the label rows into reference.json — the
    reconstruction track's G2/G3 ground truth."""
    rows = [{"t_start_s": 0.0, "t_end_s": 30.0, "rhythm": "SINUS",
             "annotator_initials": "PG", "date": "2026-08-29",
             "esc_definition_confirmed": True,
             "morphology": {"conduction_pattern": "normal",
                            "pr_ms": 162, "qrs_ms": 92, "qt_ms": 398}},
            {"t_start_s": 30.0, "t_end_s": 59.0, "rhythm": "AFIB",
             "annotator_initials": "PG", "date": "2026-08-29",
             "esc_definition_confirmed": True,
             "morphology": {"conduction_pattern": "rbbb",
                            "pr_ms": None,        # AF: no PR by definition
                            "qrs_ms": 128, "qt_ms": 366}}]
    _session(tmp_path, labels=rows)
    ref = ingest_reference(tmp_path)
    m0, m1 = ref["labels"][0]["morphology"], ref["labels"][1]["morphology"]
    assert m0 == {"conduction_pattern": "normal", "pr_ms": 162.0,
                  "qrs_ms": 92.0, "qt_ms": 398.0}
    assert m1["conduction_pattern"] == "rbbb" and m1["pr_ms"] is None
    # rows without the block stay valid (rhythm-only campaigns)
    for f in tmp_path.iterdir():
        f.unlink()
    _session(tmp_path)
    assert ingest_reference(tmp_path)["labels"][0]["morphology"] is None


def test_morphology_fails_closed(tmp_path):
    base = {"t_start_s": 0.0, "t_end_s": 10.0, "rhythm": "SINUS",
            "annotator_initials": "PG", "date": "2026-08-29",
            "esc_definition_confirmed": True}
    bad_blocks = [
        {"pr_ms": 160},                                   # no pattern
        {"conduction_pattern": "wibble", "pr_ms": 160},   # unknown pattern
        {"conduction_pattern": "normal", "qt_ms": 900},   # out of range
        {"conduction_pattern": "normal", "qrs_ms": -5},
        {"conduction_pattern": "normal", "st_mm": 2},     # unknown field
        {"conduction_pattern": "normal", "pr_ms": "long"},
    ]
    for m in bad_blocks:
        for f in tmp_path.iterdir():
            f.unlink()
        _session(tmp_path, labels=[dict(base, morphology=m)])
        with pytest.raises(ReferenceError, match="morphology"):
            ingest_reference(tmp_path)


def test_minimal_wfdb_reader_round_trip(tmp_path):
    t, mv, _ = _synth_ecg(duration_s=10.0)
    gain = 200.0
    adc = np.clip(np.round(mv * gain), -32768, 32767).astype("<i2")
    (tmp_path / "ecg.dat").write_bytes(adc.tobytes())
    (tmp_path / "ecg.hea").write_text(
        f"ecg 1 {FS:.0f} {adc.size}\necg.dat 16 {gain:.0f}(0)/mV 16 0 0 0 0 I\n")
    t2, mv2 = read_wfdb(tmp_path / "ecg.hea")
    assert t2.size == adc.size
    assert abs(float(t2[-1]) - float(t[-1])) < 0.01
    assert float(np.max(np.abs(mv2 - np.asarray(adc, float) / gain))) < 1e-9


def test_cli_ingest_reference(tmp_path):
    _session(tmp_path)
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"),
                        "ingest-reference", str(tmp_path)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["n_rpeaks"] > 50 and doc["n_labels"] == 1


def test_ingest_pi_export_maps_onto_video_clock_and_bridges(tmp_path):
    """v0.5 review findings: the pi_export.csv merge block was untested,
    and the ingested trace had no consumer-shaped bridge. Value-pinned:
    the sync model is APPLIED, and the uniform PiTrace bridge is a valid
    sidecar."""
    from datasets.schema import pi_from_dict
    t, mv, rtimes = _session(tmp_path)
    ts = np.arange(1.0, float(t[-1]) - 1.0, 0.5)
    rows = "\n".join(f"{ti:.3f},{2.0 + 0.01 * ti:.4f}" for ti in ts)
    (tmp_path / "pi_export.csv").write_text("t_s,pi_percent\n" + rows)
    ref = ingest_reference(tmp_path)
    pi = ref["perfusion_index"]
    assert pi["n"] == ts.size and pi["source_file"] == "pi_export.csv"
    # the sync model (offset + drift) is applied, not just monotonicity
    expect = ts * (1 + DRIFT_PPM * 1e-6) + ref["sync"]["offset_ms"] / 1e3
    assert np.allclose(pi["t_s_video"], expect, atol=5e-3)
    # the bridge writes a schema-valid uniform PiTrace sidecar
    bridged = pi_from_dict(json.loads(
        (tmp_path / "pi_trace.json").read_text()))
    assert bridged.fs_hz == 1.0
    assert abs(bridged.t0_video_s - expect[0]) < 0.01
    # a malformed export fails the whole ingest closed, never silently
    (tmp_path / "pi_export.csv").write_text("t_s,pi_percent\n1,2\n1,2\n")
    import pytest as _pytest
    with _pytest.raises(ValueError):
        ingest_reference(tmp_path)
