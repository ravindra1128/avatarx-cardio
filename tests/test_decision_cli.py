"""
T5 — decision logic, pipeline orchestration, CLI.

Property tests enforce the invariants directly:
  * no code path emits a class when SQI/coverage is below floor or NaN;
  * every user-facing string is exactly ScanResult.user_facing_text();
  * provenance is fully populated on every pipeline result.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import itertools
import json
import subprocess

import numpy as np
import pytest

from datasets.schema import ScanOutcome, ScanResult
from configs import load_config
from features.rhythm import RhythmFeatures
from inference.decision_logic import decide
from inference.pipeline import run as pipeline_run

REPO = str(pathlib.Path(__file__).resolve().parents[1])
CLI = str(pathlib.Path(REPO) / "cli.py")


def feats(median_abs=12.0, pnn50=0.05, dropout=0.0, median_ibi=850.0,
          n_intervals=24, irregularity=None):
    v = {"median_abs_succ_diff": median_abs, "pnn50": pnn50,
         "dropout_rate": dropout, "median_ibi": median_ibi,
         "mean_ibi": median_ibi, "n_intervals": float(n_intervals),
         "irregularity_index": (irregularity if irregularity is not None
                                else median_abs / median_ibi)}
    return RhythmFeatures(values=v, n_intervals=n_intervals,
                          mean_confidence=0.9, estimator_warnings=[])


# ---------------------------------------------------------- decision gates
def test_no_class_ever_emitted_below_quality_floors():
    """Property sweep: SQI/coverage below floor or NaN, or too few clean
    intervals -> never ACCEPT, never a predicted class, always reasons.

    The sweep includes GOOD values on each axis so every gate is exercised
    in isolation — a review found the original sweep never reached the
    coverage gate because SQI always failed first."""
    cfg = load_config()
    good = feats(median_abs=98.0, pnn50=0.8)              # blatant AF features
    sweep = itertools.product(
        [0.0, 0.1, 0.29, float("nan"), 0.9],              # incl. GOOD sqi
        [0.0, 0.2, 0.49, float("nan"), 0.9],              # incl. GOOD coverage
    )
    for sqi, cov in sweep:
        sqi_ok = np.isfinite(sqi) and sqi >= cfg["decision"]["sqi_floor"]
        cov_ok = np.isfinite(cov) and cov >= cfg["decision"]["coverage_floor"]
        if sqi_ok and cov_ok:
            continue                                      # the ACCEPT cell
        r = decide(good, sqi, cov, cfg)
        assert r.outcome is not ScanOutcome.ACCEPT, (sqi, cov)
        assert r.predicted_class is None
        assert r.afib_probability is None
        assert r.no_read_reasons, (sqi, cov)


def test_coverage_gate_fires_alone_with_good_sqi():
    """The coverage gate must fail closed on its own, not only in the
    shadow of a failing SQI."""
    cfg = load_config()
    r = decide(feats(median_abs=98.0, pnn50=0.8), 0.9, 0.2, cfg)
    assert r.outcome is not ScanOutcome.ACCEPT
    assert any("coverage" in w for w in r.no_read_reasons), r.no_read_reasons


def test_nan_in_gate_inputs_fails_closed_even_with_good_quality():
    cfg = load_config()
    r = decide(feats(median_abs=float("nan"), pnn50=float("nan")), 0.9, 0.9, cfg)
    assert r.outcome is not ScanOutcome.ACCEPT
    assert r.predicted_class is None


def test_too_few_clean_intervals_is_no_result():
    cfg = load_config()
    f = feats(n_intervals=4)
    f.values["n_intervals"] = 4.0
    r = decide(f, 0.9, 0.9, cfg)
    assert r.outcome is ScanOutcome.NO_RESULT
    assert any("interval" in w for w in r.no_read_reasons)


# ---------------------------------------------------------- interim rules
def test_interim_rule_classes():
    cfg = load_config()
    assert decide(feats(), 0.9, 0.9, cfg).predicted_class == "SINUS"
    assert decide(feats(median_abs=98, pnn50=0.8), 0.9, 0.9, cfg
                  ).predicted_class == "AFIB_SUGGESTIVE"
    assert decide(feats(median_abs=50, pnn50=0.5), 0.9, 0.9, cfg
                  ).predicted_class == "OTHER_IRREGULAR"
    # ordinary resting sinus variability (RMSSD ~40 ms, pNN50 ~0.3) is SINUS
    assert decide(feats(median_abs=26, pnn50=0.31), 0.9, 0.9, cfg
                  ).predicted_class == "SINUS"
    assert decide(feats(median_ibi=480.0, median_abs=10, pnn50=0.02), 0.9, 0.9,
                  cfg).predicted_class == "HIGH_RATE"
    # rapid AF stays AFIB_SUGGESTIVE, not HIGH_RATE
    assert decide(feats(median_ibi=480.0, median_abs=90, pnn50=0.7), 0.9, 0.9,
                  cfg).predicted_class == "AFIB_SUGGESTIVE"
    # pulse-deficit evidence upgrades moderate irregularity
    assert decide(feats(median_abs=50, pnn50=0.35, dropout=0.4), 0.9, 0.9,
                  cfg).predicted_class == "AFIB_SUGGESTIVE"


def test_decision_carries_quality_numbers():
    cfg = load_config()
    r = decide(feats(), 0.87, 0.91, cfg)
    assert r.outcome is ScanOutcome.ACCEPT
    assert abs(r.signal_quality_index - 0.87) < 1e-9


# ------------------------------------------------------------- pipeline
@pytest.fixture(scope="module")
def sinus_result(videos):
    return pipeline_run(videos["sinus30"][0])


def test_pipeline_sinus_video_reads_sinus(sinus_result):
    r = sinus_result
    assert r.outcome is ScanOutcome.ACCEPT, r.no_read_reasons
    assert r.predicted_class == "SINUS"
    assert r.usable_beats and r.usable_beats > 15
    assert r.mean_pulse_rate_bpm and 60 < r.mean_pulse_rate_bpm < 90


@pytest.mark.parametrize("name", ["af30", "af60"])
def test_pipeline_af_video_reads_irregular(videos, name):
    r = pipeline_run(videos[name][0])
    assert r.outcome is ScanOutcome.ACCEPT, r.no_read_reasons
    assert r.predicted_class in ("AFIB_SUGGESTIVE", "OTHER_IRREGULAR")


def test_pipeline_dark_video_is_no_result_with_lux_reason(videos):
    r = pipeline_run(videos["dark30"][0])
    assert r.outcome is ScanOutcome.NO_RESULT
    assert any("illuminance" in w for w in r.no_read_reasons)
    assert r.predicted_class is None and r.afib_probability is None


def test_pipeline_noface_video_is_no_result_with_face_reason(videos):
    r = pipeline_run(videos["noface30"][0])
    assert r.outcome is ScanOutcome.NO_RESULT
    assert any("face" in w.lower() for w in r.no_read_reasons)


def test_pipeline_provenance_fully_populated(sinus_result):
    r = sinus_result
    for field in ("model_version", "code_commit", "calibration_version",
                  "config_hash"):
        assert getattr(r, field) and getattr(r, field) != "unknown", field
    assert r.calibration_version.startswith("cal-")


def test_pipeline_manifest_lux_overrides_proxy(videos):
    """A metered manifest value must beat the luma proxy: the same dark
    video with a manifest asserting 400 lux passes the lux gate (and the
    proxy's failure reason disappears)."""
    r = pipeline_run(videos["dark30"][0], manifest={"illuminance_lux": 400.0})
    assert not any("illuminance" in w for w in r.no_read_reasons)


# ------------------------------------------------------------------- CLI
def _cli(*args):
    p = subprocess.run([sys.executable, CLI, *args], capture_output=True,
                       text=True, cwd=REPO)
    return p.returncode, p.stdout, p.stderr


def test_cli_process_sinus_json_and_verbatim_text(videos):
    code, out, err = _cli("process", videos["sinus30"][0])
    assert code == 0, err
    doc = json.loads(out)
    assert doc["predicted_class"] == "SINUS"
    expected = ScanResult(
        recording_id=doc["recording_id"],
        outcome=ScanOutcome(doc["outcome"]),
        predicted_class=doc["predicted_class"],
        confidence_stars=doc.get("confidence_stars"),
        confidence_limiting_factor=doc.get(
            "confidence_limiting_factor")).user_facing_text()
    assert doc["user_facing_text"] == expected          # verbatim, no edits


def test_cli_process_missing_file_exits_2():
    code, out, err = _cli("process", "/nonexistent/video.avi")
    assert code == 2
    assert err.strip()


def test_cli_validate_good_and_bad_manifest(tmp_path, videos):
    sys.path.insert(0, str(pathlib.Path(REPO) / "tests"))
    from test_integrity import make_rec
    from datasets.schema import Rhythm
    good = make_rec("p1", "siteA", "r1", [Rhythm.SINUS])
    good_p = tmp_path / "good.json"
    good_p.write_text(good.to_json())
    code, out, err = _cli("validate", str(good_p))
    assert code == 0, err
    doc = json.loads(out)
    assert doc["valid_for_beat_analysis"] is True

    bad = make_rec("p2", "siteA", "r2", [Rhythm.SINUS], codec="h264", crf=28)
    bad_p = tmp_path / "bad.json"
    bad_p.write_text(bad.to_json())
    code, out, err = _cli("validate", str(bad_p))
    assert code == 0
    doc = json.loads(out)
    assert doc["valid_for_beat_analysis"] is False
    assert any("CRF" in w for w in doc["reasons"])


def test_cli_validate_unreadable_input_exits_2(tmp_path):
    p = tmp_path / "garbage.json"
    p.write_text("{not json")
    code, out, err = _cli("validate", str(p))
    assert code == 2
    assert err.strip()
