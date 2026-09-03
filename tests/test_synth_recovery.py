"""v0.4 T1 — synthetic three-phase recovery fixtures, and the T2
acceptance run: the recovery clip through the REAL production pipeline,
recovery metrics judged against the parametric truth (HR(t) MAE <= 2 bpm,
HRR60 error <= 3 bpm, including degradations)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from features.recovery import recovery_metrics
from scripts.make_synth_recovery import (HONESTY_NOTE, make_recovery_session,
                                         recovery_hr)
from scripts.make_synth_video import synth_video

_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    d = tmp_path_factory.mktemp("recov")
    out = {}
    out["clean"] = make_recovery_session(d, "clean", seed=5)
    out["dropouts"] = make_recovery_session(
        d, "dropouts", seed=6, recovery_s=135.0,
        recovery_deficit_ms=430.0)
    out["resting"] = make_recovery_session(d, "resting", seed=7,
                                           include_activity=False,
                                           recovery_s=135.0)
    return d, out


def test_fixture_files_truth_and_honesty_rule(sessions):
    d, out = sessions
    for name in ("clean_rest.avi", "clean_activity.avi",
                 "clean_recovery.avi", "clean.session.json",
                 "clean.truth.json"):
        assert (d / name).exists(), name
    t = json.loads((d / "clean.truth.json").read_text())
    assert t["note"] == HONESTY_NOTE            # v0.1 honesty rule verbatim
    hr0 = t["hr_end_true"]
    assert abs((hr0 - recovery_hr(60.0, hr_rest=t["hr_rest"],
                                  hr0=t["hr0"], tau=t["tau"]))
               - t["hrr60_true"]) < 1e-9
    act = json.loads((d / "clean_activity.avi.truth.json").read_text())
    assert "rr_s" not in act and "rpeaks_s" not in act   # NO physiology
    assert act["reps"] == 20 and act["cadence_per_min"] == 20.0
    # resting-only session manifest carries no activity/recovery phases
    resting = json.loads((d / "resting.session.json").read_text())
    assert set(resting["phases"]) == {"rest"}


def test_rr_override_is_the_sanctioned_hook(tmp_path):
    rr = np.full(30, 0.75)
    t = synth_video(str(tmp_path / "o.avi"), kind="custom_label",
                    fps=30.0, duration_s=15.0, seed=1, rr_override=rr)
    kept = rr[np.cumsum(rr) <= 15.0]
    assert t["kind"] == "custom_label"
    assert np.allclose(t["rr_s"], kept)


@pytest.mark.parametrize("name", ["clean", "dropouts"])
def test_recovery_clip_through_production_pipeline_meets_t2(sessions, name):
    """The acceptance criterion, on the real path: ingest -> POS -> beats
    -> lattice -> features.recovery, judged against parametric truth."""
    d, out = sessions
    truth = out[name]
    from inference.pipeline import run_with_details
    res, det = run_with_details(str(d / f"{name}_recovery.avi"))
    lat = det.get("lattice")
    assert lat is not None, res.no_read_reasons
    dur = float(det["ingest"].meta.duration_s)
    m = recovery_metrics(np.asarray(lat.beat_t_s),
                         np.asarray(lat.beat_confidence), duration_s=dur)
    assert m["hrr60"] is not None, m["reasons"]
    assert abs(m["hr_end_proxy"] - truth["hr_end_true"]) <= 3.0, \
        (name, m["hr_end_proxy"], truth["hr_end_true"])
    assert abs(m["hrr60"] - truth["hrr60_true"]) <= 3.0, \
        (name, m["hrr60"], truth["hrr60_true"])
    errs = [abs(hr - float(recovery_hr(t, hr_rest=truth["hr_rest"],
                                       hr0=truth["hr0"],
                                       tau=truth["tau"])))
            for t, hr, c in m["hr_series"] if c is not None]
    assert float(np.mean(errs)) <= 2.0, (name, np.mean(errs))
