"""v0.7 regularity — the refactor's definition of SAFE.

`head_afib` and `head_flutter` are rewired onto the shared
RegularityFeatures representation. This test pins their behaviour
BIT-FOR-BIT to a golden capture taken on the pre-refactor code: the
sanctioned sentence, star grade, outcome, class, every production
feature value, the beat-evidence keys the decision reads, and the
flutter extractor's dispersion/coupling numbers, on a deterministic
synthetic corpus at 30 and 60 fps.

Regenerating the golden (AVATARX_REGEN_GOLDEN=1) is a behaviour change
by definition and requires a spec-changelog entry saying why.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import math
import os

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = _ROOT / "tests" / "fixtures" / "regularity_equivalence_golden.json"
_CONSUMER = {"capture_profile": "consumer", "assume_rig_locks": False}

# (name, kind, fps, duration_s, extra synth kwargs, respiration bar?)
CORPUS = [
    ("sinus30", "sinus", 30.0, 20.0, {}, False),
    ("sinus60", "sinus", 60.0, 20.0, {}, False),
    ("af30", "af", 30.0, 20.0, {}, False),
    ("af60", "af", 60.0, 20.0, {}, False),
    ("dark30", "sinus", 30.0, 20.0, {"lux_scale": 0.15}, False),
    ("rsa30", "rsa", 30.0, 40.0, {}, True),
    ("flutter30", "flutter_track_2to1", 30.0, 40.0, {"rr": "flutter"}, True),
    # review finding: every clip above yields ONE clean run, so the
    # golden could not see the within-run vs cross-seam pooling the
    # refactor is about, and no processed clip abstained. These do.
    ("af30_deficit", "af", 30.0, 40.0, {"deficit_drop_short_ms": 500.0},
     False),
    ("sinus30_deficit", "sinus", 30.0, 40.0,
     {"deficit_drop_short_ms": 830.0}, False),
    ("sinus30_shaky", "sinus", 30.0, 20.0, {"jitter_px": 8.0}, False),
    ("sinus30_dim", "sinus", 30.0, 20.0, {"lux_scale": 0.45}, False),
]

# Behaviour changes the v0.7 spec RECORDS (B.23), each confined to clips
# with more than one clean run: on a single run there is no seam to
# pool across and these must still match bit for bit. Everything the
# AFib decision reads (features, afib_value, sentence, stars, reasons)
# is never exempt.
DOCUMENTED_CHANGES = {
    "irregularity_value": "B.23 #5 — head_irregularity reads within-run "
                          "rel_mad/pnn80; the pre-v0.7 head counted the "
                          "run seam as a successive difference",
    "flutter": "B.23 #14 — the tachogram is built from the longest run; "
               "the pre-v0.7 tachogram bridged seams under 3 s",
}


def _num(v):
    """JSON-safe, EXACT: floats stay floats (repr round-trips bit-for-bit),
    NaN becomes the string 'nan'."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return "nan" if math.isnan(f) else f
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return {str(k): _num(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    return str(v)


def _capture(path, breathing):
    from inference.pipeline import run_with_details
    res, det = run_with_details(path, manifest=_CONSUMER,
                                recording_id="equiv")
    base = {
        "outcome": res.outcome.value,
        "predicted_class": res.predicted_class,
        "confidence_stars": res.confidence_stars,
        "user_facing_text": res.user_facing_text(),
        "no_read_reasons": list(res.no_read_reasons),
    }
    if "features" not in det:
        # capture-invalid clips exit at ingest: the sentence, stars and
        # reasons are the whole behaviour to pin
        return dict(base, features=None, flutter={"available": False},
                    irregularity_value=None)
    feats = det["features"]
    ev = det["evidence"]
    heads = {h["head"]: h for h in det["head_results"]}
    out = {
        **base,
        "n_runs": int(len(det["lattice"].runs)),
        "features": _num(feats.values),
        "n_intervals": int(feats.n_intervals),
        "mean_confidence": _num(feats.mean_confidence),
        "estimator_warnings": list(feats.estimator_warnings),
        "evidence": _num({k: ev.get(k) for k in (
            "harmonic_fraction", "n_intervals", "split_fraction",
            "timing_precision_ms", "timing_matched_fraction",
            "cross_roi_coherence", "dropout_rate", "n_beats")}),
        "head_names": [h["head"] for h in det["head_results"]],
        "afib_value": _num(heads["afib"]["value"]),
        "rate_flags_value": _num(heads["rate_flags"]["value"]),
        "rationale_features": _num(det["rationale"].get("features")),
        "gates_failed": list(det["rationale"].get("gates_failed") or []),
    }
    # the flutter extractor on the same lattice, with the torso-derived
    # respiration channel where the clip carries one
    from features.flutter import flutter_features
    resp = None
    if breathing:
        from rppg._filters import moving_average_detrend
        from rppg.respiration import (respiratory_rate_from_motion,
                                      torso_motion_series)
        t, y, fps = torso_motion_series(path)
        rate, conc = respiratory_rate_from_motion(t, y, fps)
        resp = {"t": t, "y": moving_average_detrend(y, int(12.0 * fps)),
                "rate_brpm": rate, "quality": conc}
    ff = flutter_features(res, det["lattice"], respiration=resp,
                          age_years=60)
    if ff.get("available"):
        reg = ff["regularity"]
        out["flutter"] = _num({
            "rate": {k: ff["rate"].get(k) for k in (
                "median_bpm", "band", "in_band", "band_fraction",
                "n_intervals")},
            "regularity": {k: reg.get(k) for k in (
                "rmssd_ms", "sdnn_ms", "cv", "below_floor",
                "effective_floor_ms", "measurement_limited",
                "spectral_concentration")},
            "coupling": {k: reg["coupling"].get(k) for k in (
                "available", "tachogram_resp_fraction", "resp_rate_brpm")},
        })
    else:
        out["flutter"] = {"available": False}
    from heads import get_head
    hr = get_head("irregularity").run(det["lattice"], {})
    out["irregularity_value"] = _num(hr.value)
    return out


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    pytest.importorskip("cv2")
    from scripts.make_synth_flutter import flutter_rr
    from scripts.make_synth_video import synth_video
    d = tmp_path_factory.mktemp("equiv")
    out = {}
    for name, kind, fps, dur, extra, breathing in CORPUS:
        kw = dict(extra)
        rr = None
        if kw.pop("rr", None) == "flutter":
            rr = flutter_rr(dur, ratio=2, seed=7)
        path = str(d / f"{name}.avi")
        synth_video(path, kind=kind, fps=fps, duration_s=dur, seed=7,
                    rr_override=rr,
                    torso_respiration=({"brpm": 15.0, "bob_px": 6.0}
                                       if breathing else None), **kw)
        out[name] = (path, breathing)
    return out


def _walk_equal(golden, got, path=""):
    """Exact structural equality with an informative path on mismatch."""
    if isinstance(golden, dict):
        assert isinstance(got, dict), path
        assert set(golden) == set(got), (path, set(golden) ^ set(got))
        for k in golden:
            _walk_equal(golden[k], got[k], f"{path}.{k}")
    elif isinstance(golden, list):
        assert isinstance(got, list) and len(golden) == len(got), path
        for i, (a, b) in enumerate(zip(golden, got)):
            _walk_equal(a, b, f"{path}[{i}]")
    else:
        assert golden == got, f"{path}: golden {golden!r} != got {got!r}"


def test_afib_and_flutter_are_bit_identical_to_the_golden(corpus):
    got = {name: _capture(p, b) for name, (p, b) in corpus.items()}
    if os.environ.get("AVATARX_REGEN_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True))
        pytest.skip("golden regenerated — this is a behaviour change and "
                    "needs a spec-changelog entry")
    assert GOLDEN.exists(), ("no golden on disk: run once with "
                             "AVATARX_REGEN_GOLDEN=1 on the PRE-refactor "
                             "code")
    golden = json.loads(GOLDEN.read_text())
    assert set(golden) == set(got), set(golden) ^ set(got)
    exempted = []
    for name in golden:
        g, h = golden[name], got[name]
        assert set(g) == set(h), (name, set(g) ^ set(h))
        multi = int(g.get("n_runs") or 0) > 1
        for key in g:
            if key in DOCUMENTED_CHANGES and multi:
                exempted.append((name, key))
                continue
            _walk_equal(g[key], h[key], f"{name}.{key}")
    # the exemptions exist for a reason: at least one multi-run clip
    # must be in the corpus, or the documented changes are untested
    assert any(int(v.get("n_runs") or 0) > 1 for v in golden.values()), \
        "no multi-run clip in the golden corpus"
    assert exempted, "documented changes never exercised"


def test_afib_decision_is_never_exempt_from_the_golden():
    """The exemption list can only ever name research-side outputs."""
    for key in DOCUMENTED_CHANGES:
        assert key not in ("features", "afib_value", "user_facing_text",
                           "confidence_stars", "outcome", "predicted_class",
                           "no_read_reasons", "gates_failed", "evidence",
                           "rate_flags_value", "rationale_features")


def test_golden_covers_every_outcome_and_both_frame_rates():
    if not GOLDEN.exists():
        pytest.skip("no golden yet")
    g = json.loads(GOLDEN.read_text())
    classes = {v["predicted_class"] for v in g.values()}
    outcomes = {v["outcome"] for v in g.values()}
    assert {"SINUS", "AFIB_SUGGESTIVE"} <= classes, classes
    assert {"ACCEPT", "NO_RESULT"} <= outcomes, outcomes
    # a PROCESSED clip that did not ACCEPT (review finding: the only
    # non-ACCEPT clip exited at ingest and pinned nothing downstream)
    assert any(v["outcome"] != "ACCEPT" and v.get("features") is not None
               for v in g.values()), "no processed abstention in the golden"
    assert any(int(v.get("n_runs") or 0) > 1 for v in g.values())
    assert {"HIGH_RATE"} <= classes or any(
        v["predicted_class"] == "HIGH_RATE" for v in g.values()), classes
    # 30 and 60 fps both represented, and the flutter extractor ran with
    # a real respiration channel on at least one clip
    assert any("60" in k for k in g)
    assert any(v["flutter"].get("coupling", {}).get("available") is True
               for v in g.values())
