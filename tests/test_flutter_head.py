"""v0.6 flutter T2 — head_flutter: two outputs, one head; abstention
that is distinguishable from a negative; and the F-a/F-b/F-d invariants.

Supersedes the v0.2 head_flutter_suspicion stub, whose coverage (fires
on regular 150, silent on AF and sinus, integer-ratio steps across a
series, never a sentence) is carried here with stronger assertions.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ast
import json

import numpy as np
import pytest

from beats.lattice import BeatLattice, LATTICE_VERSION
from configs import load_config
from datasets.schema import MeasurementClass, REGULAR_TACHY_SENTENCES
from heads import available_heads, enabled_heads, get_head
from heads.head_flutter import user_facing_text, WATERMARK

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _lat(rr_s, *, fps=240.0):
    rr = np.asarray(rr_s, float)
    t = np.cumsum(rr)
    return BeatLattice(
        version=LATTICE_VERSION, fps=fps, duration_s=float(t[-1] + 1),
        beat_t_s=np.concatenate([[0.0], t]),
        beat_confidence=np.ones(t.size + 1),
        beat_agreement=np.ones(t.size + 1), runs=[rr * 1000.0],
        run_confidences=[np.ones(rr.size)], n_intervals=int(rr.size),
        dropout_rate=0.0, split_fraction=0.0, per_roi_times={},
        segments=[], run_times=[t])


def _flutter(n=100, base=0.400, jitter=0.003, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(base + rng.normal(0, jitter, n), 0.25, 1.5)


def _sinus(bpm=150.0, depth=0.10, brpm=15.0, dur=60.0, noise=0.006,
           seed=0):
    rng = np.random.default_rng(seed)
    f, base, out, t = brpm / 60.0, 60.0 / bpm, [], 0.0
    while t <= dur:
        out.append(max(base * (1 + depth * np.sin(2 * np.pi * f * t))
                       + rng.normal(0, noise), 0.25))
        t += out[-1]
    return np.asarray(out)


def _af(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.62, 0.17, n), 0.28, 1.35)


def _resp(brpm=15.0, dur=70.0, fs=30.0, quality=0.9):
    t = np.arange(0, dur, 1.0 / fs)
    return {"t": t, "y": np.sin(2 * np.pi * (brpm / 60.0) * t),
            "rate_brpm": brpm, "quality": quality}


def _ctx(**over):
    d = {"scan_outcome": "ACCEPT", "cfg": {}, "respiration": _resp(),
         "age_years": 60}
    d.update(over)
    return d


# ------------------------------------------------------- registration
def test_registered_research_only_and_never_default_enabled():
    h = get_head("flutter")
    assert h.research_only is True
    assert "flutter" in available_heads()
    assert "flutter" not in [x.name for x in enabled_heads(load_config())]
    # the superseded stub is gone, not merely unregistered
    assert "flutter_suspicion" not in available_heads()
    assert not (_ROOT / "heads" / "head_flutter_suspicion.py").exists()


def test_research_class_keeps_it_out_of_consumer_payloads():
    r = get_head("flutter").run(_lat(_flutter()), _ctx())
    assert r.measurement_class is MeasurementClass.RESEARCH_RHYTHM
    from datasets.schema import public_head_results
    assert public_head_results([r.to_dict()]) == []


# --------------------------------------------------------- the flag
def test_flag_fires_on_a_sustained_regular_150_with_no_rsa():
    r = get_head("flutter").run(_lat(_flutter(seed=1)), _ctx())
    v = r.value
    assert v["regular_tachy_flag"] is True
    assert v["abstained"] is False
    assert v["conduction_band"] == "2:1"          # internal only
    assert v["sentence_key"] == "regular_tachy"
    assert v["user_facing"] is None               # never a sentence here
    c = v["criteria"]
    assert c["sustained_band"] and c["below_dispersion_floor"]
    assert c["respiratory_modulation_absent"] is True


def test_flag_stays_down_on_sinus_tach_af_and_slow_regular():
    h = get_head("flutter")
    # sinus tachycardia at the SAME rate: coupling keeps it down
    st = h.run(_lat(_sinus(seed=2)), _ctx())
    assert st.value["regular_tachy_flag"] is False
    assert st.value["abstained"] is False
    assert st.value["criteria"]["respiratory_modulation_absent"] is False
    # AF: irregular, so neither the band nor the floor holds
    af = h.run(_lat(_af(seed=2)), _ctx())
    assert af.value["regular_tachy_flag"] is False
    # 4:1 flutter at ~75: metronomic, but out of the flagging band.
    # THE documented permanent miss (F-c), asserted as behaviour.
    slow = h.run(_lat(_flutter(base=0.800, seed=2)), _ctx())
    assert slow.value["regular_tachy_flag"] is False
    assert slow.value["evidence"]["rate"]["band"] == "4:1"
    assert slow.value["evidence"]["regularity"]["below_floor"] is True
    # paced-like metronomic at 70: same story
    paced = h.run(_lat(_flutter(base=0.857, jitter=0.001, seed=2)),
                  _ctx())
    assert paced.value["regular_tachy_flag"] is False


def test_svt_is_covered_by_design_not_separated():
    """A regular 165 with no RSA flags exactly like 2:1 flutter — the
    head cannot tell them apart and the sentence names neither."""
    r = get_head("flutter").run(_lat(_flutter(base=0.370, seed=3)),
                                _ctx())
    assert r.value["regular_tachy_flag"] is True
    assert r.value["conduction_band"] == "2:1"


# ---------------------------------------------------- abstention
def test_abstains_rather_than_denying():
    """flag None (could not look) is never flag False (looked, nothing
    there). Each abstention names its cause."""
    h = get_head("flutter")
    cases = {
        "REPEAT_SCAN": (_ctx(scan_outcome="REPEAT_SCAN"), "not ACCEPT"),
        "NO_RESULT": (_ctx(scan_outcome="NO_RESULT"), "not ACCEPT"),
        "no outcome": (_ctx(scan_outcome=None), "no scan outcome"),
        "no resp": (_ctx(respiration=None), "coupling could not be"),
        "bad resp": (_ctx(respiration={"rate_brpm": 40.0}),
                     "coupling could not be"),
    }
    for name, (ctx, needle) in cases.items():
        r = h.run(_lat(_flutter()), ctx)
        assert r.value["regular_tachy_flag"] is None, name
        assert r.value["abstained"] is True, name
        assert r.value["sentence_key"] is None, name
        assert any(needle in x for x in r.reasons), (name, r.reasons)
    # too few clean intervals
    r = h.run(_lat(_flutter(n=10)), _ctx())
    assert r.value["regular_tachy_flag"] is None
    assert any("clean intervals" in x for x in r.reasons)
    # unknown frame rate -> unknown dispersion floor -> no verdict
    lat = _lat(_flutter())
    object.__setattr__(lat, "fps", float("nan"))
    r = h.run(lat, _ctx())
    assert r.value["regular_tachy_flag"] is None
    assert any("frame rate unknown" in x for x in r.reasons)


def test_measurement_limited_flags_say_so():
    """At 30 fps the dispersion floor is frame quantization, not
    physiology. A flag raised under that floor must carry the caveat."""
    r = get_head("flutter").run(_lat(_flutter(seed=4), fps=30.0), _ctx())
    assert r.value["regular_tachy_flag"] is True
    assert r.value["criteria"]["measurement_limited"] is True
    assert any("measurement-limited" in x for x in r.reasons)


# ------------------------------------------------- the series output
def test_series_signature_is_null_below_three_scans():
    h = get_head("flutter")
    for rates in (None, [150.0], [150.0, 75.0]):
        r = h.run(_lat(_flutter()), _ctx(series_rates_bpm=rates))
        assert r.value["series_ratio_signature"] is None, rates
    r = h.run(_lat(_flutter()), _ctx(series_rates_bpm=[150.0, 100.0,
                                                       75.0]))
    sig = r.value["series_ratio_signature"]
    assert sig is not None and sig["score"] > 0.9
    assert sig["atrial_bpm"] == pytest.approx(300.0, abs=3.0)
    # migrated from the v0.2 stub: a 2:1 step across a series is the
    # evidence — but now it must be a real integer-ratio FIT, and three
    # scans at one rate score zero rather than "stepping"
    flat = h.run(_lat(_flutter()),
                 _ctx(series_rates_bpm=[150.0, 150.0, 149.0]))
    assert flat.value["series_ratio_signature"]["score"] == 0.0


# ------------------------------------------------------- F-a wording
def test_no_sentence_escapes_without_an_explicit_gate_check():
    """F-a: every word comes from user_facing_text(), which is
    fail-closed — the default cannot render."""
    r = get_head("flutter").run(_lat(_flutter(seed=5)), _ctx())
    assert user_facing_text(r.value) is None                 # default
    assert user_facing_text(r.value, render_allowed=False) is None
    assert user_facing_text(r.value, render_allowed=True) == \
        REGULAR_TACHY_SENTENCES["regular_tachy"]
    # an abstention or a negative has no key, so even an allowed render
    # produces nothing
    for ctx in (_ctx(scan_outcome="REPEAT_SCAN"), _ctx()):
        v = get_head("flutter").run(_lat(_sinus(seed=5)), ctx).value
        assert user_facing_text(v, render_allowed=True) is None
    assert user_facing_text(None, render_allowed=True) is None
    assert user_facing_text({"sentence_key": "made_up"},
                            render_allowed=True) is None


def test_head_never_names_a_rhythm_anywhere_in_its_output():
    h = get_head("flutter")
    for lat, ctx in ((_lat(_flutter()), _ctx()),
                     (_lat(_sinus()), _ctx()),
                     (_lat(_af()), _ctx(respiration=None)),
                     (_lat(_flutter()), _ctx(scan_outcome="NO_RESULT"))):
        doc = h.run(lat, ctx).to_dict()
        # the registry key and the head field are identifiers, not
        # claims; strip them and no rhythm may be named in what is left
        blob = json.dumps(doc).lower().replace('"head": "flutter"', "")
        for banned in ("flutter", "svt", "atrial tachycardia",
                       "fibrillation"):
            assert banned not in blob, (banned, blob[:200])
        # and the watermark carries the disclaimer, in full
        assert doc["value"]["watermark"] == WATERMARK
        assert "NOT A DIAGNOSIS" in WATERMARK


# ------------------------------------------------ F-d: measured path
def test_head_and_extractor_cannot_reach_the_reconstruction_track():
    """F-d: the flag is measured-path only. Nothing in the head or the
    extractor may import, name or dynamically load the gated ECG
    reconstruction track."""
    banned_prefix = ("research", "evaluation.inferred_ecg", "rbcg",
                     "importlib")
    for rel in ("heads/head_flutter.py", "features/flutter.py"):
        tree = ast.parse((_ROOT / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith(banned_prefix), (rel,
                                                                  a.name)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(banned_prefix), \
                    (rel, node.module)
            elif isinstance(node, ast.Call):
                fn = node.func
                nm = getattr(fn, "id", getattr(fn, "attr", ""))
                assert nm not in ("__import__", "import_module"), rel
            elif isinstance(node, ast.Constant):
                v = node.value
                # only strings that could BE a module path are checked:
                # prose (this file's own docstring describes the
                # firewall) is not a dynamic import
                if isinstance(v, str) and "." in v and \
                        len(v.split()) == 1:
                    assert not v.lower().startswith(
                        ("research.", "rbcg.", "evaluation.inferred")), \
                        (rel, v)


def test_head_reads_only_the_lattice_and_context():
    """Heads never re-extract signal (invariant 14). The flutter head
    must work from a lattice alone — no video, no waveform."""
    r = get_head("flutter").run(_lat(_flutter(seed=6)), _ctx())
    assert r.value["regular_tachy_flag"] is True
    assert r.version.endswith("-research")


# ------------------------------------------------ through the pipeline
def test_pipeline_publishes_the_gate_verdict_and_strips_the_head(tmp_path):
    pytest.importorskip("cv2")
    from datasets.schema import public_head_results
    from inference.pipeline import run_with_details
    from scripts.make_synth_flutter import flutter_rr
    from scripts.make_synth_video import synth_video
    v = str(tmp_path / "f.avi")
    synth_video(v, kind="flutter_track_2to1", fps=30.0, duration_s=40.0,
                seed=5, rr_override=flutter_rr(40.0, ratio=2, seed=5),
                torso_respiration={"brpm": 15.0, "bob_px": 6.0})
    res, det = run_with_details(v, heads=["afib", "flutter"])
    rows = [h for h in res.head_results if h["head"] == "flutter"]
    assert len(rows) == 1
    row = rows[0]
    assert row["measurement_class"] == "RESEARCH_RHYTHM"
    # the pipeline supplies no respiration channel, so the head abstains
    # rather than flagging on an untested coupling
    assert row["value"]["regular_tachy_flag"] is None
    assert any("coupling could not be tested" in r
               for r in row["reasons"])
    assert row["value"]["user_facing"] is None
    # ... and it never reaches a consumer payload
    assert [r["head"] for r in public_head_results(res.head_results)] == \
        ["afib"]
    # a head ordered BEFORE the decision head sees no verdict and fails
    # closed rather than assuming one
    res2, _ = run_with_details(v, heads=["flutter", "afib"])
    early = [h for h in res2.head_results if h["head"] == "flutter"][0]
    assert early["value"]["regular_tachy_flag"] is None
    assert any("no scan outcome" in r for r in early["reasons"])


def test_head_measures_coupling_against_a_supplied_channel_even_with_a_pipeline_representation():
    """v0.8: the pipeline publishes context["regularity"] built without
    a respiration channel; a research caller that supplies one must get
    the coupling measured, not the stale no-channel abstention."""
    import numpy as np
    from features.regularity import regularity_from_runs
    from heads import get_head
    from scripts.make_synth_regularity import rsa_rr
    rr = rsa_rr(45.0, depth=0.16, resp_brpm=15.0, noise_ms=2.0, seed=1)
    t = np.cumsum(rr)
    runs, times = [np.diff(t) * 1000.0], [t[1:]]
    reg = regularity_from_runs(runs, run_times=times, fps=30.0)   # no channel
    assert reg.structure["coupling"]["status"] == "no_channel"
    tt = np.arange(0, 45.0, 1 / 30.0)
    resp = {"t": tt, "y": np.sin(2 * np.pi * 15.0 / 60.0 * tt),
            "rate_brpm": 15.0, "quality": 0.7}

    class _Lat:
        pass
    lat = _Lat()
    lat.runs, lat.run_times, lat.fps = runs, times, 30.0
    lat.beat_t_s = t
    hr = get_head("flutter").run(lat, {
        "runs": runs, "run_times": times, "scan_outcome": "ACCEPT",
        "respiration": resp, "regularity": reg, "age_years": 40})
    assert not any("no camera-derived respiration rate" in r
                   for r in hr.reasons), hr.reasons
    # the head LOOKED: a decided flag (RSA at ~64 bpm is not in the
    # 2:1 band), not an abstention
    assert hr.value["regular_tachy_flag"] is False, hr.value
    # and a representation with no coupling entry at all still yields
    # an abstention, never a crash, when no channel is supplied
    reg2 = regularity_from_runs(runs, run_times=times, fps=30.0)
    reg2.structure.pop("coupling", None)
    hr2 = get_head("flutter").run(lat, {
        "runs": runs, "run_times": times, "scan_outcome": "ACCEPT",
        "respiration": None, "regularity": reg2, "age_years": 40})
    assert hr2.value["regular_tachy_flag"] is None
