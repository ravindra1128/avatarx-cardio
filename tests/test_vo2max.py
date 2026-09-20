"""The fitness card's profile basis: a published non-exercise VO2max estimate.

Pinned here, in the order the scan flows:
  the equation and its conditioning  (features/vo2max.py)
  the card on both bases             (features/hemodynamics.py)
  the payload row                    (app/report_data.py)
  the request plumbing               (app/measure_api.py)
  the ShenAI route's reconcile step  (inference/shenai_route.py)

The repeatability requirement is stated as a property of the estimator: the
resting rates one seated person really produced within twenty minutes (the
tracking sheet's reference column) must not move the estimate by more than a
fraction of the owner's +/-8 target - without any smoothing, caching or
clipping, which the tests below would not be able to tell from a bug.
"""
import io
import json
import os

import pytest

os.environ.setdefault("AFIB_SHEET_ID", "test-sheet")

from app import measure_api as api  # noqa: E402
from app.report_data import report_biomarkers  # noqa: E402
from features import vo2max  # noqa: E402
from features.hemodynamics import (  # noqa: E402
    cardiorespiratory_indices, resting_hemodynamics, resting_rate_index)
from inference.shenai_route import reconcile_fitness_rate  # noqa: E402

PROFILE = {"age": 35, "sex": "male", "height_cm": 178, "weight_kg": 76, "activity_level": 3}
CLEAN = dict(rate_method="clean_interval_median", rate_intervals=20)


class _Reg:
    dispersion = {"rmssd_ms": 30.0, "sdnn_ms": 40.0}


# ------------------------------------------------------------------ equation
def test_the_published_equation_term_by_term():
    p, problems = vo2max.normalize_profile(PROFILE)
    assert problems == [] and p["bmi"] == pytest.approx(76 / 1.78 ** 2, abs=0.01)
    e = vo2max.estimate_vo2max(p, 70.0)
    mets = 18.07 + 2.77 - 0.10 * 35 - 0.17 * p["bmi"] - 0.03 * 70 + 1.06
    assert e["available"] and e["mets"] == pytest.approx(mets, abs=0.01)
    assert e["vo2max_ml_kg_min"] == pytest.approx(3.5 * mets, abs=0.05)
    assert sum(e["terms_ml_kg_min"].values()) == pytest.approx(e["vo2max_ml_kg_min"], abs=0.1)
    # who supplied what is part of the payload, never implied
    assert e["input_sources"]["resting_hr_bpm"] == "face_scan"
    assert all(v.startswith("user_entered") for k, v in e["input_sources"].items()
               if k != "resting_hr_bpm")


def test_a_woman_reads_the_sex_term_lower_and_activity_raises_it():
    man, _ = vo2max.normalize_profile(PROFILE)
    woman, _ = vo2max.normalize_profile(dict(PROFILE, sex="Female"))
    active, _ = vo2max.normalize_profile(dict(PROFILE, activity_level=5))
    v = lambda p: vo2max.estimate_vo2max(p, 70.0)["vo2max_ml_kg_min"]      # noqa: E731
    assert v(man) - v(woman) == pytest.approx(2.77 * 3.5, abs=0.1)
    assert v(active) - v(man) == pytest.approx((3.03 - 1.06) * 3.5, abs=0.1)


def test_the_estimate_is_well_conditioned_in_the_resting_rate():
    """The legacy card moved 1.8 points per bpm; this moves ~0.1 unit per bpm,
    because that is the rate's validated partial effect - not a damping."""
    p, _ = vo2max.normalize_profile(PROFILE)
    v = lambda hr: vo2max.estimate_vo2max(p, hr)["vo2max_ml_kg_min"]      # noqa: E731
    assert v(70.0) - v(74.0) == pytest.approx(0.42, abs=0.06)
    # One person, 17 minutes, reference rate 67.6 -> 77.2 bpm (2026-09-16 g3):
    # the legacy card moved 17 points on a PERFECT rate; this moves one unit.
    assert abs(v(67.6) - v(77.2)) < 1.2
    assert abs(100 * resting_rate_index(67.6) - 100 * resting_rate_index(77.2)) > 15
    # Even our worst fold (84 read as 48) stays inside the +/-8 target.
    assert abs(v(48.0) - v(84.0)) < 4.0
    # ... and the scale still separates PEOPLE: nothing is squeezed to a mean.
    unfit, _ = vo2max.normalize_profile(dict(PROFILE, age=62, weight_kg=104, activity_level=1))
    fit, _ = vo2max.normalize_profile(dict(PROFILE, age=24, weight_kg=68, activity_level=5))
    assert vo2max.estimate_vo2max(fit, 55.0)["vo2max_ml_kg_min"] \
        - vo2max.estimate_vo2max(unfit, 82.0)["vo2max_ml_kg_min"] > 25


@pytest.mark.parametrize("bad, phrase", [
    ({}, "no profile"),
    (dict(PROFILE, age=None), "age"),
    (dict(PROFILE, age=12), "age"),
    (dict(PROFILE, sex="other"), "sex"),
    (dict(PROFILE, height_cm=17.8), "height"),
    (dict(PROFILE, weight_kg=0), "weight"),
    (dict(PROFILE, activity_level=6), "activity"),
    (dict(PROFILE, activity_level=2.5), "activity"),
    (dict(PROFILE, height_cm=229, weight_kg=31), "body-mass"),
])
def test_an_incomplete_or_implausible_profile_is_no_profile(bad, phrase):
    p, problems = vo2max.normalize_profile(bad)
    assert p is None and any(phrase in x for x in problems)


def test_a_problem_names_the_field_and_never_echoes_what_was_typed():
    for field, a, b in (("age", 12, 140), ("height_cm", 17.8, 400), ("weight_kg", 0, 999),
                        ("activity_level", 6, 2.5), ("sex", "other", "prefer not to say")):
        pa = vo2max.normalize_profile(dict(PROFILE, **{field: a}))[1]
        pb = vo2max.normalize_profile(dict(PROFILE, **{field: b}))[1]
        assert pa == pb and len(pa) == 1, field


def test_inputs_outside_the_validated_cohorts_are_flagged_not_clipped():
    p, _ = vo2max.normalize_profile(dict(PROFILE, age=82))
    e = vo2max.estimate_vo2max(p, 112.0)
    assert e["available"] and set(e["extrapolated"]) == {"age", "resting_hr"}
    # where the line leaves physiology the card abstains instead of clipping
    p, _ = vo2max.normalize_profile(dict(PROFILE, sex="female", age=94, height_cm=150,
                                         weight_kg=150, activity_level=1))
    e = vo2max.estimate_vo2max(p, 110.0)
    assert not e["available"] and "no estimate" in e["reason"]


def test_the_reference_range_is_context_for_the_estimate_never_an_input():
    ref = vo2max.reference_percentiles(35, "male")
    assert (ref["p25"], ref["p50"], ref["p75"]) == (35.9, 42.4, 49.2) and ref["age_band"] == "30-39"
    assert vo2max.reference_percentiles(62, "female")["p50"] == 20.0
    assert vo2max.reference_percentiles(19, "male") is None
    assert vo2max.reference_percentiles(80, "female") is None
    # an 85-year-old still gets an estimate; there is simply no reference row for it
    p, _ = vo2max.normalize_profile(dict(PROFILE, age=85))
    e = vo2max.estimate_vo2max(p, 70.0)
    assert e["available"] and e["reference"] is None
    young, _ = vo2max.normalize_profile(dict(PROFILE, age=29))
    old, _ = vo2max.normalize_profile(dict(PROFILE, age=30))
    a, b = vo2max.estimate_vo2max(young, 70.0), vo2max.estimate_vo2max(old, 70.0)
    assert a["reference"]["p50"] != b["reference"]["p50"]                 # the band changes ...
    assert a["vo2max_ml_kg_min"] - b["vo2max_ml_kg_min"] == pytest.approx(0.35, abs=0.06)  # ... the line does not jump


# ------------------------------------------------------------- one rate
def test_rate_choice_ranks_two_readings_of_the_same_scan():
    pick = vo2max.select_resting_rate
    only = pick(72.0, "clean_interval_median", None)
    assert only["bpm"] == 72.0 and only["source"] == "clean_interval_median"
    # the live-frame rate leads; the clip's rate corroborates it within 15 %
    agree = pick(74.0, "clean_interval_median", 72.0, "shenai")
    assert agree["bpm"] == 72.0 and agree["corroborated"] is True and agree["reason"] is None
    assert agree["own_bpm"] == 74.0 and agree["source"] == "live_frame:shenai"
    fold = pick(48.0, "waveform_rhythm", 84.0, "shenai")
    assert fold["corroborated"] is False and fold["disagreement"] > 0.4
    assert fold["bpm"] == 84.0 and fold["source"] == "live_frame:shenai"
    assert "48" in fold["reason"] and "84" in fold["reason"]
    rescue = pick(None, None, 70.0, "shenai")
    assert rescue["bpm"] == 70.0 and rescue["corroborated"] is False
    assert pick(None, None, None)["bpm"] is None
    assert pick(None, None, 400.0)["bpm"] is None            # not a resting pulse


# ------------------------------------------------------------------ the card
def test_without_a_profile_the_card_is_the_proxy_it_always_was():
    out = cardiorespiratory_indices(_Reg(), 70.0, None, ref_bpm=90.0, **CLEAN)
    assert out["estimate"]["unit"] == "/100"
    assert out["estimate"]["value"] == round(100 * resting_rate_index(70.0), 1)
    assert out["oxygen_uptake_estimate"] is None and out["tier"] == "measured"
    assert out["estimator"] == "resting_rate_logistic"
    # the payload's formula now names the constants the code uses
    assert "(72-HR)/14" in out["estimate"]["method"]
    # a profile that cannot feed the equation is no profile - and says why
    bad = cardiorespiratory_indices(_Reg(), 70.0, dict(PROFILE, sex="other"), **CLEAN)
    assert bad["estimate"]["unit"] == "/100"
    assert any("sex" in x for x in bad["profile_estimate_unavailable"])
    # the proxy payload stays FENCED: no oxygen-uptake token as a value or a key
    import re
    fence = re.compile(r"m[lL]\s*/\s*kg\s*/\s*min|vo2|vo₂", re.IGNORECASE)
    assert not fence.search(json.dumps(out)) and not fence.search(json.dumps(bad))


def test_two_agreeing_readings_are_measured_and_quote_the_pages_heart_rate():
    out = cardiorespiratory_indices(_Reg(), 74.0, PROFILE, ref_bpm=72.0,
                                    reference_source="shenai", **CLEAN)
    assert out["tier"] == "measured" and out["tier_reasons"] == []
    assert out["raw_value"] == 72.0 and out["resting_rate_source"] == "live_frame:shenai"
    assert out["resting_rate_choice"]["own_bpm"] == 74.0          # ours stays on record
    # a THIN count that happens to agree does not become "measured"
    thin = cardiorespiratory_indices(_Reg(), 74.0, PROFILE, ref_bpm=72.0, reference_source="shenai",
                                     rate_method="clean_interval_median", rate_intervals=7)
    assert thin["tier"] == "provisional" and "7 clean beat intervals" in thin["tier_reasons"][0]
    assert thin["estimate"]["value"] == out["estimate"]["value"]


def test_with_a_profile_the_card_is_a_vo2max_estimate():
    out = cardiorespiratory_indices(_Reg(), 70.0, PROFILE, **CLEAN)
    est = out["estimate"]
    assert est["unit"] == "mL/kg/min" and est["name"] == "vo2max_nonexercise_jurca2005"
    assert est["value"] == out["oxygen_uptake_estimate"] == out["vo2max"]["vo2max_ml_kg_min"]
    assert out["tier"] == "measured" and out["resting_rate_source"] == "clean_interval_median"
    assert out["likely_range"][0] < est["value"] < out["likely_range"][1]
    assert out["rmssd_ms"] == 30.0 and "never enters" in est["method"]   # reported, not used
    assert out["fitness_proxy_score"] == round(100 * resting_rate_index(70.0), 1)


def test_the_clips_own_rate_logic_is_shared_by_both_bases():
    """A thin count is provisional on either basis; the profile changes the
    estimator, never what the clip is believed to have measured."""
    kw = dict(rate_method="clean_interval_median", rate_intervals=7)
    a = cardiorespiratory_indices(_Reg(), 70.0, None, **kw)
    b = cardiorespiratory_indices(_Reg(), 70.0, PROFILE, **kw)
    assert a["tier"] == b["tier"] == "provisional"
    assert a["tier_reasons"] == b["tier_reasons"] and a["raw_value"] == b["raw_value"]


def test_a_folded_clip_rate_is_overruled_by_the_live_frame_rate():
    out = cardiorespiratory_indices(_Reg(), 48.0, PROFILE, ref_bpm=84.0,
                                    reference_source="shenai", **CLEAN)
    right = cardiorespiratory_indices(_Reg(), 84.0, PROFILE, **CLEAN)
    assert out["estimate"]["value"] == right["estimate"]["value"]
    assert out["tier"] == "provisional" and out["resting_rate_source"] == "live_frame:shenai"
    assert out["raw_value"] == 84.0 and out["resting_rate_choice"]["own_bpm"] == 48.0


def test_a_scan_whose_clip_abstained_still_estimates_from_the_live_frame_rate():
    # the doubling signature with no backing: the proxy basis abstains ...
    kw = dict(rate_method="clean_interval_median", rate_intervals=20, spectral_hr_bpm=50.0,
              harmonic_fraction=0.05, spectral_snr=3.0, spectral_roi_agree=1)
    proxy = cardiorespiratory_indices(_Reg(), 100.0, None, ref_bpm=74.0, **kw)
    assert proxy["available"] is False and proxy["reason_code"] == "resting_rate_unverified"
    # ... the profile basis answers from the other reading of the same scan
    out = cardiorespiratory_indices(_Reg(), 100.0, PROFILE, ref_bpm=74.0,
                                    reference_source="shenai", **kw)
    assert out["available"] and out["raw_value"] == 74.0 and out["tier"] == "provisional"
    # and with no rate from either side it abstains, in the pipeline's words
    none = cardiorespiratory_indices(_Reg(), 100.0, PROFILE, **kw)
    assert none["available"] is False and none["estimate"] is None
    assert none["reason_code"] == "resting_rate_unverified" and none["tier"] is None


def test_no_beat_lattice_at_all_is_still_a_completed_scan_on_the_profile_basis():
    ref = {"ref_hr": 63.5, "ref_source": "shenai"}
    hemo = resting_hemodynamics({}, outcome="NO_RESULT", participant=PROFILE, reference=ref)
    assert hemo["available"] is False                       # the other cards stay blank
    fit = hemo["cardiorespiratory_fitness"]
    assert fit["available"] and fit["raw_value"] == 63.5 and fit["tier"] == "provisional"
    # neither input alone is enough
    assert "cardiorespiratory_fitness" not in resting_hemodynamics(
        {}, outcome="NO_RESULT", participant=PROFILE)
    assert "cardiorespiratory_fitness" not in resting_hemodynamics(
        {}, outcome="NO_RESULT", reference=ref)


# ------------------------------------------------------------------- payload
class _Scan:
    outcome = "NO_RESULT"


def test_the_payload_row_carries_the_estimate_and_its_basis():
    bio = report_biomarkers(_Scan(), {}, participant=PROFILE,
                            reference={"ref_hr": 70.0, "ref_source": "shenai"})
    rows = {r["key"]: r for r in bio["items"]}
    fit = rows["cardiorespiratory_fitness"]
    assert fit["status"] == "computed" and fit["unit"] == "mL/kg/min"
    assert fit["value"] == fit["oxygen_uptake_ml_kg_min"] == fit["details"]["oxygen_uptake_estimate"]
    assert fit["estimator"] == "jurca2005" and len(fit["likely_range"]) == 2
    assert fit["raw"] == {"name": "resting_heart_rate", "value": 70.0, "unit": "bpm"}
    assert "profile" in fit["limitation"] and "training" in fit["limitation"]
    # typical range: the FRIEND quartiles of MEASURED VO2max for men aged 30-39
    assert fit["typical_range"] == [35.9, 49.2] and "men aged 30-39" in fit["typical_range_label"]
    assert rows["arterial_stiffness"]["status"] == "not_computed"
    assert bio["complete"] is False
    # frozen contract: every key the webapp reads is still there
    for k in ("key", "title", "label", "status", "value", "unit", "tier", "tier_reasons",
              "typical_range", "raw", "confidence", "details"):
        assert k in fit


# ------------------------------------------------------------------ plumbing
def test_participant_block_keeps_only_what_the_equation_reads():
    doc = {"participant": dict(PROFILE, name="someone", email="a@b.c", notes={"x": 1},
                               sex="male" + "x" * 40, age=True)}
    got = api._participant_block(doc)
    assert set(got) <= set(api.PARTICIPANT_KEYS)
    assert "name" not in got and "sex" not in got and "age" not in got
    assert api._participant_block({"participant": "male"}) is None
    assert api._participant_block(None) is None


def _handler(path="/", body=b""):
    h = api.MeasureHandler.__new__(api.MeasureHandler)
    h.path = path
    h.rfile = io.BytesIO(body)
    h.headers = {"Content-Length": str(len(body)), "User-Agent": "t"}
    h.close_connection = False
    h.sent = []
    h._json = lambda code, doc: h.sent.append((code, doc))
    return h


def test_the_profile_travels_in_the_start_body_never_the_query(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "parts", raising=False)
    monkeypatch.setattr(api.result_sheet, "schedule_append", lambda *a, **k: None)
    monkeypatch.setattr(api._POOL, "submit", lambda fn: fn())           # run the job inline
    seen = {}
    monkeypatch.setattr(api.MeasureHandler, "_run_assembled", staticmethod(
        lambda path, header: seen.update(header=dict(header)) or {"outcome": "NO_RESULT"}))
    uid = "profile-upload"
    up = _handler("/api/upload-part", b"x")
    up._upload_part({"upload_id": [uid], "index": ["0"], "total": ["1"]})
    body = json.dumps({"participant": dict(PROFILE, name="someone")}).encode()
    h = _handler("/api/start", body)
    h._start_job({"upload_id": [uid], "ref_hr": ["71.5"], "ref_source": ["shenai"],
                  "age": ["99"], "sex": ["female"]})                     # query: ignored
    assert h.sent[-1][0] == 202
    assert seen["header"]["participant"] == PROFILE
    assert seen["header"]["reference"] == {"ref_hr": 71.5, "ref_source": "shenai"}


@pytest.mark.parametrize("body", [b"", b"not json", b"[1, 2]", b'{"participant": 3}',
                                  b"{" + b" " * (api.MAX_START_BODY_BYTES + 1) + b"}"])
def test_a_bad_start_body_is_no_profile_never_a_failed_scan(body, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "parts", raising=False)
    monkeypatch.setattr(api.result_sheet, "schedule_append", lambda *a, **k: None)
    monkeypatch.setattr(api._POOL, "submit", lambda fn: fn())
    seen = {}
    monkeypatch.setattr(api.MeasureHandler, "_run_assembled", staticmethod(
        lambda path, header: seen.update(header=dict(header)) or {"outcome": "NO_RESULT"}))
    uid = "bad-body-upload"
    up = _handler("/api/upload-part", b"x")
    up._upload_part({"upload_id": [uid], "index": ["0"], "total": ["1"]})
    h = _handler("/api/start", body)
    h._start_job({"upload_id": [uid]})
    assert h.sent[-1][0] == 202 and "participant" not in seen["header"]


def test_measure_video_hands_profile_and_reference_to_the_cards_only():
    import inspect
    src = inspect.getsource(api.measure_video_details)
    assert "participant=participant" in src and "reference=reference" in src
    # the pipeline call itself never sees either
    call = src[src.index("run_with_details("):]
    call = call[:call.index(")") + 1]
    assert "participant" not in call and "reference" not in call


# ----------------------------------------------------------- route reconcile
def _doc_with_card(card_row, pulse=None):
    return {"mean_pulse_rate_bpm": pulse, "biomarkers": {"items": [card_row], "complete": False}}


def _row(participant, hr, **kw):
    bio = report_biomarkers(_Scan(), {}, hemodynamics={
        "available": True, "cardiorespiratory_fitness":
            cardiorespiratory_indices(_Reg(), hr, participant, **kw)})
    return next(r for r in bio["items"] if r["key"] == "cardiorespiratory_fitness")


def test_reconcile_recomputes_the_estimate_where_the_proxy_would_abstain():
    rec = {"train_ok": True, "used": False, "train": {"bpm": 74.0}}
    # proxy basis, clip read 98 against a sound 74 bpm train: withheld, as before
    proxy = _row(None, 98.0, **CLEAN)
    out = reconcile_fitness_rate(_doc_with_card(proxy), rec)
    assert out["action"] == "abstained" and proxy["status"] == "not_computed"
    # profile basis, same scan: recomputed on the train's rate
    card = _row(PROFILE, 98.0, **CLEAN)
    before = card["value"]
    out = reconcile_fitness_rate(_doc_with_card(card), rec)
    assert out["action"] == "recomputed" and card["status"] == "computed"
    assert card["raw"]["value"] == 74.0 and card["tier"] == "provisional"
    assert card["value"] == pytest.approx(before + 0.105 * (98 - 74), abs=0.15)
    assert card["details"]["resting_rate_source"] == "live_frame:shenai_train"
    assert card["value"] == card["oxygen_uptake_ml_kg_min"] == card["details"]["oxygen_uptake_estimate"]


def test_reconcile_leaves_an_agreeing_estimate_alone_and_fills_a_missing_one():
    rec = {"train_ok": True, "used": False, "train": {"bpm": 74.0}}
    # the request already carried the SDK's rate: the train is the same SDK's
    # second account of the same beats, so the card is left alone
    card = _row(PROFILE, 72.0, ref_bpm=73.0, reference_source="shenai", **CLEAN)
    before = dict(card)
    assert reconcile_fitness_rate(_doc_with_card(card), rec)["action"] == "agree"
    assert card["value"] == before["value"] and card["tier"] == "measured"
    assert card["raw"]["value"] == 73.0
    # no reference in the request, clip and train agree: the train's rate, tier kept
    card = _row(PROFILE, 72.0, **CLEAN)
    assert reconcile_fitness_rate(_doc_with_card(card), rec)["action"] == "recomputed"
    assert card["raw"]["value"] == 74.0 and card["tier"] == "measured"
    # the clip gave no rate and the request carried no reference: the train fills it
    empty = _row(PROFILE, None)
    assert empty["status"] == "not_computed"
    doc = _doc_with_card(empty)
    assert reconcile_fitness_rate(doc, rec)["action"] == "recomputed"
    assert empty["status"] == "computed" and empty["raw"]["value"] == 74.0
    assert "reason" not in empty and doc["biomarkers"]["complete"] is True
    # an unsound train never touches the card
    assert reconcile_fitness_rate(_doc_with_card(_row(PROFILE, 98.0, **CLEAN)),
                                  dict(rec, train_ok=False))["action"] == "none"


def test_reconcile_follows_the_routes_published_pulse_when_the_route_is_used():
    rec = {"train_ok": True, "used": True, "train": {"bpm": 84.4}}
    card = _row(PROFILE, 48.0, **CLEAN)
    out = reconcile_fitness_rate(_doc_with_card(card, pulse=84.0), rec)
    assert out["action"] == "recomputed" and card["raw"]["value"] == 84.0
    # one rate per scan (owner, 2026-09-17): even a card already on the SDK's
    # own heart rate follows the pulse the rhythm result publishes
    card = _row(PROFILE, 75.0, ref_bpm=80.7, reference_source="shenai", **CLEAN)
    assert card["raw"]["value"] == 80.7
    out = reconcile_fitness_rate(_doc_with_card(card, pulse=80.8), rec)
    assert out["action"] == "recomputed" and card["raw"]["value"] == 80.8
    assert card["details"]["resting_rate_source"] == "live_frame:shenai_train_via_resolver"
