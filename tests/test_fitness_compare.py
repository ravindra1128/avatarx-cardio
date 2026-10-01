"""Test scans: formula vs manual profile vs ShenAI profile, side by side.

Owner, 2026-10-01: keep the fitness card's name, formula and calculation
exactly as they are; on TEST scans record two more readings beside it - the
non-exercise VO2max equation on the details the tester typed, and the same
equation on ShenAI's own age/BMI estimate - so the three can be compared in the
sheet. Pinned here, in the order a scan flows:

  the comparison itself            (features/fitness_compare.py)
  the card is only READ, never moved
  the start body carries both profiles, whitelisted   (app/measure_api.py)
  the six sheet columns            (app/result_sheet.py)
"""
import copy
import io
import json
import os

import pytest

os.environ.setdefault("AFIB_SHEET_ID", "test-sheet")

from app import measure_api as api  # noqa: E402
from app import result_sheet  # noqa: E402
from features import vo2max  # noqa: E402
from features.fitness_compare import compare, shenai_inputs  # noqa: E402
from features.hemodynamics import resting_rate_index  # noqa: E402

MANUAL = {"participant_id": "P1", "label": "RB", "age": 35, "sex": "male",
          "height_cm": 178, "weight_kg": 76, "activity_level": 3}
SHENAI = {"age_years": 41.2, "bmi_kg_per_m2": 26.3, "height_cm": 176.0, "weight_kg": 81.5}


def _card(hr=64.5, status="computed", source="live_frame:shenai"):
    value = round(100 * resting_rate_index(hr), 1) if status == "computed" else None
    return {"key": "cardiorespiratory_fitness", "title": "VO₂ Max / Cardiorespiratory Fitness Estimate",
            "status": status, "value": value, "unit": "/100", "tier": "provisional",
            "raw": {"name": "resting_heart_rate", "value": hr, "unit": "bpm"},
            "details": {"resting_rate_source": source}}


# ------------------------------------------------------------- the comparison
def test_three_readings_on_one_heart_rate():
    card = _card()
    out = compare(card, MANUAL, SHENAI)
    # 1) the formula: exactly what the card shows
    assert out["formula"]["score"] == card["value"] == round(100 * resting_rate_index(64.5), 1)
    assert out["formula"]["hr_bpm"] == 64.5 and out["formula"]["unit"] == "/100"
    # 2) the manual profile through the equation, on the card's heart rate
    p, _ = vo2max.normalize_profile(MANUAL)
    assert out["manual"]["vo2max_ml_kg_min"] == vo2max.estimate_vo2max(p, 64.5)["vo2max_ml_kg_min"]
    assert out["manual"]["participant_id"] == "P1" and out["manual"]["label"] == "RB"
    assert out["manual"]["summary"].startswith("male, 35 y, 178 cm, 76 kg")
    # 3) ShenAI's age and BMI; sex and activity borrowed from the manual profile
    want = vo2max.estimate_vo2max({"age_years": 41.2, "sex": "male", "bmi": 26.3, "activity_level": 3}, 64.5)
    assert out["shenai"]["vo2max_ml_kg_min"] == want["vo2max_ml_kg_min"]
    assert out["shenai"]["borrowed_from_manual"] == ["sex", "activity_level"]
    assert out["shenai"]["summary"] == "41 y, BMI 26.3, 176 cm, 82 kg"


def test_where_shenai_and_the_manual_profile_differ():
    out = compare(_card(), MANUAL, SHENAI)
    d = out["differences_shenai_minus_manual"]
    assert d["age_years"] == 6.2 and d["height_cm"] == -2.0 and d["weight_kg"] == 5.5
    assert d["bmi"] == round(26.3 - 76 / 1.78 ** 2, 1)
    assert d["vo2max_ml_kg_min"] == round(out["shenai"]["vo2max_ml_kg_min"]
                                          - out["manual"]["vo2max_ml_kg_min"], 1)
    note = out["note"]
    assert note.startswith(f"formula {out['formula']['score']:.1f}/100 | manual ")
    assert "all on 64 bpm" in note and "ShenAI minus manual: age +6.2 y" in note


def test_the_card_is_only_read():
    card = _card()
    before = copy.deepcopy(card)
    compare(card, MANUAL, SHENAI)
    assert card == before


@pytest.mark.parametrize("manual, shenai, card, m_reason, s_reason", [
    (None, SHENAI, _card(), "no manual profile", "no sex or activity level"),
    (MANUAL, None, _card(), None, "no age and BMI"),
    (MANUAL, {"height_cm": 176.0}, _card(), None, "no age and BMI"),
    (MANUAL, SHENAI, _card(status="not_computed"), "no resting heart rate", "no resting heart rate"),
    (dict(MANUAL, sex="other"), SHENAI, _card(), "sex", "no sex or activity level"),
    (MANUAL, dict(SHENAI, age_years=12.0), _card(), None, "outside the equation's input range"),
])
def test_a_missing_reading_says_why(manual, shenai, card, m_reason, s_reason):
    out = compare(card, manual, shenai)
    if m_reason:
        assert out["manual"]["available"] is False and m_reason in out["manual"]["reason"]
    else:
        assert out["manual"]["available"] is True
    assert out["shenai"]["available"] is False and s_reason in out["shenai"]["reason"]
    assert out["formula"]["score"] == card["value"]          # the card's reading is always there


def test_shenai_bmi_falls_back_to_its_own_height_and_weight_only():
    s = shenai_inputs({"age_years": 40, "height_cm": 180, "weight_kg": 81})
    assert s["bmi"] == pytest.approx(25.0) and s["bmi_source"] == "shenai_height_weight"
    assert shenai_inputs({"age_years": 40})["bmi"] is None    # never the tester's numbers


# --------------------------------------------------------------- the service
def test_the_comparison_never_moves_the_card():
    card = _card()
    doc = {"biomarkers": {"items": [copy.deepcopy(card)]}}
    api._apply_fitness_comparison(doc, {"test_profile": MANUAL, "shenai_profile": SHENAI})
    assert doc["biomarkers"]["items"][0] == card
    assert doc["fitness_comparison"]["formula"]["score"] == card["value"]
    # and the profile never reaches the card's own basis: it is not `participant`
    assert api._participant_block({"test_profile": MANUAL, "shenai_profile": SHENAI}) is None


def test_no_profiles_means_no_comparison():
    doc = {"biomarkers": {"items": [_card()]}}
    api._apply_fitness_comparison(doc, {})
    assert "fitness_comparison" not in doc


def test_only_plain_short_scalars_survive_the_whitelist():
    raw = {"test_profile": dict(MANUAL, name="Someone Long", email="a@b.c", age=True,
                                label="x" * 60, height_cm=float("nan"), extra={"k": 1}),
           "shenai_profile": dict(SHENAI, age_years="41", heartbeats=[1, 2])}
    tp, sp = api._test_profile_block(raw), api._shenai_profile_block(raw)
    assert set(tp) == {"participant_id", "sex", "weight_kg", "activity_level"}
    assert set(sp) == {"bmi_kg_per_m2", "height_cm", "weight_kg"}


def _handler(path="/", body=b""):
    h = api.MeasureHandler.__new__(api.MeasureHandler)
    h.path = path
    h.rfile = io.BytesIO(body)
    h.headers = {"Content-Length": str(len(body)), "User-Agent": "t"}
    h.close_connection = False
    h.sent = []
    h._json = lambda code, doc: h.sent.append((code, doc))
    return h


def test_both_profiles_travel_in_the_start_body(tmp_path, monkeypatch):
    from app.result_store import ResultStore
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "parts", raising=False)
    # Finished results persist in a SQLite store under the work dir; without
    # this a rerun finds this upload id "complete" and answers 200, not 202.
    monkeypatch.setattr(api, "_RESULT_STORE", ResultStore(tmp_path / "results.sqlite", api.RESULT_TTL_S))
    monkeypatch.setattr(api.result_sheet, "schedule_append", lambda *a, **k: None)
    monkeypatch.setattr(api._POOL, "submit", lambda fn: fn())
    seen = {}
    monkeypatch.setattr(api.MeasureHandler, "_run_assembled", staticmethod(
        lambda path, header: seen.update(header=dict(header)) or {"outcome": "NO_RESULT"}))
    uid = f"compare-upload-{os.getpid()}-{id(seen)}"
    _handler("/api/upload-part", b"x")._upload_part({"upload_id": [uid], "index": ["0"], "total": ["1"]})
    body = json.dumps({"test_profile": MANUAL, "shenai_profile": SHENAI}).encode()
    h = _handler("/api/start", body)
    h._start_job({"upload_id": [uid], "ref_hr": ["64.5"], "ref_source": ["shenai"]})
    assert h.sent[-1][0] == 202
    assert seen["header"]["test_profile"] == MANUAL
    assert seen["header"]["shenai_profile"] == SHENAI
    assert "participant" not in seen["header"]              # the card's basis is untouched


def test_the_assembled_job_records_the_comparison_after_the_card_is_final():
    import inspect
    src = inspect.getsource(api.MeasureHandler._run_assembled)
    assert src.index("_apply_shenai_route(") < src.index("_apply_fitness_comparison(doc, header)")
    call = src[src.index("measure_video_details("):]
    call = call[:call.index("reference=header.get(\"reference\"))")]
    assert "test_profile" not in call and "shenai_profile" not in call


# ----------------------------------------------------------------- the sheet
def test_six_comparison_columns_at_the_end_of_the_sheet():
    assert result_sheet.COLUMNS[-6:] == ["Test Participant", "Manual Profile", "Manual VO2",
                                         "ShenAI Profile", "ShenAI VO2", "Compare Note"]
    assert "Fitness" in result_sheet.COLUMNS                 # the formula column is unchanged


def test_a_test_scan_fills_them_and_a_normal_scan_leaves_them_blank():
    fc = compare(_card(), MANUAL, SHENAI)
    row = result_sheet.row_from_doc({"fitness_comparison": fc, "outcome": "REPEAT_SCAN"})
    assert row["Test Participant"] == "P1 · RB"
    assert row["Manual Profile"].startswith("male, 35 y")
    assert row["Manual VO2"] == round(fc["manual"]["vo2max_ml_kg_min"], 1)
    assert row["ShenAI Profile"] == "41 y, BMI 26.3, 176 cm, 82 kg"
    assert row["ShenAI VO2"] == round(fc["shenai"]["vo2max_ml_kg_min"], 1)
    assert row["Compare Note"] == fc["note"][:300]
    normal = result_sheet.row_from_doc({"outcome": "REPEAT_SCAN"})
    assert not any(normal.get(c) for c in result_sheet.COLUMNS[-6:])
