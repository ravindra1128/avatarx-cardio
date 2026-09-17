"""ONE AFib result per completed scan (inference/afib_result.py, owner goal
2026-09-17): AFIB_DETECTED | AFIB_NOT_DETECTED | INCONCLUSIVE, never forced.
"""
import json
import pathlib

import pytest

from inference import afib_result as ar

REPO = pathlib.Path(__file__).resolve().parents[1]
V02 = REPO / "models" / "model_a_v02_45s.json"


def _doc(outcome, cls=None, p=None, *, failed=(), fired=None, reasons=(), stars=3,
         source="video"):
    return {"outcome": outcome, "predicted_class": cls, "afib_probability": p,
            "confidence_stars": stars, "rhythm_source": source,
            "no_read_reasons": list(reasons),
            "debug": {"rationale": {"gates": [{"name": "sqi", "pass": True}],
                                    "gates_failed": list(failed),
                                    "rule": {"fired": fired}}}}


def test_every_result_is_one_of_three():
    for d in (_doc("ACCEPT", "SINUS"), _doc("ACCEPT", "AFIB_SUGGESTIVE"),
              _doc("REPEAT_SCAN"), _doc("NO_RESULT"), {}, {"outcome": "weird"}):
        assert ar.afib_result(d)["result"] in ar.RESULTS


def test_capture_failures_are_inconclusive_capture():
    d = {"outcome": "NO_RESULT", "no_read_reasons": ["no face detected for more than 2 s"],
         "debug": {}}
    r = ar.afib_result(d)
    assert r["result"] == "INCONCLUSIVE" and r["basis"]["category"] == "capture"
    d = _doc("NO_RESULT", reasons=["only 3 clean intervals (< 5)"], failed=["n_intervals_min"])
    r = ar.afib_result(d)
    assert r["result"] == "INCONCLUSIVE" and r["basis"]["category"] == "signal"


def test_signal_abstentions_are_inconclusive_signal_and_irregular_unverified_is_rhythm():
    r = ar.afib_result(_doc("REPEAT_SCAN", failed=["coverage_any_class"],
                            fired="ABSTAIN (quality/evidence gates)",
                            reasons=["only 33% of the scan yielded clean intervals"]))
    assert r["result"] == "INCONCLUSIVE" and r["basis"]["category"] == "signal"
    r = ar.afib_result(_doc("REPEAT_SCAN", failed=["afib_coherence"],
                            fired="ABSTAIN (irregular but unverified)",
                            reasons=["irregular intervals seen, but ..."]))
    assert r["result"] == "INCONCLUSIVE" and r["basis"]["category"] == "rhythm"
    r = ar.afib_result(_doc("REPEAT_SCAN", fired="ABSTAIN (confidence below 3-star floor)",
                            stars=2, reasons=["irregular rhythm seen, but scan confidence 2 of 5"]))
    assert r["result"] == "INCONCLUSIVE" and r["basis"]["category"] == "rhythm"


def test_interim_rules_without_a_probability_decide_by_class_conservatively():
    assert ar.afib_result(_doc("ACCEPT", "SINUS"))["result"] == "AFIB_NOT_DETECTED"
    assert ar.afib_result(_doc("ACCEPT", "HIGH_RATE"))["result"] == "AFIB_NOT_DETECTED"
    assert ar.afib_result(_doc("ACCEPT", "AFIB_SUGGESTIVE"))["result"] == "AFIB_DETECTED"
    r = ar.afib_result(_doc("ACCEPT", "OTHER_IRREGULAR"))
    assert r["result"] == "INCONCLUSIVE" and r["basis"]["category"] == "rhythm"
    r = ar.afib_result(_doc("ACCEPT", None))
    assert r["result"] == "INCONCLUSIVE"


def test_the_probability_band_decides_with_the_classifier():
    band = {"tau_lo": 0.4, "tau_hi": 0.8}
    cfg = {"decision": {"model_a": {"decision_band": band, "version": "t"}}}
    assert ar.afib_result(_doc("ACCEPT", "SINUS", 0.05), cfg)["result"] == "AFIB_NOT_DETECTED"
    r = ar.afib_result(_doc("ACCEPT", "SINUS", 0.55), cfg)
    assert r["result"] == "INCONCLUSIVE" and r["basis"]["category"] == "rhythm"
    assert ar.afib_result(_doc("ACCEPT", "AFIB_SUGGESTIVE", 0.9), cfg)["result"] == "AFIB_DETECTED"
    # an AF class the pipeline let through below tau_hi is not a detection
    r = ar.afib_result(_doc("ACCEPT", "AFIB_SUGGESTIVE", 0.7), cfg)
    assert r["result"] == "INCONCLUSIVE" and r["basis"]["category"] == "rhythm"
    # OTHER_IRREGULAR with a low probability IS "not detected" under the band
    assert ar.afib_result(_doc("ACCEPT", "OTHER_IRREGULAR", 0.1), cfg)["result"] == "AFIB_NOT_DETECTED"


@pytest.mark.skipif(not V02.exists(), reason="model_a_v02_45s.json not built")
def test_the_shipped_artifact_carries_a_measured_band_and_the_layer_reads_it():
    art = json.loads(V02.read_text())
    b = art["decision_band"]
    assert 0.0 < b["tau_lo"] < b["tau_hi"] < 1.0
    assert b["decided_sensitivity"] >= 0.95 and b["decided_specificity"] >= 0.97
    assert b["inconclusive_fraction"] < 0.15
    cfg = {"decision": {"model_a_path": str(V02)}}
    band = ar.decision_band(cfg)
    assert band["tau_lo"] == b["tau_lo"] and band["tau_hi"] == b["tau_hi"]
    assert band["source"] == art["version"]


def test_an_unmeasured_band_is_wide_and_named_as_such(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"type": "logreg", "version": "x"}))
    band = ar.decision_band({"decision": {"model_a_path": str(p)}})
    assert band == ar.DEFAULT_BAND
    assert band["tau_hi"] - band["tau_lo"] >= 0.5


def test_the_layer_never_raises():
    class Bad(dict):
        def get(self, *a, **k):
            raise RuntimeError("boom")
    r = ar.afib_result(Bad())
    assert r["result"] == "INCONCLUSIVE" and "failed safely" in r["basis"]["why"]


def test_launch_config_overrides_are_parsed_and_echoed(monkeypatch):
    from app import measure_api as api
    monkeypatch.setenv("AFIB_CONFIG_OVERRIDES",
                       '{"decision.classifier": "model_a", "decision.model_a_path": "models/model_a_v02_45s.json"}')
    api.LAUNCH_OVERRIDES.pop("config.decision.classifier", None)
    d = api._launch_config_overrides()
    assert d == {"decision.classifier": "model_a",
                 "decision.model_a_path": "models/model_a_v02_45s.json"}
    assert api.LAUNCH_OVERRIDES["config.decision.classifier"] == "model_a"
    monkeypatch.setenv("AFIB_CONFIG_OVERRIDES", "not json")
    assert api._launch_config_overrides() == {}
    assert "IGNORED" in api.LAUNCH_OVERRIDES["config"]


def test_the_sheet_carries_the_result_and_its_basis():
    from app import result_sheet
    row = result_sheet.row_from_doc(
        {"outcome": "ACCEPT", "afib_result": "AFIB_NOT_DETECTED", "afib_probability": 0.0071,
         "afib_result_basis": {"category": None, "why": "AF probability 0.01 <= 0.47"}}, extra={})
    assert row["AFib Result"] == "AFIB_NOT_DETECTED" and row["AFib p"] == 0.007
    assert row["AFib Basis"] == "AF probability 0.01 <= 0.47"
    row = result_sheet.row_from_doc(
        {"outcome": "REPEAT_SCAN", "afib_result": "INCONCLUSIVE",
         "afib_result_basis": {"category": "signal", "why": "only 33% ..."}}, extra={})
    assert row["AFib Basis"].startswith("signal: only 33%")
    for c in ("AFib Result", "AFib p", "AFib Basis"):
        assert c in result_sheet.COLUMNS


def test_sheet_reorder_on_start_is_opt_in_and_echoed(monkeypatch):
    from app import measure_api as api
    calls = []
    monkeypatch.setattr(api.result_sheet, "is_configured", lambda: True)
    monkeypatch.setattr(api.result_sheet, "reorder_existing_tab", lambda: (calls.append(1), "reordered 3 row(s); 90 columns")[1])
    monkeypatch.delenv("AFIB_SHEET_REORDER_ON_START", raising=False)
    api.LAUNCH_OVERRIDES.pop("sheet.reorder", None)
    api._maybe_reorder_sheet()
    assert calls == [] and "sheet.reorder" not in api.LAUNCH_OVERRIDES
    monkeypatch.setenv("AFIB_SHEET_REORDER_ON_START", "1")
    api._maybe_reorder_sheet()
    assert calls == [1] and api.LAUNCH_OVERRIDES["sheet.reorder"].startswith("reordered 3 row(s)")
    api.LAUNCH_OVERRIDES.pop("sheet.reorder", None)
