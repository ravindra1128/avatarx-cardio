"""v0.4 T5 — the three session heads and the trend store. Forbidden
output pinned at the head layer: no mL/kg/min number anywhere in any
head value; INFERRED_FITNESS heads are disabled by default; the
beta-blocker hard rule routes fitness to trend-only."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import re

import pytest

from configs import load_config
from datasets.schema import Medications, MeasurementClass, ParticipantContext
from heads.base import enabled_heads, get_head
from trend.store import TrendStore, TrendStoreError

VO2_RE = re.compile(r"m[lL]\s*/\s*kg\s*/\s*min|vo2|vo₂", re.I)


def _ctx(hrr60=28.0, age=44, meds=None, verdict="compliant", **over):
    ctx = {"recovery_metrics": {"hrr60": hrr60, "hrr30": 15.0,
                                "hrr120": None, "hr_end_proxy": 140.0,
                                "recovery_slope_bpm_min": -40.0,
                                "quality": {"n_usable": 30},
                                "reasons": []},
           "protocol_id": "sts_1min",
           "hr_rest_bpm": 72.0, "rr_rest_brpm": 14.0,
           "compliance": {"verdict": verdict},
           "workload": {"available": True, "est_mets": 5.5},
           "participant_context": ParticipantContext(
               age=age, meds=meds or Medications()),
           "trend_sessions": []}
    ctx.update(over)
    return ctx


def test_new_heads_registered_but_never_default_enabled():
    names = [h.name for h in enabled_heads(load_config())]
    for n in ("recovery", "fitness", "trend"):
        assert n not in names
        get_head(n)                                  # registered
    assert get_head("fitness").research_only
    assert get_head("trend").research_only


def test_head_recovery_is_measured_passthrough():
    r = get_head("recovery").run(None, _ctx())
    assert r.measurement_class is MeasurementClass.MEASURED
    assert r.value["hrr60_bpm"] == 28.0
    assert r.value["hr_rest_bpm"] == 72.0
    assert r.value["compliance_verdict"] == "compliant"
    assert r.value["workload_context"]["est_mets"] == 5.5


def test_head_fitness_category_bands_and_gating_reasons():
    h = get_head("fitness")
    lo = h.run(None, _ctx(hrr60=15.0, age=44))       # 40s band p20=20
    assert lo.value["category"] == "below_typical"
    hi = h.run(None, _ctx(hrr60=50.0, age=25))       # 20s band p80=45
    assert hi.value["category"] == "above_typical"
    mid = h.run(None, _ctx(hrr60=30.0, age=44))
    assert mid.value["category"] == "typical"
    assert mid.measurement_class is MeasurementClass.INFERRED_FITNESS
    assert any("§V" in r for r in mid.reasons)
    # non-compliant workload -> no category (comparability is the point)
    nc = h.run(None, _ctx(verdict="repeat"))
    assert nc.value["category"] is None
    # forbidden output: NEVER a VO2 number or unit, anywhere in the value
    for res in (lo, hi, mid, nc):
        assert not VO2_RE.search(json.dumps(res.to_dict())), res.value


def test_beta_blocker_hard_rule_routes_to_trend_only():
    h = get_head("fitness")
    for meds in (Medications(beta_blocker=True), Medications(ccb=True),
                 Medications(ivabradine=True)):
        r = h.run(None, _ctx(meds=meds))
        assert r.value["category"] is None
        assert r.value["routed"] == "trend_only"
        assert any("rate-limiting" in x for x in r.reasons)
    ok = h.run(None, _ctx(meds=Medications(thyroid=True)))
    assert ok.value["category"] is not None          # thyroid: no routing


def test_head_trend_direction_only():
    h = get_head("trend")
    few = h.run(None, _ctx(trend_sessions=[
        {"hrr60_bpm": 20.0}, {"hrr60_bpm": 22.0}]))
    assert few.value["direction"] is None
    assert any(">= 3" in r for r in few.reasons)
    up = h.run(None, _ctx(trend_sessions=[
        {"hrr60_bpm": 20.0}, {"hrr60_bpm": 24.0}, {"hrr60_bpm": 28.0}]))
    assert up.value["direction"] == "improving"
    assert up.value["magnitude_class"] is None       # V5 red: direction only
    down = h.run(None, _ctx(trend_sessions=[
        {"hrr60_bpm": 30.0}, {"hrr60_bpm": 25.0}, {"hrr60_bpm": 21.0}]))
    assert down.value["direction"] == "declining"
    flat = h.run(None, _ctx(trend_sessions=[
        {"hrr60_bpm": 25.0}, {"hrr60_bpm": 25.5}, {"hrr60_bpm": 24.8}]))
    assert flat.value["direction"] == "stable"
    assert not VO2_RE.search(json.dumps(up.to_dict()))


def test_trend_store_versioned_and_exportable(tmp_path):
    p = tmp_path / "store.json"
    s = TrendStore(p)
    s.append(session_id="a", protocol_id="sts_1min", hrr60_bpm=22.0)
    s.append(session_id="b", protocol_id="march_2min", hrr60_bpm=30.0)
    assert len(s.sessions()) == 2
    assert [r["session_id"]
            for r in s.sessions(protocol_id="sts_1min")] == ["a"]
    doc = json.loads(s.export())
    assert doc["store_version"] == 1
    p.write_text(json.dumps({"store_version": 9, "sessions": []}))
    with pytest.raises(TrendStoreError, match="newer"):
        s.sessions()
