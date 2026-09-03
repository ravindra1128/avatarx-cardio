"""M1.1 — the endpoint-head registry (v0.2 invariant 14: extension = new
head, not new pipeline). Heads are plug-ins over the BeatLattice; the
registry + per-head enable flags are the only wiring surface."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from configs import load_config
from datasets.schema import MeasurementClass
from heads.base import (EndpointHead, HeadResult, register_head, get_head,
                        available_heads, enabled_heads, HeadRegistryError)


def test_measurement_classes_are_the_sanctioned_ones():
    # v0.4 (owner-directed schema rev): INFERRED_FITNESS joins the
    # vocabulary — renderable ONLY when §V is green and signed. Any
    # further member is a spec change, not a convenience.
    # v0.4-vascular (owner-directed schema rev, spec B.20):
    # RESEARCH_VASCULAR joins for head_vascular — research-flagged,
    # never renderable while the vascular gates are red/unsigned (V-a).
    # v0.6-flutter (spec B.22): RESEARCH_RHYTHM joins for head_flutter.
    # Deliberately not INFERRED_RHYTHM: app/ renders that class on
    # sight, and an ungated flutter output must be unrenderable by
    # construction rather than by remembering to check (F-b).
    assert {m.value for m in MeasurementClass} == {
        "MEASURED", "INFERRED_RHYTHM", "INFERRED_FITNESS",
        "RESEARCH_SYNTHETIC", "RESEARCH_VASCULAR", "RESEARCH_RHYTHM"}


def test_registry_lists_builtin_heads_and_lookup_works():
    names = available_heads()
    for expected in ("afib", "rate_flags", "rhythm_map"):
        assert expected in names, names
    h = get_head("afib")
    assert isinstance(h, EndpointHead)
    assert h.name == "afib" and h.version
    assert h.required_inputs                        # declares its inputs


def test_unknown_head_and_duplicate_registration_raise():
    with pytest.raises(HeadRegistryError):
        get_head("no_such_head")

    class Dup(EndpointHead):
        name = "afib"                               # collides with builtin
        version = "0"
        required_inputs = ()

        def run(self, lattice, context):
            raise NotImplementedError

    with pytest.raises(HeadRegistryError):
        register_head(Dup)


def test_enabled_heads_follow_config_and_afib_is_mandatory():
    cfg = load_config()
    names = [h.name for h in enabled_heads(cfg)]
    assert "afib" in names                          # the decision head
    cfg2 = load_config()
    cfg2["decision"]["heads"] = {"enabled": ["rate_flags"]}
    with pytest.raises(HeadRegistryError):
        enabled_heads(cfg2)                         # afib cannot be disabled
    cfg3 = load_config()
    cfg3["decision"]["heads"] = {"enabled": ["afib", "bogus"]}
    with pytest.raises(HeadRegistryError):
        enabled_heads(cfg3)                         # unknown name is an error


def test_head_result_carries_measurement_class_and_payload():
    r = HeadResult(head="afib", version="1",
                   measurement_class=MeasurementClass.INFERRED_RHYTHM,
                   value={"predicted_class": None}, confidence=0.5,
                   reasons=["x"])
    d = r.to_dict()
    assert d["measurement_class"] == "INFERRED_RHYTHM"
    assert d["head"] == "afib" and d["value"]["predicted_class"] is None
    with pytest.raises((TypeError, ValueError)):
        HeadResult(head="x", version="1", measurement_class="MEASURED",
                   value={})                        # enum required, not str


def test_abstract_head_cannot_run():
    class NoRun(EndpointHead):
        name = "norun"
        version = "0"
        required_inputs = ()

    with pytest.raises(TypeError):
        NoRun()
