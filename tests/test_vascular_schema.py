"""v0.4-vascular T2 — stiffness ground-truth schema: the session-level
cfPWV reference block, the synchronized contact-PPG sidecar, the
tonometry-export parser, the RESEARCH_VASCULAR measurement class, and
dataset registration of the new sidecar files. Everything fails closed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import pytest

from datasets.schema import (ContactPpg, MeasurementClass, VascularReference,
                             contact_ppg_from_dict, vascular_from_dict)


def _ref(**over):
    d = {"cfpwv_mps": 8.4, "device_model": "SphygmoCor XCEL",
         "operator": "tech-07"}
    d.update(over)
    return d


# ------------------------------------------------- VascularReference
def test_vascular_reference_happy_path_and_defaults():
    r = vascular_from_dict(_ref(cavi=8.1, brachial_sbp_mmhg=128.0,
                                brachial_dbp_mmhg=82.0,
                                hr_at_measurement_bpm=64.0,
                                meds_antihypertensive=True,
                                fitzpatrick_group=4,
                                ambient_temp_note="22C office",
                                site_id="siteA", device_label="rig-1"))
    assert isinstance(r, VascularReference)
    assert r.cfpwv_mps == 8.4 and r.meds_antihypertensive is True
    assert r.meds_vasoactive is False          # default: not flagged
    assert r.cfpwv_mps_repeat is None


def test_vascular_reference_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown"):
        vascular_from_dict(_ref(vascular_age_years=55))


@pytest.mark.parametrize("missing", ["cfpwv_mps", "device_model", "operator"])
def test_vascular_reference_requires_core_fields(missing):
    d = _ref()
    del d[missing]
    with pytest.raises(ValueError, match=missing):
        vascular_from_dict(d)


@pytest.mark.parametrize("field,bad", [
    ("cfpwv_mps", 2.0), ("cfpwv_mps", 30.0),
    ("cfpwv_mps_repeat", 1.0),
    ("cavi", 1.0), ("cavi", 25.0),
    ("brachial_sbp_mmhg", 40.0), ("brachial_sbp_mmhg", 300.0),
    ("brachial_dbp_mmhg", 20.0), ("brachial_dbp_mmhg", 200.0),
    ("hr_at_measurement_bpm", 20.0), ("hr_at_measurement_bpm", 260.0),
    ("fitzpatrick_group", 0), ("fitzpatrick_group", 7),
])
def test_vascular_reference_plausibility_ranges_fail_closed(field, bad):
    with pytest.raises(ValueError):
        vascular_from_dict(_ref(**{field: bad}))


def test_vascular_reference_demographics_are_baseline_only_inputs():
    r = vascular_from_dict(_ref(age_years=54.0, sex="female"))
    assert r.age_years == 54.0 and r.sex == "female"
    with pytest.raises(ValueError):
        vascular_from_dict(_ref(age_years=10.0))
    with pytest.raises(ValueError, match="sex"):
        vascular_from_dict(_ref(sex="F"))


def test_vascular_reference_pressure_ordering():
    with pytest.raises(ValueError, match="systolic"):
        vascular_from_dict(_ref(brachial_sbp_mmhg=90.0,
                                brachial_dbp_mmhg=95.0))


def test_vascular_reference_empty_device_or_operator_rejected():
    with pytest.raises(ValueError):
        vascular_from_dict(_ref(device_model="  "))
    with pytest.raises(ValueError):
        vascular_from_dict(_ref(operator=""))


# ------------------------------------------------- contact-PPG sidecar
def _ppg(**over):
    d = {"fs_hz": 100.0, "samples": [0.0, 0.4, 1.0, 0.6, 0.1] * 200,
         "t0_video_s": 0.25, "device": "finger-clip"}
    d.update(over)
    return d


def test_contact_ppg_happy_path():
    p = contact_ppg_from_dict(_ppg())
    assert isinstance(p, ContactPpg)
    assert p.fs_hz == 100.0 and len(p.samples) == 1000
    assert p.t0_video_s == 0.25


def test_contact_ppg_fails_closed():
    with pytest.raises(ValueError):                    # fs below beat-resolving
        contact_ppg_from_dict(_ppg(fs_hz=10.0))
    with pytest.raises(ValueError):                    # too short to segment
        contact_ppg_from_dict(_ppg(samples=[0.1] * 5))
    with pytest.raises(ValueError):                    # non-finite poison
        contact_ppg_from_dict(_ppg(samples=[0.1, float("nan")] * 300))
    with pytest.raises(ValueError, match="unknown"):
        contact_ppg_from_dict(_ppg(extra_field=1))
    with pytest.raises(ValueError):
        contact_ppg_from_dict("not a dict")


# ------------------------------------------------- tonometry export parser
def test_parse_pwv_export_sphygmocor_style():
    from datasets.reference import parse_pwv_export
    text = ("Device,SphygmoCor XCEL v1.3\n"
            "Operator,tech-07\n"
            "PWV (m/s),8.4\n"
            "SP (mmHg),128\n"
            "DP (mmHg),82\n"
            "HR (bpm),64\n")
    d = parse_pwv_export(text)
    r = vascular_from_dict(d)
    assert r.cfpwv_mps == 8.4 and "SphygmoCor" in r.device_model
    assert r.brachial_sbp_mmhg == 128.0 and r.hr_at_measurement_bpm == 64.0


def test_parse_pwv_export_complior_style_and_failure():
    from datasets.reference import parse_pwv_export
    text = ("device;Complior Analyse\n"
            "operator;lab-2\n"
            "carotid-femoral pwv;9.6 m/s\n")
    r = vascular_from_dict(parse_pwv_export(text))
    assert r.cfpwv_mps == 9.6 and "Complior" in r.device_model
    with pytest.raises(ValueError, match="no cfPWV"):
        parse_pwv_export("Device,SphygmoCor\nOperator,x\nAIx (%),22\n")
    with pytest.raises(ValueError, match="device"):
        parse_pwv_export("PWV (m/s),8.0\nOperator,x\n")


# ------------------------------------------------- measurement class rev
def test_research_vascular_member_and_docstring_contract():
    # Owner-directed schema rev (v0.4-vascular prompt, spec B.20):
    # RESEARCH_VASCULAR is the head_vascular class — research-flagged,
    # never renderable on any user-facing surface while gates are red.
    assert MeasurementClass.RESEARCH_VASCULAR.value == "RESEARCH_VASCULAR"
    doc = MeasurementClass.__doc__.lower()
    assert "vascular" in doc and "never" in doc


# ------------------------------------------------- registry integration
def test_registry_hashes_pwv_and_ppg_sidecars(tmp_path, monkeypatch):
    from datasets.registry import register_dataset
    monkeypatch.setenv("AVATARX_REGISTRY", str(tmp_path / "reg.jsonl"))
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "r1.recording.json").write_text(json.dumps(
        {"participant_id": "pv01"}))
    (ds / "r1.pwv.json").write_text(json.dumps(_ref()))
    (ds / "r1.ppg.json").write_text(json.dumps(_ppg()))
    kw = {"license_class": "internal_consented",
          "consent_class": "research_v1",
          "registry_path": str(tmp_path / "reg.jsonl")}
    row1 = register_dataset(str(ds), **kw)
    # a changed reference read must change the dataset identity
    (ds / "r1.pwv.json").write_text(json.dumps(_ref(cfpwv_mps=9.9)))
    row2 = register_dataset(str(ds), **kw)
    assert row1["id"] != row2["id"]
    # and a changed contact-PPG sidecar likewise
    (ds / "r1.ppg.json").write_text(json.dumps(_ppg(t0_video_s=0.5)))
    row3 = register_dataset(str(ds), **kw)
    assert row3["id"] not in (row1["id"], row2["id"])


# ------------------------------------------------- review-pinned edges
def test_meds_flags_must_be_real_booleans():
    # review finding: a string "no" would ingest as truthy
    for bad in ("no", "false", 0, 1, [1]):
        with pytest.raises(ValueError, match="boolean"):
            vascular_from_dict(_ref(meds_antihypertensive=bad))
    r = vascular_from_dict(_ref(meds_vasoactive=True))
    assert r.meds_vasoactive is True


def test_string_context_fields_validated():
    with pytest.raises(ValueError, match="site_id"):
        vascular_from_dict(_ref(site_id="  "))
    with pytest.raises(ValueError, match="device_label"):
        vascular_from_dict(_ref(device_label=7))


def test_fitzpatrick_group_rejects_non_integers():
    for bad in (6.9, "6", True):
        with pytest.raises(ValueError):
            vascular_from_dict(_ref(fitzpatrick_group=bad))
    assert vascular_from_dict(
        _ref(fitzpatrick_group=6)).fitzpatrick_group == 6


def test_contact_ppg_t0_must_be_finite():
    with pytest.raises(ValueError, match="t0_video_s"):
        contact_ppg_from_dict(_ppg(t0_video_s=float("nan")))
    with pytest.raises(ValueError, match="t0_video_s"):
        contact_ppg_from_dict(_ppg(t0_video_s=float("inf")))


def test_parse_pwv_export_edge_cases():
    from datasets.reference import parse_pwv_export
    # a second PWV row is the retest read, never silently dropped
    two = parse_pwv_export("Device,X\nOperator,o\n"
                           "PWV (m/s),8.4\nPWV (m/s),8.9\n")
    assert two["cfpwv_mps"] == 8.4 and two["cfpwv_mps_repeat"] == 8.9
    with pytest.raises(ValueError, match="more than two"):
        parse_pwv_export("Device,X\nOperator,o\nPWV (m/s),8.4\n"
                         "PWV (m/s),8.9\nPWV (m/s),9.1\n")
    # an unattributed read cannot enter the evidence
    with pytest.raises(ValueError, match="operator"):
        parse_pwv_export("Device,X\nPWV (m/s),8.4\n")
    # malformed numerics fail closed, never coerce
    with pytest.raises(ValueError):
        parse_pwv_export("Device,X\nOperator,o\nPWV (m/s),pending\n")
    with pytest.raises(ValueError):
        parse_pwv_export("Device,X\nOperator,o\nPWV (m/s),8.4\n"
                         "SP (mmHg),n/a\n")
