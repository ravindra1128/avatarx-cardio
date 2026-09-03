"""v0.5 vasotone T2 — provocation schema + perfusion-index reference:
maneuver enum, ordered phase marks, null_optics optics-log requirement,
the PI-trace sidecar, registry hashing, and the ingest-reference PI CSV
reader. Everything fails closed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import pytest

from datasets.schema import (MANEUVERS, PiTrace, ProvocationRecord,
                             pi_from_dict, provocation_from_dict)


def _prov(**over):
    d = {"maneuver": "cold_pressor",
         "phase_marks": {"baseline": [0.0, 30.0],
                         "stimulus": [30.0, 70.0],
                         "recovery": [70.0, 100.0]}}
    d.update(over)
    return d


def test_maneuver_vocabulary_is_the_preregistered_set():
    assert set(MANEUVERS) == {"cold_pressor", "paced_breathing",
                              "mental_arithmetic", "posture_change",
                              "null_optics", "null_rest"}


def test_provocation_happy_path():
    p = provocation_from_dict(_prov(intensity=2, notes="IRB v1 cool water"))
    assert isinstance(p, ProvocationRecord)
    assert p.maneuver == "cold_pressor" and p.intensity == 2
    assert p.phase_marks["stimulus"] == (30.0, 70.0)


def test_provocation_fails_closed():
    with pytest.raises(ValueError, match="maneuver"):
        provocation_from_dict(_prov(maneuver="ice_bucket"))
    with pytest.raises(ValueError, match="unknown"):
        provocation_from_dict(_prov(extra=1))
    # phases must exist, be ordered, and not overlap
    with pytest.raises(ValueError, match="baseline"):
        provocation_from_dict({"maneuver": "cold_pressor",
                               "phase_marks": {"stimulus": [0, 10]}})
    bad = _prov()
    bad["phase_marks"]["stimulus"] = [20.0, 25.0]      # overlaps baseline
    with pytest.raises(ValueError, match="overlap"):
        provocation_from_dict(bad)
    bad2 = _prov()
    bad2["phase_marks"]["baseline"] = [10.0, 5.0]      # reversed
    with pytest.raises(ValueError, match="ordered"):
        provocation_from_dict(bad2)
    with pytest.raises(ValueError):
        provocation_from_dict(_prov(intensity=0))      # 1..3 only


def test_null_arms_phase_and_optics_rules():
    # null arms still need baseline+stimulus windows (the "stimulus" of
    # null_optics is the rig perturbation; of null_rest, just more rest)
    n = provocation_from_dict(_prov(maneuver="null_optics",
                                    optics_log="lamp sweep 200-600lux"))
    assert n.maneuver == "null_optics"
    # a null_optics arm WITHOUT a recorded optics log is unusable
    with pytest.raises(ValueError, match="optics_log"):
        provocation_from_dict(_prov(maneuver="null_optics"))
    provocation_from_dict(_prov(maneuver="null_rest"))  # log not needed


# ------------------------------------------------- PI trace sidecar
def _pi(**over):
    d = {"fs_hz": 1.0, "values": [2.0, 2.1, 1.9, 2.0] * 30,
         "t0_video_s": 0.0, "device": "oximeter-x"}
    d.update(over)
    return d


def test_pi_trace_happy_path_and_guards():
    p = pi_from_dict(_pi())
    assert isinstance(p, PiTrace) and p.fs_hz == 1.0
    with pytest.raises(ValueError):                    # too short (<30 s)
        pi_from_dict(_pi(values=[2.0] * 5))
    with pytest.raises(ValueError):                    # PI is positive
        pi_from_dict(_pi(values=[2.0, -0.5] * 60))
    with pytest.raises(ValueError):                    # non-finite
        pi_from_dict(_pi(values=[2.0, float("nan")] * 60))
    with pytest.raises(ValueError, match="unknown"):
        pi_from_dict(_pi(zzz=1))
    with pytest.raises(ValueError):
        pi_from_dict(_pi(fs_hz=0.05))                  # below 0.2 Hz


def test_registry_hashes_provocation_and_pi_sidecars(tmp_path):
    from datasets.registry import register_dataset
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "r1.recording.json").write_text(json.dumps(
        {"participant_id": "pt01"}))
    (ds / "r1.provocation.json").write_text(json.dumps(_prov()))
    (ds / "r1.pi.json").write_text(json.dumps(_pi()))
    kw = {"license_class": "internal_consented",
          "consent_class": "research_v1",
          "registry_path": str(tmp_path / "reg.jsonl")}
    row1 = register_dataset(str(ds), **kw)
    (ds / "r1.provocation.json").write_text(json.dumps(
        _prov(maneuver="null_rest")))
    row2 = register_dataset(str(ds), **kw)
    assert row1["id"] != row2["id"]
    (ds / "r1.pi.json").write_text(json.dumps(_pi(t0_video_s=1.0)))
    row3 = register_dataset(str(ds), **kw)
    assert row3["id"] not in (row1["id"], row2["id"])


def test_ingest_reference_pi_csv_reader():
    from datasets.reference import read_pi_csv
    rows = "\n".join(f"{t / 2.0},{2.0 + 0.01 * t}" for t in range(120))
    t, v = read_pi_csv("t_s,pi_percent\n" + rows)
    assert len(t) == 120 and abs(v[0] - 2.0) < 1e-9
    with pytest.raises(ValueError, match="increasing"):
        read_pi_csv("t_s,pi_percent\n1.0,2.0\n1.0,2.1\n")
    with pytest.raises(ValueError, match="header"):
        read_pi_csv("time,perfusion\n0,2\n")
    with pytest.raises(ValueError):
        read_pi_csv("t_s,pi_percent\n0.0,-1.0\n0.5,2.0\n")
