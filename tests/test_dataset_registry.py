"""M2.9 — dataset registry: content-hash identity, immutability, lineage,
registry-driven participant+session-disjoint splits, evaluate gating."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import pytest

from datasets.registry import (register_dataset, lookup, verify_registered,
                               RegistryError)
from datasets.splits import registry_dataset_splits, LeakageError


def _dataset(tmp, n=8):
    for i in range(n):
        pid = f"P{i:02d}"
        (tmp / f"r{i}.recording.json").write_text(json.dumps(
            {"recording_id": f"r{i}", "participant_id": pid,
             "session_id": f"s{i}"}))
    return tmp


def test_register_is_idempotent_and_content_addressed(tmp_path):
    ds = tmp_path / "d1"
    ds.mkdir()
    _dataset(ds)
    reg = tmp_path / "reg.jsonl"
    a = register_dataset(ds, license_class="internal_consented",
                         consent_class="research_v1", registry_path=reg)
    b = register_dataset(ds, license_class="internal_consented",
                         consent_class="research_v1", registry_path=reg)
    assert a["id"] == b["id"] and a["id"].startswith("ds-")
    assert len(reg.read_text().splitlines()) == 1        # no duplicate row
    assert lookup(a["id"], registry_path=reg)["participants"] == \
        [f"P{i:02d}" for i in range(8)]
    assert verify_registered(ds, registry_path=reg)["id"] == a["id"]


def test_mutation_breaks_identity_and_verify_fails_closed(tmp_path):
    ds = tmp_path / "d2"; ds.mkdir()
    _dataset(ds)
    reg = tmp_path / "reg.jsonl"
    a = register_dataset(ds, license_class="internal_consented",
                         consent_class="research_v1", registry_path=reg)
    (ds / "r0.recording.json").write_text(json.dumps(
        {"recording_id": "r0", "participant_id": "P00",
         "session_id": "s0", "tampered": True}))
    with pytest.raises(RegistryError, match="content changed"):
        verify_registered(ds, registry_path=reg)
    c = register_dataset(ds, license_class="internal_consented",
                         consent_class="research_v1", registry_path=reg)
    assert c["id"] != a["id"]                            # new identity


def test_bad_license_class_and_empty_dir_fail(tmp_path):
    ds = tmp_path / "d3"; ds.mkdir()
    with pytest.raises(RegistryError, match="nothing to register"):
        register_dataset(ds, license_class="internal_consented",
                         consent_class="x",
                         registry_path=tmp_path / "r.jsonl")
    _dataset(ds)
    with pytest.raises(RegistryError, match="license_class"):
        register_dataset(ds, license_class="whatever", consent_class="x",
                         registry_path=tmp_path / "r.jsonl")


def test_registry_splits_are_participant_and_session_disjoint(tmp_path):
    ds = tmp_path / "d4"; ds.mkdir()
    for i in range(24):
        pid = f"P{i % 12:02d}"                # 2 sessions per participant
        (ds / f"r{i}.recording.json").write_text(json.dumps(
            {"recording_id": f"r{i}", "participant_id": pid,
             "session_id": f"{pid}-sess{i // 12}"}))
    reg = tmp_path / "reg.jsonl"
    row = register_dataset(ds, license_class="internal_consented",
                           consent_class="research_v1", registry_path=reg)
    plan = registry_dataset_splits(row["id"], registry_path=reg)
    splits = plan["splits"]
    assert sum(len(v) for v in splits.values()) == 24
    parts = {k: set(plan["participants"][k]) for k in splits}
    sess = {k: set(plan["sessions"][k]) for k in splits}
    keys = list(splits)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not (parts[a] & parts[b])
            assert not (sess[a] & sess[b])
    # deterministic
    plan2 = registry_dataset_splits(row["id"], registry_path=reg)
    assert plan2["splits"] == splits
    with pytest.raises(RegistryError):
        registry_dataset_splits("ds-nope", registry_path=reg)


def test_participant_sidecars_are_part_of_the_content_id(tmp_path):
    """v0.6 added <id>.participant.json (skin-tone group, age, sex — the
    fairness-gate inputs) but the registry did not hash it, so editing a
    participant's group after registration left the dataset id unchanged.
    Fairness evidence must be as tamper-visible as the labels."""
    from datasets.registry import register_dataset, verify_registered
    from datasets.registry import RegistryError
    from scripts.make_synth_flutter import write_scan
    import pytest as _pt
    _pt.importorskip("cv2")
    ds = tmp_path / "ds"
    write_scan(ds, "r_a", "sinus", pid="pa", session_id="pa-s1", seed=1,
               duration_s=6.0, fitzpatrick=2)
    reg = tmp_path / "reg.jsonl"
    a = register_dataset(ds, license_class="internal_consented",
                         consent_class="research_v1", registry_path=reg)
    assert (ds / "r_a.participant.json").exists()
    verify_registered(ds, registry_path=reg)
    sidecar = ds / "r_a.participant.json"
    sidecar.write_text(sidecar.read_text().replace('"fitzpatrick_group": 2',
                                                   '"fitzpatrick_group": 6'))
    with _pt.raises(RegistryError, match="content changed"):
        verify_registered(ds, registry_path=reg)
