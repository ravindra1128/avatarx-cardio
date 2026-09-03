"""
Dataset registry (v0.2 M2.9, L6) — every dataset that feeds training or
evaluation is REGISTERED: content-hashed, tied to its campaign, carrying
a license/consent class and lineage. `cli.py evaluate` consumes
registered datasets only; an unregistered directory is not data, it is a
pile of files.

The registry is an append-only JSONL (default `datasets/registry.jsonl`,
gitignored working state; tests point at their own path). Registration is
idempotent: re-registering identical content returns the same id;
CHANGED content under the same path gets a NEW id — datasets are
immutable by identity.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import pathlib
import time
from typing import Optional

import os

DEFAULT_REGISTRY = pathlib.Path(
    os.environ.get("AVATARX_REGISTRY",
                   pathlib.Path(__file__).resolve().parent
                   / "registry.jsonl"))
LICENSE_CLASSES = ("internal_consented", "public_permissive",
                   "public_attribution", "restricted")


class RegistryError(ValueError):
    pass


def _content_hash(dataset_dir: pathlib.Path) -> tuple:
    """(dataset content id, per-file sha256 map). Hashes every manifest,
    reference and label file plus video/ecg payload sizes — enough to
    detect any content change without re-reading gigabytes."""
    files = sorted(list(dataset_dir.glob("*.recording.json"))
                   + list(dataset_dir.glob("*.ecg.json"))
                   + list(dataset_dir.glob("*.session.json"))     # v0.4
                   + list(dataset_dir.glob("*.cpet.json"))        # v0.4
                   + list(dataset_dir.glob("*.pwv.json"))         # vascular
                   + list(dataset_dir.glob("*.ppg.json"))         # vascular
                   + list(dataset_dir.glob("*.provocation.json"))  # v0.5
                   + list(dataset_dir.glob("*.pi.json"))           # v0.5
                   + list(dataset_dir.glob("*.participant.json"))  # v0.6/7
                   + list(dataset_dir.glob("**/reference.json"))
                   + list(dataset_dir.glob("**/labels.json")))
    if not files:
        raise RegistryError(f"{dataset_dir} contains no manifests "
                            "(*.recording.json or *.session.json) — "
                            "nothing to register")
    h = hashlib.sha256()
    per_file = {}
    for f in files:
        d = hashlib.sha256(f.read_bytes()).hexdigest()
        per_file[str(f.relative_to(dataset_dir))] = d
        h.update(d.encode())
    for f in sorted(dataset_dir.glob("*.avi")):
        h.update(f"{f.name}:{f.stat().st_size}".encode())
    return "ds-" + h.hexdigest()[:16], per_file


def _participants(dataset_dir: pathlib.Path) -> list:
    out = set()
    for pattern in ("*.recording.json", "*.session.json"):
        for f in dataset_dir.glob(pattern):
            with open(f) as fh:
                d = json.load(fh)
            pid = d.get("participant_id")
            if pid:
                out.add(str(pid))
    return sorted(out)


def register_dataset(dataset_dir, *, license_class: str,
                     consent_class: str,
                     campaign_ref: Optional[str] = None,
                     parent_ids: Optional[list] = None,
                     registry_path=None) -> dict:
    dataset_dir = pathlib.Path(dataset_dir).resolve()
    if license_class not in LICENSE_CLASSES:
        raise RegistryError(f"license_class must be one of "
                            f"{LICENSE_CLASSES}, got {license_class!r}")
    ds_id, per_file = _content_hash(dataset_dir)
    reg = pathlib.Path(registry_path or DEFAULT_REGISTRY)
    existing = lookup(ds_id, registry_path=reg)
    if existing:
        return existing
    row = {
        "id": ds_id, "path": str(dataset_dir),
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "registered_by": getpass.getuser(),
        "campaign_ref": campaign_ref,
        "license_class": license_class,
        "consent_class": consent_class,
        "participants": _participants(dataset_dir),
        "n_manifests": len(per_file),
        "content_files": per_file,
        "lineage": {"parent_ids": list(parent_ids or [])},
    }
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load_registry(registry_path=None) -> list:
    reg = pathlib.Path(registry_path or DEFAULT_REGISTRY)
    if not reg.exists():
        return []
    out = []
    for line in reg.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def lookup(id_or_path, registry_path=None) -> Optional[dict]:
    rows = load_registry(registry_path)
    key = str(id_or_path)
    resolved = str(pathlib.Path(key).resolve()) if pathlib.Path(key).exists() \
        else None
    hit = None
    for r in rows:                       # last row wins (append-only log)
        if r["id"] == key or (resolved and r["path"] == resolved):
            hit = r
    return hit


def verify_registered(dataset_dir, registry_path=None) -> dict:
    """The dataset at `dataset_dir` must be registered AND byte-identical
    to what was registered. Returns the row; raises RegistryError with
    the reason otherwise."""
    dataset_dir = pathlib.Path(dataset_dir).resolve()
    row = lookup(dataset_dir, registry_path=registry_path)
    if row is None:
        raise RegistryError(
            f"{dataset_dir} is not a registered dataset — register it "
            "first (datasets.registry.register_dataset / "
            "cli.py register-dataset)")
    ds_id, _ = _content_hash(dataset_dir)
    if ds_id != row["id"]:
        raise RegistryError(
            f"{dataset_dir} content changed since registration "
            f"({ds_id} != {row['id']}) — datasets are immutable by "
            "identity; register the new content")
    return row
