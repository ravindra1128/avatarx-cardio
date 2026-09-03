"""
Training runs (v0.2 M3.10). A run is a pure function of
(config, registered dataset ids, split seed, model seed):

  * datasets must be REGISTERED and byte-identical (datasets/registry);
  * splits come from datasets.splits.registry_dataset_splits —
    participant- and session-disjoint, proven;
  * features are extracted by the GATED PRODUCTION PATH
    (inference.pipeline.run_with_details) — invariant 13: no side
    harness — and cached per dataset content id so every M3 consumer
    (training, baselines, promotion) reads identical numbers;
  * the artifact is a models/baseline.train_model_a payload; the run
    record carries data lineage, config hash, git rev and DEV metrics.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Optional

import numpy as np

from configs import parse_yaml_subset, config_hash
from datasets.registry import lookup, verify_registered, RegistryError
from datasets.splits import registry_dataset_splits

_REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = _REPO / "training" / "runs"


class TrainingError(ValueError):
    pass


def rank_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC (ties averaged); NaN scores rank lowest (an
    abstaining candidate earns nothing on those records)."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    s = np.where(np.isfinite(s), s, -np.inf)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, float)
    sorted_s = s[order]
    i = 0
    r = 1.0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + r + (j - i)) / 2.0
        r += (j - i) + 1
        i = j + 1
    n1 = float(y.sum())
    n0 = float(y.size - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _label_af(manifest: dict) -> int:
    for a in manifest.get("rhythm_annotations") or []:
        if (a.get("rhythm") or "").upper() == "AFIB":
            return 1
    return 0


def extract_dataset_features(dataset_id: str, *, registry_path=None,
                             cache_dir=None, split_seed: int = 20260814
                             ) -> list:
    """Per-recording production-path features + label + split + metadata,
    cached by dataset content id (the id IS the content hash, so a stale
    cache is impossible)."""
    row = lookup(dataset_id, registry_path=registry_path)
    if row is None:
        raise RegistryError(f"unknown dataset id {dataset_id!r}")
    verify_registered(row["path"], registry_path=registry_path)
    cache_dir = pathlib.Path(cache_dir or (DEFAULT_OUT / "feature_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{row['id']}_seed{split_seed}.jsonl"
    if cache.exists():
        return [json.loads(x) for x in cache.read_text().splitlines() if x]

    from inference.pipeline import run_with_details
    plan = registry_dataset_splits(row["id"], seed=split_seed,
                                   registry_path=registry_path)
    split_of = {}
    for split, rids in plan["splits"].items():
        for rid in rids:
            split_of[rid] = split
    out = []
    base = pathlib.Path(row["path"])
    for mf in sorted(base.glob("*.recording.json")):
        with open(mf) as f:
            man = json.load(f)
        rid = str(man.get("recording_id", mf.stem))
        video = base / man["video_path"]
        res, det = run_with_details(str(video), recording_id=rid)
        feats = det.get("features")
        rec = {"recording_id": rid,
               "participant_id": str(man.get("participant_id")),
               "session_id": str(man.get("session_id", rid)),
               "split": split_of.get(rid, "UNASSIGNED"),
               "label_af": _label_af(man),
               "device": (man.get("capture") or {}).get("phone_model",
                                                        "unknown"),
               "site": man.get("site_id", "unknown"),
               "fitzpatrick": man.get("fitzpatrick"),
               "outcome": res.outcome.value,
               "features": ({k: (None if v is None or not np.isfinite(v)
                                 else float(v))
                             for k, v in feats.values.items()}
                            if feats is not None else {})}
        out.append(rec)
    with open(cache, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    return out


def matrix(records: list, feature_names: list, split: str) -> tuple:
    rows = [r for r in records if r["split"] == split]
    X = np.array([[(r["features"].get(k) if r["features"].get(k) is not None
                    else np.nan) for k in feature_names] for r in rows],
                 float) if rows else np.zeros((0, len(feature_names)))
    y = np.array([r["label_af"] for r in rows], int)
    return X, y, rows


def run_training(config_path, *, registry_path=None, out_root=None) -> dict:
    cfg = parse_yaml_subset(pathlib.Path(config_path).read_text())
    t = cfg.get("train")
    if not isinstance(t, dict):
        raise TrainingError("config must contain a top-level 'train' block")
    for k in ("run_name", "model", "dataset_ids", "seed"):
        if k not in t:
            raise TrainingError(f"train config missing {k!r}")
    if t["model"] not in ("logreg", "gbt"):
        raise TrainingError(f"model must be logreg|gbt, got {t['model']!r}")
    split_seed = int(t.get("split_seed", 20260814))

    records = []
    ds_rows = []
    for ds in t["dataset_ids"]:
        ds_rows.append(lookup(ds, registry_path=registry_path) or
                       (_ for _ in ()).throw(
                           TrainingError(f"unknown dataset {ds!r}")))
        records += extract_dataset_features(ds, registry_path=registry_path,
                                            split_seed=split_seed)

    feature_names = t.get("features")
    if not feature_names:
        seen = [r for r in records if r["features"]]
        if not seen:
            raise TrainingError("no recording produced features")
        feature_names = sorted(seen[0]["features"])
    X, y, _ = matrix(records, feature_names, "TRAIN")
    if X.shape[0] < 4 or len(set(y.tolist())) < 2:
        raise TrainingError(
            f"TRAIN split unusable: {X.shape[0]} rows, classes {set(y.tolist())}")

    np.random.seed(int(t["seed"]))                     # deterministic
    from models.baseline import train_model_a
    art = train_model_a(np.nan_to_num(X, nan=0.0), y, model=t["model"])
    art["feature_names"] = list(feature_names)

    from models.baseline import load_model_a
    model, art = load_model_a(art)
    Xd, yd, _ = matrix(records, feature_names, "DEV")
    dev_auc = rank_auc(yd, model.predict_proba(np.nan_to_num(Xd, nan=0.0))) \
        if Xd.shape[0] else float("nan")

    from inference.pipeline import _git_commit
    payload = {"config": t, "dataset_ids": [r["id"] for r in ds_rows],
               "split_seed": split_seed, "feature_names": feature_names}
    run_id = "run-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    out = pathlib.Path(out_root or DEFAULT_OUT) / run_id
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id, "run_name": t["run_name"],
        "model": t["model"], "seed": int(t["seed"]),
        "split_seed": split_seed,
        "git_commit": _git_commit(),
        "config_hash": config_hash({"train": t}),
        "datasets": [{"id": r["id"], "path": r["path"],
                      "license_class": r["license_class"]}
                     for r in ds_rows],
        "feature_names": feature_names,
        "n_train": int(X.shape[0]), "n_dev": int(Xd.shape[0]),
        "metrics": {"dev_auc": (None if not np.isfinite(dev_auc)
                                else round(dev_auc, 4))},
        "artifact": "model.json",
    }
    with open(out / "model.json", "w") as f:
        json.dump(art, f, indent=1)
    with open(out / "run_record.json", "w") as f:
        json.dump(record, f, indent=1)
    with open(out / "features.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return {"run_dir": str(out), "run_id": run_id, "record": record}
