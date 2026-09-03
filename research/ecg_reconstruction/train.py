"""
Reconstruction training loop (v0.3 T4). A run is a pure function of its
config: subjects come from the named source (synthetic fixtures, the
cached MIMIC PERform pairs, or REGISTERED facial datasets with paired
reference ECG — the flywheel), the split is participant-keyed and
therefore participant- AND session-disjoint, the named architecture
trains with deterministic seeds, and every §G metric is evaluated on the
frozen held-out side. Results land in research/runs/<run_id>/ and append
to research/runs/scoreboard.jsonl; models/registry.promote reads the
run's gate_results.json and refuses while any gate is red.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import numpy as np

from configs import config_hash, load_config, parse_yaml_subset
from evaluation.inferred_ecg.decoder import (FS, _znorm, perform_subjects,
                                             split_subjects, synth_pairs)
from research.ecg_reconstruction import WATERMARK
from research.ecg_reconstruction.decoder import ReconstructionError, train
from research.ecg_reconstruction.fidelity import (confabulation_eval,
                                                  evaluate_fidelity,
                                                  generate_blinded_set)
from research.ecg_reconstruction.gates import (DEFAULT_RUNS, append_scoreboard,
                                               evaluate_gates, load_gates)

SIGNAL_DOMAIN = {"synthetic": "synthetic", "mimic": "mimic_finger_ppg",
                 "facial": "facial_rppg"}


# ------------------------------------------------- the flywheel loader
def facial_pairs(dataset_ids: list, *, registry_path=None) -> list:
    """[(participant_id, is_af, ppg_125hz, ecg_125hz)] from REGISTERED
    facial datasets whose recording manifests name a `reference_dir`
    holding an ingested paired-ECG session (reference.json +
    ecg_export.csv). The pulse side comes from the PRODUCTION PATH
    (inference.pipeline ingest -> forehead POS), the ECG side is mapped
    onto the video clock with the session's verified sync, and both are
    resampled onto a common 125 Hz grid over their overlap."""
    from datasets.registry import lookup, verify_registered, RegistryError
    from datasets.reference import read_ecg_csv
    from inference.pipeline import run_with_details
    from rppg.pos import pos_pulse

    cfg = load_config()
    band = tuple(cfg["sqi"]["band_hz"])
    pairs, skipped = [], []
    for ds in dataset_ids:
        row = lookup(ds, registry_path=registry_path)
        if row is None:
            raise RegistryError(f"unknown dataset id {ds!r}")
        verify_registered(row["path"], registry_path=registry_path)
        base = pathlib.Path(row["path"])
        for mf in sorted(base.glob("*.recording.json")):
            with open(mf) as f:
                man = json.load(f)
            rid = str(man.get("recording_id", mf.stem))
            ref_dir = man.get("reference_dir")
            if not ref_dir:
                skipped.append((rid, "no reference_dir in manifest"))
                continue
            ref_p = base / ref_dir / "reference.json"
            csv_p = base / ref_dir / "ecg_export.csv"
            if not ref_p.exists() or not csv_p.exists():
                skipped.append((rid, "reference session not ingested"))
                continue
            with open(ref_p) as f:
                ref = json.load(f)
            _res, det = run_with_details(str(base / man["video_path"]),
                                         recording_id=rid)
            ing = det.get("ingest")
            if ing is None or not getattr(ing, "ok", False):
                skipped.append((rid, "production ingest failed"))
                continue
            fps = float(ing.meta.measured_fps_mean)
            ts = np.asarray(ing.timestamps_s, float)
            wave = pos_pulse(ing.traces["forehead"], fps, band=band)

            t_e, mv = read_ecg_csv(csv_p)
            drift = float(ref["sync"]["drift_ppm"] or 0.0)
            t_ev = t_e * (1.0 + drift * 1e-6) \
                + float(ref["sync"]["offset_ms"]) / 1000.0
            a = max(float(ts[0]), float(t_ev[0]))
            b = min(float(ts[-1]), float(t_ev[-1]))
            if b - a < 10.0:
                skipped.append((rid, f"overlap {b - a:.1f} s < 10 s"))
                continue
            grid = np.arange(a, b, 1.0 / FS)
            ppg = _znorm(np.interp(grid, ts, wave))
            ecg = _znorm(np.interp(grid, t_ev, mv))
            is_af = any((r.get("rhythm") == "AFIB")
                        for r in ref.get("labels") or [])
            pairs.append((str(man.get("participant_id", rid)),
                          int(is_af), ppg, ecg))
    if not pairs and skipped:
        raise ReconstructionError(
            "no usable facial pairs: "
            + "; ".join(f"{r}: {why}" for r, why in skipped))
    return pairs


# ------------------------------------------------- the run
def run_reconstruction_training(config_path, *, runs_root=None,
                                gates_path=None, registry_path=None) -> dict:
    cfg = parse_yaml_subset(pathlib.Path(config_path).read_text())
    r = cfg.get("reconstruction")
    if not isinstance(r, dict):
        raise ReconstructionError("config must contain a top-level "
                                  "'reconstruction' block")
    for k in ("run_name", "architecture", "source", "seed"):
        if k not in r:
            raise ReconstructionError(f"reconstruction config missing {k!r}")
    source = r["source"]
    if source not in SIGNAL_DOMAIN:
        raise ReconstructionError(
            f"source must be one of {sorted(SIGNAL_DOMAIN)}, got {source!r}")
    seed = int(r["seed"])
    steps = int(r.get("steps", 400))

    if source == "synthetic":
        subjects = synth_pairs(int(r.get("n_subjects", 8)), seed=seed)
    elif source == "mimic":
        subjects = perform_subjects(int(r.get("limit_per_class", 0)))
        if not subjects:
            raise ReconstructionError(
                "MIMIC PERform cache not present under data_cache/ — use "
                "source: synthetic or fetch the cache")
    else:
        subjects = facial_pairs(list(r.get("facial_dataset_ids") or []),
                                registry_path=registry_path)
        if not subjects:
            raise ReconstructionError("no facial pairs in the named "
                                      "datasets")

    train_s, test_s = split_subjects(
        subjects, seed=int(r.get("split_seed", seed)),
        test_fraction=float(r.get("test_fraction", 0.3)))
    if not train_s or not test_s:
        raise ReconstructionError(f"split produced an empty side "
                                  f"(train {len(train_s)}, "
                                  f"test {len(test_s)})")
    assert not ({s[0] for s in train_s} & {s[0] for s in test_s})

    artifact = train(train_s, architecture=r["architecture"], seed=seed,
                     steps=steps, batch=int(r.get("batch", 256)),
                     lr=float(r.get("lr", 3e-3)))

    fid = evaluate_fidelity(train_s, test_s, artifact)
    conf = confabulation_eval(train_s, test_s,
                              architecture=r["architecture"], seed=seed,
                              steps=int(r.get("confabulation_steps", steps)))
    evidence = {
        "data": {"signal_domain": SIGNAL_DOMAIN[source],
                 "participant_disjoint": True,
                 "session_disjoint": True,     # sessions nest in participants
                 "production_path": source == "facial",
                 "facial_dataset_ids": list(r.get("facial_dataset_ids")
                                            or []),
                 "n_train_subjects": len(train_s),
                 "n_test_subjects": len(test_s)},
        "fidelity": fid["aggregate"],
        "g1": fid["g1"],
        "confabulation": conf,
        "detection": fid["detection"],
        "blinded_reads": None,
    }
    verdict = evaluate_gates(load_gates(gates_path), evidence)

    payload = {"reconstruction": r}
    run_id = "recon-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    out = pathlib.Path(runs_root or DEFAULT_RUNS) / run_id
    out.mkdir(parents=True, exist_ok=True)

    blinded = None
    if r.get("blinded_set", True):
        blinded = generate_blinded_set(test_s, artifact, out / "blinded",
                                       seed=seed)

    record = {
        "WATERMARK": WATERMARK,
        "run_id": run_id, "run_name": r["run_name"],
        "track": "ecg_reconstruction",
        "architecture": r["architecture"], "source": source,
        "seed": seed, "steps": steps,
        "config_hash": config_hash(payload),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_subjects": sorted(s[0] for s in train_s),
        "test_subjects": sorted(s[0] for s in test_s),
        "blinded_set": blinded,
        "per_subject": fid["per_subject"],
    }
    gate_results = {"WATERMARK": WATERMARK, "run_id": run_id,
                    "evidence": evidence, **verdict}
    with open(out / "model.json", "w") as f:
        json.dump({"WATERMARK": WATERMARK, **artifact}, f)
    with open(out / "reconstruction_record.json", "w") as f:
        json.dump(record, f, indent=1)
    with open(out / "gate_results.json", "w") as f:
        json.dump(gate_results, f, indent=1)
    append_scoreboard({"run_id": run_id, "run_name": r["run_name"],
                       "trained_at": record["trained_at"],
                       "architecture": r["architecture"], "source": source,
                       "evidence": evidence,
                       "gates": verdict["gates"],
                       "all_gates_green": verdict["all_gates_green"],
                       "promotion_open": verdict["promotion_open"]},
                      runs_root=runs_root)
    return {"run_dir": str(out), "run_id": run_id,
            "gate_results": gate_results,
            "summary": {"n_train": len(train_s), "n_test": len(test_s),
                        **{k: evidence["fidelity"].get(k)
                           for k in ("decoder_qt_mae_ms",
                                     "decoder_pr_mae_ms",
                                     "decoder_qrs_mae_ms",
                                     "rr_only_qt_mae_ms")},
                        "detection": evidence["detection"],
                        "promotion": ("OPEN" if verdict["promotion_open"]
                                      else "BLOCKED")}}
