"""
Model registry + promotion (v0.2 M3.12) — invariant 13 enforced at the
door. `models/registry.jsonl` is the append-only record of every model
version; `promote()` refuses unless the mandatory-baselines gate passes
(margins per spec B.15), leak detectors are quiet, and no-read parity is
within threshold. A successful promotion appends its record to the spec
changelog — model history is spec history.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from typing import Optional

_REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REGISTRY = pathlib.Path(
    os.environ.get("AVATARX_MODEL_REGISTRY",
                   _REPO / "models" / "registry.jsonl"))
DEFAULT_SPEC = pathlib.Path(
    os.environ.get("AVATARX_SPEC_PATH",
                   _REPO / "CLAUDE_CODE_SPEC.md"))


class PromotionRefused(RuntimeError):
    pass


def _promote_reconstruction(run_dir: pathlib.Path, registry_path,
                            spec_path) -> dict:
    """§G at the door: refused unless EVERY gate is green AND the clinical
    signoff stands, with every failing reason listed (never just the
    first). Loosening a gate to get past this requires an owner-signed
    spec-changelog entry, not a code change here."""
    gp = run_dir / "gate_results.json"
    if not gp.exists():
        raise PromotionRefused(
            "promotion refused:\n  - reconstruction run has no "
            "gate_results.json — no §G evaluation on record")
    with open(gp) as f:
        gr = json.load(f)
    reasons = []
    for g in gr.get("gates") or []:
        if g.get("status") != "GREEN":
            reasons.append(f"{g.get('title', g.get('gate'))}: "
                           + ("; ".join(g.get("reasons"))
                              if g.get("reasons") else "RED"))
    if not (gr.get("gates")):
        reasons.append("gate_results.json lists no gates — fail closed")
    if not gr.get("clinical_signoff"):
        reasons.append("clinical signoff pending (configs/gates.yaml "
                       "signoff block): thresholds are provisional "
                       "defaults until the owner and a clinical advisor "
                       "confirm or tighten them")
    if reasons:
        raise PromotionRefused("promotion refused (§G):\n  - "
                               + "\n  - ".join(reasons))

    with open(run_dir / "reconstruction_record.json") as f:
        record = json.load(f)
    art_bytes = (run_dir / "model.json").read_bytes()
    rows = load_model_registry(registry_path)
    row = {
        "semver": _next_semver(rows),
        "track": "ecg_reconstruction",
        "model_sha256": hashlib.sha256(art_bytes).hexdigest(),
        "run_id": record["run_id"],
        "run_record": str(run_dir / "reconstruction_record.json"),
        "gate_results_sha256": hashlib.sha256(
            json.dumps(gr, sort_keys=True).encode()).hexdigest(),
        "gates": [{"gate": g["gate"], "status": g["status"]}
                  for g in gr["gates"]],
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reg = pathlib.Path(registry_path or DEFAULT_MODEL_REGISTRY)
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "a") as f:
        f.write(json.dumps(row) + "\n")
    spec = pathlib.Path(spec_path or DEFAULT_SPEC)
    if spec.exists():
        with open(spec, "a") as f:
            f.write(f"\n**Reconstruction promotion {row['semver']}** "
                    f"({row['promoted_at']}): run {record['run_id']} "
                    f"passed every §G gate with clinical signoff; "
                    f"gate-results sha256 "
                    f"{row['gate_results_sha256'][:16]}….\n")
    return row



def _gate_completeness(gr: dict, expected: set, reasons: list) -> None:
    """A partial gate list or a verdict the file itself does not stand
    behind is a refusal, not a promotion (review finding: a hand-crafted
    gate_results.json listing only green gates promoted)."""
    present = {g.get("gate") for g in gr.get("gates") or []}
    for missing in sorted(expected - present):
        reasons.append(f"gate {missing} missing from gate_results.json "
                       "— missing evidence is RED (fail closed)")
    if gr.get("all_gates_green") is not True:
        reasons.append("gate_results.json does not record "
                       "all_gates_green: true — fail closed")
    if gr.get("promotion_open") is not True:
        reasons.append("gate_results.json does not record "
                       "promotion_open: true — fail closed")


def _promote_vascular(run_dir: pathlib.Path, registry_path,
                      spec_path) -> dict:
    """The vascular gates at the door for head_vascular (v0.4
    arterial-stiffness track): refused unless EVERY gate V0-V5 in the
    run's recorded gate_results.json is green AND the signoff stands —
    every failing reason listed. A stiffness head that cannot beat the
    age+sex+BP baseline has measured nothing and must not ship."""
    gp = run_dir / "gate_results.json"
    if not gp.exists():
        raise PromotionRefused(
            "promotion refused:\n  - vascular run has no "
            "gate_results.json — no vascular-gate evaluation on record")
    with open(gp) as f:
        gr = json.load(f)
    reasons = []
    for g in gr.get("gates") or []:
        if g.get("status") != "GREEN":
            reasons.append(f"{g.get('title', g.get('gate'))}: "
                           + ("; ".join(g.get("reasons"))
                              if g.get("reasons") else "RED"))
    if not gr.get("gates"):
        reasons.append("gate_results.json lists no gates — fail closed")
    _gate_completeness(gr, {"v0", "v1", "v2", "v3", "v4", "v5"}, reasons)
    if not gr.get("clinical_signoff"):
        reasons.append("clinical signoff pending (configs/gates.yaml "
                       "vascular signoff block): thresholds are "
                       "provisional defaults until the owner and a "
                       "clinical advisor confirm or tighten them, and "
                       "the claim scope is recorded")
    if reasons:
        raise PromotionRefused("promotion refused (vascular gates):\n  - "
                               + "\n  - ".join(reasons))

    with open(run_dir / "vascular_record.json") as f:
        record = json.load(f)
    art_bytes = (run_dir / "model.json").read_bytes()
    rows = load_model_registry(registry_path)
    row = {
        "semver": _next_semver(rows),
        "track": "vascular_stiffness",
        "model_sha256": hashlib.sha256(art_bytes).hexdigest(),
        "run_id": record["run_id"],
        "run_record": str(run_dir / "vascular_record.json"),
        "gate_results_sha256": hashlib.sha256(
            json.dumps(gr, sort_keys=True).encode()).hexdigest(),
        "gates": [{"gate": g["gate"], "status": g["status"]}
                  for g in gr["gates"]],
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reg = pathlib.Path(registry_path or DEFAULT_MODEL_REGISTRY)
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "a") as f:
        f.write(json.dumps(row) + "\n")
    spec = pathlib.Path(spec_path or DEFAULT_SPEC)
    if spec.exists():
        with open(spec, "a") as f:
            f.write(f"\n**Vascular promotion {row['semver']}** "
                    f"({row['promoted_at']}): run {record['run_id']} "
                    f"passed every vascular gate with clinical signoff; "
                    f"gate-results sha256 "
                    f"{row['gate_results_sha256'][:16]}….\n")
    return row


def _promote_vasotone(run_dir: pathlib.Path, registry_path,
                      spec_path) -> dict:
    """§W at the door for head_vasotone (v0.5): refused unless every
    gate W0-W5 in the recorded gate_results.json is green AND the
    signoff (including claim_scope) stands — every failing reason
    listed. A reactivity metric that is secretly an HR/breathing or
    lamp detector must never ship."""
    gp = run_dir / "gate_results.json"
    if not gp.exists():
        raise PromotionRefused(
            "promotion refused:\n  - vasotone run has no "
            "gate_results.json — no \u00a7W evaluation on record")
    with open(gp) as f:
        gr = json.load(f)
    reasons = []
    for g in gr.get("gates") or []:
        if g.get("status") != "GREEN":
            reasons.append(f"{g.get('title', g.get('gate'))}: "
                           + ("; ".join(g.get("reasons"))
                              if g.get("reasons") else "RED"))
    if not gr.get("gates"):
        reasons.append("gate_results.json lists no gates — fail closed")
    _gate_completeness(gr, {"w0", "w1", "w2", "w3", "w4", "w5"}, reasons)
    if not gr.get("clinical_signoff"):
        reasons.append("clinical signoff pending (configs/gates.yaml "
                       "vasotone signoff block): thresholds are "
                       "provisional defaults until the owner and a "
                       "clinical advisor confirm or tighten them, and "
                       "the claim scope is recorded")
    if reasons:
        raise PromotionRefused("promotion refused (\u00a7W):\n  - "
                               + "\n  - ".join(reasons))

    with open(run_dir / "vasotone_record.json") as f:
        record = json.load(f)
    art_bytes = (run_dir / "model.json").read_bytes()
    rows = load_model_registry(registry_path)
    row = {
        "semver": _next_semver(rows),
        "track": "vasomotor_reactivity",
        "model_sha256": hashlib.sha256(art_bytes).hexdigest(),
        "run_id": record["run_id"],
        "run_record": str(run_dir / "vasotone_record.json"),
        "gate_results_sha256": hashlib.sha256(
            json.dumps(gr, sort_keys=True).encode()).hexdigest(),
        "gates": [{"gate": g["gate"], "status": g["status"]}
                  for g in gr["gates"]],
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reg = pathlib.Path(registry_path or DEFAULT_MODEL_REGISTRY)
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "a") as f:
        f.write(json.dumps(row) + "\n")
    spec = pathlib.Path(spec_path or DEFAULT_SPEC)
    if spec.exists():
        with open(spec, "a") as f:
            f.write(f"\n**Vasotone promotion {row['semver']}** "
                    f"({row['promoted_at']}): run {record['run_id']} "
                    f"passed every \u00a7W gate with clinical signoff; "
                    f"gate-results sha256 "
                    f"{row['gate_results_sha256'][:16]}\u2026.\n")
    return row


def _promote_flutter(run_dir: pathlib.Path, registry_path,
                     spec_path) -> dict:
    """§F at the door for head_flutter (v0.6): refused unless every gate
    F0-F5 in the recorded gate_results.json is green AND the signoff
    (including claim_scope) stands — every failing reason listed.

    The specific failure this prevents: a pattern flag that fires on
    ordinary sinus tachycardia. Regular tachycardia is usually benign,
    so an unvalidated flag is a false-referral engine, and F1's
    specificity gate is the thing standing between the two.
    """
    gp = run_dir / "gate_results.json"
    if not gp.exists():
        raise PromotionRefused(
            "promotion refused:\n  - flutter run has no "
            "gate_results.json — no §F evaluation on record")
    with open(gp) as f:
        gr = json.load(f)
    reasons = []
    for g in gr.get("gates") or []:
        if g.get("status") != "GREEN":
            reasons.append(f"{g.get('title', g.get('gate'))}: "
                           + ("; ".join(g.get("reasons"))
                              if g.get("reasons") else "RED"))
    if not gr.get("gates"):
        reasons.append("gate_results.json lists no gates — fail closed")
    _gate_completeness(gr, {"f0", "f1", "f2", "f3", "f4", "f5"}, reasons)
    if not gr.get("clinical_signoff"):
        reasons.append("clinical signoff pending (configs/gates.yaml "
                       "flutter signoff block): the sanctioned sentence "
                       "names no rhythm and routes to an ECG, and the "
                       "owner plus a clinical advisor must record that "
                       "claim scope before anything renders")
    if reasons:
        raise PromotionRefused("promotion refused (§F):\n  - "
                               + "\n  - ".join(reasons))

    with open(run_dir / "flutter_record.json") as f:
        record = json.load(f)
    art_bytes = (run_dir / "model.json").read_bytes()
    rows = load_model_registry(registry_path)
    row = {
        "semver": _next_semver(rows),
        "track": "regular_tachyarrhythmia_flag",
        "model_sha256": hashlib.sha256(art_bytes).hexdigest(),
        "run_id": record["run_id"],
        "run_record": str(run_dir / "flutter_record.json"),
        "gate_results_sha256": hashlib.sha256(
            json.dumps(gr, sort_keys=True).encode()).hexdigest(),
        "gates": [{"gate": g["gate"], "status": g["status"]}
                  for g in gr["gates"]],
        # which rule the evidence said should ship (head or B3) — part
        # of the promotion record, not a later argument
        "shipping_rule": record.get("shipping_rule"),
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reg = pathlib.Path(registry_path or DEFAULT_MODEL_REGISTRY)
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "a") as f:
        f.write(json.dumps(row) + "\n")
    spec = pathlib.Path(spec_path or DEFAULT_SPEC)
    if spec.exists():
        with open(spec, "a") as f:
            f.write(f"\n**Flutter-track promotion {row['semver']}** "
                    f"({row['promoted_at']}): run {record['run_id']} "
                    f"passed every §F gate with clinical signoff; "
                    f"shipping rule {row['shipping_rule']!r}; "
                    f"gate-results sha256 "
                    f"{row['gate_results_sha256'][:16]}….\n")
    return row


def _promote_fitness(run_dir: pathlib.Path, registry_path,
                     spec_path) -> dict:
    """§V at the door for head_fitness (v0.4): refused unless EVERY vo2
    gate in the run's recorded gate_results.json is green AND the
    clinical signoff stands — every failing reason listed. Loosening a
    gate requires an owner-signed spec-changelog entry, never a change
    here."""
    gp = run_dir / "gate_results.json"
    if not gp.exists():
        raise PromotionRefused(
            "promotion refused:\n  - fitness run has no "
            "gate_results.json — no §V evaluation on record")
    with open(gp) as f:
        gr = json.load(f)
    reasons = []
    for g in gr.get("gates") or []:
        if g.get("status") != "GREEN":
            reasons.append(f"{g.get('title', g.get('gate'))}: "
                           + ("; ".join(g.get("reasons"))
                              if g.get("reasons") else "RED"))
    if not gr.get("gates"):
        reasons.append("gate_results.json lists no gates — fail closed")
    _gate_completeness(gr, {"v1", "v2", "v3", "v4", "v5"}, reasons)
    if not gr.get("clinical_signoff"):
        reasons.append("clinical signoff pending (configs/gates.yaml "
                       "vo2 signoff block): thresholds are provisional "
                       "defaults until the owner and a clinical advisor "
                       "confirm or tighten them")
    if reasons:
        raise PromotionRefused("promotion refused (§V):\n  - "
                               + "\n  - ".join(reasons))

    with open(run_dir / "fitness_record.json") as f:
        record = json.load(f)
    art_bytes = (run_dir / "model.json").read_bytes()
    rows = load_model_registry(registry_path)
    row = {
        "semver": _next_semver(rows),
        "track": "vo2_fitness",
        "model_sha256": hashlib.sha256(art_bytes).hexdigest(),
        "run_id": record["run_id"],
        "run_record": str(run_dir / "fitness_record.json"),
        "gate_results_sha256": hashlib.sha256(
            json.dumps(gr, sort_keys=True).encode()).hexdigest(),
        "gates": [{"gate": g["gate"], "status": g["status"]}
                  for g in gr["gates"]],
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reg = pathlib.Path(registry_path or DEFAULT_MODEL_REGISTRY)
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "a") as f:
        f.write(json.dumps(row) + "\n")
    spec = pathlib.Path(spec_path or DEFAULT_SPEC)
    if spec.exists():
        with open(spec, "a") as f:
            f.write(f"\n**Fitness promotion {row['semver']}** "
                    f"({row['promoted_at']}): run {record['run_id']} "
                    f"passed every §V gate with clinical signoff; "
                    f"gate-results sha256 "
                    f"{row['gate_results_sha256'][:16]}….\n")
    return row


def _next_semver(rows: list) -> str:
    best = (0, 0, 0)
    for r in rows:
        try:
            best = max(best, tuple(int(x) for x in r["semver"].split(".")))
        except Exception:
            pass
    return f"{best[0]}.{best[1] + 1}.0" if best != (0, 0, 0) else "0.1.0"


def load_model_registry(path=None) -> list:
    p = pathlib.Path(path or DEFAULT_MODEL_REGISTRY)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _promote_regularity(run_dir: pathlib.Path, registry_path,
                        spec_path) -> dict:
    """§R at the door for head_regularity (v0.7): refused unless every
    gate R0-R5 in the recorded gate_results.json is green AND the signoff
    (including claim_scope) stands — every failing reason listed.

    The specific failure this prevents: an irregular-rhythm notification
    that fires on healthy breathing (respiratory sinus arrhythmia) in
    young people. R2's age-stratified specificity gate is the thing
    standing between a substrate index and a false-referral engine.
    """
    gp = run_dir / "gate_results.json"
    if not gp.exists():
        raise PromotionRefused(
            "promotion refused:\n  - regularity run has no "
            "gate_results.json — no §R evaluation on record")
    with open(gp) as f:
        gr = json.load(f)
    reasons = []
    for g in gr.get("gates") or []:
        if g.get("status") != "GREEN":
            reasons.append(f"{g.get('title', g.get('gate'))}: "
                           + ("; ".join(g.get("reasons"))
                              if g.get("reasons") else "RED"))
    if not gr.get("gates"):
        reasons.append("gate_results.json lists no gates — fail closed")
    _gate_completeness(gr, {"r0", "r1", "r2", "r3", "r4", "r5"}, reasons)
    if not gr.get("clinical_signoff"):
        reasons.append("clinical signoff pending (configs/gates.yaml "
                       "regularity signoff block): the sanctioned "
                       "sentence names no rhythm, offers the benign "
                       "explanation first and routes to an ECG, and the "
                       "owner plus a clinical advisor must record that "
                       "claim scope before anything renders")
    # the verdict file is not trusted on its own: the recorded EVIDENCE
    # is re-evaluated under the LIVE gates.yaml (signoff included) and
    # must open too (review finding: a forged gate_results.json promoted
    # while the yaml was unsigned)
    ep = run_dir / "evidence.json"
    if not ep.exists():
        reasons.append("run has no evidence.json — the door cannot "
                       "re-evaluate the verdict under the live gates")
    else:
        from evaluation.regularity_gates import (evaluate_regularity_gates,
                                                 load_regularity_gates)
        with open(ep) as f:
            ev = json.load(f)
        live = load_regularity_gates()
        if ev.get("gates_version") != live.get("gates_version"):
            reasons.append(f"evidence recorded under gates "
                           f"{ev.get('gates_version')!r}; live gates are "
                           f"{live.get('gates_version')!r}")
        v2 = evaluate_regularity_gates(live, ev.get("evidence") or {},
                                       ev.get("floor") or {})
        if not v2.get("promotion_open"):
            for g in v2.get("gates") or []:
                if g.get("status") != "GREEN":
                    reasons.append(f"[live re-evaluation] "
                                   f"{g.get('title', g.get('gate'))}: "
                                   + ("; ".join(g.get("reasons"))
                                      if g.get("reasons") else "RED"))
            if not v2.get("clinical_signoff"):
                reasons.append("[live re-evaluation] configs/gates.yaml "
                               "regularity signoff is not on record")
    if reasons:
        raise PromotionRefused("promotion refused (§R):\n  - "
                               + "\n  - ".join(reasons))

    with open(run_dir / "regularity_record.json") as f:
        record = json.load(f)
    art_bytes = (run_dir / "model.json").read_bytes()
    rows = load_model_registry(registry_path)
    row = {
        "semver": _next_semver(rows),
        "track": "pulse_regularity",
        "model_sha256": hashlib.sha256(art_bytes).hexdigest(),
        "run_id": record["run_id"],
        "run_record": str(run_dir / "regularity_record.json"),
        "gate_results_sha256": hashlib.sha256(
            json.dumps(gr, sort_keys=True).encode()).hexdigest(),
        "gates": [{"gate": g["gate"], "status": g["status"]}
                  for g in gr["gates"]],
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reg = pathlib.Path(registry_path or DEFAULT_MODEL_REGISTRY)
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "a") as f:
        f.write(json.dumps(row) + "\n")
    spec = pathlib.Path(spec_path or DEFAULT_SPEC)
    if spec.exists():
        with open(spec, "a") as f:
            f.write(f"\n**Regularity-track promotion {row['semver']}** "
                    f"({row['promoted_at']}): run `{row['run_id']}` passed "
                    "§R R0-R5 with clinical signoff on record; the "
                    "irregular-rhythm notification may render once app/ "
                    "is wired to `regularity_render_allowed()`.\n")
    return row


def promote(run_dir, *, registry_path=None, spec_path=None,
            eval_report: Optional[dict] = None,
            config: Optional[dict] = None) -> dict:
    """Gate + register + record. Raises PromotionRefused with EVERY
    failing reason (never just the first)."""
    from configs import load_config
    from training.baselines import mandatory_baseline_report, promotion_gate

    run_dir = pathlib.Path(run_dir)
    if (run_dir / "reconstruction_record.json").exists():
        # v0.3 §G: a reconstruction run is judged by its own pre-registered
        # gates. This reads the run's RECORDED verdicts (data, never
        # research code — the quarantine stays one-way).
        return _promote_reconstruction(run_dir, registry_path, spec_path)
    if (run_dir / "fitness_record.json").exists():
        # v0.4 §V: same discipline for head_fitness — refused while any
        # gate is red or the clinical signoff is missing.
        return _promote_fitness(run_dir, registry_path, spec_path)
    if (run_dir / "vascular_record.json").exists():
        # v0.4 vascular: same discipline for head_vascular — refused
        # while any vascular gate is red or the signoff is missing.
        return _promote_vascular(run_dir, registry_path, spec_path)
    if (run_dir / "vasotone_record.json").exists():
        # v0.5 §W: same discipline for head_vasotone.
        return _promote_vasotone(run_dir, registry_path, spec_path)
    if (run_dir / "flutter_record.json").exists():
        # v0.6 §F: same discipline for head_flutter.
        return _promote_flutter(run_dir, registry_path, spec_path)
    if (run_dir / "regularity_record.json").exists():
        # v0.7 §R: same discipline for head_regularity.
        return _promote_regularity(run_dir, registry_path, spec_path)
    with open(run_dir / "run_record.json") as f:
        record = json.load(f)
    report = eval_report or mandatory_baseline_report(run_dir)
    cfg = config or load_config()
    ok, reasons = promotion_gate(report,
                                 fairness_cfg=cfg["gates"]["fairness"])
    if not ok:
        raise PromotionRefused("promotion refused:\n  - "
                               + "\n  - ".join(reasons))

    art_bytes = (run_dir / "model.json").read_bytes()
    rows = load_model_registry(registry_path)
    row = {
        "semver": _next_semver(rows),
        "model_sha256": hashlib.sha256(art_bytes).hexdigest(),
        "run_id": record["run_id"],
        "run_record": str(run_dir / "run_record.json"),
        "eval_report_sha256": hashlib.sha256(
            json.dumps(report, sort_keys=True).encode()).hexdigest(),
        "candidate_auc": report["candidate_auc"],
        "baselines": report["baselines"],
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": record.get("git_commit"),
    }
    reg = pathlib.Path(registry_path or DEFAULT_MODEL_REGISTRY)
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "a") as f:
        f.write(json.dumps(row) + "\n")

    spec = pathlib.Path(spec_path or DEFAULT_SPEC)
    if spec.exists():
        with open(spec, "a") as f:
            f.write(
                f"\n**Model promotion {row['semver']}** "
                f"({row['promoted_at']}): run {record['run_id']} "
                f"({record['model']}, seed {record['seed']}) on datasets "
                f"{[d['id'] for d in record['datasets']]}; eval "
                f"{report['eval_split']} candidate AUC "
                f"{report['candidate_auc']} vs baselines "
                f"{report['baselines']}; leak flags: "
                f"{report['leak_flags'] or 'none'}. Artifact sha256 "
                f"{row['model_sha256'][:16]}…, eval-report sha256 "
                f"{row['eval_report_sha256'][:16]}….\n")
    return row
