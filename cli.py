#!/usr/bin/env python3
"""
AvatarX AFib v0.1 — offline research CLI.

  python3 cli.py validate  <recording.json>              capture/sync gates
  python3 cli.py process   <video> [--manifest rec.json] -> ScanResult JSON
  python3 cli.py evaluate  <dataset_dir>                 gate report (T7)
  python3 cli.py demo                                    LIVE camera scan (browser)
  python3 cli.py demo --synthetic                        offline synthetic demo

Exit codes: 0 = ran (including NO_RESULT and gate-failed outcomes — those
are RESULTS); 2 = invalid input (unreadable file, malformed JSON), with
reasons on stderr.

Every user-facing sentence comes verbatim from ScanResult.user_facing_text().
This tool never prints a diagnosis; there is no code path that could.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _fail(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 2


def _runtime_preflight(mods=("numpy", "scipy", "cv2"),
                       _find=None, _exec=None) -> int:
    """Fail FAST and NAME THE INTERPRETER when a runtime dependency is
    missing (v0.2.0.1: a demo launched with a bare `python3` that
    resolved to a dependency-less Homebrew build served the whole UI and
    only failed at scan start with 'opencv required').

    v0.3.0.1 (FOURTH wrong-interpreter launch — the v0.2.1.1 copy-paste
    suggestion was printed and the human re-ran the same command anyway):
    when a working interpreter exists, RELAUNCH with it, once, saying so.
    AVATARX_NO_REEXEC=1 opts out (set automatically before the relaunch,
    so a broken "working" interpreter cannot loop). Returns 0, exits 2
    with the fix spelled out, or does not return (exec)."""
    import importlib.util
    find = _find or importlib.util.find_spec
    missing = []
    for m in mods:
        try:
            if find(m) is None:
                missing.append(m)
        except (ImportError, ValueError):
            missing.append(m)
    if not missing:
        return 0
    working = _find_working_interpreter(mods)
    # never exec away a test process — in-process preflight tests would
    # silently replace the pytest run itself
    in_pytest = "PYTEST_CURRENT_TEST" in os.environ
    if working and not in_pytest and \
            not os.environ.get("AVATARX_NO_REEXEC"):
        print(f"this Python ({sys.executable}) is missing "
              f"{', '.join(missing)} — relaunching with {working}",
              file=sys.stderr)
        os.environ["AVATARX_NO_REEXEC"] = "1"
        (_exec or os.execv)(working, [working] + sys.argv)
        return 0        # unreachable under a real exec (tests inject _exec)
    pips = {"cv2": "opencv-python-headless"}
    msg = (
        f"this Python ({sys.executable}) is missing required packages: "
        f"{', '.join(missing)}.\n"
        f"Either install them into THIS interpreter:\n"
        f"  {sys.executable} -m pip install "
        f"{' '.join(pips.get(m, m) for m in missing)}\n"
        f"or launch with an interpreter that has them (a bare `python3` "
        f"may resolve to a different installation than you expect — "
        f"check with `which python3`; an activated venv also overrides "
        f"PATH until you `deactivate`).")
    if working:
        msg += (f"\nA working interpreter WAS found on this machine — "
                f"copy-paste:\n  {working} {' '.join(sys.argv)}")
    return _fail(msg)


def _find_working_interpreter(mods, _candidates=None, _check=None):
    """First other interpreter on this machine that imports every
    required module — so the failure message ends in a copy-paste fix
    (three real wrong-interpreter launches earned this)."""
    import shutil as _sh
    import subprocess as _sp
    cands = _candidates if _candidates is not None else [
        "/usr/bin/python3", _sh.which("python3.9"), _sh.which("python3.10"),
        _sh.which("python3.11"), _sh.which("python3"),
        "/opt/homebrew/bin/python3"]
    seen = set()
    for c in cands:
        if not c or c == sys.executable or c in seen:
            continue
        seen.add(c)
        if not pathlib.Path(c).exists():
            continue
        if _check is not None:
            ok = _check(c)
        else:
            try:
                ok = _sp.run([c, "-c",
                              "import " + ", ".join(mods)],
                             capture_output=True, timeout=20
                             ).returncode == 0
            except Exception:
                ok = False
        if ok:
            return c
    return None


def cmd_validate(args) -> int:
    from datasets.io import load_recording
    try:
        rec = load_recording(args.recording)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        return _fail(f"invalid recording JSON: {e}")
    ok_c, why_c = rec.capture.is_valid_for_beat_analysis()
    ok_s, why_s = rec.sync.is_valid_for_beat_analysis()
    ok_all, why_all = rec.is_valid_for_beat_analysis()
    print(json.dumps({
        "recording_id": rec.recording_id,
        "valid_for_beat_analysis": ok_all,
        "capture_valid": ok_c,
        "sync_valid": ok_s,
        "reasons": why_all,
    }, indent=2))
    return 0


def cmd_falsify(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    from evaluation.inferred_ecg.report import run_falsification
    try:
        out = run_falsification(source=args.source, limit=args.limit,
                                steps=args.steps, out_root=args.out)
    except RuntimeError as e:
        return _fail(str(e))
    print(json.dumps(out, indent=2))
    return 0


def cmd_train(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    from configs import parse_yaml_subset
    try:
        cfg = parse_yaml_subset(pathlib.Path(args.config).read_text())
    except (OSError, ValueError) as e:
        return _fail(f"invalid training config: {e}")
    if "reconstruction" in cfg:
        # v0.3 research track: same verb, config-dispatched; artifacts
        # stay under research/runs/ and every run auto-evaluates §G
        from research.ecg_reconstruction.train import \
            run_reconstruction_training
        from research.ecg_reconstruction.decoder import ReconstructionError
        try:
            out = run_reconstruction_training(args.config)
        except (ReconstructionError, OSError) as e:
            return _fail(str(e))
        print(json.dumps({"run_dir": out["run_dir"],
                          "run_id": out["run_id"],
                          "summary": out["summary"]}, indent=2))
        return 0
    from training.runs import run_training, TrainingError
    from datasets.registry import RegistryError
    try:
        out = run_training(args.config)
    except (TrainingError, RegistryError, OSError) as e:
        return _fail(str(e))
    print(json.dumps({"run_dir": out["run_dir"], "run_id": out["run_id"],
                      "metrics": out["record"]["metrics"]}, indent=2))
    return 0


def cmd_gate_status(args) -> int:
    track = getattr(args, "track", "ecg")
    try:
        if track == "vo2":
            from evaluation.fitness_gates import (vo2_gate_status,
                                                  render_vo2_status_html)
            doc = vo2_gate_status()
            renderer = render_vo2_status_html
        elif track == "vascular":
            from evaluation.vascular_gates import (
                vascular_gate_status, render_vascular_status_html)
            doc = vascular_gate_status()
            renderer = render_vascular_status_html
        elif track == "vasotone":
            from evaluation.vasotone_gates import (
                vasotone_gate_status, render_vasotone_status_html)
            doc = vasotone_gate_status()
            renderer = render_vasotone_status_html
        elif track == "flutter":
            from evaluation.flutter_gates import (
                flutter_gate_status, render_flutter_status_html)
            doc = flutter_gate_status()
            renderer = render_flutter_status_html
        elif track == "regularity":
            from evaluation.regularity_gates import (
                regularity_gate_status, render_regularity_status_html)
            doc = regularity_gate_status()
            renderer = render_regularity_status_html
        else:
            from research.ecg_reconstruction.gates import (
                gate_status, render_gate_status_html)
            doc = gate_status()
            renderer = render_gate_status_html
    except (OSError, ValueError) as e:
        return _fail(str(e))
    if getattr(args, "html", None):
        with open(args.html, "w") as f:
            f.write(renderer(doc))
    print(json.dumps(doc, indent=2))
    return 0


def cmd_reconstruct(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    manifest = None
    if args.manifest:
        try:
            with open(args.manifest) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            return _fail(f"invalid manifest: {e}")
    from research.ecg_reconstruction.reconstruct import reconstruct_video
    from research.ecg_reconstruction.decoder import ReconstructionError
    try:
        out = reconstruct_video(args.video, manifest=manifest)
    except (ReconstructionError, RuntimeError, OSError) as e:
        return _fail(str(e))
    print(json.dumps(out, indent=2))
    return 0


def cmd_promote(args) -> int:
    from models.registry import promote, PromotionRefused
    try:
        row = promote(args.run_dir)
    except PromotionRefused as e:
        return _fail(str(e))
    except (OSError, ValueError) as e:
        return _fail(str(e))
    print(json.dumps(row, indent=2))
    return 0


def cmd_register_dataset(args) -> int:
    from datasets.registry import register_dataset, RegistryError
    try:
        row = register_dataset(args.dataset_dir,
                               license_class=args.license_class,
                               consent_class=args.consent_class,
                               campaign_ref=args.campaign)
    except (RegistryError, OSError) as e:
        return _fail(str(e))
    print(json.dumps({"id": row["id"], "path": row["path"],
                      "participants": len(row["participants"]),
                      "n_manifests": row["n_manifests"]}, indent=2))
    return 0


def cmd_ingest_reference(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    from datasets.reference import ingest_reference, ReferenceError
    try:
        ref = ingest_reference(args.session_dir)
    except (ReferenceError, OSError, json.JSONDecodeError) as e:
        return _fail(str(e))
    print(json.dumps({"reference_json": str(pathlib.Path(args.session_dir)
                                            / "reference.json"),
                      "n_rpeaks": len(ref["rpeaks_s_ecg"]),
                      "sync": ref["sync"],
                      "n_labels": len(ref["labels"])}, indent=2))
    return 0


def cmd_campaign(args) -> int:
    from datasets.campaigns import campaign_status, CampaignError
    try:
        st = campaign_status(args.campaign_dir)
    except (CampaignError, OSError) as e:
        return _fail(str(e))
    doc = {"campaign_id": st.campaign_id, "target_n": st.target_n,
           "enrolled_participants": st.enrolled_participants,
           "sessions": st.sessions, "quota_fill": st.quota_fill,
           "unmet": st.unmet, "problems": st.problems,
           "complete": st.complete()}
    print(json.dumps(doc, indent=2))
    return 0


def cmd_process(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    if not pathlib.Path(args.video).exists():
        return _fail(f"video not found: {args.video}")
    manifest = None
    if args.manifest:
        try:
            with open(args.manifest) as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            return _fail(f"invalid manifest: {e}")
        lux = m.get("illuminance_lux")
        if lux is None:
            lux = (m.get("capture") or {}).get("illuminance_lux_mean")
        manifest = {"illuminance_lux": lux}
    heads = None
    vascular_requested = False
    vasotone_requested = False
    if getattr(args, "heads", None):
        heads = [h.strip() for h in args.heads.split(",") if h.strip()]
        # the vascular/vasotone heads are research-flagged (V-a/W-a):
        # they never run inside the pipeline result — each gets its own
        # research report
        vascular_requested = "vascular" in heads
        vasotone_requested = "vasotone" in heads
        heads = [h for h in heads
                 if h not in ("vascular", "vasotone")] or None
    try:
        from inference.pipeline import run_with_details
        result, det = run_with_details(args.video, manifest=manifest,
                                       heads=heads)
    except RuntimeError as e:
        return _fail(str(e))
    except Exception as e:
        from heads.base import HeadRegistryError
        if isinstance(e, HeadRegistryError):
            return _fail(str(e))
        raise
    if getattr(args, "report", None):
        # Export uses the same report assembler as the live API/report.  The
        # plain CLI JSON below deliberately remains the established consumer
        # ScanResult boundary; research estimates do not leak into it.
        from configs import load_config
        from app.report_data import report_capture_meta
        from app.report_render import render_report
        assembled_report = report_capture_meta(result, det, load_config())
        html_doc = render_report(result, assembled_report)
        with open(args.report, "w") as f:
            f.write(html_doc)
    doc = dataclasses.asdict(result)
    doc["outcome"] = result.outcome.value
    doc["user_facing_text"] = result.user_facing_text()
    # invariant 9 at the CLI boundary too, not only in app/: a RESEARCH_*
    # head result must never ride in the consumer JSON. The vascular and
    # vasotone heads are stripped from the pipeline entirely (above), but
    # head_flutter runs ON the measured path by design, so this filter is
    # what keeps its payload — including its sentence key — out of stdout
    # while §F is red (review finding).
    from datasets.schema import public_head_results
    doc["head_results"] = public_head_results(doc.get("head_results"))
    # explainability: every gate, feature and piece of beat evidence behind
    # the decision (v0.1.2). Never user-facing; for operators and audits.
    doc["debug"] = {"rationale": det.get("rationale"),
                    "evidence": det.get("evidence")}
    if vascular_requested:
        # research report only — stdout stays the pure consumer-free
        # ScanResult JSON; the estimate never touches it (V-a)
        try:
            from research.vascular.head_runner import run_research_head
            rr = run_research_head(result, det)
            print(f"vascular research report (watermarked, never a "
                  f"measurement): {rr['report_path']}", file=sys.stderr)
        except Exception as e:
            # the research branch must never kill the production
            # ScanResult on stdout (review finding)
            print(f"vascular research head failed "
                  f"({type(e).__name__}): {e}", file=sys.stderr)
    if vasotone_requested:
        try:
            if not getattr(args, "provocation", None):
                raise ValueError("--heads vasotone needs --provocation "
                                 "<file.provocation.json> (the timed "
                                 "phase marks)")
            from datasets.schema import (pi_from_dict,
                                         provocation_from_dict)
            from research.vascular.tone_runner import \
                run_research_tone_head
            prov = provocation_from_dict(
                json.load(open(args.provocation)))
            pi = None
            if getattr(args, "pi", None):
                pi = pi_from_dict(json.load(open(args.pi)))
            cap = None
            if args.manifest:
                with open(args.manifest) as f:
                    cap = (json.load(f).get("capture") or {})
            rr = run_research_tone_head(result, det, prov,
                                        capture=cap, pi=pi)
            print(f"vasotone research report (watermarked, never a "
                  f"measurement): {rr['report_path']}", file=sys.stderr)
        except Exception as e:
            print(f"vasotone research head failed "
                  f"({type(e).__name__}): {e}", file=sys.stderr)
    print(json.dumps(doc, indent=2, default=str))
    return 0


def _capture_manifest_from_file(path):
    """The manifest reduction `cli.py process` applies (illuminance
    only) — shared so a research report runs the pipeline on EXACTLY
    the manifest the consumer verb would (review finding). Returns
    (pipeline_manifest, raw) or raises ValueError."""
    with open(path) as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError("manifest must be a JSON object")
    lux = m.get("illuminance_lux")
    if lux is None:
        lux = (m.get("capture") or {}).get("illuminance_lux_mean")
    return {"illuminance_lux": lux}, m


def cmd_research_report(args) -> int:
    """v0.8: the Research & Investigation Report — every gated head's
    raw output for one scan, watermarked, beside its track's live gate
    status. Investigators only; never the consumer report. Exit 0 when
    it ran (values, abstentions and not-applicable tracks included),
    2 on invalid arguments or unparsable input files."""
    rc = _runtime_preflight()
    if rc:
        return rc
    from research.investigation import InvestigationError
    from research.investigation.report import (build_investigation_report,
                                               write_investigation_report)
    if not pathlib.Path(args.video).exists():
        return _fail(f"video not found: {args.video}")
    manifest = raw_manifest = None
    if getattr(args, "manifest", None):
        try:
            manifest, raw_manifest = _capture_manifest_from_file(
                args.manifest)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            return _fail(f"invalid manifest: {e}")
    if args.age is not None and not (0.0 < float(args.age) < 130.0):
        return _fail(f"--age {args.age} is not a plausible age in years")
    session = getattr(args, "session", None)
    videos = getattr(args, "videos", None) or None
    if videos and not session:
        return _fail("--videos needs --session <manifest.json>")
    if session:
        if not pathlib.Path(session).exists():
            return _fail(f"session manifest not found: {session}")
        try:
            with open(session) as f:
                if not isinstance(json.load(f), dict):
                    raise ValueError("session manifest must be a JSON "
                                     "object")
        except (OSError, ValueError, json.JSONDecodeError) as e:
            return _fail(f"invalid session manifest: {e}")
        for v in videos or []:
            if not pathlib.Path(v).exists():
                return _fail(f"video not found: {v}")
    prov = pi = None
    if getattr(args, "pi", None) and not getattr(args, "provocation", None):
        return _fail("--pi needs --provocation <file.provocation.json>")
    if getattr(args, "provocation", None):
        from datasets.schema import pi_from_dict, provocation_from_dict
        try:
            with open(args.provocation) as f:
                prov = provocation_from_dict(json.load(f))
            if getattr(args, "pi", None):
                with open(args.pi) as f:
                    pi = pi_from_dict(json.load(f))
        except Exception as e:                   # noqa: BLE001 — exit 2
            return _fail(f"invalid provocation/pi file "
                         f"({type(e).__name__}): {e}")
    try:
        doc = build_investigation_report(
            args.video, manifest=manifest, age_years=args.age,
            session_manifest=session, session_videos=videos,
            provocation=prov, pi=pi)
    except InvestigationError as e:
        return _fail(str(e))
    except RuntimeError as e:
        return _fail(str(e))
    if raw_manifest is not None:
        doc["inputs"]["capture_manifest"] = {
            "path": args.manifest,
            "applied_to_pipeline": manifest,
            "note": "reduced exactly as `cli.py process` reduces it"}
    rc = 0
    try:
        doc["written"] = write_investigation_report(doc, out_dir=args.out)
    except OSError as e:
        doc["written"] = {"error": f"could not write the report: {e}"}
        print(f"research report not written: {e}", file=sys.stderr)
        rc = 2
    print(json.dumps(doc, indent=2, default=str))
    if rc == 0:
        print(f"research & investigation report (watermarked, never a "
              f"measurement): {doc['written']['html_path']}",
              file=sys.stderr)
    return rc


def cmd_session(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    if not pathlib.Path(args.manifest).exists():
        return _fail(f"session manifest not found: {args.manifest}")
    for v in args.videos:
        if not pathlib.Path(v).exists():
            return _fail(f"video not found: {v}")
    from protocol.session import run_session, SessionError
    from protocol.challenges import ChallengeError
    try:
        result, det = run_session(args.manifest,
                                  videos_override=args.videos or None)
    except (SessionError, ChallengeError, ValueError, OSError,
            json.JSONDecodeError) as e:
        return _fail(f"invalid session: {e}")
    except RuntimeError as e:
        return _fail(str(e))
    doc = dataclasses.asdict(result)
    doc["outcome"] = result.outcome.value
    doc["user_facing_text"] = result.user_facing_text()
    doc["debug"] = {"compliance": result.compliance,
                    "phases_run": det.get("phases_run"),
                    "activity": det.get("activity")}
    print(json.dumps(doc, indent=2, default=str))
    return 0


def cmd_evaluate(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    try:
        from evaluation.report import evaluate_dataset
    except ImportError:
        return _fail("evaluate is not available in this build")
    # v0.2 M2.9: evaluation consumes REGISTERED datasets only — identity,
    # lineage and consent class are preconditions for any number leaving
    # this harness
    from datasets.registry import verify_registered, RegistryError
    try:
        verify_registered(args.dataset_dir)
    except RegistryError as e:
        return _fail(str(e))
    try:
        out = evaluate_dataset(args.dataset_dir, out_dir=args.out)
    except (OSError, ValueError) as e:
        return _fail(f"evaluate failed: {e}")
    doc = {"report_json": out["report_json"], "report_md": out["report_md"]}
    if getattr(args, "candidate", None):
        # M3.11: candidate margin over every mandatory baseline, on the
        # disjoint splits, from production-path features
        from training.baselines import mandatory_baseline_report
        try:
            doc["mandatory_baselines"] = mandatory_baseline_report(
                args.candidate)
        except (OSError, ValueError) as e:
            return _fail(f"baseline harness failed: {e}")
    print(json.dumps(doc, indent=2))
    return 0


def cmd_evaluate_fitness(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    from datasets.registry import verify_registered, RegistryError
    try:
        verify_registered(args.dataset_dir)
    except RegistryError as e:
        return _fail(str(e))
    from evaluation.fitness_metrics import evaluate_fitness_dataset
    try:
        out = evaluate_fitness_dataset(args.dataset_dir,
                                       out_dir=args.out,
                                       signal_domain=args.domain)
    except (OSError, ValueError) as e:
        return _fail(f"evaluate-fitness failed: {e}")
    print(json.dumps({"report_json": out["report_json"],
                      "report_md": out["report_md"],
                      "run_dir": out["run_dir"],
                      "promotion":
                          out["report"]["gates"]["promotion"]}, indent=2))
    return 0


def cmd_evaluate_vascular(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    from datasets.registry import verify_registered, RegistryError
    try:
        verify_registered(args.dataset_dir)
    except RegistryError as e:
        return _fail(str(e))
    from evaluation.vascular_metrics import (evaluate_vascular_dataset,
                                            VascularHarnessError)
    try:
        out = evaluate_vascular_dataset(args.dataset_dir,
                                        out_dir=args.out,
                                        signal_domain=args.domain)
    except (RuntimeError, OSError, ValueError) as e:
        return _fail(f"evaluate-vascular failed: {e}")
    doc = out["report"]
    battery = doc.get("battery") or {}
    # V-d: the summary itself carries the B3 delta or says why not
    print(json.dumps({"WATERMARK": doc["WATERMARK"],
                      "report_json": out["report_json"],
                      "report_md": out["report_md"],
                      "run_dir": out["run_dir"],
                      "rmse_improvement_vs_b3_mps":
                          battery.get("rmse_improvement_vs_b3_mps"),
                      "battery_unavailable_reason":
                          (None if battery.get("available")
                           else battery.get("reason")),
                      "promotion": doc["gates"]["promotion"]}, indent=2))
    return 0



def cmd_evaluate_vasotone(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    from datasets.registry import verify_registered, RegistryError
    try:
        verify_registered(args.dataset_dir)
    except RegistryError as e:
        return _fail(str(e))
    from evaluation.vasotone_metrics import (evaluate_vasotone_dataset,
                                            VasotoneHarnessError)
    try:
        out = evaluate_vasotone_dataset(args.dataset_dir,
                                        out_dir=args.out,
                                        signal_domain=args.domain)
    except (VasotoneHarnessError, RuntimeError, OSError,
            ValueError) as e:
        return _fail(f"evaluate-vasotone failed: {e}")
    doc = out["report"]
    b = doc.get("battery") or {}
    print(json.dumps({"WATERMARK": doc["WATERMARK"],
                      "report_json": out["report_json"],
                      "report_md": out["report_md"],
                      "run_dir": out["run_dir"],
                      "surviving_features": doc["surviving_features"],
                      "added_r2_over_b3": b.get("added_r2_over_b3"),
                      "battery_unavailable_reason":
                          (None if b.get("available")
                           else b.get("reason")),
                      "promotion": doc["gates"]["promotion"]},
                     indent=2))
    return 0


def cmd_evaluate_flutter(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    from datasets.registry import verify_registered, RegistryError
    try:
        verify_registered(args.dataset_dir)
    except RegistryError as e:
        return _fail(str(e))
    from evaluation.flutter_metrics import (evaluate_flutter_dataset,
                                            FlutterHarnessError,
                                            rows_from_dataset)
    try:
        rows = rows_from_dataset(args.dataset_dir)
        # rows_from_dataset genuinely ran the production pipeline; the
        # DOMAIN is still whatever the operator declares, and only
        # facial_rppg can ever open a gate
        out = evaluate_flutter_dataset(rows, signal_domain=args.domain,
                                       production_path=True)
    except (FlutterHarnessError, RuntimeError, OSError, ValueError) as e:
        return _fail(f"evaluate-flutter failed: {e}")
    ev = out["evidence"]
    ce = out["combined_endpoint"]
    print(json.dumps({
        "WATERMARK": ("RESEARCH ARTIFACT — regular-tachyarrhythmia "
                      "pattern flag, §F-gated, NOT VALIDATED"),
        "run_dir": out["run_dir"], "run_id": out["run_id"],
        "n_scans": ev["data"]["n_scans"],
        "rate_bias_bpm": ev["f0"].get("bias_bpm"),
        "sensitivity_2to1": ev["f1"].get("sensitivity_2to1"),
        "specificity_battery": ev["f1"].get("specificity_battery"),
        "shipping_rule": out["shipping_rule"],
        "slow_block_sensitivity": ev["f2"].get("slow_block_sensitivity"),
        "recommended_endpoint": ce.get("recommended"),
        "promotion": ("OPEN" if out["verdict"]["promotion_open"]
                      else "BLOCKED")}, indent=2))
    return 0


def _regularity_rows(args):
    from datasets.registry import verify_registered
    verify_registered(args.dataset_dir)
    from evaluation.regularity_metrics import rows_from_dataset
    return rows_from_dataset(args.dataset_dir)


def cmd_regularity_floor(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    from datasets.registry import RegistryError
    try:
        rows = _regularity_rows(args)
        from evaluation.regularity_floor import regularity_floor_report
        doc = regularity_floor_report(rows, signal_domain=args.domain,
                                      beat_error_trials=int(args.trials))
    except (RegistryError, RuntimeError, OSError, ValueError) as e:
        return _fail(f"regularity-floor failed: {e}")
    cells = doc["mdi"]["cells"]
    print(json.dumps({
        "WATERMARK": ("RESEARCH ARTIFACT — noise-floor characterization, "
                      "§R-gated; the MDI is a PUBLISHED product parameter"),
        "run_dir": doc["run_dir"], "run_id": doc["run_id"],
        "n_paired_scans": doc["n_paired_scans"],
        "interpolation_gain": doc["interpolation_gain"],
        "mdi_ms": {k: v.get("mdi_ms") for k, v in cells.items()},
        "not_characterized": [k for k, v in cells.items()
                              if v.get("mdi_ms") is None],
        "false_irregular_at_5pct_beat_error": {
            fps: next((r["clean_runs"]["false_irregular_rate"] for r in rs
                       if abs(r["beat_error_rate"] - 0.05) < 1e-9), None)
            for fps, rs in doc["beat_errors"]["fps"].items()}},
        indent=2, default=str))
    return 0


def cmd_evaluate_regularity(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    from datasets.registry import RegistryError
    try:
        rows = _regularity_rows(args)
        from evaluation.regularity_metrics import evaluate_regularity_dataset
        out = evaluate_regularity_dataset(rows, signal_domain=args.domain,
                                          production_path=True)
    except (RegistryError, RuntimeError, OSError, ValueError) as e:
        return _fail(f"evaluate-regularity failed: {e}")
    ev = out["evidence"]
    print(json.dumps({
        "WATERMARK": ("RESEARCH ARTIFACT — pulse-regularity evaluation, "
                      "§R-gated, NOT VALIDATED"),
        "run_dir": out["run_dir"], "run_id": out["run_id"],
        "n_scans": ev["data"]["n_scans"],
        "ceiling_kappa": ev["r0"].get("kappa"),
        "ceiling_index_r": ev["r0"].get("index_r"),
        "rsa_flag_rate": ev["r2"].get("rsa_flag_rate"),
        "age_strata": {b: s.get("specificity")
                       for b, s in (ev["r2"].get("age_strata") or {}).items()},
        "head_vs_b3_b4_b5": {k: ev["r3"].get(k) for k in (
            "head_balanced_accuracy", "b3_balanced_accuracy",
            "b4_balanced_accuracy", "b5_balanced_accuracy")},
        "promotion": ("OPEN" if out["verdict"]["promotion_open"]
                      else "BLOCKED")}, indent=2, default=str))
    return 0


def cmd_vascular_fidelity(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    from datasets.registry import verify_registered, RegistryError
    try:
        verify_registered(args.dataset_dir)
    except RegistryError as e:
        return _fail(str(e))
    from research.vascular.fidelity import fidelity_study
    try:
        out = fidelity_study(args.dataset_dir, out_dir=args.out,
                             signal_domain=args.domain)
    except (RuntimeError, OSError, ValueError) as e:
        # VascularError subclasses RuntimeError; the pipeline's
        # own RuntimeErrors keep the 0/2 exit contract too
        return _fail(f"vascular-fidelity failed: {e}")
    doc = out["report"]
    print(json.dumps({"WATERMARK": doc["WATERMARK"],
                      "report_json": out["report_json"],
                      "report_md": out["report_md"],
                      "run_dir": out["run_dir"],
                      "surviving_features": doc["surviving_features"],
                      "promotion": doc["gates"]["promotion"]}, indent=2))
    return 0


def cmd_demo(args) -> int:
    rc = _runtime_preflight()
    if rc:
        return rc
    if args.synthetic:
        import runpy
        runpy.run_path(str(pathlib.Path(__file__).parent / "scripts" /
                           "demo_pipeline.py"), run_name="__main__")
        return 0
    # LIVE demo: local server + browser UI. The browser owns the camera
    # (permission prompt, live preview); every frame is streamed raw to this
    # process, recorded losslessly, and analysed by the production pipeline.
    from app.server import main as live_main
    live_main(port=args.port, open_browser=not args.no_browser)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cli.py", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="capture/sync gates on a manifest")
    v.add_argument("recording")
    v.set_defaults(fn=cmd_validate)

    fz = sub.add_parser("falsify",
                        help="inferred-ECG falsification report (M4; "
                             "research-only, watermarked)")
    fz.add_argument("--source", choices=["mimic", "synthetic"],
                    default="synthetic")
    fz.add_argument("--limit", type=int, default=0,
                    help="subjects per class (0 = all)")
    fz.add_argument("--steps", type=int, default=400)
    fz.add_argument("--out", default=None)
    fz.set_defaults(fn=cmd_falsify)

    tr = sub.add_parser("train", help="config-driven training run (M3.10; "
                                      "a 'reconstruction:' config runs the "
                                      "v0.3 research track)")
    tr.add_argument("config")
    tr.set_defaults(fn=cmd_train)

    gs = sub.add_parser("gate-status",
                        help="promotion scoreboards: §G reconstruction "
                             "(default) or --track vo2 for the §V "
                             "fitness track — every gate, its evidence, "
                             "and why it is red or green")
    gs.add_argument("--track",
                    choices=["ecg", "vo2", "vascular", "vasotone",
                             "flutter", "regularity"],
                    default="ecg")
    gs.add_argument("--html", default=None, metavar="OUT_HTML",
                    help="also write the HTML scoreboard")
    gs.set_defaults(fn=cmd_gate_status)

    rc = sub.add_parser("reconstruct",
                        help="face video -> RESEARCH ECG-reconstruction "
                             "artifact + fidelity report (v0.3; "
                             "watermarked, never a consumer output)")
    rc.add_argument("video")
    rc.add_argument("--manifest", default=None,
                    help="recording manifest; a reference_dir field "
                         "enables fidelity-vs-reference numbers")
    rc.set_defaults(fn=cmd_reconstruct)

    pm = sub.add_parser("promote",
                        help="gate + register a trained model (M3.12)")
    pm.add_argument("run_dir")
    pm.set_defaults(fn=cmd_promote)

    rg = sub.add_parser("register-dataset",
                        help="register a dataset directory (M2.9)")
    rg.add_argument("dataset_dir")
    rg.add_argument("--license-class", default="internal_consented")
    rg.add_argument("--consent-class", default="research_v1")
    rg.add_argument("--campaign", default=None)
    rg.set_defaults(fn=cmd_register_dataset)

    ir = sub.add_parser("ingest-reference",
                        help="paired-ECG reference into a session dir (M2.8)")
    ir.add_argument("session_dir")
    ir.set_defaults(fn=cmd_ingest_reference)

    c = sub.add_parser("campaign", help="capture-campaign tools (M2.7)")
    c.add_argument("action", choices=["status"])
    c.add_argument("campaign_dir")
    c.set_defaults(fn=cmd_campaign)

    p = sub.add_parser("process", help="video -> ScanResult JSON")
    p.add_argument("video")
    p.add_argument("--manifest", default=None)
    p.add_argument("--report", default=None, metavar="OUT_HTML",
                   help="write the standalone Cardiac Rhythm Scan Report "
                        "(v0.2.1) to this path")
    p.add_argument("--heads", default=None,
                   help="comma list of endpoint heads to run "
                        "(e.g. afib,rate_flags,rhythm_map); afib is "
                        "mandatory; vascular/vasotone are research-"
                        "flagged and write watermarked reports instead "
                        "of touching the result")
    p.add_argument("--provocation", default=None,
                   help="provocation record for --heads vasotone "
                        "(phase marks; research surface only)")
    p.add_argument("--pi", default=None,
                   help="contact perfusion-index sidecar for --heads "
                        "vasotone (optional reference arm)")
    p.set_defaults(fn=cmd_process)

    se = sub.add_parser("session",
                        help="three-phase recovery session (v0.4): rest/"
                             "activity/recovery videos + manifest -> "
                             "SessionResult JSON (measured recovery "
                             "physiology; never a fitness claim while §V "
                             "is red)")
    se.add_argument("videos", nargs="*",
                    help="optional phase videos overriding the manifest, "
                         "in rest/activity/recovery order")
    se.add_argument("--manifest", required=True,
                    help="session manifest (see scripts/"
                         "make_synth_recovery.py for the shape)")
    se.set_defaults(fn=cmd_session)

    e = sub.add_parser("evaluate", help="dataset dir -> gate report")
    e.add_argument("dataset_dir")
    e.add_argument("--out", default=None, help="report output directory")
    e.add_argument("--candidate", default=None,
                   help="training run dir -> mandatory-baselines margins")
    e.set_defaults(fn=cmd_evaluate)

    ef = sub.add_parser("evaluate-fitness",
                        help="CPET-labeled session dataset -> §6 fitness "
                             "harness (baseline ladder, falsification "
                             "probes, §V verdict + scoreboard)")
    ef.add_argument("dataset_dir")
    ef.add_argument("--out", default=None)
    ef.add_argument("--domain", default="synthetic",
                    choices=["synthetic", "public_ppg", "facial_rppg"],
                    help="evidence domain recorded with the run; only "
                         "facial_rppg evidence can ever open a §V gate")
    ef.set_defaults(fn=cmd_evaluate_fitness)

    vf = sub.add_parser("vascular-fidelity",
                        help="paired facial+contact-PPG dataset -> V0 "
                             "signal-fidelity study (per-feature ICC / "
                             "Bland-Altman / retest, surviving set, "
                             "vascular-gate verdict; research-only, "
                             "watermarked, never a consumer output)")
    vf.add_argument("dataset_dir")
    vf.add_argument("--out", default=None)
    vf.add_argument("--domain", default="synthetic",
                    choices=["synthetic", "public_ppg", "facial_rppg"],
                    help="evidence domain recorded with the run; only "
                         "facial_rppg evidence can ever open a gate")
    vf.set_defaults(fn=cmd_vascular_fidelity)

    ev = sub.add_parser("evaluate-vascular",
                        help="cfPWV-referenced dataset -> baseline "
                             "battery (B1-B5, the T2 age-shortcut "
                             "defense) + retest/site/fairness evidence "
                             "+ vascular-gate verdict and scoreboard")
    ev.add_argument("dataset_dir")
    ev.add_argument("--out", default=None)
    ev.add_argument("--domain", default="synthetic",
                    choices=["synthetic", "public_ppg", "facial_rppg"],
                    help="evidence domain recorded with the run; only "
                         "facial_rppg evidence can ever open a gate")
    ev.set_defaults(fn=cmd_evaluate_vascular)

    et = sub.add_parser("evaluate-vasotone",
                        help="provocation dataset (facial + contact-PI "
                             "+ null arms) -> tone harness: null-arm "
                             "survival, W0 reference tracking, T2 "
                             "baseline battery, \u00a7W verdict + "
                             "scoreboard")
    et.add_argument("dataset_dir")
    et.add_argument("--out", default=None)
    et.add_argument("--domain", default="synthetic",
                    choices=["synthetic", "public_ppg", "facial_rppg"],
                    help="evidence domain recorded with the run; only "
                         "facial_rppg evidence can ever open a gate")
    et.set_defaults(fn=cmd_evaluate_vasotone)

    ef = sub.add_parser("evaluate-flutter",
                        help="labeled rhythm dataset -> §F harness: "
                             "rate accuracy in arrhythmia, the "
                             "hard-negative battery, per-ratio "
                             "sensitivity incl. the known miss, the "
                             "serial signature, head-vs-B3, and the "
                             "combined AF-or-flutter endpoint")
    ef.add_argument("dataset_dir")
    ef.add_argument("--domain", default="synthetic",
                    choices=["synthetic", "public_ppg", "facial_rppg"],
                    help="evidence domain recorded with the run; only "
                         "facial_rppg evidence can ever open a gate")
    ef.set_defaults(fn=cmd_evaluate_flutter)

    rf = sub.add_parser("regularity-floor",
                        help="paired dataset -> the camera's own noise "
                             "floor: timing-jitter budget, minimum "
                             "detectable irregularity per fps/SQI/skin "
                             "tone, and false irregularity vs beat-error "
                             "rate (§R1; the MDI is a published product "
                             "parameter)")
    rf.add_argument("dataset_dir")
    rf.add_argument("--domain", default="synthetic",
                    choices=["synthetic", "public_ppg", "facial_rppg"])
    rf.add_argument("--trials", default=30, type=int,
                    help="beat-error simulation trials per setting")
    rf.set_defaults(fn=cmd_regularity_floor)

    er = sub.add_parser("evaluate-regularity",
                        help="paired dataset -> §R harness: the camera-vs-"
                             "ECG ceiling test, RSA/benign separation with "
                             "age strata, baselines B1-B5, fairness, "
                             "verdict + scoreboard")
    er.add_argument("dataset_dir")
    er.add_argument("--domain", default="synthetic",
                    choices=["synthetic", "public_ppg", "facial_rppg"])
    er.set_defaults(fn=cmd_evaluate_regularity)

    ir = sub.add_parser("research-report",
                        help="RESEARCH ARTIFACT: every gated head's raw "
                             "output for one scan (rhythm regularity, "
                             "atrial flutter, arterial stiffness, vascular "
                             "tone, VO2/fitness) beside its track's live "
                             "gate status — for research and investigation "
                             "only, never the consumer report")
    ir.add_argument("video")
    ir.add_argument("--manifest", help="capture manifest JSON (optional)")
    ir.add_argument("--age", type=float, default=None,
                    help="participant age in years (optional)")
    ir.add_argument("--session", default=None,
                    help="three-phase session manifest for the VO2/fitness "
                         "track (optional)")
    ir.add_argument("--videos", nargs="*", default=None,
                    help="rest activity recovery videos overriding the "
                         "session manifest (optional)")
    ir.add_argument("--provocation", default=None,
                    help="timed provocation marks JSON for the vascular-"
                         "tone track (optional)")
    ir.add_argument("--pi", default=None,
                    help="contact perfusion-index reference JSON (optional)")
    ir.add_argument("--out", default=None,
                    help="output directory (default research/runs/"
                         "investigation/<recording_id>)")
    ir.set_defaults(fn=cmd_research_report)
    d = sub.add_parser("demo", help="LIVE camera scan demo in your browser "
                                    "(--synthetic for the offline demo)")
    d.add_argument("--synthetic", action="store_true",
                   help="run the offline synthetic pipeline demo instead")
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--no-browser", action="store_true",
                   help="do not auto-open the browser")
    d.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
