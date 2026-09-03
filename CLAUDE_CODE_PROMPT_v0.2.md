# Claude Code prompt — AvatarX "Inferred ECG" v0.2 (AFib = Milestone 1)

<!--
HOW TO USE
  1. Place this file in the repo root at AvatarX/code/afib/ (next to CLAUDE_CODE_SPEC.md).
  2. cd into the repo and start Claude Code:  claude
  3. Say: "Read CLAUDE_CODE_PROMPT_v0.2.md and execute it."
-->

---

You are building **AvatarX v0.2 — the "Inferred ECG" platform**. Read this
definition first, because every design decision below flows from it:

> **"Inferred ECG" means: the clinically useful conclusions a doctor would
> draw from an ECG about RHYTHM — rate, regularity, beat-to-beat pattern,
> AFib-suggestive irregularity — inferred from measured optical signals.
> It does NOT mean a generated ECG waveform.** A camera measures the
> mechanical/hemodynamic consequences of the heartbeat, not its electrical
> activity; a synthesized ECG trace would be beat-timed truth wrapped in
> hallucinated morphology (P/T/ST from training priors). This platform
> therefore renders only measured signals and rhythm inferences — and it
> contains a research module whose job is to *prove*, with our own data,
> why generated ECG waveforms don't ship.

Milestone 1 is the AFib screening head (already substantially built) plus the
**Rhythm Map** — the consumer surface that delivers the "inferred ECG" experience
on honest ground. Around it you will build the platform that everything later
grows on: an **endpoint-head registry** (so new rhythm capabilities are plug-ins,
not rewrites), a **data engine** (capture campaigns with paired-ECG ground truth),
and a **training engine** (retrain, baseline-gate, version, promote).

## Step 0 — Preflight (before writing any code)

1. Work **in the existing repo `AvatarX/code/afib`. Do NOT scaffold a new
   codebase under `AvatarX/code/ecg_afib`.** The v0.1 core (capture, PRBS sync,
   POS/CHROM rPPG, gates, beat detector, clean runs, Model A, decision logic,
   139 tests) is exactly the foundation this platform needs; a parallel repo
   would fork the safety invariants. If the owner wants the `ecg_afib` name,
   rename/move the repo folder at the end (`git mv` history intact) — one
   codebase, one set of invariants.
2. Run the suite: `python3 -m pytest tests/ -q` — **must report 139 passed.**
   If the repo is missing or tests fail, STOP and report.
3. Read `CLAUDE_CODE_SPEC.md`. **It remains the authority.** Where this prompt
   extends it, append a new changelog section (`## B.15 v0.2 changelog`) —
   never rewrite existing sections. If anything here conflicts with the spec,
   the spec wins; flag the conflict in your final summary.
4. Small commits, one per task; the full suite green at every commit.

## What v0.2 IS

A layered platform, offline-first, pure Python, permissive dependencies only:

```
L0 Capture & Sessions   capture/            (exists; extend: campaigns, paired-ECG ingestion)
L1 Signal extraction    rppg/ (rbcg/ stays ablation-only per spec B.7)
L2 Quality & SQI gates  preprocessing/, inference/ gates   (exists; extend: fairness telemetry)
L3 Beat lattice         beats/ + clean runs (exists; formalize as versioned BeatLattice schema)
L4 Endpoint heads       heads/              (NEW: registry; head_afib is head #1)
L5 Decision & report    inference/, app/    (exists; extend: ScanResult v2, Rhythm Map)
L6 Data engine          datasets/           (exists; extend: registry, adjudication, lineage)
L7 Training engine      training/           (NEW: runs, baselines, promotion gates, model registry)
L8 Falsification lab    evaluation/inferred_ecg/  (NEW: research-only; never reaches app/)
```

## What v0.2 IS NOT (do not build any of this)

- **No consumer- or clinician-facing generated/synthesized ECG waveform. Ever.**
- No QT/QTc, PR, QRS-width, or ST estimation heads. No ischemia/heart-attack
  anything. No "beat morphology classification" (PVC-vs-PAC) heads — flutter
  and ectopy exist below only as *research-flag stubs*.
- No diagnosis language; `ScanResult.user_facing_text()` remains the single
  gate for user-facing strings.
- All standing v0.1 exclusions remain: no rBCG in the production path
  (patent-encumbered; ablation-only), no RAIL-licensed checkpoints, no
  demographic features in any classifier, no cloud services.

## Non-negotiable invariants (adds 9–14 to the spec's 1–8; violating any is a bug)

9.  **Measurement-class labeling.** Every field in `ScanResult` v2 carries a
    class: `MEASURED` (pulse waveform, beat times, rate, SQI),
    `INFERRED_RHYTHM` (rhythm classification, AFib suggestion, confidence),
    or `RESEARCH_SYNTHETIC` (falsification-lab artifacts). `app/` may render
    only the first two. A test enforces the enum on every schema field.
10. **The falsification lab is quarantined.** Its outputs are watermarked
    `SYNTHETIC ECG — RESEARCH ONLY — NOT A MEASUREMENT` (burned into every
    plot and JSON header), written only under `evaluation/falsification_runs/`,
    and a test asserts no import path from `app/` or `inference/pipeline.py`
    into `evaluation/inferred_ecg/`.
11. **The word "ECG" in user-facing text may appear only inside the referral
    sentence** ("…confirm with an ECG"). The consumer surface is named
    **Rhythm Map**. Add a test that greps rendered app strings.
12. **Paired ECG ground truth flows one way** — into training/evaluation.
    It is never echoed into any consumer artifact.
13. **No model ships without beating the mandatory baselines** (Task M3) on
    participant-disjoint AND session-disjoint splits, evaluated by the same
    gated production path (`cli.py process`), never a side harness.
14. **Extension = new head, not new pipeline.** Any future endpoint must be
    implemented as a `heads/` plug-in consuming the BeatLattice; anything that
    needs to bypass L2/L3 is by definition out of scope and must be escalated.

## Ordered milestones (each task: tests first, then code; suite green per commit)

### M1 — AFib head + Rhythm Map (the shippable core)

1. **`heads/` registry.** `heads/base.py` defines `EndpointHead` (name, version,
   `required_inputs`, `run(BeatLattice, context) -> HeadResult`), a registry, and
   per-head enable flags in `configs/default.yaml`. `HeadResult` carries
   measurement-class, value payload, confidence, and reasons.
2. **`head_afib`.** Wrap the existing Model A + decision logic as the first
   registered head with **behavior locked by regression tests** (identical
   outputs to v0.1.5 on the synthetic corpus and E6 fixtures — bit-for-bit on
   the sanctioned sentence and star grade).
3. **`head_rate_flags`.** Sustained brady (<50) / tachy (>100) flags from the
   clean-run rate, with the same abstention discipline (insufficient clean
   intervals → no flag). Sanctioned sentences added to the spec changelog.
4. **`head_rhythm_map`** — the "inferred ECG" consumer surface. Renders, from
   MEASURED data only: the pulse waveform with detected-beat ticks (exists in
   demo — promote it), a tachogram strip, a Poincaré plot, per-beat confidence
   shading, capture caveats, and the head_afib sanctioned sentence + stars.
   Static SVG/HTML into the existing results page; no new frameworks.
5. **`ScanResult` v2** (schema version bump in `datasets/schema.py`): per-head
   results, measurement classes, device/capture metadata. `user_facing_text()`
   unchanged as the single string gate. Migration shim + tests for v1 readers.
6. CLI: `python3 cli.py process <video> --heads afib,rate_flags,rhythm_map`.

### M2 — Data engine (capture what the future models need)

7. **Campaign manifests.** `datasets/campaigns.py`: a YAML campaign spec —
   target N, enrollment quotas (Fitzpatrick I–VI counts, device matrix,
   lighting conditions), consent/version fields, and per-session capture
   checklist. `cli.py campaign status <dir>` reports quota fill.
8. **Paired-ECG reference ingestion.** `cli.py ingest-reference <session_dir>`:
   accepts a reference ECG export (CSV/WFDB), verifies PRBS/LED sync overlap,
   computes and stores video↔ECG clock offset + drift, R-peak series, and an
   **episode-locked rhythm-label table** (label applies to a time span, not a
   patient) with adjudicator fields (initials, date, ESC-definition checkbox).
   Fail closed on sync error > the spec's threshold.
9. **Dataset registry.** `datasets/registry.jsonl`: every registered dataset
   gets id, content hashes, campaign ref, license/consent class, and lineage.
   `datasets/splits.py` extended to read the registry and emit
   participant+session-disjoint splits by dataset id. `cli.py evaluate`
   consumes registered datasets only.

### M3 — Training engine (improve without breaking trust)

10. **`training/` runs.** Config-driven (`configs/train_*.yaml`), deterministic
    seeds, input = registry dataset ids + split id; output = a model artifact +
    a run record (data lineage, config hash, git rev, metrics).
11. **Mandatory-baselines harness.** Implement as heads-compatible models:
    tachogram-statistics (extend `models/baseline.py`), HR-only,
    participant-history (predicts from enrollment template — the memorization
    detector), and device/site (predicts from metadata alone — the leakage
    detector). `cli.py evaluate` reports the candidate's margin over every
    baseline on the disjoint splits, plus the existing leakage audit and
    no-read parity by Fitzpatrick group.
12. **Model registry + promotion.** `models/registry.jsonl` with semver, run
    record ref, and eval report hash. `cli.py promote <model>` refuses unless:
    beats all baselines with the spec-defined margins, no-read parity within
    threshold, all gates green. Promotion appends to the spec changelog.

### M4 — Inferred-ECG falsification lab (research-only, quarantined)

13. **Decoder.** `evaluation/inferred_ecg/decoder.py`: a small permissively-
    licensed PPG→ECG sequence model (conv/GRU scale, CPU-trainable) trained on
    the repo's cached MIMIC PERform AF/non-AF paired PPG+ECG, participant-
    disjoint splits only.
14. **The three tests that matter**, as an automated report
    (`cli.py falsify` → `evaluation/falsification_runs/<ts>/report.html`):
    (a) **identity-template baseline** — re-time each held-out subject's
    training-period average ECG beat to the observed pulse beats; compare to
    the decoder on identical metrics; (b) **interval-level error** — P/QRS/T
    fiducial and interval errors in ms against reference ECG (not waveform
    correlation; correlation may be reported only alongside it); (c)
    **morphology/rhythm challenge** — decoder trained without AF reconstructing
    AF segments, and vice versa. Every artifact watermarked per invariant 10.
15. **Write-up.** `docs/inferred_ecg_falsification.md`: methods, results, and
    the standing conclusion template. Expected outcome — the decoder fails to
    beat the identity baseline on morphology and fails the interval test —
    documented as the *purpose* of the module: AvatarX's quantitative,
    reproducible answer to "why don't you just show an ECG?"

### M5 — Growth stubs (flagged research, no user-facing output)

16. **`head_flutter_suspicion`** (research flag only): sustained regular
    ~140–160 rate, abnormally low interval dispersion, and cross-scan rate
    steps at integer ratios across a session series. Output = internal flag +
    logged evidence; no sanctioned sentence yet.
17. **`head_burden`**: aggregation of `head_afib` results across a registered
    series of scans (per-day AF-suggestive fraction with abstention-aware
    denominators). Internal only until a burden claim is validated.

## Definition of done

- All prior tests green throughout; every new module tested; total suite grows,
  never shrinks. Forbidden-output tests in place (invariants 9–11).
- `cli.py demo` unchanged for the user except the results page now shows the
  Rhythm Map. `cli.py process|evaluate|train|promote|ingest-reference|campaign|falsify`
  all runnable end-to-end on synthetic fixtures.
- `README.md` updated; `CLAUDE_CODE_SPEC.md` gains `## B.15 v0.2 changelog`
  (append-only) documenting: the inferred-ECG definition, invariants 9–14, the
  head registry, schema v2, and the falsification-lab quarantine.
- Final summary lists: what was built, any spec conflicts flagged, test count
  before/after, and the three things a future v0.3 should do first.

## Escalation rule

If any task appears to require displaying a synthesized ECG, estimating an ECG
interval, weakening a gate, or moving falsification-lab code toward `app/` —
**STOP and flag it in the summary instead of building it.** That boundary is
the product strategy, not a technical limitation.
