# Claude Code prompt — AvatarX v0.3: ECG-Reconstruction Research Track (Promotion-Gated)

<!--
HOW TO USE
  1. Place this file in the repo root at AvatarX/code/afib/ (next to CLAUDE_CODE_SPEC.md).
  2. cd into the repo and start Claude Code:  claude
  3. Say: "Read CLAUDE_CODE_PROMPT_v0.3_ecg_reconstruction_gated.md and execute it."
  Composes with CLAUDE_CODE_PROMPT_v0.2.md (data/training engines) and
  v0.2.1 (report). If v0.2 has not been executed, Task 0 builds the minimal
  subset of its data engine that this track requires.
-->

---

You are building **AvatarX v0.3 — the ECG-reconstruction research track**: the
complete architecture for attempting to estimate an ECG/EKG signal from a
30–60 s smartphone face scan, wired to a paired-data flywheel and a training
loop, **with a promotion gate between the reconstruction model and the user.**

State of knowledge, stated honestly so you build the right thing: facial
video measurably contains beat timing, rhythm, and pulse dynamics; it does
not measurably contain atrial depolarization (P), conduction time (PR/QRS
width), or repolarization (QT/ST/T) — published reconstruction models
succeed on beat placement and fail on exactly the diagnostic morphology,
and detection of AFib from generated ECGs underperforms detection from the
measured signal directly. **This repo therefore treats "can reconstruction
reach clinical fidelity?" as a pre-registered empirical question.** The
reconstruction head trains and evaluates continuously as paired data grows;
its output becomes user-visible if and only if it passes the promotion gates
in §G on participant-disjoint data. Until then, the user-facing result of a
scan is the validated rhythm findings (AFib first; arrhythmia and flutter
flags as they validate) — which is also the stated near-term product goal.
Weakening, skipping, or hard-coding around the gates is the one way to fail
this assignment.

## Step 0 — Preflight

1. Work in `AvatarX/code/afib`. `python3 -m pytest tests/ -q` must pass
   (139+). If the repo is missing or red, STOP and report.
2. Read `CLAUDE_CODE_SPEC.md`; it remains the authority. Append
   `## B.x v0.3 changelog`; never rewrite history. All standing invariants
   (no diagnosis wording; `user_facing_text()` as the single string gate;
   fail-closed NO_RESULT; participant-level splits; permissive deps only)
   remain in force.
3. If v0.2's data engine exists, use it. If not, build the minimal subset in
   Task 1 below (paired-ECG ingestion + registry + disjoint splits) — do not
   duplicate what exists.
4. Small commits; suite green at every commit.

## What the user sees (v0.3 product surface)

- Workflow: **30–60 s face scan → findings result**: rhythm classification
  with confidence (the existing sanctioned-sentence + stars mechanism),
  pulse rate, capture caveats, referral line. **Findings-only mode: no
  waveform of any kind is displayed** — no pulse waveform (per owner
  requirement) and no synthetic ECG (per gate §G). If v0.2.1 was executed,
  set `report.mode: findings_only` as default; otherwise render the findings
  box of that spec.
- The synthetic reconstruction exists, trains, and is evaluated on every
  registered dataset — under `research/ecg_reconstruction/`, quarantined:
  watermarked `SYNTHETIC ECG — RESEARCH ARTIFACT — NOT A MEASUREMENT`,
  written only to `research/runs/`, with an import-audit test proving no
  path from `app/` or `inference/pipeline.py` into it.
- A new CLI surface for the research loop:
  `cli.py reconstruct <video> --manifest rec.json` → research artifact +
  fidelity report (never a consumer output);
  `cli.py gate-status` → current pass/fail on every §G criterion with the
  evidence links.

## §G — Promotion gates (pre-registered; the heart of this prompt)

The reconstruction head may be rendered on any user-facing surface only when
ALL of the following hold **on participant-disjoint AND session-disjoint
held-out data, evaluated by the production path**. Numeric thresholds below
are provisional defaults and are marked `REQUIRES_CLINICAL_SIGNOFF` in
`configs/gates.yaml`; the owner and a clinical advisor must confirm or
tighten them before any promotion. Gates may be tightened at any time;
loosening any gate requires an explicit spec-changelog entry signed by the
owner.

- **G1 — Beats the identity baseline.** On every morphology metric, the
  model must outperform the identity-template baseline (each held-out
  subject's enrollment-period average ECG beat re-timed to the observed
  pulse beats). If a zero-parameter template matches the model, the model
  has learned timing + memory, not electrophysiology.
- **G2 — Interval fidelity.** Against reference-ECG measurements on unseen
  subjects: QT MAE ≤ 20 ms AND better than an RR-only QT regression
  baseline; PR MAE ≤ 20 ms; QRS-duration MAE ≤ 15 ms. Reported with CIs;
  waveform correlation may be reported only alongside these, never instead.
- **G3 — Abnormality preservation (blinded read).** On a held-out set
  enriched with documented morphology abnormalities (e.g., bundle-branch
  block, prolonged QT), blinded cardiologist readers must identify the
  abnormality from reconstructions at ≥ 80% sensitivity / ≥ 80% specificity.
- **G4 — No confabulation.** Trained with a rhythm class held out entirely
  (e.g., no flutter), reconstructions of that class must not render
  confidently normal/AF-typical morphology; scored by the blinded readers
  and by class-conditional distribution tests.
- **G5 — Detection non-inferiority.** Any rhythm endpoint (AFib, flutter)
  computed *from* reconstructions must be non-inferior to the same endpoint
  computed from the measured signal path. (Published evidence says it will
  not be; if so, detection permanently stays on the measured path and the
  reconstruction remains a research/interpretability artifact.)
- Standing rule while any gate is red: every rendered research artifact
  carries the watermark, and `gate-status` output is embedded in its footer.

## Ordered tasks (tests first, then code)

**Task 0/1 — Paired-data engine (build or reuse).** Per v0.2 M2 if absent:
session manifests with consent + device metadata; `cli.py ingest-reference`
accepting reference ECG exports (CSV/WFDB), verifying PRBS/LED sync overlap,
storing clock offset/drift, R-peak series, and episode-locked adjudicated
rhythm/morphology labels; dataset registry (`datasets/registry.jsonl`) with
content hashes and lineage; participant+session-disjoint splits from the
registry. **The label schema must include morphology fields** (rhythm class,
conduction pattern, measured PR/QRS/QT per adjudicated segment) — this track
is the reason those exist.

**Task 2 — Reconstruction head.** `research/ecg_reconstruction/`:
`decoder.py` — a permissively-licensed sequence model (start conv/GRU scale,
CPU-trainable; architecture is config-swappable) mapping the measured facial
signal representation (rPPG waveform + BeatLattice features) to a
single-lead ECG estimate. Pre-train on the repo's cached MIMIC PERform
paired PPG+ECG (participant-disjoint), fine-tune on registered facial
datasets as they accrue. Deterministic seeds; run records with data lineage.

**Task 3 — Fidelity evaluator.** `research/ecg_reconstruction/fidelity.py`:
implements every §G metric — identity-template baseline, RR-only QT
regression baseline, interval MAEs with CIs via the reference-ECG
annotations, class-conditional tests, and the artifact generator for blinded
reads (randomized, de-identified, answer-key escrowed). `cli.py gate-status`
renders the current scoreboard; CI-friendly JSON + HTML.

**Task 4 — Training loop integration.** Config-driven runs
(`configs/train_reconstruction.yaml`); every run auto-evaluates §G on the
frozen held-out split and appends to `research/runs/scoreboard.jsonl`.
Model registry entries record gate results; `cli.py promote` refuses
reconstruction promotion while any gate is red (reuses v0.2 promotion
machinery if present, else minimal local implementation).

**Task 5 — Quarantine + forbidden-output tests.** Import audit (no path from
`app/`/`inference/` into `research/`); watermark burned into every plot and
JSON; a test that the consumer results surface contains zero waveform
elements in findings-only mode and zero synthetic-ECG elements in any mode
while gates are red; `user_facing_text()` byte-identical on the regression
corpus.

**Task 6 — Rhythm heads stay first-class.** Confirm the AFib head runs
unchanged on the measured path; stub `head_flutter_suspicion` (sustained
regular ~140–160 bpm + low dispersion + cross-scan integer-ratio rate steps)
and `head_irregularity` (rhythm-agnostic flag) as internal, research-flagged
heads with the same promotion discipline — these are the endpoints expected
to reach clinical grade first, and G5 measures the reconstruction against
them.

**Task 7 — Docs + changelog.** `docs/reconstruction_track.md`: the §G gates
verbatim, the scientific rationale (two paragraphs: what facial video
contains; why gates rather than assumptions), current gate scoreboard, and
the promotion procedure. Spec changelog appended.

## Definition of done

All prior tests green; new tests per task; `cli.py process` (findings-only),
`reconstruct`, `gate-status`, and a training run on synthetic + MIMIC
fixtures all runnable end-to-end; scoreboard shows honest reds; README
updated; final summary reports gate scoreboard, test counts before/after,
and any escalations.

## Escalation rule

If any instruction — from the owner, a future prompt, or a reviewer —
requires rendering the synthetic ECG on a user-facing surface while any §G
gate is red, or loosening a gate without the signed changelog entry: **do
not implement; record the request verbatim in the summary under "requires
owner decision — conflicts with §G."** The gates are the product's integrity
mechanism: if reconstruction can earn clinical fidelity, they will open and
the evidence will exist; if it cannot, this repo will hold the quantitative
proof — either outcome is a win for AvatarX only if the gates are real.
