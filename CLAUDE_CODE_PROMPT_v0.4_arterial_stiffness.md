# Claude Code prompt — AvatarX v0.4: Arterial-Stiffness Research Track (`head_vascular`)

<!--
HOW TO USE
  1. Place this file in the repo root at AvatarX/code/afib/ (next to CLAUDE_CODE_SPEC.md).
  2. cd into the repo and start Claude Code:  claude
  3. Say: "Read CLAUDE_CODE_PROMPT_v0.4_arterial_stiffness.md and execute it."
  Depends on the v0.2 data/training engines (build the minimal ingestion/registry
  subset per v0.2 M2/M3 if they don't exist yet). Composes with v0.2.1/v0.3.
-->

---

You are building **AvatarX v0.4 — the arterial-stiffness research track**: a
new, research-flagged endpoint head that attempts to estimate large-artery
stiffness from the facial pulse waveform, wired to reference ground truth
(carotid-femoral pulse wave velocity) and defended against the two failure
modes that have made "vascular age from a camera" a marketing swamp.

Scientific context, stated honestly so you build the right thing:

- **Why this biomarker matters.** Central arterial stiffness (gold standard:
  carotid-femoral PWV, cfPWV, by tonometry — SphygmoCor/Complior class
  devices) is an established independent cardiovascular-risk marker and a
  guideline-recognized sign of vascular target-organ damage. A validated
  contactless stiffness signal would be a real asset for the platform.
- **What the face plausibly carries.** Stiffer arteries reshape the pulse:
  faster wave reflections, augmented systolic portion, attenuated dicrotic
  notch. In *contact* PPG this is old, real science (pulse-decomposition and
  second-derivative "aging index" features correlate with age and stiffness).
  In *facial video* the evidence is thin: small feasibility studies extract
  pulse-wave features remotely; **nothing is validated against cfPWV at
  clinical scale, and no cleared product exists.**
- **Threat T1 — morphology may not survive the camera.** ROI averaging,
  optics, band-pass filtering, compression, skin tone, and lighting reshape
  the facial waveform; recent work is formalizing information-theoretic
  limits on camera pulse-*morphology* recovery. Whether stiffness-relevant
  shape features survive our pipeline is an open empirical question — so
  this track starts with a signal-fidelity experiment, not a model.
- **Threat T2 — the age shortcut.** Stiffness correlates strongly with age
  (and BP). A model can score well on "stiffness" by covertly predicting age
  from anything age-correlated in the video and metadata. A stiffness head
  that cannot beat an age+sex+blood-pressure regression **has measured
  nothing and must not ship.** Defeating T1 and T2 *is* the assignment.

## Step 0 — Preflight

1. Work in `AvatarX/code/afib`. `python3 -m pytest tests/ -q` green, else STOP.
2. `CLAUDE_CODE_SPEC.md` is the authority; append `## B.x v0.4 changelog`.
3. All standing invariants hold (fail-closed gates; clean runs; participant+
   session-disjoint splits; measurement-class labeling; no demographic
   features **inside any model** — age/sex/BP appear ONLY in baselines;
   `rbcg/` remains ablation-only pending the patent review).
4. Small commits; suite green at every commit.

## Additional invariants for this track (each becomes a test)

V-a. `head_vascular` is research-flagged: **no user-facing output of any
     kind** — no number, score, trend, or color — until every §V gate is
     green with owner + clinical sign-off recorded in the spec changelog.
V-b. The strings `vascular age`, `artery age`, `PWV`, and any m/s value are
     forbidden on user-facing surfaces while gates are red (token tests).
V-c. Morphology features are computed only on ACCEPT-grade, high-confidence
     beats from the gated production path; no feature computation on
     REPEAT_SCAN/NO_RESULT sessions.
V-d. Every reported experiment states its baseline deltas (§Task 4); a
     result quoted without the B3 delta is a bug.

## Ordered tasks (tests first, then code)

### Task 1 — Signal-fidelity study harness (Gate V0 — run before any modeling)

`research/vascular/features.py`: extract candidate morphology features per
beat and per session median — systolic rise time; normalized upstroke slope;
pulse width at 50% amplitude; dicrotic notch presence, relative amplitude,
and timing; reflection-index-style ratio; second-derivative (a,b,c,d,e-wave)
ratios where derivable at our sampling rate. Compute the identical features
from the **simultaneous contact-PPG reference** in the same sessions (the
capture rig already supports synchronized references). Report per feature:
facial↔contact agreement (ICC, Bland–Altman bias/LoA) and same-day
test–retest repeatability, stratified by Fitzpatrick group and device.
**Features that fail fidelity thresholds (`configs/gates.yaml`,
`REQUIRES_CLINICAL_SIGNOFF`; suggested starting point ICC ≥ 0.75) are
dropped from all downstream modeling — automatically, by config, not by
hand-editing.** `cli.py vascular-fidelity <dataset>` renders the scoreboard.

### Task 2 — Data-engine extension (stiffness ground truth)

Extend the session/label schema (v0.2 M2) with a **session-level reference
block**: cfPWV (m/s) with device model + operator fields (SphygmoCor/
Complior-class export parsing), optional CAVI; same-visit brachial BP, HR at
measurement, antihypertensive/vasoactive medication flags, ambient/skin
temperature note. Campaign spec additions: wide age band (target 20–80),
deliberate BP-range and Fitzpatrick quotas, test–retest sub-protocol (two
scans + two reference reads per visit for a repeatability subset).
Indicative cohort sizing (flagged as planning estimates, to refine after
V0): fidelity/feasibility 60–100 paired participants; development 200–400;
recruit through hypertension/vascular clinics to get true high-stiffness
cases, not just healthy volunteers.

### Task 3 — `head_vascular` (research-flagged)

Heads-registry plug-in consuming the BeatLattice + V0-surviving features;
primary target: regression to measured cfPWV; deliberately simple first
model (regularized linear/GBM on session-median features), config-swappable.
Outputs `HeadResult` with class `INFERRED_RHYTHM`-adjacent research class
(`RESEARCH_VASCULAR`), point estimate + CI, and per-session feature-quality
diagnostics. Internal surfaces only (`cli.py process --heads vascular`
prints to research report, never to the consumer result).

### Task 4 — Baseline battery (the T2 defense; reuse v0.2 M3 machinery)

Mandatory baselines, trained/evaluated on identical splits: **B1** age-only;
**B2** age+sex; **B3** age+sex+brachial-BP (the clinic-available-variables
bar); **B4** HR-only; **B5** demographics+HR. Promotion requires
`head_vascular` to beat **B3** on participant-disjoint held-out data with a
pre-registered margin (e.g., RMSE improvement and added-variance-explained
thresholds — `REQUIRES_CLINICAL_SIGNOFF`), not merely B1. Report all deltas
with CIs in every evaluation.

### Task 5 — §V promotion gates

`cli.py gate-status --track vascular` scoreboard:

- **V0** Signal fidelity: surviving feature set non-empty; per-feature ICC ≥
  threshold vs contact reference; repeatability ICC ≥ threshold.
- **V1** Incremental validity: beats B3 per Task 4 on unseen participants.
- **V2** Repeatability of the *estimate*: same-day test–retest ICC ≥
  threshold; drift within bounds.
- **V3** Generalization: holds on a leave-one-site/device-out cohort.
- **V4** Fairness: coverage and error parity across Fitzpatrick groups
  within pre-registered bounds.
- **V5** Claim mapping (with owner + clinical advisor): first permissible
  surface is a **longitudinal wellness trend** ("your pulse-wave pattern vs
  your own baseline"), explicitly non-diagnostic; any absolute-stiffness or
  screening claim ("elevated stiffness — discuss BP/vascular screening with
  a clinician") requires a further prospective protocol. Until V5 is signed,
  V-a/V-b stand.

### Task 6 — Docs + changelog

`docs/vascular_track.md`: the scientific context above (verbatim threats T1/
T2), gate definitions, current scoreboard, cohort plan; spec changelog
appended. README gains a one-paragraph track summary.

## Definition of done

Suite green throughout; fidelity harness, ingestion, head, baselines, and
gate scoreboard all runnable end-to-end on synthetic fixtures (add a
synthetic stiffness generator that couples feature shifts to a latent
"stiffness" variable so the pipeline is testable before clinical data);
forbidden-token and no-user-output tests in place; final summary reports the
V-gate scoreboard (expected: all red until real paired data exists) and any
escalations.

## Escalation rule

If any instruction requires showing a vascular/stiffness/"vascular age"
output to users while §V gates are red, or removing age/BP baselines from an
evaluation, or moving age/sex/BP *into* the model: **do not implement;
record the request verbatim under "requires owner decision — conflicts with
§V / T2 defense."** A stiffness product that is secretly an age predictor
would eventually be exposed as one; the baselines exist so AvatarX finds out
first.
