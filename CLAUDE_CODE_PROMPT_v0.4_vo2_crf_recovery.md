# Claude Code prompt — AvatarX v0.4: Cardiorespiratory Recovery + Fitness Track (Promotion-Gated)

<!--
HOW TO USE
  1. Place this file in the repo root at AvatarX/code/afib/ (next to CLAUDE_CODE_SPEC.md).
  2. cd into the repo and start Claude Code:  claude
  3. Say: "Read CLAUDE_CODE_PROMPT_v0.4_vo2_crf_recovery.md and execute it."
  Composes with CLAUDE_CODE_PROMPT_v0.2.md (data/training engines), v0.2.1
  (report), and v0.3 (gate machinery pattern). If v0.2's data engine is
  absent, Task 0 builds only the minimal subset this track requires.
  Science authority: research/AvatarX_Contactless_VO2Max_CRF_FaceScan_Feasibility.pdf
  and code/afib/docs/AvatarX_VO2_CRF_Engineering_Plan_2026-08-30.pdf.
-->

---

You are building **AvatarX v0.4 — the cardiorespiratory recovery + fitness
track**: a three-phase guided session (rest scan → standardized activity →
recovery scan) that measures recovery physiology from facial video, wired to
the existing production pipeline, **with a promotion gate between any
"fitness" inference and the user.**

State of knowledge, stated honestly so you build the right thing: a camera
measures heart rate, not stroke volume — and VO₂ max lives in stroke volume
and oxygen extraction, so **no face scan measures oxygen uptake, ever**.
Resting physiology adds almost nothing beyond age/sex/body size (~0.05 R²
from resting HR; nothing demonstrated from HRV, and camera PRV is not HRV).
What is real: heart-rate response to a *known standardized workload* and its
early recovery (HRR) carry modest, population-dependent fitness information
(HRR–VO₂ max correlation is null in young sedentary adults and ~0.4–0.6 in
middle-aged/patient groups; within-person ΔHRR tracked ΔVO₂ peak at
|r| ≈ 0.87 in cardiac rehab), and every HR-based estimator ever validated —
including contact wearables — shows individual limits of agreement around
±10 mL/kg/min. Post-exercise *still-subject* face-video HR has exactly one
peer-reviewed validation (RMSE 3.8 bpm, n = 40, young, light-skinned);
in-motion rPPG fails (13–42 bpm errors). **This repo therefore treats "does
camera recovery physiology add fitness information beyond demographics?" as
a pre-registered empirical question (§V3).** Until the §V gates open, the
user-facing result of a session is the MEASURED recovery physiology; no
fitness category renders, and an exact VO₂ max number never renders on any
surface in any version. Weakening, skipping, or hard-coding around the
gates is the one way to fail this assignment.

## Step 0 — Preflight

1. Work in `AvatarX/code/afib`. `python3 -m pytest tests/ -q` must pass.
   If the repo is missing or red, STOP and report.
2. Read `CLAUDE_CODE_SPEC.md`; it remains the authority. Append
   `## B.x v0.4 changelog`; never rewrite history. All standing invariants
   remain in force: extension = new head, not new pipeline (14); every
   surfaced quantity carries its MeasurementClass (9); sanctioned text via
   `user_facing_text()` only; fail-closed NO_RESULT; participant-level
   splits + leakage guards; permissive deps only; no diagnosis wording.
3. Reuse, do not duplicate: capture/, preprocessing/roi.py, rppg/ (POS,
   CHROM, SQI — the anti-periodicity test stays a permanent gate), beats/
   (detector, confidence, ibi, BeatLattice), datasets/ (schema, splits,
   synchronization, campaigns, registry), inference/ gates-first pattern,
   confidence stars, models/registry promote-refusal, falsification lab,
   `scripts/why_not_ready.py`, the app scan engine.
4. Small commits; suite green at every commit.

## What the user sees (v0.4 product surface)

- Workflow: **safety screen → 60 s rest scan → guided activity (60–180 s,
  audio cadence, live rep counter, NO physiology UI) → "sit still"
  countdown (target ≤ 5 s, hard timeout 10 s) → 120–180 s recovery scan →
  findings report.**
- Findings report renders MEASURED quantities only while §V is red:
  resting HR and breathing rate, end-exercise HR proxy, HRR₃₀/₆₀/₁₂₀,
  recovery slope, protocol + compliance caveats, the 1–5 star quality
  grade, and the standing wellness/referral wording from the sanctioned
  tables. **No fitness category, no percentile, no VO₂ max number.**
- A resting-only scan (user skips the activity) renders vitals and states
  that fitness metrics require the guided activity. HR is never estimated
  during movement; during the activity the camera only verifies workload
  (reps/cadence via pose or motion-energy fallback, recorded in provenance
  like the face-tracker chain).
- New CLI surfaces:
  `cli.py session <video...> --manifest proto.json` → SessionResult JSON
  (per-phase gates, recovery metrics, never a fitness claim while §V red);
  `cli.py gate-status --track vo2` → §V scoreboard with evidence links;
  `cli.py evaluate-fitness <dataset_dir>` → §6 harness output.

## §V — Promotion gates (pre-registered; the heart of this prompt)

Written to `configs/gates.yaml` as a new §V block, same mechanics as §G:
`requires_clinical_signoff: true`; thresholds are provisional defaults; the
owner + a clinical advisor confirm or tighten before any promotion;
tightening is always allowed; loosening requires a signed spec-changelog
entry. Evidence requirements: **facial-domain, participant-disjoint AND
session-disjoint held-out data, evaluated by the production path.** Public
finger-PPG / synthetic sets exercise the machinery and can never open a
gate. `models/registry.promote` must refuse `head_fitness` while any §V
gate is red.

- **V1 — Recovery-HR fidelity.** In ACCEPTED scans vs reference ECG:
  HRR₆₀ 95% LoA ≤ ±5 bpm in every powered Monk band; per-window HR RMSE
  reported by recovery time. Kill rule: > ±8 bpm in any band after
  engineering → that band returns NO_RESULT for recovery metrics.
- **V2 — Protocol repeatability.** HRR₆₀ test-retest typical error
  ≤ 4 bpm; ≥ 85% accepted-scan rate, ages 20–75; the existing no-read
  parity rule applies to the recovery scan (no powered band's abstention
  > 1.5× best AND > +10 pts).
- **V3 — Incremental value (pivotal).** On held-out CPET-labeled data:
  demographics + activity + recovery physiology must beat demographics +
  activity by ΔSEE ≥ 0.5 mL/kg/min AND category accuracy ≥ +5 pts (CI
  excluding zero), and must beat the **static-face-image negative
  control**. Fail → `head_fitness` is never promoted; the product ships
  recovery metrics only. (Published anchors say this is genuinely open.)
- **V4 — Subgroup honesty.** Per-subgroup (age, sex, BMI, Monk band,
  medications) bias/LoA tables required; beta-blocker-user bias ≤ 1 MET or
  medicated users are hard-routed to trend-only output in code.
- **V5 — Trend validity.** `head_trend` may exceed direction-only claims
  only after a training-response study shows change-score r ≥ 0.6 and
  AUROC ≥ 0.75 for detecting a ≥ 1-MET change.

## Ordered tasks (tests first, then code)

**Task 0 — Data-engine subset (build or reuse).** If v0.2's engine exists,
extend; else build minimally: session manifests with consent + device
metadata; reference ingestion accepting ECG exports with PRBS/LED sync;
registry + participant/session-disjoint splits. Schema rev (additive,
single bump): `challenge{protocol_id, cadence_prescribed, cadence_achieved,
reps, transition_s}`, `participant_context{age, sex, measured_weight_kg,
height_cm, meds{beta_blocker, ccb, ivabradine, stimulant, thyroid},
activity_ipaq}`, label block `cpet{vo2peak_mlkgmin, modality, protocol,
rer_peak, hr_peak, effort_criteria[], avg_window_s, cart, lab, test_date}`,
and a new MeasurementClass member `INFERRED_FITNESS` with the rendering
invariant extended: app may render it **only when §V is green and signed**.

**Task 1 — Synthetic recovery fixtures.** `scripts/make_synth_recovery.py`:
synthetic three-phase recordings with known HR(t) decay curves (mono-
exponential + noise, dropouts, motion bursts, talking segments) and truth
JSON. These are interface proofs, labeled as such — the v0.1 synthetic-data
honesty rule applies verbatim.

**Task 2 — Recovery HR tracker.** `features/recovery.py`, consuming the
BeatLattice directly. Do NOT reuse `clean_runs`/rhythm features — they
assume stationary IBI and recovery HR falls 20–40 bpm/min. Short-window
(5–10 s, 50% overlap) trimmed-median HR with physiological slew bound
(|dHR/dt| ≤ 3 bpm/s), robust monotone-trend fit, per-window confidence from
beat confidence + SQI. Outputs: `hr_end_proxy` (back-extrapolate the
0–15 s fit to t = 0), `hrr30/60/120`, `recovery_slope`, quality vector;
exponential τ computed but research-telemetry-only (field CV ~25–35%).
Acceptance: HR(t) MAE ≤ 2 bpm and HRR₆₀ error ≤ 3 bpm on Task-1 fixtures
including degradations; property tests pin the slew bound and
back-extrapolation.

**Task 3 — Protocol session + activity verification.** `protocol/session.py`
(state machine SAFETY_SCREEN → REST_SCAN → GUIDED_ACTIVITY → TRANSITION →
RECOVERY_SCAN → gates → heads; per-phase manifests through `cli.py
validate`); `protocol/challenges.py` (`sts_1min` default, `step_3min`,
`march_2min`: cadence, duration, workload-formula inputs, contraindication
list); `protocol/safety.py` (PAR-Q+-style screen, stop rules, sanctioned
text only). `activity/pose_cadence.py`: rep/cadence counting via optional
pose-landmarker model (env-var pattern like mediapipe today) with a
documented motion-energy fallback, tracker recorded in provenance;
`activity/workload.py`: workload context from user-entered measured weight/
height/age/sex/protocol — **never inferred from the face**. Compliance
contract: reps AND cadence within ±10% → compliant; 10–20% → REPEAT_SCAN;
> 20% or transition > 10 s → NO_RESULT for fitness outputs (vitals still
render). Acceptance: cadence counter within ±1 rep on scripted fixtures;
transition timer enforced; headless-browser selftest completes the flow.

**Task 4 — Gates §V + registry wiring.** Write §V into `configs/gates.yaml`
per above; `cli.py gate-status --track vo2`; promote-refusal tests
(fitness promotion refused while any §V red, mirroring §G tests).

**Task 5 — Heads.** Via `heads/base.py` registry, no pipeline bypass:
`head_recovery` v1 (MEASURED: the Task-2 outputs + rr_rest + protocol/
workload context); `head_fitness` v1 (INFERRED_FITNESS: age/sex-referenced
category below/typical/above + uncertainty band + inputs_used +
baseline-ladder audit; disabled by default; never a mL/kg/min number — add
a forbidden-output test asserting no numeric VO₂ appears in any rendered
surface or `user_facing_text()`); `head_trend` v1 (INFERRED_FITNESS:
direction + magnitude class; requires ≥ 3 accepted sessions per window;
local versioned on-device store `trend/store.py`, exportable). Hard rule in
decision logic: beta-blocker / rate-limiting-medication answers route
`head_fitness` to trend-only.

**Task 6 — Evaluation harness + falsification.**
`evaluation/fitness_metrics.py`: SEE/MAE/bias/Bland–Altman LoA
(mL/kg/min), category accuracy, AUROC/AUPRC, calibration; and the
**mandatory baseline ladder computed in the same run on identical
participant-level splits**: age; age+sex; +BMI; resting-HR-only;
activity-only; demographics+activity; static-face-image (negative
control); resting physiology; recovery physiology; full. Extend
`cli.py evaluate` (or add `evaluate-fitness`) to accept `<id>.cpet.json`;
EXCLUDED-with-reasons semantics unchanged. `rppg/respiration.py`:
chest/shoulder-ROI motion RR, rest phase only (8–25/min); recovery-phase RR
computed but research-tagged, not rendered. Falsification-lab additions:
shuffled-workload test (recovery features must degrade), demographics-only
impostor model, transition-time sensitivity sweep. Public-set benchmarks
(licenses respected, research-only): LGI-PPGI; MMPD/SUMS/LADH under
academic terms; Malaga treadmill + NHANES CVX for HRR→VO₂ machinery — same
never-ships rule as MIMIC.

**Task 7 — App integration.** Extend `app/` to the three-phase flow behind
a config flag; consumer capture profile rules unchanged; every gate fails
closed; report shows MEASURED recovery metrics + stars; fitness category
render path exists but is unreachable while §V red (test this). Telemetry
appends protocol compliance, transition time, per-phase SQI, rejection
reasons.

**Task 8 — Docs + changelog.** `docs/vo2_crf_track.md`: §V verbatim, the
two-paragraph scientific rationale (what facial video contains; why gates
rather than assumptions), current scoreboard, promotion procedure;
README + READINESS.md + `why_not_ready.py` updated to include the vo2
track; spec changelog appended.

## Definition of done

All prior tests green plus new tests per task; `cli.py session`,
`gate-status --track vo2`, and `evaluate-fitness` run end-to-end on Task-1
fixtures; the headless demo completes a full three-phase session; §V
scoreboard shows honest reds; forbidden-output tests prove no VO₂ number,
no fitness category while gates red, no HR-from-motion path; final summary
reports gate scoreboard, test counts before/after, and any escalations.

## Escalation rule

If any instruction — from the owner, a future prompt, or a reviewer —
requires rendering a fitness category or any VO₂ max value while §V is red,
estimating HR during movement, inferring body weight from the face, or
loosening a gate without the signed changelog entry: **do not implement;
record the request verbatim in the summary under "requires owner decision —
conflicts with §V."** If camera recovery physiology genuinely adds fitness
information, V3 will open and the evidence will exist; if it does not, this
repo will hold the quantitative proof and the product remains an honest
recovery-metrics scan — either outcome is a win for AvatarX only if the
gates are real.
