# The cardiorespiratory recovery + fitness track (v0.4)

**Question, pre-registered (§V3):** does camera recovery physiology add
fitness information beyond demographics? This repo refuses to answer by
assumption. Until the §V gates open, the user-facing result of a
three-phase session is the MEASURED recovery physiology — resting HR and
breathing rate, end-exercise HR proxy, HRR30/60/120, recovery slope,
protocol + compliance caveats, the 1–5 star grade. **No fitness category
renders while §V is red, and an exact VO2 max number never renders on
any surface in any version.**

## Why gates rather than assumptions

A camera measures heart rate, not stroke volume — and VO2 max lives in
stroke volume and oxygen extraction, so no face scan measures oxygen
uptake, ever. Resting physiology adds almost nothing beyond age/sex/body
size (~0.05 R² from resting HR; nothing demonstrated from HRV, and
camera PRV is not HRV). What is real: heart-rate response to a KNOWN
standardized workload and its early recovery carry modest,
population-dependent fitness information — the HRR–VO2max correlation is
null in young sedentary adults and ~0.4–0.6 in middle-aged/patient
groups, while within-person ΔHRR tracked ΔVO2peak at |r| ≈ 0.87 in
cardiac rehab (which is why `head_trend` exists from day one and the
category head may never open). Every HR-based estimator ever validated —
contact wearables included — shows individual limits of agreement around
±10 mL/kg/min.

The measurement side has exactly one peer-reviewed anchor: post-exercise
*still-subject* face-video HR at RMSE 3.8 bpm (n = 40, young,
light-skinned). In-motion rPPG fails outright (13–42 bpm errors), which
is why HR is NEVER estimated during movement here — during the guided
activity the camera verifies workload (reps/cadence) and nothing else,
and body weight is user-entered, never inferred from the face. Whether
this repo's recovery measurements reach V1 fidelity across Monk bands,
and whether they add V3 incremental value on real CPET labels, are open
empirical questions — the gates hold the burden of proof.

## §V — the promotion gates (pre-registered)

Authority: the `vo2:` block of `configs/gates.yaml`
(version `V-2026-08-30-1`). Same mechanics as §G: thresholds are
provisional defaults marked `REQUIRES_CLINICAL_SIGNOFF`; the owner and a
clinical advisor confirm or tighten them before any promotion;
tightening is always allowed; **loosening requires an owner-signed
spec-changelog entry.** Evidence must be facial-domain, participant- AND
session-disjoint, through the production path; public finger-PPG and
synthetic sets exercise the machinery and can never open a gate.
`models/registry.promote` refuses `head_fitness` while any §V gate is
red.

- **V1 — Recovery-HR fidelity.** In ACCEPTED scans vs reference ECG:
  HRR60 95% LoA ≤ ±5 bpm in every powered Monk band; per-window HR RMSE
  reported by recovery time. Kill rule: > ±8 bpm in any band after
  engineering → that band returns NO_RESULT for recovery metrics.
- **V2 — Protocol repeatability.** HRR60 test-retest typical error
  ≤ 4 bpm; ≥ 85% accepted-scan rate, ages 20–75; the existing no-read
  parity rule applies to the recovery scan (no powered band's abstention
  > 1.5× best AND > +10 pts).
- **V3 — Incremental value (pivotal).** On held-out CPET-labeled data:
  demographics + activity + recovery physiology must beat demographics +
  activity by ΔSEE ≥ 0.5 mL/kg/min AND category accuracy ≥ +5 pts (CI
  excluding zero), and must beat the static-face-image negative control.
  Fail → `head_fitness` is never promoted; the product ships recovery
  metrics only. (Published anchors say this is genuinely open.)
- **V4 — Subgroup honesty.** Per-subgroup (age, sex, BMI, Monk band,
  medications) bias/LoA tables required; beta-blocker-user bias ≤ 1 MET
  or medicated users are hard-routed to trend-only output in code (they
  are, today, regardless — `head_fitness` refuses a category on any
  rate-limiting medication answer).
- **V5 — Trend validity.** `head_trend` may exceed direction-only claims
  only after a training-response study shows change-score r ≥ 0.6 and
  AUROC ≥ 0.75 for detecting a ≥ 1-MET change.

## Current scoreboard

Run `python3 cli.py gate-status --track vo2` — the live authority.
State at v0.4 ship: **all five gates RED, promotion BLOCKED.** The only
evidence run on record is the synthetic fixture dataset through
`cli.py evaluate-fitness` — surrogate domain, which exercises every
metric (the baseline ladder, the CIs, the negative control, the
shuffled-workload probe) and can never open a gate. There is no facial
CPET-labeled dataset, no reference-ECG recovery fidelity study, no
test-retest study, no blinded subgroup tables, and the clinical-signoff
block is unsigned. `head_fitness` and `head_trend` run inside sessions,
their INFERRED_FITNESS results ride `head_results` for telemetry, and
nothing renders.

## The session and the loop

```bash
# offline three-phase session (fixtures or field recordings)
python3 cli.py session --manifest proto.json

# the live app flow (safety screen -> rest -> guided activity ->
# sit-still transition -> recovery -> findings report)
AVATARX_THREE_PHASE=1 python3 cli.py demo

# CPET-labeled dataset -> §6 harness (ladder + probes + §V verdict)
python3 cli.py evaluate-fitness <dataset_dir> --domain facial_rppg

# the scoreboard
python3 cli.py gate-status --track vo2
```

Compliance contract (enforced in `protocol/session.py`): reps AND
cadence within ±10% → compliant; 10–20% → REPEAT_SCAN; > 20% or
transition > 10 s → NO_RESULT for recovery/fitness outputs, with resting
vitals still rendered. The safety screen fails closed (any yes or any
unanswered question blocks the activity; a resting scan stays
available). Recovery metrics come from `features/recovery.py` — NOT the
rhythm feature path, which assumes stationary IBI.

Benchmark sets for the machinery (research-only, licenses respected,
same never-ships rule as MIMIC): LGI-PPGI; MMPD/SUMS/LADH under academic
terms for post-exercise robustness; Malaga treadmill + NHANES CVX for
HRR→VO2 modeling. Nothing trained on restricted sets ships.

## Promotion procedure

1. A facial-domain evaluation run (registered dataset, CPET labels,
   reference-ECG fidelity study) leaves every §V gate GREEN in its
   `gate_results.json` under `evaluation/fitness_runs/`.
2. The owner and a clinical advisor sign the `vo2.signoff` block in
   `configs/gates.yaml`, confirming or tightening every threshold.
3. `python3 cli.py promote <run_dir>` — refuses with every failing
   reason while anything is red; on success it registers the model and
   appends the promotion to the spec changelog.
4. Only then may the fitness category render — the path exists in
   `app/report_session.py`, is double-gated (SessionResult field AND a
   live §V check), and is covered by an unreachability test.

If any instruction — from the owner, a future prompt, or a reviewer —
requires rendering a fitness category or any VO2 max value while §V is
red, estimating HR during movement, inferring body weight from the face,
or loosening a gate without the signed changelog entry: do not
implement; record the request verbatim under "requires owner decision —
conflicts with §V."
