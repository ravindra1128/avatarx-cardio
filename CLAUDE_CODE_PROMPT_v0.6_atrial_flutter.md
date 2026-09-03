# Claude Code prompt — AvatarX v0.6: Atrial-Flutter Track (`head_flutter`)

<!--
HOW TO USE
  1. Place this file in the repo root at AvatarX/code/afib/ (next to CLAUDE_CODE_SPEC.md).
  2. cd into the repo and start Claude Code:  claude
  3. Say: "Read CLAUDE_CODE_PROMPT_v0.6_atrial_flutter.md and execute it."
  Promotes the v0.2 M5 `head_flutter_suspicion` stub into a full track.
  Depends on v0.2 data/training engines; shares gate CLI with v0.3–v0.5.
-->

---

You are building **AvatarX v0.6 — the atrial-flutter track**. Read the
physiology first, because it inverts the AFib architecture:

> **AFib is detected because its pulse is irregular. Flutter is the
> arrhythmia our AFib detector is structurally blind to.** In typical
> flutter the atria fire ~250–300 bpm and the AV node conducts a fixed
> fraction — 2:1 (~150 bpm), 3:1 (~100), 4:1 (~75) — producing a pulse that
> is *metronomically regular*. An irregularity-based detector will read that
> as normal sinus rhythm. Flutter also cannot be *named* from the pulse:
> distinguishing it from SVT or sinus tachycardia at the same rate requires
> seeing sawtooth F waves on an ECG, which facial video does not carry.

So this track does not build a "flutter detector." It builds a **rate-and-
regularity pattern flag** that surfaces the *suspicion* of a regular
tachyarrhythmia and routes to ECG, plus the serial-scan signature that is
flutter's most distinctive camera-observable trait.

**Strategic note for the claim design (confirm with the clinical advisor
before any labeling):** clinically, stroke-prevention decisions in atrial
flutter follow essentially the same risk-based anticoagulation approach as
AFib. That means the most defensible *and* most clinically actionable
endpoint may not be "flutter vs AF" discrimination at all, but a **combined
atrial-tachyarrhythmia screen ("AF or flutter suggested — confirm with
ECG")** — where our two heads vote into one referral, and the ECG does the
naming. Task 6 evaluates that combined endpoint explicitly; it is expected
to outperform either head alone and to be the honest product surface.

**Evidence baseline (all the facial-video flutter evidence that exists):**
Cramer et al. 2025 (J Clin Monit Comput), N=51 cardioversion patients (38
AF, 13 flutter), machine-vision camera at 32 Hz / 2 m, leave-one-subject-out
SVM on rPPG interval features: AUC 0.95 arrhythmia-vs-sinus overall, AF
94.6% and flutter 92.5% correctly classified — but **"other arrhythmias"
only 61.8%**, HR bias −7.45 bpm during arrhythmia, one Fitzpatrick-VI
patient excluded. That is a *cardioversion-clinic arrhythmia-vs-sinus*
result, not evidence that flutter can be named or that slow fixed-block
flutter is detectable. Treat it as the floor to beat, not a target.

## Step 0 — Preflight

1. Work in `AvatarX/code/afib`; `pytest -q` green or STOP. `CLAUDE_CODE_SPEC.md`
   is authority; append `## B.x v0.6 changelog`.
2. Standing invariants hold, and note the two that bite hardest here:
   **clean runs only / no interval repair** (flutter's conduction changes are
   signal, not artifact) and **the SQI must never score periodicity** — a
   periodicity-based quality score would *reward* flutter's regularity while
   penalizing AF; verify the anti-periodicity test still guards this.
3. Reuse v0.4/v0.5 gate-scoreboard machinery; do not fork it.
4. Small commits; suite green at every commit.

## Track invariants (each becomes a test)

F-a. `head_flutter` never emits the word **flutter** on a user-facing
     surface, and never names a rhythm. Permitted (post-gate) sanctioned
     sentence family: *"A sustained fast, unusually regular pulse was
     detected. This can occur with several heart-rhythm conditions —
     please have an ECG."* All wording via `user_facing_text()`.
F-b. Research-flagged until §F gates are green with owner + clinical
     sign-off recorded in the changelog. Forbidden tokens on user surfaces
     while red: `flutter`, `SVT`, `atrial tachycardia`, `2:1`, `conduction
     ratio`.
F-c. **The known-miss registry is a shipped artifact, not a caveat.**
     `docs/flutter_limitations.md` must state, and tests must assert the
     product does not contradict: slow fixed-block flutter (~75 bpm,
     regular) is expected to be indistinguishable from normal sinus rhythm
     by pulse timing alone.
F-d. Flutter detection must **never** be routed through the gated ECG
     reconstruction head (v0.3 §G) — measured-path only.

## Ordered tasks (tests first, then code)

### Task 1 — Flutter-signature feature extractor

`features/flutter.py`, on ACCEPT-grade clean runs only. Three families:

1. **Rate fingerprint** — sustained median rate with tight CI in the 2:1
   band (config default 140–165, plus 3:1 ~95–110 and 4:1 ~70–80 bands
   evaluated but expected non-specific), duration of sustainment across the
   scan, absence of exertional context (see Task 3 confounders).
2. **Hyper-regularity** — "too regular" indices: interval dispersion
   (RMSSD/SDNN-class) *below* an age-adjusted floor; **absent respiratory
   modulation** (coupling between the interval series and the camera-derived
   respiration signal at the respiratory frequency — flutter's fixed
   conduction decouples the two, whereas sinus rhythm at any rate retains
   some RSA); spectral concentration of the tachogram.
3. **Serial-scan conduction-ratio steps** (flutter's most distinctive
   camera-observable trait) — across a registered scan series, rate values
   clustering at **integer-ratio steps of a common latent atrial rate**
   (e.g. 150 → 100 → 75 ⇒ atrial ~300). Implement as a latent-atrial-rate
   fit over the series' rate distribution with a goodness score; requires
   ≥3 scans in the series. Nothing else in cardiology produces this pattern
   as cleanly, so it is the track's differentiator.

Also emit **variable-block flutter** features (irregular pulse — overlaps
AF's feature space) so the AF/flutter confusion is measurable, not hidden.

### Task 2 — `head_flutter` (two outputs, one head)

Registered heads plug-in producing: (a) `regular_tachy_flag` — the
per-scan pattern flag (rate + hyper-regularity), and (b)
`series_ratio_signature` — the serial-scan score (null when <3 scans).
`HeadResult` carries per-family evidence, confidence, and the conduction-band
it triggered on (internal only). Abstains (no flag) when clean intervals are
insufficient or respiration signal quality is too low to test RSA coupling.

### Task 3 — Hard-negative battery (this track's core difficulty)

The flag's specificity is the whole product risk, because *regular
tachycardia is usually benign*. Build labeled evaluation cohorts/fixtures for
each confounder and report per-class false-positive rates:

| Confounder | Why it mimics | Required handling |
|---|---|---|
| Sinus tachycardia ~150 (exercise, anxiety, fever, caffeine, dehydration) | identical rate | RSA-coupling retained; motion/context features; **must be the dominant tested negative** |
| SVT / AVNRT ~150–180 | regular, abrupt | cannot be separated from flutter by pulse — flag covers both by design; documented in F-a wording |
| Beta-blocked or elderly low-variability sinus rhythm | hyper-regular but normal rate | rate band gate; age-adjusted dispersion floor |
| AF with regularized rate / AF-flutter conversion | overlapping features | joint evaluation with `head_afib` (Task 6) |
| Pacemaker rhythm | metronomic by design | metadata flag if available; report separately |
| Slow fixed-block flutter ~75 | looks like normal sinus | **documented permanent miss** (F-c) |
| Motion/expression artifact | can fake regularity or rate | existing SQI gates; artifact regressor baseline |

Baselines the head must beat (reuse training-engine machinery): **B1**
rate-threshold-only ("rate >140 ⇒ flag"), **B2** rate + naive dispersion,
**B3** rate + dispersion + respiration coupling *without* the learned model
(the interpretable competitor), **B4** demographic+HR. If the head cannot
beat B3, ship B3 — a transparent rule — and say so.

### Task 4 — Data-engine extension (flutter ground truth)

Extend the v0.2 label schema: rhythm label must record **atrial rate,
conduction ratio (2:1/3:1/4:1/variable), flutter type (typical/atypical)**,
and adjudicator identity — all requiring **12-lead ECG with EP-level
adjudication** (flutter naming needs F-wave morphology; single-lead is
insufficient for the positive label, though it remains adequate for the
rate/regularity comparator). Campaign design: recruit from **elective
cardioversion and EP/ablation clinics** — the only reliable source of
flutter-positive scans (Cramer's design, deliberately) — plus a **prospective
regular-tachycardia negative cohort** (exercise-recovery and anxiety/fever
presentations) so B3-beating specificity is measurable. Indicative sizing
(planning estimates): ≥40–60 ECG-confirmed flutter scans across conduction
ratios (2:1 majority is expected; oversample 3:1/4:1 deliberately), ≥200
regular-tachycardia negatives, plus the serial-scan sub-protocol (≥3 scans
per patient across a rate-varying window, e.g. pre/post rate-control
titration) to power the Task-1 family 3 signature.

### Task 5 — §F promotion gates (`cli.py gate-status --track flutter`)

- **F0** Rate accuracy in arrhythmia: pulse-rate error vs ECG during
  flutter within pre-registered bounds (Cramer measured −7.45 bpm bias and
  only 48.6% of readings within 5 bpm during arrhythmia — we must do
  materially better before a rate-based flag means anything).
- **F1** Flag performance: sensitivity for 2:1 flutter and specificity
  against the full Task-3 negative battery, subject-independent; beats B3.
- **F2** Per-ratio reporting: sensitivity broken out by 2:1 / 3:1 / 4:1 /
  variable, with the slow-fixed-block miss quantified (not merely asserted).
- **F3** Serial signature: latent-atrial-rate score separates flutter series
  from non-flutter series in the serial sub-cohort (AUC threshold).
- **F4** Fairness + coverage parity across Fitzpatrick groups (the Cramer
  study excluded its only Fitzpatrick-VI patient — we do not get to do that).
- **F5** Claim mapping (owner + clinical advisor): sanctioned sentence per
  F-a; combined-endpoint decision per Task 6; limitations doc published.

### Task 6 — Combined atrial-tachyarrhythmia endpoint (the likely product)

Implement `decision_logic` fusion of `head_afib` + `head_flutter` into one
referral output, and evaluate three candidates head-to-head on the same
disjoint splits: (i) AFib head alone, (ii) flutter head alone, (iii)
combined "AF-or-flutter suggested." Report Se/Sp/PPV/NPV at deployment
prevalence, FP per 1,000 scans, and no-read rate for each. **Recommend the
winner in the summary with numbers** — the hypothesis is that (iii) wins and
becomes the flagship referral, with the individual heads remaining internal
evidence.

### Task 7 — Docs + changelog

`docs/flutter_track.md` (physiology inversion, three feature families,
gates, cohort plan, scoreboard) and `docs/flutter_limitations.md` (F-c
known-miss registry, SVT non-separability, sinus-tach confounding). Spec
changelog appended. Synthetic generator: flutter fixtures at each conduction
ratio, including a mid-series ratio switch, so all gates run in CI pre-data.

## Definition of done

Suite green; extractor, head, hard-negative battery, baselines, combined
endpoint, and gate scoreboard runnable end-to-end on synthetic fixtures;
forbidden-token and known-miss tests in place; final summary reports the §F
scoreboard, the Task-6 head-to-head numbers with a recommendation, and any
escalations.

## Escalation rule

If any instruction requires naming flutter (or SVT) to a user, dropping the
sinus-tachycardia negative cohort, routing flutter through the gated
reconstruction head, or removing the slow-fixed-block known-miss disclosure:
**do not implement; record verbatim under "requires owner decision —
conflicts with §F / F-a–F-d."** The honest version of this feature is a
flag that says "your pulse looks fast and unusually regular — get an ECG";
the dishonest version names a rhythm the camera cannot see.
