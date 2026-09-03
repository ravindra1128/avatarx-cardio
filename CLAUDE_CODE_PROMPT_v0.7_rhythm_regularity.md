# Claude Code prompt — AvatarX v0.7: Rhythm-Regularity Track (`head_regularity`)

<!--
HOW TO USE
  1. Place this file in the repo root at AvatarX/code/afib/ (next to CLAUDE_CODE_SPEC.md).
  2. cd into the repo and start Claude Code:  claude
  3. Say: "Read CLAUDE_CODE_PROMPT_v0.7_rhythm_regularity.md and execute it."
  Refactors the substrate beneath head_afib (v0.2 M1) and head_flutter (v0.6).
  Depends on v0.2 data/training engines; shares gate CLI with v0.3–v0.6.
-->

---

You are building **AvatarX v0.7 — the rhythm-regularity track**: the
rhythm-agnostic *"is this pulse regular or irregular, and how confident are
we?"* head. Three things make it different from every other track in this
repo, and the design follows from all three.

**1. It is the substrate, not a sibling.** AFib and flutter are
*specializations* of regularity: AFib = irregular with a specific chaotic
signature; flutter = suspiciously regular at a fast rate. This head must
therefore produce the **one canonical regularity representation those heads
consume** — not a parallel feature path. The repo's P2 lesson (a matched-pair
harness passed while the production path read sinus as AF) is the reason:
two paths that can disagree will eventually disagree in production.

**2. Its ground truth is computable, not adjudicated.** The reference label
for "regular vs irregular" can be derived deterministically from the
**reference ECG's own R-R series** using the identical statistics we apply
to camera intervals. That makes this the one head we can validate at scale
on *any* paired video+ECG data — including healthy volunteers, public
corpora, and every session collected for the other tracks — with no
cardiologist adjudication needed for the base label. Exploit this: it is the
cheapest high-quality validation in the whole program, and it yields a clean
**ceiling test** — *how well do camera intervals reproduce the ECG's own
regularity verdict?*

**3. Its dominant false positive is a healthy person breathing.**
Respiratory sinus arrhythmia — genuine, benign beat-to-beat variation
phase-locked to breathing, prominent in the young and fit — is *real*
irregularity. A detector that flags it is not wrong about the intervals; it
is wrong about the meaning. Separating benign from pathological irregularity
(rather than merely measuring irregularity) is this track's actual problem.

**Regulatory anchor:** this endpoint maps most directly onto the FDA
irregular-rhythm-notification precedent (Apple DEN180042, which created 21
CFR 870.2790 / product code QDB; Fitbit K212372; Samsung K230292) — all of
which are **contact wrist PPG**; no contactless predicate exists. Build the
evidence package that pathway expects: algorithm clinical performance,
signal-quality validation, human-factors-ready labeling, and "not a
diagnosis" throughout.

## Step 0 — Preflight

1. Work in `AvatarX/code/afib`; `pytest -q` green or STOP. Spec is authority;
   append `## B.x v0.7 changelog`.
2. Standing invariants hold. Two are load-bearing here: **clean runs only /
   no interval repair** (a repaired interval erases the very evidence this
   head exists to measure) and **the SQI must never score periodicity**
   (a periodicity-based quality gate would systematically reject exactly the
   irregular rhythms we are trying to flag — re-verify the anti-periodicity
   test guards this).
3. This task **refactors** existing behavior. `head_afib`'s outputs must stay
   bit-identical on the regression corpus (sanctioned sentence + star grade)
   after it is rewired onto the shared representation. That equivalence test
   is the definition of a safe refactor here.
4. Small commits; suite green at every commit.

## Track invariants (each becomes a test)

G-a. **One representation.** `head_afib`, `head_flutter`, and
     `head_regularity` all consume `RegularityFeatures` produced by a single
     module. A second interval-statistics implementation anywhere in
     `inference/` or `heads/` is a bug (import/duplication audit test).
G-b. **Graded, not binary.** The head emits a continuous regularity index
     with a confidence interval plus a thresholded class; any code path that
     produces the class without the index and CI is a bug.
G-c. **Benign-irregularity separation is mandatory before any flag.** An
     irregular classification must carry a `benign_pattern_evidence` field
     (respiration-coupled / ectopy-pattern / chaotic / indeterminate).
     Indeterminate is allowed — silent omission is not.
G-d. **No rhythm naming.** User-facing wording never says AFib, ectopy, or
     any rhythm name; `user_facing_text()` remains the single gate.

## Ordered tasks (tests first, then code)

### Task 1 — `features/regularity.py`: the canonical representation

Extract, from ACCEPT-grade clean runs only, into a versioned
`RegularityFeatures` dataclass:

- **Dispersion family:** RMSSD-class, SDNN-class, normalized dispersion
  (CV), median absolute successive difference — each computed *within* clean
  runs and aggregated run-length-weighted (never across a run break).
- **Distribution family:** Shannon entropy of the interval histogram,
  Poincaré SD1/SD2 and their ratio, sample entropy — the classic
  AF-irregularity descriptors, and the interpretable competitor to any
  learned model.
- **Structure family (the benign/pathological discriminator):**
  respiration-coupling strength (phase/frequency coherence between the
  interval series and the camera-derived respiration signal — RSA is
  phase-locked to breathing, AF is not); short-run periodicity of the
  interval pattern (bigeminy/trigeminy-style alternation ⇒ ectopy pattern);
  outlier-interval topology (isolated long/short pairs ⇒ ectopy with
  compensatory pause, vs pervasive scatter ⇒ chaotic).
- **Confidence inputs:** number and length of clean runs, per-beat
  confidence distribution, and the propagated timing-jitter estimate (Task 2).

Refactor `head_afib` and `head_flutter` to consume this module; delete their
private duplicates; prove behavioral equivalence on the regression corpus.

### Task 2 — Noise-floor characterization (the honesty task)

Camera pulse intervals are ECG R-R intervals *plus* pulse-transit and
pre-ejection jitter *plus* frame-timing quantization *plus* beat-detection
error. Quantify our own floor before claiming to measure anyone's rhythm:

1. **Timing jitter budget:** propagate frame-rate quantization (≈13.6 ms
   RMS on intervals at 30 fps, ≈6.8 at 60 fps — recompute in code, don't
   hard-code), interpolation gain, and SQI-conditioned beat-confidence into
   a per-session jitter estimate.
2. **Minimum detectable irregularity (MDI):** the smallest true dispersion
   (as measured on simultaneous ECG R-R) that the camera path can
   distinguish from a perfectly regular rhythm at a stated confidence.
   Report MDI per fps, per SQI grade, per Fitzpatrick group.
3. **False-irregularity from beat errors:** simulate and measure — one
   missed beat inside a run inflates dispersion dramatically; confirm the
   clean-run/no-repair architecture contains it, and report the residual
   false-irregularity rate as a function of beat-detection error rate.

`cli.py regularity-floor <dataset>` renders this scoreboard. **The MDI is a
published product parameter, not an internal number**: it defines what the
product can honestly claim to notice.

### Task 3 — ECG-derived reference labeler + ceiling test

`datasets/regularity_reference.py`: compute the identical dispersion and
distribution features from the reference ECG's R-peak series, and derive the
operational reference class (regular / irregular) using a **pre-registered,
documented definition** with the thresholds in `configs/gates.yaml`
(`REQUIRES_CLINICAL_SIGNOFF`). Because the label is deterministic and
transparent, publish the definition in the docs — no hidden ground truth.

**The ceiling test (Gate R0):** run both paths — camera intervals and ECG
R-R — through the *same* feature code, and report agreement (class agreement,
index correlation, Bland–Altman on the index) across every registered
dataset. This isolates "can our optics reproduce the rhythm verdict?" from
"is the verdict clinically right?", and it runs on all paired data,
including healthy cohorts and the sessions collected for other tracks.

### Task 4 — `head_regularity` + benign-pattern classification

Registered heads plug-in emitting: regularity index + CI (`MEASURED`-adjacent
class), thresholded class, `benign_pattern_evidence` per G-c, and abstention
when clean runs or respiration quality are insufficient. Model: start with a
transparent rule/threshold model over `RegularityFeatures`; any learned model
must beat it (Task 5). Ectopy-pattern and respiration-coupled evidence are
*explanations attached to an irregular finding*, never standalone rhythm
claims (G-d).

### Task 5 — Baselines and the age/RSA stratification

Mandatory baselines on identical disjoint splits: **B1** RMSSD-threshold
only; **B2** Shannon-entropy-only (the classic AF-detection statistic);
**B3** B1+B2+rate; **B4** demographics-only (age/sex — the RSA-prior
shortcut detector); **B5** SQI-only (does signal quality alone predict the
label? — the artifact-shortcut detector). Promotion requires beating B3 and
materially beating B4/B5. Evaluate and report **age-stratified specificity**
explicitly (young/fit cohorts carry high physiological RSA; if specificity
collapses under 35, the threshold must be age-aware or the claim narrowed).

### Task 6 — §R promotion gates (`cli.py gate-status --track regularity`)

- **R0** Ceiling: camera-vs-ECG regularity agreement ≥ threshold (κ and
  index correlation) across datasets and SQI grades.
- **R1** Noise floor published: MDI characterized per fps/SQI/Fitzpatrick;
  product copy consistent with it.
- **R2** Benign separation: RSA-dominant sessions are not flagged as
  clinically irregular above a pre-registered rate; age-stratified
  specificity within bounds.
- **R3** Beats B3, B4, B5 subject-independent.
- **R4** Fairness: index bias, flag rate, and coverage parity across
  Fitzpatrick groups within bounds.
- **R5** Claim mapping (owner + clinical advisor): permitted wording family —
  *"Your pulse rhythm looked irregular during this scan. This is often
  benign (for example, normal breathing-related variation), but if it
  repeats or you have symptoms, an ECG can tell you why."* Escalation to an
  AFib-suggestive sentence remains `head_afib`'s job; this head never
  escalates on its own.

### Task 7 — Docs + changelog

`docs/regularity_track.md`: the substrate architecture (one representation,
three consumers), the published reference-label definition, the MDI table,
the RSA problem and how it is handled, gate scoreboard, and the QDB-precedent
mapping of the evidence package. Spec changelog appended, explicitly noting
the refactor and the equivalence proof.

## Definition of done

Suite green; `head_afib`/`head_flutter` behaviorally identical post-refactor
(equivalence test); no duplicate interval-statistics implementation remains;
`regularity-floor`, ceiling test, head, baselines, and gate scoreboard all
runnable end-to-end on synthetic + public fixtures; final summary reports the
§R scoreboard, the MDI table, the age-stratified specificity, and any
escalations.

## Escalation rule

If any instruction requires shipping a binary irregular/regular verdict
without the index and CI, dropping the respiration-coupling (RSA) analysis,
letting this head name a rhythm or escalate to an AFib claim, or quoting a
sensitivity below the measured MDI: **do not implement; record verbatim
under "requires owner decision — conflicts with §R / G-a–G-d."** This head's
value is that it knows the difference between an irregular heart and a
noisy camera — every gate here exists to keep that distinction honest.
