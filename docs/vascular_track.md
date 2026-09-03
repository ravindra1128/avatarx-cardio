# The arterial-stiffness research track (v0.4 vascular)

**Question, pre-registered (gates V0/V1):** do stiffness-relevant
pulse-morphology features survive the facial-video path with clinical
fidelity against a simultaneous contact-PPG reference — and does a model
on the surviving features carry information about measured
carotid-femoral pulse wave velocity (cfPWV) **beyond an
age+sex+brachial-BP regression**? Until every gate of the `vascular:`
block of `configs/gates.yaml` is green with owner + clinical signoff
recorded in the spec changelog, `head_vascular` produces **no
user-facing output of any kind** — no number, score, trend, or color
(invariant V-a) — and the strings "vascular age", "artery age", "PWV"
and any m/s value are forbidden on user-facing surfaces (V-b, token
tests).

## Scientific context (owner-directed, recorded verbatim in intent)

- **Why this biomarker matters.** Central arterial stiffness (gold
  standard: carotid-femoral PWV, cfPWV, by tonometry —
  SphygmoCor/Complior class devices) is an established independent
  cardiovascular-risk marker and a guideline-recognized sign of vascular
  target-organ damage. A validated contactless stiffness signal would be
  a real asset for the platform.
- **What the face plausibly carries.** Stiffer arteries reshape the
  pulse: faster wave reflections, augmented systolic portion, attenuated
  dicrotic notch. In *contact* PPG this is old, real science
  (pulse-decomposition and second-derivative "aging index" features
  correlate with age and stiffness). In *facial video* the evidence is
  thin: small feasibility studies extract pulse-wave features remotely;
  **nothing is validated against cfPWV at clinical scale, and no cleared
  product exists.**
- **Threat T1 — morphology may not survive the camera.** ROI averaging,
  optics, band-pass filtering, compression, skin tone, and lighting
  reshape the facial waveform; recent work is formalizing
  information-theoretic limits on camera pulse-*morphology* recovery.
  Whether stiffness-relevant shape features survive our pipeline is an
  open empirical question — so this track starts with a signal-fidelity
  experiment, not a model.
- **Threat T2 — the age shortcut.** Stiffness correlates strongly with
  age (and BP). A model can score well on "stiffness" by covertly
  predicting age from anything age-correlated in the video and metadata.
  A stiffness head that cannot beat an age+sex+blood-pressure regression
  **has measured nothing and must not ship.** Defeating T1 and T2 *is*
  the assignment.

## Why gates rather than assumptions

The repo's own miniature T1 experiment already justifies the posture:
per-beat fiducials at camera bandwidth proved noise-limited on clean
synthetic clips (single-beat reflection-index IQR up to 0.36, contrast
inverted), while per-ROI ensemble beats recover the injected morphology
— so even the *session statistic* had to be chosen empirically, and the
question of what survives on real faces belongs to a measured fidelity
study, not to optimism. "Vascular age from a camera" is a marketing
swamp precisely because that study is usually skipped.

## The machinery

- `research/vascular/features.py` — 11 candidate morphology features
  per beat and per session (systolic rise time, normalized upstroke
  slope, pulse width at 50%, dicrotic-notch presence/relative
  amplitude/timing, reflection-index-style ratio, second-derivative
  b/a, c/a, d/a and e/a where derivable at the native sampling rate —
  candidates all; the V0 study adjudicates them). Facial waveforms are
  rebuilt
  from the production ingest's retained per-ROI traces on a wide 0.5–10
  Hz morphology band; beats, gates, and segments are the production ones
  (V-c: no features off ACCEPT-grade scans, confidence floor 0.5). The
  IDENTICAL extractor runs on the synchronized contact-PPG reference.
- `research/vascular/fidelity.py` + `cli.py vascular-fidelity <ds>
  --domain facial_rppg` (real campaign days; the default `synthetic`
  domain is machinery-only) —
  the V0 study: per-feature facial↔contact ICC(2,1) + Bland–Altman,
  same-visit test–retest ICC (≥3 pairs or refused), Fitzpatrick/device
  strata. Features failing the explicit gates.yaml thresholds are
  dropped from ALL downstream modeling automatically
  (`evaluation.vascular_gates.surviving_features` — by config, never by
  hand-editing; no study on record means nothing survives).
- `evaluation/vascular_metrics.py` + `cli.py evaluate-vascular <ds>` —
  the T2 defense: mandatory baselines on identical participant-disjoint
  splits — **B1** age, **B2** age+sex, **B3** age+sex+brachial-BP (the
  clinic-available-variables bar), **B4** HR-only, **B5**
  age+sex+HR — against a morphology-only head (demographics may never
  enter the model; invariant V-d makes a report without the full
  battery and the B3 delta structurally unbuildable). Deltas carry
  multiplicity-correct participant-cluster bootstrap CIs; added
  variance explained is a held-out partial-correlation statistic.
- `heads/head_vascular.py` — registry plug-in, RESEARCH_VASCULAR class,
  inert and number-free on every surface the production pipeline can
  reach; real estimates exist only on research surfaces
  (`cli.py process --heads vascular` writes a watermarked research
  report; stdout stays the consumer-clean ScanResult).
- `scripts/make_synth_vascular.py` — a latent stiffness variable drives
  reflection/notch/rise morphology (with synchronized contact-PPG and
  cfPWV references) so every stage is testable before clinical data; a
  deliberate "age-shortcut world" fixture proves the harness reports NO
  incremental value when morphology carries only age.

## The gates (authority: the `vascular:` block of configs/gates.yaml)

| gate | question | headline thresholds (provisional, REQUIRES_CLINICAL_SIGNOFF) |
|---|---|---|
| V0 | signal fidelity | per-feature ICC ≥ 0.75 vs contact; retest ICC ≥ 0.70; ≥ 60 paired participants; surviving set non-empty |
| V1 | incremental validity (PIVOTAL) | beats B3 by ≥ 0.5 m/s RMSE with CI excluding zero AND added R² ≥ 0.05 on ≥ 40 unseen participants |
| V2 | estimate repeatability | same-day retest ICC ≥ 0.85, drift ≤ 0.5 m/s, ≥ 20 pairs |
| V3 | generalization | leave-one-site-out AND leave-one-device-out RMSE ≤ 1.25× pooled; ≥ 2 sites, ≥ 2 devices |
| V4 | fairness | coverage ratio ≥ 0.8 and RMSE ratio ≤ 1.25 on EVERY declared axis (Fitzpatrick group, capture device) |
| V5 | claim mapping | owner + clinical advisor sign the FIRST permissible surface: a longitudinal wellness trend ("your pulse-wave pattern vs your own baseline"), explicitly non-diagnostic; any absolute-stiffness or screening claim requires a further prospective protocol |

Tightening is always allowed; loosening any gate requires an
owner-signed spec-changelog entry. Surrogate-domain runs (synthetic,
public PPG) exercise the machinery and can never open a gate.
`models/registry.promote` refuses `head_vascular` while any gate is red
or the signoff (including `claim_scope`) is unsigned. Scoreboard:
`python3 cli.py gate-status --track vascular`.

## Current scoreboard

All six gates RED; promotion BLOCKED. The only recorded runs are the
synthetic fixture dataset through `vascular-fidelity` and
`evaluate-vascular` — surrogate domain, machinery evidence only, and red
on their own criteria too (fidelity cohort far below 60; retest ICC
refused below 3 pairs so no feature survives; battery blocked with an
empty surviving set; no claim mapping). This is the intended state until
paired clinical data exists.

## Cohort plan (planning estimates — refine after V0)

Recruit through hypertension/vascular clinics to get true
high-stiffness cases, not just healthy volunteers. Wide age band
(target 20–80), deliberate BP-range and Fitzpatrick quotas, and a
test–retest sub-protocol (two scans + two reference reads per visit for
a repeatability subset). Indicative sizing: fidelity/feasibility 60–100
paired participants; development 200–400. Session-level reference block
per visit: cfPWV (m/s) with device model + operator (SphygmoCor/
Complior-class export parsing in `datasets/reference.py` — extend the
alias tables against real export fixtures), optional CAVI, same-visit
brachial BP, HR at measurement, antihypertensive/vasoactive medication
flags, ambient/skin temperature note (`<id>.pwv.json`), plus the
synchronized contact-PPG sidecar (`<id>.ppg.json`).

## Escalation clause

If any instruction requires showing a vascular/stiffness/"vascular age"
output to users while the gates are red, removing age/BP baselines from
an evaluation, or moving age/sex/BP into the model: do not implement;
record the request verbatim in the spec changelog under "requires owner
decision — conflicts with §V / T2 defense". A stiffness product that is
secretly an age predictor would eventually be exposed as one; the
baselines exist so AvatarX finds out first.
