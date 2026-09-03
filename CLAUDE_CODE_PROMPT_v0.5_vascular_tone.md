# Claude Code prompt — AvatarX v0.5: Vascular-Tone Research Track (`head_vasotone`)

<!--
HOW TO USE
  1. Place this file in the repo root at AvatarX/code/afib/ (next to CLAUDE_CODE_SPEC.md).
  2. cd into the repo and start Claude Code:  claude
  3. Say: "Read CLAUDE_CODE_PROMPT_v0.5_vascular_tone.md and execute it."
  Depends on v0.2 data/training engines (build the minimal subset if absent).
  Composes with v0.4 (shares the vascular research namespace and gate CLI).
-->

---

You are building **AvatarX v0.5 — the vascular-tone research track**: a
research-flagged endpoint head that attempts to measure **vasomotor state
and reactivity** — how constricted or dilated the small vessels are, and how
they *respond* — from the facial pulse signal.

Scientific context, stated honestly so you build the right thing:

- **What vascular tone is.** The functional, minute-to-minute
  constriction/dilation of vessels (sympathetic tone, thermoregulation,
  drugs, stress) — distinct from v0.4's arterial *stiffness*, which is a
  structural wall property. Tone is dynamic; the scientifically correct
  validation is therefore **within-subject response to a controlled
  provocation**, not cross-sectional "your tone is X."
- **What the signal plausibly carries.** In contact PPG this is established
  territory: pulse amplitude (AC/DC — the oximeter "perfusion index")
  tracks local vasoconstriction; low-frequency vasomotor oscillations
  (~0.04–0.15 Hz) reflect vasomotion; notch/reflection features shift with
  peripheral resistance. On facial video the evidence tier is **case
  series**: rPPG waveform changes have been observed during vasopressor
  therapy and warm/cold fluid challenges in small clinical reports. Nothing
  validated as a tone biomarker exists — camera or otherwise consumer.
- **Threat T1 — amplitude is the most optics-confounded quantity we have.**
  rPPG AC/DC is modulated by illumination, distance, exposure/AWB, skin
  tone, makeup, and BCG-shading artifacts at least as strongly as by
  physiology. **Absolute tone levels across sessions are presumed
  non-identifiable**; the defensible objects are (a) within-session relative
  change under provocation and (b) heavily normalized indices — and even
  those must pass an optics-placebo test (below).
- **Threat T2 — the HR/respiration shortcut.** Every tone provocation (cold
  pressor, mental arithmetic, paced breathing, posture change) also changes
  heart rate and breathing. A "tone" metric that is secretly an HR-response
  or respiration-response detector has measured nothing new; baselines must
  catch it.
- **Claim ceiling.** "Vascular tone" is not itself a standard clinical
  endpoint; adjacent validated constructs (endothelial function via
  FMD/EndoPAT) require their own reference devices and prospective work.
  The realistic first surface, gated: a **wellness-tier vasomotor-reactivity
  trend**. Nothing medical.

## Step 0 — Preflight

1. Work in `AvatarX/code/afib`; suite green or STOP. Spec is authority;
   append `## B.x v0.5 changelog`.
2. Standing invariants hold (fail-closed gates; ACCEPT-grade beats only;
   participant+session-disjoint splits; measurement-class labels; no
   demographic features in models; `rbcg/` ablation-only).
3. Reuse v0.4's `research/vascular/` namespace and `gate-status` CLI
   pattern; do not fork gate machinery.

## Track invariants (each becomes a test)

W-a. `head_vasotone` is research-flagged: **no user-facing output** until
     every §W gate is green with owner + clinical sign-off in the changelog.
W-b. Forbidden on user surfaces while red (token tests): `vascular tone`,
     `vasoconstriction`, `perfusion index`, `endothelial`, any absolute
     tone score.
W-c. **No absolute cross-session tone values anywhere** — all metrics are
     within-session deltas or dimensionless normalized indices; a metric
     emitted as an absolute level is a bug.
W-d. Amplitude-derived features are computed only with AE/AWB lock verified
     in the manifest (or flagged `uncontrolled_optics` and excluded from
     gate evaluations).

## Ordered tasks (tests first, then code)

### Task 1 — Tone-feature extractor

`research/vascular/tone_features.py`, consuming the gated production path's
per-beat waveform + BeatLattice: normalized pulse amplitude (AC/DC per ROI,
median across SNR-weighted ROIs); amplitude trend within session; vasomotor
LF band power (0.04–0.15 Hz) of the amplitude envelope; notch relative
amplitude and reflection-index shift (reusing v0.4 feature code); optional
ablation: rBCG→rPPG delay shift. Every feature emitted as (baseline-window
value, response-window value, delta, normalized delta) — never absolute-only.

### Task 2 — Provocation session protocol + data engine extension

Extend session manifests with a **provocation schema**: maneuver type
(`cold_pressor` — subject's own hand in cool water per IRB protocol,
`paced_breathing`, `mental_arithmetic`, `posture_change`, `null_optics`,
`null_rest`), timed phase marks (baseline / stimulus / recovery) with
on-screen prompts from the capture app, and simultaneous **contact
reference**: pulse-oximeter perfusion-index trace (and finger-PPG waveform)
on a hand not involved in the maneuver. `cli.py ingest-reference` gains the
PI-trace type. Campaign spec: feasibility N 30–60, each subject completing
baseline + ≥2 provocations + the two null arms (below), test–retest visit
for a subset; Fitzpatrick and device quotas as standing policy.

### Task 3 — The two null arms (the T1 defense; this track's signature test)

- **`null_optics`:** physiology held at rest while the rig deliberately
  varies illumination level/spectrum, distance, and exposure within
  realistic bounds. **Gate W1 requires tone-feature "responses" in this arm
  to be below a pre-registered noise threshold.** A feature that responds to
  lamps is an optics detector and is dropped (automatically, by config).
- **`null_rest`:** quiet rest with stable optics — defines the
  within-session natural-drift distribution used to set response thresholds.

### Task 4 — `head_vasotone` + baselines

Heads-registry plug-in producing a **reactivity result**: per-provocation
response magnitude/direction vs the contact-PI reference, with confidence.
Baseline battery (reusing training-engine machinery): **B1** HR-response-only
(does ΔHR explain the "tone" response?); **B2** respiration-response-only;
**B3** HR+respiration; **B4** motion/expression-artifact regressor.
Promotion requires the facial tone response to add information beyond B3
(pre-registered margin, `REQUIRES_CLINICAL_SIGNOFF`) and to track the
contact-PI response (direction agreement and magnitude correlation
thresholds).

### Task 5 — §W promotion gates (`cli.py gate-status --track vasotone`)

- **W0** Reference tracking: facial response agrees with contact-PI response
  in direction ≥ threshold % of provocations and correlates in magnitude
  (per-subject, within-session).
- **W1** Optics invariance: null_optics responses below threshold for every
  surviving feature.
- **W2** Beats B1–B3 baselines (T2 defense) on unseen participants.
- **W3** Dose/consistency: graded or repeated provocations produce ordered/
  repeatable responses (test–retest ICC ≥ threshold).
- **W4** Fairness: response-detection parity and coverage across Fitzpatrick
  groups within bounds.
- **W5** Claim mapping (owner + clinical advisor): first permissible surface
  is a wellness **"vasomotor reactivity trend vs your own baseline"** with
  explicit non-medical labeling; any endothelial-function or clinical claim
  requires a separate EndoPAT/FMD-referenced prospective protocol (out of
  scope here).

### Task 6 — Synthetic testability + docs

Synthetic generator coupling a latent tone variable to amplitude/notch
shifts plus independent optics perturbations, so the full pipeline and both
null arms run in CI before clinical data exists. `docs/vasotone_track.md`
with the context, threats, gates, and scoreboard; spec changelog appended.

## Definition of done

Suite green throughout; extractor, provocation ingestion, both null arms,
head, baselines, and gate scoreboard runnable end-to-end on synthetic
fixtures; forbidden-token and no-user-output tests in place; final summary
reports the §W scoreboard (expected all red pre-data) and escalations.

## Escalation rule

If any instruction requires surfacing a tone/perfusion output while §W is
red, emitting absolute cross-session tone levels, or dropping the
null_optics arm or HR/respiration baselines: **do not implement; record
verbatim under "requires owner decision — conflicts with §W / T1–T2
defenses."** A camera tone metric that hasn't beaten the lamp test and the
heart-rate shortcut is a mood ring with telemetry; these gates are what
make it a biomarker.
