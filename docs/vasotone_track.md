# The vasomotor-reactivity research track (v0.5, §W)

**Question, pre-registered (gates W0–W2):** does the facial pulse
signal carry a within-session vasomotor RESPONSE to a controlled
provocation that (a) tracks a simultaneous contact perfusion-index
reference, (b) survives a deliberate optics placebo, and (c) adds
information beyond the heart-rate and respiration responses every
provocation also causes? Until every §W gate in the `vasotone:` block
of `configs/gates.yaml` is green with owner + clinical signoff,
`head_vasotone` produces **no user-facing output of any kind** (W-a),
the strings "vascular tone", "vasoconstriction"/"vasodilation"
(all vasoconstrict*/vasodilat* forms), "perfusion index" and
"endothelial" are forbidden on user surfaces (W-b, token tests), and
**no absolute cross-session tone value exists anywhere** — every metric
is a within-session delta or dimensionless normalized index (W-c,
structural).

## Scientific context (owner-directed, recorded verbatim in intent)

- **What vascular tone is.** The functional, minute-to-minute
  constriction/dilation of vessels (sympathetic tone, thermoregulation,
  drugs, stress) — distinct from v0.4's arterial *stiffness*, a
  structural wall property. Tone is dynamic; the scientifically correct
  validation is therefore **within-subject response to a controlled
  provocation**, not cross-sectional "your tone is X."
- **What the signal plausibly carries.** In contact PPG this is
  established territory: pulse amplitude (AC/DC — the oximeter
  "perfusion index") tracks local vasoconstriction; low-frequency
  vasomotor oscillations (~0.04–0.15 Hz) reflect vasomotion;
  notch/reflection features shift with peripheral resistance. On facial
  video the evidence tier is **case series**: rPPG waveform changes
  have been observed during vasopressor therapy and warm/cold fluid
  challenges in small clinical reports. Nothing validated as a tone
  biomarker exists — camera or otherwise consumer.
- **Threat T1 — amplitude is the most optics-confounded quantity we
  have.** rPPG AC/DC is modulated by illumination, distance,
  exposure/AWB, skin tone, makeup, and BCG-shading artifacts at least
  as strongly as by physiology. **Absolute tone levels across sessions
  are presumed non-identifiable**; the defensible objects are (a)
  within-session relative change under provocation and (b) heavily
  normalized indices — and even those must pass an optics-placebo test.
- **Threat T2 — the HR/respiration shortcut.** Every tone provocation
  (cold pressor, mental arithmetic, paced breathing, posture change)
  also changes heart rate and breathing. A "tone" metric that is
  secretly an HR-response or respiration-response detector has measured
  nothing new; baselines must catch it.
- **Claim ceiling.** "Vascular tone" is not itself a standard clinical
  endpoint; adjacent validated constructs (endothelial function via
  FMD/EndoPAT) require their own reference devices and prospective
  work. The realistic first surface, gated: a **wellness-tier
  vasomotor-reactivity trend**. Nothing medical.

## The two null arms (this track's signature test)

- **`null_optics`** — physiology held at rest while the rig
  deliberately varies illumination level/spectrum, distance and
  exposure within realistic bounds, with the perturbation recorded in
  the provocation sidecar's `optics_log`. Gate W1 requires every
  surviving feature's apparent "response" in this arm below the
  pre-registered noise threshold — absolutely (p95 |Δnorm| ≤ 0.10) and
  relative to natural drift (≤ 2× the null_rest p95). **A feature that
  responds to lamps is an optics detector and is dropped —
  automatically, by config** (`evaluation.vasotone_gates.
  survivors_from_study`), never by hand.
- **`null_rest`** — quiet rest with stable optics: defines each
  feature's natural within-session drift distribution, which sets the
  response-detection floor everywhere else.

## The machinery

- `research/vascular/tone_features.py` — every feature emitted as
  (baseline, response, delta, delta_norm), never absolute (W-c).
  AC/DC comes from the GREEN channel (the POS/CHROM projections
  normalize per window and destroy the amplitude scale); SNR-screened
  ROI median; Theil–Sen amplitude trend; vasomotor LF power over the
  frequencies a window actually resolves; notch/reflection shifts via
  the v0.4 ensemble morphology per phase window. W-d: without AE/AWB
  lock in the manifest, amplitude features are withheld and the session
  is flagged `uncontrolled_optics` and excluded from gate evaluations.
- `datasets/schema.py` — `ProvocationRecord` (pre-registered maneuver
  vocabulary incl. both null arms, ordered non-overlapping phase marks
  written by the capture rig's timed prompts, graded intensity 1–3,
  operator-recorded Fitzpatrick for strata only) and `PiTrace` (the
  contact perfusion-index sidecar, recorded on a hand NOT involved in
  the maneuver). `cli.py ingest-reference` ingests `pi_export.csv`
  mapped onto the video clock via the sanctioned sync model.
- `evaluation/vasotone_metrics.py` + `cli.py evaluate-vasotone <ds>
  --domain facial_rppg` — the harness: null-arm study → surviving set →
  W0 reference tracking (direction agreement with conservative
  detection floors + magnitude correlation vs the contact PI), the W2
  baseline battery (**B1** HR-response, **B2** respiration-response,
  **B3** HR+respiration, **B4** motion/artifact regressor — the facial
  response must add information beyond B3, measured as a held-out
  partial-correlation statistic with cluster-bootstrap CIs), W3
  retest/graded ordering, W4 Fitzpatrick parity.
- `heads/head_vasotone.py` — registry plug-in, research-flagged, inert
  and number-free on every pipeline-reachable surface; real readings
  only on research surfaces (`cli.py process --manifest rec.json --heads vasotone
  --provocation p.json [--pi pi.json]` → watermarked research report;
  the manifest carries the AE/AWB lock evidence W-d needs;
  stdout stays the consumer-clean ScanResult).
- `scripts/make_synth_vasotone.py` — a latent tone response drives the
  optical pulse amplitude (with 0.08 Hz vasomotion) while the PI
  sidecar tracks the same latent; null_optics uses a nonlinear
  per-frame gamma sweep (a pure brightness scale would cancel in AC/DC;
  real exposure/AWB shifts do not), so the whole pipeline and both null
  arms run in CI before clinical data exists.

## The gates (authority: the `vasotone:` block of configs/gates.yaml)

| gate | question | headline thresholds (provisional, REQUIRES_CLINICAL_SIGNOFF) |
|---|---|---|
| W0 | reference tracking | direction agreement ≥ 0.80 across ≥ 60 provocations; response-magnitude r ≥ 0.60 vs the contact-PI reference |
| W1 | optics invariance | null_optics p95 \|Δnorm\| ≤ 0.10 and ≤ 2× natural drift for every surviving feature; ≥ 20 sessions per null arm; surviving set non-empty |
| W2 | beats baselines (PIVOTAL) | facial response adds R² ≥ 0.10 beyond HR+respiration (B3) with CI excluding zero, ≥ 25 unseen participants |
| W3 | dose/consistency | retest ICC ≥ 0.60 over ≥ 10 pairs; graded provocations ordered ≥ 70% |
| W4 | fairness | response-detection parity ≥ 0.75 and coverage ratio ≥ 0.8 across Fitzpatrick groups |
| W5 | claim mapping | owner + clinical advisor sign the FIRST permissible surface: a wellness "vasomotor reactivity trend vs your own baseline", explicitly non-medical; any endothelial-function or clinical claim requires a separate EndoPAT/FMD-referenced prospective protocol (out of scope) |

Tightening is always allowed; loosening any gate requires an
owner-signed spec-changelog entry. Surrogate-domain runs exercise the
machinery and can never open a gate. `models/registry.promote` refuses
`head_vasotone` while any gate is red or the signoff (including
`claim_scope`) is unsigned. Scoreboard:
`python3 cli.py gate-status --track vasotone`.

## Current scoreboard

All six §W gates RED; promotion BLOCKED. The only recorded run is the
synthetic fixture dataset — surrogate domain, machinery evidence only,
and red on its own criteria too (null arms far below the 20-session
floors, so nothing survives and the primary response never exists;
no claim mapping). This is the intended state until provocation data
exists.

## Campaign plan (planning estimates — refine after feasibility)

Feasibility N 30–60; each subject completes baseline + ≥ 2 provocations
(cold pressor per IRB protocol — subject's own hand in cool water —
paced breathing, mental arithmetic, posture change) + BOTH null arms;
test–retest visit for a subset (≥ 10 pairs toward W3); Fitzpatrick and
device quotas per standing policy; contact reference (pulse-oximeter
PI + finger PPG) on a hand not involved in the maneuver. Template:
`configs/campaign_vasotone_template.yaml`. On-screen phase prompts come
from the collection rig; the consumer demo app is NOT the collection
instrument for provocation studies.

## Escalation clause

If any instruction requires showing a tone/reactivity output to users
while §W is red, emitting an absolute cross-session tone value,
removing the null arms or the HR/respiration baselines from an
evaluation: do not implement; record the request verbatim in the spec
changelog under "requires owner decision — conflicts with §W". A
reactivity product that is secretly a lamp detector or an HR-response
detector would eventually be exposed as one; the null arms and
baselines exist so AvatarX finds out first.
