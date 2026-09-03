# The rhythm-regularity substrate track (v0.7, §R)

**Question, pre-registered (gates R0–R5):** is this pulse regular or
irregular, and how confident are we — stated as a continuous index
with a confidence interval, a class under a published definition, and
on every irregular read an explanation of the benign pattern that could
have produced it.

Not "is this AFib". The AFib head answers that, on top of this one.

Until every §R gate in the `regularity:` block of `configs/gates.yaml`
is green with owner + clinical signoff, `head_regularity` produces **no
user-facing output** (`user_facing_text()` is fail-closed; the head
emits a sentence *key*, never a sentence), the head carries the
unrenderable `RESEARCH_RHYTHM` class, and the one sanctioned sentence
family (R5) names no rhythm and offers the benign explanation first.

## Why a substrate track, not a feature

1. **It is the representation every rhythm head already computes
   privately.** Before v0.7 the AFib path, the flutter path and the
   irregularity research head each derived interval statistics on their
   own terms. Two implementations of one statistic drift, and the drift
   is invisible until they disagree (the P2 lesson). v0.7 makes
   `features/regularity.py` the ONE place successive differences,
   dispersion, distribution and structure are computed, and turns every
   other path into a reader of it (G-a, enforced by an AST audit).
2. **Its ground truth is free.** The reference ECG's own R-R series is
   an interval series; run it through the *same code* and you have the
   label the camera is trying to reproduce. That makes the ceiling test
   (R0) the cheapest validation in the programme and a bound on
   everything downstream: no rhythm head can be more right about
   irregularity than this substrate is.
3. **Its dominant false positive is healthy people breathing.**
   Respiratory sinus arrhythmia (RSA) in a young, fit, relaxed person
   produces successive-difference statistics that overlap frankly
   irregular rhythms. Separating "irregular" from "irregular because
   breathing" is a separation problem (R2), not a detection problem —
   and it is the thing a contactless notification must get right before
   it can be worth anything.

## The precedent, and the evidence it demands

| Precedent | Route | Sensor | What it established |
|---|---|---|---|
| Apple Irregular Rhythm Notification, DEN180042 (2018) | De Novo → 21 CFR 870.2790, product code **QDB** ("photoplethysmograph analysis software for over-the-counter use") | wrist contact PPG | an OTC notification of pulse *irregularity* suggestive of AF, validated against ambulatory ECG, that names no diagnosis and tells the user to see a clinician |
| Fitbit Irregular Rhythm Notifications, K212372 (2022) | 510(k) to QDB | wrist contact PPG | the same claim on a second sensor and algorithm |
| Samsung Irregular Heart Rhythm Notification, K230292 (2023) | 510(k) to QDB | wrist contact PPG | a third |

All three are **contact** PPG. There is **no contactless predicate**:
nothing on the QDB list reads the pulse from a camera. That is the
size of the gap this track has to close, and it is why the gates below
are heavier than a 510(k) comparison would be.

What the precedent asked for, mapped onto §R:

| QDB evidence element | Where it lives here |
|---|---|
| Tachogram classification performance against ECG-adjudicated truth, per subject, with CIs | **R0** ceiling test: κ, index correlation, Bland–Altman, per SQI grade, ≥100 paired scans |
| Algorithm behaviour under degraded signal (motion, low perfusion) and the conditions under which it declines to classify | **R1** noise floor: the jitter budget, the measured interpolation gain, the MDI table per fps / SQI grade / skin tone, false irregularity vs beat-error rate; abstention is a first-class output |
| Specificity in the intended-use population, including benign irregularity | **R2** benign separation: RSA flag rate and age-stratified specificity with a hard floor per band |
| Performance beyond trivial rules | **R3** pre-registered baselines B1–B5, participant-disjoint, with margins |
| Performance across skin tones | **R4** fairness: index bias, flag-rate parity, coverage parity, darkest band present |
| Labelling: no diagnosis, benign explanation, clinician referral | **R5** claim mapping: the sanctioned sentence family, owner + clinical advisor, this head never escalates |

Where the precedent is silent — ambient lighting, camera frame rate,
face motion, distance, and skin tone as an *optical* rather than a
perfusion variable — the noise-floor gate (R1) is the answer: the
minimum detectable irregularity is a **published product parameter**,
not an internal number.

## The substrate (`features/regularity.py`)

`regularity_from_runs(runs, run_confidences, dropout_rate, mean_sqi, *,
run_times, fps, respiration)` → `RegularityFeatures`
(`version = regularity-features-v1`), computed from **clean runs only**
(ACCEPT-grade beats, the physiologic-range and missed/false-beat
splitters from `configs/default.yaml`), with **no interval repair**:
nothing interpolated, nothing bridged across a run boundary, no
successive difference ever crossing a seam.

| Family | Contents |
|---|---|
| `dispersion` | RMSSD, SDNN, SDSD, CV, median \|Δ\|, **`irregularity_index`**, pNN20/50/80, relative MAD, number of within-run differences |
| `distribution` | Shannon / sample / spectral entropy, Poincaré SD1/SD2, Markov surprise, turning-point ratio, lag-1 autocorrelation |
| `structure` | respiratory `coupling` (with a `status` ∈ ok / no_channel / poor_quality / no_tachogram / rate_uncorroborated / no_spectrum), short-run `periodicity` (lag-2/3 alternation index), `outliers` (MAD-relative outliers, paired detection, scatter fraction → topology none / paired / scattered / pervasive) |
| `confidence` | runs, run lengths, longest run, intervals, beat-confidence summary, the timing-jitter budget for this fps |
| `index` | `irregularity_index` = median \|successive difference\| / median interval, **within-run differences only**, with a moving-block percentile-bootstrap CI (B = 400, fixed seed; nominal 95 %, **measured coverage 0.88 at n = 15, 0.90 at n = 30, 0.94 at n = 70** — the percentile interval of a median under-covers at small n, so the payload carries the method and the measured coverage, pinned by test) |

**Consumers, all readers of the one object:**

- `head_afib` reads `as_rhythm_features()` — a *view* sharing the same
  values dict the pre-v0.7 `RhythmFeatures` carried, so its inputs are
  unchanged key for key. Its outputs are pinned bit-identical on a
  seven-clip golden corpus (`tests/test_regularity_equivalence.py`:
  sinus 30/60 fps, AF 30/60 fps, dark, RSA with breathing, flutter with
  breathing — sanctioned sentence, star grade, class, no-read reasons,
  head results). **The equivalence test is the definition of a safe
  refactor.**
- `features/flutter.py` reads `dispersion` and `structure.coupling`.
- `head_irregularity` reads `dispersion.rel_mad` / `pnn80` (now
  within-run; its pre-v0.7 version pooled intervals across seams and is
  the one consumer whose numbers moved — see spec B.23 #5).
- `head_rate_flags` reads the pooled clean intervals for its rate
  statistic.
- `head_regularity` reads the index, the CI, and the structure family.

The v1 raw-series estimator (`compute_rhythm_features`) is deleted; it
had no callers and was the second implementation G-a forbids.

## The published reference label (`datasets/regularity_reference.py`)

Read from `configs/gates.yaml` → `regularity.reference_label` and
nowhere else (`REQUIRES_CLINICAL_SIGNOFF`; no hidden ground truth):

```yaml
index: irregularity_index        # median |successive diff| / median interval, within-run only
irregular_if_index_at_least: 0.06
min_intervals: 15                # fewer -> indeterminate
ecg_beat_confidence: 1.0         # the reference beats are trusted; the run
                                 # discipline still applies, identically
```

The ECG R-peak series goes through the **same** `clean_runs` and the
**same** `regularity_from_runs` as the camera beats. The camera's label
IS the reference's label applied to a different interval series — one
definition object, tested to be the same on both sides.

The threshold is a *planning* value that sits deliberately inside the
RSA range (sinus at rest ≈ 0.02–0.04, deep RSA ≈ 0.06–0.10, AF ≈ 0.15+),
so that R2 has to earn the separation rather than the threshold hiding
it. Clinical signoff may move it. Under it, isolated premature beats
read *regular* (a PAC every ~12 beats moves the median of \|Δ\| very
little) — a property of the definition, recorded in the fixture table.

Why a median-based index and not RMSSD: a median ignores a minority of
wild differences, so the index is robust to **sparse** beat-detection
errors by construction; RMSSD is wrecked by one missed beat. The class
is decided on the index; RMSSD is reported beside it and never decides.
The beat-error study below measures both.

## The head (`heads/head_regularity.py`)

| Field | Meaning |
|---|---|
| `index.value`, `index.ci95`, `index.n_diffs`, `index.definition` | the graded quantity (G-b) |
| `jitter` | the timing budget at this scan's fps, with the calibrated gain |
| `class` | `regular` / `irregular` / `indeterminate` under the published definition; `threshold` beside it |
| `benign_pattern_evidence.evidence` | on **every** irregular read: `respiration_coupled` / `ectopy_pattern` / `chaotic` / `indeterminate` (G-c; `not_applicable` on a regular read) |
| `sentence_key` | `irregular` or `None` — a key, never a sentence (G-d) |
| `user_facing` | always `None` in the payload; `user_facing_text(value, render_allowed=...)` is the single gate |
| `abstained`, `reasons` | why the head declined, in words |

**Abstains** when the scan is not ACCEPT, when fewer than
`min_intervals` clean intervals exist, or when the respiration channel
is absent, poor, or yields no tachogram/spectrum — because without it
an RSA-dominant sinus rhythm and a chaotic one are the same
measurement. A "rate uncorroborated" coupling status is *not* an
abstention here: it is judged as evidence (the v0.6 refusal rule was
tuned for the flutter question and, applied here, silently abstained on
trigeminy, whose 3-beat periodicity sits inside the respiratory band).
The index and CI are reported even when the head abstains.

**Benign-evidence rule** (first match wins):

1. `respiration_coupled` needs **two signatures and a regular
   residual**: the breath must claim ≥ 50 % of the tachogram's power
   on evidence that the tachogram *moves with* the breath (phase-locking
   value ≥ 0.85 against the respiration waveform — rate-error-robust,
   and the waveform exposes a mis-estimated breathing rate), and what is
   left after that power is removed — estimated as
   0.95 · √(1 − fraction) · (tachogram σ / median) — must itself be
   regular by the published threshold. A chaotic series with some
   respiratory modulation is not "explained by breathing".
2. `ectopy_pattern` if the lag-2/3 alternation index is ≥ 0.5 or the
   outlier topology is paired.
3. `chaotic` if the scatter is pervasive and not periodic.
4. `indeterminate` otherwise. Indeterminate is allowed; omission is a
   bug (tested).

Adversarial cases pinned by test: trigeminy whose 3-beat period sits
inside the respiratory band stays ectopy; RSA with a 1.5–3.5 br/min
error in the reported breathing rate stays respiration-coupled; AF with
20 % respiratory modulation stays chaotic. **Known limitation, pinned:**
an ectopic pattern whose period equals the breathing period within the
window's resolution *and* holds a stable phase is indistinguishable
from RSA by all three signatures.

The tachogram every spectral quantity is computed from is built from
the **longest clean run only** — a run boundary is a stretch the
pipeline refused, and interpolating across it would invent interval
values inside the seam. A scan whose longest run is under 15 intervals
or 20 s has no tachogram, no coupling, and the head abstains.

**Today's consumer scan path carries no respiration channel** (the
torso second decode exists in the harness only, as in v0.6), so on a
production scan the head abstains by design, with the index in the
payload for the research reader.

## The noise floor (R1) — a published product parameter

**Budget, computed in code** (`timing_jitter_budget`): a beat located
to the nearest frame carries uniform timing error with
σ_beat = (1000 / fps) / √12 ms; an interval is the difference of two
beats, σ_interval = √2 σ_beat; and the RMSSD of a metronomic series
under that error alone is √6 σ_beat (successive differences share a
beat).

| fps | frame (ms) | σ_interval raw | RMSSD floor raw | σ_interval @ gain 0.65 | RMSSD floor @ 0.65 |
|---|---|---|---|---|---|
| 24 | 41.7 | 17.0 | 29.5 | 11.1 | 19.2 |
| 30 | 33.3 | **13.6** | 23.6 | 8.8 | **15.3** |
| 60 | 16.7 | **6.8** | 11.8 | 4.4 | **7.7** |
| 120 | 8.3 | 3.4 | 5.9 | 2.2 | 3.8 |

The budget is the **rate-averaged iid model**. Naive round-to-frame
beat picking of a true metronome would instead be a deterministic
sawtooth — zero when the pulse period sits on the frame grid, up to a
full frame period when it sits half-way — which is why the pipeline
locates beats with sub-frame interpolation and why the *measured* gain,
not the budget, is what the MDI table rests on. The ladder calibrates
that gain at one pulse rate (70 bpm); a rate sweep of the metronomic
rungs is a recorded follow-up (review finding), not a claim.

The gain 0.65 is a *planning* value for sub-frame interpolation. Every
floor run **measures** it from the metronomic ladder rungs (ECG RMSSD
under 5 ms, the reference's own dispersion removed in quadrature) at
each nominal fps and reports it beside the plan:

| fps | metronomic refs | predicted RMSSD floor raw (ms) | camera RMSSD on refs, p50 (ms) | camera-only RMSSD, p50 (ms) | measured gain |
|---|---|---|---|---|---|
| 30 | 12 | 23.6 | 19.3 | 19.2 | 0.81 |
| 60 | 12 | 11.8 | 11.2 | 10.7 | 0.91 |

**Minimum detectable irregularity (MDI).** Per (nominal fps, SQI
grade, Fitzpatrick group) cell: the floor is the 95th percentile of the
camera index on ACCEPT scans whose ECG says metronomic (RMSSD < 5 ms)
at that fps; the MDI is the median ECG-RMSSD of the lowest
true-dispersion bin the camera detects above that floor with ≥ 80 %
power, provided every better-populated bin above it does too
(detection must be monotone in true dispersion). A scan the camera
could not read is a **non-detection**, in the denominator. The gain is
the camera's own contribution — the reference's dispersion is removed
in quadrature. A cell with no references or no such bin is **not
characterized** and reds R1 — an uncharacterized cell is not a zero.
The power and confidence are read from `configs/gates.yaml`, recorded
on the floor run, and checked by R1.

| fps | SQI grade | Fitzpatrick | n scans (no-read) | floor (index p95 on refs) | MDI (ECG RMSSD, ms) | note |
|---|---|---|---|---|---|---|
| 30 | A | 1 | 4 (0) | 0.0222 | 204.9 |  |
| 30 | A | 2 | 25 (0) | 0.0222 | 162.8 |  |
| 30 | A | 3 | 4 (0) | 0.0222 | 252.0 |  |
| 30 | A | 4 | 4 (1) | 0.0222 | **not characterized** | no ECG-dispersion bin reached the required power — not characterized |
| 30 | A | 5 | 25 (0) | 0.0222 | 229.6 |  |
| 30 | A | 6 | 4 (0) | 0.0222 | 163.3 |  |
| 60 | A | 1 | 4 (0) | 0.0154 | 163.6 |  |
| 60 | A | 2 | 25 (0) | 0.0154 | 31.9 |  |
| 60 | A | 3 | 4 (0) | 0.0154 | 163.4 |  |
| 60 | A | 4 | 4 (0) | 0.0154 | 240.8 |  |
| 60 | A | 5 | 25 (0) | 0.0154 | 163.1 |  |
| 60 | A | 6 | 4 (0) | 0.0154 | 218.8 |  |

**False irregularity vs beat-error rate** (truly regular source, RMSSD
8 ms, 60 s, half the errors missed beats and half confident false
beats; 30 trials per cell):

| fps | beat error | clean runs: false-irregular (95 % CI) | clean runs: index p50 | clean runs: RMSSD p50 (ms) | raw series: false-irregular | raw: index p50 | raw: RMSSD p50 (ms) |
|---|---|---|---|---|---|---|---|
| 30 | 0% | 0.000 (0.000–0.114) | 0.0138 | 16.7 | 0.000 | 0.0138 | 16.7 |
| 30 | 2% | 0.000 (0.000–0.114) | 0.0131 | 16.3 | 0.000 | 0.0140 | 83.1 |
| 30 | 5% | 0.000 (0.000–0.114) | 0.0132 | 17.1 | 0.000 | 0.0149 | 194.5 |
| 30 | 10% | 0.000 (0.000–0.114) | 0.0151 | 17.8 | 0.000 | 0.0197 | 313.5 |
| 30 | 20% | 0.000 (0.000–0.114) | 0.0140 | 21.1 | 0.067 | 0.0248 | 428.1 |
| 30 | 30% | 0.000 (0.000–0.114) | 0.0158 | 76.3 | 0.500 | 0.0635 | 534.2 |
| 30 | 40% | 0.100 (0.035–0.256) | 0.0265 | 188.4 | 0.933 | 0.2989 | 630.5 |
| 60 | 0% | 0.000 (0.000–0.114) | 0.0087 | 11.1 | 0.000 | 0.0087 | 11.1 |
| 60 | 2% | 0.000 (0.000–0.114) | 0.0089 | 11.4 | 0.000 | 0.0095 | 150.0 |
| 60 | 5% | 0.000 (0.000–0.114) | 0.0083 | 11.0 | 0.000 | 0.0101 | 228.8 |
| 60 | 10% | 0.000 (0.000–0.114) | 0.0084 | 11.2 | 0.000 | 0.0126 | 286.2 |
| 60 | 20% | 0.000 (0.000–0.114) | 0.0089 | 38.8 | 0.067 | 0.0186 | 464.9 |
| 60 | 30% | 0.000 (0.000–0.114) | 0.0111 | 101.1 | 0.433 | 0.0304 | 520.0 |
| 60 | 40% | 0.033 (0.006–0.167) | 0.0118 | 126.5 | 0.867 | 0.2193 | 664.5 |

Read the two paths together: the clean-run architecture keeps *every*
dispersion statistic honest, while the index's median protects only
against *sparse* errors — on the raw series the index only tips past
the threshold once errors are no longer sparse, but its RMSSD is
already wrecked at 5 %.

## RSA handling (R2)

Respiration comes from the torso-motion second decode
(`rppg/respiration.py`), never from the facial trace the intervals come
from (that would be circular). `structure.coupling` reports the
tachogram's respiratory-band fraction, the fraction at the reported
breathing rate, phase locking, and a status. An irregular read whose
tachogram energy sits at the breathing rate is `respiration_coupled` —
irregular *and explained*. `clinically_irregular()` in the harness
counts only irregular reads that are **not** respiration-coupled, and
R2 requires that RSA-dominant sessions be flagged that way in ≤ 10 %
of cases (at least 20 judged RSA sessions), with specificity ≥ 0.90 in
**each** age band (<35, 35–59, ≥60; at least 20 judged sessions per
band). Every denominator is reported total / judged / no-read and the
no-read rate per band is bounded (≤ 0.30): a head that declines the
sessions it would get wrong must not be able to buy specificity with
silence. A collapse under 35 — the RSA band — is the pre-registered
failure that forces an age-aware threshold or a narrowed claim.

RSA-annotated sessions: **18** total, **18** judged (no-read rate 0.000); flagged as clinically irregular: **0.000**; explained as respiration-coupled: **0.778**; benign sessions overall specificity: **1.000** (114 judged of 114); abstained rows in the cohort: 2.

| age band | benign sessions total | judged | no-read rate | false irregular | specificity (Wilson 95 %) |
|---|---|---|---|---|---|
| <35 | 9 | 9 | 0.000 | 0 | 1.000 (0.701–1.000) |
| 35-59 | 71 | 71 | 0.000 | 0 | 1.000 (0.949–1.000) |
| >=60 | 34 | 34 | 0.000 | 0 | 1.000 (0.898–1.000) |

## What was measured (synthetic video, production path)

The cohorts (`scripts/make_synth_regularity.py`, every clip with a
breathing torso bar, a truth sidecar, an ECG R-peak sidecar from the
*same* RR series that drives the pulse, and a participant sidecar whose
Fitzpatrick group sets the clip's **skin base colour** with the pulse
amplitude scaled by its reflectance — a crude melanin model, disclosed;
group, frame rate and site are not confounded; ages span the bands
wherever physiology allows, RSA being age-specific by design):

| cohort | clips | rhythm label | ECG index (median) | camera index (median) | ECG class / camera class agree | head class (n) | head evidence (n) |
|---|---|---|---|---|---|---|---|
| af | 6 | AFIB | 0.274 | 0.220 | 6/6 | irregular (5), None (1) | chaotic (5), None (1) |
| bigeminy | 6 | PVC_FREQUENT | 0.556 | 0.596 | 5/6 | irregular (5), None (1) | ectopy_pattern (5), None (1) |
| ladder (metronomic to 32 ms RMSSD) | 84 | SINUS | 0.009 | 0.017 | 84/84 | regular (84) | not_applicable (84) |
| pac_isolated | 6 | PAC | 0.042 | 0.048 | 6/6 | regular (6) | not_applicable (6) |
| regular | 6 | SINUS | 0.006 | 0.017 | 6/6 | regular (6) | not_applicable (6) |
| rsa_mid | 6 | RESPIRATORY_SINUS_ARRHYTHMIA | 0.072 | 0.057 | 2/6 | regular (4), irregular (2) | not_applicable (4), respiration_coupled (2) |
| rsa_old | 6 | RESPIRATORY_SINUS_ARRHYTHMIA | 0.091 | 0.078 | 6/6 | irregular (6) | respiration_coupled (6) |
| rsa_young | 6 | RESPIRATORY_SINUS_ARRHYTHMIA | 0.147 | 0.125 | 6/6 | irregular (6) | respiration_coupled (6) |
| trigeminy | 6 | PVC_FREQUENT | 0.384 | 0.400 | 6/6 | irregular (6) | ectopy_pattern (6) |

**Ceiling test on the synthetic cohort (production video path, 131 paired scans of 132; 1 indeterminate on a side):** κ = **0.919**, index correlation r = **0.975**, Bland–Altman camera-minus-ECG bias 0.0003 with 95 % limits ±0.0622; worst SQI-grade κ = 0.919.

| SQI grade | n | κ | r |
|---|---|---|---|
| A | 131 | 0.919 | 0.975 |

Read the disagreements, not the κ: four of the five are `rsa_mid` clips
whose ECG index sits at 0.070–0.072 and whose camera index reads
0.053–0.059. On ECG-irregular clips the camera index reads **below the
ECG's** (median camera/ECG ratio 0.86, IQR 0.80–1.01; ≈ 0.8 on the RSA
cohorts) while regular clips carry a small positive jitter bias, so the
bias cancels and the limits of agreement widen past R0's 0.04. The
published threshold is defined on the ECG scale. Whether the camera
index is mapped onto it (a calibration needing its own validation) or
the threshold is re-set for the camera is a clinical-signoff decision,
flagged in spec B.23, not taken here. On the contact-PPG surrogate the
same ratio is 1.03, so the compression belongs to the camera path or to
the synthetic optics, not to the statistic. It is also why B3 — a
logistic rule fitted on the train half, which learns the camera's own
scale — beats the head's pre-registered threshold on this fixture.

## The first ceiling number on human hearts (public surrogate)

MIMIC PERform AF (contact fingertip PPG + ECG + impedance respiration,
125 Hz, ICU patients, 20 min per subject) is the only public corpus
with simultaneous rhythm truth, a pulse waveform and respiration. It is
run through the same beat detector, run discipline, representation and
reference label, on 90 s windows, and recorded as `signal_domain:
public_ppg` — a **surrogate that can never open a gate** (the
qualification rule reds every gate on it). There is no public
contactless corpus with ECG truth.

**MIMIC PERform AF, 35 subjects, 210 windows (114 AF windows), 210 paired:** κ = **0.873**, index r = **0.759**, Bland–Altman bias 0.0228, 95 % limits ±0.1364; worst SQI-grade κ = 0.873.

| dataset | n | κ | r |
|---|---|---|---|
| mimic_perform_af_csv | 114 | — | 0.841 |
| mimic_perform_non_af_csv | 96 | 0.101 | 0.108 |

Head on this corpus: judged 84 of 210 windows (abstained 126, almost all AF windows with an undecodable respiration channel); on the rows the head judged, balanced accuracy head 0.955 vs B1 0.818, B2 0.455, B3 0.909, B5 — (B4 has no demographics here); head no-read rate 0.778.

The AF subjects' impedance respiration is mostly undecodable (spectral
concentration ≈ 0.01), so the head abstains on most AF windows in this
corpus while the ceiling test — camera-vs-ECG index agreement, which
needs no respiration — reads them all. All AF windows read irregular on
both sides (κ within AF is undefined — a single class — and is
reported as such, never as 1.0). Within non-AF, 12 windows read
irregular on the PPG that the ECG calls regular: the surrogate
over-reads irregularity on ICU contact PPG, the direction a
beat-detection error produces. The head's balanced accuracy on this
corpus rests on 3 held-out participants behind a 0.78 no-read rate and
is a number to disbelieve; R3 would red on the no-read rate alone.

## Baselines (R3)

| | Rule | Fitted on |
|---|---|---|
| B1 | RMSSD ≥ threshold | threshold chosen on the participant-disjoint train half |
| B2 | Shannon entropy ≥ threshold | same |
| B3 | logistic on RMSSD + entropy + rate | train half |
| B4 | logistic on age + sex + skin-tone group | train half |
| B5 | logistic on SQI alone | train half |
| head | the published rule, no fitting | — |

All are scored on the held-out half against the **ECG** class; the
logistic baselines' operating points are chosen on the train half for
balanced accuracy (a fixed 0.5 collapses to the majority class under
imbalance). The head is scored on the rows it judged, **every baseline
is re-scored on those same rows** for the gate's comparison, and the
head's no-read rate is gated (≤ 0.20) beside its balanced accuracy so
the comparison cannot hide behind abstention. On a synthetic cohort where age is assigned by cohort, B4
leaks the label *by construction* and can beat the head; that is a
fixture property, not a finding about people.

| | balanced accuracy, all held-out rows | on the rows the head judged |
|---|---|---|
| B1 | 0.939 | 0.936 |
| B2 | 0.468 | 0.477 |
| B3 | 0.977 | 0.977 |
| B4 | 0.587 | 0.610 |
| B5 | 0.551 | 0.538 |
| head | — | 0.917 |
| head no-read rate | 0.018 | |
| held-out participants (judged) / rows | 57 (56) / 57 | |

## The §R scoreboard

Evaluation run `reg-7d4a714b7aad`, floor run `rfloor-c8c3760e20a4`, gates `R-2026-09-01-1`:

| gate | status | first reason |
|---|---|---|
| R0 — ceiling: camera vs ECG regularity agreement | **RED** | evidence domain 'synthetic' is not 'facial_rppg' — machinery evidence only, can never open a gate |
| R1 — noise floor published (MDI per fps / SQI / skin tone) | **RED** | evidence domain 'synthetic' is not 'facial_rppg' — machinery evidence only, can never open a gate |
| R2 — benign separation (RSA not flagged; age-stratified specificity) | **RED** | evidence domain 'synthetic' is not 'facial_rppg' — machinery evidence only, can never open a gate |
| R3 — beats B3, B4, B5 subject-independent | **RED** | evidence domain 'synthetic' is not 'facial_rppg' — machinery evidence only, can never open a gate |
| R4 — fairness (index bias, flag rate, coverage parity) | **RED** | evidence domain 'synthetic' is not 'facial_rppg' — machinery evidence only, can never open a gate |
| R5 — claim mapping (owner + clinical advisor) | **RED** | evidence domain 'synthetic' is not 'facial_rppg' — machinery evidence only, can never open a gate |

Every gate is RED on synthetic evidence by the qualification rule
(`evidence domain 'synthetic' is not 'facial_rppg' — machinery evidence
only, can never open a gate`), on top of any numeric shortfall. The
first thing that can turn a gate green is a registered, consented,
participant-disjoint facial-video cohort with simultaneous ECG.

## Invariants, each a test

| | Invariant | Test |
|---|---|---|
| G-a | one representation — a second interval-statistics implementation anywhere in `inference/` or `heads/` is a bug | `tests/test_regularity_substrate.py` (AST audit with a tested allowlist; the v1 path is gone) |
| G-b | graded, not binary — a class without index + CI is a bug | `tests/test_regularity_head.py`, `tests/test_regularity_reference.py` |
| G-c | benign-irregularity separation is mandatory on every irregular read | `tests/test_regularity_head.py` |
| G-d | no rhythm naming; `user_facing_text()` is the single gate | `tests/test_regularity_head.py`, `tests/test_scanresult_v2.py` |
| — | head_afib bit-identical after the rewiring | `tests/test_regularity_equivalence.py` (golden; regenerate only with `AVATARX_REGEN_GOLDEN=1` and a spec entry) |
| — | clean runs only, no interval repair | `tests/test_regularity_substrate.py`, `tests/test_regularity_reference.py`, `tests/test_regularity_floor.py` |
| — | SQI never scores periodicity | `tests/test_signal_quality.py` (behavioural) + `tests/test_regularity_substrate.py` (structural: the SQI has no path to the substrate) |
| — | gates read booleans and bounded, finite numbers; stale evidence (another `gates_version` or reference label) reds every gate; the registry door re-evaluates the recorded evidence under the live, signed `gates.yaml` | `tests/test_regularity_gates.py` |

## Operator loop

```bash
python3 cli.py register-dataset day_regularity/
python3 cli.py regularity-floor day_regularity/ --domain facial_rppg   # R1 evidence
python3 cli.py evaluate-regularity day_regularity/ --domain facial_rppg
python3 cli.py gate-status --track regularity --html scoreboard.html
```

Fixtures: `python3 scripts/make_synth_regularity.py out_dir/` writes the
cohorts; `make_dispersion_ladder()` writes the metronomic-to-32 ms
ladder that calibrates the gain and the MDI table.

## Honesty note

Every number in this document derives from synthetic video (an
interface proof that the stages compose and the gates fire) or from a
contact-PPG public surrogate. No clinical performance is claimed or
implied. The head renders nothing.
