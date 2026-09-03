# The atrial-flutter research track (v0.6, §F)

**Question, pre-registered (gates F0–F5):** can a facial-video scan
raise a defensible *suspicion* of a sustained regular tachyarrhythmia —
one specific enough to be worth an ECG and specific enough not to
flood clinics with people who were merely anxious, febrile or on a
treadmill an hour ago?

Not "can we detect flutter". The distinction is the whole track.

Until every §F gate in the `flutter:` block of `configs/gates.yaml` is
green with owner + clinical signoff, `head_flutter` produces **no
user-facing output of any kind** (F-b), the strings `flutter`, `SVT`,
`atrial tachycardia`, `2:1` and `conduction ratio` are forbidden on
user surfaces (F-b, token tests), the known-miss registry ships as
`docs/flutter_limitations.md` and the product may not contradict it
(F-c), and the flag is measured-path only — never routed through the
gated ECG-reconstruction head (F-d, AST test).

## The physiology, which inverts the AFib architecture

AFib is detected because its pulse is **irregular**: the AFib head is,
at bottom, an irregularity detector with a great deal of machinery
protecting it from calling detection noise "irregular".

Atrial flutter is the arrhythmia that architecture is **structurally
blind to**. The atria fire at ~250–300/min and the AV node conducts a
fixed fraction, so the ventricles — and therefore the pulse — are
*metronomically regular*:

| Conduction | Pulse | What an irregularity detector sees |
|---|---|---|
| 2:1 | ~150 | a fast but perfectly regular rhythm |
| 3:1 | ~100 | an ordinary rhythm |
| 4:1 | ~75 | normal sinus rhythm |

Flutter also **cannot be named** from the pulse. Distinguishing it from
SVT or sinus tachycardia at the same rate requires sawtooth F waves —
atrial electrical activity, on an ECG. Facial video carries the
ventricular response only. No frame rate or model changes this.

So the track builds a **rate-and-regularity pattern flag** that surfaces
suspicion and routes to an ECG, plus the serial-scan signature that is
flutter's most distinctive camera-observable trait.

## Three feature families (`features/flutter.py`)

1. **Rate fingerprint** — sustained median rate with a
   distribution-free CI inside a conduction band, plus what fraction of
   the scan's TIME sustains it. The bands are derived from the atrial
   rates they imply, not picked: 2:1 of 250–330/min is **125–165**, and
   that is the only band carrying a specific claim. 3:1 (83–110) and
   4:1 (63–83) are computed and are non-specific by arithmetic — they
   sit inside ordinary human heart rates, which is precisely why slow
   fixed-block flutter is a permanent miss.
2. **Hyper-regularity** — interval dispersion below a floor, tachogram
   spectral concentration, and **absent respiratory modulation**. The
   last is load-bearing: sinus rhythm at any rate retains some
   respiratory sinus arrhythmia, while fixed AV conduction decouples
   the ventricle from the respiratory drive.
3. **Serial conduction-ratio steps** — across a registered scan series,
   rates clustering at integer-ratio steps of ONE latent atrial rate
   (150 → 100 → 75 implies ~300). Nothing else in cardiology produces
   this pattern, which makes it the track's differentiator. It scores
   zero unless at least two *separated* divisors are occupied: a series
   sitting at one rate is divisible by any atrial rate, and calling
   that a conduction pattern would be a free pass.

Variable-block features are emitted alongside so the AF/flutter
confusion is measurable rather than hidden.

## Two things we measured before believing them

**RSA coupling is the discriminator, and it survives the camera.** On
twin clips through the production path at the same rate *and matched
dispersion* (RMSSD ≈ 25 ms both), the tachogram's respiratory-band
fraction was **0.03 for the metronomic clip and 0.91 for sinus
tachycardia** — a separation dispersion alone cannot make. This is why
the head abstains rather than flagging when no respiration channel is
available: without it, 2:1 flutter and sinus tachycardia at 150 are the
same measurement.

**The dispersion floor is the camera, not the patient.** A perfectly
metronomic 150 bpm clip reads RMSSD **15.0 ms at 30 fps** and **8.2 ms
at 60 fps** — 0.45 of a frame period in both cases, i.e. quantization.
The age-adjusted physiological floors this track pre-registered (8–18
ms, falling with age as HRV does) therefore sit *below* the 30 fps
measurement floor and are entirely masked at consumer frame rates. The
effective floor is `max(age floor, camera floor)` and any scan where
the camera wins is marked `measurement_limited`. Practical consequence:
**capture at 60 fps materially improves this track**, and the campaign
checklist asks for it.

## The hard-negative battery (Task 3) — where the product risk is

A regular fast pulse is *usually benign*, so specificity, not
sensitivity, decides whether this flag helps. Every confounder is a
labeled class with its own reported false-positive rate:

| Confounder | Why it mimics | How it is handled |
|---|---|---|
| Sinus tachycardia ~150 | identical rate | RSA coupling retained; **must dominate the battery** (gate F1 refuses a specificity earned on easy negatives) |
| SVT / AVNRT | regular, abrupt | **cannot** be separated by pulse — covered by design, and the sanctioned sentence names neither |
| Beta-blocked / elderly low-variability sinus | hyper-regular, normal rate | rate band; age-adjusted floor |
| AF with regularized rate | overlapping | joint evaluation with `head_afib` (Task 6) |
| Pacemaker rhythm | metronomic by design | rate band; reported separately |
| **Slow fixed-block flutter ~75** | looks like NSR | **documented permanent miss**, quantified by gate F2 |
| Motion/expression artifact | can fake regularity | existing SQI + evidence gates; the scan never reaches ACCEPT |

**Baselines.** B1 rate-only; B2 rate + naive dispersion; **B3 rate band
+ age-adjusted floor + respiratory coupling — the interpretable
competitor**; B4 demographics + rate. Head and B3 are compared at
matched sensitivity on participant-disjoint splits. B3 is the head's own
rule minus its abstention discipline, so the head's advantage, if any,
is that it declines what it cannot judge — which is why its specificity
is always reported beside its no-read rate. **If the head does not beat
B3 by the pre-registered margin, B3 ships**, and `shipping_rule` on the
scoreboard says so.

## The combined endpoint (Task 6) — and why it is not yet the answer

The three referral candidates are scored on the same rows against the
same combined truth (AF **or** flutter present), because that is the
question the referral actually asks. Measured on the 76-scan synthetic
cohort at planning prevalence (AF 0.03, flutter 0.003):

| candidate | Se | Sp | PPV | FP/1000 | false alarms per true case |
|---|---|---|---|---|---|
| AFib head alone | 0.088 | 0.905 | 0.031 | 92.1 | 31.6 |
| flag alone | 0.333 | 0.950 | 0.185 | 48.4 | **4.4** |
| combined | **0.412** | 0.857 | 0.090 | 138.1 | 10.2 |

The combined endpoint wins on sensitivity — but that is an *identity*,
not a finding: it is an OR of the two heads, so it dominates both
elementwise no matter how noisy either one is. It buys that extra catch
with 2.9× the false alarms per true case. Candidates are therefore
ranked by **false alarms per true case** among those clearing a minimum
sensitivity, and on this cohort **no candidate clears the floor**, so
the harness records that the decision cannot be made on the numbers and
F5 stays red. The AF head's 0.088 sensitivity here is a fixture
artifact, so this cohort cannot settle the question either way. Arm B
of the campaign is what settles it.

## §F gate table

| Gate | What it demands | Anchored to |
|---|---|---|
| **F0** rate accuracy in arrhythmia | bias ≤ ±3.0 bpm, ≥80% within 5 bpm, ≥30 arrhythmia scans | Cramer measured −7.45 bpm and 48.6% — we must do materially better before a rate-based flag means anything |
| **F1** flag performance | 2:1 sensitivity ≥0.80, specificity ≥0.95 vs the full battery, beats B3 by ≥0.02, ≥40 flutter and ≥200 negative scans, sinus tach dominant | specificity is the product risk |
| **F2** per-ratio reporting | all four ratios reported with ≥10 scans each; 4:1 sensitivity **measured** | turns F-c from assertion into number |
| **F3** serial signature | series AUC ≥0.85, ≥10 flutter and ≥20 non-flutter series, ≥3 scans each | the differentiator |
| **F4** fairness | detection parity ≥0.75, coverage ≥0.80, ≥5 participants/group, **darkest band present** | the reference study excluded its only Fitzpatrick-VI patient |
| **F5** claim mapping | owner + advisor signature, `claim_scope: regular_tachy_ecg_referral`, combined-endpoint decision recorded, limitations doc published | the sanctioned sentence names no rhythm |

Every threshold is a **planning value**, pre-registered so it cannot
drift; only data may revise it, via a spec-changelog entry.

## Campaign plan (planning estimates — `configs/campaign_flutter_template.yaml`)

Two arms, neither substitutable:

- **Arm A, positives (~45 participants):** elective cardioversion and
  EP/ablation clinics — the only reliable source of ECG-confirmed
  flutter. 12-lead with EP-level adjudication is required for every
  positive label (`flutter_label_problems` enforces it); 3:1 and 4:1
  are oversampled deliberately.
- **Arm B, negatives (~75 participants):** a prospective
  regular-tachycardia cohort — exercise recovery, anxiety/panic, fever
  presentations. Without this arm, specificity is unmeasurable and the
  flag cannot ship.
- **Serial sub-protocol:** ≥3 scans per patient across a rate-varying
  window (pre/post rate-control titration, pre/post cardioversion), 30
  series.
- Fitzpatrick VI is quota'd at 18, not hoped for.

## Evidence baseline

Cramer et al. 2025 (J Clin Monit Comput), N=51 cardioversion patients
(38 AF, 13 flutter), machine-vision camera at 32 Hz / 2 m,
leave-one-subject-out SVM on rPPG interval features: AUC 0.95
arrhythmia-vs-sinus, AF 94.6% and flutter 92.5% correctly classified —
but **"other arrhythmias" only 61.8%**, HR bias −7.45 bpm during
arrhythmia, and the single Fitzpatrick-VI patient excluded. That is a
*cardioversion-clinic arrhythmia-vs-sinus* result, not evidence that
flutter can be named or that slow fixed-block flutter is detectable.
**It is the floor to beat, not the target.**

## Operator loop

```bash
python3 cli.py register-dataset <dir> --license-class internal_consented \
    --consent-class research_v1
python3 cli.py evaluate-flutter <dir> --domain facial_rppg
python3 cli.py gate-status --track flutter --html scoreboard.html
```

`--domain` defaults to `synthetic`; only `facial_rppg` evidence can ever
open a gate, and surrogate runs are recorded with the disqualifier
attached to every gate.

## Escalation clause

If any instruction requires showing a flutter/regular-tachyarrhythmia
output to users while §F gates are red, naming a rhythm on any surface,
removing the hard-negative battery or the B3 baseline from an
evaluation, or routing the flag through the gated ECG-reconstruction
head: **do not implement**. Record the request verbatim in the spec
changelog under *"requires owner decision — conflicts with §F / F-a /
F-d"*.
