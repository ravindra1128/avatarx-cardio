# Known-miss registry — the regular-tachyarrhythmia flag (v0.6, §F)

**Status: §F is RED. Nothing described here renders to any user today.**
This document is a shipped artifact, not a caveat (invariant F-c): the
product may never contradict it, and tests assert that it does not.

The v0.6 track adds a *pattern flag* for a sustained fast, unusually
regular pulse. It is not a flutter detector, and the sections below are
the specific, named things it cannot do. They are limitations of the
signal, not of the current implementation, so most of them will not be
fixed by better engineering.

---

## 1. It cannot name a rhythm. Ever.

Atrial flutter is named from sawtooth F waves — atrial electrical
activity, visible on a 12-lead ECG in the inferior leads. Facial video
carries the **ventricular response only**: when a pulse arrived, and how
strong it was. Atrial activity is not in that channel at any frame rate,
under any lighting, with any model.

Consequently, at ~150 bpm with a metronomic pulse, these are the *same
measurement* to this system:

| Rhythm | Pulse at the camera | Clinical urgency |
|---|---|---|
| Atrial flutter, 2:1 conduction | regular, ~150 | high |
| SVT / AVNRT | regular, ~150–180 | variable |
| Sinus tachycardia (fever, exercise, anxiety, dehydration, caffeine) | regular, ~150 | usually benign |

The sanctioned sentence therefore names nothing and routes to the one
instrument that can:

> "A sustained fast, unusually regular pulse was detected. This can occur
> with several heart-rhythm conditions — please have an ECG."

**SVT is covered by design, not separated.** A flag on an SVT is not a
false positive of this design; it is the flag working. Only the ECG
decides which rhythm it was.

## 2. Slow fixed-block flutter is a permanent miss

This is the headline limitation, and the one most likely to matter
clinically.

Flutter's atria fire at ~250–300/min and the AV node conducts a fixed
fraction. The resulting pulse depends entirely on that fraction:

| Conduction | Pulse rate | Visible to this flag? |
|---|---|---|
| 2:1 | ~150 | Yes — the flag's target |
| 3:1 | ~100 | Non-specific: collides with ordinary tachycardia |
| 4:1 | ~75 | **No — indistinguishable from normal sinus rhythm** |
| Variable | irregular | No — lands in AF's feature space (see §4) |

A patient in 4:1 flutter has a pulse of about 75 beats per minute that is
*more* regular than a healthy person's. By pulse timing alone that is
normal sinus rhythm, and the rate band that keeps the flag off ordinary
resting rhythms necessarily keeps it off this patient too. The system is
expected to report "no irregular rhythm detected" for a patient in
sustained atrial flutter.

This is quantified rather than asserted: gate **F2** requires per-ratio
sensitivity to be *measured and reported* for 2:1, 3:1, 4:1 and variable
block, and refuses to go green if the 4:1 number is missing. A measured
sensitivity of zero at 4:1 passes F2. Omitting it does not.

**A negative scan does not exclude atrial flutter.** It does not exclude
atrial fibrillation either (which is already in the standard result
wording), and rate-controlled or slow fixed-block flutter is a specific,
expected reason why.

## 3. The confounder that decides the product is sinus tachycardia

A regular fast pulse is *usually benign*. At screening prevalence, sinus
tachycardia is orders of magnitude more common than flutter, so the
flag's specificity — not its sensitivity — determines whether it helps or
generates false referrals.

What separates them, when anything does, is **respiratory sinus
arrhythmia**: sinus rhythm keeps some coupling between the interval
series and breathing at every rate, while fixed AV conduction decouples
the ventricle from the respiratory drive. On the v0.6 fixtures, a
metronomic clip and a true sinus tachycardia at matched rate *and*
matched interval dispersion (RMSSD ≈ 25 ms) had tachogram
respiratory-band fractions of 0.03 vs 0.91.

Three consequences:

* **No respiration channel means no flag.** The head abstains rather than
  flagging when the camera-derived respiration signal is missing or too
  poor to test coupling. An untested coupling is not an absent one.
  Today's consumer scan path supplies no respiration channel at all, so
  the head abstains on every production scan by design.
* **The rate band is not a second exit.** The 2:1 band spans 125–165 so
  that it actually covers 2:1 conduction of the 250–300/min atrial range
  above; a narrower band would silently miss most of the flag's own
  declared target. The cost, measured: deep-RSA sinus tachycardia now
  *sustains* that band (in-band time fraction 0.88), where a narrower
  band would have excluded it on rate alone. Respiratory coupling and
  the dispersion floor therefore carry the entire discrimination against
  this confounder — there is no independent third criterion holding it
  back.
* The coupling test measures power in a narrow band around the
  *reported* breathing rate, so a wrong rate would read as absent
  modulation. The measurement is refused unless the tachogram's own
  spectrum corroborates the reported rate, and refused outright when the
  respiration channel's quality is below its validated floor. A refusal
  makes the head abstain; it never becomes a flag.
* Patients whose RSA is physiologically absent or minimal (advanced age,
  autonomic neuropathy, some medications, heart failure) lose this
  discriminator. Their sinus tachycardia looks like the flag's target.
  This population is a known specificity risk and is not yet quantified.

## 4. Variable-block flutter is AF's problem, not this flag's

When conduction varies beat to beat, flutter produces an *irregular*
pulse that overlaps atrial fibrillation's feature space. The v0.6 track
computes variable-block features so this confusion is measurable, but
the pattern flag does not fire on it, and the AF head may. The combined
referral is what covers this case; the ECG names it.

## 5. At 30 fps the "too regular" criterion is measurement-limited

Beat times are estimated from a sampled waveform, so a perfectly
metronomic source still reads a non-zero interval dispersion. Measured
through the production path: a zero-jitter 150 bpm clip reads RMSSD
**15.0 ms at 30 fps** and **8.2 ms at 60 fps** — about 0.45 of a frame
period in both cases. That is quantization, not physiology.

The age-adjusted physiological floors this track pre-registered (8–18 ms,
falling with age because heart-rate variability does) sit *below* the
30 fps measurement floor. So at 30 fps the physiological criterion is
entirely masked: what is being tested is "regular relative to the
camera", not "regular relative to a heart". Every scan where this
happens is marked `measurement_limited`, and a flag raised under such a
floor carries that caveat in its reasons.

Practical consequence: **capture at 60 fps materially improves this
track**, roughly halving the floor. Whether even 60 fps is sufficient for
the specificity F1 requires is unproven.

## 6. What the evidence base does and does not support

The only facial-video evidence involving flutter is Cramer et al. 2025
(J Clin Monit Comput): 51 cardioversion patients (38 AF, 13 flutter), a
machine-vision camera at 32 Hz and 2 m, leave-one-subject-out SVM on
rPPG interval features. It reported AUC 0.95 for arrhythmia vs sinus,
with 94.6% of AF and 92.5% of flutter correctly classified.

It does **not** support:

* naming flutter (the study classified arrhythmia vs sinus, in a
  population already known to be in arrhythmia);
* detecting slow fixed-block flutter (a cardioversion cohort is
  dominated by fast, symptomatic presentations);
* performance on the confounders that matter at screening — its "other
  arrhythmias" class was classified correctly only **61.8%** of the time;
* accurate rate during arrhythmia — HR bias was **−7.45 bpm**, with only
  **48.6%** of readings within 5 bpm, which is why §F0 gates rate
  accuracy *before* any rate-based flag is meaningful;
* equity — the single Fitzpatrick VI patient was excluded. §F4 reds the
  gate if the darkest skin-tone band is absent from our cohort. We do not
  get to make that exclusion.

## 7. Summary of what may never be said

While §F is red, no user-facing surface may contain the words *flutter*,
*SVT*, *atrial tachycardia*, a conduction ratio such as *2:1*, or the
phrase *conduction ratio* (invariant F-b, enforced by test). After §F is
green and signed, the only permitted wording is the sanctioned sentence
in §1 — which still names no rhythm.

At no point, red or green, may this product state or imply:

* that a rhythm has been identified as flutter;
* that flutter has been excluded by a negative scan;
* that a regular pulse is reassuring;
* a conduction ratio, an atrial rate, or any ECG-derived quantity.
