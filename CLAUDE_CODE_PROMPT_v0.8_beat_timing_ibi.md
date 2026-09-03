# Claude Code prompt — AvatarX v0.8: Beat-to-Beat Timing (IBI) — the Measurement Layer

<!--
HOW TO USE
  1. Place this file in the repo root at AvatarX/code/afib/ (next to CLAUDE_CODE_SPEC.md).
  2. cd into the repo and start Claude Code:  claude
  3. Say: "Read CLAUDE_CODE_PROMPT_v0.8_beat_timing_ibi.md and execute it."
  Hardens L3 (beats/) beneath v0.7 regularity, v0.2 AFib, v0.6 flutter.
  Depends on v0.2 data/training engines; shares gate CLI with v0.3–v0.7.
-->

---

You are hardening **AvatarX's beat-to-beat timing layer** — the inter-beat
interval (IBI) measurement that every rhythm endpoint in this repo stands
on. Frame the work correctly before starting:

> **IBI is not an endpoint head. It is the measurement layer (L3), and its
> error budget is the ceiling for every track above it.** Regularity cannot
> resolve irregularity smaller than our timing noise; AFib and flutter
> classifiers inherit our missed- and false-beat rates as their false
> positives; any HRV-class wellness metric is bounded by it twice over. This
> prompt therefore produces a *characterized instrument*, not a feature: a
> `BeatLattice v2` with per-beat timestamps, confidence, and a **published,
> decomposed error budget**.

Four physical facts the design must respect:

1. **Pulse timing ≠ R-peak timing.** Each optical beat arrives at the face
   at R-peak + pre-ejection period + pulse transit (~100–200 ms at the head).
   The offset itself is harmless for intervals — it cancels — but its
   *beat-to-beat variation* does not. Echo studies put resting PEP jitter at
   roughly 2–6 ms SD, rising to tens of ms across autonomic state changes
   (stress, posture, exercise). **This is an irreducible physiological floor:
   our target is not 0 ms error versus ECG R-R, and a system claiming to
   beat this floor is measuring its own overfitting.**
2. **What we measure is pulse-rate variability, not heart-rate variability.**
   PRV and HRV diverge measurably (they can differ even when HRV is absent,
   as in pacemaker cases). Label the output accordingly, everywhere.
3. **Frame rate quantizes timing.** Uniform-sampling quantization on an
   interval is ≈ √2·T/√12 — about 13.6 ms RMS at 30 fps, 6.8 at 60, 3.4 at
   120 (compute these in code; do not hard-code). Sub-frame interpolation on
   a high-SNR pulse beats these bounds — published work reaches ~19.5 ms
   interval RMSE even at 15 fps — but only when SNR is high, which makes
   interpolation gain an **SQI-conditional** quantity to measure, not assume.
4. **A single missed beat is worse than noise.** It merges two intervals and
   can inflate dispersion several-fold — which is precisely why the repo
   forbids interval repair and computes features on clean runs only. This
   layer is where that invariant earns its keep.

## Step 0 — Preflight

1. Work in `AvatarX/code/afib`; `pytest -q` green or STOP. Spec is authority;
   append `## B.x v0.8 changelog`.
2. **This is a refactor of load-bearing code.** `head_afib`, `head_flutter`,
   and `head_regularity` outputs must remain bit-identical on the regression
   corpus under the existing beat-detector configuration; any behavior change
   must be opt-in via config and separately justified by Task 6 evidence.
3. Standing invariants hold; the two that govern this layer: **no interval
   repair** (never interpolate across a suspected missed/false beat — break
   the run instead) and **clean runs only** downstream.
4. Small commits; suite green at every commit.

## Track invariants (each becomes a test)

T-a. **Every beat carries provenance**: fiducial type, source channel
     (rPPG-ROI-fused / rBCG ablation), sub-frame refinement applied, local
     SNR, and a timing-uncertainty estimate in milliseconds. A beat without
     an uncertainty estimate is a bug.
T-b. **No repair, ever** — no interpolation, no "obviously missed beat"
     insertion, no outlier smoothing. Suspicion breaks the run.
T-c. **Per-beat truth, not aggregate truth.** Any accuracy claim must be
     reported per beat and per interval. Reporting only HR-level agreement
     is a bug: 30-second averaging hides per-beat timing failure entirely
     (an HR MAE near 1 bpm coexists comfortably with unusable IBIs).
T-d. **The physiological floor is stated with every accuracy number.** IBI
     error vs ECG R-R is reported alongside the estimated PAT/PEP jitter
     contribution, so no one mistakes a physiological floor for an
     engineering defect — or claims to have beaten it.

## Ordered tasks (tests first, then code)

### Task 1 — Fiducial selection, decided by measurement

The timestamp point on each pulse is an engineering choice with measurable
consequences. Implement and compare, on identical data: **pulse foot/onset**
(intersecting-tangent and minimum methods), **maximum upslope** (first-
derivative peak), **systolic peak**, and **second-derivative onset**.
For each: timing jitter vs the ECG R-peak reference, robustness across SQI
grades, fps (30/60), Fitzpatrick group, and motion condition.
**Gate I0 requires the shipped fiducial to be the empirical winner, with the
comparison table published** — convention is not a justification (the peak
is the intuitive choice and is typically the noisiest, because a rounded
maximum localizes poorly).

### Task 2 — Sub-frame refinement + `BeatLattice v2`

Sub-frame estimators (parabolic/spline interpolation around the fiducial;
matched-filter or template cross-correlation on the beat neighborhood),
each emitting a refinement-uncertainty term. `BeatLattice v2` schema:
per-beat `{t_seconds, fiducial_type, sigma_ms, snr_local, source, run_id}`
plus run segmentation, with a v1 migration shim. Measure interpolation gain
**as a function of SQI and fps** (Task 6 reports it); if gain is negative at
low SQI, disable refinement there by config.

### Task 3 — Multi-ROI timing fusion (and its hidden bias)

Facial regions do not receive the pulse simultaneously — there is a real
spatial arrival gradient across the face. Naively averaging beat times
across ROIs can therefore *add* jitter and drift with ROI availability.
Implement and compare: single best-SNR ROI; SNR-weighted fusion with
per-ROI arrival-offset calibration (estimate each ROI's stable offset, then
fuse); and consensus voting for beat *existence* with timing taken from the
best-SNR ROI. Report which minimizes IBI jitter and — critically — which is
stable when ROI availability changes mid-scan (a common real-world event).

### Task 4 — Beat detection accuracy harness

`evaluation/beat_metrics.py` extended to report, against ECG R-peaks with
verified <10 ms sync: **beat sensitivity**, **beat precision/PPV**, missed-
beat rate, false-beat rate, and the full **timing-error distribution**
(median, IQR, 95th percentile, and the tail — not just MAE), plus
**coverage** (fraction of true beats with a confident timestamp) and
Bland–Altman on intervals. Everything stratified by SQI grade, fps,
Fitzpatrick group, motion condition, and rate band. Add the AF-specific
stress case: detection performance *during irregular rhythm*, where beat
amplitude varies markedly and pulse deficit can drop beats entirely.

### Task 5 — Error-budget model (the deliverable engineering will actually use)

`research/timing/error_budget.py`: decompose measured IBI error into
(a) frame-rate quantization, (b) sub-frame refinement residual, (c) SNR-
driven fiducial jitter, (d) beat-detection errors, (e) estimated
physiological PAT/PEP jitter, and (f) unexplained residual. Validate the
model by prediction: it must forecast measured error within tolerance across
fps and SQI strata. `cli.py timing-budget <dataset>` renders the table.
**This is the artifact that tells the team which knob matters** — whether
the next win is 60 fps capture, better ROI fusion, a different fiducial, or
lighting guidance — instead of guessing.

### Task 6 — Empirical answers to the four standing engineering questions

Run as pre-registered experiments and publish results:
1. **Does 30 fps + interpolation match 60 fps native?** (paired capture on
   the same subjects/sessions; answer per SQI grade.)
2. **What is our interpolation gain, and where does it go negative?**
3. **Which fusion strategy survives mid-scan ROI dropout?**
4. **How much of our residual error is physiological (irreducible) vs
   engineering (addressable)?** — from the Task-5 budget.

### Task 7 — §I promotion gates (`cli.py gate-status --track timing`)

- **I0** Fiducial choice empirically justified; comparison table published.
- **I1** Error budget published and predictive within tolerance.
- **I2** Per-beat accuracy targets met vs ECG at rest, stratified: e.g.
  IBI MAE ≤ 30 ms overall and ≤ 20 ms at ACCEPT-grade SQI on 60 fps capture
  (thresholds in `configs/gates.yaml`, `REQUIRES_CLINICAL_SIGNOFF`), with
  the physiological floor reported alongside per T-d.
- **I3** Beat sensitivity ≥ and false-beat rate ≤ thresholds, including
  during irregular rhythm.
- **I4** Fairness: timing error, coverage, and false-beat parity across
  Fitzpatrick groups within bounds.
- **I5** Downstream consistency: after any change, `head_afib`/`head_flutter`/
  `head_regularity` equivalence or a documented, evidence-backed improvement.
- **I6** Claim mapping: IBI is **internal substrate**. No user-facing HRV or
  stress number without the separate wellness gate, and any such surface
  must be labeled **pulse-rate variability**, never HRV (T-a/§2 above).

### Task 8 — Docs + changelog

`docs/timing_layer.md`: the four physical facts, the fiducial comparison,
the error budget table, the Task-6 answers, and the MDI hand-off to v0.7
(the regularity track's minimum detectable irregularity is derived from this
layer's noise floor — keep the two documents numerically consistent). Spec
changelog appended.

## Definition of done

Suite green; downstream heads equivalent (or improvements documented with
evidence); `BeatLattice v2` shipped with migration; beat-metrics harness,
error-budget model, and gate scoreboard runnable end-to-end on synthetic +
public fixtures (ECG-Fitness-class data with real ECG is ideal for the
ceiling work); final summary reports the §I scoreboard, the fiducial table,
the error budget, and the four Task-6 answers.

## Escalation rule

If any instruction requires interpolating across suspected missed beats,
reporting IBI accuracy only as aggregate HR agreement, quoting a timing
precision below the measured physiological floor, or shipping an HRV-labeled
consumer metric from this layer: **do not implement; record verbatim under
"requires owner decision — conflicts with §I / T-a–T-d."** Every rhythm
claim AvatarX ever makes is downstream of these milliseconds; the point of
this track is to know exactly how good they are, and to say so.
