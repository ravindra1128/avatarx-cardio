# afib-face-scan

Contactless atrial-fibrillation screening from smartphone facial video.
Reference implementation for the AvatarX technical master plan.

**Status: research pipeline, v0.1 (runnable end-to-end).** Not a medical
device. No clinical claim is made or implied.

> **Every number this repository ships derives from SYNTHETIC data or from
> the E6 public-RR experiment** (real MIMIC PERform AF RR intervals with
> synthetic rPPG degradation). Synthetic-video results are **interface
> proofs** — they demonstrate that the stages compose and the gates fire,
> not that anything works on human faces. No clinical performance is
> claimed or implied anywhere in this repository.

v0.1 builds on the pressure-tested v3 core (41 regression tests, all still
green) and adds: PRBS LED-marker synchronisation, beat-confidence
calibration, real-video ingestion (POS/CHROM on tracked facial ROIs), a
non-periodicity composite SQI, decision logic + CLI, Model A trained on
degraded real RR data, and the evaluation report harness. 139 tests.

---

## Live camera demo (v0.1.1)

```bash
python3 cli.py demo          # opens a browser: camera permission -> live face
                             # detection -> 30 s scan -> processing -> results
```

A local, self-contained web app (stdlib server, no cloud). The **browser**
owns the camera — so the OS permission prompt is real — and streams raw
frames to this process; they are recorded **losslessly** and analysed by
the **same production pipeline** `cli.py process` runs. Nothing is uploaded.
The flow: welcome → camera permission → live framing (face oval, ROI boxes,
lighting/motion/signal guidance) → 30-second scan with countdown ring →
processing → the findings report (the one sanctioned rhythm sentence,
1–5 star confidence, pulse rate, capture caveats, referral line — no
waveform of any kind since v0.3). Graceful
errors for permission denied, no face, movement, poor lighting, and weak
signal. Verified end-to-end through a real headless browser on a fake
camera; run `python3 scripts/selftest_headless.py` (needs Google Chrome).

The live demo uses the **consumer capture profile**: identical to the
research gates except it cannot require the AE/AWB lock a webcam won't
grant, and tolerates 24–30 fps — both recorded truthfully and surfaced as
caveats on the result. Every other gate still fails closed.

## CLI

```bash
python3 cli.py validate  <recording.json>              # capture/sync gates, with reasons
python3 cli.py process   <video> [--manifest rec.json] # -> ScanResult JSON (never a diagnosis)
python3 cli.py evaluate  <dataset_dir>                 # -> Gate-1/1b report, risk-coverage,
                                                       #    no-read parity, leakage audit
python3 cli.py demo --synthetic                        # offline synthetic pipeline demo
```

Exit codes: `0` = ran (a NO_RESULT is a *result*), `2` = invalid input with
reasons on stderr.

`evaluate` expects a directory of triples `<id>.recording.json`,
`<id>.avi`, `<id>.ecg.json` (generate a synthetic one with
`python3 scripts/make_synth_dataset.py <dir>`); it writes
`report/evaluation_report.{json,md}`. Recordings failing the schema gate
appear as **EXCLUDED with reasons** — they are never data points.

Reproduce the shipped artifacts:

```bash
python3 scripts/make_synth_video.py synth_videos     # test videos + truth
python3 scripts/fit_default_calibration.py           # confidence calibration
python3 scripts/e6_degradation.py                    # E6 curve + Model A (downloads MIMIC PERform AF)
```

## Dependencies and licences

| Package | Licence | Role |
|---|---|---|
| numpy | BSD-3-Clause | everywhere |
| scipy | BSD-3-Clause | filters, optimisation, peak finding |
| opencv-python-headless | Apache-2.0 | video decode/encode, tracking fallback |
| mediapipe *(optional)* | Apache-2.0 | face tracking when a local model file is provided |
| pytest *(dev)* | MIT | test runner |

Vendored asset: `capture/models/face_detection_yunet_2023mar.onnx` — the
OpenCV Zoo YuNet face detector (Apache-2.0, 232 KB), used for real-face
tracking + landmarks. The live demo's local server and browser page use
only the Python standard library and the packages above (no web framework).

Notes: mediapipe ≥ 1.0 needs a local `face_landmarker.task` file
(`AVATARX_MEDIAPIPE_MODEL=...`); without one the documented OpenCV
Haar / skin-segmentation fallbacks run and the active tracker is recorded
in provenance. mediapipe transitively installs opencv-contrib-python,
matplotlib and friends (all permissive); none are imported by this code.
Isotonic regression, logistic regression and gradient-boosted trees are
implemented in-repo so the dependency list stays exactly this table.
The MIMIC PERform AF dataset is CC-BY 4.0 (Charlton et al., Zenodo
6807403) and is downloaded to `data_cache/` (gitignored), never
redistributed.

## What is implemented (v0.1)

| Module | Notes |
|---|---|
| `datasets/schema.py` | Canonical schema; capture/sync validity are **hard gates** |
| `datasets/splits.py` | Participant-only stable-hash splitting + leakage guards |
| `datasets/synchronization.py` | PRBS LED-marker sync: sub-frame onsets, offset+drift fit, physics-floored uncertainty |
| `beats/detector.py` | Adaptive peaks, sub-sample interpolation, two-pass multi-ROI consensus (v0.1: measured fixes — see below) |
| `beats/confidence.py` | Isotonic confidence calibration, versioned artifact -> `ScanResult.calibration_version` |
| `beats/ibi.py` | Clean-run extraction (3 error channels), **no interval repair** |
| `features/rhythm.py` | Run-feature engine, bounded-stats-first, sample-size floors |
| `capture/` | Video reader (measured fps/codec/bitrate/lux proxy), face tracking chain, streaming ingest |
| `preprocessing/roi.py` | 4 separate facial ROIs -> mean RGB traces (fusion is beat-level only) |
| `rppg/pos.py`, `rppg/chrom.py` | Classical extractors, physiological band, peak-up orientation |
| `rppg/signal_quality.py` | Composite SQI; **anti-periodicity test is a permanent gate** |
| `inference/` | Decision logic (gates first, interim rules or Model A), pipeline with full provenance |
| `models/baseline.py` | In-house LR + GBT, E6 degradation, participant-level CV |
| `evaluation/` | Beat/IBI metrics vs ECG, AF metrics, gate report harness |

## E6 headline (real RR data, synthetic degradation)

Participant-level 5-fold CV on 455×90 s windows from 35 subjects
(19 AF / 16 non-AF), cluster-bootstrap CIs; production feature path
(`clean_runs` in the loop). At the moderate setting (σ = 15 ms timing
noise, pulse-deficit dropout p = 0.5, 3% false insertions of which HALF
survive the confidence threshold — the measured double-detection class,
so the false-beat axis genuinely corrupts intervals):

- logistic regression AUC **0.983** [0.949, 1.000]
- gradient-boosted trees AUC **0.980** [0.946, 1.000]

Gate (> 0.90): **PASS**. Full grid: `models/e6_degradation_curve.csv`.
Observed: raising the deficit rate slightly *raises* AUC — deficit dropout
fires only on short RRs, so it is itself an AF signature. These are
classifier-on-degraded-real-RR numbers, **not** camera numbers.

## The four rules this codebase enforces in code, not documentation

1. **Split by participant, never by window or recording.**
   `assert_no_leakage()` raises `LeakageError` on cross-split participants,
   non-site-disjoint external tests, duplicate content, and a test split
   with no AF or no hard negatives.

2. **Validity is a gate, not a field.**
   CRF > 18 / < 100 lux / unlocked exposure / sync worse than 5 ms →
   **invalid**, not weak. Unknown compression is indistinguishable from
   fatal compression and fails closed.

3. **The quality index must not measure periodicity.**
   AF is aperiodic; an autocorrelation SQI blocks exactly the patients
   being screened. The permanent anti-periodicity test asserts
   SQI(clean AF) ≥ SQI(clean sinus) − 0.05.

4. **Never emit a diagnosis.**
   `ScanResult.user_facing_text()` is the only sanctioned output path;
   the CLI prints it verbatim and the tests assert it.

## v0.8 — the Research & Investigation Report

One document, for investigators only, that carries every gated head's
raw output for one scan — rhythm regularity, the atrial-flutter pattern
flag, arterial stiffness, vasomotor reactivity and cardiorespiratory
fitness — each beside its track's **live gate status**, under one
watermark on every page and section. It is produced only by
`cli.py research-report`, written under a gitignored runs root, and
never becomes part of the Cardiac Rhythm Scan Report: nothing under
`research/` is reachable from `app/` or `inference/`, and every
participant-surface fence is unchanged. A track a single scan cannot
serve (fitness needs the three-phase session; tone needs timed
provocation marks) is listed as not applicable with what it needs.
The same five tracks also appear on the app's results screen, in the
RESEARCH / DEBUG METRICS panel, by default — screen-only, never
printed, in its own payload key, with each track beside its live gate
status. A red gate does not withhold a value; it is printed next to it.
`AVATARX_RESEARCH_TRACKS=0` turns the block off for a participant-facing
session. `docs/investigation_report.md` is the authority; spec B.24
records both owner instructions verbatim and the readings taken.

## v0.7 — the rhythm-regularity substrate track, gated (§R R0–R5)

"Is this pulse regular or irregular, and how confident are we?" is the
question every rhythm head answers privately before answering its own.
v0.7 makes it **one representation**: `features/regularity.py` is the
only place interval statistics are computed (dispersion, distribution,
structure, confidence, and a median-based `irregularity_index` with a
bootstrap CI), and the AFib, flutter, irregularity and rate heads are
readers of it. The rewiring is proven safe by a golden corpus on which
`head_afib`'s outputs are **bit-identical** before and after — the
equivalence test is the definition of a safe refactor — and an AST audit
fails the suite if a second implementation ever appears.

Its ground truth is free: the reference ECG's R-R series goes through the
*same* code, so the label the camera is judged against is the label it
is trying to reproduce (the ceiling test, R0). Its noise floor is a
**published product parameter**: the frame-quantization budget is
computed in code (13.6 ms RMS per interval at 30 fps, 6.8 at 60), the
sub-frame interpolation gain is *measured* on metronomic references at
every floor run, and the minimum detectable irregularity is tabulated
per fps / SQI grade / skin tone. Its dominant false positive is healthy
people breathing, so every irregular read carries a
`benign_pattern_evidence` (respiration-coupled / ectopy-pattern /
chaotic / indeterminate) and gate R2 demands specificity ≥ 0.90 in each
age band. Measured before believed: the median index is robust to
*sparse* beat errors by construction while RMSSD is wrecked by one
missed beat — so the class is decided on the index and the clean-run
architecture is what keeps every dispersion statistic honest.

Measured on the synthetic cohort through the production video path: camera-vs-ECG κ 0.92, index r 0.98; on MIMIC PERform contact PPG (a public surrogate that can never open a gate): κ 0.87, r 0.76 across 35 subjects. MDI characterized in 11 of 12 (fps × SQI × skin-tone) cells; the rest are *not characterized*, which is a red gate, not a zero.

The regulatory anchor is the FDA irregular-rhythm-notification precedent
(Apple DEN180042 → 21 CFR 870.2790 / QDB; Fitbit K212372; Samsung
K230292) — all contact PPG; there is no contactless predicate.
`head_regularity` renders nothing until every §R gate is green with
owner + clinical signoff, never names a rhythm, and never escalates.
`cli.py regularity-floor`, `evaluate-regularity`, `gate-status --track
regularity`; `docs/regularity_track.md` is the authority.

## v0.6 — the atrial-flutter research track, gated (§F F0–F5)

Flutter is the arrhythmia the AFib detector is **structurally blind
to**: fixed AV conduction of a ~300/min atrial circuit produces a
*metronomically regular* pulse (2:1 ≈ 150, 3:1 ≈ 100, 4:1 ≈ 75), which
an irregularity-based detector reads as normal sinus rhythm. It also
cannot be **named** from a pulse — flutter, SVT and sinus tachycardia
at the same rate are the same measurement, and separating them needs F
waves on an ECG. So this track does not build a flutter detector. It
builds a **regular-tachyarrhythmia pattern flag** that routes to an
ECG, plus the serial-scan signature (rates at integer-ratio steps of one
latent atrial rate) that is flutter's most distinctive camera-observable
trait.

Two things were measured before being believed: **RSA coupling is the
discriminator** (at matched rate *and* matched dispersion, a metronomic
clip and a sinus tachycardia had tachogram respiratory-band fractions
of 0.03 vs 0.91 — so the head abstains when no respiration channel
exists rather than guessing), and **the dispersion floor is the camera**
(a perfectly metronomic clip reads RMSSD 15.0 ms at 30 fps, 8.2 ms at
60 fps — 0.45 of a frame period, masking every physiological floor at
consumer frame rates). The known-miss registry ships as
`docs/flutter_limitations.md`: slow fixed-block flutter at ~75 bpm is a
documented permanent miss, quantified by gate F2 rather than asserted.
`head_flutter` renders nothing until every §F gate is green with owner
+ clinical signoff. `cli.py evaluate-flutter`, `gate-status --track
flutter`; `docs/flutter_track.md` is the authority.

## v0.5 — the vasomotor-reactivity research track, gated (§W W0–W5)

An attempt to measure vasomotor **reactivity** — how the small vessels
respond to a controlled provocation — from facial pulse amplitude and
morphology, validated within-subject against a contact perfusion-index
reference. Tone is dynamic and optics-confounded, so the track's
signature defenses are structural: **no absolute cross-session tone
value exists anywhere** (every metric is a within-session delta, W-c),
and every candidate feature must pass an **optics placebo** — a
null_optics arm where the rig varies lamps/exposure while physiology
rests (a feature that responds is dropped by config, W1) — plus a
null_rest arm defining natural drift, and a baseline battery proving
the response is not secretly the HR/breathing response every
provocation also causes (W2). `head_vasotone` renders nothing until
every §W gate is green with owner + clinical signoff; the first
permissible surface is a wellness reactivity trend vs your own
baseline, explicitly non-medical. `cli.py evaluate-vasotone`,
`gate-status --track vasotone`; `docs/vasotone_track.md` is the
authority. Scoreboard at ship: all six gates red on synthetic
machinery evidence, promotion BLOCKED — the intended state until
provocation data exists.

## v0.4 — the arterial-stiffness research track, gated (vascular V0–V5)

An attempt to estimate large-artery stiffness (reference: carotid-femoral
PWV by tonometry) from facial pulse-wave **morphology** — behind six
pre-registered gates in the `vascular:` block of `configs/gates.yaml`,
defended against the two failure modes that made "vascular age from a
camera" a marketing swamp: a **signal-fidelity study runs before any
model** (V0: per-feature ICC vs a simultaneous contact-PPG reference;
survivors chosen by config, never by hand), and every evaluation must
beat an **age+sex+brachial-BP baseline** on unseen participants (V1, the
age-shortcut defense — B1–B5 baselines are mandatory in every report).
`head_vascular` is research-flagged: no user-facing output of any kind —
no number, score, trend, or color — until every gate is green with owner
+ clinical signoff, and the first permissible surface is a longitudinal
wellness trend, not an absolute claim. `cli.py vascular-fidelity`,
`evaluate-vascular`, `gate-status --track vascular`;
`docs/vascular_track.md` is the authority. Scoreboard at ship: all six
gates red on synthetic machinery evidence, promotion BLOCKED — the
intended state until paired clinical data exists.

## v0.4 — the cardiorespiratory recovery + fitness track, gated (§V)

A three-phase guided session — safety screen → 60 s rest scan → guided
activity (audio cadence, live rep counter, NO physiology UI) → sit-still
transition (≤ 5 s target, 10 s hard timeout) → recovery scan → findings
report — with a promotion gate between any "fitness" inference and the
user (docs/vo2_crf_track.md):

- **What renders while §V is red: measurements only.** Resting HR and
  breathing rate, end-exercise HR proxy, HRR30/60/120, recovery slope,
  compliance caveats, stars. **No fitness category, no percentile, and
  an exact VO2 max number never renders in any version.** No face scan
  measures oxygen uptake — VO2 max lives in stroke volume and oxygen
  extraction.
- **Structural rules, in code:** HR is never estimated during movement
  (the activity phase feeds a workload verifier that reduces each frame
  to one motion float — the pulse path cannot see it); body weight and
  height are user-entered, never inferred from the face; the safety
  screen and the compliance contract (±10% compliant / 10–20% repeat /
  >20% or transition >10 s no-result, vitals still shown) fail closed.
- **The recovery tracker** (`features/recovery.py`) is not the rhythm
  path: short-window trimmed-median HR with a |dHR/dt| ≤ 3 bpm/s slew
  bound, monotone trend, back-extrapolated end-exercise proxy; verified
  to ≤ 2 bpm HR(t) MAE and ≤ 3 bpm HRR60 error on decay fixtures through
  the production pipeline, degradations included.
- **§V gates (configs/gates.yaml `vo2:`, `REQUIRES_CLINICAL_SIGNOFF`):**
  V1 HRR60 LoA ≤ ±5 bpm per Monk band (±8 kill rule); V2 retest ≤ 4 bpm
  + ≥ 85% accepted + no-read parity; V3 (pivotal) ΔSEE ≥ 0.5 mL/kg/min
  over demographics+activity with CI excluding zero AND beat the
  static-face-image negative control — fail means `head_fitness` is
  never promoted; V4 subgroup honesty (beta-blocker users are
  hard-routed to trend-only in code today); V5 trend validity before any
  magnitude claim. `cli.py promote` refuses on any red;
  `cli.py gate-status --track vo2` is the scoreboard (all red at ship).
- **Heads:** `head_recovery` (MEASURED), `head_fitness` +
  `head_trend` (INFERRED_FITNESS — a new measurement class renderable
  only when §V is green AND signed; the render path exists and is tested
  unreachable). Within-person trend is the best-evidenced use
  (ΔHRR↔ΔVO2peak |r|≈0.87 in rehab), so the local versioned trend store
  ships now, direction-only.
- **Harness:** `cli.py session` (offline three-phase),
  `cli.py evaluate-fitness` (CPET labels, SEE/LoA/category + the
  mandatory baseline ladder on identical participant splits, shuffled-
  workload and transition-sweep falsification probes),
  `AVATARX_THREE_PHASE=1 python3 cli.py demo` (live flow;
  `scripts/selftest_headless.py --session` proves it headless).

## v0.3 — the ECG-reconstruction research track, gated (§G)

The pre-registered empirical question "can an ECG reconstructed from a
face scan reach clinical fidelity?" — built as machinery with the burden
of proof on the reconstruction (docs/reconstruction_track.md):

- **What the user sees:** findings only. The report defaults to
  `report.mode: findings_only` — sanctioned sentence, stars, pulse rate,
  caveats, referral; **zero waveform elements** (no pulse waveform per
  owner requirement, no synthetic ECG per §G). `mode: full` restores the
  v0.2.1 strips.
- **What trains anyway:** `research/ecg_reconstruction/` — quarantined
  (import-audit: nothing in `app/` or `inference/` can reach it),
  watermarked `SYNTHETIC ECG — RESEARCH ARTIFACT — NOT A MEASUREMENT`,
  writing only to `research/runs/`. Architecture registry (config-
  swappable) over the v0.2 lab decoder; every training run
  auto-evaluates all five §G gates on a frozen participant-disjoint
  held-out split and appends to the scoreboard.
- **The gates (configs/gates.yaml, `REQUIRES_CLINICAL_SIGNOFF`):**
  G1 beat the identity-template baseline on every morphology metric;
  G2 QT/PR/QRS MAE ≤ 20/20/15 ms with CIs AND beat an RR-only QT
  regression; G3 blinded cardiologist reads ≥80%/80%; G4 no withheld-
  class confabulation; G5 detection non-inferior to the measured path.
  `cli.py promote` refuses while anything is red.
- **Scoreboard at ship (honest reds), MIMIC PERform 25/10 disjoint:**
  identity template wins every interval metric (QT 32 vs 49 ms, PR 12
  vs 56, QRS 8 vs 20); the decoder loses even to the RR-only QT
  baseline; withheld-AF P-hallucination excess 0.42; detection AUC
  deficit 0.000 (timing survives reconstruction — morphology doesn't).
  `python3 cli.py gate-status` is the live authority.
- **The loop:** `cli.py train configs/train_reconstruction.yaml` ·
  `cli.py reconstruct <video> --manifest rec.json` · `cli.py
  gate-status`. The flywheel: `ingest-reference` now carries adjudicated
  morphology (conduction pattern + measured PR/QRS/QT per segment);
  registered facial datasets with `reference_dir` manifests feed
  `source: facial` training — the only evidence domain that can ever
  open a gate.
- New research stub `head_irregularity` (rhythm-agnostic flag, silent)
  joins `flutter_suspicion`; the afib head is bit-for-bit unchanged.
  (v0.6: the `flutter_suspicion` stub was superseded and removed — see
  the §F track above.)

## v0.2.1 — the Cardiac Rhythm Scan Report

The results surface is now ONE unified, printable, clinician-style
document (`app/report_render.py`, pure stdlib): header block,
MEASUREMENTS and FINDINGS boxes, three 10-second strips of the measured
facial pulse waveform on a neutral gray time grid — each strip labeled
**"FACIAL PULSE WAVEFORM (rPPG) — NOT AN ECG"** — per-beat confidence
shading, a flag-controlled beat-interval trend, the sanctioned rhythm
sentence, stars, and the verbatim referral line. Familiarity comes from
the layout; the content never impersonates an ECG (no lead labels, no
mV/mm-per-s, no ECG-paper styling, no generated waveform — enforced by
`tests/test_report_forbidden.py`). NO_RESULT renders the same full page
with reasons. Export: `python3 cli.py process <video> --report out.html`;
the browser's Print gives the PDF.

<!-- screenshot placeholder: docs/img/cardiac_rhythm_scan_report.png -->

## v0.2 — the Inferred-ECG platform (M1-M5)

**"Inferred ECG" = the rhythm conclusions a clinician would draw from an
ECG, inferred from measured optical signals — NEVER a generated ECG
waveform.** v0.2 turns the v0.1 pipeline into a layered platform:
endpoint-head registry (`heads/`; afib, rate_flags, rhythm_map; new
capabilities are plug-ins over the versioned BeatLattice, never new
pipelines), the **Rhythm Map** consumer surface (measured pulse waveform
with confidence-shaded beat ticks, tachogram, Poincaré — embedded in the
demo results page), ScanResult v2 with measurement-class labeling on
every field, a data engine (campaign quotas, paired-ECG reference
ingestion with fail-closed sync, content-addressed dataset registry,
participant+session-disjoint splits), a training engine (deterministic
runs, mandatory baselines incl. memorization and metadata-leak
detectors, gated promotion), and a QUARANTINED falsification lab whose
measured MIMIC result is the standing answer to "why not show an ECG?":
the decoder wins correlation and loses the interval test 5-8x while
hallucinating P waves into AF (docs/inferred_ecg_falsification.md).
New CLI: `--heads`, `campaign`, `ingest-reference`, `register-dataset`,
`train`, `promote`, `evaluate --candidate`, `falsify`. Spec B.15.

## v0.1.6 — real-world capture hardening

Advisory mode keeps the easy-start flow: once the face, framing and hard camera
floor are usable, the countdown begins. During the scan, capture faults such as
face loss, broken framing, unusable light, motion, frozen frames, bad timestamps,
or unstable tracking can pause and recover. Provisional SNR, coherence and beat
checks remain live guidance but do not prevent the complete recording needed by
the final pipeline; unusable finished signals still return an explicit abstention.
Early finish is rejected server-side. ROI sampling is
robust to one hair/glasses/highlight-contaminated region without using a skin
colour threshold, and fallback tracking no longer has an absolute red-channel
floor. Endpoint biomarkers now require verified multi-region pulse evidence;
unusable signals remain null. Full root-cause and regression matrix:
`docs/REAL_WORLD_HARDENING.md`.

## v0.1.5 — the gate becomes a grade: 1-5 star confidence (owner-directed)

The blocking readiness gate is replaced by a **confidence rating**: in the
default `advisory` mode the scan always starts once the three blocking
checks pass (face found, framing intact, ≥ 15 fps hard floor); every
other check keeps its threshold and hint but feeds a **1-5 star score**
(`inference/confidence_stars.py`) shown live and on the result. **The bar
to start dropped; the bar to call AF did not**: every v0.1.2 AF evidence
gate is byte-identical, any failing AF gate caps the stars at 2, and the
decision downgrades an AF call carrying < 3 stars — a scan below 3 stars
can never return AFIB_SUGGESTIVE (`tests/test_false_positive_guards.py`
passes with zero edits). Even capture-invalid attempts are graded (1
star, factor named) instead of merely refused. `decision.readiness.mode:
blocking` preserves the previous gate. Star meanings and basis:
`docs/READINESS.md`; spec changelog B.14.

## v0.1.4.2 — two-region verification + ROI-integrity framing (real-use fix #2)

A real blocked attempt (screenshot on record) isolated two remaining
over-strict checks. (1) With exactly two strong ROIs — ordinary home
lighting — the ≥3-ROI coherence component reads **0.00 by construction**;
the user's signal had beat timing 27 ms, which is genuine two-region
verification (the real FP scans failed that budget at 36–49 ms). The
coherence floor is now satisfied either by ≥3-ROI coherence **or** by the
AF-grade timing budget (existing config keys, no new thresholds); the
pulse-deficit rule still demands 3-region coherence — absent beats cannot
be cross-verified by two regions. Both recorded FP evidence profiles are
permanent regression tests and still abstain. (2) The 0.70 face-width cap
(no post-scan counterpart) is replaced by **ROI integrity**: framing
vetoes only when a region keeps < 50 % of its nominal pixels. Spec B.13.

## v0.1.4 — readiness recalibrated to the parity rule (real-use fix)

Real use falsified v0.1.3: a scan could not be started in 3+ attempts
under decent conditions. Measured root cause: the gate vetoed any window
containing a timestamp hole — but a 4-frame browser drop burst every 4 s
yields a **full-scan verdict of ACCEPT/SINUS** (the pipeline is
gap-aware by design), while readiness was READY on **0 %** of
evaluations. Three checks had no post-scan counterpart at all (hole
veto, jitter, multi-ROI quota), three demanded the AF-call bars where a
*result* needs the any-class bars, the lux proxy averaged the whole
frame (a lit face in a dark room read "dark"), and the exposure check
failed slow AE ramps the 0.7 Hz high-pass removes anyway. The fix is the
**parity rule**: readiness demands exactly the evidence a RESULT needs —
the any-class gates, read from the same config keys as the decision
(coverage ≥ 0.60 of the window analysable, coherence ≥ 0.20, beat timing
≤ 40 ms, SQI ≥ 0.30) — plus capture floors and live-only checks
(face-region lighting, exposure **steps**, motion, ≥ 24 fps). A check
breaks the 3 s hold only after 2 consecutive failing evaluations. Every
session now writes `readiness_log.jsonl` (per-evaluation checks +
lifecycle events), so a session that never becomes ready is diagnosable.
AF-grade evidence (coherence ≥ 0.35, timing ≤ 30 ms) is still judged —
by the decision, on the full recording, where it accrues. Details and
the measurement table: `docs/READINESS.md`; spec changelog B.12.

## v0.1.3 — pre-scan readiness gate

The scan timer starts only when the live window already satisfies the
evidence the post-scan decision will demand — computed by the **same
function** (`inference/evidence.py`) — held for 3 s: framing,
lighting/exposure, motion, frame-rate/timestamp integrity, per-region
pulse SNR, cross-region coherence, beat-timing precision, preliminary
beats, composite SQI. The timer counts good seconds only (pause/resume;
restart if the budget is exceeded). Basis for every threshold:
`docs/READINESS.md`.

## v0.1.2 — a real false positive, root-caused

A live scan read AFIB_SUGGESTIVE while a smartwatch check did not detect
AF. The lossless recording reproduced it: **detection-error irregularity
on a weak, incoherent signal** (2-ROI chance-coincidence beats, half/double
intervals), not rhythm. The fix is layered: a short-pair false-beat splitter
in `clean_runs` (spec T5), **beat-evidence gates** in the decision (cross-ROI
coherence, interval/coverage sufficiency, harmonic and split burden; the
pulse-deficit rule needs good coherence), and a full **rationale** on every
result (`cli.py process … → debug`, demo Details). Untrustworthy or
ambiguous signals now produce **REPEAT_SCAN / NO_RESULT**, never an AF
alert. Details: spec B.10; tests: `tests/test_false_positive_guards.py`.

> A consumer wearable and this prototype are not interchangeable diagnostic
> references; ECG-confirmed rhythm is the only ground truth for AF
> performance.

## Known measured limitations (v0.1, on the record)

- Broad 2-ROI double-detections of real pulses survive the confidence
  channel (~33–59% excluded); they carry genuine pulse morphology and
  belong to the spec's short-pair splitter task (not in v0.1).
- On synthetic AF video the clean-run RMSSD understates ECG RMSSD by tens
  of ms (video-chain smoothing); AF still classifies correctly, but Gate
  1b beat-level margins are thin at 30 fps.
- Missed-beat flag recall is far below the 0.60 gate **during AF** —
  ratio-based flagging cannot see deficit gaps inside AF irregularity.
- SQI-ordered risk-coverage sheds AF scans first on the synthetic set
  (P8 abstention funnelling, made visible in the evaluate report).

## Gate 1 — run this before optimising any classifier

```python
from evaluation.beat_metrics import match_beats, ibi_agreement, GATE1_CONTROLLED

m50 = match_beats(ecg_rpeaks_s, rppg_peaks_s, tolerance_ms=50)  # PTT auto-removed
ibi = ibi_agreement(ecg_rpeaks_s, rppg_peaks_s, m50)
ok, detail = GATE1_CONTROLLED.evaluate(m50, ibi)
```

Thresholds: IBI MAE ≤ 30 ms, beat F1@50ms ≥ 0.90, missed ≤ 10%, false ≤ 10%,
|RMSSD error| ≤ 20 ms. **If this fails, do not proceed to classifier work.**

## Quickstart

```bash
pip install numpy scipy opencv-python-headless pytest   # mediapipe optional
python3 -m pytest tests/ -q          # 289 passed (+1 conditional skip)
python3 cli.py demo                  # end-to-end synthetic demo
```

See `docs/RUNBOOK.md` for the research-assistant workflow and
`CLAUDE_CODE_SPEC.md` for the full working brief and v0.1 changelog.
