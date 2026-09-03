# CLAUDE_CODE_SPEC.md — AvatarX Face-Scan AFib (v3)

> **You are implementing a contactless atrial-fibrillation screening pipeline
> from smartphone facial video.** This file is the complete working brief:
> constraints, current repo state, contracts, gates, and an ordered task list
> with acceptance criteria. The repository already contains a tested core
> (41 passing tests) — extend it; do not rewrite it. Work tests-first: every
> task's definition of done is a test that fails before your change and
> passes after. If a constraint here conflicts with something you infer from
> the code, THIS FILE WINS; flag the conflict in your summary.
>
> Quickstart: `pip install numpy pytest` ·
> `python3 -m pytest tests/ -q` (expect 41 passed) ·
> `python3 scripts/demo_pipeline.py` (synthetic end-to-end demo).

*This part is duplicated verbatim as `CLAUDE_CODE_SPEC.md` at the repository root. It is written to be handed to a coding agent as the working brief: contracts, invariants, ordered tasks, acceptance criteria. The agent should not need the PDFs.*

## B.0 Mission and hard constraints

**Mission.** Build the measurement-first pipeline for contactless AFib screening from smartphone facial video: capture → rPPG → beats-with-confidence → clean-run intervals → rhythm features → 4-class output with abstention — with every stage gated by tests against ECG ground truth.

**Hard constraints — violating any of these is a bug, not a style choice:**

1. **Never emit a diagnosis.** All user-facing text flows through `ScanResult.user_facing_text()`; the positive path must contain "not a diagnosis" + a clinician escalation; the negative path must contain "cannot rule out". Enforced by `test_user_facing_text_never_diagnoses`.
2. **Contactless inference only.** ECG and contact PPG exist in the collection rig and evaluation harness exclusively.
3. **Split by participant, never by window/recording.** Assignment is a pure function of `(participant_id, seed)` — nothing mutable may enter the hash. `assert_no_leakage()` runs in CI and its report attaches to every evaluation.
4. **Fail closed.** Invalid capture, invalid sync, insufficient clean intervals, NaN in a gate → `NO_RESULT` / gate failure, never a guess. NaN fails every gate by construction.
5. **The production feature path is the gated path.** Any metric used in a gate must be computed by the same code path the product runs (lesson P2).
6. **No demographic inputs to the classifier.** Age/sex/tone are stratification variables for evaluation, never features — shortcut learning on demographics is both a fairness and a regulatory defect.
7. **No interval repair.** Suspected missed/false beats break runs; they are never interpolated over (in AF, the "outlier" is the signal).

## B.1 Repository state (all tests green: 41)

```
afib-face-scan/
├── datasets/
│   ├── schema.py        DONE  canonical schema; capture/sync validity are HARD GATES
│   │                          v3: bpp>=0.12 lossy path · >=10-marker sync · contact-PPG channel
│   ├── splits.py        DONE  v3: identity-only hash (label-evolution-proof) ·
│   │                          split_composition() · leakage guards incl. hard-negative check
│   └── synchronization.py STUB
├── beats/
│   ├── detector.py      DONE  adaptive peaks · sub-sample interpolation · dynamic
│   │                          refractory · two-pass multi-ROI consensus fusion
│   ├── ibi.py           DONE  v3: clean_runs() — 3-channel run splitting
│   │                          (confidence / physiologic range / local-ratio)
│   └── confidence.py    STUB  → T2
├── features/
│   └── rhythm.py        DONE  v3: compute_rhythm_features_from_runs() — bounded-stats-
│                              first · run-wise diffs · longest-run-only sequence features
│                              · dropout_rate as a pulse-deficit feature
├── evaluation/
│   ├── beat_metrics.py  DONE  v3: + production_rmssd_error_ms · beat_confidence_
│   │                          calibration (ECE) · missed_beat_flag_recall
│   ├── afib_metrics.py  DONE  v3: serial_confirmation(fp_persistent_share) ·
│   │                          no_read_report() parity audit · risk-coverage · cluster
│   │                          bootstrap CIs · fairness_gate
│   ├── calibration.py   STUB  → T9
│   └── error_taxonomy.py STUB → T11
├── rppg/   pos.py chrom.py ica.py deep_rppg.py arbitration.py signal_quality.py  STUB
├── capture/  preprocessing/  rbcg/  models/  inference/                          STUB
├── scripts/
│   ├── demo_pipeline.py DONE  end-to-end synthetic demo incl. all v3 paths
│   └── timing_budget.py DONE  quantisation/extractor decomposition
└── tests/
    ├── test_integrity.py       DONE  22 tests
    └── test_pressure_fixes.py  DONE  19 regression tests, one per pressure finding
```

Run: `pip install numpy pytest` → `python3 -m pytest tests/ -q` → `python3 scripts/demo_pipeline.py`.

## B.2 Data schema (stable — do not change without updating every consumer)

`Recording` carries: identifiers (participant/session/site), paths (video, ECG, **optional contact_ppg**), UTC boundaries, `CaptureConfig` (device, resolution, measured fps + jitter, codec, **crf OR bitrate_mbps**, AE/AWB/gain locks, beautification flag, OIS/EIS state, lux, distance, mount), `SyncRecord` (method, offset, uncertainty, **n_marker_events**, end-verified), interval-level `RhythmAnnotation[]` (adjudicated, with ventricular rate and ectopy burden), `ecg_rpeaks_s`, protocol metadata, split, provenance hash. `ScanResult` carries the calibrated probability, 4-class prediction, SQI, usable beats, no-read reasons, and full provenance (model version, commit, calibration version, config hash).

**Validity gates (all raise lists of reasons):**

| Gate | Rule |
|---|---|
| Capture | lossless ∨ CRF ≤ 18 ∨ bpp ≥ 0.12; fps ≥ 30; lux ≥ 100; AE+AWB locked; beautification off |
| Sync | uncertainty ≤ 5 ms ∧ end-verified ∧ (hardware trigger ∨ ≥10 optical marker events); cross-correlation sync rejected as circular |
| Beat-level use | both above ∧ ECG ≥ 250 Hz ∧ adjudicated annotation present |

## B.3 The production inference path (normative)

```
frames ─► face track ─► 4 ROIs ─► per-ROI rPPG ─► per-ROI beat detect
      ─► two-pass consensus fusion (Beat: t, confidence, roi_agreement, sqi, amplitude)
      ─► clean_runs(min_conf = CALIBRATED threshold, ratio 1.75, physiologic [250, 2200] ms)
      ─► compute_rhythm_features_from_runs()      # never the raw series
      ─► classifier (4-class) ─► calibrated probability
      ─► decision logic: SQI/coverage gate → class → serial policy (share-aware)
      ─► SINUS | AFIB_SUGGESTIVE | OTHER_IRREGULAR | HIGH_RATE | NO_RESULT
```

**Feature hierarchy (P1):** primary = bounded/robust (pNN50, pNN20, median |Δ|, irregularity index, dropout_rate); secondary = run-wise dispersion (RMSSD, SDNN, CV, Poincaré); tertiary = longest-run-only sequence features (sample entropy ≥60 contiguous, Shannon ≥50, Markov surprise, TPR). Fragmentation surfaces as NaN + warning; it is never papered over.

## B.4 Gates v2 (supersedes all previous gate tables)

**Gate 1 — beat timing, controlled capture.** IBI MAE ≤ 30 ms · beat F1@50ms ≥ 0.90 · missed ≤ 10% · false ≤ 10% · **production-path RMSSD error ≤ 25 ms on sinus** (`production_rmssd_error_ms`) · **beat-confidence ECE ≤ 0.10** (`beat_confidence_calibration`).
**On failure → do not kill the program (P5). Branch:**

```
Gate 1 pass ──► interval-feature path (explainable; preferred for De Novo)
Gate 1 fail ──► Gate 1W: waveform-domain fallback (Model B′, Sun-style 1D CNN/TCN
                on the extracted rPPG waveform, no beat detection)
                pass = AUC ≥ 0.90 on Tier A controlled, participant-independent
Both fail  ──► program kill is now justified
```

**Gate 1b — during ECG-confirmed AF.** IBI MAE ≤ 40 ms · F1 ≥ 0.85 · **missed-beat flag recall ≥ 0.60** (the algorithm must know its gaps) · once contact PPG exists: missed-beat accounting splits extraction failures from true pulse deficit, and only extraction failures count against the gate.

**Standing gates, every evaluation:** leakage audit passes · no powered Monk band > 10 pts below best on Se or Sp · **no-read parity: no powered band's abstention > 1.5× best AND > +10 pts absolute — including rhythm-conditional (P(no-read|AF) vs P(no-read|non-AF))** · risk-coverage curve reported at 100/90/80/75/60/50% with AF cases retained at each level · every serial-confirmation number quotes its `fp_persistent_share`.

**Ectopy gate (P12 reframe):** Tier-B ectopy-arm Sp ≥ 93% enables the screening claim. 90–93%: screening claim paused, notification-claim path (Apple-precedent shape) and high-prevalence supervised deployments proceed. < 90% after two iterations: contactless *screening* is dead; the program pivots or stops.

## B.5 Ordered task list (dependency order; each task = tests first, then code)

| # | Task | Definition of done |
|---|---|---|
| **T1** | `datasets/synchronization.py`: PRBS LED-marker detector + drift estimator | Synthetic video with known offsets: recovered offset error < 2 ms, drift < 5 ppm; writes `SyncRecord` with `n_marker_events` |
| **T2** | `beats/confidence.py`: calibrate fusion confidence (isotonic over DEV) | Demo detector ECE 0.42 → **≤ 0.10** on held-out synthetic suite; `clean_runs` threshold becomes the calibrated 0.5 |
| **T3** | `rppg/pos.py`, `chrom.py`, `ica.py` (Layer 1, always on) | On the synthetic ROI suite: beat F1 within 0.05 of the shipped detector; stored-output regression test |
| **T4** | `rppg/signal_quality.py`: NSQI + skewness + spectral concentration + cross-ROI coherence | **Anti-periodicity test: clean AF must score ≥ clean sinus − 0.05** (the open-rppg failure, inverted into a test); low-light and motion cases score < 0.3 |
| **T5** | Short-pair false-beat splitter in `beats/ibi.py` (split-signature: two adjacent sub-intervals summing ≈ local median) | Sinus-with-false-beats RMSSD error improves vs v3; `test_clean_runs_do_not_recreate_the_clean_rr_failure` stays green (AF preserved) |
| **T6** | `rppg/deep_rppg.py`: waveform-output backbone wrapper + **fiducial experiment** (waveform peak vs derivative max-slope) | Both fiducials benchmarked on beat metrics; report which wins on IBI MAE — do not assume (P-finding on Diff-model rule) |
| **T7** | `models/baseline.py`: Model A (LR/RF/XGBoost) on run-features | AUC > 0.90 on rPPG-degradation-simulated real RR data (MIMIC PERform AF, CC-BY, 19 AF/16 non-AF) with participant-level CV |
| **T8** | `models/temporal.py`: Model B (IBI-sequence TCN) + Model B′ (waveform 1D CNN — **the Gate-1W fallback**) | Beats Model A on the same protocol or documents why not; B′ evaluated with no beat stage |
| **T9** | `evaluation/calibration.py`: temperature scaling + reliability diagrams | ECE reported; fitted on DEV only (`assert_preprocessing_is_split_safe` wired in) |
| **T10** | `inference/decision_logic.py`: SQI gate → class → share-aware serial policy | Property tests: no path emits a class when coverage < floor; all text via `user_facing_text()` |
| **T11** | `evaluation/error_taxonomy.py`: auto-attribution of FP/FN to {motion, lux, deficit, ectopy, rate-extreme, tone, occlusion, ambiguous} | Runs on demo outputs; report ranks causes by attributable error |
| **T12** | `inference/pipeline.py`: end-to-end orchestration with provenance | `ScanResult` fully populated incl. config hash; demo runs through it |
| **T13** | `capture/` app spec + file-based validator CLI | Rejects every invalid fixture in the test suite with the schema's reasons |
| **T14** | Experiment harnesses E0–E7 as runnable scripts | Each emits a machine-readable result + the gate it informs |

**Experiments (unchanged priorities, two updated):** **E3 spectral-regularisation test remains the highest-value open question** (does SiNC/Contrast-Phys SSL smooth away irregular rhythm? settles a direct conflict between the two prior AvatarX reports). **E2 now includes the fiducial comparison** (T6). E6 (inject measured rPPG noise + deficit dropouts into real AF RR series → classifier degradation curve) now uses `clean_runs` in the loop so the budget reflects the production path.

## B.6 Configuration defaults (single source of truth: `configs/default.yaml`)

```yaml
capture:   {fps_min: 30, fps_target: 60, lux_floor: 100, crf_ceiling: 18,
            bpp_floor: 0.12, ae_awb_gain_locked: true, beautification: off}
sync:      {max_uncertainty_ms: 5.0, min_marker_events: 10}
runs:      {min_conf: CALIBRATED, min_run_beats: 4, missed_ratio: 1.75,
            ibi_physiologic_ms: [250, 2200]}
features:  {sampen_floor: 60, shannon_floor: 50, primary: [pnn50, pnn20,
            median_abs_succ_diff, irregularity_index, dropout_rate]}
gates:
  g1:      {ibi_mae_ms: 30, f1_at_50ms: 0.90, missed: 0.10, false: 0.10,
            production_rmssd_err_ms: 25, confidence_ece: 0.10}
  g1b:     {ibi_mae_ms: 40, f1_at_50ms: 0.85, flag_recall: 0.60}
  g1w:     {auc_tier_a: 0.90}           # waveform fallback branch
  fairness:{max_subgroup_gap: 0.10, noread_max_ratio: 1.5, noread_max_abs: 0.10}
serial:    {rule: [2, 3], fp_persistent_share_planning: 0.65, within_sitting: true}
scan:      {capture_s: 120, ship_s: 90, subwindows: 3}
```

## B.7 What NOT to build (standing)

No end-to-end video→AF as the primary path (shortcut-learning + undiagnosable failures; B′ operates on the extracted waveform, not raw video). No rBCG in inference (ablation E7 only; MIT patents US 11,672,436 / US 10,638,942 unresolved). No foundation-model pretraining on web video (compression has removed the pulse; demonstrated non-convergence). No periodicity-based SQI (empirically blocks AF). No demographic features. No interval interpolation. No wellness-carve-out product strategy (WHOOP "inherent association" precedent). No raw accuracy as a headline metric, ever.

## B.8 v0.1 changelog (appended 2026-08-14; the only sanctioned edit)

Implemented against this spec, tests-first, one commit per task; the 41
v3 tests stayed green throughout (now 136 total).

* **T1** `datasets/synchronization.py`: PRBS chips + sub-frame flash-onset
  recovery from exposure-integrated brightness; offset/drift least squares;
  uncertainty floored at frame/sqrt(12·K). Design constraint discovered and
  encoded: the chip period must be incommensurate with the frame period
  (default 103.7 ms) or quantisation becomes a coherent per-burst bias.
* **T2** `beats/confidence.py`: isotonic (PAV) calibration, content-hash
  version into `ScanResult.calibration_version`; shipped synthetic-fitted
  artifact `inference/confidence_calibration_v01.json`. Two MEASURED
  detector defects fixed in `beats/detector.py` (T2's acceptance is
  unreachable with an inverted ranking): fusion spread now
  size-normalised MAD with agreement^1.5 (was range + sqrt: ranking AUC
  0.32–0.45, inverted → 0.90 sinus / 0.96 AF), and per-ROI peaks carry a
  width-at-half-prominence plausibility factor (noise spikes ~17 ms vs
  ≥67 ms real beats — no amplitude prior, so pulse deficit is not
  penalised). Known limitation on the record: broad 2-ROI
  double-detections are only ~33–59% excluded by confidence; they belong
  to T5's short-pair splitter (not in v0.1).
* **T3** `capture/`, `preprocessing/roi.py`, `rppg/pos.py`, `rppg/chrom.py`,
  `scripts/make_synth_video.py`: streaming ingest with fail-closed ordering
  (capture gate before tracking), tracker chain
  mediapipe(model-if-present)/Haar/skin-segmentation recorded in
  provenance, 4 separate ROIs, windowed POS and CHROM. Synthetic-video
  interface proof: F1@50ms = 1.00 both extractors, 30/60 fps, sinus/AF.
  (`rppg/ica.py` deferred with the rest of spec-T3's scope; v0.1 prompt
  narrowed T3 to POS+CHROM.)
* **T4** `rppg/signal_quality.py` + `configs/default.yaml` (+ in-house
  YAML-subset loader; dependency list stays pinned): five presence-only
  components; permanent anti-periodicity test (clean AF ≥ clean sinus −
  0.05; measured deltas −0.005..−0.034); noise/motion/no-pulse < 0.3.
* **T5** (v0.1 numbering) `inference/decision_logic.py`,
  `inference/pipeline.py`, `cli.py`, `datasets/io.py`: gates first, NaN
  fails closed, interim transparent rules (thresholds in config, measured
  margins sinus 12–24 ms med|Δ| vs AF 96–99 ms); full provenance
  (commit, calibration, config hash); CLI validate/process/evaluate/demo,
  exit 0=ran / 2=bad input; property tests pin the invariants.
* **T6** `models/baseline.py` + `scripts/e6_degradation.py`: in-house LR
  and GBT; E6 on MIMIC PERform AF (CC-BY 4.0, 35 subjects, 455×90 s
  windows), participant-level 5-fold CV, cluster-bootstrap CIs.
  **Gate PASS** at σ=15 ms / deficit 0.5 / 3% false insertions with the
  measured ~50% threshold-survival: LR AUC 0.983 [0.949, 1.000],
  GBT 0.980 [0.946, 1.000]; full grid in `models/e6_degradation_curve.csv`;
  artifact `models/model_a_v01.json` behind `decision.classifier: model_a`.
  Observed: deficit dropout is itself an AF signature (AUC rises with
  deficit), as B.1's design premise predicted.
* **T7** `evaluation/report.py` + `scripts/make_synth_dataset.py`:
  triple-directory harness through `run_with_details` (the production
  path); exclusions with schema reasons (CRF-28 fixture EXCLUDED);
  per-rhythm gate tables + production-RMSSD error + ECE + flag recall;
  risk-coverage with AF retention; no-read parity incl. rhythm; leakage
  audit; serial figures always quote `fp_persistent_share`. Honest
  synthetic-set findings: Gate 1b flag recall ~0.08 vs 0.60 during AF;
  AF clean-run RMSSD understated ~49 ms by the video chain; SQI-ordered
  coverage sheds AF scans first (P8 made visible).
* **T8** README (CLI, licence table, prominent synthetic-data notice),
  `docs/RUNBOOK.md`, this changelog.
* **Post-review fixes** (adversarial multi-agent review of the full v0.1
  diff; 5 findings confirmed, all fixed): (1) E6's injected false beats
  were capped below the confidence threshold — an oracle label that made
  the false-beat axis structurally unable to corrupt an interval; now a
  measured ~50% survive (`FALSE_BEAT_SURVIVAL`), and the E6 numbers above
  are the regenerated honest ones. (2) `evaluation/report.py` mapped ECG
  onto the video clock with offset only; the sanctioned
  `ecg_to_video_clock` now applies the full model incl. drift (50 ppm =
  4.5 ms at t=90 s, the whole sync budget). (3) T1's drift budget held
  only at lucky seeds (measured up to ~23 ppm error at 30 fps): rising-
  only edge estimation let shared level errors become per-burst bias;
  polarity-balanced edge anchoring cancels it, and with an order-7 PRBS
  over the 120 s capture window the worst error across 24 seeds x all
  cells is 3.5 ppm — a multi-seed distributional test now guards this.
  (4) The T5 property sweep never reached the coverage gate (SQI always
  failed first); the sweep now isolates every gate.

## B.9 v0.1.1 changelog — live camera demo (appended 2026-08-15)

Scope change requested by the project owner after v0.1: the `demo` must be a
REAL end-to-end camera experience (permission → live face detection →
30 s scan → processing → results), not a log-only synthetic run. This
DEVIATES from the v0.1 prompt's "No UI beyond the CLI"; recorded here as an
explicit, owner-authorised amendment. The research pipeline, gates and all
v0.1 acceptance criteria are unchanged; the demo is a thin consumer front
end over the SAME production path (`inference.pipeline.run_with_details`).

Added (tests-first; suite 139 → 160):
* `app/scan_engine.py` — PREVIEW→SCANNING→PROCESSING→DONE/FAILED state
  machine: live framing/lighting/motion/pre-scan-SQI feedback, lossless
  FFV1 recording + honest timestamp sidecar, fail-closed live aborts (dark,
  face-lost >2 s, movement, exposure step). Results come ONLY from the
  production pipeline — the engine computes guidance signals, never a
  rhythm result.
* `app/server.py` + `app/static/index.html` — stdlib HTTP server (no web
  framework) and a single-page app: real `getUserMedia` permission, canvas
  frame capture streamed raw (no lossy re-encode in the path), oval framing
  overlay + ROI boxes, 30 s countdown ring, processing, results report
  (verbatim `user_facing_text`, pulse rate, SQI, pulse waveform with beat
  ticks), and graceful error screens for all five failure modes.
* `capture/` — YuNet face detector (vendored OpenCV-zoo ONNX, Apache-2.0)
  with 5 landmarks → landmark-anchored ROIs on real faces; EMA geometry
  smoothing (stability judged on raw obs); timestamp sidecar overrides the
  container clock; camera-reported AE/AWB states recorded.
* **Consumer capture profile** (`capture/ingest.py`) — a documented,
  localized deviation for webcam capture: relaxes ONLY the AE/AWB lock and
  the strict 30 fps floor (to 24), records both truthfully, and returns them
  as caveats that travel with the result. Every other gate still fails
  closed (darkness, no-face, non-finite traces). Research profile unchanged.

Verification of the real user journey (not mocked): `scripts/selftest_
headless.py` drives a REAL headless Chrome with a FAKE CAMERA (Chrome's
`--use-file-for-fake-video-capture`, fed a face+pulse Y4M generated here)
through the ordinary `getUserMedia` client path — no test-source shortcut,
no hard-coded result — and reads the outcome back from the server. Measured:
30.0 fps / 2 ms jitter, YuNet tracking (stability ~0.89), SQI ~0.96;
sinus → SINUS (33–34 beats), AF → AFIB_SUGGESTIVE (45 beats). A too-short
(8 s) scan yields too few intervals to classify reliably — expected, and
the reason the shipped scan is 30 s.

Interim-rule tuning (both placeholder thresholds, replaced by Model A):
OTHER_IRREGULAR raised (med|Δ| 35→45 ms, pNN50 0.25→0.45) and made an AND
(was OR), because at 30 fps the quantisation floor inflates med|Δ| enough
that ordinary resting sinus HRV was mislabelled irregular.

Environment note: the in-app automation panes report the page as `hidden`,
which suspends media playback and throttles timers, so the browser
capture loop could not be driven there; the permission-denied path WAS
verified in-pane, and the full play-through was verified via the headless
harness above. On a user's foreground tab with a real webcam none of these
throttles apply.

## B.10 v0.1.2 changelog — real false positive, root-caused (appended 2026-08-15)

**Incident.** A live scan (v0.1.1 demo, MacBook webcam) read AFIB_SUGGESTIVE;
an immediate Apple Watch check did not detect AF. Treated as a critical
false positive. (Reference note: a consumer wearable and this prototype are
not interchangeable references — ECG-confirmed rhythm remains the only
ground truth for AF performance; the Watch reading is a red flag, not a
label.)

**Reproduction.** The demo keeps every scan losslessly with its capture
clock; both scans re-ran through the production path and reproduced the
call. Evidence (per-stage dump, now `cli.py process … → debug.rationale`):
forehead and nose independently saw a REGULAR ~1090 ms rhythm (~55 bpm);
the cheeks were noise (right cheek SNR 0.1, skewness 0) and "detected"
36–41 spurious peaks; fusion built 22/25 (18/24) beats from 2-ROI chance
coincidences — cross-ROI coherence 0.06 / 0.12; the fused series held 7
half-intervals and 2 double-intervals (the harmonic signature of false and
missed beats); the synthetic-fitted calibrator rated those clusters 0.83;
`clean_runs` had no short-interval channel; composite SQI 0.40 / 0.46 and
coverage 0.50 / 0.60 with 17 / 16 intervals sat at the floors; the interim
AF rule (med|Δ| ≥ 60 ∧ pNN50 ≥ 0.4) fired on detection-error irregularity.

**Root cause (class).** Rhythm features cannot distinguish arrhythmia from
detection error; only beat-level EVIDENCE can, and no gate demanded it. The
"strong artifact veto" (cross-ROI agreement, B.3) was averaged into a
composite that passed.

**Fix — layered, not a threshold nudge (tests first, +13; suite 175):**
1. `beats/ibi.py` — 4th splitting channel: the SHORT-PAIR false-beat
   signature (T5). No repair. `RunSet` now reports missed / false-pair
   split counts and `split_fraction` (detection-error burden). The
   AF-preservation regression stays green.
2. `inference/decision_logic.py` — beat-EVIDENCE gates after the quality
   gates. Any class: coherence ≥ 0.20, coverage ≥ 0.60, ≥ 15 clean
   intervals, split burden ≤ 0.30. AF call additionally: ≥ 20 intervals,
   coherence ≥ 0.35, harmonic fraction ≤ 0.20, split burden ≤ 0.15; the
   pulse-deficit rule counts only under coherence ≥ 0.50. Any failure →
   REPEAT_SCAN — never AFIB, and deliberately not OTHER_IRREGULAR (artifact
   vs arrhythmia is undecidable from such data). Gates are evaluated in full
   so the rationale names EVERY failing condition. Thresholds live in
   `configs/default.yaml → decision.evidence`.
3. Explainability — `decide_with_rationale` returns gates (value /
   threshold / verdict), the features the rule saw, the beat evidence and
   the rule that fired; attached to pipeline details, CLI `debug`, and the
   demo's Details panel. Every result is traceable and reproducible from
   the kept recording.
4. Demo — preview readiness requires the same coherence the decision will
   demand (a scan the pipeline is bound to reject is not started; the user
   is told to fix light/stillness first); the results screen shows why no
   result was given; the reference disclaimer is on the results screen.
5. `capture/video_reader.py` — measured fps rounded to 0.01 (a 30 fps
   camera measured 29.9999 was being flagged "below 30").

**Outcome.** Both real recordings → REPEAT_SCAN with the mechanism named
(coverage 0.44 · coherence 0.06 · 14 intervals; coherence 0.12).
Real-browser headless journeys unchanged: sinus → SINUS (31 verified
beats, coherence 1.0), AF → AFIB_SUGGESTIVE (40, 0.99). Adversarial suites:
regular rhythm + false/missed beats at low coherence → REPEAT_SCAN across
seeds; low coherence alone blocks any class; heavy split burden abstains
even when the residue looks regular; genuine coherent AF still reads AF.

**Known residual (on the record).** The confidence calibration is fitted
on synthetic data and is over-confident on real webcam beats; it is no
longer load-bearing for the decision (the evidence gates are), but a
calibration refit on real DEV recordings with ECG remains required before
any performance claim. Under this webcam/light setup the demo will now
frequently say "pulse not seen clearly across your face" and refuse to
start — that is the intended, honest behaviour.

**B.10 addendum — audit-confirmed additions (same day).** An adversarial
5-lens / 47-agent audit of the production path (with the real recordings
available for reproduction) confirmed 16 findings. Beyond the layers above,
these were fixed and tested:
* `beats/detector.py` — the detector committed to the FIRST supra-threshold
  local maximum, and the refractory test then suppressed the true, higher
  systolic peak 100-400 ms later (mis-timed beats; on weak waveforms, half/
  double intervals). Inside the refractory window a higher AND clearly more
  prominent (>1.5×) local maximum now replaces the earlier one — prominence
  guard so AF's small-but-prominent short-coupled beats are never replaced
  (the permanent anti-periodicity test caught the naive version). Measured
  on the real scans: strong-ROI IBI IQR 199→123, 133→73, 350→195 ms.
* `rppg/signal_quality.py` — the SNR/concentration energy fractions used
  the 0.7-4 Hz filter band and penalised the harmonics of fast beats (rate
  bias, masked until the detector fix raised sinus coherence). Energy
  fractions now use `sqi.energy_band_hz` [0.7, 6.0]; anti-periodicity margin
  −0.08 → −0.02, white noise 0.22 (< 0.3). The extractor band is unchanged.
* `rppg/pos.py::orient_rois_consistently` — per-ROI polarity was decided
  from each ROI's own (noise-level) skewness, so a weak ROI could be
  inverted half a period from the good ones; all ROIs now share the
  polarity of the most clearly pulse-shaped ROI.
* `inference/pipeline.py` — beat times were on the FRAME-INDEX clock; the
  audit reproduced a second manufactured-AF path (8 dropped 0.5 s browser
  batches on a perfectly regular pulse → AFIB_SUGGESTIVE in 8/10 seeds,
  measured fps still passing). Traces are now SPLIT at capture-clock gaps
  (`capture_segments`), extracted/detected per segment, and beat times are
  mapped onto the sidecar timestamps; gaps are reported in the evidence.
  The browser now reports dropped frames with every batch; they are written
  to the sidecar and surfaced as a result caveat.
* **Timing-precision evidence (Gate-1 proxy without ECG)** —
  `timing_precision_from_trains`: best ROI-pair median |Δt| of matched
  beats. Measured: references 1-5 ms (matched 1.00); the real scans 36-49 ms
  (0.63-0.75) — jitter that alone trips the AF rule with no harmonics. Any
  rhythm statement now needs ≤ 40 ms; an AF call ≤ 30 ms with ≥ 75% matched.
* Rejected by verifiers (not changed): fusion tolerance (widening it would
  inflate chance coincidences), coverage definition, class ordering, and the
  claim that readiness ignores coherence (already fixed above).
* Deferred, on the record: the spec's serial-confirmation policy (2-of-3
  within a sitting) is still not wired into the demo — implementing it needs
  a sanctioned "confirmation scan" sentence, which is a `user_facing_text`
  wording change outside this task's authority; and Model A on the real
  scans returns 0.89/0.995 (feature vector far outside its training support,
  no OOD check) — it remains behind the same evidence gates and is not the
  demo default.

Final state: 179+ tests; both real recordings → REPEAT_SCAN with the
mechanism named (scan 1: coverage 0.20 / coherence 0.13 / 7 intervals /
timing 49 ms; scan 2: coherence 0.14, otherwise clean 28-beat run at
~56 bpm — a regular rhythm the pipeline honestly declines to certify);
real-browser headless journeys unchanged (sinus → SINUS 33 beats; AF →
AFIB_SUGGESTIVE 39 beats).

## B.11 v0.1.3 changelog — pre-scan readiness gate (appended 2026-08-17)

**Problem.** After v0.1.2's evidence gates, two live scans completed and
both returned no result: the pre-scan check evaluated a 6-s window's SQI
and coherence, while the post-scan decision demanded verified beats
(coherence, timing precision, coverage, interval count) on 30 s. The user
paid 30–60 s to learn the signal was inadequate.

**Fix — the pre-scan gate IS the post-scan gate.** `inference/evidence.py`
holds ONE function (`window_evidence`) that the production pipeline and the
live engine both call; readiness (`readiness_from_evidence`) is a checklist
against the SAME thresholds the decision applies (`decision.evidence`) plus
the live-only checks (`decision.readiness`): face/framing, lighting (lux ≥
100, face luma ≥ 60), exposure stability, motion/tracking, frame rate ≥ 24,
no timestamp holes / jitter < 8 ms, ≥ 2 ROIs with in-band SNR ≥ +3 dB,
cross-ROI coherence ≥ 0.35, beat-timing precision ≤ 30 ms with ≥ 75 %
matched, ≥ 4 fused beats with ≥ 25 % multi-ROI, composite SQI ≥ 0.30 — all
held for 3 s on an 8-s rolling window. Basis for every number:
docs/READINESS.md. The server refuses `scan/start` when not ready.

**During the scan.** The timer counts GOOD seconds only: when the checklist
fails, recording pauses (an honest hole in the capture clock — the pipeline
is gap-aware, coverage is judged against captured time, fps is the median
period), the ring shows ❚❚ with the fix hint, and it resumes after 1 s of
good signal; > 20 s or > 4 pauses ⇒ restart at readiness with the reason.
Sustained darkness / violent motion still abort. Pauses are recorded in the
sidecar and shown on the results screen.

**Anti-periodicity, applied to readiness.** A first version folded the T5
split fraction into a readiness check; on an 8-s AF window (~12 intervals)
one genuine short pair reads 17 % and would have gated AF users out.
Removed; `test_readiness_is_rhythm_neutral…` now asserts READY-rate parity
AF vs sinus (measured 100 % / 100 %). Readiness judges beat PRESENCE and
measurement integrity only; rhythm burden is judged on the whole scan.

**Verification.** Real headless-browser journeys through the real gate:
sinus → READY → SINUS (32 beats); AF → READY → AFIB_SUGGESTIVE (44 beats).
Engine tests: hold before READY, actionable hints in root-cause order,
noise refuses to start, pause/resume with the pause as a capture hole,
restart on budget. Suite 189+.

## B.12 v0.1.4 changelog — readiness gate recalibrated to the parity rule (appended 2026-08-24)

**Problem (real-use falsification of B.11).** The owner could not start a
scan in 3+ attempts under decent conditions (2026-08-20/21). The v0.1.3
gate also left no telemetry — the session directories were empty, so the
failure was initially undiagnosable. Root cause, measured by replaying
degraded variants through rolling 8-s readiness windows:

1. **Hole veto vs. coverage (the decisive mechanism).** A 4-frame drop
   burst every 4 s — routine browser behaviour under transient load —
   made the `timestamps` check fail on **0 %→100 % of evaluations
   (READY 0 %, longest streak 0.0 s)** while the full-scan verdict on
   the SAME stream was **ACCEPT/SINUS**: the gate vetoed what the
   gap-aware pipeline reads by design. The headless verification in
   B.11 could not see this: the fake-camera clock never drops frames.
2. **Over-demanding thresholds.** Readiness required the AF-call bars
   (coherence ≥ 0.35, timing ≤ 30 ms with ≥ 75 % matched, ≥ 25 %
   multi-ROI beats) per 8-s window, though the decision applies those
   only to the AF call on the full recording; a RESULT needs the
   any-class gates (coherence ≥ 0.20, timing ≤ 40 ms, coverage ≥ 0.60).
   Jitter (< 8 ms) and the multi-ROI quota have **no post-scan
   counterpart at all**.
3. **Whole-frame lux proxy.** Lighting failed iff whole-frame luma <
   32/255 — a well-lit face against a dark room reads "dark" (two of
   the three failed attempts were at midnight).
4. **Exposure drift ≠ step.** The first-half/second-half luma test
   failed a slow 1.5 %/s AE ramp — a < 0.1 Hz trend the 0.7 Hz
   high-pass removes — for the whole 8-s window.
5. **Hold fragility.** 13 conjunctive checks × 7 consecutive
   evaluations: any single-eval flicker anywhere reset the 3-s hold.

**Fix — the parity rule, stated and enforced.** Readiness may demand
exactly the evidence a RESULT needs — the any-class gates, read from the
SAME `decision.evidence` keys so gate and decision cannot drift — and
nothing the decision does not gate. Concretely: `timestamps` now judges
analysable **coverage** of the window (gap-aware segments ≥ 3 s, the
`min_coverage_any` 0.60 gate; isolated drop bursts forgiven, recurring
drops that shred segments still fail); coherence ≥ `coherence_floor`
(0.20); timing ≤ `max_timing_precision_ms_any` (40 ms); jitter and the
multi-ROI quota demoted to diagnostics; lighting judged on the
**face-region** lux proxy (and `ingest` defers its proxy-lux verdict to
the same face-region estimate when the whole-frame proxy is the only
blocker — post-scan parity, fail-closed ordering preserved: a dark scene
still fails with the illuminance reason); exposure check detects
**steps** (> 12 % between adjacent 0.5-s bins in the last 2 s), not
drift; a check breaks the hold only after **2 consecutive** failing
evaluations (in-scan, engine-observed conditions — face, framing,
lighting, exposure, motion, tracking, frame rate — still pause
IMMEDIATELY: they are already frame-level-smoothed, and a face that
leaves must not be recorded as good time; the debounce covers the
evidence estimators). SNR (≥ 2 ROIs at +3 dB), prelim beats (≥ 4), SQI floor
(0.30), fps floor (24), hold (3 s) and every anti-periodicity property
are unchanged. AF-grade evidence is still judged — by the decision, on
the full recording, where it accrues.

**Honesty note.** Starting at any-result grade means a marginal signal
can still end in REPEAT_SCAN; the gate eliminates the
doomed-from-the-start class, not the marginal class. The synthetic face
cannot probe the real mid-grade regime (its pulse is far stronger than
skin), so thresholds rest on parity with the decision, not on a
synthetic sweep.

**Telemetry (new invariant-adjacent surface).** Every session appends
per-evaluation records and lifecycle events to
`<session>/readiness_log.jsonl` (state, ready, failing checks with
values, evidence summary, eval wall-time ≈ 59 ms median, outcome). A
session that never becomes ready is now diagnosable after the fact.

**Tests.** New: recurring-drops stream must reach READY ≤ 16 s AND
complete to ACCEPT (the DoD coupling, the root-cause regression);
isolated hole / pure jitter forgiven while dense drops and low fps still
denied; threshold parity asserted against the any-class config keys;
AE-ramp passes while a 20 % step fails then recovers; lit-face-dark-room
passes lighting (engine and ingest, whole-frame luma < 32 asserted as
the discriminating premise); debounce unit behaviour; telemetry log
contents. Rhythm-neutrality (AF READY-rate = sinus) retained unchanged.
`user_facing_text` untouched.

## B.13 v0.1.4.2 changelog — two-region verification + ROI-integrity framing (appended 2026-08-25)

**Problem (second real-use falsification, with evidence this time).** The
owner's live attempt on the v0.1.4 build was blocked by exactly two
checks, screenshot on record: `Framing ✗ (Move back a little)` and
`Cross-region 0.00 ✗` — while the signal was objectively decent (beat
timing 27 ms, 5 beats, SQI 0.40, tracking 0.98, 30 fps, 100 % frames,
2 SNR ROIs).

1. **Coherence 0.00 was structural, not a signal property.** The
   coherence component counts only beats seen by ≥ 3 ROIs; with exactly
   two strong regions (ordinary home lighting: one cheek shadowed, weak
   nose) it reads 0.00 by construction — the metric cannot distinguish
   "two good regions" from "garbage". The discriminating evidence
   between the two is precisely what the v0.1.2 FP investigation
   established: timing precision and matched fraction (real FP scans:
   36–49 ms, matched 0.63–0.75; this user: 27 ms).
2. **The 0.70 face-width cap had no post-scan counterpart.** A close,
   centred face with all four ROIs intact is fully analysable; the cap
   blocked the standard laptop posture.

**Fix.** (a) *Two-region verification* in the decision (any-class AND
AF-call coherence gates) and, by parity, in readiness: the coherence
floor is satisfied EITHER by ≥ 3-ROI coherence OR by the AF-grade
timing budget (median |Δt| ≤ `afib_max_timing_precision_ms`, matched ≥
`afib_min_timing_matched`) — no new thresholds, the existing AF keys
are reused as the price of missing third-region coverage. The
pulse-deficit rule is deliberately UNMOVED (still coherence ≥ 0.50):
deficit is inferred from absent beats, which two regions cannot
cross-verify. The recorded FP evidence values are a permanent
regression test and still abstain (they fail the timing budget).
(b) *ROI-integrity framing*: the veto now fires when any ROI keeps
< 50 % of its nominal pixels inside the frame
(`preprocessing/roi.py::roi_integrity`); the width floor (0.20, pixel
budget) and centring tolerance stay, the width cap is gone.

**Consequence acknowledged.** AF is now callable on two-region-verified
evidence (harmonic/split/interval/timing gates unchanged). The
alternative — permanent AF-blindness for users in ordinary home
lighting — is the P8 abstention-funnelling failure mode expressed
through capture conditions. `user_facing_text` untouched.

**Tests.** Two-region sinus → ACCEPT/SINUS; two-region AF →
ACCEPT/AFIB_SUGGESTIVE; both recorded FP profiles → abstain, never AF;
deficit rule unmoved by two-region evidence; readiness passes the
user's exact profile (coherence 0.00 / 27 ms / matched 0.85) and still
refuses the FP timing profile; close-face (width > 0.70, ROIs intact)
framing passes; `roi_integrity` unit math. Suite green.

## B.14 v0.1.5 changelog — the gate becomes a 1-5 star grade (owner-directed, appended 2026-08-25)

**Owner direction.** Even recalibrated (B.12/B.13), a blocking gate
prevents the attempt and therefore all learning. Replaced by a
confidence rating: the scan always starts unless a BLOCKING check fails
(face, framing, hard fps floor 15 — `decision.readiness.blocking_checks`);
every other check keeps its threshold and hint but feeds the 1-5 star
score (`inference/confidence_stars.py`; weights + cuts under
`decision.confidence`). `decision.readiness.mode: blocking` preserves the
previous gate verbatim (default is `advisory`).

**The rule that was not lost.** Every v0.1.2 AF evidence gate is
byte-identical. Stars EXPRESS them: (1) any failing AF evidence gate
(evaluated with the decision's own semantics, incl. the B.13 two-region
waiver; the length-dependent interval count applies to the finished
recording, not the live 8 s meter) caps stars at 2; (2) the decision
downgrades an AFIB_SUGGESTIVE carrying < 3 stars to REPEAT_SCAN (no new
outcome value) with the reason named. So "below 3 stars can never be
AFIB_SUGGESTIVE" is enforced twice, and
`tests/test_false_positive_guards.py` passes with ZERO edits.

**Carried through.** `ScanResult.confidence_stars` /
`confidence_limiting_factor`; `user_facing_text()` gains the low-star
sentence ("We could see your pulse, but not clearly enough…" + a fixed,
audited per-factor hint table) and a "Confidence: n of 5." suffix —
`test_user_facing_text_never_diagnoses` unchanged. Pipeline grades every
recording (capture-invalid attempts too: 1 star with the failing
condition named). Engine: Start available as soon as blocking checks
pass (no hold in advisory; hold kept for resume-after-pause and for
blocking mode); dark/motion hard aborts are blocking-mode-only (advisory
grades them; a lost face still pauses and the pause budget still
restarts); live star meter + limiting-factor hint in the demo UI and
telemetry.

**Deviations flagged.** (a) The prompt describes the pre-existing gate
as v0.1.3; this tree is at v0.1.4.2 — `mode: blocking` preserves the
v0.1.4.2 gate (same checks/thresholds), which is what "today's
behaviour" means here. (b) The "Confidence: n of 5." suffix is appended
to ACCEPT results at every star count (not only 3+): a 2-star
ACCEPT/SINUS (possible when only the AF-call interval count failed) must
not carry the inconclusive sentence. (c) `min_fps_hard: 15` starts scans
that the 24 fps consumer capture floor will refuse post-scan (fps
15-24): they complete and are graded, outcome NO_RESULT — consistent
with "grade, don't refuse".

**Tests.** tests/test_confidence_stars.py (11): module margins/cuts,
monotone-in-quality sweep, AF-gate star cap incl. two-region exemption,
blocking-fail = 1 star + no start, advisory/blocking mode split,
decision coupling both directions, dark30 graded 1-2 stars end-to-end,
sinus30 ≥ 3 stars with suffix, star rhythm-neutrality (AF ≥ sinus − 1).
Motion/low-signal engine tests rewritten to advisory semantics;
hold/abort tests pinned to mode: blocking.

## B.15 v0.2 changelog — the Inferred-ECG platform (append-only; opened 2026-08-25)

**Definition (governs every v0.2 decision).** "Inferred ECG" means the
clinically useful RHYTHM conclusions a doctor would draw from an ECG —
rate, regularity, beat-to-beat pattern, AF-suggestive irregularity —
inferred from measured optical signals. It does NOT mean a generated ECG
waveform: a camera measures the mechanical/hemodynamic consequences of
the heartbeat, not its electrical activity; a synthesized trace would be
beat-timed truth wrapped in hallucinated morphology. The platform renders
only measured signals and rhythm inferences; a quarantined research
module exists to PROVE with our own data why generated waveforms do not
ship (M4). The consumer surface is named **Rhythm Map**.

**Invariants 9-14 (added to the spec's 1-8):**
9.  Measurement-class labeling: every ScanResult v2 field carries
    MEASURED | INFERRED_RHYTHM | RESEARCH_SYNTHETIC
    (`datasets/schema.MeasurementClass`); app/ renders only the first
    two; a test enforces the classification table covers every field.
10. The falsification lab (`evaluation/inferred_ecg/`) is quarantined:
    artifacts watermarked "SYNTHETIC ECG — RESEARCH ONLY — NOT A
    MEASUREMENT", outputs only under `evaluation/falsification_runs/`,
    and a test asserts no import path from `app/` or
    `inference/pipeline.py` into it.
11. "ECG" in user-facing text only inside the referral sentence; the
    consumer surface is "Rhythm Map"; a test greps rendered app strings.
12. Paired ECG ground truth flows one way (training/evaluation), never
    into a consumer artifact.
13. No model ships without beating the mandatory baselines on
    participant- AND session-disjoint splits, evaluated via the gated
    production path (`cli.py process`), never a side harness.
14. Extension = new head (`heads/` plug-in over the BeatLattice), never a
    new pipeline; anything needing to bypass L2/L3 is out of scope.

**M1.1 — endpoint-head registry (this commit).** `heads/base.py`:
`EndpointHead` (name, version, required_inputs, run(BeatLattice, context)
-> HeadResult), `HeadResult` (measurement class enforced by type), a
registry with duplicate/unknown protection, and
`decision.heads.enabled` config flags. The `afib` decision head is
MANDATORY — disabling it raises (a pipeline without the gated decision is
not a production path, spec invariant 5). `beats/lattice.py` formalises
L3 as the versioned `BeatLattice` (beat-lattice-v1) with JSON round-trip.
Sanctioned rate-flag sentences added to `datasets/schema.py`
(RATE_FLAG_SENTENCES: brady, tachy) — heads never compose text.

**Preflight note (conflict flagged).** The v0.2 prompt's preflight
expects "139 passed" — the v0.1 baseline. This tree is at v0.1.5 with
213 passed + 1 conditional skip (spec B.8-B.14); the suite EXCEEDS the
precondition and the build proceeds on the newer, stricter baseline.

**M1 (complete).** head_afib wraps the gated decision bit-for-bit
(regression-locked sentences/stars/outcomes on the synthetic corpus);
head_rate_flags (brady <50 / tachy >100 from clean runs, sustained-
fraction >= 0.75, abstains under 15 intervals; sanctioned sentences in
schema.RATE_FLAG_SENTENCES); head_rhythm_map renders MEASURED-only SVG
panels (pulse waveform + confidence-shaded lattice beat ticks, tachogram
over clean runs, Poincaré) embedded in the results page; ScanResult v2
(schema_version 2, head_results with per-item classes, capture_meta,
FIELD_CLASSES covering every field, v1 migration shim);
`cli.py process --heads` with afib mandatory.

**M2 (complete).** Campaign manifests + `cli.py campaign status`
(Fitzpatrick/device/lighting quota fill, consent-version and checklist
integrity); `cli.py ingest-reference` (ECG CSV / minimal WFDB fmt-16,
sync overlap verified, offset+drift via estimate_sync, FAIL CLOSED over
5 ms or under 10 markers, in-house R peaks >= 98% recall on synthetic
truth, episode-locked labels with adjudicator fields + ESC checkbox);
dataset registry (content-hash identity, immutable, append-only JSONL) +
registry_dataset_splits (participant AND session disjointness proven) +
`cli.py evaluate` consuming registered datasets only (behaviour change
on the CLI, recorded here).

**M3 (complete).** training/ runs (config-driven, deterministic,
production-path features cached by dataset content id), mandatory
baselines (tachogram_stats, hr_only, participant_history = memorization
detector, device_site = metadata-leak detector), model registry +
`cli.py promote`. PROMOTION MARGINS (spec-defined here): candidate must
beat tachogram_stats and hr_only by >= 0.02 AUC (ceiling clause: both >=
0.97 and candidate not below); detector baselines must stay <= 0.60 AUC
(else the DATA is refused) and be exceeded by >= 0.15; no-read parity
ratio <= gates.fairness.noread_max_ratio. Every refusal lists every
failing reason.

**M4 (complete).** Quarantined falsification lab + `cli.py falsify` +
docs/inferred_ecg_falsification.md. MEASURED RESULT (MIMIC PERform,
participant-disjoint 25/10, 2026-08-25): the PPG->ECG decoder WINS
waveform correlation (0.185 vs identity-template 0.033) and LOSES the
interval test 5-8x (PR MAE 56 ms vs 12 ms; R->T 78 ms vs 10 ms); trained
without AF it hallucinates P waves into AF at 0.39 relative prominence
vs 0.04 in the reference. This is the platform's standing, reproducible
answer to "why don't you just show an ECG?".

**M5 (complete).** head_flutter_suspicion (research flag: 140-160 bpm
band + rel-MAD <= 4% + integer-ratio rate steps across a session series;
no sentence, not enabled by default) and head_burden (per-day
AF-suggestive fraction with ABSTENTION-AWARE denominators — a no-read
day has an undefined fraction, never zero; internal only).

**v0.2 deviations & notes (flagged).** (a) Preflight expected the 139-
test v0.1 baseline; the tree was at v0.1.5 (213+1) — proceeded on the
newer baseline. (b) Invariant 11 interpreted as: 'ECG' allowed only
inside clinician-referral/ground-truth deferral sentences (the two
pre-existing honesty strings in the app qualify); phrases implying the
app shows an ECG are banned outright by test. (c) M3's three tasks were
committed together (co-designed modules); every other task is one
commit. (d) `mode: blocking`/legacy behaviours from v0.1.5 unchanged.
(e) The falsification decoder is deliberately conv-scale; the write-up
codifies the burden of proof for any larger decoder claim.

## B.16 v0.2.1 changelog — the Cardiac Rhythm Scan Report (appended 2026-08-25)

**Removed** from the results screen: the metric-tile grid, the Rhythm Map
SVG panels (waveform/tachogram/Poincaré), the legacy waveform canvas +
drawWave, and the rate-flag caveat injection — replaced by ONE unified
printable document rendered by `app/report_render.py` (pure stdlib,
deterministic) with inputs assembled by `app/report_data.py` from data
the pipeline already computes. The explainability details panel stays on
the demo page. `cli.py process --report out.html` exports the standalone
file; the browser's Print-to-PDF is the PDF path (no PDF library).

**Report rules R1-R7 (each is a test in
tests/test_report_forbidden.py / test_report_render.py):** R1 strips
plot only the measured rPPG pulse with in-strip label `FACIAL PULSE
WAVEFORM (rPPG) — NOT AN ECG`; R2 neutral gray 0.2/1.0 s grid, no lead
labels / mV / mm-per-s / red-dominant colours (every hex in the document
is checked); R3 no synthesized waveform (identifier-level audit of
app/); R4 "ECG" only in the strip labels + the referral sentence — the
title is `AvatarX Cardiac Rhythm Scan Report`, "inferred ECG" is banned
from user-facing surfaces; R5 all rhythm wording via
`user_facing_text()` unchanged, findings box adds verbatim: "This
screening result is not a diagnosis. A positive or uncertain result
should be confirmed with an ECG."; R6 NO_RESULT is a first-class report
(reasons + insufficient-quality overlay over whatever signal existed);
R7 schema/gates/pipeline untouched (renderer only; `report:
show_tachogram` config flag added).

**R4/R5 reconciliation (flagged conflict).** The sanctioned
AF-suggestive sentence itself ends "…may recommend an ECG."; R5 forbids
changing it. The R4 count is therefore: strip rows + referral + any
occurrences inside the sanctioned sentence, and the test accounts for
each explicitly. The B.15 "Rhythm Map" surface name is superseded by the
report title; the rhythm_map endpoint head remains registered as a data
producer.

**Rationale (verbatim, per the prompt):** "Clinician familiarity is
delivered by layout, not by imitating the ECG trace; the measured pulse
waveform is presented under its own name."

## B.17 v0.3 changelog — the ECG-reconstruction research track, gated (opened 2026-08-29)

Owner-directed (prompt `CLAUDE_CODE_PROMPT_v0.3_ecg_reconstruction_gated.md`, repo root): build the complete architecture for ATTEMPTING ECG reconstruction from a face scan — data flywheel, training loop, fidelity evaluator — **with a pre-registered promotion gate (§G) between the reconstruction model and the user**. "Can reconstruction reach clinical fidelity?" is now a standing empirical question, not an assumption in either direction. The near-term product remains the validated rhythm findings on the measured path.

**Spec conflicts flagged (not silently resolved):**

1. **Invariant 9/10 amendment (owner-signed via the v0.3 prompt).** v0.2 stated "No consumer- or clinician-facing generated/synthesized ECG waveform. **Ever.**" The v0.3 prompt replaces "ever" with a conditional pathway: user-visible **if and only if** every §G gate passes on participant+session-disjoint facial data through the production path, with numeric thresholds `REQUIRES_CLINICAL_SIGNOFF` (owner + clinical advisor must confirm/tighten before any promotion; loosening any gate requires an owner-signed changelog entry here). Operationally nothing is displayable today and `models/registry.promote` refuses while anything is red; the v0.2 escalation stance survives as the v0.3 escalation rule (rendering while red, or loosening without signature → refuse and record). Interval ESTIMATION likewise remains lab-only evaluation (G2 numbers), never a consumer output — consistent with v0.2's escalation rule.
2. **Watermark wording.** Invariant 10 fixed `SYNTHETIC ECG — RESEARCH ONLY — NOT A MEASUREMENT`; the v0.3 prompt specifies `SYNTHETIC ECG — RESEARCH ARTIFACT — NOT A MEASUREMENT`. Both are sanctioned watermarks: the frozen v0.2 lab (`evaluation/inferred_ecg`, its runs and tests) keeps its string; everything under `research/` uses the v0.3 string, which is canonical going forward.
3. **v0.2.1 default surface superseded (owner-directed).** The report's default mode is now `report.mode: findings_only` — header, measurements, findings (sanctioned sentence + stars + caveats + referral), footer; **zero waveform elements of any kind** (no rPPG strips, no interval trend; and never a synthetic ECG while §G is red). R1–R7 stay binding for `mode: full`, which restores the v0.2.1 layout intact; the v0.2.1 renderer tests now pin `full` explicitly and `tests/test_findings_only.py` pins the default. Updated accordingly: `test_report_render.test_cli_report_export_on_fixtures`, `test_scan_engine`, `test_live_server`, `scripts/selftest_headless.py`.
4. **Prompt's "139+" preflight count** — stale as in B.13; the suite stood at 291 passed + 1 conditional skip at v0.3 start.

**Built (T1–T7, tests first):** morphology label fields on reference ingestion (`labels[].morphology`: conduction_pattern vocabulary + measured PR/QRS/QT per adjudicated segment, validated hard, optional per row); quarantined `research/ecg_reconstruction/` (architecture REGISTRY over the reused v0.2 lab decoder — config-swappable, fail-closed; deterministic artifacts with lineage); `fidelity.py` implementing every §G metric (automated QRS-onset/offset + tangent T-end intervals applied identically to both sides, identity-template baseline, RR-only QT regression baseline, subject-level bootstrap CIs, withheld-class confabulation eval, measured-path vs reconstruction detection AUC, blinded-read kit with escrowed answer key); `configs/gates.yaml` (G-2026-08-29-1) + gate machinery with two honest layers (numeric verdicts vs domain qualification — synthetic/MIMIC surrogate evidence exercises but can never open a gate); training loop `cli.py train <config with reconstruction: block>` auto-evaluating §G per run and appending `research/runs/scoreboard.jsonl`; the facial flywheel (`facial_pairs`: registered dataset manifests → `reference_dir` sessions → production-path rPPG paired with reference ECG on the video clock); `cli.py reconstruct` (watermarked research artifact + fidelity report + gate footer, fail-closed without a trained model); `cli.py gate-status` (+`--html`); `promote` refuses reconstruction runs on any red gate reading `gate_results.json` (data, never research code); findings-only surface (conflict 3); `head_irregularity` research stub (rhythm-agnostic, silent) beside the existing `flutter_suspicion`; import audit extended — **all** of `app/` and `inference/` (transitively) must reach neither `research/` nor `evaluation/inferred_ecg`. The afib head is bit-for-bit unchanged (regression suite untouched and green).

**Measured scoreboard at ship (honest reds).** MIMIC PERform run `recon-cd0892ab29ef` (25/10 participant-disjoint, windowed_mlp 400 steps): G1 RED — identity template wins every morphology metric (QT MAE 32.34 vs decoder 48.72 ms; PR 12.0 vs 56.0; QRS 8.0 vs 20.0). G2 RED — QT 48.72 ms [CI 33.13–93.36] > 20 and loses to the RR-only QT baseline (46.32); PR 56.0 [48–72] > 20; QRS 20.0 [8–24] > 15. G3 RED — no blinded read on record. G4 RED — hallucinated-P excess 0.42 (gen 0.50 vs ref 0.08 on withheld AF; reproduces B.15's 0.39-vs-0.04). G5 — AUC deficit 0.000 on the AF-irregularity endpoint (0.857 both paths; numerically non-inferior, still RED by domain: finger-PPG surrogate, not production path). Promotion: **BLOCKED**; additionally blocked by the unsigned clinical-signoff block. Every gate also carries the structural reason: no registered facial paired dataset exists yet.

Suite: 291 passed + 1 conditional skip → 321 passed + 1 conditional skip (30 new tests across test_reference_ingest, test_reconstruction_track, test_reconstruction_gates, test_reconstruction_flywheel, test_findings_only, test_growth_stub_heads; full-suite run 2026-08-29, 460 s).

**v0.3.0.1** (2026-08-29): FOURTH wrong-interpreter launch — a shell with a stale venv activation (`(venv)` prompt) resolved `python3` to the dependency-less Homebrew 3.14; the v0.2.1.1 copy-paste suggestion was printed twice and not acted on. The preflight now RELAUNCHES the CLI once with the discovered working interpreter (`os.execv`, announced on stderr): a wrong-interpreter launch self-heals instead of instructing a human. `AVATARX_NO_REEXEC=1` opts out and is set automatically before the relaunch, so a broken "working" interpreter cannot loop; the relaunch never fires inside pytest (an in-process preflight test would silently exec away the test run — it did, once, during development). Verified live: `/opt/homebrew/bin/python3 cli.py process …` announces "relaunching with /usr/bin/python3" and continues under it. Also removed `.venv/` (13 MB): a second dependency-less Python-3.14 venv left by the same codex experiment — its own `pyvenv.cfg` records it was created BY the `path/to/venv` deleted in v0.2.1.1 — unreferenced by any file. One new test (relaunch + loop guard); suite: 322 passed + 1 conditional skip.

## B.18 v0.4 changelog — the cardiorespiratory recovery + fitness track, gated (opened 2026-08-30)

Owner-directed (prompt `CLAUDE_CODE_PROMPT_v0.4_vo2_crf_recovery.md`, repo root; engineering plan `docs/AvatarX_VO2_CRF_Engineering_Plan_2026-08-30.pdf`): a three-phase guided session (safety screen → rest scan → guided activity → transition → recovery scan → findings report) measuring recovery physiology, **with the pre-registered §V gate block between any fitness inference and the user**. "Does camera recovery physiology add fitness information beyond demographics?" is a standing empirical question (§V3, pivotal); until §V is green AND clinically signed, only MEASURED recovery physiology renders, no fitness category exists on any surface, and an exact VO2 max number never renders in any version. Structural rules in code, not prose: HR is never estimated during movement (the activity phase reduces each frame to one vertical-centroid float; the pulse path cannot see it); weight/height are user-entered, never inferred from the face; safety screen and compliance contract (±10% / 10–20% / >20% or transition >10 s, vitals still render) fail closed.

**Spec conflicts and interpretations flagged (not silently resolved):**

1. **MeasurementClass grows a fourth member** (`INFERRED_FITNESS`) with the rendering invariant extended: app may render it ONLY while the `vo2:` block of configs/gates.yaml is green and signed (`evaluation/fitness_gates.fitness_render_allowed`, fail-closed; double-gated in `app/report_session.py` and tested unreachable). The v0.2 test pinning "exactly three classes" (`test_measurement_classes_are_the_three_sanctioned_ones`) was updated additively per the owner prompt's schema rev — any FURTHER member remains a spec change.
2. **Invariant 6 scope** ("no demographic inputs to the classifier"): the base-platform doc states it globally; the fitness track's V3 gate *requires* demographics as the null model (the baseline ladder) and `head_fitness` age-references its category. Recorded interpretation: invariant 6 is scoped to the RHYTHM classifier (its rationale is shortcut learning on the AF label) and is untouched; in the fitness track demographics are the baseline to beat, not a shortcut — that is what ΔSEE measures. The rhythm decision path is bit-for-bit unchanged.
3. **Fairness-scale naming**: the older platform doc stratifies by Fitzpatrick; the v0.4 prompt and plan use Monk bands. §V uses Monk (schema's MonkTone.band already exists and FDA draft guidance specifies Monk banding); Fitzpatrick remains a secondary literature field.
4. **"Vitals still render" reading**: on a non-compliant or unverifiable activity, HRR values are withheld (None) rather than shown-with-caveat — an HRR without a standardized workload is not comparable and invites misuse; resting vitals (HR, RR) always render. This is the conservative reading of "NO_RESULT for fitness outputs (vitals still render)".
5. **head_trend visibility**: the plan permits direction-only trend output pre-V5; the prompt's product surface says MEASURED only while §V red. Conservative reading implemented: `head_trend` runs and stores (local versioned exportable store), its INFERRED_FITNESS result rides head_results for telemetry, magnitude stays None pending V5, and NOTHING renders until §V green+signed. Rendering direction-only earlier would need an explicit owner surface decision.
6. **ScanResult untouched at schema_version 2** (no new fields); the additive rev = SessionResult (schema_version 1, fully classified FIELD_CLASSES, fenced SESSION_SENTENCES through the invariant-11 audit), Recording gains optional `challenge`/`participant_context` blocks (unknown-key-rejecting loaders), and the fail-closed `cpet` label reader. The v0.4 prompt's "139+" preflight count is stale as in B.13/B.17 — baseline at v0.4 start: 322 passed + 1 conditional skip.
7. **Fitness falsification probes live in `evaluation/fitness_metrics.py`**, not the quarantined lab: they are harness-side (no synthetic-ECG content), and heads/app may never import research/* (quarantine unchanged and still audited).

**Built (T0–T8, tests first, suite green at every commit):** schema rev (above); synthetic three-phase fixtures (`scripts/make_synth_recovery.py`: rr_override hook, scripted-cadence activity clips, parametric decay truth, CPET fixture datasets, honesty note verbatim); `features/recovery.py` (8 s/50% trimmed-median windows, |dHR/dt| ≤ 3 bpm/s slew bound as a tested property, PAVA monotone trend, Theil–Sen 0–15 s back-extrapolation on a denser early grid, HRR30/60/120, tau research-only — acceptance MET on fixtures THROUGH the production pipeline: HR(t) MAE ≤ 2 bpm, HRR60 ≤ 3 bpm incl. dropouts/motion/jitter); `protocol/` (challenges sts_1min/step_3min/march_2min with contraindication ids; fail-closed PAR-Q+-style screen where unanswered blocks and bp_heart_meds routes instead of blocking; session state machine + compliance contract + shared `finalize_session`); `activity/` (pose-optional cadence counter, motion-energy fallback within ±1 rep on scripted fixtures, tracker in provenance; workload context from user-entered inputs only); §V in gates.yaml (V-2026-08-30-1) + `evaluation/fitness_gates.py` (pure stdlib, importable by app/, render check fails closed) + `gate-status --track vo2` + `promote` refusing `head_fitness` (mirrors §G, every red reason listed); heads (`head_recovery` MEASURED; `head_fitness` INFERRED_FITNESS, category-only, never a mL/kg/min number — regex-fenced at head, session-JSON and report layers; beta-blocker/CCB/ivabradine hard-route to trend-only; `head_trend` direction-only + `trend/store.py`); `evaluation/fitness_metrics.py` (SEE/MAE/Bland-Altman LoA/category metrics; the mandatory baseline ladder — age → … → demographics+activity → static-face-image negative control → recovery physiology → full — on ONE participant-disjoint quantile split with cluster-bootstrap CIs; shuffled-workload probe (must degrade — it does), transition-sensitivity sweep (a +10 s start costs > 3 bpm of HRR60), rest-only respiration 8–25/min from torso motion with recovery-phase RR research-tagged); `cli.py session` / `evaluate-fitness`; the live flow behind `protocol.three_phase` (MultiPhaseSession orchestrator, vsession API, metronome + live rep counter with NO physiology UI, sit-still transition clock stopped by the recovery scan's start event, session findings report via `app/report_session.py`, per-phase + session telemetry, client-error reporting endpoint, `why_not_ready.py` vsession support, headless `--session` selftest — verified live: full flow completes with an honest fail-closed NO_RESULT + resting vitals on the fake camera, transition 3.6 s).

**Scoreboard at ship: all five §V gates RED, promotion BLOCKED** (no facial CPET data, no fidelity/retest/subgroup studies, signoff unsigned; the only recorded run is the synthetic fixture dataset through `evaluate-fitness` — surrogate domain, machinery evidence only, and on its own criteria too: 3/2-participant ladder, ΔSEE without a CI, negative control not beaten).

**Adversarial review (closing pass).** A 38-agent find-and-verify sweep over the full v0.4 diff produced 33 unique claims; 31 were CONFIRMED with reproductions and 30 fixed in the closing commit (1 accepted: the recovery-start HTTP hook is exercised by the headless `--session` selftest rather than pytest). Highest-severity fixes: the transition clock now follows the LAST recovery-recording start (a scan-engine restart no longer carries a stale transition into the metrics); the beta-blocker/rate-limiting hard rule now actually fires in the LIVE flow (the screen-level `bp_heart_meds` answer routes head_fitness to trend-only via `meds_flagged` through `finalize_session` — previously consumed nowhere); §V V2 no-read parity no longer fails OPEN when the best Monk band has zero abstention (max()-margin semantics matching the sanctioned no_read_report). Also fixed and pinned by new tests: an unrecorded transition now fails CLOSED; every §V threshold must be EXPLICIT in gates.yaml (coded defaults removed — silent contract drift impossible) and tightening provably binds; the ladder ridge no longer penalizes the intercept against uncentred VO2 (small-n SEE/bias inflation gone); the cluster bootstrap keeps draw multiplicity (the collapsed version was anti-conservative for V3's CI-excludes-zero test); the shuffled-workload probe requires STRICT degradation; the cadence counter and respiration use edge-normalized detrending (a 4 px march counted 2/20 reps; slow breathers at 8–12/min read at double their rate — both reproduce-and-fixed, band edges pinned); gated fitness/trend wording moved into sanctioned schema tables (FITNESS_CATEGORY_SENTENCES / TREND_DIRECTION_SENTENCES, fenced by the invariant-11 audit); the `_promote_fitness` green+signed path is now exercised; live/CLI session parity (compliance + stars + calibration provenance + capture_meta on partial paths); MultiPhaseSession phase guards + idempotent finalize; trend store dedupes by session_id; vsession GC; client fixes (HTTP-error feedback no longer wedges the UI, session-mode test clip keeps a monotonic capture clock, an in-capture recovery failure ends in the honest partial verdict); `AVATARX_THREE_PHASE=0` means off; session-report HTML escaping pinned.

Suite: 322 passed + 1 conditional skip → **390 passed + 1 conditional skip** (68 new tests; full-suite run 2026-08-30, 963 s). The recorded evidence run on the fitness scoreboard was regenerated under the fixed harness at the closing commit.

## B.19 v0.4.1 changelog — research/debug UI + two flagged §V requests (appended 2026-08-30)

**Owner-directed UI work (2026-08-30, via prompt; client-only — `app/static/index.html`, no server or inference change):**
1. All loaders replaced with a horizontal progress-bar component (`.pbar`): camera-permission wait (indeterminate), analysis phase ("Analysing signal…", eases to 95% and reaches 100% only on the real `done` event, freezes on failure), and the in-scan progress (the elliptical arc around the face oval removed; a labeled bar below the stage shows recording progress, amber "Paused — waiting for signal" state; the oval outline itself stays as the framing guide). Reduced-motion and ARIA supported; verified via both headless selftests plus in-browser render.
2. A **RESEARCH / DEBUG METRICS** panel on the results screen at owner request, for pipeline validation: measured signal (HR, median IBI, usable beats, SQI + components), HRV/rhythm features from the decision rationale (RMSSD, pNN50, median-abs-successive-diff, irregularity index, dropout, runs), model-estimated outputs (rhythm class, AF probability when a model runs), and for sessions the measured recovery physiology + verdict internals. Every row is labeled measured vs estimated. Numbers and labels only — participant-facing findings sentences still come solely from `user_facing_text()`; the panel is `@media print`-hidden so the printable one-pager report remains findings-only (B.16/v0.3 surface unchanged).

**Requires owner decision — conflicts with §V (recorded verbatim, not implemented):**
- 2026-08-30 (a): "I don't see VO2 max vital in the report" — disappointment report.
- 2026-08-30 (b): "override the previous UI restrictions and expose all underlying physiological metrics … At minimum display: … VO₂ max / estimated cardiorespiratory fitness output …" — every other requested metric was implemented (see above); the VO₂/fitness item was not.
Disposition: no VO₂ max number exists anywhere in the session pipeline (head_fitness is category-only by construction, and "an exact VO₂ max number NEVER renders on any surface in any version" is a §V hard rule); the fitness *category* render while §V is red is exactly what the pre-registered escalation rule forbids without an owner-signed spec change. The debug panel states the withheld status honestly and points to the sanctioned validation surfaces (`cli.py evaluate-fitness` against CPET truth; `cli.py gate-status --track vo2`). A chat instruction is not the signed changelog entry the rule requires; if the owner wants the category (or the never-a-number rule reversed), that takes a signed entry here naming the change.

Also this session: pre-existing server defect found and spun off (feedback JSON can contain bare `NaN` — invalid for browser `response.json()`; fix running as a separate task). One `--session` selftest stall reproduced the known fake-camera flake (readiness log healthy, frame batches stopped; passed on retry).


## B.20 v0.4-vascular changelog — the arterial-stiffness research track, gated (opened 2026-08-31)

**Owner prompt** (tracked verbatim: `CLAUDE_CODE_PROMPT_v0.4_arterial_stiffness.md`): build `head_vascular` — estimate large-artery stiffness from facial pulse-wave morphology against carotid-femoral PWV ground truth, behind pre-registered gates, defended against T1 (morphology may not survive the camera → a signal-fidelity study runs BEFORE any model) and T2 (the age shortcut → every evaluation must beat an age+sex+brachial-BP regression, or the head has measured nothing and must not ship). Escalation rule: any instruction to show vascular output while gates are red, remove age/BP baselines, or move age/sex/BP into the model is refused and recorded verbatim.

**Flagged interpretations (owner may overrule; none silently resolved):**
1. *Version label*: the prompt titles this track "v0.4", but B.18's v0.4 (recovery+fitness) already shipped. Recorded as **v0.4-vascular / B.20**; commits prefixed `vascular`.
2. *Gate symbol*: the prompt calls the gates "§V", which B.18 assigned to the vo2 track. This track's contract lives in the **`vascular:` block** of configs/gates.yaml (`VASC-2026-08-31-1`, gates V0–V5, `gate-status --track vascular`); prose says "the vascular gates".
3. *MeasurementClass rev* (owner-directed via prompt, INFERRED_FITNESS precedent): **RESEARCH_VASCULAR** added; the pinning tests updated additively; RESEARCH_* classes are additionally filtered out of every app payload at the serialization boundary (`datasets.schema.public_head_results`).
4. *Fitzpatrick vs Monk*: this prompt specifies Fitzpatrick stratification; the vo2 track's prompt specified Monk. Followed each prompt verbatim — the cross-track inconsistency is the owner's to reconcile.
5. *Session statistic*: the prompt says "per beat and per session median". Session-level features are computed from **per-ROI ensemble-averaged beats** instead, because per-beat medians measurably fail at camera bandwidth (repo-measured on clean synthetic 60 fps clips: per-beat reflection-index IQR 0.36 and an inverted stiffness contrast; the ensemble recovers the injected morphology). Per-beat features remain computed as dispersion/quality diagnostics and for the fidelity study's per-beat tables.
6. *Added variance explained* (V1): computed as a held-out squared partial correlation of head predictions with cfPWV given B3's predictions — never a trained demographics+morphology model, which the standing no-demographics-in-any-model invariant forbids.
7. *Tonometry export parsing*: `datasets.reference.parse_pwv_export` handles a documented generic key-value shape with SphygmoCor/Complior-style aliases and fails closed (no cfPWV row, no device, no operator, >2 PWV rows, malformed numerics). Real device export fixtures are needed to extend the alias tables; a second PWV row ingests as the retest read.
8. *Head quarantine posture*: heads/ is reachable from inference/, so `heads/head_vascular.py` contains zero research imports (AST-pinned, dynamic-import evasion banned; the repo-wide quarantine walker now also resolves `__import__`/`import_module` string literals). The head is inert and number-free on every pipeline-reachable surface; real estimates exist only where the research side supplies model+features via context (`cli.py process --heads vascular` → watermark-first research report + stderr pointer; stdout stays the consumer-clean ScanResult).
9. *Demographics for baselines*: `VascularReference` carries visit-recorded `age_years`/`sex` (with brachial BP and HR) as **baseline-only** inputs — the T2 defense needs them; the head refuses any artifact naming them.
10. *Verb naming*: the battery/evaluation verb is `evaluate-vascular` (matching the `evaluate-*` family); the prompt named only `vascular-fidelity` and `gate-status`.
11. *SDPPG candidates*: all four second-derivative ratios (b/a, c/a, d/a, e/a) ship as candidates, gated to None below 50 Hz native sampling ("where derivable"); the V0 study adjudicates them like every other feature.

**Built (T1–T6, tests first, suite green at every commit):** stiffness ground-truth schema (`VascularReference` + fail-closed `vascular_from_dict` incl. strict bool meds flags and integral Fitzpatrick; `ContactPpg` sidecar; `*.pwv.json`/`*.ppg.json` hashed by the registry); synthetic vascular fixtures (`pulse_shape` hook in pulse_series — bit-for-bit legacy default — with a latent stiffness variable driving reflection/notch/rise couplings, synchronized contact-PPG sidecars, per-visit single repeat reads, distinct scan timestamps, informative vs age-shortcut regimes); the quarantined morphology extractor (11 candidate features; wide 0.5–10 Hz band rebuilt from production ingest traces; production beats/gates/segments with the run's RESOLVED confidence floor; adjacent-in-lattice beat pairing; V-c: no features off ACCEPT); the V0 fidelity harness (participant-level ICC(2,1) verified against Shrout–Fleiss, Bland–Altman, acquisition-time-ordered same-visit retest with a 3-pair floor, per-stratum per-feature tables, config-driven `surviving_features` that also disqualifies surrogate-domain/underpowered studies); the T2 baseline battery (B1 age / B2 age+sex / B3 age+sex+BP / B4 HR / B5 age+sex+HR on one participant-disjoint split vs a morphology-only head; multiplicity-correct cluster-bootstrap CIs; V-d structural: `battery_report` refuses to exist without every baseline and the B3 delta; config-swappable trainer registry `vascular.head_model`); estimate retest, participant-disjoint leave-one-site-out AND leave-one-device-out, per-axis fairness (Fitzpatrick + device; a declared axis without tables is a named red reason); the gates module (explicit-threshold `_req`, per-kind qualification, fail-closed render guard, V5 producible only via the signed signoff's `claim_scope`); `_promote_vascular` (full gate set + recorded verdict required — the same completeness check now guards the vo2 promote); `head_vascular` (inert/number-free off research surfaces, refuses demographic/stale/corrupt artifacts); campaign machinery gains age-band/BP-range axes, retest-pair counting, fail-closed unknown axes, and a vascular template manifest; docs (vascular_track.md with verbatim T1/T2, RUNBOOK §10 with mandatory `--domain facial_rppg` on real days, README summary).

**Invariants V-a..V-d, each a test:** V-a `test_inert_and_number_free_on_every_pipeline_surface` + `test_cli_process_keeps_vascular_out_of_the_consumer_result` + the payload-boundary filter test; V-b `tests/test_vascular_forbidden.py` (sanctioned strings, client page, both report modes, forced-open session fragment, head output; regex self-tested incl. cfPWV and m/s values); V-c `test_vc_no_features_on_non_accept_scans` + the beat-pairing test; V-d `test_vd_is_structural_not_editorial`.

**Adversarial review (closing pass).** A 10-lens find-and-verify sweep (63 agents) over the full track diff produced 53 findings; **47 CONFIRMED, 6 refuted; all 47 fixed** in the review commit. Highest-severity: promote accepted a hand-crafted partial gate list (now requires the full gate set and the file's own recorded verdict — vo2 hardened identically; the §G twin left untouched deliberately, its machinery always writes complete lists — noted as follow-up); beat pairs silently spanned dropped low-confidence beats, embedding sub-floor morphology in two-cycle "beats" (pairs are now adjacent-in-lattice); meds flags ingested string "no" as truthy; four coverage gaps (config-driven surviving path, gate-status merge, head_runner success path, evaluate-vascular CLI) now tested. Statistics corrected: agreement ICC counts participants (not retest-replicated recordings), retest pairs ordered by acquisition time (drift sign was attenuated toward passing), notch detection needs prominence and takes the earliest post-systolic minimum, resample time-base stretch removed, LOSO folds made participant-disjoint. Known latent twins flagged, not fixed here: the vo2 signoff check accepts only lowercase "null" (the vascular `_unset` handles all spellings); §G promote lacks the completeness check.

**Escalations — requires owner decision, conflicts with §V / T2 defense: NONE.** No instruction in this track required rendering a vascular output while red, removing baselines, or moving demographics into a model.

**Scoreboard at ship: all six vascular gates RED, promotion BLOCKED** (fidelity run + evaluation run on the registered synthetic fixture dataset — surrogate domain, machinery evidence only, and red on their own criteria too: fidelity run fid-6b563550c01b: 6 paired participants < 60, one retest pair < the 3-pair ICC floor so retest ICC is refused and ZERO features survive; the empty survivor set blocks the battery, so V1 reads 'no baseline-battery evaluation on record'; V3/V4 have no ratios; V5 has no signed claim scope). `promote` refuses with every reason listed. Measured on the synthetic rig for the record (participant-level, n=6 — not a performance claim): rise time agrees facial↔contact at ICC 0.87, reflection index 0.77, pulse width 0.70; notch relative amplitude reads 0.18 and the deeper SDPPG waves (c/d/e) resolve in only 2 of 6 participants (refused, honest None) — Threat T1, measured, which is exactly what this track exists to do before any clinical claim.

Suite: 390 passed + 1 conditional skip (v0.4 close) → **482 passed + 1 conditional skip** (92 new tests across the vascular track and its adversarial review; closing full-suite run 2026-08-31, 1158 s). The recorded runs on the vascular scoreboard were regenerated under the review-fixed harness at the closing commit (fid-6b563550c01b, vasc-a7efaab6e06f).


## B.21 v0.5 changelog — the vasomotor-reactivity (vasotone) research track, gated §W (opened 2026-08-31)

**Owner prompt**: build `head_vasotone` — measure vasomotor state and REACTIVITY (how the small vessels respond to a controlled provocation) from the facial pulse signal, validated within-subject against a contact perfusion-index reference, defended against T1 (amplitude is the most optics-confounded quantity we have → absolute cross-session tone presumed non-identifiable + a mandatory optics-placebo arm) and T2 (every provocation also moves HR and breathing → the response must add information beyond an HR+respiration baseline). Claim ceiling: a wellness-tier reactivity trend; endothelial/clinical claims need a separate EndoPAT/FMD-referenced prospective protocol, out of scope. Escalation rule: any instruction to show tone output while §W is red, emit an absolute tone value, or drop the null arms/baselines is refused and recorded verbatim.

**Flagged interpretations (owner may overrule; none silently resolved):**
1. *Measurement class*: no new enum member was added — `head_vasotone` reuses **RESEARCH_VASCULAR** (the vascular research family class; the prompt did not direct a new member, and adding one would be a spec rev without direction). The payload-boundary filter and pinning tests cover it unchanged.
2. *Gate machinery reuse* (prompt: "do not fork"): shared mechanics extracted to `evaluation/_gate_common.py` (explicit-threshold reads, the yaml-null convention, qualification, scoreboard IO); `vascular_gates` now delegates to it behavior-identically; `vasotone_gates` is contract-only on top.
3. *"Session manifests"*: implemented as per-recording sidecars — `<id>.provocation.json` (maneuver, timed phase marks from the rig's prompts, intensity 1–3, `optics_log` REQUIRED for null_optics, operator-recorded Fitzpatrick strata-only) + `<id>.pi.json` (PiTrace). The consumer demo app is NOT the collection instrument for provocation studies; on-screen prompts are the rig's.
4. *Finger-PPG waveform reference*: the campaign template requires it at capture (`finger_ppg_recorded` checklist; the `<id>.ppg.json` sidecar format from v0.4 serves it), but the v0.5 harness SCORES against the PI trace only — waveform-level contact scoring deferred, flagged.
5. *rBCG→rPPG delay ablation* (prompt: "optional"): deferred — no rBCG extractor exists on the production path and `rbcg/` remains ablation-only pending the patent review.
6. *Primary response definition*: the facial reactivity response is pre-registered as the delta_norm of `norm_pulse_amplitude` (the perfusion-index analog) and exists only when it survives the null arms; other survivors ride as reported deltas.
7. *AC/DC channel*: computed from the GREEN channel of the production ingest traces — the POS/CHROM projections normalize per window and destroy the amplitude scale they would otherwise carry (recorded design decision).
8. *PI export parsing*: `read_pi_csv` accepts a strict documented `t_s,pi_percent` layout, fail-closed; real oximeter export fixtures are needed to extend it. `ingest-reference` maps it onto the video clock via the sanctioned sync model and bridges to a uniform `pi_trace.json` (PiTrace) when coverage permits.
9. *Disjointness flags*: the within-subject design shares participants across arms by construction; the evidence `participant/session_disjoint` flags are COMPUTED (true only when the W2 battery's held-out participant split actually ran), never asserted.
10. *W2 CI statistic*: "CI excludes zero" is tested on the SIGNED partial correlation (`added_r_ci95`) — a bootstrap CI over the squared statistic is non-negative by construction and its exclude-zero test does no statistical work (measured ~12–14% null false-pass at the config-minimum cohort before the fix). The same correction was applied to the vascular battery's reported CI.

**Built (T1–T6, tests first, suite green at every commit):** provocation + PI schema (`ProvocationRecord` with the pre-registered maneuver vocabulary incl. both null arms, ordered non-overlapping phase marks, null_optics optics-log requirement; `PiTrace`; registry hashes both sidecars; `ingest-reference` PI ingestion + bridge); synthetic testability (`amplitude_envelope` + `optics_gamma_envelope` hooks in synth_video — bit-for-bit legacy defaults; `make_synth_vasotone` couples a latent tone response to pulse amplitude with 0.08 Hz vasomotion while the PI sidecar tracks the same latent; null_optics uses randomized zero-mean nonlinear gamma sweeps because a pure brightness scale cancels in AC/DC; uncontrolled-optics W-d fuel; retest visits); the tone extractor (`tone_features.py`: every feature emitted as baseline/response/delta/delta_norm — W-c structural; green-channel AC/DC merged per PHYSICAL lattice beat pair across SNR-screened ROIs; Theil–Sen trend on a difference scale; LF power from actual data coverage with interior-gap refusal and bin-overlap band selection; notch/reflection via v0.4 ensemble morphology per phase window; W-d withholds amplitude features without AE/AWB lock); the two null arms + harness (`vasotone_metrics.py`: `survivors_from_study` is the ONE config-driven survival rule — lamp responders dropped absolutely and vs natural drift, per-feature value floors, underpowered studies produce none; W0 direction agreement with conservative detection floors + within-maneuver Fisher-z magnitude correlation vs contact PI; W2 battery B1 HR / B2 respiration (torso-band second decode — the facial-trace version was circular) / B3 HR+resp / B4 cardiac-band-stopped motion, refusing on covariate starvation, judged by held-out signed-partial-correlation with participant-cluster bootstrap; W3 like-for-like time-ordered retest pairs with worst per-maneuver ICC + graded-intensity ordering; W4 participant-level detection/coverage parity with per-group drift floors and a pre-registered group-size floor); `head_vasotone` (research-flagged, quarantine-clean, inert/number-free on every pipeline-reachable surface, W-c enforced by a recursive-key test, refuses no-rule/stale/uncontrolled/floorless inputs; `tone_runner` research glue; `process --heads vasotone --provocation --pi --manifest`); §W gates (`W-2026-08-31-1`: W0 reference tracking, W1 optics invariance, W2 pivotal T2 defense, W3 dose/consistency, W4 fairness, W5 claim mapping via the signed signoff's `claim_scope: vasomotor_reactivity_trend`; every threshold explicit incl. the PI detection floor and group-size floor; owner_confirmed must be boolean true); `_promote_vasotone` with full-gate-set completeness; `evaluate-vasotone` + `gate-status --track vasotone`; campaign machinery gains the maneuver axis (fail-closed unknown/missing axis fields, same-maneuver retest pairing) + the vasotone template with both null arms quota'd; docs (vasotone_track.md with the verbatim context/threats/claim ceiling, RUNBOOK §11, README).

**Invariants W-a..W-d, each a test:** W-a `test_inert_and_number_free_on_every_pipeline_surface` + `test_cli_process_keeps_vasotone_out_of_the_consumer_result` + the payload-boundary filter; W-b `tests/test_vasotone_forbidden.py` (hyphen-tolerant regex, self-tested; sanctioned strings, client page, both report modes, forced-open session fragment, head output, scoreboard strings); W-c the `_delta`-only emission shape + the head's recursive-key ban + `test_constriction_survives…` shape assertions; W-d `test_wd_uncontrolled_optics_withholds_amplitude` + the harness exclusion + the head refusal + the CLI manifest-lock route test.

**Adversarial review (closing pass).** A 10-lens find-and-verify sweep (56 agents) over the track diff produced 46 findings; **42 CONFIRMED, 4 refuted; all 42 fixed** in the review commit. Highest-severity: the vacuous squared-statistic CI on the pivotal W2 (fixed with the signed bootstrap; null-calibrated by test both ways); the cross-ROI merge keyed on per-ROI float midpoints (the documented median almost never executed and beat floors counted pooled entries — now keyed by the shared lattice pair); LF power clamp-extrapolating over uncovered spans; the circular B2 respiration baseline; no test proving the null_optics perturbation reaches the extractor (now probed and pinned: gamma moves the amplitude analog 14× over the plain twin); the untested ingest-PI merge (now value-pinned incl. the sync model application, plus the PiTrace bridge it lacked). Also fixed: session-count/value-count decoupling in survival, pooled-across-maneuver W0/W3 statistics, fairness floors and participant-level rates, yaml-null `owner_confirmed`, identical gamma perturbations across null sessions, campaign KeyError on missing axis fields, hyphenated forbidden tokens, and the review-noted doc/template gaps (posture_change, finger-PPG checklist, `--manifest` in the research-head command).

**Escalations — requires owner decision, conflicts with §W: NONE.** Nothing required rendering tone output while red, emitting an absolute tone value, or weakening the null arms/baselines.

**Scoreboard at ship: all six §W gates RED, promotion BLOCKED** (evaluation run vaso-6ab8fb230d70 on the registered synthetic fixture dataset — surrogate domain, machinery evidence only, and red on its own criteria too: 4 null sessions per arm < the 20-session floors so ZERO features survive (on this fixture every candidate reads as an optics detector at these floors), the primary response never exists, W0 reads 0.0 agreement over 6 provocations < 60, the battery reads zero scored provocations, W3 has no retest study, W4's groups are None-rated below the participant floor, and W5 has no signed claim scope). `promote` refuses with every reason listed. The fail-closed cascade is the design working: null arms below their 20-session floors → no feature survives → the primary response never exists → W0 scores nothing → the battery reads zero scored provocations → W3 has no study.

Suite: 482 passed + 1 conditional skip (v0.4-vascular close) → **539 passed + 1 conditional skip** (57 new tests across the vasotone track and its adversarial review; closing full-suite run 2026-08-31, 1694 s). The recorded run on the vasotone scoreboard was regenerated under the review-fixed harness at the closing commit (vaso-6ab8fb230d70).


## B.22 v0.6 changelog — the atrial-flutter research track, gated §F (opened 2026-09-01)

**Owner prompt**: build the flutter track, reading the physiology first because it INVERTS the AFib architecture — AFib is detected because its pulse is irregular; flutter is the arrhythmia our AFib detector is structurally blind to, because fixed AV conduction of a ~250-300/min atrial circuit produces a metronomically REGULAR pulse (2:1 ~150, 3:1 ~100, 4:1 ~75). Flutter also cannot be NAMED from the pulse: separating it from SVT or sinus tachycardia at the same rate needs sawtooth F waves on an ECG, which facial video does not carry. So the track builds a rate-and-regularity PATTERN FLAG that surfaces suspicion and routes to an ECG, plus the serial-scan conduction-ratio signature. Evidence baseline recorded as the floor to beat, not the target (Cramer et al. 2025, J Clin Monit Comput, N=51 cardioversion patients: AUC 0.95 arrhythmia-vs-sinus, AF 94.6% / flutter 92.5%, but "other arrhythmias" only 61.8%, HR bias -7.45 bpm with 48.6% of readings within 5 bpm, and the single Fitzpatrick-VI patient excluded). Escalation rule: any instruction to show flutter output while §F is red, to name a rhythm, to drop the hard-negative battery or the B3 baseline, or to route the flag through the gated reconstruction head is refused and recorded verbatim.

**Flagged interpretations (owner may overrule; none silently resolved):**
1. *The v0.2 stub is superseded, not kept beside*: `head_flutter_suspicion` (M5.16, B.15) is DELETED and `head_flutter` replaces it. The stub carried `INFERRED_RHYTHM`, which app/ renders on sight, so enabling it would have put the string `flutter_suspicion` into a consumer payload — the exact F-b leak this track exists to prevent. Its three pinned tests moved to `tests/test_flutter_head.py` with stronger assertions (the series signature now needs a real integer-ratio fit, so three scans at one rate score zero where the stub's pairwise check called it a 2:1 step). A config naming the removed head now raises a registry error.
2. *Measurement class*: a NEW member, `RESEARCH_RHYTHM`, rather than reusing RESEARCH_VASCULAR (v0.5's choice) — a rhythm head labeled vascular would mislabel the payload filter's own audit trail, and RESEARCH_SYNTHETIC belongs to the quarantined falsification lab that F-d firewalls. Deliberately NOT `INFERRED_RHYTHM`: app/ renders that class on sight, and an ungated flutter output must be unrenderable BY CONSTRUCTION, not by remembering to check.
3. *The lattice gained a field without a version bump*: `run_times` (each interval's end-beat time) is additive with an empty default on both `RunSet` and `BeatLattice`, and `LATTICE_VERSION` is deliberately unchanged — a bump would make every stored v1 lattice unreadable by `from_dict`, while an optional field leaves them valid. Consumers must read an empty list as "no times", never as times at zero. Without this an interval cannot be placed on the clock and the respiratory-coupling family is undefined.
4. *Respiration is not on the production path*: the pipeline computes no respiration and the synthetic generator had no torso, so BOTH were built — `synth_video(torso_respiration=...)` draws a neutral-grey shoulders bar (grey is load-bearing: the skin-chromaticity tracker keys on r>g>b, so it can never join the face mask), and the harness reads respiration from the torso-motion SECOND DECODE (`rppg/respiration.py`), never the facial trace the intervals come from, which would be circular. Consequence flagged: on today's consumer scan path there is no respiration channel, so `head_flutter` ABSTAINS on every production scan by design. That is the honest posture — without RSA coupling, sinus tachycardia at 150 and 2:1 flutter at 150 are the same measurement — and it is pinned by test rather than left as a surprise.
5. *The dispersion floor is the camera, not the patient* (measured, see below): the effective floor is `max(age-adjusted physiological floor, frame-quantization floor)` and a scan where the camera wins is marked `measurement_limited`. An unknown frame rate yields NO regularity verdict rather than a permissive one. The age-adjusted floors are retained in config because they are the PHYSIOLOGICAL claim, but at 30 fps they are entirely masked, which is disclosed in the limitations doc rather than hidden.
6. *Only the 2:1 band flags*: 3:1 (~100) and 4:1 (~75) are computed and reported per gate F2 but never raise the flag, because they collide with ordinary rhythms. This is what makes slow fixed-block flutter a documented permanent miss (F-c) rather than a bug.
7. *SVT is covered by design, not separated*: a flag on an SVT is the flag working, not a false positive, and the sanctioned sentence names neither. In the battery SVT is counted as a NEGATIVE for specificity, which is the conservative choice — the alternative (counting it as a target) would flatter the numbers.
8. *The head's confidence is the scan's own evidence grade* (stars/5), not a flutter-specific calibration — none exists, and §F0 must measure the rate accuracy that any such calibration would rest on.
9. *B3 is the head's rule minus abstention*: the interpretable competitor reads a missing coupling measurement as "not coupled". The head's only structural advantage is that it declines what it cannot judge, so its specificity is always reported beside its no-read rate, and `shipping_rule` on the scoreboard names which one the evidence says should ship.
10. *Combined endpoint scoring*: all three Task-6 candidates are scored against the COMBINED truth (AF or flutter present), which is the referral question the product actually asks; scoring the AF head against AF-only truth would compare two different questions. Prevalence used for the projection is a PLANNING value (AF 0.03, flutter 0.003).
11. *`evaluate_flutter_dataset` requires `signal_domain` and defaults `production_path=False`*: a permissive default would let hand-built or surrogate rows be recorded as the one evidence class that can open a gate.
12. *Endpoint ranking rule — a deviation from the prompt's stated hypothesis, decided on the numbers*: the prompt expected the combined endpoint to win and become the flagship referral. Candidates are ranked by FALSE ALARMS PER TRUE CASE at deployment prevalence among those clearing a minimum sensitivity, not by sensitivity — because `combined` is an OR of the two heads and dominates both elementwise, so a sensitivity-first rank crowns it by construction whatever the burden. On the measured cohort the combined endpoint has 2.9x the false-alarm burden of the flag alone, and no candidate clears the sensitivity floor, so the harness records that the decision cannot be made rather than crowning one. The owner may overrule the ranking rule; the numbers are in the record either way.
13. *Participant metadata rides its own sidecar*: skin-tone group, age and sex are Participant data, not Recording data, and the schema has no place for them on a `Recording`. `<id>.participant.json` carries them, mirroring how a real campaign supplies participant-level fields; without it the fairness gate had nothing to rate on any generated cohort.

**Built (T1-T7, tests first, suite green at every commit):** the ground-truth label schema (`ConductionRatio` with VARIABLE deliberately divisor-less, `FlutterType`, and the additive `RhythmAnnotation` block carrying atrial rate / ratio / type / adjudicator identity / adjudication lead count; `flutter_label_problems` fails closed on each and requires 12-lead EP-level adjudication for a POSITIVE while leaving single-lead recordings valid as negatives and rate comparators; `Recording.flutter_label_problems` reports the least-broken annotation and says so plainly when there is no flutter annotation at all, so "no problems" can never read as "confirmed flutter"); the fixture generator (`make_synth_flutter.py`: fixed-ratio flutter at 2:1/3:1/4:1, a mid-scan conduction-ratio STEP switching on the clock, variable block, sinus tach with deep and shallow RSA, SVT, beta-blocked sinus, paced, AF, AF with regularized rate, each cohort declaring `flag_expected` so the known misses live in the fixture table; the serial sub-protocol stepping ONE atrial rate; breathing on every clip); the extractor (`features/flutter.py`: rate fingerprint with a distribution-free median CI and a sustainment fraction, hyper-regularity against the effective floor with tachogram spectral concentration and respiratory coupling, the serial latent-atrial-rate fit that scores zero unless two SEPARATED divisors are occupied, and variable-block features so the AF/flutter confusion is measurable; ACCEPT-grade clean runs only, nothing across a run boundary, no interval repair); `head_flutter` (two outputs, RESEARCH_RHYTHM, abstains with a named cause on every missing input, emits a sentence KEY never a sentence, `user_facing_text()` fail-closed by default); the §F gates (`F-2026-09-01-1`, on the shared `_gate_common` mechanics — reused, not forked — with specificity gated ABOVE sensitivity, the battery required to be dominated by sinus tachycardia, the head required to beat B3 by a pre-registered margin or B3 ships, the 4:1 sensitivity required to be MEASURED, and an absent darkest-skin-tone band redding F4); `_promote_flutter` with full-gate-set completeness; the harness (`evaluation/flutter_metrics.py`: rate accuracy computed on arrhythmia scans only, per-confounder false-positive rates with CIs, B1-B4 with B4 fit on a participant-disjoint half, head-vs-B3 at matched sensitivity, per-ratio table, serial AUC, participant-level fairness, and the Task-6 three-way endpoint at deployment prevalence); `evaluate-flutter` + `gate-status --track flutter`; campaign axes `rhythm_class`/`conduction_ratio` plus the `series` sub-protocol (a retest PAIR is not a series) and the two-arm template; docs (`flutter_track.md`, `flutter_limitations.md` shipped as a gate dependency, RUNBOOK §12, README).

**Invariants F-a..F-d, each a test:** F-a `test_no_sentence_escapes_without_an_explicit_gate_check` + `test_head_never_names_a_rhythm_anywhere_in_its_output`; F-b `tests/test_flutter_forbidden.py` (self-tested regex incl. hyphen/space forms and any n:1 ratio, sanctioned strings, client page, both report modes, consumer payload with the head enabled, and a forced-open flagging head result); F-c `test_the_known_miss_registry_is_a_shipped_artifact` + `test_the_product_does_not_contradict_the_known_miss_registry` + gate F5's on-disk check; F-d `test_head_and_extractor_cannot_reach_the_reconstruction_track` (AST over both files, imports/dynamic imports/module-path constants).

**Measured before believed (both through the production video path):**
- **RSA coupling is the discriminator.** At matched rate AND matched dispersion (RMSSD ~25 ms both), a metronomic clip and a sinus tachycardia had tachogram respiratory-band fractions of **0.03 vs 0.91**. Dispersion alone cannot separate those. This is why the head abstains without a respiration channel instead of falling back to dispersion.
- **The dispersion floor is frame quantization.** A zero-jitter 150 bpm clip reads RMSSD **15.0 ms at 30 fps** and **8.2 ms at 60 fps** — 0.45 of a frame period in both cases. The pre-registered age-adjusted floors (8-18 ms) therefore sit BELOW the 30 fps measurement floor and are entirely masked at consumer frame rates. 60 fps roughly halves the floor, which makes this a CAPTURE finding as much as an analysis one (RUNBOOK §12 leads with it).

**Adversarial review (closing pass).** A four-lens sweep (statistical correctness, safety invariants, test quality, integration risk) over the track diff produced **21 confirmed findings, all fixed** in the review commit. The decisive one was found independently by two lenses: `_head_score` returned B3's score, so F1's head-vs-B3 criterion — the gate that decides which rule ships — compared B3 against itself (verified: inverting every head verdict left the reported number unchanged). Two further defects in the same comparison: B3's threshold was swept in-sample, letting it find razor-thin splits a fixed rule cannot answer, and the head was scored on rows it judged while B3 was scored on all rows, so the head "won" by declining what B3 got wrong. Both fixed — B3's knob is chosen on a participant-disjoint half, the comparison is like-for-like, and B3-on-everything is reported beside it so abstention's value stays visible. Second-highest: the load-bearing discriminator failed OPEN — coupling was measured in a narrow band around the REPORTED breathing rate with nothing checking it, so a 2 br/min error turned a fully coupled sinus tachycardia (0.97) into "modulation absent" (0.17), i.e. into a flag; the rate must now be corroborated by the tachogram's own spectrum and a low-quality respiration channel is refused before use. Third: the 2:1 band contradicted the shipped limitations doc (F-c) — the doc states atria fire at 250-300/min, 2:1 of which is 125-150, but the band started at 140. Widened, with the measured COST recorded in the doc rather than hidden. Fourth: `cli.py process` printed the RESEARCH_RHYTHM payload (sentence key included) to stdout — the CLI now applies the same payload filter app/ does. Also fixed: the hard-negative battery collapsed to ONE class on the production path and could never match the gate's dominant-negative name; specificity counted non-2:1 flutter as a negative; rate accuracy counted the dominant confounder as an arrhythmia scan; "sustained" counted intervals rather than time; single-subgroup cohorts scored perfect fairness; F3's shortest-series check was unfailable; `shipping_rule` fail-OPENED a missing margin to zero and ignored `gates_path`; `datasets/io.py` left the new enums as strings so half-decoded annotations passed the label bar; an abstaining head still published a fitted atrial rate; config could widen the bands and render "sustained FAST" for a 75 bpm pulse. On the test side: the sanctioned sentence was the one table exempt from the invariant-11 ECG audit (and would have failed it); `flag_expected` became an ORACLE run through the real head, which caught the SVT fixture contradicting its own declaration; the e2e now asserts per-recording verdicts; the battery is exercised at 30 fps, the rate the product actually uses.

**Task 6 — the combined atrial-tachyarrhythmia endpoint, with numbers.** Measured on a 76-scan synthetic cohort through the production video path (surrogate domain — machinery evidence, never a performance claim), at planning prevalence AF 0.03 / flutter 0.003:

| candidate | Se | Sp | PPV | NPV | FP/1000 scans | false alarms per true case | no-read |
|---|---|---|---|---|---|---|---|
| AFib head alone | 0.088 | 0.905 | 0.031 | 0.967 | 92.1 | 31.6 | 0.000 |
| flag alone | 0.333 | 0.950 | 0.185 | 0.977 | 48.4 | **4.4** | 0.040 |
| combined (AF or flag) | **0.412** | 0.857 | 0.090 | 0.977 | 138.1 | 10.2 | 0.000 |

The hypothesis in the prompt was that the combined endpoint wins. On these numbers it does NOT: it wins on sensitivity by construction (it is an OR of the two heads, so it dominates both elementwise) while generating 2.9x the false alarms per true case and less than half the PPV of the flag alone. That is why the ranking was changed from sensitivity-first — which can only ever crown the OR — to false alarms per true case among candidates clearing a pre-registered minimum sensitivity. **No candidate clears that floor on this cohort (best 0.41), so the harness records that the endpoint decision cannot be made on the numbers, and F5 stays red with that reason.** The AF head's 0.088 sensitivity here is a fixture artifact — these synthetic AF clips rarely trip the production AF path — so the honest reading is that this cohort cannot settle the question, not that the combined endpoint loses. It is a real campaign, with Arm B, that decides it.

**Escalations — requires owner decision, conflicts with §F: NONE.** Nothing required rendering flutter output while red, naming a rhythm, weakening the battery or the B3 baseline, or routing the flag through the reconstruction head. Two results DO want an owner decision, neither of which is an escalation because neither was an instruction: (a) `shipping_rule` reads `b3` — the transparent rule is what the evidence says should ship, and the prompt's own instruction is to ship it and say so; (b) the Task-6 hypothesis (combined endpoint wins) is not supported on this cohort, and the ranking rule was changed to burden-first so the OR could not win by construction (flagged interpretation 12).

**Scoreboard at ship: all six §F gates RED, promotion BLOCKED** (evaluation run `flut-fff1baed3761`, 76 scans through the production video path — surrogate domain, so every gate additionally carries the "machinery evidence only" disqualifier that no synthetic run can ever clear). On their OWN criteria: **F0 and F4 pass** — rate bias **-0.11 bpm** with **97.4%** of arrhythmia readings within 5 bpm on 39 scans (against the Cramer floor of -7.45 bpm and 48.6%), and fairness rates all six Fitzpatrick groups with worst-group detection parity 0.75 and coverage 0.92, darkest band present. **F1, F2 and F3 are red on cohort size** (29 flutter scans < 40; 3:1 has 7 scans < 10; 4 flutter series < 10), and **F5 is red on the unsigned claim scope plus the undecidable endpoint**.

The measured behaviour behind those gates, which is what the track is actually for: 2:1 sensitivity **0.917** [0.727, 1.000] at battery specificity **0.956** [0.889, 1.000] with a 3.4% no-read rate; per-confounder false-positive rates of **0.00 for sinus tachycardia (n=19)**, 0.00 for sinus (n=18) and 0.00 for AF (n=5), with SVT at 0.67 (n=3) — which is the flag working as designed, not failing. Per-ratio sensitivity is 0.917 for 2:1 and a **measured 0.00 for 3:1, 4:1 and variable block**: the documented permanent miss is now a number. The serial signature separated flutter series from controls with AUC 1.00 on 4 vs 6 series.

**And the head does not beat B3.** At matched sensitivity on held-out, participant-disjoint rows the head scores 0.9565 and B3 scores 0.9565 — identical, so the pre-registered 0.02 margin is not met and `shipping_rule` reads **`b3`**. Per the prompt's own instruction, the transparent rule ships and this says so. The baseline ladder shows where the specificity actually comes from: B1 (rate only) 0.511, B2 (+ naive dispersion) 0.872, B3 (+ respiratory coupling) 0.957, B4 (demographics + rate) 0.656. The coupling term is worth ~8.5 points of specificity over dispersion alone, and the learned head adds nothing on top of it that this cohort can detect.

Suite: 539 passed + 1 conditional skip (v0.5 vasotone close) -> **667 passed + 1 conditional skip** (128 new tests across the flutter track and its adversarial review; closing full-suite run 2026-09-01, 1942 s). The recorded run on the flutter scoreboard (`flut-fff1baed3761`, 76 scans) was regenerated under the review-fixed harness at the closing commit.

## B.23 v0.7 changelog — the rhythm-regularity substrate track, gated §R (opened 2026-09-01)

**Owner prompt**: build the rhythm-regularity track — the substrate head that answers "is this pulse regular or irregular, and how confident are we?" — and make it the ONE representation the AFib and flutter heads consume. Three reasons it is its own track: (1) it is the substrate every rhythm head already computes privately, so one canonical `RegularityFeatures` replaces every parallel path (the P2 lesson: two implementations of the same statistic drift, and the drift is invisible until it disagrees); (2) its ground truth is computable from the reference ECG's own R-R series by the SAME code, which makes it the cheapest validation in the programme and a ceiling test for everything downstream; (3) its dominant false positive is respiratory sinus arrhythmia in healthy breathing people, which is a separation problem, not a detection problem. Regulatory anchor: the FDA irregular-rhythm-notification precedent (Apple DEN180042 / 21 CFR 870.2790 product code QDB; Fitbit K212372; Samsung K230292) — all contact PPG, no contactless predicate. Standing invariants: clean runs only, no interval repair; SQI must never score periodicity; `head_afib` outputs must stay bit-identical on the regression corpus after the rewiring — the equivalence test IS the definition of a safe refactor. Escalation rule: any instruction to render an irregularity verdict while §R is red, to name a rhythm from this head, to drop the benign-separation gate, or to let this head escalate to an AFib sentence is refused and recorded verbatim.

**Flagged interpretations (owner may overrule; none silently resolved):**
1. *Measurement class*: the prompt asked for the index + CI to be "MEASURED-adjacent". There is no such class, and adding one would create a third rendering rule for app/ to get wrong. The head carries `RESEARCH_RHYTHM` (v0.6's unrenderable-by-construction class) until §R opens; the index and CI are in the payload today, under a class app/ cannot render on sight. If the owner wants a distinct class for gated measured quantities, that is a schema decision to make once, for every track.
2. *The reference label goes through `clean_runs`*: the ECG R-peak series is trusted beat by beat (confidence 1.0, published in the definition), but the physiologic-range and missed/false-beat splitters still apply, identically to the camera. The alternative — a raw R-R series on the ECG side and clean runs on the camera side — would compare two different statistics and call the difference optics. Consequence: a genuine 2x pause on the ECG (a dropped beat, a sinus pause) is cut out of the reference too; the ceiling test can never reward the camera for missing it.
3. *`irregularity_index` = median |successive difference| / median interval, within-run differences only*. A median-based index is robust to SPARSE beat errors by construction (measured: the raw series only tips past the threshold at about 30 % beat error), while RMSSD is wrecked by one missed beat. That is why the class is decided on the index, why RMSSD is reported beside it and never decides, and why the beat-error study reports both. The threshold 0.06 is a PLANNING value sitting deliberately inside the RSA range (sinus at rest ~0.02-0.04, deep RSA ~0.06-0.10, AF ~0.15+) so that R2 has to earn the separation rather than the threshold hiding it. Clinical signoff may move it; it lives in `configs/gates.yaml` and nowhere else.
4. *Abstention without a respiration channel*: the head abstains when the respiration channel is absent, poor, or yields no tachogram/spectrum, because without it an RSA-dominant sinus rhythm and a chaotic one are the same measurement. As in v0.6 (B.22 #4) today's consumer scan path carries no respiration, so `head_regularity` ABSTAINS on every production scan by design, with the index still in the payload for the research reader. A "rate uncorroborated" coupling status is NOT an abstention here: the head judges it as evidence, because the v0.6 refusal rule was tuned for the flutter question (is the pulse coupled to breathing at the reported rate?) and, applied here, silently abstained on trigeminy whose 3-beat periodicity sits inside the respiratory band.
5. *`head_irregularity` changed behaviour across run seams*: its `rel_mad` and `pnn80` are now the within-run statistics from the representation; the pre-v0.7 version pooled intervals across run boundaries and counted the seam as a successive difference. This is the one consumer whose output is NOT bit-identical after the rewiring (it is research-only, disabled by default, and had no golden). Recorded here so that nobody attributes the change to a later commit.
6. *Isolated premature beats read regular under the published definition*: a PAC every ~12 beats moves the median of |successive difference| very little (index ~0.045 < 0.06). The fixture table records `pac_isolated` as expect-regular with no benign explanation, because that is what the definition says, not because the generator is wrong. A definition that flagged isolated ectopy would flag on a smaller statistic than the median and inherit the beat-error fragility of #3.
7. *The MDI planning gain is measured, not assumed*: the jitter budget is computed in code from the frame period (`sigma_beat = (1000/fps)/sqrt(12) x gain`), giving 13.6 ms RMS on intervals at 30 fps raw and 6.8 ms at 60 fps, as the prompt derived. Sub-frame interpolation reduces it by a gain that is CALIBRATED from the metronomic ladder rungs on every floor run (`interpolation_gain_measured`) and reported beside the planning value 0.65; the MDI table is computed from measured camera indices, never from the budget.
8. *MDI is defined by power, not by the budget*: for each (fps, SQI grade, Fitzpatrick group) cell the floor is the 95th percentile of the camera index on metronomic references at that fps, and the MDI is the median ECG-RMSSD of the lowest true-dispersion bin the camera detects at >= 80 % power. A cell with no metronomic references or no detected bin is "not characterized" (None) and REDs R1 — an uncharacterized cell is not a zero.
9. *R3 baselines are fitted on a participant-disjoint half and scored on the other*: B1/B2 thresholds are chosen on the train half; B3/B4/B5 are logistic models fitted there. On a synthetic cohort where demographics are assigned by cohort, B4 (demographics-only) leaks the label BY CONSTRUCTION and can beat the head; this is a property of the fixture, recorded in the harness, not a finding about people.
10. *Public data is a surrogate, never evidence*: the only public corpus with simultaneous rhythm truth, PPG and respiration is MIMIC PERform AF (contact fingertip PPG in ICU patients). It runs through the same beat detector, run discipline and representation and is recorded under `signal_domain: public_ppg`, which the qualification rule reds on every gate. It is the first ceiling number on human hearts and nothing more. There is no public contactless corpus with ECG truth.
11. *This head never escalates*: the R5 evidence records `escalation_path_present: false` and the gate reds if it is ever true. An AFib-suggestive sentence remains `head_afib`'s job; the sanctioned R5 sentence is the only user-facing wording family, it names no rhythm, and it offers the benign explanation first.
12. *The v1 raw-series estimator is deleted*: `compute_rhythm_features` (the pre-clean-runs path) had zero callers and was the second implementation G-a forbids. `RhythmFeatures` survives as a VIEW of the representation sharing the same values dict, so every existing consumer's reads are unchanged.
13. *The G-a audit is an allowlist, not a blocklist*: three functions in `inference/` and `heads/` legitimately call banned numpy primitives on things that are not intervals (frame timestamps, harmonic fraction, display waveform); they are named in the test's allowlist, and the allowlist is itself tested to name only functions that exist, so a rename cannot silently widen it.

**Built (T1-T7, tests first, suite green at every commit):** the representation (`features/regularity.py`: `RegularityFeatures` v1 with dispersion / distribution / structure / confidence families and the index + bootstrap CI, computed from clean runs only with no successive difference crossing a seam; `features/rhythm.py` reduced to an adapter whose `RhythmFeatures` is a view sharing the same values dict; `features/flutter.py`, `head_afib`, `head_irregularity`, `head_rate_flags` and the pipeline rewired as readers; the v1 raw-series path deleted; the seven-clip golden corpus and the AST audit with its tested allowlist); the noise floor (`evaluation/regularity_floor.py`: the jitter budget in code, the interpolation gain calibrated from metronomic references on every run, the MDI table per fps / SQI grade / Fitzpatrick group with power-based detection and explicit "not characterized" cells, the beat-error study reporting index AND RMSSD on the clean-run and raw paths; `cli.py regularity-floor`); the reference label (`datasets/regularity_reference.py`: definition read from `configs/gates.yaml` only, ECG R-R through the same `clean_runs` and `regularity_from_runs`, one definition object shared with the head by test); the head (`heads/head_regularity.py`: index + CI + class + `benign_pattern_evidence` on every irregular read with a fixed precedence, abstention with a named cause, sentence key never sentence, `user_facing_text()` fail-closed; `REGULARITY_SENTENCES` on the invariant-11 ECG-wording fence); the harness (`evaluation/regularity_metrics.py`: ceiling test with κ / index r / Bland–Altman overall, per SQI grade and per dataset; benign separation with RSA flag rate, RSA explained rate and age-stratified specificity with Wilson CIs; baselines B1–B5 on a participant-disjoint split with the head's no-read rate beside its accuracy; participant-level fairness; `evaluate_regularity_dataset` requiring `signal_domain`); the §R gates (`R-2026-09-01-1` on the shared `_gate_common` mechanics, R1 reading the floor run and requiring the MDI table to be PUBLISHED in `docs/regularity_track.md`, R2 requiring every age band, R3 requiring margins over B3/B4/B5, R4 requiring the darkest band, R5 requiring the signed claim scope and redding on any escalation path; `_promote_regularity` with full-gate-set completeness; `gate-status --track regularity`); the public surrogate (`datasets/public_regularity.py`: MIMIC PERform AF through the production beat detector at 125 Hz with the corpus' own respiration channel, recorded as `public_ppg`); the fixtures (`scripts/make_synth_regularity.py`: regular, RSA young / mid, AF, bigeminy, trigeminy and isolated PAC, each with a breathing torso bar, a truth sidecar, an ECG sidecar from the same RR series, and a participant sidecar; the metronomic-to-32 ms dispersion ladder at 30 and 60 fps); docs (`regularity_track.md` with the precedent mapping and the MDI table, RUNBOOK §13, README).

**Invariants G-a..G-d, each a test:** G-a `test_no_second_interval_statistics_implementation_in_heads_or_inference` + `test_the_allowlist_names_only_functions_that_exist` + `test_the_v1_raw_series_path_is_gone`; G-b the head and reference tests asserting index + CI on every class; G-c `benign_pattern_evidence` membership on every irregular read; G-d `test_head_never_names_a_rhythm` + the invariant-11 fence on `REGULARITY_SENTENCES`; the standing invariants by `test_regularity_equivalence` (golden), `test_successive_difference_statistics_never_cross_a_run_break`, and `test_sqi_never_scores_periodicity_and_cannot_reach_the_substrate`.

**Adversarial review (12 dimensions, three refuters per finding; every confirmed finding fixed with a pinning test — recorded here so the fixes are attributable):**
14. *The tachogram is built from the longest clean run only.* The v0.6 tachogram bridged run seams shorter than 3 s by interpolation, which at ordinary heart rates covers every single-beat run break — synthesized interval values inside the very seam the run discipline exists to exclude. Consequence: a scan whose longest run is under 15 intervals or 20 s gets no tachogram, hence no coupling, hence the head abstains; flutter's spectral concentration and coupling numbers move on multi-run clips (golden exemption, see #15).
15. *The golden corpus was single-run.* Every clip yielded one clean run, so the golden could not see the within-run vs cross-seam pooling the refactor is about, and no processed clip abstained. Four clips were added (pulse-deficit AF and sinus, a shaky clip, a dim-but-processed clip), captured on the pre-refactor commit in a worktree; the AFib decision's outputs are pinned bit-for-bit on all of them, and only two research-side outputs (`head_irregularity`, #5, and the flutter extractor's tachogram numbers, #14) are exempt, on multi-run clips only, by name and with this citation.
16. *`respiration_coupled` needs two signatures and a regular residual.* The first rule awarded it on the tachogram's respiratory-band fraction alone, so trigeminy whose 3-beat period sat in the band read as breathing, RSA with a 1.5 br/min rate-estimate error read as ectopy, and AF with 20 % respiratory modulation read as breathing. Now: the breath must claim ≥ 50 % of the tachogram's power on evidence the tachogram MOVES WITH the breath (phase-locking value ≥ 0.85 against the respiration waveform; rate-error-robust, and the waveform exposes a mis-estimated rate), AND the residual after removing that power — estimated as 0.95·√(1−fraction)·(tachogram σ/median) — must itself be regular by the published threshold. Known limitation, pinned by test: an ectopic pattern whose period equals the breathing period within the window's resolution AND holds a stable phase is indistinguishable from RSA by all three signatures.
17. *The bootstrap CI is labelled by its MEASURED coverage.* The review found the iid bootstrap of the 1-dependent successive differences covered ~90 % at a nominal 95 %. A moving-block bootstrap (blocks ~n^(1/3)) was substituted and the coverage RE-MEASURED: 0.88 at n = 15, 0.90 at n = 30, 0.94 at n = 70 — identical to the iid bootstrap. The under-coverage is the percentile interval of a median at small n, not the resampling unit, and no bootstrap variant fixes it at n = 15. The interval therefore carries its method and its measured coverage in the payload (`index.ci_method`, `index.ci_coverage_measured`), the doc states them, and a test pins the coverage floor so the label cannot drift from the behaviour. Not hidden behind "95 %".
18. *The class is decided on the unrounded statistic.* The index was rounded to five decimals before the threshold was applied; 0.059996 became 0.06 and irregular.
19. *The ECG reference covers the span the camera saw.* R-peaks are mapped onto the video clock (video_t = ecg_t + sync offset) and clipped to the video's duration before the reference is computed; a longer ECG record was being compared against a shorter camera series.
20. *κ is undefined at expected agreement 1.* A single-class cohort (no irregular scan on either side) returned κ = 1.0 and opened R0; it now reads "no class agreement on record". A PAIR is a scan judged on both sides — 99 no-read scans plus one agreeing pair no longer opens R0 on n = 100 — and a SQI grade whose κ is undefined makes the worst-grade κ unrated rather than skipped. The Bland–Altman bias is gated on its own (`bland_altman_bias_max`): a tight systematic offset passed the half-width alone.
21. *The noise floor uses ACCEPT scans as references, nominal fps classes, the camera's own dispersion, monotone detection, and no-reads as non-detections.* Rejected scans had fed the floor; a measured 29.97 fps fragmented cells nothing could fill; the measured gain absorbed the reference's own RMSSD (now removed in quadrature); the MDI was the first bin reaching power even when higher bins failed; a scan the camera could not read vanished from the power denominator; and the pre-registered `mdi_power`/`mdi_confidence` were never read — they are now read, recorded on the floor run, and checked by R1, which also refuses a floor from a different signal domain than the evaluation.
22. *Abstention is disclosed and bounded (R2, R3).* Every R2 denominator is reported total / judged / no-read, `min_rsa_sessions` and `max_no_read_rate_per_age_band` are pre-registered, and the recording's rhythm label follows the schema's salience precedence rather than annotation index 0. R3 re-scores every baseline on the rows the head judged (like with like), gates `max_head_no_read_rate`, counts judged participants, and chooses each logistic baseline's operating point on the train half for balanced accuracy instead of a fixed 0.5 that collapsed to the majority class under imbalance.
23. *Gates read booleans and bounded numbers.* `"false"` had satisfied disjointness by truthiness; −∞ beat every margin; `True` read as κ 1.0; an empty pre-registered list made R4 vacuous; `require_mdi_per` was read into an error string only. Evidence recorded under another `gates_version` or reference label reds every gate as stale, and the registry door re-evaluates the run's recorded `evidence.json` under the LIVE gates.yaml (signoff included) instead of trusting `gate_results.json`.
24. *The public surrogate detects per finite segment on the real clock,* with sub-sample R-peak refinement (integer-sample peaks at 125 Hz made the median |ΔRR| of a regular heart exactly 0), marks a window without a clean run `NO_RESULT`, and shares the one `runs.min_conf` resolution with the pipeline, the reference and the beat-error study.
25. *The fixtures' skin-tone label has an optical counterpart.* Each Fitzpatrick group now sets the clip's skin base colour with the pulse amplitude scaled by its green reflectance (a crude melanin model, disclosed); group III reproduces the historical generator bit for bit. Group, frame rate and site are no longer confounded; ages span the bands wherever physiology allows (RSA stays age-specific, now in three cohorts); the ladder generates `reps` clips per (fps, rung, group) for two groups so an MDI cell can actually fill.
26. *The G-a audit carries a budget.* Alias-resolved numpy calls, hand-written slice differences and an extended statistic list are attributed to the INNERMOST function, and every allowlisted function is pinned to the exact count of each banned call it makes; `head_flutter` now passes the pipeline's representation through, and flutter's spectral concentration is composed from the shared band-fraction machinery.
27. *The jitter budget is the rate-averaged iid model, stated as such.* One refuter showed that naive round-to-frame picking of a metronome is a rate-dependent sawtooth (0 ms on the frame grid, a full frame period half-way). The pipeline does not naive-pick — beats are located with sub-frame interpolation — so the measured gain, not the budget, is what the MDI rests on; but the gain is calibrated at one pulse rate (70 bpm), and a rate sweep of the metronomic rungs is recorded as a follow-up in the doc rather than asserted away.

**Measured before believed (synthetic video through the production path, 132 scans: 8 cohorts × 6 + a 7-rung metronomic-to-32 ms ladder × 2 fps × 2 skin-tone groups × 3; and MIMIC PERform AF contact PPG as a `public_ppg` surrogate, 210 windows / 35 subjects):**
- **The ceiling on synthetic video is high but not at the threshold.** Camera-vs-ECG κ **0.92**, index r **0.98**, bias +0.0003, 95 % limits ±0.062 (R0 asks ≤ 0.04). 127 of 132 scans agree; four of the five disagreements are `rsa_mid` clips whose ECG index sits at 0.070–0.072 and whose camera index reads 0.053–0.059 — on ECG-irregular clips the camera index reads **below the ECG's** (median camera/ECG ratio 0.86, IQR 0.80–1.01; ≈ 0.8 on the three RSA cohorts) while regular clips carry a small positive jitter bias (+0.007 median), so the overall bias cancels and the limits of agreement widen. On the contact-PPG surrogate the same ratio is 1.03 (IQR 0.95–1.17), so the compression is a property of the camera path or of the synthetic optics, not of the statistic. The published threshold is defined on the ECG scale; whether the camera index is mapped onto it (a calibration, which would need its own validation) or the threshold is re-set for the camera is a clinical-signoff decision, flagged here, not taken.
- **B3 beats the head on the fixture — as R3 is built to expose.** On the head's judged held-out rows: head 0.917, B1 (RMSSD threshold chosen on train) 0.936, **B3 0.977**, B4 0.61, B5 0.54; head no-read 0.018. A logistic rule fitted on the train half learns the camera's compressed scale; the head applies the pre-registered ECG-scale threshold. On real data this is the number that would keep R3 red until the scale question above is settled.
- **RSA is separated on the fixture, and the fixture is too easy to say more.** 18 RSA sessions, 0 flagged as clinically irregular, 14 explained as respiration-coupled (the other 4 read regular on both sides); benign specificity 1.0 in every age band — but the <35 band holds 9 sessions (R2 asks 20), so the band is unrated, not passed.
- **The MDI table is mostly "not characterized" at 30 fps.** The measured index floor on metronomic references is 0.0222 at 30 fps and 0.0154 at 60 fps. At 30 fps no ladder bin up to 32 ms reaches 80 % power in any group (the 20–30 ms bin detects 0.50 light / 0.33 dark); the first characterized bin is the cohort bin above 50 ms. At 60 fps the light-skin group characterizes at **31.9 ms** RMSSD; the dark-skin group detects only 0.60 in the 20–30 ms bin and falls to the >50 ms bin. Frame rate and skin tone both move the floor, in the direction the crude melanin model predicts.
- **The measured interpolation gain is rate-dependent, and the v0.6 number was the easy case.** On metronomic references at 70 bpm the camera-only RMSSD is 19.2 ms at 30 fps (gain **0.81** of the raw 23.6 ms budget) and 10.7 ms at 60 fps (**0.91**). v0.6 measured 15.0 ms at 30 fps on a 150 bpm clip — where the pulse period (400 ms) is exactly twelve frames and naive rounding would already be error-free. The planning gain 0.65 stays a planning value; the floor scoreboard carries the measured one, and a rate sweep is the recorded follow-up (#27).
- **Beat errors are contained by the run discipline, not by the index.** Clean-run false-irregular rate 0.00 through 30 % beat error at 30 fps (0.10 at 40 %); on the raw series the RMSSD is already 83 ms at 2 % error (17 ms clean) and the index tips past the threshold at 30 %.
- **On human hearts (contact PPG surrogate, can never open a gate):** κ **0.87**, index r **0.76**, bias +0.023, 95 % limits ±0.136 over 210 windows. All 114 AF windows read irregular on both sides (κ undefined within AF, single class — reported as such, not as 1.0). Within non-AF, 12 windows read irregular on the PPG that the ECG calls regular, and one the reverse (κ 0.10 within the subset) — the surrogate over-reads irregularity on ICU contact PPG, the direction a beat-detection error produces. The head abstained on 87 of 114 AF windows (impedance respiration undecodable, concentration ≈ 0.01) and on 39 of 96 non-AF windows — no-read 0.78 on the held-out split, so its 0.95 balanced accuracy on 3 participants is a number to disbelieve, and R3 would red on the no-read rate alone.
- **Fairness on the fixture is confounded by the fixture.** The ladder runs in two skin-tone groups only, so those groups carry 84 regular clips and the others carry cohorts alone; the flag-rate parity ratio (0.16) compares different rhythm mixes and is machinery evidence, not a finding. The worst index bias (−0.054, group IV) is one rejected bigeminy scan whose camera index (0.38) is half the ECG's (0.76) — the camera missed part of the coupled beats.

**§R scoreboard:** all six gates RED — every gate first on the qualification rule (`evidence domain 'synthetic' is not 'facial_rppg' — machinery evidence only, can never open a gate`), and R0 (limits of agreement), R2 (<35 band under 20 sessions), R3 (head does not beat B3), R4 (parity, bias) and R5 (no signoff) also on their own numbers. Nothing renders.

**Escalations:** none refused. Would have been refused and recorded: rendering an irregularity verdict while §R is red; naming a rhythm from this head; dropping R2; letting this head escalate to an AFib sentence; regenerating the golden without a spec entry.

**Definition of done, checked:** suite green; head_afib bit-identical on the eleven-clip golden (four multi-run / abstaining clips added, captured on the pre-refactor commit); no duplicate interval-statistics implementation (AST audit with budgets); `regularity-floor`, the ceiling test, the head, the baselines and `gate-status --track regularity` run end to end on synthetic fixtures and on the public surrogate; scoreboard, MDI table, age strata and escalations in the final summary.

## B.24 v0.8 changelog — the Research & Investigation Report (opened 2026-09-01)

**Owner instruction (verbatim)**: "Provide all VO2 max, atrial flutter, arterial stiffness, vascular tone, rhythm-regularity in the report for research and investigation purposes only."

**Conflict, flagged, not silently resolved.** Every one of those five tracks carries a standing escalation rule in this spec — B.19 §V, B.20 vascular, B.21 §W, B.22 §F, B.23 §R — that an instruction to SHOW the track's output while its gates are red is refused and recorded verbatim. Read literally as "put them in the Cardiac Rhythm Scan Report", the instruction is exactly that, and that reading IS refused: the consumer report, the browser payload, `cli.py process` stdout and `ScanResult` are unchanged, and every test that fences them still passes. Read as what the sentence says — a document for research and investigation, not for a participant — it is inside the rules the same spec already grants: watermarked research artifacts (scoreboards, evaluation JSON, the vascular and vasotone research reports of v0.4/v0.5, the reconstruction report) exist precisely so investigators can see what participants cannot. That is what was built. The owner may overrule either half of this reading; both are recorded here.

**Flagged interpretations (owner may overrule; none silently resolved):**
1. *"The report" is a NEW document, not the consumer one.* `AvatarX Research & Investigation Report` (`research/investigation/`, `cli.py research-report`) is a second document with its own title, its own watermark on every page and every section, and no shared renderer with `app/report_render.py`. Nothing under `research/` is importable from `app/` or `inference/` (quarantine walker); `app/server.py` has no route to it; the consumer payload filter is untouched (pinned by test).
2. *Every track appears, always — as a value, an abstention with its reason, or "not applicable" with what it would need.* A single ACCEPT scan yields the regularity index/CI/class/benign evidence, the flutter pattern families, and the raw vascular morphology features (an estimate only when a trained model is on record — today none is); it cannot yield a VO2 category (three-phase recovery session required — a `--session` manifest is accepted and run) or a vasomotor reactivity (timed provocation marks required — `--provocation` is accepted). A track that cannot run says so in the report rather than vanishing, and the four statuses are kept apart: `value`, `abstained` (the head ran and declined, in its own shape — the fitness head declines by a `category` of None), `not_run` (the head never executed; the section carries the scan's or session's reasons, never invented head reasons) and `not_applicable`.
3. *Gate status rides with every value.* Each section carries the track's LIVE `gate-status` (promotion BLOCKED/OPEN, gates version, every gate's colour and first reason) read at generation time, so the number and the reason it may not be shown to anyone are never separated.
4. *Rhythm names may appear in this document.* The heads' raw payloads carry internal evidence labels (`ectopy_pattern`, band names, `flutter` in the head's own name). G-d / F-b forbid rhythm naming on USER surfaces; this is not one. The token tests continue to scan `app/`, the consumer payload and the sanctioned strings, and the new document is pinned OUTSIDE every scanned surface by a test that also asserts it is never reachable from them.
5. *No sanctioned consumer sentence is generated here.* The document quotes the consumer result (outcome, stars, the sentence the participant saw) as CONTEXT for the investigator, verbatim from `ScanResult.user_facing_text()`; it adds no wording of its own about the participant's health.
6. *Written under `research/runs/investigation/<recording_id>/`* (gitignored) as JSON + HTML, watermark first; stdout of the verb is the same JSON, watermark first. It is never written by `cli.py process`, `cli.py session` or the live demo.
7. *One extractor fix rode along, recorded here because it changes research-side behaviour.* `features/flutter.py::hyper_regularity` reused the coupling verdict of a representation handed in through context even when that representation had been built without a respiration channel (the pipeline builds it that way), so `head_flutter` would abstain on any scan whose caller handed it the pipeline's representation together with a respiration channel — latent since 09dc0b0 (no caller did both), first reached by this report. A supplied respiration channel now overrides a stale no-channel verdict; a representation with no coupling entry still falls back to measuring, as at HEAD. No gate, threshold, head contract or consumer surface moved; the eleven-clip golden is unchanged (the golden's flutter capture never handed a representation in); pinned by `test_head_measures_coupling_against_a_supplied_channel_even_with_a_pipeline_representation`, which asserts the head DECIDED (flag False on an RSA clip) rather than merely not abstaining for that reason. Two more review findings fixed in the same pass: the regularity head's rebuild-with-channel path now keeps the run confidences (the jitter budget carried no confidence penalty); the research verb reduces a capture manifest exactly as `process` does, validates every file argument up front (exit 2 with the reason, never a traceback) and prints the document even when `--out` cannot be written (exit 2).
8. *Measured on the first document (synthetic RSA clip with a breathing torso, 40 s, 30 fps):* regularity read a value (index 0.055, CI 0.029–0.056, class regular, coupling available); flutter abstained before the fix and reads its families after it; arterial stiffness abstained — "no trained vascular model on record", which is true; vascular tone and fitness were not applicable with their needs stated; all five promotions BLOCKED, twenty-nine gate rows RED (6 + 6 + 6 + 6 + 5). The consumer report on the same clip is byte-identical to what it was.

**Owner instruction, second (verbatim, with a screenshot of the results screen)**: "Per my earliest request, I still don't see VO2 max, atrial flutter, arterial stiffness, vascular tone, rhythm-regularity in the report for research and investigation purposes only in the report. See image attached. Provide the aforementioned values in the report."

**Flagged interpretations, second round (owner may overrule; none silently resolved):**
9. *"The report" was the RESULTS SCREEN, not the CLI document.* The screenshot shows the app's scan report followed by its `RESEARCH / DEBUG METRICS` panel — a surface that has existed since v0.1, is labelled "not validated outputs, not for participants, excluded from the printed report", and is hidden by print CSS. Items 1-8 above built a separate CLI document, which is not where the owner was looking. The five tracks are therefore now ALSO in that panel. The CLI document stays: it is the only surface that can carry the two tracks `app/` cannot compute.
10. *The panel block is OFF by default* (`report.research_tracks: false`, env `AVATARX_RESEARCH_TRACKS`, explicit env winning in both directions per the v0.4 lesson). This is the one place this entry departs from a literal reading of the instruction, and it is the point of the five standing escalation rules (B.19 §V, B.20 vascular, B.21 §W, B.22 §F, B.23 §R): while a track's gates are red its output may not sit in front of a participant. "For research and investigation purposes only" is read as exactly that — present when an operator asks for it, absent on a participant's scan. Default-on is a one-line change to `configs/default.yaml` and is the owner's to direct.
11. *Four fences, none moved.* (a) Screen-only: the printed and exported Cardiac Rhythm Scan Report never carries the block, pinned by test. (b) Its own payload key `research_tracks`: `datasets.schema.public_head_results` and `head_results` are byte-for-byte what they were. (c) Numbers, labels and the heads' own reasons — no sentence about the person; `user_facing` stays None on every head value and participant wording still comes solely from `user_facing_text()`. (d) Each value rides with its track's LIVE gate status, read fail-closed, so a number and the reason it may not be shown are never separated.
12. *The client page names no track and no rhythm.* `researchTracksHtml()` is generic: titles, labels, gate rows and reasons all arrive in the payload. The §F/vascular/§W token fences scan `app/static/index.html`, and this is not an evasion of them but the reason they can stay honest — a test pins that the fenced tokens are still absent from the page and that the pre-existing "irregularity index" HRV label count is unchanged. The tokens appear at runtime, in an operator-enabled block, which is the behaviour the owner asked for.
13. *One status classifier, not two.* `heads.base.head_status` now owns the "did this head decide or decline" judgement (each head declines in its own shape; the fitness head by leaving `category` None), and both research surfaces — the panel in `app/` and the document in `research/` — call it. Two implementations would drift, which is the P2 lesson this programme already paid for once.
14. *What the panel cannot compute, it says.* `app/` cannot import `research/` (quarantine walker), so arterial stiffness (needs a trained model handed in by the research surface) and vascular tone (needs timed provocation marks) are reported in the panel with their gate status, what they need, and the exact `cli.py research-report` command. Cardiorespiratory fitness reads this session's own head when the three-phase session ran, and otherwise says what it needs. Measured on a synthetic RSA clip with a breathing torso: rhythm regularity `value` (index 0.055, CI 0.029-0.056, class regular), atrial flutter `value` (flag False — it looked and found no sustained pattern), the other three stating their needs; all five promotions BLOCKED, twenty-nine gate rows RED.

**Owner instruction, third (verbatim)**: "I see no change in the report. Is there any issue with gates blocking the aforementioned bio markers to not show up in the report?"

15. *The gates were never what withheld them; a default-off flag was, and it is now ON.* The answer to the owner's question, on the record: no gate suppresses these values anywhere in this design. `research_tracks_block()` prints each track's value BESIDE its live gate status precisely so the number and the reason it may not be shown to a participant are never separated; all five promotions read BLOCKED and every track still renders. What withheld them was `report.research_tracks: false`, the operator flag added in item 10 — a caution of this implementation's own making, not a rule of the programme. It now defaults to **true**: the five tracks are in the results screen's RESEARCH / DEBUG METRICS panel on an ordinary scan. `AVATARX_RESEARCH_TRACKS=0` turns them off for a participant-facing session. Item 10's reasoning is left standing above rather than rewritten, so the reversal and its cause are both legible.
16. *What did NOT move with that default.* The panel has carried the label "not validated outputs, not for participants, excluded from the printed report" since v0.1 and is `display:none` in print CSS, so the Cardiac Rhythm Scan Report — the document a participant is handed — still carries none of this (pinned by test). The values ride their own `research_tracks` payload key; `datasets.schema.public_head_results` and `head_results` are byte-for-byte unchanged (pinned). No sentence about the person is generated: `user_facing` is None on every head value and participant wording still comes solely from `ScanResult.user_facing_text()`. The one behaviour that changed is that an ungated research payload now reaches the browser by default on the prototype's results screen, which is what the owner directed three times and is recorded here as their decision.
17. *Pinned end to end, not by construction.* `tests/test_scan_engine.py::test_research_tracks_ride_the_browser_payload` drives a real `ScanSession` over a synthetic clip and asserts the five tracks are in `session.result` with their gate rows RED, the regularity index present, the participant's `report_html` free of the watermark, and every `head_results` row still non-RESEARCH. Three earlier attempts asserted the block function in isolation; only this one tests what the owner was actually looking at.

**Built (tests first):** `research/investigation/` (`WATERMARK`, `TRACKS`, `build_investigation_report`, `render_investigation_html`, `write_investigation_report`), `cli.py research-report`, `docs/investigation_report.md`, RUNBOOK §14, README; `tests/test_investigation_report.py` pins the quarantine (no import path from app/, inference/, heads/ or datasets/; no server route; the consumer renderer and `cli.py process` know nothing of it), the watermark-first document with all five tracks, values and reasons on the single-scan tracks, stated needs on the protocol tracks, the quoted-never-authored consumer context, the HTML (watermark on every section, no waveform, no ECG-paper mimicry, nothing red), the untouched consumer report on the same scan, and the CLI verb end to end.

## B.25 v0.8 changelog — resting hemodynamics from one scan (opened 2026-09-02)

**Owner instruction (verbatim)**: "Ensure Arterial stiffness, vascular tone and VO2 max can be computed from one resting scan in the app. The goal is present users with values and we will capture data, train model and improve accuracies as we move forward. It's imperative we offer values"

**What was built.** `features/hemodynamics.py::resting_hemodynamics(det, outcome=..., participant=..., capture=...)` computes three families from a SINGLE ACCEPT-grade resting scan, and all five research tracks in the results-screen panel now read `value` rather than "needs a protocol". Every family is UNCALIBRATED by construction and carries `calibrated: False`, the definition it was computed from, and what a validated number would require.

**Flagged interpretations (owner may overrule; none silently resolved):**
1. *Two of the three asked for are real single-scan measurements; the third is not, and the difference is stated rather than papered over.* Arterial stiffness and vasomotor tone have single-waveform physics behind them and are now measured. An oxygen-uptake value does not: from a resting scan the camera contributes resting heart rate and its variability, while every published non-exercise equation is dominated by age, sex and body composition. Emitting one would report demographics as a camera measurement, and the programme's own hard rule (B.18/B.19) is that no oxygen-uptake number renders in any version. That rule STANDS. What is offered instead is real and is offered on every scan: resting heart rate, RMSSD, SDNN, and a bounded `autonomic_index` built from the two quantities a camera can measure, named after what it is rather than after a quantity it has not been validated against.
2. *Arterial stiffness is reported as CONTOUR MARKERS, never a velocity.* `reflection_index` (diastolic-peak height over systolic-peak height) and the second-derivative `aging_index` = (b − c − d − e)/a (Takazawa et al. 1998) are direct formulas over fiducials this repository already extracted for the V0 fidelity study. They need no model and no training set, they are dimensionless, and they move with stiffness. A carotid-femoral velocity in m/s still needs the referenced study the vascular gates describe, and the payload says so in `not_a_velocity`. This is the honest half of the request delivered now; the m/s number is what the captured data is for.
3. *Vasomotor tone is reported as RESTING indices, never reactivity, and nothing absolute.* Per-beat pulse amplitude is normalised by each ROI's own median before pooling, so the two outputs — `amplitude_cv` and `vasomotion_index` (power fraction in 0.04–0.15 Hz) — are dimensionless. That keeps v0.5's W-c ("nothing absolute") intact while still yielding values, because absolute amplitude is the most optics-confounded quantity in this pipeline. A capture without exposure and white-balance lock is flagged in `caveat` rather than refused, and `not_reactivity` states that a response to a timed provocation is a different measurement.
4. *One implementation of the contour math, not two.* The per-beat and ensemble morphology moved verbatim from `research/vascular/features.py` to `features/pulse_morphology.py`, with the research module reduced to an adapter that re-exports it. `app/` and `inference/` are quarantined from `research/` and could not otherwise compute a contour at all. This is the v0.7 regularity refactor applied again: the V0 fidelity study's numbers are unchanged by the move, and the vascular and vasotone suites pass untouched.
5. *A real capture limitation, disclosed rather than smoothed.* The second-derivative waves need genuine sampling rate. Measured on this repo's own fixtures: at 30 fps `sdppg_derivable` is false and `aging_index` is None; at 60 fps it is −0.14 on the same source. The index is never interpolated into existence at consumer frame rate, so on a 30 fps device the stiffness readout is the reflection index alone. Capturing at 60 fps is the difference between one marker and two.
6. *"Present users with values" was delivered on the RESEARCH panel, not the printed report.* The values are in the results-screen RESEARCH / DEBUG METRICS panel, which has carried the label "not validated outputs, not for participants, excluded from the printed report" since v0.1 and is hidden by print CSS. The Cardiac Rhythm Scan Report — the document a participant is handed — still carries none of it, and `public_head_results` is untouched. Moving any of these into the participant's document is a further decision with a regulatory shape, and it is the owner's to direct explicitly; it was not inferred from this instruction.
7. *The training path the instruction describes is what the payload is shaped for.* `cardiorespiratory_fitness.model_artifact` is a reserved slot, `missing_for_a_fitted_model` names exactly which demographic fields a scan lacked, and every family reports `calibrated: False` beside the reference points it used. Nothing here has to be rewritten to become calibrated: the fitted coefficients replace named planning constants (`HR_REF_BPM`, `HR_SPREAD_BPM`, `RMSSD_REF_MS`) and the estimate fills the reserved slot.

**Measured on this repository's fixtures (synthetic sinus, 45 s, one resting scan, exposure and white balance locked):** at 60 fps all five tracks read `value` — aging index −0.143, amplitude CV 0.128, vasomotion index 0.357, resting pulse 70.7, RMSSD 29.8 ms, autonomic index 0.356, with regularity and the flutter flag as before. At 30 fps the aging index is correctly None and the reflection index carries the stiffness readout alone. All five promotions read BLOCKED and every gate row is RED beside the values.

**Pinned by test:** `tests/test_hemodynamics.py` (13 tests) covers the Takazawa formula and its refusal on any missing component, the vasomotion band's selectivity at 0.1 Hz versus 0.35 Hz, the per-ROI self-normalisation that removes optical scale, the autonomic index's bounds and monotonicity in both inputs, the absence of any oxygen-uptake number and the presence of its reason, the 30 fps second-derivative refusal, and the non-ACCEPT refusal. `tests/test_research_tracks_panel.py` additionally pins that no sanctioned consumer sentence and no second-person wording appears anywhere in the block.

## B.26 v0.1.6 changelog — real-world capture hardening (owner-directed, appended 2026-09-02)

**Owner instruction:** pressure-test and harden the scan and biomarker pipeline for realistic smartphone capture, recover where possible, and otherwise use explicit `READY` / `REPEAT_SCAN` / `NO_RESULT` behavior without forcing biomarker values from unusable signals.

**Behavior change.** ROI extraction now rejects malformed/partial geometry and trims within-ROI shadows and highlights without a skin-color threshold. Tracking resets stale smoothing after loss and includes raw center/size jitter and detection coverage. File and live capture validate exact, finite, strictly increasing timestamps; detect dropped, collapsed, and duplicated frames; measure facial clipping, darkness, and uneven illumination; preserve capture gaps rather than filling them with stale geometry; and count only fully gated evidence toward scan duration. Biomarkers require endpoint-usable SQI, tracking, and independently verified cross-region pulse evidence. A rhythm-specific abstention may coexist with a research estimate only when those endpoint inputs survive; otherwise all three biomarker values remain null with a reason.

These upstream extraction changes intentionally move deterministic waveform, beat-time, SQI, and derived regularity numbers on the synthetic equivalence corpus. The corpus was regenerated under the final implementation only after verifying that every pinned outcome, predicted class, confidence grade, and user-facing disposition remained unchanged. This is not a gate loosening: unusable signals now abstain more often, and clean synthetic recovery remains pinned separately. The added timestamp `diff` calls inspect frame delivery only, not inter-beat variability; the regularity AST budget records them by function and count.

**Evidence boundary.** The regression matrix in `docs/REAL_WORLD_HARDENING.md` reproduces each requested failure family and names its automated test. Those fixtures establish software behavior, not clinical accuracy or population fairness; representative phones, lighting conditions, skin tones, and paired ground truth remain required before calibration or clinical claims.

## B.27 v0.1.6.1 changelog — restore the easy-start scan experience (owner-directed, appended 2026-09-02)

**Owner correction:** the hardened build made the user wait for downstream pulse evidence before the countdown and exposed `REPEAT_SCAN` on the pre-scan screen. That inverted the intended product behavior: the scan should remain as easy to begin as the previous version, with motion/illumination handling and quality validation occurring during and after capture.

**Resolution.** The browser now starts from advisory `ready` permission (face, framing, hard camera floor) and no longer requires the stricter result disposition. Result-state terminology is removed from the camera screen. Capture faults may pause good-time recording and recover, but provisional pulse SNR, cross-region coherence, beat timing and preliminary beat count are guidance rather than prerequisites for completing the scan. The final production pipeline and biomarker quality gate remain unchanged: an unsupported complete scan returns `REPEAT_SCAN` or `NO_RESULT`, with no fabricated biomarker value.

## B.28 v0.1.6.2 changelog — retain endpoint-usable values on a low-confidence rhythm scan (owner-directed, appended 2026-09-02)

**Owner correction:** a completed scan in decent lighting still showed all three resting biomarker cards as "Not computed". Replaying the retained browser capture found 18 morphology-usable beats in all four facial ROIs, SQI 0.421 and tracking stability 0.921. The rhythm decision correctly returned `REPEAT_SCAN` because whole-scan cross-ROI coherence was 0.17, below its 0.20 floor; `resting_hemodynamics` incorrectly reused that rhythm cutoff as an all-or-nothing endpoint gate and erased otherwise computable research features.

**Resolution.** Rhythm classification thresholds are unchanged. The endpoint gate now also admits an explicitly LIMITED path when cross-region coherence is at least 0.10, best-pair timing precision is at most 40 ms and at least 35% of beats match, after which the existing independent requirement for at least eight beats in at least two morphology-readable facial ROIs still applies. Such values retain the scan's `REPEAT_SCAN` outcome, 1–2-star confidence, limiting factor and a paired-data-only warning; no rhythm conclusion is promoted. Evidence below any of those limited floors still returns null values visibly. The API, desktop/mobile HTML and debug payload expose which endpoint evidence mode fired and its thresholds, and abstentions now retain the intended calculation method plus the failed signal gate rather than displaying "method unavailable".

**Measured on the retained browser scan:** the limited path yields reflection index 0.539, normalized pulse-amplitude variability 41.48% CV, and resting cardiorespiratory-fitness proxy 68.7/100 from 18 beats across four facial regions. These are uncalibrated `Research Estimate / Prototype` values; the 29.99 fps capture does not support a second-derivative aging index, unlocked optics are disclosed on amplitude variability, and no oxygen-uptake value is fabricated.

**Pinned by test:** `tests/test_hemodynamics.py` now covers the real-capture evidence margin, rejection immediately outside each limited floor, preservation of three numeric values on a `REPEAT_SCAN`, and method/confidence disclosure on abstention. Existing clean-scan and hard-invalid tests remain unchanged and green.
