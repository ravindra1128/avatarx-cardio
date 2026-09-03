# Claude Code prompt — AvatarX AFib v0.1

<!--
HOW TO USE
  1. Unzip afib-face-scan-v3.zip so this file sits NEXT TO the afib-face-scan/ directory
     (or place this file inside the repo root — either works).
  2. cd into the repo and start Claude Code:  claude
  3. Paste everything below the line, or say:
     "Read CLAUDE_CODE_PROMPT_v0.1.md and execute it."
-->

---

You are building **AvatarX AFib v0.1** — the first runnable version of a contactless
atrial-fibrillation screening research pipeline that goes from a **face video file**
to a **four-class rhythm result with abstention**, plus the evaluation harness that
gates it against ECG ground truth.

## Step 0 — Preflight (do this before writing any code)

1. Confirm the repository `afib-face-scan/` exists with its v3 core. Run:
   ```
   pip install numpy pytest
   python3 -m pytest tests/ -q        # MUST report: 41 passed
   python3 scripts/demo_pipeline.py   # must run to completion
   ```
   **If the repo is missing or tests fail, STOP and report — do not scaffold a new
   repo from scratch.** The core (schema, splits, beat detector, clean runs,
   feature engine, beat/AFib metrics) is done and tested; your job is to extend it.
2. Read `CLAUDE_CODE_SPEC.md` in the repo root. **It is the authority.** If anything
   in this prompt seems to conflict with it, the spec wins — flag the conflict in
   your final summary rather than silently choosing.
3. `git init` if needed, commit the pristine baseline as `v3-baseline`, then work in
   small commits, one per task below. Run the full test suite before every commit;
   **the existing 41 tests must stay green at every commit.**

## What v0.1 IS

An **offline research pipeline + CLI**, pure Python, that a research assistant can
run on a laptop against recorded files:

```
python3 cli.py validate  <recording.json>              # capture/sync gates, with reasons
python3 cli.py process   <video> [--manifest rec.json] # → ScanResult JSON (never a diagnosis)
python3 cli.py evaluate  <dataset_dir>/                # → Gate-1/1b report, risk-coverage,
                                                       #   no-read parity, leakage audit
python3 cli.py demo                                    # existing synthetic end-to-end demo
```

## What v0.1 IS NOT (do not build any of this)

No mobile app. No deep-learning training (Models B/B′ are v0.2). No rBCG
(patent-encumbered; ablation-only later). No cloud services, no UI beyond the CLI.
No downloading of rPPG-Toolbox checkpoints (RAIL licence restricts medical
diagnosis use). No foundation models. No demographic features into any classifier.
No attempt to "improve" the clinical output wording.

## Non-negotiable invariants (violating any of these is a bug)

1. **Never emit a diagnosis.** Every user-facing string flows through
   `ScanResult.user_facing_text()` unchanged. The existing test enforces this;
   do not weaken it.
2. **Fail closed.** Invalid capture, invalid sync, < 5 clean intervals, or NaN in
   any gate input → `NO_RESULT` with reasons — never a guess. NaN fails gates.
3. **The gated path is the production path.** Any metric you report must be
   computed by the same code path `cli.py process` runs (this repo's P2 lesson:
   a matched-pair harness passed while the raw production path read sinus as AF).
4. **Features run on clean runs only.** Always `beats/ibi.py::clean_runs()` →
   `features/rhythm.py::compute_rhythm_features_from_runs()`. Never compute
   dispersion statistics on a raw detected beat series (one undetected missed
   beat multiplies sinus RMSSD 4.6×).
5. **No interval repair.** Suspected missed/false beats break runs; never
   interpolate over them — in AF the "outlier" interval is the signal.
6. **The SQI must never measure periodicity.** AF is aperiodic; an
   autocorrelation-peak quality score blocks exactly the patients being screened.
   The anti-periodicity test (Task 4) is the acceptance criterion.
7. **Participant-level splits only**, via the existing `datasets/splits.py`.
   Never window-level or recording-level splitting, anywhere, including in
   quick experiments.
8. **Permissive dependencies only**: numpy, scipy, opencv-python-headless,
   mediapipe (optional, with fallback), pytest. Nothing GPL/RAIL. Record every
   added dependency and its licence in `README.md`.

## Tasks, in order. Tests first for every task.

For each task: write the failing tests, implement, run the FULL suite, commit.
If an acceptance criterion proves unachievable, do not lower the threshold —
mark the task blocked in your summary with the measured number.

### T1 · Synchronisation — `datasets/synchronization.py`

PRBS LED-marker sync: detect a known pseudo-random flash sequence in a video's
mean-brightness trace, cross-correlate against the marker event times recorded in
the ECG file's event channel, output a filled `SyncRecord` (offset_ms,
sync_uncertainty_ms, drift_ppm, n_marker_events, verified_at_end).

- `detect_marker_events(brightness: np.ndarray, fps: float, template: np.ndarray) -> np.ndarray`
- `estimate_sync(video_events_s, ecg_events_s) -> SyncRecord`
- Physics constraint honoured: a single flash is bounded by ±half a frame;
  only a ≥10-event sequence may claim < 5 ms (the schema gate already enforces
  `n_marker_events >= 10` — your job is to produce honest uncertainty numbers).
- **Accept:** on synthetic videos with known offsets (0–500 ms) and drifts
  (0–50 ppm) at 30 and 60 fps: recovered offset error < 2 ms, drift error < 5 ppm,
  and the returned `SyncRecord` passes `is_valid_for_beat_analysis()`.

### T2 · Beat-confidence calibration — `beats/confidence.py`

The fusion detector's confidences are honest but compressed (demo ECE 0.42 vs the
0.10 gate). Fit a monotone calibration (isotonic or Platt) mapping raw fused
confidence → P(beat is real), using ECG-matched labels from the synthetic suite.

- `fit_confidence_calibration(matches: list[BeatMatchResult], confidences: list[np.ndarray]) -> Calibrator`
- `Calibrator.apply(series: BeatSeries) -> BeatSeries` (returns a NEW series;
  never mutate), and serialise/load (JSON) with a version string carried into
  `ScanResult.calibration_version`.
- **Accept:** on a held-out synthetic suite (different seeds than fitting):
  `beat_confidence_calibration(...)["ece"] <= 0.10`, and after calibration
  `clean_runs(series, min_conf=0.5)` retains ≥ 80% of true beats on clean sinus
  while excluding ≥ 80% of injected false beats. Existing AF-preservation test
  stays green.

### T3 · Real-video ingestion — `capture/`, `preprocessing/`, `rppg/`

The first contact with actual video. Three thin, testable layers:

- `capture/video_reader.py`: decode a video file → frames + per-frame timestamps;
  measure fps mean/jitter; populate `CaptureConfig` (codec/bitrate via metadata
  when available; unknown lossy compression → the schema gate rejects it, which
  is correct behaviour, not an error to work around).
- `capture/face_tracking.py`: mediapipe FaceMesh if importable, else an OpenCV
  landmark fallback; output per-frame landmarks + a tracking-stability score;
  record which tracker ran. **No face for > 2 s → abort with NO_RESULT reason.**
- `preprocessing/roi.py`: forehead, left cheek, right cheek, nose polygons from
  landmarks → per-ROI mean RGB time series. ROIs stay SEPARATE (fusion happens
  at the beat level, in the existing detector).
- `rppg/pos.py`, `rppg/chrom.py`: the classical extractors, operating on per-ROI
  RGB traces → per-ROI pulse waveforms feeding the EXISTING
  `beats/detector.py::detect_beats_single_roi`.
- **Accept:** build `scripts/make_synth_video.py` (synthetic face: skin-tone
  ellipse, green-channel pulse modulation from a known RR series, optional
  handheld jitter, ~20 s @ 30 and 60 fps). On it: end-to-end video → fused
  beats achieves beat F1@50ms ≥ 0.85 vs the generator's truth, via BOTH POS and
  CHROM. Also: a no-face video and a < 100-lux-equivalent dark video both yield
  NO_RESULT with the correct reasons. (Synthetic-video numbers are interface
  proofs, not accuracy claims — say so in the README.)

### T4 · Signal quality — `rppg/signal_quality.py`

Composite SQI ∈ [0,1]: SNR-in-band (NSQI-style), waveform skewness, spectral
concentration **within the physiological band** (broadband energy fraction — NOT
peak sharpness), cross-ROI beat coherence, tracking stability. Weights in
`configs/default.yaml`.

- **Accept (the anti-periodicity test, non-negotiable):** on noise-matched
  synthetic waveforms, `SQI(clean AF) >= SQI(clean sinus) - 0.05`; and SQI < 0.3
  on white noise, heavy motion, and no-pulse inputs. Add this as a permanent
  test — it is the inverted form of the open-rppg defect.

### T5 · Decision logic + CLI — `inference/decision_logic.py`, `inference/pipeline.py`, `cli.py`

- `decision_logic.decide(features, sqi, coverage, config) -> ScanResult`:
  SQI/coverage gate first (insufficient → `NO_RESULT`/`REPEAT_SCAN` with
  reasons); then a **transparent, hand-set interim rule** for v0.1 (thresholds in
  config, e.g. median|Δ|, pNN50, dropout_rate, rate bands → SINUS /
  AFIB_SUGGESTIVE / OTHER_IRREGULAR / HIGH_RATE). This placeholder is replaced
  by the trained Model A in Task 6 — mark it clearly as interim; it must still
  respect every invariant.
- `pipeline.run(video_path, manifest) -> ScanResult`: orchestrates
  T3 → detector → T2 calibration → clean_runs → features → T4 → decision;
  populates full provenance (`model_version`, `code_commit` via git,
  `calibration_version`, `config_hash` = SHA of resolved config).
- `cli.py`: `validate` / `process` / `evaluate` / `demo` subcommands; JSON out;
  exit codes 0 = ran, 2 = invalid input (with schema reasons on stderr).
- **Accept:** property tests — no code path emits a class when coverage/SQI is
  below floor; every emitted string equals `user_facing_text()` output verbatim;
  ScanResult provenance fields are all populated; `process` on the T3 synthetic
  videos returns SINUS for the sinus video and AFIB_SUGGESTIVE or
  OTHER_IRREGULAR (either acceptable at v0.1) for the AF video.

### T6 · Model A baseline — `models/baseline.py` (+ `scripts/e6_degradation.py`)

Logistic regression AND gradient-boosted trees on the run-feature vector, trained
on **real RR interval data degraded to rPPG conditions** (E6): take AF/non-AF RR
series (MIMIC PERform AF, CC-BY 4.0 — download in the script, cache locally;
35 subjects: 19 AF / 16 non-AF), inject measured rPPG timing noise (Gaussian
σ 5–30 ms), pulse-deficit dropouts (drop beats following RR < 400 ms with
probability 0.3–0.7), and false-beat insertions; run through `clean_runs` →
features → classifier with **participant-level cross-validation** (reuse
`datasets/splits` hashing; `assert_preprocessing_is_split_safe` wired in).

- **Accept:** AUC > 0.90 at the moderate-degradation setting (σ = 15 ms,
  deficit 0.5), reported with participant-level bootstrap CIs from the existing
  `evaluation/afib_metrics.py`; a degradation curve (AUC vs σ and deficit rate)
  saved as CSV + printed table. If AUC ≤ 0.90, report the curve honestly and
  mark blocked — do not tune on the test folds.
- Wire the trained model behind `decision_logic` via config flag
  `classifier: interim_rules | model_a`.

### T7 · Evaluation report — `evaluation/report.py`

`cli.py evaluate <dataset_dir>` over a directory of `(recording.json, video,
ecg)` triples → one markdown + JSON report containing: Gate 1 / 1b tables
(existing `GATE1_*` objects, PLUS `production_rmssd_error_ms` and confidence
ECE), beat metrics split by rhythm (AF vs non-AF — pulse deficit hides in
aggregates), risk-coverage curve at 100/90/80/75/60/50% with AF cases retained,
`no_read_report` parity across every available axis **including rhythm**
(P(no-read|AF) vs P(no-read|non-AF)), the leakage-audit output, split
composition, and full provenance. Any serial-confirmation figure printed
anywhere must state its `fp_persistent_share`.

- **Accept:** runs end-to-end on a generated synthetic dataset of ≥ 12
  recordings (sinus/AF/AF-with-deficit × conditions); report renders; a
  recording with CRF 28 appears as EXCLUDED with the schema's reason, not as a
  data point.

### T8 · Documentation

Update `README.md`: v0.1 CLI usage, dependency licences, and — prominently —
the statement that all shipped numbers derive from synthetic data and the E6
public-RR experiment; no clinical performance is claimed or implied. Add
`docs/RUNBOOK.md`: how a research assistant processes a day's recordings and
reads the evaluate report.

## Definition of done for v0.1

- Full suite green: **all 41 existing tests + every new test** (target ≥ 75 total).
- `cli.py process` on the synthetic sinus and AF videos produces correct,
  provenance-complete ScanResults; dark/no-face inputs produce NO_RESULT.
- `cli.py evaluate` produces the full gate report on the synthetic dataset.
- E6 degradation curve exists with participant-level CIs.
- No invariant weakened; no test threshold lowered; `CLAUDE_CODE_SPEC.md`
  untouched except to append a v0.1 changelog section.
- Final summary: what passed, what is blocked (with measured numbers), the
  licence table for added dependencies, and the exact commands to reproduce
  every accepted result.

## Working agreements

- Tests first; smallest diff that passes; no drive-by refactors of DONE modules
  (extend, don't rewrite — especially `schema.py`, `splits.py`, `ibi.py`).
- When an acceptance number is missed: report the measured value and stop on
  that task. A missed gate is information, not an obstacle to route around —
  this project kills ideas with measurements; that only works if measurements
  are never negotiated.
- If a dependency fails to install (e.g. mediapipe on this platform), implement
  the documented fallback and record which path is active in `CaptureConfig`.
- Keep every module importable without optional heavy deps (guarded imports).
