# Beat-identity amplitude pooling experiment — rejected and reverted

The candidate corrected a demonstrated bookkeeping sensitivity, but **worsened repeated-scan stability**. No estimator change from this experiment remains in the application. No deployment occurred.

## Hypothesis and controlled change

Problem: the same physiological beat may have regional foot timestamps separated by more than the pool's 50-ms matching tolerance. Timestamp-only perturbations changed a tune-set score by 6.8 points without changing amplitudes.

Change: preserve the existing lattice beat time alongside each regional foot-to-foot window, and use that identity when pooling amplitudes. This changed two feature files only. Waveforms, windows, amplitude magnitudes, normalization, confidence, validity floors and score mapping were held constant.

The synthetic/API invariant check passed: unchanged selected windows and amplitudes, optional legacy helper interface retained, and bookkeeping shifts no longer changed the canonical grouping. This check proves bookkeeping behavior, not physiological validity.

## Same-record regression experiment

The optimizer indexed 33 unique recordings (25 tune, 8 holdout), skipping two exact duplicates. Baseline and candidate each replayed all eight holdout clips twice through the standard production-style prep path. Every run returned a structured result and deterministic evidence/card values.

| Holdout metric | Before | After |
|---|---:|---:|
| Structured return fraction | 1.000 | 1.000 |
| Signal metric | 0.8266 | 0.8266 |
| All-card computed fraction | 0.875 | 0.875 |
| Determinism | 1.000 | 1.000 |
| Tone cross-record CV | 0.2872 | 0.2980 |
| All-card consistency metric | 0.2595 | 0.2522 |
| Mean server job seconds | 12.6275 | 12.7850 |

These optimizer numbers use its default request window and mixed corpus; they are **not** the same-person repeatability endpoint. All unrelated card objects and signal evidence were compared per record and remained identical. The guard passed. The gate rejected because neither signal nor card coverage improved, no per-record signal improvement occurred, and the reliability improvement condition was not met.

## Confirmed same-person comparison

All 16 September 16–17 retained phone recordings were replayed on both versions with the same 70-second window request and 480×360 analysis box. The owner confirmed one person, one phone and comparable resting conditions. There were ten consecutive pairs within 20 minutes; only five pairs had numbers on both scans. All four missing scans remained missing.

| Metric | Before | After |
|---|---:|---:|
| Numeric tone coverage, all retained phone scans | 12/16 | 12/16 |
| Pairs with both results | 5/10 | 5/10 |
| Mean absolute repeated-pair difference, points | 9.42 | 13.50 |
| SD of absolute pair differences, points | 10.523 | 7.408 |
| Min–max absolute pair difference | 1.8–26.2 | 6.5–23.6 |
| Exploratory ≤20% relative-difference pass, numeric pairs | 3/5 | 2/5 |
| Same exploratory pass, all attempted pairs | 3/10 | 2/10 |
| Mean of 12 numeric scores | 46.975 | 46.050 |
| SD across those scores | 15.012 | 14.105 |
| Range across those scores | 55.4 | 48.1 |
| CV across those scores | 31.96% | 30.63% |

The across-record CV and range improved slightly, **while the actual repeat-pair endpoint worsened by 43.3%**. This is why an aggregate spread cannot substitute for repeated-scan validation. The 20% threshold is exploratory; it has no established clinical meaning for this custom score.

| Pair (UTC) | Before values | After values | Absolute difference before → after |
|---|---|---|---|
| Sep 16 08:07 → 08:12 | 48.7, 35.3 | 48.2, 29.2 | 13.4 → 19.0 |
| Sep 16 12:37 → 12:45 | 74.8, 48.6 | 71.7, 48.1 | 26.2 → 23.6 |
| Sep 17 06:10 → 06:14 | 43.9, 41.8 | 46.3, 37.9 | 2.1 → 8.4 |
| Sep 17 06:14 → 06:19 | 41.8, 40.0 | 37.9, 47.9 | 1.8 → 10.0 |
| Sep 17 06:19 → 06:26 | 40.0, 43.6 | 47.9, 41.4 | 3.6 → 6.5 |

![Repeat-pair differences](/Users/ravindrasinghbisht/Downloads/cardio/research/vascular_tone_audit/repeat_pair_comparison.png)

## Causal checks and interpretation

For all 16 before/after phone comparisons, source SHA-256, raw RGB summaries, raw beat-window amplitudes, normalized amplitude magnitudes, and signal evidence were identical. The change isolated aggregation. Four of five numeric pairs worsened. Thus eliminating the pool's time-membership sensitivity did not recover consistent physiological amplitude information from these regions. It may have included additional noisy regional amplitudes; that mechanism remains an inference, not a demonstrated source-of-noise decomposition.

A separate frontend lifecycle reproduction loaded the actual production AFibProcessor classes, started A then B, and delivered A late. The shared `afib_result` key accepted A. Existing upload/orchestration tests passed 31/31 but do not test or fix this race. No actual phone/browser end-to-end capture was performed. The production frontend files remain unchanged under the repository's Codex route ownership boundary.

## Retained work and reproducibility

Retained: investigation report, diagnostic replay and analysis tools, baseline intermediate data, comparison data and the rejected patch. Both edited feature files were restored byte-for-byte to their pre-experiment state after verifying that no concurrent edits had changed them. Existing fitness/transport work was preserved. An unrelated CLAUDE.md documentation change arrived during the work and was preserved. The rejection was appended to `data/eval_cache/iterations.jsonl`.

Artifacts:

- `research/vascular_tone_audit/replay.json`: all 33 baseline diagnostic records and intermediates.
- `research/vascular_tone_audit/analysis.json`, `per_scan.csv`: baseline statistics and feature sensitivities.
- `data/eval_cache/tone_audit/baseline.json`, `candidate.json`: twice-replayed holdout outputs.
- `data/eval_cache/tone_audit/candidate_phone.json`, `candidate_phone_analysis.json`: 16-record candidate and repeated-pair analysis.
- `data/eval_cache/tone_audit/guard.json`, `verdict.json`: structural guard and reject verdict.
- `data/eval_cache/tone_audit/candidate.patch`: exact isolated candidate, retained for inspection only.

From the repository root, using the miniconda interpreter:

```sh
/Users/ravindrasinghbisht/miniconda3/bin/python3 research/vascular_tone_audit/replay_audit.py /tmp/tone_replay.json --phone-only
/Users/ravindrasinghbisht/miniconda3/bin/python3 research/vascular_tone_audit/analyze.py /tmp/tone_replay.json /tmp/tone_analysis.json
```

A replay runs the current checkout; after the revert it reproduces the baseline. Do not mistake the saved candidate JSON for the currently active estimator. The scripts copy corpus recordings and verify original hashes. Diagnostics add overhead; use the optimizer's uninstrumented paired runs for latency comparisons.

## Next action and limits

The next scientifically stronger path is acquisition and a separately defined local pulsatile-perfusion research measurement, with the existing isolated facial-perfusion pilot as a starting point. Preserve raw RGB/DC, frame timestamps, camera exposure/gain/WB state, ROI motion and illumination controls. Collect repeated baselines plus synchronized local laser-speckle or laser-Doppler reference; finger PPG, pressure and temperature are contextual controls rather than interchangeable facial-tone labels. Hold out participants/devices and use sham/phantom controls before any physiological claim.

The retained lossy clips cannot recover missing camera telemetry or a missing reference. More threshold or score-transform experiments cannot resolve that identifiability problem. A universal clinically meaningful tolerance, valid usable-scan denominator and direct vascular-tone claim remain unestablished. The full mission is **not achieved**: this work establishes a reproducible baseline, finds concrete failure mechanisms, and rules out one proposed fix quantitatively. The next capture/reference-dependent experiment has not been run or represented as a success.
