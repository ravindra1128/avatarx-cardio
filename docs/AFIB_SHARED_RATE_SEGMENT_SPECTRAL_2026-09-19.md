# Shared resting-rate decision and capture-segment spectra

Candidate against Codex backend build `77bc9b0e6365`, 2026-09-19.

## Problem and resulting behavior

Spectral analysis previously concatenated separately extracted capture segments before Welch estimation. Its windows could cross capture gaps or extraction restarts. When the corrected spectrum lost region support, the fitness card could independently fall back to a disputed beat count even though the rate head abstained.

The candidate retains segment boundaries for spectral analysis, pools Welch estimates only within eligible continuous finite parts, and feeds the production fitness card the same rate resolver, clean-interval statistic and existing count floor as the rate head. An unresolved rate produces an explicit unavailable-card reason. A supported provisional rate stays provisional. Legacy standalone card callers keep their existing behavior.

The spectrum uses no SDK/reference rate to choose a peak. No previous scan is reused. No quality threshold, AFib classifier, source-selection gate, capture setting or retention setting changed. The existing response fields remain; rate-resolution and spectral-segment diagnostics are additive.

## Evaluation

Seventeen existing recordings were replayed twice, on input copies, with `AFIB_SCALE=640x480`, `AFIB_WINDOW_S=70`, and the existing collapsed-interval launch override. The manifest and split were unchanged. Four recordings form the existing holdout. This is one person's corpus without ECG labels, not a clinical accuracy study or a timed test–retest study.

The rate-only candidate improved consistency but failed its first latency comparison; that rejection is preserved in `data/eval_cache/shared_rate_20260919/verdict.json`. A fresh live-code holdout replay was then run because unchanged upstream stages had also slowed. The combined candidate was tested as the preplanned follow-up, not by changing scoring thresholds.

| Measure | Live baseline | Combined candidate |
|---|---:|---:|
| Holdout consistency proxy | 0.5617 | 0.6404 |
| All-record consistency proxy | 0.3838 | 0.4183 |
| Phone-record consistency proxy | 0.7285 | 0.7401 |
| Demo-record consistency proxy | 0.2662 | 0.3138 |
| All-record computed-card fraction | 0.8824 | 0.8235 |
| Holdout mean server seconds (fresh baseline) | 11.32 | 13.66 |
| Structured replay outcomes | 17/17 | 17/17 |
| Deterministic replay outcomes/evidence/cards | 17/17 | 17/17 |

The holdout gate accepts via its existing reliability clause. The three newly unavailable fitness scores have unresolved shared rates; stiffness and vascular-tone outputs are identical. The fitness card matches the rate head on every recording. Comparing only records with unchanged card availability also improves consistency: all-record paired score 0.3254 to 0.3771, phone paired score 0.7220 to 0.7427. Thus abstention alone does not explain the improvement.

AFib outcomes, failed gates, SQI, coherence, coverage, beat counts and timing precision are unchanged on every recording. This change does not yet demonstrate fewer Inconclusive scans or AFib sensitivity. The spectral candidate alone had regressed broad-corpus consistency in the earlier trial; the shared-rate correction addresses its unsupported fallback behavior.

Artifacts: `data/eval_cache/combined_rate_spectral_20260919/` includes the unchanged-threshold guard, full replay, per-record and subset review, baseline/candidate scores and accepted verdict. Raw recordings are not added to git.

## Qualification and deployment

Seventy-five focused tests passed before replay. The full existing CI selection completed across 105 modules: 1132 tests pass after the audit-budget registration fix and targeted rerun, with 3 skips. The five pre-existing individually deselected failures and five collection-error module exclusions remain exactly as recorded in `.github/ci-test-args`; no new exclusion was added. Deployment follows qualification. Only the Codex backend is in scope. The webapp, cardio and cardio-staging environments are untouched.

The full suite identified one structural-audit registration omission: `_segmented_psd` uses one `numpy.diff` on a Boolean finite-sample mask to locate holes, not on beat intervals. The existing exact-call-count audit budget now registers that one operation by function name. The detector, audit logic and interval-statistics restrictions are unchanged. The original failure and focused rerun are retained with the qualification artifacts.
