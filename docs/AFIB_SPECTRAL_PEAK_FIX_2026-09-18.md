# Spectral peak correctness candidate

Status: **owner approved the correctness exception; release verification completed**.
The exact tested patch is applied on top of build `7f5cd57f3619` and its 25
regression controls are in `tests/test_spectral_peak_validity.py`. Artifacts remain in
`data/eval_cache/spectral_peak_20260918/`. No thresholds, video retention settings,
webapp files or other cardio environments changed.

## Problem and correction

`inference/evidence.py::_fundamental` removes the first and last in-band spectrum
bins and chooses the largest remaining value. A steadily declining spectrum
therefore reports 45 bpm even when it contains no pulse peak. A half-rate search
can similarly select a declining shoulder instead of another peak.

The candidate requires a local maximum, including a bounded flat-topped peak,
for both the primary selection and the half-rate alternative. Band-edge samples
remain available as neighbours when deciding whether an interior peak exists.
The frequency band, harmonic power ratio, half-rate tolerance, signal-quality
gates and peak/median denominator are unchanged. A genuine 45 bpm peak is still
accepted; no frequency is blacklisted. Peak finding uses
[SciPy's documented local-maximum definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).

This fixes the mathematical definition of a peak. A local maximum alone does
not establish that a signal is physiological or make the AFib classifier valid.

## Validation

- **25 new controls:** the original code failed 11; the candidate passes all 25.
  Cases include declining/rising/flat spectra, valid 45–177 bpm peaks, boundary
  maxima, drift plus a real peak, supported/unsupported harmonics, flat-topped
  peaks, invalid inputs and shared drift across four facial regions.
- **69 focused tests pass**, including existing spectral, pulse cross-check,
  resting-rate resolver, doubling guard and rate-head tests.
- Existing corpus retained unchanged: **13 development / 4 held-out recordings**.
  The historical manifest and pretrimmed flags were preserved and file hashes
  verified before and after. Replay operates on temporary copies only.
- Both versions used the Codex phone profile: `640x480`, requested window `70 s`,
  collapsed-interval launch override `0.05`, retention off. Existing pretrimmed
  clips cannot test a longer retained window.
- Development replay: all 13 return, with identical outcomes, reasons, cards,
  signal evidence and spectral estimates.
- Held-out replay: each recording ran twice per version. All return and all
  repeats are deterministic. Outcomes, reasons, gate failures, cards, combined
  spectral rates and signal measurements are unchanged.
- One held-out facial region changes: `webapp_1234567890-ebe749`, left cheek,
  **45 bpm without a local peak → 81 bpm at a local peak**. Combined rate stays
  66 bpm with 3/4 regions agreeing. This is not proof that 81 bpm is the person's
  true rate; there is no ECG reference.

## Gate verdict and limits

Structural guard: **passed**, exactly one signal-chain file changed.

| Held-out proxy | Baseline | Candidate |
|---|---:|---:|
| Returned | 1.0 | 1.0 |
| Signal margin | 0.6864 | 0.6864 |
| Cards computed | 0.75 | 0.75 |
| Determinism | 1.0 | 1.0 |
| Card consistency | 0.5617 | 0.5617 |
| Latency score | 0.9459 | 0.9267 |
| Mean server job | 12.865 s | 13.3375 s |

Latency stays within the configured aggregate slack; no speed gain is claimed.
The long held-out clip took 21.49 versus 23.79 seconds on the scored first runs.
The gate rejects because signal/cards do not improve by the required 0.02 and
zero records improve on its signal-margin metric (three required). The verdict
is logged unchanged in `data/eval_cache/iterations.jsonl`.

All four held-out recordings abstain under both versions. This corpus cannot
show improved AFib sensitivity, specificity or useful classification stability.
The latest two phone recordings were not retained, so their spectra cannot be
replayed. The unit controls and one regional correction demonstrate a software
correctness property, not achievement of the screening goal.

## Owner approval and release

The owner explicitly approved the requested spectral correction exception with
“i am givng you approvance” on 2026-09-18. The original optimizer rejection is
preserved; a separate owner-exception entry was appended to the iteration log.
The exact approved patch and its controls passed release verification with the
five documented baseline failures below. Deployment targets only the isolated
cardio-codex backend; the live build receipt is recorded separately after push.
This exception does not approve the separately pending beat-matching,
SDK-input-validation or uncertainty patches, nor a fitness-card change.

The next separate investigation is the fitness card's duplicated rate decision:
it labelled the 10:00 scan measured even though the shared rate resolver reported
uncertain. No change to that card is included in this candidate.

Receipt: `docs/audits/spectral-peak-fix-2026-09-18.json`.
Patch SHA-256:
`98abaa94c99308a34bcb4b4443e4d60c73bdaf0f80abb6089b9aa48df44fb57f`.

## Release verification

Combined coverage of all 1,076 collected tests: **1,068 passed, 3 skipped,
5 documented pre-existing failures; no additional failures**. This is not an
all-green suite. The five failures match `.github/ci-test-args`: three test-module
import failures, the bit-exact historical golden comparison, and the existing
noise-only provisional-card behavior. None was excluded from this local run.

The initial full run completed 916 passing tests but exhausted disk space while
generating synthetic videos: 152 distinct tests were affected, producing one
additional call failure and 302 setup/teardown error events. Only that run's
temporary `pytest-257` fixtures were removed. All 152 affected tests then passed
when rerun serially by module with temporary-fixture cleanup between modules.
The original failure log is retained; no test or production code was changed to
make the retry pass. The existing real-recording corpus was not part of cleanup.

The prior commit's GitHub session-flow wall-clock failure passed locally here.
GitHub Actions has no repository variables or secrets configured for its
Railway deploy job. The release therefore uses the existing Railway GitHub
integration, with the live `/healthz` build and Codex sheet tab checked after
push; an Actions workflow result alone is not deployment proof.
