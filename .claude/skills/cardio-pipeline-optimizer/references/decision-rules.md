# Decision rules

Every threshold and weight lives here so the gate is auditable in one place. `score.py`,
`gate.py` and `guard.py` read these defaults; override via env vars of the same name
(uppercased). Do not change these to make a candidate pass — that is the one move the
skill exists to prevent.

## Production environment the replay must match

Replay applies the same launch overrides the deployed service runs with, so what is
scored is what production computes:

```
PIPELINE_ENV = {"AFIB_MAX_COLLAPSED_FRACTION": "0.05"}
```

If production changes its overrides, change this — and only this — to match.

## Scoring weights

```
total = W_SIG  * signal
      + W_CARD * cards
      + W_DET  * determinism
      + W_CONS * consistency
      + W_LAT  * latency
```

| Name | Value | Meaning |
|---|---|---|
| `W_SIG`  | 1.0 | gate-margin signal score — primary objective |
| `W_CARD` | 0.5 | biomarker cards computed |
| `W_DET`  | 0.3 | run-to-run determinism |
| `W_CONS` | 0.3 | cross-scan consistency of card values |
| `W_LAT`  | 0.2 | server-side latency |

### Component definitions (per record, then mean over the split)

The pipeline's own gates, taken from `configs/default.yaml` / `inference/decision_logic.py`
as of this skill's version. They are *read* here to compute margins; the guard checks they
were not *changed*.

| gate | threshold | direction |
|---|---|---|
| `COH_GATE`  | 0.20 | cross-ROI coherence, ≥ |
| `COV_GATE`  | 0.50 | clean-interval coverage, ≥ |
| `TIM_GATE`  | 40 ms | beat-timing precision, ≤ |
| `SQI_GATE`  | 0.30 | signal quality index, ≥ |

- `signal = mean( clip01(coh / COH_GATE), clip01(cov / COV_GATE),
                  clip01(TIM_GATE / max(tim, 1e-6)), clip01(sqi / SQI_GATE) )`
  — each term is "how far above the gate", capped at 1.0 so one runaway metric can't
  mask a failing one. A missing metric (no beat lattice) scores 0 for that term.
- `cards = (# of the 3 biomarker items with status == "computed") / 3`.
- `determinism = 1.0` if the `--repeat` run reproduced `outcome`, every evidence number
  (to 1e-6) and every card value; else `0.0`. Undefined (excluded) if `--repeat` < 2.
- `consistency = 1 - clip01(mean_cv / CV_TOL)`, where `mean_cv` is the coefficient of
  variation of each card's value across the records that computed it, averaged over
  cards. `CV_TOL = 0.5`. Undefined if fewer than `MIN_CARDS_FOR_CV = 3` records computed a
  given card. **Assumption:** the corpus is one person; if it ever mixes subjects, this
  term must be computed per subject or dropped.
- `latency = 1 - clip01((server_total_s - LAT_GOAL_S) / LAT_TOL_S)`, `LAT_GOAL_S = 15`,
  `LAT_TOL_S = 30`. Uses `timing.server_total_s` from the measure path (trim +
  downscale + analysis). Upload/relay time is not the pipeline's and is not scored.
- `returned = 1.0` if replay produced a structured outcome for the record, else `0.0`.

## Gate (accept/reject)

Accept the candidate iff **all** of:

0. **Guard passed.** `guard.json` reports `ok: true`. Any protected threshold moved in
   the permissive direction, or any file outside the allowed edit scope changed, is a
   reject before scoring — the reason names the exact symbol or path.
1. **Every scan returned.** `cand.returned == 1.0`.
2. **Improved.**
   `cand.signal >= base.signal + MIN_MARGIN`
   **OR** (`cand.cards >= base.cards + MIN_MARGIN` **AND** `cand.signal >= base.signal - SIG_SLACK`).
   - `MIN_MARGIN = 0.02`
   - `SIG_SLACK = 0.0`
3. **No guardrail regression.**
   - `cand.cards >= base.cards - CARD_SLACK` (`CARD_SLACK = 0.0`)
   - `cand.determinism >= base.determinism - DET_SLACK` (`DET_SLACK = 0.0`)
   - `cand.consistency >= base.consistency - CONS_SLACK` (`CONS_SLACK = 0.05`)
   - `cand.latency >= base.latency - LAT_SLACK` (`LAT_SLACK = 0.05`, ≈ 1.5 s of job time)
4. **Not a fluke.** Per-record `signal` improves on ≥ `MIN_IMPROVED_RECORDS = 3` records
   (or a majority if the holdout is small), not concentrated in one.

Otherwise reject.

## Guard: protected thresholds (never auto-overridden)

`guard.py` snapshots these before an edit and re-reads them at check time. Moving one in
the listed permissive direction, or deleting it, fails the guard.

| file | symbol | permissive direction |
|---|---|---|
| `capture/ingest.py` | `MIN_FINITE_FRACTION` | decrease |
| `capture/ingest.py` | `MAX_COLLAPSED_INTERVAL_FRACTION` | increase |
| `capture/ingest.py` | `CONSUMER_FPS_FLOOR` | decrease |
| `datasets/schema.py` | `FPS_FLOOR_BEAT` | decrease |
| `datasets/schema.py` | `BPP_FLOOR` | decrease |
| `datasets/schema.py` | `CRF_CEILING` | increase |
| `configs/default.yaml` | *(entire file)* | any change |
| `configs/gates.yaml` | *(entire file)* | any change |

Allowed edit scope for a candidate (anything else in the diff fails the guard):

```
ALLOWED_DIRS = rppg/  beats/  preprocessing/  features/  inference/
ALLOWED_FILES = app/measure_prep.py
```

`app/measure_api.py` is transport and is deliberately *not* in scope for an accuracy
iteration; latency work belongs in `measure_prep.py` (the prep steps) or the signal chain.

## Budgets

- `RETRY_BUDGET = 3` — proposals per iteration before stopping and reporting.
- `MAX_ITERATIONS = 10` — accepted steps before a human review checkpoint.

## Invariants

- Never edit the corpus, the manifest, these rules, or the scripts to make a candidate pass.
- Never widen a gate, floor, or tolerance in `configs/` to convert a `Not computed` into a value.
- A change that regresses any holdout subset is a reject even if the mean improves.
- Replay must never start the HTTP server or touch the deployed service.


## Owner goal proxies (2026-09-08), measured from the tracking sheet

`scripts/sheet_stats.py` reads the `cardio-data` tab and reports three numbers. They are
the product targets; the corpus metrics above are how a single change is judged.

| proxy | definition | target |
|---|---|---|
| `availability` | scans with `Outcome != NO_RESULT` **and** capture gates passed (a beat lattice exists) that have all three cards `computed` | ≥ 0.80 |
| `retest_cv` per card | for pairs of rows within `RETEST_WINDOW_MIN = 30` minutes and the same `User Agent`, the coefficient of variation of the card value; reported as the median across pairs | ≤ `CV_TOL` (0.5) for every card, with the fraction of pairs inside tolerance ≥ 0.80 |
| `pulse_agreement` | share of rows with `Ref HR` where abs(`Pulse - Ref HR`) ≤ `HR_TOL_BPM = 5` | ≥ 0.80 |

Report the three separately. Never combine them into one "accuracy" figure, and never
report a proxy computed on fewer than `MIN_ROWS = 10` rows as a result — say "n = …,
not yet meaningful" instead.
