---
name: cardio-pipeline-optimizer
description: >
  Use this skill whenever the user wants to improve, tune, optimize, or "make more
  accurate" the cardio / AFib rhythm pipeline in this repo (avatarx-cardio: the rPPG
  beat detection, cross-ROI fusion, biomarker cards, or the measure service), or asks
  to "run an optimization iteration", "validate this change against the recordings",
  "why do the cards say Not computed", "make the scan faster", or otherwise proposes a
  code change to rppg/, beats/, preprocessing/, features/, inference/ or app/measure_*
  and wants to know whether it actually helped. It runs ONE gated iteration —
  baseline → one change → replay the real recordings → score → accept or reject — with
  a structural guard that auto-rejects any change that loosens a validity gate, so a
  number can only improve by improving the signal. Use it even when the user just says
  "the result is not coming" or "improve the afib code": those are this skill's job. Also
  use it for "improve accuracy", "make it reliable", "results should be consistent /
  stable", "same person scanned twice gives different numbers", "80 % accuracy",
  "why is a card missing", or any question about the tracking sheet's numbers — the
  proxies for those words (availability, test-retest stability, pulse agreement) are
  defined in CLAUDE.md and measured here.
metadata:
  version: "0.1.0"
  target_repo: "avatarx-cardio"
---

# Cardio Pipeline Optimizer

Run one **gated optimization iteration** on the cardio (AFib) rhythm pipeline. A change
is accepted only when it measurably improves **signal quality, reliability, or speed**
on a held-out set of real recordings **without** loosening any validity gate, and while
the four owner constraints keep holding. Never accept a change on intuition.

## What "better" means here — read this first

There is **no ground truth** in this repo: no ECG, no CPET. So this skill cannot
measure *accuracy*, and it does not pretend to. It measures the things that can be
measured on real recordings and that accuracy depends on:

| we measure | why it is a fair proxy |
|---|---|
| cross-ROI coherence, coverage, beat-timing precision, SQI | the pipeline's own validity gates — how far a scan clears them, not just pass/fail |
| fraction of the three biomarker cards that compute | the user-visible outcome |
| determinism (same clip → identical result) | a result that changes run to run is not a measurement |
| cross-scan consistency of card values | the corpus is one person; a fitness score swinging 25→70 across their scans is not realistic |
| server-side latency per stage | the "result ASAP" constraint, measured |
| availability, test-retest stability, pulse agreement (from the tracking sheet, `scripts/sheet_stats.py`) | the owner's 2026-09-08 goal: result on every quality-passing scan, same person within 30 min agrees, pulse within ±5 bpm of the reference |

Every one of these can be gamed in exactly one way: **loosen a gate until the number
appears.** That is why the gate script runs a structural guard first — a change that
moves any protected threshold in the permissive direction, or edits config/gate files
at all, is rejected before anything is scored. Numbers improve by improving the
signal, or not at all. When real accuracy matters, the answer is a reference study, and
this skill will say so rather than imply otherwise.

## The four owner constraints (encoded in the gate, not just stated)

1. **Don't break the running code.** Edits are confined to the signal chain; the measure
   service's response contract and the test suite are checked, not trusted.
2. **A result on every scan.** Replay must produce a structured outcome
   (`ACCEPT` / `REPEAT_SCAN` / `NO_RESULT` with reasons) for every recording — an
   exception or a hang on any clip is an automatic reject. The pipeline already
   guarantees this by design; the gate makes sure a change never un-guarantees it.
3. **Realistic output.** Cross-scan consistency is scored, and the guard forbids the
   shortcut (threshold surgery) that produces unrealistic-but-present numbers.
4. **Result ASAP.** Latency is a guardrail: a change may not make the server job
   slower beyond a small slack, no matter what else it improves.

## Operating principles

- **One hypothesis, one diff, per iteration.** Small diffs make the verdict mean something.
- **Never tune on the holdout split.** Inspect `tune` freely; the gate scores `holdout` only.
- **The gate decides.** Run the scripts, read `verdict.json`, honor it. Do not override a reject.
- **Honesty over green numbers.** A mean that improves while a subset regresses is a reject; surface it.
- **Edit only the signal chain:** `rppg/`, `beats/`, `preprocessing/`, `features/`, `inference/`,
  and the prep steps in `app/measure_prep.py`. Do not touch `configs/`, `capture/ingest.py`'s
  floors, `datasets/schema.py`'s floors, the eval corpus, or anything in this skill to make
  a change pass. `scripts/guard.py` enforces this; treat a guard failure as a hard stop.

## This is the offline flow

Replay calls `app.measure_api.measure_video` **in-process** on a *copy* of each
recording — the same trim → downscale → analyse path production runs — with the same
launch overrides production runs with. It never starts the HTTP server and never
touches the Railway deployment. Deploying an accepted change stays a human step.

## Prerequisites

- Python: the **miniconda** interpreter (`/Users/ravindrasinghbisht/miniconda3/bin/python3`
  on the dev Mac). The Homebrew and system `python3` lack scipy/opencv; see the project
  memory note if `which python3` is not miniconda.
- `ffmpeg` on PATH (trim and downscale shell out to it — same as production).
- The corpus at `data/eval_corpus/` (gitignored). If empty, ask the user where the
  retained recordings are; do not synthesize clips.

Run every script from the repo root:

```bash
PY=/Users/ravindrasinghbisht/miniconda3/bin/python3
S=.claude/skills/cardio-pipeline-optimizer/scripts
```

Read `references/data-contract.md` for the exact fields replay collects, and
`references/decision-rules.md` for every threshold and weight.

Four things the first real run got wrong, so you don't:

- **`--out` takes a bare filename** (`baseline.json`); it lands in `data/eval_cache/`.
  A path with a directory in it is used as given — don't pass `data/eval_cache/x.json`.
- **Never `python replay.py | grep`** without `set -o pipefail` — the pipeline reports
  grep's exit status, and a crash in the replay (traceback on stderr) vanishes while the
  chain marches on. Send stderr to a log file and read it if a "N records ->" line is missing.
- **Replays are serial.** Latency is a scored guardrail; a second replay or diagnostic
  running at the same time inflates job seconds and can fail a good candidate.
- **The guard judges what changed since the snapshot**, not everything dirty vs HEAD — so a
  dirty tree at snapshot time (the skill's own files, an unrelated edit) is fine. It still
  fails an out-of-scope file edited *during* the iteration.

## The iteration procedure

Each script writes JSON under `data/eval_cache/`. Read each output before moving on.

### 0. Snapshot the guard (once per iteration, BEFORE any edit)

```bash
$PY $S/guard.py --snapshot
```

Records the current value of every protected threshold and the clean git state. The
candidate is later checked against this snapshot — so take it *before* you change
anything, or the guard has nothing to compare to.

### 1. Build / refresh the eval set

```bash
$PY $S/build_eval_set.py --holdout-frac 0.4 --seed 13
```

Indexes `data/eval_corpus/`, probes each recording, and writes `eval_manifest.json` with a
deterministic `tune` / `holdout` split keyed on file hash (stable across runs and
machines). Report the counts per split.

### 2. Establish the baseline

```bash
$PY $S/replay.py --split holdout --label baseline --repeat 2 --out baseline.json
$PY $S/score.py --pred baseline.json --out baseline_score.json
```

`--repeat 2` runs each holdout clip twice so determinism is measured, not assumed.
`baseline_score.json` is the bar to beat. If any record shows `returned: false`, stop —
the pipeline is already failing constraint 2 on that clip and that is the bug to fix first.

### 3. Propose one change

Form one concrete hypothesis from the `tune` split (run `replay.py --split tune` and read
`per_record` — which gate fails most, which clips lose beats, where latency goes). State
the hypothesis and the file/lines before editing. Make the smallest edit that tests it.

Things already measured on this corpus, so you don't re-derive them — see
`references/data-contract.md` § "Known results" and `data/eval_cache/iterations.jsonl`.
In short: the 1080→480 downscale was a real lift, but *further* width changes (360),
face-size arguments, POS window changes (2.4 / 3.2 s) and peak-fit tweaks all trade
clips against each other — coherence up on some, down on others, and the cards that
newly compute carry unrealistic values. No region is dead; per-ROI SNR is near 0 dB and
the four regions place the same beat ~50 ms apart. Bitrate below ~5 Mbps fails; the
narrowband orientation idea helped one clip and hurt another. Signal-side knobs inside
a one-line change are exhausted on this corpus; the remaining measured wins are
**speed** (three full decodes per scan; face detection is 60 % of analysis).

### 4. Replay the candidate

```bash
$PY $S/replay.py --split holdout --label candidate --repeat 2 --out candidate.json
$PY $S/score.py --pred candidate.json --out candidate_score.json
```

### 5. Run the guard, then the gate

```bash
$PY $S/guard.py --check --out guard.json
$PY $S/gate.py --baseline baseline_score.json --candidate candidate_score.json \
  --guard guard.json --out verdict.json
```

`verdict.json` is `{"decision": "accept"|"reject", "reasons": [...]}`. A guard failure is a
reject regardless of score, and the reason names the exact threshold or file. Honor it.

### 6a. On `accept`

```bash
$PY $S/log_iteration.py --verdict verdict.json --candidate candidate_score.json \
  --hypothesis "<one line>" --action accept
```

Appends the iteration to `data/eval_cache/iterations.jsonl`. Summarize *which* metric
moved and by how much, on how many records. The candidate is the new baseline. Then
run the repo's own tests before anyone deploys:

```bash
$PY -m pytest tests/ -q -p no:cacheprovider \
  --ignore=tests/test_findings_only.py --ignore=tests/test_report_forbidden.py \
  --ignore=tests/test_flutter_forbidden.py --ignore=tests/test_vascular_forbidden.py \
  --ignore=tests/test_vasotone_forbidden.py
```

Measured 2026-09-07 on a clean checkout of 9e0deac: **800 passed, 3 skipped, 4 failed,
54 minutes**. The five ignored modules and two of the four failures are the neurokit2
`tests`-package collision (see the project memory note); the other two are a golden
drifting in the 13th decimal (`test_regularity_equivalence`) and
`test_fitness_harness::test_transition_sensitivity_shrinks_hrr60`. Those four fail
identically with and without a candidate — a *new* failure is a reject. The card
modules alone (`test_hemodynamics`, `test_realworld_hardening`, `test_scan_engine`) run
in 6 minutes and are the fast check for anything touching `features/`.

### 6b. On `reject`

Revert the edit (`git checkout -- <files>`). Read `verdict.json` for *what* regressed and
on *which* records, and feed that into the next hypothesis. Stop after
`RETRY_BUDGET` attempts and report the failed hypotheses — do not keep guessing.

## Scoring (summary)

Per record, then averaged over the split (full definitions in `references/decision-rules.md`):

- **`signal`** — mean gate margin: how far coherence, coverage, timing precision and SQI
  sit *above* their gates, each capped at 1. Primary objective.
- **`cards`** — fraction of the three biomarker cards with `status == computed`.
- **`determinism`** — 1 if the repeat run reproduced outcome and evidence exactly, else 0.
- **`consistency`** — 1 − (cross-scan coefficient of variation of each card's value / tolerance).
- **`latency`** — 1 − ((server job seconds − goal) / tolerance), floored at 0.
- **`returned`** — fraction of records that produced a structured outcome. Must be 1.0.

`total = W_SIG·signal + W_CARD·cards + W_DET·determinism + W_CONS·consistency + W_LAT·latency`.

## The gate rule (summary)

Accept iff **all** hold:

1. **Guard passed** — no protected threshold moved permissively; diff confined to the signal chain.
2. **`returned` == 1.0** on the candidate.
3. **Improved** — `signal` up by ≥ `MIN_MARGIN`, or `cards` up by ≥ `MIN_MARGIN` with `signal` not down.
4. **No guardrail regression** — `cards`, `determinism`, `consistency` not down (within slack); `latency` not down beyond `LAT_SLACK`.
5. **Not a fluke** — improvement on ≥ `MIN_IMPROVED_RECORDS` holdout records, not one.

## Safety

This loop optimizes the signal chain against proxy metrics on one person's recordings.
It does not validate anything clinically, and an accepted change is not evidence the
pipeline is *right* — only that it is cleaner, steadier and faster on these clips.
Keep deploy manual, and keep the language honest: "coherence improved on 6 of 8 holdout
scans" is a result; "accuracy improved" is not something this skill can say.

**The corpus is not a scratch input.** `measure_video` trims in place and deletes its
input after downscaling — right for a temp upload, destructive for a corpus file. It now
copies anything under `data/` first (`_protect_input`), but the rule stands: replay and
diagnostics go through `replay._run_once` or an explicit copy, never the manifest path.
On 2026-09-09 a diagnostic broke this rule and eight phone originals were lost; two came
back from duplicates, six survive only as their trimmed 40 s equivalents (see the known
results). Before any script that reads the corpus, ask: does anything downstream write to
the path it is given?
