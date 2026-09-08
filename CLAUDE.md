# avatarx-cardio — standing goal and rules

This file is loaded into every Claude Code session that touches this repo. It is the
owner's brief; do not ask for it again. Read it, then act.

## The goal (owner's words, 2026-09-08)

Improve the scanning and vital calculation logic so the system returns a reliable result
on every successful scan:

- **Available:** a result on every scan whose capture meets the quality conditions.
- **Stable:** the same person scanned twice within ~30 minutes gets the same numbers,
  unless something physiological changed.
- **Reliable:** target ~80 % accuracy/reliability on the available comparison method.
- **Promising** enough for further validation and product testing.

Plus the four constraints already encoded in the optimizer skill: don't break the running
code; a structured result on every scan; realistic output; result as soon as possible.

## What "accuracy" can mean here — and what it cannot

There is **no ECG, no CPET, no cuff** in this repo. Accuracy against clinical truth is not
measurable and must never be claimed. The measurable proxies, in priority order:

| proxy | definition | where it is measured |
|---|---|---|
| availability | share of quality-passing scans that compute all three cards | `data/eval_cache/*_score.json` (`cards`), the tracking sheet |
| stability (test-retest) | per-card coefficient of variation across scans of the same person within 30 min | tracking sheet (`scripts/sheet_stats.py` in the skill) |
| pulse agreement | pipeline `Pulse bpm` vs the reference heart rate the client sends (`ref_hr`, ShenAI) | tracking sheet `Pulse − Ref HR` |
| signal margin | how far coherence / coverage / timing / SQI clear their gates | skill `signal` metric |
| determinism | same clip → identical result | skill `determinism` metric |
| speed | server job seconds per stage | skill `latency` metric, sheet timing columns |

"80 %" is read as: ≥ 80 % of quality-passing scans compute all three cards, AND repeat
scans within 30 min agree within the card's tolerance (`CV_TOL` in the skill), AND pulse
agrees with the reference within ±5 bpm on ≥ 80 % of scans. Report each of the three
separately; never blend them into one "accuracy" number.

## Rules that are enforced, not just stated

- **Threshold changes are the owner's decision, made in chat, one at a time.** The
  PreToolUse hook in `.claude/settings.json` blocks edits to `configs/*.yaml`,
  `capture/ingest.py`, `datasets/schema.py` and `inference/pipeline.py` unless the
  owner has approved in the conversation, in which case run
  `touch .claude/owner-approved` immediately before the single approved edit (the hook
  consumes the sentinel). Never touch the sentinel on your own judgment.
- **Every pipeline change goes through the optimizer skill's gate** — baseline → one change
  → replay the corpus → score → guard → gate. Read the per-record table, not only the
  verdict: a change that moves a record it should not touch is a reject even when the
  aggregate says accept (iteration 3 taught this).
- **Numbers improve by improving the signal, never by loosening a gate.** A card that
  appears because a floor moved is not an improvement; it is a fabricated one.
- **The response contract is frozen.** The webapp reads `outcome`, `biomarkers.items[]`,
  `timing`, `trim`, `clock`. Add fields; never rename or remove.
- **The sheet never delays a result.** `app/result_sheet.py` runs after the response is
  written; keep it that way.

## Where things are

- Interpreter: `/Users/ravindrasinghbisht/miniconda3/bin/python3` (Homebrew python lacks scipy).
- Skill: `.claude/skills/cardio-pipeline-optimizer/` — procedure, scripts, and
  `references/data-contract.md` § "Known results" (what has been tried; don't re-derive).
- Iteration log: `data/eval_cache/iterations.jsonl`. Corpus: `data/eval_corpus/` (17
  recordings, one person; gitignored). Guard snapshot: `data/eval_cache/guard_snapshot.json`.
- Tracking sheet: spreadsheet `1-CHxko3E4_xQBwxGbsUj-WjqY2_LFBVa3kKnkY3WaRo`, tab
  `cardio-data` (gid 140358874); every production scan appends a row.
- Deploy: push `main` to remote `personal` (ravindra1128/avatarx-cardio) → Railway rebuilds
  in ~1 min. Verify with `GET /healthz` → `build` (the response's `code_commit` is always
  `no-git` on Railway) and `config_hash`. Remote `origin` is the org repo; the owner pushes it.
- Local test: webapp `.claude/launch.json` starts this service on 8790 and a webapp on
  5174 pointed at it. Phone scans are ~30–50 MB; the Railway edge relays uploads at
  ~0.5–1 MB/s, so most of a phone's wait is upload, not analysis.
- Tests: `python -m pytest tests/ -q -p no:cacheprovider --ignore=tests/test_findings_only.py
  --ignore=tests/test_report_forbidden.py --ignore=tests/test_flutter_forbidden.py
  --ignore=tests/test_vascular_forbidden.py --ignore=tests/test_vasotone_forbidden.py`
  → baseline 800 passed, 4 known failures (listed in the skill), 54 min. The three card
  modules (`test_hemodynamics`, `test_realworld_hardening`, `test_scan_engine`) take 6 min.

## Owner decisions on record

| date | decision | evidence |
|---|---|---|
| 2026-09-07 | card timing ceiling 40 → 50 ms (`endpoint_max_timing_precision_ms`) | corpus screen: 26 → 31 of 51 cards, values in band; 60 ms rejected (implausible values) |
| 2026-09-07 | vascular-tone 12-beat floor applies to the whole scan, not per capture segment | gated accept; single-segment clips identical |
| 2026-09-08 | reliability over availability: the gate accepts a consistency gain (≥ +0.05, signal/determinism held) even if fewer cards compute | `gate.py` reliability clause; iterations 7–8 tried under it, both rejected on evidence |
| 2026-09-08 | lever 1 approved: additive per-patch traces in `capture/ingest.py` + SNR-based combination in evidence | three variants gated, all rejected (values scatter); reverted; design kept in the skill's known results |

## How to start a session on this repo

1. The SessionStart hook prints the last iterations and the current baseline. Read it.
2. State which proxy you are moving and the one hypothesis, in one line.
3. Run the skill. Honor the verdict. Log it. Deploy only when the owner says so.
