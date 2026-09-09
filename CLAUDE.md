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
- **A finished scan outlives its connection (2026-09-09).** `/api/process-video` answers
  with a CHUNKED body, emitting a space every 5 s while the analysis runs, so the socket
  never sits idle for the 20–45 s it takes (a real phone scan died there with
  `ERR_HTTP2_PING_FAILED` after the work was done). Leading whitespace is valid JSON, so
  parsers are unaffected; the trade is that a job which raises returns HTTP 200 with
  `{"error": ...}` instead of a 500, which clients already treat as a failure. The job
  also runs to completion when the client vanishes, and the result is held for
  `AFIB_RESULT_TTL_S` (15 min) under the client's `upload_id` — `GET /api/result?upload_id=…`
  collects it. Never key that cache on `session`: the webapp sends a constant one.
- **Evidence fields are additive.** `debug.evidence` gained `pulse_lattice_bpm`,
  `pulse_spectral_bpm`, `pulse_spectral_snr`, `pulse_spectral_roi_bpm`,
  `pulse_spectral_roi_agree`, `pulse_agreement` (2026-09-09); the cards' confidence carries
  `endpoint_evidence.pulse_check`. Legacy evidence without them reads "not_evaluated".
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
| 2026-09-09 | three changes approved, in order: pulse cross-check, one metric per stiffness card, fitness heart-rate-only | pulse cross-check built in REPORT mode (`features/hemodynamics.py::PULSE_CHECK_MODE`): the corpus cannot score an abstention rule (3 card-bearing holdout records), so every scan reports agree/disagree/unresolved on the sheet (`Pulse Check`) and the owner flips to `gate` on sheet evidence against the reference pulse |
| 2026-09-09 | pulse cross-check made BINDING (`PULSE_CHECK_MODE = "gate"`) and the fitness card given ONE basis (heart-rate-only, always) with a 15 clean-interval floor; the `calibrated_fused_beat_median` rate is inadmissible | production scan: beat count 84 bpm vs spectral 60 bpm with 2 of 4 regions backing the spectrum, reference 64 - the spectrum was right within 4 bpm and all three cards had computed on the wrong rate. Fitness: the same person read 27.1 then 50.1 out of 100 fourteen minutes apart because the card silently switched formulas, and the 50.1 needed RMSSD 286 ms against a reference of 36 ms. Gate REJECTED both (holdout cards 0.4445 -> 0.2222); shipped on the owner's decision, reliability over availability, iteration 14 in `iterations.jsonl` |
| 2026-09-09 | the client asks for a 70 s analysis window (service default stays 40) | the recorder already runs 65-100 s and the service analysed only the last 40, discarding more than half of every upload unread. The best scan yet (coherence 0.675, SQI 0.711, pulse 70.7 vs reference 70.0, 4 of 4 regions agreeing) still computed nothing: 8.5 s of its 40 s window held clean beats, 10 clean intervals where fitness needs 15. Replay of the one full-length corpus clip that could test it: 40 s -> 13 intervals, 70 s -> 22, 90 s -> 18 (older footage is poorer, so more is NOT better). Only n=1 offline - six of eight phone clips survive as 40 s trims after the corpus incident - so it is a request parameter, measured on the sheet's new `Window s` column, not a default change. Server time roughly doubles; the heartbeat added the same day covers the longer wait |

## Repeatability plan (owner-agreed 2026-09-08, after the literature review)

Target: each of the three cards within ±10–20 % on repeated scans of the same person
under similar conditions. Order of work, with the evidence in the skill's known results:

1. **Capture** (webapp): exposure/white-balance lock on the SDK track once auto-exposure
   settles; a 3 s pre-check (delivered fps, face brightness) with on-screen warnings; lock
   state, fps and brightness sent with the upload (`exposure_locked`, `awb_locked`,
   `client_fps`, `face_luma`) → manifest + sheet. Then, as separate gated changes:
   60 fps where the device allows (raises phone load and needs more bitrate) and a
   60–90 s resting recording. **Slice 1 built 2026-09-08** (`src/lib/capture/cameraCapture.js`).
   **2026-09-09 mobile finding:** on Android Chrome `exposureMode: "manual"` is auto-exposure
   OFF with the last exposure time but the driver's default gain (Chromium
   `VideoCaptureCamera2.java`); Chrome never reports the automatic ISO, the picture went
   dark (face luma 127 → ~27, "illuminance 83 lux < floor 100") and the mode read back
   "none". The lock now matches the gain to the pre-lock brightness and reverts unless the
   picture stays within ±20 %; the sheet's `Capture Note` column carries its account.
   Never ship a camera-state change without a brightness guard.
2. **Fitness card:** heart-rate-only unless the HRV term clears 2× the scan's timing-noise
   floor (`data/eval_cache/fitness_hr_only.patch`); relabel as a resting-rate proxy.
3. **Vascular tone:** require ≥ 30 amplitude beats and a locked-exposure capture; report the
   sampling bound; re-measure on longer scans.
4. **Arterial stiffness:** gate on capture (≥ 50 fps + locked exposure — the SDPPG floor);
   otherwise "needs a better capture". Single-beat morphology is not recoverable from
   30 fps consumer video (Template Collapse, arXiv 2606.03802).
5. **Signal chain:** selective best-patch extraction (few patches by SQI, never averaged —
   codec artefacts are spatially coherent) re-gated on step-1 recordings; an interval
   artefact filter in the rhythm cleaner is a separate owner-approved change (golden tests).
6. **Measure it properly:** two scans 30 min apart per session; per-card agreement from
   `scripts/sheet_stats.py`. The corpus has no such pairs.

## How to start a session on this repo

1. The SessionStart hook prints the last iterations and the current baseline. Read it.
2. State which proxy you are moving and the one hypothesis, in one line.
3. Run the skill. Honor the verdict. Log it. Deploy only when the owner says so.
