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
- **Display tiers are the one sanctioned exception (owner decision 2026-09-09 evening):** a
  card may SHOW a provisional score below a floor, from real beats, labelled provisional with
  the reason. It may never show a value that was not computed from that scan's signal, and
  the `measured` label keeps every gate.
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
- Deploy: push to remote `personal` (ravindra1128/avatarx-cardio) → Railway rebuilds in
  ~1 min. Verify with `GET /healthz` → `build` (the response's `code_commit` is always
  `no-git` on Railway) and `config_hash`. Remote `origin` is the org repo; the owner pushes it.

  **Two branches, two Railway environments (set up 2026-09-10).** Until then every
  experiment went straight at the one service the owner's phone scans, which is how a
  bitrate change that broke the SQI floor reached real scans before it could be caught.

  | branch | Railway environment | URL | who points at it |
  |---|---|---|---|
  | `main` | production | `avatarx-cardio-production.up.railway.app` | the webapp's beta/production `VITE_AFIB_URL` |
  | `staging` | staging | `avatarx-cardio-staging.up.railway.app` | the webapp's `/beta/cardio-staging` route |

  Both verified live 2026-09-10.

  **Where a candidate change gets tried.** The webapp has a second route,
  `/beta/cardio-staging`, that is a full copy of `/beta/cardio` — its own scan page,
  results page, AFib processor, camera-lock module and biomarker cards — wired to the
  staging service and to its own `afib_result_staging` key. So a change can be scanned on
  a real phone without altering `/beta/cardio` at all. Webapp-side detail and the file
  pairs are in that repo's `CLAUDE.md`; a jest test there fails if the halves cross.

  Service-side, the same discipline: push to `staging`, measure, then merge to `main`.

  Work goes to `staging`, is measured there, and only then merges to `main`. The webapp
  already splits this way: `VITE_AFIB_URL` is a per-environment GitHub secret and
  `.github/workflows/staging-deploy.yml` runs on the `staging` branch.

  Two things the Railway environment MUST get right, neither of them code:
  - its own `AFIB_SHEET_GID` pointing at a **different sheet tab**. Every comparison in
    this repo works because production scans all land in one place; a staging service
    writing into that tab silently corrupts the baseline it is being measured against.
    Resolved 2026-09-10: staging writes `cardio-staging` (gid `342232654`), production
    `cardio-data` (gid `140358874`). Check `GET /healthz` → `sheet.gid` on both before
    trusting any staging measurement; if they ever match again, staging is polluting the
    production data.
  - its own `GOOGLE_SHEETS_CREDENTIALS_JSON`. Variables are per-environment on Railway
    and are NOT inherited from production.

  Nothing in the code changes for this: the sheet id, tab and credentials are already
  read from the environment (`app/result_sheet.py`).

  **Retaining scans and ShenAI sidecars on staging** needs two more variables there, and
  the way they are read matters (learned the hard way on 2026-09-12 — eight real scans
  with both set produced nothing):
  - `AFIB_KEEP_UPLOADS` — any of `1`, `true`, `yes`, `on`, case-insensitive. Before
    2026-09-12 only the literal `1` worked and `true` failed silently.
  - `AFIB_CLIPS_TOKEN` — a long random string. Read on every request, so adding it in the
    console works immediately; before 2026-09-12 it was read once at import and a token
    added after startup was invisible until the next redeploy.
  - The gate's state is printed ONCE at startup in the Railway service log
    (`[clips] retention ON` / `OFF: <which variable>`) and again on every dropped
    sidecar. It is deliberately NOT on `/healthz`: whether a public URL holds face video
    is the oracle the token exists to deny.
  - Self-check with the real token: `GET /api/clips?token=<value>` → `200` means fully
    on, `404` means off (or wrong token — by design the two are indistinguishable).
  - `AFIB_SCALE` sets the analysis box; `640x480` (a 480×720 phone clip → 320×480) beat
    both the `480x360` default and production on every signal metric across 5 scans on
    2026-09-12. Production still runs the default.
  - The trim cuts on the packet clock (`app/measure_prep.py: choose_tail_cut`), not at
    "duration minus window". The phone's rolling recorder uploads chunk 0 (header + the
    first 3 s of frames), a hole, then the last 48 s; a copy seek into the hole resolves
    to frame 0, so until 2026-09-14 every rolling-window scan was analysed WHOLE — head,
    hole and all, with chunk 0's backwards clock step rewritten as 57 frames on one
    timestamp (the `MAX_COLLAPSED_INTERVAL_FRACTION` 0.02→0.05 launch override exists
    for that). Any session over ~115 s also failed the downscale: the phone's millisecond
    clock makes ffmpeg guess 1000 fps, and the AVI muxer refuses a hole over 60000 ticks
    = 60 s ("Too large number of skipped frames" — 4 of 11 staging scans, reproduced
    exactly in `tests/test_trim_tail.py`). The sheet's `Trim Note` column says per scan
    what was kept and dropped; `Downscale Note` keeps the first lines of ffmpeg's stderr,
    which name the cause.
- Local test: webapp `.claude/launch.json` starts this service on 8790 and a webapp on
  5174 pointed at it. Phone scans are ~30–50 MB; the Railway edge relays uploads at
  ~0.5–1 MB/s, so most of a phone's wait is upload, not analysis.
- Tests: `python -m pytest tests/ -q -p no:cacheprovider --ignore=tests/test_findings_only.py
  --ignore=tests/test_report_forbidden.py --ignore=tests/test_flutter_forbidden.py
  --ignore=tests/test_vascular_forbidden.py --ignore=tests/test_vasotone_forbidden.py`
  → baseline 800 passed, 4 known failures (listed in the skill), 54 min. The three card
  modules (`test_hemodynamics`, `test_realworld_hardening`, `test_scan_engine`) take 6 min.

## Audit of 2026-09-17 (state of the staging branch)

Deep audit of the running pipeline (63 findings, 8 of 10 stages agent-read,
the rest hand-verified). The binding constraint on phone scans is NOT the
quality floors: 5 of 12 real scans cleared every any-call gate and still
abstained because the interim irregularity rule (median|dRR| >= 60 ms AND
pNN50 >= 0.40, synthetic-era thresholds) fires on a regular rhythm at the
phone's 20-59 ms beat timing, after which only the AF-grade bar or an
abstention is reachable. The offline holdout cannot see this stage: none of
its 8 phone clips reaches the classifier under either capture profile.

Changes on `staging` since (each its own gated iteration; "owner to confirm"
= gate said no-improvement with zero regression, kept as correctness):
- #1 sheet columns MAD ms / pNN50 / Rate N (the classifier's inputs per scan).
- #6 /api/start takes the worker slot before consuming the slices; job cleanup;
  client treats a cached server error as terminal.
- #3 fusion tolerance 60 -> 90 ms + same-ROI merge — ACCEPTED (signal 0.602 -> 0.652,
  consistency 0.539 -> 0.652). 120 ms rejected: noise clears the SQI floor from 110.
- #4a POS overlap-add normalisation — REJECTED (signal 0.65 -> 0.74 but consistency
  -0.09); patch kept in data/eval_cache/pos_normalise_iter23.patch.
- #4c runs never span a capture gap — owner to confirm.
- #7 one resting-rate resolver; HIGH_RATE fails closed on an unverified count — owner
  to confirm (strict-abstain and provisional-count variants both rejected by the gate).
- #2 noise-aware irregularity rule — owner to confirm (simulation: false-irregular at
  40 ms 0.24 -> 0.01; AF pattern 1.00 -> 0.97; needs MIMIC PERform for AF on real hearts).
Not done: #5 recorder 48 -> 72 s (upload-cost A/B on the phone), #8 graded abstention,
#10 MKV passthrough clock, #11 refractory floor, #12 tracker fallback, #13 stars hint.

Added 2026-09-16 (afternoon), from the first three retained staging clips with ShenAI sidecars:
- Iteration 27 (trim, `app/measure_prep.py`) — a forward gap under `SPAN_GAP_S` (5 s)
  inside the tail is a lost recorder chunk, not the tail start; spanned, and the
  verifier accepts it. Corpus is pre-trimmed so the gate is blind; live evidence is the
  08:53 scan (14.6 s kept of 44.9 s, 12 intervals, every other gate passing). Owner to
  confirm.
- Iterations 28/28b (`capture_segments` gap_factor 1.5 -> 2.5 / 2.0) — REJECTED, reverted.
  The rule splits at the phone's normal frame wander (p99 50 ms vs a 49.5 ms threshold)
  and discards every fragment under 3 s: 17.5 s of the 08:53 window. Both variants took
  that scan from three failing gates to one (coverage 0.52 vs any-call 0.60) but the
  holdout's CARD values swung with four extra intervals (tone 54 -> 100, fitness 82 -> 20),
  consistency 0.652 -> 0.31 / 0.43. The blocker is card instability, not the segment rule:
  make the cards robust to the interval set first, then retry 28.
- Iteration 29/29b (cards, `features/hemodynamics.py`) — the fitness card abstains where
  the resolver says 'uncertain' (doubling/split signature, no fold backing), naming both
  estimates; live evidence 10:09/10:13 (counts 109/114 vs 78 bpm rhythm, reference 72-74,
  fitness 6.7/6.5 published). Holdout metric-identical; shipped on the owner's word. The
  robust tone CV tried beside it was REJECTED (phone clips' tone fell 54 -> 34, 55 -> 26
  vs rig 81 -> 72; consistency 0.652 -> 0.472). Still open: a count with no spectral
  anchor at all (10:09: 108.8 bpm from 5 intervals, "no dominant rhythm") is still shown
  provisional per the 2026-09-09 decision - nothing in the pipeline can contradict it.
- **ShenAI train route** (`inference/shenai_route.py`, wired in `app/measure_api.py`,
  owner: "build the shenai train route", 2026-09-16). The sidecar's beat train is a SECOND
  interval source for the SAME decision (`decide_with_rationale`, same gates, text, star
  coupling, rate resolver). It runs only when the video path abstained on interval gates
  alone (coverage / count / split / NaN features) with every scan gate our regions provide
  passing (sqi, coherence or two-region, timing); the train must be contiguous (a dropped
  beat is a segment break), >= 15 beats over >= 20 s, SDK quality >= 0.5, and its RMSSD must
  agree with the SDK's own lnRMSSD (regularisation control); and its rate must be
  corroborated by OUR evidence on the scan (resolved rate verified/provisional, or the
  waveform rhythm with >= 2 regions, within 15 %). Noise floor: mean_roi_agreement 1/4
  (k = 1, one unaveraged detector at the scan's timing precision). The sidecar is HELD IN
  MEMORY per in-flight upload (`_SIGNALS`, never disk; the retention gate is untouched) and
  the job waits up to `AFIB_SHENAI_WAIT_S` (8 s) for it; `AFIB_SHENAI_ROUTE=0` disables.
  Response: `rhythm_source` ("video" | "shenai_train"), `debug.shenai_route` (used or not,
  with the reason), the video path's own answer kept under `debug.video_rationale` /
  `debug.video_outcome`. Sheet: `Rhythm Source`, `ShenAI Route`, `ShenAI Rate`.
  Offline on the nine 2026-09-16 clips (consumer profile, 640x480, 45 s): used on 3 - all
  ACCEPT/SINUS, MAD 16-21 ms, pNN50 0.05-0.14, pulse 68-75 vs SDK 67-72; refused on 2 for a
  scan gate (sqi / timing) and on 4 for corroboration - in three of those our own spectral
  rhythm sat at 51-53 bpm against a 73-75 bpm train the SDK's HR agreed with, i.e. OUR
  side was the wrong one and the route still refused, by design. Nine clips, not the ten
  the harness rule asks for; AF sensitivity through the route is untested on real AF
  (same classifier, same MIMIC PERform gap as #2). Holdout gate is blind (replay calls
  measure_video, no sidecars): metric-identical.
- ShenAI comparison (`scripts/compare_shenai_signal.py`, harness fixed to follow the
  downscaled path): on the same two scans ShenAI's own beat train gave 49 and 51 clean
  intervals at coverage 0.98/0.94 (RMSSD 28/32 ms, SDNN 31/57 ms, not TOO-GOOD) against
  our 4 and 13 at 0.06/0.19. The waveform arm is VOID (no SDK sample rate) and our detector
  on its waveform over-counts 15-22 %, so the usable ShenAI signal is the TRAIN. Two clips;
  the pre-registered rule needs ten -> NOT SETTLED. The sidecar now carries the SDK's own
  lnRMSSD and bad-signal seconds (webapp `0a17167`); on the 08:53 scan RMSSD from its
  train equalled its own figure (14.8 ms) exactly. Retained clips are wiped by every
  redeploy: pull before pushing.

## Redesign of 2026-09-17: one AFib result per completed scan

**Goal (owner):** every successfully completed face scan returns exactly one of
AFIB_DETECTED | AFIB_NOT_DETECTED | INCONCLUSIVE, from a consumer camera; Inconclusive
only when the evidence cannot support a decision. Full freedom to redesign.

**What the literature settles (read 2026-09-17):**
- Every consumer-camera AF result with clinical numbers is an RR-interval-variability
  classifier on beats from LIVE, UNCOMPRESSED frames: Cardiio Rhythm on iPhone facial PPG
  (Yan 2018, JAHA: 217 inpatients, 3x20 s, sens 95 % / spec 96 %), Couderc's VPG on Android
  (2022, >= 90 %/90 %, >= 100 lux), FibriCheck fingertip camera PPG (1 min; sens 95.6 % /
  spec 96.6 %; 7-17 % insufficient quality; 8.5 % "inconclusive" in the 2026 head-to-head).
  The 453-patient facial deep-learning study (Sci Rep 2022) used an 84 fps industrial
  camera, controlled light and 10 min: 30 s segments reached only 80-95 % sensitivity.
- Compression is the wall: McDuff 2017 puts the floor for a usable rPPG signal at ~10 Mb/s;
  Android Chrome caps our recorder at ~7 Mb/s at 480x720 (measured), and our own harness
  measured cross-region correlation 0.33 (rig) -> 0.16 (phone clip). No server-side
  algorithm recovers what the codec removed.
- Facial-video AF datasets are not public (OBF: 100 healthy + 6 AF, on request). Public AF
  ground truth is RR/PPG: MIT-BIH AFDB, MIMIC PERform AF (Zenodo 6807403, CC-BY, 35 ICU
  subjects, 19 AF) - the repo already validates on the latter (E6, `models/model_a_v01.json`).

**Decision - the AFib Evidence Engine:**
1. Interval sources come from live frames (today: the ShenAI train route; next: our own
   on-device ROI traces); the compressed clip is the VERIFIER (quality, coherence, timing)
   and the source of the cards, no longer the primary interval source.
2. The classifier is the validated RR model, not the hand rule. E6b
   (`scripts/e6_window45.py`, 2026-09-17) re-fits it at OUR window: 910 x 45 s windows,
   35 subjects, participant-level OOF: AUROC 0.987 at 10 ms timing noise -> 0.967 at 60 ms
   (3 % false beats), 0.964 at 60 ms / 8 %. Deployment artifact `models/model_a_v02_45s.json`
   is a pooled fit over 10/20/30/40 ms.
3. THREE-WAY DECISION BAND from the artifact (`decision_band`): tau_lo 0.474, tau_hi 0.825,
   set on out-of-fold scores at the phone-like setting (30 ms, 8 % false): decided windows
   sens 0.954 / spec 0.972 with 9.0 % inconclusive; the same band holds spec >= 0.94 up to
   40 ms noise and degrades past it (0.90 at 50 ms, 0.79-0.81 at 60 ms) - which is why the
   existing 40 ms any-call timing gate stays as the precondition. `inference/afib_result.py`
   maps (outcome, class, probability, gates) -> the result + a basis naming which of
   capture / signal / rhythm was missing. Sheet: `AFib Result`, `AFib p`, `AFib Basis`.
   Webapp staging card: the three words as the primary chip; `user_facing_text` verbatim.
4. STAGING SWITCH (no yaml edit): Railway variable
   `AFIB_CONFIG_OVERRIDES={"decision.classifier":"model_a","decision.model_a_path":"models/model_a_v02_45s.json"}`
   - applied through `_apply_overrides` to every job, echoed on every response and on
   /healthz `launch_overrides`. Without it the interim rule runs and the result layer decides
   by class (SINUS/HIGH_RATE -> NOT_DETECTED, AFIB_SUGGESTIVE -> DETECTED, OTHER_IRREGULAR ->
   INCONCLUSIVE). Offline on the retained clips with the switch: the two ShenAI-route scans
   gave p = 0.007 / 0.003 -> AFIB_NOT_DETECTED.

**E6c (real PPG beats through OUR detector, `scripts/e6c_ppg_end_to_end.py`):** MIMIC
PERform's contact PPG at 30 Hz -> beats/detector.py -> production features -> the classifier,
each fold fitted on OTHER participants' ECG-derived windows. Detector found 0.97 of ECG beats
(IQR 0.88-1.00). OOF AUROC 0.939 [0.81, 1.00]; under the shipped band 4 % inconclusive,
decided sens 0.991 / spec 0.917; 34 of 35 subjects majority-correct. Specificity is the
weaker side on real beats (missed beats raise irregularity on non-AF windows) - the price of
the detector, not the model, and the reason the band's precondition gates matter.

**Iteration 31 (2026-09-17, route corroboration):** on the four 2026-09-17 scans (SDK 84-90
bpm) our dominant waveform rhythm read 45-48 on 3 regions - a 0.8 Hz artefact out-powering the
1.4 Hz pulse, not the subharmonic branch. `scripts/spectral_variants.py` scored four estimator
variants against the SDK's rate on 13 clips: the best agrees on 6/13 (current 4/13) - the
dominant peak of a 7 Mb/s clip is not a rate estimate, so no estimator was gated. Instead a
third corroboration: a LOCAL PEAK in our fused spectrum at the train's rate (exact bin,
>= 2x in-band median) - 9/13 true rates confirmed, 4 % false pass (a +-1 bin tolerance
triples that). Retained-clip replay with model_a: route used 9/16 (was 4), every one
AFIB_NOT_DETECTED with p <= 0.05; 7 INCONCLUSIVE (3 scan gates, 1 face, 3 uncorroborated).
STILL OPEN: the same wrong dominant rhythm drives the rate resolver's fold and the fitness
card (87/100 at a folded 45 bpm on 06:11; 5.5 at an unfolded 112 on 06:15) - the fold should
require the peak test too, or the card should abstain; owner decision.

**Caveats, stated:** MIMIC PERform is ICU ECG-derived RR with SYNTHETIC rPPG degradation,
35 subjects; no facial-video AF validation exists here or publicly; markov_surprise and
spectral_entropy are NaN on most 45 s windows and are median-imputed (near-inert).
The 3-star floor and the AF-call verification gates are unchanged: an AFIB_DETECTED still
needs coherence >= 0.35 or a two-region timing match, >= 20 intervals, and 3 stars.

**Next (in order):** (a) on-device ROI traces from the live frames (the SDK exposes
`getNormalizedFaceBbox()`; ~100x cheaper than the abandoned JPEG stream in
`afibLiveStream.js`; ship beside the clip as a second sidecar for offline validation first;
production wiring needs `run_with_details` split into ingest + analyse, a hook-protected
file the owner must open); (b) recorder to 60-70 s so one window holds >= 20 intervals
even at today's yield; (c) the corroboration rule for the train, revisited with the
sheet's `ShenAI Route` reasons once ~30 scans exist; (d) real-AF validation through the
route: the MIMIC PERform PPG channel run through OUR beat detector end to end.

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
| 2026-09-09 (evening) | DISPLAY availability over reliability: every card shows a 0-100 score on every scan that yielded beats, tiered `measured` (every floor held) or `provisional` (a floor or gate did not hold; the score comes from the beats the scan did yield, labelled with the reason); blank only with no beats at all. Scores are monotone maps of the measured marker with a typical range beside them; the raw marker travels with every value. Fitness anchors moved to population values (72 bpm centre, 14 spread) | the owner asked for a result on every scan in a normal-looking range after a day of blank cards; the morning's abstention gates now demote to provisional instead of blanking. Nothing is invented: a scan with no beats still says so. Provisional values WILL move between scans - that is thin evidence showing, and the label says so |
| 2026-09-09 (late) | stiffness reports the reflection index as a RATIO again (0.35, not 35); with no dicrotic notch it reports a BAND (High/Typical/Low) from the pulse crest time, never a second number in the same slot. The card no longer renders `limitation` or `warning` | the 0-100 rescale made a familiar 0.35 unrecognisable, and a 210.5 ms crest time in the same slot as a 0.35 ratio is what read as unrealistic. Direction checked against the PPG literature and CORRECTED: a stiffer artery carries the wave faster, so a SHORTER crest time means higher stiffness - this file had it backwards. Thresholds 120-320 ms are rules of thumb, confounded by pulse rate, hence band-only and provisional-only. The two removed lines are unchanging caveats that repeated on every card and buried the one line that changes; both still travel in the payload |

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
