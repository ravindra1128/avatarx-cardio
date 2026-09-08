# Data contract

What `replay.py` collects per record, where it comes from, and the field paths — so the
scorer never guesses at names. Everything is read from the return value of
`app.measure_api.measure_video(path, manifest={"capture_profile": "consumer"})`, which is
the exact production path (trim → downscale → sidecar clock → `inference.pipeline.run_with_details`
→ `report_biomarkers`).

## The `doc` returned by `measure_video`

| path | type | meaning |
|---|---|---|
| `doc["outcome"]` | `"ACCEPT" \| "REPEAT_SCAN" \| "NO_RESULT"` | the structured verdict — present on every successful call |
| `doc["no_read_reasons"]` | `list[str]` | why not ACCEPT; the pipeline's own wording |
| `doc["signal_quality_index"]` | float or None | SQI |
| `doc["confidence_stars"]` | int 1–5 | |
| `doc["debug"]["evidence"]` | dict | beat-fusion evidence (below) |
| `doc["debug"]["rationale"]` | dict | gates + coverage (below) |
| `doc["biomarkers"]["items"]` | list of 3 | the cards (below) |
| `doc["timing"]` | dict | server stage seconds (below) |
| `doc["trim"]`, `doc["downscale"]`, `doc["clock"]` | dict | prep provenance — record, don't score |

### `debug.evidence` (present when a beat lattice was built)

| key | used for |
|---|---|
| `cross_roi_coherence` | `signal` (COH term) |
| `timing_precision_ms` | `signal` (TIM term) |
| `n_intervals`, `n_beats` | reporting |
| `dropout_rate`, `frac_multi_roi` | reporting / hypothesis-finding |

### `debug.rationale`

| key | used for |
|---|---|
| `coverage` | `signal` (COV term) |
| `gates` | list of `{name, value, threshold, op, pass}` — reporting which gate failed |
| `sqi_components` | `{snr_in_band, skewness, spectral_concentration, cross_roi_coherence, tracking_stability}` — hypothesis-finding |

If `evidence` is `None` the scan was rejected at ingest (capture gate); every `signal`
term is 0 for that record and `no_read_reasons` says why.

### `biomarkers.items[i]`

| key | used for |
|---|---|
| `key` | `arterial_stiffness \| vascular_tone \| cardiorespiratory_fitness` |
| `status` | `"computed" \| "not_computed"` → `cards` |
| `value`, `unit` | `consistency` (per key, across records) |
| `reason` | why not computed — reporting |

**Caveat for consistency:** `arterial_stiffness` may come back as *different markers* on
different scans (`pulse_rise_time` in ms vs `reflection_index` as a ratio) depending on
which contour marker survived. Group by `(key, unit)` before computing a CV, never by
`key` alone.

### `timing`

| key | used for |
|---|---|
| `trim_s`, `downscale_s`, `analysis_s` | reporting where time goes |
| `server_total_s` | `latency` |
| `upload_received_s`, `upload_bytes` | only present via the HTTP path; absent in replay |

## The manifest (`data/eval_cache/eval_manifest.json`)

```json
{"records": [{"id": "webapp_1234567890-b45221", "path": "data/eval_corpus/webapp_1234567890-b45221.webm",
              "split": "holdout", "sha1": "…", "width": 1080, "height": 720, "codec": "vp8",
              "n_frames": 3900, "sidecar": false, "source": "webapp"}]}
```

`split` is assigned from the file's sha1 so it is identical on every machine and every
run. `source` is `webapp` (VP8/VP9 webm as the phone/laptop recorded it) or `demo`
(lossless FFV1 from the pipeline's own live demo, with an honest-clock sidecar).

## Known results on this corpus (don't re-derive)

Measured during the integration work, all on these same recordings:

- **Iteration 1 (2026-09-06), REJECTED: downscale 480 → 360 wide.** On the 6-clip
  holdout: signal 0.783 → 0.888, cards 8/18 → 10/18, determinism held, latency held —
  and consistency 0.617 → 0.419. The newly computed cards were the unrealistic ones
  (arterial-stiffness ratio 0.37/0.38 became 0.87/0.40/0.24; vascular tone 78/59 became
  14/68, one person). The native-480 demo clip also got downscaled and lost all three
  cards; one 1080×720 clip's coherence halved (0.233 → 0.109) while another's doubled.
  Frame width is the wrong variable — the size of the face in the frame is what sets
  pixels-per-ROI. Don't retry a fixed width; a face-size-normalised scale is the next test.
- **Tune screen, not taken to holdout: POS window 1.6 → 2.4 s.** Tune signal
  0.728 → 0.746 (under the margin), cards unchanged (one clip 0→3, another 3→0),
  consistency 0.425 → 0.140, latency 0.988 → 0.915. Coherence up on 6 clips, down on 4
  including the best demo clip (0.309 → 0.197). Same shape as the downscale: a trade
  between clips. A 5-point peak fit was dropped before replay: on synthetic pulses it cut
  jitter by only ~7 %. Synthetic calibration worth keeping: the 3rd-order 0.7–4 Hz
  bandpass at 30 fps gives 30–37 ms per-ROI peak jitter at 0–2 dB SNR, which is the
  ~50 ms inter-ROI disagreement observed — per-ROI SNR on these captures is near 0 dB.
  **POS 3.2 s** was worse on every axis (tune signal 0.725, cards 14/33, consistency
  0.012, mean job 21 s). The window lever is closed: 1.6 s is the best of 1.6/2.4/3.2.
  Note the job time tracks the window (11.6 → 14.2 → 21.1 s): the POS loop is a large
  share of analysis time — a speed lead, not a signal one.
- **Profile (one long webapp clip / one demo clip).** Every scan decodes the clip
  THREE times: the trim's duration probe (OpenCV loop: 13.75 s on a 1080×720 phone
  recording = 57 % of the request; 3.0 s on a demo clip it then didn't trim), the
  timestamp sidecar (`probe_video`, 1.4–3.0 s) and ingest itself. Inside analysis the
  face detector is ~60 % (YuNet per frame, 5 ms/frame); POS is 0.3 s. Only the first is
  in the skill's edit scope.
- **Iteration 2 (2026-09-06), REJECTED BY THE GATE RULE, NOT BY THE DATA: trim duration
  from packet timestamps.** `ffprobe -show_entries packet=pts_time`, second-largest pts
  (that is what the OpenCV loop returned — POS_MSEC before `read()` is the frame already
  decoded, one frame short of the true end; the cut point was kept identical on purpose).
  Verified equal to the loop within 1 ms on all 17 clips. Holdout: every evidence number
  and card byte-identical on 6/6, determinism 1.0, guard OK; trim 0.8–3.9 s → < 0.1 s,
  mean job 13.94 → 9.87 s (−29 %). gate.py said "no improvement" because its improved
  test credits signal/cards only and the latency score is capped at 1.0 under 15 s.
  Patch saved at `data/eval_cache/trim_packets.patch`; a proposed one-clause gate change
  is in the session scratchpad (`gate_speed_clause.py`) — owner's decision.
- **The cards' own gate (2026-09-07).** The three biomarker cards do not use the rhythm
  gates; they use `decision.evidence.endpoint_*`: coherence >= 0.10, timing <= 40 ms,
  matched fraction >= 0.35 (features/hemodynamics.py `limited_region_ok`), then
  >= 8 morphology-usable beats in >= 2 ROIs. On all 17 recordings every clip that
  has beats clears coherence 0.10 — the **timing ceiling is the only endpoint
  blocker**: 6/17 clear 40 ms, 8/17 clear 50, 13/17 clear 60 (before the matched and
  beats-per-ROI gates). Two webapp clips have no beats at all (no face / face out of
  frame) — capture, not pipeline. The real laptop scan of 2026-09-07 missed on timing
  alone: 0.11 / 44 ms / 0.42. Relaxing the ceiling is a frozen-config (owner) decision;
  tests/test_hemodynamics.py pins "good lighting alone cannot manufacture cross-region
  evidence" at 0.09 / 41 ms / 0.34, and a heart-rate-only fitness card on unverified
  beats is pinned as not_computed by test_abstention_still_exposes_method_and_signal_gate.
  Screen results at 40/50/60 ms (`data/eval_cache/screen_endpoint_tp.json`): cards
  computed 26 / 31 / 39 of 51; fitness values 50–80 (CV 0.13 / 0.15) at 40 and 50, but
  at 60 the 52–55 ms webapp clips yield fitness 6.6 and 7.6 (CV 0.39) — 60 admits
  implausible values, 50 does not. **Owner decision 2026-09-07: ceiling set to 50** in
  configs/default.yaml (rhythm gates unchanged); guard re-snapshotted on that config.
  Verified live: a 48 ms demo clip went 0/3 → 3/3 cards through the HTTP path.
- **Iteration 3 (2026-09-07), ACCEPTED: vascular tone's 12-beat floor moved from per
  capture segment to the scan total** (`features/hemodynamics.py`: `segment_amplitudes` +
  `pool_amplitudes`; `amplitude_series` kept as the one-segment wrapper). Root cause: phone
  recordings are split into 4–5 capture segments of 4–11 s by frame stalls; the floor was
  applied per segment, so regions holding 23–27 usable beats in total reported "0 usable
  beats". Per-segment median normalisation is KEPT (it removes the auto-exposure gain jump
  between segments — a first version that used one median per region produced 87–98 %CV
  and moved a single-segment clip's value by changing region order; rejected by hand
  despite a gate ACCEPT). v2: holdout cards 8/18 → 9/18, signal identical, determinism
  1.0, consistency 0.617 → 0.640, single-segment clips identical to the digit, new values
  51.4 / 62.2 %CV; tune: the 5-segment clip 0 → 49.6 %CV with 3/3 cards.
  Lesson for the gate: a per-record "unchanged where it should be unchanged" check is
  stronger than the aggregate metrics — read the table, not just the verdict.
- **Per-ROI diagnosis (tune split).** No region is dead: beat counts are balanced across
  forehead/cheeks/nose (≈30–40 each per clip). Coherence is low because the four regions
  place the *same* beat > 60 ms apart and fail to cluster — ~40 % of per-ROI beats pair
  with nothing; most fused beats carry only 2 regions. The tightest pair sits ≈ 20 ms, the
  pair the pipeline reports (highest matched fraction) ≈ 50 ms. It is per-ROI peak-timing
  jitter from low per-ROI SNR, not a bad region or a bad rule. Two webapp tune clips have
  zero beats for capture reasons (no face > 2 s / face out of frame) — not the pipeline's.

- **Resolution is the lever for coherence.** The webapp records ShenAI's 1080×720; the
  pipeline's own demo records 480×360. The same lossy clip as recorded: coherence 0.122,
  no cards; downscaled to 480 wide server-side: 0.312, all three cards. The service now
  downscales every upload (`app/measure_prep.py`), losslessly (FFV1 — faster *and*
  deterministic vs VP9, which varied run to run by as much as the gate margin).
- **Bitrate:** 2.4 Mbps fails the 0.12 bpp floor at 1080×720; 8 Mbps passes; 5 Mbps kept
  all three cards with *better* coherence (0.233→0.250) at 33% fewer bytes; 3.5 Mbps
  dropped under the coherence gate. The webapp records at 5 Mbps.
- **Aspect ratio must be preserved.** A forced 480×360 on a 3:2 source squeezed the face
  11% and made a geometry gate *pass* by distortion. Scale by the long edge only.
- **The non-finite ROI check was all-or-nothing** — 3 bad frames in 2824 discarded a
  whole ROI. Now `MIN_FINITE_FRACTION = 0.98` (a coverage floor, like the rest of the
  pipeline). This is a protected threshold.
- **Narrowband ROI orientation is not a fix.** It moved coherence 0.065→0.186 on one
  clip and 0.174→0.079 on another — opposite signs, n=2. Left as an opt-in launch
  override, default off. Don't propose it again without a corpus-wide result.
- **Latency:** on the dev Mac a 37 MB clip is trim 0.7 s + downscale 1.9 s + analysis
  6.9 s ≈ 11 s; on Railway the job is ~14–17 s (CPU on par with the Mac). Everything
  else in a slow request is upload and edge relay — not the pipeline's, not scored here.
- **Trim before downscale** — both are lossless, so the analysed frames are identical
  either way, but trimming first hands the encoder only the frames that will be used.
- **Every scan already returns a structured outcome** by design (`run_with_details`
  never raises for quality problems). Constraint 2 is about keeping that true.
