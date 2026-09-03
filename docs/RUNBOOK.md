# AvatarX AFib v0.1 — research-assistant runbook

How to process a day's recordings and read the evaluation report. This is
an **offline research pipeline**; nothing here is a diagnosis, and every
participant-facing sentence the tooling can produce comes from one audited
function.

## 0. One-time setup

```bash
pip install numpy scipy opencv-python-headless pytest
python3 -m pytest tests/ -q        # must say: 390 passed (1 skip is OK:
                                   # it needs the kept live recordings)
```

Optional (better face tracking on real footage): download a mediapipe
`face_landmarker.task` model and export
`AVATARX_MEDIAPIPE_MODEL=/path/to/face_landmarker.task`. The report's
provenance records which tracker actually ran.

## 1. Lay out the day's recordings

One directory per collection day; for each recording id `R` three files:

```
day_2026-08-14/
  R.recording.json   # manifest, datasets/schema.py Recording shape
  R.avi              # the face video (path referenced by the manifest)
  R.ecg.json         # reference ECG R-peaks (collection rig output)
```

The manifest is produced by the collection rig. It must carry the metered
lux, the lock states, the codec/bitrate, the PRBS sync record and the
adjudicated rhythm annotation — **the pipeline believes the manifest, and
the gates judge it.**

## 2. Validate before anything else

```bash
python3 cli.py validate day_2026-08-14/R.recording.json
```

`valid_for_beat_analysis: false` with reasons means the recording is not
usable for beat-level work. Do not argue with the gate; fix the capture
(light, locks, codec, sync) and re-record. There is deliberately no
override flag.

## 3. Process single scans (spot checks)

```bash
python3 cli.py process day_2026-08-14/R.avi --manifest day_2026-08-14/R.recording.json
```

Output is a `ScanResult` JSON. Things to check:

- `outcome`: `ACCEPT`, `REPEAT_SCAN` (quality — re-scan-able) or
  `NO_RESULT` (structural), with `no_read_reasons` spelled out;
- `predicted_class`: `SINUS | AFIB_SUGGESTIVE | OTHER_IRREGULAR |
  HIGH_RATE` — only present when every gate passed;
- `user_facing_text`: the ONLY sentence a participant may ever be shown;
- provenance: `code_commit`, `calibration_version`, `config_hash`,
  `model_version` — if any says `unknown`, stop and report the install.

## 4. Evaluate the day against ECG truth

```bash
python3 cli.py evaluate day_2026-08-14/ --out day_2026-08-14/report
```

Read `report/evaluation_report.md` top to bottom:

1. **Excluded recordings** — every exclusion has schema reasons. An
   unexpected exclusion usually means a rig misconfiguration (codec,
   lux, sync marker count).
2. **Beat metrics by rhythm** — read the AF row separately; pulse deficit
   hides in aggregates. Gate 1 (non-AF) and Gate 1b (AF) verdicts are
   per-recording pass counts plus production-RMSSD error and confidence
   ECE against their gates.
3. **Classification** — sensitivity/specificity with cluster-bootstrap
   CIs. Raw accuracy is intentionally absent; PPV is meaningful only at
   deployment prevalence.
4. **Risk-coverage** — watch the `AF retained` column, not just Se/Sp:
   quality gating that sheds AF scans is a failure mode, not a win.
5. **No-read parity** — abstention rates per axis including
   `P(no-read | AF)` vs `P(no-read | non-AF)`. A parity FAIL is a
   reportable finding even when everything else passes.
6. **Serial confirmation** — always printed with its
   `fp_persistent_share`; figures without the share are not quotable.
7. **Leakage audit / split composition** — must be PASS before any number
   from the day enters a training or evaluation pool.

## 5. What to escalate

- Any gate FAIL that repeats across a day (capture defect or drift);
- parity FAILs on any axis;
- `NO_RESULT` rates above ~15% in controlled conditions;
- any output text that did not come from `user_facing_text()` — that is a
  sev-1 defect, stop the line.

## 6. What never to do

- Never quote a synthetic-video number as performance; the report header
  says so and so must you.
- Never re-run with a lowered threshold to make a gate pass; thresholds
  change only through a reviewed spec change.
- Never split, pool or resample by recording or window — participants are
  the unit, and the leakage audit will catch it anyway.

## 7. Live camera demo (v0.1.1)

For a hands-on scan of your own pulse rhythm:

```bash
python3 cli.py demo
```

This opens a local web app in your browser. Sequence:

1. **Welcome** → click Start.
2. **Camera permission** — your browser asks; choose Allow. (The camera
   stays on your device; frames are analysed locally, nothing is uploaded.)
3. **Framing / confidence** — centre your face in the oval. In advisory mode
   the countdown begins once the face, framing and hard camera-rate checks
   pass; pulse evidence continues maturing during the scan. A live 1-5 star
   meter grades the conditions with one actionable hint at a time. A result
   below 3 stars can never be an
   AF call (docs/READINESS.md). Every evaluation is
   logged to `/tmp/avatarx_live/<id>/readiness_log.jsonl` — send that
   file with any report.
4. **30-second scan** — the ring counts GOOD seconds only. Face loss, poor or
   clipped light, movement, or unstable camera delivery/tracking can pause and
   resume. Weak/incoherent provisional pulse evidence produces live guidance
   without blocking completion; the final pipeline may still return an
   explicit no-result. Early finish is refused server-side.
5. **Processing** — the recorded scan runs through the full production
   pipeline (the same one behind `cli.py process`).
6. **Results** — the **AvatarX Cardiac Rhythm Scan Report**, findings
   only since v0.3: a printable one-pager with the measurements box and
   the findings box (sanctioned sentence + stars + caveats + referral
   line) and **no waveform of any kind**. Print/save as PDF from the
   button; or export directly:
   `python3 cli.py process <video> --report out.html`. Setting
   `report.mode: full` in configs/default.yaml restores the v0.2.1
   layout (labeled rPPG strips + beat-interval trend) for research use.

Notes for operators:

- The demo runs under the **consumer capture profile** (see §0 / spec B.8):
  it relaxes only the AE/AWB-lock and the strict 30 fps floor (down to 24),
  records both truthfully, and surfaces them as caveats on the result.
  Everything else fails closed exactly as in the research path.
- Recordings are kept lossless under `/tmp/avatarx_live/` with an honest
  per-frame timestamp sidecar (the browser's capture clock).
- To verify the whole journey without a camera:
  `python3 scripts/selftest_headless.py` drives a real headless Chrome with
  a fake camera through the actual `getUserMedia` client path and checks the
  server's result. Needs Google Chrome installed.

## 8. The reconstruction research track (v0.3)

Operator loop for the gated ECG-reconstruction question
(docs/reconstruction_track.md is the authority):

```bash
python3 cli.py train configs/train_reconstruction.yaml   # + §G auto-eval
python3 cli.py reconstruct scan.avi --manifest rec.json  # research artifact
python3 cli.py gate-status                               # the scoreboard
```

Everything it writes lives under `research/runs/`, watermarked
`SYNTHETIC ECG — RESEARCH ARTIFACT — NOT A MEASUREMENT`. Never show a
reconstruction to a participant or clinician as if it were a
measurement — while any §G gate is red (`gate-status` says which and
why), that artifact class is research-only, and `cli.py promote` will
refuse it. Paired capture sessions ingested with morphology labels
(`labels[].morphology`: conduction pattern + measured PR/QRS/QT per
adjudicated segment) are what eventually make the evidence count:
register the dataset, point `facial_dataset_ids` at it, set
`source: facial`.

## 9. The three-phase recovery session (v0.4)

Enable with `protocol.three_phase: true` in configs/default.yaml (or
`AVATARX_THREE_PHASE=1` for one launch):

```bash
AVATARX_THREE_PHASE=1 python3 cli.py demo      # live guided flow
python3 cli.py session --manifest proto.json   # offline session
python3 cli.py evaluate-fitness <dataset_dir>  # CPET harness + §V verdict
python3 cli.py gate-status --track vo2         # the fitness scoreboard
```

What to know at the bench:

- The report shows MEASURED recovery physiology only while §V is red.
  Never present a fitness category or any VO2 number to a participant —
  the surfaces cannot render one, and that is a feature, not a gap.
- HR is never read during the activity; the camera counts reps/cadence
  (tracker recorded in provenance, `motion_energy` fallback documented).
- Compliance bands: within ±10% compliant; 10–20% REPEAT_SCAN; > 20% or
  transition > 10 s → NO_RESULT for recovery metrics, vitals still
  shown. Do not coach participants to "fix" a non-compliant session by
  re-labelling — re-run the protocol.
- Diagnose a session with `python3 scripts/why_not_ready.py
  <vsession dir>` (per-phase readiness + verdict + transition time);
  verify the whole flow without a camera via
  `python3 scripts/selftest_headless.py --session`.

## 10. The arterial-stiffness research track (v0.4 vascular)

Operator loop for the gated stiffness question
(docs/vascular_track.md is the authority):

```bash
python3 cli.py register-dataset day_pwv/            # paired campaign day
python3 cli.py vascular-fidelity day_pwv/ --domain facial_rppg
python3 cli.py evaluate-vascular  day_pwv/ --domain facial_rppg
python3 cli.py gate-status --track vascular         # the scoreboard
```

`--domain` defaults to `synthetic` (surrogate/machinery-only): a real
campaign day MUST be recorded as `facial_rppg` — only facial-domain
evidence can ever open a gate, and a mislabeled run simply wastes the
day.

At the bench:

- A campaign day pairs each recording with `<id>.ppg.json` (synchronized
  contact PPG) and `<id>.pwv.json` (tonometry reference: cfPWV, device,
  operator, same-visit BP/HR, meds flags, Fitzpatrick group; age/sex are
  BASELINE-ONLY inputs). Two scans + two reference reads per visit for
  the repeatability subset.
- The fidelity study decides which morphology features any model may
  use — by the explicit thresholds in configs/gates.yaml, automatically.
  Never hand-pick features past a failed ICC; that is the swamp.
- Never present a stiffness estimate, a "vascular age", or any m/s value
  to a participant — the surfaces cannot render one while the gates are
  red, and the token tests enforce it. Research estimates exist only in
  watermarked reports under the vascular runs root
  (`python3 cli.py process <video> --heads afib,vascular` prints the
  pointer to stderr).
- A result quoted without its B3 (age+sex+BP) delta is a bug, not a
  finding (invariant V-d) — the report builder refuses to produce one.

## 11. The vasomotor-reactivity research track (v0.5, §W)

Operator loop for the gated tone question (docs/vasotone_track.md is
the authority):

```bash
python3 cli.py register-dataset day_tone/
python3 cli.py evaluate-vasotone day_tone/ --domain facial_rppg
python3 cli.py gate-status --track vasotone       # the scoreboard
```

At the bench:

- A provocation day pairs each recording with `<id>.provocation.json`
  (maneuver + timed phase marks from the rig's prompts; null_optics
  arms MUST carry the optics_log) and, for provocation arms,
  `<id>.pi.json` (oximeter perfusion-index trace from a hand NOT in
  the maneuver). Both null arms are mandatory parts of the campaign —
  a day without them cannot move gate W1.
- AE/AWB lock is not optional for this track: an unlocked capture is
  flagged uncontrolled_optics and excluded from gate evaluations (W-d).
- Everything is a within-session response; there is no absolute tone
  number to write down, and a metric emitted as one is a bug (W-c).
- Never present a tone/reactivity reading to a participant — the
  surfaces cannot render one while §W is red, and the token tests
  enforce it. Research readings live in watermarked reports under the
  vasotone runs root (`python3 cli.py process <video> --manifest
  <id>.recording.json --heads afib,vasotone --provocation p.json
  --pi pi.json` prints the pointer to stderr — the manifest is how
  the AE/AWB lock evidence reaches the head; without it every
  session reads as uncontrolled optics).

## 12. The atrial-flutter research track (v0.6, §F)

Operator loop for the gated regular-tachyarrhythmia question
(docs/flutter_track.md is the authority; docs/flutter_limitations.md is
the known-miss registry and is required reading before any clinic
conversation):

```bash
python3 cli.py register-dataset day_flutter/
python3 cli.py evaluate-flutter day_flutter/ --domain facial_rppg
python3 cli.py gate-status --track flutter        # the scoreboard
```

At the bench:

- **Capture at 60 fps whenever the device allows it.** The "too
  regular" criterion is limited by frame quantization, not physiology:
  a perfectly metronomic pulse reads RMSSD 15.0 ms at 30 fps and 8.2 ms
  at 60 fps. At 30 fps the age-adjusted physiological floors are
  entirely masked, and every such scan is marked `measurement_limited`.
- **The torso must be in frame.** Respiration comes from torso motion
  in a second decode, and without it the head ABSTAINS — it does not
  fall back to dispersion alone, because at 150 bpm sinus tachycardia
  and 2:1 flutter are the same measurement without the respiratory
  coupling. A scan of a face alone cannot move gate F1.
- **A flutter positive needs a 12-lead with EP-level adjudication**,
  with atrial rate, conduction ratio and flutter type recorded in the
  label. Single-lead ECG is fine for negatives and for the rate
  comparator, but can never carry a positive — the pulse contributes
  nothing to naming the rhythm, so the label carries all of it.
- **Record the exertional context** for every regular tachycardia
  (exercise, fever, anxiety, caffeine, dehydration). Arm B of the
  campaign is what measures specificity, and specificity is what
  decides whether this flag ships at all.
- **Never tell a participant anything about flutter, SVT, a conduction
  ratio, or a rhythm name.** The surfaces cannot render it while §F is
  red and the token tests enforce that; after §F is green the only
  permitted wording is the sanctioned sentence, which still names no
  rhythm. A negative scan does not exclude flutter — 4:1 conduction at
  ~75 bpm is a documented permanent miss.

## 13. The rhythm-regularity substrate track (v0.7, §R)

Operator loop for the gated "regular or irregular, and how sure"
question (docs/regularity_track.md is the authority; the MDI table in
it is a published product parameter, not an internal number):

```bash
python3 cli.py register-dataset day_regularity/
python3 cli.py regularity-floor day_regularity/ --domain facial_rppg   # R1: gain, MDI, beat errors
python3 cli.py evaluate-regularity day_regularity/ --domain facial_rppg
python3 cli.py gate-status --track regularity        # the scoreboard
```

At the bench:

- **Every day's capture needs metronomic references at each frame
  rate.** The noise floor is calibrated from them — the sub-frame
  interpolation gain and the MDI table are *measured* on every floor
  run, never assumed. Without at least 10 metronomic references per
  fps the floor run reds R1. A paced or beta-blocked participant at
  rest, or a bench pulse simulator, is a metronomic reference; the
  ECG RMSSD decides (< 5 ms), not the label.
- **Capture at 60 fps whenever the device allows it.** Frame
  quantization alone puts 13.6 ms RMS on every interval at 30 fps and
  6.8 ms at 60; the calibrated RMSSD floors are 15.3 ms and 7.7 ms.
  Everything the index can see sits above that floor.
- **The torso must be in frame.** The head abstains without a decodable
  respiration channel, because an irregular pulse that is *coupled to
  breathing* is the benign case the whole track exists to separate.
  A scan of a face alone yields an index for the research reader and
  no class.
- **Span the ages.** R2 requires specificity ≥ 0.90 in *each* band
  (<35, 35–59, ≥60) with at least 20 sessions per band. The under-35
  band is where respiratory sinus arrhythmia is deepest, and a collapse
  there is the pre-registered failure that forces an age-aware
  threshold or a narrowed claim — it cannot be averaged away.
- **Abstention is on the record and bounded.** Every R2 and R3
  denominator is reported total / judged / no-read, and the no-read
  rate is gated (≤ 0.30 per age band, ≤ 0.20 on the held-out set). A
  head that declines the sessions it would get wrong cannot buy
  specificity with silence, so a day's capture that produces many
  no-reads (no torso in frame, short clean runs) is a day that moves
  no gate.
- **Simultaneous ECG on every scan, and never a hand-typed label.** The
  reference label is *computed* from the ECG R-R series by the same
  code that reads the camera; there is no adjudication step to be
  generous in. Annotate the rhythm (sinus / RSA / ectopy / AF) for the
  benign-separation and fairness tables, but the class comes from the
  definition in `configs/gates.yaml`.
- **Never tell a participant their pulse was irregular, regular, or
  anything about a rhythm.** The head renders nothing while §R is red;
  after it is green the only permitted wording is the sanctioned
  sentence, which names no rhythm, offers the benign explanation first
  and routes to an ECG. This head never escalates — an AFib-suggestive
  sentence is `head_afib`'s job.

## 14. The Research & Investigation Report (v0.8)

For investigators only — never the consumer report
(docs/investigation_report.md is the authority; spec B.24 quotes the
owner instruction and the flagged reading):

```bash
python3 cli.py research-report scan.avi --age 58
python3 cli.py research-report scan.avi --session s.json --videos rest.avi act.avi rec.avi
python3 cli.py research-report scan.avi --provocation p.provocation.json --pi pi.json
```

- **Never hand it to a participant.** Every section carries its
  track's live gate status; while promotion reads BLOCKED the section
  is a research artifact, not a result. The document says so on every
  page and in every section, and the consumer report is unchanged.
- **Feed the tracks what they need.** Torso in frame for the two rhythm
  tracks; a three-phase session manifest for fitness; timed provocation
  marks for vascular tone. A track without its inputs is listed as not
  applicable with what it needs, not silently dropped.
- **It is an operator action.** `process`, `session` and the live demo
  never produce the document; only this verb does, and it writes under
  `research/runs/investigation/` (gitignored).
- **On the results screen**, the same five tracks appear in the
  RESEARCH / DEBUG METRICS panel by default, each beside its live gate
  status. Red gates do not withhold them. To turn the block off for a
  participant-facing session:

  ```bash
  AVATARX_RESEARCH_TRACKS=0 python3 cli.py demo
  ```

  Screen-only (never printed), in its own payload key.
  The two rhythm tracks compute there; arterial stiffness and vascular
  tone need the research surface and the panel says so with the command
  that produces them. Turning it on costs a second decode of the
  recording (the torso respiration channel the rhythm tracks need) plus
  two head runs, so the results screen appears a few seconds later —
  that is the flag doing work, not a stall.
