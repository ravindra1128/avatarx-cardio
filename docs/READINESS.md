# Readiness, confidence stars and the minimum signal for a valid AF analysis

**v0.1.6 — easy start, strict measurement.** In the default `advisory` mode
the API and UI start an attempt once the three **blocking** checks pass — face
found, framing intact, hard frame-rate floor (≥ 15 fps). The countdown does
not wait for downstream pulse evidence that needs a longer window to mature.
During the scan, capture faults gate recorded good time: missing/misframed face,
unusable exposure, motion, broken delivery/timestamps, low frame rate, or
unstable tracking can pause and recover. Provisional SNR, coherence and beat
checks remain visible guidance, but do not block recording the complete window
that the final pipeline needs. An unsupported finished signal still returns
`REPEAT_SCAN` or `NO_RESULT`, never a forced value. The **1–5 star confidence rating**
(`inference/confidence_stars.py`) remains the evidence grade.

**The rule that must not be lost:** making the start easy does not
lower the bar to call AF. The v0.1.2 evidence gates are untouched; stars
are their user-visible expression, enforced twice — any failing AF
evidence gate caps the stars at 2, and the decision downgrades any AF
call carrying < 3 stars to the inconclusive path. **A scan below 3 stars
can never return `AFIB_SUGGESTIVE`.**

| Stars | Meaning |
|---|---|
| 5 | every advisory check passes with margin (score ≥ 0.85) |
| 4 | all advisory checks pass (score ≥ 0.68) |
| 3 | minimum at which a rhythm call is supportable — all AF evidence gates met |
| 2 | pulse detected, rhythm call not supportable (an AF evidence gate failed) |
| 1 | conditions too poor (score < 0.30), or a blocking check failed |

Score = weighted mean of per-check margins (value vs threshold from the
same checklist below; weights `decision.confidence.weights`, cuts
`star_cuts`). The live meter (`final=False`) skips only the
length-dependent AF interval-count gate — an 8 s window can never hold
20 intervals; the finished recording applies every gate. The star suffix
on results ("Confidence: n of 5") and the low-star sentence live in
`ScanResult.user_facing_text()` — still the only sanctioned text path.

## The pre-v0.1.5 blocking gate (mode: blocking)

**Principle (the parity rule, v0.1.4).** The scan timer must not start until
the live window already satisfies **the evidence a RESULT needs** — the
any-class gates the decision itself applies — computed by the *same
function* (`inference/evidence.py::window_evidence`) reading the *same
config keys* (`decision.evidence`), so gate and decision can never drift
apart. The gate must **not** demand more than that: not the stricter
AF-call bars (the decision applies those only to the AF call itself, on
the full recording, where that evidence actually accrues), and not any
quantity the decision never gates at all. Once running, the timer counts
**good seconds only**: on quality drops the recording pauses (an honest
hole in the capture clock; the gap-aware pipeline treats it as a segment
boundary) and resumes when the signal returns; if pauses exceed a budget
the session restarts at readiness.

Flow: **Camera → face/ROI detection → live signal preview → minimum-evidence
check (held 3 s, blip-debounced) → 3-2-1 → scan (good-time timer) →
processing → result.**

## v0.1.4 recalibration — what real use falsified (measured)

A real user could not start a scan in 3+ attempts under decent conditions
(sessions of 2026-08-20/21; the v0.1.3 gate left no telemetry — itself
fixed, see below). Reconstruction on degraded variants of the reference
video, rolling 8 s windows at 0.5 s steps:

| variant | v0.1.3 READY evals | longest streak | full-scan verdict |
|---|---|---|---|
| clean | 100 % | ≥ 32 s | ACCEPT / SINUS |
| low light (0.6× gain, σ=2.5 noise, ±6 % AE wander) | 100 % | ≥ 32 s | ACCEPT / SINUS |
| side shadow (0.45–1.0 gain ramp across the face) | 100 % | ≥ 32 s | ACCEPT / SINUS |
| **4-frame drop burst every 4 s** | **0 %** | **0.0 s** | **ACCEPT / SINUS** |

The drop pattern is routine browser behaviour under transient load — and
the v0.1.3 `timestamps` check vetoed any window containing a hole, so one
small burst poisoned the entire 8 s window, every window. The gate said
NO forever to a stream the pipeline itself reads perfectly (gap-aware
segments). Three checks (hole veto, period jitter, multi-ROI quota) had
**no post-scan counterpart at all**, and three thresholds were the
AF-call bars rather than the any-result bars. Two engine checks were
mis-specified: the lux proxy averaged the **whole frame** (a lit face in
a dark room reads "dark": fails iff frame luma < 32/255), and the
exposure check compared first-half vs second-half means over 8 s (a slow
1.5 %/s AE ramp — removed entirely by the 0.7 Hz high-pass — failed the
whole window). Each transient reset the 3 s hold, which required 7
consecutive all-pass evaluations of 13 conjunctive checks.

## The checklist and where each threshold comes from

| Check | READY when | Basis |
|---|---|---|
| Face / framing | face found; box width ≥ 20 % of frame; centred within 22 %; every ROI keeps ≥ 50 % of its nominal pixels | The 20 % floor is the per-ROI pixel budget (~1200 px averaged at 640 px). v0.1.4.2: the former 0.70 face-width **cap** is gone — it had no post-scan counterpart and blocked a real close-to-laptop user whose tracking was 0.98 with beats present. The honest criterion is ROI **integrity**: a region that lost half its pixels to the frame edge is no longer the intended region (`preprocessing/roi.py::roi_integrity`) |
| Lighting | **face-region** luma ≥ 60/255, face-region lux proxy ≥ 100 lux, fewer than 3 clipped ROIs, and usable-ROI luma imbalance ≤ 0.85 | The photons that matter fall on the **skin**. The added two-sided test catches saturation and strong uneven illumination that a mean-luma floor missed |
| Exposure stability | no ≥ 12 % step between **adjacent 0.5 s luma bins in the last 2 s** | An AE **step** is a broadband transient that corrupts the waveform now. Slow drift is a < 0.1 Hz trend the 0.7 Hz high-pass removes (v0.1.3 failed 1.5 %/s ramps for 8 s) |
| Motion | median centre move < 3.5 % of face width per frame | Below this, ROI means are stable; above ~9 % beats are lost (engine abort threshold). Tracking stability (raw observations) ≥ 0.6 |
| Frame rate | median frame period ⇒ **≥ 24 fps** | Consumer capture floor: below it the **recording itself** would be invalid (research 30 fps; quantisation SD = (1000/fps)/√12) |
| Frame delivery | decoded content is advancing; batch timestamps are finite and strictly increasing | Exact duplicated frames are skipped; persistent frozen delivery pauses/restarts. Timestamps are never silently nudged into order |
| Frames (timestamps) | analysable **coverage** ≥ **0.60** and collapsed adjacent intervals ≤ 2 % | Gap-aware segments begin at >1.5 nominal periods, so one missing frame is no longer analysed as uniform sampling. Isolated holes recover; dense holes or near-zero adjacent frame times fail visibly |
| Pulse SNR | ≥ 2 ROIs with in-band SNR component ≥ 0.5 | The T4 SNR logistic is centred at **+3 dB** (in-band energy > 2× out-of-band); ≥ 2 independent regions is the minimum for cross-region verification |
| Cross-region coherence | ≥ **0.20** (`coherence_floor`) **or two-region verified**: best ROI-pair median \|Δt\| ≤ 30 ms with ≥ 75 % matched (the AF-grade timing budget, same config keys) | What ANY result needs. v0.1.4.2: with exactly two strong ROIs (ordinary home lighting — one cheek shadowed, weak nose) the ≥ 3-ROI coherence component reads **0.00 by construction**, indistinguishable from garbage; a real user was blocked at coherence 0.00 with beat timing 27 ms. Two regions placing the same beats within the AF timing budget ARE independent verification — the two real FP scans failed exactly that budget (36–49 ms, matched 0.63–0.75) and stay refused (regression-tested on their recorded values). The pulse-deficit rule still demands ≥ 3-region coherence 0.50: deficit is inferred from ABSENT beats, which two regions cannot cross-verify. Measured: references 0.98–1.00; FP scans 0.06/0.12 |
| Beat timing precision | best ROI-pair median \|Δt\| ≤ **40 ms** (= `max_timing_precision_ms_any`) | The any-class timing gate. The AF-call bar (30 ms, matched ≥ 0.75) is judged by the decision on the whole scan. Measured: references 1–5 ms; real FP scans 36–49 ms — still failing 40 ms most windows |
| Preliminary beats | ≥ 4 fused beats in the 8-s window | Presence of a fusable pulse; 4 beats = the shortest run the feature engine accepts. **No rhythm/burden test** (anti-periodicity) and **no multi-ROI quota** (cross-ROI membership is what coherence measures; the decision has no such quota) |
| Composite SQI | ≥ **0.30** (`sqi_floor`, the post-scan floor) | White noise / no-pulse measures ~0.20–0.22 on the T4 index |
| Hold | all of the above for **3 s**; a check must fail **2 consecutive** 0.5 s evaluations to break the hold | Two POS windows of stable evidence; single-eval estimator flicker must not discard accumulated readiness (persistent problems still do) |

Window: **8 s** (≥ 2 POS windows and ~7 beats at 55 bpm — enough for the
timing-precision proxy and coherence to be estimated at all).

**What the parity rule does NOT promise.** Starting at any-result grade
means a scan can still end in REPEAT_SCAN when the full recording's
evidence stays marginal — the gate minimises the doomed-from-the-start
class, it cannot guarantee AF-grade evidence will accrue. The synthetic
face cannot probe the real mid-grade regime (its pulse is far stronger
than skin), so the signal thresholds rest on parity with the decision,
and the per-session telemetry below is the instrument that will validate
them on real sessions.

## Telemetry — every session leaves evidence

`<work_dir>/<session>/readiness_log.jsonl`: one line per 0.5 s evaluation
(state, disposition, failing checks with values, evidence summary, per-ROI
photometry, duplicate/timestamp counters, lux proxy, motion, eval wall-time)
plus lifecycle events (start, pause budget
restart with reason, abort, final outcome). The 2026-08-20/21 failures
left empty directories; a session that never becomes ready is now
diagnosable after the fact.

## During the scan

The same checklist runs every 0.5 s. Engine-observed conditions (face,
framing, lighting, exposure, motion, delivery, tracking, frame rate) are already
frame-level-smoothed and pause the recording **immediately** — a face
that leaves must not be recorded as good time; the debounce applies to
the evidence estimators. **Timer = good seconds.** When a check fails
the recording pauses
(`paused: true`, ring shows ❚❚, hint says what to fix); it resumes after
1 s of good signal. Pauses > 20 s in total or > 4 pauses ⇒ restart at
readiness with the reason shown. Blocking-mode sustained darkness (> 2 s) or
violent motion (> 4 s) abort outright; advisory mode pauses/restarts. Every
pause is an honest hole in the
timestamp sidecar and is reported on the results screen.

## Anti-periodicity guarantee (readiness must not gate AF users out)

`tests/test_readiness_gate.py::test_readiness_is_rhythm_neutral…` asserts
that on noise-matched synthetic AF vs sinus, the fraction of READY
evaluations is equal within 5 points. No readiness check measures
regularity; the split/harmonic burden checks live only in the post-scan
decision, judged on the whole recording.

## What this does not do

It cannot make a weak signal strong. If fewer than two facial regions
carry a verifiable pulse the gate will keep saying so and never start —
the realistic levers are even frontal light on the forehead **and both
cheeks**, stillness, and no hair across the forehead.

Reference note: a consumer wearable and this prototype are not
interchangeable diagnostic references; ECG-confirmed rhythm is the only
ground truth for AF performance.

## v0.4 — readiness in the three-phase recovery session

The rest and recovery scans each run the SAME readiness gate and star
grade described above — nothing about the session weakens a per-scan
check, and each phase writes its own `readiness_log.jsonl`
(`scripts/why_not_ready.py <vsession dir>` summarises all phases plus
the session verdict). The guided activity has NO readiness gate because
it has no physiology: the camera only verifies workload (reps/cadence),
and in-motion pulse estimation is structurally excluded. The transition
clock (activity end -> recovery recording start) is part of the
measurement: > 10 s hard-fails the fitness outputs because the earliest
— steepest — part of the recovery curve is gone (the harness's
transition-sensitivity sweep quantifies the cost). Session stars = the
minimum of the phase stars. Fitness inference on top of these
measurements is §V-gated (docs/vo2_crf_track.md);
`python3 cli.py gate-status --track vo2` is the live authority.
