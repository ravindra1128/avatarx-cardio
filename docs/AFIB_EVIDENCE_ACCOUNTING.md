# Codex evidence accounting and SDK assessment

This is an additive diagnostics release for `avatarx-cardio-codex`. It does not
change AFib classification, source selection, cleaning, thresholds, cards, camera
settings or scan duration. It is not a signal-optimizer candidate and carries no
claim of improved AFib accuracy or conclusive yield. The existing optimizer's
card/signal score does not measure completeness of these reports.

## What every completed job reports

`debug.video_duration` describes the video path before any SDK route replaces the
final rhythm result. `debug.shenai_assessment` assesses the attached SDK train
before the routing decision, including when video accepts, when a video gate
blocks the SDK route, or when the route is disabled. Missing input, empty trains,
unavailable clocks and assessment failures have explicit states. An assessment
failure cannot skip the existing decision. Neither report contains raw waveform
samples or heartbeat arrays; both use in-memory job inputs. Retention stays off.

Both upload transports already converge on the same job implementation. Existing
response fields and sheet columns retain their meanings. New columns are appended
by name, and the sheet still runs after result delivery. No environment variables
or frontend changes are required.

## Time accounting

The recorder's elapsed time can include the rolling-buffer hole. The retained
video clock is measured after trimming and before frame rejection during face
tracking. Its first and last timestamps, span, and frame-step statistics now travel
as scalar fields in `clock`. No additional video decode is needed.

The ROI clock omits frames without usable facial regions. It cannot serve as the
full video's denominator. The audit reports both clocks and partitions time as:

```
retained frame span = unobserved head/tail time
                    + accepted processing-segment spans
                    + rejected short-fragment spans
                    + excluded ROI timestamp steps
```

All spans use endpoint differences, excluding a final frame period. Excluded
steps are **not** labelled as physically missing recording time: frame-rate
variation and rejected face observations can both trigger the existing split
rule. Full-frame and ROI-step p99 values and excluded ROI-frame counts help
distinguish those causes. The audit reports isolated frames separately.

`clean_fraction_retained` divides clean video-interval seconds by the full retained
frame span. `clean_fraction_processed` divides them by accepted processing spans.
The existing `Coverage` column and classifier denominator are unchanged. The new
`Video Whole Coverage` exposes the difference. If the full clock is unavailable,
whole-video coverage stays unknown instead of substituting the shorter ROI clock.

## What SDK assessment means

The assessment checks original-order start/end values, invalid entries, duplicate
or reversed starts, gap and overlap boundaries, and agreement between declared
durations and timestamp differences. Nulls, booleans, strings and nonfinite values
are not silently converted to physiological numbers. Invalid entries split the
descriptive statistics; successive differences never cross a gap.

Two descriptive views remain separate:

- `reported_intervals`: the SDK's timestamp-defined durations, including its final
  interval; no physiological cleaning.
- `current_route_train`: the existing start-to-start representation, using the
  unchanged route cleaner and train checks. This representation omits the final
  reported interval of each contiguous segment.

The reports include counts, span, longest clean run, RMSSD, SDNN, median absolute
successive difference, pNN50, and comparisons with SDK-reported HR/HRV. Missing
quality is explicit. Passing internal checks does not assert AFib eligibility or
clinical validity. A mismatch can reflect different windows or processing; it
does not by itself prove smoothing. The assessment does not sort away input
defects or feed any new metrics into the decision.

SDK/video alignment is currently **unknown**: there is no established mapping
between their clocks. A common scan ID, similar heart rate or shared completion
time cannot establish per-beat agreement. Verified overlap remains null, and
cross-source comparisons are explicitly labelled as unmatched windows. The PPG
sample-rate derivation and unknown origin remain visible.

ShenAI exposes heartbeat timing and PPG samples, but its quality figures and HRV
outputs are not independent AFib reference labels. Its documentation also notes
that HRV depends on measurement duration. See [SDK results](https://developer.shen.ai/video-measurement/results)
and [SDK measurement quality](https://developer.shen.ai/video-measurement/measurement).

## Validation and next decision

The focused suite passed **116 tests** covering duration partitioning, omitted face frames, invalid
clocks and beats, continuity, both transports, retention off, all route outcomes,
disabled routing, failure isolation, and legacy sheet responses.

`scripts/replay_evidence_audit.py` replays temporary copies of three previously
saved phone recordings twice each against the saved `1b031f1` replay. It checks
unchanged decisions, routes, evidence, clean runs and detected beat sequences,
repeatable audits, and unchanged hashes of original videos/sidecars. It emits
summary diagnostics only. These are older recordings; the user's latest three
scans were not retained and cannot be replayed.

All six replays passed the equality checks; the two audit reports repeated
identically for each recording. The original inputs' hashes were unchanged.
Signal, inference, capture, preprocessing, configuration and prep code have no
diff against `1b031f1`. [Verification receipt](audits/evidence-accounting-2026-09-17.json).

On the two available SDK sidecars, the unchanged route cleaner retains 60 and 64
intervals. Their video paths discard approximately 26.94 and 14.03 seconds as
short fragments. Full-train RMSSD agrees with SDK lnRMSSD, while SDNN differs by
about 11.3 and 15.4 ms; the SDK's window/processing definition still needs to be
resolved. One train also has a 3.47-second discontinuity, which is not bridged.
None of these observations establishes which rhythm classification is correct.

The next classifier decision requires source-specific validation: establish SDK
timing/processing semantics and test rhythm evidence against labelled rhythm
examples, including AF, regular rhythm, ectopy and detection errors. Contact PPG
can test interval logic; simultaneous ECG and facial recordings are needed to
validate the full screening system across users and devices. Current video gates
must not simply be removed to make the SDK train return more conclusive results.
