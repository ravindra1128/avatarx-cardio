# AFib signal handoff — 2026-09-17

The Codex route sent ShenAI evidence through a separate, best-effort POST after
job acceptance. On the single-request fallback it sent that POST only after the
result, and the service's single-request path did not invoke the train route.
A successful signal POST could also mean `held:false`. Thus upload transport and
timing could change which evidence was available for the same completed scan.

The new client sends the snapshot inside the request that starts the job. Sliced
uploads attach a bounded JSON body to `/api/start`; the single-request fallback
uses the existing length-prefixed video envelope, now with query metadata merged
in. Both transports run the same video analysis and optional ShenAI train route.
The server validates the snapshot's scan id before consuming the slices. Start
retries preserve the accepted job and never replace its inputs or queue it twice.
Incomplete or malformed input returns an explicit error without consuming parts.

The snapshot is held by the job in memory. This does not enable recording
retention. The bounded `debug.shenai_input` summary reports receipt, transport,
sample/beat counts, missing samples, sample-rate provenance and the SDK's quality,
bad-signal seconds, heart rate and lnRMSSD even when video already accepted.
It contains no waveform or beat-time arrays. Missing quality is null, not zero.
The sheet appends these summary columns plus Scan ID, retaining the old column
names and order. Its retained-sidecar columns keep their previous meaning.
Summary failures cannot stop route evaluation or change its decision.

The client preserves missing SDK values rather than converting null/empty/bool
values to zero. Missing or oversized optional snapshots are explicitly reported;
video analysis still runs. Explicitly attached/missing inputs do not wait for a
later sidecar. Legacy clients retain the bounded sidecar wait.

The first result check no longer waits out an upload-size ETA: default polling
starts within four seconds, then backs off. The former ~48-second ETA could hide
a finished ~20-second job. This removes avoidable result waiting; it does not
reduce the phone's measured 63–187-second video upload time.

## Validation

- Backend: 106 tests passed across signal handoff, routing, retention, identity,
  retry/recovery, input handling, response wiring and sheet behavior.
- Frontend: 119 tests passed across the Codex suites and existing staging
  isolation test; the production Vite build passed.
- All three recovered real phone videos were replayed on copies with the same
  consumer profile, 70-second requested window, 640x480 scale, and deployed
  collapsed-frame override. Before/after beat sequences, runs, features,
  evidence, video decisions and final decisions matched exactly. Original-file
  hashes were unchanged. The first video has no recovered SDK sidecar; the other
  two do. `audits/afib-phone-2026-09-17/handoff-replay.json` records the comparison.
- Replay outcomes are one AFIB_SUGGESTIVE, one SINUS after the existing train
  route, and one abstention. These are software outputs, not ECG-confirmed truth.
  Local replay values differ from live scans, so this is a paired local regression
  check, not a reproduction of every live numeric result.

This is an integration/diagnostic change, outside the optimizer's signal-chain
allowlist. It is not presented as an optimizer-accepted signal improvement. No
inference, feature, preprocessing, capture floor or classifier threshold changed.
The three replay comparisons cannot establish clinical accuracy or test-retest
stability. The identified beat-count/spectral disagreement, timing matcher,
noise-aware classification, SDK quality validation and source-policy issues remain
open; changing them requires a separate signal iteration and appropriate reference
data. The current corpus has no ECG-labelled AF examples.

## Release and phone check

Deploy the Codex service first, then the webapp's Codex-only changes. No new env
variable is required. Keep `AFIB_KEEP_UPLOADS=0`; the new diagnostics do not depend
on it. Keep using the dedicated Codex backend and `/beta/cardio-codex` route.

After the frontend release, refresh before starting a new scan. Check that the
result and sheet have the same Scan ID, `Signals Transport=job_request`, and
`Signals Received=TRUE` when the SDK produced a snapshot. Check the explicit
state when it did not. The selected rhythm source and route reason remain visible.
Measure several comparable scans within 10–20 minutes and inspect evidence and
outcomes independently; neither previous results nor cached classifications
influence a new scan. This test assesses the input handoff and surfaces remaining
rhythm failures; it does not by itself validate the classifier.
