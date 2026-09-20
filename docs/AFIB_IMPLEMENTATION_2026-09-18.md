# AFib implementation — 18 September 2026

This iteration addresses result delivery and capture observability in the dedicated
Codex service/route. It does not establish better AFib sensitivity or repeat-scan
classification stability. ShenAI capture, classifier thresholds and video-retention
settings are unchanged.

## Implemented independently of classifier changes

- The finished SDK measurement is copied exactly once before asynchronous
  finalization. The sidecar and reference metrics use that same snapshot. Missing
  SDK vitals cannot strand a completed scan on the scanner screen: the AFib
  results page still opens using its unique scan ID.
- The final source-selected AFib result is expressed in `afib_result` and mirrored
  into the existing AFib head. Any prior video head stays explicitly named under
  debug. Other heads, cards, outcome fields, transport errors and class decisions
  remain unchanged. Operational errors do not receive a physiological class.
- Completed-result recovery uses a 15-minute SQLite store behind the 32-entry
  memory cache. Memory eviction no longer discards unexpired results. The store
  survives process restart on the same filesystem, uses restrictive file
  permissions, removes expired rows on access/health checks, and does not store
  video or raw SDK signal inputs. Storage failure is reported through health and
  per-result recovery metadata; the immediate result can still be returned.
- Continuous acquisition diagnostics report SDK version, capture settings, SDK
  quality summaries, hidden time, presentation-frame timing and callback timing.
  These are summaries over the whole SDK measurement, not proven camera-drop
  counts or a clock alignment with the retained video tail. Histogram percentiles
  have 1-ms resolution and saturate at 3 seconds; the maximum is retained exactly.
  Missed callbacks are separated from adjacent presented-frame timing.
- Diagnostic fields are bounded/allowlisted at the backend and appended to the
  Codex sheet, asynchronously as before. The frontend does not retain frames or
  physiological sample arrays for these diagnostics, and never stops the SDK
  track. It adds a small video presentation probe; device overhead needs phone
  verification.
- The results UI labels the SDK metric as signal quality, removing its unsupported
  interpretation as a percentage probability of correct measurement.

## Verification

- Backend: 130 focused tests passed, covering response mapping, result recovery,
  input handoff, idempotent start, evidence summaries, sheet schema, retention and
  biomarker wiring.
- Frontend: 129 tests passed across nine suites, including scan completion,
  diagnostic timing, sidecar ordering, delivery, recording deadlines and route
  isolation. Vite production build passed.
- No clinical accuracy improvement is claimed. Phone timing/overhead and useful
  capture duration must be measured on the new frontend.

## Tested corrections awaiting an optimizer exception

The optimizer skill explicitly requires **“Do not override a reject.”**
Both candidates were tested, replayed, logged and restored out of active code.
They must not silently enter a deployment while awaiting an exception.

| Candidate | Focused tests | Replay observations | Optimizer decision |
|---|---:|---|---|
| One-to-one beat matching | 9 passed | Four records, twice each; decisions/cards unchanged, deterministic. Two timing metrics improve. Signal 0.6603 → 0.6670. | Reject: less than +0.02 and fewer than three improved records. |
| SDK representation/availability validation | 31 passed including existing route tests | Four records, twice each; decisions, reasons, evidence and cards unchanged. Video replay has no SDK sidecars. | Reject: no measured proxy gain. |

The first eliminates detection reuse, bounds match fraction, and removes region
order dependence. The second blocks invalid SDK quality/timestamps/durations
before route selection and keeps lnRMSSD units fixed. Neither relaxes a numeric
quality gate. Unit tests demonstrate correctness properties, not screening
accuracy. The separate, earlier uncertainty correction remains pending too.

Review artifacts: `data/eval_cache/contract_fixes_20260918/` contains baselines,
candidate replays, guards, verdicts, per-record comparisons, tests and the patches
`matching_fix.patch` and `sdk_input_fix.patch`. The scalar receipt is
`docs/audits/contract-fixes-2026-09-18.json`.

## Deployment and operational limits

Only backend `ravindra1128/avatarx-cardio-codex` and frontend Codex-owned files are
part of this change. The frontend uses its existing shared staging release; the
other cardio routes and their code/configuration are not edited.

`AFIB_RESULT_DB` can point to a file on a Railway persistent volume, for example
`/data/afib-results.sqlite3` **if `/data` is actually the mounted volume**. Without
that infrastructure, the default temporary filesystem store cannot survive a
container replacement or redeploy. A path variable alone does not create a volume.
This store is for completed results; interrupted inference is not automatically
resumed and a multi-replica shared job queue is not implemented. Video retention
must stay off.

After release verification, compare three scans approximately five minutes apart
under similar conditions. Inspect both the three-state AFib outcomes and the new
capture diagnostics. Consistent abstention alone is not evidence of success.
