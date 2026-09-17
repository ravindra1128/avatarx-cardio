# AFib delivery fixes — first implementation batch

Date: 2026-09-17. Service branch: `codex/afib-delivery`, based on `9b0d05d`.

This implements the first delivery fixes from the Phase 1 audit. It does not change the extractor, physiological features, classifier, quality thresholds, confidence rules, or clinical result labels. The remaining AFib decision defects described in `AFIB_PHASE1_AUDIT.md` are still open.

## Behavior changed

* The staging browser processor creates a unique scan ID before capture, uses it as the upload/result key, and carries it in the results URL. Results and status are checked against that identity. A late previous scan cannot overwrite the current result; previous classifications are never analysis inputs.
* AFib remains registered in the staging lifecycle even when disabled or unconfigured. Such a scan returns an explicit operational failure instead of silently omitting AFib. Pending and failed delivery states are distinct from physiological classifications.
* Duplicate finish events schedule one delivery. Finish waits for initialization. Staging opts into bounded lifecycle and recorder-stop waits; a recorder that never finishes supplies no invented/partial clip. The SDK-owned track is not stopped.
* Browser upload, start, collection, and response-body reads have deadlines. The single-shot deadline now includes reading the response body. A lost start acknowledgement triggers collection of that same scan before giving up.
* The tab stores pending job identity and can collect an accepted job after a results-page reload or a connection interruption. Recoverable failures keep collecting within the recovery window. Expiry is explicit; no previous result is substituted.
* `/api/start` checks existing accepted/completed work before worker availability or assembly. Concurrent starts reserve one job. A retry returns the existing job or a pointer to the completed result without consuming the upload again or appending a duplicate sheet row.
* Cached service responses add `scan_id`, `upload_id`, and `analysis_state`. Submission failures and invalid worker returns are collectable error documents. The cleanup sweep leaves active jobs alone. Existing response fields remain intact.

The staging results cards read the selected scan and continue observing recoverable delivery failures. Shared lifecycle helpers gained tested ordering and optional timeout support; the production AFib processor and production result cards were not edited.

Concurrent webapp commits `afde509` and `1b77235` added a three-result label renderer. Those changes were preserved when integrating this batch; this delivery work does not validate or alter that renderer's clinical mapping.

## Verification

The new server tests first reproduced six failures in the old implementation. Tests cover lost acknowledgements, simultaneous starts, completed retries, worker submission/return failures, active-upload cleanup, and invalid identifiers. Browser tests cover stale results/status, identity mismatch, unavailable processing, duplicate finish, hung headers/body/recorder, result recovery, recovery expiry, results-URL identity and storage quota failure.

Final checks: **54 service tests passed; 130 browser tests passed across 13 suites; production build passed.** Null/malformed completed responses are explicitly failed at delivery boundaries rather than reaching a renderer that could label missing data Inconclusive. Source whitespace checks passed in both repositories.

The webapp production build passed in an isolated copy using the installed dependencies. The selected browser suites use a Node Jest configuration because the installed environment lacks `jest-environment-jsdom`; this is not a real-phone capture test.

Fresh replay of 11 paired recordings, twice each:

* 22/22 structured returns; 11/11 deterministic repeat decisions.
* Every candidate video decision and final decision equals the audit baseline, including features, reasons, class, stars and source selection.
* Every source video/sidecar/timestamp hash remains unchanged.
* Final outcomes remain 4 ACCEPT/SINUS, 4 REPEAT_SCAN, 3 NO_RESULT. No AFib accuracy or human test-retest claim follows from these unlabelled recordings.

The baseline is `docs/audits/afib-phase1-2026-09-16/replay.json`. Local detailed candidate evidence is `data/eval_cache/delivery_candidate.json` and `delivery_comparison.json`.

The optimizer guard was snapshotted before edits. Its protected symbols and frozen configuration hashes are unchanged. Its scope check reports the intended HTTP handler, audit script and tests outside the signal-chain allowlist. This is the explicitly requested delivery work, not an accepted signal-optimization iteration; no guard or threshold was loosened to obtain acceptance. Delivery behavior is evaluated by fault injection, with physiological replay as a separate regression check.

## Limits and remaining work

Server jobs/results still live in one process. Browser recovery works only while that job/result remains available, within the existing cache TTL/capacity. It does not survive server restart, redeploy, cache eviction or a different worker process. Durable job/input storage and restart recovery are still required for the full completion guarantee. Browser storage failure has a volatile fallback for the current page session; it cannot promise reload recovery when persistence is unavailable.

Uploads interrupted before the job starts are bounded failures, not yet a resumable browser-persisted video queue. ShenAI sidecar acknowledgement and transport-equivalent source selection remain open; this batch intentionally preserves the existing source-selection policy. Missing numeric SDK fields, source-specific timing/confidence, uncertainty-to-SINUS behavior, stale AFib head representations, and the three-result clinical policy belong to the next decision/evidence changes and their own validation.

No phone test, webapp deployment, backend deployment, or clinical validation is claimed by this document. The requested service target remains `https://avatarx-cardio-codex-production.up.railway.app`; the staging webapp must be configured to that target when testing this service build. The original staging service fallback and local environment settings have not been repointed implicitly.

## Phone checks after the paired frontend/backend release

1. Complete a normal scan. Verify one scan ID throughout, a collected response, and no result from a previous scan.
2. Reload the results page after the job is accepted. Confirm that it retrieves the same scan without another recording or analysis.
3. Briefly disconnect after upload/start, then reconnect within the recovery window. Confirm collection of that same result.
4. Start a second scan before the first result arrives. Confirm each results URL displays its own scan.
5. Record any failed completion with its scan ID and delivery state. Count it as a completion defect, never physiological Inconclusive.

Independent repeat scans and ECG-labelled discrimination testing follow the decision fixes. A green delivery test is not evidence that the current `SINUS`/AFib rules are reliable.
