# Staging + Codex integration

Owner destination: existing `ravindra1128/avatarx-cardio` repository, `staging`
branch and staging service. Sources: staging `859e2e4` (latest VO2/fitness option A)
and Codex `de7383e`. This is a feature integration, not a newly tuned estimator.

## Resolution decisions

- Preserve staging VO2 equations, optional profile handling, request reference-rate
  selection, no-profile proxy and train-to-fitness reconciliation. Codex's shared
  rate resolver supplies the video-side rate; staging's reference policy remains.
- Preserve standalone trace ingestion, live ROI trace comparison and all existing
  trace/fitness sheet columns. Append Codex diagnostics without duplicate headers.
- Drain the start body once. Both the participant and SDK snapshot can accompany
  the same job. Signal identity/state validation and idempotency stay binding;
  malformed legacy optional profile bodies remain an absent profile.
- Both video transports use the assembled job path, including staging trace and
  fitness reconciliation and Codex input accounting/recovery.
- Preserve staging's AFib probability decision band and basis. Codex's finalizer
  synchronizes nested AFib heads to that final result instead of replacing it.
- Shared train assessment retains staging's fitness evidence even when publication
  is blocked. Codex diagnostics run on isolated inputs and cannot publish a result.
- Existing config files, capture/schema floors and inference/pipeline.py are
  unchanged. No new thresholds, calibration or score smoothing.
- Exclude the Codex repository's main-branch deployment workflow. It targets a
  different service/account setup; staging keeps its existing deployment mechanism.
  No service variables, credentials or sheet destinations are changed.

## Frontend counterpart

Webapp commit `13977bf` combines both implementations in `src/Pages/Cardio/` and
`src/lib/scan/cardio/`, routed at `/beta/cardio`. It selects the staging service,
keeps the two source routes independent, and uses the new `afib_result_cardio`
namespace so historical results cannot appear as a new combined scan.

## Validation

- Frontend combined suite: 149 passed; original Codex suite: 139 passed; build passed.
- Backend focused suite: 147 passed, including combined profile/snapshot transport,
  VO2 preservation, probability-band finalization and unique additive sheet columns.
- Broader backend regression: 173 passed initially. Four diagnostic tests expected
  Codex-only route metadata; expectations now include staging's existing fitness
  reconciliation (26 evidence tests pass on rerun). One remaining low-signal scan
  test also fails on untouched staging `859e2e4`: the assertion requires every
  biomarker value to be absent despite existing provisional values. The scan still
  abstains with REPEAT_SCAN/NO_RESULT. No gate or runtime behavior was changed to
  silence this inherited failure.
- Eight retained recordings: all returned structured results, all merged outputs
  deterministic across two runs. Rhythm outcomes, signal metrics, stiffness and
  vascular-tone cards exactly match staging. All merged cards exactly match a
  fresh replay of the Codex source. Four video-only fitness cards become unavailable
  and three values change; these are inherited Codex rate-resolver effects, not
  merge regressions. Video-only replay has no SDK reference; the additional real
  morphology-to-fitness wiring test verifies staging option A still computes from
  a valid same-scan live reference when the video's rate is unresolved (7 wiring
  tests passed). No numeric availability or accuracy improvement is claimed.
- Replay inputs are copied by the harness before processing. Replay comparisons
  and protected-file hashes are stored under the audit directory.
- Frontend label cleanup follows in commit `6d31ec7`; build passed again.

These checks verify integration behavior, not clinical accuracy. No deployment is
performed by this merge; deploy the backend before its frontend counterpart.
