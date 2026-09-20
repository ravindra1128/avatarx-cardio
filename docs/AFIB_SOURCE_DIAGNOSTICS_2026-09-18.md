# Rhythm source diagnostics — 2026-09-18

The 12:51 scan abstained on video AF-specific gates before evaluating the SDK
candidate. Its existing SDK audit showed a different interval distribution, but
there was no SDK classification to compare. This change records both source
assessments without changing the published AFib decision or source-selection policy.

`debug.rhythm_comparison` preserves the original video result and evaluates the
SDK train on isolated inputs using the same train checks and classifier as the
production route. Production eligibility is reported separately. Missing, invalid,
rejected, or failed SDK assessments have an explicit state/reason and no invented
candidate classification. Diagnostic exceptions cannot replace the scan result.

The SDK candidate still uses video quality evidence and rate corroboration. These
are not independent detectors. Their windows are not synchronized; alignment is
`unknown` and verified overlap remains null. SDK beat-window endpoints, gaps,
clean-run statistics, failed gates and classifier features remain available for
investigation. No raw video, waveform, or beat arrays are added to diagnostics.

The tracking sheet adds source results, gates, SDK policy eligibility/blocker,
agreement, alignment and diagnostic runtime. Existing columns remain intact.
No environment variables or frontend changes are required. Retention is unchanged.

## Validation scope

This is an explicitly requested observability change, not a signal optimization.
The optimizer's structural snapshot was taken before editing. Its narrow signal
file allowlist rejects the application/telemetry/test paths as out of scope; no
protected thresholds or gate/config files were changed. No optimizer acceptance or
clinical accuracy improvement is claimed. Video corpus replay cannot exercise this
hook (the offline harness calls the video-only entry point). Instead, validation
compares saved production-route outputs before/after extraction and tests the
actual service hook, source rejection paths, retention and recovery contracts.

Twenty saved production-route cases are identical before and after the shared
helper extraction. Focused regression tests cover regular/irregular controls,
scan-quality rejection, blocked/accepted/disabled production routes, missing and
malformed sidecars, strict JSON, clock uncertainty, exceptions, both upload
transports, sheet compatibility and unchanged published decisions. These fixtures
are software checks, not physiological validation or evidence of improved repeatability.
