# Real-world scan hardening audit

Status: implemented regression contract for the research prototype. Synthetic
fixtures prove software behavior and wiring; they are not evidence of clinical
accuracy or population performance.

The capture contract is:

`READY -> record verified evidence`  
`REPEAT_SCAN -> pause/recover or restart with an actionable reason`  
`NO_RESULT -> finish without a classification or biomarker value`

No failed gate is replaced by a default value. Arterial stiffness, vascular
tone, and cardiorespiratory-fitness outputs remain **Research Estimate /
Prototype** and require endpoint-usable, cross-region pulse evidence.

## Failure matrix

| Case | Reproduction | Root cause found | Implementation hardening | Expected behavior / regression |
|---|---|---|---|---|
| Low light | Dim synthetic face/scene below the face-region illuminance floor | Whole-frame brightness can misrepresent the photons on facial skin; dark pixels also destabilize tracking | Keep the face-region lux check, record per-ROI dark fractions, and exclude frames with fewer than two exposure-usable ROIs | `REPEAT_SCAN` with “face a bright, even light”; sustained invalid capture ends `NO_RESULT` (`test_consumer_profile_still_fails_closed_on_darkness`) |
| Uneven light | Opposite face halves at low/high luminance | A mean face-luma check missed strong side-to-side imbalance | Measure per-ROI luma imbalance; normalize/extract each ROI separately and use robust cross-ROI fusion | Recover when at least two regions remain coherent; otherwise `REPEAT_SCAN` with even-light guidance (`test_live_photometric_faults_are_actionable`) |
| Overexposure | Three or more facial ROIs clipped near 255 | High mean luma previously passed the one-sided darkness gate while destroying pulse amplitude | Add decoded-frame and facial-ROI clipping fractions and an overexposure gate | `REPEAT_SCAN`/`NO_RESULT`, never a fabricated pulse (`test_live_photometric_faults_are_actionable`, `test_overexposed_file_fails_before_signal_extraction`) |
| Motion artifacts | Alternating large translations during a scan | Advisory mode displayed a warning but continued accumulating corrupted frames | All physical and signal checks now gate *recorded good time* even in advisory-start mode | Timer pauses with `REPEAT_SCAN`; it resumes only after stable evidence (`test_head_motion_pauses_advisory_recording`) |
| Head movement | Synthetic face translations and detector centre motion | Centre motion and tracker quality were not coupled to advisory recording | Preserve raw tracking motion, gate good time, and keep pause holes in the capture clock | Slow/recoverable movement resumes; sustained movement restarts or fails visibly (`test_scan_timer_counts_only_good_time_and_pauses`) |
| Partial face / occlusion | Face/ROIs placed partly outside the frame | Clamped ROI boxes silently averaged the remaining wrong pixels | Enforce nominal ROI integrity before sampling; never append partial-face samples | `REPEAT_SCAN` with centre/move-back guidance and zero contaminated ROI samples (`test_partial_face_is_rejected_before_roi_sampling`) |
| Poor camera FPS | A 20 fps capture and low-rate live timestamps | Container FPS alone could look acceptable while the capture clock was slow | Use per-frame capture timestamps; keep 24 fps consumer and 30 fps research floors plus a 15 fps hard floor | 25 fps is labelled as coarser consumer evidence; 20 fps fails (`test_consumer_profile_fps_tolerance_with_caveat`) |
| Dropped frames | Isolated and recurring timestamp holes | Extraction assumed uniform sampling across a single missing frame because the gap cut was 2.5 periods | A gap over 1.5 frame periods is now a segment boundary; SQI sees only analysable segment samples | Isolated/limited holes recover; dense holes produce `REPEAT_SCAN` (`test_isolated_hole_forgiven_dense_drops_and_low_fps_denied`, `test_recurring_dropped_frames_start_and_yield_result`) |
| Duplicated frames | Lossless video with repeated decoded content | Advancing timestamps made frozen camera frames look like new evidence | Detect exact decoded-frame repeats in file probe and live capture; skip isolated repeats and pause/reject sustained repeats | Explicit camera-delivery guidance; duplicated content never enters the trace (`test_repeated_frames_fail_capture_visibly`) |
| Invalid/collapsed timestamps | Sidecar too short/long, non-monotone times, or near-zero adjacent intervals | Sidecar validation checked monotonicity but not exact frame count; median FPS hid collapsed intervals | Require one timestamp per decoded frame, validate finite/strict order, and gate collapsed-interval fraction | Bad inputs fail visibly; no timestamp is silently nudged into validity (`test_timestamp_sidecar_length_must_match_video_exactly`, `test_nonmonotone_and_mismatched_inputs_fail_closed`) |
| Weak rPPG | Skin-coloured face with noise but no pulse modulation | Treating provisional SNR/coherence as a capture failure could deadlock an otherwise usable attempt | Start on basic capture checks and keep SNR/coherence as live guidance; process the complete scan, then enforce the final evidence and biomarker gates | The scan completes, but an unsupported signal returns `REPEAT_SCAN`/`NO_RESULT` with null biomarkers (`test_weak_signal_guides_without_blocking_full_capture`, `test_low_signal_scan_completes_but_never_forces_a_result`) |
| Skin-tone variation | Light, medium, and dark synthetic reflectance with the same pulse | The fallback tracker used an absolute red-channel threshold that excluded darker/low-exposure skin | Use relative RGB plus YCrCb chroma for fallback localization; ROI filtering itself remains colour-agnostic | Pulse recovery is regression-tested across the synthetic reflectance range (`test_clean_pulse_recovery_across_synthetic_skin_reflectance`) |
| Facial hair / glasses | Dark occluder and specular highlight in one ROI; one trace corrupted | Straight spatial means and all-ROI arithmetic SQI let one region poison the scan | Trim within-ROI luminance extremes; aggregate per-ROI SQI by median while cross-region beat verification remains mandatory | One obstructed region is tolerated; fewer than two verified regions abstain (`test_robust_roi_mean_tolerates_glasses_hair_and_highlights`, `test_one_occluded_roi_does_not_poison_three_clean_regions`) |
| Distance / framing | Too-small face and out-of-frame ROI geometry | Face box width alone did not prove usable skin coverage | Combine minimum face width, center offset, and nominal ROI integrity | Actionable move-closer/move-back/centre guidance before sampling (`test_face_too_far_is_actionable_and_not_sampled`, `test_partial_face_is_rejected_before_roi_sampling`) |
| Short/interrupted scan | Call finish before the configured good-time duration | The server accepted early `finish_scan()` calls and processed a partial file | Enforce configured *good seconds* server-side; paused wall time never counts | Remains recoverable with `REPEAT_SCAN`; processing cannot begin early (`test_short_or_interrupted_scan_cannot_be_finalized`) |
| Noisy beat detection | Half/double intervals, split pairs, and cross-ROI timing disagreement | Rhythm dispersion cannot distinguish arrhythmia from detector errors by itself | Keep harmonic, split-burden, timing-precision, interval-count and cross-ROI gates ahead of classification | `REPEAT_SCAN`, never an irregular-rhythm class (`tests/test_false_positive_guards.py`) |
| Unstable ROI tracking | Alternating center/size jitter and a face reacquired after occlusion | Stability measured only center jitter and the EMA blended a reacquired face with stale geometry | Include raw size jitter and found-frame coverage; reset smoothing after a real gap; reopen detector fallback chain | Low stability lowers SQI/pauses; reacquisition starts from the current face (`test_tracker_resets_stale_geometry_after_occlusion`) |

## Biomarker availability

The biomarker stage may be independent of a rhythm-specific classification
failure, but it is not independent of measurement quality. It requires:

- composite SQI at or above the production floor;
- stable facial ROI tracking;
- pulse beats verified by cross-region coherence, or by the existing strict
  two-region timing/match rule;
- at least eight morphology-usable beats across at least two facial ROIs.

If those inputs do not survive, all three biomarker values are null with a
machine-readable reason. A clean accepted scan still produces all three
prototype estimates and exposes method, confidence, beat count, ROI count,
timestamp diagnostics, photometric diagnostics, and provenance.
