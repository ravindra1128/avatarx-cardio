# Vascular Tone investigation — 2026-09-20

## Decision and scope

The shipped calculation is an **experimental normalized facial pulse-amplitude variability index**, displayed as a custom score. It is not a direct measurement or validated surrogate of vascular smooth-muscle tone, systemic vascular resistance, endothelial function, arterial compliance, or perfusion. A lower repeatability error alone cannot validate that label.

This audit traces the `/beta/cardio` source route, inspects production and staging health, reads both tracking tabs without modifying them, and replays retained recordings through the current local checkout. These are three distinct provenance levels: local code is not proof of the exact deployed web bundle; a replay is not a live camera test. The local checkout was already dirty (fitness, transport, reporting and ShenAI work); those changes are preserved. The webapp's AGENTS.md reserves future Codex edits for `/beta/cardio-codex`; the requested production route is inspected here. No deployment is performed.

Owner confirmed in this conversation that September 16–17 retained recordings are the same person, same phone, comparable resting conditions. Other historical sheet pairs grouped by browser are only exploratory. A browser user-agent is not a participant identifier. No clinical vascular reference is available.

## Exact dependency map

Paths below are relative to `/Users/ravindrasinghbisht/Downloads/webapp` (frontend) or `/Users/ravindrasinghbisht/Downloads/cardio` (backend).

| Stage | Active implementation and behavior |
|---|---|
| Route | webapp `src/App.jsx:343`: `/beta/cardio` → `GLucosePredictionV2`; results route → `GlucosePredictionResults` |
| Scan page | `src/Pages/GlucosePrediction/GLucosePredictionv2.jsx`: imports `avatarxvitals/index.mjs`, creates ShenAI SDK, calls `initialize`; custom measurement duration 50 seconds; SDK state controls actual finish |
| Capture checks | same page, `runCapturePreCheck`, `lockCameraOncePositioned`, `lockCameraOnScanBegin`; `src/lib/capture/cameraCapture.js`: measured FPS/luma and attempted exposure/AWB lock with brightness guard. Reported booleans are not per-frame gain telemetry |
| Orchestration | `src/lib/scan/ScanOrchestrator.js`; start on active SDK measurement state; finish processors concurrently after SDK FINISHED; failures/cancel cleanup are separate |
| Recorder | `src/lib/scan/processors/MediaRecorderProcessor.js`: SDK camera track, canvas fallback at 10 FPS; MediaRecorder video, not a browser vascular feature. `/beta/cardio`: 5 Mbps request, 3-second chunks, 48-second rolling retention. Chunk zero is retained with the tail, creating a packet-clock gap on long scans. No S3 upload on this route |
| AFib processor | `src/lib/scan/processors/AFibProcessor.js`: waits for `whenBlobReady`; fire-and-forget delivery after finish; enabled only with `VITE_AFIB_URL`, unless `localStorage.afib_disabled` disables it |
| Requests | preferred `POST /api/upload-part?upload_id=...` per slice → `POST /api/start` → `GET /api/result?upload_id=...`; fallback `POST /api/process-video?...` carrying the video blob. Slice retries and start retries exist; network recovery polls by upload ID |
| Request metadata | session, unique upload_id, duration_ms, capture_profile=consumer, window_s=70, ext=webm/mp4; optional ref_hr/ref_hrv/ref_sbp/ref_dbp/ref_source; exposure_locked/awb_locked/client_fps/face_luma/capture_note. Reference values do not enter tone. No browser vascular-tone calculation or ROI traces in this route's upload |
| HTTP | `app/measure_api.py`: `MeasureHandler.do_POST`, `_upload_part`, `_start_job`, `_run_assembled` or `_run`; `_build_manifest` and `_parse_envelope` map metadata; `measure_video_details` does analysis |
| Preparation | `app/measure_prep.py`: `trim_tail`/`choose_tail_cut` use packet times; `downscale` to configured analysis box with FFV1; `build_timestamp_sidecar` selects capture clock. Requesting 70 seconds cannot recover discarded middle chunks |
| Ingestion | `inference/pipeline.py::run_with_details` → `capture/ingest.py::ingest_video`: decoding, capture checks, face tracking, `preprocessing/roi.py::mean_rgb`; forehead, left cheek, right cheek, nose RGB traces |
| Beat signal | `inference/evidence.py::extract_and_detect`, `capture_segments`; configured POS/CHROM extraction → ROI beat detectors → `beats` multi-ROI fusion → confidence calibration → runs/regularity/SQI/evidence |
| Tone waveform | `features/hemodynamics.py::resting_hemodynamics` independently re-extracts **POS**, even if the rhythm extractor changes; morphology band 0.5–min(10, 0.45×FPS) Hz; orientations aligned across regions |
| Windows | `features/pulse_morphology.py::_beat_pairs`: adjacent confidence-qualified lattice beats; `_beat_windows`: local foot-to-foot windows from waveform minima preceding each beat |
| Amplitudes | `features/hemodynamics.py::segment_amplitudes`: max-minus-min within each window; normalize by that ROI's median in that capture segment |
| Fusion | `pool_amplitudes`: longest eligible ROI series provides reference foot timestamps; other ROI amplitudes join only within 50 ms; median of currently contributing regions |
| Estimator | `vasomotor_indices` → `tone_score_from_cv`: SD/mean of pooled amplitudes, ×100, clip to [0,100], round to 0.1 |
| Serialization | `app/report_data.py::report_biomarkers`: `biomarkers.items[]`, key `vascular_tone`, value/unit/metric/tier/details/raw/confidence; response also carries outcome, timing, trim, clock. Exceptions become an explicit biomarkers error |
| Storage/UI | processor `_settle` writes `{ok,data}` to `sessionStorage.afib_result`; `src/Pages/GlucosePrediction/CardioBiomarkers.jsx` polls that key every second for up to 300 s, selects `biomarkers.items`, renders `BiomarkerCard`; score rounded to whole number for `/100` |

Additional conditional consumers on the shared scan page matter for capture load and result latency: `GM0513Processor` / `src/Pages/GlucosePrediction/gm0513LiveStream.js` stream JPEG frames to `VITE_GM0513_URL` with `/ws/scan` unless disabled; these data do not feed tone but compete for camera/browser resources. `onMeasurementComplete` also calls `${VITE_GLUCOSE_PREDICTION_API_URL}/op46/estimate` when configured, sending FormData containing SDK PPG/vitals/context, and awaits it before navigation. This call has no explicit timeout in `callEndpoint`. A completed cardio backend result can therefore coexist with delayed results navigation. It is not a tone estimator input. SDK internals are a bundled dependency; their complete camera/normalization implementation is not auditable from the route wrapper.

The beat confidence artifact is `inference/confidence_calibration_v01.json`; default `configs/default.yaml` uses `decision.extractor=pos`, `classifier=interim_rules` and SQI pulse band 0.7–4 Hz. Those determine the lattice used for tone windows, whereas tone re-extracts with its wider morphology band. There is no trained vascular-tone model in this path.

The five research heads are not this card. `heads/head_vasotone.py` and `research/vascular/tone_runner.py` concern a different research/provocation track. The existing `research/facial_perfusion_pilot/` is an isolated acquisition/reference pilot; it is not imported by this production card. ShenAI route reconciliation changes fitness/rhythm, not the tone formula.

## Definition, all inputs, transformations and failure modes

For ROI r, segment s, detected beat k:

```
A[r,s,k] = max(POS waveform over foot-to-foot window) - min(...)
N[r,s,k] = A[r,s,k] / median_k(A[r,s,k])
P[k] = median of N values whose foot is within 0.05 s of reference ROI foot
CV = population_SD(P) / mean(P)
display = round(clip(100 * round(CV, 5), 0, 100), 1)
```

CV is dimensionless; raw output is percent CV, rounded to 0.01; displayed unit is `/100`. The UI rounds the main number again to an integer. `TONE_TYPICAL=(20,60)` describes prototype observations, not a clinical normal range. The LF envelope fraction at 0.04–0.15 Hz is computed only with ≥30 s span, using interpolation to 4 Hz, Hann window and FFT. **It does not influence the displayed tone score.** At 0.04 Hz, 30 s contains only 1.2 cycles; this is weak spectral evidence and interpolation across gaps is another confound.

Dependencies of the score: pixels, ROI placement/size, exposure/gain/WB, compression, timestamps/FPS, capture segmentation, POS adaptive channel mixing and overlap-add, filter response, beat confidence and adjacency, foot locations, retained windows, normalization medians, ROI eligibility/order and pool membership, final SD/mean. HR influences beat windows and band response indirectly. HRV, demographics, ShenAI BP, notch/aging index, vasomotion power and previous scores are not direct formula inputs.

Four beats are needed for a region's segment normalization; five amplitudes over the scan qualify a ROI for provisional pooling; five pooled beats allow a provisional number; twelve support the card's own measured beat-count tier. Scan-level quality failures demote tiers. There is also a shared morphology early return before tone computation: no readable contour or too few morphology beats can suppress tone even if amplitude extraction was attempted. Treat this as a coverage dependency to investigate, not permission to bypass waveform quality.

Constant multiplication of all amplitudes cancels algebraically in CV. Within-segment median normalization does **not** remove time-varying gain, changes in POS mixing, additive noise, codec distortion or motion. Independent segment normalization also removes real between-segment amplitude differences; it cannot establish absolute perfusion. The POS implementation sums overlapping windows without dividing by overlap counts, creating edge envelopes; global normalization was already tested and rejected in iteration 23. Robust MAD CV was rejected in iteration 29. Neither should be presented as a new discovery or silently reintroduced.

The estimator clips real CV >100% to 100; this existing saturation can hide variability and is excluded as a stabilization technique. A score of 100 is not maximum physiological vascular tone.

## Transport, environment and state risks

Production health: build `c956a4574110`, sheet gid `140358874`; staging: `120edb41dbb2`, gid `342232654`. Separate tabs confirmed. Staging additionally selects `model_a`. Health snapshots are saved in `health.json`. `/beta/cardio` reads build-time `VITE_AFIB_URL`; the actual deployed frontend environment is not proven by this local source audit. Missing URL/kill switch can prevent a vascular request despite successful SDK completion.

Server cache uses upload ID; chunked heartbeat keeps a single-shot request alive and a detached job survives a lost connection. Leading spaces are JSON whitespace. HTTP 200 can contain an error; the client must inspect content. Concurrent phone runs and missing credentials must not be mistaken for estimator failures. Sheet writes run after result delivery.

The production UI is **not scan-scoped**: start removes the old shared key, but a previous processor's late `_settle` can write into it after a new scan starts. The card accepts the first stored envelope and stops polling. This was reproduced by loading the actual processor classes, starting A then B, and delivering A late: `afib_result` contained A. The UI source accepts that first envelope. This lifecycle reproduction is not proof that it caused the observed sheet variance. The Codex/staging copies already contain stronger scan-ID handling; they are distinct implementations. The production UI hides provisional reasons and most method/limitation text; raw marker text omits `% CV` although the backend sends it. This weakens interpretation, independent of numerical stability.

## Baseline and intermediate measurements

`replay_audit.py` instruments segment amplitudes and pool contributors in-process, without changing production functions on disk. It copies each corpus input and checks its SHA-256 after replay. It saves raw RGB summaries, normalized and unnormalized amplitude series, contributors, gates, tone payload, timing and trimming. `analyze.py` writes `analysis.json` with per-scan and paired metrics. Use the final JSON rather than partial running outputs. Three missing phone results have three amplitude windows per segment/ROI, below the existing four-window normalization requirement; one has no usable lattice. This is a measured availability loss, not a reason to lower the floor. Local replay uses 70 s request and 480×360 box; it is separate from the optimizer's default-window regression baseline.

Read-only historical sheet audit:

| Metric | Production | Staging |
|---|---:|---:|
| Rows | 67 | 125 |
| Numeric tone results | 52 | 104 |
| Explicit measured-tier results | 18 | 64 |
| Mean / sample SD, all numeric rows | 45.47 / 18.09 | 39.80 / 11.34 |
| CV across numeric rows | 39.77% | 28.49% |
| Min–max | 21.35–100 | 23.1–87.8 |
| Consecutive ≤20-min numeric pairs, same browser/build/config | 24 | 64 |
| Median absolute pair change | 12.0 | 6.15 |
| Maximum absolute pair change | 78.6 | 51.4 |
| Exploratory ≤20% relative-difference pass | 9/24 | 34/64 |

Production history also mixes nine `% CV` rows, 42 `/100` rows and missing-unit rows; the capped score and uncapped marker are not interchangeable at saturation. These historical statistics span code versions and conditions; they are not an A/B result or validated within-subject CV. `ACCEPT` is a rhythm outcome, not an independent definition of usable vascular information. Numeric coverage across all rows is not usable-scan coverage. Both denominator distinctions must be preserved.

The first tune record already demonstrates deterministic feature sensitivity: +10% constant amplitude gain gives 48.7→48.7; a 0→10% gain ramp gives 48.7→49.4; +10% to one amplitude can change the score by 2.3 points; shifting one ROI's foot times by +20 ms with identical amplitudes changes it by as much as 6.8 points. This isolates a pool-membership artifact. It does not estimate the fraction of real physiological variance caused by timing jitter.

The earliest observed between-record differences exist in raw RGB and selected amplitude windows. Without synchronized gain/illumination/motion and reference perfusion traces, their physiological versus optical cause cannot be uniquely identified. A variance decomposition cannot be inferred from correlated sheet features. The controlled timestamp perturbation identifies one causal software mechanism downstream.

### Completed retained-record baseline

All 33 unique recordings returned structured documents; 27 contained numeric tone values. Among the 16 owner-confirmed resting phone recordings, 12 computed (75% of all retained scans, **not** 75% of independently qualified vascular scans); only two carried the measured tier. Numeric mean 46.975, sample SD 15.012, min 19.4, max 74.8, range 55.4, across-record CV 31.96%. These recordings span two days; the pooled CV is not the repeat-pair endpoint.

Ten consecutive pairs were within 20 minutes; only five had both values. Their mean absolute difference was 9.42 points (sample SD 10.52), range 1.8–26.2. Three of five numeric pairs met the exploratory 20% relative-difference threshold; across all ten attempted pairs that is 3/10. This threshold is reported for comparison with historical work, not claimed as a scientifically justified tolerance.

| Confirmed pair, UTC | Score | Forehead mean green intensity | Forehead normalized amplitude sample CV |
|---|---|---|---|
| Sep 16 12:37 → 12:45 | 74.8 → 48.6 | 164.16 → 124.11 | 0.809 → 0.514 |
| Sep 17 06:10 → 06:14 | 43.9 → 41.8 | 171.25 → 183.51 | 0.584 → 0.443 |

The first pair diverges before ROI fusion; its green level changes by about 24.4%. That is evidence of different optical input, but without gain/light/reference traces it cannot establish how much is camera artifact. The second pair shows why stable displayed output does not establish stable intermediates. Full region-level values and contributors are in `analysis.json` and `per_scan.csv`.

For unrounded CV = sigma/mu, sensitivity to amplitude a_i is `(a_i-mu)/(n*mu*sigma) - sigma/(n*mu^2)` where sigma>0. Thus relative outliers and small sample counts can dominate changes. A +5 bpm HR perturbation has no direct algebraic effect if amplitudes/windows are held fixed; a true HR change can indirectly change window placement/filter response. A 20-ms time shift has no effect in SD/mean itself, but can discontinuously change the current ROI pooling.

## What stability target is scientifically defensible?

No validated clinical meaningful-change threshold exists for this custom CV score in the literature examined. A universal ±10–20% tolerance would be an engineering preference, not evidence. Report score differences in **percentage points of amplitude CV**, plus relative difference when both values are well away from zero. Preserve unclipped CV for analysis.

Use repeat-pair differences and Bland–Altman limits, within-subject SD and repeatability coefficient `1.96*sqrt(2)*s_w`, with participant/session-aware uncertainty. Estimate limits on development data and lock them before testing new participants/devices. Do not set the acceptance limit from the same records used to score a candidate. Require narrower limits without losing reference responsiveness or coverage, and no material worsening of individual pairs. The current sample of one person cannot define population biological variation or clinical equivalence.

Even ideal IID normal amplitudes with CV 0.4 and n=12 yield an approximate CV standard error of `0.4*sqrt((1+2*0.4^2)/(2*12)) ≈ 0.094`, or 9.4 score points. Correlated, selected, non-Gaussian beats make that approximation optimistic. A twelve-beat floor cannot promise a tight repeatability bound. This is a sampling illustration, not a clinical tolerance or fitted confidence interval.

## External evidence and realistic alternatives

1. **True tone:** active smooth-muscle constriction relative to passive vessel behavior. Direct experimental characterization uses active/passive diameter or tension under controlled pressure, not facial amplitude CV. [Primary pressure-myography protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC4545206/) and [VasoTracker 2](https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP289322) describe the measurement framework. Systemic resistance additionally requires pressure gradient and flow/cardiac output; cuff pressure alone is not tone ground truth.
2. **Facial pulsatile perfusion index, stronger candidate:** Rasche et al. studied 37 volunteers with simultaneous laser-speckle imaging and camera PPG during baseline, cognitive and heat challenges. They used 12-bit industrial imaging at 25 FPS, two-minute recordings and AC/DC amplitude processing. This supports investigating a local perfusion-related signal; it does not validate consumer-phone tone or supply our 10–20-minute score tolerance. [Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7536393/).
3. **Regional physiology cannot be assumed:** Fleischhauer et al. studied 22 healthy participants using 100-FPS, 12-bit cameras, finger PPG, continuous pressure, ECG and respiration. Facial amplitude response to cold stress differed from finger amplitude response. Extensive exclusions left 24 recordings for analysis. Finger vasoconstriction therefore cannot simply label facial amplitudes. [Primary paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVPM/Woyczyk_Impact_of_Sympathetic_Activation_in_Imaging_Photoplethysmography_ICCVW_2019_paper.pdf).
4. **Perfusion imaging implementation:** PulseCam combines motion stabilization, AC/DC normalization and reference pulsation information. Occlusion experiments involved 12 participants; incremental occlusion involved 15. The findings support relative tissue-perfusion experiments, not an absolute vascular-tone score from one uncontrolled facial clip. [Primary study](https://www.nature.com/articles/s41598-020-61576-0).
5. **Morphology:** reflection, crest time and derivatives mix pressure, heart rate, local transfer properties and acquisition effects. A vasopressor rPPG case series uses controlled palm recordings and three-minute ensemble averaging; it does not validate our short resting face route. [Case series](https://www.mdpi.com/2077-0383/15/3/1118). No justified reason to replace CV with another unvalidated morphology-to-tone map.
6. **Transit time:** distal synchronized anatomical sites and sufficiently high frame rate are needed. Four nearby facial ROIs, rolling shutter and ~30-FPS sampling cannot be treated as an established pressure/tone sensor. [Full-body rPPG study](https://openaccess.thecvf.com/content/CVPR2023W/CVPM/html/Niu_Full-Body_Cardiovascular_Sensing_With_Remote_Photoplethysmography_CVPRW_2023_paper.html).
7. **ML and datasets:** [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) and its [NeurIPS paper](https://arxiv.org/abs/2210.00716) provide POS/CHROM and learned waveform benchmarks (PURE, UBFC, BP4D+, etc.). These are useful extraction baselines, not vascular-tone labels. HR-optimized or amplitude-normalized targets can erase the amplitude information needed here. [MSPM](https://arxiv.org/abs/2402.02224) supports multi-site pulse/pressure research; pressure is not a direct tone label. No trained tone model is justified by our one-person corpus.

AC/DC is invariant to ideal constant multiplicative illumination, but not to additive offsets, nonlinear camera response, auto-exposure transitions, spatially changing illumination or compression. Preserve raw linearizable RGB/DC and acquisition metadata; do not obtain a perfusion index by dividing a POS waveform by its own median or standard deviation. HR/HRV and LF amplitude power are contextual signals, not replacements for measured vascular tone.

## Ranked, falsifiable implementation plan

| Priority | Problem → cause/evidence → change | Experiment / success / failure | Scope and complexity |
|---|---|---|---|
| 1 | Pool membership changes with ROI foot jitter; same-amplitude 20-ms perturbation changes score → associate amplitudes by their existing shared lattice beat identity | One isolated candidate; unchanged beat windows/amplitudes/gates. Require invariance to bookkeeping timestamp jitter, improved confirmed-pair repeatability, unchanged coverage/other cards, optimizer accept. Reject any unexplained record regression | Feature aggregation, small |
| 2 | Missing traceability and shared result key → bind request, response and UI to one scan ID | Finish A after B starts; A must never appear on B's page. Also test reload, failed B, polling expiry. Numeric values must be identical | Frontend delivery; use isolated Codex route, moderate |
| 3 | Optical modulation and codec artifacts corrupt amplitude → stabilized capture and per-frame exposure/gain/WB/luma/motion telemetry, fixed-duration contiguous usable window | Same subject/device, randomized lock/control, constant-light phantom and real reference. Require smaller within-session measurement error without blunting reference changes; reject darkening/FPS/SQI/coverage losses | Capture/frontend and metadata, moderate; phone acquisition required |
| 4 | CV is not perfusion or tone → separate raw-channel AC/DC perfusion-related research output | Controlled repeat baseline plus safe supervised physiological/reference protocol, phantom/sham controls, synchronized local LSI/laser Doppler; finger PPG and pressure as contextual controls. Require test-retest limits and reference responsiveness on held-out sessions/devices. Reject artifact response or absent local-reference association | Fundamental estimator/acquisition redesign, high; additional reference hardware |
| 5 | Short/noisy morphology and POS edges → investigate amplitude-preserving extraction with verified transfer response | Known-amplitude waveform/optical phantom + retained clips, then new reference recordings. Measure modulation recovery and noise separately; reject “improvement” that merely flattens envelopes | Signal processing, moderate/high; prior global overlap-normalization reject must be respected |
| 6 | Clinical label and claimed range exceed evidence → show experimental pulse-amplitude variability, raw %CV, uncertainty/provisional reason | Payload/UI contract tests; readers can distinguish research index from validated physiology. This improves honesty, not repeatability | Additive backend metadata and frontend presentation |

Expected numerical improvements are not promised in advance. None of these experiments authorizes new validity floors, deploys a replacement, or uses prior scores to stabilize the result. Record before/after metrics and retain only supported improvements. Reference recording is required before calling any redesigned output a validated surrogate.

## Verification boundary

Existing frontend `AFibProcessor.test.js` and `ScanOrchestrator.test.js`: 31/31 tests passed under Jest with Node environment. These use mocked fetch/capture and verify lifecycle, upload and delivery behavior; they are not a real-phone end-to-end validation.

## Execution record

Investigation and baseline instrumentation precede pipeline edits. Quantitative candidate results, gate verdict and retained/reverted state are recorded in `EXPERIMENT.md` after the isolated experiment. The primary source of all per-record measurements is the JSON, not a selectively chosen aggregate.
