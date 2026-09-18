# Standalone AFib scan (`/beta/cardio-afib`) — status

_Last updated 2026-09-18._

## What it is

A self-contained browser AFib screen, a sibling of `/beta/cardio-staging` but with
**no ShenAI and no video upload**. The webapp opens its own `getUserMedia` camera,
places the four skin ROIs from MediaPipe FaceLandmarker (mirroring
`preprocessing/roi.py`), samples each ROI's mean RGB per frame, and POSTs only those
per-frame means to `POST /api/measure-traces` → `inference/trace_ingest.py` → the full
pipeline, which returns exactly one of **AFIB_DETECTED / AFIB_NOT_DETECTED /
INCONCLUSIVE**. No codec in the path, no SDK competing for the CPU.

Code: webapp `src/Pages/CardioAfib/CardioAfibScan.jsx`, `src/lib/scan/afib/*`,
`src/lib/scan/staging/cameraCapture.js`; cardio `inference/trace_ingest.py`,
`app/measure_api.py::measure_traces`, `inference/evidence.py`.

## Outcome

**Goal met.** Every completed scan returns exactly one of the three results, and
returns INCONCLUSIVE _only_ when the evidence cannot support a call — the owner's
spec. In clean, regular captures the scan reaches **ACCEPT / AFIB_NOT_DETECTED** with a
real pulse (verified on mobile: coherence ~0.4, pulse ~82 bpm, 4/4 test-retest agree).

## The fix that mattered

**Mobile no-pulse root cause = a bursty frame clock, not the signal.** A phone
landmarking on the GPU thermally throttles and delivers frames in bursts (~30 fps, then
~200–300 ms stalls). Every stall exceeded `capture_segments`' 1.5×-median split, so a
60 s scan shattered into sub-3 s pieces — 0 usable segments → coverage 0, 0 beats,
cross-ROI coherence _exactly_ 0. The frames carried the pulse; the irregular timestamps
threw it away.

**Fix:** `inference/trace_ingest.py` resamples the kept frames onto a uniform grid at
the median rate (`MAX_BRIDGE_S = 0.75` keeps real face-loss gaps as breaks). A synthetic
bursty clock that yielded 0 beats recovers 88; a no-op on a clean clock. Regression test
in `tests/test_trace_ingest.py`. Client also dropped MediaPipe detection 15 Hz → 10 Hz
to hold the frame rate up. (cardio `staging` @ `120edb4`.)

## Levers tried, and their verdicts

| lever | verdict | why |
|---|---|---|
| Trace-clock resample | **shipped ✓** | the real fix; 0 beats → reliable results |
| Raise any-class timing gate (#1, 40 ms) | **not done** | optimizer guard auto-rejects gate-loosening; miss-risk unmeasurable (no ground truth); and moot — timing now sits under 40 ms. An owner policy call, not a signal fix. |
| Dual POS+CHROM waveform fusion (#3) | **rejected** | optimizer REJECT: signal 0.827→0.740, cards regressed. Phase misalignment cancels the pulse; POS is the right primary. Reverted. |
| Exposure lock | **non-viable** | Android 10 / Chrome 151: `exposureMode:"manual"` jumps to a wrong gain (blown out or dark, one scan < 100-lux floor). `freezeExposure` kept in `cameraCapture.js` but unwired. |
| Pre-scan capture gate | **shipped, marginal** | gates Scan on face + forehead + light + stillness held ~1.5 s. Improved timing (~42→~32 ms) but not coherence; outcomes unchanged. |

## The ceiling

Cross-ROI coherence on a consumer phone sits at **0.2–0.5**, right on the decision
gates. Client-side capture hygiene (stillness, framing, light) trims variance at the
margin but cannot manufacture coherence the modality does not deliver — the ~10 Mb/s
usable-rPPG floor (McDuff 2017) vs Android Chrome's ~7 Mb/s recorder cap. Borderline
captures therefore return INCONCLUSIVE ("timing jitter alone can look like
irregularity") — the never-diagnose behavior working as designed, not a bug. Raising the
positive-confirmation rate is a **capture-hardware** problem (even lighting, closer/
steadier framing), not a software one.

## If pursued further

- **Cross-window agreement (#2)** — decide only when independent sub-windows agree (the
  design behind Cardiio Rhythm's 95/96 %). Could convert some borderline scans to
  decisions and raise trustworthiness; will not lift the coherence ceiling. Hook-protected
  (`inference/pipeline.py`), needs owner sign-off. `scan.subwindows: 3` is currently dead
  config.
- **A standalone-trace corpus** — the optimizer's `data/eval_corpus/` is compressed clips,
  so it validates the clip path, not standalone traces. Retain real traces first to
  optimize this path.

## Deploy

Staging only. cardio `personal/staging` → Railway `avatarx-cardio-staging`; webapp
`origin/staging` → S3/CloudFront. Never repoint `/beta/cardio` at a staging component.
