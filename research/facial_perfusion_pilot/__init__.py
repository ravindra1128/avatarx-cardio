"""Facial Perfusion Response (FPR) pilot — measurement-system qualification
harness (QUARANTINED research module).

WHAT THIS IS
------------
A bench-rig qualification harness for a single scientific question:

    Can controlled facial video reliably detect a repeatable WITHIN-subject
    change in forehead cutaneous perfusion, WITHOUT the camera, lighting,
    compression, or processing pipeline creating the apparent signal?

It qualifies the *instrument* (stages B0-B8 below); it does NOT produce a
vascular-tone score, a clinical assessment, an ML prediction, a production
biomarker, or any physiological claim. The immediate claim under test is
only the one quoted above. See `qualify.py` for the PASS / FAIL /
INCONCLUSIVE verdict logic.

NAMING DISCIPLINE (owner brief, section 15)
-------------------------------------------
The output of this module is the "Facial Perfusion Response (FPR)". It is
NEVER "vascular tone", "vascular age", "absolute perfusion score", "clinical
vascular health" or a "diagnostic biomarker". This module is deliberately
SEPARATE from the existing `research/vascular/` tone track and does not
import, extend, or feed it.

ISOLATION
---------
Nothing here may be imported from `app/` or `inference/` (the repo's
research quarantine). This module reuses only stable *library* code that is
scientifically safe for a perfusion measurement:

  * capture.video_reader  — decode + honest frame clock (timestamp sidecar)
  * rppg._filters         — bandpass (physiological band only)
  * datasets.reference    — contact perfusion-index reference ingestion
                            (lazy-imported, only on the real-data path)

It does NOT reuse `rppg/pos.py` for the signal: POS normalises each channel
by its temporal mean (`seg/mu - 1`), which projects out exactly the DC term
that PI = AC / DC requires. FPR uses the DC-bearing spatial-mean green trace
directly (see `extract.py`).

THE BENCH TESTS (owner brief section 4; report section 9)
---------------------------------------------------------
B0  Quantisation headroom      can the camera+ROI resolve the AC/DC signal
B1  Dark-frame noise           sensor/read noise from lens-capped data
B2  Exposure/WB/focus lock     verified by challenge, not by API claim
B3  Lighting flicker           mains-related flicker / aliasing
B4  Illumination uniformity    spatial variation across the ROI
B5  Warm-up drift              illumination drift after power-on
B6  Phantom stability          non-perfused phantom must NOT show a PI
B7  Synthetic PI pipeline      processing recovers a known injected PI
B8  End-to-end null            sigma_null, the primary pilot viability metric

Only B7 can PASS/FAIL from software alone. The rest consume real rig
recordings and return INCONCLUSIVE until those recordings are present.
"""
from __future__ import annotations

WATERMARK = ("FACIAL PERFUSION RESPONSE (FPR) — MEASUREMENT-SYSTEM "
             "QUALIFICATION — RESEARCH ARTIFACT — NOT A MEASUREMENT — "
             "NOT A DIAGNOSIS — NOT A VASCULAR-TONE SCORE")


class FprError(RuntimeError):
    """Fail-closed error for the FPR pilot harness."""


# --------------------------------------------------------------------------
# Gate thresholds — the owner brief's "Hard gates" (section 5) and the
# report's section 5/9. Each carries its source. These are NOT to be
# weakened to make hardware pass; a change to any of them must be reported
# explicitly (owner brief, "Final rule").
# --------------------------------------------------------------------------

# -- Hardware gates --------------------------------------------------------
QUANT_HEADROOM_MIN = 10.0            # effective resolution >= 10x finer than
                                     # the expected AC/DC (B0)
PHANTOM_DC_DRIFT_MAX = 0.005         # phantom DC drift < 0.5% (B6)
UNIFORMITY_MAX = 0.15                # <15% corner-to-corner across ROI (B4)
FLICKER_MAINS_SNR_DB = 6.0           # a mains 50/100 or 60/120 Hz spectral
                                     # peak this far above the local floor is
                                     # "present" and fails B3
LOCK_ROI_SHIFT_MAX = 0.005           # ROI mean may move < 0.5% of full scale
                                     # under the bright-object challenge (B2)
LOCK_RATIO_SHIFT_MAX = 0.010         # channel-ratio (R/G, B/G) may move < 1%
                                     # under the challenge (B2 white balance)

# -- Signal gates ----------------------------------------------------------
SNR_MIN_DB = 6.0                     # ROI pulse SNR >= 6 dB
BEAT_YIELD_MIN = 0.80                # beat detection yield >= 80%
BEAT_REJECT_MAX = 0.30               # beat rejection < 30%
PI_BASELINE_MIN = 0.001              # PI baseline ~0.1% .. (dimensionless)
PI_BASELINE_MAX = 0.02               # .. ~2%

# -- Noise-floor gates (the primary pilot readout) -------------------------
SIGMA_NULL_PASS = 2.0                # sigma_null <= 2%  -> PASS
SIGMA_NULL_HARD_STOP = 4.0           # sigma_null  > 4%  -> hard stop (FAIL)

# --------------------------------------------------------------------------
# Capture requirements (owner brief section 6; report section 7)
# --------------------------------------------------------------------------
FPS_MIN = 25.0                       # 25-30 fps constant; 25 acceptable
FPS_TARGET = 30.0
ROI_MIN_PX = 150                     # forehead ROI >= 150 x 150 px after crop
BIT_DEPTH_PREFERRED = 10             # >= 10-bit preferred
EXPOSURE_MEAN_MIN = 0.50             # forehead mean 50-65% of full scale
EXPOSURE_MEAN_MAX = 0.65

# --------------------------------------------------------------------------
# PI processing (owner brief section 11; report section 11)
# --------------------------------------------------------------------------
PULSE_BAND_HZ = (0.7, 4.0)           # physiological pulse band (42-240 bpm)
PI_WINDOW_S = 10.0                   # 10-second windows
PI_OVERLAP_S = 5.0                   # 5-second overlap
BASELINE_TAIL_S = 60.0              # baseline = final 60 s of baseline period
GREEN_NM = (520, 550)                # analysis channel: green (520-550 nm)

__all__ = ["WATERMARK", "FprError"]
