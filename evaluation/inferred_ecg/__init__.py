"""
Inferred-ECG falsification lab (v0.2 M4) — QUARANTINED RESEARCH MODULE.

Purpose: prove, with our own data, why generated ECG waveforms do not
ship. A camera measures the mechanical/hemodynamic consequences of the
heartbeat; a PPG->ECG decoder therefore produces beat-timed truth wrapped
in hallucinated electrical morphology (P/T/ST from training priors). This
module trains such a decoder and runs the three tests that expose it
(identity-template baseline, interval-level error, cross-rhythm
morphology challenge).

INVARIANT 10: nothing under this package may be imported from `app/` or
`inference/pipeline.py` (a test walks the import graph); every artifact
is watermarked and written only under `evaluation/falsification_runs/`.
"""
WATERMARK = "SYNTHETIC ECG — RESEARCH ONLY — NOT A MEASUREMENT"
