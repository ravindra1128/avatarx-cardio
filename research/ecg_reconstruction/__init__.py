"""
ECG-reconstruction research track (v0.3) — the pre-registered empirical
question "can reconstruction reach clinical fidelity?", built as
machinery, not an assumption.

Facial video measurably contains beat timing, rhythm and pulse dynamics;
it does not measurably contain atrial depolarization (P), conduction time
(PR/QRS width) or repolarization (QT/ST/T). The v0.2 falsification lab
(evaluation/inferred_ecg) holds this repo's own numbers: the trained
decoder beats the identity-template baseline on correlation and loses
5-8x on interval fidelity, and a sinus-trained decoder hallucinates P
waves into AF. This package therefore trains and evaluates continuously
as paired data grows; its output becomes user-visible IF AND ONLY IF
every §G gate in configs/gates.yaml passes on participant- AND
session-disjoint held-out data through the production path. Until then:
research artifact, watermarked, quarantined.
"""
WATERMARK = "SYNTHETIC ECG — RESEARCH ARTIFACT — NOT A MEASUREMENT"
