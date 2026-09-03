"""AvatarX v0.8 — the Research & Investigation Report (QUARANTINED).

One document, for investigators only, that carries EVERY gated head's
raw output for one scan — VO2/fitness (§V), atrial flutter (§F),
arterial stiffness (vascular), vascular tone (§W) and rhythm regularity
(§R) — each beside its track's live gate status, under one watermark.

It is NOT the Cardiac Rhythm Scan Report and never becomes part of it:
nothing under research/ may be imported from app/ or inference/ (the
transitive quarantine walker enforces it), the consumer payload filter
(`datasets.schema.public_head_results`) is untouched, and the only way
to produce this document is the explicit `cli.py research-report` verb.
Every track's rendering invariant (V-a, F-b, W-a, §R, invariant 9)
still decides what a PARTICIPANT may see; this document decides
nothing — it records.

Owner instruction (2026-09-01, spec B.24, quoted verbatim): "Provide all
VO2 max, atrial flutter, arterial stiffness, vascular tone,
rhythm-regularity in the report for research and investigation purposes
only."
"""

WATERMARK = ("RESEARCH & INVESTIGATION REPORT — RESEARCH ARTIFACT — "
             "FOR RESEARCH AND INVESTIGATION PURPOSES ONLY — NOT VALIDATED "
             "— NOT A MEASUREMENT — NOT A DIAGNOSIS — NOT FOR PARTICIPANT "
             "OR PATIENT USE")

# (key, title, head name, gate block, gate-status import)
TRACKS = (
    ("rhythm_regularity", "Rhythm regularity (v0.7, §R)", "regularity",
     "regularity", "evaluation.regularity_gates:regularity_gate_status"),
    ("atrial_flutter", "Atrial flutter pattern flag (v0.6, §F)", "flutter",
     "flutter", "evaluation.flutter_gates:flutter_gate_status"),
    ("arterial_stiffness", "Arterial stiffness (v0.4 vascular)", "vascular",
     "vascular", "evaluation.vascular_gates:vascular_gate_status"),
    ("vascular_tone", "Vasomotor reactivity / vascular tone (v0.5, §W)",
     "vasotone", "vasotone", "evaluation.vasotone_gates:vasotone_gate_status"),
    ("cardiorespiratory_fitness", "VO2 / cardiorespiratory fitness (v0.4, §V)", "fitness",
     "vo2", "evaluation.fitness_gates:vo2_gate_status"),
)


class InvestigationError(RuntimeError):
    """Fail-closed error for the investigation report."""
