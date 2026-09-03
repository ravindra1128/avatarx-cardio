"""AvatarX v0.4 arterial-stiffness research track (QUARANTINED).

Pre-registered question, two parts: (V0) do stiffness-relevant pulse-
morphology features survive the facial-video path with clinical fidelity
against a simultaneous contact-PPG reference, and (V1) does a model on
the surviving features carry information about measured carotid-femoral
PWV beyond an age+sex+brachial-BP regression? Until every gate of the
`vascular:` block of configs/gates.yaml is green with owner + clinical
signoff recorded in the spec changelog, nothing in this package renders
anywhere a participant can see (invariants V-a/V-b): no number, score,
trend, or color — and never the strings "vascular age", "artery age",
"PWV" or any m/s value on a user-facing surface.

Nothing under research/ may be imported from app/ or inference/ — the
transitive quarantine walker in tests enforces it. The one sanctioned
bridge out is models/registry.promote, which reads gate_results.json
(data, never research code) and refuses while any gate is red.
"""

WATERMARK = ("VASCULAR MORPHOLOGY — RESEARCH ARTIFACT — "
             "NOT A MEASUREMENT")


class VascularError(RuntimeError):
    """Fail-closed error for the vascular research track."""
