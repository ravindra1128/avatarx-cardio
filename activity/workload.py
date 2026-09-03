"""
Workload context (v0.4 T3) — from USER-ENTERED measured weight/height +
age/sex + the protocol prescription. NEVER inferred from the face (hard
rule; the escalation clause names it). These numbers are protocol
context for the evaluation harness and heads — none of them is a
user-facing fitness claim.

Formulas (documented approximations, good enough for workload CONTEXT):
  sit-to-stand: vertical centre-of-mass excursion ~= 0.27 x height;
                work/rep = m*g*rise; average positive power over the
                prescribed duration.
  step test:    work/cycle = m*g*step_height; x1.33 for the eccentric
                (down) component per Ainsworth stepping convention.
  march:        no meaningful external work formula — MET-table constant.
Metabolic estimate assumes ~20% mechanical efficiency; 1 MET =
1.162 W/kg metabolic. All of it lands in `est_mets` as CONTEXT.
"""
from __future__ import annotations

G = 9.81
COM_RISE_FRACTION = 0.27         # of standing height, sit->stand
MECH_EFFICIENCY = 0.20
MET_W_PER_KG = 1.162
MARCH_METS = 4.0


def workload_context(challenge, participant_context) -> dict:
    """Challenge + ParticipantContext (or plain dict) -> workload dict.
    Fail-closed: missing required user-entered inputs -> available=False
    with reasons; nothing is guessed and nothing comes from imagery."""
    pc = participant_context
    get = (pc.get if isinstance(pc, dict)
           else lambda k, d=None: getattr(pc, k, d))
    missing = [k for k in challenge.workload_inputs
               if get(k) in (None, "")]
    if missing:
        return {"available": False, "protocol_id": challenge.protocol_id,
                "reasons": [f"user-entered {k} missing" for k in missing]}
    m = float(get("measured_weight_kg"))
    out = {"available": True, "protocol_id": challenge.protocol_id,
           "body_mass_kg": m,
           "cadence_per_min": challenge.cadence_per_min,
           "duration_s": challenge.duration_s,
           "source": "user-entered measured weight/height — never "
                     "inferred from the face"}
    n_reps = challenge.cadence_per_min / 60.0 * challenge.duration_s
    if challenge.protocol_id == "sts_1min":
        rise = COM_RISE_FRACTION * float(get("height_cm")) / 100.0
        work = m * G * rise
        power = work * n_reps / challenge.duration_s
        out.update({"work_per_rep_j": round(work, 1),
                    "avg_power_w": round(power, 1),
                    "est_mets": round(
                        (power / MECH_EFFICIENCY) / (MET_W_PER_KG * m), 2)})
    elif challenge.protocol_id == "step_3min":
        work = m * G * challenge.step_height_m * 1.33
        power = work * n_reps / challenge.duration_s
        out.update({"work_per_rep_j": round(work, 1),
                    "avg_power_w": round(power, 1),
                    "est_mets": round(
                        (power / MECH_EFFICIENCY) / (MET_W_PER_KG * m), 2)})
    else:                                          # march_2min
        out.update({"work_per_rep_j": None, "avg_power_w": None,
                    "est_mets": MARCH_METS})
    return out
