"""
Standardized activity challenges (v0.4 T3). A challenge is a fixed
prescription — cadence, duration, workload-formula inputs and the safety
contraindications that exclude it. Fitness information lives in the HR
response to a KNOWN workload; an unstandardized activity carries none,
which is why compliance is graded and non-compliance fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass


class ChallengeError(ValueError):
    pass


@dataclass(frozen=True)
class Challenge:
    protocol_id: str
    display_name: str
    cadence_per_min: float          # metronome prescription
    duration_s: float
    rep_unit: str                   # what one rep is
    workload_inputs: tuple          # ParticipantContext fields consumed
    contraindication_ids: tuple     # safety-screen answers that exclude it
    step_height_m: float = 0.0      # step_3min only


CHALLENGES = {
    "sts_1min": Challenge(
        protocol_id="sts_1min",
        display_name="1-minute sit-to-stand",
        cadence_per_min=20.0, duration_s=60.0, rep_unit="stand",
        workload_inputs=("measured_weight_kg", "height_cm", "age", "sex"),
        contraindication_ids=("bone_joint_problem", "chest_pain_activity",
                              "chest_pain_rest", "dizziness_loss_balance"),
    ),
    "step_3min": Challenge(
        protocol_id="step_3min",
        display_name="3-minute step test",
        cadence_per_min=24.0, duration_s=180.0, rep_unit="step_cycle",
        workload_inputs=("measured_weight_kg", "height_cm", "age", "sex"),
        contraindication_ids=("bone_joint_problem", "chest_pain_activity",
                              "chest_pain_rest", "dizziness_loss_balance"),
        step_height_m=0.20,
    ),
    "march_2min": Challenge(
        protocol_id="march_2min",
        display_name="2-minute march in place",
        cadence_per_min=76.0, duration_s=120.0, rep_unit="knee_raise",
        workload_inputs=("measured_weight_kg", "age", "sex"),
        contraindication_ids=("chest_pain_activity", "chest_pain_rest",
                              "dizziness_loss_balance"),
    ),
}
DEFAULT_CHALLENGE = "sts_1min"


def get_challenge(protocol_id: str) -> Challenge:
    try:
        return CHALLENGES[protocol_id]
    except KeyError:
        raise ChallengeError(
            f"unknown challenge protocol {protocol_id!r}; available: "
            f"{sorted(CHALLENGES)}") from None
