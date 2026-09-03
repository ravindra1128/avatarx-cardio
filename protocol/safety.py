"""
Pre-activity safety screen (v0.4 T3) — PAR-Q+-style. The screen is
deliberately blunt: any positive answer, or ANY unanswered question,
blocks the guided activity (a resting scan stays available) and the only
sentence the user sees is the sanctioned `safety_blocked` entry in
datasets/schema.py. This module never composes free text and never
interprets an answer medically — it routes, it does not diagnose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The screen questions (fixed, audited; ids are the contract —
# challenges name them in contraindication_ids). Yes/no; True = yes.
SCREEN_QUESTIONS = {
    "heart_condition": (
        "Has a health professional ever said you have a heart condition "
        "and that you should only do physical activity they recommend?"),
    "chest_pain_activity": (
        "Do you feel pain in your chest when you do physical activity?"),
    "chest_pain_rest": (
        "In the past month, have you had chest pain when you were not "
        "doing physical activity?"),
    "dizziness_loss_balance": (
        "Do you lose your balance because of dizziness, or do you ever "
        "lose consciousness?"),
    "bone_joint_problem": (
        "Do you have a bone or joint problem that could be made worse by "
        "standing up and sitting down repeatedly?"),
    "bp_heart_meds": (
        "Are you currently taking prescribed medication for blood "
        "pressure or a heart condition?"),
    "other_reason": (
        "Do you know of any other reason why you should not do short, "
        "light physical activity?"),
}

# Mid-activity stop rules (fixed, audited): shown before the activity and
# honored immediately — a stop always ends the session's activity phase.
STOP_RULES = (
    "Stop at once if you feel chest pain, severe shortness of breath, "
    "dizziness, or faintness — and consider speaking with a clinician.",
    "Stop if you feel any joint pain. You can end the activity at any "
    "time; a resting scan is always available.",
)

# bp_heart_meds does NOT block by itself — it routes fitness inference to
# trend-only downstream (rate-limiting medications make HR-based fitness
# categories unreliable). Everything else blocks the activity.
NON_BLOCKING = ("bp_heart_meds",)


@dataclass(frozen=True)
class ScreenResult:
    passed: bool
    blockers: tuple = ()
    unanswered: tuple = ()
    meds_flagged: bool = False


def evaluate_screen(answers: dict) -> ScreenResult:
    """Fail-closed: unknown ids are an error; unanswered questions block."""
    answers = dict(answers or {})
    unknown = set(answers) - set(SCREEN_QUESTIONS)
    if unknown:
        raise ValueError(f"safety screen: unknown answer ids "
                         f"{sorted(unknown)}")
    unanswered = tuple(q for q in SCREEN_QUESTIONS if q not in answers)
    blockers = tuple(q for q, yes in answers.items()
                     if yes and q not in NON_BLOCKING)
    meds = bool(answers.get("bp_heart_meds"))
    return ScreenResult(passed=not blockers and not unanswered,
                        blockers=blockers, unanswered=unanswered,
                        meds_flagged=meds)


def challenge_allowed(screen: ScreenResult, challenge) -> bool:
    """A passed screen can still exclude a specific challenge whose
    contraindication ids were answered yes (relevant when a future screen
    variant makes some answers non-blocking globally)."""
    return screen.passed and not (set(screen.blockers)
                                  & set(challenge.contraindication_ids))
