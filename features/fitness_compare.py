"""Three-way fitness comparison for TEST scans (owner, 2026-10-01).

The card the user sees is not touched here. It stays the resting-heart-rate
formula (features/hemodynamics.resting_rate_index on the scan's resting rate),
with its name and value exactly as computed. On a test scan the client may also
send two profiles in the body of /api/start:

  test_profile    the participant's real details, typed in by the tester:
                  age, sex, height, weight, activity level 1-5, and an id/label
                  that tells repeat scans of one person apart
  shenai_profile  ShenAI's own face-based estimates for the same scan: age,
                  BMI, height, weight. ShenAI estimates no sex and no activity
                  level (its SDK has no such output).

This module puts three results side by side, all on the SAME resting heart rate
the card used, so two columns differ only by method or by where the details
came from:

  formula   the card's value, 0-100, exactly as shown to the user
  manual    the Jurca 2005 non-exercise VO2max (features/vo2max.py, mL/kg/min)
            on the tester's details
  shenai    the same equation on ShenAI's age and BMI. Sex and activity level
            come from the tester's details because ShenAI provides neither, so
            manual vs shenai isolates exactly ShenAI's age and BMI estimates.

Recorded for comparison only: nothing here feeds the card, the rhythm result or
any other output. Pure functions, never raise on malformed input.
"""
from __future__ import annotations

import math
from typing import Optional

from features import vo2max

CARD_KEY = "cardiorespiratory_fitness"


def _num(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


def card_reading(card: Optional[dict]) -> dict:
    """What the card showed, and the resting rate it was computed on."""
    card = card if isinstance(card, dict) else {}
    raw = card.get("raw") if isinstance(card.get("raw"), dict) else {}
    details = card.get("details") if isinstance(card.get("details"), dict) else {}
    computed = card.get("status") == "computed"
    hr = _num(raw.get("value")) if raw.get("name") == "resting_heart_rate" else None
    return {"status": card.get("status"), "score": _num(card.get("value")) if computed else None,
            "unit": card.get("unit"), "hr_bpm": hr if computed else None,
            "hr_source": details.get("resting_rate_source")}


def shenai_inputs(raw) -> dict:
    """ShenAI's estimates as plain numbers. BMI is ShenAI's own when it gave
    one, else derived from its height and weight; never from the tester's."""
    raw = raw if isinstance(raw, dict) else {}
    age, height, weight = (_num(raw.get(k)) for k in ("age_years", "height_cm", "weight_kg"))
    bmi, bmi_source = _num(raw.get("bmi_kg_per_m2")), "shenai"
    if bmi is None and height and weight:
        bmi, bmi_source = weight / (height / 100.0) ** 2, "shenai_height_weight"
    return {"age_years": age, "bmi": bmi, "bmi_source": bmi_source if bmi is not None else None,
            "height_cm": height, "weight_kg": weight}


def _manual_summary(profile: Optional[dict], raw: dict) -> str:
    if profile:
        return (f"{profile['sex']}, {profile['age_years']:.0f} y, {profile['height_cm']:.0f} cm, "
                f"{profile['weight_kg']:.0f} kg, BMI {profile['bmi']:.1f}, "
                f"activity {profile['activity_level']}")
    parts = [f"{k} {raw[k]}" for k in ("sex", "age", "height_cm", "weight_kg", "activity_level")
             if raw.get(k) is not None]
    return ("incomplete: " + ", ".join(parts)) if parts else ""


def _shenai_summary(s: dict) -> str:
    bits = []
    if s["age_years"] is not None:
        bits.append(f"{s['age_years']:.0f} y")
    if s["bmi"] is not None:
        bits.append(f"BMI {s['bmi']:.1f}")
    if s["height_cm"] is not None:
        bits.append(f"{s['height_cm']:.0f} cm")
    if s["weight_kg"] is not None:
        bits.append(f"{s['weight_kg']:.0f} kg")
    return ", ".join(bits)


def _estimate(profile: dict, hr: Optional[float]) -> dict:
    if hr is None:
        return {"available": False, "reason": "the card has no resting heart rate on this scan"}
    est = vo2max.estimate_vo2max(profile, hr)
    if not est.get("available"):
        return {"available": False, "reason": est.get("reason")}
    return {"available": True, "vo2max_ml_kg_min": est["vo2max_ml_kg_min"],
            "likely_range_ml_kg_min": est.get("likely_range_ml_kg_min"),
            "extrapolated": est.get("extrapolated") or [], "inputs": est.get("inputs")}


def compare(card: Optional[dict], manual_raw=None, shenai_raw=None) -> dict:
    """The three readings of one scan and where they differ. `card` is the
    payload row of the fitness card (biomarkers.items[]); it is only read."""
    formula = card_reading(card)
    hr = formula["hr_bpm"]
    out = {"version": 1, "hr_bpm": hr, "formula": formula,
           "equation": {"id": vo2max.JURCA_2005["id"], "citation": vo2max.JURCA_2005["citation"]}}

    # --- manual: the tester's details
    mraw = manual_raw if isinstance(manual_raw, dict) else {}
    profile, problems = (vo2max.normalize_profile(mraw) if mraw else (None, ["no manual profile"]))
    manual = {"participant_id": mraw.get("participant_id"), "label": mraw.get("label"),
              "summary": _manual_summary(profile, mraw), "provided": bool(mraw)}
    if profile is None:
        manual.update(available=False, reason="; ".join(problems))
    else:
        manual.update(_estimate(profile, hr))
    out["manual"] = manual

    # --- shenai: ShenAI's age and BMI; sex and activity from the tester
    s = shenai_inputs(shenai_raw)
    shenai = {"summary": _shenai_summary(s), "provided": isinstance(shenai_raw, dict) and bool(shenai_raw),
              "age_years": _r(s["age_years"]), "bmi": _r(s["bmi"]), "bmi_source": s["bmi_source"],
              "height_cm": _r(s["height_cm"]), "weight_kg": _r(s["weight_kg"])}
    if s["age_years"] is None or s["bmi"] is None:
        shenai.update(available=False, reason="ShenAI returned no age and BMI estimate for this scan")
    elif profile is None:
        shenai.update(available=False, reason=("ShenAI estimates no sex or activity level; those come "
                                               "from the manual profile, and there is none"))
    elif not (vo2max.AGE_BOUNDS[0] <= s["age_years"] <= vo2max.AGE_BOUNDS[1]) or \
            not (vo2max.BMI_BOUNDS[0] <= s["bmi"] <= vo2max.BMI_BOUNDS[1]):
        shenai.update(available=False, reason="ShenAI's age or BMI estimate is outside the equation's input range")
    else:
        shenai.update(_estimate({"age_years": s["age_years"], "sex": profile["sex"], "bmi": s["bmi"],
                                 "activity_level": profile["activity_level"]}, hr))
        shenai["borrowed_from_manual"] = ["sex", "activity_level"]
    out["shenai"] = shenai

    # --- where ShenAI's details and result differ from the tester's
    diff = {}
    if profile is not None:
        for key, mine in (("age_years", profile["age_years"]), ("bmi", profile["bmi"]),
                          ("height_cm", profile["height_cm"]), ("weight_kg", profile["weight_kg"])):
            if s[key] is not None:
                diff[key] = round(s[key] - mine, 1)
    if manual.get("available") and shenai.get("available"):
        diff["vo2max_ml_kg_min"] = round(shenai["vo2max_ml_kg_min"] - manual["vo2max_ml_kg_min"], 1)
    out["differences_shenai_minus_manual"] = diff
    out["note"] = _note(formula, manual, shenai, diff)
    return out


def _note(formula: dict, manual: dict, shenai: dict, diff: dict) -> str:
    def val(x, unit):
        return f"{x:.1f}{unit}" if isinstance(x, (int, float)) else "-"
    head = (f"formula {val(formula['score'], '/100')} | manual "
            f"{val(manual.get('vo2max_ml_kg_min'), ' mL/kg/min')} | ShenAI "
            f"{val(shenai.get('vo2max_ml_kg_min'), ' mL/kg/min')}")
    if formula["hr_bpm"] is not None:
        head += f" | all on {formula['hr_bpm']:.0f} bpm"
    labels = (("age_years", "age", " y"), ("bmi", "BMI", ""), ("height_cm", "height", " cm"),
              ("weight_kg", "weight", " kg"), ("vo2max_ml_kg_min", "VO2", " mL/kg/min"))
    parts = [f"{name} {diff[k]:+.1f}{unit}" for k, name, unit in labels if k in diff]
    tail = ("ShenAI minus manual: " + ", ".join(parts)) if parts else ""
    why = []
    if not manual.get("available") and manual.get("reason"):
        why.append(f"manual: {manual['reason']}")
    if not shenai.get("available") and shenai.get("reason"):
        why.append(f"ShenAI: {shenai['reason']}")
    return " | ".join(x for x in [head, tail] + why if x)
