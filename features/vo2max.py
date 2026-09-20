"""Non-exercise VO2max estimate for the fitness card (2026-09-20).

WHY THE CARD CHANGED ESTIMATOR. Through 2026-09-20 the card was
100 * logistic((72 - HR) / 14): a function of ONE number, the scan's resting
pulse, with a gain of ~1.8 card points per bpm. Measured on the production
tracking sheet (67 scans, repeat pairs of one device <= 20 min apart):

  * 41 % of repeat pairs agreed within +/-8 points; mean |difference| 20.5,
    within-subject SD 20 - as large as the score's whole population spread;
  * fed the REFERENCE heart rate instead of ours (a perfect rate chain) the
    same formula still reached only 54 %: a seated adult's pulse really moves
    ~5 bpm between scans minutes apart (max 24 bpm on the sheet), and the
    formula turned every one of those beats into "fitness".

Oxygen uptake capacity is a TRAIT: it does not change in twenty minutes, so an
estimator that does is measuring the wrong thing. No published method derives
VO2max from a resting pulse alone. What IS published and cross-validated
against maximal treadmill tests is the non-exercise regression family - age,
sex, body composition, habitual activity and resting heart rate - in which the
resting rate carries a SMALL coefficient (about -0.1 mL/kg/min per bpm). That
is the validated partial effect of the one quantity the camera measures, and
it is why the estimate is repeatable: the inputs that dominate it are facts
about the person that do not move between scans. Nothing here smooths, caches
or clips a value toward a previous one; two scans agree because the person is
the same person.

WHAT THIS IS, in the vocabulary the owner's brief asks for:
  direct measurement      no  (that is CPET with gas analysis)
  validated estimation    yes (a published equation, SEE ~5-7 mL/kg/min)
  camera contribution     the resting heart rate only; every other input is
                          USER-ENTERED and is reported as such, never inferred
                          from the face (v0.4 hard rule, datasets/schema.py
                          ParticipantContext).

WHAT IT IS NOT, from the same literature (read 2026-09-20):
  * No peer-reviewed study validates VO2max from face video, at rest or
    otherwise; no equation here was validated with a camera-measured pulse.
  * An individual's error is about +/-5 (1 SD), +/-10-13 at 95 %. Across 28
    such equations only ~52 % of people land in the right fitness tertile
    (Peterman 2021, Eur J Prev Cardiol 28:142), so the card shows a number
    with its band and never a fitness category.
  * It cannot track training: the best equations called the DIRECTION of a
    measured change correctly in ~56 % of people (Peterman 2020, JAHA
    9:e015117). Two scans agreeing is repeatability, not sensitivity.
  * Rejected on the evidence: 15.3 x HRmax/HRrest (Uth 2004) - five times more
    sensitive to the resting rate, limits of agreement about +/-17, ~22 % high
    in middle-aged men; the HUNT equations (Nes 2011) - need a waist measure
    and read ~6 mL/kg/min high outside Norway; any resting-HRV term - camera
    RMSSD tracks contact PPG at r ~0.5 and adds nothing beyond age, sex and
    rate; resting-HR-only maps like the one this replaces - no published basis.

Pure functions, no I/O, never raise on malformed input.
"""
from __future__ import annotations

import math
from typing import Optional

ML_PER_MET = 3.5                      # 1 MET = 3.5 mL O2 / kg / min

# ---------------------------------------------------------------- the equation
# Jurca R, Jackson AS, LaMonte MJ, et al. Assessing cardiorespiratory fitness
# without performing exercise testing. Am J Prev Med 2005;29(3):185-193.
# doi:10.1016/j.amepre.2005.06.004. Adults aged 20-70 in three cohorts:
# NASA/JSC (n=1,863; VO2max MEASURED by gas analysis on a maximal Bruce
# treadmill test), ACLS (n=46,190; estimated from the final speed and grade of
# a maximal Balke test) and ADNFS (n=1,706; extrapolated from a submaximal
# test). Predominantly non-Hispanic white, well-educated cohorts - a stated
# limit on who the line was fitted to. Multiple R 0.81 / 0.77 /
# 0.76, SEE 1.45 / 1.50 / 1.97 METs; applied to the other two cohorts the NASA
# model kept R 0.76 / 0.75 and read low by 0.67 / 1.37 METs. These are the NASA
# coefficients - Table 5, and the worksheet the authors publish as Figure 1:
#   METs = 18.07 + 2.77*(sex: male=1, female=0) - 0.10*age - 0.17*BMI
#          - 0.03*resting HR + physical-activity score
# Resting HR there was read from the ECG after 5 minutes supine. A seated
# 50-second scan reads a few bpm higher, which at 0.03 METs per bpm costs
# about half a mL/kg/min - hence the form's "sit quietly first".
# Independent check (Peterman 2020, BALL ST, n=808, measured VO2max): this
# model 34.4 +/- 8.1 against 34.9 +/- 9.3 measured, not significantly
# different; R2 0.66, SEE 4.7 mL/kg/min, ICC 0.80 - the highest of the 27
# equations tested. The paper's own worked example (9.02 METs) is not what its
# coefficients give (9.46): two-decimal rounding leaves up to ~0.5 METs of
# systematic offset in any one person - an accuracy caveat, not a repeatability
# one, since it is the same for every scan of that person.
JURCA_2005 = {
    "id": "jurca2005",
    "citation": ("Jurca R, Jackson AS, LaMonte MJ, et al. Assessing cardiorespiratory "
                 "fitness without performing exercise testing. Am J Prev Med "
                 "2005;29(3):185-193"),
    "doi": "10.1016/j.amepre.2005.06.004",
    "intercept_mets": 18.07,
    "male_mets": 2.77,
    "age_mets_per_year": -0.10,
    "bmi_mets_per_unit": -0.17,
    "resting_hr_mets_per_bpm": -0.03,
    # SEE across the three cohorts, in METs. The card's +/- band uses the
    # middle one (ACLS, 1.50): the derivation cohort's own 1.45 is the
    # optimistic end for anyone who is not a NASA employee.
    "see_mets_range": (1.45, 1.97),
    "see_mets": 1.50,
    "multiple_r_range": (0.76, 0.81),
    # What the derivation cohorts covered. Outside it the line is an
    # extrapolation and the card says so; it is never silently clipped.
    "validated": {"age": (20.0, 70.0), "bmi": (17.0, 40.0), "resting_hr": (40.0, 105.0)},
}

# The five self-report activity levels and their scores, worded as on the
# paper's own worksheet (Figure 1): the wording IS the instrument.
_AEROBIC = ("Participate in aerobic exercises such as brisk walking, jogging or running, "
            "cycling, swimming, or vigorous sports at a comfortable pace, or other "
            "activities requiring similar levels of exertion, for ")
ACTIVITY_LEVELS = {
    1: {"score_mets": 0.00,
        "label": "Inactive or little activity other than usual daily activities"},
    2: {"score_mets": 0.32,
        "label": ("Regularly (5 or more days a week) participate in physical activities "
                  "requiring low levels of exertion that result in slight increases in "
                  "breathing and heart rate for at least 10 minutes at a time")},
    3: {"score_mets": 1.06, "label": _AEROBIC + "20 to 60 minutes per week"},
    4: {"score_mets": 1.76, "label": _AEROBIC + "1 to 3 hours per week"},
    5: {"score_mets": 3.03, "label": _AEROBIC + "over 3 hours per week"},
}

# Where measured VO2max sits in adults of the same sex and age decade: the 25th,
# 50th and 75th percentiles of treadmill CPET with gas analysis in the FRIEND
# registry (Kaminsky LA, Arena R, Myers J. Mayo Clin Proc 2015;90(11):1515-23,
# Table 3; n=7,783 tests, 4,611 men and 3,172 women aged 20-79, eight US
# laboratories). It is the card's "typical range": context for reading the
# estimate, never an input to it. A US registry - a stated limit.
FRIEND_2015 = {
    "citation": ("Kaminsky LA, Arena R, Myers J. Reference standards for cardiorespiratory "
                 "fitness measured with cardiopulmonary exercise testing: data from the "
                 "FRIEND registry. Mayo Clin Proc 2015;90(11):1515-1523"),
    "male": {20: (40.1, 48.0, 55.2), 30: (35.9, 42.4, 49.2), 40: (31.9, 37.8, 45.0),
             50: (27.1, 32.6, 39.7), 60: (23.7, 28.2, 34.5), 70: (20.4, 24.4, 30.4)},
    "female": {20: (30.5, 37.6, 44.7), 30: (25.3, 30.2, 36.1), 40: (22.1, 26.7, 32.4),
               50: (19.9, 23.4, 27.6), 60: (17.2, 20.0, 23.8), 70: (15.6, 18.3, 20.8)},
}


def reference_percentiles(age_years, sex) -> Optional[dict]:
    """The FRIEND quartiles for this sex and age decade, or None outside 20-79."""
    age = _f(age_years)
    table = FRIEND_2015.get(str(sex))
    if age is None or table is None or not 20.0 <= age < 80.0:
        return None
    decade = int(age // 10) * 10
    p25, p50, p75 = table[decade]
    return {"p25": p25, "p50": p50, "p75": p75, "age_band": f"{decade}-{decade + 9}",
            "sex": str(sex), "source": "FRIEND 2015, treadmill CPET",
            "citation": FRIEND_2015["citation"]}


# Input plausibility (refuse outside) - wider than the validated ranges (flag).
AGE_BOUNDS = (18.0, 95.0)
HEIGHT_CM_BOUNDS = (120.0, 230.0)
WEIGHT_KG_BOUNDS = (30.0, 300.0)
BMI_BOUNDS = (12.0, 70.0)
RESTING_HR_BOUNDS = (30.0, 200.0)
# An estimate outside this is not a fitness statement about a living adult, it
# is the line leaving its domain: the card abstains and says which input did it.
PLAUSIBLE_VO2MAX = (8.0, 90.0)
# Two rates "agree" within the tolerance the pulse cross-check already uses
# (features.hemodynamics.PULSE_AGREEMENT_TOL, inference.shenai_route.RATE_TOL).
RATE_AGREEMENT_TOL = 0.15

_SEX = {"male": "male", "m": "male", "man": "male",
        "female": "female", "f": "female", "woman": "female"}


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _within(x, bounds) -> bool:
    return x is not None and bounds[0] <= x <= bounds[1]


def normalize_profile(raw) -> tuple:
    """(profile | None, problems). The profile is what the equation needs, in
    its own units, from the user-entered participant block the client sends:
    age|age_years, sex, height_cm, weight_kg|measured_weight_kg, activity_level.
    None when anything required is missing or implausible - each problem is one
    short phrase naming the field, never echoing the value back."""
    problems: list = []
    if not isinstance(raw, dict) or not raw:
        return None, ["no profile was provided"]
    age = _f(raw.get("age_years") if raw.get("age_years") is not None else raw.get("age"))
    if not _within(age, AGE_BOUNDS):
        problems.append("age missing or outside 18-95 years")
    sex = _SEX.get(str(raw.get("sex") or "").strip().lower())
    if sex is None:
        problems.append("sex missing (the equation has a male/female term)")
    height = _f(raw.get("height_cm"))
    if not _within(height, HEIGHT_CM_BOUNDS):
        problems.append("height missing or outside 120-230 cm")
    weight = _f(raw.get("measured_weight_kg") if raw.get("measured_weight_kg") is not None
                else raw.get("weight_kg"))
    if not _within(weight, WEIGHT_KG_BOUNDS):
        problems.append("weight missing or outside 30-300 kg")
    level = _f(raw.get("activity_level"))
    if level is None or int(level) != level or int(level) not in ACTIVITY_LEVELS:
        problems.append("activity level missing (1-5)")
    bmi = None
    if _within(height, HEIGHT_CM_BOUNDS) and _within(weight, WEIGHT_KG_BOUNDS):
        bmi = weight / (height / 100.0) ** 2
        if not _within(bmi, BMI_BOUNDS):
            problems.append("height and weight give a body-mass index outside 12-70")
    if problems:
        return None, problems
    return {"age_years": float(age), "sex": sex, "height_cm": float(height),
            "weight_kg": float(weight), "bmi": round(float(bmi), 2),
            "activity_level": int(level)}, []


def estimate_vo2max(profile: dict, resting_hr_bpm, equation: dict = JURCA_2005) -> dict:
    """The equation, its terms, and how far the inputs sit from what it was
    validated on. `profile` is normalize_profile's output."""
    out = {"available": False, "equation": equation["id"], "citation": equation["citation"],
           "doi": equation["doi"], "reason": None}
    hr = _f(resting_hr_bpm)
    if not isinstance(profile, dict) or not _within(hr, RESTING_HR_BOUNDS):
        out["reason"] = ("no plausible resting heart rate for the equation"
                         if isinstance(profile, dict) else "no profile")
        return out
    act = ACTIVITY_LEVELS[int(profile["activity_level"])]
    terms_mets = {
        "intercept": equation["intercept_mets"],
        "sex": equation["male_mets"] if profile["sex"] == "male" else 0.0,
        "age": equation["age_mets_per_year"] * profile["age_years"],
        "bmi": equation["bmi_mets_per_unit"] * profile["bmi"],
        "resting_hr": equation["resting_hr_mets_per_bpm"] * hr,
        "activity": act["score_mets"],
    }
    mets = float(sum(terms_mets.values()))
    vo2 = mets * ML_PER_MET
    extrapolated = [name for name, val in (("age", profile["age_years"]), ("bmi", profile["bmi"]),
                                           ("resting_hr", hr))
                    if not _within(val, equation["validated"][name])]
    if not _within(vo2, PLAUSIBLE_VO2MAX):
        out["reason"] = ("the equation leaves its physiological range for these inputs ("
                         + (", ".join(extrapolated) or "combination") + " outside what it was "
                         "validated on); no estimate is reported")
        out["extrapolated"] = extrapolated
        return out
    see = equation["see_mets"] * ML_PER_MET
    out.update({
        "available": True,
        "vo2max_ml_kg_min": round(vo2, 1),
        "mets": round(mets, 2),
        "see_ml_kg_min": round(see, 1),
        "see_ml_kg_min_range": [round(s * ML_PER_MET, 1) for s in equation["see_mets_range"]],
        "likely_range_ml_kg_min": [round(vo2 - see, 1), round(vo2 + see, 1)],
        "resting_hr_bpm": round(hr, 1),
        "resting_hr_sensitivity_ml_kg_min_per_bpm": round(
            equation["resting_hr_mets_per_bpm"] * ML_PER_MET, 3),
        "terms_ml_kg_min": {k: round(v * ML_PER_MET, 2) for k, v in terms_mets.items()},
        "inputs": {"age_years": profile["age_years"], "sex": profile["sex"],
                   "bmi": profile["bmi"], "activity_level": profile["activity_level"],
                   "resting_hr_bpm": round(hr, 1)},
        "input_sources": {"age_years": "user_entered", "sex": "user_entered",
                          "bmi": "user_entered_height_weight",
                          "activity_level": "user_entered", "resting_hr_bpm": "face_scan"},
        "extrapolated": extrapolated,
        "measurement_class": "validated_nonexercise_estimation",
        "reference": reference_percentiles(profile["age_years"], profile["sex"]),
    })
    return out


def select_resting_rate(own_bpm, own_source, reference_bpm, reference_source=None) -> dict:
    """ONE resting rate for the estimate, with its provenance.

    `own_bpm` is the rate the recorded-clip pipeline resolved (None when it
    abstained); `reference_bpm` is the live-frame rate of the SAME scan when
    the client has one (the SDK's heart rate, or its beat train once the
    ShenAI route has judged it sound). Both are readings of this scan's
    signal; they differ in what the codec did to the frames in between.

      live-frame rate present -> it is the rate; the clip's rate corroborates
                                 it (within 15 %) or is recorded as disagreeing
      clip rate only          -> the clip's rate (the standalone scan, an older
                                 client)
      neither                 -> no estimate

    Why the live-frame rate leads (measured 2026-09-20, tracking sheet, repeat
    scans of one phone <= 20 min apart): within-subject SD of the SDK's rate
    5.3 bpm (production) / 4.0 (staging) against 17.1 / 10.7 for ours, and our
    rate more than 15 bpm from the SDK's on 12 of 43 production scans with our
    own failure signatures (folds to 45-48 bpm, split counts over 100). No
    contact device exists here to referee; this ranks two camera readings by
    their measured stability. It is also the heart rate the results page
    already shows, so the card never quotes a second, different pulse.
    The equation moves ~0.1 mL/kg/min per bpm, so the choice is worth about a
    unit; it decides whether a folded rate reaches the estimate at all."""
    own = _f(own_bpm)
    own = round(own, 1) if _within(own, RESTING_HR_BOUNDS) else None
    ref = _f(reference_bpm)
    ref = round(ref, 1) if _within(ref, RESTING_HR_BOUNDS) else None
    ref_name = str(reference_source or "client_reference")
    out = {"bpm": None, "source": None, "corroborated": None, "reason": None,
           "own_bpm": own, "own_source": own_source if own is not None else None,
           "reference_bpm": ref, "reference_source": ref_name if ref is not None else None,
           "disagreement": None}
    if ref is not None:
        out.update(bpm=ref, source=f"live_frame:{ref_name}", corroborated=False)
        if own is None:
            out["reason"] = (f"the recorded clip yielded no verified resting rate; the rate "
                             f"({ref:.0f} bpm) is the live-frame measurement of the same scan")
        else:
            rel = abs(own - ref) / ref
            out["disagreement"] = round(rel, 4)
            if rel <= RATE_AGREEMENT_TOL:
                out["corroborated"] = True
            else:
                out["reason"] = (f"the recorded clip read {own:.0f} bpm and the live-frame "
                                 f"measurement of the same scan {ref:.0f} bpm; the live-frame "
                                 f"rate is used")
    elif own is not None:
        out.update(bpm=own, source=own_source)
    return out
