"""The AFib rhythm result reaches the tracking sheet as the participant sees it:
the assigned class and the exact sanctioned sentence, beside the Outcome and
Stars that were always there."""
from app.result_sheet import COLUMNS, row_from_doc


def test_rhythm_columns_follow_rate_guard():
    i = COLUMNS.index("Rate Guard")
    assert COLUMNS[i + 1:i + 3] == ["Rhythm Class", "Rhythm Text"]


def test_row_carries_class_and_verbatim_text():
    text = ("We detected an irregular rhythm that can be associated with atrial "
            "fibrillation. This is not a diagnosis. Please share this result with "
            "a clinician, who may recommend an ECG. Confidence: 3 of 5.")
    row = row_from_doc({"outcome": "ACCEPT", "predicted_class": "AFIB_SUGGESTIVE",
                        "confidence_stars": 3, "user_facing_text": text})
    assert row["Outcome"] == "ACCEPT" and row["Stars"] == 3
    assert row["Rhythm Class"] == "AFIB_SUGGESTIVE"
    assert row["Rhythm Text"] == text


def test_abstained_scan_has_blank_class_but_keeps_its_sentence():
    row = row_from_doc({"outcome": "REPEAT_SCAN", "predicted_class": None,
                        "user_facing_text": "We could see your pulse, but not clearly enough."})
    assert row["Rhythm Class"] == ""
    assert row["Rhythm Text"].startswith("We could see your pulse")


def test_classifier_inputs_reach_the_sheet():
    row = row_from_doc({"outcome": "REPEAT_SCAN",
                        "debug": {"rationale": {"features": {
                            "median_abs_succ_diff": 67.4, "pnn50": 0.6087, "n_intervals": 23}}}})
    assert row["MAD ms"] == 67.4 and row["pNN50"] == 0.609 and row["Rate N"] == 23
    i = COLUMNS.index("MAD ms")
    assert COLUMNS[i:i + 3] == ["MAD ms", "pNN50", "Rate N"]   # later columns follow (route, AFib result)


def test_classifier_inputs_blank_when_no_rationale():
    row = row_from_doc({"outcome": "NO_RESULT"})
    assert row["MAD ms"] == "" and row["pNN50"] == "" and row["Rate N"] == ""
