"""M1.2 — head_afib behaviour lock. The decision now runs as endpoint
head #1 over the BeatLattice; these tests pin its outputs BIT-FOR-BIT
(sanctioned sentence, star grade, outcome, class) to the v0.1.5 values on
the synthetic corpus, so the head refactor can never drift the product."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

cv2 = pytest.importorskip("cv2")

from beats.lattice import BeatLattice, LATTICE_VERSION
from datasets.schema import ScanOutcome
from heads.base import HeadRegistryError
from inference.pipeline import run_with_details

_CONSUMER = {"capture_profile": "consumer", "assume_rig_locks": False}

# v0.1.5 golden values (bit-for-bit; from datasets/schema.py's sanctioned
# text path as shipped at commit 1c1487a)
_SINUS_TEXT = ("No irregular rhythm was detected during this scan. "
               "This scan cannot rule out atrial fibrillation, which often "
               "comes and goes. Confidence: 5 of 5.")
_AF_TEXT = ("We detected an irregular rhythm that can be associated with "
            "atrial fibrillation. This is not a diagnosis. Please share "
            "this result with a clinician, who may recommend an ECG. "
            "Confidence: 5 of 5.")
_DARK_TEXT = ("We could see your pulse, but not clearly enough to "
              "check your rhythm this time. Improve lighting: face a "
              "bright, even light. Confidence: 1 of 5.")


@pytest.fixture(scope="module")
def piped(videos):
    out = {}
    for name in ("sinus30", "af30", "dark30"):
        out[name] = run_with_details(videos[name][0], manifest=_CONSUMER,
                                     recording_id=f"lock-{name}")
    return out


def test_v015_behaviour_locked_bit_for_bit(piped):
    r, _ = piped["sinus30"]
    assert (r.outcome, r.predicted_class, r.confidence_stars) == \
        (ScanOutcome.ACCEPT, "SINUS", 5)
    assert r.user_facing_text() == _SINUS_TEXT
    r, _ = piped["af30"]
    assert (r.outcome, r.predicted_class, r.confidence_stars) == \
        (ScanOutcome.ACCEPT, "AFIB_SUGGESTIVE", 5)
    assert r.user_facing_text() == _AF_TEXT
    r, _ = piped["dark30"]
    assert (r.outcome, r.predicted_class, r.confidence_stars) == \
        (ScanOutcome.NO_RESULT, None, 1)
    assert r.user_facing_text() == _DARK_TEXT


def test_afib_head_result_mirrors_the_scan_result(piped):
    r, det = piped["sinus30"]
    hrs = {h["head"]: h for h in det["head_results"]}
    afib = hrs["afib"]
    assert afib["measurement_class"] == "INFERRED_RHYTHM"
    assert afib["value"]["user_facing_text"] == r.user_facing_text()
    assert afib["value"]["outcome"] == r.outcome.value
    assert afib["value"]["confidence_stars"] == r.confidence_stars


def test_all_enabled_heads_ran_over_the_lattice(piped):
    r, det = piped["sinus30"]
    names = [h["head"] for h in det["head_results"]]
    assert names[0] == "afib"
    assert "rate_flags" in names and "rhythm_map" in names
    lat = det["lattice"]
    assert lat.version == LATTICE_VERSION
    assert lat.n_intervals >= 15 and lat.beat_t_s.size >= 15
    rt = BeatLattice.from_dict(lat.to_dict())      # versioned round-trip
    assert rt.n_intervals == lat.n_intervals
    assert rt.beat_t_s.shape == lat.beat_t_s.shape


def test_heads_override_and_mandatory_afib(videos):
    r, det = run_with_details(videos["sinus30"][0], manifest=_CONSUMER,
                              heads=["afib"], recording_id="ovr")
    assert [h["head"] for h in det["head_results"]] == ["afib"]
    with pytest.raises(HeadRegistryError):
        run_with_details(videos["sinus30"][0], manifest=_CONSUMER,
                         heads=["rate_flags"], recording_id="bad")
