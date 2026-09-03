"""M1.6 — cli.py process --heads: head selection from the command line;
head results ride on the v2 ScanResult JSON."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess

import pytest

cv2 = pytest.importorskip("cv2")

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _cli(*args):
    r = subprocess.run([sys.executable, str(_ROOT / "cli.py"), *args],
                       capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout, r.stderr


def test_default_heads_ride_on_the_scanresult_json(videos):
    code, out, err = _cli("process", videos["sinus30"][0])
    assert code == 0, err
    doc = json.loads(out)
    assert doc["schema_version"] == 2
    names = [h["head"] for h in doc["head_results"]]
    assert names == ["afib", "rate_flags", "rhythm_map"]
    assert doc["capture_meta"]["tracker"]
    rm = doc["head_results"][2]
    assert rm["value"]["svg"].startswith("<svg")


def test_heads_selection_and_mandatory_afib(videos):
    code, out, err = _cli("process", videos["sinus30"][0],
                          "--heads", "afib,rate_flags")
    assert code == 0, err
    doc = json.loads(out)
    assert [h["head"] for h in doc["head_results"]] == ["afib", "rate_flags"]
    code2, out2, err2 = _cli("process", videos["sinus30"][0],
                             "--heads", "rate_flags")
    assert code2 == 2                      # afib cannot be disabled
    assert "afib" in err2
