"""The ShenAI beat train as a second interval source (inference/shenai_route.py,
owner decision 2026-09-16). Every test here is about what the route may NOT
do: bypass a scan gate, invent a rate, accept a smoothed train, change the
never-diagnose text, or run when the video path already answered.
"""
import json
import math

import numpy as np
import pytest

from configs import load_config
from datasets.schema import ScanOutcome, ScanResult
from inference import shenai_route as sr

CFG = load_config()


# ------------------------------------------------------------- fixtures
def _sidecar(rr_ms, *, drop_at=None, quality=0.9, bad_s=0.0, lnrmssd=None,
             hr=None, t0=1.5):
    """A sidecar whose train is contiguous (end[i] == start[i+1]) except
    where `drop_at` skips one beat. RR intervals in ms."""
    beats, t = [], t0
    for i, rr in enumerate(rr_ms):
        if drop_at is not None and i == drop_at:
            t += rr / 1000.0                  # the SDK lost this beat
            continue
        beats.append({"start_location_sec": round(t, 3),
                      "end_location_sec": round(t + rr / 1000.0, 3),
                      "duration_ms": float(rr)})
        t += rr / 1000.0
    rr = np.asarray(rr_ms, float)
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    ref = {"heart_rate_bpm": hr if hr is not None else 60000.0 / float(np.median(rr)),
           "hrv_sdnn_ms": float(np.std(rr)),
           "hrv_lnrmssd_ms": (math.log(rmssd) if lnrmssd is None else lnrmssd),
           "average_signal_quality": quality, "bad_signal_seconds": bad_s}
    return {"schema_version": 1, "ppg": {"n": 1500, "fs_hz": 30.0,
                                         "fs_source": "derived_from_beats"},
            "heartbeats": beats, "reference": ref}


def _regular(n=60, bpm=72.0, jitter_ms=12.0, seed=1):
    rng = np.random.default_rng(seed)
    return (60000.0 / bpm + rng.normal(0.0, jitter_ms, n)).tolist()


def _afib(n=60, seed=2):
    rng = np.random.default_rng(seed)
    return rng.uniform(450.0, 1150.0, n).tolist()


def _video(outcome="REPEAT_SCAN", gates_failed=("coverage", "n_intervals_min",
                                                "coverage_any_class",
                                                "n_intervals_any_class"),
           *, coherence=0.36, tp=20.0, matched=0.8, spectral=72.0, roi_agree=3,
           lattice=72.0, n_int=8, stars=3, sqi=0.45):
    """A finished video-path doc + det as measure_video_details returns them,
    abstaining on the gates named."""
    all_gates = ["sqi", "coverage", "n_intervals_min", "cross_roi_coherence",
                 "coverage_any_class", "n_intervals_any_class",
                 "split_fraction_any_class", "timing_precision_any_class"]
    gates = [{"name": g, "value": 0.0, "threshold": 0.0, "op": ">=",
              "pass": g not in gates_failed} for g in all_gates]
    ev = {"cross_roi_coherence": coherence, "timing_precision_ms": tp,
          "timing_matched_fraction": matched, "timing_pair": "cheek_l-nose",
          "split_fraction": 0.1, "harmonic_fraction": 0.05, "mean_roi_agreement": 0.7,
          "frac_multi_roi": 0.6, "n_beats": 12, "n_intervals": n_int,
          "pulse_lattice_bpm": lattice, "pulse_lattice_n_intervals": n_int,
          "pulse_spectral_bpm": spectral, "pulse_spectral_roi_agree": roi_agree,
          "pulse_spectral_snr": 3.0,
          "pulse_agreement": (abs(lattice - spectral) / spectral
                              if lattice is not None else None),
          "dropout_rate": 0.3}
    rationale = {"gates": gates, "gates_failed": list(gates_failed),
                 "confidence": {"stars": stars, "score": 0.6,
                                "limiting_factor": "signal_snr", "per_check": {}},
                 "sqi_components": {"cross_roi_coherence": coherence,
                                    "snr_in_band": 0.3},
                 "features": {"n_intervals": n_int}}
    res = ScanResult(recording_id="t", outcome=ScanOutcome(outcome),
                     predicted_class=None, signal_quality_index=sqi,
                     no_read_reasons=["only 8 clean intervals"],
                     confidence_stars=stars, confidence_limiting_factor="signal_snr")
    doc = {"recording_id": "t", "outcome": outcome, "predicted_class": None,
           "signal_quality_index": sqi, "confidence_stars": stars,
           "confidence_limiting_factor": "signal_snr", "mean_pulse_rate_bpm": None,
           "no_read_reasons": list(res.no_read_reasons),
           "user_facing_text": res.user_facing_text(),
           "usable_beats": 12, "analysed_seconds": 6.0,
           "debug": {"rationale": rationale, "evidence": ev}}

    class _Sqi:
        pass
    sq = _Sqi()
    sq.sqi = sqi
    sq.components = dict(rationale["sqi_components"])
    det = {"config": CFG, "evidence": ev, "rationale": rationale, "sqi": sq,
           "min_conf": 0.5}
    return doc, det


# ------------------------------------------------------------- the train
def test_a_dropped_beat_is_a_segment_break_never_spanned():
    t = sr.train_from_sidecar(_sidecar(_regular(30), drop_at=12))
    assert len(t["beats"]) == 29 and t["n_dropped"] == 1
    assert t["segments"][:12] == [0] * 12 and t["segments"][12:] == [1] * 17
    assert t["sdk_rmssd_ms"] == pytest.approx(
        math.exp(_sidecar(_regular(30))["reference"]["hrv_lnrmssd_ms"]), rel=1e-6)


def test_a_malformed_sidecar_yields_no_train():
    assert sr.train_from_sidecar({"heartbeats": "nope"})["beats"] == []
    assert sr.train_from_sidecar(None)["beats"] == []
    assert sr.train_from_sidecar({"heartbeats": [{"start_location_sec": "x"}]})["beats"] == []


# ------------------------------------------------------- preconditions
def test_a_scan_gate_failure_keeps_the_route_out():
    doc, det = _video(gates_failed=("cross_roi_coherence", "coverage_any_class"))
    before = dict(doc)
    rec = sr.evaluate(doc, det, _sidecar(_regular()))
    assert rec["attempted"] is False and rec["used"] is False
    assert "did not support a rhythm statement" in rec["reason"]
    assert "cross_roi_coherence" in rec["reason"]
    assert doc == before


def test_an_accepted_video_path_is_left_alone():
    doc, det = _video(outcome="ACCEPT", gates_failed=())
    doc["predicted_class"] = "SINUS"
    rec = sr.evaluate(doc, det, _sidecar(_regular()))
    assert rec["used"] is False and "not needed" in rec["reason"]
    assert doc["predicted_class"] == "SINUS" and doc["outcome"] == "ACCEPT"


def test_no_sidecar_means_no_change_and_says_so():
    doc, det = _video()
    before = dict(doc)
    rec = sr.evaluate(doc, det, None, waited_s=8.0)
    assert rec["attempted"] is False and "no ShenAI sidecar" in rec["reason"]
    assert "8.0 s" in rec["reason"]
    assert doc == before


# ------------------------------------------------------------ the route
def test_a_regular_train_turns_an_interval_abstention_into_sinus():
    doc, det = _video()
    rec = sr.evaluate(doc, det, _sidecar(_regular()))
    assert rec["used"] is True, rec
    assert doc["outcome"] == "ACCEPT" and doc["predicted_class"] == "SINUS"
    assert doc["rhythm_source"] == "shenai_train"
    assert rec["train"]["clean_intervals"] >= 50 and rec["route_coverage"] > 0.9
    # the same never-diagnose sentence the schema writes for this outcome
    same = ScanResult(recording_id="t", outcome=ScanOutcome.ACCEPT,
                      predicted_class="SINUS", confidence_stars=3,
                      confidence_limiting_factor="signal_snr")
    assert doc["user_facing_text"] == same.user_facing_text()
    # the video path's own answer is kept beside it
    assert doc["debug"]["video_outcome"]["outcome"] == "REPEAT_SCAN"
    assert doc["debug"]["video_rationale"]["gates_failed"] == det["rationale"]["gates_failed"]
    assert doc["debug"]["rationale"]["rhythm_source"] == "shenai_train"
    # the pulse the result carries is the resolver's, from the train's rate
    assert doc["mean_pulse_rate_bpm"] == pytest.approx(72.0, abs=3.0)


def test_the_noise_floor_treats_the_train_as_one_unaveraged_detector():
    doc, det = _video(tp=30.0)
    sr.evaluate(doc, det, _sidecar(_regular()))
    fl = doc["debug"]["rationale"]["rule"]["noise_floor"]
    assert fl["k"] == 1.0                       # mean_roi_agreement 0.25 -> k = 1
    assert fl["mad_floor_ms"] == pytest.approx(0.6745 * math.sqrt(6) * 30.0 / 0.954, rel=1e-3)


def test_an_uncorroborated_rate_blocks_the_route():
    # Our waveform says 50 with 3 regions and our count is a doubling case
    # the resolver refuses: the 72 bpm train has nothing on our side.
    doc, det = _video(spectral=50.0, roi_agree=3, lattice=100.0)
    before = dict(doc)
    rec = sr.evaluate(doc, det, _sidecar(_regular()))
    assert rec["attempted"] is True and rec["used"] is False
    assert "not corroborated" in rec["reason"]
    assert doc == before


def test_our_waveform_rhythm_alone_can_corroborate():
    # No usable count on our side (2 intervals), but the waveform agrees.
    doc, det = _video(lattice=None, n_int=2, spectral=71.0, roi_agree=2)
    rec = sr.evaluate(doc, det, _sidecar(_regular()))
    assert rec["used"] is True and "waveform rhythm 71 bpm" in rec["reason"]


def test_a_regularised_train_is_refused():
    # The SDK's own lnRMSSD says 45 ms; the shipped train reads ~17 ms.
    doc, det = _video()
    rec = sr.evaluate(doc, det, _sidecar(_regular(), lnrmssd=math.log(45.0)))
    assert rec["used"] is False and "regularised or truncated" in rec["reason"]


def test_too_short_or_poor_trains_are_refused():
    doc, det = _video()
    rec = sr.evaluate(doc, det, _sidecar(_regular(10)))
    assert rec["used"] is False and "only 10 beats" in rec["reason"]
    rec = sr.evaluate(doc, det, _sidecar(_regular(), quality=0.3))
    assert rec["used"] is False and "signal quality 0.30" in rec["reason"]
    rec = sr.evaluate(doc, det, _sidecar(_regular(), bad_s=12.0))
    assert rec["used"] is False and "bad signal" in rec["reason"]


def test_an_irregular_train_faces_the_same_af_gates():
    # Below 3 stars an AF call is impossible on ANY series: the coupling holds.
    doc, det = _video(stars=2, coherence=0.40)
    sc = _sidecar(_afib())
    rec = sr.evaluate(doc, det, sc)
    assert rec["used"] is True
    assert doc["outcome"] == "REPEAT_SCAN" and doc["predicted_class"] is None
    assert any("3-star floor" in r for r in doc["no_read_reasons"])
    # With the stars but WITHOUT the AF-grade coherence or two-region timing,
    # irregular intervals are seen and still not called.
    doc, det = _video(stars=3, coherence=0.25, tp=35.0, matched=0.6)
    rec = sr.evaluate(doc, det, sc)
    assert rec["used"] is True and doc["outcome"] == "REPEAT_SCAN"
    assert doc["predicted_class"] is None
    assert doc["debug"]["rationale"]["rule"]["fired"] == "ABSTAIN (irregular but unverified)"
    # Only with our regions' AF-grade verification does the call go through -
    # exactly as it would on our own lattice.
    doc, det = _video(stars=3, coherence=0.40, tp=20.0, matched=0.8)
    rec = sr.evaluate(doc, det, sc)
    assert rec["used"] is True and doc["outcome"] == "ACCEPT"
    assert doc["predicted_class"] == "AFIB_SUGGESTIVE"
    same = ScanResult(recording_id="t", outcome=ScanOutcome.ACCEPT,
                      predicted_class="AFIB_SUGGESTIVE", confidence_stars=3,
                      confidence_limiting_factor="signal_snr")
    assert doc["user_facing_text"] == same.user_facing_text()


def test_the_route_never_raises_into_a_scan():
    doc, det = _video()
    det = dict(det)
    det["config"] = {"runs": None}                  # broken config
    before = dict(doc)
    rec = sr.evaluate(doc, det, _sidecar(_regular()))
    assert rec["used"] is False and rec["reason"].startswith("route failed safely")
    assert doc == before


# ------------------------------------------------------- service wiring
def test_signals_are_held_in_memory_for_the_job_without_touching_disk(tmp_path, monkeypatch):
    import io
    from app import measure_api as api
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "parts", raising=False)
    monkeypatch.setattr(api, "CLIPS_DIR", tmp_path / "clips", raising=False)
    monkeypatch.delenv("AFIB_KEEP_UPLOADS", raising=False)
    monkeypatch.delenv("AFIB_CLIPS_TOKEN", raising=False)
    api._SIGNALS.clear()
    body = json.dumps(_sidecar(_regular())).encode()
    h = api.MeasureHandler.__new__(api.MeasureHandler)
    h.path, h.rfile = "/api/scan-signals", io.BytesIO(body)
    h.headers = {"Content-Length": str(len(body)), "User-Agent": "t"}
    h.close_connection = False
    h.sent = []
    h._json = lambda code, doc: h.sent.append((code, doc))
    # an upload that never started: nothing held, nothing written
    h._upload_signals({"upload_id": ["upl-nobody"]})
    assert h.sent[-1][0] == 200 and h.sent[-1][1]["stored"] is False
    assert h.sent[-1][1]["held"] is False and api._peek_signals("upl-nobody") is None
    # an upload in progress: held for the job, still nothing on disk
    api._part_dir("upl-abcdef").mkdir(parents=True)
    h.rfile = io.BytesIO(body)
    h._upload_signals({"upload_id": ["upl-abcdef"]})
    assert h.sent[-1][1]["stored"] is False and h.sent[-1][1]["held"] is True
    assert api._peek_signals("upl-abcdef")["reference"]["average_signal_quality"] == 0.9
    assert list((tmp_path / "parts" / "upl-abcdef").iterdir()) == []
    assert not (tmp_path / "clips").exists()
    api._discard_part_dir("upl-abcdef")
    assert api._peek_signals("upl-abcdef") is None


def test_wait_for_returns_as_soon_as_the_document_lands():
    calls = {"n": 0}

    def getter():
        calls["n"] += 1
        return {"ok": True} if calls["n"] >= 3 else None
    doc, waited = sr.wait_for(getter, budget_s=5.0, poll_s=0.01)
    assert doc == {"ok": True} and waited < 1.0
    doc, waited = sr.wait_for(lambda: None, budget_s=0.05, poll_s=0.01)
    assert doc is None and 0.04 <= waited < 1.0


def test_the_sheet_shows_the_source_and_the_route_verdict():
    from app import result_sheet
    row = result_sheet.row_from_doc(
        {"outcome": "ACCEPT", "rhythm_source": "shenai_train",
         "debug": {"shenai_route": {"used": True, "reason": "used: 56 clean intervals",
                                    "train": {"bpm": 71.8}}}}, extra={})
    assert row["Rhythm Source"] == "shenai_train"
    assert row["ShenAI Route"].startswith("used: 56 clean intervals")
    assert row["ShenAI Rate"] == 71.8
    row = result_sheet.row_from_doc(
        {"outcome": "REPEAT_SCAN", "rhythm_source": "video",
         "debug": {"shenai_route": {"used": False, "reason": "no ShenAI sidecar arrived"}}},
        extra={})
    assert row["ShenAI Route"] == "not used: no ShenAI sidecar arrived"
    assert row["ShenAI Rate"] == ""
    for c in ("Rhythm Source", "ShenAI Route", "ShenAI Rate"):
        assert c in result_sheet.COLUMNS


def test_a_peak_in_our_own_spectrum_corroborates_when_the_dominant_rhythm_is_wrong():
    """2026-09-17: on four staging scans our dominant rhythm read ~half the
    SDK's 84-90 bpm (a 0.8 Hz artefact), so the train had no corroboration.
    A local peak in our fused spectrum AT the train's rate, >= 2x the in-band
    median, is the third path (measured: 9/13 true rates, 4 % false)."""
    import numpy as np
    fb = np.round(np.arange(0.75, 3.0, 0.05), 3)
    psd = np.full(fb.size, 0.01)
    psd[np.argmin(np.abs(fb - 0.80))] = 0.12          # the artefact wins
    psd[np.argmin(np.abs(fb - 1.40))] = 0.05          # the pulse: a clear local peak, 5x median
    # our count doubled/uncertain and our dominant rhythm at 48 with 3 regions:
    # neither rate path corroborates an 84 bpm train...
    doc, det = _video(spectral=48.0, roi_agree=3, lattice=None, n_int=3)
    det = dict(det); det["_fused_psd"] = (fb, psd)
    rec = sr.evaluate(doc, det, _sidecar(_regular(bpm=84.0)))
    assert rec["used"] is True, rec
    assert "peak in our own waveform spectrum at 84 bpm" in rec["reason"]
    assert doc["outcome"] == "ACCEPT" and doc["predicted_class"] == "SINUS"
    # ...and a train whose rate has NO peak in our spectrum is still refused.
    doc, det = _video(spectral=48.0, roi_agree=3, lattice=None, n_int=3)
    det = dict(det); det["_fused_psd"] = (fb, psd)
    rec = sr.evaluate(doc, det, _sidecar(_regular(bpm=110.0)))
    assert rec["used"] is False and "not corroborated" in rec["reason"]
    assert rec["rate_check"]["peak_at_rate"]["ok"] is False


def test_peak_at_rate_is_exact_bin_and_needs_a_local_maximum():
    import numpy as np
    fb = np.round(np.arange(0.75, 3.0, 0.05), 3)
    psd = np.full(fb.size, 0.01)
    i = int(np.argmin(np.abs(fb - 1.40)))
    psd[i - 1], psd[i], psd[i + 1] = 0.03, 0.06, 0.04       # peak at 84 bpm
    assert sr.peak_at_rate(fb, psd, 84.0)["ok"] is True
    assert sr.peak_at_rate(fb, psd, 81.0)["ok"] is False      # the neighbour bin is a shoulder
    assert sr.peak_at_rate(fb, psd, 200.0)["ok"] is False     # outside the band
    flat = np.full(fb.size, 0.01)
    assert sr.peak_at_rate(fb, flat, 84.0)["ok"] is False     # no peak anywhere


def _fitness_doc(doc, bpm, value):
    doc["biomarkers"] = {"items": [
        {"key": "arterial_stiffness", "status": "computed", "value": 50.0},
        {"key": "cardiorespiratory_fitness", "status": "computed", "value": value,
         "tier": "provisional", "tier_reasons": ["only 5 clean beat intervals (a measured value needs 15)"],
         "raw": {"name": "resting_heart_rate", "value": bpm, "unit": "bpm"}, "details": {}}],
        "complete": True}
    return doc


def test_fitness_follows_the_train_rate_when_the_route_is_used():
    from features.hemodynamics import resting_rate_index
    doc, det = _video()
    _fitness_doc(doc, 45.0, 87.3)                     # the 06:11 shape: folded to half
    rec = sr.evaluate(doc, det, _sidecar(_regular(bpm=72.0)))
    assert rec["used"] is True
    r = sr.reconcile_fitness_rate(doc, rec)
    card = doc["biomarkers"]["items"][1]
    assert r["action"] == "recomputed" and card["status"] == "computed"
    assert card["raw"]["value"] == pytest.approx(doc["mean_pulse_rate_bpm"], abs=0.1)
    assert card["value"] == pytest.approx(round(100 * resting_rate_index(doc["mean_pulse_rate_bpm"]), 1), abs=0.2)
    assert any("live-frame beat train" in x for x in card["tier_reasons"])


def test_fitness_abstains_when_a_sound_train_contradicts_the_video_rate():
    # Route refused for corroboration, but the train itself is sound and its
    # rate (84) is nowhere near the card's folded 45: no number is shown.
    doc, det = _video(spectral=45.0, roi_agree=3, lattice=None, n_int=3)
    _fitness_doc(doc, 45.0, 87.3)
    rec = sr.evaluate(doc, det, _sidecar(_regular(bpm=84.0)))
    assert rec["used"] is False and rec.get("train_ok") is True
    r = sr.reconcile_fitness_rate(doc, rec)
    card = doc["biomarkers"]["items"][1]
    assert r["action"] == "abstained" and card["status"] == "not_computed" and card["value"] is None
    assert "45 bpm" in card["reason"] and "84 bpm" in card["reason"]
    assert doc["biomarkers"]["complete"] is False


def test_fitness_is_untouched_without_a_sound_train_or_when_rates_agree():
    doc, det = _video(spectral=45.0, roi_agree=3, lattice=None, n_int=3)
    _fitness_doc(doc, 80.0, 40.0)
    rec = sr.evaluate(doc, det, _sidecar(_regular(bpm=84.0)))
    r = sr.reconcile_fitness_rate(doc, rec)
    assert r["action"] == "agree" and doc["biomarkers"]["items"][1]["value"] == 40.0
    doc, det = _video()
    _fitness_doc(doc, 45.0, 87.3)
    rec = sr.evaluate(doc, det, None)                 # no sidecar at all
    assert sr.reconcile_fitness_rate(doc, rec)["action"] == "none"
    assert doc["biomarkers"]["items"][1]["value"] == 87.3


def test_the_rate_head_and_the_result_carry_one_pulse_when_the_route_is_used():
    doc, det = _video()
    doc["head_results"] = [{"head": "rate_flags", "value": {"median_bpm": 108.0}}]
    rec = sr.evaluate(doc, det, _sidecar(_regular(bpm=72.0)))
    assert rec["used"] is True
    val = doc["head_results"][0]["value"]
    assert val["median_bpm"] == doc["mean_pulse_rate_bpm"] == pytest.approx(72.0, abs=2.0)
    assert val["video_median_bpm"] == 108.0 and val["rate_source"] == "shenai_train"


def test_sheet_puts_the_afib_result_beside_the_cards():
    from app.result_sheet import COLUMNS
    i = COLUMNS.index("Fitness")
    assert COLUMNS[i + 1:i + 3] == ["AFib Result", "AFib p"]
    assert "AFib Basis" in COLUMNS

