"""Live-frame ROI traces as pipeline input (inference/trace_ingest.py)."""
import numpy as np
import pytest

from inference import trace_ingest as ti


def _doc(seconds=45.0, fps=30.0, bpm=72.0, amp=1.2, noise=0.3, drop=(), seed=3,
         luma=140.0):
    """A synthetic trace document: a pulse of `amp` counts on a `luma` base
    per region, with independent noise, at `fps`; frames in `drop` absent."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.arange(n) / fps + rng.normal(0, 0.002, n)
    t = np.maximum.accumulate(t)
    # a physiological shape: a sharp systolic upstroke every beat, not a
    # sinusoid (a broad sine gives the peak detector nothing to time)
    phase = np.mod((bpm / 60.0) * t, 1.0)
    pulse = np.exp(-((phase - 0.25) ** 2) / (2 * 0.06 ** 2)) + 0.12 * np.exp(-((phase - 0.62) ** 2) / (2 * 0.12 ** 2))
    traces = {}
    for k, r in enumerate(("forehead", "cheek_l", "cheek_r", "nose")):
        base = np.array([luma + 20, luma, luma - 15]) + 8 * k
        rows = []
        for i in range(n):
            if i in drop:
                rows.append(None)
                continue
            # the pulse modulates green most, as skin does
            g = base[1] - amp * pulse[i] + rng.normal(0, noise)
            rr = base[0] - 0.5 * amp * pulse[i] + rng.normal(0, noise)
            b = base[2] - 0.3 * amp * pulse[i] + rng.normal(0, noise)
            rows.append([float(rr), float(g), float(b)])
        traces[r] = rows
    bbox = [[0.3, 0.2, 0.4, 0.55] for _ in range(n)]
    return {"schema_version": 1, "t_s": t.tolist(), "traces": traces, "bbox": bbox,
            "width": 640, "height": 480, "fps_nominal": fps}


def test_a_trace_document_becomes_a_valid_ingest_without_a_codec_term():
    ing = ti.ingest_traces(_doc(), upload_id="t1")
    assert ing.ok, ing.reasons
    assert set(ing.traces) == {"forehead", "cheek_l", "cheek_r", "nose"}
    assert ing.traces["nose"].shape == (ing.timestamps_s.size, 3)
    assert ing.meta.codec_fourcc == "none" and ing.capture.is_lossless()
    assert 29.0 <= ing.meta.measured_fps_mean <= 31.0
    assert ing.track.tracker == "client-sdk-bbox" and ing.track.stability > 0.9
    assert any("live camera frames" in c for c in ing.capture_caveats)
    assert ing.photometric["median_usable_rois"] == 4.0


def test_absent_frames_are_holes_and_too_few_frames_fail_closed():
    ing = ti.ingest_traces(_doc(seconds=10.0, drop=set(range(100, 130))), upload_id="t2")
    assert ing.ok and ing.track.n_found == ing.track.n_frames - 30
    assert ing.track.longest_gap_s == pytest.approx(1.0, abs=0.1)
    # the kept clock has a 1 s hole exactly where the frames were absent
    assert np.max(np.diff(ing.timestamps_s)) == pytest.approx(31 / 30, abs=0.05)
    short = ti.ingest_traces(_doc(seconds=2.0), upload_id="t3")
    assert not short.ok and "need 3 s" in short.reasons[0]
    assert not ti.ingest_traces({"t_s": []}).ok
    assert not ti.ingest_traces("nonsense").ok


def test_a_low_frame_rate_still_meets_the_same_capture_gate_as_a_video():
    slow = ti.ingest_traces(_doc(fps=20.0), upload_id="t4")
    assert not slow.ok and any("fps" in r for r in slow.reasons)
    research = ti.ingest_traces(_doc(fps=28.0), capture_profile="research", upload_id="t5")
    assert not research.ok                      # 30 fps research floor
    consumer = ti.ingest_traces(_doc(fps=28.0), capture_profile="consumer", upload_id="t6")
    assert consumer.ok and any("below the 30 fps" in c for c in consumer.capture_caveats)


def test_the_registry_feeds_the_pipeline_and_leaves_real_paths_alone(tmp_path):
    import inference.pipeline as pl
    ing = ti.ingest_traces(_doc(), upload_id="t7")
    key = ti.register(ing)
    assert key.startswith("traces://")
    assert pl.ingest_video(key) is ing                      # consumed
    assert key not in ti._REGISTRY
    # a real path goes to the real reader (which refuses a missing file)
    res = pl.ingest_video(str(tmp_path / "nope.webm"), capture_profile="consumer")
    assert res.ok is False


def test_the_whole_pipeline_runs_on_traces_and_reads_the_pulse():
    from configs import load_config
    res, det = ti.run_on_traces(_doc(bpm=72.0), manifest={"capture_profile": "consumer"},
                                config=load_config(), upload_id="t8")
    assert res is not None and res.recording_id == "traces-t8"
    ev = det.get("evidence") or {}
    assert det["ingest"].ok
    # four independent regions carrying the same clean pulse: coherent, well timed
    assert ev.get("cross_roi_coherence", 0) >= 0.35
    assert ev.get("timing_precision_ms", 999) <= 40      # the any-call gate
    assert abs(ev.get("pulse_spectral_bpm", 0) - 72.0) <= 3.0
    assert res.outcome.value == "ACCEPT" and res.predicted_class == "SINUS"
    assert abs(res.mean_pulse_rate_bpm - 72.0) <= 3.0


# ------------------------------------------------------- service wiring
def test_traces_are_held_in_memory_for_the_job_and_never_written(tmp_path, monkeypatch):
    import io, json
    from app import measure_api as api
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "parts", raising=False)
    monkeypatch.setattr(api, "CLIPS_DIR", tmp_path / "clips", raising=False)
    api._TRACES.clear()
    body = json.dumps(_doc(seconds=5.0)).encode()

    def handler():
        h = api.MeasureHandler.__new__(api.MeasureHandler)
        h.path, h.rfile = "/api/scan-traces", io.BytesIO(body)
        h.headers = {"Content-Length": str(len(body)), "User-Agent": "t"}
        h.close_connection = False
        h.sent = []
        h._json = lambda code, doc: h.sent.append((code, doc))
        return h
    h = handler()
    h._upload_traces({"upload_id": ["upl-nobody"]})
    assert h.sent[-1][0] == 200 and h.sent[-1][1]["held"] is False
    assert api._peek_traces("upl-nobody") is None
    api._part_dir("upl-abcdef").mkdir(parents=True)
    h = handler()
    h._upload_traces({"upload_id": ["upl-abcdef"]})
    assert h.sent[-1][1]["held"] is True and h.sent[-1][1]["frames"] == 150
    assert api._peek_traces("upl-abcdef")["fps_nominal"] == 30.0
    assert list((tmp_path / "parts" / "upl-abcdef").iterdir()) == []
    api._discard_part_dir("upl-abcdef")
    assert api._peek_traces("upl-abcdef") is None


def test_the_trace_path_is_recorded_beside_the_video_path_and_decides_nothing(monkeypatch):
    from app import measure_api as api
    from configs import load_config
    api._TRACES.clear()
    api._hold_traces("upl-trace", _doc(bpm=72.0))
    doc = {"outcome": "REPEAT_SCAN", "predicted_class": None, "afib_result": "INCONCLUSIVE",
           "debug": {"rationale": {"gates_failed": ["coverage"]}}}
    det = {"config": load_config()}
    monkeypatch.setattr(api, "SHENAI_WAIT_S", 0.5)
    api._apply_trace_path(doc, det, "upl-trace", {"capture_profile": "consumer"})
    tp = doc["debug"]["trace_path"]
    assert tp["ran"] is True and tp["ingest_ok"] is True
    assert tp["outcome"] == "ACCEPT" and tp["predicted_class"] == "SINUS"
    assert tp["afib_result"] == "AFIB_NOT_DETECTED"
    assert abs(tp["pulse_bpm"] - 72.0) <= 3.0 and tp["n_intervals"] >= 40
    assert tp["coherence"] >= 0.35 and tp["timing_ms"] <= 40
    # the video path's own answer is untouched
    assert doc["outcome"] == "REPEAT_SCAN" and doc["afib_result"] == "INCONCLUSIVE"
    # no document -> recorded as such, quickly
    doc2 = {"outcome": "REPEAT_SCAN", "debug": {}}
    api._apply_trace_path(doc2, det, "upl-none", {"capture_profile": "consumer"})
    assert doc2["debug"]["trace_path"]["ran"] is False
    assert "no trace document" in doc2["debug"]["trace_path"]["reason"]


def test_the_sheet_carries_the_trace_path():
    from app import result_sheet
    row = result_sheet.row_from_doc(
        {"outcome": "REPEAT_SCAN", "debug": {"trace_path": {
            "ran": True, "outcome": "ACCEPT", "afib_result": "AFIB_NOT_DETECTED",
            "afib_probability": 0.0042, "sqi": 0.61, "coherence": 0.55, "timing_ms": 12.3,
            "n_intervals": 52, "coverage": 0.93, "pulse_bpm": 71.9, "fps": 29.8,
            "no_read_reasons": []}}}, extra={})
    assert row["Trace Ran"] == "TRUE" and row["Trace Outcome"] == "ACCEPT"
    assert row["Trace Result"] == "AFIB_NOT_DETECTED" and row["Trace p"] == 0.004
    assert row["Trace Intervals"] == 52 and row["Trace Pulse"] == 71.9 and row["Trace Note"] == ""
    row = result_sheet.row_from_doc({"outcome": "REPEAT_SCAN", "debug": {"trace_path": {
        "ran": False, "reason": "no trace document arrived (waited 8.0 s)"}}}, extra={})
    assert row["Trace Ran"] == "FALSE" and row["Trace Note"].startswith("no trace document")
    assert result_sheet.row_from_doc({"outcome": "ACCEPT"}, extra={})["Trace Ran"] == ""


def test_measure_traces_returns_a_full_standalone_result():
    from app import measure_api as api
    doc, det = api.measure_traces(_doc(bpm=72.0), manifest={"capture_profile": "consumer"},
                                  config_overrides={"decision.classifier": "model_a",
                                                    "decision.model_a_path": "models/model_a_v02_45s.json"})
    assert doc["outcome"] == "ACCEPT" and doc["predicted_class"] == "SINUS"
    assert doc["rhythm_source"] == "client_traces"
    assert doc["afib_result"] == "AFIB_NOT_DETECTED"
    assert isinstance(doc.get("afib_probability"), float)   # the RR classifier ran
    assert abs(doc["mean_pulse_rate_bpm"] - 72.0) <= 3.0
    assert "user_facing_text" in doc and "No irregular rhythm" in doc["user_facing_text"]
    assert doc["biomarkers"]["items"], "cards computed"
    assert doc["trace_ingest"]["ingest_ok"] is True
    # a bad document still returns exactly one of three, never throws
    bad, _ = api.measure_traces({"t_s": [0, 0.03]}, manifest={"capture_profile": "consumer"})
    assert bad["afib_result"] in ("INCONCLUSIVE", "AFIB_NOT_DETECTED", "AFIB_DETECTED")
    assert bad["outcome"] in ("NO_RESULT", "REPEAT_SCAN")


def test_measure_traces_endpoint_runs_and_writes_a_row(monkeypatch):
    import io, json
    from app import measure_api as api
    appended = {}
    monkeypatch.setattr(api.result_sheet, "schedule_append", lambda doc, extra=None: appended.update(doc=doc))
    body = json.dumps(_doc(bpm=72.0)).encode()
    h = api.MeasureHandler.__new__(api.MeasureHandler)
    h.path, h.rfile = "/api/measure-traces", io.BytesIO(body)
    h.headers = {"Content-Length": str(len(body)), "User-Agent": "t"}
    h.close_connection = False
    h.sent = []
    h._json = lambda code, doc: h.sent.append((code, doc))
    monkeypatch.setenv("AFIB_CONFIG_OVERRIDES",
                       '{"decision.classifier":"model_a","decision.model_a_path":"models/model_a_v02_45s.json"}')
    monkeypatch.setattr(api, "LAUNCH_CONFIG_OVERRIDES", api._launch_config_overrides())
    h._measure_traces_request({})
    code, doc = h.sent[-1]
    assert code == 200 and doc["afib_result"] == "AFIB_NOT_DETECTED"
    assert doc["predicted_class"] == "SINUS" and appended.get("doc") is doc
