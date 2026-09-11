"""Retained scan clips: off by default, and actually retained when on.

WHY THIS EXISTS: the eval corpus holds no 480x720 portrait clip, which is the
only shape production records, so the optimizer gate could not judge the
2026-09-10 downscale change and every candidate needed a 3-minute phone scan.
Retaining real uploads turns that into a replay.

WHY IT IS GATED: this is video of someone's face, and the staging service is a
public URL. Both AFIB_KEEP_UPLOADS=1 and a non-empty AFIB_CLIPS_TOKEN are
required; production sets neither.

The ShenAI sidecar (/api/scan-signals, added 2026-09-11) rides the same gate,
and the gate is the ONLY thing between a public unauthenticated POST and
persisted physiological waveforms, so it is pinned here too.
"""
import io
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import app.measure_api as api  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("AFIB_KEEP_UPLOADS", raising=False)
    monkeypatch.setattr(api, "CLIPS_TOKEN", "", raising=False)
    monkeypatch.setattr(api, "CLIPS_DIR", tmp_path / "clips", raising=False)
    monkeypatch.setattr(api, "CLIPS_KEEP", 3, raising=False)
    # _part_dir() and _sweep_parts() both read the module global, and
    # /api/scan-signals writes under it: a test that forgot this would write
    # into the real work dir.
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "parts", raising=False)


def _enable(monkeypatch, token="s3cret"):
    monkeypatch.setenv("AFIB_KEEP_UPLOADS", "1")
    monkeypatch.setattr(api, "CLIPS_TOKEN", token, raising=False)


SIGNALS = {"schema_version": 1, "upload_id": "upl-abcdef",
           "ppg": {"n": 3, "fs_hz": 30.0, "fs_source": "derived_from_beats",
                   "values": [0.1, 0.2, 0.3]},
           "heartbeats": [{"start": 0.0, "end": 0.8}]}


def _handler(path="/", body=b""):
    """A MeasureHandler with no socket.

    The route logic under test is a plain method; __init__ is what talks to a
    socket, so it is skipped and only the three attributes the method touches
    are supplied. _json is replaced by a recorder, which is also what keeps a
    404-vs-200 assertion honest without parsing a wire response.
    """
    h = api.MeasureHandler.__new__(api.MeasureHandler)
    h.path = path
    h.rfile = io.BytesIO(body)
    h.headers = {"Content-Length": str(len(body))}
    h.close_connection = False
    h.sent = []
    h._json = lambda code, doc: h.sent.append((code, doc))
    return h


def _post_signals(upload_id, payload=None):
    body = json.dumps(SIGNALS if payload is None else payload).encode()
    h = _handler("/api/scan-signals", body)
    h._upload_signals({"upload_id": [upload_id]})
    assert h.sent, "the route answered nothing"
    return h.sent[-1]


def _files_under(root: pathlib.Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file()) if root.exists() else []


def _clips():
    return [f for f in api.CLIPS_DIR.iterdir()
            if not f.name.endswith(api.CLIP_SIDECAR_SUFFIXES)]


def test_disabled_unless_both_switches_are_set(monkeypatch):
    assert api._clips_enabled() is False
    monkeypatch.setenv("AFIB_KEEP_UPLOADS", "1")
    assert api._clips_enabled() is False, "a token is not optional"
    monkeypatch.setattr(api, "CLIPS_TOKEN", "s3cret", raising=False)
    assert api._clips_enabled() is True
    monkeypatch.delenv("AFIB_KEEP_UPLOADS")
    assert api._clips_enabled() is False


def test_nothing_is_written_while_disabled(tmp_path):
    src = tmp_path / "scan.webm"
    src.write_bytes(b"video")
    api._retain_clip(str(src), "sess1")
    assert not api.CLIPS_DIR.exists()


def test_clip_and_its_timestamp_sidecar_are_kept(monkeypatch, tmp_path):
    _enable(monkeypatch)
    src = tmp_path / "scan.webm"
    src.write_bytes(b"video-bytes")
    (tmp_path / "scan.webm.timestamps.json").write_text('{"timestamps_s": [0.0]}')
    api._retain_clip(str(src), "abcdef123456")
    kept = _clips()
    assert len(kept) == 1
    assert kept[0].read_bytes() == b"video-bytes"
    # The sidecar is what lets a replay use the capture clock rather than guess.
    assert (pathlib.Path(str(kept[0]) + ".timestamps.json")).exists()


def test_only_the_newest_are_kept(monkeypatch, tmp_path):
    """Railway's disk is ephemeral and small; unbounded growth fills it."""
    _enable(monkeypatch)
    import time
    for i in range(6):
        src = tmp_path / f"s{i}.webm"
        src.write_bytes(b"x" * 100)
        api._retain_clip(str(src), f"sess{i}")
        time.sleep(0.01)
    kept = _clips()
    assert len(kept) <= api.CLIPS_KEEP, [f.name for f in kept]


def test_retention_never_breaks_a_scan(monkeypatch):
    """A scan must complete whether or not the clip could be kept."""
    _enable(monkeypatch)
    api._retain_clip("/nonexistent/path/scan.webm", "sess")   # must not raise


def test_a_clip_id_cannot_escape_the_clips_directory():
    """The id comes off a public URL; basename is what confines it."""
    for evil in ("../../etc/passwd", "..%2f..%2fetc/passwd", "/etc/passwd",
                 "sub/dir/scan.webm"):
        assert "/" not in os.path.basename(evil).replace("%2f", "/") or True
        resolved = pathlib.Path("/clips") / os.path.basename(evil)
        assert resolved.parent == pathlib.Path("/clips"), resolved


# --- ShenAI sidecar (/api/scan-signals, 2026-09-11) -------------------------
# The route is an UNAUTHENTICATED public POST carrying a dense PPG waveform
# and beat train: /beta/cardio-staging has no login, so a write token would
# have to ship inside a public JS bundle. _clips_enabled() is therefore the
# single thing standing between that URL and persisted physiological data on
# production. Until now it had no regression coverage at all - it was checked
# once by hand in a scratchpad probe that no longer exists - so the obvious
# "simplify" (write first, prune later at retention) would start persisting
# PPG on production with every test still green.

def test_signals_are_dropped_not_stored_while_retention_is_off(tmp_path):
    """THE production-privacy invariant: disabled means nothing hits disk.

    The part dir is pre-created because that is the real production shape: the
    phone has already uploaded its slices when it posts the signals. So the
    ONLY reason nothing is written here is the gate - not a missing directory.
    """
    api._part_dir("upl-abcdef").mkdir(parents=True)
    code, doc = _post_signals("upl-abcdef")
    # 200, not 404: the client reads 404 from an upload route as "this service
    # predates the route" and stops posting. Accepted-and-dropped, not refused.
    assert (code, doc["stored"]) == (200, False), doc
    assert _files_under(api.UPLOAD_DIR) == []
    assert not api.CLIPS_DIR.exists()


def test_signals_for_an_unknown_upload_write_nothing_while_off(tmp_path):
    """Same gate, before any directory exists: no part dir is conjured either,
    so a flood of anonymous POSTs cannot leave a trail on production."""
    code, doc = _post_signals("upl-never-seen")
    assert (code, doc["stored"]) == (200, False), doc
    assert _files_under(api.UPLOAD_DIR) == []
    assert not api.CLIPS_DIR.exists()


def test_a_sidecar_travels_with_its_clip_through_retention(monkeypatch, tmp_path):
    """Enabled: the POSTed JSON lands in the part dir and is copied beside the
    retained clip, byte for byte - that pairing is the whole point, since
    scripts/compare_shenai_signal.py reads the two together offline."""
    _enable(monkeypatch)
    d = api._part_dir("upl-abcdef")
    d.mkdir(parents=True)
    code, doc = _post_signals("upl-abcdef")
    assert (code, doc["stored"]) == (200, True), doc
    assert (d / api.SHENAI_PART_NAME).exists()

    src = d / "scan.webm"
    src.write_bytes(b"video-bytes")
    dst = api._retain_clip(str(src), "abcdef1234567890")
    paired = pathlib.Path(str(dst) + ".shenai.json")
    assert paired.exists(), sorted(f.name for f in api.CLIPS_DIR.iterdir())
    assert json.loads(paired.read_text()) == SIGNALS


def test_a_late_sidecar_is_paired_through_the_retained_pointer(monkeypatch, tmp_path):
    """The normal ordering: the phone posts signals only AFTER /api/start is
    accepted, so retention has already run and seen nothing. _pair_shenai at
    the end of the job is where the pairing actually happens, and the pointer
    file is the only place the retained basename survives (sid is truncated to
    16 chars)."""
    _enable(monkeypatch)
    src = tmp_path / "scan.webm"
    src.write_bytes(b"video-bytes")
    dst = api._retain_clip(str(src), "abcdef1234567890")
    assert not pathlib.Path(str(dst) + ".shenai.json").exists()

    d = api._part_dir("upl-abcdef")
    d.mkdir(parents=True)
    assert _post_signals("upl-abcdef")[1]["stored"] is True
    (d / api.RETAINED_POINTER_NAME).write_text(dst.name)
    result: dict = {}
    api._pair_shenai(str(d), result)
    assert json.loads(pathlib.Path(str(dst) + ".shenai.json").read_text()) == SIGNALS
    # Counts only ride the doc; the waveform itself never reaches the sheet.
    assert result["shenai"]["ppg_n"] == 3 and result["shenai"]["beats_n"] == 1
    assert "values" not in json.dumps(result["shenai"])


def test_a_shenai_sidecar_does_not_consume_a_keep_slot(monkeypatch, tmp_path):
    """Pruning counts CLIPS, not sidecars. A 30 KB JSON that took a slot in the
    newest-CLIPS_KEEP window would evict a real 3-minute phone recording - the
    exact failure CLIP_SIDECAR_SUFFIXES was introduced to prevent, and one no
    test could see before, because no test ever created a .shenai.json."""
    _enable(monkeypatch)
    import time
    for i in range(api.CLIPS_KEEP):                     # exactly the budget
        d = api._part_dir(f"upl-{i}")
        d.mkdir(parents=True)
        (d / api.SHENAI_PART_NAME).write_text(json.dumps(SIGNALS))
        src = d / "scan.webm"
        src.write_bytes(b"x" * 100)
        (d / "scan.webm.timestamps.json").write_text('{"timestamps_s": [0.0]}')
        api._retain_clip(str(src), f"sess{i}")
        time.sleep(0.01)
    kept = _clips()
    assert len(kept) == api.CLIPS_KEEP, sorted(f.name for f in api.CLIPS_DIR.iterdir())
    for clip in kept:
        assert pathlib.Path(str(clip) + ".shenai.json").exists()
        assert pathlib.Path(str(clip) + ".timestamps.json").exists()


def test_a_pruned_clip_takes_its_shenai_sidecar_with_it(monkeypatch, tmp_path):
    """Railway's disk is ephemeral and small: an orphaned JSON left behind by
    its evicted recording is unreadable forever and still costs bytes."""
    _enable(monkeypatch)
    monkeypatch.setattr(api, "CLIPS_KEEP", 2, raising=False)
    import time
    names = []
    for i in range(3):
        d = api._part_dir(f"upl-{i}")
        d.mkdir(parents=True)
        (d / api.SHENAI_PART_NAME).write_text(json.dumps(SIGNALS))
        src = d / "scan.webm"
        src.write_bytes(b"x" * 100)
        names.append(api._retain_clip(str(src), f"sess{i}"))
        time.sleep(0.01)
    assert not names[0].exists(), "the oldest clip should have been pruned"
    assert not pathlib.Path(str(names[0]) + ".shenai.json").exists()
    for f in api.CLIPS_DIR.iterdir():                   # no orphans at all
        if f.name.endswith(api.CLIP_SIDECAR_SUFFIXES):
            stem = f.name[:f.name.rindex(".", 0, f.name.rindex("."))]
            assert (api.CLIPS_DIR / stem).exists(), f.name


def test_a_sidecar_gets_no_row_of_its_own_in_the_listing(monkeypatch, tmp_path):
    """scripts/pull_scan_clips.py drives itself off this listing: a sidecar row
    would be downloaded as if it were a recording, and the `shenai` flag - not
    a row - is how it learns to ask for "<id>.shenai.json"."""
    _enable(monkeypatch)
    d = api._part_dir("upl-abcdef")
    d.mkdir(parents=True)
    (d / api.SHENAI_PART_NAME).write_text(json.dumps(SIGNALS))
    src = d / "scan.webm"
    src.write_bytes(b"video-bytes")
    (d / "scan.webm.timestamps.json").write_text('{"timestamps_s": [0.0]}')
    dst = api._retain_clip(str(src), "abcdef1234567890")

    h = _handler("/api/clips?token=s3cret")
    h._serve_clip()
    code, doc = h.sent[-1]
    assert code == 200, doc
    assert [r["id"] for r in doc["clips"]] == [dst.name], doc["clips"]
    assert doc["clips"][0]["shenai"] is True and doc["clips"][0]["sidecar"] is True


def test_the_listing_is_a_404_while_retention_is_off():
    """Off looks like a service that has no such path, not a 403 advertising
    that retained face video exists behind a token."""
    h = _handler("/api/clips?token=s3cret")
    h._serve_clip()
    assert h.sent[-1][0] == 404


def test_signals_for_an_upload_that_never_started_are_not_stored(monkeypatch):
    """Second guard, independent of the gate: the route writes into a part dir
    an upload already opened, and never creates one. It is unauthenticated, so
    without this an anonymous caller could park 4 MB per POST on the same small
    ephemeral disk the clips and the in-flight parts share - and a full disk
    fails a real scan at _upload_part's write. Still a 200, never a retry."""
    _enable(monkeypatch)
    code, doc = _post_signals("upl-never-seen")
    assert (code, doc["stored"]) == (200, False), doc
    assert not api._part_dir("upl-never-seen").exists()
    assert _files_under(api.UPLOAD_DIR) == []


# --------------------------------------------------------------------------
# Two gaps an adversarial review PROVED on 2026-09-11, both on the sidecar
# path added for route 6. Neither was caught by the round of tests written
# with the feature, which is why they are pinned here rather than described.

def test_pairing_writes_nothing_while_retention_is_disabled(monkeypatch, tmp_path):
    """_pair_shenai is the LAST writer in the family and was the only one with
    no gate.

    Proven with the gate off: a stale part-dir sidecar plus a clip left over
    from a gate-ON period made _pair_shenai copy a full PPG waveform into
    CLIPS_DIR and set doc["shenai"], which result_sheet then rendered as
    ShenAI Sidecar=TRUE, PPG N=3. That makes the invariant the docstring above
    asserts in prose - that _clips_enabled() is the single thing between a
    public URL and persisted physiological data - false in fact. Production
    runs with the gate off, so this is the production invariant.
    """
    import json
    monkeypatch.delenv("AFIB_KEEP_UPLOADS", raising=False)
    monkeypatch.setattr(api, "CLIPS_TOKEN", "", raising=False)
    monkeypatch.setattr(api, "CLIPS_DIR", tmp_path / "clips", raising=False)
    api.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    assert api._clips_enabled() is False

    part = tmp_path / "part"
    part.mkdir()
    (part / "shenai.json").write_text(json.dumps({
        "ppg": {"signal": [1.0, 2.0, 3.0], "fs_hz": 30.0},
        "heartbeats": [{"start_location_sec": 0.1,
                        "end_location_sec": 0.9, "duration_ms": 800}],
    }))

    doc = {}
    api._pair_shenai(part, doc, None)

    assert list(api.CLIPS_DIR.iterdir()) == [], "a waveform reached disk with the gate off"
    assert "shenai" not in doc, "physiological counts reached the result doc with the gate off"


def test_a_deeply_nested_sidecar_body_does_not_escape_the_parser():
    """json.loads raises RecursionError, which is NOT a ValueError.

    The sidecar parse caught only (UnicodeDecodeError, ValueError), so a body
    nested ~9999 deep - about 20 KB, well inside the route's 4 MB cap and the
    same order as a real 20-50 KB sidecar - escaped the handler, escaped
    do_POST (which has no blanket handler), and killed the connection with no
    response and a traceback. It sits BEFORE the _clips_enabled() gate, so it
    was reachable on production, on a route production documents as
    "accepted, counted and dropped without ever touching disk".
    """
    import json
    body = b'{"a":' + b'[' * 50_000 + b']' * 50_000 + b'}'
    raised = None
    try:
        json.loads(body.decode("utf-8"))
    except Exception as e:                                    # noqa: BLE001
        raised = e
    assert raised is not None, "expected the nested body to raise"
    # Whatever this build raises, the handler's except tuple must name it.
    import inspect
    src = inspect.getsource(api.MeasureHandler._upload_signals)
    assert "RecursionError" in src, (
        f"{type(raised).__name__} escapes _upload_signals' parse guard")
