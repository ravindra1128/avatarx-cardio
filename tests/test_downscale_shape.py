"""The downscale must FIT inside its box, whatever shape the source is.

This is the step that averages pixels, and averaging is what suppresses the
compression artefacts a phone's lossy clip is full of. It ran on every scan
and, on portrait video, averaged nothing: `scale={w}:-2` sets the WIDTH, which
is the long edge only on a landscape source. Every phone scan is 480x720, whose
width is already 480 — so the clip was re-encoded to 480x720, 11-16 s spent for
a 1.00x reduction (measured on all 44 tracking-sheet scans, 2026-09-10).

Nothing tested this, which is why it survived. These cases pin the shape maths
for both orientations; the landscape expectations are the pre-existing
behaviour and must not move.
"""
import os
import shutil
import subprocess
import pytest

import cv2

from app.measure_prep import downscale


def _make(path, w, h, seconds=1):
    """A tiny real video of the given shape (ffmpeg testsrc)."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=size={w}x{h}:rate=30:duration={seconds}",
         "-c:v", "libvpx", "-b:v", "1M", "-pix_fmt", "yuv420p", "-an", path],
        check=True,
    )
    return path


def _dims(path):
    cap = cv2.VideoCapture(path)
    d = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return d


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize("src_w,src_h,want_w,want_h", [
    # PORTRAIT — the shape every phone scan actually is. Was 480x720 (no-op).
    (480, 720, 240, 360),
    # LANDSCAPE — unchanged from the behaviour the corpus was validated on.
    (1080, 720, 480, 320),
    (640, 480, 480, 360),
])
def test_source_is_fitted_inside_the_box(tmp_path, src_w, src_h, want_w, want_h):
    src = _make(str(tmp_path / f"{src_w}x{src_h}.webm"), src_w, src_h)
    info = downscale(src, scale="480x360")
    out = src + ".scaled.avi"
    assert info.get("applied") is True, info
    assert os.path.exists(out)
    got = _dims(out)
    assert got == (want_w, want_h), f"{src_w}x{src_h} -> {got}, wanted {want_w}x{want_h}"
    # Neither edge may exceed the box: that is what "fit" means.
    assert got[0] <= 480 and got[1] <= 360


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_portrait_actually_loses_pixels(tmp_path):
    """The regression that mattered: it ran, cost seconds, and averaged nothing."""
    src = _make(str(tmp_path / "portrait.webm"), 480, 720)
    downscale(src, scale="480x360")
    w, h = _dims(src + ".scaled.avi")
    assert (480 * 720) / (w * h) == pytest.approx(4.0, abs=0.05)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_a_source_already_inside_the_box_is_left_alone(tmp_path):
    src = _make(str(tmp_path / "small.webm"), 480, 320)
    info = downscale(src, scale="480x360")
    assert info.get("applied") is False
    assert "already within" in (info.get("reason") or "")
    assert not os.path.exists(src + ".scaled.avi")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_aspect_ratio_is_never_changed(tmp_path):
    """A squeezed frame once let a face pass the geometry gate purely because
    it had been distorted; every downstream number came from a misshapen face."""
    for w, h in ((480, 720), (1080, 720), (640, 480)):
        src = _make(str(tmp_path / f"ar_{w}x{h}.webm"), w, h)
        downscale(src, scale="480x360")
        ow, oh = _dims(src + ".scaled.avi")
        assert (ow / oh) == pytest.approx(w / h, rel=0.02), f"{w}x{h} -> {ow}x{oh}"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_a_resize_failure_still_transcodes_rather_than_giving_up(tmp_path, monkeypatch):
    """The fallback that matters.

    Three of six real staging scans (2026-09-10) hit ffmpeg exit 234 (EINVAL)
    on the resize — their middle chunks are dropped by the client's rolling
    window and the capture rate wanders, so the stream reconfigures mid-file.
    Before this fallback those scans analysed the RAW upload and two returned
    NO_RESULT. The old no-op resize had still transcoded VP8 -> FFV1, giving
    the analysis one clean decode; that floor must survive a resize failure.
    """
    import subprocess as sp
    from app import measure_prep

    src = _make(str(tmp_path / "portrait.webm"), 480, 720)
    real = sp.run
    seen = []

    def only_resize_fails(cmd, *a, **k):
        vf = cmd[cmd.index("-vf") + 1] if "-vf" in cmd else ""
        seen.append("resize" if "scale=" in vf else "transcode")
        if "scale=" in vf:
            raise sp.CalledProcessError(234, cmd, stderr=b"Invalid argument")
        return real(cmd, *a, **k)

    monkeypatch.setattr(measure_prep.subprocess, "run", only_resize_fails)
    info = measure_prep.downscale(src, scale="480x360")

    assert seen == ["resize", "transcode"], seen
    assert info.get("applied") is True, info
    assert info.get("attempt") == "transcode-only"
    assert "resize failed" in (info.get("degraded") or "")
    out = src + ".scaled.avi"
    assert os.path.exists(out)
    # Native size kept, but it IS an FFV1 transcode — the normalisation floor.
    assert _dims(out) == (480, 720)
