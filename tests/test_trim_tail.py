"""The trim must isolate the recording's continuous TAIL, not "the last N
seconds by duration".

The client's rolling recorder keeps chunk 0 (container header + the first
~3 s of frames), drops the middle, keeps the tail. Measured on a real scan
(data/eval_corpus/webapp_trimfirst-06fcbb.webm): frames 0-2.97 s, a step
BACK to 1.10 s, frames to 3.34 s, a 134.7 s hole, then 71.8 s of tail. The
old cut, `-ss (duration - window)`, aimed into the hole; a keyframe seek into
a hole lands on the previous keyframe, frame 0, so every rolling-window scan
was "trimmed" to the whole clip and the backwards step came out as 57 frames
on one identical timestamp. These tests pin the cut selection (pure) and the
ffmpeg behaviour it works around.
"""
import os
import shutil
import subprocess

import pytest

from app import measure_prep
from app.measure_prep import GAP_S, choose_tail_cut, trim_tail


# ------------------------------------------------------------ pure selection
def _run(start, n, dt=1 / 30.0, key_every=None, first_key=True):
    """n frames from `start`, `dt` apart; keyframe on the first and every
    `key_every`-th frame."""
    out = []
    for i in range(n):
        key = (i == 0 and first_key) or (bool(key_every) and i % key_every == 0)
        out.append((round(start + i * dt, 3), key))
    return out


def _real_shape():
    """The measured shape of a rolling-window scan (see module docstring)."""
    return (_run(0.0, 90, key_every=100)            # chunk 0, part 1
            + _run(1.101, 68, first_key=False)       # chunk 0 steps BACK 1.87 s
            + _run(138.08, 2154, key_every=100))     # the tail, 71.8 s


def test_rolling_window_clip_is_cut_to_its_tail_keyframe():
    cut = choose_tail_cut(_real_shape(), window_s=70.0)
    assert cut["cut_s"] == 138.08                    # the tail's first keyframe
    assert cut["discontinuities"] == 2               # one step back, one hole
    assert cut["head_dropped_s"] == pytest.approx(3.33, abs=0.02)
    assert cut["gap_s"] == pytest.approx(134.745, abs=0.01)
    assert cut["expected_packets"] == 2154
    assert cut["kept_s"] == pytest.approx(71.77, abs=0.05)


def test_a_short_tail_is_still_isolated_from_its_head():
    # 3 s head, hole, 20 s tail: 23 s total is "within" a 40 s window, but
    # the head and the hole are not footage and must go anyway.
    tl = _run(0.0, 90, key_every=30) + _run(60.0, 600, key_every=30)
    cut = choose_tail_cut(tl, window_s=40.0)
    assert cut["cut_s"] == 60.0
    assert cut["expected_packets"] == 600
    assert cut["discontinuities"] == 1


def test_a_backwards_step_alone_is_a_discontinuity():
    tl = _run(0.0, 300, key_every=30) + _run(8.0, 300, key_every=30)   # steps back 2 s
    cut = choose_tail_cut(tl, window_s=40.0)
    assert cut["cut_s"] == 8.0
    assert cut["discontinuities"] == 1
    assert cut["expected_packets"] == 300


def test_a_continuous_short_clip_is_left_alone():
    cut = choose_tail_cut(_run(0.0, 900, key_every=90), window_s=40.0)   # 30 s
    assert cut["cut_s"] is None
    assert "already within" in cut["reason"]
    assert cut["discontinuities"] == 0


def test_a_continuous_long_clip_is_cut_by_the_window_on_a_keyframe():
    tl = _run(0.0, 3000, key_every=90)               # 100 s, keyframe every 3 s
    cut = choose_tail_cut(tl, window_s=40.0)
    target = tl[-1][0] - 40.0
    assert cut["cut_s"] <= target                    # never shorter than asked
    assert cut["cut_s"] >= target - 3.0              # within one keyframe interval
    assert tl[cut["cut_index"]][1] is True           # and ON a keyframe
    assert 40.0 <= cut["kept_s"] <= 43.1


def test_no_keyframe_after_the_hole_means_no_copy_cut():
    tl = _run(0.0, 90, key_every=30) + _run(60.0, 300, first_key=False)
    cut = choose_tail_cut(tl, window_s=40.0)
    assert cut["cut_s"] is None
    assert "keyframe" in cut["reason"]


def test_a_lost_chunk_inside_the_tail_is_spanned_not_cut():
    # The 2026-09-16 08:53 staging scan: chunk 0 with its step back, a 20 s
    # hole, then a 44.9 s tail with ONE 3.37 s chunk missing at 51.2 s.
    # Cutting after the missing chunk kept 14.6 s (12 clean intervals, three
    # short of a rhythm statement) and threw away 27.5 s of good footage.
    tl = (_run(0.0, 90, key_every=100)
          + _run(0.752, 78, first_key=False)              # steps back 2.17 s
          + _run(23.695, 825, key_every=100)              # tail, part 1
          + _run(54.028, 439, key_every=100))             # 2.85 s gap, part 2
    cut = choose_tail_cut(tl, window_s=45.0)
    assert cut["discontinuities"] == 3
    assert cut["spanned_gaps"] == [pytest.approx(2.85, abs=0.05)]
    assert cut["cut_s"] == 23.695                        # the tail's first keyframe
    assert cut["kept_s"] == pytest.approx(44.9, abs=0.1)
    assert cut["expected_packets"] == 825 + 439


def test_two_lost_chunks_are_still_a_hard_break():
    # 6.7 s of missing footage is past SPAN_GAP_S: the run after it is the
    # tail, exactly as before.
    tl = (_run(0.0, 90, key_every=30)
          + _run(60.0, 600, key_every=30)
          + _run(86.7, 300, key_every=30))                # 6.7 s gap
    cut = choose_tail_cut(tl, window_s=45.0)
    assert cut["cut_s"] == 86.7
    assert cut["spanned_gaps"] == []
    assert cut["discontinuities"] == 2


def test_a_soft_gap_in_a_short_continuous_clip_is_still_a_no_op():
    tl = _run(0.0, 300, key_every=30) + _run(12.5, 300, key_every=30)  # 2.5 s stall
    cut = choose_tail_cut(tl, window_s=45.0)
    assert cut["cut_s"] is None
    assert "already within" in cut["reason"]
    assert cut["spanned_gaps"] == [pytest.approx(2.5, abs=0.05)]


def test_jitter_is_not_a_discontinuity():
    # 30-70 ms frame intervals are the phone's normal wander.
    tl = [(0.0, True)]
    t = 0.0
    for i in range(1, 600):
        t += 0.03 if i % 2 else 0.07
        tl.append((round(t, 3), i % 100 == 0))
    cut = choose_tail_cut(tl, window_s=40.0, gap_s=GAP_S)
    assert cut["discontinuities"] == 0


# ------------------------------------------------------------ with ffmpeg
ffmpeg_missing = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None


def _gap_clip(path, head_s=3.0, hole_until_s=9.0, total_s=16.0):
    """A real webm with a hole in its packet clock, like a rolling-window
    upload: frames for t < head_s and t >= hole_until_s only, timestamps kept
    (fps_mode passthrough), a keyframe every second."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=size=160x120:rate=30:duration={total_s}",
         "-vf", f"select='lt(t,{head_s})+gte(t,{hole_until_s})'",
         "-fps_mode", "passthrough", "-c:v", "libvpx", "-g", "30",
         "-b:v", "500k", "-pix_fmt", "yuv420p", "-an", path],
        check=True)
    return path


def _timeline(path):
    return measure_prep._packet_timeline(shutil.which("ffprobe"), path)


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_trim_isolates_the_tail_of_a_clip_with_a_hole(tmp_path):
    src = _gap_clip(str(tmp_path / "scan.webm"))
    before = _timeline(src)
    assert choose_tail_cut(before, 40.0)["gap_s"] == pytest.approx(6.0, abs=0.1)

    info = trim_tail(src, window_s=40.0)

    assert info["applied"] is True, info
    assert info["discontinuities"] == 1
    assert info["trimmed_from_s"] == pytest.approx(9.0, abs=0.05)
    assert "hole" in info["note"] and info["cut_method"] == "keyframe-seek"
    after = _timeline(src)
    pts = [t for t, _ in after]
    assert len(after) == pytest.approx(210, abs=2)   # 7 s of tail at 30 fps
    assert pts[0] == pytest.approx(0.0, abs=0.05)    # re-based to zero
    assert max(b - a for a, b in zip(pts, pts[1:])) < GAP_S   # continuous
    assert min(b - a for a, b in zip(pts, pts[1:])) > 0        # monotonic


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_a_continuous_short_clip_is_not_touched(tmp_path):
    src = str(tmp_path / "short.webm")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=size=160x120:rate=30:duration=5",
                    "-c:v", "libvpx", "-b:v", "500k", "-pix_fmt", "yuv420p",
                    "-an", src], check=True)
    n = len(_timeline(src))
    info = trim_tail(src, window_s=40.0)
    assert info["applied"] is False
    assert "already within" in info["reason"]
    assert len(_timeline(src)) == n


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_why_a_seek_into_the_hole_returned_the_whole_clip(tmp_path):
    """Pins the ffmpeg behaviour the timeline cut exists for. If this ever
    fails, ffmpeg changed how a copy seek resolves inside a hole — good to
    know, and the cut above stays correct either way."""
    src = _gap_clip(str(tmp_path / "scan.webm"))
    out = str(tmp_path / "old_cut.webm")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "6.000", "-i", src,
                    "-c", "copy", "-an", out], check=True)      # 6 s is in the hole
    got = _timeline(out)
    pts = [t for t, _ in got]
    # It resolved to the last keyframe BEFORE the hole (here 2.0 s; on the
    # real scan, whose chunk 0 has one keyframe, frame 0): head frames and
    # the hole itself are still in the "trimmed" file.
    assert len(got) > 210                                        # more than the tail
    assert max(b - a for a, b in zip(pts, pts[1:])) > GAP_S      # the hole survived


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_a_cut_that_lands_elsewhere_is_rejected_not_trusted(tmp_path, monkeypatch):
    src = _gap_clip(str(tmp_path / "scan.webm"))
    calls = []
    real = measure_prep._verify_cut

    def strict(ffprobe, path, expected, spanned=0):
        calls.append(expected)
        # Pretend the first cut came back as the whole clip.
        if len(calls) == 1:
            return "got 300 packets, expected 210 (the seek landed elsewhere)"
        return real(ffprobe, path, expected, spanned)

    monkeypatch.setattr(measure_prep, "_verify_cut", strict)
    info = trim_tail(src, window_s=40.0)
    assert info["applied"] is True, info
    assert info["cut_method"] == "output-side"       # the second way was used
    assert len(_timeline(src)) == pytest.approx(210, abs=2)


# ------------------------------------------------ the downscale failure itself
def _clip_with_a_hole_longer_than_the_avi_limit(tmp_path, hole_s=2010.0):
    """A 30 fps clip whose packet clock jumps by `hole_s`. ffmpeg's AVI muxer
    writes skip frames across a hole and refuses more than 60000 of them; at
    this clip's 1/30 timebase that is 2000 s. A phone clip trips the same
    limit at 60 s, because its irregular millisecond clock makes ffmpeg guess
    a 1000 fps rate and a 1 ms timebase — reproduced exactly on a real scan,
    and the error text was the one on 4 of 11 staging scans."""
    part = str(tmp_path / "part.webm")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=size=640x480:rate=30:duration=3",
                    "-c:v", "libvpx", "-g", "30", "-b:v", "500k",
                    "-pix_fmt", "yuv420p", "-an", part], check=True)
    lst = tmp_path / "hole.txt"
    lst.write_text(f"file '{part}'\nduration {hole_s}\nfile '{part}'\n")
    out = str(tmp_path / "scan.webm")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-safe", "0", "-f", "concat",
                    "-i", str(lst), "-c", "copy", "-an", out], check=True)
    return out


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_a_hole_past_the_avi_limit_breaks_the_downscale_and_the_note_says_so(tmp_path):
    from app.measure_prep import downscale
    src = _clip_with_a_hole_longer_than_the_avi_limit(tmp_path)
    info = downscale(src, scale="480x360")
    assert info["applied"] is False
    # The first line of ffmpeg's stderr names the cause; the note must keep it.
    assert "Too large number of skipped frames" in (info.get("reason") or "")


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_the_trim_removes_the_hole_so_the_downscale_succeeds(tmp_path):
    from app.measure_prep import downscale
    src = _clip_with_a_hole_longer_than_the_avi_limit(tmp_path)
    t = trim_tail(src, window_s=40.0)
    assert t["applied"] is True and t["discontinuities"] == 1, t
    info = downscale(src, scale="480x360")
    assert info["applied"] is True and info["attempt"] == "fit", info
