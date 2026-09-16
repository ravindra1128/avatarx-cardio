"""
Clip preparation for the measure service — runs BEFORE the pipeline sees
the video. Transport-side only; nothing here analyses anything.

Two steps, both ffmpeg, both no-ops (reporting why) when ffmpeg is absent:

  downscale  — shrink the frame to the resolution the pipeline's own demo
               records at. THE fix for browser capture (evidence below).
  trim_tail  — keep the last N seconds (the settled, in-frame part).
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

DEFAULT_SCALE = os.environ.get("AFIB_SCALE", "480x360")
# VP9 -cpu-used: 0 = slowest/best, 2 = default here (~4x faster than 0,
# held or raised coherence on the two reference scans), 4+ = fast but thins
# the coherence margin. Flip via env to trade time for margin without a
# code edit; the value is echoed in every response under `downscale`.
VP9_CPU_USED = os.environ.get("AFIB_VP9_CPU_USED", "2").strip() or "2"


def downscale(video_path: str, scale: str) -> dict:
    """Downscale in place to `scale` ("WxH", or "" / "0" to disable).

    WHY THIS IS THE FIX, not bitrate: the pipeline's own live demo records
    at 480x360 and computes every resting biomarker. The webapp records
    ShenAI's full 1080x720 track — 4.5x the pixels competing for the same
    encoder bits. Measured on ONE lossless demo recording encoded at
    480x360: VP9 8 Mbps gave cross-ROI coherence 0.175 (fails the 0.2
    gate), 24 Mbps gave 0.329 (passes). Then measured on a REAL webapp scan
    recorded at 1080x720 (VP8, ~19.7 Mbps, 101 MB): as recorded, coherence
    0.122 and no card computed; the SAME lossy file downscaled to 480x360
    gave coherence 0.312, coverage 0.38 -> 0.57, all three cards computed.

    Spatial averaging before analysis raises the per-ROI signal-to-noise
    (the ROI MEAN is what rPPG reads); doing it server-side costs the
    browser nothing — no second decode pipeline, no extra memory pressure
    on ShenAI's WASM. Reported in the response as `downscale` so a scaled
    run is never mistaken for a native one."""
    info = {"applied": False, "scale": scale or None, "reason": None,
            "vp9_cpu_used": VP9_CPU_USED}
    if not scale or scale.strip() in ("0", "off", "none"):
        info["reason"] = "downscale disabled"
        return info
    try:
        w, h = (int(x) for x in scale.lower().split("x"))
        if w <= 0 or h <= 0:
            raise ValueError
    except Exception:
        info["reason"] = f"bad scale {scale!r} — expected WxH"
        return info
    ff = shutil.which("ffmpeg")
    if not ff:
        info["reason"] = "ffmpeg not on PATH — analysing at native resolution"
        return info

    import cv2
    cap = cv2.VideoCapture(video_path)
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    info["source"] = f"{sw}x{sh}"
    if sw and sh and sw <= w and sh <= h:
        info["reason"] = f"source {sw}x{sh} already within {w}x{h}"
        return info

    # ASPECT RATIO IS NEVER CHANGED. An earlier version forced an exact
    # WxH: on a 1080x720 (3:2) source that squeezed the frame 11% narrower
    # into 4:3, and a face that the pipeline correctly judged "partially
    # outside the frame or too close" then PASSED that geometry gate purely
    # because it had been compressed. Distorting the image until a gate
    # accepts it is not a fix; every downstream number would be computed
    # from a misshapen face. Fit inside the box and let the other edge
    # follow (-2 = keep AR, round to even).
    #
    # FIT, not "set the width" (fixed 2026-09-10). The previous line was
    # `scale={w}:-2`, which sets the WIDTH — the long edge only on a
    # LANDSCAPE source. Every phone scan is 480x720 PORTRAIT, whose width is
    # already 480, so the scale was a no-op: the clip was re-encoded to
    # 480x720, spending 11-16 s per scan to average exactly zero pixels
    # (measured on all 44 tracking-sheet scans; every one is 480x720).
    # Averaging is the whole point of this step — it is what suppresses the
    # compression artefacts that a 0.48 bpp VP8 phone clip is full of, and
    # a controlled A/B on 2026-09-10 showed those artefacts cost cross-ROI
    # correlation 0.555 -> 0.163, which is what shatters the beat series.
    # A 480x720 source now fits to 240x360: 4x fewer pixels, each a mean of
    # four, which is the lift this function was written to deliver.
    # Landscape sources are UNCHANGED (1080x720 -> 480x320, 640x480 ->
    # 480x360, both exactly as before) — only tall clips move.
    fit = min(w / float(sw), h / float(sh))
    tw = max(2, int(round(sw * fit / 2.0)) * 2)
    info["target"] = f"{tw}x~{int(round(sh * fit))}"
    out = video_path + ".scaled.avi"
    # VP9, not VP8, and a high bitrate: the point is to average pixels, not
    # to add a second lossy generation. Measured on the same 1080x720 scan
    # downscaled to 480x360 at 12M: VP8 (libvpx) starved its own rate
    # control to 22 MB and gave coherence 0.134 / no cards; VP9 at the
    # same settings gave 0.263 / all three cards. -sws_flags area = box
    # filter = a true mean, and beat the default scaler (0.263 vs 0.22).
    # Encoder speed: single-thread VP9 took 81 s on a 40 s clip. Measured on
    # the two real scans that pass: -cpu-used 2 with row-mt holds or raises
    # coherence (0.221 -> 0.239 on the weakest passing clip) at ~4x the
    # speed; -cpu-used 4 is 7x faster but thins the margin (0.263 -> 0.228),
    # so 2 is the setting. ffmpeg caps -threads to the cores it has.
    # LOSSLESS intermediate (FFV1), not a second lossy generation. Measured
    # on one real scan, same source, downscaled both ways: VP9 12M gave
    # beat-timing 54.5 ms / 14 intervals / no cards; FFV1 gave 25.0 ms / 51
    # intervals / all three cards, at 1/5 the encode time (16 s vs 88 s).
    # VP9 was also NON-DETERMINISTIC - threaded rate control produced a
    # different file each run, and the spread (coherence 0.188-0.205) was
    # as large as the distance to the gate, so the same scan could pass or
    # fail at random. FFV1 is bit-identical run to run (verified by md5).
    # The file is larger, but it is a server-side temp file, never uploaded.
    # Two attempts, and the SECOND one matters as much as the first.
    #
    # Measured on 6 real staging scans (2026-09-10 12:45-13:17): the resize
    # failed on 3 of them with ffmpeg exit 234 (= -22 & 0xFF = EINVAL), and
    # again on 1 of 5 on 2026-09-14. Root cause, reproduced exactly on a real
    # camera clip (tests/test_trim_tail.py pins it): the phone's packet clock
    # is irregular milliseconds, so ffmpeg guesses r_frame_rate = 1000/1 and
    # gives the AVI stream a 1 ms timebase; the AVI muxer fills a hole in the
    # clock with "skip frames" and refuses a hole longer than 60000 ticks —
    # 60 s at 1 ms — with "Too large number of skipped frames". The client's
    # rolling recorder leaves exactly such a hole (chunk 0, then nothing until
    # the last 48 s), so any session longer than ~115 s failed here, and the
    # old trim never removed the hole (see choose_tail_cut). trim_tail now
    # cuts the hole out before this runs; the fallback chain stays because a
    # clip that reaches this step with a hole is still better transcoded than
    # analysed raw.
    #
    # The old no-op `scale={w}:-2` on an already-480-wide clip was never
    # useless: it still transcoded VP8 -> FFV1, which normalises the container
    # and hands the analysis one clean decode. Losing it is why the failures
    # above degraded instead of merely staying the same — two of them returned
    # NO_RESULT. So a resize failure falls back to that plain transcode, NOT
    # to the raw upload. Native is the last resort only if ffmpeg cannot read
    # the file at all.
    base = [ff, "-y", "-v", "error", "-i", video_path]
    tail = ["-c:v", "ffv1", "-pix_fmt", "bgr0", "-an", out]
    attempts = [
        ("fit", base + ["-vf", f"scale={tw}:-2", "-sws_flags", "area"] + tail),
        ("transcode-only", base + tail),
    ]
    err = None
    for label, cmd in attempts:
        try:
            subprocess.run(cmd, check=True, timeout=300,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if os.path.getsize(out) > 0:
                info["attempt"] = label
                break
        except Exception as e:                            # noqa: BLE001
            # ffmpeg's own words, not just the exit code — the exit code alone
            # cost a whole round of scans to interpret.
            detail = getattr(e, "stderr", b"") or b""
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", "replace")
            d = detail.strip()
            if len(d) > 440:
                d = d[:240] + " … " + d[-160:]   # first line names the cause
            err = f"{label}: {e} :: {d}"
            try:
                if os.path.exists(out):
                    os.remove(out)
            except OSError:
                pass
            continue
    try:
        if err and not info.get("attempt"):
            raise RuntimeError(err)
        if info.get("attempt") == "transcode-only":
            info["degraded"] = f"resize failed, transcoded at native size ({err})"
        if os.path.getsize(out) > 0:
            # The container changed (AVI/FFV1), so the file must change name
            # too - leaving AVI bytes in a .webm would make the decoder
            # guess. Caller re-reads the returned path.
            side = video_path + ".timestamps.json"
            if os.path.exists(side):
                os.remove(side)                  # describes the old frames
            try:
                os.remove(video_path)
            except OSError:
                pass
            info["applied"] = True
            info["path"] = out
            info["scaled_to"] = (f"{tw}x(auto, AR preserved)"
                                 if info.get("attempt") == "fit"
                                 else f"{sw}x{sh} (resize failed — transcode only)")
        else:
            info["reason"] = "ffmpeg produced an empty file — keeping original"
    except Exception as e:
        info["reason"] = f"ffmpeg downscale failed ({e}) — analysing at native resolution"
    return info


def _ffprobe_path(ffmpeg_path: str) -> str | None:
    """ffprobe next to ffmpeg, else on PATH, else None."""
    ffprobe = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
    if os.path.exists(ffprobe):
        return ffprobe
    return shutil.which("ffprobe")


def _packet_timeline(ffprobe: str, video_path: str) -> list:
    """[(pts_s, is_keyframe), ...] of the first video stream, in FILE order.

    File order, not sorted: a step backwards in the clock is a finding (see
    choose_tail_cut) and sorting would hide it. I/O only, no decode — the
    per-packet pts are exactly what OpenCV's CAP_PROP_POS_MSEC reports.
    [] if ffprobe cannot read the file."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0",
             video_path],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    timeline = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pts = float(parts[0])
        except ValueError:
            continue                              # "N/A": no clock on it
        timeline.append((pts, "K" in parts[1]))
    return timeline


def _packet_duration_s(ffmpeg_path: str, video_path: str) -> float:
    """Second-to-last video packet pts in seconds via ffprobe (what the
    OpenCV decode loop reported — see trim_tail); 0.0 if unavailable."""
    ffprobe = _ffprobe_path(ffmpeg_path)
    if not ffprobe:
        return 0.0
    pts = sorted(t for t, _ in _packet_timeline(ffprobe, video_path))
    if len(pts) < 2:
        return 0.0
    return pts[-2]


# ------------------------------------------------------------------ trim
DEFAULT_WINDOW_S = float(os.environ.get("AFIB_WINDOW_S", "40"))

# A step in the packet clock larger than this is a dropped recorder chunk,
# never capture jitter: the phone's frame interval wanders 30-70 ms, a chunk
# is 3 s. A step BACKWARDS is a discontinuity at any size.
GAP_S = 1.0
# A forward gap shorter than this is a stalled or lost recorder chunk INSIDE
# the tail, not the hole: the clip keeps running on the same clock after it,
# capture_segments (inference/evidence.py) already splits the analysis around
# it, and beats never span it (Beat.segment, 2026-09-16). Measured 2026-09-16
# on a real staging scan: one 2.85 s gap at 51.2 s (exactly one 3.37 s chunk
# missing from the upload) made the cut start AFTER it, keeping 14.6 s of a
# 44.9 s tail — 12 clean intervals against the 15 needed, on a scan whose
# other gates all passed. The rolling recorder's hole is never this short in
# practice (session length minus ~51 s) and a two-chunk loss (6.7 s) is not
# spanned either, so the muxer's 60 s limit stays far away.
SPAN_GAP_S = 5.0


def choose_tail_cut(timeline: list, window_s: float,
                    gap_s: float = GAP_S,
                    span_gap_s: float = SPAN_GAP_S) -> dict:
    """Pure. Where to cut so that what remains is ONE continuous run of
    frames, about `window_s` long, starting on a keyframe.

    `timeline` is [(pts_s, is_keyframe), ...] in file order.

    WHY the timeline and not the duration. The client's rolling recorder
    keeps chunk 0 (it carries the container header — and the first 3 s of
    frames), drops the middle, keeps the tail. So a 118 s session uploads as
    frames 0–3 s, a 67 s hole, then the last 48 s; and inside chunk 0 the
    clock steps BACK once (measured on a real scan: -1.87 s at packet 90).
    Cutting at "duration minus window" put the seek target inside the hole,
    and a keyframe seek into a hole lands on the previous keyframe — frame
    0. So the "trim" returned the whole clip (measured: 2312 of 2312 packets
    for a -ss into the hole), and its only effect was to rewrite the
    backwards step as 57 frames on one identical timestamp — the collapsed
    intervals capture/ingest.py then had to be told to tolerate
    (MAX_COLLAPSED_INTERVAL_FRACTION 0.02 -> 0.05 on the staging service).
    The transcode that follows saw all of it too.

    Two kinds of discontinuity. A backwards step, or a forward gap of
    `span_gap_s` or more, is HARD: the tail starts after the last of them
    (the hole, chunk 0's clock step). A forward gap between `gap_s` and
    `span_gap_s` is SOFT — a stalled or dropped chunk inside the tail — and
    is kept inside the window, because the pipeline segments around it and
    cutting after it throws away the footage before it (see SPAN_GAP_S).

    Returns cut_s (keyframe pts to start from; None when there is nothing to
    do, or no keyframe to start a stream copy on), tail_start_s,
    discontinuities (all kinds), spanned_gaps (the soft ones kept inside the
    cut), head_dropped_s, gap_s, kept_s, expected_packets, and reason when
    cut_s is None."""
    n = len(timeline)
    if n < 2:
        return {"cut_s": None, "reason": "fewer than two video packets"}
    pts = [t for t, _ in timeline]
    tail_i, disc, hard, biggest_gap = 0, 0, 0, 0.0
    soft: list[tuple[int, float]] = []          # (index, gap) of soft ones
    for i in range(1, n):
        d = pts[i] - pts[i - 1]
        if d > gap_s or d <= 0:
            disc += 1
            biggest_gap = max(biggest_gap, d)
            if d <= 0 or d >= span_gap_s:
                hard += 1
                tail_i = i
            else:
                soft.append((i, d))
    tail_start, last = pts[tail_i], pts[-1]
    out = {
        "tail_start_s": round(tail_start, 3),
        "discontinuities": disc,
        "head_dropped_s": (round(pts[tail_i - 1] - pts[0], 3)
                           if tail_i else 0.0),
        "gap_s": round(biggest_gap, 3),
        "spanned_gaps": [round(d, 3) for _, d in soft],
    }
    if hard == 0 and last - pts[0] <= window_s + 1.0:
        out.update(cut_s=None, reason=(f"clip is {last - pts[0]:.1f}s — "
                                       "already within the window"))
        return out
    target = max(tail_start, last - window_s)
    keys = [i for i in range(tail_i, n) if timeline[i][1]]
    if not keys:
        out.update(cut_s=None, reason=("no keyframe after the last "
                                       "discontinuity — a stream copy "
                                       "cannot start there"))
        return out
    before = [i for i in keys if pts[i] <= target]
    cut_i = before[-1] if before else keys[0]
    out.update(cut_s=pts[cut_i], cut_index=cut_i,
               kept_s=round(last - pts[cut_i], 3),
               expected_packets=n - cut_i, reason=None,
               spanned_gaps=[round(d, 3) for i, d in soft if i > cut_i])
    return out


def _verify_cut(ffprobe: str, path: str, expected: int,
                spanned: int = 0) -> str | None:
    """None when `path` is about `expected` packets on one monotonic clock
    with at most `spanned` forward gaps, each under SPAN_GAP_S (the soft
    gaps choose_tail_cut kept on purpose); otherwise what is wrong with it.
    A backwards step or a gap of SPAN_GAP_S or more is never acceptable."""
    tl = _packet_timeline(ffprobe, path)
    if not tl:
        return "no readable video packets"
    if abs(len(tl) - expected) > 2:
        return (f"got {len(tl)} packets, expected {expected} "
                "(the seek landed elsewhere)")
    pts = [t for t, _ in tl]
    steps = [b - a for a, b in zip(pts, pts[1:])]
    hard = sum(1 for d in steps if d <= 0 or d >= SPAN_GAP_S)
    soft = sum(1 for d in steps if GAP_S < d < SPAN_GAP_S)
    if hard:
        return f"{hard} hard discontinuit{'y' if hard == 1 else 'ies'} remain"
    if soft > spanned:
        return (f"{soft} gap{'' if soft == 1 else 's'} remain, "
                f"{spanned} expected")
    return None


def trim_tail(video_path: str, window_s: float) -> dict:
    """Keep only the settled tail of the recording, in place. Returns
    provenance; never fails the request over trimming.

    WHY: a browser scan records the whole session — positioning, alignment,
    then the measurement — and lands at 100-150 s. This pipeline's own scan
    is 30 s. That mismatch matters more than it looks, because
    capture/ingest.py rejects an ROI if ANY sample in its trace is
    non-finite:

        bad = [roi for roi, tr in traces.items()
               if not np.all(np.isfinite(tr))]

    Measured on a real capture: 3 frames out of 2824 (0.106%) had a
    degenerate forehead box, and those 3 frames failed the entire scan. The
    longer the clip, the likelier that blip — so sending five times more
    video than the pipeline wants is actively harmful, quite apart from the
    wasted upload and decode.

    The tail is the right window: the recording ends when the measurement
    completes, so the last seconds are settled, in-frame footage. WHERE the
    tail starts is read off the packet clock (choose_tail_cut), because the
    client's rolling recorder leaves a hole in the middle of the clip and a
    cut placed by duration alone lands in it.

    No-ops (reporting why) when ffmpeg is absent or the clip is already one
    short continuous run."""
    import shutil
    import subprocess

    info = {"applied": False, "window_s": window_s, "reason": None}
    if not window_s or window_s <= 0:
        info["reason"] = "trimming disabled (window_s <= 0)"
        return info
    ff = shutil.which("ffmpeg")
    if not ff:
        info["reason"] = "ffmpeg not on PATH — analysing the whole clip"
        return info

    # Duration from the container's PACKET timestamps, not from decoding.
    # The old probe decoded every frame with OpenCV just to read the last
    # frame's clock — measured 13.75 s on a 1080x720 phone recording (57%
    # of the whole request). MediaRecorder webm carries no duration in its
    # header, which is why decoding looked necessary; the per-packet pts
    # are the same clock. One subtlety, kept on purpose: POS_MSEC read
    # before each cap.read() is the clock of the frame ALREADY decoded, so
    # the loop's answer was the SECOND-to-last frame's time; every result
    # to date was computed from it, so source_duration_s stays that value.
    ffprobe = _ffprobe_path(ff)
    timeline = _packet_timeline(ffprobe, video_path) if ffprobe else []
    if timeline:
        info["duration_probe"] = "packets"
        pts_sorted = sorted(t for t, _ in timeline)
        dur = pts_sorted[-2] if len(pts_sorted) >= 2 else 0.0
    else:
        import cv2
        cap = cv2.VideoCapture(video_path)
        dur = 0.0
        if cap.isOpened():
            while True:
                t = cap.get(cv2.CAP_PROP_POS_MSEC)
                ok, _ = cap.read()
                if not ok:
                    break
                if t and t > 0:
                    dur = t / 1000.0
        cap.release()
        info["duration_probe"] = "decode"
    info["source_duration_s"] = round(dur, 2)

    cut = None
    if timeline:
        cut = choose_tail_cut(timeline, window_s)
        for k in ("tail_start_s", "discontinuities", "head_dropped_s", "gap_s",
                  "spanned_gaps"):
            if k in cut:
                info[k] = cut[k]
        if cut.get("cut_s") is None:
            info["reason"] = cut.get("reason")
            info["note"] = cut.get("reason")
            return info
        start, expected = cut["cut_s"], cut["expected_packets"]
    else:
        # No packet clock to read (no ffprobe): the pre-2026-09-14 cut, by
        # duration alone, unverified.
        if dur <= window_s + 1.0:
            info["reason"] = f"clip is {dur:.1f}s — already within the window"
            return info
        start, expected = max(0.0, dur - window_s), None

    out = video_path + ".trim" + pathlib.Path(video_path).suffix
    # Two ways to cut a stream copy. Seeking (-ss before -i) is the fast
    # one: ffmpeg walks the packets to the keyframe at `start` and copies
    # from there — exact when `start` IS a keyframe's timestamp, and what it
    # does with a target inside a hole is documented on choose_tail_cut, so
    # the result is CHECKED against the packet count the cut should have,
    # never trusted. Output-side -ss (after -i) reads everything and drops
    # packets before `start`: slower, no seek involved.
    attempts = [
        ("keyframe-seek", [ff, "-y", "-v", "error", "-ss", f"{start:.3f}",
                           "-i", video_path, "-c", "copy", "-an", out]),
        ("output-side", [ff, "-y", "-v", "error", "-i", video_path,
                         "-ss", f"{start:.3f}", "-c", "copy", "-an", out]),
    ]
    problems = []
    for label, cmd in attempts:
        try:
            subprocess.run(cmd, check=True, timeout=120,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except Exception as e:                            # noqa: BLE001
            detail = getattr(e, "stderr", b"") or b""
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", "replace")
            problems.append(f"{label}: {e} :: {detail.strip()[:200]}")
            continue
        if not os.path.exists(out) or os.path.getsize(out) <= 0:
            problems.append(f"{label}: empty file")
            continue
        if expected is not None:
            wrong = _verify_cut(ffprobe, out, expected,
                                len((cut or {}).get("spanned_gaps") or []))
            if wrong:
                problems.append(f"{label}: {wrong}")
                continue
        os.replace(out, video_path)
        # The old sidecar describes the untrimmed clip.
        side = video_path + ".timestamps.json"
        if os.path.exists(side):
            os.remove(side)
        info["applied"] = True
        info["trimmed_from_s"] = round(start, 2)
        info["cut_method"] = label
        if cut is not None:
            info["kept_s"] = cut["kept_s"]
            note = f"kept the last {cut['kept_s']:.1f}s from keyframe {start:.2f}s"
            if info.get("discontinuities"):
                n_d = info["discontinuities"]
                note += (f"; dropped {info['head_dropped_s']:.1f}s head + "
                         f"{info['gap_s']:.1f}s hole ({n_d} discontinuit"
                         f"{'y' if n_d == 1 else 'ies'})")
            if info.get("spanned_gaps"):
                sg = info["spanned_gaps"]
                note += (f"; kept across {len(sg)} gap"
                         f"{'' if len(sg) == 1 else 's'} of "
                         + "+".join(f"{g:.1f}s" for g in sg)
                         + " (segmented, not cut)")
            info["note"] = f"{note}; {label}"
        else:
            info["note"] = (f"kept from {start:.2f}s (duration only, no packet "
                            f"timeline); {label}")
        return info
    try:
        if os.path.exists(out):
            os.remove(out)
    except OSError:
        pass
    info["reason"] = ("trim could not isolate the tail (" + "; ".join(problems)
                      + ") — analysing the whole clip")
    info["note"] = info["reason"]
    return info
