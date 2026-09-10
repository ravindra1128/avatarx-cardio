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
    cmd = [ff, "-y", "-v", "error", "-i", video_path,
           "-vf", f"scale={tw}:-2", "-sws_flags", "area",
           "-c:v", "ffv1", "-pix_fmt", "bgr0", "-an", out]
    try:
        subprocess.run(cmd, check=True, timeout=300,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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
            info["scaled_to"] = f"{tw}x(auto, AR preserved)"
        else:
            info["reason"] = "ffmpeg produced an empty file — keeping original"
    except Exception as e:
        info["reason"] = f"ffmpeg downscale failed ({e}) — analysing at native resolution"
    return info


def _packet_duration_s(ffmpeg_path: str, video_path: str) -> float:
    """Second-to-last video packet pts in seconds via ffprobe (what the
    OpenCV decode loop reported — see trim_tail); 0.0 if unavailable."""
    ffprobe = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
    if not os.path.exists(ffprobe):
        ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return 0.0
    pts = []
    for line in out.splitlines():
        tok = line.split(",", 1)[0].strip()
        try:
            pts.append(float(tok))
        except ValueError:
            continue
    if len(pts) < 2:
        return 0.0
    pts.sort()
    return pts[-2]


# ------------------------------------------------------------------ trim
DEFAULT_WINDOW_S = float(os.environ.get("AFIB_WINDOW_S", "40"))


def trim_tail(video_path: str, window_s: float) -> dict:
    """Keep only the last `window_s` seconds, in place. Returns provenance.

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
    completes, so the last seconds are settled, in-frame footage.

    No-ops (reporting why) when ffmpeg is absent or the clip is already
    short enough — never fails the request over trimming."""
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
    # of the whole request) and 3.0 s on a 30 s demo clip that then wasn't
    # even trimmed. MediaRecorder webm carries no duration in its header,
    # which is why decoding looked necessary; but the per-packet pts are
    # exactly what OpenCV's CAP_PROP_POS_MSEC reports, so scanning packets
    # (I/O only, no decode) gives the same number. One subtlety, kept on
    # purpose: POS_MSEC read before each cap.read() is the clock of the frame
    # ALREADY decoded, so the loop's answer was the SECOND-to-last frame's
    # time, one frame short of the true duration. The cut point below is
    # relative to that value, and every result to date was computed from
    # it — so this returns the same second-to-last timestamp, not the last.
    # Verified equal to the loop within 1 ms on every clip of the eval corpus
    # (webm/vp8, webm/vp9, avi/ffv1); the decode loop remains the fallback.
    dur = _packet_duration_s(ff, video_path)
    info["duration_probe"] = "packets"
    if dur <= 0:
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
    if dur <= window_s + 1.0:
        info["reason"] = f"clip is {dur:.1f}s — already within the window"
        return info

    start = max(0.0, dur - window_s)
    out = video_path + ".trim" + pathlib.Path(video_path).suffix
    cmd = [ff, "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", video_path,
           "-c", "copy", "-an", out]
    try:
        subprocess.run(cmd, check=True, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if os.path.getsize(out) > 0:
            os.replace(out, video_path)
            # The old sidecar describes the untrimmed clip.
            side = video_path + ".timestamps.json"
            if os.path.exists(side):
                os.remove(side)
            info["applied"] = True
            info["trimmed_from_s"] = round(start, 2)
        else:
            info["reason"] = "ffmpeg produced an empty file — keeping original"
    except Exception as e:
        info["reason"] = f"ffmpeg trim failed ({e}) — analysing the whole clip"
    return info


