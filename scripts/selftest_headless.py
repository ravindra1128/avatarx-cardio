"""
Headless end-to-end self-test of the LIVE demo through a REAL browser.

This is the strongest automatable proof of the consumer journey: it drives
the actual client page (getUserMedia -> live video -> canvas capture ->
POST frames -> countdown -> finish -> results) in a real, un-throttled
headless Chrome, with a FAKE CAMERA fed from a generated video file. The
page takes the ordinary camera path — there is no test-source shortcut and
no mocked result; the server runs the real production pipeline and the
outcome is read back from the server.

Requirements: Google Chrome (any recent version) + opencv. No extra Python
packages, no ffmpeg. Chrome's --use-file-for-fake-video-capture reads a Y4M
file, which we generate here from the portrait-textured pulse frames.

Run:  python3 scripts/selftest_headless.py
Exit code 0 on a verified DONE/ACCEPT run, non-zero otherwise.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import shutil
import subprocess
import tempfile
import time
import urllib.request

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("chrome") or "",
]


def find_chrome():
    for c in CHROME_CANDIDATES:
        if c and pathlib.Path(c).exists():
            return c
    return None


def write_y4m(path: str, frames_bgr, fps: int = 30) -> None:
    """Concatenate BGR frames into a Y4M (I420) file for Chrome's fake cam."""
    h, w = frames_bgr[0].shape[:2]
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C420jpeg\n".encode())
        for fr in frames_bgr:
            i420 = cv2.cvtColor(fr, cv2.COLOR_BGR2YUV_I420)   # (h*3/2, w)
            f.write(b"FRAME\n")
            f.write(np.ascontiguousarray(i420).tobytes())


def make_fake_camera_y4m(path: str, seconds: float = 12.0, fps: int = 30,
                         kind: str = "sinus") -> dict:
    """Portrait-textured pulse frames -> Y4M. Returns the truth dict."""
    from scripts.make_synth_video import (portrait_path, make_rr,
                                          pulse_series, PIXEL_NOISE_SD)
    src = portrait_path()
    w, h = 640, 480
    if src is not None:
        img = cv2.imread(src)
        crop = img[40:424, :, :]
        base = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA).astype(float)
        b, g, r = base[..., 0], base[..., 1], base[..., 2]
        skin = (r > 90) & (r > g + 10) & (g > b) & ((r - b) > 20)
        yy, xx = np.mgrid[0:h, 0:w]
        fr = ((xx - w*0.5)/(w*0.22))**2 + ((yy - h*0.42)/(h*0.30))**2 <= 1.0
        weight = cv2.GaussianBlur((skin & fr).astype(float), (0, 0), 3)
    else:                                                    # ellipse fallback
        base = np.full((h, w, 3), 40.0)
        cv2.ellipse(base, (w//2, h//2), (int(w*0.22), int(h*0.34)), 0, 0, 360,
                    (95, 140, 180), -1)
        weight = np.zeros((h, w))
        cv2.ellipse(weight, (w//2, h//2), (int(w*0.20), int(h*0.32)), 0, 0, 360, 1, -1)
        weight = cv2.GaussianBlur(weight, (0, 0), 4)

    rng = np.random.default_rng(202608)
    rr = make_rr(kind, seconds, 4242)
    pulse, peaks, onsets = pulse_series(rr, fps)
    n = int(seconds * fps)
    pulse = np.pad(pulse, (0, max(0, n - pulse.size)))[:n]
    gains = np.array([0.6, 1.0, 0.3]) * 0.02
    drift = 1.0 + 0.03 * np.sin(2*np.pi*0.1*np.arange(n)/fps)
    frames = []
    for i in range(n):
        mod = 1.0 - pulse[i] * weight[..., None] * gains[None, None, :]
        fbgr = base * mod * drift[i] + rng.normal(0, PIXEL_NOISE_SD, base.shape)
        frames.append(np.clip(fbgr, 0, 255).astype(np.uint8))
    write_y4m(path, frames, fps)
    return {"kind": kind, "fps": fps, "seconds": seconds,
            "peak_times_s": peaks.tolist(), "n_frames": n}


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read().decode())


def run_selftest(scan_seconds: float = 8.0, kind: str = "sinus",
                 timeout_s: float = 300.0, keep_open: bool = False) -> dict:
    if cv2 is None:
        raise RuntimeError("opencv required")
    chrome = find_chrome()
    if chrome is None:
        raise RuntimeError("Google Chrome / Chromium not found")

    from app.server import start_server
    work = pathlib.Path(tempfile.mkdtemp(prefix="avx_selftest_"))
    srv, port = start_server(port=0, work_dir=str(work), open_browser=False)
    base = f"http://127.0.0.1:{port}"
    y4m = work / "fakecam.y4m"
    print(f"generating fake-camera Y4M ({kind}) …")
    # long enough for readiness (window + hold + countdown ≈ 15 s) plus the
    # scan plus slack, so Chrome never loops the clip mid-scan (a loop seam is
    # a pulse-phase jump that a real camera never produces)
    truth = make_fake_camera_y4m(str(y4m), seconds=max(scan_seconds + 30, 20),
                                 kind=kind)

    prof = work / "chrome-profile"
    url = f"{base}/?scan={scan_seconds}&autostart=1"      # REAL camera path
    args = [chrome, "--headless=new", "--no-sandbox", "--no-first-run",
            "--disable-gpu", "--autoplay-policy=no-user-gesture-required",
            "--use-fake-ui-for-media-stream",             # auto-grant camera
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={y4m}",
            f"--user-data-dir={prof}",
            "--remote-debugging-port=0",
            url]
    print("launching headless Chrome on the REAL getUserMedia path …")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    # the page auto-starts only on a click; drive Start by loading a variant
    # that autostarts. We poll the server for a session reaching a verdict.
    result = None
    t0 = time.time()
    try:
        while time.time() - t0 < timeout_s:
            time.sleep(1.0)
            try:
                rows = _get(base, "/api/sessions")["sessions"]
            except Exception:
                continue
            done = [r for r in rows if r["state"] in ("done", "failed")]
            if done:
                sid = done[-1]["id"]
                result = _get(base, f"/api/status?session={sid}")
                break
            if rows:
                print(f"  session {rows[-1]['id']}: {rows[-1]['state']}")
    finally:
        if not keep_open:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            srv.shutdown()
    if result is None:
        raise TimeoutError(
            "no session reached a verdict. Known headless-Chrome flake: the "
            "fake camera sometimes stops delivering frames (occasionally "
            "before the first frame) — retry once before investigating. "
            f"Diagnose the session with: python3 scripts/why_not_ready.py "
            f"<session dir under {work}> — zero evaluations means frames "
            "never arrived (browser side); a start_scan event followed by "
            "silence means the stream died mid-scan.")
    return {"status": result, "truth": truth, "work_dir": str(work)}


def run_session_selftest(timeout_s: float = 420.0) -> dict:
    """v0.4: drive the full THREE-PHASE flow (safety screen -> rest scan
    -> guided activity -> transition -> recovery scan -> session report)
    through the real client in headless Chrome. The fake camera cannot
    perform sit-to-stands, so the honest outcome is a fail-closed
    non-compliant session WITH resting vitals — completing the flow is
    the assertion, not ACCEPT."""
    import os
    if cv2 is None:
        raise RuntimeError("opencv required")
    chrome = find_chrome()
    if chrome is None:
        raise RuntimeError("Google Chrome / Chromium not found")
    os.environ["AVATARX_THREE_PHASE"] = "1"
    from app.server import start_server
    work = pathlib.Path(tempfile.mkdtemp(prefix="avx_selftest_v_"))
    srv, port = start_server(port=0, work_dir=str(work),
                             open_browser=False)
    base = f"http://127.0.0.1:{port}"
    y4m = work / "fakecam.y4m"
    print("generating fake-camera Y4M (session mode) …")
    make_fake_camera_y4m(str(y4m), seconds=70.0, kind="sinus")
    prof = work / "chrome-profile"
    url = f"{base}/?autostart=1&rest=8&recovery=12&act=6"
    args = [chrome, "--headless=new", "--no-sandbox", "--no-first-run",
            "--disable-gpu", "--autoplay-policy=no-user-gesture-required",
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={y4m}",
            f"--user-data-dir={prof}", "--remote-debugging-port=0", url]
    print("launching headless Chrome for the three-phase session …")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    result = None
    t0 = time.time()
    try:
        while time.time() - t0 < timeout_s:
            time.sleep(1.0)
            try:
                vrows = _get(base, "/api/sessions").get("vsessions") or []
            except Exception:
                continue
            if vrows:
                vs = vrows[-1]
                print(f"  vsession {vs['id']}: {vs['phase']}")
                if vs["phase"] == "done":
                    result = _get(base, "/api/vsession/status?session="
                                  + vs["id"])
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        srv.shutdown()
        os.environ.pop("AVATARX_THREE_PHASE", None)
    if result is None:
        raise TimeoutError("three-phase session never completed — same "
                           "fake-camera flake caveats as the single-scan "
                           f"selftest; inspect {work}")
    return {"status": result, "work_dir": str(work)}


if __name__ == "__main__":
    if "--session" in sys.argv:
        out = run_session_selftest()
        r = out["status"].get("result") or {}
        rep = r.get("report_html", "")
        ok = (r.get("outcome") in ("ACCEPT", "REPEAT_SCAN", "NO_RESULT")
              and "AvatarX Recovery Session Report" in rep
              and "<svg" not in rep
              and r.get("fitness_category") is None)
        print(json.dumps({"outcome": r.get("outcome"),
                          "hr_rest_bpm": r.get("hr_rest_bpm"),
                          "compliance": (r.get("compliance")
                                         or {}).get("verdict"),
                          "no_read_reasons": r.get("no_read_reasons"),
                          "user_facing_text": r.get("user_facing_text")},
                         indent=2, default=str))
        print("work_dir:", out["work_dir"])
        print("three-phase session:", "OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    kind = sys.argv[1] if len(sys.argv) > 1 else "sinus"
    scan_s = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    out = run_selftest(kind=kind, scan_seconds=scan_s)
    st = out["status"]
    print("\nFINAL:", st["state"])
    r = st.get("result") or {}
    print(json.dumps({"outcome": r.get("outcome"),
                      "predicted_class": r.get("predicted_class"),
                      "pulse_bpm": r.get("pulse_bpm"),
                      "signal_quality": r.get("signal_quality"),
                      "no_read_reasons": r.get("no_read_reasons"),
                      "sqi_components": r.get("sqi_components"),
                      "usable_beats": r.get("usable_beats"),
                      "user_facing_text": r.get("user_facing_text"),
                      "capture": r.get("capture"),
                      "provenance": r.get("provenance"),
                      "error": st.get("error")}, indent=2, default=str))
    print("work_dir:", out["work_dir"])
    ok = st["state"] == "done" and (r.get("outcome") in
                                    ("ACCEPT", "REPEAT_SCAN", "NO_RESULT"))
    # v0.2.1/v0.3: the unified report must render end-to-end, and the
    # default surface is FINDINGS-ONLY (no waveform of any kind)...
    rep = r.get("report_html", "")
    rep_ok = ("AvatarX Cardiac Rhythm Scan Report" in rep
              and "FINDINGS" in rep
              and "<polyline" not in rep and "<svg" not in rep)
    print("report render:", "OK" if rep_ok else "MISSING")
    # ...including a NO_RESULT run (R6): a too-dark clip through the real
    # CLI --report path must still yield the full report with reasons.
    import subprocess, tempfile
    d2 = tempfile.mkdtemp(prefix="avx_selftest_noresult_")
    dark = str(pathlib.Path(d2) / "dark.avi")
    from scripts.make_synth_video import synth_video as _sv
    _sv(dark, kind="sinus", fps=30.0, duration_s=15.0, seed=9,
        lux_scale=0.15)
    out_html = str(pathlib.Path(d2) / "report.html")
    pr = subprocess.run([sys.executable,
                         str(pathlib.Path(__file__).resolve().parents[1]
                             / "cli.py"), "process", dark,
                         "--report", out_html],
                        capture_output=True, text=True, timeout=300)
    nr_ok = False
    if pr.returncode == 0 and pathlib.Path(out_html).exists():
        h = pathlib.Path(out_html).read_text()
        nr_ok = ("Why no result" in h and "<polyline" not in h)
    print("NO_RESULT report:", "OK" if nr_ok else "FAILED")
    sys.exit(0 if (ok and rep_ok and nr_ok) else 1)
