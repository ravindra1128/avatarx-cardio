"""Compare capture paths on the ONE metric that is actually binding.

Cross-ROI correlation decides whether beat fusion can match a beat across
regions. Below ~0.2 the fusion both invents beats (regions disagree) and drops
real ones (regions never line up), which shatters the interval series and is
why coverage sits at 0.47 against a 0.50 floor and ACCEPT lands on 2% of scans.
Measured 2026-09-10: lab-rig clips 0.330, webapp phone clips 0.161.

Give it any number of recordings and it reports each one on the same footing,
so a laptop reference and a phone scan of the same person in the same light
can be compared without the confounds the corpus comparison had.

  python scripts/compare_capture.py data/controlled/laptop_ref_*.avi phone.webm

READ ONLY: never writes to, trims, or deletes any input.
"""
from __future__ import annotations
import sys, os, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from capture.video_reader import probe_video, iter_frames          # noqa: E402
from capture.face_tracking import FaceTracker                      # noqa: E402
from preprocessing.roi import ROI_NAMES, mean_rgb, roi_integrity   # noqa: E402
from rppg.pos import pos_pulse                                     # noqa: E402

BAND = (0.7, 3.0)
# Reference points measured on the corpus, so a new recording can be placed.
BENCH = {"lab rig (demo_*.avi)": 0.330, "phone (webapp clips)": 0.161}


def _peak(x, fps):
    """(in-band SNR dB, peak bpm) of one ROI's pulse waveform."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 128:
        return np.nan, np.nan
    f = np.fft.rfftfreq(x.size, 1.0 / fps)
    p = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(x.size))) ** 2
    band = (f >= BAND[0]) & (f <= BAND[1])
    if not band.any():
        return np.nan, np.nan
    k = int(np.argmax(np.where(band, p, -np.inf)))
    sig = (np.abs(f - f[k]) <= 0.15) & band
    noi = band & ~sig
    if not noi.any() or p[noi].mean() <= 0:
        return np.nan, float(f[k] * 60)
    return float(10 * np.log10(p[sig].mean() / p[noi].mean())), float(f[k] * 60)


def measure(path: str) -> dict:
    meta = probe_video(path)
    fps = meta.measured_fps_mean
    tracker = FaceTracker()
    traces = {r: [] for r in ROI_NAMES}
    areas, frame_area, n = [], None, 0
    for _t, frame in iter_frames(path):
        n += 1
        obs = tracker.process(frame)
        if not obs.found:
            continue
        if frame_area is None:
            frame_area = frame.shape[0] * frame.shape[1]
        box = obs.box or (obs.cx - obs.ax, obs.cy - obs.ay, 2 * obs.ax, 2 * obs.ay)
        areas.append(float(box[2] * box[3]))
        if min(roi_integrity(obs, frame.shape).values()) < 0.50:
            continue
        mr = mean_rgb(frame, obs)
        for r in ROI_NAMES:
            traces[r].append(mr[r])

    waves = {}
    for r in ROI_NAMES:
        a = np.asarray(traces[r], float)
        if a.ndim == 2 and a.shape[0] >= 256:
            waves[r] = pos_pulse(a, fps)
    snr, bpm = [], []
    for w in waves.values():
        s, b = _peak(w, fps)
        if np.isfinite(s):
            snr.append(s); bpm.append(b)
    ks = list(waves); cors = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a = np.asarray(waves[ks[i]], float); b = np.asarray(waves[ks[j]], float)
            m = min(a.size, b.size); a, b = a[:m], b[:m]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 256:
                cors.append(abs(float(np.corrcoef(a[ok], b[ok])[0, 1])))
    return {
        "name": os.path.basename(path),
        "res": f"{meta.width}x{meta.height}",
        "fps": fps,
        "secs": n / fps if fps else float("nan"),
        "mb": os.path.getsize(path) / 1e6,
        "bpp": (os.path.getsize(path) * 8) / (meta.width * meta.height * max(n, 1)),
        "face_frac": (float(np.median(areas)) / frame_area) if areas and frame_area else float("nan"),
        "snr": float(np.median(snr)) if snr else float("nan"),
        "xroi": float(np.median(cors)) if cors else float("nan"),
        "bpm": float(np.median(bpm)) if bpm else float("nan"),
        "bpm_spread": (float(max(bpm) - min(bpm)) if len(bpm) > 1 else float("nan")),
    }


def main(paths):
    if not paths:
        print(__doc__); return 1
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"missing: {p}"); continue
        try:
            rows.append(measure(p))
        except Exception as e:                                    # noqa: BLE001
            print(f"{os.path.basename(p)}: FAILED {type(e).__name__}: {e}")
    if not rows:
        return 1
    print(f"\n{'recording':38} {'res':>10} {'s':>5} {'bpp':>6} {'face%':>6} "
          f"{'SNR dB':>7} {'xROI r':>7} {'bpm':>6} {'bpm spread':>11}")
    for r in rows:
        print(f"{r['name'][:36]:38} {r['res']:>10} {r['secs']:5.0f} {r['bpp']:6.3f} "
              f"{r['face_frac']*100:6.1f} {r['snr']:7.2f} {r['xroi']:7.3f} "
              f"{r['bpm']:6.1f} {r['bpm_spread']:11.1f}")
    print("\nxROI r is the binding metric — reference points from the corpus:")
    for k, v in BENCH.items():
        print(f"   {k:26} {v:.3f}")
    print("\n'bpm spread' is how far the four regions disagree on the pulse.")
    print("A wide spread with a healthy SNR is the fusion failure this is chasing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
