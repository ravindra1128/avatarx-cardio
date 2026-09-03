"""
Synthetic face-video generator (T3).

Draws a skin-tone ellipse "face" whose colour is modulated by a pulse
waveform built from a KNOWN RR series, writes a real video file (FFV1
lossless when available, MJPG fallback), and writes the ground truth beside
it as <video>.truth.json.

Everything produced here is an INTERFACE PROOF: it exercises decode ->
track -> ROI -> rPPG -> beats against known truth. It is not evidence about
accuracy on human skin, and no number derived from these videos may be
quoted as a performance claim (see README).

Usage:
  python3 scripts/make_synth_video.py <out_dir>     # standard v0.1 set
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

try:
    import cv2
except ImportError:                                   # keep module importable
    cv2 = None

SIZE = (320, 240)                                     # (width, height)
# RGB skin base and per-channel pulse gains: green absorbs most, red least —
# the rough haemoglobin structure POS/CHROM assume.
SKIN_RGB = np.array([180.0, 140.0, 95.0])
PULSE_GAIN_RGB = np.array([0.3, 1.0, 0.6]) * 3.0      # uint8 levels at peak
PIXEL_NOISE_SD = 2.0                                  # dithers quantisation
BACKGROUND = 120.0
# v0.6 flutter: the breathing channel. A neutral-grey bar (r == g == b is
# excluded from the skin mask by construction) in rows the face ellipse
# never reaches, so adding breathing cannot move the face box or an ROI.
TORSO_GREY = 200.0
TORSO_BAR_ROWS = 20                                   # ellipse ends 24 above
RR_BAND_BRPM_SYNTH = (8.0, 25.0)                      # rppg.respiration band


def make_rr(kind: str, duration_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "sinus":
        rr = np.clip(rng.normal(0.85, 0.03, int(duration_s / 0.5)), 0.5, 1.4)
    elif kind == "af":
        rr = np.clip(rng.normal(0.62, 0.17, int(duration_s / 0.28)), 0.28, 1.35)
    elif kind == "rsa":
        # respiratory sinus arrhythmia (hard negative): slow sinusoidal RR
        # modulation at the breathing rate, ~8% depth
        rr_list, t = [], 0.0
        while t <= duration_s:
            r = 0.85 * (1.0 + 0.08 * np.sin(2 * np.pi * 0.25 * t)) \
                + rng.normal(0, 0.01)
            rr_list.append(r)
            t += r
        rr = np.asarray(rr_list)
    else:
        raise ValueError(f"unknown rhythm kind {kind!r}")
    keep = np.cumsum(rr) <= duration_s
    return rr[keep]


# v0.4-vascular: the beat-normalized morphology model. shape=None keeps
# the historical single systolic Gaussian EXACTLY (every pre-vascular
# fixture and truth sidecar is unchanged); the extra terms exist so a
# latent stiffness variable can move reflection/notch morphology the way
# arteries do (scripts/make_synth_vascular.py owns the coupling).
DEFAULT_PULSE_SHAPE = {
    "peak_frac": 0.18, "sigma": 0.075,          # systolic wave
    "refl_amp": 0.0, "refl_frac": 0.48, "refl_sigma": 0.10,   # reflection
    "notch_depth": 0.0, "notch_frac": 0.42, "notch_sigma": 0.05,  # notch
}


def beat_waveform(t: np.ndarray, shape: dict | None = None) -> np.ndarray:
    """Waveform on beat-normalized time t in [0,1): systolic Gaussian,
    minus a dicrotic-notch dip, plus a reflected diastolic wave."""
    s = dict(DEFAULT_PULSE_SHAPE, **(shape or {}))
    w = np.exp(-((t - s["peak_frac"]) ** 2) / (2 * s["sigma"] ** 2))
    if s["refl_amp"]:
        w = w + s["refl_amp"] * np.exp(
            -((t - s["refl_frac"]) ** 2) / (2 * s["refl_sigma"] ** 2))
    if s["notch_depth"]:
        w = w - s["notch_depth"] * np.exp(
            -((t - s["notch_frac"]) ** 2) / (2 * s["notch_sigma"] ** 2))
    return np.clip(w, 0.0, None)


def pulse_series(rr_s: np.ndarray, fps: float,
                 deficit_drop_short_ms: float = 0.0,
                 shape: dict | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(per-frame pulse, VISIBLE peak times, ALL beat onset times).

    `deficit_drop_short_ms` simulates PULSE DEFICIT: a beat ending an RR
    shorter than this ejects too little volume to be seen — the beat exists
    on the ECG (its onset stays in the third return value) but produces no
    optical pulse and no visible peak.

    `shape` (v0.4-vascular) parameterizes the per-beat morphology via
    beat_waveform; None reproduces the historical single Gaussian bit for
    bit. The recorded peak time stays the systolic peak location.
    """
    onsets = np.concatenate([[0.0], np.cumsum(rr_s)])
    n = int(onsets[-1] * fps) + 1
    pulse = np.zeros(n)
    peaks = []
    peak_frac = dict(DEFAULT_PULSE_SHAPE, **(shape or {}))["peak_frac"]
    for k, (a, b) in enumerate(zip(onsets, onsets[1:])):
        if deficit_drop_short_ms and (b - a) * 1000.0 < deficit_drop_short_ms:
            continue                              # deficit: no optical pulse
        i0, i1 = int(a * fps), int(b * fps)
        if i1 - i0 > 2:
            t = np.linspace(0, 1, i1 - i0, endpoint=False)
            pulse[i0:i1] = beat_waveform(t, shape)
            peaks.append(a + peak_frac * (b - a))
    return pulse, np.asarray(peaks), onsets


def synth_video(path: str, kind: str = "sinus", fps: float = 30.0,
                duration_s: float = 20.0, lux_scale: float = 1.0,
                face: bool = True, jitter_px: float = 0.0,
                deficit_drop_short_ms: float = 0.0,
                seed: int = 0, codec: str = "FFV1",
                rr_override=None, pulse_shape: dict | None = None,
                amplitude_envelope=None,
                optics_gamma_envelope=None,
                torso_respiration: dict | None = None,
                skin_rgb=None) -> dict:
    """Write a synthetic face video + truth sidecar; return the truth dict.
    `rr_override` (v0.4): an explicit RR series in seconds — the whole
    pipeline downstream of make_rr is RR-driven, so this is the sanctioned
    hook for non-stationary HR fixtures (recovery decays); pass a
    descriptive `kind` label, it is recorded verbatim in the truth.
    `pulse_shape` (v0.4-vascular): morphology params for pulse_series,
    recorded verbatim in the truth; None = historical waveform.
    `amplitude_envelope` (v0.5 vasotone): per-frame multiplier on the
    optical pulse — the sanctioned hook for a latent TONE variable
    (vasoconstriction shrinks pulse amplitude, vasomotion wobbles it).
    `optics_gamma_envelope` (v0.5): per-frame gamma exponent applied to
    the WHOLE frame — a nonlinear optics perturbation for the
    null_optics arm (a pure brightness scale would leave AC/DC
    untouched; real exposure/AWB shifts do not). Both recorded verbatim
    in the truth; None = historical output, bit for bit.
    `torso_respiration` (v0.6 flutter): {"brpm": float, "bob_px": float}
    draws a NEUTRAL-GREY shoulders bar in the bottom rows whose top edge
    rises and falls at the breathing rate — the only camera-observable
    respiration in this generator (breathing moves the chest, not face
    colour; rppg/respiration.py reads exactly this band). Grey is
    load-bearing: the skin-segmentation tracker keys on r>g>b, so a
    neutral bar can never join the face mask, and the bar stays strictly
    BELOW the face ellipse. Needed because the flutter track's
    hyper-regularity family tests RSA coupling, which is undefined
    without a respiration channel. None = historical output, bit for
    bit."""
    if cv2 is None:
        raise RuntimeError("opencv (cv2) is required to write synthetic video")
    rng = np.random.default_rng(seed + 1000)
    if rr_override is not None:
        rr = np.asarray(rr_override, float)
        rr = rr[np.cumsum(rr) <= duration_s]
    else:
        rr = make_rr(kind, duration_s, seed)
    pulse, peaks, onsets = pulse_series(rr, fps, deficit_drop_short_ms,
                                        shape=pulse_shape)
    w, h = SIZE
    n = int(duration_s * fps)
    pulse = np.pad(pulse, (0, max(0, n - pulse.size)))[:n]
    if amplitude_envelope is not None:
        amp_env = np.asarray(amplitude_envelope, float)
        if amp_env.size != n or not np.all(np.isfinite(amp_env)) or \
                np.any(amp_env < 0):
            raise ValueError("amplitude_envelope must be a finite "
                             f"non-negative array of length {n}")
        pulse = pulse * amp_env
    else:
        amp_env = None
    if optics_gamma_envelope is not None:
        gam_env = np.asarray(optics_gamma_envelope, float)
        if gam_env.size != n or not np.all(np.isfinite(gam_env)) or \
                np.any(gam_env <= 0):
            raise ValueError("optics_gamma_envelope must be a finite "
                             f"positive array of length {n}")
    else:
        gam_env = None
    if torso_respiration is not None:
        tr = dict(torso_respiration)
        brpm, bob_px = float(tr["brpm"]), float(tr.get("bob_px", 6.0))
        if not (RR_BAND_BRPM_SYNTH[0] <= brpm <= RR_BAND_BRPM_SYNTH[1]):
            raise ValueError(
                f"torso_respiration brpm {brpm} outside the validated "
                f"resting band {RR_BAND_BRPM_SYNTH} — a fixture outside "
                "the band the reader validates is not a fixture")
        if bob_px <= 0:
            raise ValueError("torso_respiration bob_px must be positive")
        torso_top = (h - TORSO_BAR_ROWS
                     + bob_px * np.sin(2 * np.pi * (brpm / 60.0)
                                       * np.arange(n) / fps))
    else:
        torso_top = None

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    used_codec = codec
    if not writer.isOpened():
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        used_codec = "MJPG"
    if not writer.isOpened():
        raise RuntimeError("no usable video codec (tried FFV1, MJPG)")

    # smooth handheld jitter
    if jitter_px > 0:
        walk = np.cumsum(rng.normal(0, 1, (n, 2)), axis=0)
        k = max(int(fps * 0.5), 1)
        kern = np.ones(k) / k
        walk = np.stack([np.convolve(walk[:, i], kern, "same") for i in range(2)], 1)
        walk = jitter_px * walk / (np.abs(walk).max() + 1e-9)
    else:
        walk = np.zeros((n, 2))

    cx0, cy0, ax, ay = w // 2, h // 2, int(w * 0.22), int(h * 0.40)
    drift = 1.0 + 0.04 * np.sin(2 * np.pi * 0.1 * np.arange(n) / fps)
    # v0.7: `skin_rgb` (a Fitzpatrick-group base colour) is the first
    # OPTICAL counterpart of the skin-tone label. The pulse's optical
    # amplitude scales with the skin's green reflectance relative to the
    # historical base — a crude melanin model (more pigment, less
    # remitted light, smaller AC), disclosed as such; None reproduces the
    # historical output bit for bit.
    skin_base = (SKIN_RGB if skin_rgb is None
                 else np.asarray(skin_rgb, float))
    skin_gain = PULSE_GAIN_RGB * float(skin_base[1] / SKIN_RGB[1])

    for i in range(n):
        frame = np.full((h, w, 3), BACKGROUND * lux_scale * drift[i], float)
        if face:
            rgb = (skin_base - pulse[i] * skin_gain) * lux_scale * drift[i]
            bgr = tuple(float(v) for v in rgb[::-1])
            cv2.ellipse(frame,
                        (int(cx0 + walk[i, 0]), int(cy0 + walk[i, 1])),
                        (ax, ay), 0, 0, 360, bgr, thickness=-1)
        if torso_top is not None:
            # neutral grey (r == g == b): outside the skin mask by
            # construction, and below the ellipse so no ROI ever sees it
            y0 = int(round(torso_top[i]))
            frame[max(y0, 0):, :, :] = TORSO_GREY * lux_scale * drift[i]
        frame += rng.normal(0, PIXEL_NOISE_SD, frame.shape)
        if gam_env is not None:
            # nonlinear per-frame optics: gamma on normalized intensity
            # (a pure brightness scale would cancel in AC/DC; this does
            # not — the null_optics arm needs a REAL optics confound)
            frame = 255.0 * np.power(np.clip(frame, 0, 255) / 255.0,
                                     gam_env[i])
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()

    truth = {
        "skin_rgb": [float(v) for v in skin_base],
        "kind": kind, "fps": fps, "duration_s": duration_s,
        "codec_requested": codec, "codec_used": used_codec,
        "lux_scale": lux_scale, "face": face, "jitter_px": jitter_px,
        "deficit_drop_short_ms": deficit_drop_short_ms,
        "pulse_shape": (dict(pulse_shape) if pulse_shape else None),
        "amplitude_envelope": (
            [round(float(x), 5) for x in amp_env]
            if amplitude_envelope is not None else None),
        "optics_gamma_envelope": (
            [round(float(x), 5) for x in gam_env]
            if gam_env is not None else None),
        "torso_respiration": (dict(torso_respiration)
                              if torso_respiration else None),
        "seed": seed, "rr_s": rr.tolist(),
        "peak_times_s": peaks.tolist(),
        # every beat exists on the ECG even when pulse deficit hides it
        # optically; R-peak precedes ejection onset by ~20 ms here
        "rpeaks_s": (onsets - 0.02).tolist(),
        "note": "synthetic interface-proof video; not evidence about humans",
    }
    with open(path + ".truth.json", "w") as f:
        json.dump(truth, f, indent=2)
    return truth


def main(out_dir: str) -> None:
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    specs = [
        ("sinus_30fps", dict(kind="sinus", fps=30.0)),
        ("sinus_60fps", dict(kind="sinus", fps=60.0)),
        ("af_30fps", dict(kind="af", fps=30.0)),
        ("af_60fps", dict(kind="af", fps=60.0)),
        ("dark_30fps", dict(kind="sinus", fps=30.0, lux_scale=0.15)),
        ("noface_30fps", dict(kind="sinus", fps=30.0, face=False)),
    ]
    for name, kw in specs:
        p = str(d / f"{name}.avi")
        t = synth_video(p, duration_s=20.0, seed=7, **kw)
        print(f"wrote {p} ({t['codec_used']}, {len(t['peak_times_s'])} beats)")



# ---------------------------------------------------------------------------
# Portrait-textured synthetic video (v0.1.1): a REAL human face photo (the
# public-domain Grace Hopper portrait shipped with matplotlib) whose skin
# pixels are modulated by a known pulse. It exercises the real-face path
# (YuNet detection, landmark ROIs) that the ellipse cannot; the pulse is
# still synthetic and clearly labelled as such.
# ---------------------------------------------------------------------------
def portrait_path():
    """Locate matplotlib's public-domain sample portrait WITHOUT importing
    matplotlib (its import runs font-config parsing we do not need)."""
    import importlib.util
    spec = importlib.util.find_spec("matplotlib")
    if spec is None or not spec.origin:
        return None
    p = pathlib.Path(spec.origin).parent / "mpl-data" / "sample_data" \
        / "grace_hopper.jpg"
    return str(p) if p.exists() else None


def portrait_available() -> bool:
    return portrait_path() is not None


def synth_portrait_video(path: str, kind: str = "sinus", fps: float = 30.0,
                         duration_s: float = 20.0, seed: int = 0,
                         jitter_px: float = 0.0, lux_scale: float = 1.0,
                         pulse_rel: float = 0.015, codec: str = "FFV1",
                         size: tuple = (640, 480)) -> dict:
    """Write a portrait-textured synthetic face video + truth sidecar."""
    if cv2 is None:
        raise RuntimeError("opencv (cv2) is required")
    src = portrait_path()
    if src is None:
        raise RuntimeError("portrait sample (matplotlib grace_hopper.jpg) missing")
    img = cv2.imread(src)                                   # 600x512 BGR
    # 4:3 crop around the face, then resize to the capture size
    crop = img[40:424, :, :]                                # 384x512
    w, h = size
    base = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA).astype(float)

    # skin mask: warm-tone rule inside the central face region only
    b, g, r = base[..., 0], base[..., 1], base[..., 2]
    skin = (r > 90) & (r > g + 10) & (g > b) & ((r - b) > 20)
    yy, xx = np.mgrid[0:h, 0:w]
    face_region = ((xx - w * 0.5) / (w * 0.22)) ** 2 + \
                  ((yy - h * 0.42) / (h * 0.30)) ** 2 <= 1.0
    skin = skin & face_region
    weight = skin.astype(float)
    weight = cv2.GaussianBlur(weight, (0, 0), 3)

    rng = np.random.default_rng(seed + 2000)
    rr = make_rr(kind, duration_s, seed)
    pulse, peaks, onsets = pulse_series(rr, fps)
    n = int(duration_s * fps)
    pulse = np.pad(pulse, (0, max(0, n - pulse.size)))[:n]

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    used = codec
    if not writer.isOpened():
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        used = "MJPG"
    if not writer.isOpened():
        raise RuntimeError("no usable video codec")

    if jitter_px > 0:
        walk = np.cumsum(rng.normal(0, 1, (n, 2)), axis=0)
        k = max(int(fps * 0.5), 1)
        walk = np.stack([np.convolve(walk[:, i], np.ones(k) / k, "same")
                         for i in range(2)], 1)
        walk = jitter_px * walk / (np.abs(walk).max() + 1e-9)
    else:
        walk = np.zeros((n, 2))
    gains = np.array([0.6, 1.0, 0.3]) * pulse_rel               # BGR
    drift = 1.0 + 0.03 * np.sin(2 * np.pi * 0.1 * np.arange(n) / fps)
    for i in range(n):
        mod = 1.0 - pulse[i] * weight[..., None] * gains[None, None, :]
        frame = base * mod * (lux_scale * drift[i])
        if jitter_px > 0:
            M = np.float32([[1, 0, walk[i, 0]], [0, 1, walk[i, 1]]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        frame += rng.normal(0, PIXEL_NOISE_SD, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()

    truth = {
        "kind": kind, "fps": fps, "duration_s": duration_s, "codec_used": used,
        "lux_scale": lux_scale, "jitter_px": jitter_px, "seed": seed,
        "pulse_rel": pulse_rel, "source": "portrait (grace_hopper.jpg, public domain)",
        "rr_s": rr.tolist(), "peak_times_s": peaks.tolist(),
        "rpeaks_s": (onsets - 0.02).tolist(),
        "note": "real face TEXTURE, synthetic pulse; interface proof only",
    }
    with open(path + ".truth.json", "w") as f:
        json.dump(truth, f, indent=2)
    return truth


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synth_videos")
