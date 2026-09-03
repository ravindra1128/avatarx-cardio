"""
PPG->ECG sequence decoder (M4.13) — a small windowed neural regressor
(conv-scale: one hidden layer sliding over a +/-0.5 s PPG context per
output sample; ~10k parameters; numpy + manual gradients; CPU-trainable),
trained on the repo's cached MIMIC PERform AF / non-AF paired PPG+ECG
with PARTICIPANT-DISJOINT splits only.

It exists to FAIL well: the falsification report (report.py) compares it
against the identity-template baseline and measures interval-level error.
Nothing here may reach the product (invariant 10).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
from scipy.signal import butter, filtfilt

_REPO = pathlib.Path(__file__).resolve().parents[2]
FS = 125.0
CTX = 64                # +/- samples of PPG context per predicted sample
HID = 48


def perform_subjects(limit_per_class: int = 0) -> list:
    """[(subject_id, is_af, ppg, ecg)] from data_cache, z-normalised."""
    out = []
    for is_af, sub in ((1, "mimic_perform_af_csv"),
                       (0, "mimic_perform_non_af_csv")):
        d = _REPO / "data_cache" / sub
        files = sorted(d.glob("*_data.csv"))
        if limit_per_class:
            files = files[:limit_per_class]
        for f in files:
            arr = np.genfromtxt(f, delimiter=",", names=True)
            ppg = np.asarray(arr["PPG"], float)
            ecg = np.asarray(arr["ECG"], float)
            m = np.isfinite(ppg) & np.isfinite(ecg)
            ppg, ecg = ppg[m], ecg[m]
            if ppg.size < int(60 * FS):
                continue
            out.append((f.stem.replace("_data", ""), is_af,
                        _znorm(ppg), _znorm(ecg)))
    return out


def _znorm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    b, a = butter(2, [0.5 / (FS / 2), 40.0 / (FS / 2)], btype="band")
    x = filtfilt(b, a, x - np.mean(x))
    sd = float(np.std(x)) or 1.0
    return x / sd


def split_subjects(subjects: list, *, seed: int = 7,
                   test_fraction: float = 0.3) -> tuple:
    """Participant-disjoint split by stable-hash QUANTILE: deterministic,
    identity-only, and guaranteed non-empty on both sides for n >= 2
    (a raw hash threshold is lumpy at small n)."""
    keyed = sorted(
        (int(hashlib.sha256(f"{s[0]}|{seed}".encode()).hexdigest(), 16), s)
        for s in subjects)
    n_test = max(1, int(round(test_fraction * len(keyed)))) \
        if len(keyed) >= 2 else 0
    te = [s for _, s in keyed[:n_test]]
    tr = [s for _, s in keyed[n_test:]]
    return tr, te


def _windows(ppg: np.ndarray, ecg: np.ndarray, n: int, rng) -> tuple:
    idx = rng.integers(CTX, ppg.size - CTX - 1, size=n)
    X = np.stack([ppg[i - CTX:i + CTX + 1] for i in idx])
    y = ecg[idx]
    return X, y


def train_decoder(train_subjects: list, *, seed: int = 7,
                  steps: int = 400, batch: int = 256,
                  lr: float = 3e-3) -> dict:
    """One-hidden-layer windowed regressor, Adam, MSE. Returns a JSON-safe
    artifact with the training lineage."""
    rng = np.random.default_rng(seed)
    d_in = 2 * CTX + 1
    W1 = rng.normal(0, 1.0 / np.sqrt(d_in), (d_in, HID))
    b1 = np.zeros(HID)
    W2 = rng.normal(0, 1.0 / np.sqrt(HID), (HID, 1))
    b2 = np.zeros(1)
    mom = {k: 0.0 for k in ("W1", "b1", "W2", "b2")}
    vel = {k: 0.0 for k in ("W1", "b1", "W2", "b2")}
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    for step in range(1, steps + 1):
        s = train_subjects[int(rng.integers(len(train_subjects)))]
        X, y = _windows(s[2], s[3], batch, rng)
        h = np.maximum(X @ params["W1"] + params["b1"], 0.0)
        pred = (h @ params["W2"] + params["b2"]).ravel()
        err = pred - y
        gW2 = h.T @ err[:, None] / batch
        gb2 = np.array([err.mean()])
        dh = (err[:, None] @ params["W2"].T) * (h > 0)
        gW1 = X.T @ dh / batch
        gb1 = dh.mean(0)
        for k, g in (("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2)):
            mom[k] = 0.9 * mom[k] + 0.1 * g
            vel[k] = 0.999 * vel[k] + 0.001 * (g * g)
            params[k] = params[k] - lr * mom[k] / (np.sqrt(vel[k]) + 1e-8)
    return {"decoder_version": "ppg2ecg-mlpwin-v1",
            "fs": FS, "ctx": CTX, "hidden": HID, "seed": seed,
            "steps": steps,
            "train_subjects": sorted(s[0] for s in train_subjects),
            "W1": params["W1"].tolist(), "b1": params["b1"].tolist(),
            "W2": params["W2"].tolist(), "b2": params["b2"].tolist()}


def decode(artifact: dict, ppg: np.ndarray) -> np.ndarray:
    """Generate a synthetic ECG estimate from PPG. RESEARCH ONLY — the
    output is watermarked at every rendering site (invariant 10)."""
    W1 = np.asarray(artifact["W1"])
    b1 = np.asarray(artifact["b1"])
    W2 = np.asarray(artifact["W2"])
    b2 = np.asarray(artifact["b2"])
    ctx = int(artifact["ctx"])
    ppg = np.asarray(ppg, float)
    out = np.zeros(ppg.size)
    if ppg.size < 2 * ctx + 1:
        return out
    sw = np.lib.stride_tricks.sliding_window_view(ppg, 2 * ctx + 1)
    h = np.maximum(sw @ W1 + b1, 0.0)
    out[ctx:ctx + sw.shape[0]] = (h @ W2 + b2).ravel()
    return out


def synth_pairs(n_subjects: int = 6, *, seed: int = 3, af_fraction: float = 0.5,
                duration_s: float = 120.0) -> list:
    """Synthetic paired PPG/ECG for tests and fixture runs (no MIMIC
    needed): gaussian-QRS ECG (with P and T waves for sinus; no P wave and
    irregular RR for AF) and a smoothed, delayed pulse waveform driven by
    the same beat times."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / FS)
    out = []
    for i in range(n_subjects):
        is_af = 1 if i < n_subjects * af_fraction else 0
        rr = []
        cur = 0.8
        total = 0.0
        while total < duration_s:
            if is_af:
                cur = float(np.clip(rng.normal(0.75, 0.18), 0.35, 1.6))
            else:
                cur = float(np.clip(0.85 + 0.05 * np.sin(total / 4.0)
                                    + rng.normal(0, 0.02), 0.6, 1.2))
            rr.append(cur)
            total += cur
        beats = np.cumsum(rr)
        beats = beats[beats < duration_s - 1.0]
        ecg = np.zeros_like(t)
        ppg = np.zeros_like(t)
        for bt in beats:
            ecg += 1.4 * np.exp(-0.5 * ((t - bt) / 0.012) ** 2)      # QRS
            ecg += 0.3 * np.exp(-0.5 * ((t - bt - 0.24) / 0.05) ** 2)  # T
            if not is_af:
                ecg += 0.30 * np.exp(-0.5 * ((t - bt + 0.16) / 0.04) ** 2)  # P
            ppg += np.exp(-0.5 * ((t - bt - 0.25) / 0.12) ** 2)
            ppg += 0.3 * np.exp(-0.5 * ((t - bt - 0.55) / 0.10) ** 2)
        ecg += rng.normal(0, 0.03, t.size)
        ppg += rng.normal(0, 0.02, t.size)
        out.append((f"synth{i:02d}_{'af' if is_af else 'naf'}", is_af,
                    _znorm(ppg), _znorm(ecg)))
    return out
