"""Evidence for the Arterial Stiffness investigation of 2026-09-20. Re-runnable.

Every number quoted in that investigation comes from one of these subcommands, on the
cache built by `scripts/stiffness_repeatability.py cache` (the retained phone clips run
through the service's own job) and on MIMIC PERform under data_cache/.

  null       the DEPLOYED morphology path fed a shapeless sinusoid + noise: does it still
             report a dicrotic notch / a crest time?            (is the card a measurement?)
  harmonics  phase-locked 2nd/3rd harmonic of the pulse, our compressed clip vs the SDK's
             live-frame PPG, against a surrogate null           (is any shape recoverable?)
  sdk        the SDK waveform: band edges, within-scan halves, heart-rate dependence,
             invariance to its derived sample rate              (measured or imposed?)
  mimic      on real fingertip PPG (125 Hz): does the 2-harmonic descriptor track TRUE
             crest time across people, beyond heart rate?       (does it mean anything?)

Nothing here is on the request path and nothing here changes a result.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert, resample_poly, welch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from scripts.stiffness_repeatability import load_cache          # noqa: E402

HALF_BW = 0.35            # Hz either side of k*f0: wide enough for resting HRV


def _bp(x, fs, lo, hi, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [max(lo, 0.05) / nyq, min(hi, 0.99 * nyq) / nyq], btype="band")
    return filtfilt(b, a, x)


def refine_f0(x, fs, f_ref, tol=0.12):
    f, p = welch(x - x.mean(), fs=fs, nperseg=min(len(x), int(fs * 20)), noverlap=int(fs * 15))
    m = (f >= (1 - tol) * f_ref) & (f <= (1 + tol) * f_ref)
    return float(f[m][np.argmax(p[m])]) if m.any() else f_ref


def locked_harmonic(x, fs, f0, k, shift=0):
    """c_k = < z_k conj(u1)^k > / <|z1|>: harmonic k relative to the fundamental's own
    phase. Noise in band k is uncorrelated with u1^k, so it adds variance and no bias;
    no beat is detected and nothing is aligned. `shift` rolls the fundamental against
    the harmonic (the surrogate null: spectra kept, phase lock destroyed)."""
    edge = int(1.5 * fs)
    sl = slice(edge, x.size - edge)
    z1 = hilbert(_bp(x, fs, f0 - HALF_BW, f0 + HALF_BW))[sl]
    a1 = np.roll(np.abs(z1), shift)
    u1 = np.roll(z1 / np.maximum(np.abs(z1), 1e-12), shift)
    zk = hilbert(_bp(x, fs, k * (f0 - HALF_BW), k * (f0 + HALF_BW)))[sl]
    w = a1 / a1.sum()
    return complex(np.sum(w * zk * np.conj(u1) ** k) / np.sum(w * a1))


def crest_fraction_2h(c2, n=720):
    """(foot -> peak) / cycle of cos(th) + |c2| cos(2 th + arg c2)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    y = np.cos(th) + abs(c2) * np.cos(2 * th + np.angle(c2))
    return int(np.argmax(np.roll(y, -int(np.argmin(y))))) / float(n)


def _wrap(d):
    return abs((d + 180.0) % 360.0 - 180.0)


def _circ(ph_deg):
    R = abs(np.mean(np.exp(1j * np.radians(ph_deg))))
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-12))))), float(R)


def _repeat_pairs(rows, vals, max_min=20.0):
    out = []
    for (a, va), (b, vb) in itertools.combinations(list(zip(rows, vals)), 2):
        dt = abs((datetime.fromisoformat(a["t"]) - datetime.fromisoformat(b["t"])).total_seconds())
        if dt <= max_min * 60 and va is not None and vb is not None:
            out.append(_wrap(va - vb))
    return out


# ------------------------------------------------------------------ null
def cmd_null(args):
    from features.hemodynamics import stiffness_contour
    from features.pulse_morphology import (MORPH_BAND_HZ, _beat_pairs, _beat_windows,
                                           beat_morphology, ensemble_beat)
    from inference.evidence import capture_segments
    from preprocessing.roi import ROI_NAMES
    from rppg.pos import orient_rois_consistently, pos_pulse
    rng = np.random.default_rng(7)

    def contour(row, snr_db):
        A, fps = row["arrays"], float(row["fps"])
        ts, bt = A["ts"], A["beat_t"]
        pairs = _beat_pairs(bt, A["beat_conf"], float(row.get("min_conf") or 0.0))
        hi = min(MORPH_BAND_HZ[1], 0.45 * fps)
        segs = {r: [] for r in ROI_NAMES}
        for a, e in capture_segments(ts, fps):
            st = ts[a:e]
            if snr_db is None:                                   # the real waveform
                waves = orient_rois_consistently(
                    {r: pos_pulse(A[f"rgb_{r}"][a:e], fps, band=(MORPH_BAND_HZ[0], hi)) for r in ROI_NAMES})
            else:                                                # cos with crests ON the real beats + noise
                ph = np.interp(st, bt, 2 * np.pi * np.arange(bt.size), left=np.nan, right=np.nan)
                bad = ~np.isfinite(ph)
                ph[bad] = 2 * np.pi * (st[bad] - bt[0]) / np.median(np.diff(bt))
                s = np.cos(ph)
                waves = {}
                for r in ROI_NAMES:
                    nz = _bp(rng.standard_normal(st.size), fps, 0.5, hi)
                    waves[r] = s + nz * np.std(s) / np.std(nz) / (10 ** (snr_db / 20.0))
            inseg = [(t0, t1) for t0, t1 in pairs if t0 >= st[0] and t1 <= st[-1]]
            for r in ROI_NAMES:
                for f1, f2 in _beat_windows(waves[r], st, inseg):
                    segs[r].append(waves[r][f1:f2])
        feats = []
        for r in ROI_NAMES:
            ens, dur = ensemble_beat(segs[r], fps)
            f = None if ens is None else beat_morphology(ens, ens.size / dur, native_fs=fps)
            if f is not None:
                feats.append(f)
        if not feats:
            return None
        c = {k: (float(np.median([f[k] for f in feats if f.get(k) is not None]))
                 if any(f.get(k) is not None for f in feats) else None) for k in feats[0]}
        return stiffness_contour(c)

    rows = [r for r in load_cache() if "beat_t" in r["arrays"] and r["arrays"]["beat_t"].size >= 8]
    print(f"{len(rows)} scans with a beat lattice. Truth for the null input: no notch, crest time = T/2 (~400 ms).\n")
    for label, snr, reps in (("REAL scans", None, 1), ("NULL sinusoid, -5 dB (the scans' SNR)", -5.0, args.reps),
                             ("NULL sinusoid, +20 dB (clean)", 20.0, 1)):
        ri, ct, sc = [], [], []
        for _ in range(reps):
            for r in rows:
                c = contour(r, snr)
                if c is None or c.get("score") is None:
                    continue
                sc.append(c["score"])
                (ri if c.get("raw_name") == "reflection_index" else ct).append(c["raw_value"])
        n = len(sc)
        print(f"{label:<40} n={n:<3} score {np.mean(sc):5.1f} +/- {np.std(sc, ddof=1):4.1f} | "
              f"'dicrotic notch found' {len(ri)}/{n}"
              + (f", 'reflection index' {np.mean(ri):.2f} +/- {np.std(ri, ddof=1):.2f} ({min(ri):.2f}-{max(ri):.2f})" if len(ri) > 1 else "")
              + (f" | crest time {np.mean(ct):.0f} ms ({min(ct):.0f}-{max(ct):.0f})" if ct else ""))


# ------------------------------------------------------------------ harmonics
def _sources(row):
    """{'clip': (x, fs), 'sdk': (x, fs)} on the longest gap-free stretch."""
    from features.pulse_morphology import MORPH_BAND_HZ
    from preprocessing.roi import ROI_NAMES
    from rppg.pos import pos_pulse
    A, out = row["arrays"], {}
    f_ref = float(row["sdk"]["heart_rate_bpm"]) / 60.0
    if "ts" in A:
        ts = A["ts"]
        cuts = np.where(np.diff(ts) > 0.2)[0]
        ed = np.concatenate([[0], cuts + 1, [ts.size]])
        a, e = max(zip(ed[:-1], ed[1:]), key=lambda ab: ab[1] - ab[0])
        fs = (e - a - 1) / (ts[e - 1] - ts[a])
        if e - a >= fs * 25:
            hi = min(MORPH_BAND_HZ[1], 0.45 * fs)
            w = {r: pos_pulse(A[f"rgb_{r}"][a:e], fs, band=(0.5, hi)) for r in ROI_NAMES
                 if np.all(np.isfinite(A[f"rgb_{r}"][a:e]))}
            f0 = refine_f0(np.mean([v / v.std() for v in w.values()], axis=0), fs, f_ref)
            fund = {r: _bp(v, fs, f0 - HALF_BW, f0 + HALF_BW) for r, v in w.items()}
            ref = max(w, key=lambda r: np.var(fund[r]) / np.var(w[r]))
            fused = sum((np.sign(np.corrcoef(fund[r], fund[ref])[0, 1]) or 1.0)
                        * (np.var(fund[r]) / max(np.var(w[r] - fund[r]), 1e-12)) * w[r] / w[r].std() for r in w)
            out["clip"] = (fused, fs, f0)
    if "sdk_ppg" in A and A["sdk_ppg"].size > 600 and row.get("sdk_ppg_fs"):
        x, fs = A["sdk_ppg"].astype(float), float(row["sdk_ppg_fs"])
        out["sdk"] = (x, fs, refine_f0(x, fs, f_ref, tol=0.08))
    return out


def cmd_harmonics(args):
    # Significance from the SHIPPED estimator (features/pulse_harmonics.py), so this
    # report and the sheet's H2 columns can never quote two different tests. Its block
    # test replaced a first-draft surrogate test that misread a metronomic heart.
    from features import pulse_harmonics
    rows = load_cache()
    res = {"clip": [], "sdk": []}
    for r in rows:
        f_ref = float(r["sdk"]["heart_rate_bpm"]) / 60.0
        for name, (x, fs, _f0) in _sources(r).items():
            o = pulse_harmonics.estimate(x, fs, f_ref)
            if o.get("state") == "ok":
                res[name].append((r, o["r2"], o["phi2_deg"], o["p_value"]))
    for name, label in (("clip", "OUR COMPRESSED CLIP (4 regions, SNR-weighted)"), ("sdk", "SDK LIVE-FRAME PPG (sidecar)")):
        L = res[name]
        ph = [x[2] for x in L]
        sd, R = _circ(ph)
        d = _repeat_pairs([x[0] for x in L], ph)
        tested = [x[3] for x in L if x[3] is not None]
        print(f"\n{label}: {len(L)} scans")
        print(f"  2nd harmonic found (block test, p < {pulse_harmonics.DETECTABLE_P}): "
              f"{sum(p < pulse_harmonics.DETECTABLE_P for p in tested)}/{len(tested)} testable, median p {np.median(tested):.2g}")
        print(f"  phi2 across scans: circular SD {sd:.0f} deg, R = {R:.2f}   (uniformly random phases give R ~ {1/np.sqrt(len(L)):.2f})")
        print(f"  r2 {np.mean([x[1] for x in L]):.3f} +/- {np.std([x[1] for x in L], ddof=1):.3f}")
        if d:
            print(f"  repeat pairs <= 20 min: n={len(d)}  |dphi2| mean {np.mean(d):.0f} deg, median {np.median(d):.0f}, max {np.max(d):.0f}")


# ------------------------------------------------------------------ sdk
def cmd_sdk(args):
    rows = [r for r in load_cache() if "sdk" in _sources(r)]
    S = [_sources(r)["sdk"] for r in rows]
    print(f"{len(S)} scans with an SDK PPG")
    grid = np.array([0.6, 1.4, 2.6, 3.6, 4.0, 4.6, 5.0, 6.0, 8.0])
    P = []
    for x, fs, _ in S:
        f, p = welch(x - x.mean(), fs=fs, nperseg=int(fs * 16), noverlap=int(fs * 12))
        P.append(np.interp(grid, f, 10 * np.log10(np.maximum(p / p.max(), 1e-30))))
    print("A. spectrum, dB re peak: " + "  ".join(f"{g:g}Hz {v:.0f}" for g, v in zip(grid, np.median(P, axis=0))))
    d = []
    for x, fs, f0 in S:
        h = x.size // 2
        a = locked_harmonic(x[:h], fs, refine_f0(x[:h], fs, f0, 0.08), 2)
        b = locked_harmonic(x[h:], fs, refine_f0(x[h:], fs, f0, 0.08), 2)
        d.append(_wrap(np.degrees(np.angle(a) - np.angle(b))))
    print(f"B. first vs second half of one scan: |dphi2| mean {np.mean(d):.1f} deg, max {np.max(d):.1f} (0.0 everywhere = synthesised)")
    hr = np.array([60 * f0 for _, _, f0 in S])
    print("C. heart-rate dependence across scans:")
    for k in (2, 3):
        c = np.array([locked_harmonic(x, fs, f0, k) for x, fs, f0 in S])
        for nm, v in ((f"r{k}", np.abs(c)), (f"phi{k}", np.degrees(np.angle(c)))):
            print(f"   {nm:<5} {v.mean():8.3f} +/- {v.std(ddof=1):6.3f}   slope {np.polyfit(hr, v, 1)[0]:+.4f}/bpm   corr with HR {np.corrcoef(hr, v)[0,1]:+.2f}")
    worst = 0.0
    for x, fs, f0 in S:
        base = np.degrees(np.angle(locked_harmonic(x, fs, f0, 2)))
        for s in (0.97, 1.03):
            worst = max(worst, _wrap(np.degrees(np.angle(locked_harmonic(x, fs * s, refine_f0(x, fs * s, f0 * s, 0.08), 2))) - base))
    print(f"D. derived sample rate off by +/-3 %: worst |dphi2| {worst:.2f} deg")


# ------------------------------------------------------------------ mimic
def cmd_mimic(args):
    FS, rng = 125.0, np.random.default_rng(3)

    def windows(path):
        d = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(1,))
        out, step = [], int(60 * FS)
        for s in range(0, d.size - step + 1, step):
            x = d[s:s + step]
            if not np.all(np.isfinite(x)) or np.std(x) < 1e-6:
                continue
            y = _bp(x, FS, 0.5, 8.0)
            pk, _ = find_peaks(y, distance=int(0.4 * FS), prominence=0.5 * np.std(y))
            fr = [(b - (a + int(np.argmin(y[a:b])))) / (b - a) for a, b in zip(pk[:-1], pk[1:])
                  if 0.4 <= (b - a) / FS <= 1.6 and 0.06 <= (b - (a + int(np.argmin(y[a:b])))) / FS <= 0.5]
            if len(fr) < 25:
                continue
            g = resample_poly(_bp(x, FS, 0.7, 4.0), 6, 25)             # the SDK's band, 30 Hz
            nz = _bp(rng.standard_normal(g.size), 30.0, 0.7, 4.0)
            g = g + nz * np.std(g) / np.std(nz) / (10 ** (10.0 / 20.0))  # 10 dB
            f, p = welch(g - g.mean(), fs=30.0, nperseg=600, noverlap=450)
            m = (f >= 0.7) & (f <= 2.5)
            f0 = float(f[m][np.argmax(p[m])])
            out.append((float(np.median(fr)), crest_fraction_2h(locked_harmonic(g, 30.0, f0, 2)), 60 * f0))
        return out

    def resid(y, x):
        return y - np.polyval(np.polyfit(x, y, 1), x)

    base = HERE / "data_cache"
    for label, sub in (("sinus rhythm", "mimic_perform_non_af_csv"), ("AF", "mimic_perform_af_csv")):
        S = [w for w in (windows(f) for f in sorted(glob.glob(str(base / sub / "*_data.csv")))) if len(w) >= 5]
        fr, cf, hr = (np.array([np.median([w[i] for w in s]) for s in S]) for i in (0, 1, 2))
        k = np.array([len(s) for s in S], float)
        grand = np.concatenate([[w[1] for w in s] for s in S]).mean()
        msb = sum(len(s) * (np.mean([w[1] for w in s]) - grand) ** 2 for s in S) / (len(S) - 1)
        msw = sum(((np.array([w[1] for w in s]) - np.mean([w[1] for w in s])) ** 2).sum() for s in S) / (k.sum() - len(S))
        sb2 = max((msb - msw) / ((k.sum() - (k ** 2).sum() / k.sum()) / (len(S) - 1)), 0.0)
        print(f"\nMIMIC PERform, {label}: {len(S)} people, HR {hr.min():.0f}-{hr.max():.0f}  (camera-like: 0.7-4 Hz, 30 Hz, 10 dB)")
        print(f"  2-harmonic index vs TRUE crest fraction, between people: r = {np.corrcoef(cf, fr)[0,1]:+.2f}")
        print(f"  variance of the index explained by heart rate alone: R2 = {np.corrcoef(cf, hr)[0,1]**2:.2f}")
        bs = [np.corrcoef(resid(cf[i], hr[i]), resid(fr[i], hr[i]))[0, 1]
              for i in (rng.integers(0, len(S), len(S)) for _ in range(4000)) if np.unique(i).size >= 5]
        print(f"  PARTIAL r with heart rate removed from both: {np.corrcoef(resid(cf, hr), resid(fr, hr))[0,1]:+.2f}"
              f"   bootstrap 95% CI [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
        print(f"  ICC over 60 s windows: {sb2/(sb2+msw):.2f}  (between-person SD {np.sqrt(sb2):.4f}, within {np.sqrt(msw):.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("null"); n.add_argument("--reps", type=int, default=6); n.set_defaults(fn=cmd_null)
    sub.add_parser("harmonics").set_defaults(fn=cmd_harmonics)
    sub.add_parser("sdk").set_defaults(fn=cmd_sdk)
    sub.add_parser("mimic").set_defaults(fn=cmd_mimic)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
