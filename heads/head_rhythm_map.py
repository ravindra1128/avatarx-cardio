"""
head_rhythm_map (M1.4) — the "inferred ECG" consumer surface, on honest
ground. Renders ONLY MEASURED data as a static SVG: the extracted pulse
waveform with detected-beat ticks (per-beat confidence shading), a
tachogram strip (beat-to-beat interval vs time), and a Poincaré plot
(IBI_n vs IBI_n+1) from the clean runs. NO generated ECG waveform, no
electrical morphology — a camera measures the hemodynamic consequence of
the heartbeat, not its electrical activity (spec B.15). The AF sentence
and star grade are rendered by the app from the sanctioned text path;
this head deliberately returns MEASURED panels only.
"""
from __future__ import annotations

import numpy as np

from datasets.schema import MeasurementClass
from heads.base import EndpointHead, HeadResult, register_head

_ACC = "#2dd4bf"      # matches the app palette
_MUTE = "#94a3b8"
_GRID = "#243046"
_BG = "#0f172a"

W = 520               # viewBox width; panels stack vertically


def _polyline(xs, ys, colour, width=1.5, opacity=1.0) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" opacity="{opacity:.2f}"/>')


def _panel(y0, h, title) -> str:
    return (f'<rect x="0" y="{y0}" width="{W}" height="{h}" fill="{_BG}" '
            f'rx="8"/>'
            f'<text x="8" y="{y0 + 14}" fill="{_MUTE}" font-size="11" '
            f'font-family="sans-serif">{title}</text>')


def _scale(v, lo, hi, out_lo, out_hi):
    v = np.asarray(v, float)
    if hi - lo <= 0:
        return np.full_like(v, (out_lo + out_hi) / 2.0)
    return out_lo + (v - lo) / (hi - lo) * (out_hi - out_lo)


class RhythmMapHead(EndpointHead):
    name = "rhythm_map"
    version = "1.0.0"
    required_inputs = ("runs", "beat_t_s", "beat_confidence", "waveform")

    def run(self, lattice, context: dict) -> HeadResult:
        parts = [f'<svg viewBox="0 0 {W} 330" '
                 f'xmlns="http://www.w3.org/2000/svg" role="img" '
                 f'aria-label="Rhythm Map — measured pulse signals">']
        panels = []

        # ---- panel 1: measured pulse waveform + beat ticks --------------
        wf = (context or {}).get("waveform") or {}
        vals = np.asarray(wf.get("values") or [], float)
        parts.append(_panel(0, 120, "Measured pulse waveform "
                                    "(detected beats ticked)"))
        if vals.size >= 8 and wf.get("fps"):
            fps = float(wf["fps"])
            t0 = float(wf.get("t0_s", 0.0))
            tt = t0 + np.arange(vals.size) / fps
            lo, hi = float(np.min(vals)), float(np.max(vals))
            xs = _scale(tt, tt[0], tt[-1], 6, W - 6)
            ys = _scale(vals, lo, hi, 108, 24)
            parts.append(_polyline(xs, ys, _ACC, 1.6))
            # beat ticks come from the LATTICE (the beat authority), with
            # per-beat confidence shading
            for bt, bc in zip(lattice.beat_t_s, lattice.beat_confidence):
                if tt[0] <= bt <= tt[-1]:
                    x = float(_scale(float(bt), tt[0], tt[-1], 6, W - 6))
                    op = max(0.25, min(1.0, float(bc)))
                    parts.append(f'<line x1="{x:.1f}" y1="108" x2="{x:.1f}" '
                                 f'y2="118" stroke="{_ACC}" stroke-width="2" '
                                 f'opacity="{op:.2f}"/>')
            panels.append("waveform")
        else:
            parts.append(f'<text x="{W/2}" y="66" fill="{_MUTE}" '
                         f'font-size="12" text-anchor="middle" '
                         f'font-family="sans-serif">no waveform '
                         f'available</text>')

        # ---- panel 2: tachogram (beat-to-beat interval strip) ----------
        parts.append(f'<g transform="translate(0,128)">'
                     + _panel(0, 96, "Tachogram — beat-to-beat interval "
                                     "(ms), clean runs"))
        ibis = [np.asarray(r, float) for r in lattice.runs if len(r)]
        if ibis:
            allv = np.concatenate(ibis)
            lo, hi = float(np.min(allv)) - 40, float(np.max(allv)) + 40
            x_cursor = 6.0
            total = sum(len(r) for r in ibis)
            span = (W - 12) / max(total, 1)
            for r, rc in zip(ibis, lattice.run_confidences):
                xs = x_cursor + np.arange(len(r)) * span
                ys = _scale(r, lo, hi, 88, 20)
                rcm = float(np.mean(np.asarray(rc, float))) \
                    if np.asarray(rc).size else 0.8
                op = max(0.35, min(1.0, rcm))
                parts.append(_polyline(xs, ys, _ACC, 1.4, op))
                for x, y in zip(xs, ys):
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.6" '
                                 f'fill="{_ACC}" opacity="{op:.2f}"/>')
                x_cursor = float(xs[-1]) + 2 * span   # visible run gap
            for ms in (600, 1000):
                if lo < ms < hi:
                    y = float(_scale(ms, lo, hi, 88, 20))
                    parts.append(f'<line x1="6" y1="{y:.1f}" x2="{W-6}" '
                                 f'y2="{y:.1f}" stroke="{_GRID}" '
                                 f'stroke-width="1"/>'
                                 f'<text x="{W-8}" y="{y-2:.1f}" '
                                 f'fill="{_MUTE}" font-size="9" '
                                 f'text-anchor="end" '
                                 f'font-family="sans-serif">{ms} ms</text>')
            panels.append("tachogram")
        else:
            parts.append(f'<text x="{W/2}" y="54" fill="{_MUTE}" '
                         f'font-size="12" text-anchor="middle" '
                         f'font-family="sans-serif">no clean intervals'
                         f'</text>')
        parts.append("</g>")

        # ---- panel 3: Poincaré (IBI_n vs IBI_n+1, within runs) ---------
        parts.append(f'<g transform="translate(0,232)">'
                     + _panel(0, 96, "Poincaré — interval n vs n+1 (ms)"))
        pairs = []
        for r in ibis:
            if len(r) >= 2:
                pairs.append(np.stack([r[:-1], r[1:]], 1))
        if pairs:
            pp = np.concatenate(pairs)
            lo = float(np.min(pp)) - 40
            hi = float(np.max(pp)) + 40
            cx0, cx1 = W / 2 - 40, W / 2 + 40
            xs = _scale(pp[:, 0], lo, hi, cx0, cx1)
            ys = _scale(pp[:, 1], lo, hi, 88, 20)
            d0 = float(_scale(lo, lo, hi, cx0, cx1))
            d1 = float(_scale(hi, lo, hi, cx0, cx1))
            parts.append(f'<line x1="{d0:.1f}" y1="88" x2="{d1:.1f}" y2="20" '
                         f'stroke="{_GRID}" stroke-width="1"/>')
            for x, y in zip(xs, ys):
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" '
                             f'fill="{_ACC}" opacity="0.6"/>')
            panels.append("poincare")
        else:
            parts.append(f'<text x="{W/2}" y="54" fill="{_MUTE}" '
                         f'font-size="12" text-anchor="middle" '
                         f'font-family="sans-serif">not enough intervals'
                         f'</text>')
        parts.append("</g></svg>")

        return HeadResult(
            head=self.name, version=self.version,
            measurement_class=MeasurementClass.MEASURED,
            value={"svg": "".join(parts), "panels": panels,
                   "caveats": list((context or {}).get("caveats") or [])},
            reasons=[],
        )


register_head(RhythmMapHead)
