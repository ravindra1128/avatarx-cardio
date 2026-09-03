"""
AvatarX Cardiac Rhythm Scan Report renderer (v0.2.1) — PURE STDLIB.

One unified, printable, clinician-style document. The LAYOUT borrows the
conventions of consumer rhythm-report PDFs (header block, measurements
box, findings box, waveform strips on a time grid, referral line); the
CONTENT never impersonates an ECG: every plotted sample is the measured
facial pulse waveform, labeled inside every strip row as
`FACIAL PULSE WAVEFORM (rPPG) — NOT AN ECG` (rule R1). Neutral gray time
grid only (R2: no ECG-paper pink/red, no lead labels, no mV / mm/s);
amplitude is `normalized a.u.`. All rhythm wording flows through
`ScanResult.user_facing_text()` unchanged (R5); NO_RESULT renders the
same full layout with reasons and an `insufficient quality` overlay
(R6). Nothing here synthesizes a waveform (R3); the schema, gates and
pipeline are untouched (R7 — this is a renderer).

R4 note (reconciled with R5, spec B.16): "ECG" appears once per strip
label and once in the referral sentence; when the SANCTIONED sentence
itself contains "ECG" (the AF-suggestive wording mandates the clinician/
ECG referral), that occurrence is required by the text gate and counted
explicitly by the forbidden-content tests.

Deterministic: no clocks, no randomness — output is a pure function of
(scan_result, capture_meta).

v0.3: the DEFAULT product surface is `report_mode: findings_only` — no
waveform of any kind. Everything above about strips applies to `full`
mode. The resting-biomarker section is data-driven and appears in both
modes when ``capture_meta.biomarkers`` is present; the renderer performs
no biomarker calculation and visibly preserves every abstention.
"""
from __future__ import annotations

import html
import math

REPORT_TITLE = "AvatarX Cardiac Rhythm Scan Report"
STRIP_LABEL = "FACIAL PULSE WAVEFORM (rPPG) — NOT AN ECG"
REFERRAL_SENTENCE = ("This screening result is not a diagnosis. A positive "
                     "or uncertain result should be confirmed with an ECG.")
FOOTER_NOTE = "Research pipeline — not a medical device"
STRIP_SECONDS = 10.0
MAX_STRIPS = 6
GRID_MINOR_S = 0.2
GRID_MAJOR_S = 1.0

# neutral grays only (R2) + a single non-red ink for the measured trace
_INK = "#0f766e"
_GRID_MINOR = "#e8eaec"
_GRID_MAJOR = "#c9ced4"
_TXT = "#1f2937"
_MUTE = "#6b7280"

_W = 1000
_STRIP_H = 130
_TACHO_H = 150


def _f(x, nd=0, dash="—"):
    try:
        if x is None:
            return dash
        v = float(x)
        if not math.isfinite(v):
            return dash
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return dash


def _stars(n, of=5):
    if n is None:
        return "—"
    n = max(0, min(of, int(n)))
    return "★" * n + "☆" * (of - n)


def _biomarkers_html(payload: dict) -> str:
    """Render the already-computed biomarker payload; never calculate here."""
    block = dict(payload or {})
    items = list(block.get("items") or [])
    if not items:
        return ""
    e = html.escape
    cards = []
    for item in items:
        item = dict(item or {})
        computed = item.get("status") == "computed" and \
            item.get("value") is not None
        value = (_f(item.get("value"), 2) if computed else "Not computed")
        unit = (" " + e(str(item.get("unit")))) \
            if computed and item.get("unit") else ""
        metric = (f'<div class="bio-metric">'
                  f'{e(str(item.get("metric") or ""))}</div>') \
            if computed and item.get("metric") else ""
        conf = dict(item.get("confidence") or {})
        evidence = []
        if conf.get("signal_quality_index") is not None:
            evidence.append(f"SQI {_f(conf['signal_quality_index'], 2)}")
        if conf.get("confidence_stars") is not None:
            evidence.append(f"{int(conf['confidence_stars'])}/5 signal stars")
        if conf.get("n_beats_used") is not None:
            evidence.append(f"{int(conf['n_beats_used'])} beats")
        if conf.get("n_rois_used") is not None:
            evidence.append(f"{int(conf['n_rois_used'])} facial regions")
        if conf.get("confidence_limiting_factor"):
            evidence.append("limited by " +
                            str(conf["confidence_limiting_factor"]))
        evidence_line = " · ".join(e(str(x)) for x in evidence) or \
            "signal confidence unavailable"
        method = item.get("method") or "method unavailable"
        interpretation = e(str(
            conf.get("interpretation") or
            "Signal evidence is not endpoint accuracy."
        ))
        limitation = item.get("limitation")
        reason = item.get("reason")
        warning = item.get("warning")
        notes = []
        if reason:
            notes.append(f'<p class="bio-reason">Why not computed: '
                         f'{e(str(reason))}</p>')
        if limitation:
            notes.append(f'<p class="bio-limit">Limit: '
                         f'{e(str(limitation))}</p>')
        if warning:
            notes.append(f'<p class="bio-reason">Quality warning: '
                         f'{e(str(warning))}</p>')
        cards.append(
            f'<article class="bio-card" data-biomarker="'
            f'{e(str(item.get("key") or ""))}">'
            f'<div class="bio-label">'
            f'{e(str(item.get("label") or block.get("label") or ""))}</div>'
            f'<h3>{e(str(item.get("title") or "Biomarker"))}</h3>'
            f'<div class="bio-value">{value}{unit}</div>{metric}'
            f'<details><summary>Method and confidence</summary>'
            f'<p><b>Method:</b> {e(str(method))}</p>'
            f'<p><b>Signal evidence:</b> {evidence_line}</p>'
            f'<p>{interpretation}</p>'
            f'</details>{"".join(notes)}</article>')
    if block.get("scan_accepted") is False:
        note = ("The overall rhythm scan was not accepted. Numeric research "
                "features below are shown only where their own measured "
                "inputs survived; they are low-confidence prototype data, "
                "not clinical results. Missing inputs are never replaced.")
    else:
        note = ("Computed from endpoint-usable scan evidence. These "
                "uncalibrated prototype estimates are not diagnoses. A "
                "missing value is reported explicitly; it is never replaced "
                "with a default.")
    return (f'<section class="biomarkers"><div class="bio-head">'
            f'<h2>RESTING BIOMARKER ESTIMATES</h2>'
            f'<span>{e(str(block.get("label") or ""))}</span></div>'
            f'<p class="bio-note">{e(note)}</p>'
            f'<div class="bio-grid">{"".join(cards)}</div></section>')


def _grid(t0, t1, y0, y1, with_labels=True):
    out = []
    t = math.ceil(t0 / GRID_MINOR_S) * GRID_MINOR_S
    while t <= t1 + 1e-9:
        x = (t - t0) / (t1 - t0) * (_W - 70) + 60
        major = abs(t / GRID_MAJOR_S - round(t / GRID_MAJOR_S)) < 1e-6
        out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" '
                   f'stroke="{_GRID_MAJOR if major else _GRID_MINOR}" '
                   f'stroke-width="{1.2 if major else 0.6}"/>')
        if major and with_labels:
            out.append(f'<text x="{x:.1f}" y="{y1 + 14}" font-size="10" '
                       f'fill="{_MUTE}" text-anchor="middle" '
                       f'font-family="sans-serif">{t:.0f} s</text>')
        t += GRID_MINOR_S
    return "".join(out)


def _strip_svg(sig: dict, t0: float, t1: float, insufficient: bool) -> str:
    y_top, y_bot = 26, _STRIP_H - 24
    parts = [f'<svg viewBox="0 0 {_W} {_STRIP_H}" '
             f'xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="facial pulse waveform strip">',
             f'<rect width="{_W}" height="{_STRIP_H}" fill="#ffffff"/>',
             _grid(t0, t1, y_top, y_bot)]
    parts.append(f'<text x="60" y="16" font-size="11" font-weight="bold" '
                 f'fill="{_TXT}" font-family="sans-serif">'
                 f'{html.escape(STRIP_LABEL)}</text>')
    parts.append(f'<text x="{_W - 10}" y="16" font-size="11" fill="{_MUTE}" '
                 f'text-anchor="end" font-family="sans-serif">'
                 f'{t0:.0f}–{t1:.0f} s</text>')
    parts.append(f'<text x="12" y="{(y_top + y_bot) / 2:.0f}" font-size="9" '
                 f'fill="{_MUTE}" font-family="sans-serif" '
                 f'transform="rotate(-90 12 {(y_top + y_bot) / 2:.0f})" '
                 f'text-anchor="middle">normalized a.u.</text>')

    ts = list(sig.get("t_s") or [])
    vals = list(sig.get("values") or [])
    seg = [(float(a), float(b)) for a, b in zip(ts, vals)
           if t0 - 1e-9 <= float(a) <= t1 + 1e-9]
    drew = False
    if len(seg) >= 8:
        lo = min(v for _, v in seg)
        hi = max(v for _, v in seg)
        span = (hi - lo) or 1.0
        pts = []
        for t, v in seg:
            x = (t - t0) / (t1 - t0) * (_W - 70) + 60
            y = y_bot - (v - lo) / span * (y_bot - y_top - 8) - 4
            pts.append(f"{x:.1f},{y:.1f}")
        # per-beat confidence shading band + tick
        for b in sig.get("beats") or []:
            bt = float(b.get("t", -1))
            if not (t0 <= bt <= t1):
                continue
            conf = max(0.0, min(1.0, float(b.get("conf", 0.5))))
            x = (bt - t0) / (t1 - t0) * (_W - 70) + 60
            band = 0.05 + 0.30 * (1.0 - conf)      # more shade = less sure
            parts.append(f'<rect x="{x - 7:.1f}" y="{y_top}" width="14" '
                         f'height="{y_bot - y_top}" fill="#9ca3af" '
                         f'opacity="{band:.2f}"/>')
            parts.append(f'<line x1="{x:.1f}" y1="{y_bot - 10}" '
                         f'x2="{x:.1f}" y2="{y_bot}" stroke="{_INK}" '
                         f'stroke-width="2" '
                         f'opacity="{0.35 + 0.65 * conf:.2f}"/>')
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                     f'stroke="{_INK}" stroke-width="1.4"/>')
        drew = True
    if insufficient or not drew:
        parts.append(
            f'<rect x="60" y="{y_top}" width="{_W - 70}" '
            f'height="{y_bot - y_top}" fill="#f3f4f6" opacity="0.55"/>'
            f'<text x="{_W / 2}" y="{(y_top + y_bot) / 2 + 4:.0f}" '
            f'font-size="13" fill="{_MUTE}" text-anchor="middle" '
            f'font-family="sans-serif">insufficient quality — no '
            f'verified signal in this span</text>')
    parts.append("</svg>")
    return "".join(parts)


def _tachogram_svg(sig: dict, t_end: float) -> str:
    y_top, y_bot = 26, _TACHO_H - 24
    iv = [i for i in (sig.get("intervals") or [])
          if i.get("ms") is not None]
    parts = [f'<svg viewBox="0 0 {_W} {_TACHO_H}" '
             f'xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="beat interval trend">',
             f'<rect width="{_W}" height="{_TACHO_H}" fill="#ffffff"/>',
             _grid(0.0, max(t_end, 1.0), y_top, y_bot),
             f'<text x="60" y="16" font-size="11" font-weight="bold" '
             f'fill="{_TXT}" font-family="sans-serif">BEAT-INTERVAL TREND '
             f'— verified beat-to-beat interval (ms) vs time</text>']
    if len(iv) >= 2:
        ms = [float(i["ms"]) for i in iv]
        lo = min(min(ms) - 60, 560)
        hi = max(max(ms) + 60, 1040)
        for ref in (600, 800, 1000):
            if lo < ref < hi:
                y = y_bot - (ref - lo) / (hi - lo) * (y_bot - y_top)
                parts.append(f'<line x1="60" y1="{y:.1f}" x2="{_W - 10}" '
                             f'y2="{y:.1f}" stroke="{_GRID_MAJOR}" '
                             f'stroke-width="0.8" stroke-dasharray="3 3"/>'
                             f'<text x="{_W - 12}" y="{y - 3:.1f}" '
                             f'font-size="9" fill="{_MUTE}" '
                             f'text-anchor="end" '
                             f'font-family="sans-serif">{ref} ms</text>')
        prev = None
        for i in iv:
            t = float(i.get("t", 0.0))
            x = t / max(t_end, 1.0) * (_W - 70) + 60
            y = y_bot - (float(i["ms"]) - lo) / (hi - lo) * (y_bot - y_top)
            conf = max(0.0, min(1.0, float(i.get("conf", 0.8))))
            op = 0.35 + 0.65 * conf
            if prev is not None and t - prev[2] <= 2.5:
                parts.append(f'<line x1="{prev[0]:.1f}" y1="{prev[1]:.1f}" '
                             f'x2="{x:.1f}" y2="{y:.1f}" stroke="{_INK}" '
                             f'stroke-width="1.2" opacity="{op:.2f}"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" '
                         f'fill="{_INK}" opacity="{op:.2f}"/>')
            prev = (x, y, t)
    else:
        parts.append(f'<text x="{_W / 2}" y="{(y_top + y_bot) / 2:.0f}" '
                     f'font-size="13" fill="{_MUTE}" text-anchor="middle" '
                     f'font-family="sans-serif">insufficient quality — '
                     f'too few verified intervals</text>')
    parts.append("</svg>")
    return "".join(parts)


_CSS = """
.axr { background:#ffffff; color:#1f2937; border-radius:10px;
  padding:22px 26px; font:13px/1.45 -apple-system, "Segoe UI", Roboto,
  "Helvetica Neue", Arial, sans-serif; max-width:1000px; margin:0 auto;
  box-shadow:0 1px 4px rgba(0,0,0,.25); }
.axr h1 { font-size:19px; margin:0 0 2px; color:#111827; }
.axr .meta { color:#6b7280; font-size:11.5px; border-bottom:1px solid
  #d1d5db; padding-bottom:10px; margin-bottom:12px; }
.axr .cols { display:flex; gap:18px; margin-bottom:14px; }
.axr .box { border:1px solid #d1d5db; border-radius:8px; padding:10px 14px; }
.axr .box h2 { font-size:11px; letter-spacing:.08em; color:#6b7280;
  margin:0 0 8px; }
.axr .m { flex:0 0 300px; }
.axr .f { flex:1; }
.axr .m table { border-collapse:collapse; width:100%; }
.axr .m td { padding:2px 0; font-size:12.5px; }
.axr .m td:last-child { text-align:right; font-weight:600; }
.axr .sentence { font-size:14px; margin:0 0 8px; }
.axr .conf { margin:0 0 8px; }
.axr .cav { color:#6b6420; font-size:11.5px; margin:2px 0; }
.axr .reasons { color:#6b7280; font-size:11.5px; margin:4px 0; }
.axr .referral { font-size:12px; color:#374151; border-top:1px dashed
  #d1d5db; padding-top:8px; margin-top:8px; }
.axr .biomarkers { border-top:1px solid #d1d5db; padding-top:12px;
  margin:4px 0 14px; }
.axr .bio-head { display:flex; gap:10px; align-items:baseline;
  justify-content:space-between; }
.axr .bio-head h2 { font-size:11px; letter-spacing:.08em; color:#6b7280;
  margin:0; }
.axr .bio-head span, .axr .bio-label { color:#0f766e; font-size:10.5px;
  font-weight:700; }
.axr .bio-note { color:#6b7280; font-size:11px; margin:4px 0 9px; }
.axr .bio-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
  gap:10px; }
.axr .bio-card { min-width:0; border:1px solid #d1d5db; border-radius:8px;
  padding:10px 11px; background:#f8fafc; }
.axr .bio-card h3 { font-size:12.5px; margin:2px 0 5px; }
.axr .bio-value { font-size:20px; line-height:1.15; font-weight:700;
  overflow-wrap:anywhere; }
.axr .bio-metric { color:#6b7280; font-size:10.5px; overflow-wrap:anywhere; }
.axr .bio-card details { margin-top:8px; font-size:10.5px; color:#4b5563; }
.axr .bio-card summary { cursor:pointer; font-weight:600; }
.axr .bio-card details p, .axr .bio-reason, .axr .bio-limit {
  margin:5px 0 0; font-size:10.5px; overflow-wrap:anywhere; }
.axr .bio-reason { color:#6b6420; }
.axr .bio-limit { color:#6b6420; }
.axr .strip { margin:8px 0; }
.axr .strip svg, .axr .tacho svg { width:100%; height:auto; display:block;
  border:1px solid #e5e7eb; border-radius:6px; }
.axr .foot { border-top:1px solid #d1d5db; margin-top:14px; padding-top:8px;
  color:#6b7280; font-size:10.5px; }
@media print {
  @page { size: auto; margin: 12mm; }
  body { background:#ffffff !important; }
  .axr { box-shadow:none; border-radius:0; max-width:none; padding:0; }
  .axr .strip, .axr .tacho, .axr .cols { break-inside: avoid; }
  .axr .bio-card { break-inside:avoid; }
}
@media (max-width:700px) {
  .axr { padding:16px 14px; }
  .axr .cols { display:block; }
  .axr .box { margin-bottom:10px; }
  .axr .m { width:auto; }
  .axr .bio-head { display:block; }
  .axr .bio-head span { display:block; margin-top:3px; }
  .axr .bio-grid { grid-template-columns:1fr; }
}
"""


def render_report_fragment(scan_result, capture_meta: dict) -> str:
    """Embeddable report `<div class="axr">…</div>` with scoped styles.
    `capture_meta` carries header fields, optional `signal` payload
    (measured waveform/beats/intervals from the pipeline details) and
    presentation flags; the ScanResult is read-only."""
    cm = dict(capture_meta or {})
    sig = dict(cm.get("signal") or {})
    e = html.escape
    # v0.3 (owner-directed): the DEFAULT surface is findings-only — no
    # waveform of any kind (no pulse waveform, and never a synthetic ECG
    # while any §G gate is red). "full" restores the v0.2.1 strips +
    # interval trend; an absent/unknown mode fails closed to findings.
    findings_only = str(cm.get("report_mode", "findings_only")) != "full"

    outcome = getattr(scan_result.outcome, "value", str(scan_result.outcome))
    accepted = outcome == "ACCEPT"

    scan_s = cm.get("scan_seconds")
    if scan_s is None and sig.get("t_s"):
        scan_s = float(sig["t_s"][-1])
    span = float(scan_s or 0.0)
    n_rows = max(1, min(MAX_STRIPS,
                        int(math.ceil(span / STRIP_SECONDS)) or 1))

    meta_bits = [
        f"Session {e(str(cm.get('session_id') or scan_result.recording_id))}",
        f"{e(str(cm.get('datetime') or '—'))}",
        f"Scan {_f(scan_s, 0)} s",
        f"Device {e(str(cm.get('device') or cm.get('tracker') or '—'))}",
        f"{_f(cm.get('measured_fps') or cm.get('fps'), 1)} fps",
        f"Lighting {e(str(cm.get('lighting') or '—'))}",
    ]

    ivs = [i for i in (sig.get("intervals") or []) if i.get("ms")]
    bpm = getattr(scan_result, "mean_pulse_rate_bpm", None)
    band = ""
    if ivs:
        bpms = sorted(60000.0 / float(i["ms"]) for i in ivs)
        if bpm is None:
            bpm = bpms[len(bpms) // 2]
        if len(bpms) >= 8:
            q1 = bpms[len(bpms) // 4]
            q3 = bpms[(3 * len(bpms)) // 4]
            band = f" (±{(q3 - q1) / 2:.0f})"
    unusable = None
    if scan_s is not None and scan_result.analysed_seconds is not None:
        unusable = max(0.0, float(scan_s)
                       - float(scan_result.analysed_seconds))
    sqi = scan_result.signal_quality_index
    sqi_stars = None if sqi is None else max(1, min(5, round(float(sqi) * 5)))

    rows_m = [
        ("Pulse rate", f"{_f(bpm, 0)} bpm{band}"),
        ("Beats analyzed", _f(scan_result.usable_beats, 0)),
        ("Clean intervals", _f(len(ivs) if ivs else None, 0)),
        ("Signal quality", f"{_stars(sqi_stars)} ({_f(sqi, 2)})"),
        ("Unusable segments", f"{_f(unusable, 0)} s"),
    ]
    mtable = "".join(f"<tr><td>{e(k)}</td><td>{v}</td></tr>"
                     for k, v in rows_m)

    finding_lines = [f'<p class="sentence">{e(scan_result.user_facing_text())}'
                     f"</p>",
                     f'<p class="conf">Confidence: '
                     f"{_stars(scan_result.confidence_stars)}"
                     + (f" ({int(scan_result.confidence_stars)} of 5)"
                        if scan_result.confidence_stars is not None else "")
                     + "</p>"]
    for cav in (cm.get("caveats") or []):
        finding_lines.append(f'<p class="cav">{e(str(cav))}</p>')
    if cm.get("rate_flag_sentence"):
        finding_lines.append(
            f'<p class="cav">{e(str(cm["rate_flag_sentence"]))}</p>')
    if not accepted and scan_result.no_read_reasons:
        finding_lines.append(
            '<p class="reasons">Why no result: '
            + " · ".join(e(str(r))
                              for r in scan_result.no_read_reasons)
            + "</p>")
    finding_lines.append(f'<p class="referral">{e(REFERRAL_SENTENCE)}</p>')

    strips = []
    if not findings_only:
        for i in range(n_rows):
            t0 = i * STRIP_SECONDS
            t1 = min(span, t0 + STRIP_SECONDS) if span else STRIP_SECONDS
            if t1 <= t0:
                t1 = t0 + STRIP_SECONDS
            # R6: a non-ACCEPT report shows whatever signal existed UNDER
            # an insufficient-quality overlay; a row with no drawable
            # signal gets the overlay regardless (handled in _strip_svg).
            strips.append(
                f'<div class="strip">'
                f"{_strip_svg(sig, t0, t1, insufficient=not accepted)}"
                f"</div>")

    tacho = ""
    if not findings_only and cm.get("show_tachogram", True):
        tacho = (f'<div class="tacho">'
                 f"{_tachogram_svg(sig, span or STRIP_SECONDS)}</div>")

    prov = " · ".join([
        f"Pipeline {e(str(scan_result.model_version))}",
        f"calibration {e(str(scan_result.calibration_version))}",
        f"commit {e(str(scan_result.code_commit))}",
        e(FOOTER_NOTE),
        "Methods: docs/READINESS.md",
    ])
    biomarkers = _biomarkers_html(cm.get("biomarkers") or {})

    return (f'<div class="axr"><style>{_CSS}</style>'
            f"<h1>{e(REPORT_TITLE)}</h1>"
            f'<div class="meta">{" · ".join(meta_bits)}</div>'
            f'<div class="cols">'
            f'<div class="box m"><h2>MEASUREMENTS</h2>'
            f"<table>{mtable}</table></div>"
            f'<div class="box f"><h2>FINDINGS</h2>'
            f"{''.join(finding_lines)}</div></div>"
            f"{biomarkers}"
            f"{''.join(strips)}"
            f"{tacho}"
            f'<div class="foot">{prov}</div>'
            f"</div>")


def render_report(scan_result, capture_meta: dict) -> str:
    """Standalone, self-contained HTML document (the --report export and
    the print target). The title never says ECG (R4)."""
    frag = render_report_fragment(scan_result, capture_meta)
    return ("<!doctype html>\n<html lang=\"en\"><head>"
            "<meta charset=\"utf-8\">"
            "<meta name=\"viewport\" "
            "content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(REPORT_TITLE)}</title>"
            "</head><body style=\"background:#f0f1f3;margin:0;"
            "padding:16px\">"
            f"{frag}</body></html>\n")
