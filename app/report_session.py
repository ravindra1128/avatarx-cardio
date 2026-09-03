"""
Recovery-session findings report (v0.4 T7) — PURE STDLIB, findings-only
by construction: MEASURED recovery physiology + stars + protocol
caveats + referral. No waveform of any kind, ever, on this surface.

The INFERRED_FITNESS render path EXISTS here — and is unreachable while
any §V gate is red: it renders only when BOTH the SessionResult carries
a category (the session assembler populates it only when §V is open)
AND a live gate check agrees (defense in depth, fail-closed).
"""
from __future__ import annotations

import html

SESSION_REPORT_TITLE = "AvatarX Recovery Session Report"
SESSION_REFERRAL = ("This screening result is not a diagnosis. If "
                    "anything here concerns you, share it with a "
                    "clinician.")
FOOTER_NOTE = "Research pipeline — not a medical device"

_CSS = """
.axs { background:#ffffff; color:#1f2937; border-radius:10px;
  padding:22px 26px; font:13px/1.45 -apple-system, "Segoe UI", Roboto,
  Arial, sans-serif; max-width:1000px; margin:0 auto;
  box-shadow:0 1px 4px rgba(0,0,0,.25); }
.axs h1 { font-size:19px; margin:0 0 8px; color:#111827; }
.axs .cols { display:flex; gap:18px; margin-bottom:14px; }
.axs .box { border:1px solid #d1d5db; border-radius:8px;
  padding:10px 14px; flex:1; }
.axs .box h2 { font-size:11px; letter-spacing:.08em; color:#6b7280;
  margin:0 0 8px; }
.axs table { border-collapse:collapse; width:100%; }
.axs td { padding:2px 0; font-size:12.5px; }
.axs td:last-child { text-align:right; font-weight:600; }
.axs .sentence { font-size:14px; margin:0 0 8px; }
.axs .cav { color:#6b6420; font-size:11.5px; margin:2px 0; }
.axs .referral { font-size:12px; color:#374151; border-top:1px dashed
  #d1d5db; padding-top:8px; margin-top:8px; }
.axs .foot { border-top:1px solid #d1d5db; margin-top:14px;
  padding-top:8px; color:#6b7280; font-size:10.5px; }
"""


def _f(x, unit="", dash="—"):
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    return f"{v:.0f}{unit}"


def _stars(n):
    if n is None:
        return "—"
    n = max(0, min(5, int(n)))
    return "★" * n + "☆" * (5 - n)


def render_session_fragment(result, *, _render_allowed=None) -> str:
    """SessionResult -> embeddable findings div. `_render_allowed` is
    injectable for tests; production uses the live §V check."""
    e = html.escape
    if _render_allowed is None:
        from evaluation.fitness_gates import fitness_render_allowed
        _render_allowed = fitness_render_allowed

    rows = [
        ("Resting pulse", _f(result.hr_rest_bpm, " bpm")),
        ("Resting breathing", _f(result.rr_rest_brpm, " /min")),
        ("End-of-activity pulse (proxy)",
         _f(result.hr_end_proxy_bpm, " bpm")),
        ("Pulse-rate drop, 30 s", _f(result.hrr30_bpm, " bpm")),
        ("Pulse-rate drop, 60 s", _f(result.hrr60_bpm, " bpm")),
        ("Pulse-rate drop, 120 s", _f(result.hrr120_bpm, " bpm")),
        ("Early recovery slope",
         _f(result.recovery_slope_bpm_min, " bpm/min")),
    ]
    mtable = "".join(f"<tr><td>{e(k)}</td><td>{v}</td></tr>"
                     for k, v in rows)
    finding = [f'<p class="sentence">{e(result.user_facing_text())}</p>',
               f'<p>Confidence: {_stars(result.confidence_stars)}</p>']
    comp = result.compliance or {}
    if comp.get("verdict") == "compliant":
        finding.append(f'<p class="cav">Guided activity completed on '
                       f'pace ({e(str(result.protocol_id))}).</p>')
    for r in comp.get("reasons") or []:
        finding.append(f'<p class="cav">{e(str(r))}</p>')
    # ---- the §V-gated INFERRED_FITNESS path: exists, and unreachable
    # while any gate is red (both the field and the live check gate it).
    # Even when open, only SANCTIONED sentences render, keyed verbatim —
    # this module composes no fitness text of its own.
    if result.fitness_category is not None and _render_allowed():
        from datasets.schema import (FITNESS_CATEGORY_SENTENCES,
                                     TREND_DIRECTION_SENTENCES)
        cat = FITNESS_CATEGORY_SENTENCES.get(str(result.fitness_category))
        if cat:
            finding.append(f'<p class="cav">{e(cat)}</p>')
        if result.trend:
            trd = TREND_DIRECTION_SENTENCES.get(
                str(result.trend.get("direction")))
            if trd:
                finding.append(f'<p class="cav">{e(trd)}</p>')
    finding.append(f'<p class="referral">{e(SESSION_REFERRAL)}</p>')

    prov = " · ".join([f"Pipeline {e(str(result.model_version))}",
                       f"commit {e(str(result.code_commit))}",
                       f"activity tracker "
                       f"{e(str(result.activity_tracker or '—'))}",
                       e(FOOTER_NOTE)])
    return (f'<div class="axs"><style>{_CSS}</style>'
            f"<h1>{e(SESSION_REPORT_TITLE)}</h1>"
            f'<div class="cols">'
            f'<div class="box"><h2>MEASUREMENTS</h2>'
            f"<table>{mtable}</table></div>"
            f'<div class="box"><h2>FINDINGS</h2>'
            f"{''.join(finding)}</div></div>"
            f'<div class="foot">{prov}</div></div>')
