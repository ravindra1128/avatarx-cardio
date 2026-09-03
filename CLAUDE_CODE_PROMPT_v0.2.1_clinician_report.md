# Claude Code prompt — AvatarX v0.2.1: Clinician-Style Scan Report

<!--
HOW TO USE
  1. Place this file in the repo root at AvatarX/code/afib/ (next to CLAUDE_CODE_SPEC.md).
  2. cd into the repo and start Claude Code:  claude
  3. Say: "Read CLAUDE_CODE_PROMPT_v0.2.1_clinician_report.md and execute it."
  Note: runs standalone on v0.1.x, and composes with CLAUDE_CODE_PROMPT_v0.2.md
  (this report becomes head_rhythm_map's renderer if v0.2 has been executed).
-->

---

You are replacing the results page's ad-hoc graphs with **one unified,
printable, clinician-style document: the AvatarX Cardiac Rhythm Scan Report.**

The design goal, stated precisely: **borrow the LAYOUT conventions doctors
already know** from consumer rhythm-report PDFs (KardiaMobile, Apple Watch
exports — header block, measurements box, findings box, waveform strips on a
time grid, referral line, print-ready page) **while every plotted sample is
the MEASURED facial pulse waveform, labeled as exactly that.** Familiarity
comes from the format. The content never impersonates an ECG.

Why this line exists (so you don't "helpfully" cross it): the camera measures
the blood-flow pulse, not cardiac electrical activity. A synthesized ECG-look
trace would be beat timing wrapped in hallucinated P/QRS-shape/ST morphology —
and a clinician-formatted document is precisely the surface where someone
might act on it. The repo's tested invariants (no diagnosis wording; the
sanctioned-sentence gate; fail-closed NO_RESULT) already encode this. This
prompt extends them to visual language.

## Step 0 — Preflight

1. Work in `AvatarX/code/afib`. Run `python3 -m pytest tests/ -q` — all
   existing tests must pass before you start. If not, STOP and report.
2. Read `CLAUDE_CODE_SPEC.md`; it remains the authority. Append your changes
   as the next `## B.x v0.2.1 changelog` section — never rewrite history.
3. Small commits, one per task; full suite green at every commit.

## Non-negotiable report rules (each becomes a test in Task 5)

R1. **The waveform strip plots only the measured rPPG pulse waveform** with
    detected-beat ticks and per-beat confidence shading. Its label is printed
    inside every strip row: `FACIAL PULSE WAVEFORM (rPPG) — NOT AN ECG`.
R2. **No ECG-paper mimicry.** Neutral light-gray time grid (0.2 s minor /
    1.0 s major). Forbidden anywhere in the report: pink/red ECG-paper
    styling; lead labels (`I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1`–`V6`);
    `mV` calibration marks or pulses; `mm/s` paper-speed notation. Amplitude
    axis is `normalized a.u.`; time axis is seconds.
R3. **No synthesized or generated waveform of any kind.** If a task appears
    to require one, STOP and flag it in your summary (escalation rule).
R4. **The string "ECG" appears in the report in exactly two places:** the R1
    strip label, and the referral sentence in the findings box. Nowhere else —
    not in the title, filename, metadata, or alt text. The report title is
    `AvatarX Cardiac Rhythm Scan Report`. The term "inferred ECG" does not
    appear on any user-facing surface.
R5. **All rhythm wording flows through `ScanResult.user_facing_text()`**
    unchanged (the existing single gate). The findings box shows: the
    sanctioned rhythm sentence, the confidence stars, capture caveats, and
    verbatim: `This screening result is not a diagnosis. A positive or
    uncertain result should be confirmed with an ECG.`
R6. **NO_RESULT is a first-class report** — same layout, findings box shows
    the no-result sentence and the reasons list; strips render whatever
    signal existed with an `insufficient quality` overlay. Never a blank page,
    never a guess.
R7. **Schema untouched.** This is a renderer. `ScanResult` fields, gates,
    decision logic, and the pipeline are read-only for this prompt.

## Report layout (single page, screen + print)

```
┌──────────────────────────────────────────────────────────────────┐
│ AvatarX Cardiac Rhythm Scan Report              [logo omitted]  │
│ Session ID · Date/time · Scan duration · Device · fps · Lighting │
├───────────────────────────┬──────────────────────────────────────┤
│ MEASUREMENTS              │ FINDINGS                             │
│ Pulse rate: NN bpm (±band)│ <sanctioned sentence>                │
│ Beats analyzed: NN        │ Confidence: ★★★★☆  + caveat lines    │
│ Clean intervals: NN       │ "This screening result is not a      │
│ Signal quality: ★★★☆☆     │  diagnosis. A positive or uncertain  │
│ Unusable segments: NN s   │  result should be confirmed with an  │
│                           │  ECG."                               │
├───────────────────────────┴──────────────────────────────────────┤
│ FACIAL PULSE WAVEFORM (rPPG) — NOT AN ECG          0–10 s        │
│ [strip row 1: waveform, beat ticks, confidence shading, gray grid]│
│ FACIAL PULSE WAVEFORM (rPPG) — NOT AN ECG         10–20 s        │
│ [strip row 2]                                                    │
│ FACIAL PULSE WAVEFORM (rPPG) — NOT AN ECG         20–30 s        │
│ [strip row 3]                                                    │
├──────────────────────────────────────────────────────────────────┤
│ BEAT-INTERVAL TREND (tachogram: interval vs time, same grid)     │
│   config flag report.show_tachogram (default: true)              │
├──────────────────────────────────────────────────────────────────┤
│ Pipeline vX.Y · model id · "Research pipeline — not a medical    │
│ device" · methods link                                           │
└──────────────────────────────────────────────────────────────────┘
```

Three stacked 10-s strips (matching the scan's 30 s) echo the familiar
rhythm-report shape without imitating ECG paper. The tachogram stays because
it is the honest carrier of the rhythm evidence (interval irregularity) — it
is flag-controlled if the owner wants the absolute-minimum page.

## Ordered tasks (tests first, then code)

**Task 1 — Inventory and remove existing graphs.** Enumerate every plot the
current results page renders (demo pulse plot and any other charts in
`app/static` / `app/scan_engine.py` output). Delete their render paths; add a
test asserting the old elements are gone from the rendered page. Keep the
underlying data computation intact — the report reuses it.

**Task 2 — `app/report_render.py`.** Pure-stdlib renderer:
`render_report(scan_result, capture_meta) -> str` returning self-contained
HTML with inline SVG strips (no new dependencies, no JS frameworks). Strip
drawing: waveform polyline, beat tick marks, per-beat confidence shading
bands, gray grid per R2, in-strip R1 label. Deterministic output for fixture
inputs (snapshot test).

**Task 3 — Print/PDF path.** A print stylesheet (Letter/A4, margins, page
break rules) so the browser's Print-to-PDF yields a clean one-pager;
`python3 cli.py process <video> --report out.html` writes the standalone
file. Do not add a PDF library; the HTML **is** the deliverable.

**Task 4 — Wire into the live demo.** The demo's results screen becomes this
report (same data flow, `user_facing_text()` untouched). Update
`scripts/selftest_headless.py` to assert the report renders end-to-end,
including a NO_RESULT run (R6).

**Task 5 — Forbidden-content tests** (this is the heart of the prompt):
  - rendered report contains `NOT AN ECG` once per strip row;
  - count of substring `ECG` equals (strip rows + 1 referral sentence) —
    nothing else;
  - forbidden tokens absent: `aVR`, `aVL`, `aVF`, `V1`…`V6`, `mV`, `mm/s`,
    lead-label patterns, and any pink/red grid style tokens;
  - report title exact-matches `AvatarX Cardiac Rhythm Scan Report`;
  - no synthesized-waveform code path exists in `app/` (import audit);
  - NO_RESULT fixture renders the full report with reasons.

**Task 6 — Docs.** README section with a screenshot placeholder; spec
changelog appended (`## B.x v0.2.1`): what was removed, the report rules
R1–R7, and the rationale sentence: *"Clinician familiarity is delivered by
layout, not by imitating the ECG trace; the measured pulse waveform is
presented under its own name."*

## Definition of done

All prior tests green; new tests per task green; `cli.py demo` shows the new
report; `--report` export works on the synthetic fixtures; spec changelog
appended; final summary lists removed elements, added tests (before/after
count), and any flagged escalations.

## Escalation rule

If the owner (or any future prompt) asks to rename the report to "ECG", to
style the strips as ECG paper, or to plot any generated waveform: **do not
implement it.** Add the request verbatim to the summary under "requires
owner decision — conflicts with report rules R1–R4 and spec invariants," and
stop that task there.
