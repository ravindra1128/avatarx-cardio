# The Research & Investigation Report (v0.8)

**What it is.** One document, for investigators only, that carries
every gated head's raw output for one scan — rhythm regularity (§R),
the atrial-flutter pattern flag (§F), arterial stiffness (vascular),
vasomotor reactivity / vascular tone (§W) and cardiorespiratory fitness
(§V, the track the owner names VO2 max) — each beside its track's
**live gate status**, under one watermark on every page and every
section.

**What it is not.** It is not the AvatarX Cardiac Rhythm Scan Report
and never becomes part of it. Nothing under `research/` is importable
from `app/` or `inference/` (the transitive quarantine walker enforces
it), `app/server.py` has no route to it, the consumer payload filter
(`datasets.schema.public_head_results`) is untouched, and the only way
to produce the document is the explicit verb below. Every track's
rendering invariant (V-a, F-b, W-a, §R, invariant 9) still decides what
a participant may see; this document decides nothing — it records.

Owner instruction, 2026-09-01, quoted verbatim in spec B.24: *"Provide
all VO2 max, atrial flutter, arterial stiffness, vascular tone,
rhythm-regularity in the report for research and investigation purposes
only."*

## What one resting scan yields (v0.8)

`features/hemodynamics.py` computes three families from a single
ACCEPT-grade scan, so every track carries a value rather than a
protocol requirement. All are **uncalibrated** and say so, each with
the definition it was computed from:

- **Arterial stiffness**: reflection index and the Takazawa aging
  index, (b − c − d − e)/a on the pulse second derivative. Direct
  formulas, no model. Dimensionless, never a velocity.
- **Vasomotor tone**: amplitude coefficient of variation and the
  vasomotion-band (0.04–0.15 Hz) power fraction, from a per-beat
  amplitude series normalised by each region's own median so nothing
  absolute is reported.
- **Cardiorespiratory**: resting pulse, RMSSD, SDNN and a bounded
  autonomic index. No oxygen-uptake number: the camera supplies resting
  pulse and its variability, while every non-exercise equation is
  dominated by demographics, so the payload carries the reason and the
  fields a fitted model still needs.

Capture at 60 fps where the device allows it. The second-derivative
waves need real sampling rate: at 30 fps the aging index is refused
rather than interpolated, and the stiffness readout is the reflection
index alone.

## Two surfaces

**1. The results screen's RESEARCH / DEBUG METRICS panel** (v0.8, this
is the panel in the app under the scan report). The five tracks appear
there **by default**, each with its live gate status. A red gate does
not withhold a value — it is printed next to it, so the number and the
reason it may not be shown to a participant are never separated. To
turn the block off for a participant-facing session:

```bash
AVATARX_RESEARCH_TRACKS=0 python3 cli.py demo
```

or set `report.research_tracks: false` in `configs/default.yaml`. The
panel is screen-only (print CSS hides it,
so the printed Cardiac Rhythm Scan Report never carries it), the values
ride their own `research_tracks` payload key so the consumer boundary
`public_head_results` is untouched, and the page itself names no track —
every label arrives in the payload. The two rhythm tracks compute fully
there; arterial stiffness and vascular tone need the research surface
that `app/` cannot reach (quarantine), so the panel prints what they
need and the command below.

**2. The full document**, below, which carries all five including the
tracks the app cannot compute.

## Producing one

```bash
python3 cli.py research-report scan.avi --age 58
python3 cli.py research-report scan.avi --age 58 --session session.json \
        --videos rest.avi activity.avi recovery.avi          # adds the §V track
python3 cli.py research-report scan.avi --provocation p.provocation.json \
        [--pi pi.json]                                        # adds the §W track
```

Output: `research/runs/investigation/<recording_id>/investigation_report.{json,html}`
(gitignored), the JSON also on stdout, watermark first. Exit 0 whether a
track yielded a value, abstained, was not run or was not applicable;
exit 2 on invalid arguments or input files that cannot be parsed (a
missing video, a missing or malformed session manifest, `--videos`
without `--session`, `--pi` without `--provocation`, an implausible
`--age`, an unwritable `--out`). An unreadable video is not an
argument error: like `cli.py process`, it yields a NO_RESULT scan with
its reasons and exit 0. The vascular and vasotone research runners also
write their own per-scan reports under their tracks' runs roots
(gitignored), exactly as `cli.py process --heads vascular` does; the
document lists those paths and `--out` does not move them. A capture
manifest given with `--manifest` is reduced exactly as `process`
reduces it, so the quoted consumer result is the one `process` would
produce.

## What one scan yields, per track

| Track | From one facial video | Needs a protocol |
|---|---|---|
| Rhythm regularity (§R) | on an ACCEPT scan: the irregularity index + CI, and the class and benign evidence when ≥ 15 clean intervals exist and the torso-derived respiration channel is usable; a rejected scan carries no index and says why | — |
| Atrial-flutter pattern flag (§F) | the rate fingerprint, hyper-regularity and coupling families; the flag or an abstention with its reason; the serial signature needs several scans | — |
| Arterial stiffness (vascular) | **contour markers**: the reflection index and the Takazawa aging index, plus the raw morphology set. Uncalibrated, dimensionless, never a velocity. The aging index needs 60 fps; at 30 fps it is refused | a carotid-femoral velocity in m/s needs the V0 study with a contact reference |
| Vasomotor tone (§W) | **resting indices**: amplitude coefficient of variation and the vasomotion-band power fraction, both dimensionless and self-normalised so nothing absolute is reported. Unlocked optics are flagged | REACTIVITY needs `--provocation` phase marks (`--pi` optional) |
| Cardiorespiratory fitness (§V) | **resting indices**: resting pulse, RMSSD, SDNN and a bounded autonomic index. No oxygen-uptake number, by hard rule and because a resting scan cannot support one; the payload carries the reason and names the demographic fields a fitted model still needs | heart-rate recovery needs the `--session` manifest (rest, guided activity, recovery) |

Four statuses, never blurred: `value` (the head decided), `abstained`
(the head ran and declined, with its own reasons — for the fitness head
that is a `category` of None), `not_run` (the head never executed: the
scan produced no beat lattice, the session ended before its heads, or a
research runner failed — the section carries the scan's or session's
reasons, not invented head reasons) and `not_applicable` (the track
needs a protocol this scan does not carry, stated).

## Reading it

- The **consumer result** section quotes the participant's outcome,
  stars and the sanctioned sentence verbatim — context, never
  re-authored.
- Every track section shows the head, its version and measurement
  class, the raw `value` dict, the head's `reasons`, and the track's
  promotion (BLOCKED / OPEN), gates version, clinical signoff and every
  gate's colour with its first reason, read live at generation time
  through each track's own `gate_status` function and failing closed to
  BLOCKED on any error.
- Raw head payloads may contain internal evidence labels and band names
  (`ectopy_pattern`, `2:1`, latent atrial rates). Those tokens are
  forbidden on **user** surfaces (G-d, F-b, W-b, V-b); this is not one,
  and the tests that fence those surfaces are unchanged.
- No waveform, no ECG-paper conventions, nothing red: numbers, labels
  and reasons only.

## Honesty note

While a track's promotion reads BLOCKED — all five do today — nothing
in its section is a finding about the person scanned. Every number this
document has ever carried derives from synthetic video or from public
surrogates. Research pipeline; not a medical device.
