# Why AvatarX does not show a generated ECG — the falsification lab (M4.15)

**Standing conclusion.** A PPG→ECG decoder produces beat-timed truth
wrapped in hallucinated electrical morphology. On our own paired data it
loses the interval-level test to a baseline with **zero learned
morphology**, and it invents atrial activity that is not in the patient.
Therefore the AvatarX consumer surface renders **measured signals and
rhythm inferences only** (the Rhythm Map), and every artifact this lab
produces is watermarked `SYNTHETIC ECG — RESEARCH ONLY — NOT A
MEASUREMENT` and quarantined from the product (invariant 10; an
import-graph test enforces it).

## Methods

- **Data:** MIMIC PERform AF (19 subjects) + non-AF (16 subjects), paired
  PPG+ECG at 125 Hz (CC-BY 4.0, Charlton et al., Zenodo 6807403; the
  repo's existing E6 cache). Band-passed 0.5–40 Hz, z-normalised per
  subject. **Participant-disjoint** stable-hash quantile split (25 train /
  10 held-out); the synthetic-pairs mode (`--source synthetic`) exercises
  the identical harness without the cache.
- **Decoder** (`evaluation/inferred_ecg/decoder.py`): windowed one-hidden-
  layer regressor (±0.51 s PPG context per predicted ECG sample, 48 hidden
  units, ≈ 9k parameters, numpy manual gradients, Adam, MSE) — a small
  conv-scale sequence model, CPU-trainable in minutes. Deliberately
  modest; the failure mode it demonstrates (morphology from training
  priors) is architecture-independent in kind, and published large
  decoders are judged by the same tests here.
- **Identity-template baseline:** for each held-out subject, the average
  ECG beat from their own *template period* (first 60 % of the record),
  re-timed onto beats detected **from the eval-period PPG alone** (PTT
  estimated in the template period). At eval time it sees exactly what the
  decoder sees; it has learned nothing about ECG morphology beyond "this
  patient's beats look like this patient's beats".
- **Metrics** (`fiducials.py`): matched-beat R-timing MAE, PR-interval and
  R-to-T-peak interval MAE in ms, and P/T prominence relative to QRS.
  Waveform correlation is reported **only alongside** the interval errors
  — correlation is dominated by QRS timing, which the pulse gives away for
  free.
- **Cross-rhythm challenge:** a decoder trained **without AF** subjects
  reconstructs AF segments (and an AF-only decoder reconstructs sinus).
  P-wave prominence in the generated AF vs the reference AF quantifies
  hallucinated atrial activity.

Reproduce: `python3 cli.py falsify --source mimic --steps 1500`
(or `--source synthetic` without the cache). Report + watermarked plots
land under `evaluation/falsification_runs/<timestamp>/`.

## Results (MIMIC PERform, 25 train / 10 held-out, run 2026-08-25)

| Metric (held-out median) | Decoder | Identity template |
|---|---|---|
| Waveform correlation | **0.185** | 0.033 |
| PR-interval MAE | 56 ms | **12 ms** |
| R→T-peak interval MAE | 78 ms | **10 ms** |

| Cross-rhythm challenge (sinus-trained decoder on AF) | value |
|---|---|
| P prominence in **generated** AF | **0.39** |
| P prominence in **reference** AF | 0.04 |

Synthetic-corpus run (harness check, same shape of result): decoder wins
correlation 0.86 vs 0.75 and still loses the interval test (PR 56 ms vs
40 ms); hallucinated P on AF 0.125 vs reference 0.0.

## Reading

1. **The correlation win is the trap.** The decoder "looks more like an
   ECG" by correlation — because QRS timing dominates the variance and
   the pulse hands the timing over. This is precisely the metric a demo
   would quote.
2. **The interval test is where clinical reading happens, and the decoder
   fails it** — 5–8× worse than a baseline that knows nothing about ECG
   morphology. A trace whose PR interval is off by ~56 ms median is not a
   measurement of the patient's conduction; it is a drawing.
3. **The morphology is a prior, not an observation.** Trained without AF,
   the decoder paints P waves into AF at 0.39 relative prominence where
   the reference shows 0.04. Shown to a clinician, that trace asserts
   atrial depolarisation that does not exist in the patient.

Any future proposal to surface a generated ECG must first beat the
identity-template baseline on the interval test and pass the cross-rhythm
challenge on participant-disjoint data through THIS harness — the burden
of proof is codified, not rhetorical.

*Every number above derives from public paired PPG+ECG (MIMIC PERform) or
synthetic fixtures; no clinical performance of AvatarX itself is claimed
or implied. Reference ECG data flows one way (invariant 12).*
