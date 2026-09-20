# Spectral selection and interval-loss diagnostics

The three September 19 scans returned Inconclusive on build b8bcb4a7b2cd.
The video spectrum selected 45/51/51 bpm against SDK rates 87/86/82 bpm.
All SDK candidates stopped at rate corroboration; two video scans also failed
SQI. These summaries do not establish the true heart rate or cardiac rhythm.

This telemetry-only update records the exact spectra already used by the
existing selector, without retaining spectra or waveforms. Each region and the
fused spectrum report up to five strongest interior peaks, relative power,
selected peak (even outside the top five), strongest peak, band maximum,
frequency spacing and whether the existing rule selected a subharmonic.
Input/finite sample counts expose finite-sample removal. The nominal-fps clock
assumption is explicit. Peaks remain descriptive evidence, not verified beats.

Clean-run accounting partitions every adjacent detected-beat interval into:
confidence rejection, capture-segment boundary, physiological-range rejection,
unique splitter exclusions, short-run exclusions, or kept intervals. Categories
are ordered and exclusive. Splitter counters describe suspected detection errors;
they do not establish that the underlying physiological interval was erroneous.
The existing duration audit separately reports frame/ROI/short-fragment losses.

Five sheet columns add strongest spectral rate, selection type, compact spectral
audit, interval rejection counts and SQI to eight decimal places. This makes a
value just below 0.3 visible instead of displaying "0.30 below 0.3". No floors,
selection rules, source routing, frontend or retention settings change.

Validation compares 100 generated cleaner cases and 20 spectral cases to saved
pre-edit outputs, with added telemetry removed. Existing outputs are identical.
Additional tests cover exact counter conservation, half-rate selection, drift-only
spectra, bounded summaries, diagnostic failure and sheet compatibility. These are
software controls, not clinical validation or a demonstrated stability gain.

The optimizer snapshot was taken before edits. Its structural guard reports
application telemetry and tests as outside its signal-only allowlist; protected
thresholds are unchanged. This is the explicitly requested diagnostic feature,
not an accepted signal-optimization iteration. No corpus replay or accuracy gain
is claimed. No raw user recordings were saved or fetched for this change.
