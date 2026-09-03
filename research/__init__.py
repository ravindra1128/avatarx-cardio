"""
research/ — QUARANTINED research track (v0.3).

Nothing under this package may be imported from `app/` or `inference/`
(tests/test_reconstruction_quarantine.py walks the import graph). Every
artifact written by this package is watermarked and lands only under
`research/runs/`. The one sanctioned bridge OUT of quarantine is the
promotion gate (configs/gates.yaml, §G): `models/registry.promote` reads a
run's gate_results.json — data, never code — and refuses while any gate
is red.
"""
