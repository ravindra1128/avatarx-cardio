"""FPR session data contract.

Raw data is scientifically valuable and NOT reproducible; derived data is
disposable. They are stored separately and raw is WRITE-ONCE: this module
refuses to overwrite anything under `raw/`, `calib/`, `roi/` or `controls/`.

Session layout (owner brief section 10):

    <session>/
      manifest.json               provenance (subject metadata, rig config)
      raw/
        session.avi(+ .timestamps.json)   raw/lossless colour video
        session.meta.jsonl                per-frame exposure/gain/WB/focus/ts
      calib/
        dark.avi     (+ sidecar)          60 s lens-capped
        flat.avi     (+ sidecar)          60 s uniform grey card
      roi/
        roi.json                          ROI rectangle + 4x4 grid definition
        roi_verify.png                    visual verification frame
      controls/
        colorchart.csv                    per-frame colour-chart metrics
        phantom.avi  (+ sidecar)          skin-tone non-perfused phantom
        finger_ppg.csv                    raw contact PPG waveform (stimulus +ve)
        heart_rate.csv                    continuous HR
        temperature.csv                   forehead/hand/ambient/water temps
        humidity.csv
        events.csv                        immersion in/out, movement, operator
      spec/
        camera.json                       {bit_depth, roi_px, expected_ac_dc}
      derived/                             (rewritable) PI, dPI, sigma_null, ...
      report/
        qualification.json                the B0-B8 verdict

Nothing here writes into production paths and nothing imports app/inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
import json
import os

import numpy as np

from . import FprError, WATERMARK

# Which artefact each stage consumes. `None` in the value means "no external
# recording required" (B7 is synthetic; B0 needs only a spec).
STAGE_INPUTS: dict[str, Optional[str]] = {
    "B0": "spec/camera.json",
    "B1": "calib/dark.avi",
    "B2": "controls/lock_challenge.avi",
    "B3": "controls/flicker.avi",
    "B4": "calib/flat.avi",
    "B5": "controls/warmup_phantom.avi",
    "B6": "controls/phantom.avi",
    "B7": None,
    "B8": "raw/session.avi",
}

_IMMUTABLE_DIRS = ("raw", "calib", "roi", "controls", "spec")


@dataclass
class FprSession:
    root: str

    # -- path helpers ------------------------------------------------------
    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def exists(self, rel: str) -> bool:
        return bool(rel) and os.path.exists(self.path(rel))

    def ensure_dirs(self) -> None:
        for d in ("raw", "calib", "roi", "controls", "spec", "derived",
                  "report"):
            os.makedirs(self.path(d), exist_ok=True)

    # -- immutable raw write -----------------------------------------------
    def _is_immutable(self, rel: str) -> bool:
        head = rel.replace("\\", "/").split("/", 1)[0]
        return head in _IMMUTABLE_DIRS

    def write_bytes(self, rel: str, data: bytes) -> str:
        p = self.path(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if self._is_immutable(rel) and os.path.exists(p):
            raise FprError(f"refusing to overwrite immutable raw artefact "
                           f"{rel!r}; raw data is write-once")
        with open(p, "wb") as f:
            f.write(data)
        return p

    def save_json(self, rel: str, obj: Any) -> str:
        p = self.path(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if self._is_immutable(rel) and os.path.exists(p):
            raise FprError(f"refusing to overwrite immutable artefact {rel!r}")
        with open(p, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        return p

    def load_json(self, rel: str) -> Any:
        with open(self.path(rel)) as f:
            return json.load(f)

    # -- ROI ---------------------------------------------------------------
    def load_roi(self) -> tuple[int, int, int, int]:
        d = self.load_json("roi/roi.json")
        r = d.get("roi")
        if not (isinstance(r, (list, tuple)) and len(r) == 4):
            raise FprError("roi/roi.json must contain roi=[x0,y0,x1,y1]")
        return tuple(int(v) for v in r)               # type: ignore[return-value]

    # -- reference channels (finger PPG, events) ---------------------------
    def load_finger(self):
        """The finger contact-PPG control: a raw waveform
        (controls/finger_ppg.csv) if present, else a device PI export
        (controls/finger_pi.csv). Returns a reference.FingerResult."""
        from . import reference as _ref
        if self.exists("controls/finger_ppg.csv"):
            t, ppg = _ref.read_finger_ppg(self.path("controls/finger_ppg.csv"))
            return _ref.analyze_finger(ppg=ppg, t=t)
        if self.exists("controls/finger_pi.csv"):
            pt, pv = _ref.read_finger_pi_export(
                self.path("controls/finger_pi.csv"))
            return _ref.analyze_finger(pi_t=pt, pi_pct=pv)
        raise FprError("no finger control found (controls/finger_ppg.csv or "
                       "controls/finger_pi.csv)")

    def load_events(self) -> list:
        from . import reference as _ref
        if not self.exists("controls/events.csv"):
            return []
        return _ref.read_events(self.path("controls/events.csv"))

    def load_hr(self):
        """Continuous HR (t, hr_bpm) from controls/heart_rate.csv, or None."""
        from . import reference as _ref
        if not self.exists("controls/heart_rate.csv"):
            return None
        return _ref.read_hr(self.path("controls/heart_rate.csv"))

    def load_temperature(self):
        """(t, {channel: values}) from controls/temperature.csv, or None."""
        from . import reference as _ref
        if not self.exists("controls/temperature.csv"):
            return None
        return _ref.read_temperature(self.path("controls/temperature.csv"))

    def load_humidity(self):
        """(t, humidity) from controls/humidity.csv, or None."""
        from . import reference as _ref
        if not self.exists("controls/humidity.csv"):
            return None
        t, chans = _ref.read_channels(self.path("controls/humidity.csv"))
        return t, next(iter(chans.values()))

    # -- reporting ---------------------------------------------------------
    def missing_inputs(self) -> dict[str, Optional[str]]:
        """Stage -> the required input that is absent (or None if present /
        not required). Drives the INCONCLUSIVE reasons."""
        out: dict[str, Optional[str]] = {}
        for stage, rel in STAGE_INPUTS.items():
            if rel is None:
                out[stage] = None
            elif self.exists(rel):
                out[stage] = None
            else:
                out[stage] = rel
        return out

    def manifest(self) -> dict:
        if self.exists("manifest.json"):
            return self.load_json("manifest.json")
        return {}


def _json_default(o: Any):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def new_session(root: str, *, manifest: Optional[dict] = None) -> FprSession:
    """Create the directory skeleton and (optionally) a provenance manifest."""
    s = FprSession(root=root)
    s.ensure_dirs()
    if manifest is not None and not s.exists("manifest.json"):
        m = dict(manifest)
        m.setdefault("watermark", WATERMARK)
        s.save_json("manifest.json", m)
    return s
