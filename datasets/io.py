"""
Recording JSON round-trip (T5/T7).

`Recording.to_json()` serialises with dataclasses.asdict; this module is
the inverse. Unknown keys are REJECTED, not ignored — a manifest with a
misspelled field would otherwise silently lose a validity-relevant fact
and the gate would judge a record that was never fully read.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

from .schema import (Recording, CaptureConfig, SyncRecord, SyncMethod,
                     RhythmAnnotation, Rhythm, Split,
                     ChallengeRecord, ParticipantContext, Medications,
                     ConductionRatio, FlutterType)


def _build(cls, d: dict) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown field(s) {sorted(unknown)}")
    return cls(**d)


def recording_from_json(text: str) -> Recording:
    d = json.loads(text)
    if not isinstance(d, dict):
        raise ValueError("recording JSON must be an object")
    d = dict(d)
    try:
        d["capture"] = _build(CaptureConfig, d["capture"])
        sync = dict(d["sync"])
        sync["method"] = SyncMethod(sync["method"])
        d["sync"] = _build(SyncRecord, sync)
        anns = []
        for a in d.get("rhythm_annotations", []):
            a = dict(a)
            a["rhythm"] = Rhythm(a["rhythm"])
            # v0.6: the flutter block must come back as ENUMS, not
            # strings — flutter_label_problems only tests `is None`, so
            # a half-decoded annotation would pass the label bar while
            # ConductionRatio.divisor raised AttributeError downstream
            # (review finding)
            if a.get("conduction_ratio") is not None:
                a["conduction_ratio"] = ConductionRatio(
                    a["conduction_ratio"])
            if a.get("flutter_type") is not None:
                a["flutter_type"] = FlutterType(a["flutter_type"])
            anns.append(_build(RhythmAnnotation, a))
        d["rhythm_annotations"] = anns
        d["split"] = Split(d.get("split", Split.UNASSIGNED))
        # v0.4 additive blocks — optional, but validated hard when present
        if d.get("challenge") is not None:
            d["challenge"] = _build(ChallengeRecord, dict(d["challenge"]))
        if d.get("participant_context") is not None:
            pc = dict(d["participant_context"])
            if pc.get("meds") is not None:
                pc["meds"] = _build(Medications, dict(pc["meds"]))
            d["participant_context"] = _build(ParticipantContext, pc)
    except KeyError as e:
        raise ValueError(f"recording JSON missing required section: {e}")
    return _build(Recording, d)


def load_recording(path: str) -> Recording:
    with open(path) as f:
        return recording_from_json(f.read())
