"""
Capture campaigns (v0.2 M2.7, L6) — the data engine's demand side.

A campaign manifest is a YAML spec (parsed by the repo's audited YAML
subset — no new dependency) declaring WHAT the future models need:
target N, enrollment quotas (Fitzpatrick I-VI, device matrix, lighting
conditions), the consent-form version every session must match, and the
per-session capture checklist. Sessions accumulate as
`<campaign_dir>/sessions/<id>.session.json`; `campaign_status()` reports
quota fill so collection is steered by the gaps, not by convenience
sampling — the fairness properties of every later model are decided
here, at enrollment.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Optional

from configs import parse_yaml_subset

REQUIRED_SPEC_KEYS = ("id", "version", "consent_form_version", "target_n",
                      "quotas", "session_checklist")
QUOTA_AXES = ("fitzpatrick", "devices", "lighting")
# v0.4-vascular campaign axes (owner plan: wide age band, deliberate
# BP-range quotas). Optional per campaign; anything OUTSIDE the known
# axes fails closed — a quota the reporter would silently ignore is a
# fairness hole, not a convenience.
# v0.6-flutter adds rhythm_class (the recruitment problem: flutter
# positives come almost only from cardioversion/EP clinics, while the
# negatives that decide specificity come from a prospective
# regular-tachycardia cohort) and conduction_ratio (2:1 dominates
# naturally, so 3:1 and 4:1 must be quota'd deliberately or the
# per-ratio gate F2 can never be computed).
OPTIONAL_QUOTA_AXES = ("age_band", "bp_range", "maneuver",
                       "rhythm_class", "conduction_ratio")
# ONE map from quota axis to the session field that carries it. It was
# duplicated, and a new axis added to only one copy raised KeyError on
# the required-field scan instead of failing closed with a reason.
AXIS_KEY = {"fitzpatrick": "fitzpatrick", "devices": "device",
            "lighting": "lighting", "age_band": "age_band",
            "bp_range": "bp_range", "maneuver": "maneuver",
            "rhythm_class": "rhythm_class",
            "conduction_ratio": "conduction_ratio"}
_MISSING_AXIS_KEYS = (set(QUOTA_AXES) | set(OPTIONAL_QUOTA_AXES)) - set(AXIS_KEY)


class CampaignError(ValueError):
    pass


@dataclass
class CampaignStatus:
    campaign_id: str
    target_n: int
    enrolled_participants: int
    quota_fill: dict            # axis -> {bucket: {"target": t, "have": h}}
    unmet: list                 # ["fitzpatrick:V needs 3 more", ...]
    problems: list              # per-session integrity findings
    sessions: int = 0
    retest_pairs: int = 0       # participants with >= 2 valid sessions
    scan_series: int = 0        # participants with >= series.min_scans
                                # valid sessions (v0.6 F3 sub-protocol)

    def complete(self) -> bool:
        return (not self.unmet and not self.problems and
                self.enrolled_participants >= self.target_n)


def load_campaign(path) -> dict:
    if _MISSING_AXIS_KEYS:
        # fail closed in the campaign path with the module's own error
        # type, not as an AssertionError at import (which -O deletes and
        # which would break every importer)
        raise CampaignError(
            f"quota axes {sorted(_MISSING_AXIS_KEYS)} have no session "
            "field mapping — the status reporter would ignore them")
    text = pathlib.Path(path).read_text()
    spec = parse_yaml_subset(text)
    camp = spec.get("campaign")
    if not isinstance(camp, dict):
        raise CampaignError("manifest must contain a top-level 'campaign' "
                            "mapping")
    missing = [k for k in REQUIRED_SPEC_KEYS if k not in camp]
    if missing:
        raise CampaignError(f"campaign manifest missing keys: {missing}")
    for axis in QUOTA_AXES:
        if axis not in camp["quotas"] or not isinstance(camp["quotas"][axis],
                                                       dict):
            raise CampaignError(f"quotas must include the {axis!r} axis")
    unknown = set(camp["quotas"]) - set(QUOTA_AXES) - \
        set(OPTIONAL_QUOTA_AXES)
    if unknown:
        raise CampaignError(
            f"unrecognized quota axes {sorted(unknown)} — the status "
            "reporter would silently ignore them, fail closed")
    retest = camp.get("retest")
    if retest is not None and (not isinstance(retest, dict)
                               or "target_pairs" not in retest):
        raise CampaignError("retest block must declare target_pairs")
    series = camp.get("series")
    if series is not None and (not isinstance(series, dict)
                               or "target_series" not in series):
        raise CampaignError("series block must declare target_series")
    if not isinstance(camp["session_checklist"], list) or \
            not camp["session_checklist"]:
        raise CampaignError("session_checklist must be a non-empty list")
    return camp


def _load_sessions(campaign_dir: pathlib.Path) -> list:
    out = []
    sdir = campaign_dir / "sessions"
    if sdir.is_dir():
        for p in sorted(sdir.glob("*.session.json")):
            with open(p) as f:
                rec = json.load(f)
            rec["_file"] = p.name
            out.append(rec)
    return out


def campaign_status(campaign_dir) -> CampaignStatus:
    campaign_dir = pathlib.Path(campaign_dir)
    spec_path = campaign_dir / "campaign.yaml"
    if not spec_path.exists():
        raise CampaignError(f"{spec_path} not found")
    camp = load_campaign(spec_path)
    sessions = _load_sessions(campaign_dir)

    problems = []
    valid = []
    required = ["participant_id", "fitzpatrick", "device", "lighting",
                "consent_form_version", "checklist"] + [
        AXIS_KEY[a] for a in OPTIONAL_QUOTA_AXES
        if a in camp["quotas"]
        and AXIS_KEY[a] not in ("fitzpatrick", "device", "lighting")]
    for s in sessions:
        sid = s.get("_file", "?")
        for k in required:
            if k not in s:
                # fail closed: a session missing a quota'd axis field
                # must surface as a problem, never crash or silently
                # undercount (review finding)
                problems.append(f"{sid}: missing field {k!r}")
                break
        else:
            if s["consent_form_version"] != camp["consent_form_version"]:
                problems.append(
                    f"{sid}: consent form {s['consent_form_version']!r} != "
                    f"campaign {camp['consent_form_version']!r}")
                continue
            unchecked = [c for c in camp["session_checklist"]
                         if not s["checklist"].get(c)]
            if unchecked:
                problems.append(f"{sid}: checklist incomplete: {unchecked}")
                continue
            valid.append(s)

    participants = {}
    for s in valid:
        participants.setdefault(s["participant_id"], s)

    fill = {}
    unmet = []
    axis_key = AXIS_KEY
    participant_axes = ("fitzpatrick", "age_band", "bp_range")
    for axis in [a for a in QUOTA_AXES + OPTIONAL_QUOTA_AXES
                 if a in camp["quotas"]]:
        fill[axis] = {}
        for bucket, target in camp["quotas"][axis].items():
            if axis in participant_axes:
                have = sum(1 for p in participants.values()
                           if str(p.get(axis_key[axis])) == str(bucket))
            else:
                have = sum(1 for s in valid
                           if str(s[axis_key[axis]]) == str(bucket))
            fill[axis][str(bucket)] = {"target": int(target),
                                       "have": int(have)}
            if have < int(target):
                unmet.append(f"{axis}:{bucket} needs {int(target) - have} "
                             "more")

    # retest sub-protocol: a pair = a participant with >= 2 valid
    # sessions — sharing a maneuver when the campaign quotas maneuvers
    # (v0.5: 'same maneuver, second visit'; review finding)
    if "maneuver" in camp["quotas"]:
        keys = {}
        for s in valid:
            keys.setdefault((s["participant_id"],
                             s.get("maneuver")), []).append(s)
        retest_pairs = sum(
            1 for (_pid, _m), ss in keys.items() if len(ss) >= 2)
    else:
        retest_pairs = sum(
            1 for pid in {s["participant_id"] for s in valid}
            if sum(1 for s in valid
                   if s["participant_id"] == pid) >= 2)
    retest = camp.get("retest")
    if retest is not None and retest_pairs < int(retest["target_pairs"]):
        unmet.append(f"retest pairs: {retest_pairs} of "
                     f"{int(retest['target_pairs'])}")

    # v0.6 SERIES sub-protocol: the serial conduction-ratio signature is
    # undefined below three scans of one participant, so a pair is not
    # enough and the floor is explicit rather than assumed.
    series = camp.get("series")
    n_series = 0
    if series is not None:
        min_scans = int(series.get("min_scans", 3))
        counts = {}
        for s in valid:
            counts[s["participant_id"]] = counts.get(
                s["participant_id"], 0) + 1
        n_series = sum(1 for c in counts.values() if c >= min_scans)
        target_series = int(series["target_series"])
        if n_series < target_series:
            unmet.append(f"scan series (>= {min_scans} scans): "
                         f"{n_series} of {target_series}")

    return CampaignStatus(
        campaign_id=str(camp["id"]), target_n=int(camp["target_n"]),
        enrolled_participants=len(participants),
        quota_fill=fill, unmet=unmet, problems=problems,
        sessions=len(sessions), retest_pairs=retest_pairs,
        scan_series=n_series)
