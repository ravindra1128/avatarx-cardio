"""
Participant-level splitting with enforced leakage guards.

The single most common way rhythm-detection results are overstated is
partitioning at the window or recording level. This module makes
participant-level splitting the only reachable API and ships
`assert_no_leakage`, which runs in CI on every commit and is attached to every
evaluation report as evidence.

ASSIGNMENT INVARIANT (v2 — corrected after pressure test)
---------------------------------------------------------
A participant's split is a pure function of (participant_id, seed) and of
NOTHING else. Version 1 included the participant's stratum (has-AF,
has-hard-negative, Monk band) in the hash. That was a latent leakage defect:
when a participant's labels evolve — a paroxysmal AF patient whose first AF
episode is captured on visit 3 — their stratum flips, their hash changes, and
they silently migrate between splits. Measured on a 120-participant synthetic
cohort: 9/120 moved when 30 participants gained an AF label, including 2
TRAIN -> TEST migrations (the model had already trained on them). The
participants whose labels evolve are precisely the paroxysmal-AF patients —
the most valuable in the dataset.

The cost of the fix: stratification is now IN EXPECTATION only. With the
programme's case counts this is acceptable (350 AF x 20% test -> 70 +/- 7.5),
and `split_composition()` makes the realised balance visible. If a frozen
dataset's realised composition is unacceptable, the sanctioned remedy is an
APPEND-ONLY ASSIGNMENT LEDGER amendment (a recorded, one-time, minimal
reassignment before the split is ever used) — never a reshuffle.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence, Optional
import hashlib

from .schema import Recording, Split, Rhythm


class LeakageError(AssertionError):
    """Raised when split integrity is violated. Never catch this."""


# --------------------------------------------------------------------------
@dataclass
class SplitPlan:
    train: set[str]
    dev: set[str]
    internal_test: set[str]
    external_test: set[str]

    def assignment(self) -> dict[str, Split]:
        a: dict[str, Split] = {}
        for p in self.train:         a[p] = Split.TRAIN
        for p in self.dev:           a[p] = Split.DEV
        for p in self.internal_test: a[p] = Split.INTERNAL_TEST
        for p in self.external_test: a[p] = Split.EXTERNAL_TEST
        return a

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "dev": len(self.dev),
                "internal_test": len(self.internal_test),
                "external_test": len(self.external_test)}


def _stable_unit_interval(participant_id: str, seed: int) -> float:
    """Deterministic uniform value in [0,1) from participant identity ALONE.

    Cryptographic digest so the mapping is fixed across Python versions,
    platforms and process restarts — `hash()` is salted per process. No label,
    stratum, site or date may enter this function: anything mutable in the
    hash input becomes a migration channel (see module docstring).
    """
    h = hashlib.sha256(f"{seed}|{participant_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def make_participant_splits(
    recordings: Sequence[Recording],
    *,
    fractions: tuple[float, float, float] = (0.65, 0.15, 0.20),
    external_sites: Optional[set[str]] = None,
    seed: int = 20260814,
) -> SplitPlan:
    """Deterministically assign PARTICIPANTS (never recordings) to splits.

    Properties, all under test:
      * pure function of (participant_id, seed) — adding participants moves
        zero existing assignments, and label evolution moves zero assignments;
      * any participant enrolled at an `external_sites` site goes to
        EXTERNAL_TEST regardless of fractions;
      * a participant at multiple sites raises before any assignment is made.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1.0")
    external_sites = external_sites or set()

    by_participant: dict[str, list[Recording]] = defaultdict(list)
    site_of: dict[str, set[str]] = defaultdict(set)
    for r in recordings:
        by_participant[r.participant_id].append(r)
        site_of[r.participant_id].add(r.site_id)

    for p, sites in site_of.items():
        if len(sites) > 1:
            raise LeakageError(
                f"participant {p} appears at multiple sites {sorted(sites)}; "
                "resolve identity before splitting")

    ext = {p for p, s in site_of.items() if s & external_sites}
    f_tr, f_dv, _ = fractions

    tr, dv, te = set(), set(), set()
    for p in sorted(by_participant):
        if p in ext:
            continue
        u = _stable_unit_interval(p, seed)
        if u < f_tr:
            tr.add(p)
        elif u < f_tr + f_dv:
            dv.add(p)
        else:
            te.add(p)

    return SplitPlan(train=tr, dev=dv, internal_test=te, external_test=ext)


def split_composition(recordings: Sequence[Recording]) -> dict:
    """Realised per-split composition — because stratification is now only in
    expectation, the balance must be VISIBLE. Attach to every training report.
    """
    out: dict[str, dict] = {}
    by_split: dict[Split, list[Recording]] = defaultdict(list)
    for r in recordings:
        by_split[r.split].append(r)
    for s, recs in by_split.items():
        pids = {r.participant_id for r in recs}
        af_p, hn_p = set(), set()
        for r in recs:
            rh = {a.rhythm for a in r.rhythm_annotations}
            if Rhythm.AFIB in rh:
                af_p.add(r.participant_id)
            if rh & Rhythm.hard_negatives():
                hn_p.add(r.participant_id)
        out[s.value] = {"participants": len(pids), "recordings": len(recs),
                        "af_participants": len(af_p),
                        "hard_negative_participants": len(hn_p)}
    return out


# --------------------------------------------------------------------------
def assert_no_leakage(recordings: Sequence[Recording],
                      *, check_duplicates: bool = True) -> dict:
    """Hard leakage audit. Raises LeakageError on any violation.

    Run in CI. Attach the returned dict to every evaluation report — an
    auditor should be able to see that the check ran and what it covered.
    """
    report: dict = {"n_recordings": len(recordings)}
    by_split: dict[Split, set[str]] = defaultdict(set)
    for r in recordings:
        if r.split is Split.UNASSIGNED:
            raise LeakageError(f"recording {r.recording_id} has no split assignment")
        by_split[r.split].add(r.participant_id)

    # 1. No participant in more than one split.
    splits = list(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = by_split[splits[i]] & by_split[splits[j]]
            if overlap:
                raise LeakageError(
                    f"{len(overlap)} participant(s) in both {splits[i].value} and "
                    f"{splits[j].value}: {sorted(overlap)[:5]}")
    report["participants_per_split"] = {s.value: len(p) for s, p in by_split.items()}

    # 2. External test must be site-disjoint from everything else.
    ext_sites = {r.site_id for r in recordings if r.split is Split.EXTERNAL_TEST}
    other_sites = {r.site_id for r in recordings if r.split is not Split.EXTERNAL_TEST}
    if ext_sites & other_sites:
        raise LeakageError(
            f"external test shares site(s) with development data: "
            f"{sorted(ext_sites & other_sites)}")
    report["external_sites"] = sorted(ext_sites)

    # 3. No duplicate content across splits (same video hashed twice).
    if check_duplicates:
        seen: dict[str, tuple[str, Split]] = {}
        for r in recordings:
            h = r.content_hash()
            if h in seen and seen[h][1] is not r.split:
                raise LeakageError(
                    f"duplicate content {h} in {seen[h][1].value} and {r.split.value} "
                    f"({seen[h][0]} vs {r.recording_id})")
            seen[h] = (r.recording_id, r.split)
        report["unique_content_hashes"] = len(seen)

    # 4. Internal/external test must contain AF and hard negatives, else the
    #    reported specificity is meaningless.
    for s in (Split.INTERNAL_TEST, Split.EXTERNAL_TEST):
        recs = [r for r in recordings if r.split is s]
        if not recs:
            continue
        rhythms = {a.rhythm for r in recs for a in r.rhythm_annotations}
        if Rhythm.AFIB not in rhythms:
            raise LeakageError(f"{s.value} contains no AFIB annotations")
        if not (rhythms & Rhythm.hard_negatives()):
            raise LeakageError(
                f"{s.value} contains no hard-negative arrhythmias; specificity "
                "measured on it would be uninterpretable")

    report["composition"] = split_composition(recordings)
    return report


def assert_preprocessing_is_split_safe(fit_on: Sequence[Recording]) -> None:
    """Guard for any fitted preprocessing (normalisation, PCA, calibration).

    Anything fitted must be fitted on TRAIN only. Fitting a per-channel mean on
    the full corpus is a subtle and extremely common leak.
    """
    bad = {r.split for r in fit_on} - {Split.TRAIN}
    if bad:
        raise LeakageError(
            f"preprocessing/calibration fitted on non-train splits: "
            f"{sorted(s.value for s in bad)}")


def windows_are_participant_grouped(window_participant_ids: Iterable[str],
                                    split_of_participant: dict[str, Split]) -> None:
    """Assert that a windowed dataset never crosses the participant wall."""
    seen: dict[str, Split] = {}
    for pid in window_participant_ids:
        s = split_of_participant.get(pid)
        if s is None:
            raise LeakageError(f"window from unassigned participant {pid}")
        if pid in seen and seen[pid] is not s:
            raise LeakageError(f"participant {pid} windows span multiple splits")
        seen[pid] = s


def registry_dataset_splits(dataset_id: str, *, seed: int = 20260814,
                            fractions: tuple = (0.65, 0.15, 0.20),
                            registry_path=None) -> dict:
    """v0.2 M2.9: splits FROM THE REGISTRY, by dataset id.

    Reads the registered dataset's manifests and assigns recordings to
    TRAIN/DEV/INTERNAL_TEST via the same identity-only participant hash as
    make_participant_splits — then PROVES participant- AND session-
    disjointness before returning (a session straddling two splits is a
    leak even when participants are disjoint on paper)."""
    import json as _json
    import pathlib as _pl
    from datasets.registry import lookup, verify_registered, RegistryError

    row = lookup(dataset_id, registry_path=registry_path)
    if row is None:
        raise RegistryError(f"unknown dataset id {dataset_id!r}")
    verify_registered(row["path"], registry_path=registry_path)

    names = [Split.TRAIN, Split.DEV, Split.INTERNAL_TEST]
    cuts = (fractions[0], fractions[0] + fractions[1])
    out = {n.value: [] for n in names}
    part_of = {n.value: set() for n in names}
    sess_of = {n.value: set() for n in names}
    for f in sorted(_pl.Path(row["path"]).glob("*.recording.json")):
        with open(f) as fh:
            d = _json.load(fh)
        pid = str(d.get("participant_id"))
        rid = str(d.get("recording_id", f.stem))
        sid = str(d.get("session_id", rid))
        u = _stable_unit_interval(pid, seed)
        split = (names[0] if u < cuts[0]
                 else names[1] if u < cuts[1] else names[2])
        out[split.value].append(rid)
        part_of[split.value].add(pid)
        sess_of[split.value].add(sid)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = part_of[a.value], part_of[b.value]
            if pa & pb:
                raise LeakageError(f"participants straddle {a.value}/"
                                   f"{b.value}: {sorted(pa & pb)}")
            sa, sb = sess_of[a.value], sess_of[b.value]
            if sa & sb:
                raise LeakageError(f"sessions straddle {a.value}/"
                                   f"{b.value}: {sorted(sa & sb)}")
    return {"dataset_id": row["id"], "seed": seed,
            "splits": out,
            "participants": {k: sorted(v) for k, v in part_of.items()},
            "sessions": {k: sorted(v) for k, v in sess_of.items()}}
