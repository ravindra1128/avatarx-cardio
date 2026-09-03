"""
Endpoint-head registry (v0.2, M1.1) — invariant 14: EXTENSION = NEW HEAD,
NOT NEW PIPELINE.

A head is a plug-in that consumes the BeatLattice (L3) plus a context dict
and returns a HeadResult. Heads never re-extract signal, never bypass the
L2 quality gates, and never emit user-facing text outside the sanctioned
tables in `datasets/schema.py`. Anything that cannot be expressed this way
is out of scope by definition and must be escalated, not built.

The `afib` head is the decision head: it IS the production gated path
(spec invariant 5) and can never be disabled.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from datasets.schema import MeasurementClass


class HeadRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class HeadResult:
    """What a head returns. `measurement_class` is invariant 9's tag:
    app/ may render MEASURED and INFERRED_RHYTHM results only."""
    head: str
    version: str
    measurement_class: MeasurementClass
    value: dict
    confidence: Optional[float] = None
    reasons: list = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.measurement_class, MeasurementClass):
            raise TypeError("measurement_class must be a MeasurementClass "
                            f"enum member, got {self.measurement_class!r}")
        if not isinstance(self.value, dict):
            raise TypeError("value must be a dict payload")

    def to_dict(self) -> dict:
        return {"head": self.head, "version": self.version,
                "measurement_class": self.measurement_class.value,
                "value": self.value,
                "confidence": self.confidence,
                "reasons": list(self.reasons)}


class EndpointHead(ABC):
    """Contract for every endpoint head.

    Class attributes: `name` (registry key), `version`, and
    `required_inputs` — the BeatLattice/context keys the head consumes,
    declared so the pipeline can verify availability before running.
    """
    name: str = ""
    version: str = ""
    required_inputs: tuple = ()

    @abstractmethod
    def run(self, lattice, context: dict):
        """BeatLattice + context -> HeadResult (or list of HeadResult)."""


_REGISTRY: dict = {}
MANDATORY_HEAD = "afib"          # the decision head — cannot be disabled
_DEFAULT_ENABLED = ["afib", "rate_flags", "rhythm_map"]


def head_status(head_result) -> str:
    """"value" or "abstained", from each head's OWN abstention shape.

    ONE implementation, because two would drift: the research surfaces
    (app/research_tracks.py on the results screen, research/investigation
    on the CLI) both classify head results and must agree. The shapes:
    the rhythm heads say `abstained: True`; the vascular and vasotone
    heads say `available: False` (or `inert: True`); the fitness head
    declines by leaving `category` None.
    """
    hr = head_result or {}
    if not isinstance(hr, dict):
        hr = getattr(hr, "to_dict", lambda: {})()
    v = hr.get("value") or {}
    if v.get("abstained") is True or v.get("available") is False \
            or v.get("inert") is True:
        return "abstained"
    if hr.get("head") == "fitness" and "category" in v \
            and v.get("category") is None:
        return "abstained"
    return "value"


def register_head(cls) -> "EndpointHead":
    if not cls.name:
        raise HeadRegistryError("head class must set a non-empty name")
    if cls.name in _REGISTRY:
        raise HeadRegistryError(f"head {cls.name!r} is already registered")
    inst = cls()
    _REGISTRY[cls.name] = inst
    return inst


def get_head(name: str) -> "EndpointHead":
    try:
        return _REGISTRY[name]
    except KeyError:
        raise HeadRegistryError(
            f"unknown head {name!r}; available: {sorted(_REGISTRY)}") from None


def available_heads() -> list:
    return sorted(_REGISTRY)


def enabled_heads(cfg: dict, override: Optional[list] = None) -> list:
    """Heads to run, from config decision.heads.enabled (or an explicit
    CLI override). The afib decision head is mandatory: a pipeline without
    a gated decision is not a production path (spec invariant 5)."""
    hc = ((cfg or {}).get("decision", {}) or {}).get("heads") or {}
    names = list(override if override is not None
                 else hc.get("enabled", _DEFAULT_ENABLED))
    if MANDATORY_HEAD not in names:
        raise HeadRegistryError(
            f"the {MANDATORY_HEAD!r} decision head cannot be disabled")
    return [get_head(n) for n in names]
