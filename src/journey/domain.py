from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType


@functools.total_ordering
@dataclass(frozen=True, slots=True)
class Money:
    minor_units: int
    currency: str = "GBP"

    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"cannot add {self.currency} to {other.currency}")
        return Money(self.minor_units + other.minor_units, self.currency)

    def __lt__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError(f"cannot compare {self.currency} to {other.currency}")
        return self.minor_units < other.minor_units

    def __str__(self) -> str:
        amount = f"{self.minor_units / 100:.2f}"
        return f"£{amount}" if self.currency == "GBP" else f"{amount} {self.currency}"


class SourceStatus(Enum):
    """Per-source-call provenance: what happened when we asked this source."""

    FRESH = "fresh"
    STALE = "stale"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    NEVER_CALLED = "never_called"


@dataclass(frozen=True, slots=True)
class Observation:
    """A single source's result for one mode on one leg."""

    source: str
    mode: str
    status: SourceStatus
    price: Money | None = None
    duration: timedelta | None = None
    observed_at: datetime | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.price is not None and self.status not in (SourceStatus.FRESH, SourceStatus.STALE):
            raise ValueError(
                f"Observation has a price but status={self.status}: a price must be "
                "accompanied by a status that explains it (FRESH or STALE)"
            )

    def is_actionable(self) -> bool:
        return self.status in (SourceStatus.FRESH, SourceStatus.STALE) and self.price is not None


class LegStatus:
    """Sealed marker base for the four structurally distinct epistemic states
    a mode can be in on a leg. Carries no fields or behaviour of its own —
    there is no shared shape for two different states to be confused through.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class NotApplicable(LegStatus):
    """Mode is structurally implausible on this leg. No source was called."""

    reason: str
    basis: str = "physical"  # "physical" | "geometric" | "learned" | "llm_prior" | "default"


@dataclass(frozen=True, slots=True)
class Empty(LegStatus):
    """Every source we called answered: nothing available."""

    observations: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class Unknown(LegStatus):
    """A source failed, timed out, or errored — no answer was received."""

    observations: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class Conflicted(LegStatus):
    """Two or more sources answered with prices that disagree beyond threshold."""

    low: Money
    high: Money
    observations: tuple[Observation, ...]


ModeResult = Observation | NotApplicable | Empty | Unknown | Conflicted


@dataclass(frozen=True, slots=True)
class LegView:
    """The merged view of one leg across every candidate mode."""

    origin: str
    destination: str
    results: Mapping[str, ModeResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))


class StrategyKind(Enum):
    COMMIT = "commit"
    WAIT = "wait"
    USE_CACHED = "use_cached"
    REPLAN = "replan"
    ABANDON_LEG = "abandon_leg"


@dataclass(frozen=True, slots=True)
class Strategy:
    kind: StrategyKind
    leg_index: int
    reason: str
    wait_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class JourneyPlan:
    legs: tuple[LegView, ...]
    strategies: tuple[Strategy, ...] = ()
