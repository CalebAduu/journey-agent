from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Money:
    minor_units: int
    currency: str = "GBP"


class SourceStatus(Enum):
    FRESH = "fresh"
    STALE = "stale"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    NEVER_CALLED = "never_called"


@dataclass(frozen=True)
class Observation:
    source: str
    mode: str
    status: SourceStatus


class LegStatus:
    """Marker base with no fields: subclasses can never be equal to each
    other through a shared shape, only through being the same subclass."""


@dataclass(frozen=True)
class Unknown(LegStatus):
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class Empty(LegStatus):
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class NotApplicable(LegStatus):
    reason: str


@dataclass(frozen=True)
class Conflicted(LegStatus):
    low: Money
    high: Money
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class LegView:
    origin: str
    destination: str
    results: dict[str, LegStatus]
