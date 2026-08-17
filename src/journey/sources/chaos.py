"""Chaos injection: per (source, leg) scenario directives, parsed from
strings like "timeout(3)" and loaded from YAML scenario files.

SPEC.md §9 Phase 2a names six directives: ok / timeout(n) / error / empty /
price_shift(x) / slow(n).
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from journey.domain import Leg, Money


class ChaosDirective:
    """Marker base for the six directives. No shared fields."""


@dataclass(frozen=True)
class Ok(ChaosDirective):
    pass


@dataclass(frozen=True)
class Timeout(ChaosDirective):
    seconds: float


@dataclass(frozen=True)
class ErrorResponse(ChaosDirective):
    pass


@dataclass(frozen=True)
class EmptyResponse(ChaosDirective):
    pass


@dataclass(frozen=True)
class PriceShift(ChaosDirective):
    factor: float


@dataclass(frozen=True)
class Slow(ChaosDirective):
    """Answers, but late - and optionally at a shifted price. The factor
    exists because "arrives after the deadline, at this fare" is a single
    real fault: a slow source still returns a number, and which number it
    returns is what decides whether the wait was worth it."""

    seconds: float
    factor: float = 1.0


_DIRECTIVE_PATTERN = re.compile(
    r"^(?P<name>\w+)(\(\s*(?P<arg>-?[0-9.]+)\s*(,\s*(?P<arg2>-?[0-9.]+)\s*)?\))?$"
)

_NAMES = {
    "ok": Ok,
    "timeout": Timeout,
    "error": ErrorResponse,
    "empty": EmptyResponse,
    "price_shift": PriceShift,
    "slow": Slow,
}


def parse_directive(text: str) -> ChaosDirective:
    match = _DIRECTIVE_PATTERN.match(text.strip())
    directive_cls = _NAMES.get(match.group("name")) if match else None
    if directive_cls is None:
        raise ValueError(f"unrecognised chaos directive: {text!r}")
    if directive_cls in (Timeout, PriceShift, Slow):
        if match.group("arg") is None:
            raise ValueError(f"directive {text!r} requires a numeric argument")
        if directive_cls is Slow and match.group("arg2") is not None:
            return Slow(float(match.group("arg")), float(match.group("arg2")))
        return directive_cls(float(match.group("arg")))
    if match.group("arg") is not None:
        raise ValueError(f"directive {text!r} takes no argument")
    return directive_cls()


def leg_key(origin: str, destination: str) -> str:
    """The `leg:` field in a scenario's `cached:` block, e.g.
    London/Berlin -> "london_berlin"."""
    return f"{origin}_{destination}".lower().replace(" ", "_")


@dataclass(frozen=True)
class CachedFare:
    """A price stored by an earlier query, replayed into a scenario so
    UseCached is reachable. Without this nothing ever seeds the cache,
    so the strategy could be generated but never actually demonstrated."""

    source: str
    mode: str
    price: Money
    age_hours: float


@dataclass(frozen=True)
class ChaosScenario:
    name: str
    directives: Mapping[tuple[str, str, str], ChaosDirective] = field(default_factory=dict)
    # Keyed (leg_key, mode) - a cached fare belongs to the leg it was
    # queried for, never to a mode globally.
    cached: Mapping[tuple[str, str], CachedFare] = field(default_factory=dict)

    def directive_for(self, source_name: str, leg: Leg) -> ChaosDirective:
        return self.directives.get((source_name, leg.origin, leg.destination), Ok())

    def cached_for(self, origin: str, destination: str, mode: str) -> tuple[Money, float] | None:
        entry = self.cached.get((leg_key(origin, destination), mode))
        return None if entry is None else (entry.price, entry.age_hours)


def load_scenario(path) -> ChaosScenario:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    directives = {
        (entry["source"], entry["origin"], entry["destination"]): parse_directive(entry["directive"])
        for entry in data.get("directives", [])
    }
    cached = {
        (entry["leg"], entry["mode"]): CachedFare(
            source=entry["source"],
            mode=entry["mode"],
            price=Money(int(entry["price_minor"]), entry.get("currency", "GBP")),
            age_hours=float(entry["age_hours"]),
        )
        for entry in data.get("cached") or []
    }
    return ChaosScenario(name=data["name"], directives=directives, cached=cached)
