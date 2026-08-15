"""Chaos injection: per (source, leg) scenario directives, parsed from
strings like "timeout(3)" and loaded from YAML scenario files.

SPEC.md §9 Phase 2a names six directives: ok / timeout(n) / error / empty /
price_shift(x) / slow(n).
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from journey.domain import Leg


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
    seconds: float


_DIRECTIVE_PATTERN = re.compile(r"^(?P<name>\w+)(\((?P<arg>-?[0-9.]+)\))?$")

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
        return directive_cls(float(match.group("arg")))
    return directive_cls()


@dataclass(frozen=True)
class ChaosScenario:
    name: str
    directives: Mapping[tuple[str, str, str], ChaosDirective] = field(default_factory=dict)

    def directive_for(self, source_name: str, leg: Leg) -> ChaosDirective:
        return self.directives.get((source_name, leg.origin, leg.destination), Ok())


def load_scenario(path) -> ChaosScenario:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    directives = {
        (entry["source"], entry["origin"], entry["destination"]): parse_directive(entry["directive"])
        for entry in data.get("directives", [])
    }
    return ChaosScenario(name=data["name"], directives=directives)
