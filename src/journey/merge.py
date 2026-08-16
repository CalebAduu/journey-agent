"""Merge: combine every source's Observation for a leg into one LegStatus
per mode, and score how certain each outcome is
"""

from collections.abc import Sequence
from typing import Protocol

from journey.domain import (
    Available,
    Conflicted,
    Empty,
    Leg,
    LegStatus,
    LegView,
    Money,
    NotApplicable,
    Observation,
    SourceStatus,
    Unknown,
)

# Disagreement beyond this relative spread counts as a conflict 
CONFLICT_THRESHOLD = 0.15

# Partial certainty for a genuine disagreement between sources 
CONFLICTED_CERTAINTY = 0.5

# A live disruption alert doesn't make an answer wrong, just less clean -
# a third tier between confirmed and unknown
DISRUPTION_CERTAINTY_PENALTY = 0.3

_ANSWERED_STATUSES = (SourceStatus.FRESH, SourceStatus.STALE)
_FAILED_STATUSES = (SourceStatus.TIMED_OUT, SourceStatus.ERROR, SourceStatus.NEVER_CALLED)


class Feasibility(Protocol):
    def candidate_modes(self, leg: Leg) -> Sequence[str]: ...
    def reason_if_not_applicable(self, leg: Leg, mode: str) -> str | None: ...


def merge(observations: Sequence[Observation], leg: Leg, feasibility: Feasibility) -> LegView:
    by_mode: dict[str, list[Observation]] = {}
    for obs in observations:
        by_mode.setdefault(obs.mode, []).append(obs)

    modes = set(feasibility.candidate_modes(leg)) | set(by_mode.keys())

    results = []
    for mode in sorted(modes):
        reason = feasibility.reason_if_not_applicable(leg, mode)
        if reason is not None:
            results.append((mode, NotApplicable(reason=reason)))
        else:
            results.append((mode, _classify_mode(by_mode.get(mode, []))))

    return LegView(
        origin=leg.origin,
        destination=leg.destination,
        results=tuple(results),
        distance_km=leg.distance_km,
        abandonable=leg.abandonable,
    )


def _classify_mode(mode_observations: Sequence[Observation]) -> LegStatus:
    if not mode_observations:
        # Feasible, but nothing was ever attempted - honest ignorance,
        # never presented as a real "nothing available" answer.
        return Unknown(observations=())

    actionable = [
        o
        for o in mode_observations
        if o.status in _ANSWERED_STATUSES and (o.price is not None or o.duration is not None)
    ]

    if actionable:
        conflict_fields = _detect_conflict(actionable)
        if conflict_fields is not None:
            return Conflicted(observations=tuple(mode_observations), **conflict_fields)
        return Available(observations=tuple(mode_observations))

    if any(o.status in _FAILED_STATUSES for o in mode_observations):
        return Unknown(observations=tuple(mode_observations))

    return Empty(observations=tuple(mode_observations))


def _detect_conflict(actionable: Sequence[Observation]) -> dict | None:
    """Price and duration are checked independently - price-vs-price,
    duration-vs-duration - never against each other. Transitous-only legs
    never carry a price, so duration is the only dimension real sources
    can actually disagree on."""
    prices = [o.price for o in actionable if o.price is not None]
    durations = [o.duration for o in actionable if o.duration is not None]

    price_conflict = len(prices) >= 2 and _spread_exceeds_threshold([p.minor_units for p in prices])
    duration_conflict = len(durations) >= 2 and _spread_exceeds_threshold(
        [d.total_seconds() for d in durations]
    )

    if not price_conflict and not duration_conflict:
        return None

    fields: dict = {"dimension": "price" if price_conflict else "duration"}
    if price_conflict:
        fields["price_low"] = Money(min(p.minor_units for p in prices), prices[0].currency)
        fields["price_high"] = Money(max(p.minor_units for p in prices), prices[0].currency)
    if duration_conflict:
        fields["duration_low"] = min(durations)
        fields["duration_high"] = max(durations)
    return fields


def _spread_exceeds_threshold(values: Sequence[float]) -> bool:
    low, high = min(values), max(values)
    if low == 0:
        return high != 0
    return (high - low) / low > CONFLICT_THRESHOLD


def certainty(status: LegStatus) -> float:
    if isinstance(status, (NotApplicable, Empty)):
        return 1.0
    if isinstance(status, Unknown):
        return 0.0
    if isinstance(status, Conflicted):
        return CONFLICTED_CERTAINTY
    if isinstance(status, Available):
        successful = [o for o in status.observations if o.status in _ANSWERED_STATUSES]
        if any(o.detail for o in successful):
            return 1.0 - DISRUPTION_CERTAINTY_PENALTY
        return 1.0
    raise TypeError(f"unknown LegStatus variant: {type(status).__name__}")
