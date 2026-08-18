""" strategy generation filters.

Wait only if genuinely pending; UseCached only if fresh
enough; Replan only via an independent healthy alternative; AbandonLeg
only if the journey survives without the leg; Commit for any actionable
observation, never treating a missing price as zero.
"""

from datetime import timedelta

from journey.domain import (
    Available,
    LegView,
    Money,
    Observation,
    SourceStatus,
    StrategyKind,
    Unknown,
)
from journey.strategies import WAIT_DURATIONS_SECONDS, generate


class FakeRegistry:
    def __init__(self, pending_sources: set[str]):
        self.pending_sources = pending_sources

    def is_pending(self, source_name: str) -> bool:
        return source_name in self.pending_sources


class FakeCache:
    """Keyed by (origin, destination, mode): a cached fare belongs to one
    leg, not to a mode globally - the same mode on a different leg is a
    different query with a different price."""

    def __init__(self, entries: dict[tuple[str, str, str], tuple[Money, float]] | None = None):
        self.entries = entries or {}

    def get(self, origin: str, destination: str, mode: str):
        return self.entries.get((origin, destination, mode))


EMPTY_REGISTRY = FakeRegistry(pending_sources=set())
EMPTY_CACHE = FakeCache()


def test_timed_out_generates_wait_but_error_does_not():
    timed_out = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)
    errored = Observation(source="stub-flight-b", mode="flight", status=SourceStatus.ERROR)
    leg_view = LegView(
        origin="Sheffield",
        destination="London",
        results=(("flight", Unknown(observations=(timed_out, errored))),),
    )
    registry = FakeRegistry(pending_sources={"stub-flight"})  # not stub-flight-b: it already answered

    strategies = generate(leg_view, registry, budget=300.0, cache=EMPTY_CACHE)

    waits = [s for s in strategies if s.kind is StrategyKind.WAIT]
    assert any(s.source == "stub-flight" for s in waits)
    assert not any(s.source == "stub-flight-b" for s in waits)


def test_replan_never_offers_an_alternative_dependent_on_the_failed_source():
    rail_failure = Observation(source="transitous", mode="rail", status=SourceStatus.TIMED_OUT)
    # coach is "healthy" but shares the same failed source - rerouting
    # into the same blindness, not a real replan.
    coach_same_source = Observation(
        source="transitous", mode="coach", status=SourceStatus.FRESH, duration=timedelta(minutes=200)
    )
    flight_independent = Observation(
        source="stub-flight", mode="flight", status=SourceStatus.FRESH, price=Money(9000, "GBP")
    )
    leg_view = LegView(
        origin="Sheffield",
        destination="London",
        results=(
            ("rail", Unknown(observations=(rail_failure,))),
            ("coach", Available(observations=(coach_same_source,))),
            ("flight", Available(observations=(flight_independent,))),
        ),
    )

    strategies = generate(leg_view, EMPTY_REGISTRY, budget=300.0, cache=EMPTY_CACHE)

    replans = [s for s in strategies if s.kind is StrategyKind.REPLAN]
    assert {s.mode for s in replans} == {"flight"}


def test_all_healthy_yields_a_single_commit():
    rail = Observation(source="transitous", mode="rail", status=SourceStatus.FRESH, duration=timedelta(minutes=90))
    leg_view = LegView(
        origin="Sheffield",
        destination="London",
        results=(("rail", Available(observations=(rail,))),),
    )

    strategies = generate(leg_view, EMPTY_REGISTRY, budget=300.0, cache=EMPTY_CACHE)

    assert len(strategies) == 1
    assert strategies[0].kind is StrategyKind.COMMIT


def test_timeout_scenario_yields_at_least_two_strategies():
    flight_timeout = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)
    rail = Observation(source="transitous", mode="rail", status=SourceStatus.FRESH, duration=timedelta(minutes=90))
    leg_view = LegView(
        origin="Sheffield",
        destination="London",
        results=(
            ("flight", Unknown(observations=(flight_timeout,))),
            ("rail", Available(observations=(rail,))),
        ),
    )
    registry = FakeRegistry(pending_sources={"stub-flight"})

    strategies = generate(leg_view, registry, budget=300.0, cache=EMPTY_CACHE)

    assert len(strategies) >= 2


def test_wait_longer_than_remaining_budget_is_not_generated():
    assert WAIT_DURATIONS_SECONDS == (3.0, 8.0)  # test assumes these exact candidates
    timed_out = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)
    leg_view = LegView(
        origin="Sheffield",
        destination="London",
        results=(("flight", Unknown(observations=(timed_out,))),),
    )
    registry = FakeRegistry(pending_sources={"stub-flight"})

    strategies = generate(leg_view, registry, budget=5.0, cache=EMPTY_CACHE)

    waits = [s for s in strategies if s.kind is StrategyKind.WAIT]
    assert any(s.wait_seconds == 3.0 for s in waits)
    assert not any(s.wait_seconds == 8.0 for s in waits)


def test_cache_entry_older_than_max_age_yields_no_use_cached():
    timed_out = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)
    leg_view = LegView(
        origin="Sheffield",
        destination="London",
        results=(("flight", Unknown(observations=(timed_out,))),),
    )
    key = ("Sheffield", "London", "flight")
    stale_cache = FakeCache({key: (Money(9000, "GBP"), 100.0)})  # > MAX_CACHE_AGE_HOURS (72)
    fresh_cache = FakeCache({key: (Money(9000, "GBP"), 10.0)})

    stale_strategies = generate(leg_view, EMPTY_REGISTRY, budget=300.0, cache=stale_cache)
    fresh_strategies = generate(leg_view, EMPTY_REGISTRY, budget=300.0, cache=fresh_cache)

    assert not any(s.kind is StrategyKind.USE_CACHED for s in stale_strategies)
    assert any(s.kind is StrategyKind.USE_CACHED for s in fresh_strategies)


def test_cached_fare_only_applies_to_its_own_leg():
    """A cached London->Berlin flight price must not surface as a
    UseCached option on Sheffield->London - it's a different query."""
    timed_out = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)
    other_leg = LegView(
        origin="Sheffield",
        destination="London",
        results=(("flight", Unknown(observations=(timed_out,))),),
    )
    cache = FakeCache({("London", "Berlin", "flight"): (Money(7140, "GBP"), 26.0)})

    strategies = generate(other_leg, EMPTY_REGISTRY, budget=300.0, cache=cache)

    assert not any(s.kind is StrategyKind.USE_CACHED for s in strategies)


def test_use_cached_carries_its_age_and_asymmetric_drift_interval():
    """The row has to show how old the price is and which way it can
    move, so the age travels on the Strategy rather than being parsed
    back out of the reason string."""
    timed_out = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)
    leg_view = LegView(
        origin="London",
        destination="Berlin",
        results=(("flight", Unknown(observations=(timed_out,))),),
    )
    cache = FakeCache({("London", "Berlin", "flight"): (Money(7140, "GBP"), 26.0)})

    strategies = generate(leg_view, EMPTY_REGISTRY, budget=300.0, cache=cache)

    cached = next(s for s in strategies if s.kind is StrategyKind.USE_CACHED)
    assert cached.cache_age_hours == 26.0
    # cached price is the floor, drift gives the ceiling - fares ratchet up
    assert cached.cost_low == Money(7140, "GBP")
    assert cached.cost_high.minor_units > cached.cost_low.minor_units
    assert cached.cost_basis == "stale"


def test_no_use_cached_once_the_mode_has_a_live_price():
    """A cached fare is a fallback for not having a live one. Observed:
    after a wait resolved, the mode had a fresh £180 quote AND the
    26h-old cached price was still offered - and the cached strategy won,
    because with source=None it also inherited the *live* observation's
    duration. That pairs yesterday's price with today's journey time, an
    option that exists nowhere, and beats the real quote on the strength
    of it.
    """
    live = Observation(
        source="stub-flight", mode="flight", status=SourceStatus.FRESH,
        price=Money(18000, "GBP"), duration=timedelta(hours=2),
    )
    leg_view = LegView(
        origin="London",
        destination="Berlin",
        results=(("flight", Available(observations=(live,))),),
        distance_km=930.0,
    )
    cache = FakeCache({("London", "Berlin", "flight"): (Money(7140, "GBP"), 26.0)})

    strategies = generate(leg_view, EMPTY_REGISTRY, budget=300.0, cache=cache)

    assert not any(s.kind is StrategyKind.USE_CACHED for s in strategies)
    assert any(s.kind is StrategyKind.COMMIT and s.mode == "flight" for s in strategies)


def test_use_cached_still_offered_when_the_mode_has_no_live_price():
    """The converse: a timed-out mode has no live price, so the cached
    one is exactly what UseCached is for."""
    timed_out = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)
    leg_view = LegView(
        origin="London",
        destination="Berlin",
        results=(("flight", Unknown(observations=(timed_out,))),),
        distance_km=930.0,
    )
    cache = FakeCache({("London", "Berlin", "flight"): (Money(7140, "GBP"), 26.0)})

    strategies = generate(leg_view, EMPTY_REGISTRY, budget=300.0, cache=cache)

    assert any(s.kind is StrategyKind.USE_CACHED for s in strategies)
