"""Phase 5: strategy generation filters.

SPEC.md §8: Wait only if genuinely pending; UseCached only if fresh
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
    def __init__(self, entries: dict[str, tuple[Money, float]] | None = None):
        self.entries = entries or {}

    def get(self, mode: str):
        return self.entries.get(mode)


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
    stale_cache = FakeCache({"flight": (Money(9000, "GBP"), 100.0)})  # > MAX_CACHE_AGE_HOURS (72)
    fresh_cache = FakeCache({"flight": (Money(9000, "GBP"), 10.0)})

    stale_strategies = generate(leg_view, EMPTY_REGISTRY, budget=300.0, cache=stale_cache)
    fresh_strategies = generate(leg_view, EMPTY_REGISTRY, budget=300.0, cache=fresh_cache)

    assert not any(s.kind is StrategyKind.USE_CACHED for s in stale_strategies)
    assert any(s.kind is StrategyKind.USE_CACHED for s in fresh_strategies)
