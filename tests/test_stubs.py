"""Phase 2a: StubSource behaviour under each chaos directive.

Critical constraints under test: fetch() must never raise regardless of
what goes wrong, and sources must use the injected clock/rng rather than
wall-clock time or module-level randomness.
"""

import asyncio
import random
from datetime import UTC, datetime, timedelta

from journey.domain import Leg, Money, SourceStatus
from journey.sources.base import Source
from journey.sources.chaos import (
    ChaosScenario,
    EmptyResponse,
    ErrorResponse,
    PriceShift,
    Slow,
    Timeout,
)
from journey.sources.stubs import StubSource

LEG = Leg(origin="London", destination="Brussels")
BASE_PRICE = Money(15000, "GBP")
BASE_DURATION = timedelta(hours=1, minutes=30)


class FakeClock:
    """Deterministic stand-in for Clock: fixed timestamp, no real waiting."""

    def __init__(self, fixed_time: datetime):
        self.fixed_time = fixed_time
        self.slept_for: list[float] = []

    def now(self) -> datetime:
        return self.fixed_time

    async def sleep(self, seconds: float) -> None:
        self.slept_for.append(seconds)


def make_source(scenario=None, seed=42, clock=None):
    scenario = scenario if scenario is not None else ChaosScenario(name="all-ok", directives={})
    clock = clock or FakeClock(datetime(2026, 8, 15, 12, 0, tzinfo=UTC))
    return StubSource(
        name="stub-flight",
        mode="flight",
        base_price=BASE_PRICE,
        base_duration=BASE_DURATION,
        scenario=scenario,
        rng=random.Random(seed),
        clock=clock,
    )


def directive_scenario(directive):
    return ChaosScenario(name="test", directives={("stub-flight", "London", "Brussels"): directive})


def test_stub_source_satisfies_source_protocol():
    assert isinstance(make_source(), Source)


def test_ok_scenario_returns_fresh_observation_with_price_and_duration():
    observation = asyncio.run(make_source().fetch(LEG))

    assert observation.status == SourceStatus.FRESH
    assert observation.price is not None
    assert observation.duration == BASE_DURATION


def test_timeout_scenario_returns_timed_out_without_raising():
    source = make_source(scenario=directive_scenario(Timeout(3.0)))

    observation = asyncio.run(source.fetch(LEG))

    assert observation.status == SourceStatus.TIMED_OUT
    assert observation.price is None
    assert "3.0" in observation.detail


def test_error_scenario_returns_error_without_raising():
    source = make_source(scenario=directive_scenario(ErrorResponse()))

    observation = asyncio.run(source.fetch(LEG))

    assert observation.status == SourceStatus.ERROR
    assert observation.price is None


def test_empty_scenario_returns_fresh_empty_without_price_or_duration():
    source = make_source(scenario=directive_scenario(EmptyResponse()))

    observation = asyncio.run(source.fetch(LEG))

    assert observation.status == SourceStatus.FRESH_EMPTY
    assert observation.price is None
    assert observation.duration is None


def test_price_shift_scenario_changes_price_by_exact_factor():
    source = make_source(scenario=directive_scenario(PriceShift(1.3)))

    observation = asyncio.run(source.fetch(LEG))

    assert observation.price.minor_units == round(BASE_PRICE.minor_units * 1.3)


def test_slow_scenario_eventually_succeeds_and_uses_injected_clock_sleep():
    clock = FakeClock(datetime(2026, 8, 15, 12, 0, tzinfo=UTC))
    source = make_source(scenario=directive_scenario(Slow(5.0)), clock=clock)

    observation = asyncio.run(source.fetch(LEG))

    assert observation.status == SourceStatus.FRESH
    assert clock.slept_for == [5.0]


def test_fetch_never_raises_even_when_scenario_lookup_breaks():
    class BrokenScenario:
        def directive_for(self, source_name, leg):
            raise RuntimeError("boom")

    source = make_source(scenario=BrokenScenario())

    observation = asyncio.run(source.fetch(LEG))

    assert observation.status == SourceStatus.ERROR


def test_same_seed_reproduces_identical_price():
    observation_a = asyncio.run(make_source(seed=42).fetch(LEG))
    observation_b = asyncio.run(make_source(seed=42).fetch(LEG))

    assert observation_a.price == observation_b.price


def test_observed_at_comes_from_injected_clock_not_wall_clock():
    fixed_time = datetime(2020, 1, 1, tzinfo=UTC)
    source = make_source(clock=FakeClock(fixed_time))

    observation = asyncio.run(source.fetch(LEG))

    assert observation.observed_at == fixed_time
