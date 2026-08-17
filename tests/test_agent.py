"""Phase 7: the agent loop.

SPEC.md §9 Phase 7 done-when: full 3-leg run with a failure completes
and prints a coherent trace. §8's global-budget demo moment: the same
failure ranks Wait first at leg 2 and Commit first at leg 3, because
waiting gets more expensive as the remaining budget shrinks.
"""

import asyncio
import random
from datetime import UTC, datetime, timedelta

from journey.agent import PlanRequest, TimeBudget, act, plan
from journey.domain import Leg, Money, Observation, SourceStatus, Strategy, StrategyKind
from journey.fetch import PendingRegistry
from journey.sources.chaos import ChaosScenario, Slow
from journey.sources.stubs import StubSource


class FakeClock:
    def __init__(self, fixed_time: datetime):
        self.fixed_time = fixed_time

    def now(self) -> datetime:
        return self.fixed_time

    async def sleep(self, seconds: float) -> None:
        pass


class RealSleepClock:
    """FakeClock's sleep is a no-op, which makes a Slow directive return
    instantly - useless for testing a source that must still be pending
    at the harvest deadline. This one actually sleeps, in tenths."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class AllFeasible:
    def __init__(self, modes: list[str]):
        self.modes = modes

    def candidate_modes(self, leg):
        return self.modes

    def reason_if_not_applicable(self, leg, mode):
        return None


class EmptyCache:
    def get(self, origin, destination, mode):
        return None


class InstantSource:
    """Resolves immediately with a fixed price/duration."""

    def __init__(self, name: str, mode: str, price_minor: int, duration_minutes: float):
        self.name = name
        self.mode = mode
        self.price_minor = price_minor
        self.duration_minutes = duration_minutes

    async def fetch(self, leg):
        return Observation(
            source=self.name,
            mode=self.mode,
            status=SourceStatus.FRESH,
            price=Money(self.price_minor, "GBP"),
            duration=timedelta(minutes=self.duration_minutes),
        )


class EventualSource:
    """Takes a short real delay then resolves - long enough to miss a
    tight harvest deadline (shows up pending) but short enough that
    registry.wait_for() recovers it well within its own timeout."""

    def __init__(self, name: str, mode: str, delay_seconds: float, price_minor: int, duration_minutes: float):
        self.name = name
        self.mode = mode
        self.delay_seconds = delay_seconds
        self.price_minor = price_minor
        self.duration_minutes = duration_minutes

    async def fetch(self, leg):
        await asyncio.sleep(self.delay_seconds)
        return Observation(
            source=self.name,
            mode=self.mode,
            status=SourceStatus.FRESH,
            price=Money(self.price_minor, "GBP"),
            duration=timedelta(minutes=self.duration_minutes),
        )


class SlowSource:
    """Never resolves within any reasonable test timeframe."""

    def __init__(self, name: str, mode: str):
        self.name = name
        self.mode = mode

    async def fetch(self, leg):
        await asyncio.sleep(3600)
        raise AssertionError("should never actually complete in tests")


CLOCK = FakeClock(datetime(2026, 8, 16, 10, 0, tzinfo=UTC))


def test_all_healthy_single_leg_commits_without_spending_budget():
    async def run():
        leg = Leg(origin="Sheffield", destination="London", distance_km=260.0)
        sources = [InstantSource("stub-coach", "coach", 2000, 240)]
        request = PlanRequest(
            legs=(leg,),
            preference="cheapest",
            budget_seconds=300.0,
            feasibility=AllFeasible(["coach"]),
            cache=EmptyCache(),
        )
        return await plan(request, sources, PendingRegistry(), CLOCK)

    journey_plan = asyncio.run(run())

    assert len(journey_plan.trace) == 1
    decision = journey_plan.trace[0]
    assert decision.choice.kind is StrategyKind.COMMIT
    assert decision.budget_before == decision.budget_after == 300.0


def test_wait_that_resolves_commits_with_the_fresh_observation():
    async def run():
        leg = Leg(origin="Sheffield", destination="London", distance_km=260.0)
        sources = [EventualSource("stub-flight", "flight", delay_seconds=0.08, price_minor=9000, duration_minutes=90)]
        request = PlanRequest(
            legs=(leg,),
            preference="cheapest",
            budget_seconds=300.0,
            feasibility=AllFeasible(["flight"]),
            cache=EmptyCache(),
            harvest_timeout_seconds=0.02,  # shorter than the source's own delay - shows up pending
        )
        return await plan(request, sources, PendingRegistry(), CLOCK)

    journey_plan = asyncio.run(run())

    decision = journey_plan.trace[0]
    assert decision.ranked_strategies[0].kind is StrategyKind.WAIT
    assert decision.choice.kind is StrategyKind.COMMIT
    assert decision.choice.source == "stub-flight"


def test_wait_that_never_resolves_falls_back_to_best_available():
    """Tests act() directly with a hand-built ranking and a task that
    genuinely never completes, so the wait_seconds used can stay tiny -
    going through the full generate()/score pipeline here would force a
    multi-second real wait via strategies.WAIT_DURATIONS_SECONDS."""

    async def run():
        leg = Leg(origin="Sheffield", destination="London", distance_km=260.0)
        registry = PendingRegistry()
        never_task = asyncio.create_task(asyncio.sleep(3600))
        registry.register("stub-flight", never_task)

        coach_commit = Strategy(
            kind=StrategyKind.COMMIT,
            mode="coach",
            source="stub-coach",
            reason="commit to coach",
            cost_low=Money(6000, "GBP"),
            cost_high=Money(6000, "GBP"),
            cost_basis="observed",
        )
        flight_wait = Strategy(
            kind=StrategyKind.WAIT, mode="flight", source="stub-flight", wait_seconds=0.05, reason="wait for flight"
        )
        ranked = [flight_wait, coach_commit]  # Wait ranked first, by construction

        budget = TimeBudget(300.0)
        request = PlanRequest(
            legs=(leg,),
            preference="cheapest",
            budget_seconds=300.0,
            feasibility=AllFeasible(["coach", "flight"]),
            cache=EmptyCache(),
        )
        observations = [Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)]

        choice, resolved = await act(observations, ranked, leg, registry, budget, request)
        never_task.cancel()
        assert resolved is None  # the wait never resolved, so nothing landed
        return choice, budget

    choice, budget = asyncio.run(run())

    assert choice.kind is StrategyKind.COMMIT
    assert choice.source == "stub-coach"
    assert budget.remaining == 300.0 - 0.05


def test_drain_cleans_up_a_pending_task_never_explicitly_awaited():
    async def run():
        leg = Leg(origin="Sheffield", destination="London", distance_km=260.0)
        sources = [
            InstantSource("stub-coach", "coach", 500, 240),  # cheap enough that flight can never beat it
            SlowSource("stub-flight", "flight"),
        ]
        request = PlanRequest(
            legs=(leg,),
            preference="cheapest",
            budget_seconds=300.0,
            feasibility=AllFeasible(["coach", "flight"]),
            cache=EmptyCache(),
            harvest_timeout_seconds=0.02,
        )
        registry = PendingRegistry()
        journey_plan = await plan(request, sources, registry, CLOCK)
        return journey_plan, registry

    journey_plan, registry = asyncio.run(run())

    assert journey_plan.trace[0].choice.kind is StrategyKind.COMMIT
    assert journey_plan.trace[0].choice.source == "stub-coach"
    assert not registry.is_pending("stub-flight")


def test_same_failure_ranks_wait_at_leg_two_and_commit_at_leg_three():
    async def run():
        # Same distance on both legs deliberately - this test is about the
        # budget mechanism, not geography. At 260km, flight's cheapest
        # plausible price (2500) beats coach's fixed 6000, so the failure
        # is genuinely worth waiting for while budget allows it.
        leg_two = Leg(origin="London", destination="Berlin", distance_km=260.0)
        leg_three = Leg(origin="Berlin", destination="Potsdam", distance_km=260.0)
        sources = [
            InstantSource("stub-coach", "coach", 6000, 240),
            SlowSource("stub-flight", "flight"),
        ]
        request = PlanRequest(
            legs=(leg_two, leg_three),
            preference="cheapest",
            budget_seconds=9.0,  # small enough that a single Wait meaningfully shrinks it
            feasibility=AllFeasible(["coach", "flight"]),
            cache=EmptyCache(),
            harvest_timeout_seconds=0.02,
        )
        return await plan(request, sources, PendingRegistry(), CLOCK)

    journey_plan = asyncio.run(run())

    assert journey_plan.trace[0].ranked_strategies[0].kind is StrategyKind.WAIT
    assert journey_plan.trace[1].ranked_strategies[0].kind is StrategyKind.COMMIT
    assert journey_plan.trace[1].budget_before < journey_plan.trace[0].budget_before


def test_trace_records_the_observation_that_landed_during_a_wait():
    """DecisionTrace.observations/leg_view are frozen before act() runs,
    so when a wait actually resolves, nothing in the trace said what
    arrived - only that the choice had changed. Both the CLI's post-wait
    line and the narrator need the arrival itself, not just its
    consequence, or they describe a commit to a mode the trace still
    shows as pending.
    """

    async def run():
        leg = Leg(origin="London", destination="Berlin", distance_km=930.0)
        scenario = ChaosScenario(
            name="late-flight",
            directives={("stub-flight", "London", "Berlin"): Slow(0.15, 2.4194)},
        )
        clock = RealSleepClock()
        sources = [
            StubSource(
                name="stub-coach", mode="coach", base_price=Money(3886, "GBP"),
                base_duration=timedelta(hours=10), scenario=scenario,
                rng=random.Random(1), clock=clock,
            ),
            StubSource(
                name="stub-flight", mode="flight", base_price=Money(7440, "GBP"),
                base_duration=timedelta(hours=2), scenario=scenario,
                rng=random.Random(2), clock=clock,
            ),
        ]
        registry = PendingRegistry()
        request = PlanRequest(
            legs=(leg,),
            preference="fastest",
            budget_seconds=300.0,
            feasibility=AllFeasible(["coach", "flight"]),
            cache=EmptyCache(),
            harvest_timeout_seconds=0.05,  # flight still pending at the deadline
        )
        return await plan(request, sources, registry, clock)

    journey_plan = asyncio.run(run())
    decision = journey_plan.trace[0]

    # the wait was taken and the flight genuinely landed
    assert decision.resolved_observation is not None
    assert decision.resolved_observation.source == "stub-flight"
    assert decision.resolved_observation.status is SourceStatus.FRESH
    assert decision.resolved_observation.price == Money(18000, "GBP")
    # and the final choice is built on it, not on the stale pending view
    assert decision.choice.kind is StrategyKind.COMMIT
    assert decision.choice.mode == "flight"
