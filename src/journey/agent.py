"""Agent loop: harvest -> merge -> generate -> score -> act, once per
leg, carrying a shrinking TimeBudget forward. Only Wait spends budget -
Commit/UseCached/Replan/AbandonLeg are all presumed instantaneous.
registry.drain() always runs, even if a leg raises: non-negotiable #4
keeps pending tasks alive mid-journey, but nothing should outlive the
journey itself.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from journey.domain import (
    DecisionTrace,
    JourneyPlan,
    Leg,
    Observation,
    Strategy,
    StrategyKind,
    Unknown,
)
from journey.fetch import PendingRegistry, harvest
from journey.merge import Feasibility, merge
from journey.scoring import score_strategies
from journey.sources.base import Clock, Source
from journey.strategies import CacheLookup, generate

# How long harvest() waits per leg before treating a source as pending
# rather than failed. Kept on the same human/demo scale as
# strategies.WAIT_DURATIONS_SECONDS.
HARVEST_TIMEOUT_SECONDS = 3.0


class TimeBudget:
    """Total time available for the whole journey. Only Wait spends it.
    scoring.cost_of_waiting()'s downstream-slack term reads .remaining
    directly, so the same wait gets more expensive as this shrinks -
    that's what makes leg 3 rank Commit above Wait where leg 2 ranked
    Wait first for the identical failure."""

    def __init__(self, total_seconds: float):
        self.total_seconds = total_seconds
        self.remaining = total_seconds

    def spend(self, seconds: float) -> None:
        self.remaining = max(0.0, self.remaining - seconds)


@dataclass(frozen=True)
class PlanRequest:
    legs: tuple[Leg, ...]
    preference: str
    budget_seconds: float
    feasibility: Feasibility
    cache: CacheLookup
    source_reliability: dict[str, tuple[int, int]] | None = None
    harvest_timeout_seconds: float = HARVEST_TIMEOUT_SECONDS


async def plan(
    request: PlanRequest,
    sources: Sequence[Source],
    registry: PendingRegistry,
    clock: Clock,
    on_wait_tick: Callable[[str, float, float], None] | None = None,
) -> JourneyPlan:
    budget = TimeBudget(request.budget_seconds)
    trace: list[DecisionTrace] = []
    committed: list[Strategy] = []

    try:
        for leg in request.legs:
            decision = await _decide_leg(leg, sources, registry, clock, request, budget, on_wait_tick)
            trace.append(decision)
            committed.append(decision.choice)
    finally:
        await registry.drain()

    return JourneyPlan(legs=request.legs, trace=tuple(trace), committed=tuple(committed))


async def _decide_leg(
    leg: Leg,
    sources: Sequence[Source],
    registry: PendingRegistry,
    clock: Clock,
    request: PlanRequest,
    budget: TimeBudget,
    on_wait_tick: Callable[[str, float, float], None] | None = None,
) -> DecisionTrace:
    decided_at = clock.now()
    budget_before = budget.remaining

    observations = await harvest(sources, leg, request.harvest_timeout_seconds, registry)
    leg_view = merge(observations, leg, request.feasibility)
    strategies = generate(leg_view, registry, budget.remaining, request.cache)
    ranked = score_strategies(
        strategies,
        leg_view,
        request.preference,
        source_reliability=request.source_reliability,
        budget=budget.remaining,
    )

    choice, resolved = await act(observations, ranked, leg, registry, budget, request, on_wait_tick)

    return DecisionTrace(
        leg=leg,
        decided_at=decided_at,
        observations=tuple(observations),
        leg_view=leg_view,
        unknown_reasons=_unknown_reasons(leg_view),
        ranked_strategies=tuple(ranked),
        choice=choice,
        budget_before=budget_before,
        budget_after=budget.remaining,
        resolved_observation=resolved,
    )


async def act(
    observations: Sequence[Observation],
    ranked: Sequence[Strategy],
    leg: Leg,
    registry: PendingRegistry,
    budget: TimeBudget,
    request: PlanRequest,
    on_wait_tick: Callable[[str, float, float], None] | None = None,
) -> tuple[Strategy, Observation | None]:
    """Returns (chosen strategy, the observation that landed during a
    wait or None). The second element exists because the caller freezes
    observations/leg_view before this runs: when a wait resolves, the
    arrival is the evidence the final choice rests on, and without
    returning it the trace would record the consequence but not the cause.

    Commit/UseCached/Replan/AbandonLeg: no spend, the top choice
    stands as-is - Replan and AbandonLeg both just get recorded, since
    this project's Replan is a same-leg mode swap (not a multi-leg
    reroute) and needs no further machinery here (see README).

    Wait: spend wait_seconds regardless of how quickly the source
    actually resolves - the decision was to allocate that much budget,
    and using less would make VOI/cost_of_waiting's own assumptions
    inconsistent with what was actually scored. Then await the exact
    same pending task via registry.wait_for(). If it lands, fold the new
    observation in, re-merge and re-score, and commit whatever's now
    best (never offering to wait again immediately). If it times out
    again, commit the best available from the ORIGINAL ranking.
    """
    if not ranked:
        return _no_strategy_choice(leg), None

    top = ranked[0]
    if top.kind is not StrategyKind.WAIT:
        return top, None

    budget.spend(top.wait_seconds)
    tick = (lambda elapsed, total: on_wait_tick(top.source, elapsed, total)) if on_wait_tick else None
    try:
        new_observation = await registry.wait_for(top.source, top.wait_seconds, on_tick=tick)
    except TimeoutError:
        return next((s for s in ranked if s.kind is not StrategyKind.WAIT), top), None

    updated_observations = tuple(
        new_observation if o.source == top.source else o for o in observations
    )
    new_leg_view = merge(updated_observations, leg, request.feasibility)
    new_strategies = generate(new_leg_view, registry, budget.remaining, request.cache)
    new_ranked = score_strategies(
        new_strategies,
        new_leg_view,
        request.preference,
        source_reliability=request.source_reliability,
        budget=budget.remaining,
    )
    best = next((s for s in new_ranked if s.kind is not StrategyKind.WAIT), None)
    if best is not None:
        return best, new_observation
    return (new_ranked[0] if new_ranked else _no_strategy_choice(leg)), new_observation


def _unknown_reasons(leg_view) -> tuple[tuple[str, str], ...]:
    reasons = []
    for mode, status in leg_view.results:
        if isinstance(status, Unknown):
            detail_texts = [o.detail for o in status.observations if o.detail]
            reasons.append((mode, "; ".join(detail_texts) if detail_texts else "no observations"))
    return tuple(reasons)


def _no_strategy_choice(leg: Leg) -> Strategy:
    return Strategy(
        kind=StrategyKind.ABANDON_LEG,
        mode="*",
        reason=f"no viable strategy for {leg.origin} -> {leg.destination}",
    )
