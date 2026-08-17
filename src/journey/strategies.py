"""Strategy generation: turn a merged LegView into candidate Strategy
objects. No scoring or ranking here - that's Phase 6. Every filter below
is a defensible yes/no decision about whether an option belongs on the
table at all (SPEC.md §8).
"""

from typing import Protocol

from journey.domain import (
    Available,
    LegView,
    Money,
    SourceStatus,
    Strategy,
    StrategyKind,
    Unknown,
)
from journey.fetch import PendingRegistry
from journey.pricing import MAX_CACHE_AGE_HOURS, drift_cached, infer_cost

# Two candidate wait durations offered per pending source, short and
# long. Neither is offered if it exceeds the remaining budget. Kept on a
# human/demo scale - the brief's own example is "wait 5 seconds for
# flight data," and a real Wait(3) landing at ~4s reads as an actual
# decision with a real payoff, not a 30s+ dead pause on camera.
WAIT_DURATIONS_SECONDS = (3.0, 8.0)

_FAILED_STATUSES = (SourceStatus.TIMED_OUT, SourceStatus.ERROR)


class CacheLookup(Protocol):
    def get(self, origin: str, destination: str, mode: str) -> tuple[Money, float] | None:
        """(cached_price, age_hours) for this mode on this leg, or None.

        Keyed by leg as well as mode: a cached London->Berlin flight
        price says nothing about flights on any other leg, so a
        mode-only key would leak one leg's fare onto every other.
        """
        ...


def generate(leg_view: LegView, registry: PendingRegistry, budget: float, cache: CacheLookup) -> list[Strategy]:
    strategies: list[Strategy] = []

    for mode, status in leg_view.results:
        strategies.extend(_commit_strategies(mode, status, leg_view.distance_km))

        if isinstance(status, Unknown):
            strategies.extend(_wait_strategies(mode, status, registry, budget))
            strategies.extend(_replan_strategies(mode, status, leg_view))

        strategies.extend(_use_cached_strategy(mode, cache, leg_view, status))

    if leg_view.abandonable:
        strategies.append(
            Strategy(
                kind=StrategyKind.ABANDON_LEG,
                mode="*",  # not about any one mode - the leg as a whole
                reason="journey remains viable without this leg",
            )
        )

    return strategies


def _commit_strategies(mode: str, status, distance_km: float | None) -> list[Strategy]:
    """Commit for any actionable observation, wherever it lives -
    Available or Conflicted both carry actionable data."""
    observations = getattr(status, "observations", ())
    strategies = []
    for obs in observations:
        if not (obs.actionable_for_cost() or obs.actionable_for_time()):
            continue
        if obs.price is not None:
            cost_low, cost_high, cost_basis = obs.price, obs.price, "observed"
        elif distance_km is not None:
            cost_low, cost_high = infer_cost(mode, distance_km)
            cost_basis = "inferred"
        else:
            cost_low, cost_high, cost_basis = None, None, None
        strategies.append(
            Strategy(
                kind=StrategyKind.COMMIT,
                mode=mode,
                source=obs.source,
                cost_low=cost_low,
                cost_high=cost_high,
                cost_basis=cost_basis,
                reason=f"commit to {obs.source} for {mode}",
            )
        )
    return strategies


def _wait_strategies(mode: str, status: Unknown, registry: PendingRegistry, budget: float) -> list[Strategy]:
    strategies = []
    for obs in status.observations:
        # ERROR means the source answered - nothing is in flight, so no
        # wait is offered even though the mode overall is Unknown.
        if obs.status is not SourceStatus.TIMED_OUT or not registry.is_pending(obs.source):
            continue
        for wait_seconds in WAIT_DURATIONS_SECONDS:
            if wait_seconds > budget:
                continue
            strategies.append(
                Strategy(
                    kind=StrategyKind.WAIT,
                    mode=mode,
                    source=obs.source,
                    wait_seconds=wait_seconds,
                    reason=f"{obs.source} is still pending, worth waiting {wait_seconds:.0f}s more",
                )
            )
    return strategies


def _replan_strategies(mode: str, status: Unknown, leg_view: LegView) -> list[Strategy]:
    failed_sources = {o.source for o in status.observations if o.status in _FAILED_STATUSES}
    if not failed_sources:
        return []

    strategies = []
    for alt_mode, alt_status in leg_view.results:
        if alt_mode == mode or not isinstance(alt_status, Available):
            continue
        alt_sources = {
            o.source for o in alt_status.observations if o.status in (SourceStatus.FRESH, SourceStatus.STALE)
        }
        if alt_sources & failed_sources:
            continue  # depends on the same failed source - not a real replan
        strategies.append(
            Strategy(
                kind=StrategyKind.REPLAN,
                mode=alt_mode,
                reason=f"{mode} failed via {', '.join(sorted(failed_sources))}; "
                f"{alt_mode} is healthy and independent of it",
            )
        )
    return strategies


def _use_cached_strategy(mode: str, cache: CacheLookup, leg_view: LegView, status) -> list[Strategy]:
    """A cached fare is a fallback for not having a live one, so it is
    not offered once this mode has an actionable live price.

    Beyond being redundant, offering both is actively wrong: a UseCached
    strategy carries no source of its own, and scoring._expected_minutes
    matches any observation when source is None - so alongside a live
    observation the cached strategy picks up that observation's duration
    and competes as yesterday's price paired with today's journey time.
    Observed doing exactly that, and winning.
    """
    if any(o.actionable_for_cost() for o in getattr(status, "observations", ())):
        return []

    entry = cache.get(leg_view.origin, leg_view.destination, mode)
    if entry is None:
        return []
    price, age_hours = entry
    if age_hours >= MAX_CACHE_AGE_HOURS:
        return []
    low, high = drift_cached(price, age_hours)
    return [
        Strategy(
            kind=StrategyKind.USE_CACHED,
            mode=mode,
            cost_low=low,
            cost_high=high,
            cost_basis="stale",
            cache_age_hours=age_hours,
            reason=f"cached {mode} price is {age_hours:.0f}h old",
        )
    ]
