""" scoring and VOI.
"""

from datetime import timedelta

from journey.domain import (
    Available,
    Conflicted,
    LegView,
    Money,
    Observation,
    SourceStatus,
    Strategy,
    StrategyKind,
    Unknown,
)
from journey.scoring import score_strategies, voi

SHEFFIELD_LONDON_KM = 260.0


def test_wait_ranks_last_with_voi_zero_when_it_cannot_beat_current_best():
    # Coach: cheap and certain. Flight: pending, but even its cheapest
    # plausible price (MIN_FARE floor, since 260km is short for flight's
    # own per-km rate) is still worse than coach's actual price.
    coach_observation = Observation(
        source="stub-coach",
        mode="coach",
        status=SourceStatus.FRESH,
        price=Money(2000, "GBP"),
        duration=timedelta(minutes=240),
    )
    flight_timeout = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)

    leg_view = LegView(
        origin="Sheffield",
        destination="London",
        results=(
            ("coach", Available(observations=(coach_observation,))),
            ("flight", Unknown(observations=(flight_timeout,))),
        ),
        distance_km=SHEFFIELD_LONDON_KM,
    )

    coach_commit = Strategy(
        kind=StrategyKind.COMMIT,
        mode="coach",
        source="stub-coach",
        reason="commit to coach",
        cost_low=Money(2000, "GBP"),
        cost_high=Money(2000, "GBP"),
        cost_basis="observed",
    )
    flight_wait = Strategy(
        kind=StrategyKind.WAIT, mode="flight", source="stub-flight", wait_seconds=3.0, reason="wait for flight"
    )

    ranked = score_strategies([coach_commit, flight_wait], leg_view, "cheapest", budget=300.0)

    assert ranked[0].kind is StrategyKind.COMMIT
    assert ranked[-1].kind is StrategyKind.WAIT

    current_best, wait_scored = ranked[0], ranked[-1]
    assert voi(wait_scored, current_best, (0, 0), 300.0) == 0.0


def test_wait_ranks_above_commit_when_it_plausibly_can_beat_current_best():
    # Same shape, but now coach is the expensive option and flight's
    # cheapest plausible price genuinely undercuts it.
    coach_observation = Observation(
        source="stub-coach",
        mode="coach",
        status=SourceStatus.FRESH,
        price=Money(6000, "GBP"),
        duration=timedelta(minutes=240),
    )
    flight_timeout = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)

    leg_view = LegView(
        origin="Sheffield",
        destination="London",
        results=(
            ("coach", Available(observations=(coach_observation,))),
            ("flight", Unknown(observations=(flight_timeout,))),
        ),
        distance_km=SHEFFIELD_LONDON_KM,
    )

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
        kind=StrategyKind.WAIT, mode="flight", source="stub-flight", wait_seconds=3.0, reason="wait for flight"
    )

    ranked = score_strategies([coach_commit, flight_wait], leg_view, "cheapest", budget=300.0)

    assert ranked[0].kind is StrategyKind.WAIT
    assert ranked[1].kind is StrategyKind.COMMIT

    # recover the scored pair to check voi() directly too
    non_wait = next(s for s in ranked if s.kind is StrategyKind.COMMIT)
    wait_scored = next(s for s in ranked if s.kind is StrategyKind.WAIT)
    assert voi(wait_scored, non_wait, (0, 0), 300.0) > 0.0


def test_same_failure_produces_different_winners_under_cheapest_vs_reliable():
    # A cheap answer that's genuinely contested (Conflicted) vs an
    # expensive answer that's fully confirmed (Available). Same
    # candidates, same failure behind the conflict - only the preference
    # weights change.
    flight_low = Observation(source="stub-flight", mode="flight", status=SourceStatus.FRESH, price=Money(3000, "GBP"))
    flight_high = Observation(
        source="stub-flight-b", mode="flight", status=SourceStatus.FRESH, price=Money(5000, "GBP")
    )
    rail_observation = Observation(
        source="transitous", mode="rail", status=SourceStatus.FRESH, price=Money(8000, "GBP")
    )

    leg_view = LegView(
        origin="Sheffield",
        destination="London",
        results=(
            (
                "flight",
                Conflicted(
                    dimension="price",
                    observations=(flight_low, flight_high),
                    price_low=Money(3000, "GBP"),
                    price_high=Money(5000, "GBP"),
                ),
            ),
            ("rail", Available(observations=(rail_observation,))),
        ),
        distance_km=SHEFFIELD_LONDON_KM,
    )

    # Commit picks one of the conflicting observations - here, flight_low.
    cheap_but_contested = Strategy(
        kind=StrategyKind.COMMIT,
        mode="flight",
        source="stub-flight",
        reason="commit to flight",
        cost_low=Money(3000, "GBP"),
        cost_high=Money(3000, "GBP"),
        cost_basis="observed",
    )
    expensive_but_confirmed = Strategy(
        kind=StrategyKind.COMMIT,
        mode="rail",
        source="transitous",
        reason="commit to rail",
        cost_low=Money(8000, "GBP"),
        cost_high=Money(8000, "GBP"),
        cost_basis="observed",
    )
    candidates = [cheap_but_contested, expensive_but_confirmed]

    cheapest_ranked = score_strategies(candidates, leg_view, "cheapest")
    reliable_ranked = score_strategies(candidates, leg_view, "reliable")

    assert cheapest_ranked[0].mode == "flight"
    assert reliable_ranked[0].mode == "rail"


def test_cached_price_is_more_certain_than_a_dead_source_it_substitutes_for():
    """A cached price is independent evidence: it was a real quote once.

    _base_certainty used to take its base from the mode's LIVE status and
    only then apply the stale penalty - so on the very leg where UseCached
    matters (live source timed out, price on file from yesterday) the base
    was Unknown's 0.0, the penalty clamped at 0.0, and a 26h-old real
    quote scored exactly as certain as knowing nothing at all. It must
    land strictly between an observed price and a genuine unknown.
    """
    flight_timeout = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)
    coach_observation = Observation(
        source="stub-coach",
        mode="coach",
        status=SourceStatus.FRESH,
        price=Money(3886, "GBP"),
        duration=timedelta(minutes=600),
    )
    leg_view = LegView(
        origin="London",
        destination="Berlin",
        results=(
            ("coach", Available(observations=(coach_observation,))),
            ("flight", Unknown(observations=(flight_timeout,))),
        ),
        distance_km=930.0,
    )
    observed = Strategy(
        kind=StrategyKind.COMMIT,
        mode="coach",
        source="stub-coach",
        reason="commit to coach",
        cost_low=Money(3886, "GBP"),
        cost_high=Money(3886, "GBP"),
        cost_basis="observed",
    )
    cached = Strategy(
        kind=StrategyKind.USE_CACHED,
        mode="flight",
        reason="cached flight price is 26h old",
        cost_low=Money(7140, "GBP"),
        cost_high=Money(7883, "GBP"),
        cost_basis="stale",
        cache_age_hours=26.0,
    )
    genuine_unknown = Strategy(
        kind=StrategyKind.WAIT,
        mode="flight",
        source="stub-flight",
        reason="still pending",
        wait_seconds=3.0,
    )

    ranked = score_strategies([observed, cached, genuine_unknown], leg_view, "cheapest")
    by_kind = {s.kind: s for s in ranked}

    observed_certainty = by_kind[StrategyKind.COMMIT].certainty
    cached_certainty = by_kind[StrategyKind.USE_CACHED].certainty
    unknown_certainty = by_kind[StrategyKind.WAIT].certainty

    assert unknown_certainty < cached_certainty < observed_certainty
