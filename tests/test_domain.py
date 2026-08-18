"""Phase 1 thesis test.
"""


from datetime import UTC, datetime, timedelta

import pytest

from journey.domain import (
    Conflicted,
    Empty,
    LegView,
    Money,
    NotApplicable,
    Observation,
    SourceStatus,
    Unknown,
)


def test_leg_view_from_timeout_is_not_substitutable_for_leg_view_from_empty():
    timeout_observation = Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT)
    empty_observation = Observation(source="stub-flight", mode="flight", status=SourceStatus.FRESH_EMPTY)

    leg_from_timeout = LegView(
        origin="London",
        destination="Brussels",
        results=(("flight", Unknown(observations=(timeout_observation,))),),
    )
    leg_from_empty = LegView(
        origin="London",
        destination="Brussels",
        results=(("flight", Empty(observations=(empty_observation,))),),
    )

    assert leg_from_timeout != leg_from_empty
    assert type(leg_from_timeout.by_mode("flight")) is not type(leg_from_empty.by_mode("flight"))


def test_not_applicable_is_not_substitutable_for_unknown_or_empty():
    """ not calling because a mode is impossible (NOT_APPLICABLE) must
    stay distinct from calling and getting nothing back (UNKNOWN/EMPTY),
    even though none of the three carry a price."""
    not_applicable = NotApplicable(reason="no direct passenger service")
    unknown = Unknown(observations=())
    empty = Empty(observations=())

    assert not_applicable != unknown
    assert not_applicable != empty
    assert type(not_applicable) is not type(unknown)
    assert type(not_applicable) is not type(empty)


def test_conflicted_carries_both_values_not_one():
    """ two sources that disagree must keep both values as an
    interval, never collapse to a single winner."""
    conflicted = Conflicted(
        dimension="price",
        observations=(),
        price_low=Money(10000, "GBP"),
        price_high=Money(18000, "GBP"),
    )

    assert conflicted.price_low == Money(10000, "GBP")
    assert conflicted.price_high == Money(18000, "GBP")
    assert conflicted != Unknown(observations=())
    assert conflicted != Empty(observations=())
    assert conflicted != NotApplicable(reason="no direct passenger service")


def test_observation_carries_price_duration_and_provenance():
    """ a stub source's successful answer needs to report a price,
    a duration, when it was observed, and (on failure) why."""
    observed_at = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)

    observation = Observation(
        source="stub-flight",
        mode="flight",
        status=SourceStatus.FRESH,
        price=Money(15000, "GBP"),
        duration=timedelta(hours=1, minutes=30),
        observed_at=observed_at,
        detail="",
    )

    assert observation.price == Money(15000, "GBP")
    assert observation.duration == timedelta(hours=1, minutes=30)
    assert observation.observed_at == observed_at



def test_timeout_cannot_carry_a_price():
    with pytest.raises(ValueError):
        Observation("flights", "flight", SourceStatus.TIMED_OUT, price=Money(9500))


def test_fresh_with_duration_only_is_valid():
    obs = Observation("transitous", "rail", SourceStatus.FRESH,
                      duration=timedelta(minutes=127))
    assert obs.actionable_for_time()
    assert not obs.actionable_for_cost()
