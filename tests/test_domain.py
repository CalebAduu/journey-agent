"""Phase 1 thesis test.

The single failure mode this whole project exists to prevent: catching a
timeout, storing an empty result, and behaving as if the option doesn't
exist. If UNKNOWN and EMPTY can ever compare equal or be swapped for one
another, that collapse becomes possible again.
"""

from journey.domain import (
    Conflicted,
    Empty,
    LegStatus,
    LegView,
    Money,
    NotApplicable,
    Observation,
    SourceStatus,
    Unknown,
)


def test_timeout_is_not_equal_to_empty_response():
    """A source that timed out and a source that answered 'nothing available'
    must never compare equal, even when both carry no price."""
    timed_out = Unknown(
        observations=(
            Observation(source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT),
        )
    )
    empty = Empty(
        observations=(
            Observation(source="stub-flight", mode="flight", status=SourceStatus.FRESH),
        )
    )

    assert timed_out != empty
    assert type(timed_out) is not type(empty)


def test_timeout_and_empty_are_not_substitutable_in_a_leg_view():
    """The same leg, differing only in whether flight timed out or came back
    empty, must produce LegViews that are not equal — a caller cannot treat
    one as the other by accident."""
    leg_with_timeout = LegView(
        origin="London",
        destination="Brussels",
        results={"flight": Unknown(observations=())},
    )
    leg_with_empty = LegView(
        origin="London",
        destination="Brussels",
        results={"flight": Empty(observations=())},
    )

    assert leg_with_timeout != leg_with_empty


def test_not_applicable_and_unknown_are_distinct_even_with_no_observations():
    """NOT_APPLICABLE (we chose not to call) and UNKNOWN (we called and got
    nothing back) must stay distinct even though neither carries a price or
    a successful observation."""
    not_applicable = NotApplicable(reason="no direct passenger service", basis="physical")
    unknown = Unknown(observations=())

    assert not_applicable != unknown
    assert isinstance(not_applicable, LegStatus)
    assert isinstance(unknown, LegStatus)
    assert not isinstance(not_applicable, type(unknown))


def test_conflicted_carries_both_values_not_one():
    conflicted = Conflicted(
        low=Money(10000, "GBP"),
        high=Money(18000, "GBP"),
        observations=(),
    )
    assert conflicted.low == Money(10000, "GBP")
    assert conflicted.high == Money(18000, "GBP")


def test_observation_price_must_carry_an_explaining_status():
    """The hard rule: a price can only exist alongside a status that
    explains it (FRESH or STALE). A TIMED_OUT observation with a price
    makes no sense and must be rejected at construction."""
    import pytest

    with pytest.raises(ValueError):
        Observation(
            source="stub-flight",
            mode="flight",
            status=SourceStatus.TIMED_OUT,
            price=Money(5000, "GBP"),
        )


def test_is_actionable_reflects_status_not_just_price_presence():
    fresh = Observation(source="db", mode="rail", status=SourceStatus.FRESH, price=Money(7800, "GBP"))
    stale = Observation(source="db", mode="rail", status=SourceStatus.STALE, price=Money(7800, "GBP"))
    errored = Observation(source="db", mode="rail", status=SourceStatus.ERROR)

    assert fresh.is_actionable()
    assert stale.is_actionable()
    assert not errored.is_actionable()


def test_domain_types_are_frozen():
    import dataclasses

    import pytest

    money = Money(100, "GBP")
    with pytest.raises(dataclasses.FrozenInstanceError):
        money.minor_units = 200  # type: ignore[misc]
