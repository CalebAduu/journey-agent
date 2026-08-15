"""Phase 1 thesis test.

SPEC.md §9 Phase 1, done-when: "a test asserts that a LegView built from a
timeout is not equal to and not substitutable for one built from an
empty-but-successful response. Write this test first — it is the thesis
statement of the project."
"""

from datetime import UTC, datetime, timedelta

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
    empty_observation = Observation(source="stub-flight", mode="flight", status=SourceStatus.FRESH)

    leg_from_timeout = LegView(
        origin="London",
        destination="Brussels",
        results={"flight": Unknown(observations=(timeout_observation,))},
    )
    leg_from_empty = LegView(
        origin="London",
        destination="Brussels",
        results={"flight": Empty(observations=(empty_observation,))},
    )

    assert leg_from_timeout != leg_from_empty
    assert type(leg_from_timeout.results["flight"]) is not type(leg_from_empty.results["flight"])


def test_not_applicable_is_not_substitutable_for_unknown_or_empty():
    """§4: not calling because a mode is impossible (NOT_APPLICABLE) must
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
    """§4/Phase 4: two sources that disagree must keep both values as an
    interval, never collapse to a single winner."""
    conflicted = Conflicted(
        low=Money(10000, "GBP"),
        high=Money(18000, "GBP"),
        observations=(),
    )

    assert conflicted.low == Money(10000, "GBP")
    assert conflicted.high == Money(18000, "GBP")
    assert conflicted != Unknown(observations=())
    assert conflicted != Empty(observations=())
    assert conflicted != NotApplicable(reason="no direct passenger service")


def test_observation_carries_price_duration_and_provenance():
    """Phase 2a: a stub source's successful answer needs to report a price,
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
