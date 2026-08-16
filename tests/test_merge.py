"""Phase 4: merge() and certainty().

SPEC.md §9 Phase 4 done-when: £100 and £180 for the same leg yields
CONFLICTED carrying both, not one; NOT_APPLICABLE legs do not reduce the
certainty score. Critical: only Unknown reduces certainty - Empty and
NotApplicable are both full certainty.
"""

from datetime import timedelta

from journey.domain import (
    Available,
    Conflicted,
    Empty,
    Leg,
    Money,
    NotApplicable,
    Observation,
    SourceStatus,
    Unknown,
)
from journey.merge import DISRUPTION_CERTAINTY_PENALTY, certainty, merge

LEG = Leg(origin="Sheffield", destination="London")


class FakeFeasibility:
    """A minimal Feasibility: every mode is a candidate, and exactly the
    modes named in not_applicable get NotApplicable with the given reason."""

    def __init__(self, modes: list[str], not_applicable: dict[str, str] | None = None):
        self.modes = modes
        self.not_applicable = not_applicable or {}

    def candidate_modes(self, leg):
        return self.modes

    def reason_if_not_applicable(self, leg, mode):
        return self.not_applicable.get(mode)


def test_certainty_only_unknown_reduces_it():
    assert certainty(NotApplicable(reason="Doncaster Sheffield Airport closed")) == 1.0
    assert certainty(Empty(observations=())) == 1.0
    assert certainty(Unknown(observations=())) == 0.0


def test_price_conflict_keeps_both_values_as_an_interval():
    low = Observation(source="stub-flight", mode="flight", status=SourceStatus.FRESH, price=Money(10000, "GBP"))
    high = Observation(source="stub-flight-b", mode="flight", status=SourceStatus.FRESH, price=Money(18000, "GBP"))

    feasibility = FakeFeasibility(modes=["flight"])
    leg_view = merge([low, high], LEG, feasibility)

    outcome = leg_view.by_mode("flight")
    assert isinstance(outcome, Conflicted)
    assert outcome.dimension == "price"
    assert outcome.price_low == Money(10000, "GBP")
    assert outcome.price_high == Money(18000, "GBP")
    assert outcome.observations == (low, high)  # both retained, never one picked


def test_duration_conflict_is_the_only_kind_real_sources_can_produce():
    # Transitous-shaped: duration only, no price - exactly what a real
    # rail-vs-rail disagreement looks like on this route.
    fast = Observation(source="transitous", mode="rail", status=SourceStatus.FRESH, duration=timedelta(minutes=60))
    slow = Observation(source="vbb", mode="rail", status=SourceStatus.FRESH, duration=timedelta(minutes=90))

    feasibility = FakeFeasibility(modes=["rail"])
    leg_view = merge([fast, slow], LEG, feasibility)

    outcome = leg_view.by_mode("rail")
    assert isinstance(outcome, Conflicted)
    assert outcome.dimension == "duration"
    assert outcome.duration_low == timedelta(minutes=60)
    assert outcome.duration_high == timedelta(minutes=90)
    assert outcome.price_low is None
    assert outcome.price_high is None


def test_available_with_a_disruption_alert_gets_reduced_certainty():
    clean = Observation(source="transitous", mode="rail", status=SourceStatus.FRESH, duration=timedelta(minutes=60))
    disrupted = Observation(
        source="transitous",
        mode="rail",
        status=SourceStatus.FRESH,
        duration=timedelta(minutes=62),  # well within threshold of `clean` - not a conflict
        detail="Special Service",  # a live alert's headerText, per parse_transitous
    )

    feasibility = FakeFeasibility(modes=["rail"])
    clean_view = merge([clean], LEG, feasibility)
    disrupted_view = merge([disrupted], LEG, feasibility)

    clean_outcome = clean_view.by_mode("rail")
    disrupted_outcome = disrupted_view.by_mode("rail")
    assert isinstance(clean_outcome, Available)
    assert isinstance(disrupted_outcome, Available)

    assert certainty(clean_outcome) == 1.0
    assert certainty(disrupted_outcome) == 1.0 - DISRUPTION_CERTAINTY_PENALTY


def test_merge_maps_infeasible_mode_to_not_applicable_without_reducing_certainty():
    rail_observation = Observation(
        source="transitous", mode="rail", status=SourceStatus.FRESH, duration=timedelta(minutes=90)
    )
    feasibility = FakeFeasibility(
        modes=["rail", "flight"],
        not_applicable={"flight": "Doncaster Sheffield Airport closed"},
    )

    leg_view = merge([rail_observation], LEG, feasibility)

    flight_outcome = leg_view.by_mode("flight")
    assert isinstance(flight_outcome, NotApplicable)
    assert flight_outcome.reason == "Doncaster Sheffield Airport closed"
    assert certainty(flight_outcome) == 1.0

    rail_outcome = leg_view.by_mode("rail")
    assert isinstance(rail_outcome, Available)
