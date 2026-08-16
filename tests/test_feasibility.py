"""RouteFeasibility: the physical and geometric layers of §5."""

from journey.domain import Leg
from journey.feasibility import GEOMETRIC_COACH_MAX_KM, GEOMETRIC_FLIGHT_MIN_KM, RouteFeasibility

feasibility = RouteFeasibility()


def test_physical_exclusion_blocks_flight_sheffield_london():
    leg = Leg(origin="Sheffield", destination="London", distance_km=260.0)

    reason = feasibility.reason_if_not_applicable(leg, "flight")

    assert reason is not None
    assert "Doncaster Sheffield Airport closed" in reason
    assert "physical" in reason


def test_geometric_exclusion_blocks_flight_under_threshold():
    leg = Leg(origin="Berlin", destination="Potsdam", distance_km=25.0)
    assert GEOMETRIC_FLIGHT_MIN_KM > leg.distance_km  # sanity check on the fixture itself

    reason = feasibility.reason_if_not_applicable(leg, "flight")

    assert reason is not None
    assert "geometric" in reason


def test_geometric_exclusion_blocks_coach_over_threshold():
    leg = Leg(origin="London", destination="Sydney", distance_km=17000.0)
    assert GEOMETRIC_COACH_MAX_KM < leg.distance_km

    reason = feasibility.reason_if_not_applicable(leg, "coach")

    assert reason is not None
    assert "geometric" in reason


def test_all_three_modes_feasible_on_an_ordinary_leg():
    leg = Leg(origin="London", destination="Berlin", distance_km=930.0)

    assert feasibility.reason_if_not_applicable(leg, "coach") is None
    assert feasibility.reason_if_not_applicable(leg, "rail") is None
    assert feasibility.reason_if_not_applicable(leg, "flight") is None
    assert set(feasibility.candidate_modes(leg)) == {"coach", "rail", "flight"}
