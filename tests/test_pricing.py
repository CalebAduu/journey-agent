"""haversine_km, infer_cost, and drift_cached."""

import pytest

from journey.domain import Money
from journey.pricing import (
    MIN_FARE,
    STALENESS_DRIFT_PER_HOUR,
    drift_cached,
    haversine_km,
    infer_cost,
)


def test_haversine_one_degree_of_latitude_is_about_111_km():
    # A well-known geographic fact, independent of this implementation:
    # one degree of latitude is ~111.19km everywhere on the globe.
    distance = haversine_km(51.0, 0.0, 52.0, 0.0)
    assert distance == pytest.approx(111.19, rel=0.01)


def test_haversine_same_point_is_zero():
    assert haversine_km(51.5, -0.1, 51.5, -0.1) == pytest.approx(0.0, abs=1e-9)


def test_infer_cost_applies_rate_when_above_floor():
    low, high = infer_cost("rail", 1000)

    assert low == Money(round(0.12 * 1000 * 100))
    assert high == Money(round(0.35 * 1000 * 100))


def test_infer_cost_floors_low_bound_at_min_fare():
    low, _high = infer_cost("coach", 1)  # trivially short - rate alone prices under any real fare

    assert low == MIN_FARE["coach"]


def test_infer_cost_raises_high_bound_if_flooring_would_invert_the_interval():
    # Berlin -> Potsdam is ~25km; coach's own per-km high bound (0.09/km)
    # prices below MIN_FARE at this distance, so flooring low alone would
    # put low above high without also raising high to match.
    low, high = infer_cost("coach", 25)

    assert low == MIN_FARE["coach"]
    assert high.minor_units >= low.minor_units


def test_drift_cached_at_zero_age_has_no_spread():
    price = Money(10000, "GBP")

    low, high = drift_cached(price, age_hours=0)

    assert low == price
    assert high == price


def test_drift_cached_price_is_always_the_floor():
    price = Money(10000, "GBP")

    low, high = drift_cached(price, age_hours=24)

    assert low == price
    assert high.minor_units > low.minor_units


def test_drift_cached_ceiling_matches_the_stated_rate():
    price = Money(10000, "GBP")

    _low, high = drift_cached(price, age_hours=24)

    assert high == Money(round(10000 * (1 + STALENESS_DRIFT_PER_HOUR * 24)))
   
    assert high.minor_units == pytest.approx(11000, rel=0.01)
