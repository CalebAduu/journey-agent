""" parse_transitous() tested against a real saved response.

fixtures/transitous_journeys.json is a genuine api.transitous.org/api/v1/plan
response for Sheffield -> London (route leg 1), captured live
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from journey.domain import Leg, SourceStatus
from journey.sources.transitous import parse_transitous

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "transitous_journeys.json"
LEG = Leg(origin="Sheffield", destination="London")
NOW = datetime(2026, 8, 15, 20, 0, tzinfo=UTC)


def _load_itineraries() -> list[dict]:
    with open(FIXTURE_PATH, encoding="utf-8-sig") as f:
        return json.load(f)["itineraries"]


def test_real_fixture_parses_into_fresh_observation_with_duration_no_price():
    itineraries = _load_itineraries()
    # itinerary 0: not cancelled, first leg realTime=True, carries a real
    # live alert ("Special Service") - exercises status + alerts in one go.
    payload = {"itineraries": [itineraries[0]]}

    observation = parse_transitous(payload, LEG, NOW)

    assert observation.status == SourceStatus.FRESH
    assert observation.price is None
    assert observation.duration == timedelta(seconds=24420)
    assert "Special Service" in observation.detail


def test_empty_itineraries_returns_fresh_empty():
    observation = parse_transitous({"itineraries": []}, LEG, NOW)

    assert observation.status == SourceStatus.FRESH_EMPTY
    assert observation.price is None
    assert observation.duration is None


def test_all_cancelled_itineraries_returns_fresh_empty():
    itineraries = _load_itineraries()
    # itineraries 1 and 3: each has at least one leg with cancelled=True,
    # confirmed in the real fixture - both get filtered out, leaving
    # nothing usable, same as an empty response.
    payload = {"itineraries": [itineraries[1], itineraries[3]]}

    observation = parse_transitous(payload, LEG, NOW)

    assert observation.status == SourceStatus.FRESH_EMPTY
    assert observation.price is None
    assert observation.duration is None
