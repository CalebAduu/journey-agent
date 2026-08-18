""" ResponseCache record/replay, and the claim that stripping
never removes anything the parser actually reads.
"""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from journey.cache import CacheMiss, ResponseCache, _strip
from journey.domain import Leg, SourceStatus
from journey.sources.transitous import TransitousClient, parse_transitous


class FakeClock:
    def __init__(self, fixed_time: datetime):
        self.fixed_time = fixed_time

    def now(self) -> datetime:
        return self.fixed_time

    async def sleep(self, seconds: float) -> None:
        pass


def _load_real_itineraries() -> list[dict]:
    fixture_path = "C:/Users/Caleb/journey-agent/fixtures/transitous_journeys.json"
    with open(fixture_path, encoding="utf-8-sig") as f:
        return json.load(f)["itineraries"]


def test_live_mode_writes_stripped_copy_but_returns_full_payload(tmp_path):
    payload = {
        "itineraries": [
            {
                "id": "CsQBCghOb3J0aGVybhI4MjAyNjA4MTVfMjI==",
                "duration": 100,
                "legs": [
                    {
                        "mode": "WALK",
                        "realTime": False,
                        "legGeometry": {"points": "abc123"},
                        "steps": [{"distance": 12}],
                        "intermediateStops": [{"name": "somewhere"}],
                    }
                ],
            }
        ],
        "debugOutput": {"fares": 0, "execute_time": 66},
    }

    def handler(request):
        return httpx.Response(200, json=payload)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            cache = ResponseCache(mode="live", path=tmp_path)
            return await cache.get(client, "https://api.transitous.org/api/v1/plan", {"b": 2, "a": 1})

    result = asyncio.run(run())

    assert result == payload

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert "debugOutput" not in saved
    assert "id" not in saved["itineraries"][0]
    saved_leg = saved["itineraries"][0]["legs"][0]
    assert "legGeometry" not in saved_leg
    assert "steps" not in saved_leg
    assert "intermediateStops" not in saved_leg


def test_replay_mode_raises_cache_miss_when_absent(tmp_path):
    cache = ResponseCache(mode="replay", path=tmp_path)

    with pytest.raises(CacheMiss):
        asyncio.run(cache.get(None, "https://api.transitous.org/api/v1/plan", {"a": 1}))


def test_transitous_client_replay_reproduces_the_live_cache_key(tmp_path):
    payload = {"itineraries": _load_real_itineraries()}

    def handler(request):
        return httpx.Response(200, json=payload)

    fixed_clock = FakeClock(datetime(2026, 8, 17, 9, 0, tzinfo=UTC))  # a Monday
    leg = Leg(
        origin="Sheffield",
        destination="London",
        origin_ids=(("transitous", "sheffield-id"),),
        destination_ids=(("transitous", "london-id"),),
    )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            live_cache = ResponseCache(mode="live", path=tmp_path)
            live_client = TransitousClient(cache=live_cache, clock=fixed_clock, http=client)
            live_observation = await live_client.fetch(leg)

        replay_cache = ResponseCache(mode="replay", path=tmp_path)
        replay_client = TransitousClient(cache=replay_cache, clock=fixed_clock)
        replay_observation = await replay_client.fetch(leg)
        return live_observation, replay_observation

    live_observation, replay_observation = asyncio.run(run())

    assert live_observation.status != SourceStatus.ERROR
    assert replay_observation == live_observation


def test_parser_produces_identical_observation_from_stripped_and_unstripped_payload():
    itineraries = _load_real_itineraries()
    unstripped = {"itineraries": [itineraries[0]], "debugOutput": {"fares": 0}}
    stripped = _strip(unstripped)

    assert stripped != unstripped  # sanity check: stripping actually did something

    leg = Leg(origin="Sheffield", destination="London")
    now = datetime(2026, 8, 15, 20, 0, tzinfo=UTC)

    assert parse_transitous(unstripped, leg, now) == parse_transitous(stripped, leg, now)
