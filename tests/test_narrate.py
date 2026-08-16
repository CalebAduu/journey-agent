"""Phase 8: parse_intent() strict validation, narrate()'s templated
fallback, and LLM response caching.

SPEC.md Phase 8 done-when: identical run behaviour with and without
--no-llm. "LLM returns JSON only; validate strictly" + "fall back to
defaults on any validation failure" drive tests 1-3.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx

from journey.domain import (
    Available,
    DecisionTrace,
    Leg,
    LegView,
    Money,
    Observation,
    SourceStatus,
    Strategy,
    StrategyKind,
)
from journey.narrate import DEFAULT_BUDGET_SECONDS, DEFAULT_PREFERENCE, Narrator, PromptCache

NOW = datetime(2026, 8, 16, 10, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, fixed_time: datetime):
        self.fixed_time = fixed_time

    def now(self) -> datetime:
        return self.fixed_time

    async def sleep(self, seconds: float) -> None:
        pass


def _make_llm_client(reply_text: str, call_count: list[int]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        return httpx.Response(200, json={"content": [{"text": reply_text}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_parse_intent_falls_back_to_default_preference_on_invalid_value(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # "quick" is not one of cheapest/fastest/reliable - the whole result
    # is invalidated, not just the bad field.
    reply = json.dumps({"preference": "quick", "depart_by": "2026-08-23"})
    call_count = [0]

    async def run():
        async with _make_llm_client(reply, call_count) as http:
            narrator = Narrator(clock=FakeClock(NOW), cache=PromptCache(tmp_path), http=http)
            return await narrator.parse_intent("get me there quick")

    request = asyncio.run(run())

    assert request.preference == DEFAULT_PREFERENCE
    assert request.budget_seconds == DEFAULT_BUDGET_SECONDS


def test_parse_intent_falls_back_to_default_budget_on_unparseable_date(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reply = json.dumps({"preference": "reliable", "depart_by": "next Tuesday-ish"})
    call_count = [0]

    async def run():
        async with _make_llm_client(reply, call_count) as http:
            narrator = Narrator(clock=FakeClock(NOW), cache=PromptCache(tmp_path), http=http)
            return await narrator.parse_intent("reliably, sometime soon")

    request = asyncio.run(run())

    # strict + all-or-nothing: an unparseable date invalidates preference too
    assert request.preference == DEFAULT_PREFERENCE
    assert request.budget_seconds == DEFAULT_BUDGET_SECONDS


def test_parse_intent_drops_unknown_fields_but_keeps_valid_ones(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reply = json.dumps(
        {
            "preference": "fastest",
            "depart_by": "2026-08-23T09:00:00+00:00",
            "origin_city": "Sheffield",  # not part of the schema - must not break parsing
            "budget_gbp": 500,
        }
    )
    call_count = [0]

    async def run():
        async with _make_llm_client(reply, call_count) as http:
            narrator = Narrator(clock=FakeClock(NOW), cache=PromptCache(tmp_path), http=http)
            return await narrator.parse_intent("fastest way, budget 500")

    request = asyncio.run(run())

    assert request.preference == "fastest"
    expected_seconds = (datetime(2026, 8, 23, 9, 0, tzinfo=UTC) - NOW).total_seconds()
    assert request.budget_seconds == expected_seconds
    assert not hasattr(request, "origin_city")
    assert not hasattr(request, "budget_gbp")


def test_narrate_templated_fallback_produces_text_without_any_network(tmp_path):
    leg = Leg(origin="Sheffield", destination="London")
    observation = Observation(
        source="stub-coach",
        mode="coach",
        status=SourceStatus.FRESH,
        price=Money(2000, "GBP"),
        duration=timedelta(minutes=240),
    )
    choice = Strategy(
        kind=StrategyKind.COMMIT,
        mode="coach",
        source="stub-coach",
        reason="commit to coach",
        cost_low=Money(2000, "GBP"),
        cost_high=Money(2000, "GBP"),
        cost_basis="observed",
    )
    trace = DecisionTrace(
        leg=leg,
        decided_at=NOW,
        observations=(observation,),
        leg_view=LegView(
            origin=leg.origin,
            destination=leg.destination,
            results=(("coach", Available(observations=(observation,))),),
        ),
        unknown_reasons=(),
        ranked_strategies=(choice,),
        choice=choice,
        budget_before=300.0,
        budget_after=300.0,
    )

    narrator = Narrator(clock=FakeClock(NOW), cache=PromptCache(tmp_path), http=None, no_llm=True)
    text = asyncio.run(narrator.narrate(trace))

    assert isinstance(text, str) and text.strip()
    assert "coach" in text
    assert "stub-coach" in text
    # the trace itself is untouched - narrate() has no channel to affect it
    assert trace.choice is choice


def test_llm_response_is_cached_so_a_repeated_prompt_never_calls_the_network_again(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reply = json.dumps({"preference": "reliable", "depart_by": "2026-08-23"})
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        return httpx.Response(200, json={"content": [{"text": reply}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            cache = PromptCache(tmp_path)
            narrator_a = Narrator(clock=FakeClock(NOW), cache=cache, http=http)
            first = await narrator_a.parse_intent("same request text")

            narrator_b = Narrator(clock=FakeClock(NOW), cache=cache, http=http)
            second = await narrator_b.parse_intent("same request text")
            return first, second

    first, second = asyncio.run(run())

    assert call_count[0] == 1  # second call served entirely from disk cache
    assert first.preference == second.preference == "reliable"
    assert list(tmp_path.glob("*.txt"))  # the cache actually wrote to disk
