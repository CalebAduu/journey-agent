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
    Conflicted,
    DecisionTrace,
    Empty,
    Leg,
    LegView,
    Money,
    NotApplicable,
    Observation,
    SourceStatus,
    Strategy,
    StrategyKind,
    Unknown,
)
from journey.narrate import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_PREFERENCE,
    Narrator,
    PromptCache,
    _summarize_trace_for_prompt,
)

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


def test_narrate_prompt_facts_carry_sources_top_strategies_voi_and_budget_not_just_the_choice(tmp_path):
    """The LLM must be able to explain *why*, not just describe the
    outcome - which means it needs the runner-up and its VOI breakdown,
    not only the winning strategy's name and score."""
    leg = Leg(origin="London", destination="Berlin", distance_km=930.0)
    coach_obs = Observation(
        source="stub-coach",
        mode="coach",
        status=SourceStatus.FRESH,
        price=Money(3886, "GBP"),
        duration=timedelta(hours=10),
    )
    flight_obs = Observation(
        source="stub-flight",
        mode="flight",
        status=SourceStatus.TIMED_OUT,
        detail="still pending in background after 3.0s",
    )
    commit = Strategy(
        kind=StrategyKind.COMMIT,
        mode="coach",
        source="stub-coach",
        reason="commit to stub-coach for coach",
        cost_low=Money(3886, "GBP"),
        cost_high=Money(3886, "GBP"),
        cost_basis="observed",
        total_score=0.8000,
    )
    wait = Strategy(
        kind=StrategyKind.WAIT,
        mode="flight",
        source="stub-flight",
        reason="cannot beat current best at any plausible price",
        wait_seconds=3.0,
        total_score=0.7927,
        voi_value=0.0,
        voi_p=0.5,
        voi_q=0.0,
        voi_delta=-0.2652,
    )
    trace = DecisionTrace(
        leg=leg,
        decided_at=NOW,
        observations=(coach_obs, flight_obs),
        leg_view=LegView(
            origin=leg.origin,
            destination=leg.destination,
            results=(
                ("coach", Available(observations=(coach_obs,))),
                ("flight", Unknown(observations=(flight_obs,))),
            ),
            distance_km=930.0,
        ),
        unknown_reasons=(("flight", "still pending in background after 3.0s"),),
        ranked_strategies=(commit, wait),
        choice=commit,
        budget_before=120.0,
        budget_after=119.5,
    )

    facts = _summarize_trace_for_prompt(trace)

    # every source's own status/price, not just the merged per-mode state
    assert "stub-coach" in facts
    assert "38.86" in facts
    assert "stub-flight" in facts
    assert "timed_out" in facts
    # why flight is unknown
    assert "still pending in background after 3.0s" in facts
    # both top strategies, with their scores - the runner-up must be visible
    assert "0.8000" in facts
    assert "0.7927" in facts
    # the Wait's VOI breakdown, not just that it exists
    assert "0.0000" in facts
    assert "q=0.00" in facts
    # budget state, both values
    assert "120.0" in facts
    assert "119.5" in facts


def test_narrate_prompt_facts_make_empty_and_conflicted_explicit_not_left_for_the_llm_to_infer(tmp_path):
    """Observed live: when Empty/Conflicted were only visible as a raw
    per-source status buried in a list, the LLM invented a comparison to
    a NOT_APPLICABLE flight to fill the gap instead. Spelling both out
    as their own labelled facts is the same fix already applied to
    Unknown - give the model the real reason so it has no gap to fill."""
    leg = Leg(origin="Sheffield", destination="London", distance_km=260.0)
    coach_obs = Observation(source="stub-coach", mode="coach", status=SourceStatus.FRESH_EMPTY)
    rail_a = Observation(
        source="stub-rail-a", mode="rail", status=SourceStatus.FRESH,
        price=Money(8000, "GBP"), duration=timedelta(hours=8),
    )
    rail_b = Observation(
        source="stub-rail-b", mode="rail", status=SourceStatus.FRESH,
        price=Money(12000, "GBP"), duration=timedelta(hours=8),
    )
    commit = Strategy(
        kind=StrategyKind.COMMIT, mode="rail", source="stub-rail-a",
        reason="commit to stub-rail-a for rail", total_score=0.9,
    )
    trace = DecisionTrace(
        leg=leg,
        decided_at=NOW,
        observations=(coach_obs, rail_a, rail_b),
        leg_view=LegView(
            origin=leg.origin,
            destination=leg.destination,
            results=(
                ("coach", Empty(observations=(coach_obs,))),
                (
                    "rail",
                    Conflicted(
                        dimension="price",
                        observations=(rail_a, rail_b),
                        price_low=Money(8000, "GBP"),
                        price_high=Money(12000, "GBP"),
                    ),
                ),
            ),
            distance_km=260.0,
        ),
        unknown_reasons=(),
        ranked_strategies=(commit,),
        choice=commit,
        budget_before=120.0,
        budget_after=120.0,
    )

    facts = _summarize_trace_for_prompt(trace)

    # Empty must be its own labelled fact, not just inferred from the
    # raw "fresh_empty" status string sitting inside the source list
    assert "no availability" in facts.lower()
    # the conflict's actual spread must be spelled out, not left as two
    # same-mode prices the model has to notice disagree on its own
    assert "conflict" in facts.lower()
    assert "80.00" in facts and "120.00" in facts


def test_narrate_prompt_facts_exclude_the_raw_price_of_a_not_applicable_mode(tmp_path):
    """Stub sources are queried unconditionally on every leg (see cli.py's
    module docstring), so a real, normal-looking Observation exists for a
    NOT_APPLICABLE mode even though feasibility ruled it out before any
    price ever mattered. Observed live: with that observation sitting in
    the raw source list and no signal that it was excluded, the LLM used
    it as a real comparison point ("flight was the only other available
    option at £73.62") on a leg where flight was never in contention."""
    leg = Leg(origin="Sheffield", destination="London", distance_km=260.0)
    coach_obs = Observation(
        source="stub-coach", mode="coach", status=SourceStatus.FRESH,
        price=Money(3828, "GBP"), duration=timedelta(hours=10),
    )
    # a real observation - stub sources are queried unconditionally, even
    # for a mode feasibility will go on to rule out
    flight_obs = Observation(
        source="stub-flight", mode="flight", status=SourceStatus.FRESH,
        price=Money(7362, "GBP"), duration=timedelta(hours=2),
    )
    commit = Strategy(
        kind=StrategyKind.COMMIT, mode="coach", source="stub-coach",
        reason="commit to stub-coach for coach", total_score=0.8,
    )
    trace = DecisionTrace(
        leg=leg,
        decided_at=NOW,
        observations=(coach_obs, flight_obs),
        leg_view=LegView(
            origin=leg.origin,
            destination=leg.destination,
            results=(
                ("coach", Available(observations=(coach_obs,))),
                ("flight", NotApplicable(reason="Doncaster Sheffield Airport closed (physical)")),
            ),
            distance_km=260.0,
        ),
        unknown_reasons=(),
        ranked_strategies=(commit,),
        choice=commit,
        budget_before=120.0,
        budget_after=120.0,
    )

    facts = _summarize_trace_for_prompt(trace)

    # the exclusion reason must be explicit
    assert "Doncaster" in facts
    # but the raw flight price must not leak through as a phantom
    # comparison point for a mode that was never actually in contention
    assert "73.62" not in facts
    assert "stub-flight" not in facts


def test_narrate_falls_back_to_template_when_llm_describes_a_wait_as_committed(tmp_path, monkeypatch):
    """Observed live, three separate prompt rewrites in a row: the LLM
    kept describing an accepted Wait strategy as "committed to the
    flight option" - backwards, and exactly the timeout/empty conflation
    this whole project exists to prevent, just relocated into prose.
    Prompting alone didn't reliably fix it, so this is the deterministic
    backstop: don't trust the probabilistic layer to self-police a
    distinction that matters - verify it, and fall back if it's wrong."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    leg = Leg(origin="London", destination="Amsterdam", distance_km=360.0)
    flight_obs = Observation(
        source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT,
        detail="still pending in background after 3.0s",
    )
    choice = Strategy(
        kind=StrategyKind.WAIT,
        mode="flight",
        source="stub-flight",
        reason="stub-flight is still pending, worth waiting 3s more",
        wait_seconds=3.0,
        total_score=0.8213,
        voi_value=0.0286,
        voi_p=0.5,
        voi_q=0.2857,
        voi_delta=0.2000,
    )
    trace = DecisionTrace(
        leg=leg,
        decided_at=NOW,
        observations=(flight_obs,),
        leg_view=LegView(
            origin=leg.origin,
            destination=leg.destination,
            results=(("flight", Unknown(observations=(flight_obs,))),),
        ),
        unknown_reasons=(("flight", "still pending in background after 3.0s"),),
        ranked_strategies=(choice,),
        choice=choice,
        budget_before=120.0,
        budget_after=117.0,
    )
    call_count = [0]
    bad_reply = "The agent committed to the flight option despite it having timed out."

    async def run():
        async with _make_llm_client(bad_reply, call_count) as http:
            narrator = Narrator(clock=FakeClock(NOW), cache=PromptCache(tmp_path), http=http)
            return await narrator.narrate(trace), narrator.last_narration_source

    text, source = asyncio.run(run())

    assert source == "template"
    assert "commit" not in text.lower()
    assert call_count[0] == 1  # the LLM was still tried, just rejected after the fact


def test_narrate_accepts_an_llm_response_that_correctly_describes_a_wait(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    leg = Leg(origin="London", destination="Amsterdam", distance_km=360.0)
    flight_obs = Observation(
        source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT,
        detail="still pending in background after 3.0s",
    )
    choice = Strategy(
        kind=StrategyKind.WAIT,
        mode="flight",
        source="stub-flight",
        reason="stub-flight is still pending, worth waiting 3s more",
        wait_seconds=3.0,
        total_score=0.8213,
        voi_value=0.0286,
        voi_p=0.5,
        voi_q=0.2857,
        voi_delta=0.2000,
    )
    trace = DecisionTrace(
        leg=leg,
        decided_at=NOW,
        observations=(flight_obs,),
        leg_view=LegView(
            origin=leg.origin,
            destination=leg.destination,
            results=(("flight", Unknown(observations=(flight_obs,))),),
        ),
        unknown_reasons=(("flight", "still pending in background after 3.0s"),),
        ranked_strategies=(choice,),
        choice=choice,
        budget_before=120.0,
        budget_after=117.0,
    )
    call_count = [0]
    good_reply = "The agent is waiting a little longer to hear back from the flight source."

    async def run():
        async with _make_llm_client(good_reply, call_count) as http:
            narrator = Narrator(clock=FakeClock(NOW), cache=PromptCache(tmp_path), http=http)
            return await narrator.narrate(trace), narrator.last_narration_source

    text, source = asyncio.run(run())

    assert source == "llm"
    assert text == good_reply


def test_narrate_prompt_facts_note_when_the_choice_resolves_a_mode_still_listed_as_unknown(tmp_path):
    """DecisionTrace freezes observations/unknown_reasons before act() runs,
    but the choice reflects whatever act() decided after - including a
    full re-score once a waited-on source resolves (agent.py's act(),
    lines 145-171: a top-ranked Wait always gets acted on for real, and
    the final choice is whatever wins the re-score, never Wait itself).
    So a Wait-then-Commit leg's facts said "flight: pending" while Chosen
    said "commit on flight" - a genuine contradiction in the trace, not
    an invented one. This must be visible as a fact, not left for either
    narration path to silently paper over or clumsily reconcile."""
    leg = Leg(origin="London", destination="Amsterdam", distance_km=360.0)
    flight_obs = Observation(
        source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT,
        detail="still pending in background after 3.0s",
    )
    resolved_commit = Strategy(
        kind=StrategyKind.COMMIT,
        mode="flight",
        source="stub-flight",
        reason="commit to stub-flight for flight",
        cost_low=Money(8370, "GBP"),
        cost_high=Money(8370, "GBP"),
        cost_basis="observed",
        total_score=0.8213,
    )
    trace = DecisionTrace(
        leg=leg,
        decided_at=NOW,
        observations=(flight_obs,),
        leg_view=LegView(
            origin=leg.origin,
            destination=leg.destination,
            results=(("flight", Unknown(observations=(flight_obs,))),),
        ),
        unknown_reasons=(("flight", "still pending in background after 3.0s"),),
        ranked_strategies=(resolved_commit,),
        choice=resolved_commit,
        budget_before=120.0,
        budget_after=117.0,
    )

    facts = _summarize_trace_for_prompt(trace)

    # the pending fact is still there
    assert "still pending in background after 3.0s" in facts
    # but it must also say this specific mode went on to resolve before
    # the final decision, not leave "pending" standing unqualified next
    # to a Chosen line that commits to the very same mode
    assert "resolved" in facts.lower()
    assert "0.8213" in facts


def test_narrate_template_does_not_call_a_resolved_mode_unknown_next_to_committing_to_it(tmp_path):
    """Same staleness bug as the LLM facts, same file, same root cause -
    _template_narrate() built its "unknown so far" clause from the same
    pre-act unknown_reasons, so a Wait-then-Commit leg would say "unknown
    so far: flight (still pending)... Committing to flight via
    stub-flight" in the deterministic fallback too."""
    leg = Leg(origin="London", destination="Amsterdam", distance_km=360.0)
    flight_obs = Observation(
        source="stub-flight", mode="flight", status=SourceStatus.TIMED_OUT,
        detail="still pending in background after 3.0s",
    )
    resolved_commit = Strategy(
        kind=StrategyKind.COMMIT,
        mode="flight",
        source="stub-flight",
        reason="commit to stub-flight for flight",
        cost_low=Money(8370, "GBP"),
        cost_high=Money(8370, "GBP"),
        cost_basis="observed",
        total_score=0.8213,
    )
    trace = DecisionTrace(
        leg=leg,
        decided_at=NOW,
        observations=(flight_obs,),
        leg_view=LegView(
            origin=leg.origin,
            destination=leg.destination,
            results=(("flight", Unknown(observations=(flight_obs,))),),
        ),
        unknown_reasons=(("flight", "still pending in background after 3.0s"),),
        ranked_strategies=(resolved_commit,),
        choice=resolved_commit,
        budget_before=120.0,
        budget_after=117.0,
    )

    narrator = Narrator(clock=FakeClock(NOW), cache=PromptCache(tmp_path), http=None, no_llm=True)
    text = asyncio.run(narrator.narrate(trace))

    assert "Committing to flight" in text
    # must not simultaneously call flight unknown right next to
    # committing to it - the mode resolved before the final decision
    assert "unknown" not in text.lower()


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
