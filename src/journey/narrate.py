"""The LLM layer: parse_intent() turns free text into a PlanRequest,
narrate() turns a completed DecisionTrace into plain English. Both are
optional - Narrator(no_llm=True) (or a missing API key) falls through to
a deterministic templated path with identical run behaviour.

Critical constraint: narrate() takes an already-scored, frozen
DecisionTrace and returns a str. There is no channel back into the
scoring pipeline - no shared mutable state, no return value but text -
so the LLM cannot influence ranking, VOI, or conflict resolution no
matter what it produces. It narrates what already happened; it never
decides anything.

All LLM responses are cached to disk keyed on a hash of the prompt, so
a recorded demo run never depends on the network after the first pass.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from journey.agent import PlanRequest
from journey.domain import DecisionTrace, Leg, Strategy, StrategyKind
from journey.feasibility import RouteFeasibility
from journey.sources.base import Clock

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
USER_AGENT = "journey-agent (https://github.com/CalebAduu/journey-agent)"

VALID_PREFERENCES = ("cheapest", "fastest", "reliable")
DEFAULT_PREFERENCE = "cheapest"
DEFAULT_BUDGET_SECONDS = 3600.0
MIN_BUDGET_SECONDS = 60.0  # floor against a degenerate/past depart_by

# The fixed demo route (SPEC.md §6) - parse_intent doesn't invent a
# route from free text, it only derives preference/budget from it.
DEFAULT_ROUTE = (
    Leg(origin="Sheffield", destination="London", distance_km=260.0),
    Leg(origin="London", destination="Berlin", distance_km=930.0),
    Leg(origin="Berlin", destination="Potsdam", distance_km=25.0, abandonable=True),
)

_INTENT_PROMPT_TEMPLATE = """You are parsing a travel request into structured constraints for a journey-planning agent.

User request: {text}

Respond with JSON only, no other text, in exactly this shape:
{{"preference": "cheapest" | "fastest" | "reliable", "depart_by": "<ISO 8601 date or datetime, e.g. 2026-08-20>"}}

If no preference is stated, use "cheapest". If no date is given, use a date one week from now."""

_NARRATE_PROMPT_TEMPLATE = """Narrate this already-decided journey-planning outcome in one or two plain-English \
sentences, as if explaining the decision to the traveller. Do not second-guess it or suggest alternatives - the \
decision is final.

{facts}

Respond with the narration only, no preamble."""


class _EmptyCache:
    """Placeholder until a real fare cache exists: never has anything cached."""

    def get(self, mode):
        return None


def _prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class PromptCache:
    """LLM responses cached to disk, keyed on a hash of the prompt -
    identical prompt, identical cached response, no network needed."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def get(self, prompt: str) -> str | None:
        file_path = self.path / f"{_prompt_key(prompt)}.txt"
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return None

    def set(self, prompt: str, response: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / f"{_prompt_key(prompt)}.txt").write_text(response, encoding="utf-8")


@dataclass
class Narrator:
    clock: Clock
    cache: PromptCache
    http: httpx.AsyncClient | None = None
    no_llm: bool = False

    async def parse_intent(self, text: str) -> PlanRequest:
        preference = DEFAULT_PREFERENCE
        budget_seconds = DEFAULT_BUDGET_SECONDS

        if not self.no_llm:
            raw = await self._call_llm(_INTENT_PROMPT_TEMPLATE.format(text=text))
            validated = _validate_intent_json(_parse_json_object(raw)) if raw is not None else None
            if validated is not None:
                preference, depart_by = validated
                elapsed = (depart_by - self.clock.now()).total_seconds()
                budget_seconds = max(MIN_BUDGET_SECONDS, elapsed)

        return PlanRequest(
            legs=DEFAULT_ROUTE,
            preference=preference,
            budget_seconds=budget_seconds,
            feasibility=RouteFeasibility(),
            cache=_EmptyCache(),
        )

    async def narrate(self, decision_trace: DecisionTrace) -> str:
        if not self.no_llm:
            prompt = _NARRATE_PROMPT_TEMPLATE.format(facts=_summarize_trace_for_prompt(decision_trace))
            raw = await self._call_llm(prompt)
            if raw is not None and raw.strip():
                return raw.strip()

        return _template_narrate(decision_trace)

    async def _call_llm(self, prompt: str) -> str | None:
        cached = self.cache.get(prompt)
        if cached is not None:
            return cached

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or self.http is None:
            return None

        try:
            response = await self.http.post(
                ANTHROPIC_API_URL,
                json={"model": ANTHROPIC_MODEL, "max_tokens": 512, "messages": [{"role": "user", "content": prompt}]},
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "User-Agent": USER_AGENT,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            text = response.json()["content"][0]["text"]
        except Exception:  # noqa: BLE001 - a missing/broken LLM call must never break the demo
            return None

        self.cache.set(prompt, text)
        return text


def _parse_json_object(raw: str) -> dict | None:
    """LLM returns JSON only - but LLMs commonly wrap it in a markdown
    code fence despite being told not to, so that's stripped first."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json")
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _validate_intent_json(data: dict | None) -> tuple[str, datetime] | None:
    """Strict, all-or-nothing: any single validation failure invalidates
    the whole result rather than patching in partial defaults. Unknown
    fields are dropped simply by never being read."""
    if data is None:
        return None
    preference = data.get("preference")
    if preference not in VALID_PREFERENCES:
        return None
    depart_by_raw = data.get("depart_by")
    if not isinstance(depart_by_raw, str):
        return None
    try:
        depart_by = datetime.fromisoformat(depart_by_raw)
    except ValueError:
        return None
    if depart_by.tzinfo is None:
        depart_by = depart_by.replace(tzinfo=UTC)
    return preference, depart_by


def _summarize_trace_for_prompt(decision_trace: DecisionTrace) -> str:
    leg = decision_trace.leg
    lines = [f"Leg: {leg.origin} to {leg.destination}"]
    for mode, status in decision_trace.leg_view.results:
        lines.append(f"- {mode}: {type(status).__name__}")
    for mode, reason in decision_trace.unknown_reasons:
        lines.append(f"- {mode} unknown because: {reason}")
    choice = decision_trace.choice
    lines.append(f"Chosen strategy: {choice.kind.value} on {choice.mode} ({choice.reason})")
    if choice.total_score is not None:
        lines.append(f"Score: {choice.total_score:.3f}")
    return "\n".join(lines)


def _template_narrate(decision_trace: DecisionTrace) -> str:
    leg = decision_trace.leg
    parts = [f"For {leg.origin} to {leg.destination}:"]

    if decision_trace.unknown_reasons:
        unknowns = ", ".join(f"{mode} ({reason})" for mode, reason in decision_trace.unknown_reasons)
        parts.append(f"unknown so far: {unknowns}.")

    parts.append(_describe_choice(decision_trace.choice))
    return " ".join(parts)


def _describe_choice(choice: Strategy) -> str:
    cost = _describe_cost(choice)
    if choice.kind is StrategyKind.COMMIT:
        return f"Committing to {choice.mode} via {choice.source}{cost}."
    if choice.kind is StrategyKind.WAIT:
        return f"Waiting {choice.wait_seconds:.0f}s more for {choice.source} on {choice.mode}."
    if choice.kind is StrategyKind.USE_CACHED:
        return f"Using a cached {choice.mode} price{cost}."
    if choice.kind is StrategyKind.REPLAN:
        return f"Replanning via {choice.mode} instead."
    if choice.kind is StrategyKind.ABANDON_LEG:
        return f"Abandoning this leg: {choice.reason}."
    return choice.reason


def _describe_cost(strategy: Strategy) -> str:
    if strategy.cost_low is None or strategy.cost_high is None:
        return ""
    if strategy.cost_low == strategy.cost_high:
        amount = f" for £{strategy.cost_low.minor_units / 100:.2f}"
    else:
        amount = f" for £{strategy.cost_low.minor_units / 100:.2f}-£{strategy.cost_high.minor_units / 100:.2f}"
    if strategy.cost_basis and strategy.cost_basis != "observed":
        amount += f" ({strategy.cost_basis})"
    return amount
