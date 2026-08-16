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
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from journey.agent import PlanRequest
from journey.domain import (
    Conflicted,
    DecisionTrace,
    Empty,
    Leg,
    NotApplicable,
    Observation,
    Strategy,
    StrategyKind,
)
from journey.feasibility import RouteFeasibility
from journey.sources.base import Clock

logger = logging.getLogger(__name__)

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

_NARRATE_PROMPT_TEMPLATE = """You are explaining one decision already made by a deterministic \
journey-planning agent. The decision below is final - you are not choosing between options or \
second-guessing it, only explaining the reasoning behind it, using the facts given.

{facts}

Use only the facts above. Never mention a source, mode, or status that isn't listed there, and never \
speculate about anything the agent doesn't track - no travel conditions, weather, passenger experience, \
or real-time availability. A mode listed as not applicable was never in contention and has no price worth \
comparing - don't mention it beyond its listed reason, if at all.

Never calculate or state a numeric gap between two values (e.g. "10 minutes longer", "£5 cheaper"). You \
are not reliable at this arithmetic and a wrong number is worse than no number. Compare options only in \
words - cheaper, pricier, faster, slower - with no figure attached, unless that exact figure already \
appears above as its own fact.

The budget above is the agent's own time for waiting on slow sources to respond, measured in seconds. It \
has nothing to do with how long the journey itself takes. Never compare a travel time (minutes or hours) \
to the budget (seconds), and never describe a journey as fitting "within the budget" - that comparison \
doesn't mean anything.

If the strategy under "Chosen" has kind "wait": the agent has decided only to wait longer for that source \
to respond. It has NOT chosen, picked, or selected that source's mode, and nothing about it is booked, \
confirmed, or committed - there is no result yet to commit to. Say only that the agent is waiting or \
holding. The word "committed" must not appear anywhere in your response when the chosen kind is "wait".

If no source above is Unknown or Conflicted, every source gave a definite answer (including any that \
came back empty) - explain the winning choice over the runner-up in one plain sentence, with no invented \
doubt.

If a source above is Unknown or Conflicted, write 2-3 sentences: why the agent chose this strategy over \
the runner-up, what specifically is unknown or conflicted and why, and - if the runner-up is a Wait \
strategy with VOI 0 - state plainly that the missing information could not have changed the outcome and \
the agent did not wait for it.

Do not describe the journey as booked or confirmed. Do not use marketing language such as "good value" \
or "reliable service".

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
    # Set by narrate() on every call - "llm" or "template" - so a caller
    # (the CLI's Action line) can show which path actually produced the
    # last narration, not just which path was requested.
    last_narration_source: str = "template"

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
                text = raw.strip()
                if _is_narration_consistent(decision_trace.choice, text):
                    self.last_narration_source = "llm"
                    return text
                logger.warning("LLM narration rejected: described a Wait decision as committed/committing")

        self.last_narration_source = "template"
        return _template_narrate(decision_trace)

    async def _call_llm(self, prompt: str) -> str | None:
        cached = self.cache.get(prompt)
        if cached is not None:
            return cached

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("LLM path unavailable: ANTHROPIC_API_KEY is not set")
            return None
        if self.http is None:
            logger.warning("LLM path unavailable: Narrator has no http client configured")
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
        except Exception as exc:  # noqa: BLE001 - a missing/broken LLM call must never break the demo
            logger.warning("LLM call failed, falling back to template: %s: %s", type(exc).__name__, exc)
            return None

        self.cache.set(prompt, text)
        return text


def _is_narration_consistent(choice: Strategy, narration: str) -> bool:
    """A deterministic backstop, not another prompt tweak: verified live
    across three separate prompt rewrites, the LLM kept describing an
    accepted Wait as "committed to" the pending mode - backwards, and the
    exact timeout/empty conflation this project exists to prevent, just
    relocated into prose. Rather than keep tuning wording and hoping,
    reject it outright - same principle as everywhere else here: don't
    trust the probabilistic layer to self-police a distinction that
    matters."""
    return not (choice.kind is StrategyKind.WAIT and "commit" in narration.lower())


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
    """Raw facts for the LLM to reason from - not just the winning
    strategy, but the runner-up and its VOI breakdown too, since
    "why this over the alternative" is the whole point of the narration.

    Empty, Conflicted, and NotApplicable are spelled out as their own
    labelled sections, same as Unknown already was - observed live: when
    a mode's real status was only visible as a bare enum value buried
    inside the raw source list, the LLM invented a reason to fill the gap
    instead of reporting the real one (a clean-looking single price with
    no mention that a second source disagreed; a NOT_APPLICABLE mode's
    incidental stub price used as if it were a live comparison point -
    stub sources are queried unconditionally on every leg, so that
    observation genuinely exists even though feasibility ruled it out
    before price ever mattered)."""
    leg = decision_trace.leg
    not_applicable = {
        mode: status for mode, status in decision_trace.leg_view.results if isinstance(status, NotApplicable)
    }
    lines = [f"Leg: {leg.origin} to {leg.destination}", "", "Sources:"]
    for obs in decision_trace.observations:
        if obs.mode in not_applicable:
            continue  # ruled out before being asked; its price is noise, not a live option
        lines.append(f"- {obs.source} ({obs.mode}): {_describe_observation(obs)}")

    if not_applicable:
        lines.append("")
        lines.append("Not applicable (never in contention, no price was ever compared):")
        for mode, status in not_applicable.items():
            lines.append(f"- {mode}: {status.reason}")

    empty_modes = [mode for mode, status in decision_trace.leg_view.results if isinstance(status, Empty)]
    if empty_modes:
        lines.append("")
        lines.append("No availability (asked, nothing returned):")
        for mode in empty_modes:
            lines.append(f"- {mode}")

    if decision_trace.unknown_reasons:
        lines.append("")
        lines.append("Unknown at the time each source was asked:")
        choice = decision_trace.choice
        resolving_kinds = (StrategyKind.COMMIT, StrategyKind.USE_CACHED)
        for mode, reason in decision_trace.unknown_reasons:
            note = ""
            if choice.mode == mode and choice.kind in resolving_kinds:
                # the agent waited, this source answered, and the final
                # choice below is built on that answer - without this the
                # facts would say "pending" right next to a Chosen line
                # that commits to the very same mode
                note = f" - resolved before the final decision: {_describe_strategy_for_prompt(choice)}"
            lines.append(f"- {mode}: {reason}{note}")

    conflicts = [(mode, status) for mode, status in decision_trace.leg_view.results if isinstance(status, Conflicted)]
    if conflicts:
        lines.append("")
        lines.append("Conflicted (sources disagree):")
        for mode, status in conflicts:
            lines.append(f"- {mode}: {_describe_conflict(status)}")

    lines.append("")
    lines.append("Top strategies:")
    for i, strategy in enumerate(decision_trace.ranked_strategies[:2], start=1):
        lines.append(f"{i}. {_describe_strategy_for_prompt(strategy)}")

    lines.append("")
    lines.append(
        f"Budget: {decision_trace.budget_before:.1f}s before this decision, "
        f"{decision_trace.budget_after:.1f}s after."
    )

    choice = decision_trace.choice
    lines.append("")
    lines.append(f"Chosen: {choice.kind.value} on {choice.mode} ({choice.reason})")
    return "\n".join(lines)


def _describe_conflict(status: Conflicted) -> str:
    parts = [f"{status.dimension} disagreement"]
    if status.price_low is not None:
        parts.append(f"£{status.price_low.minor_units / 100:.2f}-£{status.price_high.minor_units / 100:.2f}")
    if status.duration_low is not None:
        low_min = status.duration_low.total_seconds() / 60
        high_min = status.duration_high.total_seconds() / 60
        parts.append(f"{low_min:.0f}-{high_min:.0f}min")
    sources = ", ".join(sorted({o.source for o in status.observations}))
    return f"{', '.join(parts)} across {sources}"


def _describe_observation(obs: Observation) -> str:
    parts = [obs.status.value]
    if obs.price is not None:
        parts.append(f"£{obs.price.minor_units / 100:.2f}")
    if obs.duration is not None:
        parts.append(f"{obs.duration.total_seconds() / 60:.0f}min")
    return ", ".join(parts)


def _describe_strategy_for_prompt(strategy: Strategy) -> str:
    label = f"{strategy.kind.value} {strategy.mode}"
    if strategy.source:
        label += f" via {strategy.source}"
    score = f"score {strategy.total_score:.4f}" if strategy.total_score is not None else "score n/a"
    text = f"{label} - {score}"
    if strategy.kind is StrategyKind.WAIT and strategy.voi_value is not None:
        text += f" (VOI={strategy.voi_value:.4f}, p={strategy.voi_p:.2f}, q={strategy.voi_q:.2f})"
    return text


def _template_narrate(decision_trace: DecisionTrace) -> str:
    leg = decision_trace.leg
    parts = [f"For {leg.origin} to {leg.destination}:"]

    choice = decision_trace.choice
    # unknown_reasons is frozen before act() runs, but choice reflects
    # whatever act() decided after - including a full re-score once a
    # waited-on source resolves. Without this filter, a Wait-then-Commit
    # leg would call its own chosen mode "unknown so far" one clause
    # before committing to it.
    resolved_mode = choice.mode if choice.kind in (StrategyKind.COMMIT, StrategyKind.USE_CACHED) else None
    still_unknown = [(mode, reason) for mode, reason in decision_trace.unknown_reasons if mode != resolved_mode]
    if still_unknown:
        unknowns = ", ".join(f"{mode} ({reason})" for mode, reason in still_unknown)
        parts.append(f"unknown so far: {unknowns}.")

    parts.append(_describe_choice(choice))
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
