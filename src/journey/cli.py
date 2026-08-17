"""CLI: run a scenario (or the real route) and print each decision point
as it happens, using Rich for the display. No decision logic lives here
- this only formats what agent.plan() already decided.

Rail is stubbed here, same as coach/flight, for a deliberate reason: the
only genuinely-captured Transitous fixture in fixtures/ was saved via a
raw curl call during Phase 2b reconnaissance, not through ResponseCache's
own sha256(url+params) keying - so it isn't actually discoverable by
ResponseCache.get() in replay mode. Wiring the real TransitousClient here
would mean every leg shows rail as ERROR (a real, honest CacheMiss - not
a crash - but uniformly unhelpful across all four scenarios, and fatal
to inventory_gone specifically, which needs rail to be the reliable
fallback). TransitousClient's real integration is already proven against
genuine captured data by its own Phase 2b/2c tests; this CLI's scenario
runs use stubs throughout so every scenario is self-contained and
reproducible under --seed. --live/--replay are still real flags - they
select ResponseCache's mode - but nothing in this file's default wiring
currently exercises that Transitous path.
"""

import argparse
import asyncio
import random
from datetime import UTC, datetime, timedelta

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from journey.agent import PlanRequest, plan
from journey.domain import (
    Available,
    Conflicted,
    Empty,
    Leg,
    Money,
    NotApplicable,
    Strategy,
    StrategyKind,
    Unknown,
)
from journey.feasibility import RouteFeasibility
from journey.fetch import PendingRegistry
from journey.merge import CONFLICTED_CERTAINTY
from journey.narrate import Narrator, PromptCache
from journey.pricing import infer_cost
from journey.sources.chaos import ChaosScenario, load_scenario
from journey.sources.stubs import StubSource

DEMO_BUDGET_SECONDS = 120.0
HARVEST_TIMEOUT_SECONDS = 3.0

# SPEC.md §6's fixed route.
REAL_ROUTE = (
    Leg(origin="Sheffield", destination="London", distance_km=260.0),
    Leg(origin="London", destination="Berlin", distance_km=930.0),
    Leg(origin="Berlin", destination="Potsdam", distance_km=25.0, abandonable=True),
)
# flight_timeout_valuable needs a leg where flight is geometrically
# plausible AND can plausibly beat coach - none of REAL_ROUTE's three
# legs qualify (leg 1/3 exclude flight entirely; leg 2 is long enough
# that flight's floor always exceeds coach - see flight_timeout).
AMSTERDAM_LEG = (Leg(origin="London", destination="Amsterdam", distance_km=360.0),)

SCENARIO_ROUTES = {
    "flight_timeout": REAL_ROUTE,
    "flight_timeout_valuable": AMSTERDAM_LEG,
    "price_conflict": REAL_ROUTE,
    "inventory_gone": REAL_ROUTE,
}

STATUS_STYLE = {
    Unknown: "red",
    NotApplicable: "dim",
}


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class ScenarioCache:
    """Replays a scenario's `cached:` block as the fare cache, so the
    UseCached strategy is reachable in a real run. Without this nothing
    ever seeds the cache and the strategy, though generated, could never
    actually fire."""

    def __init__(self, scenario: ChaosScenario):
        self.scenario = scenario

    def get(self, origin: str, destination: str, mode: str):
        return self.scenario.cached_for(origin, destination, mode)


def _build_sources(scenario_name: str, scenario: ChaosScenario, seed: int, clock) -> list[StubSource]:
    def rng(name: str) -> random.Random:
        return random.Random(f"{seed}-{name}")

    sources = [
        StubSource(
            name="stub-coach", mode="coach", base_price=Money(3720, "GBP"),
            base_duration=timedelta(hours=10), scenario=scenario, rng=rng("stub-coach"), clock=clock,
        ),
        StubSource(
            name="stub-flight", mode="flight", base_price=Money(7440, "GBP"),
            base_duration=timedelta(hours=2), scenario=scenario, rng=rng("stub-flight"), clock=clock,
        ),
        StubSource(
            name="stub-rail", mode="rail", base_price=Money(9100, "GBP"),
            base_duration=timedelta(hours=8), scenario=scenario, rng=rng("stub-rail"), clock=clock,
        ),
    ]
    if scenario_name == "price_conflict":
        # A second independent flight quote - standing in for a second
        # airline/booking site (no free flight-pricing API exists, per
        # SPEC.md §6). Same base_price as stub-flight so price_shift on
        # this one alone creates the conflict, not RNG jitter.
        sources.append(
            StubSource(
                name="stub-flight-b", mode="flight", base_price=Money(7440, "GBP"),
                base_duration=timedelta(hours=2), scenario=scenario, rng=rng("stub-flight-b"), clock=clock,
            )
        )
    return sources


def _fmt_money(money: Money | None) -> str:
    return f"£{money.minor_units / 100:.2f}" if money is not None else "—"


def _fmt_minutes(minutes: float | None) -> str:
    if minutes is None:
        return "—"
    hours, mins = divmod(round(minutes), 60)
    return f"{hours}h{mins:02d}" if hours else f"{mins}min"


def _fmt_money_range(low: Money | None, high: Money | None) -> str:
    if low is None or high is None:
        return "—"
    if low.minor_units == high.minor_units:
        return _fmt_money(low)
    return f"{_fmt_money(low)}–{_fmt_money(high)}"


def _cost_cell(strategy: Strategy, distance_km: float | None) -> str:
    """Real money first, normalised score in brackets. A Wait has no
    price of its own - it's a bet on what the pending source will quote -
    so it shows that source's plausible interval, the same interval
    scoring already used to compute best/worst case."""
    score = f"({strategy.cost_score:.2f})" if strategy.cost_score is not None else ""

    if strategy.kind is StrategyKind.WAIT:
        if distance_km is None:
            return f"— {score}".strip()
        low, high = infer_cost(strategy.mode, distance_km)
        return f"{_fmt_money_range(low, high)} {score}".strip()

    if strategy.cost_low is None:
        return f"— {score}".strip()

    value = _fmt_money_range(strategy.cost_low, strategy.cost_high)
    if strategy.cost_basis == "stale":
        # Asymmetric by design: the cached price is the floor and the
        # drifted value the ceiling, because fares ratchet up toward
        # departure rather than moving symmetrically either way.
        age = f"{strategy.cache_age_hours:.0f}h " if strategy.cache_age_hours is not None else ""
        return f"{value} {score} [dim]{age}stale[/dim]".strip()
    basis = f" [dim]{strategy.cost_basis}[/dim]" if strategy.cost_basis == "inferred" else ""
    return f"{value} {score}{basis}".strip()


def _time_cell(strategy: Strategy) -> str:
    """A Wait's expected_minutes is its own delay, not a journey time -
    the pending source hasn't answered, so the resolved journey duration
    genuinely isn't known yet and isn't invented here."""
    score = f"({strategy.time_score:.2f})" if strategy.time_score is not None else ""
    if strategy.kind is StrategyKind.WAIT:
        delay = f"+{strategy.wait_seconds:.0f}s" if strategy.wait_seconds is not None else "—"
        return f"{delay} {score}".strip()
    return f"{_fmt_minutes(strategy.expected_minutes)} {score}".strip()


def _certainty_cell(strategy: Strategy) -> str:
    """The word comes from the RAW certainty, the bracketed number from
    the normalised score. They can legitimately diverge: _score_batch
    min-max normalises certainty across the candidate set, so a batch
    whose lowest raw certainty is a Conflicted 0.50 maps that to 0.00 -
    identical on screen to a genuine Unknown unless the real value is
    shown alongside it."""
    score = f"({strategy.certainty_score:.2f})" if strategy.certainty_score is not None else ""
    raw = strategy.certainty
    if raw is None:
        return score or "—"
    if strategy.cost_basis == "stale":
        word = "stale"
    elif raw >= 1.0:
        word = "confirmed"
    elif raw <= 0.0:
        word = "unknown"
    elif abs(raw - CONFLICTED_CERTAINTY) < 1e-9:
        word = "conflicted"
    else:
        word = "partial"
    return f"{word} {score}".strip()


def _mode_row(mode: str, status, ranked: list[Strategy]) -> tuple[str, str, str, str]:
    if isinstance(status, NotApplicable):
        return (mode, "—", "—", f"[dim]NOT APPLICABLE — {status.reason}[/dim]")

    if isinstance(status, Unknown):
        details = "; ".join(o.detail for o in status.observations if o.detail) or "no observations"
        return (mode, "—", "—", f"[red]UNKNOWN[/red] — {details}")

    if isinstance(status, Empty):
        sources = ", ".join(sorted({o.source for o in status.observations})) or "?"
        return (mode, "—", "—", f"EMPTY — no availability ({sources})")

    if isinstance(status, Conflicted):
        price = "—"
        if status.price_low is not None:
            price = f"{_fmt_money(status.price_low)}–{_fmt_money(status.price_high)} (CONFLICTED)"
        duration = "—"
        if status.duration_low is not None:
            duration = (
                f"{_fmt_minutes(status.duration_low.total_seconds() / 60)}–"
                f"{_fmt_minutes(status.duration_high.total_seconds() / 60)} (CONFLICTED)"
            )
        sources = ", ".join(sorted({o.source for o in status.observations}))
        return (mode, price, duration, f"CONFLICTED — {sources}")

    if isinstance(status, Available):
        strategy = next(
            (s for s in ranked if s.mode == mode and s.kind in (StrategyKind.COMMIT, StrategyKind.USE_CACHED)),
            None,
        )
        price = "—"
        if strategy is not None and strategy.cost_low is not None:
            basis = f" ({strategy.cost_basis})" if strategy.cost_basis not in (None, "observed") else ""
            price = f"{_fmt_money(strategy.cost_low)}{basis}"
        duration = _fmt_minutes(strategy.expected_minutes) if strategy else "—"
        fresh_sources = sorted({o.source for o in status.observations if o.status.value in ("fresh", "stale")})
        return (mode, price, duration, f"{', '.join(fresh_sources)}: fresh" if fresh_sources else "fresh")

    return (mode, "?", "?", type(status).__name__)


def _print_leg_table(console: Console, decision) -> None:
    table = Table(title=f"{decision.leg.origin} -> {decision.leg.destination}", show_lines=False)
    table.add_column("Mode")
    table.add_column("Price")
    table.add_column("Duration")
    table.add_column("Status")
    for mode, status in decision.leg_view.results:
        table.add_row(*_mode_row(mode, status, list(decision.ranked_strategies)))
    console.print(table)


def _print_strategies(console: Console, decision) -> None:
    """Real values lead, normalised scores follow in brackets, so the
    trade-off is readable AND the weighted arithmetic stays auditable.
    Reason moves to its own line under each row: with real prices and
    durations in the cells, keeping it as a column wraps badly at 120
    columns and squeezes the numbers it's meant to explain."""
    table = Table(title="Ranked strategies")
    table.add_column("#")
    table.add_column("Kind")
    table.add_column("Mode")
    table.add_column("Cost")
    table.add_column("Time")
    table.add_column("Certainty")
    table.add_column("VOI")
    table.add_column("Total")

    distance_km = decision.leg_view.distance_km
    reasons = []

    for i, s in enumerate(decision.ranked_strategies):
        label = f"[bold green]{i + 1}*[/bold green]" if i == 0 else str(i + 1)
        if s.kind is StrategyKind.WAIT and s.wait_seconds is not None:
            kind_label = f"wait {s.wait_seconds:.0f}s"
        elif s.kind is StrategyKind.USE_CACHED:
            kind_label = "cached"
        else:
            kind_label = s.kind.value
        total = f"{s.total_score:.4f}" if s.total_score is not None else "—"
        voi = "—"
        reason = s.reason
        if s.kind is StrategyKind.WAIT and s.voi_value is not None:
            # delta is meaningless once q has already zeroed the whole
            # product out - showing it next to q=0.00 reads as broken
            # arithmetic rather than "this branch never looks at it."
            delta = "—" if s.voi_q == 0.0 else f"{s.voi_delta:.3f}"
            voi = f"{s.voi_value:.4f}"
            if s.voi_value == 0.0:
                reason = "cannot beat current best at any plausible price"
            # The p/q/delta breakdown lives on the reason line, not in the
            # column: it's needed for only the Wait rows and is far wider
            # than the value itself, so as a column it squeezes every
            # other row's real numbers into wrapping.
            # Escaped: Rich reads a bare [...] as a style tag and would
            # silently swallow the whole breakdown.
            reason += f"  \\[p={s.voi_p:.2f} q={s.voi_q:.2f} delta={delta}]"
        reasons.append(f"{i + 1}. {reason}")
        table.add_row(
            label,
            kind_label,
            s.mode,
            _cost_cell(s, distance_km),
            _time_cell(s),
            _certainty_cell(s),
            voi,
            total,
        )

    console.print(table)
    for line in reasons:
        console.print(f"  [dim]{line}[/dim]")


async def _run(args: argparse.Namespace) -> None:
    console = Console()
    clock = SystemClock()
    scenario = load_scenario(f"scenarios/{args.scenario}.yaml")
    legs = SCENARIO_ROUTES[args.scenario]
    sources = _build_sources(args.scenario, scenario, args.seed, clock)
    registry = PendingRegistry()

    request = PlanRequest(
        legs=legs,
        preference=args.preference,
        budget_seconds=DEMO_BUDGET_SECONDS,
        feasibility=RouteFeasibility(),
        cache=ScenarioCache(scenario),
        harvest_timeout_seconds=HARVEST_TIMEOUT_SECONDS,
    )

    progress = Progress(
        TextColumn("[bold yellow]waiting for {task.fields[source]}[/bold yellow]"),
        BarColumn(),
        TextColumn("{task.completed:.1f}s / {task.total:.1f}s"),
        console=console,
    )
    progress_tasks: dict[str, int] = {}

    def on_wait_tick(source: str, elapsed: float, total: float) -> None:
        if source not in progress_tasks:
            progress.start()
            progress_tasks[source] = progress.add_task("wait", source=source, total=total)
        progress.update(progress_tasks[source], completed=elapsed, total=total)

    console.print(f"\n[bold]Scenario:[/bold] {scenario.name}  [bold]Preference:[/bold] {args.preference}\n")

    # A real client so the LLM path (parse_intent/narrate) can actually be
    # reached under --no-llm=False - narrate.py's _call_llm() checks
    # self.http is None and returns early otherwise, which is exactly the
    # bug this shared client fixes.
    async with httpx.AsyncClient() as http:
        narrator = Narrator(clock=clock, cache=PromptCache(".llm_cache"), http=http, no_llm=args.no_llm)

        journey_plan = await plan(request, sources, registry, clock, on_wait_tick=on_wait_tick)

        if progress_tasks:
            progress.stop()

        for decision in journey_plan.trace:
            _print_leg_table(console, decision)
            _print_strategies(console, decision)
            narration = await narrator.narrate(decision)
            console.print(
                f"[bold]Action:[/bold] [dim]\\[narration: {narrator.last_narration_source}][/dim] {narration}"
            )
            console.print(f"[bold]Budget:[/bold] {decision.budget_before:.1f}s -> {decision.budget_after:.1f}s\n")

    console.print(f"[bold green]Done.[/bold green] {len(journey_plan.committed)} leg(s) decided.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="journey", description="Journey Agent - reasons honestly about what it doesn't know.")
    parser.add_argument("--scenario", choices=sorted(SCENARIO_ROUTES), required=True)
    parser.add_argument("--preference", choices=("cheapest", "fastest", "reliable"), default="cheapest")
    parser.add_argument("--replay", action="store_true", help="ResponseCache in replay mode (no network)")
    parser.add_argument("--live", action="store_true", help="ResponseCache in live mode (real network, records fixtures)")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM narrator, use the templated fallback")
    parser.add_argument("--seed", type=int, default=42, help="seeds every stub source's RNG for reproducible runs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.live and args.replay:
        raise SystemExit("--live and --replay are mutually exclusive")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
