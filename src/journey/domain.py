from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


@dataclass(frozen=True)
class Money:
    minor_units: int
    currency: str = "GBP"


class SourceStatus(Enum):
    FRESH = "fresh"
    FRESH_EMPTY = "fresh_empty"
    STALE = "stale"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    NEVER_CALLED = "never_called"

@dataclass(frozen=True)
class Leg:
    origin: str
    destination: str
    origin_ids: tuple[tuple[str, str], ...] = ()       # (mode, stop_id)
    destination_ids: tuple[tuple[str, str], ...] = ()
    # Copied onto the LegView by merge() - see LegView for why each
    # exists. Fixed geographic/route facts, so they belong on the Leg
    # itself rather than being reconstructed at merge() time.
    distance_km: float | None = None
    abandonable: bool = False

    def stop_id(self, place: str, mode: str) -> str | None:
        pairs = self.origin_ids if place == "origin" else self.destination_ids
        return next((sid for m, sid in pairs if m == mode), None)


@dataclass(frozen=True)
class Observation:
    source: str
    mode: str
    status: SourceStatus
    price: Money | None = None
    duration: timedelta | None = None
    observed_at: datetime | None = None
    detail: str = ""

    def __post_init__(self):
        if self.status is SourceStatus.FRESH and self.price is None and self.duration is None:
            raise ValueError("FRESH must carry a price or a duration")
        no_data_statuses = (
            SourceStatus.TIMED_OUT,
            SourceStatus.ERROR,
            SourceStatus.NEVER_CALLED,
            SourceStatus.FRESH_EMPTY,
        )
        if self.status in no_data_statuses and (self.price is not None or self.duration is not None):
            raise ValueError(f"{self.status.value} cannot carry data")

    def actionable_for_cost(self) -> bool:
        return (
            self.status in (SourceStatus.FRESH, SourceStatus.STALE)
            and self.price is not None
        )

    def actionable_for_time(self) -> bool:
        return (
            self.status in (SourceStatus.FRESH, SourceStatus.STALE)
            and self.duration is not None
        )

class LegStatus:
    """Marker base with no fields: subclasses can never be equal to each
    other through a shared shape, only through being the same subclass."""


@dataclass(frozen=True)
class Unknown(LegStatus):
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class Empty(LegStatus):
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class NotApplicable(LegStatus):
    reason: str


@dataclass(frozen=True)
class Conflicted(LegStatus):
    dimension: str          # "price" | "duration"
    observations: tuple[Observation, ...]
    price_low: Money | None = None
    price_high: Money | None = None
    duration_low: timedelta | None = None
    duration_high: timedelta | None = None

@dataclass(frozen=True)
class LegView:
    origin: str
    destination: str
    results: tuple[tuple[str, LegStatus], ...]
    # Both added for Phase 5 (strategies.py): distance_km lets Commit
    # infer a cost interval when an observation has no price; abandonable
    # tells AbandonLeg whether the journey survives without this leg.
    # Neither is known at merge() time - whoever builds the LegView for a
    # real run (the planner, Phase 7) is responsible for setting them.
    distance_km: float | None = None
    abandonable: bool = False

    def by_mode(self, mode: str) -> LegStatus | None:
        return next((s for m, s in self.results if m == mode), None)

@dataclass(frozen=True)
class Available(LegStatus):
    observations: tuple[Observation, ...]


class StrategyKind(Enum):
    COMMIT = "commit"
    WAIT = "wait"
    USE_CACHED = "use_cached"
    REPLAN = "replan"
    ABANDON_LEG = "abandon_leg"


@dataclass(frozen=True)
class Strategy:
    kind: StrategyKind
    mode: str
    reason: str
    cost_low: Money | None = None
    cost_high: Money | None = None
    cost_basis: str | None = None  # "observed" | "inferred" | "stale"
    wait_seconds: float | None = None
    source: str | None = None
    # Phase 6 (scoring.py): the scored components, kept on the Strategy so
    # they can be printed alongside the total, not just used internally.
    certainty: float | None = None
    expected_minutes: float | None = None
    cost_score: float | None = None
    time_score: float | None = None
    certainty_score: float | None = None
    total_score: float | None = None
    # Wait only: this strategy's total_score at the best/worst plausible
    # outcome for its mode. voi()'s q is the fraction of the
    # [worst_case_score, best_case_score] range that beats current_best -
    # not just whether it's possible to win, but how much of the range does.
    best_case_score: float | None = None
    worst_case_score: float | None = None
    # The raw p x q x delta value voi() computed, for Wait strategies -
    # distinct from total_score, which is current_best.total_score
    # adjusted by this minus cost_of_waiting.
    voi_value: float | None = None
    # The breakdown behind voi_value, kept for display (Phase 9 CLI):
    # p = P(source responds in time), q = fraction of the plausible range
    # that beats current_best, delta = best_case_score - incumbent.
    voi_p: float | None = None
    voi_q: float | None = None
    voi_delta: float | None = None


@dataclass(frozen=True)
class DecisionTrace:
    """One decision point's full record, as data - a narrator formats
    this into text, this dataclass never does."""

    leg: Leg
    decided_at: datetime
    observations: tuple[Observation, ...]              # every source status harvested this leg
    leg_view: LegView                                    # merged per-mode outcome (Unknown carries its own reasons)
    unknown_reasons: tuple[tuple[str, str], ...]        # (mode, reason) for every Unknown mode, pulled out for convenience
    ranked_strategies: tuple[Strategy, ...]             # scored + ranked, with components and VOI, BEFORE any Wait is acted on
    choice: Strategy                                     # what was actually committed to, after acting on the top choice
    budget_before: float
    budget_after: float


@dataclass(frozen=True)
class JourneyPlan:
    legs: tuple[Leg, ...]
    trace: tuple[DecisionTrace, ...]
    committed: tuple[Strategy, ...]