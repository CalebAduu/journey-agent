"""Scoring: weight each Strategy's cost/time/certainty against a
preference, and voi() for deciding whether a pending Wait is worth its
cost. The two VOI tests here are the ones the video points at.

All priors are hand-specified, not measured - see README.
"""

from dataclasses import replace

from journey.domain import LegView, Strategy, StrategyKind
from journey.merge import certainty as leg_certainty
from journey.pricing import infer_cost

# cost, time, certainty weights per preference mode (SPEC.md §8).
PREFERENCE_WEIGHTS = {
    "cheapest": (0.7, 0.2, 0.1),
    "fastest": (0.2, 0.7, 0.1),
    "reliable": (0.1, 0.2, 0.7),
}

# A cost_basis of "inferred" or "stale" means the price/interval itself
# is a guess, not a quote - certainty takes a flat hit regardless of how
# certain the underlying LegStatus otherwise is.
COST_BASIS_CERTAINTY_PENALTY = 0.2

# UseCached does NOT inherit the live status of the mode it prices. A
# cached fare is independent evidence - it was a real quote when it was
# taken - and the situation where it's worth using is precisely the one
# where the live source just died. Inheriting Unknown's 0.0 there made a
# real 26h-old quote score exactly as certain as knowing nothing, which
# is the timeout/empty conflation this project exists to avoid, wearing
# a different hat. This base minus COST_BASIS_CERTAINTY_PENALTY puts a
# cached price strictly between an observed quote and a true unknown.
CACHED_BASE_CERTAINTY = 0.8

# Weak symmetric Beta prior pseudo-counts, combined with this session's
# observed (successes, failures) for a source to estimate p = P(responds
# in time). With no observations yet, p = ALPHA / (ALPHA + BETA) = 0.5.
BETA_PRIOR_ALPHA = 2.0
BETA_PRIOR_BETA = 2.0

# Cost of waiting
VALUE_OF_TIME = 0.002

DOWNSTREAM_SLACK_RISK_COEFFICIENT = 0.05


def score_strategies(
    strategies: list[Strategy],
    leg_view: LegView,
    preference: str,
    *,
    source_reliability: dict[str, tuple[int, int]] | None = None,
    budget: float = float("inf"),
) -> list[Strategy]:
    """Score every strategy's cost/time/certainty (normalised across this
    candidate set), weight by preference, then rank best-first.

    A Wait strategy's final total_score is NOT the weighted formula - it
    comes from current_best's score adjusted by voi() minus
    cost_of_waiting, since VOI already measures expected improvement in
    the same scoring units; running it back through the weighted formula
    would double-count it.
    """
    weights = PREFERENCE_WEIGHTS[preference]
    scored = _score_batch(strategies, leg_view, weights)

    non_wait = [s for s in scored if s.kind is not StrategyKind.WAIT]
    waits = [s for s in scored if s.kind is StrategyKind.WAIT]

    if not non_wait:
        return sorted(scored, key=lambda s: s.total_score, reverse=True)

    current_best = max(non_wait, key=lambda s: s.total_score)

    final = list(non_wait)
    for w in waits:
        source_rel = (source_reliability or {}).get(w.source, (0, 0))
        p, q, delta = _voi_components(w, current_best, source_rel)
        value = p * q * delta if q != 0.0 else 0.0
        cost = cost_of_waiting(w.wait_seconds, budget)
        final.append(
            replace(
                w,
                total_score=current_best.total_score + value - cost,
                voi_value=value,
                voi_p=p,
                voi_q=q,
                voi_delta=delta,
            )
        )

    return sorted(final, key=lambda s: s.total_score, reverse=True)


def voi(strategy: Strategy, current_best: Strategy, source_reliability: tuple[int, int], budget: float) -> float:
    """p x q x delta.

    q is the fraction of [worst_case_score, best_case_score] that beats
    current_best - not just whether winning is possible, but how much of
    the plausible range does. Scores here are "higher is better", the
    mirror of a price-space (lower is better) version of the same
    formula: q = (incumbent - pending_low) / (pending_high - pending_low).

    Critically: q is 0 - and so the whole result is exactly 0 - whenever
    even the best plausible outcome can't beat current_best, regardless
    of how likely a response is.
    """
    p, q, delta = _voi_components(strategy, current_best, source_reliability)
    if q == 0.0:
        return 0.0  # information that can't change the decision is worth nothing
    return p * q * delta


def _voi_components(
    strategy: Strategy, current_best: Strategy, source_reliability: tuple[int, int]
) -> tuple[float, float, float]:
    """p, q, delta - shared by voi() and score_strategies(), which also
    stores them on the Strategy for display (Phase 9 CLI)."""
    best, worst = strategy.best_case_score, strategy.worst_case_score
    incumbent = current_best.total_score

    if best is None or worst is None or best <= incumbent:
        q = 0.0
    elif worst >= incumbent:
        q = 1.0
    else:
        q = (best - incumbent) / (best - worst)

    successes, failures = source_reliability
    p = (successes + BETA_PRIOR_ALPHA) / (successes + failures + BETA_PRIOR_ALPHA + BETA_PRIOR_BETA)
    delta = (best - incumbent) if best is not None else 0.0

    return p, q, delta


def cost_of_waiting(wait_seconds: float, budget: float) -> float:
    time_cost = wait_seconds * VALUE_OF_TIME
    slack_cost = DOWNSTREAM_SLACK_RISK_COEFFICIENT * (wait_seconds / budget) if budget > 0 else float("inf")
    return time_cost + slack_cost


def _score_batch(strategies: list[Strategy], leg_view: LegView, weights: tuple[float, float, float]) -> list[Strategy]:
    cost_weight, time_weight, certainty_weight = weights

    raw_cost = {i: _cost_midpoint(s) for i, s in enumerate(strategies)}
    raw_minutes = {i: _expected_minutes(s, leg_view) for i, s in enumerate(strategies)}
    raw_certainty = {i: _base_certainty(s, leg_view) for i, s in enumerate(strategies)}

    cost_scores = _normalize({k: (-v if v is not None else None) for k, v in raw_cost.items()})
    time_scores = _normalize({k: (-v if v is not None else None) for k, v in raw_minutes.items()})
    certainty_scores = _normalize(raw_certainty)

    result = []
    for i, s in enumerate(strategies):
        total = (
            cost_weight * cost_scores[i] + time_weight * time_scores[i] + certainty_weight * certainty_scores[i]
        )

        best_case_score = worst_case_score = None
        if s.kind is StrategyKind.WAIT and leg_view.distance_km is not None:
           
            best_price, worst_price = infer_cost(s.mode, leg_view.distance_km)
            best_case_score = _hypothetical_total(raw_cost, raw_minutes, raw_certainty, best_price, weights)
            worst_case_score = _hypothetical_total(raw_cost, raw_minutes, raw_certainty, worst_price, weights)

        result.append(
            replace(
                s,
                certainty=raw_certainty[i],
                expected_minutes=raw_minutes[i],
                cost_score=cost_scores[i],
                time_score=time_scores[i],
                certainty_score=certainty_scores[i],
                total_score=total,
                best_case_score=best_case_score,
                worst_case_score=worst_case_score,
            )
        )
    return result


def _hypothetical_total(raw_cost: dict, raw_minutes: dict, raw_certainty: dict, price, weights) -> float:
    """Score one hypothetical outcome (a resolved price, zero further
    delay, full certainty) against the real candidates already scored in
    this batch, without mutating or being seen by any other hypothetical."""
    cost_weight, time_weight, certainty_weight = weights
    cost_batch = {**{k: (-v if v is not None else None) for k, v in raw_cost.items()}, "_hyp": -price.minor_units}
    minutes_batch = {**{k: (-v if v is not None else None) for k, v in raw_minutes.items()}, "_hyp": -0.0}
    certainty_batch = {**raw_certainty, "_hyp": 1.0}

    return (
        cost_weight * _normalize(cost_batch)["_hyp"]
        + time_weight * _normalize(minutes_batch)["_hyp"]
        + certainty_weight * _normalize(certainty_batch)["_hyp"]
    )


def _cost_midpoint(strategy: Strategy) -> float | None:
    if strategy.cost_low is None or strategy.cost_high is None:
        return None
    return (strategy.cost_low.minor_units + strategy.cost_high.minor_units) / 2


def _expected_minutes(strategy: Strategy, leg_view: LegView) -> float | None:
    if strategy.kind is StrategyKind.WAIT:
        return strategy.wait_seconds / 60.0
    status = leg_view.by_mode(strategy.mode)
    if status is None:
        return None
    observations = getattr(status, "observations", ())
    matching = [
        o for o in observations if o.duration is not None and (strategy.source is None or o.source == strategy.source)
    ]
    if not matching:
        return None
    return matching[0].duration.total_seconds() / 60.0


def _base_certainty(strategy: Strategy, leg_view: LegView) -> float:
    if strategy.kind is StrategyKind.ABANDON_LEG:
        base = 1.0
    elif strategy.kind is StrategyKind.USE_CACHED:
        base = CACHED_BASE_CERTAINTY  # see the constant: never inherits the live status
    else:
        status = leg_view.by_mode(strategy.mode)
        base = leg_certainty(status) if status is not None else 1.0
    if strategy.cost_basis in ("inferred", "stale"):
        base -= COST_BASIS_CERTAINTY_PENALTY
    return max(0.0, min(1.0, base))


def _normalize(raw: dict) -> dict:
    """Min-max normalise across the candidate set - higher raw value =
    higher score. Missing data (None) gets a neutral 0.5: neither
    rewarded nor punished for lacking a comparable value on this axis."""
    present = [v for v in raw.values() if v is not None]
    if not present or max(present) == min(present):
        return {k: (0.5 if v is None else 1.0) for k, v in raw.items()}
    lo, hi = min(present), max(present)
    return {k: (0.5 if v is None else (v - lo) / (hi - lo)) for k, v in raw.items()}
