# Journey Agent

An agent that plans a multi-leg journey and reasons honestly about what it doesn't know, rather than guessing or falling back silently. The task isn't "build a travel app" - it's: does the type system make it structurally impossible to confuse a timeout with an empty result?

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate   # source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
```

`--replay` isn't load-bearing right now (see "What's unfinished" - the real network path isn't wired into the CLI yet), but it's the flag you should get in the habit of passing.

**Quickstart (no API key needed):**

```bash
python -m journey.cli --scenario flight_timeout_worthless --replay --no-llm
```

**With the LLM layer** - natural-language intent parsing and plain-English narration of each decision:

```bash
export ANTHROPIC_API_KEY=sk-...
python -m journey.cli --scenario flight_timeout_worthless --replay
```

The LLM is confined to the edges: parsing what the traveller asked for, and explaining what the agent decided. All ranking, VOI, and conflict resolution are deterministic and unit-tested. `--no-llm` runs the full agent with templated text - same decisions either way, just described differently.

Two more scenarios worth running: `--scenario flight_timeout_valuable` (the same kind of failure as above, but where waiting genuinely pays off) and `--scenario price_conflict` (two sources disagree, both values kept). A fourth, `inventory_gone`, exists too. Preference defaults to `cheapest`; pass `--preference fastest` or `--preference reliable` to see the same failure produce a different winner.

## The core idea

A source can fail in four structurally different ways, and the type system makes it impossible to write code that treats one as another:

| State | Meaning | Did we call it? | Reduces certainty? |
|---|---|---|---|
| `NotApplicable` | The mode is physically or geometrically implausible on this leg | No, by design | No |
| `Empty` | The source answered: nothing available | Yes | No |
| `Unknown` | The source timed out, errored, or never came back | Yes, but no answer | Yes |
| `Conflicted` | Two sources answered and disagree beyond a threshold | Yes, both answered | Partially |

These are five separate frozen dataclasses under a shared marker base with no fields of its own (`Available` is the fifth - the ordinary "we got a usable answer" case). Python's dataclass-generated equality checks the class before any field, so `Unknown(...)` and `Empty(...)` can never compare equal no matter what they contain. The single failure mode this project exists to prevent - catching a timeout, storing an empty result, and behaving as if the option doesn't exist - is not just discouraged by convention here, it doesn't type-check.

`NotApplicable` never reduces the certainty score - knowing a mode is impossible is knowledge, not doubt. Only `Unknown` does.

## How VOI works

`value_of_waiting = p × q × Δ`, compared against `cost_of_waiting`.

- **p** - probability the pending source responds in time. A Beta-Bernoulli prior (`ALPHA = BETA = 2.0`) updated from this session's own observed successes/failures per source. With no observations yet, `p = 2/(2+2) = 0.5`.
- **q** - the fraction of the pending source's plausible outcome range that would actually beat the current best. Computed from `infer_cost()`'s low/high bounds scored through the same weighting as every other candidate, giving a best-case and worst-case total for the pending strategy. If even the best case can't beat the incumbent, `q = 0` outright - `p` and `Δ` never get multiplied in.
- **Δ** - `best_case_score - incumbent_score`, in the same 0-1 weighted-score units as everything else being ranked.

`cost_of_waiting = wait_seconds × VALUE_OF_TIME + DOWNSTREAM_SLACK_RISK_COEFFICIENT × (wait_seconds / remaining_budget)` - the second term is why the identical failure ranks `Wait` first on an early leg and `Commit` first later: the same wait costs more as the journey's remaining slack shrinks.

**Worked example, both taken from an actual run** (`--scenario flight_timeout_worthless` / `flight_timeout_valuable`, `--seed 42`):

- London → Berlin (930km), flight pending: coach's incumbent score is `0.8000`. Flight's best plausible case scores `0.5058` - already below the incumbent - so `q = 0.0000` and `voi = 0.0000` exactly, regardless of `p`. The agent commits to coach without waiting.
- London → Amsterdam (360km), same kind of failure, coach priced at its own high/peak bound instead of its floor: flight's best case now scores `1.0000` against an incumbent of `0.8000`. `p = 0.5000`, `q = 0.2857`, `Δ = 0.2000`, `voi = 0.0286` - small, but enough to beat `cost_of_waiting` at that leg's budget, so `Wait` ranks first and the agent holds.

**The weakest link, stated plainly:** `q` assumes the pending source's real outcome is uniformly distributed between its worst and best plausible values. Real fare distributions aren't uniform - budget-carrier prices cluster near the floor with a long thin tail toward the expensive end, not spread evenly across the range. A production version would fit `q` from historical query logs instead of assuming a flat distribution between two hand-specified bounds.

## Sources: what's real and what isn't

- **Transitous** (`api.transitous.org`) - real, fully integrated. `TransitousClient` and `parse_transitous()` are tested against a genuine captured Sheffield→London response, not assumptions. Confirmed live: it never returns a price - `"fares": 0` in every response, only an empty `agencyFareUrl`.
- **VBB** (`v6.vbb.transport.rest`) - its `/locations` endpoint is confirmed real and reachable (two genuine captured responses in `fixtures/`, real stop IDs resolved). No client was built to consume it - the one `/journeys` query attempted came back empty, and nothing further was built on top of that.
- **BVG** (`v6.bvg.transport.rest`) - never actually queried. It's part of the intended route design (Berlin-Brandenburg local rail, same API family as DB/VBB) but nothing was verified against it this build.
- **DB** (`v6.db.transport.rest`) - confirmed returning `503 Service Unavailable` on every endpoint (`/locations`, `/stations`, `/journeys`) via direct investigation, with the docs page loading fine - the backend itself is down, not a bad query. No `DbClient` was built: there was nothing live to build a parser against.
- **Flights and coach** - stubbed, on every leg, unconditionally. No free flight-pricing API exists: Amadeus's Self-Service sandbox shut down 17 July 2026, and nothing replaced it.

## The pricing priors

`COST_PER_KM` and `MIN_FARE` (`pricing.py`) are hand-specified, not derived from fare data - back-solved from a handful of known real fares rather than measured systematically. The ranges are wide on purpose: UK rail on the same train can vary roughly threefold depending on booking horizon alone, so a narrow prior would be confidently wrong more often than a wide one is uselessly vague. In production these would come from historical per-route fare observations instead of hand-set bounds.

## Design decisions, with trade-offs

- **No agent framework.** Plain Python + `asyncio` throughout. Costs some boilerplate (the harvest/merge/generate/score/act pipeline is wired by hand); buys the ability to defend every line on camera instead of trusting a framework's internals.
- **Deterministic core, LLM only at the edges.** Ranking, scoring, VOI, and conflict resolution are plain Python, unit-tested, and the LLM never touches them - `narrate()` takes an already-scored, frozen `DecisionTrace` and returns a string, with no channel back into the pipeline. Costs some narrative flexibility (the LLM can't explain a decision the deterministic core didn't make); buys a demo that can't be talked into a different answer by a bad prompt.
- **Feasibility as belief, not boolean.** `RouteFeasibility` implements the physical and geometric layers of §5 (a small hardcoded exclusion list, and generalizable distance thresholds) - a false "infeasible" would silently and permanently eliminate a valid option, so the default is to call unless there's a concrete reason not to. The LLM-prior/belief-update layers described in §5 point 4 weren't built (see below); everything not explicitly excluded is just attempted.
- **Failures as return values, not exceptions.** Every source's `fetch()` is contractually unable to raise - a timeout, a malformed response, an unexpected internal error all become an `Observation` carrying the right `SourceStatus`, never a crash. Costs a small amount of ceremony (a catch-all in every source); buys a harvest loop that never needs to guess whether a missing entry means "didn't respond" or "the whole run crashed."

## What's unfinished

- Full VBB/BVG parsers were never built - only VBB's `/locations` endpoint was verified live; BVG was never queried.
- No `DbClient` exists - DB's 503 was confirmed by direct investigation, not by an integrated, tested client.
- The feasibility belief-update layer (§5 point 4 - LLM priors, updated from observations) wasn't built. Only the physical and geometric layers exist.
- `Replan` is a same-leg mode swap (fly instead of coach), not a multi-leg reroute - it can't route around a leg entirely.
- `AbandonLeg` records the choice but doesn't reconnect route topology - if a middle leg were abandoned, the next leg's origin wouldn't automatically adjust to wherever the journey actually ended up.
- `--live` and `--replay` are real, parsed CLI flags, but nothing in the current scenario wiring exercises the real-network path - every scenario uses stub sources throughout, for reasons explained in `cli.py`'s own module docstring (the one genuinely captured Transitous fixture isn't keyed the way `ResponseCache` would need to find it in replay mode).

## Tests

```bash
python -m pytest
```

71 tests across 13 files, no network in any test's hot path. The two that matter most (`tests/test_scoring.py`):

- `test_wait_ranks_last_with_voi_zero_when_it_cannot_beat_current_best` - proves VOI returns exactly `0.0` when even the best plausible outcome can't beat the incumbent, regardless of how likely a response is.
- `test_wait_ranks_above_commit_when_it_plausibly_can_beat_current_best` - proves `Wait` outranks `Commit` when the numbers say it should.

A third, `test_same_failure_produces_different_winners_under_cheapest_vs_reliable`, proves the same injected failure produces a different chosen strategy purely from changing the preference weights - same code, same failure, different, legible reason.
