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
python -m journey.cli --scenario flight_timeout --replay --no-llm
```

**With the LLM layer** - natural-language intent parsing and plain-English narration of each decision:

```bash
$env:ANTHROPIC_API_KEY = "sk..."   # export ANTHROPIC_API_KEY=sk-... on macOS/Linux
python -m journey.cli --scenario flight_timeout --replay
```

The LLM is confined to the edges: parsing what the traveller asked for, and explaining what the agent decided. All ranking, VOI, and conflict resolution are deterministic and unit-tested. `--no-llm` runs the full agent with templated text - same decisions either way, just described differently.

Two more scenarios worth running: `--scenario flight_timeout_valuable` (the same kind of failure as above, but where waiting genuinely pays off) and `--scenario price_conflict` (two sources disagree, both values kept). A fourth, `inventory_gone`, exists too. Preference defaults to `cheapest`; pass `--preference fastest` or `--preference reliable` to see the same failure produce a different winner.

**The three runs the walkthrough covers**, in order — the arithmetic behind each is broken down in [The three demo runs, in numbers](#the-three-demo-runs-in-numbers):

```bash
python -m journey.cli --scenario flight_timeout --preference cheapest --replay
python -m journey.cli --scenario inventory_gone --replay
python -m journey.cli --scenario flight_timeout --preference fastest --replay
```

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

## How the numbers work

Every component score is 0-1, higher is better, and `Total` is their weighted sum. The CLI prints the real value first and the normalised score in brackets - `£38.86 (1.00)` - so the arithmetic below can be checked directly off any row. Every figure in this section is taken from an actual `--seed 42 --replay` run, not worked out by hand.

### The preference weights

`PREFERENCE_WEIGHTS` (`scoring.py`) is the whole of what `--preference` does:

| Preference | cost | time | certainty |
|---|---|---|---|
| `cheapest` (default) | **0.7** | 0.2 | 0.1 |
| `fastest` | 0.2 | **0.7** | 0.1 |
| `reliable` | 0.1 | 0.2 | **0.7** |

They appear in the CLI header and in the column titles (`Cost x0.7`), so the multiplier is never hidden. Certainty is deliberately never the dominant term under `cheapest` or `fastest`: it breaks ties and penalises guesses, but a traveller who asked for the cheapest option should still get it when the cheap option is well-evidenced.

### Normalisation, and why it is necessary

The three axes arrive in incompatible units: cost in pence, time in minutes, certainty already 0-1. Weighting and adding those directly is meaningless - `9234` pence would swamp `480` minutes purely because pence are smaller units, and the weights would express nothing. So each axis is **min-max normalised across the candidate set for that leg**, with cost and time negated first so that "higher score" always means "better".

Three consequences worth stating plainly, because they surprise people reading the table:

- **Scores are relative to the candidates on offer, not absolute.** `1.00` on cost means *cheapest of what is available on this leg*, not cheap. In `inventory_gone` leg 1, coach is `Empty` and flight is `NotApplicable`, leaving rail as the only candidate - so rail scores `1.00` on every axis and totals `1.0000` at £89.44, the most expensive fare in the whole run. A perfect score there means "only option", not "great deal".
- **Missing values get a neutral 0.5**, neither rewarded nor punished. `Replan` carries no price of its own, so its cost cell reads `— (0.50)`. Same for a `Wait`, whose cost is not yet known.
- **Adding a candidate re-scores every other candidate.** The range shifts, so all the fractions move. This is not a bug, but it does mean scores are only comparable within one leg's table.

Worked, from `flight_timeout` leg 2 (London->Berlin) under `cheapest`. Cost values present in the batch are coach `3886p`, cached flight midpoint `7511.5p`, rail `9234p`:

```
negated:  coach -3886   cached -7511.5   rail -9234
min = -9234, max = -3886, range = 5348

coach   (-3886  + 9234) / 5348 = 1.00
cached  (-7511.5 + 9234) / 5348 = 0.32
rail    (-9234  + 9234) / 5348 = 0.00
```

and the winning row reconciles exactly:

```
commit coach = 0.7 x 1.00  +  0.2 x 0.00  +  0.1 x 1.00  =  0.8000
commit rail  = 0.7 x 0.00  +  0.2 x 0.20  +  0.1 x 1.00  =  0.1400
```

### Certainty by state

The `Certainty` column shows the word and the number, because `0.50` alone doesn't say *why*:

| Shown | Value | State | Meaning |
|---|---|---|---|
| `confirmed` | **1.0** | `Available` | A source answered with usable data |
| `confirmed` | **1.0** | `Empty` / `NotApplicable` | A definite answer, or a mode ruled out by design - both are knowledge, not doubt |
| `partial` | **0.8** | `Available`, inferred price | The state is sound but the price is a distance estimate, not a quote |
| `partial` | **0.7** | `Available` + disruption detail | Answered, but carrying a live disruption note (`DISRUPTION_CERTAINTY_PENALTY = 0.3`) |
| `stale` | **0.6** | `UseCached` | `CACHED_BASE_CERTAINTY = 0.8` minus the `0.2` non-quote penalty |
| `conflicted` | **0.5** | `Conflicted` | Two sources answered and disagree: a bounded interval, strictly more than nothing |
| `unknown` | **0.0** | `Unknown` | Asked, no answer - the only state that actually reduces certainty |

(`partial` is the catch-all label for any value strictly between the named tiers; the number beside it is always the real one.)

Two of these are load-bearing for the thesis. `NotApplicable` sits at `1.0`: knowing Doncaster Sheffield Airport is closed is *information*, and docking certainty for it would punish the agent for knowing something. And `stale` at `0.6` does **not** inherit the certainty of the dead live source it is standing in for - a cached fare was a real quote when it was taken, and the situation where it is worth consulting is exactly the one where the live source just failed.

An `inferred` or `stale` cost basis costs a flat `COST_BASIS_CERTAINTY_PENALTY = 0.2` on top of the state's own value, because the *price* is a guess even when the state is sound.

### The cost priors — assumptions, not measurements

`COST_PER_KM` and `MIN_FARE` (`pricing.py`) are **hand-specified, back-solved from a handful of known real fares**. They are not derived from fare data and should not be read as measurements:

| Mode | £/km low | £/km high | Minimum fare |
|---|---|---|---|
| rail | 0.12 | 0.35 | £5.00 |
| coach | 0.04 | 0.09 | £3.00 |
| flight | 0.08 | 0.40 | £25.00 |

The ranges are wide deliberately: UK rail on the same train varies roughly threefold on booking horizon alone, so a narrow prior would be confidently wrong more often than a wide one is uselessly vague. `infer_cost()` floors the low bound at `MIN_FARE` and raises the high bound to match if flooring would invert the interval. For the London->Berlin leg, `infer_cost("flight", 930)` gives **£74.40 - £372.00**, which is what the `Wait` rows display as flight's plausible range.

### Asymmetric staleness drift

`drift_cached()` treats a cached fare as a **floor, not a midpoint**, because fares ratchet upward toward departure as advance quotas sell out - a day-old quote is far more likely to be an underestimate than an overestimate.

```
STALENESS_DRIFT_PER_HOUR = 0.004     # ~10% over 24h, assumed, not measured
MAX_CACHE_AGE_HOURS      = 72        # older than this, UseCached is not offered

high = price x (1 + 0.004 x age_hours)
```

The `flight_timeout` scenario seeds a £71.40 flight price at 26 hours old, which is inside the 72-hour limit and drifts to:

```
£71.40 x (1 + 0.004 x 26) = £71.40 x 1.104 = £78.83
```

so the row reads `£71.40–£78.83 (0.32) 26h stale`. The interval is one-sided on purpose: the cached price is the best case, never the expected case.

### VOI = p × q × Δ

Waiting is only worth it if the information could actually change the decision. `voi()` (`scoring.py`) computes:

- **p** — probability the pending source answers in time. A Beta-Bernoulli prior (`ALPHA = BETA = 2.0`) updated from this session's observed successes and failures for that source. With nothing observed yet, `p = 2/(2+2) = 0.5000`.
- **q** — the fraction of the pending source's plausible outcome range that would actually beat the incumbent. Both bounds from `infer_cost()` are scored through the same weighted pipeline as every real candidate, giving `best_case_score` and `worst_case_score`; then `q = (best − incumbent) / (best − worst)`, clamped to `1.0` when even the worst case wins. **If the best case cannot beat the incumbent, `q = 0` and the whole product is exactly `0`, regardless of how likely a response is.**
- **Δ** — `best_case_score − incumbent_score`, in the same 0-1 units as everything else being ranked.

**The q = 0 case** — `flight_timeout` leg 2, `--preference cheapest`. Coach is confirmed at **£38.86**. Flight's cheapest plausible price is its **£74.40** floor — already nearly double coach — so even a perfect flight result loses:

```
incumbent (commit coach)  = 0.8000
best_case  (flight @ £74.40)  = 0.5348      <- below the incumbent
worst_case (flight @ £372.00) = 0.3000

best_case < incumbent  ->  q = 0.0000
VOI = 0.5000 x 0.0000 x (-0.2652) = 0.0000
```

The agent commits to coach immediately. This is the headline: it does not wait, and it does not need to see the flight price to know that. Note `Δ` is negative here and displayed as `—`, because once `q` has zeroed the product `Δ` is never read.

**The q = 1 case** — the same leg, same failure, `--preference fastest`. Now time carries 0.7 and flight's two-hour journey is worth far more, so *both* bounds beat the incumbent:

```
incumbent (use_cached flight) = 0.4744
best_case  (flight @ £74.40)  = 0.8671
worst_case (flight @ £372.00) = 0.8000     <- still above the incumbent

worst_case >= incumbent  ->  q = 1.0000
VOI = 0.5000 x 1.0000 x 0.3927 = 0.1963
```

**The weakest link, stated plainly:** `q` assumes the outcome is uniformly distributed between the two bounds. Real fare distributions are not uniform — budget-carrier prices cluster near the floor with a long thin tail upward. A production version would fit `q` from historical query logs rather than assuming a flat distribution between two hand-set bounds.

### Why Wait is scored differently

Every other strategy gets the weighted sum. A `Wait` instead gets:

```
Total = best non-wait total  +  VOI  -  cost_of_waiting
```

The reason is double-counting. VOI is *already* an expected improvement measured in these same weighted score units — it was produced by running hypothetical outcomes through the identical pipeline. Feeding it back through the weights would count the same benefit twice. So a `Wait` is scored as "what I have now, plus what asking might gain, minus what asking costs".

```
cost_of_waiting = wait_seconds x VALUE_OF_TIME
                + DOWNSTREAM_SLACK_RISK_COEFFICIENT x (wait_seconds / remaining_budget)

VALUE_OF_TIME = 0.002        DOWNSTREAM_SLACK_RISK_COEFFICIENT = 0.05
```

The second term is why the *same* pending failure ranks `Wait` first early in a journey and `Commit` first later: an identical wait costs more as the remaining budget shrinks. Both leg-2 wait rows reconcile:

```
wait 3s (cheapest) = 0.8000 + 0.0000 - (3 x 0.002 + 0.05 x 3/120)  = 0.7927
wait 8s (cheapest) = 0.8000 + 0.0000 - (8 x 0.002 + 0.05 x 8/120)  = 0.7807
wait 3s (fastest)  = 0.4744 + 0.1963 - 0.00725                      = 0.6635
```

Because this formula differs, the CLI prints a legend under every table naming the exception — otherwise a reviewer checking a `Wait` row against the column weights would correctly conclude the arithmetic was broken.

### The three demo runs, in numbers

**1. `--scenario flight_timeout --preference cheapest`** — *information that cannot change the decision is worth nothing.*

Leg 1 (Sheffield->London) is the quiet baseline: coach £38.28 wins at `0.8000`, flight is `NOT APPLICABLE` because Doncaster Sheffield Airport is closed — a hardcoded physical exclusion, and one that costs no certainty. Leg 2 is the point of the whole project: flight misses the 3s harvest deadline and merges as `Unknown`, and the agent ranks two `Wait` options **below** committing, because `q = 0`. The `cached` row appears here too, at `£71.40–£78.83 (0.32) 26h stale`, scoring `0.3855` — real information, correctly discounted, still not competitive. Leg 3 offers `abandon_leg` at `0.5500` (Berlin->Potsdam is the one abandonable leg) and coach still wins at `0.8000`.

The flight in this scenario does eventually answer, at £180.00 — but under `cheapest` the agent never waits, so it never finds out. Declining to spend three seconds on information that provably cannot change the answer *is* the result.

**2. `--scenario inventory_gone`** — *empty is not the same as unknown.*

Leg 1: coach comes back `EMPTY` — the source answered, and the answer was "nothing available". Certainty stays at `1.0`, because a definite "no" is knowledge. Flight is `NOT APPLICABLE`. Rail is the only candidate left and totals `1.0000` at £89.44 — the relative-normalisation point above, made concrete. Had coach instead *timed out*, it would be `UNKNOWN` at certainty `0.0` and the agent would have had a live `Wait` decision to make. Same visible outcome on the surface, entirely different epistemic state, and the type system keeps them apart.

**3. `--scenario flight_timeout --preference fastest`** — *the same failure, a different answer.*

Nothing about the world changed: identical scenario, identical seed, identical injected fault. Only the weights moved. Leg 1 flips from coach to **rail** (`0.8000` vs coach's `0.3000` — the exact mirror of run 1). Leg 2 flips harder: `wait 3s` now ranks **first** at `0.6635`, because with time weighted at 0.7 the flight's two-hour journey means even its worst plausible price beats the incumbent, so `q` goes from `0.0000` to `1.0000` and VOI from `0.0000` to `0.1963`.

So the agent waits — and this time the wait resolves. The CLI prints the step, because the committed strategy is no longer row 1 of the table above it and would otherwise look unexplained:

```
Waited: stub-flight answered £180.00, 2h00 -> re-ranked -> commit flight via stub-flight
```

The arrival re-runs the whole pipeline — re-merge, re-generate, re-score — against the new evidence. Flight is now `Available` rather than `Unknown`, which also drops the `cached` row: a stale price is a fallback for not having a live one, and there is now a live one. Against coach and rail, flight commits at `0.8000` — `0.2 × 0.0000 + 0.7 × 1.0000 + 0.1 × 1.0000`, winning on the 2h journey while being the most expensive fare on the leg. Budget drops `120.0s -> 117.0s`, the honest cost of the decision, and the only thing in the whole run that consumes it.

Worth saying plainly: £180.00 loses under `cheapest` and wins under `fastest`. The agent did not get a better flight by waiting — it got the *same* flight, and only one of the two preferences had any reason to pay three seconds to find out about it.

## Sources: what's real and what isn't

- **Transitous** (`api.transitous.org`) - real, fully integrated. `TransitousClient` and `parse_transitous()` are tested against a genuine captured Sheffield→London response, not assumptions. Confirmed live: it never returns a price - `"fares": 0` in every response, only an empty `agencyFareUrl`.
- **VBB** (`v6.vbb.transport.rest`) - its `/locations` endpoint is confirmed real and reachable (two genuine captured responses in `fixtures/`, real stop IDs resolved). No client was built to consume it - the one `/journeys` query attempted came back empty, and nothing further was built on top of that.
- **BVG** (`v6.bvg.transport.rest`) - never actually queried. It's part of the intended route design (Berlin-Brandenburg local rail, same API family as DB/VBB) but nothing was verified against it this build.
- **DB** (`v6.db.transport.rest`) - confirmed returning `503 Service Unavailable` on every endpoint (`/locations`, `/stations`, `/journeys`) via direct investigation, with the docs page loading fine - the backend itself is down, not a bad query. No `DbClient` was built: there was nothing live to build a parser against.
- **Flights and coach** - stubbed, on every leg, unconditionally. No free flight-pricing API exists: Amadeus's Self-Service sandbox shut down 17 July 2026, and nothing replaced it.

## Chaos injection

Failures are injected via a chaos layer wrapping the same source protocol as the real clients, so they're reproducible for testing and demonstration. The agent's response to them is not scripted — the *same* injected fault produces a different decision depending on the economics. `flight_timeout` is the cleanest demonstration: one scenario, one seed, one slow flight source, and the agent declines to wait under `--preference cheapest` but waits and commits to the flight under `--preference fastest`. Deutsche Bahn's outage during development was a genuine, uninjected failure handled by the same code path.

## Design decisions, with trade-offs

- **No agent framework.** Plain Python + `asyncio` throughout. Costs some boilerplate (the harvest/merge/generate/score/act pipeline is wired by hand); buys the ability to defend every line on camera instead of trusting a framework's internals.
- **Deterministic core, LLM only at the edges.** Ranking, scoring, VOI, and conflict resolution are plain Python, unit-tested, and the LLM never touches them - `narrate()` takes an already-scored, frozen `DecisionTrace` and returns a string, with no channel back into the pipeline. Costs some narrative flexibility (the LLM can't explain a decision the deterministic core didn't make); buys a demo that can't be talked into a different answer by a bad prompt.
- **Feasibility as belief, not boolean.** `RouteFeasibility` implements the physical and geometric layers of §5 (a small hardcoded exclusion list, and generalizable distance thresholds) - a false "infeasible" would silently and permanently eliminate a valid option, so the default is to call unless there's a concrete reason not to. The LLM-prior/belief-update layers described in §5 point 4 weren't built (see below); everything not explicitly excluded is just attempted.
- **Failures as return values, not exceptions.** Every source's `fetch()` is contractually unable to raise - a timeout, a malformed response, an unexpected internal error all become an `Observation` carrying the right `SourceStatus`, never a crash. Costs a small amount of ceremony (a catch-all in every source); buys a harvest loop that never needs to guess whether a missing entry means "didn't respond" or "the whole run crashed."

## Hard constraints vs. soft preferences

The agent optimises a **weighted objective**, not a constrained one. Under `fastest`, time carries 70% of the weight, so a fast-but-expensive option wins and the cheaper alternative stays visible beside it with the trade-off legible. Run 3 is exactly that shape: flight commits at £180.00, while coach sits two rows below at £38.86 and `0.3000` — not hidden, just outranked, and you can read why straight off the row.

What it does **not** support is a hard constraint — *fastest under £200* — where an option breaching the ceiling is removed from consideration before scoring rather than down-weighted. Today there is no way to express that: at `--max-cost 150` the £180 flight should never have been scored at all, and instead it would simply win.

This is a deliberate scope decision rather than an oversight. A soft weight and a hard constraint act at different stages of the pipeline — scoring versus eligibility filtering — and I chose to build the reasoning layer thoroughly rather than half-build a constraint system alongside it. With more time I'd add `--max-cost` and `--arrive-by` as pre-scoring filters, including the case that makes constraints genuinely interesting: when a constraint eliminates every option, the agent has to say so honestly rather than return nothing or quietly relax the constraint. That failure mode is the same class of problem as the rest of this project — the difference between "no answer" and "no acceptable answer" is exactly the kind of distinction the type system here exists to keep separate.



## What I'd build next given more time

A second integrated real source. Transitous is fully wired and tested against recorded fixtures; VBB's /locations was verified live and BVG confirmed reachable, but neither has a parser. This is the highest-value next step because it turns duration conflict on the Berlin legs from injected into observed — the conflict-detection code path already exists and is unit-tested, so the work is a VbbClient parser mapping their response shape into Observation, not new logic. The two services share DB's API shape, so one parser covers both.

Feasibility belief-updates. The physical and geometric layers work; the belief-update layer (LLM-seeded priors that shift as real observations arrive) is specified in SPEC.md §5 but unbuilt. The structure is there — FeasibilityBelief carries p_feasible and an observation count — so the remaining work is the update rule and wiring it to harvest results. I stopped at the two deterministic layers because they're defensible without a calibration story; the learned layer needs data I'd want to validate before trusting.

Hard constraints. --preference fastest is a scoring weight, not a cutoff — there's no --max-cost or --arrive-by, so nothing removes an option for breaching a ceiling rather than merely down-weighting it. This is a deliberate boundary, not an oversight: a soft weight and a hard constraint act at different pipeline stages (scoring vs. eligibility filtering), and I built the reasoning layer thoroughly rather than half-building both. Adding them means a pre-scoring filter plus the edge case where a constraint eliminates every option and the agent must report that honestly rather than returning nothing. See Hard constraints vs. soft preferences.

Richer replanning. Replan currently swaps modes within a leg; it can't reroute across legs via a different hub. AbandonLeg records the choice but doesn't reconnect route topology — abandon a middle leg and the next leg's origin wouldn't adjust. Both are safe in the current demo because only the final contingency leg is abandonable, but a general solution needs the journey represented as a graph the agent can re-solve, rather than a fixed three-leg list.

Exercising the real-network path. --live and --replay are parsed and real, but every shipped scenario uses stubs, because the one captured Transitous fixture isn't keyed the way ResponseCache looks it up in replay. Reconciling that — so --replay serves recorded real responses and --live records fresh ones — is small but I deprioritised it below getting the reasoning layer correct.

## Tests

```bash
python -m pytest
```

88 tests across 13 files, no network in any test's hot path. The two that matter most (`tests/test_scoring.py`):

- `test_wait_ranks_last_with_voi_zero_when_it_cannot_beat_current_best` - proves VOI returns exactly `0.0` when even the best plausible outcome can't beat the incumbent, regardless of how likely a response is.
- `test_wait_ranks_above_commit_when_it_plausibly_can_beat_current_best` - proves `Wait` outranks `Commit` when the numbers say it should.

A third, `test_same_failure_produces_different_winners_under_cheapest_vs_reliable`, proves the same injected failure produces a different chosen strategy purely from changing the preference weights - same code, same failure, different, legible reason.
