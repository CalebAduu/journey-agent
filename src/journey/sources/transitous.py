"""TransitousClient: real Source for api.transitous.org (MOTIS/GTFS).

Parsing lives in parse_transitous(), a pure function tested directly
against a saved real fixture (fixtures/transitous_journeys.json) - see
tests/test_transitous.py. fetch() is everything around that: HTTP,
timeouts, and the never-raise contract.

Confirmed against real live responses:
  - Transitous never returns a fare/price - only "agencyFareUrl" (often
    empty) and "debugOutput.fares": 0. Every Observation from this client
    has price=None.
  - "realTime" (bool) and "alerts" (list, each with "headerText") are
    present per leg. "cancelled" (bool) is present on transit legs and
    absent on WALK legs (no cancellation concept for a walk).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

from journey.cache import ResponseCache
from journey.domain import Leg, Observation, SourceStatus
from journey.sources.base import Clock

SOURCE_NAME = "transitous"
MODE = "rail"
BASE_URL = "https://api.transitous.org"
TIMEOUT_SECONDS = 30.0
USER_AGENT = "journey-agent (https://github.com/CalebAduu/journey-agent)"


def _itinerary_has_cancelled_leg(itinerary: dict) -> bool:
    return any(leg.get("cancelled", False) for leg in itinerary.get("legs", []))


def parse_transitous(payload: dict, leg: Leg, now: datetime) -> Observation:
    itineraries = payload.get("itineraries", [])
    usable = [it for it in itineraries if not _itinerary_has_cancelled_leg(it)]

    if not usable:
        detail = (
            f"no itineraries {leg.origin} -> {leg.destination}"
            if not itineraries
            else f"all itineraries cancelled {leg.origin} -> {leg.destination}"
        )
        return Observation(
            source=SOURCE_NAME,
            mode=MODE,
            status=SourceStatus.FRESH_EMPTY,
            observed_at=now,
            detail=detail,
        )

    # The itinerary this Observation reports on: shortest duration among
    # the usable (non-cancelled) options. realTime/alerts are read from
    # this same itinerary's legs, not aggregated across every alternative
    # Transitous returned - the Observation describes the one answer it's
    # actually giving.
    chosen = min(usable, key=lambda it: it["duration"])
    chosen_legs = chosen.get("legs", [])

    status = SourceStatus.FRESH if any(leg_.get("realTime", False) for leg_ in chosen_legs) else SourceStatus.STALE

    alert_texts = [
        alert["headerText"]
        for leg_ in chosen_legs
        for alert in (leg_.get("alerts") or [])
        if alert.get("headerText")
    ]

    return Observation(
        source=SOURCE_NAME,
        mode=MODE,
        status=status,
        price=None,
        duration=timedelta(seconds=chosen["duration"]),
        observed_at=now,
        detail="; ".join(alert_texts),
    )


def _next_weekday_mid_morning(now: datetime) -> datetime:
    """A deterministic, realistic departure time for live queries: the
    next Mon-Fri at 10:00, so a run started on a Saturday evening still
    asks Transitous about typical weekday service rather than whatever
    moment happens to be "now"."""
    candidate = now
    while True:
        candidate += timedelta(days=1)
        if candidate.weekday() < 5:
            break
    return candidate.replace(hour=10, minute=0, second=0, microsecond=0)


@dataclass
class TransitousClient:
    cache: ResponseCache
    clock: Clock
    http: httpx.AsyncClient | None = None  # unused in replay mode
    name: str = SOURCE_NAME
    mode: str = MODE

    async def fetch(self, leg: Leg) -> Observation:
        now = self.clock.now()
        try:
            departure = _next_weekday_mid_morning(now)
            payload = await self.cache.get(
                self.http,
                f"{BASE_URL}/api/v1/plan",
                {
                    "fromPlace": leg.stop_id("origin", SOURCE_NAME),
                    "toPlace": leg.stop_id("destination", SOURCE_NAME),
                    "time": departure.isoformat(),
                },
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT_SECONDS,
            )
            return parse_transitous(payload, leg, now)
        except (TimeoutError, httpx.TimeoutException):
            # httpx raises its own TimeoutException family (ReadTimeout,
            # ConnectTimeout, ...) on a real timeout - it does NOT inherit
            # from TimeoutError/asyncio.TimeoutError, so both are caught
            # here to actually map every timeout onto TIMED_OUT. CacheMiss
            # falls through to the except Exception below, same as any
            # other unhandled failure - it becomes ERROR.
            return Observation(
                source=SOURCE_NAME,
                mode=MODE,
                status=SourceStatus.TIMED_OUT,
                detail=f"timed out after {TIMEOUT_SECONDS}s",
                observed_at=now,
            )
        except Exception as exc:  # noqa: BLE001 - fetch must never raise, by design
            return Observation(
                source=SOURCE_NAME,
                mode=MODE,
                status=SourceStatus.ERROR,
                detail=f"{type(exc).__name__}: {exc}",
                observed_at=now,
            )
