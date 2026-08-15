"""Stub Source for modes with no free real API (flight, coach, ferry).
One reusable class parametrized per mode, not a subclass per mode.

fetch() must never raise: every failure path, including an unexpected
internal error, returns an Observation carrying the SourceStatus that
explains it.
"""

import random
from dataclasses import dataclass
from datetime import timedelta

from journey.domain import Leg, Money, Observation, SourceStatus
from journey.sources.base import Clock
from journey.sources.chaos import (
    ChaosDirective,
    ChaosScenario,
    EmptyResponse,
    ErrorResponse,
    Ok,
    PriceShift,
    Slow,
    Timeout,
)


@dataclass
class StubSource:
    name: str
    mode: str
    base_price: Money
    base_duration: timedelta
    scenario: ChaosScenario
    rng: random.Random
    clock: Clock

    async def fetch(self, leg: Leg) -> Observation:
        try:
            directive = self.scenario.directive_for(self.name, leg)
            return await self._respond(directive)
        except Exception as exc:  # noqa: BLE001 - fetch must never raise, by design
            return Observation(
                source=self.name,
                mode=self.mode,
                status=SourceStatus.ERROR,
                detail=f"unexpected error: {exc}",
                observed_at=self.clock.now(),
            )

    async def _respond(self, directive: ChaosDirective) -> Observation:
        if isinstance(directive, Ok):
            return self._ok()
        if isinstance(directive, Timeout):
            await self.clock.sleep(directive.seconds)
            return Observation(
                source=self.name,
                mode=self.mode,
                status=SourceStatus.TIMED_OUT,
                detail=f"source timed out after {directive.seconds}s",
                observed_at=self.clock.now(),
            )
        if isinstance(directive, ErrorResponse):
            return Observation(
                source=self.name,
                mode=self.mode,
                status=SourceStatus.ERROR,
                detail="source returned an error",
                observed_at=self.clock.now(),
            )
        if isinstance(directive, EmptyResponse):
            return Observation(
                source=self.name,
                mode=self.mode,
                status=SourceStatus.FRESH,
                detail="no availability",
                observed_at=self.clock.now(),
            )
        if isinstance(directive, PriceShift):
            return self._ok(factor=directive.factor, jitter=False)
        if isinstance(directive, Slow):
            await self.clock.sleep(directive.seconds)
            return self._ok()
        return Observation(
            source=self.name,
            mode=self.mode,
            status=SourceStatus.ERROR,
            detail=f"unrecognised chaos directive: {directive}",
            observed_at=self.clock.now(),
        )

    def _ok(self, factor: float = 1.0, jitter: bool = True) -> Observation:
        multiplier = factor * (self.rng.uniform(0.95, 1.05) if jitter else 1.0)
        price = Money(round(self.base_price.minor_units * multiplier), self.base_price.currency)
        return Observation(
            source=self.name,
            mode=self.mode,
            status=SourceStatus.FRESH,
            price=price,
            duration=self.base_duration,
            observed_at=self.clock.now(),
        )
