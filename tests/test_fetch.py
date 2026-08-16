"""Phase 3: async harvest against a deadline, never cancelling what's
still running.

SPEC.md §9 Phase 3 done-when: a test with one instant + one slow source
returns at the deadline with one FRESH and one TIMED_OUT, and the slow
task is still pending, not cancelled. Durations here are scaled down
from the spec's illustrative 2s/10s so the suite stays fast - the
mechanism under test (asyncio.wait leaving pending tasks alone) doesn't
care about the actual numbers.
"""

import asyncio
from datetime import timedelta

from journey.domain import Leg, Observation, SourceStatus
from journey.fetch import PendingRegistry, harvest

LEG = Leg(origin="Sheffield", destination="London")


class InstantSource:
    name = "instant"
    mode = "flight"

    async def fetch(self, leg: Leg) -> Observation:
        return Observation(
            source=self.name,
            mode=self.mode,
            status=SourceStatus.FRESH,
            duration=timedelta(minutes=90),
        )


class SlowSource:
    name = "slow"
    mode = "flight"

    def __init__(self, delay_seconds: float):
        self.delay_seconds = delay_seconds

    async def fetch(self, leg: Leg) -> Observation:
        await asyncio.sleep(self.delay_seconds)
        return Observation(
            source=self.name,
            mode=self.mode,
            status=SourceStatus.FRESH,
            duration=timedelta(minutes=120),
        )


def test_partial_harvest_returns_at_deadline_with_slow_task_still_pending():
    async def run():
        registry = PendingRegistry()
        sources = [InstantSource(), SlowSource(delay_seconds=0.3)]

        observations = await harvest(sources, LEG, timeout_seconds=0.05, registry=registry)

        assert [o.source for o in observations] == ["instant", "slow"]  # sorted
        instant_obs, slow_obs = observations
        assert instant_obs.status == SourceStatus.FRESH
        assert slow_obs.status == SourceStatus.TIMED_OUT
        assert "pending" in slow_obs.detail

        pending_task = registry.get("slow")
        assert pending_task is not None
        assert not pending_task.done()
        assert not pending_task.cancelled()

        await registry.drain()  # cleanup so the test doesn't leak a running task

    asyncio.run(run())


def test_subsequent_wait_for_recovers_the_late_observation_as_fresh():
    async def run():
        registry = PendingRegistry()
        sources = [SlowSource(delay_seconds=0.05)]

        first_pass = await harvest(sources, LEG, timeout_seconds=0.01, registry=registry)
        assert first_pass[0].status == SourceStatus.TIMED_OUT
        assert "slow" in registry

        recovered = await registry.wait_for("slow", timeout_seconds=1.0)

        assert recovered.status == SourceStatus.FRESH
        assert recovered.source == "slow"
        assert "slow" not in registry  # resolved, no longer pending

    asyncio.run(run())
