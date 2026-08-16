"""Async harvest: call every source for a leg concurrently, return within
a deadline, and never lose a source that's still running.

Non-negotiable #4: never cancel a pending source task on timeout. That's
why this uses asyncio.wait (which just reports done/pending and touches
neither) rather than asyncio.gather or asyncio.wait_for directly on the
whole batch - both of those would either block until every task finishes
or cancel what's left when the deadline hits.
"""

import asyncio
from collections.abc import Callable, Sequence

from journey.domain import Leg, Observation, SourceStatus
from journey.sources.base import Source

# How often wait_for()'s optional on_tick callback fires while polling a
# pending task - fine-grained enough for a smooth CLI countdown.
TICK_INTERVAL_SECONDS = 0.1


class PendingRegistry:
    """Tasks still running when a harvest deadline passed. Kept alive -
    never cancelled - so a later Wait strategy can await the exact same
    task instead of issuing a fresh call. Only drain() cancels anything,
    and only at journey end."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}

    def register(self, source_name: str, task: asyncio.Task) -> None:
        self._tasks[source_name] = task

    def get(self, source_name: str) -> asyncio.Task | None:
        return self._tasks.get(source_name)

    def is_pending(self, source_name: str) -> bool:
        return source_name in self._tasks

    def __contains__(self, source_name: str) -> bool:
        return self.is_pending(source_name)

    async def wait_for(
        self,
        source_name: str,
        timeout_seconds: float,
        on_tick: Callable[[float, float], None] | None = None,
    ) -> Observation:
        """Re-await a still-pending task. Wrapped in asyncio.shield so
        that if THIS wait itself times out, the underlying task keeps
        running rather than being cancelled - a second, later wait_for
        can still recover it.

        on_tick, if given, fires roughly every TICK_INTERVAL_SECONDS with
        (elapsed, timeout_seconds) - purely for a live display (Phase 9
        CLI); omitting it preserves the exact original single-await
        behaviour.
        """
        task = self._tasks[source_name]
        shielded = asyncio.shield(task)

        if on_tick is None:
            observation = await asyncio.wait_for(shielded, timeout=timeout_seconds)
            self._tasks.pop(source_name, None)
            return observation

        elapsed = 0.0
        while elapsed < timeout_seconds:
            step = min(TICK_INTERVAL_SECONDS, timeout_seconds - elapsed)
            done, _pending = await asyncio.wait({shielded}, timeout=step)
            elapsed += step
            if shielded in done:
                self._tasks.pop(source_name, None)
                return shielded.result()
            on_tick(elapsed, timeout_seconds)

        raise TimeoutError(f"{source_name} did not respond within {timeout_seconds}s")

    async def drain(self) -> None:
        """Cancel every remaining pending task. Call at journey end, not
        between legs - a task registered here is meant to survive until
        something explicitly decides it's no longer wanted."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()


async def harvest(
    sources: Sequence[Source],
    leg: Leg,
    timeout_seconds: float,
    registry: PendingRegistry,
) -> list[Observation]:
    """Call every source's fetch() concurrently. Sources that finish
    within timeout_seconds contribute their real Observation. Sources
    still running at the deadline are registered (not cancelled) and get
    a synthesized TIMED_OUT placeholder here - every source yields
    exactly one Observation, pending or not. Sorted by source name so
    output is deterministic regardless of asyncio scheduling order.
    """
    tasks = {asyncio.create_task(source.fetch(leg)): source for source in sources}

    done, pending = await asyncio.wait(tasks.keys(), timeout=timeout_seconds)

    observations = [task.result() for task in done]

    for task in pending:
        source = tasks[task]
        registry.register(source.name, task)
        observations.append(
            Observation(
                source=source.name,
                mode=source.mode,
                status=SourceStatus.TIMED_OUT,
                detail=f"still pending in background after {timeout_seconds}s",
            )
        )

    observations.sort(key=lambda observation: observation.source)
    return observations
