from datetime import datetime
from typing import Protocol, runtime_checkable

from journey.domain import Leg, Observation


class Clock(Protocol):
    """Time as a dependency: sources ask this instead of touching
    datetime.now() or asyncio.sleep() directly, so runs stay reproducible
    under a fixed seed and tests never wait on real wall-clock time."""

    def now(self) -> datetime: ...

    async def sleep(self, seconds: float) -> None: ...


@runtime_checkable
class Source(Protocol):
    async def fetch(self, leg: Leg) -> Observation: ...
