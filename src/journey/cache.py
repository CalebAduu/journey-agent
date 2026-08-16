"""ResponseCache: record/replay layer for real HTTP sources.

Two modes:
  - "live": make the real request, write a stripped copy to disk, return
    the FULL (unstripped) payload to the caller.
  - "replay": read the stripped copy from disk, no network. Raises
    CacheMiss if nothing was recorded for this exact request.

"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import httpx


class CacheMiss(Exception):
    """Raised in replay mode when no recorded response exists for this
    exact request."""


def _cache_key(url: str, params: dict) -> str:
    canonical = url + "|" + "|".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _strip(payload: dict) -> dict:
    stripped = deepcopy(payload)
    stripped.pop("debugOutput", None)
    for itinerary in stripped.get("itineraries", []):
        itinerary.pop("id", None)
        for leg in itinerary.get("legs", []):
            leg.pop("legGeometry", None)
            leg.pop("steps", None)
            leg.pop("intermediateStops", None)
    return stripped


class ResponseCache:
    def __init__(self, mode: str, path: Path):
        if mode not in ("live", "replay"):
            raise ValueError(f"unknown cache mode: {mode!r} (must be 'live' or 'replay')")
        self.mode = mode
        self.path = Path(path)

    async def get(
        self,
        http: httpx.AsyncClient | None,
        url: str,
        params: dict,
        *,
        headers: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        key = _cache_key(url, params)
        file_path = self.path / f"{key}.json"

        if self.mode == "replay":
            if not file_path.exists():
                raise CacheMiss(f"no cached response for {url} {params}")
            return json.loads(file_path.read_text(encoding="utf-8"))

        response = await http.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        self.path.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(_strip(payload), ensure_ascii=False, indent=2), encoding="utf-8")

        return payload
