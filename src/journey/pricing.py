"""Cost priors and distance-derived cost intervals.

Every constant used anywhere in cost estimation lives here - no magic
numbers elsewhere in the codebase.
"""

import math

from journey.domain import Money

# Mean earth radius used by haversine_km.
EARTH_RADIUS_KM = 6371.0

# Hand-specified priors — NOT derived from fare data.
# Wide on purpose: UK rail on the same train varies ~3x by booking horizon.
# In production these would come from historical per-route fare observations.
COST_PER_KM = {
    "rail":   (0.12, 0.35),   # £/km low, high
    "coach":  (0.04, 0.09),
    "flight": (0.08, 0.40),
}
MIN_FARE = {"rail": Money(500), "coach": Money(300), "flight": Money(2500)}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def infer_cost(mode: str, distance_km: float) -> tuple[Money, Money]:
    """A hand-specified cost interval for `mode` over `distance_km`.

    MIN_FARE floors the low bound - on a short leg the per-km rate alone
    can price below what any real fare would charge (e.g. coach over
    Berlin-Potsdam's ~25km). If flooring the low bound would push it past
    the per-km high bound, the high bound is raised to match, so the
    interval never inverts.
    """
    low_rate, high_rate = COST_PER_KM[mode]
    low = Money(round(low_rate * distance_km * 100))
    high = Money(round(high_rate * distance_km * 100))

    floor = MIN_FARE[mode]
    if low.minor_units < floor.minor_units:
        low = floor
    if high.minor_units < low.minor_units:
        high = low

    return (low, high)


# Fares ratchet up toward departure as advance quotas sell out, so a stale
# quote is a lower bound rather than a symmetric estimate. Rate is assumed,
# not measured — see README.
STALENESS_DRIFT_PER_HOUR = 0.004      # ~10% over 24h
MAX_CACHE_AGE_HOURS = 72              # beyond this, don't offer UseCached


def drift_cached(price: Money, age_hours: float) -> tuple[Money, Money]:
    """Asymmetric interval: cached price is the floor, drift gives the ceiling."""
    high = Money(round(price.minor_units * (1 + STALENESS_DRIFT_PER_HOUR * age_hours)))
    return price, high
