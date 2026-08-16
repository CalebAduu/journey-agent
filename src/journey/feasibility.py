"""Feasibility beliefs (§5): physical and geometric priors for whether a
mode is even worth attempting on a leg.

Only the physical and geometric layers are built here - nothing in this
project yet needs the LLM-prior/belief-update layers (§5 points 3-4).
The asymmetry §5 argues for (a false "infeasible" silently eliminates a
valid option forever; a false "feasible" costs one wasted call and
self-corrects) is exactly why those layers default to just calling: only
entries in _PHYSICAL_EXCLUSIONS and the two geometric thresholds below
ever produce NOT_APPLICABLE. Everything else is attempted.
"""

from journey.domain import Leg

# Hand-specified, small, stable (§5: ~10 entries max). Both directions of
# a pair are listed explicitly - order isn't assumed.
_PHYSICAL_EXCLUSIONS: dict[tuple[str, str, str], str] = {
    ("Sheffield", "London", "flight"): "Doncaster Sheffield Airport closed",
    ("London", "Sheffield", "flight"): "Doncaster Sheffield Airport closed",
}

# Computed, not tabulated - generalises to any city pair (§5).
GEOMETRIC_FLIGHT_MIN_KM = 150
GEOMETRIC_COACH_MAX_KM = 1500


class RouteFeasibility:
    def candidate_modes(self, leg: Leg) -> list[str]:
        return ["coach", "rail", "flight"]

    def reason_if_not_applicable(self, leg: Leg, mode: str) -> str | None:
        physical_reason = _PHYSICAL_EXCLUSIONS.get((leg.origin, leg.destination, mode))
        if physical_reason is not None:
            return f"{physical_reason} (physical)"

        if leg.distance_km is None:
            return None

        if mode == "flight" and leg.distance_km < GEOMETRIC_FLIGHT_MIN_KM:
            return (
                f"~{leg.distance_km:.0f}km, below the geometric "
                f"<{GEOMETRIC_FLIGHT_MIN_KM}km flight-implausible threshold"
            )
        if mode == "coach" and leg.distance_km > GEOMETRIC_COACH_MAX_KM:
            return (
                f"~{leg.distance_km:.0f}km, beyond the geometric "
                f">{GEOMETRIC_COACH_MAX_KM}km coach-implausible threshold"
            )

        return None
