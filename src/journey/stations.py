"""Station IDs, resolved via each source's own geocoder.

Namespaces do NOT match across sources: Transitous uses feed-prefixed
GTFS IDs, VBB/BVG use their own numeric IDs, DB uses EVA numbers.
Never share an ID between sources.
"""

# Transitous (api.transitous.org)
SHEFFIELD_RAIL  = "gb-great-britain_910GSHEFFLD"   # LONG_DISTANCE, REGIONAL_RAIL, BUS
SHEFFIELD_COACH = "gb-flixbus_9b69e519-3ecb-11ea-8017-02437075395e"  # COACH, BUS
LONDON_EUSTON   = "gb-great-britain_910GEUSTON"    # LONG_DISTANCE, NIGHT_RAIL, REGIONAL_RAIL

# VBB / BVG namespace
BERLIN_HBF  = "900003201"    # S+U Berlin Hauptbahnhof
POTSDAM_HBF = "900230999"    # S Potsdam Hauptbahnhof