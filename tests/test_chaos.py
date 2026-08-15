"""Phase 2a: chaos directive parsing, scenario lookup, and YAML loading."""

import pytest

from journey.domain import Leg
from journey.sources.chaos import (
    ChaosScenario,
    EmptyResponse,
    ErrorResponse,
    Ok,
    PriceShift,
    Slow,
    Timeout,
    load_scenario,
    parse_directive,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ok", Ok()),
        ("timeout(3)", Timeout(3.0)),
        ("error", ErrorResponse()),
        ("empty", EmptyResponse()),
        ("price_shift(1.15)", PriceShift(1.15)),
        ("slow(5)", Slow(5.0)),
    ],
)
def test_parse_directive(text, expected):
    assert parse_directive(text) == expected


def test_parse_directive_rejects_unrecognised_text():
    with pytest.raises(ValueError):
        parse_directive("not-a-real-directive")


def test_scenario_directive_for_defaults_to_ok_when_unconfigured():
    scenario = ChaosScenario(name="all-clear", directives={})
    leg = Leg(origin="London", destination="Brussels")

    assert scenario.directive_for("stub-flight", leg) == Ok()


def test_scenario_directive_for_returns_configured_directive():
    leg = Leg(origin="Cologne", destination="Munich")
    scenario = ChaosScenario(
        name="flight_timeout",
        directives={("stub-flight", "Cologne", "Munich"): Timeout(3.0)},
    )

    assert scenario.directive_for("stub-flight", leg) == Timeout(3.0)


def test_load_scenario_from_yaml(tmp_path):
    yaml_path = tmp_path / "flight_timeout.yaml"
    yaml_path.write_text(
        "name: flight_timeout\n"
        "directives:\n"
        "  - source: stub-flight\n"
        "    origin: Cologne\n"
        "    destination: Munich\n"
        '    directive: "timeout(3)"\n'
    )

    scenario = load_scenario(yaml_path)

    assert scenario.name == "flight_timeout"
    leg = Leg(origin="Cologne", destination="Munich")
    assert scenario.directive_for("stub-flight", leg) == Timeout(3.0)
