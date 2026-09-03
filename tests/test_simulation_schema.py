"""Validation for the design-only simulation configuration schema."""

from pathlib import Path

import yaml


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "simulation" / "schema.yaml"
)


def test_simulation_schema_has_required_sections() -> None:
    """The YAML template should parse and expose the planned config structure."""
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))

    required_sections = {
        "simulation",
        "population",
        "financial_variables",
        "loan_property_variables",
        "context_variables",
        "derived_variables",
        "approval_model",
        "scenario_effects",
        "outcomes",
        "output",
    }
    assert required_sections.issubset(schema)


def test_research_parameters_remain_unset() -> None:
    """The schema must not masquerade as a calibrated experiment config."""
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["simulation"]["random_seed"] is None
    assert schema["simulation"]["n_samples"] is None
    assert schema["simulation"]["scenario"] is None
    assert schema["approval_model"]["intercept"] is None
    assert schema["scenario_effects"]["direct"]["log_odds_effects"] is None
    assert schema["scenario_effects"]["upstream"]["group_parameter_shifts"] is None


def test_schema_names_exactly_the_four_core_scenarios() -> None:
    """Scenario choices should stay aligned with the research design."""
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["simulation"]["allowed_scenarios"] == [
        "fair_baseline",
        "direct_discrimination",
        "upstream_inequality",
        "mixed_mechanism",
    ]
