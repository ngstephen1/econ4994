"""Structural tests for the calibrated, design-only simulation configs."""

import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "simulation"
LEVELS = ("mild", "moderate", "strong")


def load_yaml(path: Path) -> dict:
    """Load a YAML mapping from a project-relative configuration file."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def baseline() -> dict:
    return load_yaml(CONFIG_DIR / "baseline.yaml")


def effects() -> dict:
    return load_yaml(CONFIG_DIR / "effect_sizes.yaml")


def schema() -> dict:
    return load_yaml(CONFIG_DIR / "schema.yaml")


def scenario_configs() -> dict[str, dict]:
    return {
        path.stem: load_yaml(path)["scenario"]
        for path in sorted((CONFIG_DIR / "scenarios").glob("*.yaml"))
    }


def test_calibrated_shares_are_valid_probabilities() -> None:
    """Demographic and loan-category shares must each define a distribution."""
    config = baseline()
    share_maps = [
        item["shares"] for item in config["population"]["demographics"].values()
    ]
    share_maps.extend(
        config["loan_property_variables"][name]["shares"]
        for name in ("loan_purpose", "loan_type", "occupancy_type")
    )

    for shares in share_maps:
        assert all(0.0 <= value <= 1.0 for value in shares.values())
        assert abs(sum(shares.values()) - 1.0) < 1e-12

    zero_probability = config["financial_variables"]["existing_monthly_debt"][
        "zero_probability"
    ]
    assert 0.0 <= zero_probability <= 1.0


def test_race_categories_and_focal_comparison_match_schema() -> None:
    """The first experiment must retain the schema races and compare Black/White."""
    config = baseline()
    schema_races = schema()["population"]["demographics"]["race"]["categories"]
    calibrated_races = list(
        config["population"]["demographics"]["race"]["shares"].keys()
    )

    assert calibrated_races == schema_races
    assert config["simulation"]["focal_protected_attribute"] == "race"
    assert config["simulation"]["focal_comparison"] == {
        "reference": "White",
        "comparison": "Black",
    }


def test_direct_effect_levels_are_ordered_and_isolated() -> None:
    """Only Black receives increasingly negative direct log-odds treatments."""
    library = effects()["direct_effects"]["levels"]
    races = schema()["population"]["demographics"]["race"]["categories"]
    black_effects = []

    for level in LEVELS:
        race_effects = library[level]["log_odds_by_race"]
        assert list(race_effects) == races
        assert race_effects["White"] == 0.0
        assert race_effects["Black"] < 0.0
        assert all(
            race_effects[race] == 0.0
            for race in races
            if race not in {"White", "Black"}
        )
        black_effects.append(abs(race_effects["Black"]))

    assert black_effects[0] < black_effects[1] < black_effects[2]


def test_upstream_levels_are_supported_ordered_and_isolated() -> None:
    """Upstream treatments must affect only supported fields and grow by level."""
    config = baseline()
    library = effects()["upstream_effects"]
    races = schema()["population"]["demographics"]["race"]["categories"]
    supported = set(config["financial_variables"])

    assert set(library["affected_variables"]).issubset(supported)
    for variable in library["affected_variables"]:
        magnitudes = []
        for level in LEVELS:
            treatment = library["levels"][level][variable]
            shifts = treatment["additive_shift_by_race"]
            assert list(shifts) == races
            assert shifts["White"] == 0.0
            assert shifts["Black"] < 0.0
            assert all(
                shifts[race] == 0.0
                for race in races
                if race not in {"White", "Black"}
            )
            magnitudes.append(abs(shifts["Black"]))
            if treatment["shifted_parameter"] == "log_location":
                assert math.isclose(
                    math.exp(shifts["Black"]),
                    treatment["conditional_black_multiplier"],
                    rel_tol=1e-5,
                )
        assert magnitudes[0] < magnitudes[1] < magnitudes[2]


def test_scenario_switch_matrix_is_clean() -> None:
    """Each scenario must activate exactly its intended mechanisms."""
    scenarios = scenario_configs()
    expected = {
        "fair_baseline": (False, False),
        "direct_discrimination": (False, True),
        "upstream_inequality": (True, False),
        "mixed_mechanism": (True, True),
    }

    assert set(scenarios) == set(expected)
    for name, (upstream, direct) in expected.items():
        scenario = scenarios[name]
        assert scenario["name"] == name
        assert scenario["upstream"]["enabled"] is upstream
        assert scenario["direct"]["enabled"] is direct
        for mechanism in ("upstream", "direct"):
            effect_level = scenario[mechanism]["effect_level"]
            if scenario[mechanism]["enabled"]:
                assert effect_level in LEVELS
            else:
                assert effect_level is None


def test_bounds_target_and_v1_outcome_policy_are_valid() -> None:
    """Bounds must be ordered, the target valid, and denial reasons disabled."""
    config = baseline()

    def inspect_bounds(value: object) -> None:
        if isinstance(value, dict):
            if "minimum" in value and "maximum" in value:
                assert value["minimum"] < value["maximum"]
            for nested in value.values():
                inspect_bounds(nested)
        elif isinstance(value, list):
            for nested in value:
                inspect_bounds(nested)

    inspect_bounds(config)
    target = config["approval_model"]["intercept"]["target_mean_probability"]
    assert 0.0 < target < 1.0
    assert config["outcomes"]["denial_reason"]["enabled"] is False
    assert config["outcomes"]["denial_reason"]["fill_value"] is None


def test_payment_factor_and_derived_identities_are_coherent() -> None:
    """The DTI intermediate and persisted ratio identities must be explicit."""
    config = baseline()
    payment = config["internal_housing_payment"]
    monthly_rate = payment["annual_interest_rate"] / 12
    months = payment["amortization_months"]
    expected_factor = (
        monthly_rate
        * (1 + monthly_rate) ** months
        / ((1 + monthly_rate) ** months - 1)
    )

    assert payment["persist"] is False
    assert math.isclose(
        payment["monthly_principal_interest_factor"],
        expected_factor,
        rel_tol=1e-6,
    )
    loan = config["loan_property_variables"]
    assert loan["loan_amount"]["construction"] == (
        "property_value * loan_to_value_ratio"
    )
    assert loan["income_to_loan_ratio"]["construction"] == (
        "annual_income / loan_amount"
    )


def test_config_checks_do_not_create_generated_data() -> None:
    """Reading every config must leave generated-data directories unchanged."""
    data_root = ROOT / "data"
    before = sorted(path.relative_to(data_root) for path in data_root.rglob("*"))

    baseline()
    effects()
    schema()
    scenario_configs()

    after = sorted(path.relative_to(data_root) for path in data_root.rglob("*"))
    assert after == before
