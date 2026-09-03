"""Reproduce the Prompt 5 DTI calibration diagnosis and candidate comparison."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from fair_lending.simulation.config import PROJECT_ROOT, resolve_simulation_config
from fair_lending.simulation.diagnostics import (
    calculate_dti_components,
    component_distribution_table,
    dti_correlation_table,
    high_dti_comparison_table,
    summarize_diagnostic_run,
)
from fair_lending.simulation.population import create_random_streams, generate_population


N_ROWS = 100_000
SEED = 4_994
PREVIOUS_INTERCEPT = 2.2422267822548747
ORIGINAL_PROPERTY_INCOME_ELASTICITY = 0.55
ORIGINAL_PROPERTY_RESIDUAL_SD = 0.30
SELECTED_PROPERTY_INCOME_ELASTICITY = 0.90
SELECTED_PROPERTY_RESIDUAL_SD = 0.18

# A deliberately small, interpretable sequence rather than a tuning grid.
CANDIDATES = [
    ("original", 0.55, 0.30, 0.35),
    ("property_residual_sd_0.25", 0.55, 0.25, 0.35),
    ("property_income_elasticity_0.70", 0.70, 0.30, 0.35),
    ("property_income_elasticity_1.00", 1.00, 0.30, 0.35),
    ("existing_debt_income_elasticity_1.00", 0.55, 0.30, 1.00),
    ("property_0.90_debt_1.00", 0.90, 0.30, 1.00),
    ("property_0.90_residual_0.25", 0.90, 0.25, 0.35),
    ("property_0.85_residual_0.20", 0.85, 0.20, 0.35),
    ("property_0.90_residual_0.20", 0.90, 0.20, 0.35),
    ("selected_property_0.90_residual_0.18", 0.90, 0.18, 0.35),
]


def candidate_config(
    base: dict, property_income_elasticity: float, property_residual_sd: float,
    debt_income_elasticity: float,
) -> dict:
    config = copy.deepcopy(base)
    property_spec = config["loan_property_variables"]["property_value"]
    property_spec["income_elasticity"] = property_income_elasticity
    property_spec["log_residual_standard_deviation"] = property_residual_sd
    config["financial_variables"]["existing_monthly_debt"][
        "income_elasticity"
    ] = debt_income_elasticity
    return config


def generate(config: dict) -> pd.DataFrame:
    streams, _ = create_random_streams(SEED)
    data, _ = generate_population(config, streams)
    return data


def main() -> None:
    base = resolve_simulation_config(
        "fair_baseline", "moderate", n_rows=N_ROWS, seed=SEED
    )
    output = PROJECT_ROOT / "results" / "tables"
    output.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    generated: dict[str, tuple[pd.DataFrame, dict]] = {}
    for name, property_elasticity, property_residual, debt_elasticity in CANDIDATES:
        config = candidate_config(
            base, property_elasticity, property_residual, debt_elasticity
        )
        data = generate(config)
        generated[name] = (data, config)
        candidate_rows.append(
            summarize_diagnostic_run(
                data, config, candidate=name, intercept=PREVIOUS_INTERCEPT
            )
        )

    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(output / "dti_diagnostic_candidates.csv", index=False)
    candidates.loc[candidates["candidate"] == "original"].to_csv(
        output / "dti_diagnostic_before.csv", index=False
    )
    candidates.loc[
        candidates["candidate"] == "selected_property_0.90_residual_0.18"
    ].to_csv(output / "dti_diagnostic_after.csv", index=False)

    before_data, before_config = generated["original"]
    after_data, after_config = generated["selected_property_0.90_residual_0.18"]
    pd.concat(
        [
            high_dti_comparison_table(
                before_data, before_config, calibration_label="before"
            ),
            high_dti_comparison_table(
                after_data, after_config, calibration_label="after"
            ),
        ],
        ignore_index=True,
    ).to_csv(output / "high_dti_comparison.csv", index=False)
    components = pd.concat(
        [
            component_distribution_table(
                before_data,
                calculate_dti_components(before_data, before_config),
                calibration_label="before",
            ),
            component_distribution_table(
                after_data,
                calculate_dti_components(after_data, after_config),
                calibration_label="after",
            ),
        ],
        ignore_index=True,
    )
    components.to_csv(output / "dti_component_distributions.csv", index=False)
    pd.concat(
        [
            dti_correlation_table(
                before_data, before_config, calibration_label="before"
            ),
            dti_correlation_table(after_data, after_config, calibration_label="after"),
        ],
        ignore_index=True,
    ).to_csv(output / "dti_correlations.csv", index=False)


if __name__ == "__main__":
    main()
