"""Deterministic tests for dashboard support logic (not UI pixel tests)."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fair_lending.dashboard.data import (
    DashboardArtifactError,
    build_mechanism_matrix,
    filter_benchmark_row,
    load_result_table,
)
from fair_lending.dashboard.formatting import format_currency, format_percentage_points
from fair_lending.dashboard.simulation_service import (
    CustomTreatments,
    SimulationRequest,
    export_dataset_csv,
    export_dataset_parquet,
    generate_dashboard_simulation,
    resolve_dashboard_config,
    summarize_simulation,
)
from fair_lending.simulation.config import CONFIG_DIRECTORY, PROJECT_ROOT
from fair_lending.simulation.generator import OUTPUT_COLUMNS


INTERCEPT = 2.0362202310934663


def request(
    scenario: str = "fair_baseline", seed: int = 4_994
) -> SimulationRequest:
    return SimulationRequest(scenario, "moderate", 500, seed)


def test_result_loader_finds_declared_table(tmp_path: Path) -> None:
    table_directory = tmp_path / "results" / "tables"
    table_directory.mkdir(parents=True)
    pd.DataFrame({"scenario": ["fair_baseline"], "value": [1.0]}).to_csv(
        table_directory / "race_approval_gaps.csv", index=False
    )
    loaded = load_result_table("race_approval_gaps", tmp_path)
    assert loaded.to_dict("records") == [{"scenario": "fair_baseline", "value": 1.0}]


def test_missing_result_has_controlled_recovery_message(tmp_path: Path) -> None:
    with pytest.raises(DashboardArtifactError, match="run_statistical_recovery.py first"):
        load_result_table("statistical_recovery", tmp_path)


def test_dashboard_generation_is_deterministic() -> None:
    first, _ = generate_dashboard_simulation(request(), intercept=INTERCEPT)
    second, _ = generate_dashboard_simulation(request(), intercept=INTERCEPT)
    assert_frame_equal(first, second, check_exact=True)


def test_changing_seed_changes_dashboard_sample() -> None:
    first, _ = generate_dashboard_simulation(request(seed=7), intercept=INTERCEPT)
    second, _ = generate_dashboard_simulation(request(seed=8), intercept=INTERCEPT)
    assert not first.drop(columns="application_id").equals(
        second.drop(columns="application_id")
    )


def test_scenario_changes_only_the_declared_treatment_switches() -> None:
    fair = resolve_dashboard_config(request("fair_baseline"))
    direct = resolve_dashboard_config(request("direct_discrimination"))
    upstream = resolve_dashboard_config(request("upstream_inequality"))
    assert fair["scenario_effects"]["direct"]["enabled"] is False
    assert fair["scenario_effects"]["upstream"]["enabled"] is False
    assert direct["scenario_effects"]["direct"]["enabled"] is True
    assert direct["scenario_effects"]["upstream"]["enabled"] is False
    assert upstream["scenario_effects"]["direct"]["enabled"] is False
    assert upstream["scenario_effects"]["upstream"]["enabled"] is True


def test_custom_overrides_do_not_mutate_yaml() -> None:
    yaml_paths = sorted(CONFIG_DIRECTORY.rglob("*.yaml"))
    before = {path: path.read_bytes() for path in yaml_paths}
    custom = CustomTreatments(-0.40, 0.82, -44.0, 0.61)
    resolved = resolve_dashboard_config(
        SimulationRequest("fair_baseline", "moderate", 200, 9, custom)
    )
    after = {path: path.read_bytes() for path in yaml_paths}
    assert before == after
    assert resolved["scenario_effects"]["direct"]["log_odds_by_race"]["Black"] == -0.40
    assert resolved["scenario_effects"]["upstream"]["treatments"]["credit_score"]["additive_shift_by_race"]["Black"] == -44.0


def test_custom_treatments_preserve_frozen_intercept() -> None:
    custom = CustomTreatments(-0.50, 0.70, -75.0, 0.30)
    _, metadata = generate_dashboard_simulation(
        SimulationRequest("mixed_mechanism", "strong", 300, 4_994, custom),
        intercept=INTERCEPT,
    )
    assert metadata["intercept"] == INTERCEPT
    assert metadata["intercept_calibration_source"] == "frozen_fair_baseline_artifact"


def test_summary_black_white_gap_matches_hand_calculation() -> None:
    data, metadata = generate_dashboard_simulation(request(), intercept=INTERCEPT)
    toy = data.iloc[:4].copy()
    toy["race"] = ["White", "White", "Black", "Black"]
    toy["approved"] = [1, 1, 0, 1]
    summary = summarize_simulation(toy, metadata)
    assert summary["white_approval_rate"] == 1.0
    assert summary["black_approval_rate"] == 0.5
    assert summary["black_white_approval_gap"] == -0.5


def test_dashboard_unit_formatting_is_explicit() -> None:
    assert format_percentage_points(-0.035) == "-3.50 pp"
    assert format_percentage_points(0.0) == "+0.00 pp"
    assert format_currency(123456.4) == "$123,456"


def test_download_exports_preserve_the_24_field_schema() -> None:
    data, _ = generate_dashboard_simulation(request(), intercept=INTERCEPT)
    csv = pd.read_csv(BytesIO(export_dataset_csv(data)))
    parquet = pd.read_parquet(BytesIO(export_dataset_parquet(data)))
    assert list(csv.columns) == OUTPUT_COLUMNS
    assert list(parquet.columns) == OUTPUT_COLUMNS
    assert len(csv) == len(data) == len(parquet)


def test_benchmark_filter_returns_the_exact_saved_row() -> None:
    path = PROJECT_ROOT / "results" / "tables" / "ml_benchmark.csv"
    if not path.exists():
        pytest.skip("Generated Prompt 7 result table is intentionally Git-ignored")
    benchmark = load_result_table("ml_benchmark")
    expected = benchmark.loc[
        benchmark["scenario"].eq("direct_discrimination")
        & benchmark["model"].eq("logistic_regression")
        & benchmark["feature_regime"].eq("race_blind")
    ].iloc[0]
    selected = filter_benchmark_row(
        benchmark, "direct_discrimination", "logistic_regression", "race_blind"
    )
    assert selected["predicted_probability_gap"] == expected["predicted_probability_gap"]
    assert selected["roc_auc"] == expected["roc_auc"]


def test_mechanism_matrix_matches_saved_statistical_and_ml_values() -> None:
    statistical_path = PROJECT_ROOT / "results" / "tables" / "statistical_recovery.csv"
    ml_path = PROJECT_ROOT / "results" / "tables" / "ml_benchmark.csv"
    if not statistical_path.exists() or not ml_path.exists():
        pytest.skip("Generated Prompt 6/7 result tables are intentionally Git-ignored")
    statistical = load_result_table("statistical_recovery")
    ml = load_result_table("ml_benchmark")
    matrix = build_mechanism_matrix(statistical, ml).set_index("scenario")
    source_stat = statistical.loc[
        statistical["scenario"].eq("mixed_mechanism")
        & statistical["model"].eq("model_2")
    ].iloc[0]
    source_ml = ml.loc[
        ml["scenario"].eq("mixed_mechanism")
        & ml["model"].eq("logistic_regression")
        & ml["feature_regime"].eq("race_aware_sensitivity")
    ].iloc[0]
    assert matrix.loc["mixed_mechanism", "observed_gap"] == source_stat["raw_approval_gap"]
    assert matrix.loc["mixed_mechanism", "race_aware_ml_gap"] == source_ml["predicted_probability_gap"]
    assert bool(matrix.loc["mixed_mechanism", "upstream_differences"])
    assert bool(matrix.loc["mixed_mechanism", "direct_race_effect"])


def test_custom_treatment_probabilities_remain_valid() -> None:
    custom = CustomTreatments(-0.50, 0.70, -75.0, 0.30)
    data, _ = generate_dashboard_simulation(
        SimulationRequest("fair_baseline", "moderate", 1_000, 22, custom),
        intercept=INTERCEPT,
    )
    assert np.isfinite(data["approval_probability_true"]).all()
    assert data["approval_probability_true"].between(0.0, 1.0).all()
