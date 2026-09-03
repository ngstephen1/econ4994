"""Deterministic safeguards and behavioral tests for the ML benchmark."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fair_lending.models.evaluation import evaluate_predictions, true_probability_recovery
from fair_lending.models.features import (
    FORBIDDEN_RACE_BLIND_FIELDS,
    PROXY_FIELDS,
    RACE_AWARE,
    RACE_BLIND,
    get_feature_spec,
    select_features,
)
from fair_lending.models.group_audit import (
    black_white_disparities,
    group_prediction_metrics,
)
from fair_lending.models.training import (
    build_pipeline,
    candidate_parameters,
    deterministic_split,
    tune_on_validation,
)
from fair_lending.simulation.generator import generate_synthetic_data


INTERCEPT = 2.0362202310934663
SEED = 4_994
BEHAVIOR_N = 40_000
FAIR_GAP_TOLERANCE = 0.02
DIRECT_INCREMENT_TOLERANCE = 0.015
UPSTREAM_GAP_MINIMUM = 0.03


@lru_cache(maxsize=None)
def scenario_split(scenario: str):
    data, _ = generate_synthetic_data(
        scenario, "moderate", BEHAVIOR_N, SEED, intercept=INTERCEPT
    )
    return deterministic_split(data, SEED)


@lru_cache(maxsize=None)
def logistic_result(scenario: str, regime: str):
    split = scenario_split(scenario)
    selected = tune_on_validation(split, regime, "logistic_regression", SEED)
    probability = selected["pipeline"].predict_proba(
        select_features(split.test, regime)
    )[:, 1]
    audit = group_prediction_metrics(split.test, probability)
    return selected, probability, black_white_disparities(audit)


def test_feature_regimes_enforce_race_and_leakage_policy() -> None:
    blind = get_feature_spec(RACE_BLIND)
    aware = get_feature_spec(RACE_AWARE)
    assert "race" not in blind.columns
    assert "race" in aware.columns
    assert set(aware.columns) == set(blind.columns) | {"race"}
    assert not FORBIDDEN_RACE_BLIND_FIELDS.intersection(blind.columns)
    assert not PROXY_FIELDS.intersection(blind.columns)
    assert "approval_probability_true" not in aware.columns
    assert "denial_reason" not in aware.columns
    assert "application_id" not in aware.columns


def test_split_is_deterministic_exact_and_non_overlapping() -> None:
    data, _ = generate_synthetic_data(
        "fair_baseline", "moderate", 1_000, SEED, intercept=INTERCEPT
    )
    first = deterministic_split(data, SEED)
    second = deterministic_split(data, SEED)
    assert (len(first.train), len(first.validation), len(first.test)) == (600, 200, 200)
    first_ids = [set(part["application_id"]) for part in first.__dict__.values()]
    assert first_ids[0].isdisjoint(first_ids[1])
    assert first_ids[0].isdisjoint(first_ids[2])
    assert first_ids[1].isdisjoint(first_ids[2])
    for name in ("train", "validation", "test"):
        assert getattr(first, name)["application_id"].equals(
            getattr(second, name)["application_id"]
        )


def test_shared_population_gets_same_split_assignments_across_scenarios() -> None:
    fair = scenario_split("fair_baseline")
    mixed = scenario_split("mixed_mechanism")
    for name in ("train", "validation", "test"):
        assert getattr(fair, name)["application_id"].equals(
            getattr(mixed, name)["application_id"]
        )


def test_preprocessing_is_inside_fitted_sklearn_pipeline() -> None:
    selected, _, _ = logistic_result("fair_baseline", RACE_BLIND)
    pipeline = selected["pipeline"]
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocess", "classifier"]
    assert hasattr(pipeline.named_steps["preprocess"], "transformers_")


def test_fixed_seed_model_probabilities_are_deterministic_and_valid() -> None:
    split = scenario_split("fair_baseline")
    first = tune_on_validation(split, RACE_BLIND, "logistic_regression", SEED)
    second = tune_on_validation(split, RACE_BLIND, "logistic_regression", SEED)
    features = select_features(split.test, RACE_BLIND)
    first_probability = first["pipeline"].predict_proba(features)[:, 1]
    second_probability = second["pipeline"].predict_proba(features)[:, 1]
    np.testing.assert_array_equal(first_probability, second_probability)
    assert ((0.0 <= first_probability) & (first_probability <= 1.0)).all()


def test_all_model_families_are_deterministic_for_fixed_random_state() -> None:
    data, _ = generate_synthetic_data(
        "fair_baseline", "moderate", 1_500, SEED, intercept=INTERCEPT
    )
    split = deterministic_split(data, SEED)
    spec = get_feature_spec(RACE_BLIND)
    x_train = select_features(split.train, RACE_BLIND)
    y_train = split.train["approved"]
    x_test = select_features(split.test, RACE_BLIND)
    for model_name in (
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ):
        parameters = candidate_parameters(model_name)[0]
        first = build_pipeline(model_name, parameters, spec, SEED).fit(
            x_train, y_train
        )
        second = build_pipeline(model_name, parameters, spec, SEED).fit(
            x_train, y_train
        )
        np.testing.assert_allclose(
            first.predict_proba(x_test),
            second.predict_proba(x_test),
            rtol=0.0,
            atol=1e-15,
        )


def test_test_partition_cannot_change_hyperparameter_selection() -> None:
    split = scenario_split("fair_baseline")
    changed_test = split.test.copy()
    changed_test["approved"] = 1 - changed_test["approved"]
    altered = type(split)(split.train, split.validation, changed_test)
    original = tune_on_validation(split, RACE_BLIND, "logistic_regression", SEED)
    repeated = tune_on_validation(altered, RACE_BLIND, "logistic_regression", SEED)
    assert original["selected_parameters"] == repeated["selected_parameters"]
    assert original["selection_records"] == repeated["selection_records"]


def test_fair_baseline_race_blind_gap_is_small() -> None:
    _, _, gaps = logistic_result("fair_baseline", RACE_BLIND)
    assert abs(gaps["predicted_probability_gap"]) < FAIR_GAP_TOLERANCE


def test_direct_race_aware_model_reproduces_more_negative_disparity() -> None:
    _, _, blind = logistic_result("direct_discrimination", RACE_BLIND)
    _, _, aware = logistic_result("direct_discrimination", RACE_AWARE)
    assert aware["predicted_probability_gap"] < (
        blind["predicted_probability_gap"] - DIRECT_INCREMENT_TOLERANCE
    )


def test_upstream_race_blind_model_has_meaningful_predicted_gap() -> None:
    _, _, gaps = logistic_result("upstream_inequality", RACE_BLIND)
    assert gaps["predicted_probability_gap"] < -UPSTREAM_GAP_MINIMUM


def test_group_metrics_match_hand_checked_example() -> None:
    test = pd.DataFrame(
        {
            "race": ["White", "White", "Black", "Black"],
            "approved": [1, 0, 1, 0],
            "approval_probability_true": [0.8, 0.3, 0.6, 0.2],
        }
    )
    probability = np.array([0.9, 0.4, 0.4, 0.1])
    metrics = group_prediction_metrics(test, probability).set_index("race")
    assert metrics.loc["White", "accuracy"] == 1.0
    assert metrics.loc["White", "tpr"] == 1.0
    assert metrics.loc["White", "fpr"] == 0.0
    assert metrics.loc["Black", "accuracy"] == 0.5
    assert metrics.loc["Black", "fnr"] == 1.0
    gaps = black_white_disparities(metrics.reset_index())
    assert gaps["observed_black_white_gap"] == 0.0
    assert gaps["predicted_probability_gap"] == pytest.approx(-0.4)
    assert gaps["predicted_approval_gap"] == pytest.approx(-0.5)


def test_true_probability_recovery_matches_hand_calculation() -> None:
    predicted = np.array([0.2, 0.8])
    truth = np.array([0.1, 0.6])
    result = true_probability_recovery(predicted, truth)
    assert result["true_probability_mae"] == pytest.approx(0.15)
    assert result["true_probability_squared_error"] == pytest.approx(0.025)
    assert result["true_probability_rmse"] == pytest.approx(np.sqrt(0.025))
    assert result["mean_prediction_minus_true_probability"] == pytest.approx(0.15)


def test_prediction_evaluation_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        evaluate_predictions(
            np.array([0, 1]), np.array([-0.1, 1.1]), np.array([0.2, 0.8])
        )
