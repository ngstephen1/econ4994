"""Deterministic tests for the Version 2 calibrated applicant population."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from fair_lending.economic_lending.applicants import generate_applicant_population
from fair_lending.economic_lending.config import create_random_streams, load_economic_config
from fair_lending.economic_lending.contracts import (
    make_fixed_loan_options,
    make_requested_loan_options,
    payment_schedule,
)
from fair_lending.economic_lending.outcomes import simulate_repayment_outcomes
from fair_lending.economic_lending.population import generate_economic_population
from fair_lending.economic_lending.profit import (
    expected_profit,
    expected_profit_from_options,
)
from fair_lending.economic_lending.repayment import (
    TRUE_RISK_PREDICTORS,
    true_repayment_probabilities,
    true_repayment_probabilities_from_score,
)
from fair_lending.economic_lending.schema import APPLICANTS_COLUMNS, BaselineEconomicParameters
from fair_lending.economic_lending.applicants import tiny_fixture_applicants


def _config():
    return load_economic_config()


def _result(n: int = 500, seed: int = 4994):
    return generate_economic_population(n, _config(), seed=seed)


def test_valid_group_categories_and_exact_cohort_partition() -> None:
    result = _result(100)
    assert set(result.applicants["group"].unique()) <= {"A", "B"}
    assert result.applicants["cohort"].value_counts().to_dict() == {
        "historical_train": 60,
        "historical_validation": 20,
        "evaluation": 20,
    }
    assert result.applicants["applicant_id"].is_unique


def test_group_share_does_not_change_financial_generation() -> None:
    config_a = _config()
    config_b = copy.deepcopy(config_a)
    config_a["population"]["group_shares"] = {"A": 1.0, "B": 0.0}
    config_b["population"]["group_shares"] = {"A": 0.0, "B": 1.0}
    streams_a, _ = create_random_streams(91, config_a["randomness"]["stream_names"])
    streams_b, _ = create_random_streams(91, config_b["randomness"]["stream_names"])
    applicants_a, _ = generate_applicant_population(200, config_a, streams_a)
    applicants_b, _ = generate_applicant_population(200, config_b, streams_b)
    pd.testing.assert_frame_equal(
        applicants_a.drop(columns="group"), applicants_b.drop(columns="group")
    )


def test_age_employment_and_monetary_constraints() -> None:
    result = _result()
    applicants = result.applicants
    assert (applicants["employment_years"] >= 0).all()
    assert (applicants["employment_years"] <= applicants["age_years"] - 18).all()
    monetary = [
        "annual_income",
        "liquid_assets",
        "existing_monthly_debt",
        "property_value",
        "requested_loan_amount",
    ]
    assert (applicants[monetary] >= 0).all().all()


def test_credit_and_requested_loan_bounds() -> None:
    config = _config()
    result = generate_economic_population(500, config)
    credit = config["population"]["credit_score"]
    ltv = config["population"]["requested_ltv"]
    assert result.applicants["credit_score"].between(
        credit["minimum"], credit["maximum"]
    ).all()
    ratio = result.applicants["requested_loan_amount"] / result.applicants["property_value"]
    assert ratio.between(ltv["minimum"] - 1e-6, ltv["maximum"] + 1e-6).all()


def test_dti_and_ltv_diagnostics_match_declared_arithmetic() -> None:
    result = _result()
    joined = result.applicants.merge(result.loan_options, on="applicant_id")
    expected_ltv = joined["requested_loan_amount"] / joined["property_value"]
    expected_dti = (
        joined["existing_monthly_debt"] + joined["first_period_payment"]
    ) / (joined["annual_income"] / 12.0)
    np.testing.assert_allclose(joined["requested_ltv"], expected_ltv)
    np.testing.assert_allclose(joined["first_period_dti"], expected_dti)


def test_truth_probabilities_and_term_identity() -> None:
    result = _result()
    truth = result.simulation_truth.merge(
        result.loan_options[["applicant_id", "term_periods"]], on="applicant_id"
    )
    assert truth["repayment_probability_per_period_true"].between(0, 1).all()
    assert truth["full_repayment_probability_true"].between(0, 1).all()
    np.testing.assert_allclose(
        truth["full_repayment_probability_true"],
        truth["repayment_probability_per_period_true"] ** truth["term_periods"],
    )


def test_group_and_identifiers_are_excluded_from_true_risk_predictors() -> None:
    assert "group" not in TRUE_RISK_PREDICTORS
    assert "cohort" not in TRUE_RISK_PREDICTORS
    assert "applicant_id" not in TRUE_RISK_PREDICTORS
    assert not any("true" in column for column in APPLICANTS_COLUMNS)


def test_same_seed_reproduces_all_four_outputs() -> None:
    first = _result(100, 123)
    second = _result(100, 123)
    for name in ("applicants", "loan_options", "simulation_truth", "loan_outcomes"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))


def test_different_seed_changes_population_and_outcomes() -> None:
    first = _result(100, 123)
    second = _result(100, 124)
    assert not first.applicants.equals(second.applicants)
    assert not first.loan_outcomes.equals(second.loan_outcomes)


class _HalfDrawRng:
    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def random(self, size: int) -> np.ndarray:
        self.call_sizes.append(size)
        return np.full(size, 0.5)


def test_repayment_stops_after_first_default_and_completed_receipts_are_full() -> None:
    applicants = tiny_fixture_applicants().head(2)
    params = BaselineEconomicParameters(term_periods=3)
    loans = make_fixed_loan_options(applicants, params)
    truth = pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"],
            "repayment_probability_per_period_true": [0.0, 1.0],
            "full_repayment_probability_true": [0.0, 1.0],
        }
    )
    rng = _HalfDrawRng()
    outcomes = simulate_repayment_outcomes(loans, truth, rng)  # type: ignore[arg-type]
    assert rng.call_sizes == [2, 1, 1]
    first = outcomes.set_index("applicant_id").loc[1]
    second = outcomes.set_index("applicant_id").loc[2]
    assert first["default_period"] == 1
    assert first["realized_total_receipts"] == 0
    assert bool(second["completed_all_payments"])
    promised = params.requested_principal * (
        1 + params.periodic_interest_rate * params.term_periods
    )
    assert second["realized_total_receipts"] == pytest.approx(promised)
    assert second["realized_profit"] == pytest.approx(
        promised - params.requested_principal - params.transaction_cost
    )


def test_schedule_free_expected_profit_matches_period_schedule_accounting() -> None:
    applicants = tiny_fixture_applicants()
    loans = make_fixed_loan_options(applicants)
    truth = true_repayment_probabilities_from_score(applicants, loans)
    scheduled = expected_profit(loans, payment_schedule(loans), truth)
    schedule_free = expected_profit_from_options(loans, truth)
    pd.testing.assert_frame_equal(scheduled, schedule_free)


def test_group_flip_does_not_change_true_risk() -> None:
    config = _config()
    result = generate_economic_population(200, config)
    flipped = result.applicants.copy()
    flipped["group"] = flipped["group"].map({"A": "B", "B": "A"})
    original = true_repayment_probabilities(
        result.applicants, result.loan_options, config["true_risk"]
    )
    changed = true_repayment_probabilities(
        flipped, result.loan_options, config["true_risk"]
    )
    pd.testing.assert_frame_equal(original, changed)


def test_validation_report_matches_source_calculations() -> None:
    result = _result(250)
    report = result.validation_report
    assert report["population"]["total_rows"] == 250
    assert report["risk"]["mean_rho_true"] == pytest.approx(
        result.simulation_truth["repayment_probability_per_period_true"].mean()
    )
    assert report["risk"]["realized_default_rate"] == pytest.approx(
        (~result.loan_outcomes["completed_all_payments"]).mean()
    )
    assert report["economics"]["positive_expected_profit_share"] + report[
        "economics"
    ]["nonpositive_expected_profit_share"] == pytest.approx(1.0)


def test_cohorts_are_applicant_level_and_disjoint() -> None:
    result = _result(300)
    membership = result.applicants.groupby("applicant_id", observed=True)["cohort"].nunique()
    assert (membership == 1).all()
    cohort_sets = [
        set(result.applicants.loc[result.applicants["cohort"] == name, "applicant_id"])
        for name in result.applicants["cohort"].cat.categories
    ]
    assert all(left.isdisjoint(right) for i, left in enumerate(cohort_sets) for right in cohort_sets[i + 1 :])
