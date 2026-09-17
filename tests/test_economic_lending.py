"""Tests for the Version 2 economic lending baseline."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from fair_lending.economic_lending.allocation import (
    allocate_fixed_requests,
    enumerate_allocations,
)
from fair_lending.economic_lending.applicants import (
    generate_applicant_population,
    tiny_fixture_applicants,
)
from fair_lending.economic_lending.config import create_random_streams, load_economic_config
from fair_lending.economic_lending.contracts import (
    make_fixed_loan_options,
    make_requested_loan_options,
    payment_schedule,
    realized_receipts,
)
from fair_lending.economic_lending.examples import tiny_fixture_pipeline
from fair_lending.economic_lending.profit import expected_profit, expected_receipts
from fair_lending.economic_lending.repayment import (
    survival_probability,
    true_repayment_probabilities,
    true_repayment_probabilities_from_score,
)
from fair_lending.economic_lending.schema import (
    BaselineEconomicParameters,
    SCHEMA_CONTRACTS,
    SCHEMA_VERSION,
    WORKING_PAYMENT_RULE_ID,
)
from fair_lending.economic_lending.validation import (
    require_unique_applicant_ids,
    require_valid_probabilities,
)


def _fixture_tables():
    params = BaselineEconomicParameters()
    applicants = tiny_fixture_applicants()
    loans = make_fixed_loan_options(applicants, params)
    schedule = payment_schedule(loans)
    truth = true_repayment_probabilities_from_score(applicants, loans)
    assessments = expected_profit(loans, schedule, truth)
    return params, applicants, loans, schedule, truth, assessments


def _generated_applicants(
    n_applicants: int,
    seed: int,
    group_shares: dict[str, float] | None = None,
):
    config = load_economic_config()
    if group_shares is not None:
        config = copy.deepcopy(config)
        config["population"]["group_shares"] = group_shares
    streams, _ = create_random_streams(seed, config["randomness"]["stream_names"])
    applicants, _ = generate_applicant_population(n_applicants, config, streams)
    return config, applicants


def test_schema_keys_are_declared() -> None:
    assert set(SCHEMA_CONTRACTS) == {
        "applicants",
        "loan_options",
        "payment_schedule",
        "simulation_truth",
        "policy_assessments",
        "lending_decisions",
        "loan_outcomes",
        "run_manifest",
    }
    assert SCHEMA_VERSION == "v2-economic-lending-0.3"
    assert "payment_rule_id" in SCHEMA_CONTRACTS["loan_options"]
    assert "payment_rule_id" in SCHEMA_CONTRACTS["payment_schedule"]
    assert SCHEMA_CONTRACTS["simulation_truth"] == (
        "applicant_id",
        "repayment_probability_per_period_true",
        "full_repayment_probability_true",
    )


def test_applicant_ids_are_unique() -> None:
    _, applicants = _generated_applicants(n_applicants=25, seed=4994)
    require_unique_applicant_ids(applicants)
    assert applicants["applicant_id"].is_unique


def test_repayment_probabilities_are_valid() -> None:
    config, applicants = _generated_applicants(n_applicants=100, seed=4994)
    loans = make_requested_loan_options(applicants, config["contract"])
    truth = true_repayment_probabilities(applicants, loans, config["true_risk"])
    require_valid_probabilities(truth)
    assert truth["repayment_probability_per_period_true"].between(0, 1).all()
    assert truth["full_repayment_probability_true"].between(0, 1).all()


def test_full_repayment_probability_equals_rho_to_term() -> None:
    _, _, _, _, truth, _ = _fixture_tables()
    expected = truth["repayment_probability_per_period_true"] ** 2
    np.testing.assert_allclose(truth["full_repayment_probability_true"], expected)


def test_survival_probability_is_rho_to_period_and_nonincreasing() -> None:
    periods = np.array([1, 2, 3])
    survival = survival_probability(0.8, periods)
    np.testing.assert_allclose(survival, [0.8, 0.64, 0.512])
    assert np.all(np.diff(survival) <= 0)


def test_payment_schedule_arithmetic() -> None:
    _, _, _, schedule, _, _ = _fixture_tables()
    by_applicant = schedule.groupby("applicant_id")["scheduled_payment"].sum()
    assert by_applicant.to_dict() == {1: 140.0, 2: 140.0, 3: 140.0, 4: 140.0}
    assert schedule["scheduled_payment"].unique().tolist() == [70.0]
    assert schedule["payment_rule_id"].unique().tolist() == [WORKING_PAYMENT_RULE_ID]


def test_full_repayment_gives_all_promised_payments() -> None:
    _, _, _, schedule, _, _ = _fixture_tables()
    assert realized_receipts(schedule, applicant_id=1, default_period=None) == 140.0


def test_immediate_default_gives_zero_later_payments() -> None:
    _, _, _, schedule, _, _ = _fixture_tables()
    assert realized_receipts(schedule, applicant_id=1, default_period=1) == 0.0


def test_default_in_period_two_preserves_only_period_one_receipt() -> None:
    _, _, _, schedule, _, _ = _fixture_tables()
    assert realized_receipts(schedule, applicant_id=1, default_period=2) == 70.0


def test_positive_infinity_represents_full_repayment() -> None:
    _, _, _, schedule, _, _ = _fixture_tables()
    assert realized_receipts(schedule, applicant_id=1, default_period=float("inf")) == 140.0


def test_invalid_default_period_is_rejected() -> None:
    _, _, _, schedule, _, _ = _fixture_tables()
    with pytest.raises(ValueError, match="default_period"):
        realized_receipts(schedule, applicant_id=1, default_period=0)


def test_expected_receipt_calculation() -> None:
    _, _, _, schedule, truth, _ = _fixture_tables()
    receipts = expected_receipts(schedule, truth)
    assert receipts.set_index("applicant_id")["expected_receipts_true"].to_dict() == {
        1: pytest.approx(119.7),
        2: pytest.approx(91.875),
        3: pytest.approx(52.5),
        4: pytest.approx(129.675),
    }


def test_one_period_expected_receipts_equal_rho_times_payment() -> None:
    params = BaselineEconomicParameters(term_periods=1)
    applicants = tiny_fixture_applicants().head(1)
    loans = make_fixed_loan_options(applicants, params)
    schedule = payment_schedule(loans)
    truth = true_repayment_probabilities_from_score(applicants, loans)
    receipts = expected_receipts(schedule, truth)
    payment = schedule.loc[0, "scheduled_payment"]
    rho = truth.loc[0, "repayment_probability_per_period_true"]
    assert receipts.loc[0, "expected_receipts_true"] == pytest.approx(rho * payment)


def test_multi_period_expected_receipts_use_period_survival() -> None:
    _, _, _, schedule, truth, _ = _fixture_tables()
    receipts = expected_receipts(schedule, truth)
    rho = truth.loc[0, "repayment_probability_per_period_true"]
    expected = 70.0 * rho + 70.0 * rho**2
    old_whole_loan_formula = rho * 140.0
    assert receipts.loc[0, "expected_receipts_true"] == pytest.approx(expected)
    assert receipts.loc[0, "expected_receipts_true"] != pytest.approx(old_whole_loan_formula)


def test_expected_profit_calculation() -> None:
    _, _, _, _, _, assessments = _fixture_tables()
    assert assessments.set_index("applicant_id")["expected_profit_true"].to_dict() == {
        1: pytest.approx(14.7),
        2: pytest.approx(-13.125),
        3: pytest.approx(-52.5),
        4: pytest.approx(24.675),
    }


def test_expected_profit_subtracts_principal_exactly_once() -> None:
    _, _, loans, _, _, assessments = _fixture_tables()
    merged = loans.merge(assessments, on="applicant_id")
    reconstructed = (
        merged["expected_receipts_true"]
        - merged["requested_principal"]
        - merged["transaction_cost"]
    )
    np.testing.assert_allclose(merged["expected_profit_true"], reconstructed)


def test_zero_loan_gives_zero_lending_profit() -> None:
    params = BaselineEconomicParameters(requested_principal=0.0)
    applicants = tiny_fixture_applicants().head(1)
    loans = make_fixed_loan_options(applicants, params)
    schedule = payment_schedule(loans)
    truth = true_repayment_probabilities_from_score(applicants, loans)
    assessments = expected_profit(loans, schedule, truth)
    assert assessments.loc[0, "expected_receipts_true"] == 0.0
    assert assessments.loc[0, "expected_profit_true"] == 0.0


def test_transaction_cost_charged_exactly_once() -> None:
    params = BaselineEconomicParameters(transaction_cost=7.0)
    applicants = tiny_fixture_applicants().head(1)
    loans = make_fixed_loan_options(applicants, params)
    schedule = payment_schedule(loans)
    truth = true_repayment_probabilities_from_score(applicants, loans)
    assessments = expected_profit(loans, schedule, truth)
    assert assessments.loc[0, "expected_profit_true"] == pytest.approx(119.7 - 100.0 - 7.0)


def test_bank_budget_never_exceeded() -> None:
    params, _, loans, _, _, assessments = _fixture_tables()
    allocation = allocate_fixed_requests(loans, assessments, params.bank_budget)
    assert allocation.total_funded_principal <= params.bank_budget


def test_at_most_one_loan_per_applicant() -> None:
    params, _, loans, _, _, assessments = _fixture_tables()
    allocation = allocate_fixed_requests(loans, assessments, params.bank_budget)
    assert allocation.decisions["applicant_id"].is_unique


def test_exhaustive_enumeration_matches_allocation_solver() -> None:
    params, _, loans, _, _, assessments = _fixture_tables()
    allocation = allocate_fixed_requests(loans, assessments, params.bank_budget)
    enumerated = enumerate_allocations(loans, assessments, params.bank_budget)
    best = enumerated.loc[enumerated["feasible"], "total_expected_profit_true"].max()
    assert allocation.total_expected_profit_true == pytest.approx(best)
    assert allocation.selected_applicant_ids == (1, 4)


def test_same_seed_produces_same_applicants() -> None:
    _, first = _generated_applicants(n_applicants=30, seed=4994)
    _, second = _generated_applicants(n_applicants=30, seed=4994)
    pd.testing.assert_frame_equal(first, second)


def test_baseline_applicant_distributions_do_not_depend_on_group() -> None:
    _, applicants_a = _generated_applicants(
        n_applicants=20, seed=4994, group_shares={"A": 1.0, "B": 0.0}
    )
    _, applicants_b = _generated_applicants(
        n_applicants=20, seed=4994, group_shares={"A": 0.0, "B": 1.0}
    )
    compare_cols = [
        "annual_income",
        "credit_score",
        "employment_years",
        "liquid_assets",
        "existing_monthly_debt",
        "property_value",
        "requested_loan_amount",
    ]
    np.testing.assert_allclose(applicants_a[compare_cols], applicants_b[compare_cols])


def test_group_does_not_enter_repayment_model() -> None:
    applicants = tiny_fixture_applicants()
    flipped = applicants.copy()
    flipped["group"] = flipped["group"].map({"A": "B", "B": "A"})
    pd.testing.assert_frame_equal(
        true_repayment_probabilities_from_score(applicants, make_fixed_loan_options(applicants)),
        true_repayment_probabilities_from_score(flipped, make_fixed_loan_options(flipped)),
    )


def test_group_does_not_enter_profit_function() -> None:
    _, applicants, loans, schedule, truth, assessments = _fixture_tables()
    flipped = applicants.copy()
    flipped["group"] = flipped["group"].map({"A": "B", "B": "A"})
    flipped_truth = true_repayment_probabilities_from_score(flipped, loans)
    flipped_assessments = expected_profit(loans, schedule, flipped_truth)
    pd.testing.assert_frame_equal(assessments, flipped_assessments)


def test_group_does_not_enter_allocation_objective() -> None:
    params, applicants, loans, schedule, _, assessments = _fixture_tables()
    flipped = applicants.copy()
    flipped["group"] = flipped["group"].map({"A": "B", "B": "A"})
    flipped_assessments = expected_profit(
        loans,
        schedule,
        true_repayment_probabilities_from_score(flipped, loans),
    )
    allocation = allocate_fixed_requests(loans, assessments, params.bank_budget)
    flipped_allocation = allocate_fixed_requests(loans, flipped_assessments, params.bank_budget)
    assert allocation.selected_applicant_ids == flipped_allocation.selected_applicant_ids


def test_tiny_fixture_pipeline_is_hand_checkable() -> None:
    display, allocation = tiny_fixture_pipeline()
    assert display["expected_profit_true"].round(6).tolist() == [
        14.7,
        -13.125,
        -52.5,
        24.675,
    ]
    assert allocation.selected_applicant_ids == (1, 4)
    assert allocation.total_funded_principal == 200.0
    assert allocation.total_expected_profit_true == pytest.approx(39.375)
    assert allocation.unused_funds == 0.0
