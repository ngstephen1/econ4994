"""Schema contracts for the Version 2 economic lending baseline."""

from __future__ import annotations

from dataclasses import dataclass


APPLICANTS_COLUMNS = (
    "applicant_id",
    "group",
    "cohort",
    "age_years",
    "annual_income",
    "credit_score",
    "employment_years",
    "liquid_assets",
    "existing_monthly_debt",
    "property_value",
    "requested_loan_amount",
)

LOAN_OPTIONS_COLUMNS = (
    "applicant_id",
    "option_id",
    "requested_principal",
    "periodic_interest_rate",
    "term_periods",
    "transaction_cost",
    "payment_rule_id",
    "first_period_payment",
    "first_period_dti",
    "requested_ltv",
)

PAYMENT_SCHEDULE_COLUMNS = (
    "applicant_id",
    "period",
    "scheduled_payment",
    "payment_rule_id",
)

SIMULATION_TRUTH_COLUMNS = (
    "applicant_id",
    "repayment_probability_per_period_true",
    "full_repayment_probability_true",
)

TRUE_ECONOMIC_ASSESSMENT_COLUMNS = (
    "applicant_id",
    "expected_receipts_true",
    "expected_profit_true",
)

POLICY_ASSESSMENTS_COLUMNS = (
    "applicant_id",
    "option_id",
    "policy_id",
    "repayment_probability_base",
    "repayment_probability_used",
    "expected_receipts_perceived",
    "expected_profit_perceived",
    "assessment_status",
)

LENDING_DECISIONS_COLUMNS = (
    "applicant_id",
    "funded",
    "funded_principal",
)

LOAN_OUTCOMES_COLUMNS = (
    "applicant_id",
    "default_period",
    "completed_all_payments",
    "realized_total_receipts",
    "realized_profit",
)

RUN_MANIFEST_COLUMNS = (
    "run_id",
    "random_seed",
    "n_applicants",
    "bank_budget",
    "schema_version",
)

SCHEMA_VERSION = "v2-economic-lending-0.3"

WORKING_PAYMENT_RULE_ID = "equal_payments_simple_interest_v1"


@dataclass(frozen=True)
class BaselineEconomicParameters:
    """Small executable assumptions for the baseline accounting model."""

    requested_principal: float = 100.0
    periodic_interest_rate: float = 0.20
    term_periods: int = 2
    transaction_cost: float = 5.0
    bank_budget: float = 200.0
    payment_rule_id: str = WORKING_PAYMENT_RULE_ID


SCHEMA_CONTRACTS = {
    "applicants": APPLICANTS_COLUMNS,
    "loan_options": LOAN_OPTIONS_COLUMNS,
    "payment_schedule": PAYMENT_SCHEDULE_COLUMNS,
    "simulation_truth": SIMULATION_TRUTH_COLUMNS,
    "policy_assessments": POLICY_ASSESSMENTS_COLUMNS,
    "lending_decisions": LENDING_DECISIONS_COLUMNS,
    "loan_outcomes": LOAN_OUTCOMES_COLUMNS,
    "run_manifest": RUN_MANIFEST_COLUMNS,
}
