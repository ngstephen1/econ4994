"""Small hand-checkable Version 2 baseline examples."""

from __future__ import annotations

import pandas as pd

from fair_lending.economic_lending.allocation import AllocationResult, allocate_fixed_requests
from fair_lending.economic_lending.applicants import tiny_fixture_applicants
from fair_lending.economic_lending.contracts import make_fixed_loan_options, payment_schedule
from fair_lending.economic_lending.profit import expected_profit
from fair_lending.economic_lending.repayment import true_repayment_probabilities_from_score
from fair_lending.economic_lending.schema import BaselineEconomicParameters


def tiny_fixture_pipeline() -> tuple[pd.DataFrame, AllocationResult]:
    """Run the four-applicant fixture from applicants through allocation."""

    params = BaselineEconomicParameters()
    applicants = tiny_fixture_applicants()
    loan_options = make_fixed_loan_options(applicants, params)
    schedule = payment_schedule(loan_options)
    truth = true_repayment_probabilities_from_score(applicants, loan_options)
    assessments = expected_profit(loan_options, schedule, truth)
    display = (
        applicants.loc[:, ["applicant_id", "group"]]
        .merge(loan_options, on="applicant_id")
        .merge(truth, on="applicant_id")
        .merge(assessments, on="applicant_id")
    )
    allocation = allocate_fixed_requests(loan_options, assessments, params.bank_budget)
    return display, allocation
