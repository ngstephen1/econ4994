"""Version 2 economic lending baseline.

This package is separate from the Version 1 approval-disparity simulation.
It contains the Version 2 repayment truth, cash-flow accounting, traditional
and flexible risk estimation, and fixed-request allocation foundations.
"""

from fair_lending.economic_lending.allocation import AllocationResult, allocate_fixed_requests
from fair_lending.economic_lending.applicants import generate_applicant_population
from fair_lending.economic_lending.contracts import (
    make_fixed_loan_options,
    make_requested_loan_options,
    payment_schedule,
)
from fair_lending.economic_lending.outcomes import simulate_repayment_outcomes
from fair_lending.economic_lending.population import (
    EconomicPopulationResult,
    generate_economic_population,
)
from fair_lending.economic_lending.profit import (
    expected_profit,
    expected_profit_from_options,
    expected_receipts,
)
from fair_lending.economic_lending.repayment import (
    survival_probability,
    true_repayment_probabilities,
    true_repayment_probabilities_from_score,
)

__all__ = [
    "AllocationResult",
    "EconomicPopulationResult",
    "allocate_fixed_requests",
    "expected_profit",
    "expected_profit_from_options",
    "expected_receipts",
    "generate_applicant_population",
    "generate_economic_population",
    "make_fixed_loan_options",
    "make_requested_loan_options",
    "payment_schedule",
    "simulate_repayment_outcomes",
    "survival_probability",
    "true_repayment_probabilities",
    "true_repayment_probabilities_from_score",
]
