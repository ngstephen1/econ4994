"""End-to-end Version 2 applicant, truth, and observed-history pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from fair_lending.economic_lending.applicants import generate_applicant_population
from fair_lending.economic_lending.config import create_random_streams
from fair_lending.economic_lending.contracts import make_requested_loan_options
from fair_lending.economic_lending.diagnostics import population_validation_report
from fair_lending.economic_lending.outcomes import simulate_repayment_outcomes
from fair_lending.economic_lending.repayment import true_repayment_probabilities


@dataclass(frozen=True)
class EconomicPopulationResult:
    """The four logical outputs and diagnostics for one generated population."""

    applicants: pd.DataFrame
    loan_options: pd.DataFrame
    simulation_truth: pd.DataFrame
    loan_outcomes: pd.DataFrame
    validation_report: dict[str, Any]
    group_diagnostics: pd.DataFrame


def generate_economic_population(
    n_applicants: int,
    config: dict[str, Any],
    seed: int | None = None,
) -> EconomicPopulationResult:
    """Generate the reproducible Version 2 baseline population."""

    run_seed = int(config["randomness"]["seed"] if seed is None else seed)
    resolved = dict(config)
    resolved["randomness"] = dict(config["randomness"])
    resolved["randomness"]["seed"] = run_seed
    streams, spawn_keys = create_random_streams(
        run_seed, resolved["randomness"]["stream_names"]
    )
    applicants, clipping = generate_applicant_population(
        n_applicants, resolved, streams
    )
    loan_options = make_requested_loan_options(applicants, resolved["contract"])
    truth = true_repayment_probabilities(
        applicants, loan_options, resolved["true_risk"]
    )
    outcomes = simulate_repayment_outcomes(
        loan_options, truth, streams["repayment_outcomes"]
    )
    report, group_table = population_validation_report(
        applicants,
        loan_options,
        truth,
        outcomes,
        resolved,
        clipping,
        spawn_keys,
    )
    return EconomicPopulationResult(
        applicants=applicants,
        loan_options=loan_options,
        simulation_truth=truth,
        loan_outcomes=outcomes,
        validation_report=report,
        group_diagnostics=group_table,
    )
