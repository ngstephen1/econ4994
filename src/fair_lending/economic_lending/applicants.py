"""Applicant population generation for Version 2 economic lending."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fair_lending.economic_lending.schema import APPLICANTS_COLUMNS


def _clip(
    values: np.ndarray,
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray, dict[str, float]]:
    outside = (values < minimum) | (values > maximum)
    return np.clip(values, minimum, maximum), {
        "below_minimum_share": float(np.mean(values < minimum)),
        "above_maximum_share": float(np.mean(values > maximum)),
        "total_clipped_share": float(np.mean(outside)),
    }


def _draw_cohorts(
    n_applicants: int,
    shares: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    names = list(shares)
    raw_counts = np.asarray([shares[name] * n_applicants for name in names])
    counts = np.floor(raw_counts).astype(int)
    remainder = n_applicants - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw_counts - counts), kind="stable")
        counts[order[:remainder]] += 1
    labels = np.concatenate(
        [np.repeat(name, count) for name, count in zip(names, counts, strict=True)]
    )
    return labels[rng.permutation(n_applicants)]


def generate_applicant_population(
    n_applicants: int,
    config: dict[str, Any],
    streams: dict[str, np.random.Generator],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Generate one baseline population with group-independent finances.

    The calibration uses two correlated latent factors: financial stability
    and creditworthiness. They are internal causes and are not persisted.
    """

    if n_applicants <= 0:
        raise ValueError("n_applicants must be positive")
    required_streams = {
        "group",
        "demographics",
        "latent_factors",
        "financial_variables",
        "loan_requests",
        "cohorts",
    }
    missing = required_streams - set(streams)
    if missing:
        raise ValueError(f"missing random streams: {sorted(missing)}")

    population = config["population"]
    clipping: dict[str, dict[str, float]] = {}

    group_names = list(population["group_shares"])
    group = streams["group"].choice(
        group_names,
        size=n_applicants,
        p=[population["group_shares"][name] for name in group_names],
    )
    cohort = _draw_cohorts(n_applicants, config["cohorts"]["shares"], streams["cohorts"])

    age_spec = population["age"]
    age_unit = streams["demographics"].beta(
        age_spec["beta_alpha"], age_spec["beta_beta"], n_applicants
    )
    age_years = np.rint(
        age_spec["minimum"] + age_unit * (age_spec["maximum"] - age_spec["minimum"])
    ).astype(np.int16)

    latent = streams["latent_factors"].standard_normal((n_applicants, 2))
    financial_stability = latent[:, 0]
    correlation = float(population["latent_factors"]["correlation"])
    creditworthiness = (
        correlation * financial_stability
        + np.sqrt(1.0 - correlation**2) * latent[:, 1]
    )

    financial_rng = streams["financial_variables"]
    employment_spec = population["employment_years"]
    feasible_history = np.maximum(age_years - employment_spec["entry_age"], 0).astype(float)
    raw_employment_fraction = (
        employment_spec["fraction_intercept"]
        + employment_spec["stability_coefficient"] * financial_stability
        + financial_rng.normal(0.0, employment_spec["residual_sd"], n_applicants)
    )
    employment_fraction, clipping["employment_fraction"] = _clip(
        raw_employment_fraction,
        employment_spec["minimum_fraction"],
        employment_spec["maximum_fraction"],
    )
    employment_years = feasible_history * employment_fraction

    income_spec = population["annual_income"]
    raw_income = np.exp(
        np.log(income_spec["reference_median"])
        + income_spec["stability_coefficient"] * financial_stability
        + income_spec["employment_coefficient"] * (employment_years - 10.0)
        + income_spec["creditworthiness_coefficient"] * creditworthiness
        + financial_rng.normal(0.0, income_spec["log_residual_sd"], n_applicants)
    )
    annual_income, clipping["annual_income"] = _clip(
        raw_income, income_spec["minimum"], income_spec["maximum"]
    )

    credit_spec = population["credit_score"]
    raw_credit = (
        credit_spec["reference_mean"]
        + credit_spec["creditworthiness_coefficient"] * creditworthiness
        + credit_spec["stability_coefficient"] * financial_stability
        + financial_rng.normal(0.0, credit_spec["residual_sd"], n_applicants)
    )
    credit_score, clipping["credit_score"] = _clip(
        np.rint(raw_credit), credit_spec["minimum"], credit_spec["maximum"]
    )
    credit_score = credit_score.astype(np.int16)

    assets_spec = population["liquid_assets"]
    raw_assets = np.exp(
        np.log(assets_spec["reference_median"])
        + assets_spec["income_elasticity"]
        * np.log(annual_income / income_spec["reference_median"])
        + assets_spec["stability_coefficient"] * financial_stability
        + assets_spec["age_coefficient"] * (age_years - 40.0)
        + financial_rng.normal(0.0, assets_spec["log_residual_sd"], n_applicants)
    )
    liquid_assets, clipping["liquid_assets"] = _clip(
        raw_assets, assets_spec["minimum"], assets_spec["maximum"]
    )

    debt_spec = population["existing_monthly_debt"]
    debt_logit = (
        debt_spec["logit_intercept"]
        + debt_spec["stability_coefficient"] * financial_stability
        + debt_spec["creditworthiness_coefficient"] * creditworthiness
        + financial_rng.normal(0.0, debt_spec["residual_sd"], n_applicants)
    )
    debt_unit = 1.0 / (1.0 + np.exp(-debt_logit))
    debt_income_share = debt_spec["minimum_income_share"] + debt_unit * (
        debt_spec["maximum_income_share"] - debt_spec["minimum_income_share"]
    )
    existing_monthly_debt = debt_income_share * (annual_income / 12.0)

    loan_rng = streams["loan_requests"]
    property_spec = population["property_value"]
    raw_property_value = np.exp(
        np.log(property_spec["reference_median"])
        + property_spec["income_elasticity"]
        * np.log(annual_income / income_spec["reference_median"])
        + property_spec["stability_coefficient"] * financial_stability
        + loan_rng.normal(0.0, property_spec["log_residual_sd"], n_applicants)
    )
    property_value, clipping["property_value"] = _clip(
        raw_property_value, property_spec["minimum"], property_spec["maximum"]
    )

    ltv_spec = population["requested_ltv"]
    asset_to_property = liquid_assets / property_value
    raw_ltv = (
        ltv_spec["intercept"]
        + ltv_spec["asset_to_property_coefficient"] * asset_to_property
        + ltv_spec["stability_coefficient"] * financial_stability
        + ltv_spec["creditworthiness_coefficient"] * creditworthiness
        + loan_rng.normal(0.0, ltv_spec["residual_sd"], n_applicants)
    )
    requested_ltv, clipping["requested_ltv"] = _clip(
        raw_ltv, ltv_spec["minimum"], ltv_spec["maximum"]
    )
    requested_loan_amount = property_value * requested_ltv

    applicants = pd.DataFrame(
        {
            "applicant_id": np.char.mod("V2APP%09d", np.arange(1, n_applicants + 1)),
            "group": pd.Categorical(group, categories=["A", "B"]),
            "cohort": pd.Categorical(
                cohort,
                categories=["historical_train", "historical_validation", "evaluation"],
            ),
            "age_years": age_years,
            "annual_income": np.round(annual_income, 2),
            "credit_score": credit_score,
            "employment_years": np.round(employment_years, 3),
            "liquid_assets": np.round(liquid_assets, 2),
            "existing_monthly_debt": np.round(existing_monthly_debt, 2),
            "property_value": np.round(property_value, 2),
            "requested_loan_amount": np.round(requested_loan_amount, 2),
        }
    )
    return applicants.loc[:, APPLICANTS_COLUMNS], clipping


def tiny_fixture_applicants() -> pd.DataFrame:
    """Four hand-checkable applicants retained for accounting validation."""

    return pd.DataFrame(
        {
            "applicant_id": [1, 2, 3, 4],
            "group": ["A", "B", "A", "B"],
            "annual_income": [90_000.0, 90_000.0, 70_000.0, 110_000.0],
            "credit_score": [720.0, 700.0, 650.0, 760.0],
            "repayment_score": [
                2.1972245773362196,
                1.0986122886681098,
                0.0,
                2.9444389791664403,
            ],
        }
    )
