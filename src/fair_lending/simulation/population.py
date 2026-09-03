"""Vectorized generation of pre-decision mortgage application features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from fair_lending.simulation.config import approval_transform_parameters


RANDOM_STREAM_NAMES = (
    "demographics",
    "latent_factors",
    "context",
    "financials",
    "loans",
    "approval",
)


def create_random_streams(
    seed: int,
) -> tuple[dict[str, np.random.Generator], dict[str, list[int]]]:
    """Create stable, independent child streams from one recorded run seed."""
    children = np.random.SeedSequence(seed).spawn(len(RANDOM_STREAM_NAMES))
    streams = {
        name: np.random.default_rng(child)
        for name, child in zip(RANDOM_STREAM_NAMES, children, strict=True)
    }
    spawn_keys = {
        name: list(child.spawn_key)
        for name, child in zip(RANDOM_STREAM_NAMES, children, strict=True)
    }
    return streams, spawn_keys


def _categorical_draw(
    rng: np.random.Generator, n_rows: int, shares: dict[str, float]
) -> np.ndarray:
    categories = np.asarray(list(shares), dtype=object)
    probabilities = np.asarray(list(shares.values()), dtype=float)
    return rng.choice(categories, size=n_rows, p=probabilities)


def _values_for_categories(
    categories: np.ndarray | pd.Series,
    mapping: dict[str, float],
) -> np.ndarray:
    values = np.empty(len(categories), dtype=float)
    category_array = np.asarray(categories)
    assigned = np.zeros(len(categories), dtype=bool)
    for category, value in mapping.items():
        mask = category_array == category
        values[mask] = float(value)
        assigned |= mask
    if not assigned.all():
        unknown = np.unique(category_array[~assigned]).tolist()
        raise ValueError(f"Categories missing from parameter map: {unknown}")
    return values


def _clip_with_diagnostics(
    raw: np.ndarray, minimum: float, maximum: float
) -> tuple[np.ndarray, dict[str, float]]:
    below = raw < minimum
    above = raw > maximum
    clipped = np.clip(raw, minimum, maximum)
    return clipped, {
        "below_minimum_share": float(np.mean(below)),
        "above_maximum_share": float(np.mean(above)),
        "total_clipped_share": float(np.mean(below | above)),
    }


def _age_draws(
    age_group: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    lower_map = {
        "18-24": 18,
        "25-34": 25,
        "35-44": 35,
        "45-54": 45,
        "55-64": 55,
        "65+": 65,
    }
    upper_exclusive_map = {
        "18-24": 25,
        "25-34": 35,
        "35-44": 45,
        "45-54": 55,
        "55-64": 65,
        "65+": 81,
    }
    low = _values_for_categories(age_group, lower_map).astype(int)
    high = _values_for_categories(age_group, upper_exclusive_map).astype(int)
    ages = np.floor(low + rng.random(len(age_group)) * (high - low)).astype(int)
    feasible_history = np.minimum(ages - 18, 45).astype(float)
    return ages, feasible_history


def generate_population(
    config: dict[str, Any],
    streams: dict[str, np.random.Generator],
    *,
    include_application_id: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate every persisted pre-outcome feature in documented causal order."""
    n_rows = int(config["simulation"]["n_samples"])
    demographics_rng = streams["demographics"]
    latent_rng = streams["latent_factors"]
    context_rng = streams["context"]
    financial_rng = streams["financials"]
    loan_rng = streams["loans"]
    clipping: dict[str, dict[str, float]] = {}

    demographic_specs = config["population"]["demographics"]
    race = _categorical_draw(demographics_rng, n_rows, demographic_specs["race"]["shares"])
    ethnicity = _categorical_draw(
        demographics_rng, n_rows, demographic_specs["ethnicity"]["shares"]
    )
    sex = _categorical_draw(demographics_rng, n_rows, demographic_specs["sex"]["shares"])
    age_group = _categorical_draw(
        demographics_rng, n_rows, demographic_specs["age_group"]["shares"]
    )

    rho = float(config["latent_factors"]["dependence"]["correlation"])
    latent_draws = latent_rng.standard_normal((n_rows, 2))
    financial_stability = latent_draws[:, 0]
    neighborhood_advantage = (
        rho * latent_draws[:, 0] + np.sqrt(1.0 - rho**2) * latent_draws[:, 1]
    )

    context = config["context_variables"]
    neighborhood_income_spec = context["neighborhood_income_index"]
    equation = neighborhood_income_spec["equation"]
    raw_neighborhood_income = (
        equation["intercept"]
        + equation["neighborhood_advantage_coefficient"] * neighborhood_advantage
        + context_rng.normal(0.0, equation["residual_standard_deviation"], n_rows)
    )
    bounds = neighborhood_income_spec["bounds"]
    neighborhood_income_index, clipping["neighborhood_income_index"] = (
        _clip_with_diagnostics(raw_neighborhood_income, bounds["minimum"], bounds["maximum"])
    )

    minority_spec = context["neighborhood_minority_share"]["equation"]
    minority_score = (
        minority_spec["intercept"]
        + minority_spec["neighborhood_advantage_coefficient"] * neighborhood_advantage
        + context_rng.normal(0.0, minority_spec["residual_standard_deviation"], n_rows)
    )
    neighborhood_minority_share = expit(minority_score)
    clipping["neighborhood_minority_share"] = {
        "below_minimum_share": 0.0,
        "above_maximum_share": 0.0,
        "total_clipped_share": 0.0,
    }

    unemployment_spec = context["local_unemployment_rate"]
    equation = unemployment_spec["equation"]
    raw_unemployment = (
        equation["intercept"]
        + equation["neighborhood_advantage_coefficient"] * neighborhood_advantage
        + context_rng.normal(0.0, equation["residual_standard_deviation"], n_rows)
    )
    bounds = unemployment_spec["bounds"]
    local_unemployment_rate, clipping["local_unemployment_rate"] = (
        _clip_with_diagnostics(raw_unemployment, bounds["minimum"], bounds["maximum"])
    )

    financial = config["financial_variables"]
    upstream = config["scenario_effects"]["upstream"]["treatments"]
    income_spec = financial["annual_income"]
    income_shift = _values_for_categories(
        race, upstream["annual_income"]["additive_shift_by_race"]
    )
    income_age_multiplier = _values_for_categories(
        age_group, income_spec["age_group_multipliers"]
    )
    raw_log_income = (
        np.log(income_spec["reference_median_usd"])
        + np.log(income_age_multiplier)
        + income_spec["financial_stability_coefficient"] * financial_stability
        + income_spec["neighborhood_advantage_coefficient"] * neighborhood_advantage
        + income_shift
        + financial_rng.normal(0.0, income_spec["log_residual_standard_deviation"], n_rows)
    )
    raw_income = np.exp(raw_log_income)
    bounds = income_spec["bounds"]
    annual_income, clipping["annual_income"] = _clip_with_diagnostics(
        raw_income, bounds["minimum"], bounds["maximum"]
    )
    income_transform = approval_transform_parameters(config, "annual_income")
    standardized_log_income = (
        np.log(annual_income) - np.log(float(income_transform["center"]))
    ) / float(income_transform["scale"])

    _, feasible_history = _age_draws(age_group, financial_rng)
    employment_spec = financial["employment_years"]
    fraction_spec = employment_spec["employment_fraction"]
    raw_employment_fraction = (
        fraction_spec["intercept"]
        + fraction_spec["financial_stability_coefficient"] * financial_stability
        + financial_rng.normal(0.0, fraction_spec["residual_standard_deviation"], n_rows)
    )
    employment_fraction = np.clip(
        raw_employment_fraction, fraction_spec["minimum"], fraction_spec["maximum"]
    )
    employment_years = feasible_history * employment_fraction
    clipping["employment_fraction"] = {
        "below_minimum_share": float(np.mean(raw_employment_fraction < fraction_spec["minimum"])),
        "above_maximum_share": float(np.mean(raw_employment_fraction > fraction_spec["maximum"])),
        "total_clipped_share": float(
            np.mean(
                (raw_employment_fraction < fraction_spec["minimum"])
                | (raw_employment_fraction > fraction_spec["maximum"])
            )
        ),
    }

    credit_spec = financial["credit_score"]
    credit_shift = _values_for_categories(
        race, upstream["credit_score"]["additive_shift_by_race"]
    )
    raw_credit = (
        credit_spec["reference_mean"]
        + credit_spec["financial_stability_coefficient"] * financial_stability
        + credit_spec["standardized_log_income_coefficient"] * standardized_log_income
        + credit_shift
        + financial_rng.normal(0.0, credit_spec["residual_standard_deviation"], n_rows)
    )
    rounded_credit = np.rint(raw_credit)
    bounds = credit_spec["bounds"]
    credit_score, clipping["credit_score"] = _clip_with_diagnostics(
        rounded_credit, bounds["minimum"], bounds["maximum"]
    )
    credit_score = credit_score.astype(np.int16)

    assets_spec = financial["liquid_assets"]
    assets_shift = _values_for_categories(
        race, upstream["liquid_assets"]["additive_shift_by_race"]
    )
    asset_age_multiplier = _values_for_categories(
        age_group, assets_spec["age_group_multipliers"]
    )
    raw_log_assets = (
        np.log(assets_spec["reference_median_usd"])
        + assets_spec["log_income_elasticity"]
        * (np.log(annual_income) - np.log(income_spec["reference_median_usd"]))
        + assets_spec["financial_stability_coefficient"] * financial_stability
        + np.log(asset_age_multiplier)
        + assets_shift
        + financial_rng.normal(0.0, assets_spec["log_residual_standard_deviation"], n_rows)
    )
    raw_assets = np.exp(raw_log_assets)
    bounds = assets_spec["bounds"]
    liquid_assets, clipping["liquid_assets"] = _clip_with_diagnostics(
        raw_assets, bounds["minimum"], bounds["maximum"]
    )

    debt_spec = financial["existing_monthly_debt"]
    raw_positive_debt = np.exp(
        np.log(debt_spec["positive_reference_median_usd"])
        + debt_spec["income_elasticity"]
        * (np.log(annual_income) - np.log(income_spec["reference_median_usd"]))
        + debt_spec["financial_stability_coefficient"] * financial_stability
        + financial_rng.normal(0.0, debt_spec["log_residual_standard_deviation"], n_rows)
    )
    positive_debt, debt_clip = _clip_with_diagnostics(
        raw_positive_debt,
        debt_spec["bounds"]["minimum"],
        debt_spec["bounds"]["maximum"],
    )
    has_zero_debt = financial_rng.random(n_rows) < debt_spec["zero_probability"]
    existing_monthly_debt = np.where(has_zero_debt, 0.0, positive_debt)
    clipping["existing_monthly_debt"] = {
        key: value * float(np.mean(~has_zero_debt)) for key, value in debt_clip.items()
    }

    loan = config["loan_property_variables"]
    loan_purpose = _categorical_draw(loan_rng, n_rows, loan["loan_purpose"]["shares"])
    loan_type = _categorical_draw(loan_rng, n_rows, loan["loan_type"]["shares"])
    occupancy_type = _categorical_draw(
        loan_rng, n_rows, loan["occupancy_type"]["shares"]
    )

    property_spec = loan["property_value"]
    purpose_multiplier = _values_for_categories(
        loan_purpose, property_spec["purpose_multipliers"]
    )
    occupancy_multiplier = _values_for_categories(
        occupancy_type, property_spec["occupancy_multipliers"]
    )
    raw_property_value = np.exp(
        np.log(property_spec["reference_median_usd"])
        + property_spec["income_elasticity"]
        * (np.log(annual_income) - np.log(income_spec["reference_median_usd"]))
        + property_spec["neighborhood_income_elasticity"]
        * np.log(neighborhood_income_index)
        + np.log(purpose_multiplier)
        + np.log(occupancy_multiplier)
        + loan_rng.normal(0.0, property_spec["log_residual_standard_deviation"], n_rows)
    )
    bounds = property_spec["bounds"]
    property_value, clipping["property_value"] = _clip_with_diagnostics(
        raw_property_value, bounds["minimum"], bounds["maximum"]
    )

    ltv_spec = loan["loan_to_value_ratio"]
    base_ltv = ltv_spec["base_minimum"] + (
        ltv_spec["base_maximum"] - ltv_spec["base_minimum"]
    ) * loan_rng.beta(
        ltv_spec["beta_shape_alpha"], ltv_spec["beta_shape_beta"], n_rows
    )
    raw_ltv = (
        base_ltv
        + _values_for_categories(loan_type, ltv_spec["loan_type_offsets"])
        + _values_for_categories(loan_purpose, ltv_spec["purpose_offsets"])
        + _values_for_categories(occupancy_type, ltv_spec["occupancy_offsets"])
    )
    bounds = ltv_spec["bounds"]
    loan_to_value_ratio, clipping["loan_to_value_ratio"] = _clip_with_diagnostics(
        raw_ltv, bounds["minimum"], bounds["maximum"]
    )
    loan_amount = property_value * loan_to_value_ratio
    income_to_loan_ratio = annual_income / loan_amount

    payment = config["internal_housing_payment"]
    projected_housing_payment = (
        loan_amount * payment["monthly_principal_interest_factor"]
        + property_value * payment["monthly_tax_insurance_factor_of_property_value"]
    )
    raw_dti = (existing_monthly_debt + projected_housing_payment) / (
        annual_income / 12.0
    )
    dti_bounds = financial["debt_to_income_ratio"]["bounds"]
    debt_to_income_ratio, clipping["debt_to_income_ratio"] = _clip_with_diagnostics(
        raw_dti, dti_bounds["minimum"], dti_bounds["maximum"]
    )
    clipping["debt_to_income_ratio"]["upper_boundary_share"] = float(
        np.mean(raw_dti >= dti_bounds["maximum"])
    )

    frame_data: dict[str, Any] = {}
    if include_application_id:
        frame_data["application_id"] = np.char.mod(
            "APP%09d", np.arange(1, n_rows + 1)
        )
    frame_data.update(
        {
            "race": pd.Categorical(race, categories=list(demographic_specs["race"]["shares"])),
            "ethnicity": pd.Categorical(
                ethnicity, categories=list(demographic_specs["ethnicity"]["shares"])
            ),
            "sex": pd.Categorical(sex, categories=list(demographic_specs["sex"]["shares"])),
            "age_group": pd.Categorical(
                age_group, categories=list(demographic_specs["age_group"]["shares"])
            ),
            "annual_income": annual_income,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "liquid_assets": liquid_assets,
            "existing_monthly_debt": existing_monthly_debt,
            "debt_to_income_ratio": debt_to_income_ratio,
            "loan_amount": loan_amount,
            "property_value": property_value,
            "loan_to_value_ratio": loan_to_value_ratio,
            "loan_purpose": pd.Categorical(
                loan_purpose, categories=list(loan["loan_purpose"]["shares"])
            ),
            "loan_type": pd.Categorical(
                loan_type, categories=list(loan["loan_type"]["shares"])
            ),
            "occupancy_type": pd.Categorical(
                occupancy_type, categories=list(loan["occupancy_type"]["shares"])
            ),
            "neighborhood_income_index": neighborhood_income_index,
            "neighborhood_minority_share": neighborhood_minority_share,
            "local_unemployment_rate": local_unemployment_rate,
            "income_to_loan_ratio": income_to_loan_ratio,
        }
    )
    diagnostics = {
        "clipping_rates": clipping,
        "latent_correlation_realized": float(
            np.corrcoef(financial_stability, neighborhood_advantage)[0, 1]
        ),
        "zero_debt_share": float(np.mean(has_zero_debt)),
    }
    return pd.DataFrame(frame_data), diagnostics
