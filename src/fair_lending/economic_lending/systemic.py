"""Matched upstream-opportunity worlds for the Version 2 systemic experiment."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
import yaml
from scipy.optimize import brentq

from fair_lending.economic_lending.config import (
    PROJECT_ROOT,
    config_fingerprint,
    create_random_streams,
)
from fair_lending.economic_lending.contracts import make_requested_loan_options
from fair_lending.economic_lending.repayment import (
    TRUE_RISK_PREDICTORS,
    sigmoid,
    transform_observable_risk_features,
)
from fair_lending.economic_lending.schema import APPLICANTS_COLUMNS


SYSTEMIC_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "economic_lending" / "systemic_opportunity.yaml"
)
SYSTEMIC_MECHANISM_ID = "systemic_opportunity_v1"
SYSTEMIC_LENDER_FORBIDDEN = frozenset(
    {
        "group",
        "opportunity_access",
        "systemic_strength",
        "financial_stability_latent",
        "creditworthiness_latent",
        "opportunity_uniform",
    }
)


def load_systemic_config(path: Path | str = SYSTEMIC_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the frozen upstream-opportunity specification."""

    path = Path(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("mechanism_id") != SYSTEMIC_MECHANISM_ID:
        raise ValueError(f"systemic config must declare mechanism_id={SYSTEMIC_MECHANISM_ID}")
    if float(loaded["direct_belief_delta"]) != 0.0:
        raise ValueError("the pure systemic experiment requires direct_belief_delta=0")
    strengths = [float(value) for value in loaded["systemic_strengths"]]
    if strengths != [0.0, 0.1, 0.2, 0.4]:
        raise ValueError("systemic strengths must remain the predeclared [0, .1, .2, .4]")
    config = copy.deepcopy(loaded)
    config["metadata"] = {
        "config_path": str(path),
        "config_fingerprint": config_fingerprint(loaded),
    }
    return config


def opportunity_probability(
    group: np.ndarray | pd.Series,
    financial_stability: np.ndarray | pd.Series,
    strength: float,
    opportunity_config: dict[str, Any],
) -> np.ndarray:
    """Return access probabilities from pre-treatment variables only."""

    if strength < 0:
        raise ValueError("systemic strength must be nonnegative")
    alpha = opportunity_config.get("solved_intercept")
    if alpha is None:
        raise ValueError("opportunity intercept must be calibrated and frozen")
    treated = np.asarray(group).astype(str) == "B"
    score = (
        float(alpha)
        + float(opportunity_config["beta_stability"])
        * np.asarray(financial_stability, dtype=float)
        - float(strength) * treated
    )
    return np.asarray(sigmoid(score), dtype=float)


def calibrate_opportunity_intercept(
    financial_stability: np.ndarray,
    opportunity_config: dict[str, Any],
) -> float:
    """Solve only the neutral intercept for the declared aggregate target."""

    target = float(opportunity_config["neutral_access_target"])
    beta = float(opportunity_config["beta_stability"])
    tolerance = float(opportunity_config["tolerance"])

    def objective(alpha: float) -> float:
        return float(np.mean(sigmoid(alpha + beta * financial_stability)) - target)

    return float(brentq(objective, -10.0, 10.0, xtol=tolerance))


def draw_systemic_population_inputs(
    n_applicants: int,
    baseline_config: dict[str, Any],
    systemic_config: dict[str, Any],
    *,
    seed: int,
) -> pd.DataFrame:
    """Draw and retain every exogenous input used by matched systemic worlds."""

    if n_applicants <= 0:
        raise ValueError("n_applicants must be positive")
    names = list(baseline_config["randomness"]["stream_names"])
    opportunity_name = systemic_config["randomness"]["opportunity_stream_name"]
    if opportunity_name not in names:
        names.append(opportunity_name)
    streams, _ = create_random_streams(seed, names)
    population = baseline_config["population"]

    group_names = list(population["group_shares"])
    group = streams["group"].choice(
        group_names,
        size=n_applicants,
        p=[population["group_shares"][name] for name in group_names],
    )
    cohort_names = list(baseline_config["cohorts"]["shares"])
    raw_counts = np.asarray(
        [baseline_config["cohorts"]["shares"][name] * n_applicants for name in cohort_names]
    )
    counts = np.floor(raw_counts).astype(int)
    remainder = n_applicants - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw_counts - counts), kind="stable")
        counts[order[:remainder]] += 1
    cohorts = np.concatenate(
        [np.repeat(name, count) for name, count in zip(cohort_names, counts, strict=True)]
    )
    cohorts = cohorts[streams["cohorts"].permutation(n_applicants)]

    age_spec = population["age"]
    age_unit = streams["demographics"].beta(
        age_spec["beta_alpha"], age_spec["beta_beta"], n_applicants
    )
    age = np.rint(
        age_spec["minimum"] + age_unit * (age_spec["maximum"] - age_spec["minimum"])
    ).astype(np.int16)
    latent = streams["latent_factors"].standard_normal((n_applicants, 2))
    stability = latent[:, 0]
    corr = float(population["latent_factors"]["correlation"])
    creditworthiness = corr * stability + np.sqrt(1.0 - corr**2) * latent[:, 1]

    financial_rng = streams["financial_variables"]
    loan_rng = streams["loan_requests"]
    return pd.DataFrame(
        {
            "applicant_id": np.char.mod("V2APP%09d", np.arange(1, n_applicants + 1)),
            "group": group,
            "cohort": cohorts,
            "age_years": age,
            "financial_stability_latent": stability,
            "creditworthiness_latent": creditworthiness,
            "employment_residual": financial_rng.normal(0.0, population["employment_years"]["residual_sd"], n_applicants),
            "income_residual": financial_rng.normal(0.0, population["annual_income"]["log_residual_sd"], n_applicants),
            "credit_residual": financial_rng.normal(0.0, population["credit_score"]["residual_sd"], n_applicants),
            "assets_residual": financial_rng.normal(0.0, population["liquid_assets"]["log_residual_sd"], n_applicants),
            "debt_residual": financial_rng.normal(0.0, population["existing_monthly_debt"]["residual_sd"], n_applicants),
            "property_residual": loan_rng.normal(0.0, population["property_value"]["log_residual_sd"], n_applicants),
            "ltv_residual": loan_rng.normal(0.0, population["requested_ltv"]["residual_sd"], n_applicants),
            "opportunity_uniform": streams[opportunity_name].random(n_applicants),
        }
    )


def _clipped(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    return np.clip(values, float(spec["minimum"]), float(spec["maximum"]))


def generate_systemic_applicants(
    draws: pd.DataFrame,
    baseline_config: dict[str, Any],
    systemic_config: dict[str, Any],
    strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate all descendants from common inputs and opportunity access."""

    population = baseline_config["population"]
    effects = systemic_config["downstream_effects"]
    probability = opportunity_probability(
        draws["group"],
        draws["financial_stability_latent"],
        strength,
        systemic_config["opportunity_model"],
    )
    access = draws["opportunity_uniform"].to_numpy(dtype=float) <= probability
    access_float = access.astype(float)
    stability = draws["financial_stability_latent"].to_numpy(dtype=float)
    creditworthiness = draws["creditworthiness_latent"].to_numpy(dtype=float)
    age = draws["age_years"].to_numpy(dtype=float)

    employment_spec = population["employment_years"]
    feasible_history = np.maximum(age - float(employment_spec["entry_age"]), 0.0)
    employment_fraction = np.clip(
        float(employment_spec["fraction_intercept"])
        + float(employment_spec["stability_coefficient"]) * stability
        + draws["employment_residual"].to_numpy(dtype=float),
        float(employment_spec["minimum_fraction"]),
        float(employment_spec["maximum_fraction"]),
    )
    employment_years = np.clip(
        feasible_history * employment_fraction
        + float(effects["employment_years_additive"]) * access_float,
        0.0,
        feasible_history,
    )

    income_spec = population["annual_income"]
    annual_income = _clipped(
        np.exp(
            np.log(float(income_spec["reference_median"]))
            + float(income_spec["stability_coefficient"]) * stability
            + float(income_spec["employment_coefficient"]) * (employment_years - 10.0)
            + float(income_spec["creditworthiness_coefficient"]) * creditworthiness
            + draws["income_residual"].to_numpy(dtype=float)
            + float(effects["log_annual_income_additive"]) * access_float
        ),
        income_spec,
    )

    credit_spec = population["credit_score"]
    credit_score = np.rint(
        _clipped(
            float(credit_spec["reference_mean"])
            + float(credit_spec["creditworthiness_coefficient"]) * creditworthiness
            + float(credit_spec["stability_coefficient"]) * stability
            + draws["credit_residual"].to_numpy(dtype=float),
            credit_spec,
        )
    ).astype(np.int16)

    assets_spec = population["liquid_assets"]
    liquid_assets = _clipped(
        np.exp(
            np.log(float(assets_spec["reference_median"]))
            + float(assets_spec["income_elasticity"])
            * np.log(annual_income / float(income_spec["reference_median"]))
            + float(assets_spec["stability_coefficient"]) * stability
            + float(assets_spec["age_coefficient"]) * (age - 40.0)
            + draws["assets_residual"].to_numpy(dtype=float)
            + float(effects["log_liquid_assets_additive"]) * access_float
        ),
        assets_spec,
    )

    debt_spec = population["existing_monthly_debt"]
    debt_logit = (
        float(debt_spec["logit_intercept"])
        + float(debt_spec["stability_coefficient"]) * stability
        + float(debt_spec["creditworthiness_coefficient"]) * creditworthiness
        + draws["debt_residual"].to_numpy(dtype=float)
    )
    debt_unit = np.asarray(sigmoid(debt_logit), dtype=float)
    debt_share = float(debt_spec["minimum_income_share"]) + debt_unit * (
        float(debt_spec["maximum_income_share"])
        - float(debt_spec["minimum_income_share"])
    )
    existing_monthly_debt = debt_share * annual_income / 12.0

    property_spec = population["property_value"]
    property_value = _clipped(
        np.exp(
            np.log(float(property_spec["reference_median"]))
            + float(property_spec["income_elasticity"])
            * np.log(annual_income / float(income_spec["reference_median"]))
            + float(property_spec["stability_coefficient"]) * stability
            + draws["property_residual"].to_numpy(dtype=float)
        ),
        property_spec,
    )
    ltv_spec = population["requested_ltv"]
    requested_ltv = _clipped(
        float(ltv_spec["intercept"])
        + float(ltv_spec["asset_to_property_coefficient"]) * (liquid_assets / property_value)
        + float(ltv_spec["stability_coefficient"]) * stability
        + float(ltv_spec["creditworthiness_coefficient"]) * creditworthiness
        + draws["ltv_residual"].to_numpy(dtype=float),
        ltv_spec,
    )
    requested_loan = property_value * requested_ltv

    applicants = pd.DataFrame(
        {
            "applicant_id": draws["applicant_id"].to_numpy(),
            "group": pd.Categorical(draws["group"], categories=["A", "B"]),
            "cohort": pd.Categorical(
                draws["cohort"],
                categories=["historical_train", "historical_validation", "evaluation"],
            ),
            "age_years": draws["age_years"].to_numpy(dtype=np.int16),
            "annual_income": np.round(annual_income, 2),
            "credit_score": credit_score,
            "employment_years": np.round(employment_years, 3),
            "liquid_assets": np.round(liquid_assets, 2),
            "existing_monthly_debt": np.round(existing_monthly_debt, 2),
            "property_value": np.round(property_value, 2),
            "requested_loan_amount": np.round(requested_loan, 2),
        }
    ).loc[:, APPLICANTS_COLUMNS]
    mechanism = draws.loc[
        :,
        [
            "applicant_id",
            "financial_stability_latent",
            "creditworthiness_latent",
            "opportunity_uniform",
        ],
    ].copy()
    mechanism.insert(1, "systemic_strength", float(strength))
    mechanism["opportunity_probability"] = probability
    mechanism["opportunity_access"] = access.astype(np.int8)
    return applicants, mechanism


def generate_systemic_world(
    draws: pd.DataFrame,
    baseline_config: dict[str, Any],
    systemic_config: dict[str, Any],
    strength: float,
) -> dict[str, pd.DataFrame]:
    """Build applicant and contract descendants for one matched opportunity world."""

    applicants, mechanism = generate_systemic_applicants(
        draws, baseline_config, systemic_config, strength
    )
    loans = make_requested_loan_options(applicants, baseline_config["contract"])
    return {"applicants": applicants, "loan_options": loans, "mechanism": mechanism}


def identify_opportunity_switchers(
    neutral_mechanism: pd.DataFrame,
    treated_mechanism: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """Identify Group B applicants whose access changes from one to zero."""

    paired = (
        neutral_mechanism[["applicant_id", "opportunity_access"]]
        .rename(columns={"opportunity_access": "opportunity_access_neutral"})
        .merge(
            treated_mechanism[["applicant_id", "opportunity_access"]].rename(
                columns={"opportunity_access": "opportunity_access_systemic"}
            ),
            on="applicant_id",
            validate="one_to_one",
        )
        .merge(groups[["applicant_id", "group"]], on="applicant_id", validate="one_to_one")
    )
    paired["upstream_displaced"] = (
        paired["group"].astype(str).eq("B")
        & paired["opportunity_access_neutral"].eq(1)
        & paired["opportunity_access_systemic"].eq(0)
    )
    return paired


def validate_matched_worlds(
    draws: pd.DataFrame,
    neutral: dict[str, pd.DataFrame],
    treated: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Enforce paired inputs, exact Group A invariance, and accounting identities."""

    if draws["applicant_id"].duplicated().any():
        raise ValueError("pre-treatment applicant IDs must be unique")
    first = neutral["applicants"].merge(
        treated["applicants"], on="applicant_id", suffixes=("_neutral", "_systemic")
    )
    group_a = first["group_neutral"].astype(str).eq("A")
    comparison_columns = [
        "age_years",
        "annual_income",
        "credit_score",
        "employment_years",
        "liquid_assets",
        "existing_monthly_debt",
        "property_value",
        "requested_loan_amount",
    ]
    for column in comparison_columns:
        if not np.array_equal(
            first.loc[group_a, f"{column}_neutral"].to_numpy(),
            first.loc[group_a, f"{column}_systemic"].to_numpy(),
        ):
            raise AssertionError(f"Group A {column} changed across matched worlds")
    for world in (neutral, treated):
        joined = world["applicants"].merge(world["loan_options"], on="applicant_id")
        np.testing.assert_allclose(
            joined["requested_ltv"],
            joined["requested_loan_amount"] / joined["property_value"],
        )
        np.testing.assert_allclose(
            joined["first_period_dti"],
            (joined["existing_monthly_debt"] + joined["first_period_payment"])
            / (joined["annual_income"] / 12.0),
        )
    return {
        "group_a_rows": int(group_a.sum()),
        "group_a_exactly_invariant": True,
        "pre_treatment_rows": int(len(draws)),
        "dti_ltv_identities_valid": True,
    }


def controlled_funding_audit(
    decisions: pd.DataFrame,
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
    true_risk_config: dict[str, Any],
) -> pd.DataFrame:
    """Fit descriptive U0/U1 funding logits; U1 conditions on descendants."""

    source = (
        applicants.merge(
            loan_options[["applicant_id", "first_period_dti", "requested_ltv"]],
            on="applicant_id",
            validate="one_to_one",
        )
        .merge(decisions[["applicant_id", "approved"]], on="applicant_id", validate="one_to_one")
    )
    group_b = source["group"].astype(str).eq("B").astype(float)
    transformed = transform_observable_risk_features(source, true_risk_config)
    designs = {
        "U0_race_only": pd.DataFrame({"group_B": group_b}),
        "U1_downstream_controls": pd.concat(
            [pd.DataFrame({"group_B": group_b}), transformed.loc[:, TRUE_RISK_PREDICTORS]],
            axis=1,
        ),
    }
    rows: list[dict[str, Any]] = []
    for model_id, design in designs.items():
        if SYSTEMIC_LENDER_FORBIDDEN.intersection(design.columns):
            raise AssertionError("mechanism or group label leaked beyond the declared indicator")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = sm.GLM(
                source["approved"].astype(int),
                sm.add_constant(design, has_constant="add"),
                family=sm.families.Binomial(),
            ).fit(maxiter=100, disp=0)
        raw_coefficient = float(fitted.params["group_B"])
        raw_standard_error = float(fitted.bse["group_B"])
        separated = any(issubclass(item.category, PerfectSeparationWarning) for item in caught)
        stable = bool(
            fitted.converged
            and np.isfinite(raw_coefficient)
            and np.isfinite(raw_standard_error)
            and abs(raw_coefficient) < 50.0
            and raw_standard_error < 50.0
            and not separated
        )
        ci = fitted.conf_int().loc["group_B"] if stable else pd.Series([np.nan, np.nan])
        rows.append(
            {
                "model": model_id,
                "n": int(len(source)),
                "group_b_coefficient": raw_coefficient if stable else np.nan,
                "group_b_standard_error": raw_standard_error if stable else np.nan,
                "group_b_p_value": float(fitted.pvalues["group_B"]) if stable else np.nan,
                "group_b_ci_low": float(ci.iloc[0]),
                "group_b_ci_high": float(ci.iloc[1]),
                "converged": bool(fitted.converged),
                "identified": stable,
                "identification_status": "identified" if stable else "perfect_or_quasi_separation",
                "raw_group_b_coefficient": raw_coefficient,
                "raw_group_b_standard_error": raw_standard_error,
            }
        )
    return pd.DataFrame(rows)
