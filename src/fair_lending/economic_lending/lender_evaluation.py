"""Evaluator-only diagnostics for traditional lender risk estimates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from fair_lending.economic_lending.profit import expected_profit_from_options


def realized_label_metrics(
    payment_history: pd.DataFrame,
    applicant_probabilities: pd.DataFrame,
    probability_column: str,
) -> dict[str, float]:
    """Evaluate next-payment predictions against realized at-risk labels."""

    scored = payment_history.loc[:, ["applicant_id", "paid_this_period"]].merge(
        applicant_probabilities.loc[:, ["applicant_id", probability_column]],
        on="applicant_id",
        how="inner",
        validate="many_to_one",
    )
    target = scored["paid_this_period"].to_numpy(dtype=int)
    probability = scored[probability_column].to_numpy(dtype=float)
    return {
        "n_payment_rows": int(len(scored)),
        "brier_score": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, probability))
        if np.unique(target).size == 2
        else float("nan"),
        "average_precision": float(average_precision_score(target, probability))
        if np.unique(target).size == 2
        else float("nan"),
    }


def probability_recovery_metrics(
    predictions: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    probability_column: str,
) -> dict[str, float]:
    """Compare lender estimates with evaluator-only hidden conditional truth."""

    joined = predictions.merge(simulation_truth, on="applicant_id", validate="one_to_one")
    error = (
        joined[probability_column]
        - joined["repayment_probability_per_period_true"]
    )
    return {
        "n_applicants": int(len(joined)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_prediction_error": float(error.mean()),
        "correlation_with_true_rho": float(
            joined[[probability_column, "repayment_probability_per_period_true"]]
            .corr()
            .iloc[0, 1]
        ),
    }


def probability_quantiles(
    predictions: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    probability_column: str,
) -> dict[str, dict[str, float]]:
    joined = predictions.merge(simulation_truth, on="applicant_id", validate="one_to_one")
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    return {
        "predicted": {
            f"p{int(q * 100):02d}": float(joined[probability_column].quantile(q))
            for q in quantiles
        },
        "true": {
            f"p{int(q * 100):02d}": float(
                joined["repayment_probability_per_period_true"].quantile(q)
            )
            for q in quantiles
        },
    }


def probability_error_quantiles(
    predictions: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    probability_column: str,
) -> dict[str, float]:
    """Return quantiles of predicted conditional probability minus truth."""

    joined = predictions.merge(simulation_truth, on="applicant_id", validate="one_to_one")
    error = joined[probability_column] - joined["repayment_probability_per_period_true"]
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    return {f"p{int(q * 100):02d}": float(error.quantile(q)) for q in quantiles}


def paired_prediction_comparison(
    traditional: pd.DataFrame,
    ml: pd.DataFrame,
) -> dict[str, float]:
    """Summarize paired ML-minus-traditional applicant risk estimates."""

    joined = traditional.merge(ml, on="applicant_id", validate="one_to_one")
    difference = (
        joined["repayment_probability_ml"]
        - joined["repayment_probability_traditional"]
    )
    return {
        "mean_ml_minus_traditional": float(difference.mean()),
        "median_ml_minus_traditional": float(difference.median()),
        "standard_deviation": float(difference.std(ddof=1)),
        "p05": float(difference.quantile(0.05)),
        "p95": float(difference.quantile(0.95)),
        "maximum_absolute_difference": float(difference.abs().max()),
        "prediction_correlation": float(
            joined[
                ["repayment_probability_traditional", "repayment_probability_ml"]
            ].corr().iloc[0, 1]
        ),
    }


def break_even_disagreement_table(
    traditional_policy: pd.DataFrame,
    ml_policy: pd.DataFrame,
    loan_options: pd.DataFrame,
    simulation_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Create the 2x2 perceived break-even table with evaluator-only true profit."""

    truth = expected_profit_from_options(loan_options, simulation_truth)
    joined = (
        traditional_policy.loc[:, ["applicant_id", "expected_profit_perceived"]]
        .rename(columns={"expected_profit_perceived": "traditional_expected_profit"})
        .merge(
            ml_policy.loc[:, ["applicant_id", "expected_profit_perceived"]].rename(
                columns={"expected_profit_perceived": "ml_expected_profit"}
            ),
            on="applicant_id",
            validate="one_to_one",
        )
        .merge(truth, on="applicant_id", validate="one_to_one")
    )
    joined["traditional_classification"] = np.where(
        joined["traditional_expected_profit"] > 0, "positive", "nonpositive"
    )
    joined["ml_classification"] = np.where(
        joined["ml_expected_profit"] > 0, "positive", "nonpositive"
    )
    summarized = (
        joined.groupby(
            ["traditional_classification", "ml_classification"], observed=True
        )
        .agg(
            n_applicants=("applicant_id", "size"),
            mean_true_expected_profit=("expected_profit_true", "mean"),
        )
    )
    full_index = pd.MultiIndex.from_product(
        [["positive", "nonpositive"], ["positive", "nonpositive"]],
        names=["traditional_classification", "ml_classification"],
    )
    summarized = summarized.reindex(full_index)
    summarized["n_applicants"] = summarized["n_applicants"].fillna(0).astype(int)
    return summarized.reset_index()


def calibration_table(
    payment_history: pd.DataFrame,
    predictions: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    probability_column: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    applicants = predictions.merge(simulation_truth, on="applicant_id", validate="one_to_one")
    applicants["calibration_bin"] = pd.qcut(
        applicants[probability_column], q=n_bins, duplicates="drop"
    ).astype(str)
    payment = payment_history.loc[:, ["applicant_id", "paid_this_period"]].merge(
        applicants.loc[:, ["applicant_id", "calibration_bin"]],
        on="applicant_id",
        validate="many_to_one",
    )
    applicant_summary = applicants.groupby("calibration_bin", observed=True).agg(
        n_applicants=("applicant_id", "size"),
        mean_predicted_rho=(probability_column, "mean"),
        mean_true_rho=("repayment_probability_per_period_true", "mean"),
    )
    payment_summary = payment.groupby("calibration_bin", observed=True).agg(
        n_payment_rows=("paid_this_period", "size"),
        observed_payment_rate=("paid_this_period", "mean"),
    )
    return applicant_summary.join(payment_summary).reset_index()


def group_risk_audit(
    applicants: pd.DataFrame,
    predictions: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    probability_column: str,
) -> pd.DataFrame:
    joined = (
        applicants.loc[:, ["applicant_id", "group"]]
        .merge(predictions, on="applicant_id", validate="one_to_one")
        .merge(simulation_truth, on="applicant_id", validate="one_to_one")
    )
    joined["prediction_error"] = (
        joined[probability_column] - joined["repayment_probability_per_period_true"]
    )
    joined["absolute_error"] = joined["prediction_error"].abs()
    joined["squared_error"] = joined["prediction_error"] ** 2
    audit = joined.groupby("group", observed=True).agg(
        n_applicants=("applicant_id", "size"),
        mean_predicted_rho=(probability_column, "mean"),
        mean_true_rho=("repayment_probability_per_period_true", "mean"),
        mean_prediction_error=("prediction_error", "mean"),
        mae=("absolute_error", "mean"),
        mean_squared_error=("squared_error", "mean"),
    ).reset_index()
    audit["rmse"] = np.sqrt(audit.pop("mean_squared_error"))
    return audit


def profit_estimation_diagnostics(
    loan_options: pd.DataFrame,
    policy_assessments: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    applicants: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compare perceived economics with evaluator-only true economics."""

    joined = profit_evaluation_table(
        loan_options, policy_assessments, simulation_truth, applicants
    )
    true_nonpositive = ~joined["true_positive_profit"]
    true_positive = joined["true_positive_profit"]
    overall = {
        "mean_error": float(joined["profit_error"].mean()),
        "mae": float(joined["profit_error"].abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(joined["profit_error"])))),
        "wrong_profit_sign_rate": float(joined["wrong_profit_sign"].mean()),
        "agreement_rate": float((~joined["wrong_profit_sign"]).mean()),
        "false_profitable_rate": float(
            joined.loc[true_nonpositive, "perceived_positive_profit"].mean()
        ),
        "false_unprofitable_rate": float(
            (~joined.loc[true_positive, "perceived_positive_profit"]).mean()
        ),
    }
    grouped_rows: list[dict[str, Any]] = []
    for group, frame in joined.groupby("group", observed=True):
        grouped_rows.append(
            {
                "group": str(group),
                "n_applicants": int(len(frame)),
                "mean_error": float(frame["profit_error"].mean()),
                "mae": float(frame["profit_error"].abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(frame["profit_error"])))),
                "wrong_profit_sign_rate": float(frame["wrong_profit_sign"].mean()),
            }
        )
    return overall, pd.DataFrame(grouped_rows)


def profit_evaluation_table(
    loan_options: pd.DataFrame,
    policy_assessments: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    applicants: pd.DataFrame,
) -> pd.DataFrame:
    """Create an evaluator-only row-level perceived-versus-true profit table."""

    truth = expected_profit_from_options(loan_options, simulation_truth)
    joined = (
        applicants.loc[:, ["applicant_id", "group"]]
        .merge(
            policy_assessments.loc[
                :, ["applicant_id", "expected_profit_perceived"]
            ],
            on="applicant_id",
            validate="one_to_one",
        )
        .merge(truth, on="applicant_id", validate="one_to_one")
    )
    joined["profit_error"] = (
        joined["expected_profit_perceived"] - joined["expected_profit_true"]
    )
    joined["true_positive_profit"] = joined["expected_profit_true"] > 0
    joined["perceived_positive_profit"] = joined["expected_profit_perceived"] > 0
    joined["wrong_profit_sign"] = (
        joined["true_positive_profit"] != joined["perceived_positive_profit"]
    )
    return joined


def survival_selection_diagnostics(
    training_history: pd.DataFrame,
    training_applicants: pd.DataFrame,
    training_loans: pd.DataFrame,
    training_truth: pd.DataFrame,
) -> dict[str, Any]:
    counts = training_history.groupby("applicant_id").size().rename("payment_rows")
    joined = (
        training_applicants.loc[:, ["applicant_id", "credit_score"]]
        .merge(
            training_loans.loc[:, ["applicant_id", "first_period_dti", "requested_ltv"]],
            on="applicant_id",
            validate="one_to_one",
        )
        .merge(training_truth, on="applicant_id", validate="one_to_one")
        .merge(counts, left_on="applicant_id", right_index=True, validate="one_to_one")
    )
    correlation_fields = [
        "repayment_probability_per_period_true",
        "credit_score",
        "first_period_dti",
        "requested_ltv",
    ]
    correlations = {
        field: float(joined["payment_rows"].corr(joined[field]))
        for field in correlation_fields
    }
    joined["rho_quintile"] = pd.qcut(
        joined["repayment_probability_per_period_true"], 5, labels=False, duplicates="drop"
    )
    by_quintile = (
        joined.groupby("rho_quintile", observed=True)
        .agg(
            n_applicants=("applicant_id", "size"),
            mean_rho=("repayment_probability_per_period_true", "mean"),
            mean_payment_rows=("payment_rows", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "mean_payment_rows_per_applicant": float(counts.mean()),
        "median_payment_rows_per_applicant": float(counts.median()),
        "correlation_with_payment_rows": correlations,
        "rho_quintile_summary": by_quintile,
    }
