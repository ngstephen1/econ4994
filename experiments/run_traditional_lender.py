"""Fit and evaluate the Version 2 traditional lender risk model."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fair_lending.economic_lending.config import PROJECT_ROOT, load_economic_config
from fair_lending.economic_lending.history import build_at_risk_payment_history
from fair_lending.economic_lending.lender_evaluation import (
    calibration_table,
    group_risk_audit,
    probability_quantiles,
    probability_recovery_metrics,
    profit_evaluation_table,
    profit_estimation_diagnostics,
    realized_label_metrics,
    survival_selection_diagnostics,
)
from fair_lending.economic_lending.traditional import (
    TRADITIONAL_LENDER_FEATURES,
    build_policy_assessments,
    fit_linear_probability_model,
    fit_traditional_logit,
    predict_applicant_risk,
)
from fair_lending.economic_lending.schema import SCHEMA_VERSION


def _revision() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty_worktree": dirty}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _coefficient_table(model, true_risk_config: dict[str, Any]) -> pd.DataFrame:
    true_coefficients = {"intercept": float(true_risk_config["intercept"])}
    true_coefficients.update(
        {
            feature: float(true_risk_config["predictors"][feature]["coefficient"])
            for feature in TRADITIONAL_LENDER_FEATURES
        }
    )
    rows = []
    for feature, true_value in true_coefficients.items():
        fitted = model.coefficients[feature]
        rows.append(
            {
                "term": feature,
                "true_coefficient": true_value,
                "fitted_coefficient": fitted,
                "difference": fitted - true_value,
                "relative_error": (fitted - true_value) / true_value
                if true_value != 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    config = load_economic_config()
    data_directory = PROJECT_ROOT / config["artifacts"]["data_directory"]
    applicants = pd.read_parquet(data_directory / "applicants.parquet")
    loans = pd.read_parquet(data_directory / "loan_options.parquet")
    truth = pd.read_parquet(data_directory / "simulation_truth.parquet")
    outcomes = pd.read_parquet(data_directory / "loan_outcomes.parquet")

    history = build_at_risk_payment_history(applicants, loans, outcomes)
    histories = {
        cohort: history.loc[history["cohort"] == cohort].reset_index(drop=True)
        for cohort in ("historical_train", "historical_validation", "evaluation")
    }
    applicant_sets = {
        cohort: applicants.loc[applicants["cohort"] == cohort].reset_index(drop=True)
        for cohort in histories
    }
    loan_sets = {
        cohort: loans.loc[loans["applicant_id"].isin(frame["applicant_id"])].reset_index(drop=True)
        for cohort, frame in applicant_sets.items()
    }
    truth_sets = {
        cohort: truth.loc[truth["applicant_id"].isin(frame["applicant_id"])].reset_index(drop=True)
        for cohort, frame in applicant_sets.items()
    }

    primary = fit_traditional_logit(histories["historical_train"], config["true_risk"])
    weighted = fit_traditional_logit(
        histories["historical_train"], config["true_risk"], applicant_weighted=True
    )
    lpm = fit_linear_probability_model(histories["historical_train"], config["true_risk"])

    primary_predictions = {
        cohort: predict_applicant_risk(
            primary, applicant_sets[cohort], loan_sets[cohort], config["true_risk"]
        )
        for cohort in ("historical_validation", "evaluation")
    }
    weighted_predictions = {
        cohort: predict_applicant_risk(
            weighted,
            applicant_sets[cohort],
            loan_sets[cohort],
            config["true_risk"],
            probability_column="repayment_probability_applicant_weighted",
        )
        for cohort in ("historical_validation", "evaluation")
    }
    lpm_predictions = {
        cohort: predict_applicant_risk(
            lpm,
            applicant_sets[cohort],
            loan_sets[cohort],
            config["true_risk"],
            probability_column="repayment_probability_lpm_raw",
        )
        for cohort in ("historical_validation", "evaluation")
    }

    realized: dict[str, Any] = {}
    oracle: dict[str, Any] = {}
    weighted_metrics: dict[str, Any] = {}
    lpm_metrics: dict[str, Any] = {}
    for cohort in ("historical_validation", "evaluation"):
        realized[cohort] = realized_label_metrics(
            histories[cohort],
            primary_predictions[cohort],
            "repayment_probability_traditional",
        )
        oracle_probabilities = truth_sets[cohort].rename(
            columns={
                "repayment_probability_per_period_true": "true_risk_oracle_probability"
            }
        )
        oracle[cohort] = realized_label_metrics(
            histories[cohort], oracle_probabilities, "true_risk_oracle_probability"
        )
        weighted_metrics[cohort] = {
            "realized_labels": realized_label_metrics(
                histories[cohort],
                weighted_predictions[cohort],
                "repayment_probability_applicant_weighted",
            ),
            "truth_recovery": probability_recovery_metrics(
                weighted_predictions[cohort],
                truth_sets[cohort],
                "repayment_probability_applicant_weighted",
            ),
        }
        raw_lpm = lpm_predictions[cohort]["repayment_probability_lpm_raw"]
        clipped = lpm_predictions[cohort].copy()
        clipped["repayment_probability_lpm_clipped"] = raw_lpm.clip(0.0, 1.0)
        lpm_metrics[cohort] = {
            "raw_prediction_minimum": float(raw_lpm.min()),
            "raw_prediction_maximum": float(raw_lpm.max()),
            "fraction_below_zero": float((raw_lpm < 0).mean()),
            "fraction_above_one": float((raw_lpm > 1).mean()),
            "clipped_realized_labels": realized_label_metrics(
                histories[cohort], clipped, "repayment_probability_lpm_clipped"
            ),
            "clipped_truth_recovery": probability_recovery_metrics(
                clipped, truth_sets[cohort], "repayment_probability_lpm_clipped"
            ),
        }

    evaluation_predictions = primary_predictions["evaluation"]
    evaluation_policy = build_policy_assessments(
        loan_sets["evaluation"],
        evaluation_predictions,
        policy_id=config["traditional_lender"]["policy_id"],
    )
    policy_path = PROJECT_ROOT / config["artifacts"]["policy_assessments"]
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_policy.to_parquet(policy_path, index=False)

    group_audit = group_risk_audit(
        applicant_sets["evaluation"],
        evaluation_predictions,
        truth_sets["evaluation"],
        "repayment_probability_traditional",
    )
    profit_overall, profit_by_group = profit_estimation_diagnostics(
        loan_sets["evaluation"],
        evaluation_policy,
        truth_sets["evaluation"],
        applicant_sets["evaluation"],
    )
    profit_detail = profit_evaluation_table(
        loan_sets["evaluation"],
        evaluation_policy,
        truth_sets["evaluation"],
        applicant_sets["evaluation"],
    )
    calibration = calibration_table(
        histories["evaluation"],
        evaluation_predictions,
        truth_sets["evaluation"],
        "repayment_probability_traditional",
        n_bins=int(config["traditional_lender"]["calibration_bins"]),
    )
    coefficients = _coefficient_table(primary, config["true_risk"])
    selection = survival_selection_diagnostics(
        histories["historical_train"],
        applicant_sets["historical_train"],
        loan_sets["historical_train"],
        truth_sets["historical_train"],
    )

    model_metadata = {
        "policy_id": config["traditional_lender"]["policy_id"],
        "model_family": primary.model_family,
        "training_cohort": "historical_train",
        "validation_cohort": "historical_validation",
        "evaluation_cohort_excluded_from_fit_and_selection": True,
        "features": list(primary.features),
        "transforms": config["true_risk"]["predictors"],
        "coefficients": primary.coefficients,
        "regularization": "none",
        "settings": {"link": "logit", "period_index_used": False},
        "training_applicant_count": int(applicant_sets["historical_train"].shape[0]),
        "training_payment_row_count": int(histories["historical_train"].shape[0]),
        "config_fingerprint": config["metadata"]["config_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "code_revision": _revision(),
        "random_state": None,
        "sensitivity_coefficients": {
            "applicant_weighted_logit": weighted.coefficients,
            "linear_probability_model": lpm.coefficients,
        },
    }
    _write_json(PROJECT_ROOT / config["artifacts"]["traditional_model"], model_metadata)

    metrics = {
        "payment_history": {
            "all_rows": int(len(history)),
            "rows_by_cohort": {
                cohort: int(len(frame)) for cohort, frame in histories.items()
            },
            "applicants_by_cohort": {
                cohort: int(len(frame)) for cohort, frame in applicant_sets.items()
            },
        },
        "primary_model": {
            "realized_label_metrics": realized,
            "evaluation_truth_recovery": probability_recovery_metrics(
                evaluation_predictions,
                truth_sets["evaluation"],
                "repayment_probability_traditional",
            ),
            "evaluation_probability_quantiles": probability_quantiles(
                evaluation_predictions,
                truth_sets["evaluation"],
                "repayment_probability_traditional",
            ),
        },
        "oracle_realized_label_metrics": oracle,
        "survival_selection": selection,
        "applicant_weighted_sensitivity": weighted_metrics,
        "linear_probability_sensitivity": lpm_metrics,
        "profit_estimation": profit_overall,
    }
    _write_json(PROJECT_ROOT / config["artifacts"]["traditional_metrics"], metrics)

    tables_directory = PROJECT_ROOT / "results" / "tables"
    tables_directory.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(tables_directory / "traditional_lender_coefficients.csv", index=False)
    calibration.to_csv(tables_directory / "traditional_lender_calibration.csv", index=False)
    group_audit.to_csv(tables_directory / "traditional_lender_group_audit.csv", index=False)
    profit_by_group.to_csv(
        tables_directory / "traditional_lender_profit_by_group.csv", index=False
    )
    profit_detail.to_csv(
        tables_directory / "traditional_lender_profit_evaluation.csv", index=False
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
