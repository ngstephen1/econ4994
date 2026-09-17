"""Compare the frozen traditional lender with one flexible repayment-risk model."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from fair_lending.economic_lending.config import PROJECT_ROOT, load_economic_config
from fair_lending.economic_lending.history import build_at_risk_payment_history
from fair_lending.economic_lending.lender_evaluation import (
    break_even_disagreement_table,
    calibration_table,
    group_risk_audit,
    paired_prediction_comparison,
    probability_error_quantiles,
    probability_recovery_metrics,
    profit_estimation_diagnostics,
    profit_evaluation_table,
    realized_label_metrics,
)
from fair_lending.economic_lending.ml import (
    ML_FEATURES,
    load_ml_config,
    ml_feature_matrix,
    predict_ml_applicant_risk,
    tune_hist_gradient_boosting,
)
from fair_lending.economic_lending.ml_figures import (
    save_calibration_plot,
    save_profit_error_distribution,
    save_risk_error_distribution,
    save_risk_truth_scatter,
)
from fair_lending.economic_lending.schema import SCHEMA_VERSION
from fair_lending.economic_lending.traditional import (
    FittedTraditionalModel,
    build_policy_assessments,
    predict_applicant_risk,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _revision() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {"commit": commit, "dirty_worktree": dirty}


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    baseline_config = load_economic_config()
    ml_config = load_ml_config()
    data_directory = PROJECT_ROOT / baseline_config["artifacts"]["data_directory"]
    applicants = pd.read_parquet(data_directory / "applicants.parquet")
    loans = pd.read_parquet(data_directory / "loan_options.parquet")
    truth = pd.read_parquet(data_directory / "simulation_truth.parquet")
    outcomes = pd.read_parquet(data_directory / "loan_outcomes.parquet")
    history = build_at_risk_payment_history(applicants, loans, outcomes)

    cohorts = ("historical_train", "historical_validation", "evaluation")
    histories = {
        cohort: history.loc[history["cohort"] == cohort].reset_index(drop=True)
        for cohort in cohorts
    }
    applicant_sets = {
        cohort: applicants.loc[applicants["cohort"] == cohort].reset_index(drop=True)
        for cohort in cohorts
    }
    loan_sets = {
        cohort: loans.loc[loans["applicant_id"].isin(frame["applicant_id"])].reset_index(drop=True)
        for cohort, frame in applicant_sets.items()
    }
    truth_sets = {
        cohort: truth.loc[truth["applicant_id"].isin(frame["applicant_id"])].reset_index(drop=True)
        for cohort, frame in applicant_sets.items()
    }

    selection = tune_hist_gradient_boosting(
        histories["historical_train"], histories["historical_validation"], ml_config
    )
    ml_predictions = {
        cohort: predict_ml_applicant_risk(
            selection.estimator, applicant_sets[cohort], loan_sets[cohort]
        )
        for cohort in ("historical_validation", "evaluation")
    }

    traditional_metadata_path = PROJECT_ROOT / baseline_config["artifacts"]["traditional_model"]
    traditional_metadata = json.loads(traditional_metadata_path.read_text(encoding="utf-8"))
    if traditional_metadata["config_fingerprint"] != baseline_config["metadata"]["config_fingerprint"]:
        raise RuntimeError("traditional model and population configuration fingerprints differ")
    traditional_model = FittedTraditionalModel(
        model_family=traditional_metadata["model_family"],
        coefficients=traditional_metadata["coefficients"],
        features=tuple(traditional_metadata["features"]),
    )
    traditional_predictions = {
        cohort: predict_applicant_risk(
            traditional_model,
            applicant_sets[cohort],
            loan_sets[cohort],
            baseline_config["true_risk"],
        )
        for cohort in ("historical_validation", "evaluation")
    }

    realized_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    for cohort in ("historical_validation", "evaluation"):
        oracle = truth_sets[cohort].rename(
            columns={"repayment_probability_per_period_true": "oracle_probability"}
        )
        model_inputs = [
            ("traditional_logit_v1", traditional_predictions[cohort], "repayment_probability_traditional"),
            ("ml_histgb_v1", ml_predictions[cohort], "repayment_probability_ml"),
            ("oracle", oracle, "oracle_probability"),
        ]
        for model_name, predictions, probability_column in model_inputs:
            realized_rows.append(
                {
                    "cohort": cohort,
                    "model": model_name,
                    **realized_label_metrics(
                        histories[cohort], predictions, probability_column
                    ),
                }
            )
            if model_name != "oracle":
                recovery_rows.append(
                    {
                        "cohort": cohort,
                        "model": model_name,
                        **probability_recovery_metrics(
                            predictions, truth_sets[cohort], probability_column
                        ),
                        **{
                            f"error_{key}": value
                            for key, value in probability_error_quantiles(
                                predictions, truth_sets[cohort], probability_column
                            ).items()
                        },
                    }
                )
        if cohort == "evaluation":
            for model_name, predictions, probability_column in model_inputs:
                frame = calibration_table(
                    histories[cohort],
                    predictions,
                    truth_sets[cohort],
                    probability_column,
                    n_bins=10,
                )
                frame.insert(0, "model", model_name)
                calibration_frames.append(frame)

    realized_table = pd.DataFrame(realized_rows)
    recovery_table = pd.DataFrame(recovery_rows)
    calibration = pd.concat(calibration_frames, ignore_index=True)

    evaluation_predictions = traditional_predictions["evaluation"].merge(
        ml_predictions["evaluation"], on="applicant_id", validate="one_to_one"
    )
    paired = paired_prediction_comparison(
        traditional_predictions["evaluation"], ml_predictions["evaluation"]
    )

    traditional_policy_path = PROJECT_ROOT / baseline_config["artifacts"]["policy_assessments"]
    saved_policy = pd.read_parquet(traditional_policy_path)
    traditional_policy = saved_policy.loc[
        saved_policy["policy_id"] == "traditional_logit_v1"
    ].reset_index(drop=True)
    if len(traditional_policy) != len(applicant_sets["evaluation"]):
        raise RuntimeError("frozen traditional policy assessment is missing evaluation rows")
    ml_policy = build_policy_assessments(
        loan_sets["evaluation"],
        ml_predictions["evaluation"],
        policy_id=ml_config["policy_id"],
        probability_column="repayment_probability_ml",
    )
    combined_policy = pd.concat([traditional_policy, ml_policy], ignore_index=True).sort_values(
        ["applicant_id", "policy_id"], kind="mergesort"
    )
    if any("true" in column for column in combined_policy.columns):
        raise AssertionError("hidden truth entered policy assessments")
    combined_policy.to_parquet(traditional_policy_path, index=False)

    traditional_profit, traditional_profit_group = profit_estimation_diagnostics(
        loan_sets["evaluation"], traditional_policy, truth_sets["evaluation"], applicant_sets["evaluation"]
    )
    ml_profit, ml_profit_group = profit_estimation_diagnostics(
        loan_sets["evaluation"], ml_policy, truth_sets["evaluation"], applicant_sets["evaluation"]
    )
    profit_comparison = pd.DataFrame(
        [
            {"model": "traditional_logit_v1", **traditional_profit},
            {"model": "ml_histgb_v1", **ml_profit},
        ]
    )
    break_even = break_even_disagreement_table(
        traditional_policy, ml_policy, loan_sets["evaluation"], truth_sets["evaluation"]
    )

    traditional_group = group_risk_audit(
        applicant_sets["evaluation"], traditional_predictions["evaluation"], truth_sets["evaluation"],
        "repayment_probability_traditional",
    ).rename(
        columns={
            "mean_predicted_rho": "mean_predicted_traditional_rho",
            "mean_prediction_error": "traditional_mean_error",
            "mae": "traditional_mae",
            "rmse": "traditional_rmse",
        }
    )
    ml_group = group_risk_audit(
        applicant_sets["evaluation"], ml_predictions["evaluation"], truth_sets["evaluation"],
        "repayment_probability_ml",
    ).rename(
        columns={
            "n_applicants": "ml_n_applicants",
            "mean_predicted_rho": "mean_predicted_ml_rho",
            "mean_true_rho": "ml_mean_true_rho",
            "mean_prediction_error": "ml_mean_error",
            "mae": "ml_mae",
            "rmse": "ml_rmse",
        }
    )
    group_audit = traditional_group.merge(ml_group, on="group", validate="one_to_one")
    group_error_differences = {
        "traditional_B_minus_A_prediction_error": float(
            group_audit.set_index("group").loc["B", "traditional_mean_error"]
            - group_audit.set_index("group").loc["A", "traditional_mean_error"]
        ),
        "ml_B_minus_A_prediction_error": float(
            group_audit.set_index("group").loc["B", "ml_mean_error"]
            - group_audit.set_index("group").loc["A", "ml_mean_error"]
        ),
    }

    importance_result = permutation_importance(
        selection.estimator,
        ml_feature_matrix(histories["historical_validation"]),
        histories["historical_validation"]["paid_this_period"].to_numpy(dtype=int),
        scoring=ml_config["permutation_importance"]["scoring"],
        n_repeats=int(ml_config["permutation_importance"]["n_repeats"]),
        random_state=int(ml_config["random_state"]),
    )
    importance = pd.DataFrame(
        {
            "feature": ML_FEATURES,
            "mean_log_loss_increase": importance_result.importances_mean,
            "standard_deviation": importance_result.importances_std,
        }
    ).sort_values("mean_log_loss_increase", ascending=False, kind="mergesort")

    model_path = PROJECT_ROOT / ml_config["artifacts"]["model_joblib"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selection.estimator, model_path)
    model_metadata = {
        "policy_id": ml_config["policy_id"],
        "model_family": ml_config["model_family"],
        "features": list(ML_FEATURES),
        "selected_parameters": {
            key: _native(value) for key, value in selection.selected_parameters.items()
        },
        "fixed_parameters": ml_config["fixed_parameters"],
        "selection_metric": ml_config["selection"]["primary_metric"],
        "training_cohort": "historical_train",
        "validation_cohort": "historical_validation",
        "evaluation_cohort_excluded_from_selection": True,
        "training_payment_rows": int(len(histories["historical_train"])),
        "validation_payment_rows": int(len(histories["historical_validation"])),
        "baseline_config_fingerprint": baseline_config["metadata"]["config_fingerprint"],
        "ml_config_fingerprint": ml_config["metadata"]["config_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "code_revision": _revision(),
        "random_state": int(ml_config["random_state"]),
    }
    _write_json(PROJECT_ROOT / ml_config["artifacts"]["model_metadata"], model_metadata)

    metrics = {
        "payment_rows": {cohort: int(len(frame)) for cohort, frame in histories.items()},
        "selected_parameters": model_metadata["selected_parameters"],
        "realized_label_metrics": realized_table.to_dict(orient="records"),
        "truth_recovery": recovery_table.to_dict(orient="records"),
        "paired_prediction_comparison": paired,
        "group_prediction_error_differences": group_error_differences,
        "profit_comparison": profit_comparison.to_dict(orient="records"),
        "break_even_disagreement": break_even.to_dict(orient="records"),
        "feature_importance": importance.to_dict(orient="records"),
    }
    _write_json(PROJECT_ROOT / ml_config["artifacts"]["metrics"], metrics)

    tables_directory = PROJECT_ROOT / "results" / "tables"
    figures_directory = PROJECT_ROOT / "results" / "figures"
    tables_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)
    risk_comparison = realized_table.merge(
        recovery_table,
        on=["cohort", "model"],
        how="left",
        validate="one_to_one",
    )
    risk_comparison.to_csv(tables_directory / "v2_risk_model_comparison.csv", index=False)
    group_audit.to_csv(tables_directory / "v2_risk_group_audit.csv", index=False)
    profit_comparison.to_csv(tables_directory / "v2_profit_model_comparison.csv", index=False)
    selection.trials.to_csv(tables_directory / "v2_ml_hyperparameter_selection.csv", index=False)
    calibration.to_csv(tables_directory / "v2_risk_calibration_bins.csv", index=False)
    break_even.to_csv(tables_directory / "v2_break_even_disagreement.csv", index=False)
    importance.to_csv(tables_directory / "v2_ml_permutation_importance.csv", index=False)

    comparison = (
        applicant_sets["evaluation"].loc[:, ["applicant_id", "group"]]
        .merge(evaluation_predictions, on="applicant_id", validate="one_to_one")
        .merge(truth_sets["evaluation"], on="applicant_id", validate="one_to_one")
    )
    comparison["traditional_risk_error"] = (
        comparison["repayment_probability_traditional"]
        - comparison["repayment_probability_per_period_true"]
    )
    comparison["ml_risk_error"] = (
        comparison["repayment_probability_ml"]
        - comparison["repayment_probability_per_period_true"]
    )
    traditional_profit_detail = profit_evaluation_table(
        loan_sets["evaluation"], traditional_policy, truth_sets["evaluation"], applicant_sets["evaluation"]
    ).assign(model="traditional_logit_v1")
    ml_profit_detail = profit_evaluation_table(
        loan_sets["evaluation"], ml_policy, truth_sets["evaluation"], applicant_sets["evaluation"]
    ).assign(model="ml_histgb_v1")
    profit_errors = pd.concat([traditional_profit_detail, ml_profit_detail], ignore_index=True)
    save_risk_truth_scatter(comparison, figures_directory / "v2_risk_predictions_vs_truth.png")
    save_calibration_plot(calibration, figures_directory / "v2_risk_calibration.png")
    save_risk_error_distribution(comparison, figures_directory / "v2_risk_error_distribution.png")
    save_profit_error_distribution(profit_errors, figures_directory / "v2_profit_error_distribution.png")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
