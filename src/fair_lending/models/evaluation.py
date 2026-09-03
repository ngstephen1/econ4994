"""Overall predictive, probability-recovery, oracle, and calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def true_probability_recovery(
    predicted_probability: np.ndarray, true_probability: np.ndarray
) -> dict[str, float]:
    """Measure recovery of the DGP probability rather than its Bernoulli draw."""
    predicted = np.asarray(predicted_probability, dtype=float)
    truth = np.asarray(true_probability, dtype=float)
    if len(predicted) != len(truth):
        raise ValueError("Predicted and true probability arrays must have equal length")
    error = predicted - truth
    if len(predicted) < 2 or np.std(predicted) == 0.0 or np.std(truth) == 0.0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(predicted, truth)[0, 1])
    squared_error = float(np.mean(np.square(error)))
    return {
        "true_probability_mae": float(np.mean(np.abs(error))),
        "true_probability_rmse": float(np.sqrt(squared_error)),
        "true_probability_squared_error": squared_error,
        "true_probability_correlation": correlation,
        "mean_prediction_minus_true_probability": float(np.mean(error)),
    }


def evaluate_predictions(
    approved: np.ndarray,
    predicted_probability: np.ndarray,
    true_probability: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Evaluate one model on a held-out set."""
    y = np.asarray(approved, dtype=int)
    probability = np.asarray(predicted_probability, dtype=float)
    in_range = (0.0 <= probability) & (probability <= 1.0)
    if not np.isfinite(probability).all() or not in_range.all():
        raise ValueError("Predicted probabilities must be finite and in [0, 1]")
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "observed_approval_rate": float(np.mean(y)),
        "predicted_approval_rate": float(np.mean(predicted)),
        "mean_predicted_probability": float(np.mean(probability)),
        "threshold": float(threshold),
        **true_probability_recovery(probability, true_probability),
    }


def evaluate_oracle(test: pd.DataFrame, threshold: float = 0.50) -> dict[str, float]:
    """Evaluate the non-trained synthetic true probability reference."""
    values = test["approval_probability_true"].to_numpy(dtype=float)
    result = evaluate_predictions(
        test["approved"].to_numpy(dtype=int), values, values, threshold
    )
    result["benchmark"] = "synthetic_oracle_probability"
    return result


def calibration_bins(
    approved: np.ndarray,
    predicted_probability: np.ndarray,
    true_probability: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build equal-frequency reliability and true-probability recovery bins."""
    frame = pd.DataFrame(
        {
            "approved": np.asarray(approved, dtype=int),
            "predicted_probability": np.asarray(predicted_probability, dtype=float),
            "true_probability": np.asarray(true_probability, dtype=float),
        }
    )
    frame["bin"] = pd.qcut(
        frame["predicted_probability"], q=n_bins, labels=False, duplicates="drop"
    )
    return (
        frame.groupby("bin", observed=True)
        .agg(
            n=("approved", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_approval_rate=("approved", "mean"),
            mean_true_probability=("true_probability", "mean"),
        )
        .reset_index()
    )
