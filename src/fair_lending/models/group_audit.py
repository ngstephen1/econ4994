"""Race-group audits relative to simulated lender approval decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from fair_lending.models.evaluation import true_probability_recovery


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def group_prediction_metrics(
    test: pd.DataFrame,
    predicted_probability: np.ndarray,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Compute decision-agreement and probability metrics by race."""
    audit = test.loc[:, ["race", "approved", "approval_probability_true"]].copy()
    audit["predicted_probability"] = np.asarray(predicted_probability, dtype=float)
    audit["predicted"] = (audit["predicted_probability"] >= threshold).astype(int)
    rows = []
    for race, group in audit.groupby("race", observed=True, sort=False):
        y = group["approved"].to_numpy(dtype=int)
        prediction = group["predicted"].to_numpy(dtype=int)
        probability = group["predicted_probability"].to_numpy(dtype=float)
        truth = group["approval_probability_true"].to_numpy(dtype=float)
        tp = int(((prediction == 1) & (y == 1)).sum())
        fp = int(((prediction == 1) & (y == 0)).sum())
        tn = int(((prediction == 0) & (y == 0)).sum())
        fn = int(((prediction == 0) & (y == 1)).sum())
        tpr = _ratio(tp, tp + fn)
        tnr = _ratio(tn, tn + fp)
        rows.append(
            {
                "race": str(race),
                "n": len(group),
                "actual_approval_rate": float(np.mean(y)),
                "mean_predicted_probability": float(np.mean(probability)),
                "predicted_approval_rate": float(np.mean(prediction)),
                "accuracy": float(np.mean(prediction == y)),
                "balanced_accuracy": (
                    float(np.mean([tpr, tnr]))
                    if np.isfinite(tpr) and np.isfinite(tnr)
                    else float("nan")
                ),
                "tpr": tpr,
                "fpr": _ratio(fp, fp + tn),
                "tnr": tnr,
                "fnr": _ratio(fn, fn + tp),
                "ppv": _ratio(tp, tp + fp),
                "npv": _ratio(tn, tn + fn),
                "roc_auc": (
                    float(roc_auc_score(y, probability))
                    if len(np.unique(y)) == 2
                    else float("nan")
                ),
                "brier_score": float(brier_score_loss(y, probability)),
                "true_probability_mean": float(np.mean(truth)),
                **true_probability_recovery(probability, truth),
            }
        )
    return pd.DataFrame(rows)


def black_white_disparities(group_metrics: pd.DataFrame) -> dict[str, float]:
    """Return Black-minus-White actual and predicted disparity estimands."""
    indexed = group_metrics.set_index("race")
    missing = {"White", "Black"}.difference(indexed.index)
    if missing:
        raise ValueError(f"Missing focal audit groups: {sorted(missing)}")
    black = indexed.loc["Black"]
    white = indexed.loc["White"]
    observed = float(black["actual_approval_rate"] - white["actual_approval_rate"])
    probability = float(
        black["mean_predicted_probability"] - white["mean_predicted_probability"]
    )
    predicted_rate = float(
        black["predicted_approval_rate"] - white["predicted_approval_rate"]
    )
    true_probability = float(
        black["true_probability_mean"] - white["true_probability_mean"]
    )
    reproduction_error = probability - observed
    return {
        "black_mean_prediction": float(black["mean_predicted_probability"]),
        "white_mean_prediction": float(white["mean_predicted_probability"]),
        "predicted_probability_gap": probability,
        "black_predicted_approval_rate": float(black["predicted_approval_rate"]),
        "white_predicted_approval_rate": float(white["predicted_approval_rate"]),
        "predicted_approval_gap": predicted_rate,
        "observed_black_white_gap": observed,
        "true_probability_black_white_gap": true_probability,
        "probability_gap_reproduction_error": reproduction_error,
        "absolute_reproduction_error": abs(reproduction_error),
    }
