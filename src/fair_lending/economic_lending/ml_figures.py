"""Small publication-style figures for the Version 2 risk-model benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODEL_COLORS = {"traditional_logit_v1": "#1f77b4", "ml_histgb_v1": "#d95f02", "oracle": "#4d4d4d"}


def save_risk_truth_scatter(comparison: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, (label, column, color) in zip(
        axes,
        [
            ("Traditional logit", "repayment_probability_traditional", "#1f77b4"),
            ("HistGradientBoosting", "repayment_probability_ml", "#d95f02"),
        ],
        strict=True,
    ):
        axis.scatter(
            comparison["repayment_probability_per_period_true"],
            comparison[column],
            s=9,
            alpha=0.35,
            color=color,
            edgecolors="none",
        )
        lower = min(comparison["repayment_probability_per_period_true"].min(), comparison[column].min())
        axis.plot([lower, 1], [lower, 1], color="black", linewidth=1, linestyle="--")
        axis.set_title(label)
        axis.set_xlabel("True conditional repayment probability")
    axes[0].set_ylabel("Estimated conditional repayment probability")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_calibration_plot(calibration: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    for model, frame in calibration.groupby("model", observed=True):
        axis.plot(
            frame["mean_predicted_rho"],
            frame["observed_payment_rate"],
            marker="o",
            linewidth=1.5,
            label=model,
            color=MODEL_COLORS.get(str(model)),
        )
    lower = calibration[["mean_predicted_rho", "observed_payment_rate"]].min().min()
    axis.plot([lower, 1], [lower, 1], color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Mean predicted conditional probability")
    axis.set_ylabel("Observed at-risk payment rate")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_risk_error_distribution(comparison: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for label, column, color in [
        ("Traditional logit", "traditional_risk_error", "#1f77b4"),
        ("HistGradientBoosting", "ml_risk_error", "#d95f02"),
    ]:
        axis.hist(comparison[column], bins=45, density=True, alpha=0.45, label=label, color=color)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("Predicted rho minus true rho")
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_profit_error_distribution(profit_errors: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    values = [
        profit_errors.loc[profit_errors["model"] == model, "profit_error"]
        for model in ["traditional_logit_v1", "ml_histgb_v1"]
    ]
    axis.boxplot(values, tick_labels=["Traditional logit", "HistGradientBoosting"], showfliers=False)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("Perceived minus true expected profit ($)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
