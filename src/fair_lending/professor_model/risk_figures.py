"""Three static scientific diagnostics; each uses all saved applicants."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


def save_risk_figures(applicants: pd.DataFrame, truth: pd.DataFrame, directory: Path) -> list[Path]:
    """Plot probability in percent units with labels explaining the log scale."""
    joined = applicants.merge(truth, on="applicant_id", validate="one_to_one")
    directory.mkdir(parents=True, exist_ok=True)
    probability_percent = 100 * joined.true_PD.to_numpy()
    paths: list[Path] = []
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
        edges = np.geomspace(probability_percent.min(), probability_percent.max(), 46)
        ax.hist(probability_percent, bins=edges, color="#198B8D", alpha=.9, edgecolor="white", linewidth=.4)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.axvline(probability_percent.mean(), color="#17365D", linestyle="--", label=f"Mean {probability_percent.mean():.4f}%")
        ax.axvline(np.median(probability_percent), color="#B46737", linestyle=":", label=f"Median {np.median(probability_percent):.4f}%")
        ax.set(xlabel="True default probability (%) — logarithmic scale", ylabel="Applicants per logarithmic bin", title=f"True default-risk distribution | n = {len(joined):,}")
        ax.legend(frameon=False)
        path = directory / "true_pd_histogram.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
        for column, multiplier, label, filename, title in (
            ("log_Y", 1, "Natural log of annual income (Y in synthetic dollars)", "true_pd_vs_log_income.png", "True default risk versus log income"),
            ("D", 100, "Debt-to-income ratio (%) — untruncated", "true_pd_vs_dti.png", "True default risk versus DTI"),
        ):
            fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
            density = ax.hexbin(multiplier * joined[column], probability_percent, gridsize=55, bins="log", mincnt=1, cmap="viridis", linewidths=0)
            fig.colorbar(density, ax=ax, label="Applicants per hexagon (log color scale)")
            ax.set(xlabel=label, ylabel="True default probability (%)", title=title)
            ax.text(.02, .98, "All applicants; other inputs vary", transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
            path = directory / filename
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths
