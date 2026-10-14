"""Two diagnostic figures with full, linear measurement axes."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_signal_figures(applicants: pd.DataFrame, targets: pd.DataFrame, directory: Path) -> list[Path]:
    data = applicants.loc[:, ["applicant_id", "true_PD"]].merge(targets, on="applicant_id", validate="one_to_one")
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
        ax.hist(data.PN, bins=65, color="#198B8D", edgecolor="white", linewidth=.4)
        ax.axvline(0, linestyle="--", color="#17365D", label="Zero measurement")
        ax.set(xlabel="PN measurement (raw probability units)", ylabel="Applicants", title=f"Raw noisy risk measurements | n = {len(data):,}")
        ax.legend(frameon=False)
        path = directory / "pn_histogram.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
        fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
        density = ax.hexbin(data.true_PD, data.PN, gridsize=55, bins="log", mincnt=1, cmap="viridis", linewidths=0)
        fig.colorbar(density, ax=ax, label="Applicants per hexagon (log color scale)")
        ax.set(xlabel="Hidden true_PD (raw probability units)", ylabel="PN measurement (raw probability units)", title="Noisy measurement versus hidden true risk")
        ax.axhline(0, color="#17365D", linestyle="--", linewidth=.8)
        path = directory / "pn_vs_true_pd.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths
