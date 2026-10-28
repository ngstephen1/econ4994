"""Static figures of six selected diagnostic cases; no inference or optimization."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from . import raw_ordering as s


def render(tables: Path, destination: Path):
    policies = pd.read_csv(tables / "policies.csv")
    paired = pd.read_csv(tables / "comparisons.csv")
    within = paired[paired.comparison_kind.eq("within_model_ordering")]
    if len(policies) != 36 or len(within) != 18:
        raise ValueError("figure requires all six diagnostic cases")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    colors = {s.BOUNDED: "#8995A3", s.RAW: "#167D8D"}
    labels = ["Canonical", "Draw 008", "Draw 064", "Draw 089", "Draw 092", "Draw 154"]
    for row, (source, title) in enumerate(zip(s.POLICIES, ("W: frozen aware", "Z: B contribution removed", "F: frozen blind"))):
        subset = policies[policies.source_policy.eq(source)].set_index(["case_id", "ordering_regime"])
        axis, profit = axes[row]
        y = np.arange(len(s.CASES))
        for regime, offset, label in ((s.BOUNDED, -.17, "Original bounded ordering"), (s.RAW, .17, "Raw ordering; same accounting PD")):
            values = [subset.loc[(c, regime), "number_approved"] for c in s.CASES]
            axis.barh(y+offset, values, height=.3, color=colors[regime], label=label)
        axis.set_yticks(y, labels); axis.invert_yaxis(); axis.set_xlabel("Applicants funded")
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
        pairs = within[within.policy_first.eq(f"{source}__{s.RAW}")].set_index("case_id")
        values = np.array([pairs.loc[c, "true_profit_difference_USD"] for c in s.CASES]) / 1e9
        profit.axvline(0, color="#87929F", linewidth=.8)
        profit.scatter(values, y, color="#167D8D", s=34, zorder=3)
        for yi, value in zip(y, values):
            profit.annotate(f"{value*1000:+,.2f}M", (value, yi), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
        profit.set_yticks(y, labels); profit.invert_yaxis()
        profit.set_xlim(min(-.08, float(values.min())-.08), max(.3, float(values.max())+.65))
        profit.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"${v:g}B"))
        profit.set_xlabel("Raw-order minus original true expected profit")
        for ax in (axis, profit):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    fig.suptitle("Raw ordering can preserve information lost in clipped-zero ties\nSix selected mechanism diagnostics; same population, probabilities and capital", fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.025), ncol=2, fontsize=10, frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return destination
