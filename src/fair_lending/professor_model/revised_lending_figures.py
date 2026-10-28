"""Two static Step 9B figures; no optimization, fitting, or scientific changes."""
from __future__ import annotations

import numpy as np

from .revised_lending import POLICY_ORDER, SCENARIO_IDS

LABELS = ("No-direct\naware", "Direct\nblind − 0.1B", "Blind\ncontrol",
          "Frozen\naware bank", "Frozen\nblind bank", "Frozen\noracle")


def save_figures(portfolio, groups, profit_path, approval_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    table = portfolio.set_index("policy_id").loc[list(POLICY_ORDER)]
    x = np.arange(len(POLICY_ORDER))
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(11.5, 6.2), layout="constrained")
        profit = table.true_expected_portfolio_profit_USD.to_numpy() / 1e9
        colors = ["#861F41", "#E5751F", "#253746", "#89939B", "#89939B", "#89939B"]
        bars = ax.bar(x, profit, color=colors, edgecolor="#253746", linewidth=.6)
        for index, bar in enumerate(bars):
            if index >= len(SCENARIO_IDS): bar.set_hatch("//")
        ax.bar_label(bars, labels=[f"${v:.4f}B" for v in profit], padding=4)
        ax.set_xticks(x, LABELS)
        ax.set_ylabel("True expected portfolio profit (USD billions)")
        ax.set_ylim(min(0., float(profit.min()) * 1.15), max(.01, float(profit.max()) * 1.15))
        ax.axhline(0, color="#68737D", linewidth=.8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.set_title("Scenario lending comparison: evaluator-only true expected profit", pad=16)
        ax.legend(handles=[Patch(facecolor="#89939B", edgecolor="#253746", hatch="//", label="Frozen reference, not reoptimized")], loc="upper right")
        fig.supxlabel("Same applicants, contracts and capital. Nominal expected values under a stylized contract, not realized returns.", fontsize=9)
        with profit_path.open("xb") as stream: fig.savefig(stream, format="png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.5, 6.2), layout="constrained")
        indexed = groups.set_index(["policy_id", "B"])
        for group, offset, color in ((0, -.19, "#253746"), (1, .19, "#E5751F")):
            rates = np.array([100 * indexed.loc[(p, group), "approval_rate"] for p in POLICY_ORDER])
            bars = ax.bar(x + offset, rates, width=.36, color=color, edgecolor="#253746", linewidth=.6, label=f"B={group}")
            for index, bar in enumerate(bars):
                if index >= len(SCENARIO_IDS): bar.set_hatch("//")
            ax.bar_label(bars, labels=[f"{v:.1f}%" for v in rates], padding=3, fontsize=9)
        ax.set_xticks(x, LABELS)
        ax.set_ylim(0, 110)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel("Approval rate within group (%)")
        ax.set_title("Group approval rates under identical contracts and capital", pad=16)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", ncol=2)
        fig.supxlabel("Hatched bars: frozen references. Denominators include all evaluation applicants in each group. Synthetic evidence only.", fontsize=9)
        with approval_path.open("xb") as stream: fig.savefig(stream, format="png", dpi=170)
        plt.close(fig)
