"""Static Step 9C figures with separate diagnostic and valid-policy panels."""
from __future__ import annotations

import numpy as np

from . import mechanism_audit as audit
from .revised_lending import REFERENCE_IDS

NAVY, MAROON, TEAL, ORANGE = "#253746", "#861F41", "#207F86", "#E5751F"
ORDER = (*audit.POLICIES, *REFERENCE_IDS)
LABELS = ("A: direct", "D: positive\nfloor", "E: blind\ncontrol", "Frozen\naware", "Frozen\nblind", "Frozen\noracle")


def save_figures(risks, ties, portfolio, groups, derivation, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def save(fig, key):
        with output[key].open("xb") as stream:
            fig.savefig(stream, format="png", dpi=170)
        plt.close(fig)

    style = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
             "axes.titleweight": "semibold", "axes.labelcolor": NAVY, "text.color": NAVY}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), layout="constrained")
        raw = risks.loc[risks.variant_id == audit.B]
        bins = np.linspace(float(raw.score.min()) * 100, float(raw.score.max()) * 100, 70)
        for group, color in ((0, NAVY), (1, ORANGE)):
            values = raw.loc[raw.B == group, "score"].to_numpy() * 100
            axes[0].hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
                         alpha=.7, color=color, label=f"B={group}; {int((values < 0).sum()):,} negative")
        axes[0].axvline(0, color=MAROON, linewidth=1, linestyle="--")
        axes[0].set(title="B: raw direct scores — diagnostic only", xlabel="Raw score × 100 (percentage-point units)",
                    ylabel="Share within group per bin (%)")
        axes[0].legend(loc="upper left", fontsize=9)

        labels = []
        for index, variant in enumerate(audit.POLICIES):
            for group, color in ((0, NAVY), (1, ORANGE)):
                y = 2 * index + group
                values = risks.loc[(risks.variant_id == variant) & (risks.B == group), "score"].to_numpy() * 100
                lo, q1, median, q3, hi = np.quantile(values, [0, .25, .5, .75, 1])
                axes[1].plot([lo, hi], [y, y], color=color, linewidth=1)
                axes[1].plot([q1, q3], [y, y], color=color, linewidth=7, solid_capstyle="butt")
                axes[1].scatter([median], [y], color="white", edgecolor=color, s=38, zorder=3)
                labels.append(f"{variant[0]} / B={group}")
        axes[1].axvline(100 * derivation["floor_raw_probability"], color=TEAL, linewidth=1.3,
                        linestyle="--", label="Predeclared positive floor")
        axes[1].set_yticks(range(6), labels)
        axes[1].set_ylim(6.2, -.5)  # Leave room below the last distribution for its legend.
        axes[1].set(title="A, D, E: valid policy-score distributions", xlabel="Bounded score × 100 (percentage-point units)")
        axes[1].legend(loc="lower right", fontsize=9)
        for ax in axes:
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
        fig.suptitle("Direct adjustment and probability transformations", fontsize=16)
        fig.supxlabel("Separate score scales. Right: min–max, interquartile range and median. "
                      f"C is diagnostic only; C equals A: {derivation['C_exactly_equals_A']}. No hidden truth used.", fontsize=9)
        save(fig, "risk_figure")

        ordered = ties.set_index("variant_id").loc[list(audit.POLICIES)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
        x = np.arange(3)
        for offset, prefix, color, label in ((-.19, "zero", MAROON, "Exact-zero block"),
                                              (.19, "floor", TEAL, "Relocated positive-floor block (D)")):
            counts = ordered[f"{prefix}_block_count"].to_numpy()
            shares = ordered[f"{prefix}_block_budget_share"].to_numpy() * 100
            for ax, values, fmt in ((axes[0], counts, lambda v: f"{v:,.0f}"),
                                    (axes[1], shares, lambda v: f"{v:.2f}%")):
                bars = ax.bar(x + offset, values, .36, color=color, label=label)
                ax.bar_label(bars, labels=[fmt(v) for v in values], padding=4, fontsize=9)
        axes[0].set_ylabel("Applicants in the tied block")
        axes[1].set_ylabel("Block principal / fixed capital budget (%)")
        axes[1].axhline(100, color=NAVY, linewidth=1, linestyle="--", label="100% of budget")
        axes[1].set_ylim(0, max(115., float(ordered[["zero_block_budget_share", "floor_block_budget_share"]].max().max()) * 120))
        axes[0].set_ylim(0, max(1, float(ordered[["zero_block_count", "floor_block_count"]].max().max())) * 1.28)
        for ax in axes:
            ax.set_xticks(x, LABELS[:3]); ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
        fig.suptitle("A positive floor relocates the tie; it does not split it", fontsize=16)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[1].text(.98, .94, "Dashed line: full capital budget", transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=9)
        axes[0].set_title("Tie-block size")
        axes[1].set_title("Capital needed for the block alone")
        fig.supxlabel("The cutoff must also fund all lower scores. Full-prefix feasibility is recorded separately in tie_capital_diagnostics.csv.", fontsize=9)
        save(fig, "tie_figure")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), layout="constrained")
        indexed = groups.set_index(["policy_id", "B"])
        x = np.arange(6)
        for group, offset, color in ((0, -.19, NAVY), (1, .19, ORANGE)):
            values = np.array([indexed.loc[(p, group), "approval_rate"] for p in ORDER]) * 100
            bars = axes[0].bar(x + offset, values, .36, color=color, label=f"B={group}")
            for i, bar in enumerate(bars):
                if i >= 3: bar.set_hatch("//")
            axes[0].bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=8)
        axes[0].set_ylim(0, 113); axes[0].set_yticks(np.arange(0, 101, 20))
        axes[0].set(title="Approval rates by group", ylabel="Approval rate within group (%)")
        axes[0].legend(ncol=2, loc="upper right")
        values = portfolio.set_index("policy_id").loc[list(ORDER), "true_expected_portfolio_profit_USD"].to_numpy() / 1e9
        bars = axes[1].bar(x, values, color=[MAROON, TEAL, NAVY, "#89939B", "#89939B", "#89939B"])
        for i, bar in enumerate(bars):
            if i >= 3: bar.set_hatch("//")
        axes[1].bar_label(bars, labels=[f"${v:.4f}B" for v in values], padding=4, fontsize=8)
        axes[1].set_ylim(min(0., float(values.min()) * 1.2), max(.001, float(values.max()) * 1.17))
        axes[1].set(title="Evaluator-only true expected portfolio profit", ylabel="Nominal expected profit (USD billions)")
        axes[1].legend(handles=[Patch(facecolor="#89939B", hatch="//", label="Frozen reference; not reoptimized")], loc="upper right", fontsize=9)
        for ax in axes:
            ax.set_xticks(x, LABELS, fontsize=9); ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
        fig.suptitle("Frozen capital and contracts: lending outcomes", fontsize=16)
        fig.supxlabel("Only A/D/E are new policy comparisons. Synthetic, stylized expected values; smaller gaps are not a fairness guarantee. "
                      "The oracle is a common-cutoff reference.", fontsize=9)
        save(fig, "outcome_figure")
