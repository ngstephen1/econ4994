"""Four static six-bar figures from validated November result tables only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("ols_control", "ols_penalty", "ml_histgb")
SCENARIOS = ("scenario_1_no_systemic", "scenario_2_systemic")
METHOD_LABELS = ("OLS control", "OLS + 0.01B", "Gradient boosting")
SCENARIO_LABELS = {
    SCENARIOS[0]: "Scenario 1: no systemic differences",
    SCENARIOS[1]: "Scenario 2: systemic epsilon shift",
}


def save_figures(groups: pd.DataFrame, directory: Path) -> list[Path]:
    """Use within-group rates and additive group profit contributions.

    No inference, fitting, selection or optimization occurs here. The same
    profit limits are used for both worlds; negative profits are not hidden.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profit_column = "true_expected_portfolio_profit_USD"
    required = {"scenario_id", "method_id", "B", "approval_rate", profit_column}
    if not required.issubset(groups.columns) or len(groups) != 12:
        raise ValueError("exact twelve scenario/method/group cells required")
    indexed = groups.set_index(["scenario_id", "method_id", "B"])
    expected = pd.MultiIndex.from_product([SCENARIOS, METHODS, [0, 1]], names=indexed.index.names)
    if not indexed.index.is_unique or set(indexed.index) != set(expected):
        raise ValueError("missing or duplicated figure cells")
    profits = groups[profit_column].to_numpy(dtype=float) / 1e6
    if not np.isfinite(profits).all() or not groups.approval_rate.between(0, 1).all():
        raise ValueError("finite profits and valid approval rates required")
    low = min(0.0, float(profits.min()) * 1.15)
    high = max(1.0, float(profits.max()) * 1.18)
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        for scenario in SCENARIOS:
            for metric in ("approval_rate", profit_column):
                stem = f"{scenario}_{'approval_rates' if metric == 'approval_rate' else 'true_profit'}"
                outputs = [directory / f"{stem}.{extension}" for extension in ("png", "pdf")]
                if any(p.exists() for p in outputs):
                    raise FileExistsError("figure outputs are immutable")
                fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
                for group, offset, color in ((0, -.19, "#253746"), (1, .19, "#E5751F")):
                    values = np.array([indexed.loc[(scenario, method, group), metric] for method in METHODS], dtype=float)
                    values = 100 * values if metric == "approval_rate" else values / 1e6
                    bars = ax.bar(np.arange(3) + offset, values, width=.36, color=color,
                                  edgecolor="#253746", linewidth=.5, label=f"B={group}")
                    labels = [f"{v:.1f}%" for v in values] if metric == "approval_rate" else [f"${v:,.1f}M" for v in values]
                    ax.bar_label(bars, labels=labels, padding=4, fontsize=10)
                ax.set_xticks(np.arange(3), METHOD_LABELS)
                ax.set_title(SCENARIO_LABELS[scenario], pad=18, color="#861F41", weight="bold")
                if metric == "approval_rate":
                    ax.set_ylim(0, 100)
                    ax.set_ylabel("Within-group approval rate (%)")
                else:
                    ax.set_ylim(low, high)
                    ax.set_ylabel("True expected profit from funded loans (USD millions)")
                ax.axhline(0, color="#68737D", linewidth=.7)
                ax.grid(axis="y", alpha=.2)
                ax.set_axisbelow(True)
                ax.legend(ncol=2, loc="upper right", frameon=False)
                fig.supxlabel("One matched synthetic realization. Fixed capital and contracts. Group sizes differ.\nNominal expected values, not realized or annual bank returns.", fontsize=9)
                for path in outputs:
                    with path.open("xb") as stream:
                        fig.savefig(stream, format=path.suffix[1:], dpi=180)
                    saved.append(path)
                plt.close(fig)
    return saved
