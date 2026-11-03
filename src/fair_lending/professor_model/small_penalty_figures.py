"""Prediction-only group approval and true-profit figures, with positive counts."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .small_penalty import SCENARIOS, METHODS


def approval_label(rate, approved):
    """Small positive approvals must never display as a rounded zero."""
    if isinstance(approved, (bool, np.bool_)) or not np.isfinite(approved) or int(approved) != approved or approved < 0:
        raise ValueError("nonnegative integer approval count required")
    if not np.isfinite(rate) or not 0 <= rate <= 1 or (rate > 0) != (approved > 0):
        raise ValueError("consistent finite rate/count required")
    text = f"{100*rate:.2f}%"
    if rate > 0 and float(text[:-1]) == 0:
        text = f"{100*rate:.4g}%"
    return f"{text}\n(n={int(approved):,})"


def profit_label(value):
    if not np.isfinite(value): raise ValueError("finite profit required")
    if value != 0 and abs(value) < 1:
        return f"${value:.3g}"
    if value != 0 and abs(value) < 10000:
        return f"${value:,.2f}"
    return f"${value/1e6:,.2f}M"


def save_figures(groups, directory: Path, delta: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    required = {"scenario_id", "method_id", "B", "approval_rate", "number_approved", "true_expected_portfolio_profit_USD"}
    if not required.issubset(groups.columns) or len(groups) != 12:
        raise ValueError("twelve complete group cells required")
    indexed = groups.set_index(["scenario_id", "method_id", "B"])
    expected = {(s,m,b) for s in SCENARIOS for m in METHODS for b in (0,1)}
    if not indexed.index.is_unique or set(indexed.index) != expected:
        raise ValueError("missing or duplicate group cells")
    profit = groups.true_expected_portfolio_profit_USD.to_numpy(dtype=float)
    if not np.isfinite(profit).all() or not groups.approval_rate.between(0,1).all():
        raise ValueError("invalid plotting values")
    low, high = min(0., float(profit.min()/1e6)*1.15), max(1., float(profit.max()/1e6)*1.2)
    labels = ("OLS control", f"OLS + {delta:g}B", "Gradient boosting")
    directory.mkdir(parents=True,exist_ok=True); outputs=[]
    with plt.rc_context({"font.size":11,"axes.spines.top":False,"axes.spines.right":False}):
        for index, scenario in enumerate(SCENARIOS,1):
            for metric,stem in (("approval_rate","approval_rates"),("true_expected_portfolio_profit_USD","true_profit")):
                files = [directory/f"{scenario}_{stem}.{ext}" for ext in ("png","pdf")]
                if any(p.exists() for p in files): raise FileExistsError("figures are immutable")
                fig,ax=plt.subplots(figsize=(10.5,6),layout="constrained")
                for b,offset,color,name in ((0,-.19,"#253746","Group A (B=0)"),(1,.19,"#E5751F","Minority Group B (B=1)")):
                    rows=[indexed.loc[(scenario,m,b)] for m in METHODS]
                    values=np.array([r[metric] for r in rows],float)
                    shown=values*100 if metric=="approval_rate" else values/1e6
                    bars=ax.bar(np.arange(3)+offset,shown,width=.36,color=color,edgecolor="#253746",linewidth=.5,label=name)
                    text=[approval_label(r.approval_rate,r.number_approved) if metric=="approval_rate" else profit_label(r[metric]) for r in rows]
                    ax.bar_label(bars,labels=text,padding=5,fontsize=9)
                ax.set_xticks(np.arange(3),labels)
                ax.set_title(f"Scenario {index}: {'no systemic differences' if index==1 else 'systemic differences'}",color="#861F41",weight="bold",pad=18)
                ax.set_ylim((0,100) if metric=="approval_rate" else (low,high))
                ax.set_ylabel("Within-group approval rate (%)" if metric=="approval_rate" else "True expected bank profit from funded loans (USD millions)")
                ax.axhline(0,color="#68737D",linewidth=.7);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
                ax.legend(loc="upper right",ncol=2,frameon=False,fontsize=10)
                fig.supxlabel("80,000 training / 20,000 prediction applicants. Profits use prediction loans only.\nIllustrative approval-calibrated penalty; one matched noise realization; unequal group sizes.",fontsize=9)
                for p in files:
                    with p.open("xb") as stream:fig.savefig(stream,format=p.suffix[1:],dpi=180)
                    outputs.append(p)
                plt.close(fig)
    return outputs
