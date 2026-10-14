"""Evaluator-only true-risk cutoff benchmark against frozen bank decisions.

The existing exact search is reused through a local schema adapter. No bank
function receives truth, no bank policy is refitted/reoptimized, and the adapter
never leaves this evaluator boundary under the old bank-facing column names.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from fair_lending.professor_model.bank_regression import MODEL_ID, validate_ids
from fair_lending.professor_model.lending import BANK_INPUT_COLUMNS, choose_cutoff
from fair_lending.professor_model.profit import TRUE_PROFIT_COLUMNS

ORACLE_ID = "professor_true_risk_oracle_v1"
SCOPE = "EVALUATOR_ONLY"
ORACLE_COLUMNS = (
    "information_scope", "applicant_id", "policy_id", "true_PD", "optimal_tau",
    "approved", "requested_principal", "funded_principal", "decision_status",
)


def optimize_oracle(contracts: pd.DataFrame, truth: pd.DataFrame, budget: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Use the unchanged exact cutoff algorithm on true-PD-ordered true profits.

    Internal legacy names are an algorithm adapter, NOT actual bank inputs or
    a new bank assessment. Returned/persisted oracle columns are truth-labeled.
    """
    for name, frame in (("contracts", contracts), ("oracle truth", truth)):
        validate_ids(frame, name)
    if tuple(truth.columns) != TRUE_PROFIT_COLUMNS or set(truth.applicant_id) != set(contracts.applicant_id):
        raise ValueError("oracle requires exactly matching saved truth and contract IDs")
    if not np.isfinite(truth.iloc[:, 1:].to_numpy()).all() or not truth.true_PD.between(0, 1).all():
        raise ValueError("invalid true-risk/profit values")
    joined = contracts[["applicant_id", "L_i"]].merge(truth, on="applicant_id", validate="one_to_one")
    adapter = pd.DataFrame({
        "applicant_id": joined.applicant_id, "model_id": MODEL_ID,
        "bank_PD_bounded": joined.true_PD, "requested_principal": joined.L_i,
        "expected_profit_bank": joined.expected_profit_true,
    }).loc[:, BANK_INPUT_COLUMNS]
    internal_decisions, internal_candidates, internal_summary = choose_cutoff(adapter, budget)
    decisions = internal_decisions.rename(columns={"model_id": "policy_id", "bank_PD_bounded": "true_PD"})
    decisions["policy_id"] = ORACLE_ID
    decisions.insert(0, "information_scope", SCOPE)
    candidates = internal_candidates.rename(columns={"bank_expected_portfolio_profit": "true_expected_portfolio_profit"})
    candidates.insert(0, "information_scope", SCOPE)
    candidates.insert(1, "policy_id", ORACLE_ID)
    # Preserve search diagnostics, but do not mislabel true objective as bank belief.
    summary = {key.replace("bank_expected_portfolio_profit", "true_expected_portfolio_profit").replace("average_bank_profit_approved", "average_true_profit_approved").replace("all_requests_bank_profitable", "all_requests_true_profitable"): value for key, value in internal_summary.items() if key != "model_id"}
    summary.update(policy_id=ORACLE_ID, information_scope=SCOPE, risk_column="true_PD", objective_column="expected_profit_true")
    return decisions.loc[:, ORACLE_COLUMNS], candidates, summary


def compare_policies(
    contracts: pd.DataFrame, truth: pd.DataFrame, audit: pd.DataFrame,
    bank_decisions: pd.DataFrame, bank_selection: dict[str, Any], oracle_decisions: pd.DataFrame,
    oracle_selection: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize immutable matched policies; never select a different bank set."""
    for name, frame in (("contracts", contracts), ("truth", truth), ("audit", audit), ("bank decisions", bank_decisions), ("oracle decisions", oracle_decisions)):
        validate_ids(frame, name)
        if set(frame.applicant_id) != set(contracts.applicant_id):
            raise ValueError("comparison IDs must match exactly")
    if tuple(oracle_decisions.columns) != ORACLE_COLUMNS or not oracle_decisions.information_scope.eq(SCOPE).all() or not oracle_decisions.policy_id.eq(ORACLE_ID).all():
        raise ValueError("oracle decisions must be evaluator-only and truth-labeled")
    if not audit.B.isin([0, 1]).all():
        raise ValueError("invalid audit groups")
    if bank_selection["available_bank_capital"] != oracle_selection["available_bank_capital"]:
        raise ValueError("oracle must use the exact saved bank budget")
    base = contracts[["applicant_id", "L_i"]].merge(truth, on="applicant_id", validate="one_to_one").merge(audit[["applicant_id", "B"]], on="applicant_id", validate="one_to_one").sort_values("applicant_id").reset_index(drop=True)
    rows, groups, masks = [], [], {}
    for policy, decisions, selection, score in (
        ("bank", bank_decisions, bank_selection, "bank_PD_bounded"),
        ("oracle", oracle_decisions, oracle_selection, "true_PD"),
    ):
        aligned = decisions.sort_values("applicant_id").reset_index(drop=True)
        if not pd.api.types.is_bool_dtype(aligned.approved):
            raise ValueError("approval must be boolean")
        mask = aligned.approved.to_numpy()
        tau = selection["optimal_tau"]
        risk = aligned[score].to_numpy()
        expected = np.zeros(len(mask), dtype=bool) if tau is None else risk <= tau
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(aligned.requested_principal, base.L_i)
        np.testing.assert_array_equal(aligned.funded_principal, np.where(mask, base.L_i, 0.))
        funded = math.fsum(aligned.funded_principal)
        budget = selection["available_bank_capital"]
        if funded > budget or funded != selection["total_funded_principal"] or int(mask.sum()) != selection["number_approved"]:
            raise ValueError("policy capital/count differs from saved selection")
        if policy == "oracle":
            np.testing.assert_array_equal(aligned.true_PD, base.true_PD)
        masks[policy] = mask
        selected = base.loc[mask]
        total_true_profit = math.fsum(selected.expected_profit_true)
        rows.append({
            "information_scope": SCOPE, "policy": policy, "cutoff_score": score,
            "optimal_cutoff": tau, "n_applicants": len(base), "number_approved": int(mask.sum()),
            "approval_rate": float(mask.mean()), "funded_principal": funded,
            "available_bank_capital": budget, "capital_utilization": funded / budget if budget else None,
            "unused_capital": budget - funded, "true_expected_portfolio_profit": total_true_profit,
            "mean_true_PD_approved": float(selected.true_PD.mean()) if len(selected) else None,
            "average_true_profit_per_approved": total_true_profit / len(selected) if len(selected) else None,
        })
        for group, frame in base.groupby("B", sort=True):
            group_mask = mask[frame.index]
            issued = frame.loc[group_mask]
            group_funded = math.fsum(issued.L_i)
            groups.append({"information_scope": SCOPE, "policy": policy, "B": int(group), "n_applicants": len(frame), "number_approved": len(issued), "approval_rate": float(group_mask.mean()), "total_funded_principal": group_funded, "funded_principal_per_applicant": group_funded / len(frame), "mean_true_PD_approved": float(issued.true_PD.mean()) if len(issued) else None})
    summary, group_table = pd.DataFrame(rows), pd.DataFrame(groups)
    indexed = summary.set_index("policy")
    bank_profit = float(indexed.loc["bank", "true_expected_portfolio_profit"])
    oracle_profit = float(indexed.loc["oracle", "true_expected_portfolio_profit"])
    regret = oracle_profit - bank_profit  # signed; never clip a negative comparison
    bank_mask, oracle_mask = masks["bank"], masks["oracle"]
    disagreements = []
    for name, mask in (("funded_by_both", bank_mask & oracle_mask), ("bank_only", bank_mask & ~oracle_mask), ("oracle_only", oracle_mask & ~bank_mask), ("rejected_by_both", ~bank_mask & ~oracle_mask)):
        frame = base.loc[mask]
        disagreements.append({"information_scope": SCOPE, "allocation_group": name, "n_applicants": len(frame), "share_of_evaluation": len(frame) / len(base), "total_requested_principal": math.fsum(frame.L_i), "total_true_expected_profit": math.fsum(frame.expected_profit_true), "mean_true_PD": float(frame.true_PD.mean()) if len(frame) else None})
    def gap(policy: str) -> float | None:
        by_group = group_table.loc[group_table.policy == policy].set_index("B")
        return float(by_group.loc[1, "approval_rate"] - by_group.loc[0, "approval_rate"]) if {0, 1}.issubset(by_group.index) else None
    bank_gap, oracle_gap = gap("bank"), gap("oracle")
    union = int((bank_mask | oracle_mask).sum())
    comparison = {
        "information_scope": SCOPE, "economic_regret": regret,
        "economic_regret_pct_of_oracle": 100 * regret / oracle_profit if oracle_profit != 0 else None,
        "regret_nonnegative_within_tolerance": regret >= -1e-6,
        "regret_numerical_tolerance_USD": 1e-6,
        "jaccard_overlap": int((bank_mask & oracle_mask).sum()) / union if union else None,
        "jaccard_status": "defined" if union else "undefined_empty_union",
        "bank_group_approval_gap_pp": 100 * bank_gap if bank_gap is not None else None,
        "oracle_group_approval_gap_pp": 100 * oracle_gap if oracle_gap is not None else None,
        "bank_minus_oracle_group_gap_pp": 100 * (bank_gap - oracle_gap) if bank_gap is not None and oracle_gap is not None else None,
        "bank_minus_oracle_mean_true_PD_approved": rows[0]["mean_true_PD_approved"] - rows[1]["mean_true_PD_approved"] if rows[0]["mean_true_PD_approved"] is not None and rows[1]["mean_true_PD_approved"] is not None else None,
        "limitation": "Same cutoff form and capital, but score-dependent portfolio families differ; nonnegative regret is not a universal guarantee.",
    }
    for key in ("economic_regret", "economic_regret_pct_of_oracle", "jaccard_overlap", "bank_minus_oracle_group_gap_pp"):
        summary[key] = comparison[key]
    return summary, group_table, pd.DataFrame(disagreements), comparison
