"""One common risk cutoff, exact ties, and perceived-profit optimization only.

Feasible portfolios are nested prefixes of exact risk groups, not unrestricted
subsets. All arithmetic comparisons use exact sums of the input binary floats;
no budget guard, slack, currency rounding, or approximate objective tie is used.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from fair_lending.professor_model.bank_regression import MODEL_ID, validate_ids

BANK_INPUT_COLUMNS = ("applicant_id", "model_id", "bank_PD_bounded", "requested_principal", "expected_profit_bank")
DECISION_COLUMNS = ("applicant_id", "model_id", "bank_PD_bounded", "optimal_tau", "approved", "requested_principal", "funded_principal", "decision_status")


def validate_bank_inputs(bank: pd.DataFrame) -> None:
    validate_ids(bank, "bank lending view")
    if tuple(bank.columns) != BANK_INPUT_COLUMNS or len(bank) == 0 or bank.isna().any().any():
        raise ValueError("bank lending view requires the exact observable-only schema")
    if not bank.model_id.eq(MODEL_ID).all():
        raise ValueError("unexpected frozen bank model")
    if not np.isfinite(bank.loc[:, BANK_INPUT_COLUMNS[2:]].to_numpy(dtype=float)).all():
        raise ValueError("lending inputs must be finite")
    if not bank.bank_PD_bounded.between(0, 1).all() or not (bank.requested_principal > 0).all():
        raise ValueError("bounded PD and strictly positive full requests are required")


def capital_budget(bank: pd.DataFrame, fraction: float) -> tuple[float, float]:
    validate_bank_inputs(bank)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("budget fraction must be finite and in [0,1]")
    total = math.fsum(bank.requested_principal)
    return total, float(fraction * total)


def _integer_units(values: list[float]) -> tuple[list[int], int]:
    """Every finite IEEE float is numerator/2**e; align the exact denominators."""
    ratios = [float(value).as_integer_ratio() for value in values]
    exponent = max(denominator.bit_length() - 1 for _, denominator in ratios)
    return [numerator << (exponent - denominator.bit_length() + 1) for numerator, denominator in ratios], exponent


def _risk_distribution(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"n": 0, **{key: None for key in ("mean", "median", "sample_SD", "minimum", "p05", "p95", "maximum")}}
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "sample_SD": float(values.std(ddof=1)) if len(values) > 1 else None, "minimum": float(values.min()), "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95)), "maximum": float(values.max())}


def choose_cutoff(bank: pd.DataFrame, budget: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Enumerate each unique cutoff and reject-all; ties choose the lowest cutoff.

    A null selected cutoff denotes the separate no-lending policy (equivalent
    to tau=-infinity), never tau=0, which would fund every clipped-zero score.
    Complexity is O(n log n) sorting plus O(n) exact integer accumulation.
    """
    validate_bank_inputs(bank)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    ordered = bank.sort_values(["bank_PD_bounded", "applicant_id"]).reset_index(drop=True)
    principal_units, principal_exponent = _integer_units([*ordered.requested_principal, float(budget)])
    budget_units = principal_units.pop()
    profit_units, profit_exponent = _integer_units(ordered.expected_profit_bank.tolist())
    records = [{"candidate_id": 0, "candidate_type": "no_lending", "tau": np.nan, "number_approved": 0, "funded_principal": 0., "budget_feasible": True, "bank_expected_portfolio_profit": 0., "risk_group_size": 0}]
    cumulative_principal = cumulative_profit = best_profit = 0
    best_id, group_start = 0, 0
    candidate_capital_units = [0]
    risks = ordered.bank_PD_bounded.to_numpy()
    for index, (principal, profit) in enumerate(zip(principal_units, profit_units)):
        cumulative_principal += principal
        cumulative_profit += profit
        if index + 1 != len(ordered) and risks[index] == risks[index + 1]:
            continue  # a tied group is never split
        feasible = cumulative_principal <= budget_units
        candidate_id = len(records)
        records.append({"candidate_id": candidate_id, "candidate_type": "cutoff", "tau": float(risks[index]), "number_approved": index + 1, "funded_principal": cumulative_principal / (1 << principal_exponent), "budget_feasible": feasible, "bank_expected_portfolio_profit": cumulative_profit / (1 << profit_exponent), "risk_group_size": index - group_start + 1})
        candidate_capital_units.append(cumulative_principal)
        # Strict exact comparison preserves the first/lowest cutoff on ties.
        if feasible and cumulative_profit > best_profit:
            best_id, best_profit = candidate_id, cumulative_profit
        group_start = index + 1
    candidates = pd.DataFrame(records)
    candidates["selected"] = candidates.candidate_id.eq(best_id)
    chosen = records[best_id]
    tau = None if best_id == 0 else chosen["tau"]
    by_id = bank.sort_values("applicant_id").reset_index(drop=True)
    approved = np.zeros(len(by_id), dtype=bool) if tau is None else by_id.bank_PD_bounded.to_numpy() <= tau
    decisions = by_id.loc[:, BANK_INPUT_COLUMNS[:3]].copy()
    decisions["optimal_tau"] = np.nan if tau is None else tau
    decisions["approved"] = approved
    decisions["requested_principal"] = by_id.requested_principal.to_numpy()
    decisions["funded_principal"] = np.where(approved, by_id.requested_principal, 0.)
    decisions["decision_status"] = np.where(approved, "approved_full_request", "rejected_no_lending_policy" if tau is None else "rejected_above_cutoff")
    funded = math.fsum(decisions.funded_principal)
    if funded > budget:
        raise RuntimeError("original capital budget exceeded")
    bank_profit = math.fsum(by_id.loc[approved, "expected_profit_bank"])
    if funded != chosen["funded_principal"] or bank_profit != chosen["bank_expected_portfolio_profit"]:
        raise RuntimeError("exact candidate sums do not match issued full loans")
    n_approved = int(approved.sum())
    feasible_ids = candidates.loc[candidates.budget_feasible, "candidate_id"]
    next_candidate = records[best_id + 1] if best_id + 1 < len(records) else None
    risk_groups = ordered.groupby("bank_PD_bounded", sort=True).size()
    summary = {
        "model_id": MODEL_ID, "optimal_tau": tau, "selected_candidate_id": best_id,
        "n_applicants": len(by_id), "number_approved": n_approved, "number_rejected": len(by_id) - n_approved,
        "approval_rate": n_approved / len(by_id), "total_requested_principal": math.fsum(by_id.requested_principal),
        "available_bank_capital": float(budget), "total_funded_principal": funded,
        "budget_utilization": funded / budget if budget else None, "unused_capital": budget - funded,
        "bank_expected_portfolio_profit": bank_profit,
        "average_bank_profit_approved": bank_profit / n_approved if n_approved else None,
        "exact_budget_exhausted": candidate_capital_units[best_id] == budget_units,
        "next_cutoff": next_candidate["tau"] if next_candidate else None,
        "next_cutoff_budget_feasible": next_candidate["budget_feasible"] if next_candidate else None,
        "next_cutoff_funded_principal": next_candidate["funded_principal"] if next_candidate else None,
        "capital_limits_next_cutoff": next_candidate is not None and not next_candidate["budget_feasible"],
        "largest_feasible_cutoff_selected": best_id == int(feasible_ids.iloc[-1]),
        "all_requests_bank_profitable": bool((by_id.expected_profit_bank > 0).all()),
        "n_candidates_including_no_lending": len(candidates), "n_feasible_candidates": int(candidates.budget_feasible.sum()),
        "n_tied_risk_groups": int((risk_groups > 1).sum()), "largest_tied_group_size": int(risk_groups.max()),
        "zero_risk_applicants": int((by_id.bank_PD_bounded == 0).sum()),
        "zero_risk_approved": int(((by_id.bank_PD_bounded == 0) & approved).sum()),
        "selected_cutoff_group_size": chosen["risk_group_size"],
        **{f"{label}_PD_{key}": value for label, values in (("approved", by_id.loc[approved, "bank_PD_bounded"].to_numpy()), ("rejected", by_id.loc[~approved, "bank_PD_bounded"].to_numpy())) for key, value in _risk_distribution(values).items()},
    }
    return decisions.loc[:, DECISION_COLUMNS], candidates, summary


def validate_solution(bank: pd.DataFrame, budget: float, decisions: pd.DataFrame, candidates: pd.DataFrame, summary: dict[str, Any]) -> dict[str, bool]:
    """Verify full coverage, exact decision rule, ties, feasibility, and optimality."""
    validate_ids(decisions, "lending decisions")
    if tuple(decisions.columns) != DECISION_COLUMNS or set(decisions.applicant_id) != set(bank.applicant_id):
        raise ValueError("truth-free decision schema or IDs are invalid")
    aligned = bank.sort_values("applicant_id").reset_index(drop=True)
    result = decisions.sort_values("applicant_id").reset_index(drop=True)
    expected_mask = np.zeros(len(aligned), dtype=bool) if summary["optimal_tau"] is None else aligned.bank_PD_bounded <= summary["optimal_tau"]
    np.testing.assert_array_equal(result.approved, expected_mask)
    np.testing.assert_array_equal(result.funded_principal, np.where(expected_mask, aligned.requested_principal, 0.))
    rebuilt_decisions, rebuilt_candidates, rebuilt_summary = choose_cutoff(bank, budget)
    pd.testing.assert_frame_equal(result, rebuilt_decisions, check_exact=True)
    pd.testing.assert_frame_equal(candidates, rebuilt_candidates, check_exact=True)
    if summary != rebuilt_summary:
        raise ValueError("selected cutoff or reported summary does not match exact bank-only search")
    return {"exact_threshold_rule": True, "tied_risks_together": True, "no_skipped_applicants": True, "full_loans_only": True, "original_budget_not_exceeded": True, "all_distinct_candidates_and_no_lending": True, "maximum_perceived_profit_among_feasible_cutoffs": True, "truth_free_decisions": True}


def evaluate_portfolio(bank: pd.DataFrame, decisions: pd.DataFrame, audit: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Post-freeze evaluator audit only; cannot influence a previously chosen tau."""
    for name, frame in (("bank", bank), ("decisions", decisions), ("observed audit", audit), ("true profit", truth)):
        validate_ids(frame, name)
        if set(frame.applicant_id) != set(bank.applicant_id):
            raise ValueError("portfolio audit IDs do not match")
    if tuple(audit.columns) != ("applicant_id", "B", "bank_PD_raw") or tuple(truth.columns) != ("applicant_id", "true_PD", "expected_receipts_true", "expected_profit_true"):
        raise ValueError("invalid evaluator audit schema")
    if not audit.B.isin([0, 1]).all() or not np.isfinite(audit.bank_PD_raw).all() or not np.isfinite(truth.iloc[:, 1:].to_numpy()).all() or not truth.true_PD.between(0, 1).all():
        raise ValueError("invalid evaluator risk/group values")
    joined = decisions.merge(audit, on="applicant_id", validate="one_to_one").merge(truth, on="applicant_id", validate="one_to_one").merge(bank[["applicant_id", "expected_profit_bank"]], on="applicant_id", validate="one_to_one").sort_values("applicant_id")
    groups = []
    for group, frame in joined.groupby("B", sort=True):
        issued = frame.loc[frame.approved]
        funded = math.fsum(frame.funded_principal)
        groups.append({"B": int(group), "n_applicants": len(frame), "number_approved": len(issued), "approval_rate": len(issued) / len(frame), "total_funded_principal": funded, "average_funded_principal_per_applicant": funded / len(frame), "mean_bank_PD_approved": float(issued.bank_PD_bounded.mean()) if len(issued) else None, "mean_true_PD_approved": float(issued.true_PD.mean()) if len(issued) else None, "total_true_expected_profit": math.fsum(issued.expected_profit_true)})
    group_table = pd.DataFrame(groups)
    bank_profit = math.fsum(joined.loc[joined.approved, "expected_profit_bank"])
    true_profit = math.fsum(joined.loc[joined.approved, "expected_profit_true"])
    indexed = group_table.set_index("B")
    both_groups = {0, 1}.issubset(indexed.index)
    evaluation = {
        "information_scope": "EVALUATOR_ONLY", "bank_expected_portfolio_profit": bank_profit,
        "true_expected_portfolio_profit": true_profit, "profit_estimation_error": bank_profit - true_profit,
        "B1_minus_B0_approval_gap": float(indexed.loc[1, "approval_rate"] - indexed.loc[0, "approval_rate"]) if both_groups else None,
        "B1_minus_B0_approval_gap_pp": float(100 * (indexed.loc[1, "approval_rate"] - indexed.loc[0, "approval_rate"])) if both_groups else None,
        "B1_minus_B0_funded_principal_per_applicant_gap_USD": float(indexed.loc[1, "average_funded_principal_per_applicant"] - indexed.loc[0, "average_funded_principal_per_applicant"]) if both_groups else None,
        "B1_minus_B0_total_funded_principal_gap_USD": float(indexed.loc[1, "total_funded_principal"] - indexed.loc[0, "total_funded_principal"]) if both_groups else None,
        "raw_negative_predictions": int((joined.bank_PD_raw < 0).sum()),
        "raw_negative_predictions_approved": int(((joined.bank_PD_raw < 0) & joined.approved).sum()),
    }
    return group_table, evaluation
