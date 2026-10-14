"""Step 7B: one raw-PN OLS sensitivity, then matched frozen-policy evaluation.

No generation or private-data lookup. Only the explicit fit view reaches OLS.
Old bank/oracle code and feature contracts remain unchanged.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .bank_regression import MODEL_ID as AWARE_ID, error_metrics, ids_fingerprint, validate_ids
from .config import configuration_fingerprint
from .lending import BANK_INPUT_COLUMNS, choose_cutoff
from .profit import expected_cashflows

MODEL_ID = "professor_bank_ols_group_blind_v1"
FEATURES = ("Y", "D", "C")
DESIGN = ("const", *FEATURES)
FIT_COLUMNS = ("applicant_id", "split", "Y", "D", "C", "PN")
RAW = "bank_PD_group_blind_raw"
BOUNDED = "bank_PD_group_blind_bounded"
PREDICTION_COLUMNS = ("applicant_id", "split", "model_id", RAW, BOUNDED, "prediction_out_of_bounds")
PROFIT_COLUMNS = ("applicant_id", "model_id", BOUNDED, "expected_receipts_bank", "expected_profit_bank")
DECISION_COLUMNS = ("applicant_id", "model_id", BOUNDED, "optimal_tau", "approved", "requested_principal", "funded_principal", "decision_status")


def design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Exactly [1, Y dollar levels, D ratio, C original signal]."""
    values = frame.loc[:, FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values[:, 0] > 0).all():
        raise ValueError("finite observed predictors and positive income required")
    result = pd.DataFrame(values, index=frame.index, columns=FEATURES)
    result.insert(0, "const", 1.)
    return result


def fit_group_blind(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit once on saved training IDs; no evaluation target enters the solver."""
    validate_ids(frame, "group-blind fit")
    if tuple(frame.columns) != FIT_COLUMNS or set(frame.split) != {"train", "evaluation"}:
        raise ValueError("exact truth-free group-blind fit view and saved cohorts required")
    train = frame.loc[frame.split == "train"].sort_values("applicant_id")
    matrix = design_matrix(train)
    target = train.PN.to_numpy(dtype=float)
    if not np.isfinite(target).all() or len(train) <= len(DESIGN) or np.linalg.matrix_rank(matrix) != len(DESIGN):
        raise ValueError("finite raw target and full-rank training design required")
    fit = sm.OLS(target, matrix, missing="raise").fit(method="pinv")
    coefficients = {term: float(fit.params[term]) for term in DESIGN}
    if not np.isfinite(list(coefficients.values())).all():
        raise ValueError("nonfinite coefficients")
    return {
        "schema_version": "professor-group-blind-model-v1", "model_id": MODEL_ID,
        "family": "statsmodels_OLS", "solver": "pinv", "regularization": None,
        "feature_order": list(FEATURES), "design_order": list(DESIGN), "intercept": True,
        "target": "PN", "target_transform": "none", "prediction_rescaling": "none",
        "bounding": "clip_0_1_separate_column", "coefficients": coefficients,
        "training_count": len(train), "training_ID_fingerprint": ids_fingerprint(train.applicant_id),
        "training_target_fingerprint": configuration_fingerprint({"IDs": train.applicant_id.tolist(), "PN": target.tolist()}),
        "evaluation_ID_fingerprint": ids_fingerprint(frame.loc[frame.split == "evaluation", "applicant_id"]),
        "training_mean_PN": float(target.mean()), "design_rank": len(DESIGN),
        "design_condition_number": float(np.linalg.cond(matrix)),
        "units": {"const": "raw PN units", "Y": "annual synthetic USD, levels", "D": "DTI ratio", "C": "original continuous signal"},
    }


def predict_group_blind(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    """Explicit selection makes B/private/target changes irrelevant to prediction."""
    if model["model_id"] != MODEL_ID or tuple(model["feature_order"]) != FEATURES or tuple(model["design_order"]) != DESIGN or set(model["coefficients"]) != set(DESIGN) or model["intercept"] is not True:
        raise ValueError("incompatible group-blind model contract")
    coef = np.array([model["coefficients"][term] for term in DESIGN])
    raw = design_matrix(frame).to_numpy() @ coef
    if not np.isfinite(coef).all() or not np.isfinite(raw).all():
        raise ValueError("nonfinite predictions")
    return raw


def risk_estimates(frame: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    validate_ids(frame, "group-blind predictions")
    if not frame.split.isin(["train", "evaluation"]).all():
        raise ValueError("invalid prediction cohorts")
    raw = predict_group_blind(frame, model)
    return pd.DataFrame({"applicant_id": frame.applicant_id.to_numpy(), "split": frame.split.to_numpy(), "model_id": MODEL_ID, RAW: raw, BOUNDED: np.clip(raw, 0, 1), "prediction_out_of_bounds": (raw < 0) | (raw > 1)}).loc[:, PREDICTION_COLUMNS]


def assess_profit(contracts: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    """Accept exactly ID/model/bounded score, not raw scores or evaluator truth."""
    validate_ids(contracts, "contracts")
    validate_ids(estimates, "group-blind bounded risk")
    if tuple(estimates.columns) != ("applicant_id", "model_id", BOUNDED) or not estimates.model_id.eq(MODEL_ID).all() or set(estimates.applicant_id) != set(contracts.applicant_id):
        raise ValueError("bounded group-blind risk schema/IDs required")
    aligned = contracts[["applicant_id"]].merge(estimates, on="applicant_id", validate="one_to_one")
    receipts, profit = expected_cashflows(contracts, aligned[BOUNDED].to_numpy())
    return aligned.assign(expected_receipts_bank=receipts, expected_profit_bank=profit).loc[:, PROFIT_COLUMNS]


def optimize_cutoff(contracts: pd.DataFrame, assessments: pd.DataFrame, budget: float):
    """Reuse the old exact prefix search with a local, observable-only adapter.

    The old search requires its original model ID. Map only inside this call,
    then immediately restore the new ID/score name; never modify old allowlists.
    """
    validate_ids(assessments, "group-blind assessments")
    if tuple(assessments.columns) != PROFIT_COLUMNS or not assessments.model_id.eq(MODEL_ID).all() or set(assessments.applicant_id) != set(contracts.applicant_id):
        raise ValueError("truth-free group-blind assessments required")
    adapter = contracts[["applicant_id", "L_i"]].merge(assessments, on="applicant_id", validate="one_to_one").rename(columns={"L_i": "requested_principal", BOUNDED: "bank_PD_bounded"})
    adapter["model_id"] = AWARE_ID
    decisions, candidates, selection = choose_cutoff(adapter.loc[:, BANK_INPUT_COLUMNS], budget)
    decisions["model_id"] = MODEL_ID
    decisions = decisions.rename(columns={"bank_PD_bounded": BOUNDED})
    selection["model_id"] = MODEL_ID
    return decisions.loc[:, DECISION_COLUMNS], candidates, selection


def _same_ids(reference: pd.DataFrame, *frames: pd.DataFrame) -> None:
    for frame in (reference, *frames):
        validate_ids(frame, "comparison")
        if set(frame.applicant_id) != set(reference.applicant_id):
            raise ValueError("comparison requires identical complete applicant IDs")


def compare_risk(observed: pd.DataFrame, aware: pd.DataFrame, blind: pd.DataFrame, truth: pd.DataFrame):
    """Evaluator-only held-out PN/hidden-risk metrics, group audit, paired changes."""
    evaluation = observed.loc[observed.split == "evaluation"].copy()
    aware = aware.loc[aware.split == "evaluation"]
    blind = blind.loc[blind.split == "evaluation"]
    _same_ids(evaluation, aware, blind, truth)
    joined = evaluation.merge(aware[["applicant_id", "bank_PD_raw", "bank_PD_bounded"]], on="applicant_id", validate="one_to_one").merge(blind[["applicant_id", RAW, BOUNDED]], on="applicant_id", validate="one_to_one").merge(truth[["applicant_id", "true_PD"]], on="applicant_id", validate="one_to_one")
    rows, groups, changes = [], [], []
    for policy, columns in (("group_aware", ("bank_PD_raw", "bank_PD_bounded")), ("group_blind", (RAW, BOUNDED))):
        for kind, column in zip(("raw", "bounded"), columns):
            for target, reference in (("held_out_PN", "PN"), ("hidden_true_PD", "true_PD")):
                rows.append({"information_scope": "EVALUATOR_ONLY", "policy": policy, "prediction": kind, "reference": target, "n": len(joined), **error_metrics(joined[reference].to_numpy(), joined[column].to_numpy())})
            for group, frame in joined.groupby("B", sort=True):
                groups.append({"policy": policy, "prediction": kind, "B": int(group), "n": len(frame), "mean_true_PD": float(frame.true_PD.mean()), "mean_predicted_PD": float(frame[column].mean()), **error_metrics(frame.true_PD.to_numpy(), frame[column].to_numpy())})
    for scope, frame in [("overall", joined), *[(f"B={int(b)}", f) for b, f in joined.groupby("B", sort=True)]]:
        for kind, a, b in (("raw", "bank_PD_raw", RAW), ("bounded", "bank_PD_bounded", BOUNDED)):
            delta = frame[b].to_numpy() - frame[a].to_numpy()
            changes.append({"scope": scope, "prediction": kind, "n": len(frame), "mean_blind_minus_aware": float(delta.mean()), "mean_absolute_change": float(np.abs(delta).mean()), "RMSE_change": float(np.sqrt(np.mean(delta**2))), "minimum_change": float(delta.min()), "maximum_change": float(delta.max())})
    return pd.DataFrame(rows), pd.DataFrame(groups), pd.DataFrame(changes)


def compare_allocations(contracts, truth, audit, policies, budget):
    """Read three frozen decisions; do not fit or optimize any policy here.

    policies maps label -> (decisions, saved selection, score column,
    perceived profit by applicant or None for oracle).
    """
    _same_ids(contracts, truth, audit, *[item[0] for item in policies.values()])
    base = contracts[["applicant_id", "L_i"]].merge(truth, on="applicant_id", validate="one_to_one").merge(audit[["applicant_id", "B"]], on="applicant_id", validate="one_to_one").sort_values("applicant_id").reset_index(drop=True)
    if not audit.B.isin([0, 1]).all():
        raise ValueError("invalid audit group")
    rows, groups, masks = [], [], {}
    for policy, (decisions, selection, score, profit) in policies.items():
        aligned = decisions.sort_values("applicant_id").reset_index(drop=True)
        if not pd.api.types.is_bool_dtype(aligned.approved) or selection["available_bank_capital"] != budget:
            raise ValueError("boolean decisions and exact shared saved budget required")
        mask = aligned.approved.to_numpy()
        tau = selection["optimal_tau"]
        np.testing.assert_array_equal(mask, np.zeros(len(mask), dtype=bool) if tau is None else aligned[score].to_numpy() <= tau)
        np.testing.assert_array_equal(aligned.requested_principal, base.L_i)
        np.testing.assert_array_equal(aligned.funded_principal, np.where(mask, base.L_i, 0.))
        funded = math.fsum(aligned.funded_principal)
        if funded > budget or funded != selection["total_funded_principal"] or int(mask.sum()) != selection["number_approved"]:
            raise ValueError("frozen count/capital mismatch")
        if policy == "oracle":
            np.testing.assert_array_equal(aligned[score], base.true_PD)
        perceived = None
        if profit is not None:
            _same_ids(base, profit)
            values = base[["applicant_id"]].merge(profit, on="applicant_id", validate="one_to_one")
            perceived = math.fsum(values.loc[mask, "expected_profit_bank"])
            if perceived != selection["bank_expected_portfolio_profit"]:
                raise ValueError("frozen perceived objective mismatch")
        issued = base.loc[mask]
        masks[policy] = mask
        rows.append({"information_scope": "EVALUATOR_ONLY", "policy": policy, "cutoff_score": score, "optimal_cutoff": tau, "n_applicants": len(base), "number_approved": len(issued), "approval_rate": float(mask.mean()), "funded_principal": funded, "available_bank_capital": budget, "capital_utilization": funded / budget if budget else None, "perceived_portfolio_profit": perceived, "true_expected_portfolio_profit": math.fsum(issued.expected_profit_true), "mean_true_PD_approved": float(issued.true_PD.mean()) if len(issued) else None})
        for group, frame in base.groupby("B", sort=True):
            selected = frame.loc[mask[frame.index]]
            group_funds = math.fsum(selected.L_i)
            groups.append({"information_scope": "EVALUATOR_ONLY", "policy": policy, "B": int(group), "n_applicants": len(frame), "number_approved": len(selected), "approval_rate": len(selected) / len(frame), "funded_principal": group_funds, "funded_principal_per_applicant": group_funds / len(frame)})
    summary, group_table = pd.DataFrame(rows), pd.DataFrame(groups)
    by_policy = summary.set_index("policy")
    oracle_profit = by_policy.loc["oracle", "true_expected_portfolio_profit"]
    summary["signed_oracle_minus_policy_true_profit"] = oracle_profit - summary.true_expected_portfolio_profit
    summary["signed_difference_pct_of_oracle"] = (100 * summary.signed_oracle_minus_policy_true_profit / oracle_profit) if oracle_profit else np.nan
    for policy in policies:
        g = group_table.loc[group_table.policy == policy].set_index("B")
        for field, source, scale in (("approval_gap_pp", "approval_rate", 100), ("funded_principal_gap_per_applicant_USD", "funded_principal_per_applicant", 1)):
            summary.loc[summary.policy == policy, field] = scale * (g.loc[1, source] - g.loc[0, source]) if {0, 1}.issubset(g.index) else np.nan
    a, b = masks["group_aware"], masks["group_blind"]
    union = int((a | b).sum())
    overlap = []
    for scope, scope_mask in [("overall", np.ones(len(base), dtype=bool)), ("B=0", (base.B == 0).to_numpy()), ("B=1", (base.B == 1).to_numpy())]:
        for label, mask in (("approved_by_both", a & b), ("group_aware_only", a & ~b), ("group_blind_only", b & ~a), ("rejected_by_both", ~a & ~b)):
            frame = base.loc[mask & scope_mask]
            overlap.append({"information_scope": "EVALUATOR_ONLY", "scope": scope, "allocation_group": label, "n_applicants": len(frame), "mean_true_PD": float(frame.true_PD.mean()) if len(frame) else None, "mean_true_expected_profit": float(frame.expected_profit_true.mean()) if len(frame) else None, "total_true_expected_profit": math.fsum(frame.expected_profit_true)})
    indexed = summary.set_index("policy")
    comparison = {"jaccard_overlap": int((a & b).sum()) / union if union else None,
                  "group_blind_minus_aware_approval_gap_pp": float(indexed.loc["group_blind", "approval_gap_pp"] - indexed.loc["group_aware", "approval_gap_pp"]),
                  "group_blind_minus_aware_true_profit_USD": float(indexed.loc["group_blind", "true_expected_portfolio_profit"] - indexed.loc["group_aware", "true_expected_portfolio_profit"]),
                  "group_blind_minus_aware_funded_principal_gap_per_applicant_USD": float(indexed.loc["group_blind", "funded_principal_gap_per_applicant_USD"] - indexed.loc["group_aware", "funded_principal_gap_per_applicant_USD"])}
    for group in (0, 1):
        g = group_table.loc[group_table.B == group].set_index("policy")
        comparison[f"B{group}_blind_minus_aware_approval_rate_pp"] = float(100 * (g.loc["group_blind", "approval_rate"] - g.loc["group_aware", "approval_rate"])) if len(g) else None
    return summary, group_table, pd.DataFrame(overlap), comparison
