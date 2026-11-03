"""Approval-calibrated smaller penalty; no fitting or evaluator information.

This successor reuses frozen scores. The search intentionally uses prediction-
cohort group approval counts for an illustrative policy calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import nov2_followup as parent
from .bank_regression import ids_fingerprint, validate_ids
from .contracts import validate_contracts

SCENARIOS = parent.SCENARIOS
METHODS = parent.METHODS
CANDIDATES = (.005, .0025, .00125, .000625, .0003125, .00015625)
EXPERIMENT_ID = "professor-small-penalty-v1"


def _audit(reference, audit):
    validate_ids(reference, "prediction rows")
    validate_ids(audit, "group audit")
    if tuple(audit.columns) != ("applicant_id", "B") or not audit.B.isin([0, 1]).all():
        raise ValueError("exact observable-only ID/B audit required")
    reference = reference.sort_values("applicant_id").reset_index(drop=True)
    audit = audit.sort_values("applicant_id").reset_index(drop=True)
    if reference.applicant_id.tolist() != audit.applicant_id.tolist() or set(audit.B) != {0, 1}:
        raise ValueError("matching IDs and both groups required")
    return reference, audit


def adjust_penalty(control_predictions, audit, delta):
    """Add delta to Group B raw scores before clipping; Group A is unchanged."""
    if (isinstance(delta, (bool, np.bool_)) or not isinstance(delta, (int, float, np.number))
            or not np.isfinite(delta) or delta <= 0):
        raise ValueError("positive finite penalty required")
    if tuple(control_predictions.columns) != parent.PREDICTION_COLUMNS:
        raise ValueError("exact truth-free prediction schema required")
    pred, groups = _audit(control_predictions, audit)
    if (not pred.method_id.eq("ols_control").all() or pred.scenario_id.nunique() != 1
            or pred.scenario_id.iloc[0] not in SCENARIOS
            or not np.isfinite(pred[["raw_PD", "bounded_PD"]].to_numpy()).all()
            or not np.array_equal(pred.bounded_PD, np.clip(pred.raw_PD, 0, 1))):
        raise ValueError("one valid frozen OLS control required")
    result = pred.copy()
    result["method_id"] = "ols_penalty"
    result["raw_PD"] = pred.raw_PD.to_numpy() + float(delta) * groups.B.to_numpy()
    result["bounded_PD"] = np.clip(result.raw_PD, 0, 1)
    np.testing.assert_array_equal(result.loc[groups.B.eq(0), "raw_PD"], pred.loc[groups.B.eq(0), "raw_PD"])
    return result.loc[:, parent.PREDICTION_COLUMNS]


def group_criterion(decisions, audit):
    """Strict positive B approval and lower B rate, using exact integers."""
    allowed = {"applicant_id", "approved", "scenario_id", "method_id", "bounded_PD",
               "optimal_tau", "requested_principal", "funded_principal", "decision_status"}
    if not set(decisions.columns).issubset(allowed):
        raise ValueError("group criterion accepts bank decision columns only")
    if "approved" not in decisions or not pd.api.types.is_bool_dtype(decisions.approved):
        raise ValueError("boolean approval decisions required")
    decisions, groups = _audit(decisions, audit)
    n_a, n_b = int(groups.B.eq(0).sum()), int(groups.B.eq(1).sum())
    a = int(decisions.loc[groups.B.eq(0), "approved"].sum())
    b = int(decisions.loc[groups.B.eq(1), "approved"].sum())
    return {"n_A": n_a, "n_B": n_b, "approved_A": a, "approved_B": b,
            "rate_A": a / n_a, "rate_B": b / n_b,
            "criterion_met": b > 0 and b * n_a < a * n_b}


def select_common_penalty(contracts, controls_by_scenario, audits_by_scenario, budget, *,
                          candidates=CANDIDATES, optimizer=parent.optimize_policy):
    """Stop at first common success. No truth or true-profit argument exists."""
    if set(controls_by_scenario) != set(SCENARIOS) or set(audits_by_scenario) != set(SCENARIOS):
        raise ValueError("exactly the two declared scenarios required")
    grid = tuple(candidates)
    if not grid or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
                       or not np.isfinite(v) or v <= 0 for v in grid):
        raise ValueError("finite positive candidate sequence required")
    if any(a <= b for a, b in zip(grid, grid[1:])):
        raise ValueError("candidates must be strictly descending")
    rows, records = [], []
    for index, delta in enumerate(grid):
        policies, current = {}, []
        for scenario in SCENARIOS:
            predictions = adjust_penalty(controls_by_scenario[scenario], audits_by_scenario[scenario], delta)
            if not predictions.scenario_id.eq(scenario).all():
                raise ValueError("control scenario identity mismatch")
            policy = optimizer(contracts, predictions, budget)
            criteria = group_criterion(policy["decisions"], audits_by_scenario[scenario])
            selection = policy["selection"]
            if selection["available_bank_capital"] != budget or selection["total_funded_principal"] > budget:
                raise ValueError("original budget not respected")
            row = {"candidate_index": index, "delta": float(delta), "scenario_id": scenario,
                   **criteria, "optimal_tau": selection["optimal_tau"],
                   "funded_principal_USD": selection["total_funded_principal"],
                   "available_capital_USD": float(budget)}
            current.append(row); policies[scenario] = policy
            records.append({"candidate_index": index, "delta": float(delta), "scenario_id": scenario,
                            "group_criterion": criteria, "selection": selection,
                            "checks": policy["checks"]})
        common = all(row["criterion_met"] for row in current)
        for row in current:
            row.update(common_criterion_met=common, selected=common)
        rows.extend(current)
        if common:
            return {"status": "criterion_met", "selected_delta": float(delta),
                    "attempts": pd.DataFrame(rows), "attempt_records": records,
                    "selected_policies": policies}
    return {"status": "criterion_unmet", "selected_delta": None,
            "attempts": pd.DataFrame(rows), "attempt_records": records,
            "selected_policies": {}}


def validate_prediction_cohort(splits, contracts, predictions, audit):
    """Prediction IDs alone may receive contracts, decisions and profit audits."""
    if tuple(splits.columns) != ("applicant_id", "split") or set(splits.split) != {"train", "evaluation"}:
        raise ValueError("original train/evaluation split required")
    validate_ids(splits, "original split"); validate_contracts(contracts)
    evaluation = set(splits.loc[splits.split.eq("evaluation"), "applicant_id"])
    training = set(splits.loc[splits.split.eq("train"), "applicant_id"])
    if evaluation & training or set(contracts.applicant_id) != evaluation:
        raise ValueError("contracts must contain exactly the disjoint prediction cohort")
    _, audit = _audit(contracts, audit)
    if tuple(predictions.columns) != parent.PREDICTION_COLUMNS or predictions.empty:
        raise ValueError("exact prediction columns required")
    if predictions.duplicated(["scenario_id", "method_id", "applicant_id"]).any():
        raise ValueError("duplicate prediction cells")
    for (scenario, method), frame in predictions.groupby(["scenario_id", "method_id"]):
        if scenario not in SCENARIOS or method not in METHODS or set(frame.applicant_id) != evaluation:
            raise ValueError("prediction cells must contain exactly evaluation IDs")
    return {"training_rows": len(training), "prediction_rows": len(evaluation),
            "training_prediction_disjoint": True, "prediction_only_contracts_and_audit": True,
            "training_ID_fingerprint": ids_fingerprint(pd.Series(sorted(training))),
            "prediction_ID_fingerprint": ids_fingerprint(pd.Series(sorted(evaluation)))}
