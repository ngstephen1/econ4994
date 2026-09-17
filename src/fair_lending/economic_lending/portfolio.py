"""Population-scale fixed-request portfolio allocation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from fair_lending.economic_lending.recovery import atomic_json, digest

OPTIMALITY_ROUNDOFF = 32 * np.finfo(float).eps


@dataclass(frozen=True)
class PortfolioSolution:
    decisions: pd.DataFrame
    solver_metadata: dict[str, Any]


def solve_fixed_request_portfolio(
    loan_options: pd.DataFrame,
    perceived_assessments: pd.DataFrame,
    bank_budget: float,
    *,
    world_id: str,
    budget_id: str,
    policy_id: str,
    time_limit: float = 60.0,
    diagnostic_directory: Path | None = None,
) -> PortfolioSolution:
    """Solve an exact binary fixed-request knapsack with SciPy/HiGHS MILP."""

    if not np.isfinite(bank_budget) or bank_budget < 0:
        raise ValueError("bank_budget must be nonnegative")
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    required = {"applicant_id", "option_id", "expected_profit_perceived"}
    if not required.issubset(perceived_assessments):
        raise ValueError(f"assessments missing {sorted(required-set(perceived_assessments))}")
    if loan_options["applicant_id"].duplicated().any() or perceived_assessments["applicant_id"].duplicated().any():
        raise ValueError("fixed-request allocation requires one option and assessment per applicant")
    source = loan_options.loc[:, ["applicant_id", "option_id", "requested_principal"]].merge(
        perceived_assessments.loc[:, ["applicant_id", "option_id", "expected_profit_perceived"]],
        on=["applicant_id", "option_id"], validate="one_to_one"
    )
    principal = source["requested_principal"].to_numpy(dtype=float)
    profit = source["expected_profit_perceived"].to_numpy(dtype=float)
    if len(source) != len(loan_options) or len(source) != len(perceived_assessments):
        raise ValueError("loan and assessment keys must match exactly")
    if not np.isfinite(principal).all() or not np.isfinite(profit).all() or (principal < 0).any():
        raise ValueError("finite profits and finite nonnegative principal required")
    upper = (profit > 0).astype(float)
    # A cent-valued request has an integer cost. Flooring K in cents preserves
    # EXACTLY the feasible set; no arbitrary budget guard is subtracted.
    cent_values = [Decimal(str(x)) * 100 for x in principal]
    cent_exact = all(x == x.to_integral_value() and x < 2**52 for x in cent_values)
    if cent_exact:
        cents = np.array([int(x) for x in cent_values], dtype=np.int64)
        divisor = max(1, int(np.gcd.reduce(cents)) if len(cents) else 1)
        budget_cents = int((Decimal(str(bank_budget)) * 100).to_integral_value(rounding=ROUND_FLOOR))
        constraint_coefficients = (cents // divisor).astype(float)
        constraint_upper_bound = float(budget_cents // divisor)
        scaling = "integer_cents_gcd"
    else:
        divisor = None
        constraint_coefficients = principal
        constraint_upper_bound = float(bank_budget)
        scaling = "original_dollars"
    started = perf_counter()
    solver_options = {
        "disp": False,
        "mip_rel_gap": 0.0,
        "time_limit": float(time_limit),
        "mip_feasibility_tolerance": 1e-9,
    }
    case = {
        "world_id": world_id, "budget_id": budget_id, "policy_id": policy_id,
        "bank_budget": float(bank_budget), "solver_options": solver_options,
        "loans": source[["applicant_id", "option_id", "requested_principal"]].to_dict("records"),
        "assessments": source[["applicant_id", "option_id", "expected_profit_perceived"]].to_dict("records"),
    }
    diagnostic_path = None
    if diagnostic_directory is not None:
        diagnostic_path = Path(diagnostic_directory) / f"{digest(case)}.json"
        atomic_json(diagnostic_path, {**case, "status": "started"})
    # SciPy forwards this native HiGHS option, announcing it as unrecognized.
    # Suppress ONLY that forwarding notice; the setting is saved in diagnostics.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*mip_feasibility_tolerance.*", category=RuntimeWarning)
        result = milp(c=-profit, integrality=np.ones(len(source), dtype=int),
                      bounds=Bounds(np.zeros(len(source)), upper),
                      constraints=LinearConstraint(constraint_coefficients.reshape(1, -1),
                                                   lb=-np.inf, ub=constraint_upper_bound),
                      options=solver_options)
    elapsed = perf_counter() - started
    candidate = result.x
    residual = float(principal @ (candidate > .5) - bank_budget) if candidate is not None else None
    integrality_error = float(np.max(np.abs(candidate - np.round(candidate)))) if candidate is not None else None
    if diagnostic_path is not None:
        atomic_json(diagnostic_path, {**case, "status": "solver_returned",
                    "solver_status": int(result.status), "message": str(result.message),
                    "candidate": candidate.tolist() if candidate is not None else None,
                    "integrality_error": integrality_error, "budget_residual_dollars": residual,
                    "reported_mip_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None and np.isfinite(result.mip_gap) else None,
                    "objective": float(-result.fun) if getattr(result, "fun", None) is not None else None,
                    "objective_bound": float(-result.mip_dual_bound) if getattr(result, "mip_dual_bound", None) is not None and np.isfinite(result.mip_dual_bound) else None,
                    "runtime_seconds": elapsed, "constraint_scaling": scaling})
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")
    selected = result.x > 0.5
    funded = np.where(selected, principal, 0.0)
    exact_funded = sum((Decimal(str(p)) for p in principal[selected]), Decimal(0))
    if exact_funded > Decimal(str(bank_budget)):
        raise RuntimeError("MILP returned an infeasible portfolio")
    if integrality_error > 1e-9 or (selected & (upper == 0)).any():
        raise RuntimeError("MILP returned invalid binary decisions")
    gap = float(getattr(result, "mip_gap", np.inf))
    if not np.isfinite(gap) or gap < 0 or gap > OPTIMALITY_ROUNDOFF:
        raise RuntimeError(f"MILP did not certify optimality within floating roundoff: gap={gap}")
    if not np.isclose(-result.fun, profit @ selected, rtol=1e-10, atol=1e-6):
        raise RuntimeError("MILP rounded objective differs from reported objective")
    decisions = pd.DataFrame(
        {
            "world_id": world_id,
            "budget_id": budget_id,
            "policy_id": policy_id,
            "applicant_id": source["applicant_id"].to_numpy(),
            "selected_option_id": source["option_id"].where(selected, pd.NA),
            "approved": selected,
            "funded_amount": funded,
            "decision_status": np.where(selected, "funded", "not_funded"),
            "perceived_expected_profit": profit,
            "perceived_profit_per_dollar": np.divide(profit, principal, out=np.zeros_like(profit), where=principal > 0),
        }
    )
    dual_bound = getattr(result, "mip_dual_bound", np.nan)
    metadata = {
        "world_id": world_id, "budget_id": budget_id, "policy_id": policy_id,
        "solver": "scipy.optimize.milp (HiGHS)", "solver_status_code": int(result.status),
        "solver_status": str(result.message), "success": bool(result.success),
        "perceived_objective_value": float(-result.fun),
        "objective_upper_bound": float(-dual_bound) if np.isfinite(dual_bound) else None,
        "optimality_gap": float(getattr(result, "mip_gap", np.nan)),
        "mip_node_count": int(getattr(result, "mip_node_count", 0)),
        "runtime_seconds": float(elapsed), "bank_budget": float(bank_budget),
        "budget_constraint_scaling": scaling,
        "cent_gcd": divisor,
        "time_limit_seconds": time_limit,
        "mip_feasibility_tolerance": 1e-9,
        "optimality_roundoff_tolerance": OPTIMALITY_ROUNDOFF,
        "budget_constraint_numerical_guard_dollars": 0.0,
        "integrality_error": integrality_error,
        "budget_residual_dollars": float(exact_funded - Decimal(str(bank_budget))),
        "total_funded_principal": float(funded.sum()),
        "unused_funds": float(bank_budget-funded.sum()),
        "budget_feasible": True,
    }
    return PortfolioSolution(decisions=decisions, solver_metadata=metadata)


def evaluate_portfolio(
    decisions: pd.DataFrame,
    loan_options: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    true_economics: pd.DataFrame,
    outcomes: pd.DataFrame,
    perceived_probabilities: pd.DataFrame,
    bank_budget: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a frozen allocation using evaluator-only truth and outcomes."""

    source = (decisions.merge(loan_options[["applicant_id","requested_principal"]],on="applicant_id",validate="one_to_one")
              .merge(simulation_truth,on="applicant_id",validate="one_to_one")
              .merge(true_economics,on="applicant_id",validate="one_to_one")
              .merge(outcomes[["applicant_id","realized_total_receipts","realized_profit"]],on="applicant_id",validate="one_to_one")
              .merge(perceived_probabilities,on="applicant_id",validate="one_to_one"))
    selected = source[source.approved].copy()
    true_profit = selected["expected_profit_true"]
    funded = selected["funded_amount"]
    nonpositive = true_profit <= 0
    total_true = float(true_profit.sum())
    summary = {
        "n_applicants": int(len(source)), "applicants_funded": int(len(selected)),
        "funding_rate": float(len(selected)/len(source)),
        "total_principal_funded": float(funded.sum()),
        "budget_utilization": float(funded.sum()/bank_budget) if bank_budget else 0.0,
        "unused_funds": float(bank_budget-funded.sum()),
        "mean_funded_loan": float(funded.mean()) if len(selected) else 0.0,
        "median_funded_loan": float(funded.median()) if len(selected) else 0.0,
        "mean_funded_principal_per_applicant": float(source.funded_amount.mean()),
        "mean_true_rho_funded": float(selected.repayment_probability_per_period_true.mean()) if len(selected) else np.nan,
        "mean_perceived_rho_funded": float(selected.repayment_probability_used.mean()) if len(selected) else np.nan,
        "mean_true_expected_profit_funded": float(true_profit.mean()) if len(selected) else np.nan,
        "true_expected_portfolio_profit": total_true,
        "perceived_expected_portfolio_profit": float(selected.perceived_expected_profit.sum()),
        "realized_portfolio_receipts": float(selected.realized_total_receipts.sum()),
        "realized_portfolio_profit": float(selected.realized_profit.sum()),
        "realized_minus_true_expected_profit": float(selected.realized_profit.sum()-total_true),
        "truly_nonpositive_funded_count": int(nonpositive.sum()),
        "truly_nonpositive_funded_fraction": float(nonpositive.mean()) if len(selected) else 0.0,
        "signed_true_profit_nonpositive_funded": float(true_profit[nonpositive].sum()),
        "total_true_expected_loss_nonpositive_funded": float(-true_profit[nonpositive].sum()),
    }
    quantiles = true_profit.quantile([.05,.25,.75,.95]) if len(selected) else pd.Series({q:np.nan for q in [.05,.25,.75,.95]})
    tail = {
        "mean": float(true_profit.mean()) if len(selected) else np.nan,
        "median": float(true_profit.median()) if len(selected) else np.nan,
        "p05": float(quantiles.loc[.05]), "p25": float(quantiles.loc[.25]),
        "p75": float(quantiles.loc[.75]), "p95": float(quantiles.loc[.95]),
        "minimum": float(true_profit.min()) if len(selected) else np.nan,
        "largest_true_expected_loss": float(max(0.0,-true_profit.min())) if len(selected) else 0.0,
        "total_true_expected_loss_nonpositive": float(-true_profit[nonpositive].sum()),
    }
    return summary, tail


def allocation_overlap(first: pd.DataFrame, second: pd.DataFrame, loan_options: pd.DataFrame) -> dict[str, float | int]:
    """Return paired selection overlap without treating disagreements as errors."""

    a=set(first.loc[first.approved,"applicant_id"]); b=set(second.loc[second.approved,"applicant_id"])
    both=a & b; union=a | b
    principal=loan_options.set_index("applicant_id")["requested_principal"]
    return {"selected_by_both":len(both),"selected_by_first_only":len(a-b),"selected_by_second_only":len(b-a),
            "jaccard_overlap":float(len(both)/len(union)) if union else 1.0,
            "overlap_principal":float(principal.loc[list(both)].sum()) if both else 0.0,
            "overlap_share_first_principal":float(principal.loc[list(both)].sum()/principal.loc[list(a)].sum()) if a else 1.0}


def allocation_disagreements(oracle: pd.DataFrame, policy: pd.DataFrame, true_economics: pd.DataFrame) -> pd.DataFrame:
    """Summarize oracle-policy disagreements and their true-profit consequence."""

    o=set(oracle.loc[oracle.approved,"applicant_id"]); p=set(policy.loc[policy.approved,"applicant_id"])
    truth=true_economics.set_index("applicant_id")["expected_profit_true"]
    rows=[]
    for label, ids in [("oracle_funded_policy_not_funded",o-p),("policy_funded_oracle_not_funded",p-o)]:
        values=truth.loc[list(ids)] if ids else pd.Series(dtype=float)
        rows.append({"disagreement_type":label,"n_applicants":len(ids),"mean_true_expected_profit":float(values.mean()) if len(values) else np.nan,"total_true_expected_profit":float(values.sum())})
    return pd.DataFrame(rows)


def portfolio_group_audit(decisions: pd.DataFrame, applicants: pd.DataFrame, loan_options: pd.DataFrame,
                          simulation_truth: pd.DataFrame, true_economics: pd.DataFrame) -> pd.DataFrame:
    """Audit neutral-group allocation using explicit access-to-credit denominators."""

    source=(applicants[["applicant_id","group"]].merge(decisions[["applicant_id","approved","funded_amount"]],on="applicant_id",validate="one_to_one")
            .merge(loan_options[["applicant_id","requested_principal"]],on="applicant_id",validate="one_to_one")
            .merge(simulation_truth,on="applicant_id",validate="one_to_one").merge(true_economics,on="applicant_id",validate="one_to_one"))
    rows=[]
    for group,frame in source.groupby("group",observed=True):
        funded=frame[frame.approved]
        rows.append({"group":group,"applicant_count":len(frame),"funding_rate":float(frame.approved.mean()),
                     "total_funded_principal":float(frame.funded_amount.sum()),"mean_funded_principal_per_applicant":float(frame.funded_amount.mean()),
                     "mean_funded_principal_among_funded":float(funded.funded_amount.mean()) if len(funded) else 0.0,
                     "funded_requested_principal_ratio":float(frame.funded_amount.sum()/frame.requested_principal.sum()),
                     "mean_true_rho_funded":float(funded.repayment_probability_per_period_true.mean()) if len(funded) else np.nan,
                     "mean_true_expected_profit_funded":float(funded.expected_profit_true.mean()) if len(funded) else np.nan})
    result=pd.DataFrame(rows); idx=result.set_index("group")
    gap={"group":"B_minus_A","applicant_count":0,"funding_rate":float(idx.loc["B","funding_rate"]-idx.loc["A","funding_rate"]),
         "total_funded_principal":np.nan,"mean_funded_principal_per_applicant":float(idx.loc["B","mean_funded_principal_per_applicant"]-idx.loc["A","mean_funded_principal_per_applicant"]),
         "mean_funded_principal_among_funded":np.nan,"funded_requested_principal_ratio":float(idx.loc["B","funded_requested_principal_ratio"]-idx.loc["A","funded_requested_principal_ratio"]),
         "mean_true_rho_funded":np.nan,"mean_true_expected_profit_funded":np.nan}
    return pd.concat([result,pd.DataFrame([gap])],ignore_index=True)
