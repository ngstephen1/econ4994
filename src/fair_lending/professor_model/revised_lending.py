"""Step 9B: delegate cutoff search; audit frozen policies after truth release.

No fitting, risk adjustment, or true-profit recalculation lives here. The
legacy model-ID mapping exists only inside the bank adapter, never in exports.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import math

import numpy as np
import pandas as pd

from .bank_regression import MODEL_ID, validate_ids
from .contracts import validate_contracts
from .lending import BANK_INPUT_COLUMNS, choose_cutoff, validate_solution
from .noise_pilot import exact_cutoff_diagnostics
from .profit import PREVIEW_SCOPE, TRUE_PROFIT_COLUMNS, assess_bank_profit
from .revised_scenarios import SCENARIOS

SCHEMA_VERSION = "professor-revised-lending-v1"
SCENARIO_IDS = tuple(s["scenario_id"] for s in SCENARIOS)
REFERENCE_IDS = ("frozen_group_aware_bank", "frozen_group_blind_bank", "frozen_true_risk_oracle")
POLICY_ORDER = (*SCENARIO_IDS, *REFERENCE_IDS)
BANK_SCOPE = "BANK_ONLY_FROZEN_BEFORE_EVALUATION"
EVALUATOR_SCOPE = PREVIEW_SCOPE
RISK_VIEW = ("applicant_id", "scenario_id", "estimated_PD_bounded")
DECISION_COLUMNS = ("information_scope", "scenario_id", "policy_id", "applicant_id",
                    "estimated_PD_bounded", "optimal_tau", "approved",
                    "requested_principal", "funded_principal", "decision_status")


def optimize_scenario(contracts, risk, budget):
    """Accept only saved bounded scores and contracts, then delegate unchanged."""
    validate_contracts(contracts)
    validate_ids(risk, "scenario bank risk")
    if tuple(risk.columns) != RISK_VIEW or risk.scenario_id.nunique() != 1 or risk.scenario_id.iloc[0] not in SCENARIO_IDS or set(risk.applicant_id) != set(contracts.applicant_id):
        raise ValueError("exact truth/group-free single-scenario risk view required")
    scenario = risk.scenario_id.iloc[0]
    aligned = contracts[["applicant_id"]].merge(risk, on="applicant_id", validate="one_to_one")
    adapter = aligned[["applicant_id", "estimated_PD_bounded"]].rename(columns={"estimated_PD_bounded": "bank_PD_bounded"})
    adapter.insert(1, "model_id", MODEL_ID)
    assessment = assess_bank_profit(contracts, adapter)
    bank = contracts[["applicant_id", "L_i"]].merge(assessment, on="applicant_id", validate="one_to_one").rename(columns={"L_i": "requested_principal"}).loc[:, BANK_INPUT_COLUMNS]
    decisions, candidates, selection = choose_cutoff(bank, budget)
    checks = validate_solution(bank, budget, decisions, candidates, selection)
    checks.update(exact_cutoff_diagnostics(bank, decisions, candidates, selection, budget))

    # Restore honest scenario identity; the legacy ID above is just an adapter.
    decisions = decisions.drop(columns="model_id").rename(columns={"bank_PD_bounded": "estimated_PD_bounded"})
    assessment = assessment.drop(columns="model_id").rename(columns={"bank_PD_bounded": "estimated_PD_bounded"})
    for frame in (decisions, assessment, candidates):
        frame.insert(0, "policy_id", scenario)
        frame.insert(0, "scenario_id", scenario)
        frame.insert(0, "information_scope", BANK_SCOPE)
    selection = {k: v for k, v in selection.items() if k != "model_id"}
    selection.update(scenario_id=scenario, policy_id=scenario,
                     base_model_id=next(s["base_model_id"] for s in SCENARIOS if s["scenario_id"] == scenario))
    return {"decisions": decisions.loc[:, DECISION_COLUMNS], "assessments": assessment,
            "candidates": candidates, "selection": selection, "checks": checks}


def evaluation_policy(decisions, selection, *, score_column, role, oracle=False):
    """Evaluator-only in-memory adapter; original decision files stay untouched."""
    return {
        "role": role, "is_oracle": oracle, "selection": selection,
        "decisions": decisions[["applicant_id", score_column, "approved", "requested_principal", "funded_principal"]].rename(columns={score_column: "score"}).copy(),
    }


def _aligned(frame, reference, name):
    validate_ids(frame, name)
    if set(frame.applicant_id) != set(reference.applicant_id):
        raise ValueError(f"{name}: evaluation IDs differ")
    return frame.sort_values("applicant_id").reset_index(drop=True)


def evaluate_policies(policies, contracts, audit, truth, budget):
    """Aggregate already-frozen masks; no cutoff selection is called here."""
    validate_contracts(contracts)
    base = contracts[["applicant_id", "L_i"]].sort_values("applicant_id").reset_index(drop=True)
    audit, truth = _aligned(audit, base, "group audit"), _aligned(truth, base, "true profit")
    if tuple(audit.columns) != ("applicant_id", "B") or not audit.B.isin([0, 1]).all() or tuple(truth.columns) != TRUE_PROFIT_COLUMNS:
        raise ValueError("exact evaluator group/truth schemas required")
    if not np.isfinite(truth.iloc[:, 1:]).all().all() or not truth.true_PD.between(0, 1).all():
        raise ValueError("invalid evaluator truth")
    if set(policies) != set(POLICY_ORDER):
        raise ValueError("all three scenarios and all three frozen references required")
    base = base.merge(audit, on="applicant_id", validate="one_to_one").merge(truth, on="applicant_id", validate="one_to_one")
    rows, groups, masks = [], [], {}
    for policy_id in POLICY_ORDER:
        policy = policies[policy_id]
        decisions = _aligned(policy["decisions"], base, "frozen decisions")
        selection = policy["selection"]
        if tuple(decisions.columns) != ("applicant_id", "score", "approved", "requested_principal", "funded_principal") or not pd.api.types.is_bool_dtype(decisions.approved):
            raise ValueError("invalid normalized frozen decisions")
        if not np.isfinite(decisions.score).all() or not decisions.score.between(0, 1).all() or selection["available_bank_capital"] != budget:
            raise ValueError("invalid score or unequal capital")
        mask = decisions.approved.to_numpy()
        tau = selection["optimal_tau"]
        np.testing.assert_array_equal(mask, np.zeros(len(base), dtype=bool) if tau is None else decisions.score <= tau)
        np.testing.assert_array_equal(decisions.requested_principal, base.L_i)
        np.testing.assert_array_equal(decisions.funded_principal, np.where(mask, base.L_i, 0.))
        exact_cost = sum((Fraction(float(x)) for x in decisions.funded_principal), Fraction())
        if exact_cost > Fraction(float(budget)) or int(mask.sum()) != selection["number_approved"] or float(exact_cost) != selection["total_funded_principal"]:
            raise ValueError("frozen count/funding/budget mismatch")
        if policy["is_oracle"]:
            np.testing.assert_array_equal(decisions.score, base.true_PD)
        issued = base.loc[mask]
        masks[policy_id] = mask
        true_profit = math.fsum(issued.expected_profit_true)
        row = {"information_scope": EVALUATOR_SCOPE, "policy_id": policy_id,
               "policy_role": policy["role"], "cutoff_score": "true_PD" if policy["is_oracle"] else "estimated_PD_bounded",
               "optimal_tau": tau, "n_applicants": len(base), "number_approved": len(issued),
               "number_rejected": len(base) - len(issued), "approval_rate": len(issued) / len(base),
               "funded_principal_USD": float(exact_cost), "available_capital_USD": budget,
               "capital_utilization": float(exact_cost) / budget if budget else None,
               "unused_capital_USD": budget - float(exact_cost),
               "bank_expected_portfolio_profit_USD": None if policy["is_oracle"] else selection["bank_expected_portfolio_profit"],
               "true_expected_portfolio_profit_USD": true_profit,
               "mean_true_profit_per_approved_USD": true_profit / len(issued) if len(issued) else None,
               "mean_true_PD_approved": float(issued.true_PD.mean()) if len(issued) else None}
        per_group = {}
        for group in (0, 1):
            members = base.B.eq(group).to_numpy()
            funded = members & mask
            count, approved = int(members.sum()), int(funded.sum())
            rate = approved / count if count else None
            per_group[group] = rate
            group_funds = math.fsum(base.loc[funded, "L_i"])
            groups.append({"information_scope": EVALUATOR_SCOPE, "policy_id": policy_id, "policy_role": policy["role"],
                           "B": group, "n_applicants": count, "number_approved": approved,
                           "number_rejected": count - approved, "approval_rate": rate,
                           "funded_principal_USD": group_funds,
                           "funded_principal_per_applicant_USD": group_funds / count if count else None,
                           "mean_estimated_bounded_PD_approved": float(decisions.loc[funded, "score"].mean()) if approved and not policy["is_oracle"] else None,
                           "mean_true_PD_approved": float(base.loc[funded, "true_PD"].mean()) if approved else None,
                           "true_expected_portfolio_profit_USD": math.fsum(base.loc[funded, "expected_profit_true"])})
        row["approval_gap_pp"] = 100 * (per_group[1] - per_group[0]) if None not in per_group.values() else None
        rows.append(row)
    portfolio, group_table = pd.DataFrame(rows), pd.DataFrame(groups)
    indexed = portfolio.set_index("policy_id")
    for label, reference in (("frozen_bank", REFERENCE_IDS[0]), ("frozen_blind", REFERENCE_IDS[1])):
        portfolio[f"approval_gap_minus_{label}_pp"] = portfolio.approval_gap_pp - indexed.loc[reference, "approval_gap_pp"]
    for label, reference in (("frozen_bank", REFERENCE_IDS[0]), ("frozen_oracle", REFERENCE_IDS[2])):
        portfolio[f"true_profit_minus_{label}_USD"] = portfolio.true_expected_portfolio_profit_USD - indexed.loc[reference, "true_expected_portfolio_profit_USD"]
    group_table = group_table.merge(portfolio[["policy_id", "approval_gap_pp", "approval_gap_minus_frozen_bank_pp", "approval_gap_minus_frozen_blind_pp"]], on="policy_id", validate="many_to_one")

    overlaps = []
    for first, second in combinations(POLICY_ORDER, 2):
        a, b = masks[first], masks[second]
        union = int((a | b).sum())
        jaccard = int((a & b).sum()) / union if union else None
        for label, selected, funded_a, funded_b in (
            ("approved_by_both", a & b, True, True), ("first_only", a & ~b, True, False),
            ("second_only", ~a & b, False, True), ("rejected_by_both", ~a & ~b, False, False),
        ):
            requests = base.loc[selected]
            potential = math.fsum(requests.expected_profit_true)
            overlaps.append({"information_scope": EVALUATOR_SCOPE, "policy_first": first, "policy_second": second,
                             "allocation_category": label, "n_applicants": len(requests),
                             "share_of_evaluation": len(requests) / len(base), "jaccard_overlap": jaccard,
                             "requested_principal_USD": math.fsum(requests.L_i),
                             "potential_true_expected_profit_USD": potential,
                             "first_policy_expected_profit_contribution_USD": potential if funded_a else 0.,
                             "second_policy_expected_profit_contribution_USD": potential if funded_b else 0.,
                             "profit_scope": "hypothetical_unfunded_by_both" if not funded_a and not funded_b else "expected_contribution_only_to_funding_policy_or_policies"})
    return portfolio, group_table, pd.DataFrame(overlaps)


def zero_tie_outcomes(risks, decisions, contracts, budget):
    """Post-freeze tie audit, including group denominators and selection status."""
    rows = []
    for scenario in SCENARIO_IDS:
        risk = risks.loc[risks.scenario_id == scenario]
        selected = decisions.loc[decisions.scenario_id == scenario]
        validate_ids(risk, "tie risk")
        selected = _aligned(selected, risk, "tie decisions")
        joined = risk.merge(selected[["applicant_id", "approved"]], on="applicant_id", validate="one_to_one").merge(contracts[["applicant_id", "L_i"]], on="applicant_id", validate="one_to_one")
        if len(joined) != len(risk):
            raise ValueError("incomplete tie join")
        zero = joined.loc[joined.estimated_PD_bounded.eq(0)]
        count, approved = len(zero), int(zero.approved.sum())
        if approved not in (0, count):
            raise ValueError("selected policy splits zero-score tie")
        cost = sum((Fraction(float(v)) for v in zero.L_i), Fraction())
        reject_all = not bool(joined.approved.any())
        row = {"information_scope": EVALUATOR_SCOPE, "scenario_id": scenario,
               "zero_score_count": count, "zero_score_share": count / len(joined),
               "zero_block_requested_principal_USD": float(cost), "available_capital_USD": budget,
               "zero_block_budget_pct": 100 * float(cost) / budget if budget else None,
               "zero_block_feasible": cost <= Fraction(float(budget)),
               "zero_block_approved_count": approved, "whole_zero_block_approved": bool(count and approved == count),
               "reject_all_selected": reject_all,
               "zero_block_action": "no_zero_scores" if not count else "approved_whole" if approved == count else "rejected_all"}
        for group in (0, 1):
            n = int(joined.B.eq(group).sum())
            k = int(zero.B.eq(group).sum())
            row.update({f"B{group}_applicants": n, f"B{group}_zero_count": k, f"B{group}_zero_share": k / n if n else None})
        rows.append(row)
    return pd.DataFrame(rows)
