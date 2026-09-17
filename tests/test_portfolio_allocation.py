"""Deterministic tests for population-scale fixed-request allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fair_lending.economic_lending.allocation import allocate_fixed_requests
from fair_lending.economic_lending.portfolio import (
    allocation_disagreements,
    allocation_overlap,
    evaluate_portfolio,
    portfolio_group_audit,
    solve_fixed_request_portfolio,
)


def _loans() -> pd.DataFrame:
    return pd.DataFrame({"applicant_id":[1,2,3,4],"option_id":["a","b","c","d"],"requested_principal":[60.,50.,40.,30.]})


def _assessment(profits=(12.,10.,7.,-2.)) -> pd.DataFrame:
    return pd.DataFrame({"applicant_id":[1,2,3,4],"option_id":["a","b","c","d"],"expected_profit_perceived":profits,"repayment_probability_used":[.9,.8,.7,.6]})


def _solve(budget=100., profits=(12.,10.,7.,-2.)):
    return solve_fixed_request_portfolio(_loans(),_assessment(profits),budget,world_id="world",budget_id="budget",policy_id="policy")


def test_milp_matches_tiny_exhaustive_enumeration() -> None:
    milp_solution=_solve()
    exhaustive_assessment=_assessment().rename(columns={"expected_profit_perceived":"expected_profit_true"})
    exhaustive=allocate_fixed_requests(_loans(),exhaustive_assessment,100.)
    selected=tuple(sorted(milp_solution.decisions.loc[milp_solution.decisions.approved,"applicant_id"]))
    assert selected==exhaustive.selected_applicant_ids
    assert milp_solution.solver_metadata["optimality_gap"]==pytest.approx(0.)


def test_budget_option_ownership_and_single_loan_constraints() -> None:
    result=_solve(90.)
    assert result.decisions.funded_amount.sum() <= 90.+1e-8
    assert result.decisions.applicant_id.is_unique
    selected=result.decisions.loc[result.decisions.approved]
    valid=set(map(tuple,_loans()[["applicant_id","option_id"]].to_numpy()))
    assert all((row.applicant_id,row.selected_option_id) in valid for row in selected.itertuples())


def test_nonpositive_perceived_profit_is_never_selected() -> None:
    result=_solve(1000.)
    selected=result.decisions.loc[result.decisions.approved,"applicant_id"].tolist()
    assert 4 not in selected
    assert result.solver_metadata["unused_funds"] > 0


def test_policy_objective_uses_only_supplied_perceived_profit() -> None:
    first=_solve(60.,profits=(100.,1.,1.,1.)).decisions
    second=_solve(60.,profits=(1.,100.,1.,1.)).decisions
    assert first.loc[first.approved,"applicant_id"].tolist()==[1]
    assert 2 in second.loc[second.approved,"applicant_id"].tolist()


def test_decision_schema_contains_no_truth_fields() -> None:
    decisions=_solve().decisions.drop(columns=["perceived_expected_profit","perceived_profit_per_dollar"])
    assert not any("true" in column for column in decisions)
    assert {"approved","funded_amount","selected_option_id","decision_status"}.issubset(decisions)


def test_solver_is_reproducible() -> None:
    pd.testing.assert_frame_equal(_solve().decisions,_solve().decisions)


def test_true_and_realized_portfolio_accounting() -> None:
    decisions=_solve().decisions
    truth=pd.DataFrame({"applicant_id":[1,2,3,4],"repayment_probability_per_period_true":[.9,.8,.7,.6],"full_repayment_probability_true":[.81,.64,.49,.36]})
    economics=pd.DataFrame({"applicant_id":[1,2,3,4],"expected_receipts_true":[75.,62.,45.,25.],"expected_profit_true":[15.,12.,5.,-5.]})
    outcomes=pd.DataFrame({"applicant_id":[1,2,3,4],"realized_total_receipts":[80.,65.,40.,0.],"realized_profit":[20.,15.,0.,-30.]})
    summary,tail=evaluate_portfolio(decisions,_loans(),truth,economics,outcomes,_assessment()[["applicant_id","repayment_probability_used"]],100.)
    ids=decisions.loc[decisions.approved,"applicant_id"]
    assert summary["true_expected_portfolio_profit"]==pytest.approx(economics.loc[economics.applicant_id.isin(ids),"expected_profit_true"].sum())
    assert summary["realized_portfolio_profit"]==pytest.approx(outcomes.loc[outcomes.applicant_id.isin(ids),"realized_profit"].sum())
    assert tail["minimum"]==pytest.approx(economics.loc[economics.applicant_id.isin(ids),"expected_profit_true"].min())


def test_regret_arithmetic_and_oracle_zero() -> None:
    oracle=100.; policy=92.5
    assert oracle-oracle==pytest.approx(0.)
    assert oracle-policy==pytest.approx(7.5)
    assert 100*(oracle-policy)/oracle==pytest.approx(7.5)


def test_overlap_and_jaccard_are_correct() -> None:
    first=_solve(100.).decisions
    second=_solve(60.,profits=(100.,1.,1.,1.)).decisions
    overlap=allocation_overlap(first,second,_loans())
    a=set(first.loc[first.approved,"applicant_id"]); b=set(second.loc[second.approved,"applicant_id"])
    assert overlap["selected_by_both"]==len(a&b)
    assert overlap["jaccard_overlap"]==pytest.approx(len(a&b)/len(a|b))


def test_disagreements_do_not_label_all_unfunded_profitable_loans_errors() -> None:
    oracle=_solve(60.,profits=(100.,1.,1.,1.)).decisions
    policy=_solve(50.,profits=(1.,100.,1.,1.)).decisions
    economics=pd.DataFrame({"applicant_id":[1,2,3,4],"expected_profit_true":[20.,15.,10.,5.]})
    result=allocation_disagreements(oracle,policy,economics)
    assert set(result.disagreement_type)=={"oracle_funded_policy_not_funded","policy_funded_oracle_not_funded"}
    assert "false_rejection" not in result.columns


def test_group_audit_uses_per_applicant_and_funded_denominators() -> None:
    decisions=_solve(100.).decisions
    applicants=pd.DataFrame({"applicant_id":[1,2,3,4],"group":["A","A","B","B"]})
    truth=pd.DataFrame({"applicant_id":[1,2,3,4],"repayment_probability_per_period_true":[.9,.8,.7,.6],"full_repayment_probability_true":[.81,.64,.49,.36]})
    economics=pd.DataFrame({"applicant_id":[1,2,3,4],"expected_receipts_true":[75.,62.,45.,25.],"expected_profit_true":[15.,12.,5.,-5.]})
    audit=portfolio_group_audit(decisions,applicants,_loans(),truth,economics).set_index("group")
    assert audit.loc["A","mean_funded_principal_per_applicant"]==pytest.approx(decisions.loc[:1,"funded_amount"].mean())
    assert audit.loc["B_minus_A","funding_rate"]==pytest.approx(audit.loc["B","funding_rate"]-audit.loc["A","funding_rate"])


def test_same_absolute_budget_can_be_reused_across_policies_and_worlds() -> None:
    total=_loans().requested_principal.sum(); budget=.4*total
    values=[_solve(budget).solver_metadata["bank_budget"] for _ in range(3)]
    assert values==[budget,budget,budget]
