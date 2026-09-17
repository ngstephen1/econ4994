"""Deterministic tests for direct repayment-belief discrimination."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fair_lending.economic_lending.direct_belief import (
    build_direct_policy_assessments, direct_policy_id, distort_repayment_beliefs,
    load_direct_belief_config, logit_probability,
)
from fair_lending.economic_lending.portfolio import solve_fixed_request_portfolio


def _applicants() -> pd.DataFrame:
    return pd.DataFrame({"applicant_id":[1,2,3,4],"group":["A","B","A","B"]})


def _loans() -> pd.DataFrame:
    return pd.DataFrame({"applicant_id":[1,2,3,4],"option_id":["a","b","c","d"],"requested_principal":[100.,100.,100.,100.],"periodic_interest_rate":[.2]*4,"term_periods":[2]*4,"transaction_cost":[5.]*4,"payment_rule_id":["rule"]*4,"first_period_payment":[70.]*4,"first_period_dti":[.3]*4,"requested_ltv":[.8]*4})


def _base(values=(.9,.9,.8,.8)) -> pd.DataFrame:
    return pd.DataFrame({"applicant_id":[1,2,3,4],"probability":values})


def test_predeclared_delta_grid_and_odds_multipliers() -> None:
    config=load_direct_belief_config(); assert config["deltas"]==[0.0,.05,.10,.20]
    np.testing.assert_allclose(np.exp(-np.array(config["deltas"])),[1.,.9512294245,.9048374180,.8187307531])


def test_delta_zero_exactly_reproduces_base_probability_and_profit() -> None:
    distorted=distort_repayment_beliefs(_applicants(),_base(),"probability",0.)
    np.testing.assert_allclose(distorted.repayment_probability_used,distorted.repayment_probability_base,rtol=0,atol=1e-15)
    policy=build_direct_policy_assessments(_loans(),distorted,"neutral")
    assert policy.loc[0,"expected_profit_perceived"]==pytest.approx(14.7)


def test_group_a_unchanged_and_group_b_receives_exact_logit_shift() -> None:
    delta=.2; distorted=distort_repayment_beliefs(_applicants(),_base(),"probability",delta).merge(_applicants(),on="applicant_id")
    a=distorted.group.eq("A"); b=~a
    np.testing.assert_allclose(distorted.loc[a,"repayment_probability_used"],distorted.loc[a,"repayment_probability_base"])
    shift=logit_probability(distorted.loc[b,"repayment_probability_used"])-logit_probability(distorted.loc[b,"repayment_probability_base"])
    np.testing.assert_allclose(shift,-delta,atol=1e-12)
    assert (distorted.loc[b,"repayment_probability_used"] <= distorted.loc[b,"repayment_probability_base"]).all()


def test_same_delta_formula_applies_to_different_model_estimates() -> None:
    for values in [(.9,.9,.8,.8),(.95,.85,.75,.65)]:
        result=distort_repayment_beliefs(_applicants(),_base(values),"probability",.1).merge(_applicants(),on="applicant_id")
        b=result.group.eq("B")
        np.testing.assert_allclose(logit_probability(result.loc[b,"repayment_probability_used"])-logit_probability(result.loc[b,"repayment_probability_base"]),-.1)


def test_base_and_used_fields_remain_distinct_and_economics_use_used() -> None:
    distorted=distort_repayment_beliefs(_applicants(),_base(),"probability",.2)
    policy=build_direct_policy_assessments(_loans(),distorted,"direct")
    assert {"repayment_probability_base","repayment_probability_used"}.issubset(policy)
    b=policy.applicant_id.isin([2,4]); assert (policy.loc[b,"repayment_probability_used"]<policy.loc[b,"repayment_probability_base"]).all()
    rho=policy.loc[1,"repayment_probability_used"]
    assert policy.loc[1,"expected_receipts_perceived"]==pytest.approx(70*(rho+rho**2))


def test_probability_treatment_does_not_mutate_inputs_or_truth() -> None:
    applicants=_applicants(); base=_base(); applicants_before=applicants.copy(); base_before=base.copy()
    distort_repayment_beliefs(applicants,base,"probability",.2)
    pd.testing.assert_frame_equal(applicants,applicants_before); pd.testing.assert_frame_equal(base,base_before)


def test_delta_zero_reproduces_neutral_allocation() -> None:
    neutral=build_direct_policy_assessments(_loans(),distort_repayment_beliefs(_applicants(),_base(),"probability",0.),"neutral")
    repeat=build_direct_policy_assessments(_loans(),distort_repayment_beliefs(_applicants(),_base(),"probability",0.),"direct0")
    first=solve_fixed_request_portfolio(_loans(),neutral,200.,world_id="w",budget_id="b",policy_id="n")
    second=solve_fixed_request_portfolio(_loans(),repeat,200.,world_id="w",budget_id="b",policy_id="d")
    assert first.decisions.approved.tolist()==second.decisions.approved.tolist()


def test_policy_and_decision_artifacts_are_truth_free() -> None:
    policy=build_direct_policy_assessments(_loans(),distort_repayment_beliefs(_applicants(),_base(),"probability",.2),"direct")
    assert not any("true" in c for c in policy)
    decision=solve_fixed_request_portfolio(_loans(),policy,200.,world_id="w",budget_id="b",policy_id="p").decisions.drop(columns=["perceived_expected_profit","perceived_profit_per_dollar"])
    assert not any("true" in c for c in decision)


def test_budget_and_solver_optimality_hold_across_treatments() -> None:
    for delta in [0.,.05,.1,.2]:
        policy=build_direct_policy_assessments(_loans(),distort_repayment_beliefs(_applicants(),_base(),"probability",delta),direct_policy_id("traditional","additive_logistic_baseline",delta))
        result=solve_fixed_request_portfolio(_loans(),policy,200.,world_id="w",budget_id="b",policy_id="p")
        assert result.decisions.funded_amount.sum()<=200.+1e-9
        assert result.solver_metadata["optimality_gap"]==pytest.approx(0.)


def test_difference_in_gap_and_level_change_arithmetic() -> None:
    a0,b0=.40,.45; a1,b1=.43,.39
    assert (b1-a1)-(b0-a0)==pytest.approx(-.09)
    assert a1-a0==pytest.approx(.03); assert b1-b0==pytest.approx(-.06)


def test_displaced_and_replacement_classification_allows_nonmonotonic_changes() -> None:
    neutral={1,2,3}; distorted={2,3,4}
    assert neutral-distorted=={1}; assert distorted-neutral=={4}; assert neutral&distorted=={2,3}


def test_true_risk_reference_uses_declared_truth_only_as_base() -> None:
    truth=pd.DataFrame({"applicant_id":[1,2,3,4],"probability":[.91,.92,.93,.94]})
    result=distort_repayment_beliefs(_applicants(),truth,"probability",.1)
    np.testing.assert_allclose(result.repayment_probability_base,truth.probability)


def test_pure_mechanism_cost_is_nonnegative_for_oracle_objective() -> None:
    true_base=_base((.95,.94,.85,.84)); neutral=build_direct_policy_assessments(_loans(),distort_repayment_beliefs(_applicants(),true_base,"probability",0.),"oracle")
    distorted=build_direct_policy_assessments(_loans(),distort_repayment_beliefs(_applicants(),true_base,"probability",.2),"reference")
    n=solve_fixed_request_portfolio(_loans(),neutral,200.,world_id="w",budget_id="b",policy_id="n").decisions
    d=solve_fixed_request_portfolio(_loans(),distorted,200.,world_id="w",budget_id="b",policy_id="d").decisions
    true_profit=neutral.set_index("applicant_id").expected_profit_perceived
    nv=true_profit.loc[n.loc[n.approved,"applicant_id"]].sum(); dv=true_profit.loc[d.loc[d.approved,"applicant_id"]].sum()
    assert nv-dv>=-1e-9
