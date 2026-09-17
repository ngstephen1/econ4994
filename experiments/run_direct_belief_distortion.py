"""Run direct discrimination treatments through distorted repayment beliefs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fair_lending.economic_lending.config import PROJECT_ROOT, load_economic_config
from fair_lending.economic_lending.direct_belief import (
    build_direct_policy_assessments, direct_policy_id, distort_repayment_beliefs,
    load_direct_belief_config,
)
from fair_lending.economic_lending.nonlinear import load_nonlinear_config
from fair_lending.economic_lending.portfolio import (
    allocation_overlap, evaluate_portfolio, portfolio_group_audit,
    solve_fixed_request_portfolio,
)
from fair_lending.economic_lending.profit import (
    expected_economics_from_probability, expected_profit_from_options,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _load_world(world_id: str, directory: Path) -> dict[str, Any]:
    frames={name:pd.read_parquet(directory/f"{name}.parquet") for name in
            ["applicants","loan_options","simulation_truth","loan_outcomes","policy_assessments"]}
    ids=frames["applicants"].loc[frames["applicants"].cohort.eq("evaluation"),"applicant_id"]
    for name in ["applicants","loan_options","simulation_truth","loan_outcomes"]:
        frames[name]=frames[name].loc[frames[name].applicant_id.isin(ids)].reset_index(drop=True)
    frames["world_id"]=world_id
    return frames


def _base_probabilities(world: dict[str, Any]) -> dict[str,pd.DataFrame]:
    saved=world["policy_assessments"]
    traditional=saved.loc[saved.policy_id.str.startswith("traditional_logit"),["applicant_id","repayment_probability_base"]].rename(columns={"repayment_probability_base":"probability"})
    ml=saved.loc[saved.policy_id.str.startswith("ml_histgb"),["applicant_id","repayment_probability_base"]].rename(columns={"repayment_probability_base":"probability"})
    reference=world["simulation_truth"][["applicant_id","repayment_probability_per_period_true"]].rename(columns={"repayment_probability_per_period_true":"probability"})
    if not all(len(x)==2000 for x in [traditional,ml,reference]): raise RuntimeError("frozen evaluation probabilities incomplete")
    return {"traditional":traditional.reset_index(drop=True),"ml":ml.reset_index(drop=True),"true_risk_reference":reference.reset_index(drop=True)}


def _preallocation(world_id: str, family: str, delta: float, applicants: pd.DataFrame,
                   loans: pd.DataFrame, probabilities: pd.DataFrame, policy: pd.DataFrame) -> dict[str,Any]:
    joined=(applicants[["applicant_id","group"]].merge(probabilities,on="applicant_id")
            .merge(policy[["applicant_id","repayment_probability_base","repayment_probability_used","expected_receipts_perceived","expected_profit_perceived"]],on="applicant_id"))
    base_econ=expected_economics_from_probability(loans,policy,"repayment_probability_base","base_receipts","base_profit")
    joined=joined.merge(base_econ,on="applicant_id")
    b=joined[joined.group.eq("B")]; term=int(loans.term_periods.iloc[0])
    return {"world_id":world_id,"family":family,"delta":delta,"odds_multiplier":float(np.exp(-delta)),
            "group_b_n":len(b),"group_b_mean_rho_base":float(b.repayment_probability_base.mean()),
            "group_b_mean_rho_used":float(b.repayment_probability_used.mean()),
            "group_b_mean_rho_change":float((b.repayment_probability_used-b.repayment_probability_base).mean()),
            "group_b_mean_full_repayment_base":float(np.mean(b.repayment_probability_base**term)),
            "group_b_mean_full_repayment_used":float(np.mean(b.repayment_probability_used**term)),
            "group_b_mean_full_repayment_change":float(np.mean(b.repayment_probability_used**term-b.repayment_probability_base**term)),
            "group_b_mean_receipts_change":float((b.expected_receipts_perceived-b.base_receipts).mean()),
            "group_b_mean_profit_change":float((b.expected_profit_perceived-b.base_profit).mean())}


def _subset_quality(ids: set, applicants: pd.DataFrame, truth: pd.DataFrame, economics: pd.DataFrame) -> dict[str,Any]:
    frame=(applicants.loc[applicants.applicant_id.isin(ids),["applicant_id","group"]]
           .merge(truth,on="applicant_id").merge(economics,on="applicant_id"))
    return {"n":len(frame),"mean_rho_true":float(frame.repayment_probability_per_period_true.mean()) if len(frame) else np.nan,
            "median_rho_true":float(frame.repayment_probability_per_period_true.median()) if len(frame) else np.nan,
            "mean_true_expected_profit":float(frame.expected_profit_true.mean()) if len(frame) else np.nan,
            "median_true_expected_profit":float(frame.expected_profit_true.median()) if len(frame) else np.nan,
            "total_true_expected_profit":float(frame.expected_profit_true.sum())}


def _matching(frame: pd.DataFrame, **values: Any) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, value in values.items():
        mask &= frame[column].eq(value)
    return frame.loc[mask]


def main() -> int:
    baseline=load_economic_config(); nonlinear=load_nonlinear_config(); config=load_direct_belief_config()
    worlds={"additive_logistic_baseline":_load_world("additive_logistic_baseline",PROJECT_ROOT/baseline["artifacts"]["data_directory"]),
            "nonlinear_v1":_load_world("nonlinear_v1",PROJECT_ROOT/nonlinear["artifacts"]["data_directory"])}
    pd.testing.assert_frame_equal(worlds["additive_logistic_baseline"]["applicants"],worlds["nonlinear_v1"]["applicants"])
    pd.testing.assert_frame_equal(worlds["additive_logistic_baseline"]["loan_options"],worlds["nonlinear_v1"]["loan_options"])
    budgets={k:float(v) for k,v in config["budgets"].items()}; deltas=[float(x) for x in config["deltas"]]
    policies={}; pre=[]
    for world_id,world in worlds.items():
        for family,base in _base_probabilities(world).items():
            for delta in deltas:
                distorted=distort_repayment_beliefs(world["applicants"],base,"probability",delta)
                pid=direct_policy_id(family,world_id,delta)
                policy=build_direct_policy_assessments(world["loan_options"],distorted,pid)
                policies[(world_id,family,delta)]=policy
                pre.append(_preallocation(world_id,family,delta,world["applicants"],world["loan_options"],base,policy))

    decisions={}; solver=[]; summaries=[]; group_rows=[]; nonpositive=[]
    for world_id,world in worlds.items():
        economics=expected_profit_from_options(world["loan_options"],world["simulation_truth"])
        for family in ["traditional","ml","true_risk_reference"]:
            for delta in deltas:
                policy=policies[(world_id,family,delta)]
                for budget_id,budget in budgets.items():
                    solution=solve_fixed_request_portfolio(world["loan_options"],policy,budget,world_id=world_id,budget_id=budget_id,policy_id=str(policy.policy_id.iloc[0]))
                    key=(world_id,family,delta,budget_id); decisions[key]=solution.decisions; solver.append({"family":family,"delta":delta,**solution.solver_metadata})
                    summary,_=evaluate_portfolio(solution.decisions,world["loan_options"],world["simulation_truth"],economics,world["loan_outcomes"],policy[["applicant_id","repayment_probability_used"]],budget)
                    summaries.append({"world_id":world_id,"family":family,"delta":delta,"budget_id":budget_id,"bank_budget":budget,"policy_id":str(policy.policy_id.iloc[0]),**summary})
                    audit=portfolio_group_audit(solution.decisions,world["applicants"],world["loan_options"],world["simulation_truth"],economics)
                    for _,row in audit.iterrows(): group_rows.append({"world_id":world_id,"family":family,"delta":delta,"budget_id":budget_id,**row.to_dict()})
                    selected=(solution.decisions.loc[solution.decisions.approved,["applicant_id"]].merge(world["applicants"][["applicant_id","group"]],on="applicant_id").merge(economics,on="applicant_id"))
                    bad=selected[selected.expected_profit_true<=0]
                    counts=bad.groupby("group",observed=True).size().to_dict()
                    nonpositive.append({"world_id":world_id,"family":family,"delta":delta,"budget_id":budget_id,"nonpositive_count":len(bad),"nonpositive_fraction":len(bad)/len(selected) if len(selected) else 0.,"total_true_expected_loss":float(-bad.expected_profit_true.sum()),"group_a_count":int(counts.get("A",0)),"group_b_count":int(counts.get("B",0))})

    pre_df=pd.DataFrame(pre); summary=pd.DataFrame(summaries).merge(pd.DataFrame(nonpositive),on=["world_id","family","delta","budget_id"])
    groups=pd.DataFrame(group_rows); solver_df=pd.DataFrame(solver)
    group_effect=[]; overlap=[]; displaced=[]; replacements=[]; interactions=[]; oracle_reference=[]
    for world_id,world in worlds.items():
        economics=expected_profit_from_options(world["loan_options"],world["simulation_truth"])
        oracle_values={b:float(_matching(summary,world_id=world_id,family="true_risk_reference",delta=0.0,budget_id=b).true_expected_portfolio_profit.iloc[0]) for b in budgets}
        for family in ["traditional","ml","true_risk_reference"]:
            for budget_id in budgets:
                neutral_dec=decisions[(world_id,family,0.0,budget_id)]
                oracle_dec=decisions[(world_id,"true_risk_reference",0.0,budget_id)]
                neutral_value=float(_matching(summary,world_id=world_id,family=family,delta=0.0,budget_id=budget_id).true_expected_portfolio_profit.iloc[0])
                for delta in deltas:
                    current=decisions[(world_id,family,delta,budget_id)]
                    current_value=float(_matching(summary,world_id=world_id,family=family,delta=delta,budget_id=budget_id).true_expected_portfolio_profit.iloc[0])
                    neutral_audit=_matching(groups,world_id=world_id,family=family,delta=0.0,budget_id=budget_id).set_index("group")
                    audit=_matching(groups,world_id=world_id,family=family,delta=delta,budget_id=budget_id).set_index("group")
                    row={"world_id":world_id,"family":family,"delta":delta,"budget_id":budget_id}
                    for group in ["A","B"]:
                        row[f"group_{group.lower()}_funding_rate"]=audit.loc[group,"funding_rate"]
                        row[f"group_{group.lower()}_funding_rate_change"]=audit.loc[group,"funding_rate"]-neutral_audit.loc[group,"funding_rate"]
                        row[f"group_{group.lower()}_principal_per_applicant"]=audit.loc[group,"mean_funded_principal_per_applicant"]
                        row[f"group_{group.lower()}_principal_per_applicant_change"]=audit.loc[group,"mean_funded_principal_per_applicant"]-neutral_audit.loc[group,"mean_funded_principal_per_applicant"]
                        row[f"group_{group.lower()}_funded_requested_ratio"]=audit.loc[group,"funded_requested_principal_ratio"]
                    row["funding_gap_b_minus_a"]=row["group_b_funding_rate"]-row["group_a_funding_rate"]
                    row["direct_effect_on_funding_gap"]=row["funding_gap_b_minus_a"]-(neutral_audit.loc["B","funding_rate"]-neutral_audit.loc["A","funding_rate"])
                    row["principal_gap_b_minus_a"]=row["group_b_principal_per_applicant"]-row["group_a_principal_per_applicant"]
                    row["direct_effect_on_principal_gap"]=row["principal_gap_b_minus_a"]-(neutral_audit.loc["B","mean_funded_principal_per_applicant"]-neutral_audit.loc["A","mean_funded_principal_per_applicant"])
                    row["funded_request_gap_b_minus_a"]=row["group_b_funded_requested_ratio"]-row["group_a_funded_requested_ratio"]
                    row["direct_effect_on_funded_request_gap"]=row["funded_request_gap_b_minus_a"]-(neutral_audit.loc["B","funded_requested_principal_ratio"]-neutral_audit.loc["A","funded_requested_principal_ratio"])
                    group_effect.append(row)
                    for comparison,target in [("same_model_neutral",neutral_dec),("neutral_oracle",oracle_dec)]:
                        overlap.append({"world_id":world_id,"family":family,"delta":delta,"budget_id":budget_id,"comparison":comparison,**allocation_overlap(current,target,world["loan_options"])})
                    oracle_regret=oracle_values[budget_id]-current_value; neutral_regret=oracle_values[budget_id]-neutral_value
                    interactions.append({"world_id":world_id,"family":family,"delta":delta,"budget_id":budget_id,"neutral_model_regret":neutral_regret,"distorted_model_regret":oracle_regret,"incremental_distortion_regret":oracle_regret-neutral_regret,"true_profit_change_from_neutral":current_value-neutral_value,"neutral_minus_distorted_true_profit":neutral_value-current_value})
                    if family=="true_risk_reference": oracle_reference.append({"world_id":world_id,"delta":delta,"budget_id":budget_id,"oracle_true_profit":oracle_values[budget_id],"distorted_reference_true_profit":current_value,"pure_mechanism_cost":oracle_values[budget_id]-current_value,"pure_mechanism_cost_percent":100*(oracle_values[budget_id]-current_value)/oracle_values[budget_id],"oracle_overlap":allocation_overlap(current,oracle_dec,world["loan_options"])["jaccard_overlap"]})
                    if delta>0:
                        neutral_ids=set(neutral_dec.loc[neutral_dec.approved,"applicant_id"]); current_ids=set(current.loc[current.approved,"applicant_id"])
                        group_map=world["applicants"].set_index("applicant_id").group
                        b_displaced={i for i in neutral_ids-current_ids if group_map.loc[i]=="B"}; a_replaced={i for i in current_ids-neutral_ids if group_map.loc[i]=="A"}
                        displaced.append({"world_id":world_id,"family":family,"delta":delta,"budget_id":budget_id,**_subset_quality(b_displaced,world["applicants"],world["simulation_truth"],economics)})
                        replacements.append({"world_id":world_id,"family":family,"delta":delta,"budget_id":budget_id,**_subset_quality(a_replaced,world["applicants"],world["simulation_truth"],economics)})

    effects=pd.DataFrame(group_effect); overlaps=pd.DataFrame(overlap); interaction=pd.DataFrame(interactions); oracle_ref=pd.DataFrame(oracle_reference)
    portfolio_profit=summary.merge(interaction,on=["world_id","family","delta","budget_id"])
    treatment_summary=summary.merge(pre_df,on=["world_id","family","delta"]).merge(effects,on=["world_id","family","delta","budget_id"])

    tables=PROJECT_ROOT/"results/tables"; figures=PROJECT_ROOT/"results/figures"; tables.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    outputs={"v2_direct_treatment_summary.csv":treatment_summary,"v2_direct_group_effects.csv":effects,"v2_direct_portfolio_profit.csv":portfolio_profit,"v2_direct_allocation_overlap.csv":overlaps,"v2_direct_displaced_borrowers.csv":pd.DataFrame(displaced),"v2_direct_replacement_borrowers.csv":pd.DataFrame(replacements),"v2_direct_oracle_reference.csv":oracle_ref,"v2_direct_model_interaction.csv":interaction}
    for name,frame in outputs.items(): frame.to_csv(tables/name,index=False)
    policy_path=PROJECT_ROOT/config["artifacts"]["policy_assessments"]; policy_path.parent.mkdir(parents=True,exist_ok=True)
    fitted_policies=[frame for (world,family,delta),frame in policies.items() if family in {"traditional","ml"}]
    reference_policies=[frame for (world,family,delta),frame in policies.items() if family=="true_risk_reference"]
    pd.concat(fitted_policies,ignore_index=True).to_parquet(policy_path,index=False)
    reference_path=PROJECT_ROOT/config["artifacts"]["evaluator_reference_assessments"]
    pd.concat(reference_policies,ignore_index=True).to_parquet(reference_path,index=False)
    decision_path=PROJECT_ROOT/config["artifacts"]["decisions"]; decision_path.parent.mkdir(parents=True,exist_ok=True)
    persisted=[]
    for key,frame in decisions.items():
        item=frame.drop(columns=["perceived_expected_profit","perceived_profit_per_dollar"]).copy(); item.insert(3,"delta",key[2]); item.insert(3,"family",key[1]); persisted.append(item)
    pd.concat(persisted,ignore_index=True).to_parquet(decision_path,index=False)

    model_effects=effects[effects.family.isin(["traditional","ml"])]
    def lineplot(column,ylabel,title,path,data=model_effects):
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
        for ax,((world,budget),frame) in zip(axes.ravel(),data.groupby(["world_id","budget_id"],sort=False)):
            for family,color in [("traditional","#2878B5"),("ml","#E07A1F")]:
                f=frame[frame.family.eq(family)].sort_values("delta"); ax.plot(f.delta,f[column],marker="o",label=family,color=color)
            ax.axhline(0,color="0.5",lw=.8); ax.set_title(f"{('Additive' if world.startswith('additive') else 'Nonlinear')} · {budget.split('_')[-1].replace('pct','%')}"); ax.set_xlabel("Log-odds distortion δ"); ax.set_ylabel(ylabel)
        axes[0,0].legend(); fig.suptitle(title); fig.tight_layout(); fig.savefig(figures/path,dpi=180); plt.close(fig)
    lineplot("group_b_funding_rate_change","Change from neutral","Group B funding-rate response","v2_direct_group_b_funding_change.png")
    lineplot("direct_effect_on_funding_gap","Difference in gap","Direct effect on B-minus-A funding gap","v2_direct_gap_change.png")
    lineplot("true_profit_change_from_neutral","True profit change ($)","Portfolio-profit effect of belief distortion","v2_direct_true_profit_change.png",portfolio_profit[portfolio_profit.family.isin(["traditional","ml"])])
    lineplot("incremental_distortion_regret","Incremental regret ($)","Incremental regret from belief distortion","v2_direct_incremental_regret.png",interaction[interaction.family.isin(["traditional","ml"])])
    neutral_overlap=overlaps[(overlaps.comparison=="same_model_neutral")&overlaps.family.isin(["traditional","ml"])]
    lineplot("jaccard_overlap","Jaccard overlap","Allocation overlap with same-model neutral portfolio","v2_direct_neutral_overlap.png",neutral_overlap)

    metrics={"mechanism_id":config["mechanism_id"],"config_fingerprint":config["metadata"]["config_fingerprint"],"deltas":deltas,"odds_multipliers":{str(d):float(np.exp(-d)) for d in deltas},"n_unique_policies":len(policies),"n_fitted_policies":len(fitted_policies),"n_evaluator_reference_policies":len(reference_policies),"n_portfolios":len(decisions),"solver_runs":solver,"policy_artifact":str(policy_path.relative_to(PROJECT_ROOT)),"evaluator_reference_artifact":str(reference_path.relative_to(PROJECT_ROOT)),"decision_artifact":str(decision_path.relative_to(PROJECT_ROOT))}
    _write_json(PROJECT_ROOT/config["artifacts"]["metrics"],metrics)
    print(json.dumps(metrics,indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
