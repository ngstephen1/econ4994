"""Allocate evaluation-cohort lending portfolios under frozen V2 risk policies."""

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
from fair_lending.economic_lending.nonlinear import load_nonlinear_config
from fair_lending.economic_lending.portfolio import (
    allocation_disagreements, allocation_overlap, evaluate_portfolio,
    portfolio_group_audit, solve_fixed_request_portfolio,
)
from fair_lending.economic_lending.profit import expected_profit_from_options


BUDGETS = {"nonbinding_100pct": 1.0, "moderate_40pct": .4, "tight_20pct": .2}
CURVE_FRACTIONS = tuple(np.arange(.1, 1.01, .1).round(2))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _load_world(world_id: str, directory: Path) -> dict[str, pd.DataFrame]:
    frames={name:pd.read_parquet(directory/f"{name}.parquet") for name in
            ["applicants","loan_options","simulation_truth","loan_outcomes","policy_assessments"]}
    ids=frames["applicants"].loc[frames["applicants"].cohort.eq("evaluation"),"applicant_id"]
    frames["applicants"]=frames["applicants"].loc[frames["applicants"].applicant_id.isin(ids)].reset_index(drop=True)
    for name in ["loan_options","simulation_truth","loan_outcomes"]:
        frames[name]=frames[name].loc[frames[name].applicant_id.isin(ids)].reset_index(drop=True)
    frames["world_id"]=world_id
    return frames


def _policy_inputs(world: dict[str, Any]) -> dict[str, pd.DataFrame]:
    loans=world["loan_options"]; truth=world["simulation_truth"]
    true_econ=expected_profit_from_options(loans,truth)
    oracle=(loans[["applicant_id","option_id"]].merge(truth[["applicant_id","repayment_probability_per_period_true"]],on="applicant_id")
            .merge(true_econ,on="applicant_id").rename(columns={"repayment_probability_per_period_true":"repayment_probability_used","expected_profit_true":"expected_profit_perceived"}))
    saved=world["policy_assessments"]
    traditional=saved.loc[saved.policy_id.str.startswith("traditional_logit")].reset_index(drop=True)
    ml=saved.loc[saved.policy_id.str.startswith("ml_histgb")].reset_index(drop=True)
    if len(oracle)!=2000 or len(traditional)!=2000 or len(ml)!=2000:
        raise RuntimeError("each frozen evaluation policy must contain 2,000 applicants")
    return {"oracle":oracle,"traditional":traditional,"ml":ml}


def _policy_probability(assessment: pd.DataFrame) -> pd.DataFrame:
    return assessment[["applicant_id","repayment_probability_used"]]


def main() -> int:
    baseline=load_economic_config(); nonlinear=load_nonlinear_config()
    worlds={
        "additive_logistic_baseline":_load_world("additive_logistic_baseline",PROJECT_ROOT/baseline["artifacts"]["data_directory"]),
        "nonlinear_v1":_load_world("nonlinear_v1",PROJECT_ROOT/nonlinear["artifacts"]["data_directory"]),
    }
    pd.testing.assert_frame_equal(worlds["additive_logistic_baseline"]["applicants"],worlds["nonlinear_v1"]["applicants"])
    pd.testing.assert_frame_equal(worlds["additive_logistic_baseline"]["loan_options"],worlds["nonlinear_v1"]["loan_options"])
    total_requested=float(worlds["additive_logistic_baseline"]["loan_options"].requested_principal.sum())
    budgets={name:total_requested*fraction for name,fraction in BUDGETS.items()}

    decisions=[]; solver=[]; summaries=[]; tails=[]; groups=[]; overlaps=[]; disagreements=[]; regrets=[]
    primary_solutions: dict[tuple[str,str,str],pd.DataFrame]={}
    for world_id,world in worlds.items():
        inputs=_policy_inputs(world); true_econ=expected_profit_from_options(world["loan_options"],world["simulation_truth"])
        policy_ids={"oracle":f"oracle_true_risk_{'additive' if world_id.startswith('additive') else 'nonlinear'}_v1",
                    "traditional":str(inputs["traditional"].policy_id.iloc[0]),"ml":str(inputs["ml"].policy_id.iloc[0])}
        for budget_id,budget in budgets.items():
            local={}
            for policy,assessment in inputs.items():
                solution=solve_fixed_request_portfolio(world["loan_options"],assessment,budget,world_id=world_id,budget_id=budget_id,policy_id=policy_ids[policy])
                local[policy]=solution.decisions; primary_solutions[(world_id,budget_id,policy)]=solution.decisions
                solver.append(solution.solver_metadata)
                summary,tail=evaluate_portfolio(solution.decisions,world["loan_options"],world["simulation_truth"],true_econ,world["loan_outcomes"],_policy_probability(assessment),budget)
                key={"world_id":world_id,"budget_id":budget_id,"budget_fraction":BUDGETS[budget_id],"bank_budget":budget,"policy":policy,"policy_id":policy_ids[policy]}
                summaries.append({**key,**summary}); tails.append({**key,**tail})
                audit=portfolio_group_audit(solution.decisions,world["applicants"],world["loan_options"],world["simulation_truth"],true_econ)
                for _,row in audit.iterrows(): groups.append({**key,**row.to_dict()})
                persist=solution.decisions.drop(columns=["perceived_expected_profit","perceived_profit_per_dollar"])
                decisions.append(persist)
            oracle_true=next(x["true_expected_portfolio_profit"] for x in summaries if x["world_id"]==world_id and x["budget_id"]==budget_id and x["policy"]=="oracle")
            for policy in ["oracle","traditional","ml"]:
                value=next(x["true_expected_portfolio_profit"] for x in summaries if x["world_id"]==world_id and x["budget_id"]==budget_id and x["policy"]==policy)
                regret=oracle_true-value
                regrets.append({"world_id":world_id,"budget_id":budget_id,"bank_budget":budget,"policy":policy,"oracle_true_expected_profit":oracle_true,"policy_true_expected_profit":value,"regret_dollars":regret,"regret_percent_oracle":100*regret/oracle_true if oracle_true else np.nan})
            for first,second in [("traditional","oracle"),("ml","oracle"),("traditional","ml")]:
                overlaps.append({"world_id":world_id,"budget_id":budget_id,"first_policy":first,"second_policy":second,**allocation_overlap(local[first],local[second],world["loan_options"])})
            for policy in ["traditional","ml"]:
                frame=allocation_disagreements(local["oracle"],local[policy],true_econ)
                for _,row in frame.iterrows(): disagreements.append({"world_id":world_id,"budget_id":budget_id,"policy":policy,**row.to_dict()})

    summary_df=pd.DataFrame(summaries); regret_df=pd.DataFrame(regrets); overlap_df=pd.DataFrame(overlaps)
    tail_df=pd.DataFrame(tails); group_df=pd.DataFrame(groups); disagreement_df=pd.DataFrame(disagreements); solver_df=pd.DataFrame(solver)
    cross=summary_df.merge(regret_df[["world_id","budget_id","policy","regret_dollars","regret_percent_oracle"]],on=["world_id","budget_id","policy"])
    oracle_overlap=overlap_df.loc[overlap_df.second_policy.eq("oracle"),["world_id","budget_id","first_policy","jaccard_overlap"]].rename(columns={"first_policy":"policy","jaccard_overlap":"oracle_jaccard_overlap"})
    cross=cross.merge(oracle_overlap,on=["world_id","budget_id","policy"],how="left")
    cross.loc[cross.policy.eq("oracle"),"oracle_jaccard_overlap"]=1.0

    # Exploratory fixed-grid scarcity curve; primary regimes above remain unchanged.
    curve=[]
    for world_id,world in worlds.items():
        inputs=_policy_inputs(world); true_econ=expected_profit_from_options(world["loan_options"],world["simulation_truth"])
        for fraction in CURVE_FRACTIONS:
            budget=total_requested*float(fraction); local={}
            for policy,assessment in inputs.items():
                sol=solve_fixed_request_portfolio(world["loan_options"],assessment,budget,world_id=world_id,budget_id=f"curve_{fraction:.1f}",policy_id=policy)
                value=float(sol.decisions.loc[sol.decisions.approved,"applicant_id"].to_frame().merge(true_econ,on="applicant_id").expected_profit_true.sum())
                local[policy]=value
            for policy,value in local.items(): curve.append({"world_id":world_id,"budget_fraction":float(fraction),"policy":policy,"true_expected_portfolio_profit":value,"regret_dollars":local["oracle"]-value})
    curve_df=pd.DataFrame(curve)

    tables=PROJECT_ROOT/"results/tables"; figures=PROJECT_ROOT/"results/figures"; tables.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    outputs={"v2_portfolio_summary.csv":summary_df,"v2_portfolio_regret.csv":regret_df,"v2_portfolio_overlap.csv":overlap_df,"v2_portfolio_group_audit.csv":group_df,"v2_portfolio_disagreements.csv":disagreement_df,"v2_portfolio_tail_diagnostics.csv":tail_df,"v2_additive_nonlinear_portfolio_comparison.csv":cross,"v2_portfolio_solver_diagnostics.csv":solver_df,"v2_portfolio_scarcity_curve.csv":curve_df}
    for name,frame in outputs.items(): frame.to_csv(tables/name,index=False)
    decision_path=PROJECT_ROOT/"data/processed/economic_lending/v2_portfolio/decisions.parquet"; decision_path.parent.mkdir(parents=True,exist_ok=True); pd.concat(decisions,ignore_index=True).to_parquet(decision_path,index=False)

    colors={"oracle":"#333333","traditional":"#2878B5","ml":"#E07A1F"}
    world_order=["additive_logistic_baseline","nonlinear_v1"]; budget_order=["nonbinding_100pct","moderate_40pct","tight_20pct"]
    labels=[f"{('Additive' if w.startswith('additive') else 'Nonlinear')}\n{b.split('_')[-1].replace('pct','%')}" for w in world_order for b in budget_order]
    def ordered_pivot(frame,value,policies):
        pivot=frame.pivot_table(index=["world_id","budget_id"],columns="policy",values=value)
        return pivot.reindex(pd.MultiIndex.from_product([world_order,budget_order],names=["world_id","budget_id"]))[policies]
    def bars(value,ylabel,title,path):
        pivot=ordered_pivot(summary_df,value,["oracle","traditional","ml"])
        ax=pivot.plot(kind="bar",figsize=(11,5),color=[colors[x] for x in pivot.columns]); ax.set(ylabel=ylabel,title=title,xlabel=""); ax.set_xticklabels(labels,rotation=0); plt.tight_layout(); plt.savefig(figures/path,dpi=180); plt.close()
    bars("true_expected_portfolio_profit","True expected portfolio profit ($)","Portfolio value under fixed capital budgets","v2_portfolio_true_profit.png")
    piv=ordered_pivot(regret_df,"regret_dollars",["traditional","ml"]); ax=piv.plot(kind="bar",figsize=(11,5),color=[colors["traditional"],colors["ml"]]); ax.set(ylabel="Expected-profit regret ($)",title="Economic regret relative to true-risk oracle",xlabel=""); ax.set_xticklabels(labels,rotation=0); plt.tight_layout(); plt.savefig(figures/"v2_portfolio_regret.png",dpi=180); plt.close()
    op_source=overlap_df.loc[overlap_df.second_policy.eq("oracle")].rename(columns={"first_policy":"policy"}); op=ordered_pivot(op_source,"jaccard_overlap",["traditional","ml"]); ax=op.plot(kind="bar",figsize=(11,5),color=[colors["traditional"],colors["ml"]]); ax.set(ylabel="Jaccard overlap with oracle",title="Portfolio overlap with oracle allocation",xlabel="",ylim=(0,1)); ax.set_xticklabels(labels,rotation=0); plt.tight_layout(); plt.savefig(figures/"v2_portfolio_oracle_overlap.png",dpi=180); plt.close()
    ga=group_df.loc[group_df.group.isin(["A","B"])].copy(); ga["label"]=ga.apply(lambda r:f"{('Add' if str(r.world_id).startswith('additive') else 'Nonlin')}-{str(r.budget_id).split('_')[-1].replace('pct','%')}\n{str(r.policy).title()}",axis=1); pivot=ga.pivot_table(index="label",columns="group",values="funding_rate"); ax=pivot.plot(kind="bar",figsize=(14,5)); ax.set(ylabel="Funding rate",title="Neutral group audit: funding rates",xlabel=""); plt.xticks(rotation=35,ha="right"); plt.tight_layout(); plt.savefig(figures/"v2_portfolio_group_funding.png",dpi=180); plt.close()
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));
    for world_id,ls in [("additive_logistic_baseline","-"),("nonlinear_v1","--")]:
        for policy in ["oracle","traditional","ml"]:
            f=curve_df[(curve_df.world_id==world_id)&(curve_df.policy==policy)]; axes[0].plot(f.budget_fraction,f.true_expected_portfolio_profit,label=f"{world_id}: {policy}",ls=ls,color=colors[policy]); axes[1].plot(f.budget_fraction,f.regret_dollars,label=f"{world_id}: {policy}",ls=ls,color=colors[policy])
    axes[0].set(xlabel="Budget fraction",ylabel="True expected profit ($)",title="Exploratory capital-scarcity curve"); axes[1].set(xlabel="Budget fraction",ylabel="Regret ($)",title="Regret across capital scarcity"); axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(figures/"v2_portfolio_scarcity_curve.png",dpi=180); plt.close(fig)

    metrics={"total_requested_principal":total_requested,"budgets":budgets,"budget_fractions":BUDGETS,"solver_runs":solver,"portfolio_summary":summaries,"regret":regrets,"overlap":overlaps,"disagreements":disagreements,"decision_artifact":str(decision_path.relative_to(PROJECT_ROOT)),"primary_decision_rows":int(sum(len(x) for x in decisions)),"exploratory_curve_points":int(len(curve_df))}
    _write_json(PROJECT_ROOT/"results/metrics/v2_portfolio_allocation_benchmark.json",metrics)
    print(json.dumps(metrics,indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
