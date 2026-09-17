"""Run the matched nonlinear true-risk misspecification experiment."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fair_lending.economic_lending.config import PROJECT_ROOT, load_economic_config
from fair_lending.economic_lending.history import build_at_risk_payment_history
from fair_lending.economic_lending.lender_evaluation import (
    calibration_table, group_risk_audit, probability_recovery_metrics,
    profit_estimation_diagnostics, profit_evaluation_table, realized_label_metrics,
)
from fair_lending.economic_lending.ml import (
    load_ml_config, predict_ml_applicant_risk, tune_hist_gradient_boosting,
)
from fair_lending.economic_lending.nonlinear import (
    load_nonlinear_config, nonlinear_repayment_probabilities,
)
from fair_lending.economic_lending.outcomes import simulate_repayment_outcomes_from_uniforms
from fair_lending.economic_lending.profit import expected_profit_from_options
from fair_lending.economic_lending.repayment import true_repayment_probabilities
from fair_lending.economic_lending.traditional import (
    build_policy_assessments, fit_traditional_logit, predict_applicant_risk,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _revision() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                check=True, capture_output=True, text=True).stdout.strip())
    return {"commit": commit, "dirty_worktree": dirty}


def _distribution(series: pd.Series, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(series.mean()), f"{prefix}_median": float(series.median()),
        f"{prefix}_p05": float(series.quantile(.05)), f"{prefix}_p25": float(series.quantile(.25)),
        f"{prefix}_p75": float(series.quantile(.75)), f"{prefix}_p95": float(series.quantile(.95)),
    }


def _world_summary(name: str, truth: pd.DataFrame, loans: pd.DataFrame) -> dict[str, Any]:
    economics = expected_profit_from_options(loans, truth)
    return {
        "risk_world": name,
        **_distribution(truth["repayment_probability_per_period_true"], "rho"),
        **_distribution(truth["full_repayment_probability_true"], "full_repayment"),
        "mean_true_expected_profit": float(economics["expected_profit_true"].mean()),
        "median_true_expected_profit": float(economics["expected_profit_true"].median()),
        "positive_profit_share": float((economics["expected_profit_true"] > 0).mean()),
    }


def _risk_abs_metrics(pred: pd.DataFrame, truth: pd.DataFrame, column: str) -> dict[str, float]:
    joined = pred.merge(truth, on="applicant_id", validate="one_to_one")
    error = pred[column].to_numpy() - joined["repayment_probability_per_period_true"].to_numpy()
    absolute = np.abs(error)
    return {
        **probability_recovery_metrics(pred, truth, column),
        "absolute_error_p05": float(np.quantile(absolute, .05)),
        "absolute_error_p50": float(np.quantile(absolute, .50)),
        "absolute_error_p95": float(np.quantile(absolute, .95)),
    }


def _save_dgp_figures(applicants: pd.DataFrame, loans: pd.DataFrame, baseline: dict,
                      nonlinear: dict, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    base_app = applicants.iloc[[0]].copy()
    for field in ["annual_income", "credit_score", "employment_years", "liquid_assets"]:
        base_app[field] = applicants[field].median()
    base_loan = loans.iloc[[0]].copy()
    base_loan["requested_ltv"] = .80
    dti = np.linspace(.20, .70, 200)
    a = pd.concat([base_app] * len(dti), ignore_index=True); a["applicant_id"] = np.arange(len(dti))
    l = pd.concat([base_loan] * len(dti), ignore_index=True); l["applicant_id"] = np.arange(len(dti)); l["first_period_dti"] = dti
    rho = nonlinear_repayment_probabilities(a, l, baseline["true_risk"], nonlinear)
    assets = np.geomspace(500, 500000, 200)
    aa = a.copy(); aa["liquid_assets"] = assets
    la = l.copy(); la["first_period_dti"] = .45
    arho = nonlinear_repayment_probabilities(aa, la, baseline["true_risk"], nonlinear)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(dti, rho["repayment_probability_per_period_true"]); axes[0].axvline(.45, ls="--", c="0.4")
    axes[0].set(xlabel="First-period DTI", ylabel="True conditional repayment probability", title="Convex high-DTI penalty")
    axes[1].semilogx(assets, arho["repayment_probability_per_period_true"])
    axes[1].set(xlabel="Liquid assets ($)", ylabel="True conditional repayment probability", title="Bounded asset buffer")
    fig.tight_layout(); fig.savefig(figures / "v2_nonlinear_true_dti_effect.png", dpi=180); plt.close(fig)

    credit = np.linspace(550, 825, 80); ltv = np.linspace(.50, .95, 80)
    cc, ll = np.meshgrid(credit, ltv)
    n = cc.size
    sa = pd.concat([base_app] * n, ignore_index=True); sa["applicant_id"] = np.arange(n); sa["credit_score"] = cc.ravel()
    sl = pd.concat([base_loan] * n, ignore_index=True); sl["applicant_id"] = np.arange(n); sl["requested_ltv"] = ll.ravel(); sl["first_period_dti"] = .45
    srho = nonlinear_repayment_probabilities(sa, sl, baseline["true_risk"], nonlinear)["repayment_probability_per_period_true"].to_numpy().reshape(cc.shape)
    fig, ax = plt.subplots(figsize=(7, 5)); im=ax.pcolormesh(credit, ltv, srho, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="True conditional repayment probability")
    ax.set(xlabel="Credit score", ylabel="Requested LTV", title="Nonlinear credit × LTV risk surface")
    fig.tight_layout(); fig.savefig(figures / "v2_nonlinear_credit_ltv_surface.png", dpi=180); plt.close(fig)


def main() -> int:
    baseline = load_economic_config(); nonlinear = load_nonlinear_config(); ml_config = load_ml_config()
    if nonlinear["baseline_config_fingerprint"] != baseline["metadata"]["config_fingerprint"]:
        raise RuntimeError("nonlinear world is not anchored to the frozen baseline config")
    base_dir = PROJECT_ROOT / baseline["artifacts"]["data_directory"]
    applicants = pd.read_parquet(base_dir / "applicants.parquet")
    loans = pd.read_parquet(base_dir / "loan_options.parquet")
    additive_truth = pd.read_parquet(base_dir / "simulation_truth.parquet")
    regenerated_additive = true_repayment_probabilities(applicants, loans, baseline["true_risk"])
    pd.testing.assert_frame_equal(additive_truth, regenerated_additive)
    nonlinear_truth = nonlinear_repayment_probabilities(applicants, loans, baseline["true_risk"], nonlinear)

    rng = np.random.default_rng(int(nonlinear["repayment_pairing"]["seed"]))
    uniforms = rng.random((len(applicants), int(loans["term_periods"].max())))
    paired_additive_outcomes = simulate_repayment_outcomes_from_uniforms(loans, additive_truth, uniforms)
    nonlinear_outcomes = simulate_repayment_outcomes_from_uniforms(loans, nonlinear_truth, uniforms)
    history = build_at_risk_payment_history(applicants, loans, nonlinear_outcomes)
    cohorts = ("historical_train", "historical_validation", "evaluation")
    histories = {c: history.loc[history.cohort == c].reset_index(drop=True) for c in cohorts}
    app = {c: applicants.loc[applicants.cohort == c].reset_index(drop=True) for c in cohorts}
    loan = {c: loans.loc[loans.applicant_id.isin(app[c].applicant_id)].reset_index(drop=True) for c in cohorts}
    truth = {c: nonlinear_truth.loc[nonlinear_truth.applicant_id.isin(app[c].applicant_id)].reset_index(drop=True) for c in cohorts}

    traditional_model = fit_traditional_logit(histories["historical_train"], baseline["true_risk"])
    selection = tune_hist_gradient_boosting(histories["historical_train"], histories["historical_validation"], ml_config)
    predictions: dict[str, dict[str, pd.DataFrame]] = {}
    for c in ("historical_validation", "evaluation"):
        predictions[c] = {
            "traditional": predict_applicant_risk(traditional_model, app[c], loan[c], baseline["true_risk"]),
            "ml": predict_ml_applicant_risk(selection.estimator, app[c], loan[c]),
        }

    realized=[]; recovery=[]; calibration=[]
    for c in ("historical_validation", "evaluation"):
        oracle=truth[c].rename(columns={"repayment_probability_per_period_true":"oracle_probability"})
        models=[("traditional",predictions[c]["traditional"],"repayment_probability_traditional"),
                ("ml",predictions[c]["ml"],"repayment_probability_ml"),("oracle",oracle,"oracle_probability")]
        for name,p,col in models:
            realized.append({"cohort":c,"model":name,**realized_label_metrics(histories[c],p,col)})
            if name != "oracle": recovery.append({"cohort":c,"model":name,**_risk_abs_metrics(p,truth[c],col)})
            if c == "evaluation":
                ct=calibration_table(histories[c],p,truth[c],col); ct.insert(0,"model",name); calibration.append(ct)
    realized_df=pd.DataFrame(realized); recovery_df=pd.DataFrame(recovery); calibration_df=pd.concat(calibration,ignore_index=True)

    ep=predictions["evaluation"]
    paired=ep["traditional"].merge(ep["ml"],on="applicant_id").merge(truth["evaluation"],on="applicant_id")
    ta=(paired.repayment_probability_traditional-paired.repayment_probability_per_period_true).abs()
    ma=(paired.repayment_probability_ml-paired.repayment_probability_per_period_true).abs()
    paired_abs={"mean_ml_minus_traditional_absolute_error":float((ma-ta).mean()),
                "median_ml_minus_traditional_absolute_error":float((ma-ta).median()),
                "fraction_ml_closer":float((ma < ta).mean())}
    tpolicy=build_policy_assessments(loan["evaluation"],ep["traditional"],policy_id=nonlinear["policies"]["traditional"])
    mpolicy=build_policy_assessments(loan["evaluation"],ep["ml"],policy_id=nonlinear["policies"]["ml"],probability_column="repayment_probability_ml")
    tprofit,tpg=profit_estimation_diagnostics(loan["evaluation"],tpolicy,truth["evaluation"],app["evaluation"])
    mprofit,mpg=profit_estimation_diagnostics(loan["evaluation"],mpolicy,truth["evaluation"],app["evaluation"])
    profit_df=pd.DataFrame([{"model":"traditional",**tprofit},{"model":"ml",**mprofit}])
    td=profit_evaluation_table(loan["evaluation"],tpolicy,truth["evaluation"],app["evaluation"])
    md=profit_evaluation_table(loan["evaluation"],mpolicy,truth["evaluation"],app["evaluation"])
    disagreement=td[["applicant_id","expected_profit_true","expected_profit_perceived"]].rename(columns={"expected_profit_perceived":"traditional_profit"}).merge(
        md[["applicant_id","expected_profit_perceived"]].rename(columns={"expected_profit_perceived":"ml_profit"}),on="applicant_id")
    disagreement["traditional_classification"]=np.where(disagreement.traditional_profit>0,"positive","nonpositive")
    disagreement["ml_classification"]=np.where(disagreement.ml_profit>0,"positive","nonpositive")
    break_even=disagreement.groupby(["traditional_classification","ml_classification"],observed=True).agg(
        n_applicants=("applicant_id","size"),mean_true_expected_profit=("expected_profit_true","mean"),
        median_true_expected_profit=("expected_profit_true","median"),true_positive_profit_share=("expected_profit_true",lambda x:float((x>0).mean()))).reset_index()

    tg=group_risk_audit(app["evaluation"],ep["traditional"],truth["evaluation"],"repayment_probability_traditional").rename(columns={"mean_predicted_rho":"traditional_mean_rho","mean_prediction_error":"traditional_mean_error","mae":"traditional_risk_mae","rmse":"traditional_risk_rmse"})
    mg=group_risk_audit(app["evaluation"],ep["ml"],truth["evaluation"],"repayment_probability_ml").rename(columns={"n_applicants":"ml_n","mean_predicted_rho":"ml_mean_rho","mean_true_rho":"ml_mean_true_rho","mean_prediction_error":"ml_mean_error","mae":"ml_risk_mae","rmse":"ml_risk_rmse"})
    group=tg.merge(mg,on="group").merge(tpg[["group","mae"]].rename(columns={"mae":"traditional_profit_mae"}),on="group").merge(mpg[["group","mae"]].rename(columns={"mae":"ml_profit_mae"}),on="group")

    worlds=pd.DataFrame([_world_summary("additive_logistic_baseline",additive_truth,loans),_world_summary("nonlinear_v1",nonlinear_truth,loans)])
    change=nonlinear_truth.repayment_probability_per_period_true-additive_truth.repayment_probability_per_period_true
    paired_world_change={"mean":float(change.mean()),"sd":float(change.std()),"p05":float(change.quantile(.05)),"median":float(change.median()),"p95":float(change.quantile(.95))}
    baseline_risk=pd.read_csv(PROJECT_ROOT/"results/tables/v2_risk_model_comparison.csv"); baseline_profit=pd.read_csv(PROJECT_ROOT/"results/tables/v2_profit_model_comparison.csv")
    br=baseline_risk[baseline_risk.cohort.eq("evaluation")].set_index("model"); bp=baseline_profit.set_index("model")
    nr=recovery_df[recovery_df.cohort.eq("evaluation")].set_index("model"); npf=profit_df.set_index("model")
    cross=pd.DataFrame([
        {"risk_world":"additive_logistic_baseline","model":"traditional","risk_mae":br.loc["traditional_logit_v1","mae"],"profit_mae":bp.loc["traditional_logit_v1","mae"],"wrong_profit_sign_rate":bp.loc["traditional_logit_v1","wrong_profit_sign_rate"]},
        {"risk_world":"additive_logistic_baseline","model":"ml","risk_mae":br.loc["ml_histgb_v1","mae"],"profit_mae":bp.loc["ml_histgb_v1","mae"],"wrong_profit_sign_rate":bp.loc["ml_histgb_v1","wrong_profit_sign_rate"]},
        {"risk_world":"nonlinear_v1","model":"traditional","risk_mae":nr.loc["traditional","mae"],"profit_mae":npf.loc["traditional","mae"],"wrong_profit_sign_rate":npf.loc["traditional","wrong_profit_sign_rate"]},
        {"risk_world":"nonlinear_v1","model":"ml","risk_mae":nr.loc["ml","mae"],"profit_mae":npf.loc["ml","mae"],"wrong_profit_sign_rate":npf.loc["ml","wrong_profit_sign_rate"]},
    ])

    out=PROJECT_ROOT/nonlinear["artifacts"]["data_directory"]; out.mkdir(parents=True,exist_ok=True)
    applicants.to_parquet(out/"applicants.parquet",index=False); loans.to_parquet(out/"loan_options.parquet",index=False)
    nonlinear_truth.to_parquet(out/"simulation_truth.parquet",index=False); nonlinear_outcomes.to_parquet(out/"loan_outcomes.parquet",index=False)
    pd.concat([tpolicy,mpolicy],ignore_index=True).to_parquet(out/"policy_assessments.parquet",index=False)
    model_dir=PROJECT_ROOT/"results/models"; joblib.dump(selection.estimator,PROJECT_ROOT/nonlinear["artifacts"]["ml_model"])
    model_dir.mkdir(parents=True,exist_ok=True)
    common={"risk_world":"nonlinear_v1","nonlinear_config_fingerprint":nonlinear["metadata"]["config_fingerprint"],"baseline_config_fingerprint":baseline["metadata"]["config_fingerprint"],"code_revision":_revision()}
    _write_json(PROJECT_ROOT/nonlinear["artifacts"]["traditional_model"],{**common,"policy_id":nonlinear["policies"]["traditional"],"model_family":traditional_model.model_family,"features":list(traditional_model.features),"coefficients":traditional_model.coefficients})
    _write_json(PROJECT_ROOT/nonlinear["artifacts"]["ml_metadata"],{**common,"policy_id":nonlinear["policies"]["ml"],"features":list(ml_config["features"]),"selected_parameters":{k:(v.item() if isinstance(v,np.generic) else v) for k,v in selection.selected_parameters.items()},"evaluation_excluded_from_selection":True})

    tables=PROJECT_ROOT/"results/tables"; figures=PROJECT_ROOT/"results/figures"; tables.mkdir(parents=True,exist_ok=True)
    worlds.to_csv(tables/"v2_nonlinear_world_comparison.csv",index=False); recovery_df.merge(realized_df,on=["cohort","model"],how="outer").to_csv(tables/"v2_nonlinear_risk_model_comparison.csv",index=False)
    profit_df.to_csv(tables/"v2_nonlinear_profit_model_comparison.csv",index=False); group.to_csv(tables/"v2_nonlinear_group_audit.csv",index=False)
    calibration_df.to_csv(tables/"v2_nonlinear_calibration_bins.csv",index=False); break_even.to_csv(tables/"v2_nonlinear_break_even_disagreement.csv",index=False); cross.to_csv(tables/"v2_additive_vs_nonlinear_summary.csv",index=False)
    selection.trials.to_csv(tables/"v2_nonlinear_ml_hyperparameter_selection.csv",index=False)

    _save_dgp_figures(applicants,loans,baseline,nonlinear,figures)
    fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].scatter(paired.repayment_probability_per_period_true,paired.repayment_probability_traditional,s=7,alpha=.35); axes[1].scatter(paired.repayment_probability_per_period_true,paired.repayment_probability_ml,s=7,alpha=.35)
    for ax,title in zip(axes,["Traditional additive logit","HistGradientBoosting"]): ax.plot([.85,1],[.85,1],ls="--",c="black"); ax.set(xlabel="True nonlinear rho",ylabel="Predicted rho",title=title)
    fig.tight_layout(); fig.savefig(figures/"v2_nonlinear_risk_predictions_vs_truth.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.hist(ta,bins=50,alpha=.6,label="Traditional"); ax.hist(ma,bins=50,alpha=.6,label="ML"); ax.set(xlabel="Absolute rho error",ylabel="Applicants",title="Nonlinear-world probability error"); ax.legend(); fig.tight_layout(); fig.savefig(figures/"v2_nonlinear_risk_error_distribution.png",dpi=180); plt.close(fig)
    pivot=cross.pivot(index="model",columns="risk_world",values="risk_mae"); pivot.plot(kind="bar",figsize=(7,4)); plt.ylabel("Risk MAE"); plt.title("Model recovery across true-risk worlds"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(figures/"v2_additive_vs_nonlinear_model_comparison.png",dpi=180); plt.close()

    metrics={**common,"selected_ml_parameters":{k:(v.item() if isinstance(v,np.generic) else v) for k,v in selection.selected_parameters.items()},"payment_rows":{c:int(len(histories[c])) for c in cohorts},"world_comparison":worlds.to_dict("records"),"paired_risk_change":paired_world_change,"risk_recovery":recovery_df.to_dict("records"),"realized_metrics":realized_df.to_dict("records"),"paired_absolute_error":paired_abs,"profit":profit_df.to_dict("records"),"break_even":break_even.to_dict("records"),"group_audit":group.to_dict("records"),"paired_outcome_full_repayment_rates":{"additive":float(paired_additive_outcomes.completed_all_payments.mean()),"nonlinear":float(nonlinear_outcomes.completed_all_payments.mean())}}
    _write_json(PROJECT_ROOT/nonlinear["artifacts"]["metrics"],metrics)
    print(json.dumps(metrics,indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
