"""Run matched upstream-opportunity worlds through risk estimation and allocation."""

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
from fair_lending.economic_lending.history import build_at_risk_payment_history
from fair_lending.economic_lending.ml import (
    load_ml_config,
    predict_ml_applicant_risk,
    tune_hist_gradient_boosting,
)
from fair_lending.economic_lending.nonlinear import (
    load_nonlinear_config,
    nonlinear_repayment_probabilities,
)
from fair_lending.economic_lending.outcomes import simulate_repayment_outcomes_from_uniforms
from fair_lending.economic_lending.portfolio import (
    allocation_overlap,
    evaluate_portfolio,
    portfolio_group_audit,
    solve_fixed_request_portfolio,
)
from fair_lending.economic_lending.profit import expected_profit_from_options
from fair_lending.economic_lending.repayment import true_repayment_probabilities
from fair_lending.economic_lending.systemic import (
    controlled_funding_audit,
    draw_systemic_population_inputs,
    generate_systemic_world,
    identify_opportunity_switchers,
    load_systemic_config,
    validate_matched_worlds,
)
from fair_lending.economic_lending.traditional import (
    build_policy_assessments,
    fit_traditional_logit,
    predict_applicant_risk,
)


RISK_WORLDS = ("additive_logistic_baseline", "nonlinear_v1")
FAMILIES = ("traditional", "ml", "true_risk_reference")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _policy_id(family: str, risk_world: str, strength: float) -> str:
    token = f"{strength:.2f}".replace(".", "p")
    return f"{family}_systemic_{risk_world}_s{token}"


def _oracle_policy(loans: pd.DataFrame, truth: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    probability = truth[["applicant_id", "repayment_probability_per_period_true"]].rename(
        columns={"repayment_probability_per_period_true": "repayment_probability_oracle"}
    )
    return build_policy_assessments(
        loans,
        probability,
        policy_id=policy_id,
        probability_column="repayment_probability_oracle",
    )


def _matching(frame: pd.DataFrame, **values: Any) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, value in values.items():
        mask &= frame[column].eq(value)
    return frame.loc[mask]


def _paired_pathway(
    risk_world: str,
    strength: float,
    neutral: dict[str, pd.DataFrame],
    current: dict[str, pd.DataFrame],
    switchers: pd.DataFrame,
) -> list[dict[str, Any]]:
    fields = ["employment_years", "annual_income", "liquid_assets", "property_value", "requested_loan_amount"]
    neutral_frame = (
        neutral["applicants"][["applicant_id", "group", *fields]]
        .merge(neutral["loan_options"][["applicant_id", "first_period_dti", "requested_ltv"]], on="applicant_id")
        .merge(neutral["truth"], on="applicant_id")
        .merge(neutral["economics"], on="applicant_id")
    )
    current_frame = (
        current["applicants"][["applicant_id", *fields]]
        .merge(current["loan_options"][["applicant_id", "first_period_dti", "requested_ltv"]], on="applicant_id")
        .merge(current["truth"], on="applicant_id")
        .merge(current["economics"], on="applicant_id")
    )
    paired = neutral_frame.merge(current_frame, on="applicant_id", suffixes=("_neutral", "_systemic"))
    switcher_ids = set(switchers.loc[switchers["upstream_displaced"], "applicant_id"])
    rows = []
    variables = [
        *fields,
        "first_period_dti",
        "requested_ltv",
        "repayment_probability_per_period_true",
        "full_repayment_probability_true",
        "expected_profit_true",
    ]
    for population, sample in (
        ("group_b_switchers", paired[paired["applicant_id"].isin(switcher_ids)]),
        ("all_group_b", paired[paired["group"].astype(str).eq("B")]),
    ):
        row: dict[str, Any] = {
            "risk_world": risk_world,
            "systemic_strength": strength,
            "population": population,
            "n": int(len(sample)),
            "share_of_group_b": float(len(sample) / (paired["group"].astype(str).eq("B").sum())),
        }
        for variable in variables:
            difference = sample[f"{variable}_systemic"] - sample[f"{variable}_neutral"]
            row[f"mean_{variable}_neutral"] = float(sample[f"{variable}_neutral"].mean()) if len(sample) else np.nan
            row[f"mean_{variable}_systemic"] = float(sample[f"{variable}_systemic"].mean()) if len(sample) else np.nan
            row[f"mean_change_{variable}"] = float(difference.mean()) if len(sample) else np.nan
        rows.append(row)
    return rows


def _transition_counts(neutral: pd.DataFrame, current: pd.DataFrame, ids: set[str]) -> dict[str, int]:
    first = neutral.set_index("applicant_id")["approved"].astype(bool)
    second = current.set_index("applicant_id")["approved"].astype(bool)
    index = pd.Index(sorted(ids)).intersection(first.index).intersection(second.index)
    a = first.loc[index]
    b = second.loc[index]
    return {
        "funded_both": int((a & b).sum()),
        "neutral_only": int((a & ~b).sum()),
        "systemic_only": int((~a & b).sum()),
        "unfunded_both": int((~a & ~b).sum()),
    }


def _mechanism_figure(path: Path) -> None:
    labels = ["Group + pre-treatment\nlatent characteristics", "Opportunity access", "Employment, income,\nand assets", "Repayment risk", "Group-blind lending\nand portfolio allocation"]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels, strict=True)):
        ax.text(x, .5, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=.45", "facecolor": "#eef3f8", "edgecolor": "#315b7d"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(xs[index + 1] - .09, .5), xytext=(x + .09, .5),
                        arrowprops={"arrowstyle": "->", "color": "#315b7d", "lw": 1.5})
    fig.suptitle("Synthetic systemic-opportunity pathway", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    baseline = load_economic_config()
    systemic = load_systemic_config()
    nonlinear = load_nonlinear_config()
    ml_config = load_ml_config()
    n = int(systemic["artifacts"]["development_rows"])
    seed = int(systemic["randomness"]["development_seed"])
    strengths = [float(value) for value in systemic["systemic_strengths"]]
    budgets = {name: float(value) for name, value in systemic["budgets"].items()}

    draws = draw_systemic_population_inputs(n, baseline, systemic, seed=seed)
    base_worlds = {strength: generate_systemic_world(draws, baseline, systemic, strength) for strength in strengths}
    validations = {
        str(strength): validate_matched_worlds(draws, base_worlds[0.0], base_worlds[strength])
        for strength in strengths
    }
    neutral_mechanism = base_worlds[0.0]["mechanism"]
    switchers = {
        strength: identify_opportunity_switchers(
            neutral_mechanism, base_worlds[strength]["mechanism"], base_worlds[0.0]["applicants"]
        )
        for strength in strengths
    }
    if not any(switchers[s]["upstream_displaced"].any() for s in strengths if s > 0):
        raise RuntimeError("predeclared strength grid produced no Group B opportunity switchers")

    rng = np.random.default_rng(int(systemic["randomness"]["repayment_uniform_seed"]))
    term = int(baseline["contract"]["term_periods"])
    repayment_uniforms = rng.random((n, term))

    worlds: dict[tuple[str, float], dict[str, Any]] = {}
    policies: dict[tuple[str, float, str], pd.DataFrame] = {}
    model_metadata: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    mechanism_reference: list[pd.DataFrame] = []

    for risk_world in RISK_WORLDS:
        for strength in strengths:
            common = base_worlds[strength]
            applicants = common["applicants"]
            loans = common["loan_options"]
            if risk_world == "additive_logistic_baseline":
                truth = true_repayment_probabilities(applicants, loans, baseline["true_risk"])
            else:
                truth = nonlinear_repayment_probabilities(
                    applicants, loans, baseline["true_risk"], nonlinear
                )
            outcomes = simulate_repayment_outcomes_from_uniforms(loans, truth, repayment_uniforms)
            economics = expected_profit_from_options(loans, truth)
            history = build_at_risk_payment_history(applicants, loans, outcomes)
            train = history[history["cohort"].eq("historical_train")].reset_index(drop=True)
            validation = history[history["cohort"].eq("historical_validation")].reset_index(drop=True)
            eval_ids = applicants.loc[applicants["cohort"].astype(str).eq("evaluation"), "applicant_id"]
            evaluation = {
                "applicants": applicants[applicants["applicant_id"].isin(eval_ids)].reset_index(drop=True),
                "loan_options": loans[loans["applicant_id"].isin(eval_ids)].reset_index(drop=True),
                "truth": truth[truth["applicant_id"].isin(eval_ids)].reset_index(drop=True),
                "outcomes": outcomes[outcomes["applicant_id"].isin(eval_ids)].reset_index(drop=True),
                "economics": economics[economics["applicant_id"].isin(eval_ids)].reset_index(drop=True),
                "mechanism": common["mechanism"][common["mechanism"]["applicant_id"].isin(eval_ids)].reset_index(drop=True),
            }
            worlds[(risk_world, strength)] = evaluation

            traditional = fit_traditional_logit(train, baseline["true_risk"])
            traditional_prediction = predict_applicant_risk(
                traditional,
                evaluation["applicants"],
                evaluation["loan_options"],
                baseline["true_risk"],
            )
            traditional_policy = build_policy_assessments(
                evaluation["loan_options"], traditional_prediction,
                policy_id=_policy_id("traditional", risk_world, strength),
            )
            selection = tune_hist_gradient_boosting(train, validation, ml_config)
            ml_prediction = predict_ml_applicant_risk(
                selection.estimator, evaluation["applicants"], evaluation["loan_options"]
            )
            ml_policy = build_policy_assessments(
                evaluation["loan_options"], ml_prediction,
                policy_id=_policy_id("ml", risk_world, strength),
                probability_column="repayment_probability_ml",
            )
            oracle_policy = _oracle_policy(
                evaluation["loan_options"], evaluation["truth"],
                _policy_id("true_risk_reference", risk_world, strength),
            )
            policies[(risk_world, strength, "traditional")] = traditional_policy
            policies[(risk_world, strength, "ml")] = ml_policy
            policies[(risk_world, strength, "true_risk_reference")] = oracle_policy
            model_metadata.extend(
                [
                    {
                        "risk_world": risk_world,
                        "systemic_strength": strength,
                        "family": "traditional",
                        "training_applicants": int(train["applicant_id"].nunique()),
                        "training_payment_rows": int(len(train)),
                        "validation_payment_rows": int(len(validation)),
                        "features": list(traditional.features),
                        "coefficients": traditional.coefficients,
                    },
                    {
                        "risk_world": risk_world,
                        "systemic_strength": strength,
                        "family": "ml",
                        "training_applicants": int(train["applicant_id"].nunique()),
                        "training_payment_rows": int(len(train)),
                        "validation_payment_rows": int(len(validation)),
                        "features": list(ml_config["features"]),
                        "selected_parameters": {k: _safe(v) for k, v in selection.selected_parameters.items()},
                        "selection_metric": "historical_validation log loss",
                    },
                ]
            )
            full_world = {**common, "truth": truth, "economics": economics}
            neutral_full = worlds.get((risk_world, 0.0))
            if strength == 0.0:
                pathway_rows.extend(_paired_pathway(risk_world, strength, full_world, full_world, switchers[strength]))
            else:
                neutral_common = base_worlds[0.0]
                if risk_world == "additive_logistic_baseline":
                    neutral_truth = true_repayment_probabilities(neutral_common["applicants"], neutral_common["loan_options"], baseline["true_risk"])
                else:
                    neutral_truth = nonlinear_repayment_probabilities(neutral_common["applicants"], neutral_common["loan_options"], baseline["true_risk"], nonlinear)
                neutral_economics = expected_profit_from_options(neutral_common["loan_options"], neutral_truth)
                pathway_rows.extend(_paired_pathway(risk_world, strength, {**neutral_common, "truth": neutral_truth, "economics": neutral_economics}, full_world, switchers[strength]))

            reference = (
                applicants[["applicant_id", "group", "cohort", "age_years", "annual_income", "credit_score", "employment_years", "liquid_assets", "existing_monthly_debt", "property_value", "requested_loan_amount"]]
                .merge(common["mechanism"], on="applicant_id")
                .merge(loans[["applicant_id", "first_period_dti", "requested_ltv"]], on="applicant_id")
                .merge(truth, on="applicant_id")
                .merge(economics, on="applicant_id")
                .merge(outcomes[["applicant_id", "default_period", "completed_all_payments", "realized_profit"]], on="applicant_id")
            )
            reference.insert(0, "risk_world", risk_world)
            mechanism_reference.append(reference)

    decisions: dict[tuple[str, float, str, str], pd.DataFrame] = {}
    solver_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for risk_world in RISK_WORLDS:
        for strength in strengths:
            world = worlds[(risk_world, strength)]
            for family in FAMILIES:
                policy = policies[(risk_world, strength, family)]
                for budget_id, budget in budgets.items():
                    solution = solve_fixed_request_portfolio(
                        world["loan_options"], policy, budget,
                        world_id=f"{risk_world}__systemic_s{strength:.2f}",
                        budget_id=budget_id,
                        policy_id=str(policy["policy_id"].iloc[0]),
                    )
                    key = (risk_world, strength, family, budget_id)
                    decisions[key] = solution.decisions
                    solver_rows.append({"risk_world": risk_world, "systemic_strength": strength, "family": family, **solution.solver_metadata})
                    summary, _ = evaluate_portfolio(
                        solution.decisions, world["loan_options"], world["truth"],
                        world["economics"], world["outcomes"],
                        policy[["applicant_id", "repayment_probability_used"]], budget,
                    )
                    portfolio_rows.append({"risk_world": risk_world, "systemic_strength": strength, "family": family, "budget_id": budget_id, "bank_budget": budget, **summary})
                    audit = portfolio_group_audit(solution.decisions, world["applicants"], world["loan_options"], world["truth"], world["economics"])
                    for row in audit.to_dict("records"):
                        group_rows.append({"risk_world": risk_world, "systemic_strength": strength, "family": family, "budget_id": budget_id, **row})
                    controlled = controlled_funding_audit(solution.decisions, world["applicants"], world["loan_options"], baseline["true_risk"])
                    for row in controlled.to_dict("records"):
                        audit_rows.append({"risk_world": risk_world, "systemic_strength": strength, "family": family, "budget_id": budget_id, **row})

    portfolio = pd.DataFrame(portfolio_rows)
    groups = pd.DataFrame(group_rows)
    group_effects: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    switcher_lending: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for risk_world in RISK_WORLDS:
        group_map = worlds[(risk_world, 0.0)]["applicants"].set_index("applicant_id")["group"].astype(str)
        for family in FAMILIES:
            for budget_id in budgets:
                neutral_decision = decisions[(risk_world, 0.0, family, budget_id)]
                neutral_group = _matching(groups, risk_world=risk_world, systemic_strength=0.0, family=family, budget_id=budget_id).set_index("group")
                neutral_profit = float(_matching(portfolio, risk_world=risk_world, systemic_strength=0.0, family=family, budget_id=budget_id)["true_expected_portfolio_profit"].iloc[0])
                for strength in strengths:
                    current_decision = decisions[(risk_world, strength, family, budget_id)]
                    current_group = _matching(groups, risk_world=risk_world, systemic_strength=strength, family=family, budget_id=budget_id).set_index("group")
                    row: dict[str, Any] = {"risk_world": risk_world, "systemic_strength": strength, "family": family, "budget_id": budget_id}
                    for group in ("A", "B"):
                        prefix = f"group_{group.lower()}"
                        row[f"{prefix}_funding_rate"] = float(current_group.loc[group, "funding_rate"])
                        row[f"{prefix}_funding_rate_change"] = float(current_group.loc[group, "funding_rate"] - neutral_group.loc[group, "funding_rate"])
                        row[f"{prefix}_principal_per_applicant"] = float(current_group.loc[group, "mean_funded_principal_per_applicant"])
                        row[f"{prefix}_principal_per_applicant_change"] = float(current_group.loc[group, "mean_funded_principal_per_applicant"] - neutral_group.loc[group, "mean_funded_principal_per_applicant"])
                        row[f"{prefix}_funded_requested_ratio"] = float(current_group.loc[group, "funded_requested_principal_ratio"])
                    row["funding_gap_b_minus_a"] = row["group_b_funding_rate"] - row["group_a_funding_rate"]
                    neutral_gap = float(neutral_group.loc["B", "funding_rate"] - neutral_group.loc["A", "funding_rate"])
                    row["systemic_effect_on_funding_gap"] = row["funding_gap_b_minus_a"] - neutral_gap
                    row["principal_gap_b_minus_a"] = row["group_b_principal_per_applicant"] - row["group_a_principal_per_applicant"]
                    neutral_principal_gap = float(neutral_group.loc["B", "mean_funded_principal_per_applicant"] - neutral_group.loc["A", "mean_funded_principal_per_applicant"])
                    row["systemic_effect_on_principal_gap"] = row["principal_gap_b_minus_a"] - neutral_principal_gap
                    row["funded_request_gap_b_minus_a"] = row["group_b_funded_requested_ratio"] - row["group_a_funded_requested_ratio"]
                    neutral_ratio_gap = float(neutral_group.loc["B", "funded_requested_principal_ratio"] - neutral_group.loc["A", "funded_requested_principal_ratio"])
                    row["systemic_effect_on_funded_request_gap"] = row["funded_request_gap_b_minus_a"] - neutral_ratio_gap
                    neutral_selected = set(neutral_decision.loc[neutral_decision["approved"], "applicant_id"])
                    current_selected = set(current_decision.loc[current_decision["approved"], "applicant_id"])
                    row["group_a_portfolio_spillovers"] = sum(group_map.loc[item] == "A" for item in neutral_selected.symmetric_difference(current_selected))
                    group_effects.append(row)
                    overlap_rows.append({"risk_world": risk_world, "systemic_strength": strength, "family": family, "budget_id": budget_id, "comparison": "same_policy_s0", **allocation_overlap(current_decision, neutral_decision, worlds[(risk_world, strength)]["loan_options"])})
                    current_profit = float(_matching(portfolio, risk_world=risk_world, systemic_strength=strength, family=family, budget_id=budget_id)["true_expected_portfolio_profit"].iloc[0])
                    if family == "true_risk_reference":
                        oracle_rows.append({"risk_world": risk_world, "systemic_strength": strength, "budget_id": budget_id, "true_expected_portfolio_profit": current_profit, "change_in_maximum_attainable_true_profit": current_profit - neutral_profit, "allocation_overlap_with_s0": overlap_rows[-1]["jaccard_overlap"]})
                    if strength > 0:
                        switcher_ids = set(switchers[strength].loc[switchers[strength]["upstream_displaced"], "applicant_id"])
                        transition = _transition_counts(neutral_decision, current_decision, switcher_ids)
                        a_ids = set(group_map[group_map.eq("A")].index)
                        switcher_lending.append({"risk_world": risk_world, "systemic_strength": strength, "family": family, "budget_id": budget_id, "population": "group_b_opportunity_switchers", "n": sum(transition.values()), **transition})
                        a_transition = _transition_counts(neutral_decision, current_decision, a_ids)
                        switcher_lending.append({"risk_world": risk_world, "systemic_strength": strength, "family": family, "budget_id": budget_id, "population": "group_a_portfolio_spillovers", "n": sum(a_transition.values()), **a_transition})

    effects = pd.DataFrame(group_effects)
    overlaps = pd.DataFrame(overlap_rows)
    oracle_reference = pd.DataFrame(oracle_rows)
    fitted = portfolio[portfolio["family"].isin(["traditional", "ml"])].copy()
    oracle_profit = portfolio[portfolio["family"].eq("true_risk_reference")][["risk_world", "systemic_strength", "budget_id", "true_expected_portfolio_profit"]].rename(columns={"true_expected_portfolio_profit": "oracle_true_expected_portfolio_profit"})
    fitted = fitted.merge(oracle_profit, on=["risk_world", "systemic_strength", "budget_id"], validate="many_to_one")
    fitted["oracle_regret"] = fitted["oracle_true_expected_portfolio_profit"] - fitted["true_expected_portfolio_profit"]
    neutral_fitted = fitted[fitted["systemic_strength"].eq(0)][["risk_world", "family", "budget_id", "true_expected_portfolio_profit", "realized_portfolio_profit", "oracle_regret"]].rename(columns={"true_expected_portfolio_profit": "neutral_true_expected_portfolio_profit", "realized_portfolio_profit": "neutral_realized_portfolio_profit", "oracle_regret": "neutral_oracle_regret"})
    fitted = fitted.merge(neutral_fitted, on=["risk_world", "family", "budget_id"], validate="many_to_one")
    fitted["true_profit_change_from_s0"] = fitted["true_expected_portfolio_profit"] - fitted["neutral_true_expected_portfolio_profit"]
    fitted["realized_profit_change_from_s0"] = fitted["realized_portfolio_profit"] - fitted["neutral_realized_portfolio_profit"]
    fitted["oracle_regret_change_from_s0"] = fitted["oracle_regret"] - fitted["neutral_oracle_regret"]

    opportunity_rows = []
    financial_rows = []
    for strength in strengths:
        applicants = base_worlds[strength]["applicants"]
        mechanism = base_worlds[strength]["mechanism"]
        joined = applicants.merge(mechanism[["applicant_id", "opportunity_probability", "opportunity_access"]], on="applicant_id")
        for group, frame in joined.groupby("group", observed=True):
            opportunity_rows.append({"systemic_strength": strength, "group": str(group), "n": len(frame), "mean_opportunity_probability": float(frame["opportunity_probability"].mean()), "opportunity_access_rate": float(frame["opportunity_access"].mean()), "odds_multiplier_for_group_b": float(np.exp(-strength)) if str(group) == "B" else 1.0})
            financial_rows.append({"systemic_strength": strength, "group": str(group), "n": len(frame), "mean_employment_years": float(frame["employment_years"].mean()), "mean_annual_income": float(frame["annual_income"].mean()), "mean_liquid_assets": float(frame["liquid_assets"].mean()), "mean_property_value": float(frame["property_value"].mean()), "mean_requested_loan_amount": float(frame["requested_loan_amount"].mean())})
    opportunity_summary = pd.DataFrame(opportunity_rows)
    financial_pathway = pd.DataFrame(financial_rows)
    switcher_effects = pd.DataFrame(pathway_rows)
    controlled = pd.DataFrame(audit_rows)
    comparison = effects.merge(fitted[["risk_world", "systemic_strength", "family", "budget_id", "oracle_regret", "true_profit_change_from_s0"]], on=["risk_world", "systemic_strength", "family", "budget_id"], how="left")

    tables = PROJECT_ROOT / "results" / "tables"
    figures = PROJECT_ROOT / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    outputs = {
        "v2_systemic_opportunity_summary.csv": opportunity_summary,
        "v2_systemic_group_effects.csv": effects,
        "v2_systemic_switcher_effects.csv": switcher_effects,
        "v2_systemic_switcher_lending.csv": pd.DataFrame(switcher_lending),
        "v2_systemic_financial_pathway.csv": financial_pathway,
        "v2_systemic_portfolio_profit.csv": fitted,
        "v2_systemic_allocation_overlap.csv": overlaps,
        "v2_systemic_oracle_reference.csv": oracle_reference,
        "v2_systemic_controlled_audit.csv": controlled,
        "v2_systemic_additive_nonlinear_comparison.csv": comparison,
    }
    for name, frame in outputs.items():
        frame.sort_values(list(frame.columns[: min(4, len(frame.columns))]), kind="mergesort").to_csv(tables / name, index=False)

    data_dir = PROJECT_ROOT / systemic["artifacts"]["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    fitted_policies = []
    reference_policies = []
    for (risk_world, strength, family), frame in policies.items():
        item = frame.copy()
        item.insert(0, "family", family)
        item.insert(0, "systemic_strength", strength)
        item.insert(0, "risk_world", risk_world)
        (reference_policies if family == "true_risk_reference" else fitted_policies).append(item)
    pd.concat(fitted_policies, ignore_index=True).to_parquet(PROJECT_ROOT / systemic["artifacts"]["policy_assessments"], index=False)
    pd.concat(reference_policies, ignore_index=True).to_parquet(PROJECT_ROOT / systemic["artifacts"]["evaluator_reference_assessments"], index=False)
    persisted_decisions = []
    for (risk_world, strength, family, budget_id), frame in decisions.items():
        item = frame.drop(columns=["perceived_expected_profit", "perceived_profit_per_dollar"]).copy()
        item.insert(3, "family", family)
        item.insert(3, "systemic_strength", strength)
        item.insert(3, "risk_world", risk_world)
        persisted_decisions.append(item)
    pd.concat(persisted_decisions, ignore_index=True).to_parquet(PROJECT_ROOT / systemic["artifacts"]["decisions"], index=False)
    pd.concat(mechanism_reference, ignore_index=True).to_parquet(PROJECT_ROOT / systemic["artifacts"]["mechanism_reference"], index=False)

    _mechanism_figure(figures / "v2_systemic_mechanism_diagram.png")
    b_access = opportunity_summary[opportunity_summary["group"].eq("B")]
    fig, ax = plt.subplots(figsize=(6.5, 4)); ax.plot(b_access["systemic_strength"], b_access["opportunity_access_rate"], marker="o"); ax.set(xlabel="Systemic strength s", ylabel="Group B access rate", title="Upstream opportunity access"); fig.tight_layout(); fig.savefig(figures / "v2_systemic_group_b_opportunity_access.png", dpi=180); plt.close(fig)
    b_fin = financial_pathway[financial_pathway["group"].eq("B")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6));
    for ax, column, label in zip(axes, ["mean_employment_years", "mean_annual_income", "mean_liquid_assets"], ["Employment years", "Annual income ($)", "Liquid assets ($)"], strict=True): ax.plot(b_fin["systemic_strength"], b_fin[column], marker="o"); ax.set(xlabel="Systemic strength s", ylabel=label)
    fig.suptitle("Group B downstream financial pathway"); fig.tight_layout(); fig.savefig(figures / "v2_systemic_group_b_financial_pathway.png", dpi=180); plt.close(fig)
    plotted = effects[effects["family"].isin(["traditional", "ml", "true_risk_reference"])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, ((risk_world, budget_id), frame) in zip(axes.ravel(), plotted.groupby(["risk_world", "budget_id"], sort=False)):
        for family, color in [("traditional", "#2878B5"), ("ml", "#E07A1F"), ("true_risk_reference", "#3B8C5A")]:
            part = frame[frame["family"].eq(family)].sort_values("systemic_strength"); ax.plot(part["systemic_strength"], part["systemic_effect_on_funding_gap"], marker="o", label=family, color=color)
        ax.axhline(0, color="0.5", lw=.8); ax.set(title=f"{risk_world.replace('_',' ')} · {budget_id}", xlabel="s", ylabel="Change in B−A funding gap")
    axes[0, 0].legend(fontsize=8); fig.tight_layout(); fig.savefig(figures / "v2_systemic_funding_gap_change.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, ((risk_world, budget_id), frame) in zip(axes.ravel(), fitted.groupby(["risk_world", "budget_id"], sort=False)):
        oracle = oracle_reference[(oracle_reference["risk_world"].eq(risk_world)) & (oracle_reference["budget_id"].eq(budget_id))].sort_values("systemic_strength")
        ax.plot(oracle["systemic_strength"], oracle["true_expected_portfolio_profit"], marker="o", color="#3B8C5A", label="oracle true profit")
        twin = ax.twinx()
        for family, color in [("traditional", "#2878B5"), ("ml", "#E07A1F")]:
            part = frame[frame["family"].eq(family)].sort_values("systemic_strength"); twin.plot(part["systemic_strength"], part["oracle_regret"], marker="x", ls="--", color=color, label=f"{family} regret")
        ax.set(title=f"{risk_world.replace('_',' ')} · {budget_id}", xlabel="s", ylabel="Oracle true profit ($)"); twin.set_ylabel("Oracle regret ($)")
    fig.tight_layout(); fig.savefig(figures / "v2_systemic_profit_and_regret.png", dpi=180); plt.close(fig)

    config_path = Path(systemic["metadata"]["config_path"])
    raw_fingerprint = systemic["metadata"]["config_fingerprint"]
    metrics = {
        "mechanism_id": systemic["mechanism_id"],
        "systemic_config_fingerprint": raw_fingerprint,
        "baseline_config_fingerprint": baseline["metadata"]["config_fingerprint"],
        "nonlinear_config_fingerprint": nonlinear["metadata"]["config_fingerprint"],
        "ml_config_fingerprint": ml_config["metadata"]["config_fingerprint"],
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "n_applicants": n,
        "strengths": strengths,
        "odds_multipliers": {str(s): float(np.exp(-s)) for s in strengths},
        "neutral_access_target": systemic["opportunity_model"]["neutral_access_target"],
        "solved_opportunity_intercept": systemic["opportunity_model"]["solved_intercept"],
        "downstream_effects": systemic["downstream_effects"],
        "direct_belief_delta": 0.0,
        "matched_world_validation": validations,
        "controlled_audit_identification": {
            "identified": int(controlled["identified"].sum()),
            "not_identified_due_to_separation": int((~controlled["identified"]).sum()),
        },
        "model_metadata": model_metadata,
        "solver_runs": solver_rows,
        "tables": list(outputs),
        "figures": [
            "v2_systemic_mechanism_diagram.png",
            "v2_systemic_group_b_opportunity_access.png",
            "v2_systemic_group_b_financial_pathway.png",
            "v2_systemic_funding_gap_change.png",
            "v2_systemic_profit_and_regret.png",
        ],
    }
    _write_json(PROJECT_ROOT / systemic["artifacts"]["metrics"], metrics)
    print(json.dumps({k: metrics[k] for k in ["mechanism_id", "systemic_config_fingerprint", "n_applicants", "strengths", "solved_opportunity_intercept"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
