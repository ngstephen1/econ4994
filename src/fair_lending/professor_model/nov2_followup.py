"""Matched November scenarios, train-only risk regression, and frozen-policy audits.

The simulator constructs PN and hidden risk separately from lender fitting.
The new +0.01B experiment deliberately does not alter historical -0.1B APIs.
All monetary assessments retain the original stylized contract/accounting.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from . import bank_regression as aware
from . import group_blind as blind
from .config import BANK_COLUMNS, PRIVATE_COLUMNS, configuration_fingerprint
from .contracts import validate_contracts
from .lending import BANK_INPUT_COLUMNS, choose_cutoff, validate_solution
from .noise_pilot import exact_cutoff_diagnostics
from .profit import TRUE_PROFIT_COLUMNS, expected_cashflows
from .risk import TRUTH_COLUMNS, compute_true_risk, load_true_risk_config

SCENARIOS = ("scenario_1_no_systemic", "scenario_2_systemic")
METHODS = ("ols_control", "ols_penalty", "ml_histgb")
FEATURES = {SCENARIOS[0]: ("Y", "D", "C"), SCENARIOS[1]: ("Y", "D", "C", "B")}
DIRECT_PENALTY = 0.01
EPSILON_SHIFT = 1.0
EXACT_BUDGET = 1826087220.7466326
PREDICTION_COLUMNS = ("applicant_id", "scenario_id", "method_id", "raw_PD", "bounded_PD")
ML_SETTINGS = {"loss": "squared_error", "learning_rate": .05,
               "min_samples_leaf": 1000, "l2_regularization": 10,
               "early_stopping": False, "random_state": 4994,
               "max_leaf_nodes": [3, 7], "max_iter": [25, 75],
               "selection": "validation_PN_MSE",
               "exact_tie_rule": "fewer_leaves_then_fewer_iterations"}


def _frame(frame, columns, name):
    """Enforce an exact allowlist, unique IDs, finite values, and stable order."""
    if tuple(frame.columns) != tuple(columns) or frame.empty:
        raise ValueError(f"{name}: exact columns {tuple(columns)} required")
    aware.validate_ids(frame, name)
    if frame.isna().any().any():
        raise ValueError(f"{name}: missing values")
    numeric = frame.select_dtypes(include=np.number)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{name}: nonfinite values")
    return frame.sort_values("applicant_id").reset_index(drop=True).copy()


def _align(reference, frame, columns, name):
    result = _frame(frame, columns, name)
    if result.applicant_id.tolist() != reference.applicant_id.tolist():
        raise ValueError(f"{name}: applicant IDs do not match")
    return result


def _bank(frame):
    result = _frame(frame, BANK_COLUMNS, "bank observables")
    if not result.Y.gt(0).all() or not result.B.isin([0, 1]).all():
        raise ValueError("positive income and binary B required")
    return result


def construct_world(bank, private, targets, eta, truth, scenario_id):
    """Simulator only: shift B=1 epsilon and its C signal with common residuals.

    Scenario 1 is copied bit-for-bit. Scenario 2 touches B=1 rows only, so B=0
    floating-point representations are preserved even when CSV inputs differ
    slightly from a fresh evaluation of their structural expressions.
    """
    if scenario_id not in SCENARIOS:
        raise ValueError("unknown November scenario")
    bank = _bank(bank)
    private = _align(bank, private, PRIVATE_COLUMNS, "private population")
    targets = _align(bank, targets, ("applicant_id", "PN"), "observed targets")
    eta = _align(bank, eta, ("applicant_id", "eta"), "private target noise")
    truth = _align(bank, truth, TRUTH_COLUMNS, "hidden truth")
    if not truth.loc[:, TRUTH_COLUMNS[1:]].apply(lambda s: s.between(0, 1)).all().all():
        raise ValueError("truth components must be probabilities")
    np.testing.assert_allclose(np.log(bank.Y), private.log_Y, rtol=0, atol=2e-14)
    np.testing.assert_allclose(bank.C, .4 * private.epsilon + private.omega, rtol=0, atol=1e-15)
    np.testing.assert_allclose(targets.PN, .8 * truth.true_PD + eta.eta, rtol=0, atol=1e-15)
    risk_input = private[["applicant_id", "log_Y", "epsilon"]].assign(D=bank.D)
    reconstructed = compute_true_risk(risk_input, load_true_risk_config())
    np.testing.assert_allclose(truth.iloc[:, 1:], reconstructed.iloc[:, 1:], rtol=0, atol=1e-15)
    b1 = bank.B.eq(1).to_numpy()
    new_bank, new_private, new_targets, new_truth = [f.copy() for f in (bank, private, targets, truth)]
    if scenario_id == SCENARIOS[1] and b1.any():
        new_private.loc[b1, "epsilon"] = private.loc[b1, "epsilon"] + EPSILON_SHIFT
        new_bank.loc[b1, "C"] = .4 * new_private.loc[b1, "epsilon"] + private.loc[b1, "omega"]
        changed = new_private.loc[b1, ["applicant_id", "log_Y", "epsilon"]].assign(D=bank.loc[b1, "D"])
        recalculated = compute_true_risk(changed, load_true_risk_config())
        new_truth.loc[b1, list(TRUTH_COLUMNS[1:])] = recalculated.loc[:, TRUTH_COLUMNS[1:]].to_numpy()
        new_targets.loc[b1, "PN"] = .8 * new_truth.loc[b1, "true_PD"] + eta.loc[b1, "eta"]
        if not (new_truth.loc[b1, "true_PD"].to_numpy() >= truth.loc[b1, "true_PD"].to_numpy()).all():
            raise ValueError("positive epsilon treatment cannot lower fixed-feature true risk")
    for before, after in ((bank, new_bank), (private, new_private), (targets, new_targets), (truth, new_truth)):
        pd.testing.assert_frame_equal(before.loc[~b1], after.loc[~b1], check_exact=True)
    pd.testing.assert_frame_equal(bank[["applicant_id", "Y", "D", "B"]], new_bank[["applicant_id", "Y", "D", "B"]], check_exact=True)
    return {"bank": new_bank, "private": new_private, "targets": new_targets,
            "truth": new_truth, "checks": {"baseline_structural_equations_verified": True,
            "B0_exactly_invariant": True, "IDs_income_D_B_omega_unchanged": True,
            "saved_eta_reused": True, "epsilon_shift": 0.0 if scenario_id == SCENARIOS[0] else EPSILON_SHIFT}}


def make_inner_splits(saved_splits, *, fit_rows=64000, validation_rows=16000,
                      seed=4994, spawn_key=(1102, 0)):
    """Partition only saved training IDs; evaluation applicants cannot enter."""
    split = _frame(saved_splits, ("applicant_id", "split"), "saved split")
    if not split.split.isin(["train", "evaluation"]).all() or set(split.split) != {"train", "evaluation"}:
        raise ValueError("original train/evaluation cohorts required")
    train = split.loc[split.split.eq("train"), ["applicant_id"]].reset_index(drop=True)
    if any(type(v) is not int or v < 1 for v in (fit_rows, validation_rows)) or fit_rows + validation_rows != len(train):
        raise ValueError("inner counts must cover training applicants exactly")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=tuple(spawn_key))))
    permutation = rng.permutation(len(train))
    labels = np.full(len(train), "validation", dtype=object)
    labels[permutation[:fit_rows]] = "fit"
    return train.assign(inner_split=labels)


def _ml_configuration(settings):
    settings = ML_SETTINGS if settings is None else settings
    if settings != ML_SETTINGS:
        raise ValueError("November ML settings and four-point grid are frozen")
    return {k: v for k, v in settings.items() if k not in ("max_leaf_nodes", "max_iter", "selection", "exact_tie_rule")}


def fit_scenario_models(bank, targets, splits, inner_splits, scenario_id, *,
                        frozen_blind_model=None, frozen_blind_predictions=None, ml_settings=None):
    """Fit on bank information only, select using training-only PN validation.

    Raw evaluation PN is present only in the input target table: it is not read
    during fitting/selection. Evaluation diagnostics belong to the evaluator.
    """
    if scenario_id not in FEATURES:
        raise ValueError("unknown scenario")
    bank = _bank(bank)
    targets = _align(bank, targets, ("applicant_id", "PN"), "bank targets")
    splits = _align(bank, splits, ("applicant_id", "split"), "saved split")
    if not splits.split.isin(["train", "evaluation"]).all() or set(splits.split) != {"train", "evaluation"}:
        raise ValueError("original cohorts required")
    observed = bank.merge(targets, on="applicant_id", validate="one_to_one").merge(splits, on="applicant_id", validate="one_to_one")
    train = observed.loc[observed.split.eq("train")].reset_index(drop=True)
    evaluation = observed.loc[observed.split.eq("evaluation")].reset_index(drop=True)
    inner = _align(train, inner_splits, ("applicant_id", "inner_split"), "inner split")
    if not inner.inner_split.isin(["fit", "validation"]).all() or set(inner.inner_split) != {"fit", "validation"}:
        raise ValueError("both inner fitting and validation applicants required")
    features = FEATURES[scenario_id]
    if scenario_id == SCENARIOS[0]:
        if frozen_blind_model is None or frozen_blind_predictions is None:
            raise ValueError("Scenario 1 requires the frozen blind model and predictions")
        model = frozen_blind_model.copy()
        if model["training_count"] != len(train) or model["training_ID_fingerprint"] != aware.ids_fingerprint(train.applicant_id):
            raise ValueError("frozen blind model training identity mismatch")
        fingerprint = configuration_fingerprint({"IDs": train.applicant_id.tolist(), "PN": train.PN.tolist()})
        if model["training_target_fingerprint"] != fingerprint:
            raise ValueError("frozen blind model target identity mismatch")
        frozen = _align(bank, frozen_blind_predictions, blind.PREDICTION_COLUMNS, "frozen blind predictions")
        if not frozen.model_id.eq(blind.MODEL_ID).all() or not np.array_equal(frozen.split, splits.split):
            raise ValueError("frozen blind model/split mismatch")
        if not np.array_equal(frozen[blind.BOUNDED], np.clip(frozen[blind.RAW], 0, 1)):
            raise ValueError("frozen clipping mismatch")
        if not pd.api.types.is_bool_dtype(frozen.prediction_out_of_bounds) or not np.array_equal(
            frozen.prediction_out_of_bounds, (frozen[blind.RAW] < 0) | (frozen[blind.RAW] > 1)
        ):
            raise ValueError("frozen out-of-bounds flags mismatch")
        replay = blind.predict_group_blind(bank, model)
        np.testing.assert_allclose(replay, frozen[blind.RAW], rtol=0, atol=1e-15)
        raw_ols = frozen.loc[frozen.split.eq("evaluation"), blind.RAW].to_numpy(copy=True)
    else:
        # Explicit observable-only view; the old OLS helper reads train rows.
        model = aware.fit_bank_ols(observed.loc[:, ["applicant_id", "split", *features, "PN"]])
        raw_ols = aware.predict_bank_ols(evaluation.loc[:, features], model)
    fixed = _ml_configuration(ml_settings)
    fitting = inner.inner_split.eq("fit").to_numpy()
    validation = ~fitting
    x_train, y_train = train.loc[:, features].to_numpy(), train.PN.to_numpy()
    x_eval = evaluation.loc[:, features].to_numpy()
    rows = []
    # Explicit single-thread limits avoid opaque OpenMP oversubscription and
    # make the deterministic single-run benchmark practical on a laptop.
    with threadpool_limits(limits=1):
        for leaves, iterations in product(ML_SETTINGS["max_leaf_nodes"], ML_SETTINGS["max_iter"]):
            candidate = HistGradientBoostingRegressor(**fixed, max_leaf_nodes=leaves, max_iter=iterations)
            candidate.fit(x_train[fitting], y_train[fitting])
            prediction = candidate.predict(x_train[validation])
            rows.append({"scenario_id": scenario_id, "max_leaf_nodes": leaves, "max_iter": iterations,
                         "validation_PN_MSE": float(np.mean((prediction - y_train[validation]) ** 2)),
                         "n_fit": int(fitting.sum()), "n_validation": int(validation.sum())})
        winner = min(rows, key=lambda row: (row["validation_PN_MSE"], row["max_leaf_nodes"], row["max_iter"]))
        ml = HistGradientBoostingRegressor(**fixed, max_leaf_nodes=winner["max_leaf_nodes"], max_iter=winner["max_iter"])
        ml.fit(x_train, y_train)
        raw_ml = ml.predict(x_eval)
    for row in rows:
        row["selected"] = row is winner
    if not np.isfinite(raw_ml).all():
        raise ValueError("nonfinite ML prediction")
    predictions = []
    for method, raw in ((METHODS[0], raw_ols), (METHODS[1], raw_ols + DIRECT_PENALTY * evaluation.B.to_numpy()), (METHODS[2], raw_ml)):
        predictions.append(pd.DataFrame({"applicant_id": evaluation.applicant_id, "scenario_id": scenario_id,
                                        "method_id": method, "raw_PD": raw, "bounded_PD": np.clip(raw, 0, 1)}))
    # Existing prediction helpers require their historical IDs internally;
    # exported models identify this study and disclose their original source.
    model = {**model, "source_api_model_id": model["model_id"],
             "model_id": f"nov2_{scenario_id}_ols_control"}
    metadata = {"scenario_id": scenario_id, "features": list(features), "target": "raw_PN",
                "training_count": len(train), "evaluation_count": len(evaluation),
                "training_ID_fingerprint": aware.ids_fingerprint(train.applicant_id),
                "evaluation_ID_fingerprint": aware.ids_fingerprint(evaluation.applicant_id),
                "fit_ID_fingerprint": aware.ids_fingerprint(train.loc[fitting, "applicant_id"]),
                "validation_ID_fingerprint": aware.ids_fingerprint(train.loc[validation, "applicant_id"]),
                "training_mean_PN": float(y_train.mean()), "ols_frozen_replay": scenario_id == SCENARIOS[0],
                "OLS": model, "ML": {"family": "HistGradientBoostingRegressor", "settings": ml.get_params(),
                                      "selection_metric": "validation_PN_MSE", "selected_validation_MSE": winner["validation_PN_MSE"],
                                      "validation_mean_reference_MSE": float(np.mean((y_train[fitting].mean() - y_train[validation]) ** 2)),
                                      "fit_rows": int(fitting.sum()), "validation_rows": int(validation.sum()),
                                      "final_refit_rows": len(train), "thread_limit": 1}}
    return {"predictions": pd.concat(predictions, ignore_index=True).loc[:, PREDICTION_COLUMNS],
            "ols_model": model, "ml_model": ml, "model_metadata": metadata,
            "validation_grid": pd.DataFrame(rows)}


def optimize_policy(contracts, predictions, budget):
    """Bank-only assessment/search; adapt the legacy ID only inside this call."""
    validate_contracts(contracts)
    contracts = contracts.sort_values("applicant_id").reset_index(drop=True)
    predictions = _align(contracts, predictions, PREDICTION_COLUMNS, "single policy predictions")
    if predictions.scenario_id.nunique() != 1 or predictions.method_id.nunique() != 1 or predictions.scenario_id.iloc[0] not in SCENARIOS or predictions.method_id.iloc[0] not in METHODS:
        raise ValueError("one declared scenario/method required")
    if not np.array_equal(predictions.bounded_PD, np.clip(predictions.raw_PD, 0, 1)):
        raise ValueError("bounded predictions must equal clipped raw scores")
    scenario, method = predictions.scenario_id.iloc[0], predictions.method_id.iloc[0]
    receipts, profits = expected_cashflows(contracts, predictions.bounded_PD.to_numpy())
    adapter = pd.DataFrame({"applicant_id": contracts.applicant_id, "model_id": aware.MODEL_ID,
                            "bank_PD_bounded": predictions.bounded_PD, "requested_principal": contracts.L_i,
                            "expected_profit_bank": profits}).loc[:, BANK_INPUT_COLUMNS]
    decisions, candidates, selection = choose_cutoff(adapter, budget)
    checks = validate_solution(adapter, budget, decisions, candidates, selection)
    checks.update(exact_cutoff_diagnostics(adapter, decisions, candidates, selection, budget))
    decisions = decisions.drop(columns="model_id").rename(columns={"bank_PD_bounded": "bounded_PD"})
    assessments = pd.DataFrame({"applicant_id": contracts.applicant_id, "bounded_PD": predictions.bounded_PD,
                                "expected_receipts_bank": receipts, "expected_profit_bank": profits})
    for frame in (assessments, decisions, candidates):
        frame.insert(0, "method_id", method)
        frame.insert(0, "scenario_id", scenario)
    selection = {k: v for k, v in selection.items() if k != "model_id"}
    selection.update(scenario_id=scenario, method_id=method)
    return {"predictions": predictions, "assessments": assessments, "decisions": decisions,
            "candidates": candidates, "selection": selection, "checks": checks}


def evaluate_policies(policies, contracts, audit, truth_profit, targets=None):
    """Post-freeze evaluator: aggregate three fixed masks without optimization."""
    if set(policies) != set(METHODS):
        raise ValueError("exactly three declared methods required")
    validate_contracts(contracts)
    base = contracts[["applicant_id", "L_i"]].sort_values("applicant_id").reset_index(drop=True)
    audit = _align(base, audit, ("applicant_id", "B"), "group audit")
    if not audit.B.isin([0, 1]).all() or set(audit.B) != {0, 1}:
        raise ValueError("both audit groups required")
    truth = _align(base, truth_profit, TRUE_PROFIT_COLUMNS, "evaluator true profit")
    if not truth.true_PD.between(0, 1).all():
        raise ValueError("true default probabilities outside [0,1]")
    target = None if targets is None else _align(base, targets, ("applicant_id", "PN"), "held-out targets")
    base = base.merge(audit, on="applicant_id", validate="one_to_one").merge(truth, on="applicant_id", validate="one_to_one")
    portfolio, groups, risks, masks = [], [], [], {}
    scenario = budget = None
    for method in METHODS:
        policy = policies[method]
        prediction = _align(base, policy["predictions"], PREDICTION_COLUMNS, "frozen policy predictions")
        decision = policy["decisions"].sort_values("applicant_id").reset_index(drop=True)
        selection = policy["selection"]
        current = selection["scenario_id"]
        if current not in SCENARIOS or (scenario is not None and scenario != current) or selection["method_id"] != method:
            raise ValueError("mixed scenario or method identity")
        scenario = current
        current_budget = selection["available_bank_capital"]
        if not np.isfinite(current_budget) or current_budget < 0 or (budget is not None and current_budget != budget):
            raise ValueError("all policies require the same finite capital budget")
        budget = current_budget
        if not prediction.scenario_id.eq(scenario).all() or not prediction.method_id.eq(method).all() or not decision.scenario_id.eq(scenario).all() or not decision.method_id.eq(method).all():
            raise ValueError("frozen identity mismatch")
        if decision.applicant_id.tolist() != base.applicant_id.tolist() or not pd.api.types.is_bool_dtype(decision.approved):
            raise ValueError("complete boolean frozen decisions required")
        if not np.array_equal(prediction.bounded_PD, np.clip(prediction.raw_PD, 0, 1)):
            raise ValueError("frozen clipped scores mismatch")
        np.testing.assert_array_equal(decision.bounded_PD, prediction.bounded_PD)
        np.testing.assert_array_equal(decision.requested_principal, base.L_i)
        mask = decision.approved.to_numpy()
        tau = selection["optimal_tau"]
        np.testing.assert_array_equal(mask, np.zeros(len(base), dtype=bool) if tau is None else prediction.bounded_PD <= tau)
        np.testing.assert_array_equal(decision.funded_principal, np.where(mask, base.L_i, 0.))
        if int(mask.sum()) != selection["number_approved"] or math.fsum(decision.funded_principal) != selection["total_funded_principal"]:
            raise ValueError("frozen count or principal mismatch")
        if sum((Fraction(float(v)) for v in decision.funded_principal), Fraction()) > Fraction(float(budget)):
            raise ValueError("frozen capital budget exceeded")
        assessment = _align(base, policy["assessments"],
                            ("scenario_id", "method_id", "applicant_id", "bounded_PD", "expected_receipts_bank", "expected_profit_bank"),
                            "perceived assessments")
        if not assessment.scenario_id.eq(scenario).all() or not assessment.method_id.eq(method).all():
            raise ValueError("assessment scenario/method mismatch")
        np.testing.assert_array_equal(assessment.bounded_PD, prediction.bounded_PD)
        if math.fsum(assessment.loc[mask, "expected_profit_bank"]) != selection["bank_expected_portfolio_profit"]:
            raise ValueError("frozen perceived objective mismatch")
        masks[method] = mask
        profit = math.fsum(base.loc[mask, "expected_profit_true"])
        row = {"scenario_id": scenario, "method_id": method, "n_applicants": len(base),
               "number_approved": int(mask.sum()), "approval_rate": float(mask.mean()),
               "funded_principal_USD": selection["total_funded_principal"],
               "available_capital_USD": selection["available_bank_capital"],
               "capital_utilization": selection["budget_utilization"], "optimal_tau": tau,
               "bank_expected_portfolio_profit_USD": selection["bank_expected_portfolio_profit"],
               "true_expected_portfolio_profit_USD": profit,
               "profit_estimation_error_USD": selection["bank_expected_portfolio_profit"] - profit,
               "true_profit_per_eligible_applicant_USD": profit / len(base),
               "reject_all_selected": tau is None,
               "bounded_zero_count": int(prediction.bounded_PD.eq(0).sum()),
               "negative_raw_count": int(prediction.raw_PD.lt(0).sum()),
               "above_one_raw_count": int(prediction.raw_PD.gt(1).sum()),
               "n_tied_score_groups": policy["checks"]["n_tied_score_groups"],
               "zero_block_principal_USD": policy["checks"]["zero_block_principal_USD"],
               "zero_block_feasible": policy["checks"]["zero_block_feasible"]}
        group_rates = {}
        for group in (0, 1):
            members = base.B.eq(group).to_numpy()
            funded = mask & members
            count, approved = int(members.sum()), int(funded.sum())
            total = math.fsum(base.loc[funded, "expected_profit_true"])
            group_rates[group] = approved / count
            groups.append({"scenario_id": scenario, "method_id": method, "B": group,
                           "n_applicants": count, "number_approved": approved,
                           "approval_rate": approved / count,
                           "funded_principal_USD": math.fsum(base.loc[funded, "L_i"]),
                           "true_expected_portfolio_profit_USD": total,
                           "true_profit_per_eligible_applicant_USD": total / count,
                           "true_profit_per_approved_loan_USD": total / approved if approved else None,
                           "mean_true_PD": float(base.loc[members, "true_PD"].mean()),
                           "mean_true_PD_approved": float(base.loc[funded, "true_PD"].mean()) if approved else None,
                           "mean_raw_PD": float(prediction.loc[members, "raw_PD"].mean()),
                           "mean_bounded_PD": float(prediction.loc[members, "bounded_PD"].mean())})
        row["approval_gap_pp"] = 100 * (group_rates[1] - group_rates[0])
        portfolio.append(row)
        for kind, column in (("raw", "raw_PD"), ("bounded", "bounded_PD")):
            for reference, actual in [("true_PD", base.true_PD.to_numpy()), *([] if target is None else [("PN", target.PN.to_numpy())])]:
                risks.append({"scenario_id": scenario, "method_id": method, "prediction": kind,
                              "reference": reference, **aware.error_metrics(actual, prediction[column].to_numpy())})
    comparisons = []
    control, penalty = masks[METHODS[0]], masks[METHODS[1]]
    for scope, members in [("overall", np.ones(len(base), dtype=bool)), ("B=0", base.B.eq(0).to_numpy()), ("B=1", base.B.eq(1).to_numpy())]:
        lost, gained = control & ~penalty & members, penalty & ~control & members
        loss, gain = math.fsum(base.loc[lost, "expected_profit_true"]), math.fsum(base.loc[gained, "expected_profit_true"])
        comparisons.append({"scenario_id": scenario, "scope": scope, "comparison": "ols_penalty_minus_ols_control",
                            "n_applicants": int(members.sum()), "lost_approvals": int(lost.sum()),
                            "gained_approvals": int(gained.sum()), "net_approval_change": int(gained.sum() - lost.sum()),
                            "true_profit_lost_USD": loss, "true_profit_gained_USD": gain,
                            "true_profit_change_USD": gain - loss,
                            "approval_rate_change_pp": 100 * (int(gained.sum()) - int(lost.sum())) / int(members.sum())})
    return {"portfolio": pd.DataFrame(portfolio), "groups": pd.DataFrame(groups),
            "risk_diagnostics": pd.DataFrame(risks), "penalty_comparisons": pd.DataFrame(comparisons)}
