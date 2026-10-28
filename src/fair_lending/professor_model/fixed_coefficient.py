"""Step 10B: subtract one saved additive contribution, never fit a model.

W/F are frozen policy references. Only Z is optimized. Evaluator aggregation
is separate from bank construction and never selects a treatment or cutoff.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import math

import numpy as np
import pandas as pd

from . import bank_regression as aware, group_blind as blind, lending
from .config import configuration_fingerprint as fp
from .contracts import validate_contracts
from .noise_pilot import exact_cutoff_diagnostics, frame_fingerprint
from .profit import assess_bank_profit, TRUE_PROFIT_COLUMNS

SCHEMA = "professor-fixed-coefficient-v1"
W, Z, F = "frozen_aware", "aware_B_zero", "frozen_blind"
POLICIES = (W, Z, F)
MODEL_ID = "professor_aware_B_zero_v1"
OBSERVED = ("applicant_id", "Y", "D", "B", "C", "split")
PAIRS = ((Z, W), (Z, F), (F, W))
SCOPE = "EVALUATOR_ONLY_NOT_FOR_LENDER_DECISIONS"
SUMMARY_UNITS = {"true_profit_difference_USD": "USD", "approval_count_difference": "applications",
                 "approval_rate_difference_pp": "pp", "approval_gap_difference_pp": "pp",
                 "absolute_gap_change_pp": "pp", "funded_principal_difference_USD": "USD",
                 "raw_true_PD_MAE_difference": "raw_probability", "bounded_true_PD_MAE_difference": "raw_probability"}


def validate_observed(observed, contracts):
    validate_contracts(contracts)
    aware.validate_ids(observed, "evaluation observables")
    if tuple(observed.columns) != OBSERVED or not observed.split.eq("evaluation").all() or set(observed.applicant_id) != set(contracts.applicant_id):
        raise ValueError("exact observed-only evaluation view and contract IDs required")
    aware.design_matrix(observed)
    if set(observed.B) != {0, 1}: raise ValueError("both audit groups required")


def validate_model(model, policy):
    mod = aware if policy == W else blind
    design = aware.DESIGN_COLUMNS if policy == W else blind.DESIGN
    expected = {"family": "statsmodels_OLS", "solver": "pinv", "intercept": True, "target": "PN",
                "target_transform": "none", "prediction_rescaling": "none", "regularization": None,
                "bounding": "clip_0_1_separate_column", "model_id": mod.MODEL_ID}
    if policy not in (W, F) or any(model.get(k) != v for k, v in expected.items()) or model.get("feature_order") != list(mod.FEATURES) or model.get("design_order") != list(design) or set(model.get("coefficients", {})) != set(design):
        raise ValueError("only the exact saved additive raw-PN model contract is supported")
    if any(k in model for k in ("offset", "interactions", "transforms", "link")) or not np.isfinite(list(model["coefficients"].values())).all():
        raise ValueError("unsupported offset/interaction/transform or nonfinite coefficient")


def reconstruct_decisions(predictions, contracts, selection, policy):
    """Apply the SAVED cutoff. Never call choose_cutoff for a reference."""
    bound = blind.BOUNDED if policy == F else "bank_PD_bounded"
    columns = blind.DECISION_COLUMNS if policy == F else lending.DECISION_COLUMNS
    data = predictions.sort_values("applicant_id").reset_index(drop=True)
    principal = contracts.set_index("applicant_id").loc[data.applicant_id, "L_i"].to_numpy()
    tau = selection["optimal_tau"]
    mask = np.zeros(len(data), dtype=bool) if tau is None else data[bound].to_numpy() <= tau
    result = data[["applicant_id", "model_id", bound]].copy()
    result["optimal_tau"] = np.nan if tau is None else tau
    result["approved"] = mask
    result["requested_principal"] = principal
    result["funded_principal"] = np.where(mask, principal, 0.)
    result["decision_status"] = np.where(mask, "approved_full_request", "rejected_no_lending_policy" if tau is None else "rejected_above_cutoff")
    if int(mask.sum()) != selection["number_approved"] or math.fsum(result.funded_principal) != selection["total_funded_principal"]:
        raise ValueError("reference count/capital differs from saved selection")
    if sum((Fraction(float(v)) for v in result.funded_principal), Fraction()) > Fraction(float(selection["available_bank_capital"])):
        raise ValueError("reference exceeds original dollar budget")
    return result.loc[:, columns]


def replay_reference(observed, contracts, payload, policy):
    validate_observed(observed, contracts); validate_model(payload["model"], policy)
    mod = aware if policy == W else blind
    pred = mod.risk_estimates(observed, payload["model"])
    bound = "bank_PD_bounded" if policy == W else blind.BOUNDED
    profit = (assess_bank_profit if policy == W else blind.assess_profit)(contracts, pred[["applicant_id", "model_id", bound]])
    decisions = reconstruct_decisions(pred, contracts, payload["selection"], policy)
    for name, value in (("predictions", pred), ("profit", profit), ("decisions", decisions)):
        if frame_fingerprint(value) != payload[f"{name}_fingerprint"]:
            raise ValueError(f"{policy} frozen {name} fingerprint mismatch")
    issued = set(decisions.loc[decisions.approved, "applicant_id"])
    if math.fsum(profit.loc[profit.applicant_id.isin(issued), "expected_profit_bank"]) != payload["selection"]["bank_expected_portfolio_profit"]:
        raise ValueError("reference perceived objective differs")
    return {"model": deepcopy(payload["model"]), "predictions": pred, "profit": profit,
            "decisions": decisions, "selection": deepcopy(payload["selection"]), "diagnostics": deepcopy(payload.get("diagnostics", {}))}


def counterfactual_scores(observed, aware_predictions, model):
    """Authoritative operation: saved/replayed raw W minus its beta_B * B."""
    validate_model(model, W); aware.validate_ids(aware_predictions, "frozen aware scores")
    if tuple(aware_predictions.columns) != aware.ESTIMATE_COLUMNS or set(observed.applicant_id) != set(aware_predictions.applicant_id):
        raise ValueError("exact frozen aware prediction schema/IDs required")
    obs = observed.sort_values("applicant_id").reset_index(drop=True)
    source = aware_predictions.sort_values("applicant_id").reset_index(drop=True)
    replay = aware.risk_estimates(obs, model)
    pd.testing.assert_frame_equal(source, replay, check_exact=True)
    before = fp(model); raw = source.bank_PD_raw.to_numpy() - model["coefficients"]["B"] * obs.B.to_numpy()
    zero = deepcopy(model); zero["coefficients"]["B"] = 0.
    np.testing.assert_allclose(raw, aware.predict_bank_ols(obs, zero), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(raw[obs.B.eq(0)], source.loc[obs.B.eq(0), "bank_PD_raw"])
    if fp(model) != before: raise ValueError("original coefficient artifact mutated")
    pred = source.copy(); pred["model_id"] = MODEL_ID
    pred["bank_PD_raw"] = raw; pred["bank_PD_bounded"] = np.clip(raw, 0, 1)
    pred["prediction_out_of_bounds"] = (raw < 0) | (raw > 1)
    artifact = {"family": "fixed_coefficient_intervention_not_a_fit", "policy_id": MODEL_ID,
                "source_model_fingerprint": before, "source_coefficients": deepcopy(model["coefficients"]),
                "coefficients_used": zero["coefficients"], "operation": "raw_W_minus_saved_beta_B_times_B",
                "beta_B_removed": model["coefficients"]["B"], "bounding": "clip_0_1_separate_column",
                "positive_floor": None, "new_fits": 0, "raw_replay_atol": 1e-15}
    return pred, artifact


def optimize_counterfactual(contracts, predictions):
    """A private legacy-ID adapter; public outputs retain the intervention ID."""
    if tuple(predictions.columns) != aware.ESTIMATE_COLUMNS or not predictions.model_id.eq(MODEL_ID).all():
        raise ValueError("exact counterfactual prediction schema/ID required")
    bounded = predictions[["applicant_id", "model_id", "bank_PD_bounded"]].copy()
    bounded["model_id"] = aware.MODEL_ID
    profit = assess_bank_profit(contracts, bounded)
    bank = contracts[["applicant_id", "L_i"]].merge(profit, on="applicant_id", validate="one_to_one").rename(columns={"L_i": "requested_principal"}).loc[:, lending.BANK_INPUT_COLUMNS]
    return bank, profit


def choose_counterfactual(contracts, predictions, budget):
    bank, profit = optimize_counterfactual(contracts, predictions)
    decisions, candidates, selection = lending.choose_cutoff(bank, budget)
    checks = exact_cutoff_diagnostics(bank, decisions, candidates, selection, budget)
    for frame in (profit, decisions): frame["model_id"] = MODEL_ID
    selection["model_id"] = MODEL_ID
    return {"predictions": predictions, "profit": profit, "decisions": decisions,
            "candidates": candidates, "selection": selection, "diagnostics": checks}


def normalized_predictions(bank, policy):
    pred = bank["predictions"].sort_values("applicant_id").reset_index(drop=True)
    if policy == F: pred = pred.rename(columns={blind.RAW: "bank_PD_raw", blind.BOUNDED: "bank_PD_bounded"})
    return pred


def evaluate_case(banks, observed, contracts, truth, case_id, targets=None):
    """Pure evaluator: all policy masks must already be frozen by the IO gate."""
    validate_observed(observed, contracts)
    if set(banks) != set(POLICIES) or tuple(truth.columns) != TRUE_PROFIT_COLUMNS:
        raise ValueError("all three policies and exact saved truth required")
    aware.validate_ids(truth, "evaluator truth")
    if set(truth.applicant_id) != set(observed.applicant_id) or not np.isfinite(truth.iloc[:, 1:]).all().all() or not truth.true_PD.between(0, 1).all():
        raise ValueError("invalid evaluator truth")
    data = observed[["applicant_id", "B"]].merge(contracts[["applicant_id", "L_i"]], on="applicant_id", validate="one_to_one").merge(truth, on="applicant_id", validate="one_to_one").sort_values("applicant_id").reset_index(drop=True)
    if targets is not None:
        aware.validate_ids(targets, "saved canonical PN")
        if tuple(targets.columns) != ("applicant_id", "PN") or set(targets.applicant_id) != set(data.applicant_id): raise ValueError("canonical PN IDs/schema differ")
        data = data.merge(targets, on="applicant_id", validate="one_to_one")
    rows, groups, masks = [], [], {}
    for policy in POLICIES:
        b = banks[policy]; pred = normalized_predictions(b, policy)
        decisions = b["decisions"].sort_values("applicant_id").reset_index(drop=True)
        np.testing.assert_array_equal(pred.applicant_id, data.applicant_id); np.testing.assert_array_equal(decisions.applicant_id, data.applicant_id)
        mask = decisions.approved.to_numpy(); tau = b["selection"]["optimal_tau"]
        np.testing.assert_array_equal(mask, np.zeros(len(data), dtype=bool) if tau is None else pred.bank_PD_bounded <= tau)
        np.testing.assert_array_equal(decisions.funded_principal, np.where(mask, data.L_i, 0.))
        masks[policy] = mask; issued = data.loc[mask]
        score_counts = pred.bank_PD_bounded.value_counts(); zero = pred.bank_PD_bounded.eq(0).to_numpy()
        zero_cost = math.fsum(data.loc[zero, "L_i"])
        assessed = b["profit"].set_index("applicant_id").loc[data.applicant_id, "expected_profit_bank"].to_numpy()
        row = {"information_scope": SCOPE, "case_id": case_id, "policy": policy, "policy_id": b["selection"]["model_id"],
               "n_applicants": len(data), "number_approved": int(mask.sum()), "approval_rate": float(mask.mean()),
               "optimal_tau": tau, "funded_principal_USD": math.fsum(decisions.funded_principal),
               "available_capital_USD": b["selection"]["available_bank_capital"],
               "capital_utilization": b["selection"]["budget_utilization"], "unused_capital_USD": b["selection"]["unused_capital"],
               "perceived_profit_USD": b["selection"]["bank_expected_portfolio_profit"],
               "true_profit_USD": math.fsum(issued.expected_profit_true),
               "mean_true_PD_approved": float(issued.true_PD.mean()) if len(issued) else None,
               "mean_true_profit_per_approved_USD": math.fsum(issued.expected_profit_true) / len(issued) if len(issued) else None,
               "unique_score_count": len(score_counts), "tied_block_count": int((score_counts > 1).sum()),
               "tied_block_sizes_JSON": json.dumps(sorted(int(n) for n in score_counts[score_counts > 1])),
               "zero_score_count": int(zero.sum()), "zero_block_principal_USD": zero_cost,
               "zero_block_feasible": sum((Fraction(float(v)) for v in data.loc[zero, "L_i"]), Fraction()) <= Fraction(float(b["selection"]["available_bank_capital"])),
               "zero_block_approved": int((zero & mask).sum()), "selected_boundary_size": b["selection"]["selected_cutoff_group_size"],
               "reject_all_selected": tau is None, "negative_raw_count": int(pred.bank_PD_raw.lt(0).sum()),
               "above_one_raw_count": int(pred.bank_PD_raw.gt(1).sum()),
               "mean_raw_PD": float(pred.bank_PD_raw.mean()), "mean_bounded_PD": float(pred.bank_PD_bounded.mean()),
               "PN_metrics_status": "computed_from_saved_canonical_targets" if targets is not None else "unavailable_no_noise_replay",
               "profit_estimation_mean_error_USD": float(np.mean(assessed - data.expected_profit_true)),
               "profit_estimation_MAE_USD": float(np.mean(np.abs(assessed - data.expected_profit_true)))}
        rates = {}
        for group in (0, 1):
            member = data.B.eq(group).to_numpy(); funded = member & mask; rates[group] = float(mask[member].mean())
            g = {"information_scope": SCOPE, "case_id": case_id, "policy": policy, "B": group,
                 "n_applicants": int(member.sum()), "number_approved": int(funded.sum()), "approval_rate": rates[group],
                 "funded_principal_USD": math.fsum(data.loc[funded, "L_i"]),
                 "funded_principal_per_applicant_USD": math.fsum(data.loc[funded, "L_i"]) / int(member.sum()),
                 "true_profit_USD": math.fsum(data.loc[funded, "expected_profit_true"]),
                 "mean_estimated_PD_approved": float(pred.loc[funded, "bank_PD_bounded"].mean()) if funded.any() else None,
                 "mean_true_PD_approved": float(data.loc[funded, "true_PD"].mean()) if funded.any() else None}
            for kind, column in (("raw", "bank_PD_raw"), ("bounded", "bank_PD_bounded")):
                err = aware.error_metrics(data.loc[member, "true_PD"].to_numpy(), pred.loc[member, column].to_numpy())
                g.update({f"{kind}_true_PD_{k}": err[k] for k in ("MAE", "RMSE", "mean_signed_error")})
            groups.append(g)
        row["approval_gap_pp"] = 100 * (rates[1] - rates[0])
        for kind, column in (("raw", "bank_PD_raw"), ("bounded", "bank_PD_bounded")):
            for reference in (["true_PD", "PN"] if targets is not None else ["true_PD"]):
                err = aware.error_metrics(data[reference].to_numpy(), pred[column].to_numpy())
                row.update({f"{kind}_{reference}_{k}": value for k, value in err.items()})
        rows.append(row)
    portfolio = pd.DataFrame(rows); by_policy = portfolio.set_index("policy"); pairs, overlap = [], []
    for first, second in PAIRS:
        x, y = masks[first], masks[second]; union = int((x | y).sum())
        delta = {"information_scope": SCOPE, "case_id": case_id, "comparison": f"{first}_minus_{second}",
                 "policy_first": first, "policy_second": second,
                 "true_profit_difference_USD": by_policy.loc[first, "true_profit_USD"] - by_policy.loc[second, "true_profit_USD"],
                 "approval_count_difference": int(x.sum() - y.sum()), "approval_rate_difference_pp": 100 * (x.mean() - y.mean()),
                 "approval_gap_difference_pp": by_policy.loc[first, "approval_gap_pp"] - by_policy.loc[second, "approval_gap_pp"],
                 "absolute_gap_change_pp": abs(by_policy.loc[first, "approval_gap_pp"]) - abs(by_policy.loc[second, "approval_gap_pp"]),
                 "funded_principal_difference_USD": by_policy.loc[first, "funded_principal_USD"] - by_policy.loc[second, "funded_principal_USD"],
                 "jaccard_overlap": int((x & y).sum()) / union if union else None}
        for kind in ("raw", "bounded"): delta[f"{kind}_true_PD_MAE_difference"] = by_policy.loc[first, f"{kind}_true_PD_MAE"] - by_policy.loc[second, f"{kind}_true_PD_MAE"]
        pairs.append(delta)
        for label, mask, in_first, in_second in (("approved_by_both", x & y, True, True), ("first_only", x & ~y, True, False), ("second_only", ~x & y, False, True), ("rejected_by_both", ~x & ~y, False, False)):
            value = math.fsum(data.loc[mask, "expected_profit_true"])
            overlap.append({"information_scope": SCOPE, "case_id": case_id, "policy_first": first, "policy_second": second,
                            "allocation_category": label, "n_applicants": int(mask.sum()), "jaccard_overlap": delta["jaccard_overlap"],
                            "potential_true_profit_USD": value, "first_policy_profit_contribution_USD": value if in_first else 0.,
                            "second_policy_profit_contribution_USD": value if in_second else 0.,
                            "profit_scope": "hypothetical_unfunded_by_both" if not in_first and not in_second else "expected_only_for_funding_policies"})
    return {"policies": portfolio, "groups": pd.DataFrame(groups), "paired": pd.DataFrame(pairs), "overlap": pd.DataFrame(overlap)}


def summarize_noise(paired):
    """Empirical noise variation, never pool the separate canonical case."""
    if paired.empty or paired.case_id.eq("canonical").any() or paired.duplicated(["case_id", "comparison"]).any():
        raise ValueError("unique noise-only paired cells required")
    rows = []
    for comparison, frame in paired.groupby("comparison", sort=True):
        for metric, unit in SUMMARY_UNITS.items():
            values = frame[metric].dropna().to_numpy(dtype=float); n = len(values)
            if not n or not np.isfinite(values).all(): raise ValueError("finite noise comparison values required")
            sd = float(values.std(ddof=1)) if n > 1 else None
            rows.append({"comparison": comparison, "metric": metric, "unit": unit, "n_defined": n,
                         "mean": float(values.mean()), "median": float(np.median(values)), "sample_SD": sd,
                         "MCSE_mean": sd / math.sqrt(n) if sd is not None else None,
                         "p2_5": float(np.quantile(values, .025)), "p97_5": float(np.quantile(values, .975)),
                         "minimum": float(values.min()), "maximum": float(values.max()),
                         "positive_count": int((values > 0).sum()), "negative_count": int((values < 0).sum()), "zero_count": int((values == 0).sum())})
    return pd.DataFrame(rows)
