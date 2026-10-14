"""Five fixed-population, noise-only replications using the frozen model APIs.

The simulator creates PN. Bank functions receive observables and PN only.
Evaluator functions are separate and run after a compact bank freeze.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math

import numpy as np
import pandas as pd

from . import bank_regression as aware
from . import group_blind as blind
from .config import BANK_COLUMNS, configuration_fingerprint
from .contracts import validate_contracts
from .lending import BANK_INPUT_COLUMNS, choose_cutoff
from .noisy_signal import raw_noisy_measurement
from .profit import assess_bank_profit

MASTER_SEED = 499408
REPLICATIONS = 5
POLICIES = ("group_aware", "group_blind")
OBSERVED_COLUMNS = (*BANK_COLUMNS, "split")


def frame_fingerprint(frame):
    """Exact value/order/type identity; pandas version is recorded in provenance."""
    digest = hashlib.sha256()
    digest.update(configuration_fingerprint({"columns": list(frame.columns), "dtypes": [str(t) for t in frame.dtypes]}).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def streams():
    """Always spawn exactly five independent children, never reuse canonical [5]."""
    return np.random.SeedSequence(MASTER_SEED).spawn(REPLICATIONS)


def draw_targets(truth, replication):
    """SIMULATOR ONLY: attach independent eta in sorted applicant-ID order."""
    if type(replication) is not int or not 0 <= replication < REPLICATIONS:
        raise ValueError("Step 8A permits only replication indices 0..4")
    if tuple(truth.columns) != ("applicant_id", "true_PD"):
        raise ValueError("simulator target view requires only ID and saved true_PD")
    aware.validate_ids(truth, "fixed truth")
    ordered = truth.sort_values("applicant_id").reset_index(drop=True)
    child = streams()[replication]
    eta = np.random.Generator(np.random.PCG64(child)).normal(0., .1, len(ordered))
    targets = ordered[["applicant_id"]].assign(PN=raw_noisy_measurement(ordered.true_PD.to_numpy(), eta))
    return targets, {
        "master_seed": MASTER_SEED, "spawn_key": list(child.spawn_key), "pool_size": child.pool_size,
        "bit_generator": "PCG64", "applicant_order": "ascending_lexicographic_applicant_id",
        "eta_sha256": hashlib.sha256(eta.astype("<f8").tobytes()).hexdigest(),
        "target_fingerprint": frame_fingerprint(targets), "ordered_ID_fingerprint": aware.ids_fingerprint(ordered.applicant_id),
        "eta_mean": float(eta.mean()), "eta_sample_SD": float(eta.std(ddof=1)),
        "PN_mean": float(targets.PN.mean()), "PN_negative_count": int((targets.PN < 0).sum()),
    }


def exact_cutoff_diagnostics(bank, decisions, candidates, selection, budget):
    """Independent rational audit, not a new lending optimizer or policy.

    Check every risk-group endpoint against the old search, using Fraction
    rather than its aligned-integer implementation. Residual = cost - budget.
    """
    ordered = bank.sort_values(["bank_PD_bounded", "applicant_id"]).reset_index(drop=True)
    risk = ordered.bank_PD_bounded.to_numpy()
    limit = Fraction(float(budget))
    capital = profit = best = Fraction(0)
    best_tau, best_count, candidate_index, feasible_count = None, 0, 1, 1
    max_feasible_residual = -limit  # reject-all is feasible
    for i, row in enumerate(ordered.itertuples(index=False)):
        capital += Fraction(float(row.requested_principal))
        profit += Fraction(float(row.expected_profit_bank))
        if i + 1 < len(ordered) and risk[i] == risk[i + 1]:
            continue
        candidate = candidates.iloc[candidate_index]
        feasible = capital <= limit
        if bool(candidate.budget_feasible) != feasible or candidate.number_approved != i + 1 or candidate.tau != risk[i] or candidate.funded_principal != float(capital) or candidate.bank_expected_portfolio_profit != float(profit):
            raise ValueError("candidate differs from independent exact enumeration")
        if feasible:
            feasible_count += 1
            max_feasible_residual = max(max_feasible_residual, capital - limit)
            if profit > best:
                best, best_tau, best_count = profit, float(risk[i]), i + 1
        candidate_index += 1
    if candidate_index != len(candidates) or not candidates.selected.sum() == 1 or selection["optimal_tau"] != best_tau or selection["number_approved"] != best_count or selection["bank_expected_portfolio_profit"] != float(best) or selection["available_bank_capital"] != budget:
        raise ValueError("selection differs from independent exact optimum")
    first = candidates.iloc[0]
    if first.candidate_type != "no_lending" or not first.budget_feasible or first.number_approved != 0:
        raise ValueError("separate reject-all candidate missing")
    aligned = bank.sort_values("applicant_id").reset_index(drop=True)
    result = decisions.sort_values("applicant_id").reset_index(drop=True)
    np.testing.assert_array_equal(result.applicant_id, aligned.applicant_id)
    mask = np.zeros(len(aligned), dtype=bool) if best_tau is None else aligned.bank_PD_bounded.to_numpy() <= best_tau
    np.testing.assert_array_equal(result.approved, mask)
    np.testing.assert_array_equal(result.funded_principal, np.where(mask, aligned.requested_principal, 0.))
    cost = sum((Fraction(float(v)) for v in result.funded_principal), Fraction(0))
    if cost > limit or result.approved.sum() != best_count:
        raise ValueError("budget violation or partial/skipped funding")
    zero = aligned.bank_PD_bounded.eq(0)
    zero_cost = sum((Fraction(float(v)) for v in aligned.loc[zero, "requested_principal"]), Fraction(0))
    groups = aligned.groupby("bank_PD_bounded").size()
    return {"independent_exact_enumeration_passed": True,
            "bounded_zero_count": int(zero.sum()), "tied_zero_applicants": int(zero.sum()) if zero.sum() > 1 else 0,
            "n_tied_score_groups": int((groups > 1).sum()), "zero_block_principal_USD": float(zero_cost),
            "zero_block_feasible": zero_cost <= limit, "feasible_cutoff_candidates_including_reject_all": feasible_count,
            "selected_cutoff": best_tau, "remaining_capital_USD": budget - float(cost),
            "selected_budget_residual_USD": float(cost - limit),
            "maximum_feasible_candidate_budget_residual_USD": float(max_feasible_residual),
            "maximum_budget_violation_USD": float(max(Fraction(0), cost - limit)),
            "reject_all_selected": best_tau is None}


def fit_bank_policies(observed, targets, contracts, budget):
    """BANK ONLY: both reused OLS fits, predictions, profit, and exact cutoffs."""
    if tuple(observed.columns) != OBSERVED_COLUMNS:
        raise ValueError("exact observable-only applicant/split view required")
    joined = aware.join_bank_inputs(observed.loc[:, BANK_COLUMNS], targets, len(observed))
    joined = aware.attach_splits(joined, observed[["applicant_id", "split"]])
    validate_contracts(contracts)
    if set(contracts.applicant_id) != set(joined.loc[joined.split == "evaluation", "applicant_id"]):
        raise ValueError("contracts must cover exactly the frozen evaluation IDs")
    result = {}
    for policy in POLICIES:
        if policy == "group_aware":
            fit_view = joined[["applicant_id", "split", *aware.FEATURES, "PN"]]
            model = aware.fit_bank_ols(fit_view)
            evaluation = fit_view.loc[fit_view.split == "evaluation"]
            predictions = aware.risk_estimates(evaluation, model)
            profit = assess_bank_profit(contracts, predictions[["applicant_id", "model_id", "bank_PD_bounded"]])
            bank = contracts[["applicant_id", "L_i"]].merge(profit, on="applicant_id", validate="one_to_one").rename(columns={"L_i": "requested_principal"}).loc[:, BANK_INPUT_COLUMNS]
            decisions, candidates, selection = choose_cutoff(bank, budget)
        else:
            fit_view = joined.loc[:, blind.FIT_COLUMNS]
            model = blind.fit_group_blind(fit_view)
            evaluation = fit_view.loc[fit_view.split == "evaluation"]
            predictions = blind.risk_estimates(evaluation, model)
            profit = blind.assess_profit(contracts, predictions[["applicant_id", "model_id", blind.BOUNDED]])
            decisions, candidates, selection = blind.optimize_cutoff(contracts, profit, budget)
            # A private adapter for the shared numerical audit, not a new model.
            bank = contracts[["applicant_id", "L_i"]].merge(profit, on="applicant_id", validate="one_to_one").rename(columns={"L_i": "requested_principal", blind.BOUNDED: "bank_PD_bounded"}).loc[:, BANK_INPUT_COLUMNS]
        diagnostics = exact_cutoff_diagnostics(bank, decisions, candidates, selection, budget)
        result[policy] = {"model": model, "predictions": predictions, "profit": profit, "decisions": decisions,
                          "selection": selection, "diagnostics": diagnostics, "candidate_fingerprint": frame_fingerprint(candidates)}
    return result


def bank_freeze_payload(banks):
    """Compact truth-free freeze; raw applicant/prediction/decision rows not saved."""
    return {policy: {"model": banks[policy]["model"], "selection": banks[policy]["selection"],
                     "diagnostics": banks[policy]["diagnostics"], "candidate_fingerprint": banks[policy]["candidate_fingerprint"],
                     **{f"{name}_fingerprint": frame_fingerprint(banks[policy][name]) for name in ("predictions", "profit", "decisions")}}
            for policy in POLICIES}


def evaluate_replication(banks, observed, targets, contracts, truth, oracle, oracle_selection, budget, frozen_payload):
    """EVALUATOR ONLY: called after IO verifies a persisted bank freeze."""
    if frozen_payload != bank_freeze_payload(banks):
        raise ValueError("bank outputs differ from the frozen allocation")
    evaluation = observed.loc[observed.split == "evaluation"]
    policy_data = {p: (banks[p]["decisions"], banks[p]["selection"], "bank_PD_bounded" if p == "group_aware" else blind.BOUNDED, banks[p]["profit"][["applicant_id", "expected_profit_bank"]]) for p in POLICIES}
    policy_data["oracle"] = (oracle, oracle_selection, "true_PD", None)
    portfolio, groups, overlap, comparison = blind.compare_allocations(contracts, truth, evaluation[["applicant_id", "B"]], policy_data, budget)
    rows = []
    for policy in POLICIES:
        pred = banks[policy]["predictions"]
        raw_col, bounded_col = ("bank_PD_raw", "bank_PD_bounded") if policy == "group_aware" else (blind.RAW, blind.BOUNDED)
        joined = evaluation[["applicant_id"]].merge(targets, on="applicant_id", validate="one_to_one").merge(pred, on="applicant_id", validate="one_to_one").merge(truth[["applicant_id", "true_PD"]], on="applicant_id", validate="one_to_one")
        row = {"policy": policy, **banks[policy]["selection"], **banks[policy]["diagnostics"],
               **portfolio.loc[portfolio.policy == policy].iloc[0].to_dict(),
               "mean_raw_PD": float(joined[raw_col].mean()), "mean_bounded_PD": float(joined[bounded_col].mean()),
               "negative_raw_count": int((joined[raw_col] < 0).sum()), "raw_above_one_count": int((joined[raw_col] > 1).sum())}
        for kind, column in (("raw", raw_col), ("bounded", bounded_col)):
            for name, actual in (("PN", joined.PN), ("true_PD", joined.true_PD)):
                row.update({f"{kind}_{name}_{key}": value for key, value in aware.error_metrics(actual.to_numpy(), joined[column].to_numpy()).items()})
        for b in (0, 1):
            row[f"B{b}_approval_rate"] = float(groups.loc[(groups.policy == policy) & (groups.B == b), "approval_rate"].iloc[0])
        rows.append(row)
    whole = overlap.loc[overlap.scope == "overall"].set_index("allocation_group")
    comparison.update({f"overlap_{name}": int(whole.loc[name, "n_applicants"]) for name in whole.index})
    return pd.DataFrame(rows), groups.loc[groups.policy.isin(POLICIES)].reset_index(drop=True), comparison


def result_tables(records):
    """Tidy deterministic consolidation; require the complete five-run pilot."""
    if len(records) != REPLICATIONS or sorted(r["replication"] for r in records) != list(range(REPLICATIONS)) or any(r["status"] != "success" for r in records):
        raise ValueError("exactly five distinct successful replications required")
    replications, coefficients, groups = [], [], []
    for record in sorted(records, key=lambda r: r["replication"]):
        i = record["replication"]
        row = {"replication": i, "beta_B_hat": record["banks"]["group_aware"]["model"]["coefficients"]["B"], "runtime_seconds": record["runtime_seconds"],
               "gap_change_pp": record["comparison"]["group_blind_minus_aware_approval_gap_pp"],
               "true_profit_change": record["comparison"]["group_blind_minus_aware_true_profit_USD"], **record["comparison"]}
        for policy, prefix in (("group_aware", "aware"), ("group_blind", "blind")):
            metrics = next(m for m in record["policy_metrics"] if m["policy"] == policy)
            # Keep the CSV readable; the immutable JSON retains every detailed
            # policy metric, threshold diagnostic, and raw/bounded error.
            columns = ("optimal_tau", "B0_approval_rate", "B1_approval_rate", "signed_oracle_minus_policy_true_profit",
                       "raw_PN_RMSE", "raw_true_PD_MAE", "raw_true_PD_RMSE", "bounded_true_PD_MAE", "bounded_true_PD_RMSE",
                       "zero_block_principal_USD", "zero_block_feasible", "maximum_budget_violation_USD")
            row.update({f"{prefix}_{key}": metrics[key] for key in columns})
            row.update({f"{prefix}_approval_gap_pp": metrics["approval_gap_pp"], f"{prefix}_approved": metrics["number_approved"], f"{prefix}_true_profit": metrics["true_expected_portfolio_profit"], f"{prefix}_zero_predictions": metrics["bounded_zero_count"]})
            model = record["banks"][policy]["model"]
            coefficients.extend({"replication": i, "policy": policy, "term": term, "coefficient": model["coefficients"][term], "units": model["units"][term]} for term in model["design_order"])
        replications.append(row)
        groups.extend({"replication": i, **g} for g in record["group_approval"])
    frame = pd.DataFrame(replications)
    summary = []
    # Summarize every quantitative estimand; keep flags/IDs out of numeric SDs.
    for column in frame.select_dtypes(include="number").columns:
        if column == "replication":
            continue
        values = frame[column].dropna()
        summary.append({"metric": column, "n_replications": len(frame), "n_defined": len(values),
                        "mean": float(values.mean()) if len(values) else None, "median": float(values.median()) if len(values) else None,
                        "minimum": float(values.min()) if len(values) else None, "maximum": float(values.max()) if len(values) else None,
                        "sample_SD": float(values.std(ddof=1)) if len(values) > 1 else None})
    return frame, pd.DataFrame(coefficients), pd.DataFrame(groups), pd.DataFrame(summary)
