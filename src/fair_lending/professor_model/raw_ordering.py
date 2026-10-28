"""Raw-score prefixes with unchanged bounded-PD economics, isolated from V1/V2.

An ordering score may be negative. It is never passed to a survival/profit API.
Only the bank-only input allowlist can reach the new exact prefix optimizer.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pandas as pd

from .bank_regression import validate_ids
from .fixed_coefficient import W, Z, F, POLICIES, PAIRS, TRUE_PROFIT_COLUMNS
from .lending import _integer_units

SCHEMA = "professor-raw-ordering-v1"
CASES = ("canonical", "noise_008", "noise_064", "noise_089", "noise_092", "noise_154")
BOUNDED, RAW = "bounded_frozen", "raw_order"
INPUT_COLUMNS = ("applicant_id", "source_policy", "ordering_score", "accounting_PD", "requested_principal", "expected_profit_perceived")
DECISION_COLUMNS = (*INPUT_COLUMNS[:4], "cutoff_score", "approved", "requested_principal", "funded_principal", "decision_status")


def validate_inputs(bank):
    validate_ids(bank, "raw-ordering bank inputs")
    if tuple(bank.columns) != INPUT_COLUMNS or len(bank) == 0 or bank.isna().any().any():
        raise ValueError("exact truth-free bank input allowlist required")
    if bank.source_policy.nunique() != 1 or bank.source_policy.iloc[0] not in POLICIES:
        raise ValueError("one declared source policy required")
    if not np.isfinite(bank.iloc[:, 2:].to_numpy(dtype=float)).all() or not bank.requested_principal.gt(0).all():
        raise ValueError("finite scores/profits and positive full requests required")
    if not bank.accounting_PD.between(0, 1).all() or not np.array_equal(bank.accounting_PD, np.clip(bank.ordering_score, 0, 1)):
        raise ValueError("accounting PD must equal unchanged clip(raw,0,1)")


def choose_raw_cutoff(bank, budget):
    """Optimize raw prefixes using exact binary-float integer sums; no skip/tie split."""
    validate_inputs(bank)
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (float, int)) or not math.isfinite(budget) or budget < 0:
        raise ValueError("finite nonnegative original budget required")
    ordered = bank.sort_values(["ordering_score", "applicant_id"]).reset_index(drop=True)
    capital_units, ce = _integer_units([*ordered.requested_principal, float(budget)])
    limit = capital_units.pop()
    profit_units, pe = _integer_units(ordered.expected_profit_perceived.tolist())
    rows = [{"candidate_id": 0, "candidate_type": "reject_all", "cutoff_score": np.nan,
             "number_approved": 0, "funded_principal": 0., "budget_feasible": True,
             "perceived_profit_USD": 0., "raw_tie_size": 0}]
    capital = profit = best = best_id = start = 0
    scores = ordered.ordering_score.to_numpy()
    for i, (cost, value) in enumerate(zip(capital_units, profit_units)):
        capital += cost; profit += value
        if i + 1 < len(ordered) and scores[i] == scores[i + 1]:
            continue
        feasible = capital <= limit
        row = {"candidate_id": len(rows), "candidate_type": "raw_cutoff", "cutoff_score": float(scores[i]),
               "number_approved": i + 1, "funded_principal": capital / (1 << ce),
               "budget_feasible": feasible, "perceived_profit_USD": profit / (1 << pe), "raw_tie_size": i - start + 1}
        rows.append(row)
        if feasible and profit > best:  # exact objective tie: retain first/lowest endpoint
            best, best_id = profit, row["candidate_id"]
        start = i + 1
    candidates = pd.DataFrame(rows)
    candidates["selected"] = candidates.candidate_id.eq(best_id)
    chosen = rows[best_id]; cutoff = None if best_id == 0 else chosen["cutoff_score"]
    aligned = bank.sort_values("applicant_id").reset_index(drop=True)
    mask = np.zeros(len(aligned), dtype=bool) if cutoff is None else aligned.ordering_score.to_numpy() <= cutoff
    decisions = aligned.loc[:, INPUT_COLUMNS[:4]].copy()
    decisions["cutoff_score"] = np.nan if cutoff is None else cutoff
    decisions["approved"] = mask
    decisions["requested_principal"] = aligned.requested_principal.to_numpy()
    decisions["funded_principal"] = np.where(mask, aligned.requested_principal, 0.)
    decisions["decision_status"] = np.where(mask, "approved_full_request", "rejected_no_lending_policy" if cutoff is None else "rejected_above_raw_cutoff")
    selected_cost = math.fsum(decisions.funded_principal)
    selected_profit = math.fsum(aligned.loc[mask, "expected_profit_perceived"])
    if selected_cost != chosen["funded_principal"] or selected_profit != chosen["perceived_profit_USD"] or selected_cost > budget:
        raise ValueError("selected prefix sum or original budget mismatch")
    selection = {"source_policy": str(aligned.source_policy.iloc[0]), "ordering_regime": RAW,
                 "cutoff_score": cutoff, "selected_candidate_id": best_id, "number_approved": int(mask.sum()),
                 "n_applicants": len(bank), "available_capital_USD": float(budget),
                 "funded_principal_USD": selected_cost, "perceived_profit_USD": selected_profit,
                 "unused_capital_USD": budget - selected_cost, "candidate_count": len(candidates),
                 "selected_raw_tie_size": chosen["raw_tie_size"], "reject_all_selected": cutoff is None}
    return {"decisions": decisions.loc[:, DECISION_COLUMNS], "candidates": candidates, "selection": selection}


def audit_raw_cutoff(bank, result, budget):
    """Independently enumerate every raw endpoint using Fraction, not integer units."""
    validate_inputs(bank)
    ordered = bank.sort_values(["ordering_score", "applicant_id"]).reset_index(drop=True)
    candidates = result["candidates"]
    first = candidates.iloc[0]
    if first.candidate_type != "reject_all" or first.number_approved != 0 or first.funded_principal != 0 or first.perceived_profit_USD != 0 or not first.budget_feasible or not pd.isna(first.cutoff_score):
        raise ValueError("separate reject-all missing")
    limit = Fraction(float(budget)); capital = profit = best = Fraction(0)
    best_id, best_tau, count, endpoint, start = 0, None, 0, 1, 0
    scores = ordered.ordering_score.to_numpy()
    candidate_rows = iter(candidates.iloc[1:].itertuples(index=False))
    for i, row in enumerate(ordered.itertuples(index=False)):
        capital += Fraction(float(row.requested_principal)); profit += Fraction(float(row.expected_profit_perceived))
        if i + 1 < len(ordered) and scores[i] == scores[i + 1]:
            continue
        candidate = next(candidate_rows, None); feasible = capital <= limit
        if candidate is None or candidate.candidate_id != endpoint or candidate.candidate_type != "raw_cutoff" or candidate.cutoff_score != scores[i] or candidate.number_approved != i + 1 or candidate.funded_principal != float(capital) or candidate.perceived_profit_USD != float(profit) or bool(candidate.budget_feasible) != feasible or candidate.raw_tie_size != i - start + 1:
            raise ValueError("raw endpoint differs from independent rational enumeration")
        if feasible and profit > best:
            best, best_id, best_tau, count = profit, endpoint, float(scores[i]), i + 1
        endpoint += 1; start = i + 1
    selection = result["selection"]
    if next(candidate_rows, None) is not None or len(candidates) != endpoint or candidates.selected.sum() != 1 or int(candidates.loc[candidates.selected, "candidate_id"].iloc[0]) != best_id:
        raise ValueError("raw candidate coverage/selection differs")
    if any(selection[k] != v for k, v in {"cutoff_score": best_tau, "selected_candidate_id": best_id,
        "number_approved": count, "perceived_profit_USD": float(best), "available_capital_USD": budget}.items()):
        raise ValueError("raw selection is not the exact optimal prefix")
    aligned = bank.sort_values("applicant_id").reset_index(drop=True)
    decisions = result["decisions"].sort_values("applicant_id").reset_index(drop=True)
    if tuple(decisions.columns) != DECISION_COLUMNS:
        raise ValueError("unexpected decision schema")
    np.testing.assert_array_equal(decisions.applicant_id, aligned.applicant_id)
    np.testing.assert_array_equal(decisions.ordering_score, aligned.ordering_score)
    np.testing.assert_array_equal(decisions.source_policy, aligned.source_policy)
    np.testing.assert_array_equal(decisions.requested_principal, aligned.requested_principal)
    mask = np.zeros(len(bank), dtype=bool) if best_tau is None else aligned.ordering_score.to_numpy() <= best_tau
    np.testing.assert_array_equal(decisions.approved, mask)
    np.testing.assert_array_equal(decisions.accounting_PD, aligned.accounting_PD)
    np.testing.assert_array_equal(decisions.funded_principal, np.where(mask, aligned.requested_principal, 0.))
    cost = sum((Fraction(float(v)) for v in decisions.funded_principal), Fraction())
    if cost > limit or selection["funded_principal_USD"] != float(cost):
        raise ValueError("original budget violated")
    return {"every_raw_endpoint_fraction_checked": True, "raw_candidate_count": endpoint,
            "original_dollar_budget_respected": True, "exact_raw_ties_indivisible": True,
            "perceived_profit_USD": float(best), "budget_residual_USD": float(cost - limit)}


def bounded_prefix_mapping(bank, result, baseline_decisions, baseline_selection):
    """Map all bounded endpoints into raw candidates; do not reoptimize a baseline."""
    validate_inputs(bank)
    ordered = bank.sort_values(["ordering_score", "applicant_id"]).reset_index(drop=True)
    bounded = ordered.accounting_PD.to_numpy()
    if (np.diff(bounded) < 0).any():
        raise ValueError("clipping must preserve order")
    ends = np.flatnonzero(np.r_[bounded[:-1] != bounded[1:], True])
    counts = [0, *(int(i + 1) for i in ends)]
    candidates = result["candidates"].set_index("number_approved")
    if any(n not in candidates.index for n in counts):
        raise ValueError("a bounded prefix is missing from the raw feasible family")
    mapping = candidates.loc[counts].reset_index().copy()
    mapping["bounded_cutoff"] = [np.nan, *bounded[ends]]
    mapping = mapping.rename(columns={"cutoff_score": "raw_endpoint"})
    mapping["is_frozen_selection"] = mapping.number_approved.eq(baseline_selection["number_approved"])
    if len(mapping) != len(np.unique(bounded)) + 1 or mapping.is_frozen_selection.sum() != 1:
        raise ValueError("bounded-prefix or frozen-selection coverage differs")
    ref = baseline_decisions.sort_values("applicant_id").reset_index(drop=True)
    aligned = bank.sort_values("applicant_id").reset_index(drop=True)
    np.testing.assert_array_equal(ref.applicant_id, aligned.applicant_id)
    tau = baseline_selection["optimal_tau"]
    old_mask = np.zeros(len(bank), dtype=bool) if tau is None else aligned.accounting_PD.to_numpy() <= tau
    np.testing.assert_array_equal(ref.approved, old_mask)
    frozen = mapping.loc[mapping.is_frozen_selection].iloc[0]
    if frozen.funded_principal != baseline_selection["total_funded_principal"] or frozen.perceived_profit_USD != baseline_selection["bank_expected_portfolio_profit"] or not frozen.budget_feasible:
        raise ValueError("frozen baseline cost/objective changed")
    selected = result["decisions"].sort_values("applicant_id").approved.to_numpy()
    values = aligned.expected_profit_perceived.to_numpy()
    old_value = sum((Fraction(float(v)) for v in values[old_mask]), Fraction())
    new_value = sum((Fraction(float(v)) for v in values[selected]), Fraction())
    if new_value < old_value:
        raise ValueError("expanded prefix family lowered optimized perceived profit")
    return mapping, {"all_bounded_prefixes_in_raw_family": True, "bounded_endpoint_count": len(mapping),
                     "additional_raw_endpoints": len(result["candidates"]) - len(mapping),
                     "frozen_selected_mask_available": True, "perceived_objective_nondecreasing_exact": True}


def evaluate_case(banks, observed, truth, case_id):
    """Evaluator-only aggregates; bank policies must already be saved and frozen."""
    if set(banks) != set(POLICIES) or tuple(truth.columns) != TRUE_PROFIT_COLUMNS or tuple(observed.columns) != ("applicant_id", "B"):
        raise ValueError("exact three-source evaluator views required")
    validate_ids(observed, "audit groups"); validate_ids(truth, "saved truth")
    if not observed.B.isin([0, 1]).all() or set(observed.B) != {0, 1} or not np.isfinite(truth.iloc[:, 1:]).all().all() or not truth.true_PD.between(0, 1).all():
        raise ValueError("invalid audit truth/groups")
    data = observed.merge(truth, on="applicant_id", validate="one_to_one").sort_values("applicant_id").reset_index(drop=True)
    if len(data) != len(observed) or set(truth.applicant_id) != set(observed.applicant_id):
        raise ValueError("evaluator IDs differ")
    rows, groups, masks, inputs = [], [], {}, {}
    for source in POLICIES:
        bank = banks[source]; inp = bank["inputs"].sort_values("applicant_id").reset_index(drop=True)
        validate_inputs(inp); np.testing.assert_array_equal(inp.applicant_id, data.applicant_id)
        for regime in (BOUNDED, RAW):
            key = f"{source}__{regime}"
            result = bank["baseline"] if regime == BOUNDED else bank["raw"]
            selection = result["selection"]; decisions = result["decisions"].sort_values("applicant_id").reset_index(drop=True)
            np.testing.assert_array_equal(decisions.applicant_id, data.applicant_id)
            mask = decisions.approved.to_numpy(dtype=bool)
            tau = selection["optimal_tau"] if regime == BOUNDED else selection["cutoff_score"]
            score = inp.accounting_PD if regime == BOUNDED else inp.ordering_score
            np.testing.assert_array_equal(mask, np.zeros(len(inp), dtype=bool) if tau is None else score <= tau)
            np.testing.assert_array_equal(decisions.funded_principal, np.where(mask, inp.requested_principal, 0.))
            masks[key] = mask; inputs[key] = inp
            budget = selection["available_bank_capital"] if regime == BOUNDED else selection["available_capital_USD"]
            capital = math.fsum(inp.loc[mask, "requested_principal"])
            if capital > budget: raise ValueError("evaluation portfolio exceeds original budget")
            row = {"case_id": case_id, "policy": key, "source_policy": source, "ordering_regime": regime,
                   "n_applicants": len(data), "number_approved": int(mask.sum()), "approval_rate": float(mask.mean()),
                   "cutoff_score": tau, "cutoff_units": "raw_PN_score_not_probability" if regime == RAW else "bounded_probability",
                   "available_capital_USD": budget, "funded_principal_USD": capital,
                   "unused_capital_USD": budget - capital, "capital_utilization": capital / budget if budget else None,
                   "perceived_profit_USD": math.fsum(inp.loc[mask, "expected_profit_perceived"]),
                   "true_profit_USD": math.fsum(data.loc[mask, "expected_profit_true"]),
                   "mean_accounting_PD_approved": float(inp.loc[mask, "accounting_PD"].mean()) if mask.any() else None,
                   "mean_true_PD_approved": float(data.loc[mask, "true_PD"].mean()) if mask.any() else None,
                   "zero_PD_approved": int((inp.accounting_PD.eq(0).to_numpy() & mask).sum()),
                   "negative_raw_approved": int((inp.ordering_score.lt(0).to_numpy() & mask).sum()),
                   "reject_all_selected": tau is None}
            rates = {}
            for group in (0, 1):
                member = data.B.eq(group).to_numpy(); issued = member & mask
                rates[group] = float(mask[member].mean())
                groups.append({"case_id": case_id, "policy": key, "source_policy": source, "ordering_regime": regime,
                    "B": group, "n_applicants": int(member.sum()), "number_approved": int(issued.sum()),
                    "approval_rate": rates[group], "funded_principal_USD": math.fsum(inp.loc[issued, "requested_principal"]),
                    "true_profit_USD": math.fsum(data.loc[issued, "expected_profit_true"])})
            row["approval_gap_pp"] = 100 * (rates[1] - rates[0]); rows.append(row)
    indexed = pd.DataFrame(rows).set_index("policy")
    pairs = [(f"{p}__{RAW}", f"{p}__{BOUNDED}", "within_model_ordering") for p in POLICIES]
    pairs += [(f"{a}__{r}", f"{b}__{r}", "between_models_same_ordering") for r in (BOUNDED, RAW) for a, b in PAIRS]
    comparisons, overlap = [], []
    for first, second, kind in pairs:
        a, b = indexed.loc[first], indexed.loc[second]; ma, mb = masks[first], masks[second]
        union = int((ma | mb).sum())
        comparisons.append({"case_id": case_id, "comparison_kind": kind, "policy_first": first, "policy_second": second,
            "true_profit_difference_USD": a.true_profit_USD - b.true_profit_USD,
            "perceived_profit_difference_USD": a.perceived_profit_USD - b.perceived_profit_USD,
            "approval_count_difference": int(a.number_approved - b.number_approved),
            "approval_gap_difference_pp": a.approval_gap_pp - b.approval_gap_pp,
            "funded_principal_difference_USD": a.funded_principal_USD - b.funded_principal_USD,
            "jaccard_overlap": int((ma & mb).sum()) / union if union else None})
        for label, mask in (("approved_by_both", ma & mb), ("first_only", ma & ~mb), ("second_only", ~ma & mb), ("rejected_by_both", ~ma & ~mb)):
            overlap.append({"case_id": case_id, "policy_first": first, "policy_second": second,
                "allocation_category": label, "n_applicants": int(mask.sum()),
                "potential_true_profit_USD": math.fsum(data.loc[mask, "expected_profit_true"]),
                "potential_values_not_earned_on_unfunded_requests": True})
    return {"policies": pd.DataFrame(rows), "groups": pd.DataFrame(groups),
            "comparisons": pd.DataFrame(comparisons), "overlap": pd.DataFrame(overlap)}
