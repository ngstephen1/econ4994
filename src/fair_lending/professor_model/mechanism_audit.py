"""Step 9C: predeclared score diagnostics and zero-only positive-floor audit."""
from __future__ import annotations

from fractions import Fraction
import json

import numpy as np
import pandas as pd

from . import revised_lending as lending
from . import revised_scenarios as original
from .bank_regression import validate_ids

SCHEMA_VERSION = "professor-mechanism-audit-v1"
A, B, C, D, E = ("A_frozen_direct", "B_unclipped_direct_diagnostic", "C_zero_floor_diagnostic",
                 "D_positive_floor_sensitivity", "E_frozen_blind_control")
VARIANTS = (A, B, C, D, E)
POLICIES = (A, D, E)
RISK_SCOPE = "RISK_DIAGNOSTIC_NO_HIDDEN_TRUTH"
RISK_COLUMNS = ("information_scope", "variant_id", "applicant_id", "B", "origin_raw_score", "score",
                "eligible_for_lending", "is_valid_probability", "lower_clipped", "upper_clipped", "positive_floor_replaced")
POLICY_VIEW = ("applicant_id", "variant_id", "estimated_PD_bounded")
# These labels only adapt the fixed-width existing evaluator. No underlying
# regression, group treatment, or risk value changes in the adapter.
EVALUATION_SLOTS = {A: lending.SCENARIO_IDS[1], D: lending.SCENARIO_IDS[0], E: lending.SCENARIO_IDS[2]}


def build_variants(frozen):
    """Use existing raw/bounded risks only. Never read PN, truth, or outcomes."""
    if tuple(frozen.columns) != original.RISK_COLUMNS:
        raise ValueError("exact frozen Step 9A risk schema required")
    references = {}
    for key, scenario in (("direct", original.SCENARIOS[1]["scenario_id"]), ("blind", original.SCENARIOS[2]["scenario_id"])):
        frame = frozen.loc[frozen.scenario_id == scenario].sort_values("applicant_id").reset_index(drop=True)
        validate_ids(frame, scenario)
        if frame.empty or not frame.B.isin([0, 1]).all() or not np.isfinite(frame.loc[:, original.RISK_COLUMNS[4:8]].to_numpy(dtype=float)).all():
            raise ValueError("finite, nonempty frozen risks with binary B required")
        np.testing.assert_array_equal(frame.estimated_PD_adjusted_raw, frame.estimated_PD_base_raw + frame.group_adjustment)
        np.testing.assert_array_equal(frame.estimated_PD_bounded, np.clip(frame.estimated_PD_adjusted_raw, 0, 1))
        references[key] = frame
    direct, blind = references["direct"], references["blind"]
    pd.testing.assert_frame_equal(direct[["applicant_id", "B"]], blind[["applicant_id", "B"]], check_exact=True)
    np.testing.assert_array_equal(direct.estimated_PD_base_raw, blind.estimated_PD_adjusted_raw)
    np.testing.assert_array_equal(direct.group_adjustment, np.where(direct.B.eq(1), -.1, 0.))
    if not blind.group_adjustment.eq(0).all(): raise ValueError("control must have no manual adjustment")
    control = blind.estimated_PD_bounded.to_numpy()
    positive = control[control > 0]
    if len(positive) == 0: raise ValueError("no strictly positive control scores: floor undefined")
    floor = float(np.quantile(positive, .01, method="linear"))
    if not 0 < floor <= 1: raise ValueError("derived floor must be a positive probability")
    raw = direct.estimated_PD_adjusted_raw.to_numpy()
    bounded = direct.estimated_PD_bounded.to_numpy()
    zeros = bounded == 0
    values = {A: bounded, B: raw, C: np.maximum(raw, 0), D: np.where(zeros, floor, bounded), E: control}
    records = []
    for variant in VARIANTS:
        origin = blind.estimated_PD_adjusted_raw.to_numpy() if variant == E else raw
        score = values[variant]
        records.append(pd.DataFrame({
            "information_scope": RISK_SCOPE, "variant_id": variant, "applicant_id": direct.applicant_id,
            "B": direct.B, "origin_raw_score": origin, "score": score,
            "eligible_for_lending": variant in POLICIES, "is_valid_probability": (score >= 0) & (score <= 1),
            "lower_clipped": (origin < 0) & (variant != B),
            "upper_clipped": (origin > 1) & (variant in (A, D, E)),
            "positive_floor_replaced": zeros if variant == D else np.zeros(len(direct), dtype=bool),
        }))
    risks = pd.concat(records, ignore_index=True).loc[:, RISK_COLUMNS].sort_values(["variant_id", "applicant_id"]).reset_index(drop=True)
    derivation = {"quantile": .01, "method": "linear", "source": "frozen_blind_control_strictly_positive_bounded_evaluation_scores",
                  "source_count": len(control), "strictly_positive_source_count": len(positive), "floor_raw_probability": floor,
                  "floor_pp": 100 * floor, "affected_applicants": int(zeros.sum()),
                  "B0_affected": int((zeros & direct.B.eq(0).to_numpy()).sum()),
                  "B1_affected": int((zeros & direct.B.eq(1).to_numpy()).sum()),
                  "existing_nonzero_scores_equal_floor": int(((bounded != 0) & (bounded == floor)).sum()),
                  "direct_raw_above_one": int((raw > 1).sum()),
                  "C_exactly_equals_A": bool(np.array_equal(values[C], bounded)), "outcome_tuning": False}
    return risks, derivation


def validate_variants(risks, frozen, derivation):
    rebuilt, expected = build_variants(frozen)
    pd.testing.assert_frame_equal(risks, rebuilt, check_exact=True)
    if derivation != expected: raise ValueError("floor derivation differs from predeclared rule")
    return {"exact_A_and_E_reproduction": True, "exact_raw_adjustment": True,
            "diagnostics_B_C_excluded_from_lending": True, "positive_floor_rule_valid": True,
            "C_equals_A_when_no_upper_exceedance": expected["C_exactly_equals_A"]}


def definitions(derivation):
    formulas = {A: "clip(frozen_blind_raw - 0.1*B, 0, 1)", B: "frozen_blind_raw - 0.1*B",
                C: "max(frozen_blind_raw - 0.1*B, 0)", D: "replace only exact A zeros by floor", E: "frozen_blind_bounded"}
    return pd.DataFrame([{"information_scope": RISK_SCOPE, "variant_id": v,
                          "role": "valid_policy" if v in POLICIES else "diagnostic_only_never_lend",
                          "formula": formulas[v], "floor_quantile": .01 if v == D else None,
                          "floor_method": "linear" if v == D else "not_applicable",
                          "floor_raw_probability": derivation["floor_raw_probability"] if v == D else None,
                          "floor_affected_applicants": derivation["affected_applicants"] if v == D else 0} for v in VARIANTS])


def risk_diagnostics(risks):
    rows = []
    for variant, frame in risks.groupby("variant_id", sort=True):
        for scope, subset in [("overall", frame), *[(f"B={b}", f) for b, f in frame.groupby("B", sort=True)]]:
            score = subset.score.to_numpy()
            origin = subset.origin_raw_score.to_numpy()
            counts = subset.score.value_counts()
            rows.append({"information_scope": RISK_SCOPE, "variant_id": variant, "scope": scope,
                         "eligible_for_lending": variant in POLICIES, "n_applicants": len(subset),
                         "mean_origin_raw_pp": float(100 * origin.mean()), "mean_score_pp": float(100 * score.mean()),
                         "median_score_pp": float(100 * np.median(score)), "minimum_score_pp": float(100 * score.min()),
                         "maximum_score_pp": float(100 * score.max()), "p01_score_pp": float(100 * np.quantile(score, .01)),
                         "p99_score_pp": float(100 * np.quantile(score, .99)),
                         "mean_score_minus_origin_pp": float(100 * np.mean(score - origin)),
                         "negative_score_count": int((score < 0).sum()), "above_one_count": int((score > 1).sum()),
                         "lower_clipped_count": int(subset.lower_clipped.sum()), "upper_clipped_count": int(subset.upper_clipped.sum()),
                         "floor_replaced_count": int(subset.positive_floor_replaced.sum()),
                         "zero_count": int((score == 0).sum()), "zero_share": float((score == 0).mean()),
                         "unique_score_count": len(counts), "tied_block_count": int((counts > 1).sum()),
                         "tied_block_sizes_JSON": json.dumps(sorted(int(n) for n in counts[counts > 1])),
                         "largest_tied_block_size": int(counts.max()) if (counts > 1).any() else 0})
    return pd.DataFrame(rows)


def optimize_policy(contracts, view, budget):
    """Only A/D/E may become policies. Delegate the exact Step 9B algorithm."""
    validate_ids(view, "mechanism policy scores")
    if tuple(view.columns) != POLICY_VIEW or view.empty or view.variant_id.nunique() != 1 or view.variant_id.iloc[0] not in POLICIES:
        raise ValueError("only explicit valid-policy variants A/D/E may be optimized")
    variant = view.variant_id.iloc[0]
    adapter = view.rename(columns={"variant_id": "scenario_id"}).copy()
    adapter["scenario_id"] = lending.SCENARIO_IDS[1]  # Internal compatibility label, not a fit.
    bank = lending.optimize_scenario(contracts, adapter.loc[:, lending.RISK_VIEW], budget)
    for key in ("decisions", "assessments", "candidates"):
        bank[key] = bank[key].rename(columns={"scenario_id": "variant_id"})
        bank[key]["variant_id"] = bank[key]["policy_id"] = variant
    bank["selection"] = {k: v for k, v in bank["selection"].items() if k != "scenario_id"}
    bank["selection"].update(variant_id=variant, policy_id=variant)
    return bank


def tie_diagnostics(risks, decisions, contracts, banks, budget, floor):
    rows = []
    for variant in POLICIES:
        frame = risks.loc[risks.variant_id == variant].merge(contracts[["applicant_id", "L_i"]], on="applicant_id", validate="one_to_one")
        selection = decisions.loc[decisions.variant_id == variant, ["applicant_id", "approved"]]
        validate_ids(selection, "tie decisions")
        if set(selection.applicant_id) != set(frame.applicant_id): raise ValueError("tie audit IDs differ")
        frame = frame.merge(selection, on="applicant_id", validate="one_to_one")
        counts = frame.score.value_counts()
        selected = banks[variant]["selection"]
        row = {"information_scope": lending.BANK_SCOPE, "variant_id": variant, "n_applicants": len(frame),
               "unique_score_count": len(counts), "tied_block_count": int((counts > 1).sum()),
               "tied_block_sizes_JSON": json.dumps(sorted(int(n) for n in counts[counts > 1])),
               "largest_tied_block_size": int(counts.max()) if (counts > 1).any() else 0,
               "selected_cutoff": selected["optimal_tau"], "selected_boundary_size": selected["selected_cutoff_group_size"],
               "selected_boundary_contains_ties": selected["selected_cutoff_group_size"] > 1,
               "number_approved": selected["number_approved"], "funded_principal_USD": selected["total_funded_principal"],
               "capital_utilization": selected["budget_utilization"], "unused_capital_USD": selected["unused_capital"],
               "reject_all_selected": selected["optimal_tau"] is None}
        for prefix, mask in (("zero", frame.score.eq(0)), ("floor", frame.score.eq(floor) if variant == D else pd.Series(False, index=frame.index))):
            block = frame.loc[mask]
            cost = sum((Fraction(float(v)) for v in block.L_i), Fraction())
            approved = int(block.approved.sum())
            if approved not in (0, len(block)): raise ValueError("selected policy split a tied block")
            row.update({f"{prefix}_block_count": len(block), f"{prefix}_block_principal_USD": float(cost),
                        f"{prefix}_block_budget_share": float(cost) / budget if budget else None,
                        f"{prefix}_block_feasible": cost <= Fraction(float(budget)),
                        f"{prefix}_block_approved": approved,
                        f"{prefix}_block_action": "no_block" if block.empty else "approved_whole" if approved == len(block) else "reject_all_policy" if selected["optimal_tau"] is None else "excluded_whole"})
            for group in (0, 1):
                n = int(frame.B.eq(group).sum()); k = int(block.B.eq(group).sum())
                row[f"B{group}_{prefix}_count"] = k
                row[f"B{group}_{prefix}_share"] = k / n if n else None
        row["positive_scores_below_floor"] = int(((frame.score > 0) & (frame.score < floor)).sum()) if variant == D else 0
        if variant == D:
            prefix = frame.loc[frame.score <= floor]
            cost = sum((Fraction(float(v)) for v in prefix.L_i), Fraction())
            row.update(floor_prefix_applicants=len(prefix), floor_prefix_principal_USD=float(cost),
                       floor_prefix_budget_share=float(cost) / budget if budget else None,
                       floor_prefix_feasible=cost <= Fraction(float(budget)))
        else:
            row.update(floor_prefix_applicants=None, floor_prefix_principal_USD=None,
                       floor_prefix_budget_share=None, floor_prefix_feasible=None)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_policies(banks, references, contracts, audit, truth, budget):
    """Reuse Step 9B's tested six-policy evaluator through label-only adapters."""
    policies = {key: references[key] for key in lending.REFERENCE_IDS}
    for variant, slot in EVALUATION_SLOTS.items():
        policies[slot] = lending.evaluation_policy(banks[variant]["decisions"], banks[variant]["selection"],
                                                   score_column="estimated_PD_bounded", role="step9c_valid_policy")
    portfolio, groups, overlap = lending.evaluate_policies(policies, contracts, audit, truth, budget)
    rename = {slot: variant for variant, slot in EVALUATION_SLOTS.items()}
    for table, columns in ((portfolio, ("policy_id",)), (groups, ("policy_id",)), (overlap, ("policy_first", "policy_second"))):
        for column in columns: table[column] = table[column].replace(rename)
    indexed = portfolio.set_index("policy_id")
    baseline = indexed.loc[A]
    for target, source in (("true_profit_minus_frozen_direct_USD", "true_expected_portfolio_profit_USD"),
                           ("approval_gap_minus_direct_pp", "approval_gap_pp"), ("funded_principal_minus_direct_USD", "funded_principal_USD"),
                           ("approval_count_minus_direct", "number_approved")):
        portfolio[target] = portfolio[source] - baseline[source]
    portfolio["true_profit_minus_blind_control_USD"] = portfolio.true_expected_portfolio_profit_USD - indexed.loc[E, "true_expected_portfolio_profit_USD"]
    group_baseline = groups.loc[groups.policy_id == A].set_index("B")
    groups["approval_rate_minus_direct_pp"] = [100 * (row.approval_rate - group_baseline.loc[row.B, "approval_rate"]) for row in groups.itertuples()]
    return portfolio, groups, overlap
