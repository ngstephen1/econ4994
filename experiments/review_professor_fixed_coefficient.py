"""Post-hoc explanation of frozen Step 10B evidence; never fits or optimizes.

Default: read and print only. --write saves a new, content-addressed review.
No simulation, repayment draw, regression fit, or lending search is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "cd7f3ee268cfc4d04ab906f0dbb6caab0de730aa21fdd96743c7eae815efebc7"
METRICS = Path("results/metrics/professor_model/fixed_coefficient")
TABLES = Path("results/tables/professor_model/fixed_coefficient") / EXPERIMENT[:16]
W, Z, F = "frozen_aware", "aware_B_zero", "frozen_blind"
POLICIES = (W, Z, F)
CASES = {"canonical", *(f"noise_{i:03d}" for i in range(200))}
ORDER = ("const", "Y", "D", "C")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        if file_hash(root / name) != expected:
            raise ValueError(f"source changed: {name}")


def sealed_json(path):
    value = json.loads(Path(path).read_text())
    saved = value.pop("content_fingerprint")
    if fingerprint(value) != saved:
        raise ValueError(f"invalid content fingerprint: {path}")
    return value


def validate_policy_cells(frame, expected_cases):
    """Require a complete three-policy panel before calculating diagnostics."""
    if frame.duplicated(["case_id", "policy"]).any():
        raise ValueError("duplicate policy cells")
    if set(map(tuple, frame[["case_id", "policy"]].to_numpy())) != {(c, p) for c in expected_cases for p in POLICIES}:
        raise ValueError("incomplete/unexpected policy cells")
    columns = ["true_profit_USD", "funded_principal_USD", "available_capital_USD", "zero_block_principal_USD"]
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError("nonfinite economic values")
    if (frame.funded_principal_USD < 0).any() or (frame.funded_principal_USD > frame.available_capital_USD).any():
        raise ValueError("original dollar budget violated")
    for name in ("reject_all_selected", "zero_block_feasible"):
        if not frame[name].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f"invalid boolean {name}")
    if not np.array_equal(frame.reject_all_selected, frame.number_approved.eq(0)):
        raise ValueError("reject-all count inconsistent")


def tail_attribution(policies, paired):
    """Partition a saved mean by discordant reject-all status, without trimming.

    The conditional means are post-hoc descriptions, not replacement estimates
    or a causal mediation decomposition. Contributions retain the full n.
    """
    if paired.duplicated(["case_id", "comparison"]).any():
        raise ValueError("duplicate paired observations")
    out = []
    for comparison, rows in paired.groupby("comparison", sort=True):
        if rows.policy_first.nunique() != 1 or rows.policy_second.nunique() != 1:
            raise ValueError("inconsistent comparison labels")
        first, second = rows.iloc[0][["policy_first", "policy_second"]]
        left = policies.loc[policies.policy.eq(first)].set_index("case_id")
        right = policies.loc[policies.policy.eq(second)].set_index("case_id")
        ids = rows.case_id.tolist()
        if set(ids) != set(left.index) or set(ids) != set(right.index):
            raise ValueError("paired cases differ from policy panel")
        values = rows.true_profit_difference_USD.to_numpy()
        np.testing.assert_allclose(values, left.loc[ids, "true_profit_USD"] - right.loc[ids, "true_profit_USD"], atol=1e-6, rtol=0)
        different = (left.loc[ids, "reject_all_selected"] != right.loc[ids, "reject_all_selected"]).to_numpy()
        n = len(values)
        contributions = [math.fsum(values[mask]) / n for mask in (different, ~different)]
        np.testing.assert_allclose(math.fsum(contributions), math.fsum(values) / n, atol=1e-6, rtol=0)
        out.append({"comparison": comparison, "n": n, "full_mean_USD": math.fsum(values) / n,
                    "discordant_reject_all_cases": sorted(rows.loc[different, "case_id"].tolist()),
                    "discordant_count": int(different.sum()),
                    "discordant_contribution_to_full_mean_USD": contributions[0],
                    "remaining_count": int((~different).sum()),
                    "remaining_contribution_to_full_mean_USD": contributions[1],
                    "remaining_conditional_mean_USD": math.fsum(values[~different]) / int((~different).sum()) if (~different).any() else None})
    return out


def implied_group_projection(aware, blind):
    """Algebraic diagnostic from saved coefficients, not a new B~X fit.

    With identical training X/target, OLS gives b_F=b_W+b_B*delta where delta
    is the training projection of B on X. A zero b_B cannot identify delta.
    """
    beta = aware["B"]
    if beta == 0:
        return None
    return np.array([(blind[k] - aware[k]) / beta for k in ORDER])


def build_review(root=ROOT):
    directory = root / METRICS / EXPERIMENT[:16]
    validation = json.loads((directory / "final_validation.json").read_text())
    preservation = json.loads((root / METRICS / "preservation_manifest.json").read_text())
    verify_hashes(root, preservation["original_checksums"])
    verify_hashes(root, validation["artifact_hashes"])
    index = json.loads((root / "results/metrics/professor_model/professor_model_evidence_index.json").read_text())
    if fingerprint({k: v for k, v in index.items() if k != "extensions"}) != preservation["prior_index_core_fingerprint"]:
        raise ValueError("historical Step 10A index core changed")
    verify_hashes(root, {a["path"]: a["sha256"] for a in index["extensions"]["step10b"]["artifacts"]})
    freezes = {}
    for path in sorted((directory / "freezes").glob("*.json")):
        value = sealed_json(path)
        if value["case_id"] in freezes or value["experiment_fingerprint"] != EXPERIMENT or set(value["banks"]) != set(POLICIES):
            raise ValueError("duplicate/incompatible bank freeze")
        freezes[value["case_id"]] = value
    if set(freezes) != CASES:
        raise ValueError("all 201 frozen cases required")
    read = lambda name: pd.read_csv(root / TABLES / f"{name}.csv", float_precision="round_trip")
    noise, canonical, paired, summary = map(read, ("noise_policies", "canonical_policies", "noise_paired", "noise_summary"))
    validate_policy_cells(noise, CASES - {"canonical"})
    validate_policy_cells(canonical, {"canonical"})
    combined = pd.concat([canonical, noise], ignore_index=True)
    # A saved assertion alone is insufficient: replay raw linear predictions
    # and independently compare their maximum with the contract break-even PD.
    observed = pd.read_parquet(root / "data/synthetic/professor_model/baseline/applicants_bank_view.parquet")
    splits = pd.read_csv(root / "data/synthetic/professor_model/baseline/applicant_splits.csv")
    observed = observed.merge(splits, on="applicant_id", validate="one_to_one").sort_values("applicant_id")
    x = np.column_stack([np.ones(len(observed)), observed[["Y", "D", "C"]].to_numpy()])
    group = observed.B.to_numpy(); evaluation = observed.split.eq("evaluation").to_numpy()
    if len(observed) != 100000 or evaluation.sum() != 20000:
        raise ValueError("unexpected saved population/split")
    contracts = pd.read_parquet(root / "data/processed/professor_model/loan_contracts.parquet")
    contracts = contracts.sort_values("applicant_id").reset_index(drop=True)
    np.testing.assert_array_equal(observed.loc[evaluation, "applicant_id"], contracts.applicant_id)
    if not contracts["T"].eq(30).all() or not contracts.r.eq(.05).all() or not contracts.theta.eq(.1).all():
        raise ValueError("unexpected frozen economics")
    break_even = .05 / 1.05
    flags = {p: {k: 0 for k in ("all_requests_bank_profitable", "largest_feasible_cutoff_selected", "capital_limits_next_cutoff")} for p in POLICIES}
    algebra, max_pd, block_residual = [], 0., 0.
    saved_cells = combined.set_index(["case_id", "policy"])
    for case, freeze in sorted(freezes.items()):
        banks = freeze["banks"]
        aware, blind = [banks[p]["source_model"]["coefficients"] for p in (W, F)]
        raw_z = x @ np.array([aware[k] for k in ORDER])
        raw_f = x @ np.array([blind[k] for k in ORDER])
        raw_w = raw_z + aware["B"] * group
        raw_z = raw_w - aware["B"] * group  # authoritative subtract-before-clip operation
        for p, raw in ((W, raw_w), (Z, raw_z), (F, raw_f)):
            bounded = np.clip(raw[evaluation], 0, 1)
            saved = saved_cells.loc[(case, p)]
            zero = bounded == 0
            cost = math.fsum(contracts.loc[zero, "L_i"])
            block_residual = max(block_residual, abs(cost - saved.zero_block_principal_USD))
            if int(zero.sum()) != saved.zero_score_count or block_residual > 1e-6 or bool(cost <= saved.available_capital_USD) != saved.zero_block_feasible:
                raise ValueError("replayed zero-block count/cost/feasibility differs")
            max_pd = max(max_pd, float(bounded.max()))
            if not (bounded < break_even).all():
                raise ValueError("not every saved request is profitable under the current contract")
            for key in flags[p]:
                if banks[p]["selection"][key] is not True:
                    raise ValueError(f"unexpected prefix diagnostic: {case}/{p}/{key}")
                flags[p][key] += 1
        delta = implied_group_projection(aware, blind)
        if delta is None:
            raise ValueError("saved zero B coefficient: projection diagnostic undefined")
        projected = x @ delta
        residual = raw_f - raw_z - aware["B"] * projected
        # Check the projection interpretation through its normal equations;
        # the score identity alone follows from the definition of delta.
        train_x = x[~evaluation]
        moments = np.mean(train_x * (group[~evaluation] - projected[~evaluation])[:, None], axis=0)
        normalized_moments = moments / np.sqrt(np.mean(train_x ** 2, axis=0))
        if np.max(np.abs(normalized_moments)) > 1e-7:
            raise ValueError("saved coefficient differences do not imply the training B projection")
        algebra.append({"case_id": case, "beta_B": aware["B"],
                        "implied_projection_coefficients": dict(zip(ORDER, delta.tolist())),
                        "implied_projection_training_mean": float(projected[~evaluation].mean()),
                        "implied_projection_evaluation_mean": float(projected[evaluation].mean()),
                        "implied_projection_evaluation_SD": float(projected[evaluation].std(ddof=1)),
                        "implied_projection_evaluation_min": float(projected[evaluation].min()),
                        "implied_projection_evaluation_max": float(projected[evaluation].max()),
                        "F_minus_Z_mean_raw_score": float((raw_f - raw_z)[evaluation].mean()),
                        "F_minus_Z_SD_raw_score": float((raw_f - raw_z)[evaluation].std(ddof=1)),
                        "F_Z_raw_rank_correlation": float(spearmanr(raw_f[evaluation], raw_z[evaluation]).statistic),
                        "max_normalized_projection_moment": float(np.max(np.abs(normalized_moments))),
                        "max_algebra_residual": float(np.max(np.abs(residual)))})
    truth = pd.read_parquet(root / "data/processed/professor_model/true_expected_profit.parquet")
    mismatch = combined.reject_all_selected.ne(~combined.zero_block_feasible)
    if mismatch.any() or not truth.expected_profit_true.gt(0).all():
        raise ValueError("saved reject-all mechanism or true profitability differs")
    cases = {c for row in tail_attribution(noise, paired) for c in row["discordant_reject_all_cases"]}
    selected = combined[combined.case_id.isin(cases)].copy()
    selected["zero_block_budget_share"] = selected.zero_block_principal_USD / selected.available_capital_USD
    selected = selected[["case_id", "policy", "number_approved", "zero_score_count", "zero_block_budget_share", "true_profit_USD"]]
    source_names = [str(METRICS / "preservation_manifest.json"), str(METRICS / EXPERIMENT[:16] / "final_validation.json"),
                    "experiments/review_professor_fixed_coefficient.py", "tests/test_professor_post10b_review.py"]
    report = {"schema_version": "professor-post10b-review-v1", "scope": "post_hoc_saved_evidence_review_no_new_policy",
              "experiment_fingerprint": EXPERIMENT, "new_fits": 0, "new_draws": 0, "new_cutoff_searches": 0,
              "input_hashes": {p: file_hash(root / p) for p in source_names},
              "step10a_core_fingerprint": preservation["prior_index_core_fingerprint"],
              "step10b_index_extension_fingerprint": fingerprint(index["extensions"]["step10b"]),
              "checks": {"original_files_unchanged": len(preservation["original_checksums"]),
                         "step10b_artifact_hashes_match": len(validation["artifact_hashes"]),
                         "step10b_index_artifacts_match": len(index["extensions"]["step10b"]["artifacts"]),
                         "cases": len(freezes), "policy_cells": len(combined), "prefix_flags": flags,
                         "zero_blocks_replayed": len(combined), "max_zero_block_principal_residual_USD": block_residual,
                         "zero_block_replay_tolerance_USD": 1e-6},
              "economics": {"T": 30, "r_per_period": .05, "theta": .1, "break_even_PD": break_even,
                            "max_replayed_evaluation_bounded_PD": max_pd, "all_evaluation_requests_true_profitable": True,
                            "true_PD_max": float(truth.true_PD.max()), "reject_all_iff_zero_block_exceeds_budget": True},
              "canonical": canonical.replace({np.nan: None}).to_dict("records"),
              "noise_profit_summaries": summary[summary.metric.eq("true_profit_difference_USD")].to_dict("records"),
              "tail_attribution": tail_attribution(noise, paired), "tail_policy_details": selected.to_dict("records"),
              "coefficient_algebra": algebra, "documented_prior_tests": validation["targeted_dependency_tests_including_new"],
              "full_suite_rerun": False,
              "limitations": ["one fixed population and split", "post-hoc diagnostic partition, not a mediation decomposition",
                              "no new B-on-X regression fit; projection implied algebraically", "stylized undiscounted growing payments",
                              "no universal profit/fairness ranking", "four known V2 lookup failures remain unresolved"]}
    # Check preservation again after the review; none of the old files is an output.
    verify_hashes(root, preservation["original_checksums"])
    verify_hashes(root, validation["artifact_hashes"])
    return report


def write_review(report, root=ROOT):
    digest = fingerprint(report)
    path = root / "results/metrics/professor_model/post10b_review" / f"{digest[:16]}.json"
    content = json.dumps({**report, "content_fingerprint": digest}, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError("review identity collision")
    else:
        with path.open("x") as stream:
            stream.write(content)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="save a new review JSON; otherwise print only")
    args = parser.parse_args(argv)
    report = build_review()
    print(json.dumps({k: report[k] for k in ("scope", "checks", "economics", "tail_attribution")}, indent=2))
    if args.write:
        print(f"Saved review: {write_review(report)}")


if __name__ == "__main__":
    main()
