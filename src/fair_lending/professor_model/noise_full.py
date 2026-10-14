"""Step 8B extensions only: streams, evaluator OLS decomposition and MC summaries.

All operational fits, economics, exact cutoffs and audits reuse noise_pilot.
The pilot's code and schemas are not modified to expand its scope.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.tools import pinv_extended

from . import bank_regression as aware
from . import noise_pilot as pilot
from .noisy_signal import raw_noisy_measurement

R = 200
ORIGINAL_ABS_GAP_PP = 14.385705
DECOMPOSITION_ATOL = 1e-12
DECOMPOSITION_RTOL = 1e-10


def draw_targets(truth, replication):
    if type(replication) is not int or not 0 <= replication < R:
        raise ValueError("full study permits only indices 0..199")
    if tuple(truth.columns) != ("applicant_id", "true_PD"):
        raise ValueError("simulator-only saved truth view required")
    aware.validate_ids(truth, "fixed simulator truth")
    ordered = truth.sort_values("applicant_id").reset_index(drop=True)
    child = np.random.SeedSequence(499408).spawn(R)[replication]
    eta = np.random.Generator(np.random.PCG64(child)).normal(0., .1, len(ordered))
    targets = ordered[["applicant_id"]].assign(PN=raw_noisy_measurement(ordered.true_PD.to_numpy(), eta))
    # Same stream contract as Step 8A, allowing exact first-five validation.
    stream = {"master_seed": 499408, "spawn_key": list(child.spawn_key), "pool_size": child.pool_size,
              "bit_generator": "PCG64", "applicant_order": "ascending_lexicographic_applicant_id",
              "eta_sha256": hashlib.sha256(eta.astype("<f8").tobytes()).hexdigest(),
              "target_fingerprint": pilot.frame_fingerprint(targets), "ordered_ID_fingerprint": aware.ids_fingerprint(ordered.applicant_id),
              "eta_mean": float(eta.mean()), "eta_sample_SD": float(eta.std(ddof=1)),
              "PN_mean": float(targets.PN.mean()), "PN_negative_count": int((targets.PN < 0).sum())}
    return targets, eta, stream


def reference_design(observed, truth):
    """Evaluator-only, same sorted training IDs and [1,Y,D,C,B] matrix."""
    ordered = observed.sort_values("applicant_id").reset_index(drop=True)
    aligned = ordered[["applicant_id"]].merge(truth, on="applicant_id", validate="one_to_one")
    if len(aligned) != len(ordered) or set(aligned.applicant_id) != set(truth.applicant_id):
        raise ValueError("signal-reference IDs differ")
    mask = ordered.split.eq("train").to_numpy()
    matrix = aware.design_matrix(ordered.loc[mask])
    return matrix, .8 * aligned.loc[mask, "true_PD"].to_numpy(), mask


def fit_signal_reference(matrix, target):
    """One evaluator-only OLS; never passed to operational lending code."""
    result = sm.OLS(target, matrix, missing="raise").fit(method="pinv")
    return {"information_scope": "EVALUATOR_ONLY_NOT_AN_OPERATIONAL_BANK",
            "feature_order": list(aware.FEATURES), "design_order": list(aware.DESIGN_COLUMNS),
            "target": "0.8 * saved_true_PD", "coefficients": {k: float(result.params[k]) for k in aware.DESIGN_COLUMNS},
            "beta_B_signal": float(result.params["B"]), "training_count": len(matrix),
            "matrix_fingerprint": pilot.frame_fingerprint(matrix.reset_index(drop=True)),
            "target_sha256": hashlib.sha256(np.asarray(target, dtype="<f8").tobytes()).hexdigest()}


def decomposition(beta_B, eta, train_mask, B_pseudoinverse_row, signal_reference):
    """OLS coefficient of eta via the same fixed pseudoinverse, not a bank fit."""
    noise = float(B_pseudoinverse_row @ eta[train_mask])
    signal = signal_reference["beta_B_signal"]
    residual = float(beta_B - signal - noise)
    if not np.isclose(beta_B, signal + noise, rtol=DECOMPOSITION_RTOL, atol=DECOMPOSITION_ATOL):
        raise ValueError("OLS signal/noise decomposition failed")
    return {"beta_B_signal": signal, "beta_B_noise": noise, "beta_B_total": beta_B,
            "residual": residual, "valid": True, "atol": DECOMPOSITION_ATOL, "rtol": DECOMPOSITION_RTOL}


def B_row(matrix):
    # statsmodels OLS(method='pinv') uses this same 1e-15-cutoff SVD utility.
    return pinv_extended(matrix.to_numpy())[0][list(matrix.columns).index("B")]


def distribution(values):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(a)
    if not n:
        return {"n_defined": 0, **{k: None for k in ("mean", "median", "sample_SD", "MCSE_mean", "p2.5", "p5", "p25", "p75", "p95", "p97.5", "minimum", "maximum")}}
    sd = float(a.std(ddof=1)) if n > 1 else None
    return {"n_defined": n, "mean": float(a.mean()), "median": float(np.median(a)), "sample_SD": sd,
            "MCSE_mean": sd / math.sqrt(n) if sd is not None else None,
            **{f"p{q:g}": float(np.quantile(a, q / 100, method="linear")) for q in (2.5, 5, 25, 75, 95, 97.5)},
            "minimum": float(a.min()), "maximum": float(a.max())}


def paired_table(records):
    rows = []
    for r in sorted(records, key=lambda r: r["replication"]):
        indexed = {m["policy"]: m for m in r["policy_metrics"]}
        a, b = (indexed[p] for p in pilot.POLICIES)
        rows.append({"replication": r["replication"], "beta_B_hat": r["banks"]["group_aware"]["model"]["coefficients"]["B"],
                     "aware_gap_pp": a["approval_gap_pp"], "blind_gap_pp": b["approval_gap_pp"],
                     "gap_change_pp": b["approval_gap_pp"] - a["approval_gap_pp"],
                     "aware_abs_gap_pp": abs(a["approval_gap_pp"]), "blind_abs_gap_pp": abs(b["approval_gap_pp"]),
                     "abs_gap_change_pp": abs(b["approval_gap_pp"]) - abs(a["approval_gap_pp"]),
                     "true_profit_change_USD": b["true_expected_portfolio_profit"] - a["true_expected_portfolio_profit"],
                     **{f"{prefix}_{key}": m[key] for prefix, m in (("aware", a), ("blind", b)) for key in
                        ("true_expected_portfolio_profit", "signed_oracle_minus_policy_true_profit", "number_approved", "funded_principal", "capital_utilization", "B0_approval_rate", "B1_approval_rate")}})
    return pd.DataFrame(rows)


def summarize_records(records, expected=R):
    if len(records) != expected or sorted(r["replication"] for r in records) != list(range(expected)) or any(r["status"] != "success" for r in records):
        raise ValueError("incomplete or duplicate full-study replication coverage")
    paired = paired_table(records)
    tidy, coefficients, group_rows, ties = [], [], [], []
    for r in sorted(records, key=lambda r: r["replication"]):
        pair = paired.loc[paired.replication == r["replication"]].iloc[0].to_dict()
        for m in r["policy_metrics"]:
            tidy.append({**pair, **m, "runtime_seconds": r["runtime_seconds"], "origin": r["origin"],
                         "negative_raw_share": m["negative_raw_count"] / m["n_applicants"],
                         "zero_block_budget_ratio": m["zero_block_principal_USD"] / m["available_bank_capital"],
                         **r["decomposition"], **r["comparison"]})
            ties.append({"replication": r["replication"], "policy": m["policy"], **{k: m[k] for k in
                         ("bounded_zero_count", "zero_block_principal_USD", "zero_block_feasible", "number_approved", "optimal_tau", "feasible_cutoff_candidates_including_reject_all", "unused_capital", "reject_all_selected", "maximum_budget_violation_USD")},
                         "zero_block_budget_ratio": m["zero_block_principal_USD"] / m["available_bank_capital"]})
        for p in pilot.POLICIES:
            model = r["banks"][p]["model"]
            coefficients.extend({"replication": r["replication"], "policy": p, "term": t, "coefficient": model["coefficients"][t]} for t in model["design_order"])
        group_rows.extend({"replication": r["replication"], **g} for g in r["group_approval"])
    summary = [{"metric": c, "scope": "paired_replications", "n_successful": expected, **distribution(paired[c])} for c in paired if c != "replication"]
    tidy = pd.DataFrame(tidy).sort_values(["replication", "policy"]).reset_index(drop=True)
    for policy, frame in tidy.groupby("policy", sort=True):
        for kind in ("raw", "bounded"):
            for target in ("PN", "true_PD"):
                for metric in ("MAE", "RMSE", "mean_signed_error"):
                    column = f"{kind}_{target}_{metric}"
                    summary.append({"metric": column, "scope": policy, "n_successful": expected, **distribution(frame[column])})
    coef = pd.DataFrame(coefficients)
    coef_summary = pd.DataFrame([{"policy": p, "term": t, **distribution(f.coefficient)} for (p, t), f in coef.groupby(["policy", "term"], sort=True)])
    group = pd.DataFrame(group_rows)
    group_summary = pd.DataFrame([{"policy": p, "B": int(b), "metric": column, **distribution(f[column])}
                                  for (p, b), f in group.groupby(["policy", "B"], sort=True)
                                  for column in ("number_approved", "approval_rate", "funded_principal", "funded_principal_per_applicant")])
    frequencies = []
    def frequency(name, mask, scope="replications"):
        values = np.asarray(mask, dtype=bool)
        frequencies.append({"event": name, "scope": scope, "count": int(values.sum()), "denominator": len(values), "fraction": float(values.mean())})
    for column in ("beta_B_hat", "aware_gap_pp", "blind_gap_pp", "true_profit_change_USD"):
        for sign, mask in (("positive", paired[column] > 0), ("negative", paired[column] < 0), ("zero", paired[column] == 0)):
            frequency(f"{column}_{sign}", mask)
    frequency("blind_smaller_absolute_gap", paired.blind_abs_gap_pp < paired.aware_abs_gap_pp)
    frequency("aware_abs_gap_at_least_original_14.385705pp", paired.aware_abs_gap_pp >= ORIGINAL_ABS_GAP_PP)
    for scope, frame in [("pooled_policy_replications", tidy), *list(tidy.groupby("policy", sort=True))]:
        frequency("reject_all", frame.reject_all_selected, scope)
        frequency("zero_block_over_90pct_capital", frame.zero_block_budget_ratio > .9, scope)
        frequency("zero_block_exceeds_capital", ~frame.zero_block_feasible, scope)
    return {"results": tidy, "coefficients": coef_summary, "groups": group_summary, "ties": pd.DataFrame(ties),
            "summary": pd.DataFrame(summary), "frequencies": frequencies, "paired": paired}
