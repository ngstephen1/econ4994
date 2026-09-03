"""Monte Carlo summaries, signatures, and practical detection thresholds."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PRACTICAL_GAP = 0.01


def mechanism_signature(row: dict[str, Any] | pd.Series) -> str:
    """Classify a synthetic diagnostic pattern using predeclared rules."""
    raw_nonzero = bool(row["raw_ci_high"] < 0.0 or row["raw_ci_low"] > 0.0)
    adjusted_nonzero = bool(row["model_2_black_p_value"] < 0.05)
    blind = abs(float(row["race_blind_predicted_probability_gap"])) >= PRACTICAL_GAP
    aware = abs(float(row["race_aware_predicted_probability_gap"])) >= PRACTICAL_GAP
    aware_larger = (
        abs(float(row["race_aware_predicted_probability_gap"]))
        - abs(float(row["race_blind_predicted_probability_gap"]))
        >= PRACTICAL_GAP
    )
    if not raw_nonzero and not adjusted_nonzero and not blind:
        return "Pattern A"
    if raw_nonzero and adjusted_nonzero and not blind and aware:
        return "Pattern B"
    if raw_nonzero and not adjusted_nonzero and blind:
        return "Pattern C"
    if raw_nonzero and adjusted_nonzero and blind and aware and aware_larger:
        return "Pattern D"
    return "Other / ambiguous"


def relative_recovery(
    estimate: pd.Series | np.ndarray, truth: pd.Series | np.ndarray
) -> np.ndarray:
    """Return 1-relative-error, floored at zero; zero truth is undefined."""
    estimated = np.asarray(estimate, dtype=float)
    target = np.asarray(truth, dtype=float)
    result = np.full(len(estimated), np.nan, dtype=float)
    nonzero = np.abs(target) > 1e-12
    result[nonzero] = np.maximum(
        0.0, 1.0 - np.abs(estimated[nonzero] - target[nonzero]) / np.abs(target[nonzero])
    )
    return result


def summarize_estimate(
    estimate: pd.Series,
    truth: pd.Series | None = None,
) -> dict[str, float | int]:
    """Summarize one estimand with optional run-specific synthetic truth."""
    values = pd.to_numeric(estimate, errors="coerce").dropna().to_numpy(dtype=float)
    result: dict[str, float | int] = {
        "n_completed": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else float("nan"),
        "mc_sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "q025": float(np.quantile(values, 0.025)) if len(values) else float("nan"),
        "q975": float(np.quantile(values, 0.975)) if len(values) else float("nan"),
    }
    if truth is None:
        result.update(truth_mean=float("nan"), bias=float("nan"), rmse=float("nan"))
        return result
    aligned = pd.concat(
        [pd.to_numeric(estimate, errors="coerce"), pd.to_numeric(truth, errors="coerce")],
        axis=1,
    ).dropna()
    if aligned.empty:
        result.update(truth_mean=float("nan"), bias=float("nan"), rmse=float("nan"))
    else:
        error = aligned.iloc[:, 0].to_numpy() - aligned.iloc[:, 1].to_numpy()
        result.update(
            truth_mean=float(aligned.iloc[:, 1].mean()),
            bias=float(np.mean(error)),
            rmse=float(np.sqrt(np.mean(np.square(error)))),
        )
    return result


def _group_columns(family: str) -> list[str]:
    columns = ["n_rows", "direct_black_log_odds", "upstream_strength", "scenario_family"]
    return ["experiment_family", *columns]


ESTIMANDS = {
    "raw_approval_gap": None,
    "model_0_black_coefficient": "true_direct_log_odds",
    "model_1_black_coefficient": "true_direct_log_odds",
    "model_2_black_coefficient": "true_direct_log_odds",
    "model_3_black_coefficient": "true_direct_log_odds",
    "model_2_adjusted_probability_gap": "true_direct_probability_gap",
    "race_blind_predicted_probability_gap": "test_true_probability_gap",
    "race_aware_predicted_probability_gap": "test_true_probability_gap",
    "race_blind_true_probability_mae": None,
    "race_aware_true_probability_mae": None,
}


def summarize_runs(runs: pd.DataFrame, family: str) -> pd.DataFrame:
    """Create a long/tidy Monte Carlo summary for one experiment family."""
    if runs.empty:
        return pd.DataFrame()
    group_columns = _group_columns(family)
    rows: list[dict[str, Any]] = []
    for keys, group in runs.groupby(group_columns, dropna=False, sort=True):
        common = dict(zip(group_columns, keys, strict=True))
        for estimand, truth_column in ESTIMANDS.items():
            if estimand not in group:
                continue
            summary = summarize_estimate(
                group[estimand], group[truth_column] if truth_column else None
            )
            row = {**common, "estimand": estimand, **summary}
            if estimand == "model_2_black_coefficient":
                detected = group["model_2_black_p_value"].lt(0.05)
                row["detection_probability"] = float(detected.mean())
                truth = group["true_direct_log_odds"]
                nonzero = truth.ne(0.0)
                row["sign_recovery_rate"] = (
                    float(
                        np.mean(
                            np.sign(group.loc[nonzero, estimand])
                            == np.sign(truth.loc[nonzero])
                        )
                    )
                    if nonzero.any()
                    else float("nan")
                )
                row["coverage_probability"] = float(
                    (
                        group["model_2_black_ci_low"].le(truth)
                        & group["model_2_black_ci_high"].ge(truth)
                    ).mean()
                )
            else:
                row.update(
                    detection_probability=float("nan"),
                    sign_recovery_rate=float("nan"),
                    coverage_probability=float("nan"),
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns + ["estimand"]).reset_index(drop=True)


def detection_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detection, false positives, coverage, and ML recovery rates."""
    groups = [
        "n_rows",
        "direct_black_log_odds",
        "upstream_strength",
        "scenario_family",
    ]
    rows = []
    for keys, group in runs.groupby(groups, dropna=False, sort=True):
        truth = group["true_direct_log_odds"]
        significant = group["model_2_black_p_value"].lt(0.05)
        negative_detection = significant & group["model_2_black_coefficient"].lt(0.0)
        coverage = group["model_2_black_ci_low"].le(truth) & group[
            "model_2_black_ci_high"
        ].ge(truth)
        aware_recovery = relative_recovery(
            group["race_aware_predicted_probability_gap"],
            group["test_true_probability_gap"],
        )
        blind_recovery = relative_recovery(
            group["race_blind_predicted_probability_gap"],
            group["test_true_probability_gap"],
        )
        aware_sign = np.sign(group["race_aware_predicted_probability_gap"]) == np.sign(
            group["test_true_probability_gap"]
        )
        rows.append(
            {
                **dict(zip(groups, keys, strict=True)),
                "n_replications": len(group),
                "model_2_detection_probability": float(significant.mean()),
                "model_2_negative_detection_probability": float(negative_detection.mean()),
                "model_2_false_positive_rate": (
                    float(significant.mean()) if bool(truth.eq(0.0).all()) else float("nan")
                ),
                "model_2_coverage_probability": float(coverage.mean()),
                "raw_gap_below_minus_2pp_probability": float(
                    group["raw_approval_gap"].le(-0.02).mean()
                ),
                "race_aware_75pct_recovery_probability": float(
                    np.nanmean((aware_recovery >= 0.75) & aware_sign)
                ),
                "race_blind_below_50pct_recovery_probability": float(
                    np.nanmean(blind_recovery < 0.50)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def coverage_table(runs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "n_rows",
        "direct_black_log_odds",
        "upstream_strength",
        "scenario_family",
        "n_replications",
        "model_2_coverage_probability",
    ]
    return detection_table(runs).loc[:, columns]


def signature_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Return signature frequencies and the modal diagnostic per setting."""
    groups = ["n_rows", "direct_black_log_odds", "upstream_strength", "scenario_family"]
    frame = runs.copy()
    if "mechanism_signature" not in frame:
        frame["mechanism_signature"] = frame.apply(mechanism_signature, axis=1)
    counts = (
        frame.groupby(groups + ["mechanism_signature"], dropna=False)
        .size()
        .rename("n_replications")
        .reset_index()
    )
    counts["signature_share"] = counts["n_replications"] / counts.groupby(groups)[
        "n_replications"
    ].transform("sum")
    counts["is_modal_signature"] = counts.groupby(groups)["n_replications"].transform(
        "max"
    ).eq(counts["n_replications"])
    return counts.sort_values(groups + ["mechanism_signature"]).reset_index(drop=True)


def infer_detection_thresholds(detection: pd.DataFrame, n_rows: int = 100_000) -> pd.DataFrame:
    """Infer four predeclared practical thresholds from routine summaries."""
    at_size = detection.loc[detection["n_rows"].eq(n_rows)].copy()

    def first_value(frame: pd.DataFrame, condition: pd.Series, field: str, ascending: bool = True):
        eligible = frame.loc[condition].sort_values(field, ascending=ascending)
        return (float(eligible.iloc[0][field]), "reached") if len(eligible) else (float("nan"), "not reached")

    direct = at_size.loc[at_size["upstream_strength"].eq(0.0)].copy()
    direct["direct_magnitude"] = direct["direct_black_log_odds"].abs()
    upstream = at_size.loc[at_size["direct_black_log_odds"].eq(0.0)].copy()
    values = []
    threshold, status = first_value(
        direct,
        direct["model_2_negative_detection_probability"].ge(0.80)
        & direct["direct_magnitude"].gt(0.0),
        "direct_magnitude",
    )
    values.append(("model_2_negative_detection_80pct", "absolute direct log-odds penalty", threshold, status))
    threshold, status = first_value(
        upstream,
        upstream["raw_gap_below_minus_2pp_probability"].ge(0.80)
        & upstream["upstream_strength"].gt(0.0),
        "upstream_strength",
    )
    values.append(("raw_gap_below_minus_2pp_80pct", "upstream strength", threshold, status))
    threshold, status = first_value(
        direct,
        direct["race_aware_75pct_recovery_probability"].ge(0.80)
        & direct["direct_magnitude"].gt(0.0),
        "direct_magnitude",
    )
    values.append(("race_aware_75pct_recovery_in_80pct_runs", "absolute direct log-odds penalty", threshold, status))
    threshold, status = first_value(
        direct,
        direct["race_blind_below_50pct_recovery_probability"].ge(0.80)
        & direct["direct_magnitude"].gt(0.0),
        "direct_magnitude",
    )
    values.append(("race_blind_below_50pct_recovery_in_80pct_runs", "absolute direct log-odds penalty", threshold, status))
    return pd.DataFrame(values, columns=["threshold", "scale", "value", "status"]).assign(n_rows=n_rows)
