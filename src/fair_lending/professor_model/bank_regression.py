"""Step 5: raw-PN OLS with an explicit observable-feature information boundary.

No generation, truth lookup, fitting choices, lending, or economic accounting
occurs in this module. Evaluator functions accept truth only after model freeze.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fair_lending.professor_model.config import BANK_COLUMNS, configuration_fingerprint

MODEL_ID = "professor_bank_ols_v1"
FEATURES = ("Y", "D", "C", "B")
DESIGN_COLUMNS = ("const", *FEATURES)
ESTIMATE_COLUMNS = (
    "applicant_id", "split", "model_id", "bank_PD_raw", "bank_PD_bounded",
    "prediction_out_of_bounds",
)
PREVIEW_COLUMNS = (
    "information_scope", "applicant_id", "Y", "D", "C", "B", "PN", "true_PD",
    "bank_PD_raw", "bank_PD_bounded",
)


def validate_ids(frame: pd.DataFrame, name: str) -> None:
    """A unique, nonempty string key is required; never silently drop records."""
    if "applicant_id" not in frame:
        raise ValueError(f"{name}: missing applicant_id")
    ids = frame.applicant_id
    if ids.isna().any() or not ids.is_unique or not ids.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{name}: missing, duplicate, or invalid applicant IDs")


def _finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("model inputs must be finite and nonmissing")
    return values


def join_bank_inputs(bank: pd.DataFrame, targets: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    """Read-time contract for observed covariates and the raw measurement only."""
    for name, frame, columns in (
        ("bank", bank, BANK_COLUMNS), ("targets", targets, ("applicant_id", "PN")),
    ):
        if tuple(frame.columns) != columns or len(frame) != expected_rows:
            raise ValueError(f"{name}: wrong schema or row count")
        validate_ids(frame, name)
        _finite(frame, columns[1:])
    if set(bank.applicant_id) != set(targets.applicant_id):
        raise ValueError("bank/target applicant IDs do not match")
    if not (bank.Y > 0).all() or not bank.B.isin([0, 1]).all():
        raise ValueError("income must be positive and B must be binary")
    return bank.merge(targets, on="applicant_id", validate="one_to_one").sort_values("applicant_id").reset_index(drop=True)


def make_splits(ids: pd.Series, split_config: dict[str, Any]) -> pd.DataFrame:
    """Sort first, then permute once with the dedicated PCG64 child [6]."""
    frame = pd.DataFrame({"applicant_id": ids})
    validate_ids(frame, "split")
    n_train, n_eval = split_config["train_rows"], split_config["evaluation_rows"]
    if min(n_train, n_eval) < 1 or n_train + n_eval != len(frame):
        raise ValueError("split counts do not cover the full population")
    if split_config["spawn_key"] != [6] or split_config["bit_generator"] != "PCG64" or split_config["applicant_order"] != "ascending_lexicographic_applicant_id":
        raise ValueError("Step 5 requires sorted IDs and reserved PCG64 child [6]")
    frame = frame.sort_values("applicant_id").reset_index(drop=True)
    child = np.random.SeedSequence(entropy=split_config["master_seed"], spawn_key=(6,))
    permutation = np.random.Generator(np.random.PCG64(child)).permutation(len(frame))
    labels = np.full(len(frame), "evaluation", dtype=object)
    labels[permutation[:n_train]] = "train"
    frame["split"] = labels
    return frame


def attach_splits(applicants: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    validate_ids(applicants, "applicants")
    validate_ids(splits, "splits")
    if tuple(splits.columns) != ("applicant_id", "split") or not splits.split.isin(["train", "evaluation"]).all():
        raise ValueError("invalid split schema or labels")
    if set(applicants.applicant_id) != set(splits.applicant_id) or set(splits.split) != {"train", "evaluation"}:
        raise ValueError("splits must cover every applicant, with both cohorts")
    return applicants.merge(splits, on="applicant_id", validate="one_to_one").sort_values("applicant_id").reset_index(drop=True)


def design_matrix(applicants: pd.DataFrame) -> pd.DataFrame:
    """Exactly [1,Y,D,C,B]; levels/ratio/original signal units, no transforms."""
    values = _finite(applicants, FEATURES)
    if not (values[:, 0] > 0).all() or not np.isin(values[:, 3], [0, 1]).all():
        raise ValueError("invalid observed income or group")
    matrix = pd.DataFrame(values, index=applicants.index, columns=FEATURES)
    matrix.insert(0, "const", 1.0)
    return matrix


def ids_fingerprint(ids: pd.Series) -> str:
    return configuration_fingerprint({"sorted_applicant_ids": sorted(ids.tolist())})


def fit_bank_ols(applicants: pd.DataFrame) -> dict[str, Any]:
    """Only train rows enter statsmodels. Evaluation PN is never consulted."""
    validate_ids(applicants, "fitting view")
    if not applicants.split.isin(["train", "evaluation"]).all():
        raise ValueError("invalid split label")
    train = applicants.loc[applicants.split == "train"].sort_values("applicant_id")
    matrix = design_matrix(train)
    target = _finite(train, ("PN",)).ravel()  # raw; no clipping or scale correction
    rank = int(np.linalg.matrix_rank(matrix.to_numpy()))
    if len(train) <= len(DESIGN_COLUMNS) or rank != len(DESIGN_COLUMNS):
        raise ValueError("training design must have full column rank and residual degrees of freedom")
    result = sm.OLS(target, matrix, missing="raise").fit(method="pinv")
    if not np.isfinite(result.params.to_numpy()).all():
        raise ValueError("nonfinite fitted coefficients")
    return {
        "schema_version": "professor-bank-regression-v1", "model_id": MODEL_ID,
        "family": "statsmodels_OLS", "solver": "pinv", "regularization": None,
        "feature_order": list(FEATURES), "design_order": list(DESIGN_COLUMNS),
        "intercept": True, "target": "PN", "target_transform": "none",
        "prediction_rescaling": "none", "bounding": "clip_0_1_separate_column",
        "coefficients": {name: float(result.params[name]) for name in DESIGN_COLUMNS},
        "training_mean_PN": float(target.mean()), "training_count": len(train),
        "training_ID_fingerprint": ids_fingerprint(train.applicant_id),
        "design_rank": rank, "design_condition_number": float(np.linalg.cond(matrix)),
        "units": {"const": "raw PN units", "Y": "annual synthetic USD, levels", "D": "DTI ratio", "C": "original continuous signal", "B": "binary 0/1"},
    }


def predict_bank_ols(applicants: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    """Replay transparent coefficients; labels, PN, and private columns ignored."""
    if model["model_id"] != MODEL_ID or tuple(model["design_order"]) != DESIGN_COLUMNS or tuple(model["feature_order"]) != FEATURES or model["intercept"] is not True:
        raise ValueError("incompatible model/feature contract")
    if set(model["coefficients"]) != set(DESIGN_COLUMNS):
        raise ValueError("incompatible coefficient schema")
    coefficients = np.array([model["coefficients"][key] for key in DESIGN_COLUMNS], dtype=float)
    raw = design_matrix(applicants).to_numpy() @ coefficients
    if not np.isfinite(coefficients).all() or not np.isfinite(raw).all():
        raise ValueError("predictions/coefficients must be finite")
    return raw


def risk_estimates(applicants: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    validate_ids(applicants, "prediction view")
    if not applicants.split.isin(["train", "evaluation"]).all():
        raise ValueError("invalid prediction split")
    raw = predict_bank_ols(applicants, model)
    return pd.DataFrame({
        "applicant_id": applicants.applicant_id.to_numpy(), "split": applicants.split.to_numpy(),
        "model_id": MODEL_ID, "bank_PD_raw": raw, "bank_PD_bounded": np.clip(raw, 0, 1),
        "prediction_out_of_bounds": (raw < 0) | (raw > 1),
    }).loc[:, ESTIMATE_COLUMNS]


def error_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 1 or len(actual) == 0 or not np.isfinite([actual, prediction]).all():
        raise ValueError("metrics require matching finite nonempty vectors")
    error = prediction - actual
    constant = np.ptp(actual) == 0 or np.ptp(prediction) == 0
    total_ss = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "mean_signed_error": float(error.mean()),
        "R_squared": None if total_ss == 0 else float(1 - np.sum(error ** 2) / total_ss),
        "Pearson_correlation": None if constant else float(np.corrcoef(actual, prediction)[0, 1]),
        "correlation_status": "undefined_constant_vector" if constant else "defined",
    }


def prediction_summary(raw: np.ndarray, bounded: np.ndarray) -> dict[str, Any]:
    raw, bounded = np.asarray(raw, dtype=float), np.asarray(bounded, dtype=float)
    if len(raw) < 2 or not np.isfinite([raw, bounded]).all() or not np.array_equal(bounded, np.clip(raw, 0, 1)):
        raise ValueError("distribution requires finite raw values and their separate clipping")
    return {
        "n": len(raw), "mean": float(raw.mean()), "median": float(np.median(raw)),
        "sample_SD": float(raw.std(ddof=1)), "minimum": float(raw.min()), "maximum": float(raw.max()),
        **{f"p{q:02d}": float(np.quantile(raw, q / 100)) for q in (1, 5, 50, 95, 99)},
        "below_zero_count": int((raw < 0).sum()), "below_zero_share": float((raw < 0).mean()),
        "above_one_count": int((raw > 1).sum()), "above_one_share": float((raw > 1).mean()),
        "bounded_zero_count": int((bounded == 0).sum()), "bounded_one_count": int((bounded == 1).sum()),
    }


def evaluate_bank_risk(
    applicants: pd.DataFrame, predictions: pd.DataFrame, truth: pd.DataFrame,
    model: dict[str, Any], preview_rows: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluator only. No fitted model or raw predictions are modified here."""
    for name, frame in (("applicants", applicants), ("predictions", predictions), ("truth", truth)):
        validate_ids(frame, name)
        if set(frame.applicant_id) != set(applicants.applicant_id):
            raise ValueError("evaluation join has mismatched IDs")
    if tuple(predictions.columns) != ESTIMATE_COLUMNS or tuple(truth.columns) != ("applicant_id", "P_income", "P_other", "true_PD"):
        raise ValueError("evaluation input schema mismatch")
    _finite(truth, ("P_income", "P_other", "true_PD"))
    if not truth.loc[:, ["P_income", "P_other", "true_PD"]].apply(lambda x: x.between(0, 1, inclusive="neither")).all().all():
        raise ValueError("invalid saved true probability")
    # Check labels align before using only the held-out subset.
    joined = applicants.merge(predictions, on=["applicant_id", "split"], validate="one_to_one")
    if len(joined) != len(applicants):
        raise ValueError("prediction split labels do not match applicants")
    evaluation = joined.loc[joined.split == "evaluation"].merge(truth, on="applicant_id", validate="one_to_one").sort_values("applicant_id")
    if len(evaluation) < 2:
        raise ValueError("at least two evaluation applicants are needed")
    pn, hidden = evaluation.PN.to_numpy(), evaluation.true_PD.to_numpy()
    raw, bounded = evaluation.bank_PD_raw.to_numpy(), evaluation.bank_PD_bounded.to_numpy()
    metrics = []
    for reference, estimate, actual, predicted in (
        ("held_out_PN", "OLS_raw", pn, raw),
        ("held_out_PN", "constant_training_mean_PN", pn, np.full(len(pn), model["training_mean_PN"])),
        ("hidden_true_PD", "OLS_raw", hidden, raw),
        ("hidden_true_PD", "OLS_bounded", hidden, bounded),
        ("scaled_true_risk_reference_0.8", "OLS_raw", .8 * hidden, raw),
    ):
        metrics.append({"reference": reference, "estimate": estimate, "split": "evaluation", "n": len(pn), **error_metrics(actual, predicted)})
    groups = []
    for group, frame in evaluation.groupby("B", sort=True):
        error = error_metrics(frame.true_PD.to_numpy(), frame.bank_PD_raw.to_numpy())
        groups.append({"B": int(group), "n_applicants": len(frame), "mean_true_PD": float(frame.true_PD.mean()), "mean_bank_PD_raw": float(frame.bank_PD_raw.mean()), "mean_bank_PD_bounded": float(frame.bank_PD_bounded.mean()), "raw_MAE": error["MAE"], "raw_mean_signed_error": error["mean_signed_error"]})
    preview = evaluation.head(preview_rows).copy()
    preview["information_scope"] = "EVALUATOR_ONLY_NOT_FOR_LENDER_TRAINING"
    summary = prediction_summary(raw, bounded)
    return pd.DataFrame(metrics), pd.DataFrame(groups), preview.loc[:, PREVIEW_COLUMNS], summary
