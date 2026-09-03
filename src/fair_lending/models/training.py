"""Deterministic splitting, sklearn pipelines, and validation-only selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fair_lending.models.features import FeatureSpec, get_feature_spec, select_features


MODEL_ORDER = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
}


@dataclass(frozen=True)
class DataSplit:
    """Non-overlapping application-level development and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def deterministic_split(data: pd.DataFrame, seed: int = 4_994) -> DataSplit:
    """Create an exact 60/20/20 split from one seeded row permutation.

    Assignment depends only on row order, sample size, and seed—not on the
    outcome—so shared application IDs occupy the same partition in every
    scenario.
    """
    n_rows = len(data)
    if n_rows < 5:
        raise ValueError("At least five observations are required")
    order = np.random.default_rng(seed).permutation(n_rows)
    train_end = int(0.60 * n_rows)
    validation_end = train_end + int(0.20 * n_rows)

    def take(indices: np.ndarray) -> pd.DataFrame:
        return data.iloc[indices].copy().reset_index(drop=True)

    return DataSplit(
        train=take(order[:train_end]),
        validation=take(order[train_end:validation_end]),
        test=take(order[validation_end:]),
    )


def candidate_parameters(model_name: str) -> list[dict[str, Any]]:
    """Return the small, predeclared validation grid for one model family."""
    grids = {
        "logistic_regression": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
        "random_forest": [
            {"n_estimators": 160, "max_depth": 10, "min_samples_leaf": 20},
            {"n_estimators": 160, "max_depth": None, "min_samples_leaf": 20},
            {"n_estimators": 160, "max_depth": None, "min_samples_leaf": 5},
        ],
        "hist_gradient_boosting": [
            {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 15},
            {"learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31},
            {"learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 31},
        ],
    }
    try:
        return grids[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown model {model_name!r}; expected {MODEL_ORDER}") from error


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Construct train-fitted scaling and one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(spec.numeric)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(spec.categorical),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    model_name: str,
    parameters: dict[str, Any],
    spec: FeatureSpec,
    seed: int,
) -> Pipeline:
    """Build one deterministic preprocessing/classifier pipeline."""
    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            C=float(parameters["C"]), solver="lbfgs", max_iter=1_000, random_state=seed
        )
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            **parameters, max_features="sqrt", n_jobs=-1, random_state=seed
        )
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            **parameters,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model {model_name!r}; expected {MODEL_ORDER}")
    return Pipeline(
        [("preprocess", build_preprocessor(spec)), ("classifier", estimator)]
    )


def tune_on_validation(
    split: DataSplit,
    regime: str,
    model_name: str,
    seed: int = 4_994,
) -> dict[str, Any]:
    """Fit candidates on train only and select by validation log loss.

    The test partition is absent from selection. The selected pipeline remains
    fit only on the 60% training partition.
    """
    spec = get_feature_spec(regime)
    x_train = select_features(split.train, regime)
    y_train = split.train["approved"].to_numpy(dtype=int)
    x_validation = select_features(split.validation, regime)
    y_validation = split.validation["approved"].to_numpy(dtype=int)
    records: list[dict[str, Any]] = []
    fitted_candidates: list[Pipeline] = []

    for candidate_index, parameters in enumerate(candidate_parameters(model_name)):
        pipeline = build_pipeline(model_name, parameters, spec, seed)
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_validation)[:, 1]
        records.append(
            {
                "candidate_index": candidate_index,
                "parameters": parameters,
                "validation_log_loss": float(log_loss(y_validation, probability)),
                "validation_brier_score": float(
                    brier_score_loss(y_validation, probability)
                ),
                "validation_roc_auc": float(roc_auc_score(y_validation, probability)),
            }
        )
        fitted_candidates.append(pipeline)

    selected_index = min(
        range(len(records)),
        key=lambda index: (
            records[index]["validation_log_loss"],
            records[index]["validation_brier_score"],
            -records[index]["validation_roc_auc"],
            index,
        ),
    )
    for index, record in enumerate(records):
        record["selected"] = index == selected_index
    return {
        "pipeline": fitted_candidates[selected_index],
        "selected_parameters": records[selected_index]["parameters"],
        "selection_records": records,
        "selection_metric": "minimum validation log loss",
        "n_train": len(split.train),
        "n_validation": len(split.validation),
    }
