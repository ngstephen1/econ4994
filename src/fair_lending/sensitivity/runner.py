"""Single-run evaluation and resumable Monte Carlo orchestration."""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fair_lending.analysis.descriptive import black_white_raw_gap
from fair_lending.analysis.estimands import (
    standardized_black_white_contrast,
    true_direct_effect,
)
from fair_lending.analysis.logit import MODEL_ORDER, fit_logit_model, prepare_regression_sample
from fair_lending.models.evaluation import evaluate_predictions
from fair_lending.models.features import RACE_AWARE, RACE_BLIND, select_features
from fair_lending.models.group_audit import black_white_disparities, group_prediction_metrics
from fair_lending.models.training import deterministic_split, tune_on_validation
from fair_lending.sensitivity.design import (
    ANALYSIS_VERSION,
    MODEL_POLICY,
    RunSpec,
    resolve_sensitivity_config,
    run_id,
    run_identity_payload,
    upstream_treatment,
)
from fair_lending.sensitivity.metrics import mechanism_signature
from fair_lending.sensitivity.persistence import (
    load_matching_record,
    record_path,
    write_record_atomic,
)
from fair_lending.simulation.config import stable_fingerprint
from fair_lending.simulation.generator import generate_from_resolved_config


def _prefix_metrics(prefix: str, values: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {f"{prefix}_{field}": values[field] for field in fields}


def evaluate_run(
    spec: RunSpec,
    *,
    intercept: float,
    git_revision: str | None,
    git_worktree_dirty: bool | None,
) -> dict[str, Any]:
    """Generate, analyze, and release one Monte Carlo replication."""
    started = perf_counter()
    config = resolve_sensitivity_config(spec)
    fingerprint = stable_fingerprint(config)
    identifier = run_id(spec, fingerprint)
    identity = run_identity_payload(spec, fingerprint)
    data, generation = generate_from_resolved_config(config, intercept)
    raw = black_white_raw_gap(data, spec.scenario_family)
    sample = prepare_regression_sample(data, config)
    truth = true_direct_effect(sample, config, intercept)

    record: dict[str, Any] = {
        "status": "completed",
        "run_id": identifier,
        **identity,
        "families": list(spec.families),
        "scenario_family": spec.scenario_family,
        "frozen_intercept": float(intercept),
        **upstream_treatment(spec.upstream_strength),
        "raw_approval_gap": raw["raw_approval_gap"],
        "raw_standard_error": raw["standard_error"],
        "raw_ci_low": raw["ci_low"],
        "raw_ci_high": raw["ci_high"],
        "white_approval_rate": raw["white_approval_rate"],
        "black_approval_rate": raw["black_approval_rate"],
        **truth,
        "git_revision": git_revision,
        "git_worktree_dirty": git_worktree_dirty,
        "random_stream_spawn_keys": generation["random_stream_spawn_keys"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    statistical_fields = (
        "black_coefficient",
        "black_se",
        "black_z",
        "black_p_value",
        "black_ci_low",
        "black_ci_high",
        "black_odds_ratio",
        "converged",
        "optimizer",
        "optimizer_fallback_used",
        "dropped_separation_columns",
        "iterations",
        "condition_number",
        "maximum_standard_error",
    )
    for model_name in MODEL_ORDER:
        fitted = fit_logit_model(data, config, model_name)
        record.update(_prefix_metrics(model_name, fitted["summary"], statistical_fields))
        record[f"{model_name}_adjusted_probability_gap"] = (
            standardized_black_white_contrast(
                fitted["result"], fitted["design_matrix"]
            )
        )

    split = deterministic_split(data, spec.child_seed)
    record.update(
        n_train=len(split.train),
        n_validation=len(split.validation),
        n_test=len(split.test),
    )
    test_truth_gap: float | None = None
    for regime, prefix in ((RACE_BLIND, "race_blind"), (RACE_AWARE, "race_aware")):
        selected = tune_on_validation(
            split, regime, "logistic_regression", spec.child_seed
        )
        probability = selected["pipeline"].predict_proba(
            select_features(split.test, regime)
        )[:, 1]
        overall = evaluate_predictions(
            split.test["approved"].to_numpy(dtype=int),
            probability,
            split.test["approval_probability_true"].to_numpy(dtype=float),
            0.50,
        )
        audit = group_prediction_metrics(split.test, probability, 0.50)
        gaps = black_white_disparities(audit)
        indexed = audit.set_index("race")
        record.update(
            _prefix_metrics(
                prefix,
                overall,
                (
                    "roc_auc",
                    "log_loss",
                    "brier_score",
                    "true_probability_mae",
                    "true_probability_rmse",
                    "true_probability_correlation",
                    "mean_prediction_minus_true_probability",
                ),
            )
        )
        record[f"{prefix}_predicted_probability_gap"] = gaps[
            "predicted_probability_gap"
        ]
        record[f"{prefix}_thresholded_approval_gap"] = gaps[
            "predicted_approval_gap"
        ]
        record[f"{prefix}_observed_test_approval_gap"] = gaps[
            "observed_black_white_gap"
        ]
        record[f"{prefix}_true_probability_mae_white"] = float(
            indexed.loc["White", "true_probability_mae"]
        )
        record[f"{prefix}_true_probability_mae_black"] = float(
            indexed.loc["Black", "true_probability_mae"]
        )
        record[f"{prefix}_prediction_minus_truth_white"] = float(
            indexed.loc["White", "mean_prediction_minus_true_probability"]
        )
        record[f"{prefix}_prediction_minus_truth_black"] = float(
            indexed.loc["Black", "mean_prediction_minus_true_probability"]
        )
        record[f"{prefix}_selected_hyperparameters"] = selected[
            "selected_parameters"
        ]
        if test_truth_gap is None:
            test_truth_gap = float(gaps["true_probability_black_white_gap"])

    record["observed_test_approval_gap"] = record[
        "race_blind_observed_test_approval_gap"
    ]
    record["test_true_probability_gap"] = float(test_truth_gap)
    for prefix in ("race_blind", "race_aware"):
        record[f"{prefix}_disparity_recovery_error"] = (
            record[f"{prefix}_predicted_probability_gap"]
            - record["test_true_probability_gap"]
        )
    record["mechanism_signature"] = mechanism_signature(record)
    record["runtime_seconds"] = float(perf_counter() - started)
    return record


def failed_record(
    spec: RunSpec,
    *,
    intercept: float,
    git_revision: str | None,
    git_worktree_dirty: bool | None,
    error: BaseException,
) -> dict[str, Any]:
    config = resolve_sensitivity_config(spec)
    fingerprint = stable_fingerprint(config)
    return {
        "status": "failed",
        "run_id": run_id(spec, fingerprint),
        **run_identity_payload(spec, fingerprint),
        "families": list(spec.families),
        "scenario_family": spec.scenario_family,
        "frozen_intercept": float(intercept),
        "git_revision": git_revision,
        "git_worktree_dirty": git_worktree_dirty,
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def _worker(payload: tuple[RunSpec, float, str | None, bool | None]) -> dict[str, Any]:
    spec, intercept, revision, dirty = payload
    try:
        # NumPy/SciPy may otherwise create several BLAS threads inside each
        # process. One numerical thread per process avoids oversubscription and
        # makes --workers scale predictably on a laptop.
        with threadpool_limits(limits=1):
            return evaluate_run(
                spec,
                intercept=intercept,
                git_revision=revision,
                git_worktree_dirty=dirty,
            )
    except Exception as error:  # pragma: no cover - exercised by integration failures
        return failed_record(
            spec,
            intercept=intercept,
            git_revision=revision,
            git_worktree_dirty=dirty,
            error=error,
        )


def execute_design(
    specs: list[RunSpec],
    *,
    intercept: float,
    revision: str | None,
    dirty: bool | None,
    resume: bool,
    workers: int,
    project_root: Path | str | None = None,
    progress_every: int = 25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Execute or resume a stable design, saving every result immediately."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    pending: list[RunSpec] = []
    resumed = 0
    for spec in specs:
        config = resolve_sensitivity_config(spec)
        fingerprint = stable_fingerprint(config)
        identifier = run_id(spec, fingerprint)
        existing = (
            load_matching_record(
                record_path(identifier, project_root),
                expected_run_id=identifier,
                expected_config_fingerprint=fingerprint,
            )
            if resume
            else None
        )
        if existing is not None:
            existing["families"] = list(spec.families)
            completed.append(existing)
            resumed += 1
        else:
            pending.append(spec)

    payloads = [(spec, intercept, revision, dirty) for spec in pending]
    if workers == 1:
        iterator = map(_worker, payloads)
        for index, record in enumerate(iterator, start=1):
            write_record_atomic(record, record_path(record["run_id"], project_root))
            (completed if record["status"] == "completed" else failed).append(record)
            if index % progress_every == 0 or index == len(pending):
                print(f"completed {index}/{len(pending)} pending runs", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, payload): payload[0] for payload in payloads}
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                write_record_atomic(record, record_path(record["run_id"], project_root))
                (completed if record["status"] == "completed" else failed).append(record)
                if index % progress_every == 0 or index == len(pending):
                    print(f"completed {index}/{len(pending)} pending runs", flush=True)

    completed.sort(
        key=lambda row: (
            row["n_rows"],
            row["direct_black_log_odds"],
            row["upstream_strength"],
            row["replication_index"],
        )
    )
    failed.sort(key=lambda row: row["run_id"])
    return completed, failed, resumed


def records_frame(records: list[dict[str, Any]], families: set[str]) -> pd.DataFrame:
    """Project unique scientific runs into one requested output family."""
    rows = []
    for record in records:
        memberships = families.intersection(record["families"])
        for family in sorted(memberships):
            row = dict(record)
            row["experiment_family"] = family
            for key in ("families", "random_stream_spawn_keys"):
                if key in row:
                    row[key] = json.dumps(row[key], sort_keys=True)
            for key in (
                "race_blind_selected_hyperparameters",
                "race_aware_selected_hyperparameters",
            ):
                if key in row:
                    row[key] = json.dumps(row[key], sort_keys=True)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "experiment_family",
            "n_rows",
            "direct_black_log_odds",
            "upstream_strength",
            "replication_index",
        ]
    ).reset_index(drop=True)
