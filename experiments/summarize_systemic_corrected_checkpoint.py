"""Validate and summarize the predeclared five-run corrected checkpoint.

This script never generates populations, fits models, or solves portfolios.
It reads corrected records through the production integrity validator and keeps
legacy records in a diagnostic-only comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fair_lending.economic_lending.config import PROJECT_ROOT
from fair_lending.economic_lending.recovery import atomic_json, digest
from fair_lending.economic_lending.systemic_mc import (
    MC_ANALYSIS_VERSION,
    code_revision,
    expected_record_counts,
    load_systemic_mc_config,
    load_valid_replication,
    records_to_frames,
    replication_id,
    replication_path,
    replication_seeds,
)


CHECKPOINT_INDICES = (0, 1, 2, 3, 13)
EVIDENCE_STATUS = "INTERIM CORRECTED 5-REPLICATION SUMMARY"
REQUIRED_RUN_KEY = ("risk_world", "systemic_strength", "family", "budget_id")
PRIMARY_DELTA_METRICS = (
    "delta_a_funding_rate",
    "delta_b_funding_rate",
    "delta_funding_gap",
    "delta_principal_gap",
    "delta_funded_requested_gap",
    "delta_true_expected_portfolio_profit",
)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_corrected_records(indices=CHECKPOINT_INDICES) -> tuple[list[dict], dict]:
    config = load_systemic_mc_config()
    revision = code_revision()
    records = []
    for index in indices:
        seeds = replication_seeds(config["design"]["master_seed"], index)
        run_id = replication_id(config["metadata"]["config_fingerprint"], seeds, revision)
        path = replication_path(config, run_id)
        record = load_valid_replication(
            path,
            expected_run_id=run_id,
            expected_config_fingerprint=config["metadata"]["config_fingerprint"],
            expected_revision=revision,
            expected_counts=expected_record_counts(config),
            expected_design=config["design"],
        )
        if record is None:
            raise RuntimeError(f"corrected replication {index} is absent or invalid: {path}")
        records.append(record)
    return records, config


def provenance_summary(records: list[dict]) -> dict[str, Any]:
    fields = {
        "scientific_config_fingerprint": [r["config_fingerprint"] for r in records],
        "systemic_config_fingerprint": [r["systemic_config_fingerprint"] for r in records],
        "source_content_fingerprint": [r["provenance"]["fingerprint"] for r in records],
        "dependency_fingerprint": [_hash_json(r["provenance"]["dependencies"]) for r in records],
        "analysis_schema_version": [r["analysis_version"] for r in records],
    }
    solver_settings = []
    for record in records:
        unique = {
            _hash_json({k: row["solver_metadata"].get(k) for k in [
                "solver", "budget_constraint_scaling", "time_limit_seconds",
                "mip_feasibility_tolerance", "optimality_roundoff_tolerance",
                "budget_constraint_numerical_guard_dollars",
            ]})
            for row in record["runs"]
        }
        if len(unique) != 1:
            raise RuntimeError(f"replication {record['replication_index']} has inconsistent solver settings")
        solver_settings.append(next(iter(unique)))
    fields["solver_settings_fingerprint"] = solver_settings
    result = {}
    for name, values in fields.items():
        unique = sorted(set(values))
        result[name] = unique[0] if len(unique) == 1 else unique
        result[f"{name}_consistent"] = len(unique) == 1
    result["all_consistent"] = all(result[f"{name}_consistent"] for name in fields)
    return result


def _stats(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="raise")
    if not np.isfinite(values).all():
        raise ValueError("checkpoint summary received nonfinite values")
    n = len(values)
    sd = float(values.std(ddof=1)) if n > 1 else math.nan
    return {
        "n_corrected": n,
        "mean": float(values.mean()),
        "sd": sd,
        "mcse": sd / math.sqrt(n) if n > 1 else math.nan,
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def build_interim_summary(records: list[dict]) -> pd.DataFrame:
    frames = records_to_frames(records)
    runs = frames["runs"]
    rows: list[dict] = []
    for keys, group in runs.groupby(["risk_world", "systemic_strength", "family", "budget_id"], sort=True):
        for metric in PRIMARY_DELTA_METRICS:
            rows.append({"evidence_status": EVIDENCE_STATUS, "estimand": metric,
                         **dict(zip(REQUIRED_RUN_KEY, keys, strict=True)), **_stats(group[metric])})
    regret = runs[runs["family"].isin(["traditional", "ml"])]
    for keys, group in regret.groupby(list(REQUIRED_RUN_KEY), sort=True):
        rows.append({"evidence_status": EVIDENCE_STATUS, "estimand": "oracle_regret",
                     **dict(zip(REQUIRED_RUN_KEY, keys, strict=True)), **_stats(group["oracle_regret"])})

    opportunity_rows = []
    for record in records:
        frame = pd.DataFrame(record["opportunity"])
        neutral = frame.loc[frame.systemic_strength.eq(0), "group_b_opportunity_access_rate"].item()
        frame["opportunity_access_change"] = frame["group_b_opportunity_access_rate"] - neutral
        frame["replication_index"] = record["replication_index"]
        opportunity_rows.append(frame)
    opportunity = pd.concat(opportunity_rows, ignore_index=True)
    for strength, group in opportunity.groupby("systemic_strength", sort=True):
        for metric in ["opportunity_access_change", "switcher_count"]:
            rows.append({"evidence_status": EVIDENCE_STATUS, "estimand": metric,
                         "risk_world": "not_applicable", "systemic_strength": strength,
                         "family": "not_applicable", "budget_id": "not_applicable",
                         **_stats(group[metric])})
    return pd.DataFrame(rows).sort_values(
        ["estimand", "risk_world", "systemic_strength", "family", "budget_id"], kind="mergesort"
    ).reset_index(drop=True)


def solver_summary(records: list[dict]) -> dict[str, Any]:
    rows = [row for record in records for row in record["runs"]]
    gaps = np.array([row["solver_mip_gap"] for row in rows], dtype=float)
    signed_residuals = np.array([row["solver_metadata"]["budget_residual_dollars"] for row in rows], dtype=float)
    budgets = np.array([row["bank_budget"] for row in rows], dtype=float)
    statuses = [row["solver_metadata"]["solver_status_code"] for row in rows]
    messages = [row["solver_status"].lower() for row in rows]
    diagnostic_counts: dict[int, int] = {}
    roots = [PROJECT_ROOT / "results/recovery/prompt17b_solver_cases",
             PROJECT_ROOT / "results/recovery/replication13_solver_cases"]
    for index in CHECKPOINT_INDICES:
        candidates = [(directory, len(list(directory.glob("*.json"))))
                      for root in roots for directory in root.glob(f"r{index}_*")]
        complete = [count for _, count in candidates if count == 72]
        diagnostic_counts[index] = 72 if complete else 0
    return {
        "portfolios_attempted": len(rows),
        "portfolios_optimal": int(sum(row["solver_success"] and row["solver_mip_gap"] <= row["solver_metadata"]["optimality_roundoff_tolerance"] for row in rows)),
        "portfolios_time_limited": sum("time limit" in message for message in messages),
        "portfolios_failed": sum(status != 0 for status in statuses),
        "maximum_reported_mip_gap": float(gaps.max()),
        "maximum_signed_budget_residual_dollars": float(signed_residuals.max()),
        "maximum_budget_violation_dollars": float(np.maximum(signed_residuals, 0).max()),
        "maximum_normalized_feasibility_violation": float(np.maximum(signed_residuals / budgets, 0).max()),
        "solver_input_diagnostics_captured": sum(diagnostic_counts.values()),
        "diagnostic_files_by_replication": diagnostic_counts,
        "diagnostic_capture_enabled_for_all": sum(diagnostic_counts.values()) == len(rows),
        "error_replay_diagnostics_triggered": 0,
    }


def runtime_summary(records: list[dict]) -> dict[str, Any]:
    frame = pd.DataFrame([{"replication_index": r["replication_index"], **r["timing"]} for r in records])
    attempts = []
    for path in (PROJECT_ROOT / "results/metrics/systemic_mc_recovery_runs/attempts").glob("*/*.json"):
        attempt = json.loads(path.read_text())
        if attempt.get("analysis_version") == MC_ANALYSIS_VERSION and attempt.get("replication_index") in CHECKPOINT_INDICES:
            attempts.append(attempt)
    cumulative = (pd.DataFrame([{"replication_index": a["replication_index"],
                                 "attempt_seconds": a.get("timing", {}).get("total_seconds", 0),
                                 "attempt_status": a.get("status")} for a in attempts])
                  .groupby("replication_index", as_index=False)
                  .agg(cumulative_attempt_seconds=("attempt_seconds", "sum"),
                       recorded_attempt_count=("attempt_seconds", "size"),
                       recorded_failed_attempts=("attempt_status", lambda s: int((s != "success").sum()))))
    frame = frame.merge(cumulative, on="replication_index", validate="one_to_one")
    runtime = frame["cumulative_attempt_seconds"]
    stats = _stats(runtime)
    return {
        "per_replication": frame.sort_values("replication_index").to_dict("records"),
        **stats,
        "estimated_remaining_45_seconds_using_median": 45 * stats["median"],
        "estimated_remaining_45_seconds_using_mean": 45 * stats["mean"],
    }


def resume_validation() -> dict[str, Any]:
    path = PROJECT_ROOT / "results/metrics/v2_systemic_mc_recovery_manifest.json"
    manifest = json.loads(path.read_text())
    indices = [row["replication_index"] for row in manifest["replications"]]
    passed = (manifest["newly_executed_replications"] == 0 and manifest["successful_replications"] == 1
              and manifest["failed_replications"] == 0 and indices == [0])
    return {
        "tested_replication_index": 0,
        "newly_executed_replications": manifest["newly_executed_replications"],
        "invocation_wall_seconds": manifest["invocation_wall_seconds"],
        "record_reused": passed,
        "generation_model_fitting_and_optimization_launched": False if passed else "validation_failed",
    }


def compare_legacy(records: list[dict]) -> tuple[pd.DataFrame, dict[str, Any]]:
    legacy_by_index = {}
    for path in (PROJECT_ROOT / "results/metrics/systemic_mc_runs").glob("*.json"):
        record = json.loads(path.read_text())
        legacy_by_index[record["replication_index"]] = record
    comparisons = []
    for corrected in records:
        index = corrected["replication_index"]
        legacy = legacy_by_index.get(index)
        if legacy is None or legacy.get("status") != "success":
            comparisons.append({"replication_index": index, "legacy_status": legacy.get("status") if legacy else "absent",
                                "corrected_status": corrected["status"],
                                "portfolio_decisions_comparison": "unavailable_not_persisted"})
            continue
        old = pd.DataFrame(legacy["runs"]).set_index(list(REQUIRED_RUN_KEY)).sort_index()
        new = pd.DataFrame(corrected["runs"]).set_index(list(REQUIRED_RUN_KEY)).sort_index()
        administrative = {"solver_runtime_seconds", "solver_mip_gap", "replication_index", "bank_budget"}
        numeric = sorted(set(old.select_dtypes(include=[np.number]).columns)
                         & set(new.select_dtypes(include=[np.number]).columns) - administrative)
        diffs = (new[numeric].astype(float) - old[numeric].astype(float)).abs().to_numpy().ravel()
        gap_diffs = (new["solver_mip_gap"].astype(float) - old["solver_mip_gap"].astype(float)).abs()
        solver_status_differences = int((new["solver_status"].astype(str) != old["solver_status"].astype(str)).sum())
        comparisons.append({"replication_index": index, "legacy_status": "provisional_legacy",
                            "corrected_status": "validated_corrected", "downstream_values_compared": len(diffs),
                            "exact_downstream_values": int((diffs == 0).sum()),
                            "differing_downstream_values": int((diffs != 0).sum()),
                            "maximum_downstream_absolute_difference": float(diffs.max()) if len(diffs) else 0.0,
                            "solver_gap_values_differing": int((gap_diffs != 0).sum()),
                            "maximum_solver_gap_difference": float(gap_diffs.max()),
                            "solver_status_differences": solver_status_differences,
                            "portfolio_decisions_comparison": "unavailable_not_persisted"})
    defaults = {"downstream_values_compared": 0, "exact_downstream_values": 0,
                "differing_downstream_values": 0, "maximum_downstream_absolute_difference": None,
                "solver_gap_values_differing": 0, "maximum_solver_gap_difference": None,
                "solver_status_differences": None}
    comparisons = [{**defaults, **row} for row in comparisons]
    frame = pd.DataFrame(comparisons).sort_values("replication_index")
    report = {
        "matching_indices": frame.replication_index.tolist(),
        "exact_downstream_values": int(frame.exact_downstream_values.sum()),
        "differing_downstream_values": int(frame.differing_downstream_values.sum()),
        "maximum_downstream_absolute_difference": float(frame.maximum_downstream_absolute_difference.dropna().max()),
        "solver_gap_values_differing": int(frame.solver_gap_values_differing.sum()),
        "maximum_solver_gap_difference": float(frame.maximum_solver_gap_difference.dropna().max()),
        "solver_status_differences": int(frame.solver_status_differences.dropna().sum()),
        "portfolio_decisions_comparison": "unavailable: neither legacy nor corrected compact records persist applicant decisions",
        "legacy_scientifically_reusable": False,
        "reason": "legacy records lack corrected source provenance and complete decision vectors; diagnostic aggregate agreement cannot certify scientific equivalence",
    }
    return frame, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    records, _ = load_corrected_records()
    provenance = provenance_summary(records)
    if not provenance["all_consistent"]:
        raise RuntimeError("corrected provenance differs; refusing aggregation")
    runs = records_to_frames(records)["runs"].copy()
    runs.insert(0, "evidence_status", "validated_corrected")
    runs = runs.sort_values(["replication_index", *REQUIRED_RUN_KEY], kind="mergesort")
    summary = build_interim_summary(records)
    legacy_frame, legacy_report = compare_legacy(records)
    runtime = runtime_summary(records)
    solvers = solver_summary(records)
    completion_ready = (
        len(records) == 5 and len(runs) == 360 and solvers["portfolios_optimal"] == 360
        and solvers["portfolios_failed"] == 0 and solvers["portfolios_time_limited"] == 0
        and provenance["all_consistent"]
    )
    decision = "A. READY_FOR_FULL_50" if completion_ready else "B. NEEDS_FURTHER_SOLVER_DEBUGGING"
    metrics = {
        "evidence_status": EVIDENCE_STATUS,
        "selected_replication_indices": list(CHECKPOINT_INDICES),
        "attempted": 5, "corrected_successful": len(records), "failed": 5 - len(records),
        "provenance": provenance, "runtime": runtime, "solver": solvers,
        "resume_validation": resume_validation(),
        "legacy_comparison": legacy_report, "go_no_go": decision,
        "full_50_launched": False,
    }
    table_dir = args.output_root / "tables"; metric_dir = args.output_root / "metrics"
    table_dir.mkdir(parents=True, exist_ok=True); metric_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(table_dir / "v2_systemic_mc_corrected_checkpoint_runs.csv", index=False)
    summary.to_csv(table_dir / "v2_systemic_mc_corrected_checkpoint_summary.csv", index=False)
    legacy_frame.to_csv(table_dir / "v2_systemic_mc_corrected_vs_legacy.csv", index=False)
    atomic_json(metric_dir / "v2_systemic_mc_corrected_checkpoint.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0 if completion_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
