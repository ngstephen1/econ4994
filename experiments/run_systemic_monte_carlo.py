"""Run and summarize repeated-seed systemic-opportunity uncertainty experiments."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fair_lending.economic_lending.config import PROJECT_ROOT
from fair_lending.economic_lending.systemic_mc import (
    MC_CONFIG_PATH,
    MC_ANALYSIS_VERSION,
    code_revision,
    expected_record_counts,
    load_systemic_mc_config,
    load_valid_replication,
    model_performance_summary,
    monte_carlo_summary,
    monotonicity_summary,
    records_to_frames,
    replication_id,
    replication_path,
    replication_seeds,
    run_systemic_mc_replication,
    summarize_scalar,
    switcher_summary,
    write_replication_atomic,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _worker(index: int, config_path: str, diagnostic_directory: str | None = None) -> dict[str, Any]:
    return run_systemic_mc_replication(index, mc_config_path=config_path,
                                      diagnostic_directory=Path(diagnostic_directory) if diagnostic_directory else None)


def _controlled_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["risk_world", "systemic_strength", "family", "budget_id", "model"]
    for key, group in frame.groupby(keys, sort=True):
        valid = group[group["identified"].astype(bool)]
        failed_audit = group.get("audit_status", pd.Series("", index=group.index)).eq("failed_secondary_audit")
        row = {
            **dict(zip(keys, key, strict=True)),
            "n_attempted": int(len(group)),
            "n_identified": int(len(valid)),
            "identified_fraction": float(len(valid) / len(group)),
            "converged_fraction": float(group["converged"].mean()),
            "unidentified_fraction": float((~group["identified"].astype(bool)).mean()),
            "secondary_failure_count": int(failed_audit.sum()),
        }
        coefficient = summarize_scalar(valid["group_b_coefficient"])
        row.update({f"coefficient_{name}": value for name, value in coefficient.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _add_model_difference_rows(summary: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    fitted = runs[runs["family"].isin(["traditional", "ml"])]
    pivot = fitted.pivot_table(
        index=["replication_index", "risk_world", "systemic_strength", "budget_id"],
        columns="family",
        values="oracle_regret",
    ).reset_index()
    pivot["ml_minus_traditional_oracle_regret"] = pivot["ml"] - pivot["traditional"]
    rows = []
    for key, frame in pivot.groupby(["risk_world", "systemic_strength", "budget_id"], sort=True):
        rows.append(
            {
                "record_type": "model_comparison",
                "risk_world": key[0],
                "systemic_strength": key[1],
                "family": "ml_minus_traditional",
                "budget_id": key[2],
                "estimand": "oracle_regret_difference",
                **summarize_scalar(frame["ml_minus_traditional_oracle_regret"]),
            }
        )
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _render_figures(frames: dict[str, pd.DataFrame], output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    runs = frames["runs"]
    files: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        files.append(name)

    focus = runs[runs["systemic_strength"].isin([.2, .4])]
    for estimand, title, name in [
        ("delta_b_funding_rate", "Systemic change in Group B funding", "v2_systemic_mc_b_funding_distribution.png"),
        ("delta_funding_gap", "Systemic change in B−A funding gap", "v2_systemic_mc_gap_distribution.png"),
        ("delta_true_expected_portfolio_profit", "True expected portfolio-profit change", "v2_systemic_mc_true_profit_distribution.png"),
    ]:
        labels, values = [], []
        for key, frame in focus.groupby(["risk_world", "family", "systemic_strength", "budget_id"], sort=True):
            labels.append(f"{key[0].split('_')[0]}\n{key[1]}\ns={key[2]:.1f}\n{key[3]}")
            values.append(frame[estimand].to_numpy())
        fig, ax = plt.subplots(figsize=(24, 6))
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.axhline(0, color="0.4", lw=.8)
        ax.set(title=title, ylabel=estimand)
        ax.tick_params(axis="x", labelsize=7)
        save(fig, name)

    regret = runs[(runs["family"].isin(["traditional", "ml"])) & runs["systemic_strength"].isin([.2, .4])]
    labels, values = [], []
    for key, frame in regret.groupby(["risk_world", "family", "systemic_strength", "budget_id"], sort=True):
        labels.append(f"{key[0].split('_')[0]}\n{key[1]}\ns={key[2]:.1f}\n{key[3]}")
        values.append(frame["oracle_regret"].to_numpy())
    fig, ax = plt.subplots(figsize=(11, 5)); ax.boxplot(values, tick_labels=labels, showfliers=False); ax.set(title="Oracle-regret distribution", ylabel="Oracle regret ($)"); ax.tick_params(axis="x", labelsize=8); save(fig, "v2_systemic_mc_oracle_regret.png")

    summary = monte_carlo_summary(runs)
    selected = summary[(summary["estimand"].eq("delta_funding_gap")) & (summary["family"].eq("true_risk_reference")) & (summary["budget_id"].eq("moderate_40pct"))]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for risk_world, frame in selected.groupby("risk_world"):
        frame = frame.sort_values("systemic_strength")
        ax.plot(frame["systemic_strength"], frame["mean"], marker="o", label=risk_world)
        ax.fill_between(frame["systemic_strength"], frame["p2_5"], frame["p97_5"], alpha=.2)
    ax.axhline(0, color="0.4", lw=.8); ax.set(xlabel="Systemic strength s", ylabel="Funding-gap effect", title="Oracle strength response with replicate intervals"); ax.legend(); save(fig, "v2_systemic_mc_strength_intervals.png")

    opportunity = runs.drop_duplicates(["replication_index", "systemic_strength"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strength, frame in opportunity[opportunity["systemic_strength"] > 0].groupby("systemic_strength"):
        ax.hist(frame["switcher_count"], bins="auto", alpha=.45, label=f"s={strength:.1f}")
    ax.set(xlabel="Group B switcher count", ylabel="Replications", title="Opportunity-switcher distribution"); ax.legend(); save(fig, "v2_systemic_mc_switcher_counts.png")
    return files


def _load_or_run(
    indices: list[int],
    config: dict[str, Any],
    *,
    resume: bool,
    workers: int,
    summarize_only: bool = False,
    diagnostic_directory: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    revision = code_revision()
    fingerprint = config["metadata"]["config_fingerprint"]
    master_seed = int(config["design"]["master_seed"])
    counts = expected_record_counts(config)
    records: dict[int, dict[str, Any]] = {}
    pending: list[int] = []
    for index in indices:
        seeds = replication_seeds(master_seed, index)
        run_id = replication_id(fingerprint, seeds, revision)
        path = replication_path(config, run_id)
        existing = load_valid_replication(
            path,
            expected_run_id=run_id,
            expected_config_fingerprint=fingerprint,
            expected_revision=revision,
            expected_counts=counts,
            expected_design=config["design"],
        ) if resume or summarize_only else None
        if existing is None:
            pending.append(index)
        else:
            records[index] = existing
            print(f"replication {index}: resumed")
    if summarize_only:
        if pending:
            raise RuntimeError(f"summary-only: missing or invalid replications {pending}; no work launched")
        return [records[index] for index in indices], 0

    def persist(index, record):
        path = replication_path(config, record["run_id"])
        write_replication_atomic(record, path)
        if record["status"] == "success" and load_valid_replication(
            path, expected_run_id=record["run_id"], expected_config_fingerprint=fingerprint,
            expected_revision=revision, expected_counts=counts, expected_design=config["design"],
        ) is None:
            record = {**record, "status": "failed_validation", "failure_message": "post-write integrity check failed"}
            write_replication_atomic(record, path)
        records[index] = record
        print(f"replication {index}: {record['status']} ({record['timing']['total_seconds']:.1f}s)", flush=True)

    def stop_required():
        return sum(r["status"] != "success" for r in records.values()) / max(1, len(records)) > float(config["failure_policy"]["maximum_failure_fraction"])

    if workers == 1:
        for index in pending:
            record = _worker(index, str(config["metadata"]["config_path"]), diagnostic_directory)
            persist(index, record)
            if stop_required():
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Bounded batches: do not enqueue the entire experiment past a stop rule.
            for offset in range(0, len(pending), workers):
                futures = {executor.submit(_worker, index, str(config["metadata"]["config_path"]), diagnostic_directory): index for index in pending[offset:offset+workers]}
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        record = future.result()
                    except Exception as error:
                        seeds = replication_seeds(master_seed, index)
                        record = {"status": "failed_worker", "replication_index": index,
                                  "run_id": replication_id(fingerprint, seeds, revision), "seeds": vars(seeds),
                                  "failure_message": str(error), "timing": {"total_seconds": 0}}
                    persist(index, record)
                if stop_required():
                    break
    return [records[index] for index in sorted(records)], sum(index in records for index in pending)


def _summarize(records: list[dict[str, Any]], config: dict[str, Any], wall_seconds: float, newly_run: int, mode: str) -> int:
    attempted = len(records)
    successful = [record for record in records if record["status"] == "success"]
    failures = [record for record in records if record["status"] != "success"]
    failure_fraction = len(failures) / attempted if attempted else 0.0
    manifest = {
        "experiment_id": config["experiment_id"],
        "analysis_version": MC_ANALYSIS_VERSION,
        "config_fingerprint": config["metadata"]["config_fingerprint"],
        "systemic_config_fingerprint": config["systemic_config_fingerprint"],
        "code_revision": code_revision(),
        "mode": mode,
        "attempted_replications": attempted,
        "successful_replications": len(successful),
        "failed_replications": len(failures),
        "failure_fraction": failure_fraction,
        "newly_executed_replications": newly_run,
        "invocation_wall_seconds": wall_seconds,
        "recorded_latest_attempt_compute_seconds": sum(r.get("timing", {}).get("total_seconds", 0) for r in records),
        "target_successful_replications": config["design"]["target_successful_replications"],
        "completion_status": "target_complete" if len(successful) >= config["design"]["target_successful_replications"] and not failures else "interim_incomplete",
        "replications": [
            {
                "replication_index": record["replication_index"],
                "run_id": record["run_id"],
                "status": record["status"],
                "seeds": record["seeds"],
                "total_seconds": record.get("timing", {}).get("total_seconds"),
                "failure_stage": record.get("failure_stage"),
                "failure_type": record.get("failure_type"),
                "failure_message": record.get("failure_message"),
            }
            for record in records
        ],
    }
    attempts_directory = PROJECT_ROOT / config["storage"]["replication_directory"] / "attempts"
    attempt_records = [json.loads(p.read_text()) for p in attempts_directory.glob("*/*.json")]
    manifest["recorded_cumulative_attempt_compute_seconds"] = sum(r.get("timing", {}).get("total_seconds", 0) for r in attempt_records)
    manifest["recorded_attempt_count"] = len(attempt_records)
    manifest["runtime_note"] = "Compute seconds sum recorded attempts, not wall time; interrupted legacy work is not reconstructable."
    _write_json(PROJECT_ROOT / config["storage"]["manifest"], manifest)
    if failure_fraction > float(config["failure_policy"]["maximum_failure_fraction"]):
        print(f"failure fraction {failure_fraction:.1%} exceeds 5%; summaries not presented")
        return 1
    if not successful:
        return 1
    frames = records_to_frames(records)
    runs = frames["runs"].sort_values(["replication_index", "risk_world", "systemic_strength", "family", "budget_id"], kind="mergesort")
    summary = monte_carlo_summary(runs)
    switchers = switcher_summary(frames["pathway"], frames["switcher_lending"], runs)
    model = model_performance_summary(frames["model_performance"], runs, frames["ml_selections"])
    model = _add_model_difference_rows(model, runs)
    controlled = _controlled_summary(frames["controlled_audits"])
    monotonicity = monotonicity_summary(runs)
    output_frames = {
        "runs": runs,
        "summary": summary,
        "switchers": switchers,
        "model_performance": model,
        "controlled_audit": controlled,
        "monotonicity": monotonicity,
    }
    for key, frame in output_frames.items():
        sort_keys = [k for k in ["replication_index", "risk_world", "systemic_strength", "family", "budget_id", "record_type", "estimand", "model", "population", "variable"] if k in frame]
        if sort_keys:
            frame = frame.sort_values(sort_keys, kind="mergesort")
        path = PROJECT_ROOT / config["outputs"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    figures = _render_figures(frames, PROJECT_ROOT / "results" / "figures" / "systemic_mc_recovery") if len(successful) > 1 else []
    timing = frames["replications"]
    metrics = {
        **{key: value for key, value in manifest.items() if key != "replications"},
        "master_seed": config["design"]["master_seed"],
        "n_applicants_per_replication": config["design"]["n_applicants"],
        "cohort_sizes": {"historical_train": 6000, "historical_validation": 2000, "evaluation": 2000},
        "portfolios_solved": int(len(runs)),
        "mean_replication_seconds": float(timing.loc[timing["status"].eq("success"), "total_seconds"].mean()),
        "median_replication_seconds": float(timing.loc[timing["status"].eq("success"), "total_seconds"].median()),
        "controlled_audit_identified_fraction": float(frames["controlled_audits"]["identified"].mean()),
        "raw_replication_rows_persisted": False,
        "tables": {key: config["outputs"][key] for key in output_frames},
        "figures": figures,
    }
    _write_json(PROJECT_ROOT / config["storage"]["metrics"], metrics)
    print(json.dumps({key: metrics[key] for key in ["attempted_replications", "successful_replications", "failed_replications", "invocation_wall_seconds", "mean_replication_seconds", "portfolios_solved"]}, indent=2))
    return int(bool(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MC_CONFIG_PATH)
    parser.add_argument("--replications", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--replication-index", type=int, help="Run only this index; no implicit batch")
    parser.add_argument("--diagnostic-directory", type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_systemic_mc_config(args.config)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    count = int(config["design"]["pilot_replications"] if args.pilot else config["design"]["target_successful_replications"])
    if args.replications is not None:
        if args.replications <= 0:
            raise ValueError("replications must be positive")
        count = args.replications
    indices = list(range(count))
    if args.replication_index is not None:
        if args.replication_index < 0:
            raise ValueError("replication index must be nonnegative")
        indices = [args.replication_index]
    started = perf_counter()
    records, newly_run = _load_or_run(
        indices,
        config,
        resume=args.resume or args.summarize_only,
        workers=args.workers,
        summarize_only=args.summarize_only,
        diagnostic_directory=args.diagnostic_directory,
    )
    if args.summarize_only and newly_run:
        raise RuntimeError("summarize-only requires every requested replication to exist")
    return _summarize(records, config, perf_counter() - started, newly_run, "pilot" if args.pilot else "full")


if __name__ == "__main__":
    raise SystemExit(main())
