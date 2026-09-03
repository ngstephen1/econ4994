"""Consolidated result tables and paper-ready summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fair_lending.sensitivity.metrics import (
    coverage_table,
    detection_table,
    infer_detection_thresholds,
    signature_table,
    summarize_runs,
)
from fair_lending.sensitivity.runner import records_frame
from fair_lending.simulation.config import PROJECT_ROOT


FAMILY_OUTPUTS = {
    "direct": ("sensitivity_direct_runs.csv", "sensitivity_direct_summary.csv", {"direct"}),
    "upstream": ("sensitivity_upstream_runs.csv", "sensitivity_upstream_summary.csv", {"upstream"}),
    "mixed": ("sensitivity_mixed_runs.csv", "sensitivity_mixed_summary.csv", {"mixed"}),
    "sample_size": (
        "sensitivity_sample_size_runs.csv",
        "sensitivity_sample_size_summary.csv",
        {"sample_size", "detection"},
    ),
}


def paper_ready_summary(runs: pd.DataFrame, detection: pd.DataFrame) -> pd.DataFrame:
    """Select representative 100,000-row mechanism settings for presentation."""
    choices = [
        ("Fair", 0.0, 0.0),
        ("Mild direct", -0.10, 0.0),
        ("Moderate direct", -0.25, 0.0),
        ("Strong direct", -0.50, 0.0),
        ("Mild upstream", 0.0, 0.50),
        ("Moderate upstream", 0.0, 1.00),
        ("Strong upstream", 0.0, 1.50),
        ("Moderate mixed", -0.25, 1.00),
        ("Strong mixed", -0.50, 1.50),
    ]
    rows = []
    for label, direct, upstream in choices:
        group = runs.loc[
            runs["n_rows"].eq(100_000)
            & runs["direct_black_log_odds"].eq(direct)
            & runs["upstream_strength"].eq(upstream)
        ].drop_duplicates("run_id")
        if group.empty:
            continue
        diagnostic = detection.loc[
            detection["n_rows"].eq(100_000)
            & detection["direct_black_log_odds"].eq(direct)
            & detection["upstream_strength"].eq(upstream)
        ]
        rows.append(
            {
                "scenario_setting": label,
                "direct_black_log_odds": direct,
                "upstream_strength": upstream,
                "raw_approval_gap": group["raw_approval_gap"].mean(),
                "adjusted_probability_gap": group[
                    "model_2_adjusted_probability_gap"
                ].mean(),
                "race_blind_ml_gap": group[
                    "race_blind_predicted_probability_gap"
                ].mean(),
                "race_aware_ml_gap": group[
                    "race_aware_predicted_probability_gap"
                ].mean(),
                "true_direct_probability_contrast": group[
                    "true_direct_probability_gap"
                ].mean(),
                "detection_probability": (
                    diagnostic["model_2_detection_probability"].iloc[0]
                    if len(diagnostic)
                    else float("nan")
                ),
                "coverage": (
                    diagnostic["model_2_coverage_probability"].iloc[0]
                    if len(diagnostic)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def write_output_tables(
    records: list[dict[str, Any]], project_root: Path | str | None = None
) -> dict[str, Path]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    directory = root / "results" / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    all_frames = []
    for family, (runs_name, summary_name, memberships) in FAMILY_OUTPUTS.items():
        frame = records_frame(records, memberships)
        if frame.empty:
            continue
        runs_path = directory / runs_name
        summary_path = directory / summary_name
        frame.to_csv(runs_path, index=False)
        summarize_runs(frame, family).to_csv(summary_path, index=False)
        paths[f"{family}_runs"] = runs_path
        paths[f"{family}_summary"] = summary_path
        all_frames.append(frame)

    all_runs = (
        pd.concat(all_frames, ignore_index=True)
        .drop_duplicates("run_id")
        .sort_values(
            ["n_rows", "direct_black_log_odds", "upstream_strength", "replication_index"]
        )
        .reset_index(drop=True)
    )
    detection = detection_table(all_runs)
    coverage = coverage_table(all_runs)
    signatures = signature_table(all_runs)
    thresholds = infer_detection_thresholds(detection)
    paper = paper_ready_summary(all_runs, detection)
    extras = {
        "detection": ("monte_carlo_detection.csv", detection),
        "coverage": ("monte_carlo_coverage.csv", coverage),
        "signatures": ("mechanism_signatures.csv", signatures),
        "thresholds": ("detection_thresholds.csv", thresholds),
        "paper": ("sensitivity_paper_summary.csv", paper),
    }
    for key, (filename, frame) in extras.items():
        path = directory / filename
        frame.to_csv(path, index=False)
        paths[key] = path
    return paths
